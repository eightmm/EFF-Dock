import csv
import json
import sys
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem

from scripts.external_models import prepare_diffdock_pocket_inputs
from scripts.external_models.evaluate_native_outputs import load_pose_records, no_align_rmsd
from scripts.external_models.prepare_vina_recovery_shards import write_recovery_shards
from scripts.external_models.repair_diffbindfr_output_pdb import repair_pdb
from scripts.external_models.run_posebench_vina import parse_vina_results
from scripts.external_models.sigmadock_compat import (
    assign_missing_stereochemistry_from_3d,
    recover_nonstandard_inter_residue_bonds,
    remove_residues_without_ca,
)

ROOT = Path(__file__).resolve().parents[1]


def test_parse_vina_results_preserves_model_order(tmp_path: Path) -> None:
    pdbqt = tmp_path / "poses.pdbqt"
    pdbqt.write_text(
        "MODEL 1\n"
        "REMARK VINA RESULT: -7.500 0.000 0.000\n"
        "ENDMDL\n"
        "MODEL 2\n"
        "REMARK VINA RESULT: -7.100 1.250 2.500\n"
        "ENDMDL\n"
    )

    assert parse_vina_results(pdbqt) == [
        (-7.5, 0.0, 0.0),
        (-7.1, 1.25, 2.5),
    ]


def test_vina_recovery_shards_exclude_completed_targets(tmp_path: Path) -> None:
    input_csv = tmp_path / "inputs.csv"
    input_csv.write_text("complex_name,value\nA001_LIG,1\nA002_LIG,2\nA003_LIG,3\n")
    completed = tmp_path / "run" / "predictions" / "A002_LIG"
    completed.mkdir(parents=True)
    (completed / "A002_LIG.sdf").write_text("pose\n$$$$\n")

    summary = write_recovery_shards(
        input_csv=input_csv,
        run_roots=[tmp_path / "run"],
        output_root=tmp_path / "recovery",
        num_shards=2,
    )

    assert summary["remaining_target_ids"] == ["A001_LIG", "A003_LIG"]
    emitted = "".join(
        path.read_text() for path in sorted((tmp_path / "recovery").rglob("*.csv"))
    )
    assert "A002_LIG" not in emitted


def test_vina_evaluator_reads_all_multirecord_sdf_poses(tmp_path: Path) -> None:
    sdf = tmp_path / "poses.sdf"
    molecule = Chem.MolFromSmiles("CCO")
    writer = Chem.SDWriter(str(sdf))
    for seed in (11, 12, 13):
        pose = Chem.AddHs(Chem.Mol(molecule))
        params = AllChem.ETKDGv3()
        params.randomSeed = seed
        assert AllChem.EmbedMolecule(pose, params) == 0
        writer.write(Chem.RemoveHs(pose))
    writer.close()

    records, errors, available = load_pose_records([sdf], "posebench_vina")

    assert available == 3
    assert errors == []
    assert [candidate_index for candidate_index, _, _ in records] == [0, 1, 2]


def test_interformer_records_follow_pose_score_not_native_rank(tmp_path: Path) -> None:
    sdf = tmp_path / "1abc_docked.sdf"
    writer = Chem.SDWriter(str(sdf))
    for x in (0.0, 1.0, 2.0):
        molecule = Chem.AddHs(Chem.MolFromSmiles("CC"))
        params = AllChem.ETKDGv3()
        params.randomSeed = 1
        assert AllChem.EmbedMolecule(molecule, params) == 0
        molecule.GetConformer().SetAtomPosition(0, (x, 0.0, 0.0))
        writer.write(Chem.RemoveHs(molecule))
    writer.close()
    scores = tmp_path / "query.round0_ensemble.csv"
    scores.write_text(
        "Target,pose_rank,pred_pose\n"
        "1abc,0,0.1\n"
        "1abc,1,0.9\n"
        "1abc,2,0.2\n"
    )

    records, errors, available = load_pose_records([sdf, scores], "interformer")

    assert errors == []
    assert available == 3
    assert "#record=1;" in str(records[0][1])


def test_ranked_pose_loader_does_not_promote_missing_rank1(tmp_path: Path) -> None:
    paths = []
    for rank in (2, 3):
        path = tmp_path / f"rank{rank}_ligand_lddt0.5_affinity1.0.sdf"
        molecule = Chem.AddHs(Chem.MolFromSmiles("CCO"))
        params = AllChem.ETKDGv3()
        params.randomSeed = rank
        assert AllChem.EmbedMolecule(molecule, params) == 0
        Chem.MolToMolFile(Chem.RemoveHs(molecule), str(path))
        paths.append(path)

    records, errors, available = load_pose_records(paths, "posebench_dynamicbind")

    assert available == 2
    assert errors == []
    assert [candidate_index for candidate_index, _, _ in records] == [1, 2]


def test_heavy_atom_rmsd_removes_stereo_defining_explicit_hydrogen() -> None:
    pose = Chem.MolFromSmiles("[H]/N=C(/N)c1ccccc1")
    params = AllChem.ETKDGv3()
    params.randomSeed = 17
    assert AllChem.EmbedMolecule(pose, params) == 0
    reference = Chem.RemoveAllHs(pose)

    rmsd, method = no_align_rmsd(pose, reference)

    assert rmsd == 0.0
    assert method.startswith("rdkit_calc_rms")


def test_heavy_atom_rmsd_selects_primary_ligand_from_cofactors() -> None:
    primary = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    cofactor = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1"))
    params = AllChem.ETKDGv3()
    params.randomSeed = 21
    assert AllChem.EmbedMolecule(primary, params) == 0
    params.randomSeed = 22
    assert AllChem.EmbedMolecule(cofactor, params) == 0
    primary = Chem.RemoveAllHs(primary)
    cofactor = Chem.RemoveAllHs(cofactor)
    combined = Chem.CombineMols(primary, cofactor)

    rmsd, method = no_align_rmsd(combined, primary)

    assert rmsd == 0.0
    assert method.endswith("primary_fragment_0")


def _pdb_atom(
    serial: int,
    atom: str,
    residue: int,
    x: float,
    y: float,
    z: float,
    element: str,
) -> str:
    return (
        f"ATOM  {serial:5d} {atom:^4s} ALA A{residue:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00          {element:>2s}"
    )


def test_predicted_receptor_can_use_native_pre_esm_pocket_crop(
    tmp_path: Path,
    monkeypatch,
) -> None:
    target_id = "TEST_LIG"
    predicted_dir = tmp_path / "predicted"
    predicted_dir.mkdir()
    protein = predicted_dir / f"{target_id}_holo_aligned_predicted_protein.pdb"
    protein.write_text(
        "\n".join(
            [
                _pdb_atom(1, "N", 1, -1.0, 0.0, 0.0, "N"),
                _pdb_atom(2, "CA", 1, 0.0, 0.0, 0.0, "C"),
                _pdb_atom(3, "C", 1, 1.0, 0.0, 0.0, "C"),
                _pdb_atom(4, "N", 2, 29.0, 0.0, 0.0, "N"),
                _pdb_atom(5, "CA", 2, 30.0, 0.0, 0.0, "C"),
                _pdb_atom(6, "C", 2, 31.0, 0.0, 0.0, "C"),
                "END",
            ]
        )
        + "\n"
    )
    ligand = tmp_path / "ligand.sdf"
    ligand.write_text(
        "ligand\n"
        "  EFF-Dock\n"
        "\n"
        "  1  0  0  0  0  0  0  0  0  0999 V2000\n"
        "    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n"
        "M  END\n"
        "$$$$\n"
    )
    source_csv = tmp_path / "source.csv"
    with source_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["complex_name", "experimental_protein", "ligand"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "complex_name": target_id,
                "experimental_protein": protein,
                "ligand": ligand,
            }
        )

    output_csv = tmp_path / "prepared" / "inputs.csv"
    protein_output_dir = tmp_path / "prepared" / "proteins"
    monkeypatch.setitem(
        prepare_diffdock_pocket_inputs.DATASET_TO_SIZE,
        "astex_diverse",
        1,
    )
    monkeypatch.syspath_prepend(str(ROOT / "scripts" / "external_models"))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_diffdock_pocket_inputs.py",
            "--source-csv",
            str(source_csv),
            "--dataset",
            "astex_diverse",
            "--receptor-mode",
            "holo_aligned_predicted_pocket_crop",
            "--predicted-protein-dir",
            str(predicted_dir),
            "--protein-output-dir",
            str(protein_output_dir),
            "--output-csv",
            str(output_csv),
        ],
    )

    prepare_diffdock_pocket_inputs.main()

    with output_csv.open(newline="") as handle:
        row = next(csv.DictReader(handle))
    cropped = Path(row["experimental_protein"])
    assert cropped.is_file()
    assert "A   1" in cropped.read_text()
    assert "A   2" not in cropped.read_text()
    provenance = json.loads(output_csv.with_suffix(".json").read_text())
    stats = provenance["protein_stats"][target_id]["pre_esm_pocket_crop"]
    assert stats["max_selected_chain_residues"] == 1
    assert provenance["receptor_mode"] == "holo_aligned_predicted_pocket_crop"


def test_external_install_recovery_uses_model_local_uv_projects() -> None:
    tools_root = ROOT / "benchmarks/external_models/tools"
    sync_script = (tools_root / "sync_model.sh").read_text()
    runner = (tools_root / "run_model.sh").read_text()

    assert "micromamba" not in sync_script
    assert 'model_root="$repo_root/benchmarks/external_models/environments/$model"' in sync_script
    assert 'UV_PROJECT_ENVIRONMENT="$model_root/.venv"' in sync_script
    assert 'uv sync --project "$model_root"' in sync_script
    assert 'uv run --project "$model_root" --no-sync' in runner
    assert 'sigmadock_cudnn_lib' in runner

    expected_python = {
        "sigmadock": "3.12",
        "surfdock": "3.10",
        "diffbindfr": "3.9",
        "interformer": "3.12",
    }
    for model, python_version in expected_python.items():
        project_root = ROOT / "benchmarks/external_models/environments" / model
        assert (project_root / "pyproject.toml").is_file()
        assert (project_root / ".python-version").read_text().strip() == python_version

    environments = ROOT / "benchmarks/external_models/environments"
    sigmadock = (environments / "sigmadock/pyproject.toml").read_text()
    surfdock = (environments / "surfdock/pyproject.toml").read_text()
    diffbindfr = (environments / "diffbindfr/pyproject.toml").read_text()
    interformer = (environments / "interformer/pyproject.toml").read_text()
    assert '"torch==2.13.0"' in sigmadock
    assert '"torchvision==0.28.0"' in sigmadock
    assert 'name = "pytorch-cu126"' in sigmadock
    assert "torch-2.2.0+cu121.html" in surfdock
    assert "pymesh2-0.3.1-cp310-cp310-linux_x86_64.whl" in surfdock
    assert '"ipython==8.37.0"' in surfdock
    assert '"setuptools==80.9.0"' in surfdock
    assert '"torch==1.13.1+cu117"' in diffbindfr
    assert "torch-1.13.1+cu117.html" in diffbindfr
    assert '"setuptools==80.9.0"' in diffbindfr
    assert '"torch==2.4.0"' in interformer

    interformer_wrapper = tools_root / "interformer_obrms.sh"
    interformer_bootstrap = tools_root / "bootstrap_interformer_native.py"
    assert interformer_wrapper.is_file()
    assert interformer_bootstrap.is_file()
    assert "obrms-lib" in interformer_wrapper.read_text()
    bootstrap_text = interformer_bootstrap.read_text()
    assert "libboost-headers-1.84.0-ha770c72_5.conda" in bootstrap_text
    assert "openbabel-3.1.1-py312hbfe4552_9.conda" in bootstrap_text
    assert "obrms-lib" in sync_script

    for model in expected_python:
        sbatch = (ROOT / f"scripts/slurm/external_{model}_inference.sbatch")
        assert f"others/{model}" in sbatch.read_text()

    surfdock_sbatch = (
        ROOT / "scripts/slurm/external_surfdock_inference.sbatch"
    ).read_text()
    interformer_sbatch = (
        ROOT / "scripts/slurm/external_interformer_inference.sbatch"
    ).read_text()
    assert ".cache/precomputed_arrays" in surfdock_sbatch
    assert "--normalize-db-markers" in interformer_sbatch

    sigmadock_sbatch = (
        ROOT / "scripts/slurm/external_sigmadock_inference.sbatch"
    ).read_text()
    assert "diffusion.num_steps=$INFERENCE_STEPS" in sigmadock_sbatch
    assert "run_sigmadock_sample.py" in sigmadock_sbatch
    assert "postprocessing.scoring=vinardo" in sigmadock_sbatch
    assert "postprocessing.bust_config=null" in sigmadock_sbatch


def test_sigmadock_parser_compat_removes_only_nonstandard_residue_links() -> None:
    residue = [
        ("N", -1.3, 0.0, 0.0, "N"),
        ("CA", 0.0, 0.0, 0.0, "C"),
        ("C", 1.5, 0.0, 0.0, "C"),
        ("O", 2.7, 0.0, 0.0, "O"),
        ("CB", 0.0, 1.5, 0.0, "C"),
    ]
    lines: list[str] = []
    serial = 1
    for residue_number, shift in ((1, 0.0), (100, 0.2)):
        for atom, x, y, z, element in residue:
            lines.append(
                _pdb_atom(
                    serial,
                    atom,
                    residue_number,
                    x + shift,
                    y,
                    z,
                    element,
                )
            )
            serial += 1
    pdb_block = "\n".join([*lines, "END"]) + "\n"

    assert Chem.MolFromPDBBlock(pdb_block, removeHs=True) is None
    recovered, removed = recover_nonstandard_inter_residue_bonds(pdb_block)
    assert recovered is not None
    assert recovered.GetNumAtoms() == 10
    assert removed > 0


def test_sigmadock_parser_compat_removes_whole_residue_without_ca() -> None:
    complete = [
        _pdb_atom(1, "N", 1, -1.0, 0.0, 0.0, "N"),
        _pdb_atom(2, "CA", 1, 0.0, 0.0, 0.0, "C"),
        _pdb_atom(3, "C", 1, 1.0, 0.0, 0.0, "C"),
    ]
    incomplete = [
        _pdb_atom(4, "N", 2, 2.3, 0.0, 0.0, "N"),
        _pdb_atom(5, "C", 2, 3.8, 0.0, 0.0, "C"),
        _pdb_atom(6, "O", 2, 5.0, 0.0, 0.0, "O"),
    ]
    hetero_without_ca = _pdb_atom(7, "ZN", 900, 8.0, 0.0, 0.0, "Zn").replace(
        "ATOM  ", "HETATM", 1
    )
    molecule = Chem.MolFromPDBBlock(
        "\n".join([*complete, *incomplete, hetero_without_ca, "END"]) + "\n",
        removeHs=True,
    )
    assert molecule is not None

    cleaned, removed_residues, removed_atoms = remove_residues_without_ca(molecule)

    assert removed_residues == 1
    assert removed_atoms == 3
    assert cleaned.GetNumAtoms() == 4
    assert {
        residue_number
        for atom in cleaned.GetAtoms()
        if (residue_number := atom.GetPDBResidueInfo().GetResidueNumber())
    } == {1, 900}
    assert any(
        atom.GetPDBResidueInfo().GetIsHeteroAtom() for atom in cleaned.GetAtoms()
    )


def test_sigmadock_parser_compat_restores_3d_double_bond_stereo() -> None:
    mol = Chem.AddHs(Chem.MolFromSmiles("CC=CC"))
    params = AllChem.ETKDGv3()
    params.randomSeed = 7
    assert AllChem.EmbedMolecule(mol, params) == 0
    mol = Chem.RemoveHs(mol)
    double_bond = next(
        bond
        for bond in mol.GetBonds()
        if bond.GetBondType() == Chem.BondType.DOUBLE
    )
    double_bond.SetStereo(Chem.BondStereo.STEREONONE)

    assert assign_missing_stereochemistry_from_3d(mol) == 1
    assert double_bond.GetStereo() != Chem.BondStereo.STEREONONE


def test_diffbindfr_chain_repair_is_python39_compatible(tmp_path: Path) -> None:
    source = tmp_path / "source.pdb"
    exported = tmp_path / "exported.pdb"
    source.write_text(
        _pdb_atom(1, "CA", 1, 0.0, 0.0, 0.0, "C") + "\nEND\n"
    )
    line = _pdb_atom(1, "CA", 1, 0.0, 0.0, 0.0, "C")
    exported.write_text(line[:21] + "AA" + line[22:] + "\nEND\n")

    repaired, audit = repair_pdb(source, exported)

    assert audit["repaired"] is True
    assert audit["chain_map"] == {"AA": "A"}
    assert repaired.read_text().splitlines()[0][21] == "A"
