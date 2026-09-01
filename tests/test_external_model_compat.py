import csv
import json
import runpy
import sys
from pathlib import Path

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from scripts.external_models.aggregate_inference_coverage import main as aggregate_coverage
from scripts.external_models.posebench_vina_compat import install_rdkit_six_compat
from scripts.external_models.postprocess_rldiff_rlpp import prepare_smina_receptor
from scripts.external_models.prepare_posebench_vina_receptor import (
    meeko_output_prefix,
    select_meeko_receptor_source,
)
from scripts.external_models.repair_diffbindfr_output_pdb import repair_pdb
from scripts.external_models.run_posebench_dynamicbind import prepare_compatible_runner
from scripts.external_models.run_seeded_diffdock import (
    TargetFailureRecorder,
    _has_complete_output,
)
from scripts.external_models.surfdock_compat import (
    assign_missing_stereochemistry_from_3d,
    preserve_biopython_pdb_on_rdkit_failure,
)


def test_posebench_vina_compat_provides_legacy_string_io(monkeypatch) -> None:
    import io
    import sys

    monkeypatch.delitem(sys.modules, "rdkit.six", raising=False)

    assert install_rdkit_six_compat() is True
    assert sys.modules["rdkit.six"].StringIO is io.StringIO


def test_posebench_vina_receptor_adapter_preserves_requested_pdbqt_name(
    tmp_path: Path,
) -> None:
    output = tmp_path / "receptor.pdbqt"
    assert meeko_output_prefix(output) == tmp_path / "receptor"


def test_posebench_vina_receptor_adapter_recovers_original_posebench_temp_pdb(
    tmp_path: Path,
) -> None:
    original = tmp_path / "target.pdb"
    original.write_text("END\n")
    prepared = tmp_path / "target_reduced_prepped.pdb"
    prepared.write_text("END\n")

    assert select_meeko_receptor_source(prepared) == original


def _pdb_atom(serial: int, chain: str, x: float) -> str:
    return (
        f"ATOM  {serial:5d}  CA  ALA {chain}{1:4d}    "
        f"{x:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00 20.00           C  \n"
    )


def test_diffbindfr_chain_repair_restores_fixed_width_columns(tmp_path: Path) -> None:
    source = tmp_path / "source.pdb"
    source.write_text(_pdb_atom(1, "A", 1.0) + _pdb_atom(2, "a", 2.0) + "END\n")

    standard_a = _pdb_atom(1, "A", 1.0)
    standard_lower = _pdb_atom(2, "a", 2.0)
    malformed_aa = standard_lower[:21] + "AA" + standard_lower[22:]
    exported = tmp_path / "prot_final.pdb"
    exported.write_text(standard_a + malformed_aa + "END\n")

    compatible, audit = repair_pdb(source, exported)

    assert compatible != exported
    assert audit["repaired"] is True
    assert audit["chain_map"] == {"A": "A", "AA": "a"}
    repaired_atoms = [
        line for line in compatible.read_text().splitlines() if line.startswith("ATOM")
    ]
    assert [line[21] for line in repaired_atoms] == ["A", "a"]
    assert [float(line[30:38]) for line in repaired_atoms] == [1.0, 2.0]


def test_diffbindfr_chain_repair_allows_bounded_flexible_atom_omission(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.pdb"
    source.write_text(
        "".join(_pdb_atom(index, "a", float(index)) for index in range(1, 21))
        + "END\n"
    )
    standard = source.read_text().splitlines(keepends=True)
    # DiffBindFR may omit a small number of flexible side-chain atoms.  This
    # fixture omits one of twenty source atoms (95% identity overlap).
    malformed = [line[:21] + "AA" + line[22:] for line in standard[:19]]
    exported = tmp_path / "prot_final.pdb"
    exported.write_text("".join(malformed) + "END\n")

    compatible, audit = repair_pdb(source, exported)

    assert compatible.is_file()
    validation = audit["chain_validation"]["AA->a"]
    assert validation["source_atoms"] == 20
    assert validation["exported_atoms"] == 19
    assert validation["exported_signature_overlap"] == pytest.approx(1.0)
    assert validation["source_signature_coverage"] == pytest.approx(0.95)


def test_surfdock_preserves_existing_pdbio_fallback(tmp_path: Path) -> None:
    receptor = tmp_path / "pocket.pdb"
    receptor.write_text(_pdb_atom(1, "A", 1.0))

    def writer(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("writer must not be called for a missing RDKit molecule")

    assert preserve_biopython_pdb_on_rdkit_failure(None, str(receptor), writer) is None
    assert receptor.read_text().startswith("ATOM")

    with pytest.raises(ValueError, match="no PDBIO fallback"):
        preserve_biopython_pdb_on_rdkit_failure(
            None, str(tmp_path / "missing.pdb"), writer
        )


def test_surfdock_stereo_recovery_makes_polyene_embedding_reproducible() -> None:
    mol = Chem.AddHs(Chem.MolFromSmiles("CC=CC"))
    params = AllChem.ETKDGv3()
    params.randomSeed = 7
    assert AllChem.EmbedMolecule(mol, params) == 0
    mol = Chem.RemoveHs(mol)
    double_bond = next(
        bond for bond in mol.GetBonds() if bond.GetBondType() == Chem.BondType.DOUBLE
    )
    double_bond.SetStereo(Chem.BondStereo.STEREONONE)

    assert assign_missing_stereochemistry_from_3d(mol) == 1
    assert double_bond.GetStereo() != Chem.BondStereo.STEREONONE


def test_rldiff_smina_receptor_compat_removes_complete_unsupported_hetero_residue(
    tmp_path: Path,
) -> None:
    source = tmp_path / "receptor.pdb"
    source.write_text(
        _pdb_atom(1, "A", 1.0)
        + "HETATM    2  V   VO4 A 998       2.000   0.000   0.000  1.00 10.00           V  \n"
        + "HETATM    3  O1  VO4 A 998       3.000   0.000   0.000  1.00 10.00           O  \n"
        + "HETATM    4 ZN    ZN A 999       4.000   0.000   0.000  1.00 10.00          Zn  \n"
        + "END\n"
    )
    prepared = tmp_path / "prepared.pdb"

    audit = prepare_smina_receptor(source, prepared)

    text = prepared.read_text()
    assert "VO4" not in text
    assert " ZN " in text
    assert text.startswith("ATOM")
    assert audit["unsupported_elements"] == ["V"]
    assert audit["removed_atom_count"] == 2


def test_rldiff_smina_receptor_compat_fails_on_unsupported_protein_atom(
    tmp_path: Path,
) -> None:
    source = tmp_path / "receptor.pdb"
    source.write_text(
        "ATOM      1  V   ALA A   1       1.000   0.000   0.000  1.00 10.00           V  \n"
    )

    with pytest.raises(ValueError, match="ATOM record"):
        prepare_smina_receptor(source, tmp_path / "prepared.pdb")


def test_dynamicbind_compat_runner_uses_two_phase_rank_rename(tmp_path: Path) -> None:
    root = tmp_path / "DynamicBind"
    root.mkdir()
    source = '''import os
import uuid
from typing import Literal
def rename_files_by_confidence(directory_path, molecule_type: Literal["ligand", "receptor"] = "ligand"):
    os.rename("unsafe", "rank1")

def swap_dir_names(dir1, dir2):
    pass

file_path = os.path.realpath(__file__)
script_folder = os.path.dirname(file_path)
'''
    (root / "run_single_protein_inference.py").write_text(source)
    output = tmp_path / "output"
    (output / "workdirs").mkdir(parents=True)

    runner = prepare_compatible_runner(root, output)
    patched = runner.read_text()

    assert ".effdock_rank_tmp_" in patched
    assert 'os.rename("unsafe", "rank1")' not in patched
    assert f"script_folder = {str(root)!r}" in patched
    assert (output / "dynamicbind_compatibility.json").is_file()

    pose_dir = tmp_path / "poses"
    pose_dir.mkdir()
    low = pose_dir / "rank1_ligand_lddt0.10_affinity1.00.sdf"
    high = pose_dir / "rank2_ligand_lddt0.90_affinity2.00.sdf"
    low.write_text("low")
    high.write_text("high")
    namespace = runpy.run_path(str(runner))
    namespace["rename_files_by_confidence"](pose_dir)

    ranked = sorted(pose_dir.glob("rank*_ligand*.sdf"))
    assert len(ranked) == 2
    assert ranked[0].read_text() == "high"
    assert ranked[1].read_text() == "low"


def test_diffdock_failure_recorder_distinguishes_native_limit_from_oom() -> None:
    emitted: list[tuple[object, ...]] = []
    recorder = TargetFailureRecorder(lambda *values, **_kwargs: emitted.append(values))

    recorder("Skipping 7FRX_O88 because of the error:")
    recorder("The receptor is too large 3405")
    recorder("Failed on", ["7PJQ_OWH"], "CUDA driver error: out of memory")

    assert recorder.failures == {
        "7FRX_O88": {
            "kind": "native_unsupported",
            "message": "The receptor is too large 3405",
        },
        "7PJQ_OWH": {
            "kind": "inference_failure",
            "message": "CUDA driver error: out of memory",
        },
    }
    assert emitted[-1] == (
        "Failed on",
        ["7PJQ_OWH"],
        "CUDA driver error: out of memory",
    )


def test_diffdock_exact_target_skip_requires_every_expected_rank(tmp_path: Path) -> None:
    target_dir = tmp_path / "7PT3_3KK"
    target_dir.mkdir()
    for rank in range(1, 6):
        (target_dir / f"rank{rank}_confidence0.00.sdf").write_text("pose")
    # A truncated PDB-code directory must not satisfy the exact target check.
    (tmp_path / "7PT3").mkdir()

    assert _has_complete_output(tmp_path, "7PT3_3KK", 5) is True
    assert _has_complete_output(tmp_path, "7PT3", 5) is False
    (target_dir / "rank5_confidence0.00.sdf").unlink()
    assert _has_complete_output(tmp_path, "7PT3_3KK", 5) is False


def test_coverage_strict_mode_can_count_audited_native_limit(
    tmp_path: Path, monkeypatch
) -> None:
    expected = tmp_path / "expected.csv"
    with expected.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["complex_name"])
        writer.writeheader()
        writer.writerows([{"complex_name": "complete"}, {"complex_name": "large"}])
    shard = tmp_path / "runs" / "seed_0" / "shard_000"
    shard.mkdir(parents=True)
    (shard / "run_metadata.json").write_text(
        json.dumps({"samples_per_complex": 5})
    )
    (shard / "coverage.json").write_text(
        json.dumps(
            {
                "coverage": {
                    "complete": {"pose_count": 5, "status": "complete"},
                    "large": {"pose_count": 0, "status": "native_unsupported"},
                }
            }
        )
    )
    output = tmp_path / "aggregate.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "aggregate_inference_coverage.py",
            "--expected-csv",
            str(expected),
            "--run-root",
            str(tmp_path / "runs"),
            "--output-json",
            str(output),
            "--strict",
            "--allow-native-unsupported",
        ],
    )

    aggregate_coverage()

    payload = json.loads(output.read_text())
    assert payload["targets_with_expected_pose_count"] == 1
    assert payload["targets_native_unsupported"] == 1
    assert payload["targets_with_terminal_outcome"] == 2
