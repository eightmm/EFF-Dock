import csv
import json
import sys
from pathlib import Path

from scripts.external_models import prepare_diffdock_pocket_inputs


ROOT = Path(__file__).resolve().parents[1]


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
    sync_script = (ROOT / "scripts/others/sync_model.sh").read_text()
    runner = (ROOT / "scripts/others/run_model.sh").read_text()

    assert "micromamba" not in sync_script
    assert 'model_root="$repo_root/others/$model"' in sync_script
    assert 'UV_PROJECT_ENVIRONMENT="$model_root/.venv"' in sync_script
    assert 'uv sync --project "$model_root"' in sync_script
    assert 'uv run --project "$model_root" --no-sync' in runner

    expected_python = {
        "surfdock": "3.10",
        "diffbindfr": "3.9",
        "interformer": "3.12",
    }
    for model, python_version in expected_python.items():
        project_root = ROOT / "others" / model
        assert (project_root / "pyproject.toml").is_file()
        assert (project_root / ".python-version").read_text().strip() == python_version

    surfdock = (ROOT / "others/surfdock/pyproject.toml").read_text()
    diffbindfr = (ROOT / "others/diffbindfr/pyproject.toml").read_text()
    interformer = (ROOT / "others/interformer/pyproject.toml").read_text()
    assert "torch-2.2.0+cu121.html" in surfdock
    assert "pymesh2-0.3.1-cp310-cp310-linux_x86_64.whl" in surfdock
    assert '"ipython==8.37.0"' in surfdock
    assert '"setuptools==80.9.0"' in surfdock
    assert '"torch==1.13.1+cu117"' in diffbindfr
    assert "torch-1.13.1+cu117.html" in diffbindfr
    assert '"setuptools==80.9.0"' in diffbindfr
    assert '"torch==2.4.0"' in interformer

    interformer_wrapper = ROOT / "scripts/others/interformer_obrms.sh"
    interformer_bootstrap = ROOT / "scripts/others/bootstrap_interformer_native.py"
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
