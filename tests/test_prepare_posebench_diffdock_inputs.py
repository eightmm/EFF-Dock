import csv
from pathlib import Path

from rdkit import Chem

from scripts.external_models.prepare_posebench_diffdock_inputs import localize_manifest


def test_reference_sdf_mode_uses_documented_file_input(tmp_path: Path) -> None:
    posebench = tmp_path / "posebench"
    data = tmp_path / "data"
    source = posebench / "forks/DiffDock/inference/diffdock_astex_diverse_inputs.csv"
    source.parent.mkdir(parents=True)
    rows = [
        {
            "complex_name": f"T{i:03d}_LIG",
            "protein_path": f"/released/T{i:03d}_protein.pdb",
            "ligand_description": "CCO",
            "protein_sequence": "AAA",
        }
        for i in range(85)
    ]
    with source.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    protein_root = data / "astex_diverse_set/astex_diverse_holo_aligned_predicted_structures"
    protein_root.mkdir(parents=True)
    target = rows[0]["complex_name"]
    (protein_root / "T000_protein.pdb").write_text("END\n")
    ligand = data / f"astex_diverse_set/{target}/{target}_ligand.sdf"
    ligand.parent.mkdir(parents=True)
    ligand.write_text("ligand\n")

    localized, _ = localize_manifest(
        posebench_root=posebench,
        data_root=data,
        dataset="astex_diverse",
        target_ids=[target],
        ligand_description_source="reference_sdf",
    )

    assert localized[0]["ligand_description"] == str(ligand.resolve())


def test_reference_smiles_mode_uses_frozen_primary_ligand(tmp_path: Path) -> None:
    posebench = tmp_path / "posebench"
    data = tmp_path / "data"
    source = posebench / "forks/DiffDock/inference/diffdock_astex_diverse_inputs.csv"
    source.parent.mkdir(parents=True)
    rows = [
        {
            "complex_name": f"T{i:03d}_LIG",
            "protein_path": f"/released/T{i:03d}_protein.pdb",
            "ligand_description": "CC.[Fe]",
            "protein_sequence": "AAA",
        }
        for i in range(85)
    ]
    with source.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    protein_root = data / "astex_diverse_set/astex_diverse_holo_aligned_predicted_structures"
    protein_root.mkdir(parents=True)
    target = rows[0]["complex_name"]
    (protein_root / "T000_protein.pdb").write_text("END\n")
    ligand = data / f"astex_diverse_set/{target}/{target}_ligand.sdf"
    ligand.parent.mkdir(parents=True)
    writer = Chem.SDWriter(str(ligand))
    writer.write(Chem.AddHs(Chem.MolFromSmiles("C[C@H](O)F")))
    writer.close()

    localized, _ = localize_manifest(
        posebench_root=posebench,
        data_root=data,
        dataset="astex_diverse",
        target_ids=[target],
        ligand_description_source="reference_smiles",
    )

    parsed = Chem.MolFromSmiles(localized[0]["ligand_description"])
    assert parsed is not None
    assert len(Chem.GetMolFrags(parsed)) == 1
    assert "Fe" not in localized[0]["ligand_description"]
