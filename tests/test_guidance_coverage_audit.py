from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from rdkit import Chem

from effdock.guidance.errors import UnsupportedPhysicalChemistryError
from effdock.workflows.benchmark_inputs import BenchmarkInputMismatchError
from effdock.workflows.evaluate import (
    build_arg_parser as build_evaluate_parser,
)
from effdock.workflows.evaluate import (
    receptor_guidance_metadata,
    serialize_evaluation_failure,
)
from effdock.workflows.guidance_coverage_audit import (
    AUDIT_SCHEMA_VERSION,
    _crystal_coords,
    build_arg_parser,
    build_audit_report,
    merge_audit_reports,
)

PROTEIN_PDB = """\
ATOM      1  N   ALA A   1       8.800  10.000  10.000  1.00 20.00           N
ATOM      2  CA  ALA A   1      10.000  10.000  10.000  1.00 20.00           C
ATOM      3  C   ALA A   1      11.200  10.000  10.000  1.00 20.00           C
ATOM      4  O   ALA A   1      12.000  10.800  10.000  1.00 20.00           O
ATOM      5  CB  ALA A   1      10.000   8.500  10.000  1.00 20.00           C
ATOM      6  N   GLY A   2      11.500   8.800  10.000  1.00 20.00           N
ATOM      7  CA  GLY A   2      12.700   8.800  10.000  1.00 20.00           C
ATOM      8  C   GLY A   2      13.500   9.900  10.000  1.00 20.00           C
ATOM      9  O   GLY A   2      14.700   9.800  10.000  1.00 20.00           O
END
"""

SO4_LINE = "HETATM   10  O1  SO4 B 101      10.000  12.500  10.000  1.00 20.00           O\n"


def _write_reference_ligand(path: Path) -> None:
    mol = Chem.MolFromSmiles("CCO")
    conformer = Chem.Conformer(mol.GetNumAtoms())
    for index, xyz in enumerate(([9.0, 11.0, 10.0], [10.4, 11.0, 10.0], [11.3, 11.8, 10.0])):
        conformer.SetAtomPosition(index, xyz)
    conformer.Set3D(True)
    mol.AddConformer(conformer)
    writer = Chem.SDWriter(str(path))
    writer.write(mol)
    writer.close()


def _tiny_dataset(tmp_path: Path) -> tuple[Path, Path, Path]:
    data_dir = tmp_path / "astex"
    external_dir = tmp_path / "external"
    data_dir.mkdir()
    external_dir.mkdir()
    centers = {}
    smiles = {}
    for complex_id, has_obstacle in (("clean", False), ("cofactor", True)):
        complex_dir = data_dir / complex_id
        complex_dir.mkdir()
        pdb = PROTEIN_PDB
        if has_obstacle:
            pdb = pdb.replace("END\n", SO4_LINE + "END\n")
        (complex_dir / f"{complex_id}_protein.pdb").write_text(pdb)
        _write_reference_ligand(complex_dir / f"{complex_id}_ligand.sdf")
        centers[complex_id] = [10.0, 10.0, 10.0]
        smiles[complex_id] = {"smiles": "CCO"}
    center_path = tmp_path / "centers.json"
    center_path.write_text(json.dumps(centers))
    (external_dir / "astex_smiles.json").write_text(json.dumps(smiles))
    return data_dir, external_dir, center_path


def test_evaluator_receptor_policy_defaults_to_fail_closed() -> None:
    parser = build_evaluate_parser()
    args = parser.parse_args(
        [
            "--dataset",
            "astex",
            "--data-dir",
            "data",
            "--pocket-centers",
            "centers.json",
        ]
    )
    assert args.unified_guidance_receptor_policy == "fail_closed"
    assert args.benchmark_input_manifest is None
    assert args.require_full_ligand_atom_mapping is False

    args = parser.parse_args(
        [
            "--dataset",
            "astex",
            "--data-dir",
            "data",
            "--pocket-centers",
            "centers.json",
            "--unified-guidance-receptor-policy",
            "geometry_only",
        ]
    )
    assert args.unified_guidance_receptor_policy == "geometry_only"


def test_evaluator_failure_preserves_repr_and_structured_chemistry() -> None:
    error = UnsupportedPhysicalChemistryError(
        "unsupported_test_site",
        "site cannot be typed",
        details={"element": "X"},
    )
    failure = serialize_evaluation_failure("tiny", error)
    assert failure["error"] == repr(error)
    assert failure["code"] == "unsupported_test_site"
    assert failure["message"] == "site cannot be typed"
    assert failure["details"] == {"element": "X"}
    assert failure["unsupported_physical_chemistry"] == error.as_dict()


def test_receptor_metadata_preserves_returned_identity_and_provenance() -> None:
    system = SimpleNamespace(
        receptor_policy="descriptive policy",
        receptor_policy_identity={"name": "geometry_only", "sha256": "abc123"},
        receptor_provenance={"metal_fallbacks": [{"code": "unsupported_metal"}]},
    )
    metadata = receptor_guidance_metadata(system, "geometry_only")
    assert metadata["mode"] == "geometry_only"
    assert metadata["identity_sha256"] == "abc123"
    assert metadata["identity"]["name"] == "geometry_only"
    assert metadata["provenance"]["metal_fallbacks"][0]["code"] == "unsupported_metal"


def test_tiny_audit_uses_full_discovery_and_crystal_numerical_gate(tmp_path: Path) -> None:
    data_dir, external_dir, center_path = _tiny_dataset(tmp_path)
    args = build_arg_parser().parse_args(
        [
            "--dataset",
            "astex",
            "--data-dir",
            str(data_dir),
            "--external-dir",
            str(external_dir),
            "--pocket-centers",
            str(center_path),
            "--receptor-policy",
            "geometry_only",
            "--output",
            str(tmp_path / "audit.json"),
        ]
    )
    report = build_audit_report(args)
    dataset = report["datasets"]["astex"]

    assert report["schema_version"] == AUDIT_SCHEMA_VERSION
    assert len(report["implementation"]["sha256"]) == 64
    assert len(report["receptor_policy_identity"]["sha256"]) == 64
    assert report["information_boundary"]["coefficient_tuning_allowed"] is False
    assert dataset["complete"] is True
    assert dataset["ids"] == ["clean", "cofactor"]
    assert dataset["success"] == 2
    assert dataset["failed"] == 0
    assert dataset["chemistry_slices"]["strict_supported"]["ids"] == ["clean"]
    assert dataset["chemistry_slices"]["nonprotein_only"]["ids"] == ["cofactor"]
    assert dataset["ligand_representation_slices"]["exact_graph"]["ids"] == [
        "clean",
        "cofactor",
    ]
    assert (
        dataset["ligand_representation_slices"][
            "same_connectivity_representation_mismatch"
        ]["count"]
        == 0
    )
    assert dataset["strict_supported_equivalence"]["passed"] is True
    assert dataset["complexes"]["clean"]["seed"] == 43
    assert dataset["complexes"]["cofactor"]["seed"] == 44
    assert dataset["complexes"]["cofactor"]["receptor"]["obstacle_count"] == 1
    numerical = dataset["complexes"]["cofactor"]["crystal_numerical_preflight"]
    assert numerical["energy_finite"] is True
    assert numerical["gradient_finite"] is True
    assert "total" in numerical["energies"]
    assert "total" in numerical["max_abs_gradient"]


def test_partial_reference_mapping_fails_closed(tmp_path: Path) -> None:
    reference = Chem.MolFromSmiles("CCO")
    reference_conformer = Chem.Conformer(reference.GetNumAtoms())
    reference_conformer.SetAtomPosition(0, [5.0, 5.0, 5.0])
    reference_conformer.SetAtomPosition(1, [6.4, 5.0, 5.0])
    reference_conformer.SetAtomPosition(2, [6.4, 6.2, 5.0])
    reference_conformer.Set3D(True)
    reference.AddConformer(reference_conformer)
    reference_path = tmp_path / "partial_reference.sdf"
    writer = Chem.SDWriter(str(reference_path))
    writer.write(reference)
    writer.close()

    inference = Chem.MolFromSmiles("CCCO")
    inference_conformer = Chem.Conformer(inference.GetNumAtoms())
    inference_conformer.SetAtomPosition(0, [0.0, -1.4, 0.0])
    inference_conformer.SetAtomPosition(1, [0.0, 0.0, 0.0])
    inference_conformer.SetAtomPosition(2, [0.0, 1.4, 0.0])
    inference_conformer.SetAtomPosition(3, [1.2, 1.4, 0.0])
    inference_conformer.Set3D(True)
    inference.AddConformer(inference_conformer)
    item = SimpleNamespace(ligand_ref=reference_path, ligand_format="sdf")

    with pytest.raises(BenchmarkInputMismatchError) as captured:
        _crystal_coords(
            item,
            inference,
            torch.tensor([5.0, 5.0, 5.0]),
        )
    assert captured.value.code == "benchmark_ligand_atom_mapping_mismatch"
    assert captured.value.details["matched_atoms"] == 3
    assert captured.value.details["input_atoms"] == 4
    assert captured.value.details["reference_atoms"] == 3


def test_reference_with_extra_atom_also_fails_closed(tmp_path: Path) -> None:
    reference = Chem.MolFromSmiles("CCCO")
    reference_conformer = Chem.Conformer(reference.GetNumAtoms())
    for index in range(reference.GetNumAtoms()):
        reference_conformer.SetAtomPosition(index, [float(index), 0.0, 0.0])
    reference_conformer.Set3D(True)
    reference.AddConformer(reference_conformer)
    reference_path = tmp_path / "reference_extra.sdf"
    writer = Chem.SDWriter(str(reference_path))
    writer.write(reference)
    writer.close()

    inference = Chem.MolFromSmiles("CCO")
    inference_conformer = Chem.Conformer(inference.GetNumAtoms())
    for index in range(inference.GetNumAtoms()):
        inference_conformer.SetAtomPosition(index, [float(index), 0.0, 0.0])
    inference_conformer.Set3D(True)
    inference.AddConformer(inference_conformer)
    item = SimpleNamespace(ligand_ref=reference_path, ligand_format="sdf")

    with pytest.raises(BenchmarkInputMismatchError) as captured:
        _crystal_coords(item, inference, torch.zeros(3))
    assert captured.value.code == "benchmark_ligand_atom_mapping_mismatch"
    assert captured.value.details["matched_atoms"] == 3
    assert captured.value.details["input_atoms"] == 3
    assert captured.value.details["reference_atoms"] == 4
    assert captured.value.details["full_bijection"] is False


def test_merge_audits_keeps_dataset_native_payloads(tmp_path: Path) -> None:
    paths = []
    for dataset in ("astex", "posebusters"):
        path = tmp_path / f"{dataset}.json"
        path.write_text(
            json.dumps(
                {
                    "schema_version": AUDIT_SCHEMA_VERSION,
                    "protocol_id": "test",
                    "receptor_policy": "geometry_only",
                    "implementation": {"sha256": "implementation"},
                    "parameter_set": {"sha256": "parameters"},
                    "receptor_policy_identity": {"sha256": "policy"},
                    "datasets": {dataset: {"discovered": 1, "ids": [dataset]}},
                }
            )
        )
        paths.append(path)
    merged = merge_audit_reports(paths, "combined")
    assert sorted(merged["datasets"]) == ["astex", "posebusters"]
    assert merged["protocol_id"] == "combined"
    assert merged["receptor_policy"] == "geometry_only"
    assert merged["implementation"]["sha256"] == "implementation"
    assert merged["parameter_set"]["sha256"] == "parameters"
    assert merged["receptor_policy_identity"]["sha256"] == "policy"

    drifted = json.loads(paths[1].read_text())
    drifted["parameter_set"]["sha256"] = "different"
    paths[1].write_text(json.dumps(drifted))
    with pytest.raises(ValueError, match="different parameter_set"):
        merge_audit_reports(paths, "combined")
