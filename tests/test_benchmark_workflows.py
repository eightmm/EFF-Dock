from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest
import torch
from rdkit import Chem

from effdock.evaluation.benchmark import compute_pose_rmsd, compute_pose_rmsd_with_method
from effdock.workflows import posebusters_report
from effdock.workflows.benchmark_data import strip_reference_ligand
from effdock.workflows.evaluate import (
    ComplexInput,
    require_complete_evaluation,
    require_expected_discovered_count,
    shard_complexes,
    summarize_rows,
)
from effdock.workflows.guidance_direct_drift_posebusters_report import (
    _require_input_hash_verification,
)
from effdock.workflows.posebusters_report import (
    EXPECTED_POSEBUSTERS_VERSION,
    require_complete_posebusters_run,
    require_posebusters_runtime_version,
    verify_posebusters_input_hashes,
)


def _atom_line(record: str, serial: int, atom: str, residue: str, number: int, xyz) -> str:
    x, y, z = xyz
    return (
        f"{record:<6}{serial:5d} {atom:^4s} {residue:>3s} A{number:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00           C  \n"
    )


def _molecule_with_conformer(
    smiles: str,
    coordinates: list[tuple[float, float, float]],
) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None and mol.GetNumAtoms() == len(coordinates)
    conformer = Chem.Conformer(mol.GetNumAtoms())
    for index, xyz in enumerate(coordinates):
        conformer.SetAtomPosition(index, xyz)
    mol.AddConformer(conformer)
    return mol


def test_pose_rmsd_reports_symmetry_aware_method() -> None:
    coordinates = [(0.0, 0.0, 0.0), (1.4, 0.0, 0.0)]
    mol_dock = _molecule_with_conformer("CC", coordinates)
    mol_ref = _molecule_with_conformer("CC", coordinates)
    pose = torch.tensor(coordinates, dtype=torch.float32)
    center = torch.zeros(3)

    value, method = compute_pose_rmsd_with_method(
        pose,
        pose,
        center,
        [0, 1],
        mol_dock,
        mol_ref,
    )

    assert value == pytest.approx(0.0, abs=1e-6)
    assert method == "rdkit_calc_rms_symmetry_no_align"
    assert compute_pose_rmsd(pose, pose, center, [0, 1], mol_dock, mol_ref) == pytest.approx(
        value
    )


def test_pose_rmsd_reports_mapped_index_fallback() -> None:
    coordinates = [(0.0, 0.0, 0.0), (1.4, 0.0, 0.0)]
    mol_dock = _molecule_with_conformer("CC", coordinates)
    mol_ref = _molecule_with_conformer("CC", coordinates)
    pose = torch.tensor(coordinates, dtype=torch.float32)
    mapped_ref = torch.tensor([[1.0, 0.0, 0.0]])

    value, method = compute_pose_rmsd_with_method(
        pose,
        mapped_ref,
        torch.zeros(3),
        [0],
        mol_dock,
        mol_ref,
    )

    assert value == pytest.approx(1.0)
    assert method == "mapped_index_fallback"


def test_strip_reference_ligand_handles_atom_encoded_peptide() -> None:
    ligand = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    pdb = "".join(
        [
            _atom_line("ATOM", 1, "C1", "LIG", 9, ligand[0]),
            _atom_line("ATOM", 2, "C2", "LIG", 9, ligand[1]),
            _atom_line("ATOM", 3, "C3", "LIG", 9, ligand[2]),
            _atom_line("ATOM", 4, "CA", "ALA", 10, [10.0, 0.0, 0.0]),
            "END\n",
        ]
    )
    cleaned, removed = strip_reference_ligand(pdb, ligand)
    assert removed == ["A:9:LIG"]
    assert "LIG" not in cleaned
    assert "ALA" in cleaned


def test_strip_reference_ligand_fails_without_coordinate_match() -> None:
    pdb = "".join(
        _atom_line("HETATM", i, f"C{i}", "LIG", 9, [10.0 + i, 0.0, 0.0]) for i in range(1, 4)
    )
    with pytest.raises(ValueError, match="no RCSB residue matched"):
        strip_reference_ligand(pdb, np.zeros((3, 3)))


def test_shard_complexes_is_disjoint_and_complete() -> None:
    complexes = [ComplexInput(str(i), None, None, "sdf", None, (0.0, 0.0, 0.0)) for i in range(9)]
    shards = [shard_complexes(complexes, index, 4) for index in range(4)]
    observed = [item.complex_id for shard in shards for item in shard]
    assert sorted(observed, key=int) == [str(i) for i in range(9)]
    assert len(observed) == len(set(observed))


def test_full_cohort_discovery_gate_rejects_missing_complex() -> None:
    require_expected_discovered_count(85, 85)
    with pytest.raises(RuntimeError, match="expected 85 complexes, discovered 84"):
        require_expected_discovered_count(84, 85)


@pytest.mark.parametrize(
    "gate, label",
    [
        (require_complete_evaluation, "benchmark shard"),
        (require_complete_posebusters_run, "PoseBusters shard"),
    ],
)
def test_shard_completion_gates_reject_recorded_failures(gate, label: str) -> None:
    gate(num_assigned=2, num_success=2, failures=[])
    with pytest.raises(RuntimeError, match=label):
        gate(
            num_assigned=2,
            num_success=1,
            failures=[{"id": "missing_case", "error": "boom"}],
        )


def test_posebusters_input_hash_gate_detects_saved_pose_drift(tmp_path) -> None:
    protein = tmp_path / "protein.pdb"
    ligand = tmp_path / "ligand.sdf"
    pose = tmp_path / "pose.sdf"
    protein.write_text("protein")
    ligand.write_text("ligand")
    pose.write_text("pose")

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    row = {
        "id": "case",
        "protein": str(protein),
        "ligand_ref": str(ligand),
        "protein_sha256": digest(protein),
        "ligand_reference_sha256": digest(ligand),
        "saved_pose_sha256_json": json.dumps({"oracle": digest(pose)}),
    }
    verify_posebusters_input_hashes(row, pose, "oracle")
    pose.write_text("drifted")
    with pytest.raises(ValueError, match="selected pose changed after sampling"):
        verify_posebusters_input_hashes(row, pose, "oracle")


def test_posebusters_runtime_version_is_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert require_posebusters_runtime_version() == EXPECTED_POSEBUSTERS_VERSION
    monkeypatch.setattr(posebusters_report, "distribution_version", lambda _: "0.6.6")
    with pytest.raises(RuntimeError, match="runtime version mismatch"):
        require_posebusters_runtime_version()


def test_direct_official_report_requires_hash_verified_shards(tmp_path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    summary_path = run_dir / "shard-000-of-001.summary.json"
    summary = {
        "input_hashes_verified": False,
        "num_input_hashes_verified": 0,
        "num_assigned": 1,
    }
    summary_path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="sampling-time input hashes not verified"):
        _require_input_hash_verification(tmp_path, {"run"}, 1)

    summary["input_hashes_verified"] = True
    summary["num_input_hashes_verified"] = 1
    summary_path.write_text(json.dumps(summary))
    _require_input_hash_verification(tmp_path, {"run"}, 1)


def test_summarize_rows_keeps_selector_metrics_separate() -> None:
    rows = [
        {
            "first_rmsd": 3.0,
            "vina_rmsd": 1.5,
            "oracle_rmsd": 0.8,
            "confidence_rmsd": 1.2,
            "confidence_final_rmsd": 1.1,
            "first_fast_valid": False,
            "vina_fast_valid": True,
            "oracle_fast_valid": True,
            "confidence_fast_valid": True,
            "confidence_final_fast_valid": True,
        },
        {
            "first_rmsd": 1.0,
            "vina_rmsd": 2.5,
            "oracle_rmsd": 1.0,
            "confidence_rmsd": 2.2,
            "confidence_final_rmsd": 1.8,
            "first_fast_valid": True,
            "vina_fast_valid": False,
            "oracle_fast_valid": True,
            "confidence_fast_valid": False,
            "confidence_final_fast_valid": True,
        },
    ]
    stats = summarize_rows(rows)
    assert stats["first"]["pct_lt_2A"] == 50.0
    assert stats["vina"]["pct_lt_2A"] == 50.0
    assert stats["oracle"]["pct_lt_2A"] == 100.0
    assert stats["confidence"]["pct_lt_2A"] == 50.0
    assert stats["confidence_final"]["pct_lt_2A"] == 100.0
    assert stats["oracle"]["fast_valid_pct"] == 100.0


def test_summarize_rows_reports_candidate_set_joint_oracle() -> None:
    rows = [
        {
            "first_rmsd": 3.0,
            "oracle_rmsd": 0.8,
            "first_fast_valid": False,
            "oracle_fast_valid": False,
            "num_fast_valid_candidates": 2,
            "num_rmsd_lt2_candidates": 3,
            "num_fast_valid_rmsd_lt2_candidates": 1,
            "fast_valid_oracle_rmsd": 1.2,
            "joint_fast_valid_and_rmsd_lt2": True,
        },
        {
            "first_rmsd": 4.0,
            "oracle_rmsd": 1.0,
            "first_fast_valid": False,
            "oracle_fast_valid": False,
            "num_fast_valid_candidates": 0,
            "num_rmsd_lt2_candidates": 0,
            "num_fast_valid_rmsd_lt2_candidates": 0,
            "fast_valid_oracle_rmsd": float("inf"),
            "joint_fast_valid_and_rmsd_lt2": False,
        },
    ]
    stats = summarize_rows(rows)
    candidate = stats["candidate_set"]
    assert candidate["any_fast_valid_pct"] == 50.0
    assert candidate["mean_fast_valid_candidates"] == 1.0
    assert candidate["joint_fast_valid_and_rmsd_lt2_pct"] == 50.0
    assert candidate["fast_valid_oracle_coverage_pct"] == 50.0
    assert candidate["fast_valid_oracle_median_rmsd"] == 1.2
    assert candidate["mean_rmsd_lt2_candidates"] == 1.5
    assert candidate["complexes_with_any_rmsd_lt2_pct"] == 50.0
    assert candidate["mean_fast_valid_rmsd_lt2_candidates"] == 0.5
    assert candidate["complexes_with_any_fast_valid_rmsd_lt2_pct"] == 50.0
