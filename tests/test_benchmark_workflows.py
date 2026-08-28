from __future__ import annotations

import pytest
import torch
from rdkit import Chem

from effdock.evaluation.benchmark import compute_pose_rmsd, compute_pose_rmsd_with_method
from effdock.workflows.benchmark_report import EXPECTED_COUNTS
from effdock.workflows.evaluate import ComplexInput, shard_complexes, summarize_rows


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


def test_benchmark_aggregate_scope_is_astex_and_posebusters() -> None:
    assert EXPECTED_COUNTS == {"astex": 85, "posebusters": 308}


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


def test_shard_complexes_is_disjoint_and_complete() -> None:
    complexes = [ComplexInput(str(i), None, None, "sdf", None, (0.0, 0.0, 0.0)) for i in range(9)]
    shards = [shard_complexes(complexes, index, 4) for index in range(4)]
    observed = [item.complex_id for shard in shards for item in shard]
    assert sorted(observed, key=int) == [str(i) for i in range(9)]
    assert len(observed) == len(set(observed))


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
