"""Tests for the label-blind fragment-geometry audit."""

from __future__ import annotations

import math

import pytest
import torch
from rdkit import Chem

from effdock.evaluation.fragment_geometry import (
    enumerate_full_atom_mappings,
    fragment_rigid_fit_floor,
    partitions_equivalent,
)
from effdock.inference.preprocess import generate_smiles_conformer, load_ligand
from scripts.analyze_rdkit_fragment_geometry import _apply_hydrogen_policy


def _rotation_z(angle: float) -> torch.Tensor:
    return torch.tensor(
        [
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float64,
    )


def test_independent_fragment_rigid_motion_has_zero_floor() -> None:
    crystal = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [4.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [4.0, 0.0, 1.0],
        ],
        dtype=torch.float64,
    )
    fragment_id = torch.tensor([0, 0, 0, 1, 1, 1])
    inference = crystal.clone()
    inference[:3] = crystal[:3] @ _rotation_z(0.7).T + torch.tensor([8.0, -2.0, 1.0])
    inference[3:] = crystal[3:] @ _rotation_z(-1.1).T + torch.tensor([-3.0, 5.0, 2.0])

    result = fragment_rigid_fit_floor(
        crystal,
        inference,
        fragment_id,
        fragment_id,
        list(range(6)),
    )

    assert result["rigid_fragment_floor_rmsd"] == pytest.approx(0.0, abs=1e-7)
    assert result["pair_distance_rmse"] == pytest.approx(0.0, abs=1e-7)
    assert result["stored_partition_equal"] is True


def test_two_atom_bond_length_error_survives_rigid_fit() -> None:
    crystal = torch.tensor([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    inference = torch.tensor([[-1.5, 0.0, 0.0], [1.5, 0.0, 0.0]])
    fragment_id = torch.tensor([0, 0])

    result = fragment_rigid_fit_floor(
        crystal,
        inference,
        fragment_id,
        fragment_id,
        [0, 1],
    )

    assert result["rigid_fragment_floor_rmsd"] == pytest.approx(0.5)
    assert result["pair_distance_rmse"] == pytest.approx(1.0)
    assert result["fragments"][0]["fit_rank"] == 1
    assert result["fragments"][0]["orientation_observable"] is False


def test_single_atom_fragment_has_zero_floor_and_unobservable_rotation() -> None:
    crystal = torch.tensor([[2.0, -3.0, 4.0]])
    inference = torch.tensor([[-7.0, 8.0, 9.0]])
    fragment_id = torch.tensor([0])

    result = fragment_rigid_fit_floor(
        crystal,
        inference,
        fragment_id,
        fragment_id,
        [0],
    )

    assert result["rigid_fragment_floor_rmsd"] == pytest.approx(0.0)
    assert result["fragments"][0]["fit_rank"] == 0
    assert result["fragments"][0]["orientation_observable"] is False


def test_reflection_is_not_removed_by_proper_rotation() -> None:
    crystal = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    inference = crystal.clone()
    inference[:, 0] *= -1.0
    fragment_id = torch.zeros(4, dtype=torch.long)

    result = fragment_rigid_fit_floor(
        crystal,
        inference,
        fragment_id,
        fragment_id,
        [0, 1, 2, 3],
    )

    assert result["rigid_fragment_floor_rmsd"] > 0.1


def test_different_stored_partition_is_reported_but_not_dropped() -> None:
    coords = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]
    )
    stored_fragment_id = torch.tensor([0, 0, 1, 1])
    inference_fragment_id = torch.tensor([0, 0, 0, 1])

    result = fragment_rigid_fit_floor(
        coords,
        coords,
        stored_fragment_id,
        inference_fragment_id,
        [0, 1, 2, 3],
    )

    assert result["rigid_fragment_floor_rmsd"] == pytest.approx(0.0)
    assert result["stored_partition_equal"] is False
    assert result["fragment_count"] == 2


def test_fragment_labels_do_not_affect_partition_equivalence() -> None:
    assert partitions_equivalent(
        torch.tensor([0, 0, 1, 1]),
        torch.tensor([5, 5, 2, 2]),
    )
    assert not partitions_equivalent(
        torch.tensor([0, 0, 1, 1]),
        torch.tensor([0, 1, 1, 1]),
    )


def test_stereo_preserving_mapping_rejects_opposite_enantiomer() -> None:
    crystal = Chem.MolFromSmiles("F[C@](Cl)(Br)I")
    same = Chem.MolFromSmiles("F[C@](Cl)(Br)I")
    opposite = Chem.MolFromSmiles("F[C@@](Cl)(Br)I")
    assert crystal is not None and same is not None and opposite is not None

    mappings, method, _ = enumerate_full_atom_mappings(crystal, same)
    assert mappings
    assert method == "strict_stereo"

    mappings, method, _ = enumerate_full_atom_mappings(crystal, opposite)
    assert not mappings
    assert method.startswith("fallback_stereo_rejected:")


def test_symmetry_cap_is_reported_exactly() -> None:
    crystal = Chem.MolFromSmiles("c1ccccc1")
    inference = Chem.MolFromSmiles("c1ccccc1")
    assert crystal is not None and inference is not None

    mappings, method, truncated = enumerate_full_atom_mappings(
        crystal,
        inference,
        max_matches=1,
    )

    assert len(mappings) == 1
    assert method == "strict_stereo"
    assert truncated is True


def test_audited_smiles_helper_matches_public_loader() -> None:
    audited, metadata = generate_smiles_conformer("CCCO", random_seed=17)
    public, has_pose = load_ligand("CCCO", random_seed=17)

    assert not has_pose
    assert metadata["embed_status"] == 0
    assert isinstance(metadata["mmff_status"], int)
    torch.testing.assert_close(
        torch.as_tensor(audited.GetConformer().GetPositions()),
        torch.as_tensor(public.GetConformer().GetPositions()),
        rtol=0.0,
        atol=0.0,
    )


def test_heavy_only_policy_removes_stereo_defining_explicit_hydrogen() -> None:
    mol, _ = generate_smiles_conformer("C/C=N/[H]", random_seed=0)

    assert any(atom.GetAtomicNum() == 1 for atom in mol.GetAtoms())
    normalized = _apply_hydrogen_policy(mol, "remove_all_hs")

    assert normalized.GetNumAtoms() == 3
    assert all(atom.GetAtomicNum() != 1 for atom in normalized.GetAtoms())
    assert normalized.GetNumConformers() == 1
    assert normalized.GetConformer().Is3D()
