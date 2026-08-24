"""Rigid-fragment geometry comparison utilities.

The docking model may translate and rotate every fragment independently, but
it cannot deform the coordinates inside a fragment.  This module therefore
measures only the residual that remains after an independent proper rigid fit
of every inference fragment onto its crystal counterpart.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from rdkit import Chem
from torch import Tensor

from effdock.evaluation.benchmark import match_atoms
from effdock.preprocess.fragments import decompose_fragments
from effdock.workflows.benchmark_inputs import full_heavy_atom_mapping_metadata


class FragmentPartitionMismatchError(ValueError):
    """Raised when crystal and inference molecules do not have the same fragments."""


def _stereochemistry_compatible(
    mol_crystal: Chem.Mol,
    mol_inference: Chem.Mol,
    inference_to_crystal: Sequence[int],
) -> bool:
    crystal = Chem.Mol(mol_crystal)
    inference = Chem.Mol(mol_inference)
    Chem.AssignStereochemistry(crystal, cleanIt=True, force=True)
    Chem.AssignStereochemistry(inference, cleanIt=True, force=True)
    mapping = tuple(int(index) for index in inference_to_crystal)

    for inference_index, crystal_index in enumerate(mapping):
        input_atom = inference.GetAtomWithIdx(inference_index)
        crystal_atom = crystal.GetAtomWithIdx(crystal_index)
        if input_atom.GetIsotope() != crystal_atom.GetIsotope():
            return False
        input_cip = input_atom.GetProp("_CIPCode") if input_atom.HasProp("_CIPCode") else None
        crystal_cip = (
            crystal_atom.GetProp("_CIPCode") if crystal_atom.HasProp("_CIPCode") else None
        )
        if input_cip != crystal_cip:
            return False

    specified = {
        Chem.rdchem.BondStereo.STEREOZ,
        Chem.rdchem.BondStereo.STEREOE,
        Chem.rdchem.BondStereo.STEREOCIS,
        Chem.rdchem.BondStereo.STEREOTRANS,
    }
    for input_bond in inference.GetBonds():
        crystal_bond = crystal.GetBondBetweenAtoms(
            mapping[input_bond.GetBeginAtomIdx()],
            mapping[input_bond.GetEndAtomIdx()],
        )
        if crystal_bond is None:
            return False
        input_stereo = input_bond.GetStereo()
        crystal_stereo = crystal_bond.GetStereo()
        if (input_stereo in specified or crystal_stereo in specified) and (
            input_stereo != crystal_stereo
        ):
            return False
    return True


def _canonical_partition(fragment_id: Tensor) -> frozenset[frozenset[int]]:
    fragment_id = torch.as_tensor(fragment_id, dtype=torch.long).view(-1)
    return frozenset(
        frozenset((fragment_id == int(fragment)).nonzero(as_tuple=True)[0].tolist())
        for fragment in torch.unique(fragment_id).tolist()
    )


def enumerate_full_atom_mappings(
    mol_crystal: Chem.Mol,
    mol_inference: Chem.Mol,
    *,
    max_matches: int = 1024,
) -> tuple[list[tuple[int, ...]], str, bool]:
    """Enumerate full input-to-crystal heavy-atom mappings.

    Exact stereo-preserving RDKit graph matches are preferred and enumerate
    valid chemical symmetries.  When representations differ, the established
    benchmark mapper supplies one fallback which is accepted only after a full
    element- and connectivity-preserving bijection check.  We deliberately do
    not enumerate element-only graph automorphisms because those can swap atoms
    that are distinguished by bond order, charge, or stereochemistry and make
    the geometry floor spuriously optimistic.
    """
    if max_matches <= 0:
        raise ValueError("max_matches must be positive")
    if mol_crystal.GetNumAtoms() != mol_inference.GetNumAtoms():
        return [], "atom_count_mismatch", False

    strict = mol_crystal.GetSubstructMatches(
        mol_inference,
        uniquify=False,
        maxMatches=max_matches + 1,
        useChirality=True,
    )
    strict = [tuple(int(index) for index in match) for match in strict]
    if strict:
        return strict[:max_matches], "strict_stereo", len(strict) > max_matches

    inference_indices, crystal_indices, method = match_atoms(
        mol_crystal,
        mol_inference,
    )
    metadata = full_heavy_atom_mapping_metadata(
        mol_crystal,
        mol_inference,
        inference_indices,
        crystal_indices,
        method,
    )
    if not bool(metadata["accepted"]):
        return [], f"fallback_rejected:{method}", False
    mapping_by_input = dict(zip(inference_indices, crystal_indices, strict=True))
    mapping = tuple(mapping_by_input[index] for index in range(mol_inference.GetNumAtoms()))
    if not _stereochemistry_compatible(mol_crystal, mol_inference, mapping):
        return [], f"fallback_stereo_rejected:{method}", False
    return [mapping], f"fallback_full:{method}:{metadata['relation']}", False


def _proper_rigid_fit(moving: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:
    moving = torch.as_tensor(moving, dtype=torch.float64)
    target = torch.as_tensor(target, dtype=torch.float64)
    if moving.shape != target.shape or moving.ndim != 2 or moving.shape[1] != 3:
        raise ValueError("moving and target coordinates must both have shape [N, 3]")
    if moving.shape[0] == 0:
        raise ValueError("a fragment must contain at least one atom")
    if not bool(torch.isfinite(moving).all() and torch.isfinite(target).all()):
        raise ValueError("fragment coordinates must be finite")

    moving_centered = moving - moving.mean(dim=0, keepdim=True)
    target_centered = target - target.mean(dim=0, keepdim=True)
    covariance = moving_centered.T @ target_centered
    u, _, vh = torch.linalg.svd(covariance)
    rotation = vh.T @ u.T
    if float(torch.linalg.det(rotation)) < 0.0:
        correction = torch.eye(3, dtype=torch.float64)
        correction[-1, -1] = -1.0
        rotation = vh.T @ correction @ u.T
    residual = moving_centered @ rotation.T - target_centered
    return rotation, residual


def _pair_distance_squared_error(moving: Tensor, target: Tensor) -> tuple[float, int]:
    if moving.shape[0] < 2:
        return 0.0, 0
    difference = torch.pdist(moving.to(torch.float64)) - torch.pdist(target.to(torch.float64))
    return float(difference.square().sum().item()), int(difference.numel())


def fragment_rigid_fit_floor(
    crystal_coords: Tensor,
    inference_coords: Tensor,
    crystal_fragment_id: Tensor,
    inference_fragment_id: Tensor,
    inference_to_crystal: Sequence[int],
) -> dict[str, object]:
    """Return the optimistic RMSD floor under independent fragment SE(3) fits."""
    crystal_coords = torch.as_tensor(crystal_coords, dtype=torch.float64)
    inference_coords = torch.as_tensor(inference_coords, dtype=torch.float64)
    crystal_fragment_id = torch.as_tensor(crystal_fragment_id, dtype=torch.long).view(-1)
    inference_fragment_id = torch.as_tensor(inference_fragment_id, dtype=torch.long).view(-1)
    mapping = torch.as_tensor(inference_to_crystal, dtype=torch.long).view(-1)

    n_crystal = int(crystal_coords.shape[0])
    n_inference = int(inference_coords.shape[0])
    if crystal_coords.shape != (n_crystal, 3) or inference_coords.shape != (n_inference, 3):
        raise ValueError("coordinate tensors must have shape [N, 3]")
    if crystal_fragment_id.numel() != n_crystal:
        raise ValueError("crystal fragment IDs must match crystal atom count")
    if inference_fragment_id.numel() != n_inference or mapping.numel() != n_inference:
        raise ValueError("inference fragment IDs and mapping must match inference atom count")
    if n_crystal != n_inference or sorted(mapping.tolist()) != list(range(n_crystal)):
        raise ValueError("inference_to_crystal must be a full atom bijection")

    mapped_inference_partition = torch.empty_like(inference_fragment_id)
    mapped_inference_partition[mapping] = inference_fragment_id
    stored_partition_equal = _canonical_partition(crystal_fragment_id) == _canonical_partition(
        mapped_inference_partition
    )

    total_squared_error = 0.0
    total_pair_squared_error = 0.0
    total_pairs = 0
    fragment_records: list[dict[str, object]] = []
    for inference_fragment in torch.unique(inference_fragment_id).tolist():
        inference_indices = (
            inference_fragment_id == int(inference_fragment)
        ).nonzero(as_tuple=True)[0]
        crystal_indices = mapping[inference_indices]
        crystal_fragments = torch.unique(crystal_fragment_id[crystal_indices])
        moving = inference_coords[inference_indices]
        target = crystal_coords[crystal_indices]
        _, residual = _proper_rigid_fit(moving, target)
        squared_error = float(residual.square().sum().item())
        pair_squared_error, pair_count = _pair_distance_squared_error(moving, target)
        atom_count = int(inference_indices.numel())
        fit_rank = int(torch.linalg.matrix_rank(moving - moving.mean(dim=0)).item())
        total_squared_error += squared_error
        total_pair_squared_error += pair_squared_error
        total_pairs += pair_count
        fragment_records.append(
            {
                "inference_fragment_id": int(inference_fragment),
                "stored_crystal_fragment_ids": [
                    int(fragment) for fragment in crystal_fragments.tolist()
                ],
                "crystal_atom_indices": [int(index) for index in crystal_indices.tolist()],
                "atom_count": atom_count,
                "fit_rank": fit_rank,
                "orientation_observable": fit_rank >= 2,
                "rigid_fit_rmsd": math.sqrt(squared_error / atom_count),
                "max_atom_residual": float(torch.linalg.vector_norm(residual, dim=-1).max()),
                "pair_distance_rmse": (
                    math.sqrt(pair_squared_error / pair_count) if pair_count else 0.0
                ),
            }
        )

    atom_count = n_inference
    return {
        "atom_count": atom_count,
        "fragment_count": len(fragment_records),
        "stored_fragment_count": int(torch.unique(crystal_fragment_id).numel()),
        "stored_partition_equal": stored_partition_equal,
        "rigid_fragment_floor_rmsd": math.sqrt(total_squared_error / atom_count),
        "pair_distance_rmse": (
            math.sqrt(total_pair_squared_error / total_pairs) if total_pairs else 0.0
        ),
        "squared_error": total_squared_error,
        "pair_squared_error": total_pair_squared_error,
        "pair_count": total_pairs,
        "max_fragment_rmsd": max(
            float(record["rigid_fit_rmsd"]) for record in fragment_records
        ),
        "fragments": fragment_records,
    }


def analyze_fragment_geometry_pair(
    mol_crystal: Chem.Mol,
    mol_inference: Chem.Mol,
    crystal_coords: Tensor,
    crystal_fragment_id: Tensor,
    *,
    max_matches: int = 1024,
) -> dict[str, object]:
    """Find the symmetry mapping with the smallest rigid-fragment RMSD floor."""
    inference_coords = torch.as_tensor(
        mol_inference.GetConformer().GetPositions(),
        dtype=torch.float64,
    )
    inference_fragments = decompose_fragments(mol_inference, inference_coords)
    if inference_fragments is None:
        raise ValueError("inference fragment decomposition failed")

    mappings, mapping_method, mapping_truncated = enumerate_full_atom_mappings(
        mol_crystal,
        mol_inference,
        max_matches=max_matches,
    )
    if not mappings:
        raise ValueError(f"no full atom mapping ({mapping_method})")

    candidates: list[tuple[float, tuple[int, ...], dict[str, object]]] = []
    partition_mismatch_count = 0
    for mapping in mappings:
        result = fragment_rigid_fit_floor(
            crystal_coords,
            inference_coords,
            crystal_fragment_id,
            inference_fragments["fragment_id"],
            mapping,
        )
        if not bool(result["stored_partition_equal"]):
            partition_mismatch_count += 1
        candidates.append((float(result["rigid_fragment_floor_rmsd"]), mapping, result))
    _, selected_mapping, selected = min(candidates, key=lambda item: item[0])
    return {
        **selected,
        "mapping_method": mapping_method,
        "mapping_count": len(mappings),
        "mapping_truncated": mapping_truncated,
        "symmetry_complete": mapping_method == "strict_stereo" and not mapping_truncated,
        "partition_mismatch_mapping_count": partition_mismatch_count,
        "any_mapping_stored_partition_equal": partition_mismatch_count < len(mappings),
        "inference_to_crystal": list(selected_mapping),
    }


def partitions_equivalent(left: Tensor, right: Tensor) -> bool:
    """Return whether two atom-indexed fragment assignments encode one partition."""
    return _canonical_partition(left) == _canonical_partition(right)


__all__ = [
    "FragmentPartitionMismatchError",
    "analyze_fragment_geometry_pair",
    "enumerate_full_atom_mappings",
    "fragment_rigid_fit_floor",
    "partitions_equivalent",
]
