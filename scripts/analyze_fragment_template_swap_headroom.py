#!/usr/bin/env python
"""Measure crystal fragment-template headroom on saved docking ensembles.

This is an oracle, post-hoc diagnostic.  It replaces the internal coordinates
of each inference fragment with the corresponding crystal fragment while
preserving that saved pose's independently fitted fragment frame.  It never
copies or fits a whole-ligand crystal placement.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import statistics
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from rdkit import Chem, rdBase

from effdock.evaluation.benchmark import compute_pose_rmsd, load_ligand
from effdock.evaluation.fragment_geometry import (
    enumerate_full_atom_mappings,
    fragment_rigid_fit_floor,
)
from effdock.preprocess.fragments import decompose_fragments
from effdock.workflows.benchmark_inputs import (
    file_sha256,
    load_benchmark_inputs,
    load_benchmark_ligand,
)

SCHEMA_VERSION = "effdock.fragment_template_swap_headroom.v1"
RMSD_THRESHOLD_ANGSTROM = 2.0
DEFAULT_MAX_MATCHES = 4096
STEREO_POLICIES = ("require", "connectivity_sensitivity")
SDF_FRAGMENT_PAIR_DISTANCE_TOLERANCE_ANGSTROM = 5e-4
IMPLEMENTATION_FILES = (
    Path("scripts/analyze_fragment_template_swap_headroom.py"),
    Path("src/effdock/evaluation/fragment_geometry.py"),
    Path("src/effdock/evaluation/benchmark.py"),
    Path("src/effdock/inference/preprocess.py"),
    Path("src/effdock/inference/io.py"),
    Path("src/effdock/preprocess/fragments.py"),
    Path("src/effdock/workflows/benchmark_inputs.py"),
)


def _proper_rotation(moving: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return the proper Kabsch rotation taking centered ``moving`` to ``target``."""
    moving = torch.as_tensor(moving, dtype=torch.float64)
    target = torch.as_tensor(target, dtype=torch.float64)
    if moving.shape != target.shape or moving.ndim != 2 or moving.shape[1] != 3:
        raise ValueError("moving and target coordinates must have shape [N, 3]")
    if moving.shape[0] == 0:
        raise ValueError("a fragment must contain at least one atom")
    if not bool(torch.isfinite(moving).all() and torch.isfinite(target).all()):
        raise ValueError("fragment coordinates must be finite")

    moving_centered = moving - moving.mean(dim=0, keepdim=True)
    target_centered = target - target.mean(dim=0, keepdim=True)
    u, _, vh = torch.linalg.svd(moving_centered.T @ target_centered)
    rotation = vh.T @ u.T
    if float(torch.linalg.det(rotation)) < 0.0:
        correction = torch.eye(3, dtype=torch.float64)
        correction[-1, -1] = -1.0
        rotation = vh.T @ correction @ u.T
    return rotation


def replace_fragment_internal_geometry(
    pose_coords: torch.Tensor,
    crystal_coords: torch.Tensor,
    inference_fragment_id: torch.Tensor,
    inference_to_crystal: Sequence[int],
) -> torch.Tensor:
    """Swap fragment-local templates without preserving whole-ligand placement."""
    pose = torch.as_tensor(pose_coords, dtype=torch.float64)
    crystal = torch.as_tensor(crystal_coords, dtype=torch.float64)
    fragment_id = torch.as_tensor(inference_fragment_id, dtype=torch.long).view(-1)
    mapping = torch.as_tensor(inference_to_crystal, dtype=torch.long).view(-1)
    atom_count = int(pose.shape[0])
    if pose.shape != (atom_count, 3) or crystal.shape != (atom_count, 3):
        raise ValueError("pose and crystal coordinates must both have shape [N, 3]")
    if fragment_id.numel() != atom_count or mapping.numel() != atom_count:
        raise ValueError("fragment IDs and mapping must match the pose atom count")
    if sorted(mapping.tolist()) != list(range(atom_count)):
        raise ValueError("inference_to_crystal must be a full atom bijection")

    swapped = torch.empty_like(pose)
    for fragment in torch.unique(fragment_id).tolist():
        indices = (fragment_id == int(fragment)).nonzero(as_tuple=True)[0]
        crystal_fragment = crystal.index_select(0, mapping.index_select(0, indices))
        pose_fragment = pose.index_select(0, indices)
        rotation = _proper_rotation(crystal_fragment, pose_fragment)
        placed = (
            (crystal_fragment - crystal_fragment.mean(dim=0, keepdim=True))
            @ rotation.T
            + pose_fragment.mean(dim=0, keepdim=True)
        )
        swapped.index_copy_(0, indices, placed)
    return swapped


def benchmark_symmetry_aware_rmsd(
    pose_coords: torch.Tensor,
    crystal_coords: torch.Tensor,
    inference_to_crystal: Sequence[int],
    mol_input: Chem.Mol,
    mol_ref: Chem.Mol,
) -> float:
    """Match the production benchmark's RDKit CalcRMS metric and full-map fallback."""
    pose = torch.as_tensor(pose_coords, dtype=torch.float64)
    crystal = torch.as_tensor(crystal_coords, dtype=torch.float64)
    mapping = torch.as_tensor(inference_to_crystal, dtype=torch.long).view(-1)
    if pose.ndim != 2 or pose.shape[1] != 3 or crystal.shape != pose.shape:
        raise ValueError("pose and crystal coordinates must both have shape [N, 3]")
    if mapping.numel() != pose.shape[0] or sorted(mapping.tolist()) != list(
        range(pose.shape[0])
    ):
        raise ValueError("inference_to_crystal must be a full atom bijection")
    return compute_pose_rmsd(
        pose,
        crystal.index_select(0, mapping),
        torch.zeros(3, dtype=torch.float64),
        list(range(pose.shape[0])),
        mol_input,
        mol_ref,
    )


def _reference_automorphism_expansion(
    mol_ref: Chem.Mol,
    base_mapping: Sequence[int],
    *,
    max_matches: int,
) -> tuple[list[tuple[int, ...]], bool]:
    """Compose a validated fallback mapping with exact reference automorphisms."""
    automorphisms = mol_ref.GetSubstructMatches(
        mol_ref,
        uniquify=False,
        maxMatches=max_matches + 1,
        useChirality=True,
    )
    truncated = len(automorphisms) > max_matches
    base = tuple(int(index) for index in base_mapping)
    composed = {
        tuple(int(automorphism[crystal_index]) for crystal_index in base)
        for automorphism in automorphisms[:max_matches]
    }
    return sorted(composed), truncated


def enumerate_symmetry_mappings(
    mol_ref: Chem.Mol,
    mol_input: Chem.Mol,
    *,
    max_matches: int,
    stereo_policy: str = "require",
) -> tuple[list[tuple[int, ...]], dict[str, Any]]:
    """Enumerate exact input-to-reference symmetries or fail explicitly."""
    if stereo_policy not in STEREO_POLICIES:
        raise ValueError(f"unsupported stereo policy: {stereo_policy}")
    mappings, method, truncated = enumerate_full_atom_mappings(
        mol_ref,
        mol_input,
        max_matches=max_matches,
    )
    if not mappings:
        if stereo_policy == "require":
            raise ValueError(f"no stereo-preserving full atom mapping ({method})")
        if not method.startswith("fallback_stereo_rejected:"):
            raise ValueError(
                "connectivity sensitivity may relax stereochemistry only after "
                f"a complete constitutional mapping ({method})"
            )
        nonchiral = mol_ref.GetSubstructMatches(
            mol_input,
            uniquify=False,
            maxMatches=max_matches + 1,
            useChirality=False,
        )
        truncated = len(nonchiral) > max_matches
        mappings = sorted(
            {
                tuple(int(index) for index in mapping)
                for mapping in nonchiral[:max_matches]
            }
        )
        if truncated:
            raise ValueError(
                "exact non-chiral symmetry enumeration exceeded "
                f"max_matches={max_matches}"
            )
        if not mappings:
            raise ValueError(
                "stereo-relaxed sensitivity found no exact constitutional mapping"
            )
        return mappings, {
            "base_mapping_method": "strict_connectivity_stereo_mismatch",
            "symmetry_expansion": "direct_nonchiral_input_to_reference_matches",
            "mapping_count": len(mappings),
            "mapping_truncated": False,
            "symmetry_complete": True,
            "stereo_preserving": False,
            "sensitivity_only": True,
            "stereo_failure_reason": method,
        }
    expansion = "direct_input_to_reference_matches"
    if method != "strict_stereo":
        mappings, auto_truncated = _reference_automorphism_expansion(
            mol_ref,
            mappings[0],
            max_matches=max_matches,
        )
        truncated = bool(truncated or auto_truncated)
        expansion = "validated_fallback_composed_with_reference_stereo_automorphisms"
    if truncated:
        raise ValueError(
            f"exact symmetry enumeration exceeded max_matches={max_matches} ({method})"
        )
    if not mappings:
        raise ValueError("reference automorphism expansion produced no mapping")
    return mappings, {
        "base_mapping_method": method,
        "symmetry_expansion": expansion,
        "mapping_count": len(mappings),
        "mapping_truncated": False,
        "symmetry_complete": True,
        "stereo_preserving": True,
        "sensitivity_only": False,
    }


def _select_swap_mapping(
    mol_ref: Chem.Mol,
    mol_input: Chem.Mol,
    crystal_coords: torch.Tensor,
    inference_fragment_id: torch.Tensor,
    mappings: Sequence[Sequence[int]],
) -> tuple[tuple[int, ...], dict[str, Any]]:
    crystal_fragments = decompose_fragments(mol_ref, crystal_coords)
    if crystal_fragments is None:
        raise ValueError("crystal fragment decomposition failed")
    input_coords = torch.as_tensor(
        mol_input.GetConformer().GetPositions(),
        dtype=torch.float64,
    )
    candidates: list[tuple[float, tuple[int, ...], dict[str, Any]]] = []
    for raw_mapping in mappings:
        mapping = tuple(int(index) for index in raw_mapping)
        fit = fragment_rigid_fit_floor(
            crystal_coords,
            input_coords,
            crystal_fragments["fragment_id"],
            inference_fragment_id,
            mapping,
        )
        candidates.append((float(fit["rigid_fragment_floor_rmsd"]), mapping, fit))
    _, selected_mapping, selected_fit = min(candidates, key=lambda item: item[0])
    return selected_mapping, selected_fit


def _graph_order_signature(mol: Chem.Mol) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    atoms = tuple(
        (
            atom.GetAtomicNum(),
            atom.GetIsotope(),
            atom.GetFormalCharge(),
        )
        for atom in mol.GetAtoms()
    )
    bonds = tuple(
        sorted(
            (
                min(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()),
                max(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()),
                float(bond.GetBondTypeAsDouble()),
                bool(bond.GetIsAromatic()),
            )
            for bond in mol.GetBonds()
        )
    )
    return atoms, bonds


def _load_all_poses(path: Path) -> list[Chem.Mol]:
    # Do not use ``list(Chem.SDMolSupplier(path))`` here.  ``list`` asks the
    # indexed supplier for a length hint, which switches RDKit onto its random
    # access path.  RDKit 2025.09.5 can then mis-seek an otherwise valid record
    # whose byte offset sits on a 4096-byte buffer boundary (observed for
    # PoseBusters 7rou_66i, sample 95).  The forward supplier reads the exact
    # serialized stream and preserves every record/property without recovery
    # or input mutation.
    with path.open("rb") as handle:
        poses = [
            mol
            for mol in Chem.ForwardSDMolSupplier(
                handle,
                sanitize=True,
                removeHs=False,
                strictParsing=True,
            )
        ]
    if not poses or any(mol is None for mol in poses):
        failed = [index for index, mol in enumerate(poses) if mol is None]
        raise ValueError(f"strict SDF parsing failed for candidate records {failed[:8]}")
    return [mol for mol in poses if mol is not None]


def _property_values(poses: Sequence[Chem.Mol], name: str) -> list[str]:
    missing = [index for index, mol in enumerate(poses) if not mol.HasProp(name)]
    if missing:
        raise ValueError(f"saved poses lack required property {name!r}: {missing[:8]}")
    return [mol.GetProp(name).strip() for mol in poses]


def _single_property(poses: Sequence[Chem.Mol], name: str) -> str:
    values = _property_values(poses, name)
    unique = sorted(set(values))
    if len(unique) != 1:
        raise ValueError(f"saved pose property {name!r} is inconsistent: {unique[:8]}")
    return unique[0]


def _load_summary_seed_rows(paths: Sequence[Path]) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    for path in paths:
        with path.open(newline="") as handle:
            for raw in csv.DictReader(handle):
                complex_id = str(raw.get("id", "")).strip().lower()
                seed = str(raw.get("sampling_seed", "")).strip()
                if not complex_id or not seed:
                    continue
                record = {
                    "sampling_seed": seed,
                    "all_poses_sdf": str(raw.get("all_poses_sdf", "")).strip(),
                    "all_poses_sdf_sha256": str(
                        raw.get("all_poses_sdf_sha256", "")
                    ).strip(),
                    "source_csv": str(path),
                }
                previous = rows.get(complex_id)
                if previous is not None and previous["sampling_seed"] != seed:
                    raise ValueError(f"conflicting summary seeds for {complex_id}")
                rows[complex_id] = record
    return rows


def _resolve_sampling_seed(
    complex_id: str,
    pose_path: Path,
    poses: Sequence[Chem.Mol],
    summary_row: dict[str, str] | None,
) -> tuple[int, str]:
    sdf_seed: int | None = None
    if all(mol.HasProp("sampling_seed") for mol in poses):
        sdf_seed = int(_single_property(poses, "sampling_seed"))
    elif any(mol.HasProp("sampling_seed") for mol in poses):
        raise ValueError("sampling_seed is present on only part of the saved ensemble")

    csv_seed: int | None = None
    if summary_row is not None:
        csv_seed = int(summary_row["sampling_seed"])
        bound_path = summary_row.get("all_poses_sdf", "")
        if bound_path and Path(bound_path).resolve() != pose_path.resolve():
            raise ValueError(f"summary row for {complex_id} binds a different all_poses SDF")
        bound_hash = summary_row.get("all_poses_sdf_sha256", "")
        if bound_hash and bound_hash != file_sha256(pose_path):
            raise ValueError(f"summary row for {complex_id} has a stale all_poses hash")

    if sdf_seed is not None and csv_seed is not None and sdf_seed != csv_seed:
        raise ValueError(f"SDF and summary sampling seeds disagree for {complex_id}")
    if sdf_seed is not None:
        return sdf_seed, (
            "all_poses_sdf.sampling_seed+summary_csv"
            if csv_seed is not None
            else "all_poses_sdf.sampling_seed"
        )
    if csv_seed is not None:
        return csv_seed, "summary_csv.sampling_seed"
    raise ValueError(
        "sampling seed is absent from the saved SDF and no bound summary CSV row exists"
    )


def _validate_saved_ensemble(
    *,
    dataset: str,
    complex_id: str,
    poses: Sequence[Chem.Mol],
    template: Chem.Mol,
    expected_poses: int,
    expected_sigma: float,
) -> dict[str, Any]:
    if len(poses) != expected_poses:
        raise ValueError(f"expected {expected_poses} poses, found {len(poses)}")
    if _single_property(poses, "dataset").lower() != dataset:
        raise ValueError("saved ensemble dataset property does not match --dataset")
    if _single_property(poses, "complex_id").lower() != complex_id:
        raise ValueError("saved ensemble complex_id property does not match its filename")
    if int(_single_property(poses, "num_samples")) != expected_poses:
        raise ValueError("saved ensemble num_samples property is not the expected pose count")
    sigma = float(_single_property(poses, "sample_sigma"))
    if not math.isclose(sigma, expected_sigma, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"expected sample_sigma={expected_sigma}, found {sigma}")
    sample_indices = [int(value) for value in _property_values(poses, "sample_index")]
    if sample_indices != list(range(expected_poses)):
        raise ValueError("saved ensemble sample_index order is not canonical 0..N-1")
    record_names = [mol.GetProp("_Name").strip() for mol in poses]
    if record_names != [f"docked_pose_{index}" for index in range(expected_poses)]:
        raise ValueError("saved ensemble record names do not match the production writer")
    template_signature = _graph_order_signature(template)
    mismatched = [
        index
        for index, pose in enumerate(poses)
        if _graph_order_signature(pose) != template_signature
    ]
    if mismatched:
        raise ValueError(
            "saved SDF atom order/graph is not identical to the regenerated production "
            f"template for records {mismatched[:8]}"
        )
    coordinate_failures = [
        index
        for index, pose in enumerate(poses)
        if not bool(
            torch.isfinite(
                torch.as_tensor(pose.GetConformer().GetPositions(), dtype=torch.float64)
            ).all()
        )
    ]
    if coordinate_failures:
        raise ValueError(f"saved poses contain non-finite coordinates: {coordinate_failures[:8]}")
    template_coords = torch.as_tensor(
        template.GetConformer().GetPositions(),
        dtype=torch.float64,
    )
    fragments = decompose_fragments(template, template_coords)
    if fragments is None:
        raise ValueError("regenerated production-template fragment decomposition failed")
    fragment_id = fragments["fragment_id"].to(torch.long)
    maximum_pair_distance_error = 0.0
    for pose in poses:
        pose_coords = torch.as_tensor(
            pose.GetConformer().GetPositions(),
            dtype=torch.float64,
        )
        for fragment in torch.unique(fragment_id).tolist():
            indices = (fragment_id == int(fragment)).nonzero(as_tuple=True)[0]
            if indices.numel() < 2:
                continue
            error = (
                torch.pdist(pose_coords.index_select(0, indices))
                - torch.pdist(template_coords.index_select(0, indices))
            ).abs()
            maximum_pair_distance_error = max(
                maximum_pair_distance_error,
                float(error.max().item()),
            )
    if maximum_pair_distance_error > SDF_FRAGMENT_PAIR_DISTANCE_TOLERANCE_ANGSTROM:
        raise ValueError(
            "saved poses are not rigid transports of the regenerated production "
            f"fragment template: max pair-distance error={maximum_pair_distance_error:.6g} A"
        )
    ensemble_hash = _single_property(poses, "candidate_ensemble_sha256")
    if len(ensemble_hash) != 64 or any(
        character not in "0123456789abcdef" for character in ensemble_hash.lower()
    ):
        raise ValueError("candidate_ensemble_sha256 is not a canonical SHA-256 digest")
    return {
        "pose_count": len(poses),
        "sample_sigma": sigma,
        "num_steps": int(_single_property(poses, "num_steps")),
        "sample_indices_canonical": True,
        "production_writer_record_names_canonical": True,
        "saved_constitutional_graph_and_atom_order_match_regenerated_template": True,
        "saved_fragments_match_regenerated_template_up_to_rigid_motion": True,
        "maximum_fragment_pair_distance_error_angstrom": maximum_pair_distance_error,
        "fragment_pair_distance_tolerance_angstrom": (
            SDF_FRAGMENT_PAIR_DISTANCE_TOLERANCE_ANGSTROM
        ),
        "candidate_ensemble_sha256": ensemble_hash,
    }


def _find_reference_ligand(dataset_root: Path, complex_id: str) -> tuple[Path, str]:
    directories = [path for path in dataset_root.iterdir() if path.is_dir()]
    exact = [path for path in directories if path.name.lower() == complex_id]
    if exact:
        matching_dirs = exact
    else:
        pdb_prefix = complex_id.split("_", maxsplit=1)[0] + "_"
        matching_dirs = [
            path for path in directories if path.name.lower().startswith(pdb_prefix)
        ]
    if len(matching_dirs) != 1:
        raise ValueError(
            f"reference directory resolution is ambiguous for {complex_id}: "
            f"{[path.name for path in matching_dirs]}"
        )
    complex_dir = matching_dirs[0]
    candidates = [
        path
        for path in complex_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in {".sdf", ".mol2"}
        and path.stem.lower().endswith("_ligand")
    ]
    if len(candidates) != 1:
        raise ValueError(
            f"reference ligand resolution is ambiguous in {complex_dir}: "
            f"{[path.name for path in candidates]}"
        )
    path = candidates[0]
    return path, path.suffix.lower().lstrip(".")


def _analyze_complex(
    *,
    dataset: str,
    dataset_root: Path,
    pose_path: Path,
    smiles: str,
    summary_row: dict[str, str] | None,
    expected_poses: int,
    expected_sigma: float,
    max_matches: int,
    stereo_policy: str,
) -> dict[str, Any]:
    complex_id = pose_path.stem.lower()
    stage = "load_saved_poses"
    record: dict[str, Any] = {"id": complex_id, "status": "failed"}
    try:
        poses = _load_all_poses(pose_path)
        stage = "resolve_sampling_seed"
        sampling_seed, seed_source = _resolve_sampling_seed(
            complex_id,
            pose_path,
            poses,
            summary_row,
        )
        stage = "regenerate_production_template"
        template, has_pose = load_benchmark_ligand(smiles, random_seed=sampling_seed)
        if has_pose:
            raise ValueError("frozen benchmark SMILES unexpectedly loaded as a posed ligand")
        stage = "validate_saved_ensemble"
        saved_contract = _validate_saved_ensemble(
            dataset=dataset,
            complex_id=complex_id,
            poses=poses,
            template=template,
            expected_poses=expected_poses,
            expected_sigma=expected_sigma,
        )

        stage = "load_crystal_reference"
        reference_path, reference_format = _find_reference_ligand(
            dataset_root,
            complex_id,
        )
        mol_ref = load_ligand(reference_path, reference_format)
        crystal_coords = torch.as_tensor(
            mol_ref.GetConformer().GetPositions(),
            dtype=torch.float64,
        )
        stage = "decompose_inference_fragments"
        template_coords = torch.as_tensor(
            template.GetConformer().GetPositions(),
            dtype=torch.float64,
        )
        fragments = decompose_fragments(template, template_coords)
        if fragments is None:
            raise ValueError("production-template fragment decomposition failed")
        inference_fragment_id = fragments["fragment_id"].to(torch.long)

        stage = "enumerate_stereo_symmetries"
        mappings, mapping_metadata = enumerate_symmetry_mappings(
            mol_ref,
            template,
            max_matches=max_matches,
            stereo_policy=stereo_policy,
        )
        selected_mapping, selected_fit = _select_swap_mapping(
            mol_ref,
            template,
            crystal_coords,
            inference_fragment_id,
            mappings,
        )

        stage = "swap_and_score_saved_poses"
        pose_rows: list[dict[str, Any]] = []
        before_values: list[float] = []
        after_values: list[float] = []
        for index, pose_mol in enumerate(poses):
            pose_coords = torch.as_tensor(
                pose_mol.GetConformer().GetPositions(),
                dtype=torch.float64,
            )
            swapped = replace_fragment_internal_geometry(
                pose_coords,
                crystal_coords,
                inference_fragment_id,
                selected_mapping,
            )
            before = benchmark_symmetry_aware_rmsd(
                pose_coords,
                crystal_coords,
                selected_mapping,
                template,
                mol_ref,
            )
            after = benchmark_symmetry_aware_rmsd(
                swapped,
                crystal_coords,
                selected_mapping,
                template,
                mol_ref,
            )
            before_success = before < RMSD_THRESHOLD_ANGSTROM
            after_success = after < RMSD_THRESHOLD_ANGSTROM
            if not before_success and after_success:
                crossing = "entered_lt2"
            elif before_success and not after_success:
                crossing = "exited_lt2"
            elif before_success:
                crossing = "remained_lt2"
            else:
                crossing = "remained_ge2"
            before_values.append(before)
            after_values.append(after)
            pose_rows.append(
                {
                    "sample_index": index,
                    "before_rmsd_angstrom": before,
                    "after_rmsd_angstrom": after,
                    "delta_rmsd_angstrom": after - before,
                    "threshold_crossing": crossing,
                }
            )

        k2_before = sum(value < RMSD_THRESHOLD_ANGSTROM for value in before_values)
        k2_after = sum(value < RMSD_THRESHOLD_ANGSTROM for value in after_values)
        crossing_counts = Counter(row["threshold_crossing"] for row in pose_rows)
        stage = "complete"
        record.update(
            {
                "status": "ok",
                "sampling_seed": sampling_seed,
                "sampling_seed_source": seed_source,
                "smiles_sha256": hashlib.sha256(smiles.encode("utf-8")).hexdigest(),
                "input_sha256": {
                    "all_poses_sdf": file_sha256(pose_path),
                    "crystal_reference": file_sha256(reference_path),
                },
                "paths": {
                    "all_poses_sdf": str(pose_path),
                    "crystal_reference": str(reference_path),
                },
                "saved_ensemble_contract": saved_contract,
                "mapping": {
                    **mapping_metadata,
                    "selected_inference_to_crystal": list(selected_mapping),
                    "selection_rule": (
                        "minimum production-template rigid-fragment floor; fixed for all poses"
                    ),
                    "production_template_rigid_fragment_floor_rmsd_angstrom": float(
                        selected_fit["rigid_fragment_floor_rmsd"]
                    ),
                    "stored_partition_equal": bool(
                        selected_fit["stored_partition_equal"]
                    ),
                },
                "fragment_count": int(torch.unique(inference_fragment_id).numel()),
                "atom_count": int(template.GetNumAtoms()),
                "before": {
                    "k2": k2_before,
                    "k2_fraction": k2_before / len(before_values),
                    "best_rmsd_angstrom": min(before_values),
                    "best_sample_index": int(min(range(len(before_values)), key=before_values.__getitem__)),
                    "median_rmsd_angstrom": statistics.median(before_values),
                },
                "after": {
                    "k2": k2_after,
                    "k2_fraction": k2_after / len(after_values),
                    "best_rmsd_angstrom": min(after_values),
                    "best_sample_index": int(min(range(len(after_values)), key=after_values.__getitem__)),
                    "median_rmsd_angstrom": statistics.median(after_values),
                },
                "delta": {
                    "k2": k2_after - k2_before,
                    "best_rmsd_angstrom": min(after_values) - min(before_values),
                    "pose_threshold_crossings": dict(sorted(crossing_counts.items())),
                },
                "poses": pose_rows,
            }
        )
    except Exception as error:
        record.update(
            {
                "failure_stage": stage,
                "failure_code": f"{stage}:{type(error).__name__}",
                "error_type": type(error).__name__,
                "error": str(error),
            }
        )
    return record


def _arm_summary(records: Sequence[dict[str, Any]], label: str) -> dict[str, Any]:
    counts = [int(record[label]["k2"]) for record in records]
    fractions = [float(record[label]["k2_fraction"]) for record in records]
    best = [float(record[label]["best_rmsd_angstrom"]) for record in records]
    total_candidates = sum(int(record["saved_ensemble_contract"]["pose_count"]) for record in records)
    return {
        "mean_k2": statistics.fmean(counts) if counts else None,
        "median_k2": statistics.median(counts) if counts else None,
        "macro_k2_fraction": statistics.fmean(fractions) if fractions else None,
        "macro_k2_per_100": (
            100.0 * statistics.fmean(fractions) if fractions else None
        ),
        "micro_k2_fraction": sum(counts) / total_candidates if total_candidates else None,
        "p_k_ge_1": sum(count >= 1 for count in counts) / len(counts) if counts else None,
        "p_k_ge_5": sum(count >= 5 for count in counts) / len(counts) if counts else None,
        "p_k_ge_10": sum(count >= 10 for count in counts) / len(counts) if counts else None,
        "best_rmsd_angstrom": {
            "mean": statistics.fmean(best) if best else None,
            "median": statistics.median(best) if best else None,
            "max": max(best) if best else None,
        },
    }


def aggregate_records(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    valid = [record for record in records if record["status"] == "ok"]
    failures = [record for record in records if record["status"] != "ok"]

    def mapping_subgroup(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
        before_total = sum(int(record["before"]["k2"]) for record in rows)
        after_total = sum(int(record["after"]["k2"]) for record in rows)
        return {
            "complex_count": len(rows),
            "complex_ids": sorted(str(record["id"]) for record in rows),
            "before_k2_total": before_total,
            "after_k2_total": after_total,
            "delta_k2_total": after_total - before_total,
            "mean_delta_k2": (
                (after_total - before_total) / len(rows) if rows else None
            ),
        }

    sensitivity_rows = [
        record
        for record in valid
        if bool(record.get("mapping", {}).get("sensitivity_only", False))
    ]
    stereo_rows = [
        record
        for record in valid
        if not bool(record.get("mapping", {}).get("sensitivity_only", False))
    ]
    complex_crossings: dict[str, dict[str, int]] = {}
    for threshold in (1, 5, 10):
        gained = sum(
            int(record["before"]["k2"]) < threshold <= int(record["after"]["k2"])
            for record in valid
        )
        lost = sum(
            int(record["after"]["k2"]) < threshold <= int(record["before"]["k2"])
            for record in valid
        )
        complex_crossings[f"k_ge_{threshold}"] = {"gained": gained, "lost": lost}
    pose_crossings = Counter(
        pose["threshold_crossing"] for record in valid for pose in record["poses"]
    )
    return {
        "attempted_complexes": len(records),
        "successful_complexes": len(valid),
        "failed_complexes": len(failures),
        "coverage_fraction": len(valid) / len(records) if records else 0.0,
        "failure_codes": dict(
            sorted(Counter(record["failure_code"] for record in failures).items())
        ),
        "mapping_subgroups": {
            "stereo_preserving": mapping_subgroup(stereo_rows),
            "connectivity_sensitivity_only": mapping_subgroup(sensitivity_rows),
        },
        "before": _arm_summary(valid, "before"),
        "after": _arm_summary(valid, "after"),
        "delta": {
            "mean_k2": (
                statistics.fmean(
                    int(record["after"]["k2"]) - int(record["before"]["k2"])
                    for record in valid
                )
                if valid
                else None
            ),
            "median_k2": (
                statistics.median(
                    int(record["after"]["k2"]) - int(record["before"]["k2"])
                    for record in valid
                )
                if valid
                else None
            ),
            "complex_threshold_crossings": complex_crossings,
            "pose_threshold_crossings": dict(sorted(pose_crossings.items())),
        },
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("astex", "posebusters"), required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--all-poses-dir", type=Path, required=True)
    parser.add_argument("--benchmark-input-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-csv", type=Path, action="append", default=[])
    parser.add_argument("--expected-poses", type=int, default=100)
    parser.add_argument("--expected-sigma", type=float, default=2.0)
    parser.add_argument("--max-matches", type=int, default=DEFAULT_MAX_MATCHES)
    parser.add_argument(
        "--stereo-policy",
        choices=STEREO_POLICIES,
        default="require",
        help=(
            "require stereo-preserving maps, or run a separately labelled "
            "non-chiral connectivity sensitivity analysis"
        ),
    )
    parser.add_argument("--only-id", action="append", default=[])
    parser.add_argument("--require-complete-success", action="store_true")
    parser.add_argument("--protocol-id", default="")
    parser.add_argument("--protocol-path", type=Path)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.expected_poses <= 0:
        raise ValueError("--expected-poses must be positive")
    if not math.isfinite(args.expected_sigma) or args.expected_sigma < 0.0:
        raise ValueError("--expected-sigma must be finite and non-negative")
    if args.max_matches <= 0:
        raise ValueError("--max-matches must be positive")
    stereo_policy = getattr(args, "stereo_policy", "require")
    if stereo_policy not in STEREO_POLICIES:
        raise ValueError(f"unsupported stereo policy: {stereo_policy}")
    if not args.dataset_root.is_dir():
        raise FileNotFoundError(f"dataset root does not exist: {args.dataset_root}")
    if not args.all_poses_dir.is_dir():
        raise FileNotFoundError(f"all_poses directory does not exist: {args.all_poses_dir}")
    if not args.benchmark_input_manifest.is_file():
        raise FileNotFoundError(
            f"benchmark input manifest does not exist: {args.benchmark_input_manifest}"
        )
    protocol_path = getattr(args, "protocol_path", None)
    if protocol_path is not None and not protocol_path.is_file():
        raise FileNotFoundError(f"protocol file does not exist: {protocol_path}")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {args.output}")
    for path in args.summary_csv:
        if not path.is_file():
            raise FileNotFoundError(f"summary CSV does not exist: {path}")

    smiles_by_id, manifest_identity = load_benchmark_inputs(
        args.dataset,
        args.dataset_root,
        args.benchmark_input_manifest,
    )
    summary_rows = _load_summary_seed_rows(args.summary_csv)
    pose_paths = sorted(args.all_poses_dir.glob("*.sdf"), key=lambda path: path.stem.lower())
    requested = {str(complex_id).lower() for complex_id in args.only_id}
    if requested:
        pose_paths = [path for path in pose_paths if path.stem.lower() in requested]
        missing = sorted(requested - {path.stem.lower() for path in pose_paths})
        if missing:
            raise FileNotFoundError(f"requested all_poses SDFs are missing: {missing}")
    if not pose_paths:
        raise ValueError("no all_poses SDF files were selected")
    duplicate_ids = [
        complex_id
        for complex_id, count in Counter(path.stem.lower() for path in pose_paths).items()
        if count > 1
    ]
    if duplicate_ids:
        raise ValueError(f"duplicate all_poses complex IDs: {duplicate_ids}")

    records: list[dict[str, Any]] = []
    for pose_path in pose_paths:
        complex_id = pose_path.stem.lower()
        smiles = smiles_by_id.get(complex_id)
        if smiles is None:
            records.append(
                {
                    "id": complex_id,
                    "status": "failed",
                    "failure_stage": "resolve_manifest_smiles",
                    "failure_code": "resolve_manifest_smiles:KeyError",
                    "error_type": "KeyError",
                    "error": "complex ID is absent from the frozen benchmark input manifest",
                }
            )
            continue
        records.append(
            _analyze_complex(
                dataset=args.dataset,
                dataset_root=args.dataset_root,
                pose_path=pose_path,
                smiles=smiles,
                summary_row=summary_rows.get(complex_id),
                expected_poses=args.expected_poses,
                expected_sigma=args.expected_sigma,
                max_matches=args.max_matches,
                stereo_policy=stereo_policy,
            )
        )

    valid = [record for record in records if record["status"] == "ok"]
    sorted_manifest_ids = sorted(smiles_by_id)
    base_seed_candidates = {
        int(record["sampling_seed"])
        - (sorted_manifest_ids.index(str(record["id"])) + 1)
        for record in valid
    }
    inferred_seed_contract: dict[str, Any] = {
        "formula_checked": "sampling_seed = base_seed + 1-based globally sorted manifest index",
        "base_seed_candidates": sorted(base_seed_candidates),
        "consistent_across_successes": len(base_seed_candidates) <= 1,
    }
    if len(base_seed_candidates) == 1:
        inferred_seed_contract["base_seed"] = next(iter(base_seed_candidates))

    implementation_hashes = {
        str(path): file_sha256(path) for path in IMPLEMENTATION_FILES if path.is_file()
    }
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "diagnostic_boundary": (
            "crystal fragment-internal geometry oracle; independently fitted per fragment; "
            "no whole-conformer crystal placement is preserved; "
            + (
                "stereo-preserving mappings required"
                if stereo_policy == "require"
                else "connectivity-only stereo-mismatch rows are sensitivity-only"
            )
        ),
        "metric_contract": {
            "rmsd": (
                "production benchmark compute_pose_rmsd: RDKit CalcRMS without alignment, "
                "with the selected complete atom map as the fallback; connectivity-only "
                "maps are permitted only in explicitly labelled sensitivity runs"
            ),
            "k2": "number of saved candidates with symmetry-aware RMSD < 2.0 angstrom",
            "macro_k2_per_100": "100 times the mean per-complex K2/N fraction",
        },
        "arguments": {
            "dataset": args.dataset,
            "dataset_root": str(args.dataset_root),
            "all_poses_dir": str(args.all_poses_dir),
            "benchmark_input_manifest": str(args.benchmark_input_manifest),
            "summary_csv": [str(path) for path in args.summary_csv],
            "expected_poses": args.expected_poses,
            "expected_sigma": args.expected_sigma,
            "max_matches": args.max_matches,
            "stereo_policy": stereo_policy,
            "only_id": sorted(requested),
        },
        "provenance": {
            "protocol_id": getattr(args, "protocol_id", ""),
            "protocol": (
                {
                    "path": str(protocol_path),
                    "sha256": file_sha256(protocol_path),
                }
                if protocol_path is not None
                else None
            ),
            "benchmark_input_identity": manifest_identity,
            "sampling_seed_contract_observed": inferred_seed_contract,
            "implementation_sha256": implementation_hashes,
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "rdkit": rdBase.rdkitVersion,
        },
        "aggregate": aggregate_records(records),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    return result


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = run(args)
    print(json.dumps(result["aggregate"], indent=2, sort_keys=True, allow_nan=False))
    if getattr(args, "require_complete_success", False) and result["aggregate"][
        "failed_complexes"
    ]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
