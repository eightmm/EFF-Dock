#!/usr/bin/env python3
"""Evaluate an EFF-Dock checkpoint on frozen external benchmarks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem

from effdock.evaluation.benchmark import (
    apply_refinement,
    compute_pose_rmsd_with_method,
    compute_stats,
    detect_complex_files,
    match_atoms,
    select_by_score,
    select_pose,
)
from effdock.evaluation.benchmark import (
    load_ligand as load_ref_ligand,
)
from effdock.evaluation.pose_scoring import build_protein_vina_inputs, score_poses
from effdock.evaluation.pose_validity import check_validity, ligand_bounds, vdw_radii
from effdock.guidance.feynman_kac import (
    DEFAULT_FK_CONSTRAINT_TERMS,
    FeynmanKacConstraintResampler,
    FKConstraintConfig,
    parse_fk_resample_times,
)
from effdock.guidance.provenance import (
    guidance_implementation_identity,
    physical_system_reference_sha256,
)
from effdock.inference.defaults import (
    DEFAULT_CONFIDENCE_CHECKPOINT,
    DEFAULT_CONFIG,
    DEFAULT_DOCKING_CHECKPOINT,
    DEFAULT_NUM_SAMPLES,
    DEFAULT_NUM_STEPS,
    DEFAULT_POCKET_CUTOFF,
    DEFAULT_SCHEDULE_POWER,
    DEFAULT_SIGMA,
    DEFAULT_TIME_SCHEDULE,
)
from effdock.inference.docking import load_model
from effdock.inference.io import write_multi_sdf
from effdock.inference.preprocess import preprocess_complex
from effdock.inference.sampler import (
    parse_sigma_list,
    sample_shared_prior_states,
    sample_unified,
    sample_unified_multi_sigma,
)
from effdock.workflows.benchmark_inputs import (
    BenchmarkInputMismatchError,
    full_heavy_atom_mapping_metadata,
    load_benchmark_inputs,
    load_benchmark_ligand,
)

UNIFIED_GUIDANCE_RECEPTOR_POLICIES = ("fail_closed", "geometry_only")
SELECTOR_PROFILES = ("legacy", "confidence_cluster_free", "candidate_only")
DEFAULT_SELECTOR_PROFILE = "legacy"
CANDIDATE_ENSEMBLE_HASH_CONTRACT = "EFFDOCK_CANDIDATE_ENSEMBLE_V1"
CONFIDENCE_SCORE_LEDGER_CONTRACT = "EFFDOCK_CONFIDENCE_SCORE_LEDGER_V1"
POSE_DIVERSITY_CONTRACT = "EFFDOCK_HEAVY_ATOM_RECEPTOR_FRAME_DIVERSITY_V2"
POSE_DIVERSITY_ROUND_DECIMALS = 3
TRANSLATION_SDE_SEED_XOR = 0x54534445
CONFIDENCE_SCORE_LEDGER_FIELDS = (
    "confidence_rmsd",
    "confidence_success_logit",
    "confidence_success",
    "confidence_atom_rmsd",
    "confidence_atom_q90",
    "confidence_atom_ok",
    "pl_clash_1p6",
)


def fk_post_resampling_mutation_classification(
    *,
    translation_sde_enabled: bool,
    translation_jitter: float,
    rotation_jitter: float,
) -> str:
    """Describe the post-resampling mutation without conflating it with dynamics."""
    if translation_sde_enabled:
        return "none_sde_is_continuous_translation_dynamics"
    if translation_jitter > 0.0 or rotation_jitter > 0.0:
        return "heuristic_not_marginal_preserving_sde"
    return "none"


@dataclass(frozen=True)
class ComplexInput:
    complex_id: str
    protein: Path
    ligand_ref: Path
    ligand_format: str
    smiles: str | None
    pocket_center: tuple[float, float, float]
    ligand_input_identity_sha256: str = ""
    ligand_input_canonical_smiles: str | None = None
    enforce_benchmark_heavy_atom_policy: bool = False


def load_smiles(
    dataset: str,
    external_dir: Path,
    benchmark_input_manifest: Path | None = None,
) -> dict[str, str]:
    """Compatibility wrapper around the content-addressed benchmark loader."""
    mapping, _ = load_benchmark_inputs(dataset, external_dir, benchmark_input_manifest)
    return mapping


def load_pocket_centers(path: Path) -> dict[str, tuple[float, float, float]]:
    """Load frozen pocket centers keyed by complex ID.

    The center provenance is deliberately not inferred here. Prospective and
    reference-defined benchmark centers must be kept in distinct manifests.
    """
    with path.open() as handle:
        raw = json.load(handle)
    centers: dict[str, tuple[float, float, float]] = {}
    for complex_id, value in raw.items():
        if isinstance(value, dict):
            value = value.get("pocket_center", value.get("center"))
        if not isinstance(value, list | tuple) or len(value) != 3:
            raise ValueError(f"invalid pocket center for {complex_id!r}: {value!r}")
        centers[complex_id.lower()] = tuple(float(x) for x in value)
    return centers


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def shared_prior_sha256(translations: torch.Tensor, rotations: torch.Tensor) -> str:
    """Hash the exact deterministic CPU prior pool used for one complex."""
    digest = hashlib.sha256()
    digest.update(b"EFFDOCK_SHARED_PRIOR_V1\0")
    for name, tensor in (("translation", translations), ("rotation", rotations)):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(str(array.dtype).encode())
        digest.update(b"\0")
        digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode())
        digest.update(b"\0")
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def candidate_ensemble_sha256(poses: list[torch.Tensor]) -> str:
    """Hash one ordered post-refinement candidate ensemble."""
    if not poses:
        raise ValueError("poses must be non-empty")
    array = (
        torch.stack([pose.detach().cpu().to(torch.float32) for pose in poses], dim=0)
        .contiguous()
        .numpy()
    )
    digest = hashlib.sha256()
    digest.update(CANDIDATE_ENSEMBLE_HASH_CONTRACT.encode("utf-8"))
    digest.update(b"\0")
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(b"\0")
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("utf-8"))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def pose_diversity_metrics(
    poses: list[torch.Tensor],
    mol: Chem.Mol,
) -> dict[str, int | float]:
    """Measure heavy-atom clone count and receptor-frame candidate spread."""
    if not poses:
        raise ValueError("poses must be non-empty")
    coordinates = (
        torch.stack([pose.detach().cpu().to(torch.float64) for pose in poses], dim=0)
        .contiguous()
        .numpy()
    )
    if coordinates.ndim != 3 or coordinates.shape[2] != 3:
        raise ValueError("poses must have shape (samples, atoms, 3)")
    if coordinates.shape[1] != mol.GetNumAtoms():
        raise ValueError("pose atom count does not match ligand molecule")
    if not np.isfinite(coordinates).all():
        raise ValueError("pose coordinates must be finite")
    heavy_mask = np.asarray(
        [atom.GetAtomicNum() > 1 for atom in mol.GetAtoms()],
        dtype=bool,
    )
    if not bool(heavy_mask.any()):
        raise ValueError("ligand must contain at least one heavy atom")
    heavy = coordinates[:, heavy_mask, :]
    scale = 10**POSE_DIVERSITY_ROUND_DECIMALS
    coordinate_unique_count = len(
        {
            np.rint(pose * scale).astype("<i8", copy=False).tobytes()
            for pose in heavy
        }
    )
    if len(heavy) == 1:
        pairwise_mean = 0.0
        pairwise_median = 0.0
        pairwise_ge2_fraction = 0.0
        nearest_median = 0.0
        c2_connected_component_count = 1
    else:
        delta = heavy[:, None, :, :] - heavy[None, :, :, :]
        pairwise = np.sqrt(np.mean(np.sum(delta * delta, axis=-1), axis=-1))
        upper = pairwise[np.triu_indices(len(heavy), k=1)]
        pairwise_mean = float(np.mean(upper))
        pairwise_median = float(np.median(upper))
        pairwise_ge2_fraction = float(np.mean(upper >= 2.0))
        adjacency = pairwise < 2.0
        unseen = set(range(len(heavy)))
        c2_connected_component_count = 0
        while unseen:
            c2_connected_component_count += 1
            stack = [unseen.pop()]
            while stack:
                node = stack.pop()
                neighbors = {
                    int(index)
                    for index in np.flatnonzero(adjacency[node])
                    if int(index) in unseen
                }
                unseen.difference_update(neighbors)
                stack.extend(neighbors)
        np.fill_diagonal(pairwise, np.inf)
        nearest_median = float(np.median(np.min(pairwise, axis=1)))
    return {
        "diversity_heavy_atom_count": int(heavy_mask.sum()),
        "coordinate_unique_count": coordinate_unique_count,
        "pairwise_heavy_atom_rmsd_mean": pairwise_mean,
        "pairwise_heavy_atom_rmsd_median": pairwise_median,
        "pairwise_heavy_atom_rmsd_ge2_fraction": pairwise_ge2_fraction,
        "nearest_neighbor_heavy_atom_rmsd_median": nearest_median,
        "c2_connected_component_count": c2_connected_component_count,
    }


def confidence_score_ledger_json(scores: list[dict[str, float]]) -> str:
    """Serialize the finite selector inputs for every ordered candidate."""
    if not scores:
        raise ValueError("confidence scores must be non-empty")
    ledger: list[dict[str, float]] = []
    for candidate_index, score in enumerate(scores):
        entry: dict[str, float] = {}
        for field in CONFIDENCE_SCORE_LEDGER_FIELDS:
            if field not in score:
                raise ValueError(
                    f"confidence candidate {candidate_index} lacks required field {field!r}"
                )
            value = score[field]
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise ValueError(
                    f"confidence candidate {candidate_index} field {field!r} must be numeric"
                )
            numeric = float(value)
            if not math.isfinite(numeric):
                raise ValueError(
                    f"confidence candidate {candidate_index} field {field!r} must be finite"
                )
            entry[field] = numeric
        ledger.append(entry)
    return json.dumps(
        ledger,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def select_confidence_cluster_free(
    poses: list[torch.Tensor],
    scores: list[dict[str, float]],
    graph: dict[str, torch.Tensor],
    pocket_center: torch.Tensor,
) -> dict[str, int]:
    """Select pure confidence and its fixed cluster-free diagnostic filter."""
    from effdock.confidence.selectors import (
        protein_ligand_clash_rates,
        select_confidence_filter,
    )

    if not poses or len(poses) != len(scores):
        raise ValueError("poses and confidence scores must be non-empty and have equal length")
    predicted_rmsd = torch.tensor(
        [score["confidence_rmsd"] for score in scores],
        dtype=torch.float32,
    )
    if not torch.isfinite(predicted_rmsd).all():
        raise ValueError("confidence RMSD predictions must be finite")
    clash = protein_ligand_clash_rates(poses, graph, pocket_center)
    if clash.numel() != len(scores) or not torch.isfinite(clash).all():
        raise ValueError("protein-ligand clash rates must be finite and match candidates")
    for index, score in enumerate(scores):
        score["pl_clash_1p6"] = float(clash[index])
    filter_index, _ = select_confidence_filter(scores, clash)
    return {
        "confidence": int(torch.argmin(predicted_rmsd)),
        "confidence_filter": int(filter_index),
    }


def sorted_id_sha256(ids: list[str]) -> str:
    """Hash one exact, ordered complex-ID list with a versioned contract."""
    digest = hashlib.sha256()
    digest.update(b"EFFDOCK_SORTED_COMPLEX_IDS_V1\0")
    for complex_id in ids:
        digest.update(complex_id.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def global_seed_by_id(
    complexes: list[ComplexInput],
    base_seed: int,
) -> dict[str, int]:
    """Map globally sorted benchmark IDs to stable per-complex seeds."""
    ids = [item.complex_id for item in complexes]
    if ids != sorted(ids) or len(ids) != len(set(ids)):
        raise ValueError("complexes must have unique globally sorted IDs")
    return {
        complex_id: int(base_seed) + global_index
        for global_index, complex_id in enumerate(ids, start=1)
    }


def _json_safe(value):
    """Convert small provenance payloads to deterministic JSON-compatible data."""
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        detached = value.detach().cpu()
        return detached.item() if detached.ndim == 0 else detached.tolist()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, set):
        return [_json_safe(item) for item in sorted(value, key=str)]
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    if hasattr(value, "as_dict") and callable(value.as_dict):
        return _json_safe(value.as_dict())
    return str(value)


def _canonical_json(value: object) -> str:
    return json.dumps(
        _json_safe(value),
        separators=(",", ":"),
        sort_keys=True,
    )


def receptor_guidance_metadata(system, requested_policy: str) -> dict[str, object]:
    """Read optional policy provenance without coupling to one system revision."""
    system_policy = getattr(system, "receptor_policy", None)
    raw_identity = getattr(system, "receptor_policy_identity", None)
    if callable(raw_identity):
        raw_identity = raw_identity()
    raw_provenance = getattr(system, "receptor_provenance", None)
    if callable(raw_provenance):
        raw_provenance = raw_provenance()
    identity = _json_safe(raw_identity)
    if not isinstance(identity, dict):
        identity = {
            "mode": requested_policy,
            "system_receptor_policy": _json_safe(system_policy),
            "identity_available": False,
        }
    else:
        identity = dict(identity)
        identity.setdefault("mode", requested_policy)
    identity_payload = _canonical_json(identity).encode("utf-8")
    declared_sha256 = identity.get("sha256")
    identity_sha256 = (
        declared_sha256
        if isinstance(declared_sha256, str) and declared_sha256
        else hashlib.sha256(b"EFFDOCK_RECEPTOR_POLICY_IDENTITY_V1\0" + identity_payload).hexdigest()
    )
    provenance = _json_safe(raw_provenance)
    if not isinstance(provenance, dict):
        provenance = {
            "mode": requested_policy,
            "obstacle_count": 0,
            "metal_fallback_count": 0,
            "metal_fallback_reasons": {},
            "provenance_available": False,
        }
    else:
        provenance = dict(provenance)
        provenance.setdefault("mode", requested_policy)
    return {
        "mode": requested_policy,
        "system_receptor_policy": _json_safe(system_policy),
        "identity": identity,
        "identity_sha256": identity_sha256,
        "provenance": provenance,
    }


def serialize_evaluation_failure(complex_id: str, exc: Exception) -> dict[str, object]:
    """Preserve the historical repr while exposing structured chemistry failures."""
    failure: dict[str, object] = {
        "id": complex_id,
        "error": repr(exc),
        "error_type": type(exc).__name__,
        "message": str(exc),
    }
    try:
        from effdock.guidance.errors import UnsupportedPhysicalChemistryError
    except ImportError:  # pragma: no cover - the package always includes guidance
        UnsupportedPhysicalChemistryError = ()  # type: ignore[assignment]
    if isinstance(exc, UnsupportedPhysicalChemistryError):
        structured = _json_safe(exc.as_dict())
        failure.update(structured)
        failure["unsupported_physical_chemistry"] = structured
    elif isinstance(exc, BenchmarkInputMismatchError):
        structured = _json_safe(exc.as_dict())
        failure.update(structured)
        failure["benchmark_input_mismatch"] = structured
    return failure


def shard_complexes(
    complexes: list[ComplexInput], shard_index: int, num_shards: int
) -> list[ComplexInput]:
    if num_shards < 1:
        raise ValueError("num_shards must be >= 1")
    if not 0 <= shard_index < num_shards:
        raise ValueError("shard_index must satisfy 0 <= shard_index < num_shards")
    return complexes[shard_index::num_shards]


def require_expected_discovered_count(actual: int, expected: int | None) -> None:
    """Fail before sampling when filesystem discovery drops a frozen benchmark ID."""
    if expected is not None and actual != expected:
        raise RuntimeError(
            "benchmark discovery count mismatch: "
            f"expected {expected} complexes, discovered {actual}"
        )


def require_complete_evaluation(
    *, num_assigned: int, num_success: int, failures: list[dict]
) -> None:
    """Turn recorded per-complex failures into a nonzero shard exit."""
    if num_success != num_assigned or failures:
        failed_ids = sorted(str(failure.get("id", "<unknown>")) for failure in failures)
        preview = ", ".join(failed_ids[:8])
        suffix = " ..." if len(failed_ids) > 8 else ""
        raise RuntimeError(
            "benchmark shard did not complete successfully: "
            f"assigned={num_assigned} success={num_success} failures={len(failures)}"
            + (f" failed_ids={preview}{suffix}" if failed_ids else "")
        )


def _write_pose(mol: Chem.Mol, pose: torch.Tensor, center: torch.Tensor, path: Path) -> None:
    pose_mol = Chem.RWMol(mol)
    conf = pose_mol.GetConformer()
    absolute = pose.detach().cpu() + center.detach().cpu()
    for atom_index, xyz in enumerate(absolute.tolist()):
        conf.SetAtomPosition(atom_index, xyz)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = Chem.SDWriter(str(path))
    writer.write(pose_mol)
    writer.close()


def summarize_rows(rows: list[dict]) -> dict:
    summary: dict[str, dict] = {}
    selectors = [
        selector
        for selector in (
            "selected",
            "first",
            "vina",
            "confidence",
            "confidence_filter",
            "confidence_final",
            "oracle",
        )
        if rows and f"{selector}_rmsd" in rows[0]
    ]
    for selector in selectors:
        values = np.asarray([row[f"{selector}_rmsd"] for row in rows], dtype=float)
        summary[selector] = compute_stats(values) if len(values) else {}
    for selector in selectors:
        valid = [bool(row[f"{selector}_fast_valid"]) for row in rows]
        summary[selector]["fast_valid_pct"] = float(np.mean(valid) * 100) if valid else None
    if rows and "num_fast_valid_candidates" in rows[0]:
        valid_counts = np.asarray(
            [int(row["num_fast_valid_candidates"]) for row in rows],
            dtype=float,
        )
        valid_oracle = np.asarray(
            [float(row["fast_valid_oracle_rmsd"]) for row in rows],
            dtype=float,
        )
        finite = np.isfinite(valid_oracle)
        summary["candidate_set"] = {
            "any_fast_valid_pct": float((valid_counts > 0).mean() * 100),
            "mean_fast_valid_candidates": float(valid_counts.mean()),
            "joint_fast_valid_and_rmsd_lt2_pct": float(
                np.asarray(
                    [bool(row["joint_fast_valid_and_rmsd_lt2"]) for row in rows],
                    dtype=float,
                ).mean()
                * 100
            ),
            "fast_valid_oracle_coverage_pct": float(finite.mean() * 100),
            "fast_valid_oracle_median_rmsd": (
                float(np.median(valid_oracle[finite])) if bool(finite.any()) else None
            ),
        }
        if "num_rmsd_lt2_candidates" in rows[0]:
            k2 = np.asarray(
                [int(row["num_rmsd_lt2_candidates"]) for row in rows],
                dtype=float,
            )
            fast_k2 = np.asarray(
                [int(row["num_fast_valid_rmsd_lt2_candidates"]) for row in rows],
                dtype=float,
            )
            summary["candidate_set"].update(
                {
                    "mean_rmsd_lt2_candidates": float(k2.mean()),
                    "complexes_with_any_rmsd_lt2_pct": float((k2 > 0).mean() * 100),
                    "mean_fast_valid_rmsd_lt2_candidates": float(fast_k2.mean()),
                    "complexes_with_any_fast_valid_rmsd_lt2_pct": float(
                        (fast_k2 > 0).mean() * 100
                    ),
                }
            )
    return summary


def candidate_ids(dataset: str, smiles_by_id: dict[str, str]) -> list[str]:
    ids = sorted(smiles_by_id)
    if dataset == "astex":
        return ids
    return ids


def find_file(root: Path, complex_id: str, kind: str) -> Path | None:
    cid = complex_id.lower()
    suffixes = (".sdf", ".mol2") if kind == "ligand" else (".pdb",)
    hits: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        rel = str(path.relative_to(root)).lower()
        stem = path.stem.lower()
        if cid not in rel and cid.split("_")[0] not in rel:
            continue
        if kind == "protein":
            if "protein" in stem and "predicted" not in stem:
                hits.insert(0, path)
            elif "holo_aligned_predicted_protein" in stem:
                hits.append(path)
            elif "protein" in stem:
                hits.append(path)
        else:
            if "ligand" in stem or "crystal" in stem:
                hits.append(path)
    if not hits:
        return None
    return sorted(set(hits), key=lambda p: (len(str(p)), str(p)))[0]


def discover_complexes(
    dataset: str,
    data_dir: Path,
    external_dir: Path,
    pocket_centers: dict[str, tuple[float, float, float]],
    benchmark_input_manifest: Path | None = None,
) -> list[ComplexInput]:
    smiles_by_id, input_identity = load_benchmark_inputs(
        dataset,
        external_dir,
        benchmark_input_manifest,
    )
    dir_index = {p.name.lower(): p for p in data_dir.iterdir() if p.is_dir()}
    complexes: list[ComplexInput] = []
    missing_centers: list[str] = []
    for cid in candidate_ids(dataset, smiles_by_id):
        if cid not in pocket_centers:
            missing_centers.append(cid)
            continue
        direct = dir_index.get(cid)
        if direct is None:
            prefix = cid.split("_")[0] + "_"
            direct = next((p for name, p in dir_index.items() if name.startswith(prefix)), None)
        detected = detect_complex_files(direct, direct.name) if direct and direct.exists() else None
        if detected is not None:
            protein, ligand, fmt = detected
        else:
            search_root = direct if direct is not None else data_dir
            protein = find_file(search_root, cid, "protein")
            ligand = find_file(search_root, cid, "ligand")
            if protein is None or ligand is None:
                continue
            fmt = ligand.suffix.lower().lstrip(".")
        complexes.append(
            ComplexInput(
                complex_id=cid,
                protein=protein,
                ligand_ref=ligand,
                ligand_format=fmt,
                smiles=smiles_by_id.get(cid),
                pocket_center=pocket_centers[cid],
                ligand_input_identity_sha256=str(input_identity["per_id"][cid]["sha256"]),
                ligand_input_canonical_smiles=str(
                    input_identity["per_id"][cid]["canonical_heavy_isomeric_smiles"]
                ),
                enforce_benchmark_heavy_atom_policy=benchmark_input_manifest is not None,
            )
        )
    if missing_centers:
        preview = ", ".join(missing_centers[:5])
        raise ValueError(
            f"frozen pocket centers missing for {len(missing_centers)} benchmark IDs "
            f"(first: {preview})"
        )
    return complexes


def _sample_center_jitter(seed: int, sigma: float) -> torch.Tensor:
    """Sample a paired center perturbation without consuming the global RNG."""
    jitter_generator = torch.Generator(device="cpu")
    jitter_generator.manual_seed(seed)
    return sigma * torch.randn(3, dtype=torch.float32, generator=jitter_generator)


def evaluate_one(
    model: torch.nn.Module,
    item: ComplexInput,
    *,
    dataset: str,
    confidence_model: torch.nn.Module | None,
    device: torch.device,
    num_samples: int,
    num_steps: int,
    sigma: float,
    sigma_list: list[float],
    sigma_counts: list[int],
    center_jitter_sigma: float,
    pocket_cutoff: float,
    pose_objective: str,
    score_rot_sigma_max: float,
    score_alpha_min: float,
    time_schedule: str,
    schedule_power: float,
    vina_guidance_scale: float,
    vina_guidance_start_t: float,
    vina_guidance_ramp_power: float,
    vina_guidance_max_force: float,
    vina_guidance_max_velocity: float,
    vina_guidance_max_angular_velocity: float,
    vina_guidance_protein_shell: float,
    vina_guidance_w_strain: float,
    unified_guidance_scale: float,
    unified_guidance_start_t: float,
    unified_guidance_ramp_power: float,
    unified_guidance_max_force: float,
    unified_guidance_max_velocity: float,
    unified_guidance_max_angular_velocity: float,
    unified_guidance_max_atom_displacement: float,
    unified_guidance_max_backtracks: int,
    unified_guidance_protein_shell: float,
    unified_guidance_receptor_policy: str,
    unified_guidance_mode: str,
    prior_pool_size: int,
    seed: int,
    refine: str,
    pose_dir: Path | None,
    trajectory_dir: Path | None,
    require_full_ligand_atom_mapping: bool,
    selector_profile: str = DEFAULT_SELECTOR_PROFILE,
    unified_guidance_steric_radius_scale: float | None = None,
    unified_guidance_chiral_improper_scale: float = 1.0,
    fk_constraint_beta: float = 0.0,
    fk_resample_times: tuple[float, ...] = (),
    fk_resample_method: str = "systematic",
    fk_resample_translation_jitter: float = 0.0,
    fk_resample_rotation_jitter: float = 0.0,
    translation_sde_base_sigma: float = 0.0,
    ligand_conformer_seed: int | None = None,
) -> dict:
    if selector_profile not in SELECTOR_PROFILES:
        raise ValueError(f"unknown selector profile: {selector_profile!r}")
    cluster_free_profile = selector_profile == "confidence_cluster_free"
    candidate_only_profile = selector_profile == "candidate_only"
    if cluster_free_profile and confidence_model is None:
        raise ValueError("confidence_cluster_free selector profile requires a confidence model")
    if (cluster_free_profile or candidate_only_profile) and trajectory_dir is not None:
        raise ValueError(
            f"{selector_profile} selector profile does not support trajectory export"
        )
    if ligand_conformer_seed is not None and ligand_conformer_seed < 0:
        raise ValueError("ligand_conformer_seed must be non-negative when specified")
    if vina_guidance_scale != 0.0 and unified_guidance_scale != 0.0:
        raise ValueError("Vina and unified guidance cannot be active together")
    if not math.isfinite(fk_constraint_beta) or fk_constraint_beta < 0.0:
        raise ValueError("fk_constraint_beta must be finite and non-negative")
    if not math.isfinite(translation_sde_base_sigma) or translation_sde_base_sigma < 0.0:
        raise ValueError("translation_sde_base_sigma must be finite and non-negative")
    translation_sde_enabled = translation_sde_base_sigma > 0.0
    fk_enabled = fk_constraint_beta > 0.0
    if fk_enabled != bool(fk_resample_times):
        raise ValueError(
            "positive fk_constraint_beta and non-empty fk_resample_times are required together"
        )
    if fk_enabled and (vina_guidance_scale != 0.0 or unified_guidance_scale != 0.0):
        raise ValueError("FK resampling cannot be combined with Vina or unified guidance")
    if fk_enabled and pose_objective.lower() != "linear_fm":
        raise ValueError("experimental FK resampling currently requires linear_fm")
    if translation_sde_enabled and pose_objective.lower() != "linear_fm":
        raise ValueError("score-corrected translation SDE currently requires linear_fm")
    if translation_sde_enabled and (
        vina_guidance_scale != 0.0 or unified_guidance_scale != 0.0
    ):
        raise ValueError(
            "score-corrected translation SDE cannot be combined with gradient guidance"
        )
    if fk_resample_method not in {"systematic", "multinomial"}:
        raise ValueError(f"unknown FK resample method: {fk_resample_method!r}")
    if (
        not math.isfinite(fk_resample_translation_jitter)
        or not math.isfinite(fk_resample_rotation_jitter)
        or fk_resample_translation_jitter < 0.0
        or fk_resample_rotation_jitter < 0.0
    ):
        raise ValueError("FK post-resampling jitter scales must be finite and non-negative")
    if not fk_enabled and (
        fk_resample_translation_jitter != 0.0 or fk_resample_rotation_jitter != 0.0
    ):
        raise ValueError("FK post-resampling jitter requires FK resampling")
    if translation_sde_enabled and (
        fk_resample_translation_jitter != 0.0 or fk_resample_rotation_jitter != 0.0
    ):
        raise ValueError(
            "score-corrected translation SDE cannot be combined with heuristic FK jitter"
        )
    if unified_guidance_scale < 0.0:
        raise ValueError("unified_guidance_scale must be non-negative")
    if (
        unified_guidance_steric_radius_scale is not None
        and unified_guidance_steric_radius_scale <= 0.0
    ):
        raise ValueError("unified_guidance_steric_radius_scale must be positive")
    if unified_guidance_chiral_improper_scale < 0.0:
        raise ValueError("unified_guidance_chiral_improper_scale must be non-negative")
    if unified_guidance_mode not in {"operator_split", "normalized_drift"}:
        raise ValueError(f"unknown unified guidance mode: {unified_guidance_mode!r}")
    if prior_pool_size and prior_pool_size < num_samples:
        raise ValueError("prior_pool_size must be zero or at least num_samples")
    if prior_pool_size and sigma_list:
        raise ValueError("shared prior pools currently require one scalar sigma")
    torch.manual_seed(seed)
    translation_sde_seed = (
        (int(seed) ^ TRANSLATION_SDE_SEED_XOR) & ((1 << 63) - 1)
        if translation_sde_enabled
        else None
    )
    translation_sde_generator = None
    if translation_sde_seed is not None:
        translation_sde_generator = torch.Generator(device=device)
        translation_sde_generator.manual_seed(translation_sde_seed)
    mol_ref = load_ref_ligand(item.ligand_ref, item.ligand_format)
    ligand_input = item.smiles if item.smiles else str(item.ligand_ref)
    effective_ligand_conformer_seed = (
        seed if ligand_conformer_seed is None else ligand_conformer_seed
    )
    if item.smiles and item.enforce_benchmark_heavy_atom_policy:
        mol_in, _ = load_benchmark_ligand(
            ligand_input,
            random_seed=effective_ligand_conformer_seed,
        )
    else:
        from effdock.inference.preprocess import load_ligand

        mol_in, _ = load_ligand(
            ligand_input,
            random_seed=effective_ligand_conformer_seed,
        )
    dock_idx, ref_idx, match_method = match_atoms(mol_ref, mol_in)
    mapping_metadata = full_heavy_atom_mapping_metadata(
        mol_ref,
        mol_in,
        dock_idx,
        ref_idx,
        match_method,
    )
    if require_full_ligand_atom_mapping and not mapping_metadata["accepted"]:
        raise BenchmarkInputMismatchError(
            "benchmark_ligand_atom_mapping_mismatch",
            "frozen benchmark input requires a complete connectivity-preserving atom map",
            details={
                **mapping_metadata,
                "input_identity_sha256": item.ligand_input_identity_sha256,
            },
        )
    ref_pos_abs = torch.tensor(mol_ref.GetConformer().GetPositions(), dtype=torch.float32)
    pocket_center = torch.tensor(item.pocket_center, dtype=torch.float32)
    if center_jitter_sigma > 0.0:
        # Keep the sampling RNG paired across jitter conditions.  A dedicated
        # CPU generator also makes 1A and 2A use the same perturbation direction
        # for a given complex, differing only in magnitude.
        pocket_center = pocket_center + _sample_center_jitter(seed, center_jitter_sigma)
    graph, lig_data, meta = preprocess_complex(
        item.protein,
        mol_in,
        pocket_center=pocket_center,
        pocket_cutoff=pocket_cutoff,
    )
    guidance_fn = None
    guidance_operator_split = False
    guidance_direct_drift = False
    guidance_is_unified = False
    guidance_metadata: dict | None = None
    fk_resampler: FeynmanKacConstraintResampler | None = None
    if vina_guidance_scale != 0.0:
        if vina_guidance_scale < 0.0:
            raise ValueError("vina_guidance_scale must be non-negative")
        from effdock.evaluation.vina_guidance import VinaGuidanceConfig, build_vina_guidance

        guidance_fn = build_vina_guidance(
            mol_in,
            item.protein,
            pocket_center=meta["pocket_center"],
            frag_id=lig_data["fragment_id"],
            device=device,
            protein_shell_cutoff=vina_guidance_protein_shell,
            config=VinaGuidanceConfig(
                start_t=vina_guidance_start_t,
                ramp_power=vina_guidance_ramp_power,
                max_atom_force=vina_guidance_max_force,
                max_translation_velocity=vina_guidance_max_velocity,
                max_angular_velocity=vina_guidance_max_angular_velocity,
                w_strain=vina_guidance_w_strain,
            ),
        )
    elif unified_guidance_scale != 0.0 or fk_enabled:
        from effdock.guidance import (
            GuidanceEnergyConfig,
            PhysicalEnergyConfig,
            UnifiedGuidance,
            UnifiedGuidanceConfig,
            build_physical_system,
        )
        from effdock.guidance.parameterization import guidance_parameter_identity

        system = build_physical_system(
            mol_in,
            item.protein,
            fragment_id=lig_data["fragment_id"],
            near_coords=meta["pocket_center"].view(1, 3),
            protein_cutoff=unified_guidance_protein_shell,
            coordinate_origin=meta["pocket_center"],
            receptor_policy=unified_guidance_receptor_policy,
        )
        receptor_metadata = receptor_guidance_metadata(
            system,
            unified_guidance_receptor_policy,
        )
        physical_config = PhysicalEnergyConfig(
            steric_radius_scale=(
                PhysicalEnergyConfig().steric_radius_scale
                if unified_guidance_steric_radius_scale is None
                else unified_guidance_steric_radius_scale
            ),
            chiral_improper_scale=unified_guidance_chiral_improper_scale,
        )
        term_coefficients = {
            "protein_ligand_steric_radius_scale": physical_config.steric_radius_scale,
            "ligand_chiral_improper_scale": physical_config.chiral_improper_scale,
        }
        if fk_enabled:
            coupling_metadata = {
                "mode": (
                    "feynman_kac_constraint_resampling_translation_sde"
                    if translation_sde_enabled
                    else "feynman_kac_constraint_resampling"
                ),
                "weight": "exp(-beta * delta_constraint_potential)",
                "beta": fk_constraint_beta,
                "potential_schedule": "difference",
                "endpoint_estimator": "constant_velocity_euler_terminal",
                "requested_resample_times": list(fk_resample_times),
                "resample_method": fk_resample_method,
                "group_by_prior_sigma": True,
                "constraint_terms": list(DEFAULT_FK_CONSTRAINT_TERMS),
                "excluded_term_classes": [
                    "attractive_physical_terms",
                    "interaction_terms",
                    "learned_confidence_or_oracle_terms",
                ],
                "dynamics": (
                    "translation_score_corrected_sde_deterministic_so3"
                    if translation_sde_enabled
                    else "deterministic_flow_without_score_corrected_sde"
                ),
                "translation_sde": {
                    "base_sigma": translation_sde_base_sigma,
                    "seed": translation_sde_seed,
                    "seed_domain_separation": "sampling_seed XOR 0x54534445",
                    "diffusion_schedule": "g(t)=base_sigma*(1-t)",
                    "score": "(t*v-T)/((1-t)*prior_sigma^2)",
                    "prior": "pocket_centered_isotropic_gaussian_translation",
                    "rotation_dynamics": "deterministic_so3_flow",
                },
                "post_resampling_mutation": {
                    "classification": fk_post_resampling_mutation_classification(
                        translation_sde_enabled=translation_sde_enabled,
                        translation_jitter=fk_resample_translation_jitter,
                        rotation_jitter=fk_resample_rotation_jitter,
                    ),
                    "translation_sigma": fk_resample_translation_jitter,
                    "rotation_sigma_radians": fk_resample_rotation_jitter,
                },
            }
        else:
            coupling_metadata = {
                "mode": unified_guidance_mode,
                "normalized_space": (
                    "pose_atom_induced_velocity_rms"
                    if unified_guidance_mode == "normalized_drift"
                    else None
                ),
                "time_ramp_quadrature": (
                    "interval_average"
                    if unified_guidance_mode == "normalized_drift"
                    else "right_endpoint"
                ),
            }
        guidance_metadata = {
            "parameter_set": guidance_parameter_identity(),
            "runtime_term_coefficients": term_coefficients,
            "coupling": coupling_metadata,
            "system_reference_sha256": physical_system_reference_sha256(system),
            "topology_reference_sha256": system.topology.reference_sha256(),
            "interaction_reference_sha256": (
                system.interaction_topology.reference_sha256()
                if system.interaction_topology is not None
                else None
            ),
            "term_counts": (
                system.interaction_topology.term_counts()
                if system.interaction_topology is not None
                else {}
            ),
            "receptor": receptor_metadata,
        }
        system = system.to(device=device, dtype=torch.float32)
        if fk_enabled:
            fk_resampler = FeynmanKacConstraintResampler(
                system,
                FKConstraintConfig(
                    beta=fk_constraint_beta,
                    resample_method=fk_resample_method,
                    seed=seed,
                    dynamics=(
                        "translation_score_corrected_sde_deterministic_so3"
                        if translation_sde_enabled
                        else "deterministic_flow_without_score_corrected_sde"
                    ),
                    translation_sde_base_sigma=translation_sde_base_sigma,
                    energy=physical_config,
                ),
            )
        else:
            guidance_fn = UnifiedGuidance(
                system,
                UnifiedGuidanceConfig(
                    start_t=unified_guidance_start_t,
                    ramp_power=unified_guidance_ramp_power,
                    max_atom_force=unified_guidance_max_force,
                    max_translation_velocity=unified_guidance_max_velocity,
                    max_angular_velocity=unified_guidance_max_angular_velocity,
                    max_atom_displacement=unified_guidance_max_atom_displacement,
                    max_backtracks=unified_guidance_max_backtracks,
                    energy=GuidanceEnergyConfig(physical=physical_config),
                ),
            )
            guidance_is_unified = True
            guidance_operator_split = unified_guidance_mode == "operator_split"
            guidance_direct_drift = unified_guidance_mode == "normalized_drift"
    initial_T_frag = None
    initial_q_frag = None
    prior_pool_sha256 = ""
    if prior_pool_size:
        prior_T, prior_q = sample_shared_prior_states(
            prior_pool_size,
            int(meta["num_frag"]),
            lig_data["frag_sizes"],
            translation_sigma=sigma,
            seed=seed,
        )
        prior_pool_sha256 = shared_prior_sha256(prior_T, prior_q)
        initial_T_frag = prior_T[:num_samples]
        initial_q_frag = prior_q[:num_samples]
    if sigma_list:
        results = sample_unified_multi_sigma(
            model,
            graph,
            lig_data,
            meta,
            sigma_list=sigma_list,
            samples_per_sigma=sigma_counts,
            num_steps=num_steps,
            time_schedule=time_schedule,
            schedule_power=schedule_power,
            pose_objective=pose_objective,
            score_rot_sigma_max=score_rot_sigma_max,
            score_alpha_min=score_alpha_min,
            device=device,
            save_traj=trajectory_dir is not None,
            guidance_fn=guidance_fn,
            guidance_scale=(unified_guidance_scale if guidance_is_unified else vina_guidance_scale),
            guidance_min_t=(
                unified_guidance_start_t if guidance_is_unified else vina_guidance_start_t
            ),
            guidance_operator_split=guidance_operator_split,
            guidance_direct_drift=guidance_direct_drift,
            translation_sde_base_sigma=translation_sde_base_sigma,
            translation_sde_generator=translation_sde_generator,
            fk_resample_times=fk_resample_times if fk_enabled else None,
            fk_resampler=fk_resampler,
            fk_resample_trans_sigma=fk_resample_translation_jitter,
            fk_resample_rot_sigma=fk_resample_rotation_jitter,
        )
    else:
        results = sample_unified(
            model,
            graph,
            lig_data,
            meta,
            num_samples=num_samples,
            num_steps=num_steps,
            translation_sigma=sigma,
            time_schedule=time_schedule,
            schedule_power=schedule_power,
            pose_objective=pose_objective,
            score_rot_sigma_max=score_rot_sigma_max,
            score_alpha_min=score_alpha_min,
            device=device,
            save_traj=trajectory_dir is not None,
            guidance_fn=guidance_fn,
            guidance_scale=(unified_guidance_scale if guidance_is_unified else vina_guidance_scale),
            guidance_min_t=(
                unified_guidance_start_t if guidance_is_unified else vina_guidance_start_t
            ),
            guidance_operator_split=guidance_operator_split,
            guidance_direct_drift=guidance_direct_drift,
            translation_sde_base_sigma=translation_sde_base_sigma,
            translation_sde_generator=translation_sde_generator,
            initial_T_frag=initial_T_frag,
            initial_q_frag=initial_q_frag,
            fk_resample_times=fk_resample_times if fk_enabled else None,
            fk_resampler=fk_resampler,
            fk_resample_trans_sigma=fk_resample_translation_jitter,
            fk_resample_rot_sigma=fk_resample_rotation_jitter,
        )
    fk_diagnostics = fk_resampler.diagnostics() if fk_resampler is not None else {}
    if fk_resampler is not None:
        fk_initial_ancestors = torch.tensor(
            [int(result["initial_sample_index"]) for result in results],
            dtype=torch.long,
        )
        resampler_ancestors = fk_resampler.final_initial_ancestors()
        if resampler_ancestors is None or not torch.equal(
            fk_initial_ancestors,
            resampler_ancestors,
        ):
            raise RuntimeError("FK sampler and resampler genealogy disagree")
    poses = [result["atom_pos_pred"].detach().cpu() for result in results]
    poses = apply_refinement(refine, poses, mol_in, meta["pocket_center"])
    diversity_metrics = pose_diversity_metrics(poses, mol_in)

    ref_pos = ref_pos_abs - meta["pocket_center"].cpu()
    if not dock_idx:
        raise RuntimeError("atom matching failed")
    ref_pos = ref_pos.index_select(0, torch.as_tensor(ref_idx, dtype=torch.long))
    rmsd_results = [
        compute_pose_rmsd_with_method(
            pose.cpu(),
            ref_pos,
            meta["pocket_center"].cpu(),
            dock_idx,
            mol_in,
            mol_ref,
        )
        for pose in poses
    ]
    rmsds = [value for value, _ in rmsd_results]
    rmsd_methods = [method for _, method in rmsd_results]
    first_i = 0
    oracle_i = select_pose("oracle", rmsds)
    selector_indices = {"first": first_i, "oracle": oracle_i}
    if candidate_only_profile:
        selector_indices = {"selected": first_i, **selector_indices}
    vina_scores: list[dict[str, float]] | None = None
    vina_i: int | None = None
    if not cluster_free_profile and not candidate_only_profile:
        vina_scores = score_poses(
            mol_in,
            poses,
            item.protein,
            pocket_center=meta["pocket_center"].cpu(),
            frag_id=lig_data["fragment_id"].cpu(),
        )
        vina_i = select_by_score([score["total"] for score in vina_scores])
        selector_indices = {"first": first_i, "vina": vina_i, "oracle": oracle_i}
    confidence_scores: list[dict[str, float]] | None = None
    candidate_ensemble_hash = candidate_ensemble_sha256(poses)
    confidence_score_ledger = ""
    if confidence_model is not None and not candidate_only_profile:
        from effdock.confidence.runtime import sample_sigmas, score_poses_with_confidence

        confidence_scores = score_poses_with_confidence(
            confidence_model,
            model,
            graph,
            lig_data,
            meta,
            poses,
            sigma=sample_sigmas(results, sigma),
            device=device,
        )
        if cluster_free_profile:
            confidence_indices = select_confidence_cluster_free(
                poses,
                confidence_scores,
                graph,
                meta["pocket_center"],
            )
            selector_indices.update(confidence_indices)
        else:
            from effdock.confidence.selectors import select_confidence_poses

            confidence_indices = select_confidence_poses(
                poses, confidence_scores, graph, meta["pocket_center"]
            )
            selector_indices["confidence"] = confidence_indices["confidence"]
            selector_indices["confidence_filter"] = confidence_indices["confidence_filter_v1"]
            selector_indices["confidence_final"] = confidence_indices[
                "pair_gate_density_rank_vote_plclash_ambig"
            ]
        confidence_score_ledger = confidence_score_ledger_json(confidence_scores)

    absolute_poses = [pose.cpu() + meta["pocket_center"].cpu() for pose in poses]
    prot = build_protein_vina_inputs(
        item.protein,
        torch.cat(absolute_poses, dim=0),
        cutoff=10.0,
    )
    bounds = ligand_bounds(mol_in)
    lig_atomic_nums = torch.tensor([atom.GetAtomicNum() for atom in mol_in.GetAtoms()])
    lig_r = vdw_radii(lig_atomic_nums)
    prot_r = vdw_radii(prot["atomic_nums"])

    candidate_validity = [
        check_validity(
            pose,
            bounds,
            prot_xyz=prot["coords"],
            prot_r=prot_r,
            lig_r=lig_r,
            return_terms=True,
        )
        for pose in absolute_poses
    ]
    fast_valid_indices = [
        index for index, terms in enumerate(candidate_validity) if bool(terms["valid"])
    ]
    num_rmsd_lt2_candidates = sum(float(rmsd) < 2.0 for rmsd in rmsds)
    num_fast_valid_rmsd_lt2_candidates = sum(
        float(rmsds[index]) < 2.0 for index in fast_valid_indices
    )
    fast_valid_oracle_i = (
        min(fast_valid_indices, key=lambda index: rmsds[index]) if fast_valid_indices else -1
    )
    fast_valid_oracle_rmsd = (
        float(rmsds[fast_valid_oracle_i]) if fast_valid_oracle_i >= 0 else float("inf")
    )

    validity: dict[str, dict[str, bool]] = {}
    saved_pose_sha256: dict[str, str] = {}
    all_poses_sdf = ""
    all_poses_sdf_sha256 = ""
    if pose_dir is not None:
        all_poses_path = pose_dir / "all_poses" / f"{item.complex_id}.sdf"
        all_poses_path.parent.mkdir(parents=True, exist_ok=True)
        candidate_scores = [
            {
                **(vina_scores[index] if vina_scores is not None else {}),
                **(confidence_scores[index] if confidence_scores is not None else {}),
            }
            for index in range(len(poses))
        ]
        write_multi_sdf(
            mol_in,
            poses,
            meta["pocket_center"],
            all_poses_path,
            scores=candidate_scores,
            props={
                "dataset": dataset,
                "complex_id": item.complex_id,
                "sampling_seed": seed,
                "ligand_conformer_seed": effective_ligand_conformer_seed,
                "num_samples": num_samples,
                "num_steps": num_steps,
                "candidate_ensemble_sha256": candidate_ensemble_hash,
            },
            per_pose_props=[
                {
                    "sample_sigma": float(results[index].get("sample_sigma", sigma)),
                    "fast_valid": bool(candidate_validity[index]["valid"]),
                    **(
                        {"fk_initial_sample_index": int(results[index]["initial_sample_index"])}
                        if fk_enabled
                        else {}
                    ),
                }
                for index in range(len(poses))
            ],
        )
        all_poses_sdf = str(all_poses_path)
        all_poses_sdf_sha256 = file_sha256(all_poses_path)
    if cluster_free_profile:
        saved_selectors = {"confidence", "confidence_filter"}
    elif candidate_only_profile:
        saved_selectors = {"selected"}
    else:
        saved_selectors = set(selector_indices)
    for selector, index in selector_indices.items():
        validity[selector] = candidate_validity[index]
        if pose_dir is not None and selector in saved_selectors:
            pose_path = pose_dir / selector / f"{item.complex_id}.sdf"
            _write_pose(
                mol_in,
                poses[index],
                meta["pocket_center"],
                pose_path,
            )
            saved_pose_sha256[selector] = file_sha256(pose_path)
    if (
        pose_dir is not None
        and not cluster_free_profile
        and not candidate_only_profile
        and fast_valid_oracle_i >= 0
    ):
        _write_pose(
            mol_in,
            poses[fast_valid_oracle_i],
            meta["pocket_center"],
            pose_dir / "fast_valid_oracle" / f"{item.complex_id}.sdf",
        )

    guidance_stats = (
        guidance_fn.diagnostics() if guidance_is_unified and guidance_fn is not None else {}
    )
    guidance_direct_step_trace = (
        guidance_fn.direct_step_trace() if guidance_direct_drift and guidance_fn is not None else []
    )

    row = {
        "id": item.complex_id,
        "selector_profile": selector_profile,
        "protein": str(item.protein),
        "ligand_ref": str(item.ligand_ref),
        "protein_sha256": file_sha256(item.protein),
        "ligand_reference_sha256": file_sha256(item.ligand_ref),
        "saved_pose_sha256_json": _canonical_json(saved_pose_sha256),
        "all_poses_sdf": all_poses_sdf,
        "all_poses_sdf_sha256": all_poses_sdf_sha256,
        "all_poses_count": len(poses) if pose_dir is not None else 0,
        "num_samples": num_samples,
        **(
            {
                "selected_index": first_i,
                "selected_rmsd": float(rmsds[first_i]),
            }
            if candidate_only_profile
            else {}
        ),
        "first_index": first_i,
        **({"vina_index": vina_i} if vina_i is not None else {}),
        "oracle_index": oracle_i,
        "first_rmsd": float(rmsds[first_i]),
        **({"vina_rmsd": float(rmsds[vina_i])} if vina_i is not None else {}),
        "oracle_rmsd": float(rmsds[oracle_i]),
        "candidate_rmsds_json": _canonical_json([float(rmsd) for rmsd in rmsds]),
        "candidate_rmsd_method_json": _canonical_json(rmsd_methods),
        "num_mapped_index_rmsd_fallback_candidates": sum(
            method == "mapped_index_fallback" for method in rmsd_methods
        ),
        "candidate_fast_valid_json": _canonical_json(
            [bool(terms["valid"]) for terms in candidate_validity]
        ),
        "pose_diversity_contract": POSE_DIVERSITY_CONTRACT,
        "pose_diversity_round_decimals": POSE_DIVERSITY_ROUND_DECIMALS,
        **diversity_metrics,
        "num_fast_valid_candidates": len(fast_valid_indices),
        "num_rmsd_lt2_candidates": num_rmsd_lt2_candidates,
        "fraction_rmsd_lt2_candidates": num_rmsd_lt2_candidates / len(rmsds),
        "num_fast_valid_rmsd_lt2_candidates": num_fast_valid_rmsd_lt2_candidates,
        "fast_valid_oracle_index": fast_valid_oracle_i,
        "fast_valid_oracle_rmsd": fast_valid_oracle_rmsd,
        "joint_fast_valid_and_rmsd_lt2": bool(fast_valid_oracle_rmsd < 2.0),
        **(
            {
                "vina_score": vina_scores[vina_i]["vina"],
                "vina_strain": vina_scores[vina_i]["strain"],
                "vina_total": vina_scores[vina_i]["total"],
            }
            if vina_scores is not None and vina_i is not None
            else {}
        ),
        "mean_sample_rmsd": float(np.mean(rmsds)),
        **{
            f"{selector}_fast_{term}": value
            for selector, terms in validity.items()
            for term, value in terms.items()
        },
        "match_method": match_method,
        "num_match_atoms": len(dock_idx),
        "num_input_atoms": mol_in.GetNumAtoms(),
        "num_ref_atoms": mol_ref.GetNumAtoms(),
        "full_heavy_atom_bijection": bool(mapping_metadata["accepted"]),
        "ligand_graph_relation": str(mapping_metadata["relation"]),
        "ligand_mapping_metadata_json": _canonical_json(mapping_metadata),
        "exact_full_heavy_atom_graph": mapping_metadata["relation"] == "exact_graph",
        "ligand_input_identity_sha256": item.ligand_input_identity_sha256,
        "ligand_input_canonical_smiles": item.ligand_input_canonical_smiles or "",
        "prior_pool_size": prior_pool_size,
        "sampling_seed": seed,
        "ligand_conformer_seed": effective_ligand_conformer_seed,
        "prior_pool_sha256": prior_pool_sha256,
        "guidance_mode": (
            (
                "feynman_kac_constraint_resampling_translation_sde"
                if translation_sde_enabled
                else "feynman_kac_constraint_resampling"
            )
            if fk_enabled
            else (
                "translation_score_corrected_sde"
                if translation_sde_enabled
                else (
                    "unified_operator_split"
                    if guidance_operator_split
                    else (
                        "unified_normalized_drift"
                        if guidance_direct_drift
                        else ("vina_additive" if vina_guidance_scale != 0.0 else "none")
                    )
                )
            )
        ),
        "sampling_dynamics": (
            "translation_score_corrected_sde_deterministic_so3"
            if translation_sde_enabled
            else "deterministic_ode"
        ),
        "translation_sde_base_sigma": translation_sde_base_sigma,
        "translation_sde_seed": translation_sde_seed,
        **(
            {
                "fk_constraint_beta": fk_constraint_beta,
                "fk_resample_times_json": _canonical_json(list(fk_resample_times)),
                "fk_resample_method": fk_resample_method,
                "fk_resample_translation_jitter": fk_resample_translation_jitter,
                "fk_resample_rotation_jitter": fk_resample_rotation_jitter,
                "fk_num_resampling_events": int(fk_diagnostics.get("num_resampling_events", 0)),
                "fk_final_unique_initial_ancestors": fk_diagnostics.get(
                    "final_unique_initial_ancestors"
                ),
                "fk_diagnostics_json": _canonical_json(fk_diagnostics),
            }
            if fk_enabled
            else {}
        ),
        "guidance_parameter_sha256": (
            guidance_metadata["parameter_set"]["sha256"] if guidance_metadata is not None else ""
        ),
        "guidance_topology_reference_sha256": (
            guidance_metadata["topology_reference_sha256"] if guidance_metadata is not None else ""
        ),
        "guidance_system_reference_sha256": (
            guidance_metadata["system_reference_sha256"] if guidance_metadata is not None else ""
        ),
        "guidance_interaction_reference_sha256": (
            guidance_metadata["interaction_reference_sha256"]
            if guidance_metadata is not None
            else ""
        ),
        "guidance_receptor_policy": (
            guidance_metadata["receptor"]["mode"] if guidance_metadata is not None else ""
        ),
        "guidance_receptor_policy_identity_sha256": (
            guidance_metadata["receptor"]["identity_sha256"]
            if guidance_metadata is not None
            else ""
        ),
        "guidance_receptor_policy_identity_json": (
            _canonical_json(guidance_metadata["receptor"]["identity"])
            if guidance_metadata is not None
            else ""
        ),
        "guidance_receptor_provenance_json": (
            _canonical_json(guidance_metadata["receptor"]["provenance"])
            if guidance_metadata is not None
            else ""
        ),
        "guidance_metadata_json": (
            _canonical_json(guidance_metadata) if guidance_metadata is not None else ""
        ),
        "guidance_direct_step_trace_json": _canonical_json(guidance_direct_step_trace),
        **{f"guidance_{name}": value for name, value in guidance_stats.items()},
    }
    if confidence_scores is not None:
        row["candidate_ensemble_sha256"] = candidate_ensemble_hash
        row["confidence_candidate_scores_json"] = confidence_score_ledger
        confidence_selectors = (
            ("confidence", "confidence_filter")
            if cluster_free_profile
            else ("confidence", "confidence_filter", "confidence_final")
        )
        for selector in confidence_selectors:
            index = selector_indices[selector]
            row[f"{selector}_index"] = index
            row[f"{selector}_rmsd"] = float(rmsds[index])
            row[f"{selector}_pred_rmsd"] = confidence_scores[index]["confidence_rmsd"]
            row[f"{selector}_pred_success"] = confidence_scores[index]["confidence_success"]
    elif candidate_only_profile:
        row["candidate_ensemble_sha256"] = candidate_ensemble_hash
    if trajectory_dir is not None:
        if confidence_scores is None:
            raise ValueError("--trajectory-dir requires --confidence-checkpoint")
        if refine != "none":
            raise ValueError("trajectory export requires --refine none")
        selected_index = selector_indices["confidence_final"]
        selected_result = results[selected_index]
        trajectory_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "schema_version": 1,
                "dataset": dataset,
                "complex_id": item.complex_id,
                "protein": str(item.protein),
                "ligand_ref": str(item.ligand_ref),
                "selector": "pair_gate_density_rank_vote_plclash_ambig",
                "selected_index": selected_index,
                "seed": seed,
                "num_samples": num_samples,
                "num_steps": num_steps,
                "pocket_cutoff": pocket_cutoff,
                "pocket_center": meta["pocket_center"].detach().cpu(),
                "fragment_id": lig_data["fragment_id"].detach().cpu(),
                "atomic_numbers": torch.tensor(
                    [atom.GetAtomicNum() for atom in mol_in.GetAtoms()], dtype=torch.long
                ),
                "bonds": [
                    (
                        bond.GetBeginAtomIdx(),
                        bond.GetEndAtomIdx(),
                        float(bond.GetBondTypeAsDouble()),
                    )
                    for bond in mol_in.GetBonds()
                ],
                "guidance_metadata": guidance_metadata,
                **(
                    {
                        "fk_diagnostics": fk_diagnostics,
                        "fk_initial_sample_index": int(selected_result["initial_sample_index"]),
                    }
                    if fk_enabled
                    else {}
                ),
                "traj": [
                    frame.detach().cpu() + meta["pocket_center"].detach().cpu()
                    for frame in selected_result["traj"]
                ],
                "traj_times": list(selected_result["traj_times"]),
                "selected_rmsd": float(rmsds[selected_index]),
                "oracle_index": oracle_i,
                "oracle_rmsd": float(rmsds[oracle_i]),
                "confidence_pred_rmsd": confidence_scores[selected_index]["confidence_rmsd"],
                "confidence_pred_success": confidence_scores[selected_index]["confidence_success"],
            },
            trajectory_dir / f"{item.complex_id}.pt",
        )
    return row


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=("astex", "posebusters", "phibench", "foldbench", "openbind"),
        required=True,
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--external-dir", type=Path, default=Path("data/external_test"))
    parser.add_argument(
        "--benchmark-input-manifest",
        type=Path,
        default=None,
        help="Optional versioned ligand-input mapping; FULL-V2 requires this manifest.",
    )
    parser.add_argument(
        "--pocket-centers",
        type=Path,
        required=True,
        help="Frozen JSON mapping benchmark IDs to declared [x,y,z] pocket centers.",
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_DOCKING_CHECKPOINT)
    parser.add_argument(
        "--confidence-checkpoint",
        type=Path,
        default=DEFAULT_CONFIDENCE_CHECKPOINT,
        help="Learned confidence checkpoint evaluated on the same sampled poses.",
    )
    parser.add_argument(
        "--no-confidence",
        action="store_const",
        dest="confidence_checkpoint",
        const=None,
        help="Disable learned confidence evaluation.",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--output-dir", dest="out_dir", type=Path, default=Path("outputs/external_benchmarks")
    )
    parser.add_argument("--num-samples", type=int, default=DEFAULT_NUM_SAMPLES)
    parser.add_argument("--num-steps", type=int, default=DEFAULT_NUM_STEPS)
    parser.add_argument("--sigma", type=float, default=DEFAULT_SIGMA)
    parser.add_argument(
        "--sigma-list",
        type=str,
        default=None,
        help='Multi-sigma sampling, e.g. "2.5,3.0,3.5" or "2.5:14,3.0:13,3.5:13".',
    )
    parser.add_argument("--time-schedule", type=str, default=DEFAULT_TIME_SCHEDULE)
    parser.add_argument("--schedule-power", type=float, default=DEFAULT_SCHEDULE_POWER)
    parser.add_argument("--vina-guidance-scale", type=float, default=0.0)
    parser.add_argument("--vina-guidance-start-t", type=float, default=0.5)
    parser.add_argument("--vina-guidance-ramp-power", type=float, default=1.0)
    parser.add_argument("--vina-guidance-max-force", type=float, default=10.0)
    parser.add_argument("--vina-guidance-max-velocity", type=float, default=5.0)
    parser.add_argument("--vina-guidance-max-angular-velocity", type=float, default=5.0)
    parser.add_argument("--vina-guidance-protein-shell", type=float, default=18.0)
    parser.add_argument("--vina-guidance-w-strain", type=float, default=1.0)
    parser.add_argument(
        "--unified-guidance-scale",
        type=float,
        default=0.0,
        help=(
            "Experimental GuidanceEnergy coupling strength. Mutually exclusive "
            "with Vina guidance; 0 disables it."
        ),
    )
    parser.add_argument(
        "--unified-guidance-mode",
        choices=("operator_split", "normalized_drift"),
        default="operator_split",
        help=(
            "Experimental GuidanceEnergy coupling. operator_split preserves the "
            "guarded post-step corrector; normalized_drift adds a pose-normalized "
            "control field directly to the learned ODE velocity."
        ),
    )
    parser.add_argument("--unified-guidance-start-t", type=float, default=0.5)
    parser.add_argument("--unified-guidance-ramp-power", type=float, default=1.0)
    parser.add_argument("--unified-guidance-max-force", type=float, default=20.0)
    parser.add_argument("--unified-guidance-max-velocity", type=float, default=5.0)
    parser.add_argument(
        "--unified-guidance-max-angular-velocity",
        type=float,
        default=5.0,
    )
    parser.add_argument(
        "--unified-guidance-max-atom-displacement",
        type=float,
        default=0.25,
    )
    parser.add_argument("--unified-guidance-max-backtracks", type=int, default=8)
    parser.add_argument(
        "--unified-guidance-steric-radius-scale",
        type=float,
        default=None,
        help=(
            "Optional runtime override for the protein-ligand steric contact radius "
            "scale used by unified guidance or FK constraints; omitted uses the "
            "versioned EFF-FF default."
        ),
    )
    parser.add_argument(
        "--unified-guidance-chiral-improper-scale",
        type=float,
        default=1.0,
        help=(
            "Dimensionless scale on non-planar ligand improper restraints used "
            "by unified guidance or FK constraints."
        ),
    )
    parser.add_argument("--unified-guidance-protein-shell", type=float, default=18.0)
    parser.add_argument(
        "--unified-guidance-receptor-policy",
        choices=UNIFIED_GUIDANCE_RECEPTOR_POLICIES,
        default="fail_closed",
        help=(
            "Receptor chemistry admission used when unified guidance or FK constraints "
            "are active. The default preserves the historical fail-closed contract."
        ),
    )
    parser.add_argument(
        "--translation-sde-base-sigma",
        type=float,
        default=0.0,
        help=(
            "Experimental linear-FM score-corrected diffusion scale in angstrom. "
            "Uses g(t)=sigma*(1-t) for Gaussian fragment translations only; "
            "SO(3) rotations remain deterministic. 0 disables it."
        ),
    )
    parser.add_argument(
        "--fk-constraint-beta",
        type=float,
        default=0.0,
        help=(
            "Experimental constraint-only Feynman-Kac inverse temperature beta. "
            "0 disables FK; larger values more strongly favor lower potential."
        ),
    )
    parser.add_argument(
        "--fk-resample-times",
        type=parse_fk_resample_times,
        default=(),
        metavar="T1,T2,...",
        help=(
            "Strictly increasing resampling times inside (0,1). Required with "
            "positive --fk-constraint-beta."
        ),
    )
    parser.add_argument(
        "--fk-resample-method",
        choices=("systematic", "multinomial"),
        default="systematic",
    )
    parser.add_argument(
        "--fk-resample-translation-jitter",
        type=float,
        default=0.0,
        help=(
            "Optional post-resampling translation mutation sigma in angstrom; "
            "heuristic, not the paper's marginal-preserving SDE."
        ),
    )
    parser.add_argument(
        "--fk-resample-rotation-jitter",
        type=float,
        default=0.0,
        help=(
            "Optional post-resampling rotation-vector mutation sigma in radians; "
            "heuristic, not the paper's marginal-preserving SDE."
        ),
    )
    parser.add_argument(
        "--prior-pool-size",
        type=int,
        default=0,
        help=(
            "Generate this many deterministic initial poses and use the first "
            "--num-samples, enabling nested fixed-budget comparisons."
        ),
    )
    parser.add_argument("--center-jitter-sigma", type=float, default=0.0)
    parser.add_argument("--pocket-cutoff", type=float, default=DEFAULT_POCKET_CUTOFF)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--ligand-conformer-seed",
        type=int,
        default=None,
        help=(
            "Optional deterministic ligand conformer seed independent of the per-complex "
            "sampling seed. Omitted preserves the historical coupled-seed behavior."
        ),
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--only-id",
        action="append",
        default=[],
        help="Evaluate only this ID while preserving its full-dataset seed; repeatable.",
    )
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument(
        "--expected-discovered-count",
        type=int,
        default=None,
        help=(
            "Fail before sampling unless filesystem discovery yields exactly this many "
            "complexes. Intended for frozen full-cohort runs."
        ),
    )
    parser.add_argument(
        "--require-complete-success",
        action="store_true",
        help="Write the shard summary, then exit nonzero if any assigned complex failed.",
    )
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--protocol-id", type=str, default=None)
    parser.add_argument(
        "--selector-profile",
        choices=SELECTOR_PROFILES,
        default=DEFAULT_SELECTOR_PROFILE,
        help=(
            "Selector/output contract. confidence_cluster_free skips legacy Torch-Vina "
            "ranking and pairwise cluster/density confidence selection; candidate_only "
            "skips all Torch-Vina and learned-confidence selection."
        ),
    )
    parser.add_argument(
        "--eligibility-manifest",
        type=Path,
        default=None,
        help="Optional frozen chemistry-eligibility manifest recorded in run provenance.",
    )
    parser.add_argument(
        "--require-full-ligand-atom-mapping",
        "--require-exact-ligand-graph",
        dest="require_full_ligand_atom_mapping",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Fail before sampling unless input/reference have a complete element- and "
            "connectivity-preserving heavy-atom bijection."
        ),
    )
    parser.add_argument(
        "--save-selected-poses",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Save the complete all_poses multi-record SDF and selector-specific "
            "single-pose convenience SDFs."
        ),
    )
    parser.add_argument(
        "--trajectory-dir",
        type=Path,
        default=None,
        help="Optionally save the actual confidence-selected ODE trajectory as a PT bundle.",
    )
    parser.add_argument("--refine", choices=("none", "mmff"), default="none")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.vina_guidance_scale != 0.0 and args.unified_guidance_scale != 0.0:
        parser.error("--vina-guidance-scale and --unified-guidance-scale are exclusive")
    if not math.isfinite(args.fk_constraint_beta) or args.fk_constraint_beta < 0.0:
        parser.error("--fk-constraint-beta must be finite and non-negative")
    if (
        not math.isfinite(args.translation_sde_base_sigma)
        or args.translation_sde_base_sigma < 0.0
    ):
        parser.error("--translation-sde-base-sigma must be finite and non-negative")
    translation_sde_enabled = args.translation_sde_base_sigma > 0.0
    fk_enabled = args.fk_constraint_beta > 0.0
    if fk_enabled != bool(args.fk_resample_times):
        parser.error(
            "positive --fk-constraint-beta and non-empty --fk-resample-times are required together"
        )
    if fk_enabled and (args.vina_guidance_scale != 0.0 or args.unified_guidance_scale != 0.0):
        parser.error("FK resampling cannot be combined with Vina or unified guidance")
    if (
        not math.isfinite(args.fk_resample_translation_jitter)
        or not math.isfinite(args.fk_resample_rotation_jitter)
        or args.fk_resample_translation_jitter < 0.0
        or args.fk_resample_rotation_jitter < 0.0
    ):
        parser.error("FK post-resampling jitter scales must be finite and non-negative")
    if not fk_enabled and (
        args.fk_resample_translation_jitter != 0.0 or args.fk_resample_rotation_jitter != 0.0
    ):
        parser.error("FK post-resampling jitter requires FK resampling")
    if translation_sde_enabled and (
        args.fk_resample_translation_jitter != 0.0
        or args.fk_resample_rotation_jitter != 0.0
    ):
        parser.error(
            "--translation-sde-base-sigma cannot be combined with heuristic FK jitter"
        )
    if translation_sde_enabled and (
        args.vina_guidance_scale != 0.0 or args.unified_guidance_scale != 0.0
    ):
        parser.error(
            "--translation-sde-base-sigma cannot be combined with gradient guidance"
        )
    if args.unified_guidance_scale < 0.0:
        parser.error("--unified-guidance-scale must be non-negative")
    if (
        args.unified_guidance_steric_radius_scale is not None
        and args.unified_guidance_steric_radius_scale <= 0.0
    ):
        parser.error("--unified-guidance-steric-radius-scale must be positive")
    if args.unified_guidance_chiral_improper_scale < 0.0:
        parser.error("--unified-guidance-chiral-improper-scale must be non-negative")
    if args.prior_pool_size < 0:
        parser.error("--prior-pool-size must be non-negative")
    if args.prior_pool_size and args.prior_pool_size < args.num_samples:
        parser.error("--prior-pool-size must be zero or at least --num-samples")
    if args.prior_pool_size and args.sigma_list:
        parser.error("--prior-pool-size currently requires scalar --sigma")
    if args.expected_discovered_count is not None and args.expected_discovered_count < 1:
        parser.error("--expected-discovered-count must be positive")
    if args.ligand_conformer_seed is not None and args.ligand_conformer_seed < 0:
        parser.error("--ligand-conformer-seed must be non-negative")
    if args.selector_profile == "confidence_cluster_free":
        if args.confidence_checkpoint is None:
            parser.error(
                "--selector-profile confidence_cluster_free requires --confidence-checkpoint"
            )
        if args.trajectory_dir is not None:
            parser.error(
                "--selector-profile confidence_cluster_free does not support --trajectory-dir"
            )
    if args.selector_profile == "candidate_only" and args.trajectory_dir is not None:
        parser.error("--selector-profile candidate_only does not support --trajectory-dir")

    pocket_centers = load_pocket_centers(args.pocket_centers)
    _, benchmark_input_identity = load_benchmark_inputs(
        args.dataset,
        args.external_dir,
        args.benchmark_input_manifest,
    )
    complexes = discover_complexes(
        args.dataset,
        args.data_dir,
        args.external_dir,
        pocket_centers,
        args.benchmark_input_manifest,
    )
    total_discovered = len(complexes)
    require_expected_discovered_count(total_discovered, args.expected_discovered_count)
    seed_by_id = global_seed_by_id(complexes, args.seed)
    if args.limit:
        complexes = complexes[: args.limit]
    if args.only_id:
        requested = {complex_id.lower() for complex_id in args.only_id}
        complexes = [item for item in complexes if item.complex_id.lower() in requested]
        found = {item.complex_id.lower() for item in complexes}
        if missing := requested - found:
            raise ValueError(f"requested benchmark IDs not found: {sorted(missing)}")
    complexes = shard_complexes(complexes, args.shard_index, args.num_shards)
    if not complexes:
        raise SystemExit(f"No complexes found in {args.data_dir}")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, cfg, ckpt = load_model(args.config, args.checkpoint, device)
    confidence_model = None
    confidence_ckpt = None
    if args.confidence_checkpoint is not None and args.selector_profile != "candidate_only":
        from effdock.confidence.runtime import load_pose_confidence_model

        confidence_model, confidence_ckpt = load_pose_confidence_model(
            args.confidence_checkpoint, device
        )
    sigma = float(args.sigma if args.sigma is not None else cfg["data"].get("prior_sigma", 3.5))
    pose_objective = cfg.get("data", {}).get("pose_objective", "linear_fm")
    if fk_enabled and pose_objective.lower() != "linear_fm":
        parser.error("experimental FK resampling currently requires linear_fm")
    if translation_sde_enabled and pose_objective.lower() != "linear_fm":
        parser.error("score-corrected translation SDE currently requires linear_fm")
    score_rot_sigma_max = float(cfg.get("data", {}).get("score_rot_sigma_max", torch.pi))
    score_alpha_min = float(cfg.get("data", {}).get("score_alpha_min", 0.0))
    sigma_list, sigma_counts = parse_sigma_list(args.sigma_list, args.num_samples)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    failures: list[dict] = []
    print(
        f"checkpoint={args.checkpoint} step={ckpt.get('step')} "
        f"dataset={args.dataset} complexes={len(complexes)}/{total_discovered} "
        f"shard={args.shard_index}/{args.num_shards} samples={args.num_samples}"
    )
    for i, item in enumerate(complexes, start=1):
        try:
            row = evaluate_one(
                model,
                item,
                dataset=args.dataset,
                confidence_model=confidence_model,
                device=device,
                num_samples=args.num_samples,
                num_steps=args.num_steps,
                sigma=sigma,
                sigma_list=sigma_list,
                sigma_counts=sigma_counts,
                center_jitter_sigma=args.center_jitter_sigma,
                pocket_cutoff=args.pocket_cutoff,
                pose_objective=pose_objective,
                score_rot_sigma_max=score_rot_sigma_max,
                score_alpha_min=score_alpha_min,
                time_schedule=args.time_schedule,
                schedule_power=args.schedule_power,
                vina_guidance_scale=args.vina_guidance_scale,
                vina_guidance_start_t=args.vina_guidance_start_t,
                vina_guidance_ramp_power=args.vina_guidance_ramp_power,
                vina_guidance_max_force=args.vina_guidance_max_force,
                vina_guidance_max_velocity=args.vina_guidance_max_velocity,
                vina_guidance_max_angular_velocity=args.vina_guidance_max_angular_velocity,
                vina_guidance_protein_shell=args.vina_guidance_protein_shell,
                vina_guidance_w_strain=args.vina_guidance_w_strain,
                unified_guidance_scale=args.unified_guidance_scale,
                unified_guidance_start_t=args.unified_guidance_start_t,
                unified_guidance_ramp_power=args.unified_guidance_ramp_power,
                unified_guidance_max_force=args.unified_guidance_max_force,
                unified_guidance_max_velocity=args.unified_guidance_max_velocity,
                unified_guidance_max_angular_velocity=(args.unified_guidance_max_angular_velocity),
                unified_guidance_max_atom_displacement=(
                    args.unified_guidance_max_atom_displacement
                ),
                unified_guidance_max_backtracks=args.unified_guidance_max_backtracks,
                unified_guidance_protein_shell=args.unified_guidance_protein_shell,
                unified_guidance_receptor_policy=(args.unified_guidance_receptor_policy),
                unified_guidance_mode=args.unified_guidance_mode,
                prior_pool_size=args.prior_pool_size,
                seed=seed_by_id[item.complex_id],
                refine=args.refine,
                pose_dir=(
                    args.out_dir / "poses" / (args.run_name or args.dataset) / args.dataset
                    if args.save_selected_poses
                    else None
                ),
                trajectory_dir=args.trajectory_dir,
                require_full_ligand_atom_mapping=args.require_full_ligand_atom_mapping,
                selector_profile=args.selector_profile,
                unified_guidance_steric_radius_scale=(args.unified_guidance_steric_radius_scale),
                unified_guidance_chiral_improper_scale=(
                    args.unified_guidance_chiral_improper_scale
                ),
                fk_constraint_beta=args.fk_constraint_beta,
                fk_resample_times=args.fk_resample_times,
                fk_resample_method=args.fk_resample_method,
                fk_resample_translation_jitter=(args.fk_resample_translation_jitter),
                fk_resample_rotation_jitter=args.fk_resample_rotation_jitter,
                translation_sde_base_sigma=args.translation_sde_base_sigma,
                ligand_conformer_seed=args.ligand_conformer_seed,
            )
            rows.append(row)
            selector_log = f"first={row['first_rmsd']:.3f} "
            if "vina_rmsd" in row:
                selector_log += f"vina={row['vina_rmsd']:.3f} "
            if confidence_model is not None:
                selector_log += (
                    f"confidence={row['confidence_rmsd']:.3f} "
                    f"confidence_filter={row['confidence_filter_rmsd']:.3f} "
                )
                if "confidence_final_rmsd" in row:
                    selector_log += f"confidence_final={row['confidence_final_rmsd']:.3f} "
            print(
                f"[{i:04d}/{len(complexes)}] {item.complex_id} "
                f"{selector_log}oracle={row['oracle_rmsd']:.3f}"
            )
        except Exception as exc:
            failures.append(serialize_evaluation_failure(item.complex_id, exc))
            print(f"[{i:04d}/{len(complexes)}] {item.complex_id} FAIL {exc!r}")

    sigma_tag = (
        "mix" + "-".join(f"{s:g}x{n}" for s, n in zip(sigma_list, sigma_counts))
        if sigma_list
        else f"sig{sigma:g}"
    )
    jitter_tag = f"_cj{args.center_jitter_sigma:g}" if args.center_jitter_sigma > 0.0 else ""
    cutoff_tag = f"_pc{args.pocket_cutoff:g}" if args.pocket_cutoff != DEFAULT_POCKET_CUTOFF else ""
    prior_pool_tag = f"_pool{args.prior_pool_size}" if args.prior_pool_size else ""
    unified_guidance_tag = (
        f"_ug{args.unified_guidance_scale:g}_{args.unified_guidance_mode}"
        if args.unified_guidance_scale != 0.0
        else ""
    )
    translation_sde_tag = (
        f"_tsde{args.translation_sde_base_sigma:g}" if translation_sde_enabled else ""
    )
    fk_tag = (
        f"_fkb{args.fk_constraint_beta:g}_{args.fk_resample_method}_t"
        + "-".join(f"{time:g}" for time in args.fk_resample_times)
        + (
            f"_jt{args.fk_resample_translation_jitter:g}"
            if args.fk_resample_translation_jitter > 0.0
            else ""
        )
        + (
            f"_jr{args.fk_resample_rotation_jitter:g}"
            if args.fk_resample_rotation_jitter > 0.0
            else ""
        )
        if fk_enabled
        else ""
    )
    base_tag = args.run_name or (
        f"{args.dataset}_{args.checkpoint.stem}_n{args.num_samples}_s{args.num_steps}_"
        f"{sigma_tag}{jitter_tag}{cutoff_tag}{prior_pool_tag}{unified_guidance_tag}"
        f"{translation_sde_tag}{fk_tag}"
    )
    shard_tag = f".shard-{args.shard_index:03d}-of-{args.num_shards:03d}"
    tag = f"{base_tag}{shard_tag}" if args.num_shards > 1 else base_tag
    csv_path = args.out_dir / f"{tag}.csv"
    if rows:
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    summary = {
        "dataset": args.dataset,
        "run_name": args.run_name,
        "protocol_id": args.protocol_id,
        "selector_profile": args.selector_profile,
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": ckpt.get("step"),
        "confidence_checkpoint": (
            str(args.confidence_checkpoint) if confidence_model is not None else None
        ),
        "confidence_scoring_enabled": confidence_model is not None,
        "confidence_step": confidence_ckpt.get("step") if confidence_ckpt else None,
        "config": str(args.config),
        "data_dir": str(args.data_dir),
        "benchmark_input_identity": benchmark_input_identity,
        "require_full_ligand_atom_mapping": args.require_full_ligand_atom_mapping,
        "expected_discovered_count": args.expected_discovered_count,
        "require_complete_success": args.require_complete_success,
        "num_discovered_total": total_discovered,
        "num_assigned": len(complexes),
        "num_success": len(rows),
        "num_failed": len(failures),
        "num_samples": args.num_samples,
        "num_steps": args.num_steps,
        "model_pose_step_budget": args.num_samples * args.num_steps,
        "sigma": sigma,
        "sigma_list": sigma_list,
        "sigma_counts": sigma_counts,
        "pose_objective": pose_objective,
        "score_rot_sigma_max": score_rot_sigma_max,
        "score_alpha_min": score_alpha_min,
        "translation_sde_base_sigma": args.translation_sde_base_sigma,
        "sampling_dynamics_contract": (
            {
                "mode": "translation_score_corrected_sde_deterministic_so3",
                "diffusion_schedule": "g(t)=base_sigma*(1-t)",
                "translation_score": "(t*v-T)/((1-t)*prior_sigma^2)",
                "translation_prior": "pocket_centered_isotropic_gaussian",
                "rotation": "deterministic_so3_flow_non_gaussian_prior",
                "reference": "arXiv:2509.01543 Eq.12 generalized to prior_sigma",
                "seed_domain_separation": "sampling_seed XOR 0x54534445",
            }
            if translation_sde_enabled
            else {"mode": "deterministic_ode"}
        ),
        "center_jitter_sigma": args.center_jitter_sigma,
        "pocket_cutoff": args.pocket_cutoff,
        "time_schedule": args.time_schedule,
        "schedule_power": args.schedule_power,
        "vina_guidance_scale": args.vina_guidance_scale,
        "vina_guidance_start_t": args.vina_guidance_start_t,
        "vina_guidance_ramp_power": args.vina_guidance_ramp_power,
        "vina_guidance_max_force": args.vina_guidance_max_force,
        "vina_guidance_max_velocity": args.vina_guidance_max_velocity,
        "vina_guidance_max_angular_velocity": args.vina_guidance_max_angular_velocity,
        "vina_guidance_protein_shell": args.vina_guidance_protein_shell,
        "vina_guidance_w_strain": args.vina_guidance_w_strain,
        "unified_guidance_scale": args.unified_guidance_scale,
        "unified_guidance_mode": args.unified_guidance_mode,
        "unified_guidance_start_t": args.unified_guidance_start_t,
        "unified_guidance_ramp_power": args.unified_guidance_ramp_power,
        "unified_guidance_max_force": args.unified_guidance_max_force,
        "unified_guidance_max_velocity": args.unified_guidance_max_velocity,
        "unified_guidance_max_angular_velocity": (args.unified_guidance_max_angular_velocity),
        "unified_guidance_max_atom_displacement": (args.unified_guidance_max_atom_displacement),
        "unified_guidance_max_backtracks": args.unified_guidance_max_backtracks,
        "unified_guidance_steric_radius_scale": (args.unified_guidance_steric_radius_scale),
        "unified_guidance_chiral_improper_scale": (args.unified_guidance_chiral_improper_scale),
        "unified_guidance_protein_shell": args.unified_guidance_protein_shell,
        "unified_guidance_receptor_policy": args.unified_guidance_receptor_policy,
        "guidance_implementation": guidance_implementation_identity(),
        "prior_pool_size": args.prior_pool_size,
        "prior_pool_hash_contract": "EFFDOCK_SHARED_PRIOR_V1",
        "refine": args.refine,
        "seed": args.seed,
        "ligand_conformer_seed": args.ligand_conformer_seed,
        "ligand_conformer_seed_policy": (
            "fixed_explicit"
            if args.ligand_conformer_seed is not None
            else "per_complex_sampling_seed_legacy"
        ),
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "pocket_centers": str(args.pocket_centers),
        "pocket_centers_sha256": file_sha256(args.pocket_centers),
        "eligibility_manifest": (
            str(args.eligibility_manifest) if args.eligibility_manifest is not None else None
        ),
        "eligibility_manifest_sha256": (
            file_sha256(args.eligibility_manifest)
            if args.eligibility_manifest is not None
            else None
        ),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "confidence_checkpoint_sha256": (
            file_sha256(args.confidence_checkpoint) if confidence_model is not None else None
        ),
        "config_sha256": file_sha256(args.config),
        "runtime": {
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "gpu_total_memory_bytes": (
                int(torch.cuda.get_device_properties(device).total_memory)
                if device.type == "cuda"
                else None
            ),
            "cuda_max_memory_allocated_bytes": (
                int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
            ),
            "cuda_max_memory_reserved_bytes": (
                int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else None
            ),
        },
        "stats": summarize_rows(rows),
        "failures": failures,
        "csv": str(csv_path) if rows else None,
    }
    if confidence_model is not None:
        summary["candidate_ensemble_hash_contract"] = CANDIDATE_ENSEMBLE_HASH_CONTRACT
        summary["confidence_score_ledger_contract"] = CONFIDENCE_SCORE_LEDGER_CONTRACT
    if fk_enabled:
        summary.update(
            {
                "fk_constraint_beta": args.fk_constraint_beta,
                "fk_resample_times": list(args.fk_resample_times),
                "fk_resample_method": args.fk_resample_method,
                "fk_resample_translation_jitter": (args.fk_resample_translation_jitter),
                "fk_resample_rotation_jitter": args.fk_resample_rotation_jitter,
                "fk_contract": {
                    "weight": "exp(-beta * delta_constraint_potential)",
                    "endpoint_estimator": "constant_velocity_euler_terminal",
                    "dynamics": (
                        "translation_score_corrected_sde_deterministic_so3"
                        if translation_sde_enabled
                        else "deterministic_flow_without_score_corrected_sde"
                    ),
                    "translation_sde_base_sigma": args.translation_sde_base_sigma,
                    "constraint_terms": list(DEFAULT_FK_CONSTRAINT_TERMS),
                    "post_resampling_mutation": fk_post_resampling_mutation_classification(
                        translation_sde_enabled=translation_sde_enabled,
                        translation_jitter=args.fk_resample_translation_jitter,
                        rotation_jitter=args.fk_resample_rotation_jitter,
                    ),
                },
            }
        )
    if args.unified_guidance_scale != 0.0 or fk_enabled:
        from effdock.guidance.parameterization import guidance_parameter_identity

        summary["guidance_parameter_set"] = guidance_parameter_identity()
        receptor_identities = {
            row["guidance_receptor_policy_identity_sha256"]: json.loads(
                row["guidance_receptor_policy_identity_json"]
            )
            for row in rows
            if row.get("guidance_receptor_policy_identity_sha256")
        }
        summary["guidance_receptor_policy_identities"] = receptor_identities
        summary["guidance_receptor_provenance_by_id"] = {
            row["id"]: json.loads(row["guidance_receptor_provenance_json"])
            for row in rows
            if row.get("guidance_receptor_provenance_json")
        }
        stat_names = sorted(
            {
                key.removeprefix("guidance_")
                for row in rows
                for key in row
                if key.startswith("guidance_")
                and key
                not in {
                    "guidance_mode",
                    "guidance_parameter_sha256",
                    "guidance_topology_reference_sha256",
                    "guidance_system_reference_sha256",
                    "guidance_interaction_reference_sha256",
                    "guidance_receptor_policy",
                    "guidance_receptor_policy_identity_sha256",
                    "guidance_receptor_policy_identity_json",
                    "guidance_receptor_provenance_json",
                    "guidance_metadata_json",
                    "guidance_direct_step_trace_json",
                }
            }
        )
        runtime_stats: dict[str, int | float | None] = {}
        for name in stat_names:
            values = [row.get(f"guidance_{name}") for row in rows]
            present = [value for value in values if value is not None]
            if not present:
                runtime_stats[name] = None
            elif name.startswith("max_") or name.startswith("direct_max_"):
                runtime_stats[name] = max(float(value) for value in present)
            elif name.startswith("min_"):
                runtime_stats[name] = min(float(value) for value in present)
            elif all(isinstance(value, int) and not isinstance(value, bool) for value in present):
                runtime_stats[name] = sum(int(value) for value in present)
            else:
                runtime_stats[name] = sum(float(value) for value in present)
        summary["guidance_runtime_stats"] = runtime_stats
        if args.unified_guidance_scale != 0.0 and args.unified_guidance_mode == "operator_split":
            summary["guidance_operator_stats"] = runtime_stats
    summary_path = args.out_dir / f"{tag}.summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    if args.require_complete_success:
        require_complete_evaluation(
            num_assigned=len(complexes),
            num_success=len(rows),
            failures=failures,
        )


if __name__ == "__main__":
    main()
