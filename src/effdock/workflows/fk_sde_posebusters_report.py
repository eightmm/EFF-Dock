"""Audit all-pose official PoseBusters results for frozen FK-ODE/FK-SDE."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import statistics
import tempfile
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from posebusters import PoseBusters
from rdkit import Chem

from effdock.workflows.guidance_budget_posebusters_report import VALIDITY_CHECKS
from effdock.workflows.posebusters_report import require_posebusters_runtime_version

PROTOCOL_ID = "EFFDOCK-FK-TRANSLATION-SDE-POSEBUSTERS-V1"
SCHEMA_VERSION = "effdock.fk_sde_posebusters_manifest.v1"
OFFICIAL_SCHEMA_VERSION = "effdock.fk_sde_posebusters_official.v1"
REPORT_SCHEMA_VERSION = "effdock.fk_sde_posebusters_report.v1"
POSEBUSTERS_VERSION = "0.6.5"
POSEBUSTERS_CONFIG = "redock"
NUM_SAMPLES = 40
NUM_STEPS = 25
NUM_SHARDS = 8
OFFICIAL_SHARDS = 16
EXPECTED_COMPLEXES = 308
SMOKE_ID = "7b2c_tp7"
COORDINATE_ROUND_DECIMALS = 3


@dataclass(frozen=True)
class ArmSpec:
    name: str
    slug: str
    sde_sigma: float

    @property
    def dynamics(self) -> str:
        if self.sde_sigma > 0.0:
            return "translation_score_corrected_sde_deterministic_so3"
        return "deterministic_ode"

    @property
    def guidance_mode(self) -> str:
        if self.sde_sigma > 0.0:
            return "feynman_kac_constraint_resampling_translation_sde"
        return "feynman_kac_constraint_resampling"


ARM_SPECS = {
    spec.name: spec
    for spec in (
        ArmSpec("fk_ode", "fk-ode", 0.0),
        ArmSpec("fk_sde", "fk-sde", 0.3),
    )
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError(f"{path}: missing or duplicate CSV fields")
        return list(reader)


def _as_bool(value: object, *, label: str) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"{label}: expected a boolean, got {value!r}")


def _finite_float(value: object, *, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label}: expected a finite value")
    return result


def _require_close(actual: object, expected: float, *, label: str) -> None:
    value = _finite_float(actual, label=label)
    if not math.isclose(value, expected, rel_tol=0.0, abs_tol=1.0e-9):
        raise ValueError(f"{label}: expected {expected}, got {value}")


def _cohort_ids(path: Path) -> list[str]:
    manifest = _read_json(path)
    try:
        ids = manifest["datasets"]["posebusters"]["audited_ids"]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"{path}: missing PoseBusters audited IDs") from exc
    if not isinstance(ids, list) or any(not isinstance(value, str) for value in ids):
        raise ValueError(f"{path}: audited IDs must be a string list")
    if len(ids) != EXPECTED_COMPLEXES or len(set(ids)) != EXPECTED_COMPLEXES:
        raise ValueError(f"{path}: expected {EXPECTED_COMPLEXES} unique IDs")
    return sorted(ids)


def _run_name(spec: ArmSpec, *, mode: str) -> str:
    prefix = "effdock-fk-sde-posebusters-v1"
    if mode == "smoke":
        prefix += "-smoke"
    return f"{prefix}-{spec.slug}"


def _sampling_tag(run_name: str, *, shard_index: int, num_shards: int) -> str:
    if num_shards == 1:
        return run_name
    return f"{run_name}.shard-{shard_index:03d}-of-{num_shards:03d}"


def _validate_summary(
    summary: dict[str, Any],
    spec: ArmSpec,
    *,
    mode: str,
    shard_index: int,
    num_shards: int,
) -> None:
    label = f"{mode}.{spec.name}.shard{shard_index}"
    expected = {
        "dataset": "posebusters",
        "protocol_id": PROTOCOL_ID,
        "run_name": _run_name(spec, mode=mode),
        "selector_profile": "confidence_cluster_free",
        "checkpoint": "weights/effdock_geometry_ft_100k_best.pt",
        "confidence_checkpoint": (
            "weights/effdock_confidence_extmatch_n80_s25_step42500.pt"
        ),
        "config": "configs/train.yaml",
        "require_full_ligand_atom_mapping": True,
        "num_samples": NUM_SAMPLES,
        "num_steps": NUM_STEPS,
        "model_pose_step_budget": NUM_SAMPLES * NUM_STEPS,
        "prior_pool_size": 100,
        "time_schedule": "late",
        "schedule_power": 3.0,
        "seed": 42,
        "refine": "none",
        "unified_guidance_receptor_policy": "geometry_only",
        "unified_guidance_scale": 0.0,
        "vina_guidance_scale": 0.0,
        "expected_discovered_count": EXPECTED_COMPLEXES,
        "require_complete_success": True,
        "num_shards": num_shards,
        "shard_index": shard_index,
    }
    for key, value in expected.items():
        if summary.get(key) != value:
            raise ValueError(f"{label}.{key}: expected {value!r}, got {summary.get(key)!r}")
    _require_close(summary.get("sigma"), 0.5, label=f"{label}.sigma")
    _require_close(summary.get("pocket_cutoff"), 10.0, label=f"{label}.pocket_cutoff")
    _require_close(summary.get("center_jitter_sigma"), 0.0, label=f"{label}.jitter")
    _require_close(
        summary.get("translation_sde_base_sigma"),
        spec.sde_sigma,
        label=f"{label}.translation_sde_base_sigma",
    )
    _require_close(summary.get("fk_constraint_beta"), 0.01, label=f"{label}.beta")
    if summary.get("fk_resample_times") != [0.3, 0.6, 0.8]:
        raise ValueError(f"{label}: FK schedule drifted")
    if summary.get("fk_resample_method") != "systematic":
        raise ValueError(f"{label}: FK resampling method drifted")
    dynamics = summary.get("sampling_dynamics_contract")
    if not isinstance(dynamics, dict) or dynamics.get("mode") != spec.dynamics:
        raise ValueError(f"{label}: sampling dynamics drifted")
    if int(summary.get("num_failed", -1)) != 0:
        raise ValueError(f"{label}: recorded sampling failures")


def _coordinate_key(coords: np.ndarray) -> bytes:
    if coords.ndim != 2 or coords.shape[1] != 3 or not np.isfinite(coords).all():
        raise ValueError("SDF record contains invalid coordinates")
    scale = 10**COORDINATE_ROUND_DECIMALS
    return np.rint(coords * scale).astype("<i8", copy=False).tobytes()


def pose_diversity_metrics(coordinates: np.ndarray, heavy_mask: np.ndarray) -> dict[str, Any]:
    """Measure exact clone collapse and receptor-frame heavy-atom pose spread."""
    if coordinates.ndim != 3 or coordinates.shape[0] != NUM_SAMPLES:
        raise ValueError(f"expected coordinates shaped ({NUM_SAMPLES}, atoms, 3)")
    if coordinates.shape[2] != 3 or not np.isfinite(coordinates).all():
        raise ValueError("coordinates must be finite xyz arrays")
    if heavy_mask.shape != (coordinates.shape[1],) or not bool(heavy_mask.any()):
        raise ValueError("heavy-atom mask does not match coordinates")
    heavy = coordinates[:, heavy_mask, :]
    delta = heavy[:, None, :, :] - heavy[None, :, :, :]
    pairwise = np.sqrt(np.mean(np.sum(delta * delta, axis=-1), axis=-1))
    upper = pairwise[np.triu_indices(NUM_SAMPLES, k=1)]
    with_diagonal_masked = pairwise.copy()
    np.fill_diagonal(with_diagonal_masked, np.inf)
    nearest = np.min(with_diagonal_masked, axis=1)
    return {
        "coordinate_unique_count": len({_coordinate_key(value) for value in coordinates}),
        "pairwise_heavy_atom_rmsd_mean": float(np.mean(upper)),
        "pairwise_heavy_atom_rmsd_median": float(np.median(upper)),
        "pairwise_heavy_atom_rmsd_ge2_fraction": float(np.mean(upper >= 2.0)),
        "nearest_neighbor_heavy_atom_rmsd_median": float(np.median(nearest)),
    }


def _load_pose_ensemble(path: Path) -> tuple[np.ndarray, list[int], np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    molecules = [mol for mol in Chem.SDMolSupplier(str(path), removeHs=False) if mol is not None]
    if len(molecules) != NUM_SAMPLES:
        raise ValueError(f"{path}: expected {NUM_SAMPLES} records, got {len(molecules)}")
    atomic_numbers = np.asarray(
        [atom.GetAtomicNum() for atom in molecules[0].GetAtoms()], dtype=np.int64
    )
    coordinates: list[np.ndarray] = []
    ancestry: list[int] = []
    for index, molecule in enumerate(molecules):
        observed = np.asarray([atom.GetAtomicNum() for atom in molecule.GetAtoms()])
        if not np.array_equal(observed, atomic_numbers):
            raise ValueError(f"{path}: atom ordering changed at record {index}")
        coords = np.asarray(molecule.GetConformer().GetPositions(), dtype=np.float64)
        _coordinate_key(coords)
        coordinates.append(coords)
        if not molecule.HasProp("fk_initial_sample_index"):
            raise ValueError(f"{path}: record {index} lacks FK ancestry")
        ancestor = int(molecule.GetProp("fk_initial_sample_index"))
        if not 0 <= ancestor < NUM_SAMPLES:
            raise ValueError(f"{path}: record {index} has invalid FK ancestry")
        ancestry.append(ancestor)
    return np.stack(coordinates), ancestry, atomic_numbers > 1


def _validate_fk_diagnostics(
    row: dict[str, str], spec: ArmSpec, *, label: str
) -> tuple[list[float], int]:
    diagnostics = json.loads(row["fk_diagnostics_json"])
    if diagnostics.get("schema_version") != "effdock.fk_constraint_resampling.v2":
        raise ValueError(f"{label}: FK diagnostics schema drifted")
    expected_dynamics = (
        "translation_score_corrected_sde_deterministic_so3"
        if spec.sde_sigma > 0.0
        else "deterministic_flow_without_score_corrected_sde"
    )
    if diagnostics.get("dynamics") != expected_dynamics:
        raise ValueError(f"{label}: FK diagnostic dynamics drifted")
    _require_close(
        diagnostics.get("translation_sde_base_sigma"),
        spec.sde_sigma,
        label=f"{label}.diagnostic_sigma",
    )
    events = diagnostics.get("events")
    if diagnostics.get("num_resampling_events") != 3 or not isinstance(events, list):
        raise ValueError(f"{label}: expected exactly three FK events")
    if len(events) != 3:
        raise ValueError(f"{label}: expected exactly three FK event rows")
    ess: list[float] = []
    for index, (event, requested) in enumerate(zip(events, (0.3, 0.6, 0.8), strict=True)):
        if not isinstance(event, dict) or event.get("event_index") != index:
            raise ValueError(f"{label}: invalid FK event index")
        _require_close(event.get("requested_time"), requested, label=f"{label}.time")
        actual_time = _finite_float(event.get("actual_time"), label=f"{label}.actual_time")
        ess_fraction = _finite_float(event.get("ess_fraction"), label=f"{label}.ess")
        if not 0.0 < actual_time < 1.0 or not 0.0 < ess_fraction <= 1.0:
            raise ValueError(f"{label}: invalid FK event time or ESS")
        ess.append(ess_fraction)
        for key in (
            "potential_min",
            "potential_median",
            "potential_max",
            "delta_min",
            "delta_median",
            "delta_max",
            "max_group_weight",
        ):
            _finite_float(event.get(key), label=f"{label}.{key}")
    final_ancestors = int(diagnostics.get("final_unique_initial_ancestors", 0))
    if not 1 <= final_ancestors <= NUM_SAMPLES:
        raise ValueError(f"{label}: invalid final ancestry count")
    return ess, final_ancestors


def _validate_sampling_row(
    row: dict[str, str],
    spec: ArmSpec,
    *,
    label: str,
    source_csv: Path,
) -> dict[str, Any]:
    if row.get("guidance_mode") != spec.guidance_mode:
        raise ValueError(f"{label}: guidance mode drifted")
    if row.get("sampling_dynamics") != spec.dynamics:
        raise ValueError(f"{label}: sampling dynamics drifted")
    _require_close(
        row.get("translation_sde_base_sigma"),
        spec.sde_sigma,
        label=f"{label}.translation_sde_base_sigma",
    )
    _require_close(row.get("fk_constraint_beta"), 0.01, label=f"{label}.beta")
    if int(row.get("num_samples", -1)) != NUM_SAMPLES:
        raise ValueError(f"{label}: sample count drifted")
    if int(row.get("prior_pool_size", -1)) != 100:
        raise ValueError(f"{label}: prior-pool size drifted")
    if not _as_bool(row.get("full_heavy_atom_bijection"), label=f"{label}.bijection"):
        raise ValueError(f"{label}: full heavy-atom mapping is required")

    prior_hash = row.get("prior_pool_sha256", "")
    if len(prior_hash) != 64 or any(char not in "0123456789abcdef" for char in prior_hash):
        raise ValueError(f"{label}: invalid prior-pool hash")
    sampling_seed = int(row["sampling_seed"])
    sde_seed = row.get("translation_sde_seed", "")
    if spec.sde_sigma > 0.0:
        if int(sde_seed) != (sampling_seed ^ 0x54534445):
            raise ValueError(f"{label}: translation-SDE seed contract drifted")
    elif sde_seed not in {"", "None"}:
        raise ValueError(f"{label}: deterministic arm unexpectedly records SDE seed")

    ess_fractions, final_ancestors = _validate_fk_diagnostics(row, spec, label=label)
    pose_path = Path(row.get("all_poses_sdf", ""))
    if int(row.get("all_poses_count", -1)) != NUM_SAMPLES:
        raise ValueError(f"{label}: all-pose count drifted")
    pose_hash = row.get("all_poses_sdf_sha256", "")
    if _sha256(pose_path) != pose_hash:
        raise ValueError(f"{label}: all-pose SDF hash mismatch")
    coordinates, ancestry, heavy_mask = _load_pose_ensemble(pose_path)
    if len(set(ancestry)) != final_ancestors:
        raise ValueError(f"{label}: SDF ancestry and diagnostics disagree")
    diversity = pose_diversity_metrics(coordinates, heavy_mask)

    protein = Path(row["protein"])
    ligand_ref = Path(row["ligand_ref"])
    protein_hash = row.get("protein_sha256", "")
    ligand_ref_hash = row.get("ligand_reference_sha256", "")
    if _sha256(protein) != protein_hash or _sha256(ligand_ref) != ligand_ref_hash:
        raise ValueError(f"{label}: protein/reference ligand hash mismatch")
    selected_index = int(row["confidence_index"])
    if not 0 <= selected_index < NUM_SAMPLES:
        raise ValueError(f"{label}: invalid confidence-selected index")
    selected_rmsd = _finite_float(row["confidence_rmsd"], label=f"{label}.confidence_rmsd")
    oracle_rmsd = _finite_float(row["oracle_rmsd"], label=f"{label}.oracle_rmsd")

    return {
        "arm": spec.name,
        "id": row["id"],
        "sampling_seed": sampling_seed,
        "prior_pool_sha256": prior_hash,
        "translation_sde_seed": int(sde_seed) if spec.sde_sigma > 0.0 else None,
        "selected_index": selected_index,
        "selected_rmsd": selected_rmsd,
        "evaluator_oracle_rmsd": oracle_rmsd,
        "pose_path": str(pose_path),
        "pose_sha256": pose_hash,
        "protein": str(protein),
        "protein_sha256": protein_hash,
        "ligand_ref": str(ligand_ref),
        "ligand_ref_sha256": ligand_ref_hash,
        "sampling_csv": str(source_csv),
        "sampling_csv_sha256": _sha256(source_csv),
        "sampling_row_sha256": _canonical_sha256(row),
        "ess_fractions": ess_fractions,
        "final_unique_initial_ancestors": final_ancestors,
        "diversity": diversity,
    }


def build_manifest(
    *,
    input_root: Path,
    cohort_manifest: Path,
    mode: str,
) -> dict[str, Any]:
    if mode not in {"smoke", "full"}:
        raise ValueError(f"unknown manifest mode: {mode}")
    all_ids = _cohort_ids(cohort_manifest)
    expected_ids = [SMOKE_ID] if mode == "smoke" else all_ids
    num_shards = 1 if mode == "smoke" else NUM_SHARDS
    records: list[dict[str, Any]] = []
    by_arm: dict[str, dict[str, dict[str, Any]]] = {}
    for spec in ARM_SPECS.values():
        arm_rows: dict[str, dict[str, Any]] = {}
        for shard_index in range(num_shards):
            run_name = _run_name(spec, mode=mode)
            tag = _sampling_tag(
                run_name,
                shard_index=shard_index,
                num_shards=num_shards,
            )
            arm_dir = input_root / spec.name
            summary_path = arm_dir / f"{tag}.summary.json"
            csv_path = arm_dir / f"{tag}.csv"
            summary = _read_json(summary_path)
            _validate_summary(
                summary,
                spec,
                mode=mode,
                shard_index=shard_index,
                num_shards=num_shards,
            )
            rows = _read_csv(csv_path)
            if len(rows) != int(summary.get("num_success", -1)):
                raise ValueError(f"{tag}: CSV/summary success count mismatch")
            for row in rows:
                complex_id = row.get("id", "")
                if complex_id in arm_rows:
                    raise ValueError(f"{spec.name}: duplicate complex {complex_id}")
                arm_rows[complex_id] = _validate_sampling_row(
                    row,
                    spec,
                    label=f"{spec.name}.{complex_id}",
                    source_csv=csv_path,
                )
        if sorted(arm_rows) != expected_ids:
            raise ValueError(
                f"{spec.name}: expected exact {mode} ID set; "
                f"observed={len(arm_rows)}, expected={len(expected_ids)}"
            )
        by_arm[spec.name] = arm_rows
        records.extend(arm_rows[complex_id] for complex_id in sorted(arm_rows))

    for complex_id in expected_ids:
        paired = {
            (rows[complex_id]["sampling_seed"], rows[complex_id]["prior_pool_sha256"])
            for rows in by_arm.values()
        }
        if len(paired) != 1:
            raise ValueError(f"{complex_id}: paired arms do not share prior identity")

    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "mode": mode,
        "claim_boundary": "paired_descriptive_external_evidence_only",
        "posebusters_version": POSEBUSTERS_VERSION,
        "posebusters_config": POSEBUSTERS_CONFIG,
        "validity_definition": "all 27 non-RMSD redock checks",
        "validity_checks": list(VALIDITY_CHECKS),
        "poses_per_cell": NUM_SAMPLES,
        "expected_complexes_per_arm": len(expected_ids),
        "expected_cells": len(expected_ids) * len(ARM_SPECS),
        "expected_poses": len(expected_ids) * len(ARM_SPECS) * NUM_SAMPLES,
        "cohort_manifest": str(cohort_manifest),
        "cohort_manifest_sha256": _sha256(cohort_manifest),
        "ordered_ids_sha256": _canonical_sha256(expected_ids),
        "records": records,
    }


def _validate_manifest_identity(manifest: dict[str, Any], *, expected_mode: str | None) -> None:
    exact = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "claim_boundary": "paired_descriptive_external_evidence_only",
        "posebusters_version": POSEBUSTERS_VERSION,
        "posebusters_config": POSEBUSTERS_CONFIG,
        "validity_definition": "all 27 non-RMSD redock checks",
        "validity_checks": list(VALIDITY_CHECKS),
        "poses_per_cell": NUM_SAMPLES,
    }
    for key, expected in exact.items():
        if manifest.get(key) != expected:
            raise ValueError(f"manifest {key} mismatch")
    mode = manifest.get("mode")
    if mode not in {"smoke", "full"} or (expected_mode is not None and mode != expected_mode):
        raise ValueError("manifest mode mismatch")
    complexes = 1 if mode == "smoke" else EXPECTED_COMPLEXES
    expected_counts = {
        "expected_complexes_per_arm": complexes,
        "expected_cells": complexes * len(ARM_SPECS),
        "expected_poses": complexes * len(ARM_SPECS) * NUM_SAMPLES,
    }
    for key, expected in expected_counts.items():
        if manifest.get(key) != expected:
            raise ValueError(f"manifest {key} mismatch")
    records = manifest.get("records")
    if not isinstance(records, list) or len(records) != expected_counts["expected_cells"]:
        raise ValueError("manifest record inventory mismatch")
    keys = [(row.get("arm"), row.get("id")) for row in records if isinstance(row, dict)]
    if len(keys) != len(records) or len(set(keys)) != len(keys):
        raise ValueError("manifest contains invalid or duplicate cell records")


def _validate_frame_row(raw: dict[str, Any], *, label: str) -> tuple[str, dict[str, bool]]:
    rmsd_checks = [key for key in raw if str(key).startswith("rmsd_")]
    if len(rmsd_checks) != 1:
        raise ValueError(f"{label}: expected exactly one separate RMSD check")
    rmsd_check = rmsd_checks[0]
    expected = {*VALIDITY_CHECKS, rmsd_check}
    if set(raw) != expected:
        raise ValueError(
            f"{label}: official schema mismatch; "
            f"missing={sorted(expected - set(raw))}, extra={sorted(set(raw) - expected)}"
        )
    return rmsd_check, {
        key: False if pd.isna(value) else bool(value) for key, value in raw.items()
    }


def _write_csv_gz(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with gzip.open(path, "wt", newline="", encoding="utf-8", compresslevel=6) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def score_official_shard(
    *,
    manifest_path: Path,
    manifest_sha256: str,
    output_root: Path,
    num_shards: int,
    shard_index: int,
    workers: int,
) -> dict[str, Any]:
    if _sha256(manifest_path) != manifest_sha256:
        raise ValueError("manifest SHA-256 mismatch")
    manifest = _read_json(manifest_path)
    _validate_manifest_identity(manifest, expected_mode=None)
    mode = str(manifest["mode"])
    expected_shards = 1 if mode == "smoke" else OFFICIAL_SHARDS
    if num_shards != expected_shards or not 0 <= shard_index < num_shards:
        raise ValueError(f"{mode}: official shard contract mismatch")
    if workers < 1:
        raise ValueError("workers must be positive")
    records = list(manifest["records"])[shard_index::num_shards]
    if not records:
        raise ValueError("empty official shard assignment")

    final_dir = output_root / mode / f"shard-{shard_index:03d}-of-{num_shards:03d}"
    if final_dir.exists():
        raise FileExistsError(f"refusing to overwrite completed shard: {final_dir}")
    incomplete = output_root / mode / ".incomplete"
    incomplete.mkdir(parents=True, exist_ok=True)
    attempt = Path(tempfile.mkdtemp(prefix=f"{final_dir.name}.attempt-", dir=incomplete))
    started = time.monotonic()
    observed_version = require_posebusters_runtime_version()
    if observed_version != POSEBUSTERS_VERSION:
        raise RuntimeError(
            f"expected PoseBusters {POSEBUSTERS_VERSION}, got {observed_version}"
        )
    buster = PoseBusters(
        config=POSEBUSTERS_CONFIG,
        max_workers=workers,
        chunk_size=math.ceil(NUM_SAMPLES / workers),
    )
    pose_rows: list[dict[str, Any]] = []
    cell_rows: list[dict[str, Any]] = []
    rmsd_check_name: str | None = None
    for cell_index, record in enumerate(records, start=1):
        arm = str(record["arm"])
        complex_id = str(record["id"])
        pose_path = Path(record["pose_path"])
        protein = Path(record["protein"])
        ligand_ref = Path(record["ligand_ref"])
        assets = (
            (pose_path, str(record["pose_sha256"])),
            (protein, str(record["protein_sha256"])),
            (ligand_ref, str(record["ligand_ref_sha256"])),
        )
        for path, expected_hash in assets:
            if not path.is_file() or _sha256(path) != expected_hash:
                raise ValueError(f"{arm}/{complex_id}: changed asset {path}")
        frame = buster.bust(pose_path, ligand_ref, protein, full_report=False)
        if len(frame.index) != NUM_SAMPLES:
            raise ValueError(
                f"{arm}/{complex_id}: expected {NUM_SAMPLES} PB rows, got {len(frame.index)}"
            )
        check_counts = {check: 0 for check in VALIDITY_CHECKS}
        valid_count = 0
        joint_count = 0
        for pose_index, (_, frame_row) in enumerate(frame.iterrows()):
            current_rmsd, checks = _validate_frame_row(
                frame_row.to_dict(), label=f"{arm}/{complex_id}/pose-{pose_index}"
            )
            if rmsd_check_name is None:
                rmsd_check_name = current_rmsd
            elif current_rmsd != rmsd_check_name:
                raise ValueError("PoseBusters RMSD column changed within the shard")
            valid = all(checks[check] for check in VALIDITY_CHECKS)
            rmsd_success = checks[current_rmsd]
            valid_count += int(valid)
            joint_count += int(valid and rmsd_success)
            for check in VALIDITY_CHECKS:
                check_counts[check] += int(checks[check])
            pose_rows.append(
                {
                    "arm": arm,
                    "id": complex_id,
                    "pose_index": pose_index,
                    "posebusters_valid": valid,
                    "separate_rmsd_check": rmsd_success,
                    **{check: checks[check] for check in VALIDITY_CHECKS},
                }
            )
        cell_rows.append(
            {
                "arm": arm,
                "id": complex_id,
                "pose_count": NUM_SAMPLES,
                "valid_count": valid_count,
                "joint_valid_rmsd_count": joint_count,
                **{f"{check}_pass_count": check_counts[check] for check in VALIDITY_CHECKS},
            }
        )
        print(
            f"[{cell_index:03d}/{len(records):03d}] {arm} {complex_id} "
            f"PB-valid={valid_count}/{NUM_SAMPLES} joint={joint_count}/{NUM_SAMPLES}",
            flush=True,
        )
    if rmsd_check_name is None:
        raise RuntimeError("no official PoseBusters results were produced")

    pose_fields = [
        "arm",
        "id",
        "pose_index",
        "posebusters_valid",
        "separate_rmsd_check",
        *VALIDITY_CHECKS,
    ]
    cell_fields = [
        "arm",
        "id",
        "pose_count",
        "valid_count",
        "joint_valid_rmsd_count",
        *[f"{check}_pass_count" for check in VALIDITY_CHECKS],
    ]
    poses_output = attempt / "poses.csv.gz"
    cells_output = attempt / "cells.csv"
    _write_csv_gz(poses_output, pose_rows, pose_fields)
    with cells_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=cell_fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(cell_rows)
    summary = {
        "schema_version": OFFICIAL_SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": "complete",
        "mode": mode,
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "posebusters_version": observed_version,
        "posebusters_config": POSEBUSTERS_CONFIG,
        "validity_definition": "all 27 non-RMSD redock checks",
        "validity_checks": list(VALIDITY_CHECKS),
        "rmsd_check": rmsd_check_name,
        "num_shards": num_shards,
        "shard_index": shard_index,
        "assigned_cells": len(records),
        "assigned_poses": len(records) * NUM_SAMPLES,
        "result_cells": len(cell_rows),
        "result_poses": len(pose_rows),
        "valid_poses": sum(int(row["posebusters_valid"]) for row in pose_rows),
        "joint_valid_rmsd_poses": sum(
            int(row["posebusters_valid"] and row["separate_rmsd_check"])
            for row in pose_rows
        ),
        "workers": workers,
        "posebusters_chunk_size": math.ceil(NUM_SAMPLES / workers),
        "runtime": {
            "elapsed_seconds": time.monotonic() - started,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
            "cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
            "finished_at_utc": datetime.now(UTC).isoformat(),
        },
        "artifacts": {
            "poses_csv_gz": str(final_dir / "poses.csv.gz"),
            "poses_csv_gz_sha256": _sha256(poses_output),
            "cells_csv": str(final_dir / "cells.csv"),
            "cells_csv_sha256": _sha256(cells_output),
            "summary": str(final_dir / "summary.json"),
        },
    }
    (attempt / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    os.rename(attempt, final_dir)
    return summary


def _load_official_results(
    *,
    manifest: dict[str, Any],
    manifest_sha256: str,
    official_root: Path,
    num_shards: int,
) -> tuple[dict[tuple[str, str], list[dict[str, Any]]], str]:
    mode = str(manifest["mode"])
    expected_shards = 1 if mode == "smoke" else OFFICIAL_SHARDS
    if num_shards != expected_shards:
        raise ValueError(f"{mode}: report official shard contract mismatch")
    expected_cells = {(str(row["arm"]), str(row["id"])) for row in manifest["records"]}
    by_cell: dict[tuple[str, str], list[dict[str, Any]]] = {}
    observed_rmsd_check: str | None = None
    total_summary_cells = 0
    total_summary_poses = 0
    for shard_index in range(num_shards):
        shard_dir = official_root / mode / f"shard-{shard_index:03d}-of-{num_shards:03d}"
        summary_path = shard_dir / "summary.json"
        poses_path = shard_dir / "poses.csv.gz"
        cells_path = shard_dir / "cells.csv"
        summary = _read_json(summary_path)
        exact = {
            "schema_version": OFFICIAL_SCHEMA_VERSION,
            "protocol_id": PROTOCOL_ID,
            "status": "complete",
            "mode": mode,
            "manifest_sha256": manifest_sha256,
            "posebusters_version": POSEBUSTERS_VERSION,
            "posebusters_config": POSEBUSTERS_CONFIG,
            "validity_definition": "all 27 non-RMSD redock checks",
            "validity_checks": list(VALIDITY_CHECKS),
            "num_shards": num_shards,
            "shard_index": shard_index,
        }
        for key, expected in exact.items():
            if summary.get(key) != expected:
                raise ValueError(f"{summary_path}: {key} mismatch")
        artifacts = summary.get("artifacts")
        if not isinstance(artifacts, dict):
            raise ValueError(f"{summary_path}: missing artifacts")
        if _sha256(poses_path) != artifacts.get("poses_csv_gz_sha256"):
            raise ValueError(f"{poses_path}: hash mismatch")
        if _sha256(cells_path) != artifacts.get("cells_csv_sha256"):
            raise ValueError(f"{cells_path}: hash mismatch")
        rmsd_check = str(summary.get("rmsd_check", ""))
        if not rmsd_check.startswith("rmsd_"):
            raise ValueError(f"{summary_path}: invalid RMSD check")
        if observed_rmsd_check is None:
            observed_rmsd_check = rmsd_check
        elif observed_rmsd_check != rmsd_check:
            raise ValueError("official RMSD check differs across shards")
        with gzip.open(poses_path, "rt", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            expected_fields = {
                "arm",
                "id",
                "pose_index",
                "posebusters_valid",
                "separate_rmsd_check",
                *VALIDITY_CHECKS,
            }
            if reader.fieldnames is None or set(reader.fieldnames) != expected_fields:
                raise ValueError(f"{poses_path}: official pose CSV schema mismatch")
            shard_pose_count = 0
            for line_number, raw in enumerate(reader, start=2):
                key = (raw["arm"], raw["id"])
                if key not in expected_cells:
                    raise ValueError(f"{poses_path}:{line_number}: cell outside manifest")
                pose_index = int(raw["pose_index"])
                if not 0 <= pose_index < NUM_SAMPLES:
                    raise ValueError(f"{poses_path}:{line_number}: invalid pose index")
                checks = {
                    check: _as_bool(raw[check], label=f"{poses_path}:{line_number}.{check}")
                    for check in VALIDITY_CHECKS
                }
                valid = _as_bool(
                    raw["posebusters_valid"],
                    label=f"{poses_path}:{line_number}.posebusters_valid",
                )
                if valid != all(checks.values()):
                    raise ValueError(f"{poses_path}:{line_number}: PB-valid conjunction mismatch")
                by_cell.setdefault(key, []).append(
                    {
                        "pose_index": pose_index,
                        "posebusters_valid": valid,
                        "rmsd_success": _as_bool(
                            raw["separate_rmsd_check"],
                            label=f"{poses_path}:{line_number}.separate_rmsd_check",
                        ),
                        "checks": checks,
                    }
                )
                shard_pose_count += 1
        assigned_cells = int(summary.get("assigned_cells", -1))
        assigned_poses = int(summary.get("assigned_poses", -1))
        if shard_pose_count != assigned_poses or assigned_poses != assigned_cells * NUM_SAMPLES:
            raise ValueError(f"{summary_path}: assigned/result pose counts disagree")
        total_summary_cells += assigned_cells
        total_summary_poses += assigned_poses

    if set(by_cell) != expected_cells:
        raise ValueError("official result cell set differs from manifest")
    for key, rows in by_cell.items():
        indexes = [int(row["pose_index"]) for row in rows]
        if sorted(indexes) != list(range(NUM_SAMPLES)):
            raise ValueError(f"{key}: official pose indexes are incomplete or duplicated")
        rows.sort(key=lambda row: int(row["pose_index"]))
    if total_summary_cells != len(expected_cells):
        raise ValueError("official summary cell total mismatch")
    if total_summary_poses != len(expected_cells) * NUM_SAMPLES:
        raise ValueError("official summary pose total mismatch")
    if observed_rmsd_check is None:
        raise ValueError("no official RMSD check was observed")
    return by_cell, observed_rmsd_check


def _pct(count: int, total: int) -> float:
    return 100.0 * count / total


def _arm_metrics(
    records: dict[str, dict[str, Any]],
    official: dict[tuple[str, str], list[dict[str, Any]]],
    *,
    arm: str,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    per_complex: dict[str, dict[str, Any]] = {}
    check_counts = {check: 0 for check in VALIDITY_CHECKS}
    for complex_id in sorted(records):
        record = records[complex_id]
        poses = official[(arm, complex_id)]
        selected = poses[int(record["selected_index"])]
        for pose in poses:
            for check in VALIDITY_CHECKS:
                check_counts[check] += int(pose["checks"][check])
        per_complex[complex_id] = {
            "selected_pb_valid": bool(selected["posebusters_valid"]),
            "selected_rmsd_success": bool(selected["rmsd_success"]),
            "selected_joint": bool(
                selected["posebusters_valid"] and selected["rmsd_success"]
            ),
            "selected_rmsd": float(record["selected_rmsd"]),
            "evaluator_oracle_rmsd": float(record["evaluator_oracle_rmsd"]),
            "oracle_rmsd_success": any(bool(pose["rmsd_success"]) for pose in poses),
            "oracle_joint": any(
                bool(pose["posebusters_valid"] and pose["rmsd_success"])
                for pose in poses
            ),
            "valid_candidates": sum(bool(pose["posebusters_valid"]) for pose in poses),
            "joint_candidates": sum(
                bool(pose["posebusters_valid"] and pose["rmsd_success"]) for pose in poses
            ),
            "diversity": record["diversity"],
            "ess_fractions": [float(value) for value in record["ess_fractions"]],
            "final_unique_initial_ancestors": int(
                record["final_unique_initial_ancestors"]
            ),
        }
    rows = [per_complex[key] for key in sorted(per_complex)]
    complexes = len(rows)
    poses_total = complexes * NUM_SAMPLES
    selected_valid_count = sum(row["selected_pb_valid"] for row in rows)
    selected_rmsd_count = sum(row["selected_rmsd_success"] for row in rows)
    selected_joint_count = sum(row["selected_joint"] for row in rows)
    oracle_rmsd_count = sum(row["oracle_rmsd_success"] for row in rows)
    oracle_joint_count = sum(row["oracle_joint"] for row in rows)
    valid_candidates = sum(int(row["valid_candidates"]) for row in rows)
    joint_candidates = sum(int(row["joint_candidates"]) for row in rows)
    diversity = [row["diversity"] for row in rows]
    coordinate_unique = sum(int(value["coordinate_unique_count"]) for value in diversity)
    ess = [value for row in rows for value in row["ess_fractions"]]
    metrics = {
        "complexes": complexes,
        "confidence_selected": {
            "posebusters_valid_count": selected_valid_count,
            "posebusters_valid_pct": _pct(selected_valid_count, complexes),
            "official_rmsd_le2_count": selected_rmsd_count,
            "official_rmsd_le2_pct": _pct(selected_rmsd_count, complexes),
            "joint_valid_rmsd_le2_count": selected_joint_count,
            "joint_valid_rmsd_le2_pct": _pct(selected_joint_count, complexes),
            "median_mapped_rmsd": float(
                statistics.median(float(row["selected_rmsd"]) for row in rows)
            ),
        },
        "oracle": {
            "official_rmsd_le2_count": oracle_rmsd_count,
            "official_rmsd_le2_pct": _pct(oracle_rmsd_count, complexes),
            "joint_valid_rmsd_le2_count": oracle_joint_count,
            "joint_valid_rmsd_le2_pct": _pct(oracle_joint_count, complexes),
            "median_evaluator_oracle_rmsd": float(
                statistics.median(float(row["evaluator_oracle_rmsd"]) for row in rows)
            ),
        },
        "candidate_set": {
            "poses": poses_total,
            "posebusters_valid_count": valid_candidates,
            "posebusters_valid_pct": _pct(valid_candidates, poses_total),
            "joint_valid_rmsd_le2_count": joint_candidates,
            "joint_valid_rmsd_le2_pct": _pct(joint_candidates, poses_total),
            "mean_valid_candidates_per_complex": float(
                statistics.fmean(int(row["valid_candidates"]) for row in rows)
            ),
            "terminal_unique_coordinate_fraction": coordinate_unique / poses_total,
            "complexes_with_all_terminal_coordinates_unique": sum(
                int(value["coordinate_unique_count"]) == NUM_SAMPLES for value in diversity
            ),
            "median_complex_pairwise_heavy_atom_rmsd": float(
                statistics.median(
                    float(value["pairwise_heavy_atom_rmsd_median"]) for value in diversity
                )
            ),
            "mean_pairwise_heavy_atom_rmsd_ge2_fraction": float(
                statistics.fmean(
                    float(value["pairwise_heavy_atom_rmsd_ge2_fraction"])
                    for value in diversity
                )
            ),
            "median_complex_nearest_neighbor_heavy_atom_rmsd": float(
                statistics.median(
                    float(value["nearest_neighbor_heavy_atom_rmsd_median"])
                    for value in diversity
                )
            ),
        },
        "feynman_kac": {
            "events": len(ess),
            "ess_fraction_min": min(ess),
            "ess_fraction_median": float(statistics.median(ess)),
            "final_unique_initial_ancestor_fraction_mean": float(
                statistics.fmean(
                    int(row["final_unique_initial_ancestors"]) / NUM_SAMPLES for row in rows
                )
            ),
        },
        "all_pose_per_check_pass_pct": {
            check: _pct(count, poses_total) for check, count in check_counts.items()
        },
    }
    return metrics, per_complex


def _paired_metrics(
    baseline: dict[str, dict[str, Any]], comparison: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    if sorted(baseline) != sorted(comparison):
        raise ValueError("paired arms have different complex IDs")
    ids = sorted(baseline)

    def delta(key: str) -> tuple[int, float]:
        before = sum(bool(baseline[complex_id][key]) for complex_id in ids)
        after = sum(bool(comparison[complex_id][key]) for complex_id in ids)
        return after - before, _pct(after - before, len(ids))

    selected_joint_count, selected_joint_pct = delta("selected_joint")
    selected_valid_count, selected_valid_pct = delta("selected_pb_valid")
    selected_rmsd_count, selected_rmsd_pct = delta("selected_rmsd_success")
    oracle_joint_count, oracle_joint_pct = delta("oracle_joint")
    oracle_rmsd_count, oracle_rmsd_pct = delta("oracle_rmsd_success")
    return {
        "complexes": len(ids),
        "selected_joint_valid_rmsd_le2_count_delta": selected_joint_count,
        "selected_joint_valid_rmsd_le2_pct_delta": selected_joint_pct,
        "selected_posebusters_valid_count_delta": selected_valid_count,
        "selected_posebusters_valid_pct_delta": selected_valid_pct,
        "selected_official_rmsd_le2_count_delta": selected_rmsd_count,
        "selected_official_rmsd_le2_pct_delta": selected_rmsd_pct,
        "selected_joint_gained_complexes": sum(
            not baseline[complex_id]["selected_joint"]
            and comparison[complex_id]["selected_joint"]
            for complex_id in ids
        ),
        "selected_joint_lost_complexes": sum(
            baseline[complex_id]["selected_joint"]
            and not comparison[complex_id]["selected_joint"]
            for complex_id in ids
        ),
        "selected_paired_median_mapped_rmsd_delta": float(
            statistics.median(
                float(comparison[complex_id]["selected_rmsd"])
                - float(baseline[complex_id]["selected_rmsd"])
                for complex_id in ids
            )
        ),
        "oracle_joint_valid_rmsd_le2_count_delta": oracle_joint_count,
        "oracle_joint_valid_rmsd_le2_pct_delta": oracle_joint_pct,
        "oracle_official_rmsd_le2_count_delta": oracle_rmsd_count,
        "oracle_official_rmsd_le2_pct_delta": oracle_rmsd_pct,
        "candidate_posebusters_valid_pct_delta": float(
            statistics.fmean(
                (
                    int(comparison[complex_id]["valid_candidates"])
                    - int(baseline[complex_id]["valid_candidates"])
                )
                / NUM_SAMPLES
                * 100.0
                for complex_id in ids
            )
        ),
        "candidate_joint_valid_rmsd_le2_pct_delta": float(
            statistics.fmean(
                (
                    int(comparison[complex_id]["joint_candidates"])
                    - int(baseline[complex_id]["joint_candidates"])
                )
                / NUM_SAMPLES
                * 100.0
                for complex_id in ids
            )
        ),
        "terminal_unique_coordinate_fraction_delta": float(
            statistics.fmean(
                (
                    int(comparison[complex_id]["diversity"]["coordinate_unique_count"])
                    - int(baseline[complex_id]["diversity"]["coordinate_unique_count"])
                )
                / NUM_SAMPLES
                for complex_id in ids
            )
        ),
        "pairwise_heavy_atom_rmsd_median_paired_delta": float(
            statistics.median(
                float(
                    comparison[complex_id]["diversity"][
                        "pairwise_heavy_atom_rmsd_median"
                    ]
                )
                - float(
                    baseline[complex_id]["diversity"][
                        "pairwise_heavy_atom_rmsd_median"
                    ]
                )
                for complex_id in ids
            )
        ),
        "nearest_neighbor_heavy_atom_rmsd_median_paired_delta": float(
            statistics.median(
                float(
                    comparison[complex_id]["diversity"][
                        "nearest_neighbor_heavy_atom_rmsd_median"
                    ]
                )
                - float(
                    baseline[complex_id]["diversity"][
                        "nearest_neighbor_heavy_atom_rmsd_median"
                    ]
                )
                for complex_id in ids
            )
        ),
    }


def build_report(
    *,
    manifest_path: Path,
    manifest_sha256: str,
    official_root: Path,
    num_shards: int,
    expected_mode: str,
) -> dict[str, Any]:
    if _sha256(manifest_path) != manifest_sha256:
        raise ValueError("report manifest SHA-256 mismatch")
    manifest = _read_json(manifest_path)
    _validate_manifest_identity(manifest, expected_mode=expected_mode)
    official, rmsd_check = _load_official_results(
        manifest=manifest,
        manifest_sha256=manifest_sha256,
        official_root=official_root,
        num_shards=num_shards,
    )
    records_by_arm: dict[str, dict[str, dict[str, Any]]] = {
        arm: {} for arm in ARM_SPECS
    }
    for record in manifest["records"]:
        records_by_arm[str(record["arm"])][str(record["id"])] = record
    ids = sorted(next(iter(records_by_arm.values())))
    for complex_id in ids:
        paired = {
            (
                rows[complex_id]["sampling_seed"],
                rows[complex_id]["prior_pool_sha256"],
            )
            for rows in records_by_arm.values()
        }
        if len(paired) != 1:
            raise ValueError(f"{complex_id}: manifest prior identity is no longer paired")
    arm_metrics: dict[str, Any] = {}
    per_complex: dict[str, dict[str, dict[str, Any]]] = {}
    for arm in ARM_SPECS:
        arm_metrics[arm], per_complex[arm] = _arm_metrics(
            records_by_arm[arm], official, arm=arm
        )
    paired = _paired_metrics(per_complex["fk_ode"], per_complex["fk_sde"])
    expected_poses = int(manifest["expected_poses"])
    result: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "mode": expected_mode,
        "status": "complete" if expected_mode == "full" else "smoke_passed",
        "claim_boundary": "paired_descriptive_external_evidence_only",
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "posebusters_version": POSEBUSTERS_VERSION,
        "posebusters_config": POSEBUSTERS_CONFIG,
        "posebusters_validity_definition": "all 27 non-RMSD redock checks",
        "official_rmsd_check": rmsd_check,
        "checks": {
            "paired_sampling_seed_and_prior_identity": True,
            "all_pose_sdf_hashes_and_counts_exact": True,
            "fk_diagnostics_complete_and_finite": True,
            "official_cells_complete": len(official) == int(manifest["expected_cells"]),
            "official_pose_rows_complete": (
                sum(len(rows) for rows in official.values()) == expected_poses
            ),
        },
        "arms": arm_metrics,
        "primary_contrast_fk_sde_minus_fk_ode": paired,
    }
    if expected_mode == "full":
        selected_joint_gate = paired["selected_joint_valid_rmsd_le2_pct_delta"] >= 2.0
        selected_rmsd_guard = paired["selected_official_rmsd_le2_pct_delta"] >= -2.0
        oracle_joint_guard = paired["oracle_joint_valid_rmsd_le2_pct_delta"] >= -2.0
        diversity_gate = (
            arm_metrics["fk_sde"]["candidate_set"][
                "terminal_unique_coordinate_fraction"
            ]
            > 0.90
        )
        result["predeclared_utility_gate"] = {
            "selected_joint_gain_at_least_2pp": selected_joint_gate,
            "selected_rmsd_decrease_not_worse_than_2pp": selected_rmsd_guard,
            "oracle_joint_decrease_not_worse_than_2pp": oracle_joint_guard,
            "fk_sde_terminal_unique_fraction_above_0p90": diversity_gate,
            "supported": (
                selected_joint_gate
                and selected_rmsd_guard
                and oracle_joint_guard
                and diversity_gate
            ),
        }
    return result


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# FK Translation-SDE PoseBusters Results",
        "",
        f"- Protocol: `{report['protocol_id']}`",
        "- Claim boundary: paired descriptive external evidence only",
        f"- Official RMSD check: `{report['official_rmsd_check']}`",
        "",
        "| Arm | Selected PB-valid | Selected RMSD <=2 A | Selected joint | "
        "Oracle joint | All-pose PB-valid | Terminal unique | Pairwise RMSD |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    complexes = report["arms"]["fk_ode"]["complexes"]
    for arm in ARM_SPECS:
        metrics = report["arms"][arm]
        selected = metrics["confidence_selected"]
        oracle = metrics["oracle"]
        candidates = metrics["candidate_set"]
        lines.append(
            f"| {arm} | {selected['posebusters_valid_count']}/{complexes} "
            f"({selected['posebusters_valid_pct']:.2f}%) | "
            f"{selected['official_rmsd_le2_count']}/{complexes} "
            f"({selected['official_rmsd_le2_pct']:.2f}%) | "
            f"{selected['joint_valid_rmsd_le2_count']}/{complexes} "
            f"({selected['joint_valid_rmsd_le2_pct']:.2f}%) | "
            f"{oracle['joint_valid_rmsd_le2_count']}/{complexes} "
            f"({oracle['joint_valid_rmsd_le2_pct']:.2f}%) | "
            f"{candidates['posebusters_valid_pct']:.2f}% | "
            f"{candidates['terminal_unique_coordinate_fraction']:.3f} | "
            f"{candidates['median_complex_pairwise_heavy_atom_rmsd']:.3f} A |"
        )
    primary = report["primary_contrast_fk_sde_minus_fk_ode"]
    lines.extend(
        [
            "",
            "## Primary contrast: FK-SDE minus FK-ODE",
            "",
            f"- Selected joint: "
            f"{primary['selected_joint_valid_rmsd_le2_pct_delta']:+.2f} pp",
            f"- Selected PB-valid: "
            f"{primary['selected_posebusters_valid_pct_delta']:+.2f} pp",
            f"- Selected official RMSD <=2 A: "
            f"{primary['selected_official_rmsd_le2_pct_delta']:+.2f} pp",
            f"- Oracle joint: "
            f"{primary['oracle_joint_valid_rmsd_le2_pct_delta']:+.2f} pp",
            f"- All-pose PB-valid: "
            f"{primary['candidate_posebusters_valid_pct_delta']:+.2f} pp",
            f"- Terminal uniqueness: "
            f"{primary['terminal_unique_coordinate_fraction_delta']:+.3f}",
            f"- Pairwise heavy-atom RMSD median: "
            f"{primary['pairwise_heavy_atom_rmsd_median_paired_delta']:+.3f} A",
            "",
            "PoseBusters was already opened; these values cannot tune settings or "
            "admit the method.",
        ]
    )
    if "predeclared_utility_gate" in report:
        lines.insert(
            -2,
            f"- Predeclared utility gate supported: "
            f"`{str(report['predeclared_utility_gate']['supported']).lower()}`",
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_json_exclusive(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifest = subparsers.add_parser("manifest")
    manifest.add_argument("--input-root", type=Path, required=True)
    manifest.add_argument("--cohort-manifest", type=Path, required=True)
    manifest.add_argument("--mode", choices=("smoke", "full"), required=True)
    manifest.add_argument("--output", type=Path, required=True)

    score = subparsers.add_parser("score")
    score.add_argument("--manifest", type=Path, required=True)
    score.add_argument("--manifest-sha256", required=True)
    score.add_argument("--output-root", type=Path, required=True)
    score.add_argument("--num-shards", type=int, required=True)
    score.add_argument("--shard-index", type=int, required=True)
    score.add_argument("--workers", type=int, default=4)

    smoke = subparsers.add_parser("smoke-audit")
    smoke.add_argument("--manifest", type=Path, required=True)
    smoke.add_argument("--manifest-sha256", required=True)
    smoke.add_argument("--official-root", type=Path, required=True)
    smoke.add_argument("--output", type=Path, required=True)

    report = subparsers.add_parser("report")
    report.add_argument("--manifest", type=Path, required=True)
    report.add_argument("--manifest-sha256", required=True)
    report.add_argument("--official-root", type=Path, required=True)
    report.add_argument("--num-shards", type=int, default=OFFICIAL_SHARDS)
    report.add_argument("--output-json", type=Path, required=True)
    report.add_argument("--output-markdown", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    if args.command == "manifest":
        result = build_manifest(
            input_root=args.input_root,
            cohort_manifest=args.cohort_manifest,
            mode=args.mode,
        )
        _write_json_exclusive(args.output, result)
        print(
            json.dumps(
                {
                    "status": "complete",
                    "mode": args.mode,
                    "cells": result["expected_cells"],
                    "poses": result["expected_poses"],
                    "sha256": _sha256(args.output),
                },
                sort_keys=True,
            )
        )
    elif args.command == "score":
        result = score_official_shard(
            manifest_path=args.manifest,
            manifest_sha256=args.manifest_sha256,
            output_root=args.output_root,
            num_shards=args.num_shards,
            shard_index=args.shard_index,
            workers=args.workers,
        )
        print(json.dumps(result, sort_keys=True))
    elif args.command == "smoke-audit":
        result = build_report(
            manifest_path=args.manifest,
            manifest_sha256=args.manifest_sha256,
            official_root=args.official_root,
            num_shards=1,
            expected_mode="smoke",
        )
        _write_json_exclusive(args.output, result)
        print(json.dumps(result["checks"], sort_keys=True))
    elif args.command == "report":
        result = build_report(
            manifest_path=args.manifest,
            manifest_sha256=args.manifest_sha256,
            official_root=args.official_root,
            num_shards=args.num_shards,
            expected_mode="full",
        )
        _write_json_exclusive(args.output_json, result)
        _write_markdown(result, args.output_markdown)
        print(json.dumps(result["primary_contrast_fk_sde_minus_fk_ode"], sort_keys=True))
    else:  # pragma: no cover
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
