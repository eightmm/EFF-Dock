#!/usr/bin/env python3
"""Fail-closed replay identity gate for the eta-sweep confidence extension.

The confidence checkpoint is intentionally evaluated in a second process.  This
module proves exact discrete identity and bounded numerical equivalence to the
frozen no-confidence sampling run, including atom-ordered comparisons of the
three saved-pose sentinels, before downstream PoseBusters results are allowed
to use the new selectors.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import torch
from rdkit import Chem

from effdock.confidence.selectors import ConfidenceFilterConfig, select_confidence_filter
from effdock.workflows.evaluate import sorted_id_sha256
from effdock.workflows.guidance_budget_full_report import (
    EXPECTED_BENCHMARK_INPUT_MANIFEST_SHA256,
    EXPECTED_DATASET_COUNTS,
    RECEPTOR_POLICY,
)
from effdock.workflows.guidance_budget_report import (
    DATASETS,
    EXPECTED_CHECKPOINT_SHA256,
    EXPECTED_CONFIG_SHA256,
    EXPECTED_GUIDANCE_PARAMETER_SHA256,
    EXPECTED_POCKET_CENTERS_SHA256,
)
from effdock.workflows.guidance_eta_sweep_report import (
    _REQUIRED_SUMMARY_KEYS,
    ETA_TAGS,
    ETA_VALUES,
    EXPECTED_GUIDANCE_MODE,
    NUM_SAMPLES,
    NUM_STEPS,
    _validate_direct_runtime,
    _validate_direct_step_trace,
    _validate_trace_runtime_consistency,
    eta_tag,
    expected_run_name,
)
from effdock.workflows.guidance_eta_sweep_report import (
    PROTOCOL_ID as PARENT_PROTOCOL_ID,
)

PROTOCOL_ID = "EFFDOCK-UNIFIED-GUIDANCE-ETA-SWEEP-CONFIDENCE-PB-V1"
AUDIT_CONTRACT = "EFFDOCK_CONFIDENCE_REPLAY_IDENTITY_V2"
SCHEMA_VERSION = "effdock.guidance_eta_sweep_confidence_identity.v2"
EXPECTED_CONFIDENCE_CHECKPOINT_SHA256 = (
    "e31fde6f351284205c78f7a1510002779c43312e94d9f82003d47a14d72bc78f"
)
EXPECTED_CONFIDENCE_STEP = 42_500
GPU_NAME_FRAGMENTS_BY_PARTITION = {
    "6000ada": ("RTX 6000 Ada",),
    "heavy": ("H100", "RTX PRO 6000"),
}
ALLOWED_GPU_NAME_FRAGMENTS = tuple(
    fragment for fragments in GPU_NAME_FRAGMENTS_BY_PARTITION.values() for fragment in fragments
)
EXPECTED_GPU_FRAGMENT = ALLOWED_GPU_NAME_FRAGMENTS[0]
MIN_GPU_TOTAL_MEMORY_BYTES = 48_000 * 1024**2
GPU_MEMORY_LIMIT_BYTES = 48 * 1024**3
EXPECTED_RECEPTOR_POLICY_SHA256 = "7bd75b1ff265b46fb556f7770ed5c393ad349304ae4ceedc0564dde93e26c5fd"
CANDIDATE_ENSEMBLE_HASH_CONTRACT = "EFFDOCK_CANDIDATE_ENSEMBLE_V1"
CONFIDENCE_SCORE_LEDGER_CONTRACT = "EFFDOCK_CONFIDENCE_SCORE_LEDGER_V1"
SCALAR_REL_TOL = 2e-5
SCALAR_ABS_TOL = 1e-4
TELEMETRY_REL_TOL = 2e-4
TELEMETRY_ABS_TOL = 2e-4
POSE_COORDINATE_RMSD_TOL_ANGSTROM = 5e-4
POSE_MAX_ATOM_DISPLACEMENT_TOL_ANGSTROM = 1e-3
EXPECTED_BENCHMARK_IDENTITIES = {
    "astex": {
        "sha256": "331d88531a0e57ac0b11053d47797b1ceec76c48f96d96154fe9067347b9547c",
        "ids_sha256": "79a5056ccb451a3c94cc29d07806e97d1c1231ba394955e1be6eedfab606bad4",
        "mapping_sha256": "4f9b8c5f2753949bd14ba9c45f920cab0ebd63f2a0110126ca429645070200ca",
    },
    "posebusters": {
        "sha256": "31f95bfd6710c6d993970037606b2cab51aac8d9d14919b4ba676c94fe775781",
        "ids_sha256": "ecbeb7b6a01c1a4eddf44e5b4e19cf9222042ebc100f563bf50ccabde33715af",
        "mapping_sha256": "6f3f0cf229131559ea7ff095cc6b87beaa55e4e6b09d5626f555643a1e06f74f",
    },
}
SMOKE_IDS = {"astex": "1jje", "posebusters": "7b2c_tp7"}

LEGACY_SELECTORS = ("first", "vina", "oracle")
CONFIDENCE_SELECTORS = ("confidence", "confidence_filter", "confidence_final")
_FAST_TERMS = ("valid", "bond", "angle", "internal_clash", "protein_clash")
_RUNTIME_SCORE_FIELDS = (
    "confidence_rmsd",
    "confidence_success_logit",
    "confidence_success",
    "confidence_atom_rmsd",
    "confidence_atom_q90",
    "confidence_atom_ok",
)
_CONFIDENCE_COLUMNS = {
    "candidate_ensemble_sha256",
    "confidence_candidate_scores_json",
    *(f"{selector}_fast_{term}" for selector in CONFIDENCE_SELECTORS for term in _FAST_TERMS),
    *(
        f"{selector}_{suffix}"
        for selector in CONFIDENCE_SELECTORS
        for suffix in ("index", "rmsd", "pred_rmsd", "pred_success")
    ),
}
_SUMMARY_DIFFERENCES = {
    "protocol_id",
    "confidence_checkpoint",
    "confidence_step",
    "confidence_checkpoint_sha256",
    "csv",
    "runtime",
    "stats",
    "guidance_implementation",
    "guidance_runtime_stats",
}
_CONFIDENCE_ONLY_SUMMARY_KEYS = {
    "candidate_ensemble_hash_contract",
    "confidence_score_ledger_contract",
}
_EXTRA_REQUIRED_SUMMARY_KEYS = {
    "benchmark_input_identity",
    "checkpoint_step",
    "confidence_checkpoint",
    "confidence_step",
    "expected_discovered_count",
    "num_assigned",
    "num_failed",
    "num_success",
    "require_complete_success",
    "stats",
}
_EXPECTED_FILTER_CONFIG = {
    "pred_rmsd_margin": 0.03,
    "success_gain": 0.0,
    "atom_ok_gain": 0.0,
    "clash_limit": 0.0,
    "fallback_head_tolerance": 0.0,
    "mode": "strict_both",
    "atom_rmsd_gain": 0.0,
    "success_tolerance": 0.05,
    "atom_ok_tolerance": 0.02,
}
_SCALAR_CSV_FIELDS = {
    "first_rmsd",
    "vina_rmsd",
    "oracle_rmsd",
    "fast_valid_oracle_rmsd",
    "vina_score",
    "vina_strain",
    "vina_total",
    "mean_sample_rmsd",
}
_DISCRETE_SUFFIXES = (
    "_index",
    "_count",
    "_attempted",
    "_accepted",
    "_rejected",
    "_backtracks",
    "_evaluations",
    "_trials",
    "_poses",
    "_applied",
)


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{label} must be a 64-character SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{label} must be hexadecimal") from exc
    return value


def _ordered_sha256(values: list[str], *, domain: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(domain)
    for value in values:
        digest.update(value.encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _records_sha256(records: list[dict[str, Any]], *, domain: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(domain)
    for record in records:
        digest.update(_canonical_json(record).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _new_delta_bucket() -> dict[str, Any]:
    return {
        "comparisons": 0,
        "max_absolute_delta": 0.0,
        "max_absolute_delta_location": None,
        "max_absolute_delta_parent": None,
        "max_absolute_delta_replay": None,
        "max_relative_delta": 0.0,
        "max_relative_delta_location": None,
        "max_meaningful_relative_delta": 0.0,
        "max_meaningful_relative_delta_location": None,
    }


def _new_delta_observations() -> dict[str, dict[str, Any]]:
    return {
        "legacy_csv_scalars": _new_delta_bucket(),
        "legacy_summary_stats": _new_delta_bucket(),
        "summary_guidance_telemetry": _new_delta_bucket(),
        "direct_step_telemetry": _new_delta_bucket(),
    }


def _record_numeric_delta(
    bucket: dict[str, Any],
    parent: float,
    replay: float,
    *,
    label: str,
    rel_tol: float,
    abs_tol: float,
) -> None:
    absolute = abs(parent - replay)
    scale = max(abs(parent), abs(replay))
    relative = absolute / scale if scale > 0.0 else 0.0
    bucket["comparisons"] += 1
    if absolute > float(bucket["max_absolute_delta"]):
        bucket["max_absolute_delta"] = absolute
        bucket["max_absolute_delta_location"] = label
        bucket["max_absolute_delta_parent"] = parent
        bucket["max_absolute_delta_replay"] = replay
    if relative > float(bucket["max_relative_delta"]):
        bucket["max_relative_delta"] = relative
        bucket["max_relative_delta_location"] = label
    absolute_floor_scale = abs_tol / rel_tol if rel_tol > 0.0 else math.inf
    if scale > absolute_floor_scale and relative > float(bucket["max_meaningful_relative_delta"]):
        bucket["max_meaningful_relative_delta"] = relative
        bucket["max_meaningful_relative_delta_location"] = label


def _compare_finite_numeric(
    parent: Any,
    replay: Any,
    *,
    label: str,
    rel_tol: float,
    abs_tol: float,
    bucket: dict[str, Any],
) -> None:
    if isinstance(parent, bool) or isinstance(replay, bool):
        raise ValueError(f"{label}: numerical replay values may not be booleans")
    try:
        parent_value = float(parent)
        replay_value = float(replay)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}: replay values must both be numeric") from exc
    if not math.isfinite(parent_value) or not math.isfinite(replay_value):
        raise ValueError(f"{label}: replay values must both be finite")
    _record_numeric_delta(
        bucket,
        parent_value,
        replay_value,
        label=label,
        rel_tol=rel_tol,
        abs_tol=abs_tol,
    )
    if not math.isclose(
        parent_value,
        replay_value,
        rel_tol=rel_tol,
        abs_tol=abs_tol,
    ):
        raise ValueError(
            f"{label}: numerical replay mismatch; parent={parent_value!r}, "
            f"replay={replay_value!r}, rel_tol={rel_tol}, abs_tol={abs_tol}"
        )


def _is_discrete_metric(name: str) -> bool:
    return name in {"count", "pose_count", "finite_count", "applied_count", "eta"} or name.endswith(
        _DISCRETE_SUFFIXES
    )


def _compare_numeric_tree(
    parent: Any,
    replay: Any,
    *,
    label: str,
    rel_tol: float,
    abs_tol: float,
    bucket: dict[str, Any],
) -> None:
    if isinstance(parent, dict):
        if not isinstance(replay, dict) or set(parent) != set(replay):
            raise ValueError(f"{label}: replay object keys differ")
        for key in sorted(parent):
            child_label = f"{label}.{key}"
            if _is_discrete_metric(key):
                if parent[key] != replay[key]:
                    raise ValueError(f"{child_label}: discrete replay value differs")
            else:
                _compare_numeric_tree(
                    parent[key],
                    replay[key],
                    label=child_label,
                    rel_tol=rel_tol,
                    abs_tol=abs_tol,
                    bucket=bucket,
                )
        return
    if isinstance(parent, list):
        if not isinstance(replay, list) or len(parent) != len(replay):
            raise ValueError(f"{label}: replay list length differs")
        for index, (parent_value, replay_value) in enumerate(zip(parent, replay, strict=True)):
            _compare_numeric_tree(
                parent_value,
                replay_value,
                label=f"{label}[{index}]",
                rel_tol=rel_tol,
                abs_tol=abs_tol,
                bucket=bucket,
            )
        return
    if parent is None or replay is None:
        if parent is not replay:
            raise ValueError(f"{label}: replay nullability differs")
        return
    if isinstance(parent, (int, float)) and not isinstance(parent, bool):
        _compare_finite_numeric(
            parent,
            replay,
            label=label,
            rel_tol=rel_tol,
            abs_tol=abs_tol,
            bucket=bucket,
        )
        return
    if parent != replay:
        raise ValueError(f"{label}: categorical replay value differs")


def _merge_delta_observations(
    target: dict[str, dict[str, Any]], source: dict[str, dict[str, Any]]
) -> None:
    for name, source_bucket in source.items():
        target_bucket = target[name]
        target_bucket["comparisons"] += int(source_bucket["comparisons"])
        for metric in (
            "max_absolute_delta",
            "max_relative_delta",
            "max_meaningful_relative_delta",
        ):
            if float(source_bucket[metric]) <= float(target_bucket[metric]):
                continue
            target_bucket[metric] = source_bucket[metric]
            target_bucket[f"{metric}_location"] = source_bucket[f"{metric}_location"]
            if metric == "max_absolute_delta":
                target_bucket["max_absolute_delta_parent"] = source_bucket[
                    "max_absolute_delta_parent"
                ]
                target_bucket["max_absolute_delta_replay"] = source_bucket[
                    "max_absolute_delta_replay"
                ]


def _is_tolerant_csv_scalar(field: str) -> bool:
    if field in _SCALAR_CSV_FIELDS:
        return True
    if not field.startswith("guidance_"):
        return False
    return not (
        field.endswith("_json")
        or "sha256" in field
        or field in {"guidance_mode", "guidance_receptor_policy"}
        or _is_discrete_metric(field)
    )


def _compare_common_csv_field(
    parent: str,
    replay: str,
    *,
    field: str,
    label: str,
    observations: dict[str, dict[str, Any]],
) -> None:
    if field == "guidance_direct_step_trace_json":
        try:
            parent_trace = json.loads(parent)
            replay_trace = json.loads(replay)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{label}: invalid direct-step telemetry JSON") from exc
        _compare_numeric_tree(
            parent_trace,
            replay_trace,
            label=label,
            rel_tol=TELEMETRY_REL_TOL,
            abs_tol=TELEMETRY_ABS_TOL,
            bucket=observations["direct_step_telemetry"],
        )
        return
    if _is_tolerant_csv_scalar(field):
        if parent == "" or replay == "":
            if parent != replay:
                raise ValueError(f"{label}: scalar replay nullability differs")
            return
        if (
            field == "fast_valid_oracle_rmsd"
            and parent == replay
            and parent
            in {
                "inf",
                "-inf",
            }
        ):
            return
        _compare_finite_numeric(
            parent,
            replay,
            label=label,
            rel_tol=SCALAR_REL_TOL,
            abs_tol=SCALAR_ABS_TOL,
            bucket=observations["legacy_csv_scalars"],
        )
        return
    if parent != replay:
        raise ValueError(f"{label}: categorical/discrete replay value differs")


def _same_scalar(actual: Any, expected: Any) -> bool:
    if isinstance(expected, float):
        try:
            return math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-12)
        except (TypeError, ValueError):
            return False
    return actual == expected


def _require_setting(summary: dict[str, Any], key: str, expected: Any, *, label: str) -> None:
    if key not in summary or not _same_scalar(summary[key], expected):
        raise ValueError(f"{label}: {key} must be {expected!r}, got {summary.get(key)!r}")


def _summary_filename(run_name: str, shard_index: int, *, smoke: bool) -> str:
    if smoke:
        return f"{run_name}.summary.json"
    return f"{run_name}.shard-{shard_index:03d}-of-008.summary.json"


def _expected_summary_paths(
    root: Path,
    *,
    smoke: bool,
    eta_values: tuple[float, ...] = ETA_VALUES,
    eta_tags: tuple[str, ...] = ETA_TAGS,
    expected_run_name_fn: Callable[[str, float], str] = expected_run_name,
) -> dict[tuple[str, str, int], Path]:
    shards = 1 if smoke else 8
    expected = {
        (dataset, tag, shard): root
        / _summary_filename(expected_run_name_fn(dataset, eta), shard, smoke=smoke)
        for dataset in DATASETS
        for eta, tag in zip(eta_values, eta_tags, strict=True)
        for shard in range(shards)
    }
    observed = set(root.glob("*.summary.json")) if root.is_dir() else set()
    required = set(expected.values())
    if observed != required:
        raise ValueError(
            f"{root}: summary inventory mismatch; "
            f"missing={[path.name for path in sorted(required - observed)[:5]]}, "
            f"extra={[path.name for path in sorted(observed - required)[:5]]}"
        )
    return expected


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _companion_csv(summary_path: Path, summary: dict[str, Any], root: Path) -> Path:
    expected = summary_path.with_name(summary_path.name.removesuffix(".summary.json") + ".csv")
    if not expected.is_file():
        raise FileNotFoundError(f"{summary_path}: missing companion CSV {expected}")
    raw_value = summary.get("csv")
    if not isinstance(raw_value, str) or not raw_value:
        raise ValueError(f"{summary_path}: summary csv path must be non-empty")
    raw = Path(raw_value)
    candidates = (
        [raw] if raw.is_absolute() else [Path.cwd() / raw, root / raw, summary_path.parent / raw]
    )
    existing = {candidate.resolve() for candidate in candidates if candidate.is_file()}
    if existing != {expected.resolve()}:
        raise ValueError(
            f"{summary_path}: summary csv path does not resolve uniquely to companion CSV"
        )
    return expected


def _read_raw_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        if not fields:
            raise ValueError(f"{path}: missing CSV header")
        if len(fields) != len(set(fields)):
            raise ValueError(f"{path}: duplicate CSV field names")
        rows = list(reader)
    for line_number, row in enumerate(rows, start=2):
        if None in row or any(value is None for value in row.values()):
            raise ValueError(f"{path}:{line_number}: malformed or missing CSV value")
        if not row.get("id"):
            raise ValueError(f"{path}:{line_number}: empty complex ID")
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path}: duplicate complex IDs")
    return fields, rows


def _validate_benchmark_identity(summary: dict[str, Any], dataset: str, *, label: str) -> None:
    identity = summary.get("benchmark_input_identity")
    if not isinstance(identity, dict):
        raise ValueError(f"{label}: benchmark_input_identity must be an object")
    exact = {
        "dataset": dataset,
        "count": EXPECTED_DATASET_COUNTS[dataset],
        "mode": "frozen_manifest",
        **EXPECTED_BENCHMARK_IDENTITIES[dataset],
    }
    for key, expected in exact.items():
        _require_setting(identity, key, expected, label=f"{label}.benchmark_input_identity")
    source = identity.get("sources", {}).get("frozen_manifest")
    if (
        not isinstance(source, dict)
        or source.get("sha256") != EXPECTED_BENCHMARK_INPUT_MANIFEST_SHA256
    ):
        raise ValueError(f"{label}: frozen benchmark-input manifest hash mismatch")


def _validate_runtime(summary: dict[str, Any], *, label: str) -> None:
    runtime = summary.get("runtime")
    if not isinstance(runtime, dict):
        raise ValueError(f"{label}: runtime must be an object")
    partition = runtime.get("slurm_partition")
    expected_gpu_fragments = (
        GPU_NAME_FRAGMENTS_BY_PARTITION.get(partition) if isinstance(partition, str) else None
    )
    if expected_gpu_fragments is None:
        allowed = ", ".join(GPU_NAME_FRAGMENTS_BY_PARTITION)
        raise ValueError(f"{label}: runtime.slurm_partition must be one of {allowed}")
    gpu = runtime.get("gpu")
    if (
        runtime.get("device") != "cuda"
        or not isinstance(gpu, str)
        or not any(fragment in gpu for fragment in expected_gpu_fragments)
    ):
        allowed = ", ".join(expected_gpu_fragments)
        raise ValueError(f"{label}: sampling GPU on {partition} must match one of {allowed}")
    total_memory = runtime.get("gpu_total_memory_bytes")
    if (
        isinstance(total_memory, bool)
        or not isinstance(total_memory, int)
        or total_memory < MIN_GPU_TOTAL_MEMORY_BYTES
    ):
        raise ValueError(
            f"{label}: runtime.gpu_total_memory_bytes must be at least {MIN_GPU_TOTAL_MEMORY_BYTES}"
        )
    for key in ("torch", "cuda"):
        if not isinstance(runtime.get(key), str) or not runtime[key]:
            raise ValueError(f"{label}: runtime.{key} must be non-empty")
    for key in ("cuda_max_memory_allocated_bytes", "cuda_max_memory_reserved_bytes"):
        value = runtime.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 0 < value < GPU_MEMORY_LIMIT_BYTES
        ):
            raise ValueError(f"{label}: runtime.{key} must be positive and below 48 GiB")


def _validate_summary(
    summary: dict[str, Any],
    *,
    path: Path,
    dataset: str,
    eta: float,
    shard_index: int,
    smoke: bool,
    confidence: bool,
    expected_run_name_fn: Callable[[str, float], str] = expected_run_name,
    expected_guidance_parameter_sha256: str = EXPECTED_GUIDANCE_PARAMETER_SHA256,
    expected_receptor_policy_sha256: str = EXPECTED_RECEPTOR_POLICY_SHA256,
) -> None:
    label = str(path)
    required = set(_REQUIRED_SUMMARY_KEYS) | _EXTRA_REQUIRED_SUMMARY_KEYS
    missing = sorted(required - set(summary))
    if missing:
        raise ValueError(f"{label}: missing required summary fields {missing}")
    shards = 1 if smoke else 8
    exact = {
        "protocol_id": PROTOCOL_ID if confidence else PARENT_PROTOCOL_ID,
        "run_name": expected_run_name_fn(dataset, eta),
        "dataset": dataset,
        "num_samples": NUM_SAMPLES,
        "num_steps": NUM_STEPS,
        "model_pose_step_budget": NUM_SAMPLES * NUM_STEPS,
        "num_shards": shards,
        "shard_index": shard_index,
        "seed": 42,
        "unified_guidance_scale": eta,
        "unified_guidance_mode": EXPECTED_GUIDANCE_MODE,
        "prior_pool_size": 100,
        "prior_pool_hash_contract": "EFFDOCK_SHARED_PRIOR_V1",
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "checkpoint_step": 100_000,
        "config_sha256": EXPECTED_CONFIG_SHA256,
        "pocket_centers_sha256": EXPECTED_POCKET_CENTERS_SHA256[dataset],
        "num_discovered_total": EXPECTED_DATASET_COUNTS[dataset],
        "expected_discovered_count": EXPECTED_DATASET_COUNTS[dataset],
        "require_complete_success": True,
        "require_full_ligand_atom_mapping": True,
        "sigma": 0.5,
        "sigma_list": [],
        "sigma_counts": [],
        "pose_objective": "linear_fm",
        "score_rot_sigma_max": math.pi,
        "score_alpha_min": 0.0,
        "time_schedule": "late",
        "schedule_power": 3.0,
        "pocket_cutoff": 10.0,
        "center_jitter_sigma": 0.0,
        "vina_guidance_scale": 0.0,
        "vina_guidance_start_t": 0.5,
        "vina_guidance_ramp_power": 1.0,
        "vina_guidance_max_force": 10.0,
        "vina_guidance_max_velocity": 5.0,
        "vina_guidance_max_angular_velocity": 5.0,
        "vina_guidance_protein_shell": 18.0,
        "vina_guidance_w_strain": 1.0,
        "unified_guidance_start_t": 0.5,
        "unified_guidance_ramp_power": 1.0,
        "unified_guidance_max_force": 20.0,
        "unified_guidance_max_velocity": 5.0,
        "unified_guidance_max_angular_velocity": 5.0,
        "unified_guidance_max_atom_displacement": 0.25,
        "unified_guidance_max_backtracks": 8,
        "unified_guidance_protein_shell": 18.0,
        "unified_guidance_receptor_policy": RECEPTOR_POLICY,
        "refine": "none",
        "num_failed": 0,
    }
    for key, expected in exact.items():
        _require_setting(summary, key, expected, label=label)
    if summary.get("failures") != []:
        raise ValueError(f"{label}: complete replay rejects recorded failures")
    assigned = summary.get("num_assigned")
    success = summary.get("num_success")
    if isinstance(assigned, bool) or not isinstance(assigned, int) or assigned < 1:
        raise ValueError(f"{label}: num_assigned must be a positive integer")
    if success != assigned:
        raise ValueError(f"{label}: num_success must equal num_assigned")
    if smoke and assigned != 1:
        raise ValueError(f"{label}: smoke shard must contain exactly one success")
    _require_sha256(
        summary.get("eligibility_manifest_sha256"), label=f"{label}.eligibility_manifest_sha256"
    )
    _validate_benchmark_identity(summary, dataset, label=label)
    _validate_runtime(summary, label=label)

    if confidence:
        _require_setting(
            summary,
            "confidence_checkpoint_sha256",
            EXPECTED_CONFIDENCE_CHECKPOINT_SHA256,
            label=label,
        )
        _require_setting(summary, "confidence_step", EXPECTED_CONFIDENCE_STEP, label=label)
        if (
            not isinstance(summary.get("confidence_checkpoint"), str)
            or not summary["confidence_checkpoint"]
        ):
            raise ValueError(f"{label}: confidence checkpoint path must be non-empty")
        _require_setting(
            summary,
            "candidate_ensemble_hash_contract",
            CANDIDATE_ENSEMBLE_HASH_CONTRACT,
            label=label,
        )
        _require_setting(
            summary,
            "confidence_score_ledger_contract",
            CONFIDENCE_SCORE_LEDGER_CONTRACT,
            label=label,
        )
    else:
        for key in ("confidence_checkpoint", "confidence_step", "confidence_checkpoint_sha256"):
            _require_setting(summary, key, None, label=label)

    if eta > 0.0:
        parameter_set = summary.get("guidance_parameter_set")
        if (
            not isinstance(parameter_set, dict)
            or parameter_set.get("sha256") != expected_guidance_parameter_sha256
        ):
            raise ValueError(f"{label}: guidance parameter hash mismatch")
        identities = summary.get("guidance_receptor_policy_identities")
        if not isinstance(identities, dict) or set(identities) != {expected_receptor_policy_sha256}:
            raise ValueError(f"{label}: receptor-policy identity mismatch")
        _validate_direct_runtime(summary, expected_successes=assigned)
    else:
        if summary.get("guidance_runtime_stats") not in (None, {}):
            raise ValueError(f"{label}: eta=0 unexpectedly has guidance runtime telemetry")
        if summary.get("guidance_parameter_set") not in (None, {}):
            raise ValueError(f"{label}: eta=0 unexpectedly has guidance parameters")


def _normalized_implementation(value: Any, *, label: str) -> tuple[str, dict[str, Any]]:
    if not isinstance(value, dict):
        raise ValueError(f"{label}: guidance_implementation must be an object")
    digest = _require_sha256(value.get("sha256"), label=f"{label}.sha256")
    normalized = dict(value)
    normalized.pop("sha256")
    return digest, normalized


def _compare_summaries(
    parent: dict[str, Any],
    confidence: dict[str, Any],
    *,
    label: str,
    observations: dict[str, dict[str, Any]],
) -> tuple[str, str]:
    if set(parent) & _CONFIDENCE_ONLY_SUMMARY_KEYS:
        raise ValueError(f"{label}: parent unexpectedly has confidence-only summary fields")
    if set(confidence) != set(parent) | _CONFIDENCE_ONLY_SUMMARY_KEYS:
        raise ValueError(f"{label}: parent/confidence summary field sets differ")
    for key in sorted(set(parent) - _SUMMARY_DIFFERENCES):
        if parent[key] != confidence[key]:
            raise ValueError(f"{label}: summary field {key} differs across replay")
    for key in ("torch", "cuda", "device", "gpu"):
        if parent["runtime"].get(key) != confidence["runtime"].get(key):
            raise ValueError(f"{label}: runtime.{key} differs across replay")

    parent_impl_hash, parent_impl = _normalized_implementation(
        parent["guidance_implementation"], label=f"{label}.parent_implementation"
    )
    confidence_impl_hash, confidence_impl = _normalized_implementation(
        confidence["guidance_implementation"], label=f"{label}.confidence_implementation"
    )
    if parent_impl != confidence_impl:
        raise ValueError(f"{label}: implementation environment/file inventory differs")

    parent_stats = parent.get("stats")
    confidence_stats = confidence.get("stats")
    if not isinstance(parent_stats, dict) or set(parent_stats) != {
        "first",
        "vina",
        "oracle",
        "candidate_set",
    }:
        raise ValueError(f"{label}: parent selector statistics are incomplete")
    expected_confidence_stats = set(parent_stats) | set(CONFIDENCE_SELECTORS)
    if not isinstance(confidence_stats, dict) or set(confidence_stats) != expected_confidence_stats:
        raise ValueError(f"{label}: confidence selector statistics are incomplete")
    for key, value in parent_stats.items():
        _compare_numeric_tree(
            value,
            confidence_stats[key],
            label=f"{label}.stats.{key}",
            rel_tol=SCALAR_REL_TOL,
            abs_tol=SCALAR_ABS_TOL,
            bucket=observations["legacy_summary_stats"],
        )
    if "guidance_runtime_stats" in parent:
        _compare_numeric_tree(
            parent["guidance_runtime_stats"],
            confidence["guidance_runtime_stats"],
            label=f"{label}.guidance_runtime_stats",
            rel_tol=TELEMETRY_REL_TOL,
            abs_tol=TELEMETRY_ABS_TOL,
            bucket=observations["summary_guidance_telemetry"],
        )
    return parent_impl_hash, confidence_impl_hash


def _parse_pose_ledger(value: str, *, label: str, expected: set[str]) -> dict[str, str]:
    try:
        ledger = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label}: invalid saved-pose hash ledger") from exc
    if not isinstance(ledger, dict) or set(ledger) != expected:
        raise ValueError(f"{label}: saved-pose selectors must be exactly {sorted(expected)}")
    return {key: _require_sha256(raw, label=f"{label}.{key}") for key, raw in ledger.items()}


def _verify_pose_file(
    root: Path,
    *,
    run_name: str,
    dataset: str,
    complex_id: str,
    selector: str,
    expected_sha256: str,
) -> Path:
    if Path(complex_id).name != complex_id:
        raise ValueError(f"unsafe complex ID in pose path: {complex_id!r}")
    path = root / "poses" / run_name / dataset / selector / f"{complex_id}.sdf"
    if not path.is_file():
        raise FileNotFoundError(f"missing saved pose: {path}")
    if _file_sha256(path) != expected_sha256:
        raise ValueError(f"{path}: saved pose SHA-256 differs from CSV ledger")
    return path


def _read_single_sdf(path: Path) -> Chem.Mol:
    supplier = Chem.SDMolSupplier(
        str(path),
        sanitize=False,
        removeHs=False,
        strictParsing=True,
    )
    molecules = list(supplier)
    if len(molecules) != 1 or molecules[0] is None:
        raise ValueError(f"{path}: expected exactly one parseable SDF molecule")
    molecule = molecules[0]
    if molecule.GetNumConformers() != 1:
        raise ValueError(f"{path}: expected exactly one coordinate conformer")
    return molecule


def _atom_identity(molecule: Chem.Mol) -> list[tuple[Any, ...]]:
    return [
        (
            atom.GetAtomicNum(),
            atom.GetIsotope(),
            atom.GetFormalCharge(),
            str(atom.GetChiralTag()),
            atom.GetIsAromatic(),
            atom.GetNoImplicit(),
            atom.GetNumExplicitHs(),
            atom.GetAtomMapNum(),
        )
        for atom in molecule.GetAtoms()
    ]


def _bond_identity(molecule: Chem.Mol) -> list[tuple[Any, ...]]:
    return [
        (
            bond.GetBeginAtomIdx(),
            bond.GetEndAtomIdx(),
            str(bond.GetBondType()),
            str(bond.GetStereo()),
            tuple(bond.GetStereoAtoms()),
            bond.GetIsAromatic(),
            bond.GetIsConjugated(),
        )
        for bond in molecule.GetBonds()
    ]


def _compare_legacy_pose(
    parent_path: Path,
    replay_path: Path,
    *,
    parent_sha256: str,
    replay_sha256: str,
    label: str,
) -> dict[str, Any]:
    if parent_sha256 == replay_sha256:
        return {
            "parent_sha256": parent_sha256,
            "replay_sha256": replay_sha256,
            "hash_equal": True,
            "atom_count": None,
            "bond_count": None,
            "coordinate_rmsd_angstrom": 0.0,
            "max_atom_displacement_angstrom": 0.0,
        }

    parent_molecule = _read_single_sdf(parent_path)
    replay_molecule = _read_single_sdf(replay_path)
    parent_atoms = _atom_identity(parent_molecule)
    replay_atoms = _atom_identity(replay_molecule)
    if parent_atoms != replay_atoms:
        raise ValueError(f"{label}: legacy selected pose atom identity/order differs")
    parent_bonds = _bond_identity(parent_molecule)
    replay_bonds = _bond_identity(replay_molecule)
    if parent_bonds != replay_bonds:
        raise ValueError(f"{label}: legacy selected pose bond identity/order differs")
    if not parent_atoms:
        raise ValueError(f"{label}: legacy selected pose has no atoms")

    parent_conformer = parent_molecule.GetConformer()
    replay_conformer = replay_molecule.GetConformer()
    squared_displacements: list[float] = []
    max_displacement = 0.0
    for atom_index in range(len(parent_atoms)):
        parent_position = parent_conformer.GetAtomPosition(atom_index)
        replay_position = replay_conformer.GetAtomPosition(atom_index)
        displacement = math.sqrt(
            (float(parent_position.x) - float(replay_position.x)) ** 2
            + (float(parent_position.y) - float(replay_position.y)) ** 2
            + (float(parent_position.z) - float(replay_position.z)) ** 2
        )
        if not math.isfinite(displacement):
            raise ValueError(f"{label}: legacy selected pose has non-finite coordinates")
        squared_displacements.append(displacement**2)
        max_displacement = max(max_displacement, displacement)
    coordinate_rmsd = math.sqrt(math.fsum(squared_displacements) / len(parent_atoms))
    if coordinate_rmsd > POSE_COORDINATE_RMSD_TOL_ANGSTROM:
        raise ValueError(
            f"{label}: legacy selected pose coordinate RMSD {coordinate_rmsd:.8g} Å "
            f"exceeds {POSE_COORDINATE_RMSD_TOL_ANGSTROM:.8g} Å"
        )
    if max_displacement > POSE_MAX_ATOM_DISPLACEMENT_TOL_ANGSTROM:
        raise ValueError(
            f"{label}: legacy selected pose max atom displacement "
            f"{max_displacement:.8g} Å exceeds "
            f"{POSE_MAX_ATOM_DISPLACEMENT_TOL_ANGSTROM:.8g} Å"
        )
    return {
        "parent_sha256": parent_sha256,
        "replay_sha256": replay_sha256,
        "hash_equal": False,
        "atom_count": len(parent_atoms),
        "bond_count": len(parent_bonds),
        "coordinate_rmsd_angstrom": coordinate_rmsd,
        "max_atom_displacement_angstrom": max_displacement,
    }


def _numeric_score(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _validate_candidate_scores(row: dict[str, str], *, label: str) -> list[dict[str, float]]:
    try:
        raw = json.loads(row["confidence_candidate_scores_json"])
    except (KeyError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label}: invalid confidence_candidate_scores_json") from exc
    if not isinstance(raw, list) or len(raw) != NUM_SAMPLES:
        raise ValueError(f"{label}: confidence candidate score list must have length {NUM_SAMPLES}")
    scores: list[dict[str, float]] = []
    required = {*_RUNTIME_SCORE_FIELDS, "pl_clash_1p6"}
    for index, entry in enumerate(raw):
        if not isinstance(entry, dict) or not required <= set(entry):
            raise ValueError(f"{label}: candidate {index} lacks required confidence fields")
        converted = {
            key: _numeric_score(value, label=f"{label}.candidate[{index}].{key}")
            for key, value in entry.items()
        }
        for key in (
            "confidence_rmsd",
            "confidence_atom_rmsd",
            "confidence_atom_q90",
            "pl_clash_1p6",
        ):
            if converted[key] < 0.0:
                raise ValueError(f"{label}: candidate {index} {key} must be non-negative")
        for key in ("confidence_success", "confidence_atom_ok"):
            if not 0.0 <= converted[key] <= 1.0:
                raise ValueError(f"{label}: candidate {index} {key} must be in [0, 1]")
        scores.append(converted)
    return scores


def _parse_index(row: dict[str, str], key: str, *, label: str) -> int:
    try:
        value = int(row[key])
    except (KeyError, ValueError) as exc:
        raise ValueError(f"{label}: {key} must be an integer") from exc
    if not 0 <= value < NUM_SAMPLES:
        raise ValueError(f"{label}: {key} is outside the candidate pool")
    return value


def _validate_recomputed_selectors(row: dict[str, str], *, label: str) -> None:
    scores = _validate_candidate_scores(row, label=label)
    pred = torch.tensor([score["confidence_rmsd"] for score in scores], dtype=torch.float32)
    expected_primary = int(torch.argmin(pred))
    actual_primary = _parse_index(row, "confidence_index", label=label)
    if actual_primary != expected_primary:
        raise ValueError(f"{label}: confidence_index does not equal argmin confidence_rmsd")
    clash = torch.tensor([score["pl_clash_1p6"] for score in scores], dtype=torch.float32)
    expected_filter, _ = select_confidence_filter(scores, clash)
    actual_filter = _parse_index(row, "confidence_filter_index", label=label)
    if actual_filter != expected_filter:
        raise ValueError(f"{label}: confidence_filter_index does not match frozen filter")

    for selector in CONFIDENCE_SELECTORS:
        index = _parse_index(row, f"{selector}_index", label=label)
        for suffix, score_key in (
            ("pred_rmsd", "confidence_rmsd"),
            ("pred_success", "confidence_success"),
        ):
            try:
                recorded = float(row[f"{selector}_{suffix}"])
            except (KeyError, ValueError) as exc:
                raise ValueError(f"{label}: invalid {selector}_{suffix}") from exc
            if not math.isfinite(recorded) or recorded != scores[index][score_key]:
                raise ValueError(f"{label}: {selector}_{suffix} differs from candidate score")
        try:
            rmsd = float(row[f"{selector}_rmsd"])
        except (KeyError, ValueError) as exc:
            raise ValueError(f"{label}: invalid {selector}_rmsd") from exc
        if not math.isfinite(rmsd):
            raise ValueError(f"{label}: {selector}_rmsd must be finite")
        for term in _FAST_TERMS:
            if row.get(f"{selector}_fast_{term}") not in {"True", "False"}:
                raise ValueError(f"{label}: invalid {selector}_fast_{term}")


def _compare_rows(
    parent_fields: list[str],
    parent_rows: list[dict[str, str]],
    confidence_fields: list[str],
    confidence_rows: list[dict[str, str]],
    *,
    parent_root: Path,
    confidence_root: Path,
    run_name: str,
    dataset: str,
    eta: float,
    shard_index: int,
    observations: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    label = f"{run_name}/shard-{shard_index:03d}"
    if "saved_pose_sha256_json" not in parent_fields:
        raise ValueError(f"{label}: parent CSV lacks saved_pose_sha256_json")
    if set(parent_fields) & _CONFIDENCE_COLUMNS:
        raise ValueError(f"{label}: frozen parent unexpectedly contains confidence replay fields")
    if set(confidence_fields) != set(parent_fields) | _CONFIDENCE_COLUMNS:
        raise ValueError(f"{label}: confidence CSV field set differs from the frozen contract")
    parent_ids = [row["id"] for row in parent_rows]
    confidence_ids = [row["id"] for row in confidence_rows]
    if parent_ids != confidence_ids:
        raise ValueError(f"{label}: parent/confidence ID order differs")

    row_records: list[dict[str, Any]] = []
    sentinel_records: list[dict[str, Any]] = []
    confidence_pose_records: list[dict[str, Any]] = []
    for parent_row, confidence_row in zip(parent_rows, confidence_rows, strict=True):
        complex_id = parent_row["id"]
        row_label = f"{label}/{complex_id}"
        row_observations = _new_delta_observations()
        for field in parent_fields:
            if field == "saved_pose_sha256_json":
                continue
            _compare_common_csv_field(
                parent_row[field],
                confidence_row[field],
                field=field,
                label=f"{row_label}.{field}",
                observations=row_observations,
            )
        parent_ledger = _parse_pose_ledger(
            parent_row["saved_pose_sha256_json"],
            label=f"{row_label}.parent",
            expected=set(LEGACY_SELECTORS),
        )
        confidence_ledger = _parse_pose_ledger(
            confidence_row["saved_pose_sha256_json"],
            label=f"{row_label}.confidence",
            expected=set(LEGACY_SELECTORS) | set(CONFIDENCE_SELECTORS),
        )
        legacy_pose_replay: dict[str, dict[str, Any]] = {}
        for selector in LEGACY_SELECTORS:
            parent_pose_path = _verify_pose_file(
                parent_root,
                run_name=run_name,
                dataset=dataset,
                complex_id=complex_id,
                selector=selector,
                expected_sha256=parent_ledger[selector],
            )
            replay_pose_path = _verify_pose_file(
                confidence_root,
                run_name=run_name,
                dataset=dataset,
                complex_id=complex_id,
                selector=selector,
                expected_sha256=confidence_ledger[selector],
            )
            legacy_pose_replay[selector] = _compare_legacy_pose(
                parent_pose_path,
                replay_pose_path,
                parent_sha256=parent_ledger[selector],
                replay_sha256=confidence_ledger[selector],
                label=f"{row_label}.{selector}",
            )
        for selector in CONFIDENCE_SELECTORS:
            _verify_pose_file(
                confidence_root,
                run_name=run_name,
                dataset=dataset,
                complex_id=complex_id,
                selector=selector,
                expected_sha256=confidence_ledger[selector],
            )

        candidate_hash = _require_sha256(
            confidence_row["candidate_ensemble_sha256"],
            label=f"{row_label}.candidate_ensemble_sha256",
        )
        _validate_recomputed_selectors(confidence_row, label=row_label)
        trace = _validate_direct_step_trace(parent_row, eta=eta)
        if (eta > 0.0 and len(trace) != 8) or (eta == 0.0 and trace):
            raise ValueError(f"{row_label}: direct-step trace does not match eta")

        legacy_payload = {
            field: parent_row[field] for field in parent_fields if field != "saved_pose_sha256_json"
        }
        replay_legacy_payload = {
            field: confidence_row[field]
            for field in parent_fields
            if field != "saved_pose_sha256_json"
        }
        _merge_delta_observations(observations, row_observations)
        row_records.append(
            {
                "id": complex_id,
                "parent_legacy_row_sha256": hashlib.sha256(
                    _canonical_json(legacy_payload).encode()
                ).hexdigest(),
                "replay_legacy_row_sha256": hashlib.sha256(
                    _canonical_json(replay_legacy_payload).encode()
                ).hexdigest(),
                "numerical_deltas": {
                    name: row_observations[name]
                    for name in (
                        "legacy_csv_scalars",
                        "direct_step_telemetry",
                    )
                },
                "confidence_extension_sha256": hashlib.sha256(
                    _canonical_json(
                        {field: confidence_row[field] for field in sorted(_CONFIDENCE_COLUMNS)}
                    ).encode()
                ).hexdigest(),
                "candidate_ensemble_sha256": candidate_hash,
            }
        )
        sentinel_records.append(
            {
                "id": complex_id,
                "selectors": legacy_pose_replay,
            }
        )
        confidence_pose_records.append(
            {
                "id": complex_id,
                **{selector: confidence_ledger[selector] for selector in CONFIDENCE_SELECTORS},
            }
        )
    return row_records, sentinel_records, confidence_pose_records, parent_ids


def _interleave_shards(shards: list[list[str]]) -> list[str]:
    return [
        values[position]
        for position in range(max((len(values) for values in shards), default=0))
        for values in shards
        if position < len(values)
    ]


def _new_pose_observations() -> dict[str, Any]:
    return {
        "comparisons": 0,
        "hash_equal": 0,
        "hash_mismatch_but_numerically_equivalent": 0,
        "max_coordinate_rmsd_angstrom": 0.0,
        "max_coordinate_rmsd_location": None,
        "max_atom_displacement_angstrom": 0.0,
        "max_atom_displacement_location": None,
    }


def _accumulate_pose_observations(
    target: dict[str, Any],
    records: list[dict[str, Any]],
    *,
    run_name: str,
) -> None:
    for record in records:
        for selector, pose in record["selectors"].items():
            location = f"{run_name}/{record['id']}/{selector}"
            target["comparisons"] += 1
            if pose["hash_equal"]:
                target["hash_equal"] += 1
            else:
                target["hash_mismatch_but_numerically_equivalent"] += 1
            rmsd = float(pose["coordinate_rmsd_angstrom"])
            if rmsd > float(target["max_coordinate_rmsd_angstrom"]):
                target["max_coordinate_rmsd_angstrom"] = rmsd
                target["max_coordinate_rmsd_location"] = location
            displacement = float(pose["max_atom_displacement_angstrom"])
            if displacement > float(target["max_atom_displacement_angstrom"]):
                target["max_atom_displacement_angstrom"] = displacement
                target["max_atom_displacement_location"] = location


def build_identity_audit(
    parent_dir: Path,
    confidence_dir: Path,
    *,
    smoke: bool = False,
) -> dict[str, Any]:
    """Validate one complete smoke/full replay and return its immutable audit."""
    if asdict(ConfidenceFilterConfig()) != _EXPECTED_FILTER_CONFIG:
        raise RuntimeError("ConfidenceFilterConfig defaults differ from the frozen replay contract")
    parent_dir = parent_dir.resolve()
    confidence_dir = confidence_dir.resolve()
    parent_paths = _expected_summary_paths(parent_dir, smoke=smoke)
    confidence_paths = _expected_summary_paths(confidence_dir, smoke=smoke)
    shard_count = 1 if smoke else 8

    details: dict[str, Any] = {"datasets": {}}
    global_records: list[dict[str, Any]] = []
    coverage_per_dataset: dict[str, dict[str, int]] = {}
    total_rows = 0
    parent_impl_hashes: set[str] = set()
    confidence_impl_hashes: set[str] = set()
    confidence_cuda_runtimes: list[dict[str, Any]] = []
    delta_observations = _new_delta_observations()
    pose_observations = _new_pose_observations()
    for dataset in DATASETS:
        dataset_cells: dict[str, Any] = {}
        reference_ids: list[str] | None = None
        dataset_rows = 0
        for eta, tag in zip(ETA_VALUES, ETA_TAGS, strict=True):
            run_name = expected_run_name(dataset, eta)
            row_records: list[dict[str, Any]] = []
            sentinel_records: list[dict[str, Any]] = []
            confidence_pose_records: list[dict[str, Any]] = []
            shard_ids: list[list[str]] = []
            artifacts: list[dict[str, Any]] = []
            parent_traces: list[list[dict[str, Any]]] = []
            confidence_traces: list[list[dict[str, Any]]] = []
            for shard_index in range(shard_count):
                parent_path = parent_paths[(dataset, tag, shard_index)]
                confidence_path = confidence_paths[(dataset, tag, shard_index)]
                parent_summary = _load_json_object(parent_path)
                confidence_summary = _load_json_object(confidence_path)
                _validate_summary(
                    parent_summary,
                    path=parent_path,
                    dataset=dataset,
                    eta=eta,
                    shard_index=shard_index,
                    smoke=smoke,
                    confidence=False,
                )
                _validate_summary(
                    confidence_summary,
                    path=confidence_path,
                    dataset=dataset,
                    eta=eta,
                    shard_index=shard_index,
                    smoke=smoke,
                    confidence=True,
                )
                parent_impl, confidence_impl = _compare_summaries(
                    parent_summary,
                    confidence_summary,
                    label=f"{run_name}/shard-{shard_index:03d}",
                    observations=delta_observations,
                )
                parent_impl_hashes.add(parent_impl)
                confidence_impl_hashes.add(confidence_impl)
                confidence_cuda_runtimes.append(confidence_summary["runtime"])

                parent_csv = _companion_csv(parent_path, parent_summary, parent_dir)
                confidence_csv = _companion_csv(confidence_path, confidence_summary, confidence_dir)
                parent_fields, parent_rows = _read_raw_csv(parent_csv)
                confidence_fields, confidence_rows = _read_raw_csv(confidence_csv)
                if len(parent_rows) != parent_summary["num_success"]:
                    raise ValueError(f"{parent_csv}: row count differs from summary")
                if len(confidence_rows) != confidence_summary["num_success"]:
                    raise ValueError(f"{confidence_csv}: row count differs from summary")
                records, sentinels, confidence_poses, ids = _compare_rows(
                    parent_fields,
                    parent_rows,
                    confidence_fields,
                    confidence_rows,
                    parent_root=parent_dir,
                    confidence_root=confidence_dir,
                    run_name=run_name,
                    dataset=dataset,
                    eta=eta,
                    shard_index=shard_index,
                    observations=delta_observations,
                )
                row_records.extend(records)
                sentinel_records.extend(sentinels)
                confidence_pose_records.extend(confidence_poses)
                shard_ids.append(ids)
                parent_traces.extend(
                    [_validate_direct_step_trace(row, eta=eta) for row in parent_rows]
                )
                confidence_traces.extend(
                    [_validate_direct_step_trace(row, eta=eta) for row in confidence_rows]
                )
                if eta > 0.0:
                    _validate_trace_runtime_consistency(
                        parent_summary, parent_traces[-len(parent_rows) :]
                    )
                    _validate_trace_runtime_consistency(
                        confidence_summary, confidence_traces[-len(confidence_rows) :]
                    )
                artifacts.append(
                    {
                        "shard_index": shard_index,
                        "rows": len(ids),
                        "ordered_ids_sha256": _ordered_sha256(
                            ids, domain=b"EFFDOCK_CONFIDENCE_REPLAY_SHARD_IDS_V1\0"
                        ),
                        "parent": {
                            "summary": str(parent_path),
                            "summary_sha256": _file_sha256(parent_path),
                            "csv": str(parent_csv),
                            "csv_sha256": _file_sha256(parent_csv),
                        },
                        "confidence": {
                            "summary": str(confidence_path),
                            "summary_sha256": _file_sha256(confidence_path),
                            "csv": str(confidence_csv),
                            "csv_sha256": _file_sha256(confidence_csv),
                        },
                    }
                )

            ordered_ids = _interleave_shards(shard_ids)
            expected_count = 1 if smoke else EXPECTED_DATASET_COUNTS[dataset]
            if len(ordered_ids) != expected_count or len(set(ordered_ids)) != expected_count:
                raise ValueError(f"{run_name}: exact cohort coverage mismatch")
            if smoke and ordered_ids != [SMOKE_IDS[dataset]]:
                raise ValueError(f"{run_name}: smoke ID must be {SMOKE_IDS[dataset]}")
            if (
                not smoke
                and sorted_id_sha256(ordered_ids)
                != EXPECTED_BENCHMARK_IDENTITIES[dataset]["ids_sha256"]
            ):
                raise ValueError(f"{run_name}: IDs differ from the frozen benchmark manifest")
            if reference_ids is None:
                reference_ids = ordered_ids
            elif ordered_ids != reference_ids:
                raise ValueError(f"{dataset}: eta cells do not have identical ID order")
            dataset_rows += len(ordered_ids)
            total_rows += len(ordered_ids)
            cell_pose_observations = _new_pose_observations()
            _accumulate_pose_observations(
                cell_pose_observations,
                sentinel_records,
                run_name=run_name,
            )
            _accumulate_pose_observations(
                pose_observations,
                sentinel_records,
                run_name=run_name,
            )
            cell = {
                "run_name": run_name,
                "eta": eta,
                "eta_tag": eta_tag(eta),
                "shards": shard_count,
                "rows": len(ordered_ids),
                "ordered_ids_sha256": _ordered_sha256(
                    ordered_ids, domain=b"EFFDOCK_CONFIDENCE_REPLAY_CELL_IDS_V1\0"
                ),
                "row_equivalence_ledger_sha256": _records_sha256(
                    row_records, domain=b"EFFDOCK_CONFIDENCE_REPLAY_ROWS_V2\0"
                ),
                "parent_selector_pose_ledger_sha256": _records_sha256(
                    sentinel_records, domain=b"EFFDOCK_CONFIDENCE_REPLAY_SENTINELS_V2\0"
                ),
                "confidence_selector_pose_ledger_sha256": _records_sha256(
                    confidence_pose_records,
                    domain=b"EFFDOCK_CONFIDENCE_REPLAY_NEW_SELECTOR_POSES_V1\0",
                ),
                "artifact_ledger_sha256": _records_sha256(
                    artifacts, domain=b"EFFDOCK_CONFIDENCE_REPLAY_ARTIFACTS_V1\0"
                ),
                "row_equivalence": {
                    "mode": "numerical_replay_equivalence",
                    "records": row_records,
                },
                "legacy_pose_equivalence": {
                    **cell_pose_observations,
                    "records": sentinel_records,
                },
                "artifacts": artifacts,
            }
            dataset_cells[tag] = cell
            global_records.append(
                {
                    key: cell[key]
                    for key in (
                        "run_name",
                        "eta",
                        "rows",
                        "ordered_ids_sha256",
                        "row_equivalence_ledger_sha256",
                        "parent_selector_pose_ledger_sha256",
                        "confidence_selector_pose_ledger_sha256",
                        "artifact_ledger_sha256",
                    )
                }
            )
        ids_per_cell = 1 if smoke else EXPECTED_DATASET_COUNTS[dataset]
        coverage_per_dataset[dataset] = {
            "cells": len(ETA_VALUES),
            "shards": len(ETA_VALUES) * shard_count,
            "rows": dataset_rows,
            "ids_per_cell": ids_per_cell,
        }
        details["datasets"][dataset] = {
            "ids_per_cell": ids_per_cell,
            "ordered_ids_sha256": _ordered_sha256(
                reference_ids or [], domain=b"EFFDOCK_CONFIDENCE_REPLAY_DATASET_IDS_V1\0"
            ),
            "cells": dataset_cells,
        }

    expected_rows = (
        len(DATASETS) * len(ETA_VALUES)
        if smoke
        else sum(EXPECTED_DATASET_COUNTS.values()) * len(ETA_VALUES)
    )
    if total_rows != expected_rows:
        raise ValueError(f"replay row total must be {expected_rows}, got {total_rows}")
    if len(parent_impl_hashes) != 1 or len(confidence_impl_hashes) != 1:
        raise ValueError("implementation identity is inconsistent across replay shards")

    frozen_hashes = {
        "docking_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "config_sha256": EXPECTED_CONFIG_SHA256,
        "confidence_checkpoint_sha256": EXPECTED_CONFIDENCE_CHECKPOINT_SHA256,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "parent_sampling_protocol_id": PARENT_PROTOCOL_ID,
        "audit_contract": AUDIT_CONTRACT,
        "mode": "smoke" if smoke else "full",
        "status": "passed",
        "parent_sentinels_verified": True,
        "summary_contracts_verified": True,
        "candidate_ensemble_hashes_present": True,
        "selector_recomputed": True,
        "frozen_hashes": frozen_hashes,
        "numerical_replay_contract": {
            "contract_id": AUDIT_CONTRACT,
            "categorical_discrete_and_legacy_selector_indices": "exact",
            "legacy_scalar_and_summary_stats": {
                "rel_tol": SCALAR_REL_TOL,
                "abs_tol": SCALAR_ABS_TOL,
                "finite_required": True,
            },
            "direct_step_and_summary_guidance_telemetry": {
                "rel_tol": TELEMETRY_REL_TOL,
                "abs_tol": TELEMETRY_ABS_TOL,
                "finite_required": True,
                "counters": "exact",
            },
            "legacy_selected_pose": {
                "atom_identity_and_order": "exact",
                "bond_identity_and_order": "exact",
                "coordinate_rmsd_max_angstrom": POSE_COORDINATE_RMSD_TOL_ANGSTROM,
                "max_atom_displacement_angstrom": (POSE_MAX_ATOM_DISPLACEMENT_TOL_ANGSTROM),
                "file_sha256": "exact fast path; otherwise numerical pose gate",
                "current_file_hash_binding": "exact",
            },
        },
        "maximum_observed_deltas": {
            **delta_observations,
            "legacy_selected_poses": pose_observations,
        },
        "parent_dir": str(parent_dir),
        "confidence_dir": str(confidence_dir),
        "coverage": {
            "datasets": len(DATASETS),
            "cells": len(DATASETS) * len(ETA_VALUES),
            "shards": len(DATASETS) * len(ETA_VALUES) * shard_count,
            "rows": total_rows,
            "per_dataset": coverage_per_dataset,
        },
        "cuda_runtime": {
            "slurm_partitions": sorted(
                {str(runtime["slurm_partition"]) for runtime in confidence_cuda_runtimes}
            ),
            "gpu_names": sorted({str(runtime["gpu"]) for runtime in confidence_cuda_runtimes}),
            "gpu_total_memory_bytes": sorted(
                {int(runtime["gpu_total_memory_bytes"]) for runtime in confidence_cuda_runtimes}
            ),
            "minimum_required_total_memory_bytes": MIN_GPU_TOTAL_MEMORY_BYTES,
            "max_memory_allocated_bytes": max(
                int(runtime["cuda_max_memory_allocated_bytes"])
                for runtime in confidence_cuda_runtimes
            ),
            "max_memory_reserved_bytes": max(
                int(runtime["cuda_max_memory_reserved_bytes"])
                for runtime in confidence_cuda_runtimes
            ),
            "limit_bytes": GPU_MEMORY_LIMIT_BYTES,
        },
        "checks": {
            "summary_inventory_complete": True,
            "protocols_and_hashes_frozen": True,
            "scientific_config_exact": True,
            "complete_success": True,
            "id_order_and_coverage_exact": True,
            "all_parent_csv_fields_numerically_equivalent": True,
            "parent_selector_poses_numerically_equivalent": True,
            "confidence_selector_fields_complete": True,
            "confidence_selector_pose_hashes_complete": True,
            "candidate_ensemble_hashes_present": True,
            "selector_recomputed": True,
            "summary_contracts_verified": True,
        },
        "details": {
            "frozen_hashes": frozen_hashes,
            "confidence_filter_config": _EXPECTED_FILTER_CONFIG,
            "parent_implementation_sha256": next(iter(parent_impl_hashes)),
            "confidence_implementation_sha256": next(iter(confidence_impl_hashes)),
            **details,
        },
        "global_equivalence_ledger_sha256": _records_sha256(
            global_records, domain=b"EFFDOCK_CONFIDENCE_REPLAY_GLOBAL_V2\0"
        ),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-dir", type=Path, required=True)
    parser.add_argument("--confidence-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    audit = build_identity_audit(
        args.parent_dir,
        args.confidence_dir,
        smoke=args.smoke,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(audit, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
