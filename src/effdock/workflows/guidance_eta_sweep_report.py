#!/usr/bin/env python3
"""Strict full-cohort report for the descriptive direct-guidance eta sweep."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from effdock.inference.sampler import build_time_grid
from effdock.workflows.evaluate import sorted_id_sha256, summarize_rows
from effdock.workflows.guidance_budget_full_report import (
    EXPECTED_BENCHMARK_INPUT_MANIFEST_SHA256,
    EXPECTED_DATASET_COUNTS,
    RECEPTOR_POLICY,
    _paired_comparison,
    _validate_sampling_input_rows,
    load_full_cohort_audits,
)
from effdock.workflows.guidance_budget_report import (
    DATASETS,
    DEFAULT_BOOTSTRAP_RESAMPLES,
    DEFAULT_BOOTSTRAP_SEED,
    DEFAULT_EXPECTED_SHARDS,
    EXPECTED_CHECKPOINT_SHA256,
    EXPECTED_CONFIG_SHA256,
    EXPECTED_POCKET_CENTERS_SHA256,
    _read_shard_rows,
    _stable_shard_metadata,
    _target_metrics,
    _validate_failure_records,
)
from effdock.workflows.guidance_coverage_audit import ID_HASH_CONTRACT

PROTOCOL_ID = "EFFDOCK-UNIFIED-GUIDANCE-ETA-SWEEP-V2"
ETA_VALUES = (0.0, 0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5)
ETA_TAGS = (
    "eta0000",
    "eta0025",
    "eta0050",
    "eta0100",
    "eta0200",
    "eta0300",
    "eta0400",
    "eta0500",
)
NUM_SAMPLES = 100
NUM_STEPS = 10
CONDITION = "n100_s10"
EXPECTED_GUIDANCE_MODE = "normalized_drift"
EXPECTED_SHARD_TASKS = len(DATASETS) * len(ETA_VALUES) * DEFAULT_EXPECTED_SHARDS

_REQUIRED_SUMMARY_KEYS = (
    "run_name",
    "protocol_id",
    "dataset",
    "num_samples",
    "num_steps",
    "model_pose_step_budget",
    "num_discovered_total",
    "num_shards",
    "shard_index",
    "seed",
    "unified_guidance_scale",
    "unified_guidance_mode",
    "prior_pool_size",
    "prior_pool_hash_contract",
    "checkpoint_sha256",
    "confidence_checkpoint_sha256",
    "config_sha256",
    "pocket_centers_sha256",
    "eligibility_manifest_sha256",
    "benchmark_input_identity",
    "guidance_implementation",
    "require_full_ligand_atom_mapping",
    "sigma",
    "time_schedule",
    "schedule_power",
    "pocket_cutoff",
    "center_jitter_sigma",
    "vina_guidance_scale",
    "unified_guidance_start_t",
    "unified_guidance_ramp_power",
    "unified_guidance_max_force",
    "unified_guidance_max_velocity",
    "unified_guidance_max_angular_velocity",
    "unified_guidance_max_atom_displacement",
    "unified_guidance_max_backtracks",
    "unified_guidance_protein_shell",
    "unified_guidance_receptor_policy",
    "refine",
    "runtime",
    "csv",
    "failures",
)


def eta_tag(eta: float) -> str:
    """Return the one filesystem-safe tag for a frozen eta value."""
    matches = [
        tag
        for value, tag in zip(ETA_VALUES, ETA_TAGS, strict=True)
        if math.isclose(float(eta), value, rel_tol=0.0, abs_tol=1e-12)
    ]
    if len(matches) != 1:
        raise ValueError(f"eta must be one of {ETA_VALUES}, got {eta!r}")
    return matches[0]


def eta_value(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid unified_guidance_scale: {value!r}") from exc
    eta_tag(numeric)
    return next(
        expected
        for expected in ETA_VALUES
        if math.isclose(numeric, expected, rel_tol=0.0, abs_tol=1e-12)
    )


def expected_run_name(dataset: str, eta: float) -> str:
    if dataset not in DATASETS:
        raise ValueError(f"unexpected dataset: {dataset!r}")
    return (
        "effdock-guidance-direct-drift-eta-sweep-v2-"
        f"{dataset}-n{NUM_SAMPLES}-s{NUM_STEPS}-{eta_tag(eta)}"
    )


def _active_intervals() -> int:
    grid = build_time_grid(NUM_STEPS, schedule="late", power=3.0)
    return sum(float(right) > 0.5 for right in grid[1:])


def _interval_ramp(left: float, right: float) -> float:
    active_left = max(left, 0.5)
    if right <= 0.5:
        return 0.0
    numerator = (right - 0.5) ** 2 - (active_left - 0.5) ** 2
    return max(0.0, min(1.0, numerator / (right - left)))


def _expected_trace_intervals() -> tuple[dict[str, float], ...]:
    grid = build_time_grid(NUM_STEPS, schedule="late", power=3.0)
    result: list[dict[str, float]] = []
    for left_tensor, right_tensor in zip(grid[:-1], grid[1:], strict=True):
        left, right = float(left_tensor), float(right_tensor)
        if right > 0.5:
            result.append(
                {
                    "t": left,
                    "t_end": right,
                    "dt": right - left,
                    "ramp": _interval_ramp(left, right),
                }
            )
    return tuple(result)


def _require_exact_setting(summary: dict[str, Any], key: str, expected: Any) -> None:
    actual = summary[key]
    matches = (
        math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-12)
        if isinstance(expected, float)
        else actual == expected
    )
    if not matches:
        raise ValueError(f"{summary['run_name']}: {key} must be {expected!r}, got {actual!r}")


def _numeric_scalar(value: Any, *, label: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a numeric scalar or null")
    if not math.isfinite(float(value)):
        raise ValueError(f"{label} must be finite")
    return value


def _combine_runtime_stats(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate every scalar telemetry field without a name allow-list.

    ``direct_*_sum`` fields are pose-level sums by contract and therefore also
    receive a finite-pose mean. Counter/sum fields are added, maxima/minima are
    reduced accordingly, and unknown non-scalar telemetry fails closed.
    """
    mappings = [summary.get("guidance_runtime_stats") for summary in summaries]
    if any(not isinstance(mapping, dict) for mapping in mappings):
        raise ValueError("every nonzero-eta shard requires guidance_runtime_stats")
    typed = [mapping for mapping in mappings if isinstance(mapping, dict)]
    key_sets = [set(mapping) for mapping in typed]
    if not key_sets or any(keys != key_sets[0] for keys in key_sets[1:]):
        raise ValueError("guidance telemetry fields differ across shards")

    combined: dict[str, Any] = {}
    for name in sorted(key_sets[0]):
        raw_values = [mapping[name] for mapping in typed]
        present = [
            _numeric_scalar(value, label=f"guidance_runtime_stats.{name}")
            for value in raw_values
            if value is not None
        ]
        if not present:
            combined[name] = None
        elif any(value is None for value in raw_values):
            raise ValueError(f"guidance telemetry {name} is null in only some shards")
        elif name.startswith(("max_", "direct_max_")):
            combined[name] = max(float(value) for value in present)
        elif name.startswith(("min_", "direct_min_")):
            combined[name] = min(float(value) for value in present)
        elif all(isinstance(value, int) for value in present):
            combined[name] = sum(int(value) for value in present)
        else:
            combined[name] = sum(float(value) for value in present)

    pose_count = int(combined.get("direct_pose_evaluations", 0) or 0)
    nonfinite = int(combined.get("direct_nonfinite_poses", 0) or 0)
    finite_pose_count = pose_count - nonfinite
    mean_contract = {
        "direct_reference_atom_speed_rms_sum": "direct_atom_speed_rms_valid_count",
        "direct_model_atom_speed_rms_sum": "direct_atom_speed_rms_valid_count",
        "direct_raw_atom_speed_rms_sum": "direct_atom_speed_rms_valid_count",
        "direct_applied_atom_speed_rms_sum": "direct_atom_speed_rms_valid_count",
        "direct_total_atom_speed_rms_sum": "direct_atom_speed_rms_valid_count",
        "direct_applied_to_model_rms_ratio_sum": ("direct_applied_to_model_rms_ratio_valid_count"),
        "direct_model_guide_cosine_sum": "direct_model_guide_cosine_valid_count",
        "direct_guide_parallel_to_model_ratio_sum": (
            "direct_guide_parallel_to_model_ratio_valid_count"
        ),
        "direct_cap_scale_sum": "direct_cap_scale_valid_count",
    }
    means: dict[str, float | None] = {}
    for sum_name, count_name in mean_contract.items():
        if sum_name not in combined or count_name not in combined:
            continue
        count = int(combined[count_name] or 0)
        means[f"{sum_name.removesuffix('_sum')}_mean"] = (
            float(combined[sum_name]) / count if count > 0 else None
        )
    active_intervals = _active_intervals()
    trajectory_count = pose_count // active_intervals if active_intervals else 0
    if trajectory_count * active_intervals != pose_count:
        raise ValueError("direct pose evaluations are not whole active trajectories")
    for sum_name in (
        "direct_model_rms_path_proxy_sum",
        "direct_applied_rms_path_proxy_sum",
        "direct_total_rms_path_proxy_sum",
    ):
        if sum_name in combined:
            means[f"{sum_name.removesuffix('_sum')}_mean_per_trajectory"] = (
                float(combined[sum_name]) / trajectory_count if trajectory_count > 0 else None
            )
    return {
        "scalars": combined,
        "finite_pose_count": finite_pose_count,
        "trajectory_count": trajectory_count,
        "means": means,
    }


_TRACE_DISTRIBUTIONS = (
    "applied_to_model_rms_ratio",
    "model_guide_cosine",
    "guide_parallel_to_model_ratio",
    "cap_scale",
)
_TRACE_SUM_FIELDS = (
    "model_atom_speed_rms_sum",
    "applied_atom_speed_rms_sum",
    "total_atom_speed_rms_sum",
    "model_rms_path_proxy_sum",
    "applied_rms_path_proxy_sum",
    "total_rms_path_proxy_sum",
)
_TRACE_COUNT_FIELDS = (
    "pose_count",
    "finite_count",
    "applied_count",
    "atom_speed_rms_valid_count",
    "translation_cap_trigger_count",
    "angular_cap_trigger_count",
    "displacement_cap_trigger_count",
    "any_cap_trigger_count",
    "multiple_cap_trigger_count",
)


def _validate_direct_step_trace(row: dict[str, Any], *, eta: float) -> list[dict[str, Any]]:
    label = f"{row['id']}.guidance_direct_step_trace_json"
    raw = row.get("guidance_direct_step_trace_json")
    if not isinstance(raw, str):
        raise ValueError(f"{label} must be a JSON string")
    try:
        trace = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is invalid JSON") from exc
    if not isinstance(trace, list):
        raise ValueError(f"{label} must decode to a list")
    if eta == 0.0:
        if trace:
            raise ValueError(f"{label} must be [] when eta=0")
        return []

    expected = _expected_trace_intervals()
    if len(trace) != len(expected):
        raise ValueError(f"{label} must contain exactly {len(expected)} active steps")
    required = {
        "t",
        "t_end",
        "dt",
        "ramp",
        "eta",
        *_TRACE_SUM_FIELDS,
        *_TRACE_COUNT_FIELDS,
    }
    for distribution in _TRACE_DISTRIBUTIONS:
        required.update(
            {
                f"{distribution}_sum",
                f"{distribution}_valid_count",
                f"{distribution}_p05",
                f"{distribution}_p50",
                f"{distribution}_p95",
                f"{distribution}_p99",
            }
        )

    validated: list[dict[str, Any]] = []
    for index, (step, expected_step) in enumerate(zip(trace, expected, strict=True)):
        step_label = f"{label}[{index}]"
        if not isinstance(step, dict):
            raise ValueError(f"{step_label} must be an object")
        missing = sorted(required - set(step))
        if missing:
            raise ValueError(f"{step_label} is missing fields {missing}")
        for key, expected_value in {**expected_step, "eta": eta}.items():
            value = _numeric_scalar(step[key], label=f"{step_label}.{key}")
            if not math.isclose(float(value), expected_value, rel_tol=0.0, abs_tol=2e-7):
                raise ValueError(f"{step_label}.{key} must be {expected_value}, got {value}")
        for key in _TRACE_SUM_FIELDS:
            value = _numeric_scalar(step[key], label=f"{step_label}.{key}")
            if float(value) < -1e-8:
                raise ValueError(f"{step_label}.{key} must be non-negative")
        for key in _TRACE_COUNT_FIELDS:
            value = _numeric_scalar(step[key], label=f"{step_label}.{key}")
            if not isinstance(value, int) or value < 0:
                raise ValueError(f"{step_label}.{key} must be a non-negative integer")
        if int(step["pose_count"]) != NUM_SAMPLES:
            raise ValueError(f"{step_label}.pose_count must equal {NUM_SAMPLES}")
        if int(step["finite_count"]) != NUM_SAMPLES:
            raise ValueError(f"{step_label}.finite_count must equal {NUM_SAMPLES}")
        if int(step["atom_speed_rms_valid_count"]) != int(step["finite_count"]):
            raise ValueError(f"{step_label}: atom-speed valid count mismatch")
        if not 0 <= int(step["applied_count"]) <= NUM_SAMPLES:
            raise ValueError(f"{step_label}: invalid applied_count")
        for key in _TRACE_COUNT_FIELDS[4:]:
            if int(step[key]) > NUM_SAMPLES:
                raise ValueError(f"{step_label}.{key} exceeds pose_count")

        individual_cap_count = sum(
            int(step[key])
            for key in (
                "translation_cap_trigger_count",
                "angular_cap_trigger_count",
                "displacement_cap_trigger_count",
            )
        )
        any_cap_count = int(step["any_cap_trigger_count"])
        multiple_cap_count = int(step["multiple_cap_trigger_count"])
        if multiple_cap_count > any_cap_count:
            raise ValueError(f"{step_label}: multiple-cap count exceeds any-cap count")
        if not (
            any_cap_count + multiple_cap_count
            <= individual_cap_count
            <= any_cap_count + 2 * multiple_cap_count
        ):
            raise ValueError(f"{step_label}: cap trigger counters are inconsistent")

        for distribution in _TRACE_DISTRIBUTIONS:
            sum_key = f"{distribution}_sum"
            count_key = f"{distribution}_valid_count"
            _numeric_scalar(step[sum_key], label=f"{step_label}.{sum_key}")
            count = _numeric_scalar(step[count_key], label=f"{step_label}.{count_key}")
            if not isinstance(count, int) or not 0 <= count <= NUM_SAMPLES:
                raise ValueError(f"{step_label}.{count_key} is invalid")
            distribution_sum = float(step[sum_key])
            if distribution in {"applied_to_model_rms_ratio", "cap_scale"} and (
                distribution_sum < -1e-8
            ):
                raise ValueError(f"{step_label}.{sum_key} must be non-negative")
            if distribution == "cap_scale" and distribution_sum > count + 1e-6:
                raise ValueError(f"{step_label}.{sum_key} exceeds its valid count")
            if distribution == "model_guide_cosine" and abs(distribution_sum) > count + 1e-6:
                raise ValueError(f"{step_label}.{sum_key} is outside cosine bounds")
            quantiles = [
                step[f"{distribution}_{suffix}"] for suffix in ("p05", "p50", "p95", "p99")
            ]
            if count == 0:
                if any(value is not None for value in quantiles):
                    raise ValueError(f"{step_label}: empty distribution has quantiles")
            else:
                values = [
                    float(_numeric_scalar(value, label=f"{step_label}.{distribution}"))
                    for value in quantiles
                ]
                if values != sorted(values):
                    raise ValueError(f"{step_label}: distribution quantiles are unordered")
                if distribution == "model_guide_cosine" and (
                    values[0] < -1.000001 or values[-1] > 1.000001
                ):
                    raise ValueError(f"{step_label}: cosine quantile is outside [-1, 1]")
                if distribution == "applied_to_model_rms_ratio" and values[0] < -1e-8:
                    raise ValueError(f"{step_label}: RMS-ratio quantile is negative")
                if distribution == "cap_scale" and (values[0] < 0.0 or values[-1] > 1.000001):
                    raise ValueError(f"{step_label}: cap quantile is outside [0, 1]")

        dt = float(step["dt"])
        for base in ("model", "applied", "total"):
            if not math.isclose(
                float(step[f"{base}_rms_path_proxy_sum"]),
                dt * float(step[f"{base}_atom_speed_rms_sum"]),
                rel_tol=1e-6,
                abs_tol=1e-6,
            ):
                raise ValueError(f"{step_label}: {base} path proxy is inconsistent")
        validated.append(step)
    return validated


def _cohort_scalar_summary(values: list[float]) -> dict[str, float | int | None]:
    """Summarize one value per complex without implying pooled-pose quantiles."""
    if not values:
        return {
            "complex_count": 0,
            "mean": None,
            "min": None,
            "p05": None,
            "p50": None,
            "p95": None,
            "max": None,
        }
    ordered = sorted(values)

    def percentile(fraction: float) -> float:
        position = fraction * (len(ordered) - 1)
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return float(ordered[lower])
        weight = position - lower
        return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)

    return {
        "complex_count": len(ordered),
        "mean": float(sum(ordered) / len(ordered)),
        "min": float(ordered[0]),
        "p05": percentile(0.05),
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "max": float(ordered[-1]),
    }


def _aggregate_interval_telemetry(
    traces: list[list[dict[str, Any]]],
    *,
    eta: float,
) -> list[dict[str, Any]]:
    if eta == 0.0:
        return []
    expected = _expected_trace_intervals()
    result: list[dict[str, Any]] = []
    additive_fields = {
        *_TRACE_SUM_FIELDS,
        *_TRACE_COUNT_FIELDS,
        *(f"{name}_sum" for name in _TRACE_DISTRIBUTIONS),
        *(f"{name}_valid_count" for name in _TRACE_DISTRIBUTIONS),
    }
    for index, interval in enumerate(expected):
        scalars = {
            name: sum(float(trace[index][name]) for trace in traces)
            for name in sorted(additive_fields)
        }
        for name in additive_fields:
            if name.endswith("_count") or name in _TRACE_COUNT_FIELDS:
                scalars[name] = int(scalars[name])
        means: dict[str, float | None] = {}
        for name in ("model", "applied", "total"):
            count = int(scalars["atom_speed_rms_valid_count"])
            means[f"{name}_atom_speed_rms_mean"] = (
                float(scalars[f"{name}_atom_speed_rms_sum"]) / count if count else None
            )
            means[f"{name}_rms_path_proxy_mean_per_pose"] = (
                float(scalars[f"{name}_rms_path_proxy_sum"]) / count if count else None
            )
        for name in _TRACE_DISTRIBUTIONS:
            count = int(scalars[f"{name}_valid_count"])
            means[f"{name}_mean"] = float(scalars[f"{name}_sum"]) / count if count else None
        valid_count = int(scalars["finite_count"])
        cap_rates = {
            name.removesuffix("_count") + "_pct": (
                float(scalars[name]) / valid_count * 100.0 if valid_count else None
            )
            for name in _TRACE_COUNT_FIELDS[4:]
        }
        per_complex_pose_quantiles = {
            distribution: {
                suffix: _cohort_scalar_summary(
                    [
                        float(trace[index][f"{distribution}_{suffix}"])
                        for trace in traces
                        if trace[index][f"{distribution}_{suffix}"] is not None
                    ]
                )
                for suffix in ("p05", "p50", "p95", "p99")
            }
            for distribution in _TRACE_DISTRIBUTIONS
        }
        result.append(
            {
                "active_step_index": index,
                **interval,
                "eta": eta,
                "complexes": len(traces),
                "pose_trajectories": len(traces) * NUM_SAMPLES,
                "scalars": scalars,
                "means": means,
                "cap_trigger_pct": cap_rates,
                "per_complex_pose_quantiles": {
                    "semantics": (
                        "cohort summaries of each complex's within-complex 100-pose "
                        "quantile; these are not pooled-pose quantiles"
                    ),
                    "metrics": per_complex_pose_quantiles,
                },
            }
        )
    return result


def _validate_trace_runtime_consistency(
    summary: dict[str, Any],
    traces: list[list[dict[str, Any]]],
) -> None:
    stats = summary["guidance_runtime_stats"]
    mapping = {
        "direct_pose_evaluations": ("pose_count",),
        "direct_pose_applied": ("applied_count",),
        "direct_model_atom_speed_rms_sum": ("model_atom_speed_rms_sum",),
        "direct_applied_atom_speed_rms_sum": ("applied_atom_speed_rms_sum",),
        "direct_total_atom_speed_rms_sum": ("total_atom_speed_rms_sum",),
        "direct_atom_speed_rms_valid_count": ("atom_speed_rms_valid_count",),
        "direct_applied_to_model_rms_ratio_sum": ("applied_to_model_rms_ratio_sum",),
        "direct_applied_to_model_rms_ratio_valid_count": (
            "applied_to_model_rms_ratio_valid_count",
        ),
        "direct_model_guide_cosine_sum": ("model_guide_cosine_sum",),
        "direct_model_guide_cosine_valid_count": ("model_guide_cosine_valid_count",),
        "direct_guide_parallel_to_model_ratio_sum": ("guide_parallel_to_model_ratio_sum",),
        "direct_guide_parallel_to_model_ratio_valid_count": (
            "guide_parallel_to_model_ratio_valid_count",
        ),
        "direct_model_rms_path_proxy_sum": ("model_rms_path_proxy_sum",),
        "direct_applied_rms_path_proxy_sum": ("applied_rms_path_proxy_sum",),
        "direct_total_rms_path_proxy_sum": ("total_rms_path_proxy_sum",),
        "direct_cap_scale_sum": ("cap_scale_sum",),
        "direct_cap_scale_valid_count": ("cap_scale_valid_count",),
        "direct_translation_cap_trigger_count": ("translation_cap_trigger_count",),
        "direct_angular_cap_trigger_count": ("angular_cap_trigger_count",),
        "direct_displacement_cap_trigger_count": ("displacement_cap_trigger_count",),
        "direct_any_cap_trigger_count": ("any_cap_trigger_count",),
        "direct_multiple_cap_trigger_count": ("multiple_cap_trigger_count",),
    }
    for runtime_name, (trace_name,) in mapping.items():
        if runtime_name not in stats:
            raise ValueError(f"{summary['run_name']}: runtime lacks {runtime_name}")
        traced = sum(float(step[trace_name]) for trace in traces for step in trace)
        runtime = float(stats[runtime_name])
        if not math.isclose(traced, runtime, rel_tol=1e-7, abs_tol=1e-5):
            raise ValueError(f"{summary['run_name']}: trace/runtime mismatch for {runtime_name}")
    finite = sum(int(step["finite_count"]) for trace in traces for step in trace)
    expected_finite = int(stats["direct_pose_evaluations"]) - int(stats["direct_nonfinite_poses"])
    if finite != expected_finite:
        raise ValueError(f"{summary['run_name']}: trace/runtime finite-count mismatch")


def _combine_cuda_runtime(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    runtimes = [summary.get("runtime") for summary in summaries]
    if any(not isinstance(runtime, dict) for runtime in runtimes):
        raise ValueError("every sampling shard requires a runtime object")
    typed = [runtime for runtime in runtimes if isinstance(runtime, dict)]
    required = ("cuda_max_memory_allocated_bytes", "cuda_max_memory_reserved_bytes")
    for runtime in typed:
        missing = [key for key in required if key not in runtime]
        if missing:
            raise ValueError(f"sampling runtime is missing CUDA fields {missing}")
        for key in required:
            _numeric_scalar(runtime[key], label=f"runtime.{key}")
    return {
        "device": sorted({str(runtime.get("device")) for runtime in typed}),
        "gpu": sorted({str(runtime.get("gpu")) for runtime in typed}),
        "torch": sorted({str(runtime.get("torch")) for runtime in typed}),
        "cuda": sorted({str(runtime.get("cuda")) for runtime in typed}),
        "max_memory_allocated_bytes": max(
            int(runtime["cuda_max_memory_allocated_bytes"]) for runtime in typed
        ),
        "max_memory_reserved_bytes": max(
            int(runtime["cuda_max_memory_reserved_bytes"]) for runtime in typed
        ),
    }


def _load_sampling_inventory(
    input_dir: Path,
    *,
    expected_shards: int,
) -> dict[tuple[str, str], list[tuple[Path, dict[str, Any]]]]:
    expected_total = len(DATASETS) * len(ETA_VALUES) * expected_shards
    paths = sorted(input_dir.glob("*.summary.json")) if input_dir.is_dir() else []
    if len(paths) != expected_total:
        raise ValueError(
            f"eta-sweep requires exactly {expected_total} sampling shard summaries, "
            f"got {len(paths)}"
        )

    grouped: dict[tuple[str, str], list[tuple[Path, dict[str, Any]]]] = {}
    for path in paths:
        summary = json.loads(path.read_text())
        if not isinstance(summary, dict):
            raise ValueError(f"{path}: summary must be a JSON object")
        if summary.get("protocol_id") != PROTOCOL_ID:
            raise ValueError(f"{path}: protocol mismatch")
        missing = [key for key in _REQUIRED_SUMMARY_KEYS if key not in summary]
        if missing:
            raise ValueError(f"{path}: missing required keys {missing}")
        dataset = str(summary["dataset"])
        if dataset not in DATASETS:
            raise ValueError(f"{path}: unexpected dataset {dataset!r}")
        eta = eta_value(summary["unified_guidance_scale"])
        tag = eta_tag(eta)
        if summary["run_name"] != expected_run_name(dataset, eta):
            raise ValueError(f"{path}: run_name mismatch")
        if (int(summary["num_samples"]), int(summary["num_steps"])) != (
            NUM_SAMPLES,
            NUM_STEPS,
        ):
            raise ValueError(f"{path}: only N100/S10 belongs to this protocol")
        if int(summary["model_pose_step_budget"]) != 1000:
            raise ValueError(f"{path}: learned-model pose-step budget must equal 1000")
        grouped.setdefault((dataset, tag), []).append((path, summary))

    expected_cells = {(dataset, tag) for dataset in DATASETS for tag in ETA_TAGS}
    if set(grouped) != expected_cells:
        raise ValueError(
            "eta-sweep run-cell mismatch; "
            f"missing={sorted(expected_cells - set(grouped))}, "
            f"extra={sorted(set(grouped) - expected_cells)}"
        )
    for key, values in grouped.items():
        if len(values) != expected_shards:
            raise ValueError(
                f"{key}: expected {expected_shards} shard summaries, got {len(values)}"
            )
    return grouped


def _validate_direct_runtime(
    summary: dict[str, Any],
    *,
    expected_successes: int,
) -> None:
    stats = summary.get("guidance_runtime_stats")
    if not isinstance(stats, dict):
        raise ValueError(f"{summary['run_name']}: nonzero eta lacks runtime statistics")
    for name, value in stats.items():
        if value is not None:
            _numeric_scalar(value, label=f"{summary['run_name']}.{name}")
        if "nonfinite" in name.lower() and value is not None and float(value) != 0.0:
            raise ValueError(f"{summary['run_name']}: {name} must be zero")

    expected_steps = expected_successes * _active_intervals()
    expected_poses = expected_steps * NUM_SAMPLES
    exact_counters = {
        "direct_steps_attempted": expected_steps,
        "direct_pose_evaluations": expected_poses,
        "direct_batched_energy_evaluations": expected_steps,
        "direct_pose_energy_evaluations": expected_poses,
        "direct_nonfinite_poses": 0,
    }
    for key, expected in exact_counters.items():
        if int(stats.get(key, -1)) != expected:
            raise ValueError(
                f"{summary['run_name']}: {key} must be {expected}, got {stats.get(key)!r}"
            )
    applied = int(stats.get("direct_pose_applied", -1))
    if not 0 <= applied <= expected_poses:
        raise ValueError(f"{summary['run_name']}: invalid direct_pose_applied")
    cap_limits = {
        "direct_max_translation_velocity": 5.000001,
        "direct_max_angular_velocity": 5.000001,
        "direct_max_estimated_atom_displacement": 0.250001,
    }
    for key, limit in cap_limits.items():
        if float(stats.get(key, math.inf)) > limit:
            raise ValueError(f"{summary['run_name']}: {key} exceeded its cap")


def _aggregate_cell(
    paths_and_summaries: list[tuple[Path, dict[str, Any]]],
    ids: tuple[str, ...],
    audit: dict[str, Any],
    *,
    dataset: str,
    eta: float,
    input_dir: Path,
    expected_shards: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    run_name = expected_run_name(dataset, eta)
    ordered_summaries = sorted(
        paths_and_summaries,
        key=lambda item: int(item[1]["shard_index"]),
    )
    indices = [int(summary["shard_index"]) for _, summary in ordered_summaries]
    if indices != list(range(expected_shards)):
        raise ValueError(f"{run_name}: incomplete or duplicate shard indices")
    reference_metadata = _stable_shard_metadata(ordered_summaries[0][1])
    rows_by_id: dict[str, dict[str, Any]] = {}
    summaries: list[dict[str, Any]] = []
    traces: list[list[dict[str, Any]]] = []

    exact_settings = {
        "model_pose_step_budget": 1000,
        "num_shards": expected_shards,
        "seed": 42,
        "unified_guidance_scale": eta,
        "unified_guidance_mode": EXPECTED_GUIDANCE_MODE,
        "prior_pool_size": 100,
        "prior_pool_hash_contract": "EFFDOCK_SHARED_PRIOR_V1",
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "confidence_checkpoint_sha256": None,
        "config_sha256": EXPECTED_CONFIG_SHA256,
        "sigma": 0.5,
        "time_schedule": "late",
        "schedule_power": 3.0,
        "pocket_cutoff": 10.0,
        "center_jitter_sigma": 0.0,
        "vina_guidance_scale": 0.0,
        "unified_guidance_start_t": 0.5,
        "unified_guidance_ramp_power": 1.0,
        "unified_guidance_max_force": 20.0,
        "unified_guidance_max_velocity": 5.0,
        "unified_guidance_max_angular_velocity": 5.0,
        "unified_guidance_max_atom_displacement": 0.25,
        "unified_guidance_max_backtracks": 8,
        "unified_guidance_protein_shell": 18.0,
        "unified_guidance_receptor_policy": RECEPTOR_POLICY,
        "require_full_ligand_atom_mapping": True,
        "refine": "none",
    }
    for path, summary in ordered_summaries:
        if summary["protocol_id"] != PROTOCOL_ID:
            raise ValueError(f"{path}: protocol mismatch")
        if summary["run_name"] != run_name or summary["dataset"] != dataset:
            raise ValueError(f"{path}: run identity mismatch")
        if int(summary["num_discovered_total"]) != EXPECTED_DATASET_COUNTS[dataset]:
            raise ValueError(f"{path}: full-cohort discovered count mismatch")
        if summary["pocket_centers_sha256"] != EXPECTED_POCKET_CENTERS_SHA256[dataset]:
            raise ValueError(f"{path}: pocket-center hash mismatch")
        if summary["benchmark_input_identity"] != audit["benchmark_input_identity"]:
            raise ValueError(f"{path}: benchmark input identity differs from audit")
        if summary["eligibility_manifest_sha256"] != audit["source_sha256"]:
            raise ValueError(f"{path}: fresh cohort-audit hash mismatch")
        if summary["guidance_implementation"] != audit["implementation"]:
            raise ValueError(f"{path}: guidance implementation differs from audit")
        if _stable_shard_metadata(summary) != reference_metadata:
            raise ValueError(f"{path}: inconsistent metadata across shards")
        for key, expected in exact_settings.items():
            _require_exact_setting(summary, key, expected)

        shard_rows = _read_shard_rows(path, summary, input_dir)
        shard_index = int(summary["shard_index"])
        expected_ids = set(ids[shard_index::expected_shards])
        observed_ids = {row["id"] for row in shard_rows}
        if observed_ids != expected_ids:
            raise ValueError(
                f"{path}: deterministic shard ID mismatch; "
                f"missing={sorted(expected_ids - observed_ids)[:5]}, "
                f"outside={sorted(observed_ids - expected_ids)[:5]}"
            )
        if eta > 0.0:
            _validate_direct_runtime(summary, expected_successes=len(shard_rows))
            if summary.get("guidance_parameter_set") != audit["parameter_set"]:
                raise ValueError(f"{path}: guidance parameter set differs from audit")
            policy = audit["receptor_policy_identity"]
            if summary.get("guidance_receptor_policy_identities") != {policy["sha256"]: policy}:
                raise ValueError(f"{path}: receptor-policy identity differs from audit")
        else:
            if summary.get("guidance_runtime_stats") not in (None, {}):
                raise ValueError(f"{path}: eta=0 unexpectedly has guidance telemetry")
            if summary.get("guidance_parameter_set") not in (None, {}):
                raise ValueError(f"{path}: eta=0 unexpectedly has guidance parameters")

        expected_row_mode = "unified_normalized_drift" if eta > 0.0 else "none"
        shard_traces: list[list[dict[str, Any]]] = []
        for row in shard_rows:
            complex_id = row["id"]
            if complex_id in rows_by_id:
                raise ValueError(f"{run_name}: duplicate row for {complex_id}")
            if row["guidance_mode"] != expected_row_mode:
                raise ValueError(f"{run_name}/{complex_id}: guidance mode mismatch")
            if int(row["prior_pool_size"]) != 100:
                raise ValueError(f"{run_name}/{complex_id}: prior pool must equal 100")
            parameter_hash = str(row["guidance_parameter_sha256"])
            if eta > 0.0:
                if parameter_hash != audit["parameter_set"]["sha256"]:
                    raise ValueError(f"{run_name}/{complex_id}: guidance hash mismatch")
                if row.get("guidance_receptor_policy") != RECEPTOR_POLICY:
                    raise ValueError(f"{run_name}/{complex_id}: receptor policy mismatch")
            elif parameter_hash:
                raise ValueError(f"{run_name}/{complex_id}: eta=0 has guidance hash")
            trace = _validate_direct_step_trace(row, eta=eta)
            traces.append(trace)
            shard_traces.append(trace)
            rows_by_id[complex_id] = row
        if eta > 0.0:
            _validate_trace_runtime_consistency(summary, shard_traces)
        summaries.append(summary)

    id_set = set(ids)
    _validate_failure_records(
        ordered_summaries,
        set(rows_by_id),
        id_set,
        run_name=run_name,
    )
    if set(rows_by_id) != id_set:
        raise ValueError(f"{run_name}: exact full-cohort coverage mismatch")
    for global_index, complex_id in enumerate(ids, start=1):
        row = rows_by_id[complex_id]
        expected_seed = 42 + global_index
        if int(row["sampling_seed"]) != expected_seed:
            raise ValueError(f"{run_name}/{complex_id}: sampling_seed must be {expected_seed}")
        prior_hash = str(row["prior_pool_sha256"])
        if len(prior_hash) != 64:
            raise ValueError(f"{run_name}/{complex_id}: invalid prior-pool hash")
        try:
            int(prior_hash, 16)
        except ValueError as exc:
            raise ValueError(f"{run_name}/{complex_id}: invalid prior-pool hash") from exc
    _validate_sampling_input_rows(
        rows_by_id,
        audit,
        dataset=dataset,
        arm="guided" if eta > 0.0 else "unguided",
    )
    stats = summarize_rows([rows_by_id[complex_id] for complex_id in ids])
    guidance_telemetry = None
    if eta > 0.0:
        guidance_telemetry = _combine_runtime_stats(summaries)
        guidance_telemetry["active_intervals"] = _aggregate_interval_telemetry(
            traces,
            eta=eta,
        )
    return rows_by_id, {
        "run_name": run_name,
        "eta": eta,
        "eta_tag": eta_tag(eta),
        "count": len(ids),
        "ids_sha256": sorted_id_sha256(list(ids)),
        "stats": stats,
        "target_metrics": _target_metrics(stats),
        "guidance_telemetry": guidance_telemetry,
        "cuda_runtime": _combine_cuda_runtime(summaries),
        "shard_summaries": [str(path) for path, _ in ordered_summaries],
    }


def build_report(
    input_dir: Path,
    cohort_audit: Path,
    *,
    expected_shards: int = DEFAULT_EXPECTED_SHARDS,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
) -> dict[str, Any]:
    if expected_shards < 1 or bootstrap_resamples < 1:
        raise ValueError("expected_shards and bootstrap_resamples must be >= 1")
    audits = load_full_cohort_audits(cohort_audit)
    grouped = _load_sampling_inventory(input_dir, expected_shards=expected_shards)
    if audits["astex"]["implementation"] != audits["posebusters"]["implementation"]:
        raise ValueError("guidance implementation differs across dataset audits")
    if audits["astex"]["parameter_set"] != audits["posebusters"]["parameter_set"]:
        raise ValueError("guidance parameter set differs across dataset audits")

    all_rows: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    report: dict[str, Any] = {
        "protocol_id": PROTOCOL_ID,
        "status": "complete_strict_full_cohort_paired_descriptive_eta_sweep",
        "claim_boundary": (
            "paired descriptive Astex/PoseBusters reference-pocket redocking; all eta "
            "values are reported without automatic selection or production admission"
        ),
        "coupling": {
            "mode": "unified_normalized_drift",
            "normalization": "one pose-wise scalar in induced atom-velocity RMS space",
            "time_ramp": "interval-average quadrature",
        },
        "condition": {
            "name": CONDITION,
            "num_samples": NUM_SAMPLES,
            "num_steps": NUM_STEPS,
            "model_pose_step_budget": NUM_SAMPLES * NUM_STEPS,
        },
        "eta_grid": [
            {"eta": eta, "tag": tag} for eta, tag in zip(ETA_VALUES, ETA_TAGS, strict=True)
        ],
        "sampling_inventory": {
            "run_cells": len(DATASETS) * len(ETA_VALUES),
            "shards_per_cell": expected_shards,
            "total_shard_tasks": len(DATASETS) * len(ETA_VALUES) * expected_shards,
        },
        "benchmark_input_manifest_sha256": EXPECTED_BENCHMARK_INPUT_MANIFEST_SHA256,
        "guidance_implementation": audits["astex"]["implementation"],
        "guidance_parameter_set": audits["astex"]["parameter_set"],
        "bootstrap": {
            "method": "paired complex-ID bootstrap, percentile 95% CI",
            "seed": bootstrap_seed,
            "resamples": bootstrap_resamples,
        },
        "datasets": {},
    }
    for dataset in DATASETS:
        audit = audits[dataset]
        ids = audit["ids"]
        cells: dict[str, Any] = {}
        for eta, tag in zip(ETA_VALUES, ETA_TAGS, strict=True):
            rows, aggregate = _aggregate_cell(
                grouped[(dataset, tag)],
                ids,
                audit,
                dataset=dataset,
                eta=eta,
                input_dir=input_dir,
                expected_shards=expected_shards,
            )
            all_rows[(dataset, tag)] = rows
            cells[tag] = aggregate

        baseline = all_rows[(dataset, ETA_TAGS[0])]
        eta_vs_eta0 = {
            tag: _paired_comparison(
                baseline,
                all_rows[(dataset, tag)],
                ids,
                baseline_label=ETA_TAGS[0],
                comparison_label=tag,
                seed=bootstrap_seed,
                resamples=bootstrap_resamples,
            )
            for tag in ETA_TAGS
        }
        pairing_ledger = hashlib.sha256()
        pairing_ledger.update(b"EFFDOCK_ETA_SWEEP_SHARED_PRIOR_V2\0")
        for complex_id in ids:
            records = [all_rows[(dataset, tag)][complex_id] for tag in ETA_TAGS]
            seeds = {int(record["sampling_seed"]) for record in records}
            prior_hashes = {str(record["prior_pool_sha256"]) for record in records}
            if len(seeds) != 1 or len(prior_hashes) != 1:
                raise ValueError(
                    f"{dataset}/{complex_id}: eta cells do not share the exact prior pool"
                )
            pairing_ledger.update(
                f"{complex_id}\t{next(iter(seeds))}\t{next(iter(prior_hashes))}\n".encode()
            )

        report["datasets"][dataset] = {
            "coverage": {
                "count": len(ids),
                "expected": EXPECTED_DATASET_COUNTS[dataset],
                "ids_sha256": sorted_id_sha256(list(ids)),
                "ids_hash_contract": ID_HASH_CONTRACT,
                "audit_path": audit["source_path"],
                "audit_sha256": audit["source_sha256"],
            },
            "cells": cells,
            "eta_vs_eta0": eta_vs_eta0,
            "prior_pairing": {
                "verified": True,
                "complexes": len(ids),
                "contract": "same seed and exact 100-pose prior hash across all eta values",
                "pairing_ledger_sha256": pairing_ledger.hexdigest(),
            },
        }
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--cohort-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-shards", type=int, default=DEFAULT_EXPECTED_SHARDS)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument("--bootstrap-resamples", type=int, default=DEFAULT_BOOTSTRAP_RESAMPLES)
    args = parser.parse_args(argv)
    result = build_report(
        args.input_dir,
        args.cohort_audit,
        expected_shards=args.expected_shards,
        bootstrap_seed=args.bootstrap_seed,
        bootstrap_resamples=args.bootstrap_resamples,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
