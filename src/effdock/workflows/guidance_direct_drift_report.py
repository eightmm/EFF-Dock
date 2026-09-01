#!/usr/bin/env python3
"""Strict full-cohort report for normalized direct GuidanceEnergy drift."""

from __future__ import annotations

import argparse
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
    CONDITIONS,
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

PROTOCOL_ID = "EFFDOCK-UNIFIED-GUIDANCE-DIRECT-DRIFT-BUDGET1000-V1"
ARMS = ("unguided", "direct")
ARM_SCALES = {"unguided": 0.0, "direct": 0.1}
EXPECTED_GUIDANCE_MODE = "normalized_drift"
EXPECTED_ROW_MODES = {"unguided": "none", "direct": "unified_normalized_drift"}

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


def expected_run_name(dataset: str, num_samples: int, num_steps: int, arm: str) -> str:
    return (
        f"effdock-guidance-direct-drift-v1-{dataset}-"
        f"n{num_samples}-s{num_steps}-{arm}"
    )


def _active_intervals(num_steps: int) -> int:
    grid = build_time_grid(num_steps, schedule="late", power=3.0)
    return sum(float(right) > 0.5 for right in grid[1:])


def _combine_runtime_stats(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    names = sorted(
        {
            name
            for summary in summaries
            for name in (summary.get("guidance_runtime_stats") or {})
        }
    )
    result: dict[str, Any] = {}
    for name in names:
        values = [
            summary["guidance_runtime_stats"].get(name)
            for summary in summaries
            if isinstance(summary.get("guidance_runtime_stats"), dict)
        ]
        present = [value for value in values if value is not None]
        if not present:
            result[name] = None
        elif name.startswith("max_") or name.startswith("direct_max_"):
            result[name] = max(float(value) for value in present)
        elif name.startswith("min_"):
            result[name] = min(float(value) for value in present)
        elif all(isinstance(value, int) and not isinstance(value, bool) for value in present):
            result[name] = sum(int(value) for value in present)
        else:
            result[name] = sum(float(value) for value in present)
    pose_count = int(result.get("direct_pose_evaluations", 0) or 0)
    finite_count = pose_count - int(result.get("direct_nonfinite_poses", 0) or 0)
    if finite_count > 0:
        for source, target in (
            ("direct_reference_atom_speed_rms_sum", "direct_mean_reference_atom_speed_rms"),
            ("direct_raw_atom_speed_rms_sum", "direct_mean_raw_atom_speed_rms"),
            ("direct_applied_atom_speed_rms_sum", "direct_mean_applied_atom_speed_rms"),
            ("direct_cap_scale_sum", "direct_mean_cap_scale"),
        ):
            result[target] = float(result.get(source, 0.0)) / finite_count
    return result


def _combine_cuda_runtime(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    """Retain the peak CUDA allocation evidence across every shard in a cell."""
    runtimes = [summary.get("runtime") for summary in summaries]
    if any(not isinstance(runtime, dict) for runtime in runtimes):
        raise ValueError("every sampling shard requires a runtime object")
    typed = [runtime for runtime in runtimes if isinstance(runtime, dict)]
    required = ("cuda_max_memory_allocated_bytes", "cuda_max_memory_reserved_bytes")
    for runtime in typed:
        missing = [key for key in required if key not in runtime]
        if missing:
            raise ValueError(f"sampling runtime is missing CUDA fields {missing}")
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


def _require_exact_setting(summary: dict[str, Any], key: str, expected: Any) -> None:
    actual = summary[key]
    matches = (
        math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-12)
        if isinstance(expected, float)
        else actual == expected
    )
    if not matches:
        raise ValueError(
            f"{summary['run_name']}: {key} must be {expected!r}, got {actual!r}"
        )


def _aggregate_cell(
    paths_and_summaries: list[tuple[Path, dict[str, Any]]],
    ids: tuple[str, ...],
    audit: dict[str, Any],
    *,
    dataset: str,
    arm: str,
    num_samples: int,
    num_steps: int,
    input_dir: Path,
    expected_shards: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    run_name = expected_run_name(dataset, num_samples, num_steps, arm)
    if len(paths_and_summaries) != expected_shards:
        raise ValueError(
            f"{run_name}: expected {expected_shards} summaries, "
            f"got {len(paths_and_summaries)}"
        )
    paths_and_summaries = sorted(
        paths_and_summaries,
        key=lambda item: int(item[1]["shard_index"]),
    )
    if [int(value[1]["shard_index"]) for value in paths_and_summaries] != list(
        range(expected_shards)
    ):
        raise ValueError(f"{run_name}: incomplete or duplicate shard indices")
    reference_metadata = _stable_shard_metadata(paths_and_summaries[0][1])
    rows_by_id: dict[str, dict[str, Any]] = {}
    summaries: list[dict[str, Any]] = []

    exact_settings = {
        "model_pose_step_budget": 1000,
        "num_shards": expected_shards,
        "seed": 42,
        "unified_guidance_scale": ARM_SCALES[arm],
        "unified_guidance_mode": EXPECTED_GUIDANCE_MODE,
        "prior_pool_size": 100,
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
    for path, summary in paths_and_summaries:
        missing = [key for key in _REQUIRED_SUMMARY_KEYS if key not in summary]
        if missing:
            raise ValueError(f"{path}: missing required keys {missing}")
        if summary["protocol_id"] != PROTOCOL_ID:
            raise ValueError(f"{path}: protocol mismatch")
        if summary["run_name"] != run_name or summary["dataset"] != dataset:
            raise ValueError(f"{path}: run identity mismatch")
        if int(summary["num_samples"]) != num_samples or int(summary["num_steps"]) != num_steps:
            raise ValueError(f"{path}: budget-cell mismatch")
        if int(summary["num_discovered_total"]) != EXPECTED_DATASET_COUNTS[dataset]:
            raise ValueError(f"{path}: full-cohort discovered count mismatch")
        if summary["pocket_centers_sha256"] != EXPECTED_POCKET_CENTERS_SHA256[dataset]:
            raise ValueError(f"{path}: pocket-center hash mismatch")
        if summary["benchmark_input_identity"] != audit["benchmark_input_identity"]:
            raise ValueError(f"{path}: benchmark input identity differs from audit")
        if summary["eligibility_manifest_sha256"] != audit["source_sha256"]:
            raise ValueError(f"{path}: fresh cohort-audit hash mismatch")
        if summary["guidance_implementation"] != audit["implementation"]:
            raise ValueError(f"{path}: implementation differs from fresh audit")
        if _stable_shard_metadata(summary) != reference_metadata:
            raise ValueError(f"{path}: inconsistent metadata across shards")
        for key, expected in exact_settings.items():
            _require_exact_setting(summary, key, expected)
        if arm == "direct":
            runtime_stats = summary.get("guidance_runtime_stats")
            if not isinstance(runtime_stats, dict):
                raise ValueError(f"{path}: direct shard lacks runtime statistics")
            expected_steps = int(summary["num_success"]) * _active_intervals(num_steps)
            expected_poses = expected_steps * num_samples
            if int(runtime_stats.get("direct_steps_attempted", -1)) != expected_steps:
                raise ValueError(f"{path}: direct step count mismatch")
            if int(runtime_stats.get("direct_pose_evaluations", -1)) != expected_poses:
                raise ValueError(f"{path}: direct pose-evaluation count mismatch")
            if int(runtime_stats.get("direct_nonfinite_poses", -1)) != 0:
                raise ValueError(f"{path}: direct guidance contains non-finite poses")
            if float(runtime_stats.get("direct_max_translation_velocity", math.inf)) > 5.000001:
                raise ValueError(f"{path}: direct translation cap exceeded")
            if float(runtime_stats.get("direct_max_angular_velocity", math.inf)) > 5.000001:
                raise ValueError(f"{path}: direct angular cap exceeded")
            if (
                float(runtime_stats.get("direct_max_estimated_atom_displacement", math.inf))
                > 0.250001
            ):
                raise ValueError(f"{path}: direct displacement cap exceeded")
            parameter_set = summary.get("guidance_parameter_set")
            if parameter_set != audit["parameter_set"]:
                raise ValueError(f"{path}: guidance parameter set differs from audit")
            expected_policy = audit["receptor_policy_identity"]
            if summary.get("guidance_receptor_policy_identities") != {
                expected_policy["sha256"]: expected_policy
            }:
                raise ValueError(f"{path}: receptor-policy identity differs from audit")
        elif summary.get("guidance_runtime_stats") not in (None, {}):
            raise ValueError(f"{path}: unguided shard unexpectedly has guidance statistics")

        for row in _read_shard_rows(path, summary, input_dir):
            complex_id = row["id"]
            if complex_id in rows_by_id:
                raise ValueError(f"{run_name}: duplicate row for {complex_id}")
            if row["guidance_mode"] != EXPECTED_ROW_MODES[arm]:
                raise ValueError(f"{run_name}/{complex_id}: guidance mode mismatch")
            if int(row["prior_pool_size"]) != 100:
                raise ValueError(f"{run_name}/{complex_id}: prior pool must equal 100")
            parameter_hash = str(row["guidance_parameter_sha256"])
            if arm == "direct" and parameter_hash != audit["parameter_set"]["sha256"]:
                raise ValueError(f"{run_name}/{complex_id}: guidance hash mismatch")
            if arm == "direct" and row.get("guidance_receptor_policy") != RECEPTOR_POLICY:
                raise ValueError(f"{run_name}/{complex_id}: receptor policy mismatch")
            if arm == "unguided" and parameter_hash:
                raise ValueError(f"{run_name}/{complex_id}: unguided row has guidance hash")
            rows_by_id[complex_id] = row
        summaries.append(summary)

    id_set = set(ids)
    _validate_failure_records(
        paths_and_summaries,
        set(rows_by_id),
        id_set,
        run_name=run_name,
    )
    if set(rows_by_id) != id_set:
        raise ValueError(f"{run_name}: exact full-cohort coverage mismatch")
    _validate_sampling_input_rows(
        rows_by_id,
        audit,
        dataset=dataset,
        arm="guided" if arm == "direct" else "unguided",
    )
    ordered = [rows_by_id[complex_id] for complex_id in ids]
    stats = summarize_rows(ordered)
    return rows_by_id, {
        "count": len(ordered),
        "ids_sha256": sorted_id_sha256(list(ids)),
        "stats": stats,
        "target_metrics": _target_metrics(stats),
        "runtime_stats": _combine_runtime_stats(summaries) if arm == "direct" else None,
        "cuda_runtime": _combine_cuda_runtime(summaries),
        "shard_summaries": [str(path) for path, _ in paths_and_summaries],
    }


def build_report(
    input_dir: Path,
    cohort_audit: Path,
    *,
    expected_shards: int = DEFAULT_EXPECTED_SHARDS,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
) -> dict[str, Any]:
    audits = load_full_cohort_audits(cohort_audit)
    grouped: dict[tuple[str, str, str], list[tuple[Path, dict[str, Any]]]] = {}
    for path in sorted(input_dir.glob("*.summary.json")):
        summary = json.loads(path.read_text())
        if not isinstance(summary, dict) or summary.get("protocol_id") != PROTOCOL_ID:
            continue
        dataset = str(summary.get("dataset"))
        arm = "direct" if math.isclose(
            float(summary.get("unified_guidance_scale", -1.0)),
            ARM_SCALES["direct"],
            rel_tol=0.0,
            abs_tol=1e-12,
        ) else "unguided"
        condition = next(
            (
                name
                for name, samples, steps in CONDITIONS
                if (int(summary.get("num_samples", -1)), int(summary.get("num_steps", -1)))
                == (samples, steps)
            ),
            None,
        )
        if dataset not in DATASETS or condition is None:
            raise ValueError(f"{path}: unexpected dataset or budget cell")
        grouped.setdefault((dataset, condition, arm), []).append((path, summary))

    expected = {
        (dataset, condition, arm)
        for dataset in DATASETS
        for condition, _, _ in CONDITIONS
        for arm in ARMS
    }
    if set(grouped) != expected:
        raise ValueError(
            f"direct-drift protocol cell mismatch; missing={sorted(expected - set(grouped))}, "
            f"extra={sorted(set(grouped) - expected)}"
        )

    rows: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = {}
    report: dict[str, Any] = {
        "protocol_id": PROTOCOL_ID,
        "status": "complete_strict_full_cohort_paired",
        "claim_boundary": (
            "paired descriptive reference-pocket redocking; Astex/PoseBusters were "
            "already opened and no value is a production-admission decision"
        ),
        "coupling": {
            "mode": "unified_normalized_drift",
            "normalization": "one pose-wise scalar in induced atom-velocity RMS space",
            "time_ramp": "interval-average quadrature",
            "strength": 0.1,
        },
        "benchmark_input_manifest_sha256": EXPECTED_BENCHMARK_INPUT_MANIFEST_SHA256,
        "expected_shards_per_cell": expected_shards,
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
        dataset_result: dict[str, Any] = {
            "coverage": {
                "count": len(ids),
                "expected": EXPECTED_DATASET_COUNTS[dataset],
                "ids_sha256": sorted_id_sha256(list(ids)),
                "ids_hash_contract": ID_HASH_CONTRACT,
                "audit_path": audit["source_path"],
                "audit_sha256": audit["source_sha256"],
            },
            "cells": {},
        }
        for condition, num_samples, num_steps in CONDITIONS:
            cell: dict[str, Any] = {}
            for arm in ARMS:
                current_rows, aggregate = _aggregate_cell(
                    grouped[(dataset, condition, arm)],
                    ids,
                    audit,
                    dataset=dataset,
                    arm=arm,
                    num_samples=num_samples,
                    num_steps=num_steps,
                    input_dir=input_dir,
                    expected_shards=expected_shards,
                )
                rows[(dataset, condition, arm)] = current_rows
                cell[arm] = aggregate
            cell["direct_minus_unguided"] = _paired_comparison(
                rows[(dataset, condition, "unguided")],
                rows[(dataset, condition, "direct")],
                ids,
                baseline_label="unguided",
                comparison_label="direct",
                seed=bootstrap_seed,
                resamples=bootstrap_resamples,
            )
            dataset_result["cells"][condition] = cell

        for complex_id in ids:
            records = [
                rows[(dataset, condition, arm)][complex_id]
                for condition, _, _ in CONDITIONS
                for arm in ARMS
            ]
            if len({int(record["sampling_seed"]) for record in records}) != 1:
                raise ValueError(f"{dataset}/{complex_id}: sampling seeds are not paired")
            if len({str(record["prior_pool_sha256"]) for record in records}) != 1:
                raise ValueError(f"{dataset}/{complex_id}: prior pools are not paired")
        dataset_result["prior_pairing"] = {
            "verified": True,
            "complexes": len(ids),
            "contract": "same seed and exact 100-pose pool across all cells and arms",
        }
        report["datasets"][dataset] = dataset_result
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
