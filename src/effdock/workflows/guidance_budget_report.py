#!/usr/bin/env python3
"""Strict paired report for the unified-guidance 1,000-step budget study."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from effdock.workflows.evaluate import summarize_rows

PROTOCOL_ID = "EFFDOCK-UNIFIED-GUIDANCE-BUDGET1000-V1"
DATASETS = ("astex", "posebusters")
CONDITIONS = (
    ("n100_s10", 100, 10),
    ("n50_s20", 50, 20),
    ("n40_s25", 40, 25),
)
ARM_SCALES = {"unguided": 0.0, "guided": 0.1}
DEFAULT_EXPECTED_SHARDS = 8
DEFAULT_BOOTSTRAP_SEED = 20260731
DEFAULT_BOOTSTRAP_RESAMPLES = 10_000
EXPECTED_CHECKPOINT_SHA256 = "6932fb3ba6ebac770f714453529656a44b8f33cf15119d23c9e675d2d60b36db"
EXPECTED_CONFIG_SHA256 = "39aa62e4a48ed6f3aa4ff59345fb43a81220e2baba22edfd5beb0c4981b307ec"
EXPECTED_GUIDANCE_PARAMETER_SHA256 = (
    "7851dfe3cb2f290d3fce6e3ae2e2fe1d785cd5bc2c730e6d13bbcfb67e2b6012"
)
EXPECTED_POCKET_CENTERS_SHA256 = {
    "astex": "1ac4d8629a7ee2adb785173db56fb69ec4140d68e3057631ae10df6ef88d0d85",
    "posebusters": "2d3db55c8cc75650cff85d8e3c12445fb8f45fbe2673d8bbc32045ee8c0f6ad0",
}

_REQUIRED_SUMMARY_KEYS = (
    "run_name",
    "protocol_id",
    "dataset",
    "num_samples",
    "num_steps",
    "model_pose_step_budget",
    "num_shards",
    "shard_index",
    "seed",
    "unified_guidance_scale",
    "prior_pool_size",
    "checkpoint_sha256",
    "confidence_checkpoint_sha256",
    "config_sha256",
    "pocket_centers_sha256",
    "eligibility_manifest_sha256",
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
    "refine",
    "csv",
    "failures",
)
_SHARD_VARYING_KEYS = {
    "csv",
    "failures",
    "guidance_operator_stats",
    "guidance_runtime_stats",
    "guidance_receptor_provenance_by_id",
    "num_assigned",
    "num_failed",
    "num_success",
    "runtime",
    "shard_index",
    "stats",
}
_REQUIRED_ROW_KEYS = {
    "id",
    "oracle_rmsd",
    "oracle_fast_valid",
    "num_fast_valid_candidates",
    "fast_valid_oracle_rmsd",
    "joint_fast_valid_and_rmsd_lt2",
    "prior_pool_size",
    "sampling_seed",
    "prior_pool_sha256",
    "guidance_mode",
    "guidance_parameter_sha256",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ids_sha256(ids: list[str] | tuple[str, ...]) -> str:
    payload = "".join(f"{complex_id}\n" for complex_id in sorted(ids))
    return hashlib.sha256(payload.encode()).hexdigest()


def _expected_run_name(dataset: str, num_samples: int, num_steps: int, arm: str) -> str:
    return f"effdock-guidance-budget1000-v1-{dataset}-n{num_samples}-s{num_steps}-{arm}"


def _require_keys(record: dict[str, Any], keys: tuple[str, ...], *, label: str) -> None:
    missing = [key for key in keys if key not in record]
    if missing:
        raise ValueError(f"{label}: missing required keys {missing}")


def _load_eligibility(path: Path) -> tuple[dict[str, tuple[str, ...]], dict[str, Any]]:
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError("eligibility manifest must be a JSON object")
    if raw.get("protocol_id") != PROTOCOL_ID:
        raise ValueError(
            "eligibility protocol_id mismatch: "
            f"expected {PROTOCOL_ID!r}, got {raw.get('protocol_id')!r}"
        )
    datasets = raw.get("datasets")
    if not isinstance(datasets, dict):
        raise ValueError("eligibility manifest requires a datasets object")
    if set(datasets) != set(DATASETS):
        raise ValueError(
            f"eligibility datasets must be exactly {list(DATASETS)}, got {sorted(datasets)}"
        )

    eligible_by_dataset: dict[str, tuple[str, ...]] = {}
    manifest_summary: dict[str, Any] = {}
    for dataset in DATASETS:
        entry = datasets[dataset]
        if not isinstance(entry, dict) or not isinstance(entry.get("eligible_ids"), list):
            raise ValueError(f"eligibility datasets.{dataset}.eligible_ids must be a list")
        ids = entry["eligible_ids"]
        if not ids or any(not isinstance(value, str) or not value.strip() for value in ids):
            raise ValueError(f"eligibility {dataset} IDs must be non-empty strings")
        if any(value != value.strip() for value in ids):
            raise ValueError(f"eligibility {dataset} IDs may not contain surrounding whitespace")
        if len(ids) != len(set(ids)):
            raise ValueError(f"eligibility {dataset} contains duplicate IDs")
        sorted_ids = tuple(sorted(ids))
        digest = _ids_sha256(sorted_ids)
        if "eligible" in entry and int(entry["eligible"]) != len(sorted_ids):
            raise ValueError(f"eligibility {dataset} eligible count does not match eligible_ids")
        if entry.get("eligible_ids_sha256", digest) != digest:
            raise ValueError(f"eligibility {dataset} eligible_ids_sha256 mismatch")
        discovered = entry.get("discovered")
        if discovered is not None and int(discovered) < len(sorted_ids):
            raise ValueError(f"eligibility {dataset} discovered count is smaller than eligible")
        eligible_by_dataset[dataset] = sorted_ids
        manifest_summary[dataset] = {
            "discovered": int(discovered) if discovered is not None else None,
            "eligible": len(sorted_ids),
            "eligible_ids_sha256": digest,
        }
    return eligible_by_dataset, manifest_summary


def _arm_for_scale(value: Any) -> str:
    try:
        scale = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid unified_guidance_scale: {value!r}") from exc
    matches = [
        arm
        for arm, expected in ARM_SCALES.items()
        if math.isclose(scale, expected, rel_tol=0.0, abs_tol=1e-12)
    ]
    if len(matches) != 1:
        raise ValueError(f"unified_guidance_scale must be one of {ARM_SCALES}, got {scale}")
    return matches[0]


def _cell_key(num_samples: int, num_steps: int) -> str:
    for name, expected_samples, expected_steps in CONDITIONS:
        if (num_samples, num_steps) == (expected_samples, expected_steps):
            return name
    raise ValueError(
        "unexpected budget condition "
        f"n{num_samples}/s{num_steps}; expected "
        + ", ".join(f"n{n}/s{s}" for _, n, s in CONDITIONS)
    )


def _stable_shard_metadata(summary: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in summary.items() if key not in _SHARD_VARYING_KEYS}


def _parse_bool(value: Any, *, field: str) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"{field} must be true or false, got {value!r}")


def _convert_row(raw: dict[str, str], *, label: str) -> dict[str, Any]:
    missing = sorted(_REQUIRED_ROW_KEYS - set(raw))
    if missing:
        raise ValueError(f"{label}: CSV row missing required fields {missing}")
    row: dict[str, Any] = {}
    for key, value in raw.items():
        if value is None:
            raise ValueError(f"{label}: missing CSV value for {key}")
        if key == "id":
            row[key] = value.strip()
        elif key.endswith("_rmsd"):
            try:
                row[key] = float(value)
            except ValueError as exc:
                raise ValueError(f"{label}: invalid float for {key}: {value!r}") from exc
        elif key.endswith("_fast_valid") or key in {
            "joint_fast_valid_and_rmsd_lt2",
            "full_heavy_atom_bijection",
            "exact_full_heavy_atom_graph",
        }:
            row[key] = _parse_bool(value, field=f"{label}.{key}")
        elif (
            key
            in {
                "fast_valid_oracle_index",
                "num_fast_valid_candidates",
                "num_input_atoms",
                "num_match_atoms",
                "num_ref_atoms",
                "prior_pool_size",
                "sampling_seed",
            }
            or key.startswith("guidance_")
            and key.endswith(
                (
                    "attempted",
                    "accepted",
                    "rejected",
                    "backtracks",
                    "evaluations",
                    "trials",
                )
            )
        ):
            try:
                row[key] = int(value)
            except ValueError as exc:
                raise ValueError(f"{label}: invalid integer for {key}: {value!r}") from exc
        else:
            row[key] = value
    if not row["id"]:
        raise ValueError(f"{label}: empty complex ID")
    if not math.isfinite(row["oracle_rmsd"]):
        raise ValueError(f"{label}: oracle_rmsd must be finite")
    return row


def _resolve_csv(summary_path: Path, csv_value: Any, input_dir: Path) -> Path:
    if not isinstance(csv_value, str) or not csv_value:
        raise ValueError(f"{summary_path}: completed shard requires a non-empty csv path")
    raw = Path(csv_value)
    if raw.is_absolute():
        candidates = [raw]
    else:
        candidates = [Path.cwd() / raw, summary_path.parent / raw, input_dir / raw]
    hits: list[Path] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.exists() and resolved not in hits:
            hits.append(resolved)
    if not hits:
        raise FileNotFoundError(f"{summary_path}: CSV does not exist: {csv_value}")
    if len(hits) > 1:
        raise ValueError(f"{summary_path}: ambiguous relative CSV path: {csv_value}")
    return hits[0]


def _read_shard_rows(
    summary_path: Path,
    summary: dict[str, Any],
    input_dir: Path,
) -> list[dict[str, Any]]:
    csv_path = _resolve_csv(summary_path, summary["csv"], input_dir)
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{csv_path}: missing CSV header")
        rows = [
            _convert_row(row, label=f"{csv_path}:{line_number}")
            for line_number, row in enumerate(reader, start=2)
        ]
    if "num_success" in summary and int(summary["num_success"]) != len(rows):
        raise ValueError(
            f"{summary_path}: num_success={summary['num_success']} but CSV has {len(rows)} rows"
        )
    failures = summary["failures"]
    if not isinstance(failures, list):
        raise ValueError(f"{summary_path}: failures must be a list")
    if "num_failed" in summary and int(summary["num_failed"]) != len(failures):
        raise ValueError(
            f"{summary_path}: num_failed={summary['num_failed']} "
            f"but failures has {len(failures)} entries"
        )
    if "num_assigned" in summary and int(summary["num_assigned"]) != len(rows) + len(failures):
        raise ValueError(f"{summary_path}: num_assigned does not equal successes plus failures")
    return rows


def _validate_failure_records(
    summaries: list[tuple[Path, dict[str, Any]]],
    row_ids: set[str],
    eligible_ids: set[str],
    *,
    run_name: str,
) -> None:
    failure_ids: list[str] = []
    for path, summary in summaries:
        for failure in summary["failures"]:
            if not isinstance(failure, dict) or not isinstance(failure.get("id"), str):
                raise ValueError(f"{path}: every failure requires a string id")
            failure_ids.append(failure["id"])
    duplicates = sorted(
        complex_id for complex_id in set(failure_ids) if failure_ids.count(complex_id) > 1
    )
    if duplicates:
        raise ValueError(f"{run_name}: duplicate failure IDs: {duplicates[:5]}")
    outside = sorted(set(failure_ids) - eligible_ids)
    if outside:
        raise ValueError(f"{run_name}: failure IDs outside eligibility: {outside[:5]}")
    overlap = sorted(set(failure_ids) & row_ids)
    if overlap:
        raise ValueError(f"{run_name}: IDs appear as both success and failure: {overlap[:5]}")
    if failure_ids:
        preview = ", ".join(sorted(failure_ids)[:5])
        raise ValueError(
            f"{run_name}: {len(failure_ids)} eligible sampling failures; "
            f"strict paired report rejects survivor-only aggregation (first: {preview})"
        )


def _bootstrap_ci(
    baseline: np.ndarray,
    comparison: np.ndarray,
    *,
    reducer: str,
    seed: int,
    resamples: int,
) -> list[float]:
    if baseline.shape != comparison.shape or baseline.ndim != 1 or not len(baseline):
        raise ValueError("paired bootstrap inputs must be non-empty aligned vectors")
    if resamples < 1:
        raise ValueError("bootstrap_resamples must be >= 1")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(baseline), size=(resamples, len(baseline)))
    if reducer == "mean_pp":
        samples = (comparison[indices].mean(axis=1) - baseline[indices].mean(axis=1)) * 100.0
    elif reducer == "median":
        samples = np.median(comparison[indices], axis=1) - np.median(baseline[indices], axis=1)
    else:
        raise ValueError(f"unknown bootstrap reducer: {reducer}")
    return [float(value) for value in np.percentile(samples, [2.5, 97.5])]


def _paired_metric(
    baseline: np.ndarray,
    comparison: np.ndarray,
    *,
    reducer: str,
    unit: str,
    seed: int,
    resamples: int,
) -> dict[str, Any]:
    if reducer == "mean_pp":
        baseline_value = float(baseline.mean() * 100.0)
        comparison_value = float(comparison.mean() * 100.0)
    elif reducer == "median":
        baseline_value = float(np.median(baseline))
        comparison_value = float(np.median(comparison))
    else:
        raise ValueError(f"unknown paired reducer: {reducer}")
    return {
        "baseline": baseline_value,
        "comparison": comparison_value,
        "delta": comparison_value - baseline_value,
        "ci95": _bootstrap_ci(
            baseline,
            comparison,
            reducer=reducer,
            seed=seed,
            resamples=resamples,
        ),
        "unit": unit,
    }


def _paired_comparison(
    baseline_rows: dict[str, dict[str, Any]],
    comparison_rows: dict[str, dict[str, Any]],
    ids: tuple[str, ...],
    *,
    baseline_label: str,
    comparison_label: str,
    seed: int,
    resamples: int,
) -> dict[str, Any]:
    if set(ids) != set(baseline_rows) or set(ids) != set(comparison_rows):
        raise ValueError("paired comparison requires exact, aligned ID coverage")
    oracle_baseline = np.asarray(
        [float(baseline_rows[complex_id]["oracle_rmsd"]) for complex_id in ids]
    )
    oracle_comparison = np.asarray(
        [float(comparison_rows[complex_id]["oracle_rmsd"]) for complex_id in ids]
    )
    joint_baseline = np.asarray(
        [bool(baseline_rows[complex_id]["joint_fast_valid_and_rmsd_lt2"]) for complex_id in ids],
        dtype=float,
    )
    joint_comparison = np.asarray(
        [bool(comparison_rows[complex_id]["joint_fast_valid_and_rmsd_lt2"]) for complex_id in ids],
        dtype=float,
    )
    return {
        "direction": f"{comparison_label}_minus_{baseline_label}",
        "common_ids": len(ids),
        "common_ids_sha256": _ids_sha256(ids),
        "metrics": {
            "oracle_lt2": _paired_metric(
                (oracle_baseline < 2.0).astype(float),
                (oracle_comparison < 2.0).astype(float),
                reducer="mean_pp",
                unit="percentage_points",
                seed=seed,
                resamples=resamples,
            ),
            "oracle_median_rmsd": _paired_metric(
                oracle_baseline,
                oracle_comparison,
                reducer="median",
                unit="angstrom",
                seed=seed,
                resamples=resamples,
            ),
            "joint_fast_valid_and_rmsd_lt2": _paired_metric(
                joint_baseline,
                joint_comparison,
                reducer="mean_pp",
                unit="percentage_points",
                seed=seed,
                resamples=resamples,
            ),
        },
    }


def _target_metrics(stats: dict[str, Any]) -> dict[str, float]:
    return {
        "oracle_lt2_pct": float(stats["oracle"]["pct_lt_2A"]),
        "oracle_median_rmsd": float(stats["oracle"]["median_rmsd"]),
        "joint_fast_valid_and_rmsd_lt2_pct": float(
            stats["candidate_set"]["joint_fast_valid_and_rmsd_lt2_pct"]
        ),
    }


def _aggregate_cell(
    summaries: list[tuple[Path, dict[str, Any]]],
    eligible_ids: tuple[str, ...],
    *,
    arm: str,
    input_dir: Path,
    expected_shards: int,
    manifest_discovered: int | None,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    if len(summaries) != expected_shards:
        run_name = summaries[0][1]["run_name"] if summaries else "unknown"
        raise ValueError(
            f"{run_name}: expected {expected_shards} shard summaries, got {len(summaries)}"
        )
    summaries = sorted(summaries, key=lambda item: int(item[1]["shard_index"]))
    shard_indices = [int(summary["shard_index"]) for _, summary in summaries]
    if shard_indices != list(range(expected_shards)):
        raise ValueError(
            f"{summaries[0][1]['run_name']}: shard indices must be "
            f"0..{expected_shards - 1}, got {shard_indices}"
        )
    reference_metadata = _stable_shard_metadata(summaries[0][1])
    for path, summary in summaries:
        if int(summary["num_shards"]) != expected_shards:
            raise ValueError(
                f"{path}: num_shards={summary['num_shards']} "
                f"does not match expected {expected_shards}"
            )
        if _stable_shard_metadata(summary) != reference_metadata:
            raise ValueError(f"{path}: inconsistent shard metadata")
        if (
            manifest_discovered is not None
            and "num_discovered_total" in summary
            and int(summary["num_discovered_total"]) != manifest_discovered
        ):
            raise ValueError(f"{path}: discovered count does not match eligibility manifest")
        operator_stats = summary.get("guidance_operator_stats")
        if arm == "guided":
            if not isinstance(operator_stats, dict):
                raise ValueError(f"{path}: guided shard lacks guidance_operator_stats")
            required_counters = {
                "steps_attempted",
                "pose_corrections_attempted",
                "pose_corrections_accepted",
                "pose_corrections_rejected",
                "nonfinite_base_poses",
                "nonfinite_trials",
                "max_accepted_atom_displacement",
            }
            missing_counters = sorted(required_counters - set(operator_stats))
            if missing_counters:
                raise ValueError(f"{path}: guidance_operator_stats missing {missing_counters}")
            attempted = int(operator_stats["pose_corrections_attempted"])
            accepted = int(operator_stats["pose_corrections_accepted"])
            rejected = int(operator_stats["pose_corrections_rejected"])
            if attempted <= 0 or accepted + rejected != attempted:
                raise ValueError(f"{path}: inconsistent guidance correction counters")
            if int(operator_stats["nonfinite_base_poses"]) != 0:
                raise ValueError(f"{path}: guided shard contains non-finite base poses")
            if int(operator_stats["nonfinite_trials"]) != 0:
                raise ValueError(f"{path}: guided shard contains non-finite trials")
            if float(operator_stats["max_accepted_atom_displacement"]) > 0.250001:
                raise ValueError(f"{path}: guidance trust-region bound was exceeded")
        elif operator_stats not in (None, {}):
            raise ValueError(f"{path}: unguided shard unexpectedly has operator stats")

    rows_by_id: dict[str, dict[str, Any]] = {}
    field_names: set[str] | None = None
    for path, summary in summaries:
        for row in _read_shard_rows(path, summary, input_dir):
            current_fields = set(row)
            if field_names is None:
                field_names = current_fields
            elif current_fields != field_names:
                raise ValueError(f"{path}: inconsistent CSV columns across shards")
            complex_id = row["id"]
            if complex_id in rows_by_id:
                raise ValueError(f"{summary['run_name']}: duplicate success row for {complex_id}")
            expected_mode = "unified_operator_split" if arm == "guided" else "none"
            if row["guidance_mode"] != expected_mode:
                raise ValueError(
                    f"{summary['run_name']}: {complex_id} guidance_mode must be {expected_mode!r}"
                )
            if int(row["prior_pool_size"]) != 100:
                raise ValueError(
                    f"{summary['run_name']}: {complex_id} prior_pool_size must equal 100"
                )
            prior_hash = str(row["prior_pool_sha256"])
            if len(prior_hash) != 64:
                raise ValueError(f"{summary['run_name']}: {complex_id} invalid prior-pool hash")
            try:
                int(prior_hash, 16)
            except ValueError as exc:
                raise ValueError(
                    f"{summary['run_name']}: {complex_id} invalid prior-pool hash"
                ) from exc
            parameter_hash = str(row["guidance_parameter_sha256"])
            if arm == "guided" and not parameter_hash:
                raise ValueError(
                    f"{summary['run_name']}: {complex_id} lacks guidance parameter hash"
                )
            if arm == "unguided" and parameter_hash:
                raise ValueError(
                    f"{summary['run_name']}: {complex_id} unexpectedly has guidance hash"
                )
            if arm == "guided" and parameter_hash != str(
                summary["guidance_parameter_set"]["sha256"]
            ):
                raise ValueError(f"{summary['run_name']}: {complex_id} guidance hash mismatch")
            rows_by_id[complex_id] = row

    eligible_set = set(eligible_ids)
    _validate_failure_records(
        summaries,
        set(rows_by_id),
        eligible_set,
        run_name=summaries[0][1]["run_name"],
    )
    missing = sorted(eligible_set - set(rows_by_id))
    outside = sorted(set(rows_by_id) - eligible_set)
    if missing or outside:
        raise ValueError(
            f"{summaries[0][1]['run_name']}: eligibility coverage mismatch; "
            f"missing={missing[:5]}, outside={outside[:5]}"
        )
    ordered_rows = [rows_by_id[complex_id] for complex_id in eligible_ids]
    stats = summarize_rows(ordered_rows)
    if "oracle" not in stats or "candidate_set" not in stats:
        raise ValueError(f"{summaries[0][1]['run_name']}: required metrics are absent")
    aggregate = {
        "eligible": len(eligible_ids),
        "success": len(ordered_rows),
        "failed": 0,
        "coverage_pct": 100.0,
        "eligible_ids_sha256": _ids_sha256(eligible_ids),
        "stats": stats,
        "target_metrics": _target_metrics(stats),
        "metadata": reference_metadata,
        "shard_summaries": [str(path) for path, _ in summaries],
    }
    return rows_by_id, aggregate


def _validate_cross_cell_metadata(
    grouped: dict[tuple[str, str, str], list[tuple[Path, dict[str, Any]]]],
    *,
    eligibility_sha256: str,
) -> None:
    flat = [summary for summaries in grouped.values() for _, summary in summaries]
    for key in ("checkpoint_sha256", "confidence_checkpoint_sha256", "config_sha256", "seed"):
        values = {json.dumps(summary[key], sort_keys=True) for summary in flat}
        if len(values) != 1:
            raise ValueError(f"cross-cell metadata mismatch for {key}")
    if any(summary["checkpoint_sha256"] != EXPECTED_CHECKPOINT_SHA256 for summary in flat):
        raise ValueError("protocol docking checkpoint hash mismatch")
    if any(summary["config_sha256"] != EXPECTED_CONFIG_SHA256 for summary in flat):
        raise ValueError("protocol config hash mismatch")
    if any(summary["confidence_checkpoint_sha256"] is not None for summary in flat):
        raise ValueError("protocol requires confidence to be disabled")
    if any(summary["eligibility_manifest_sha256"] != eligibility_sha256 for summary in flat):
        raise ValueError("summary eligibility manifest hash mismatch")
    if any(int(summary["prior_pool_size"]) != 100 for summary in flat):
        raise ValueError("protocol requires prior_pool_size=100 for nested prior comparisons")
    exact_protocol_values = {
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
        "refine": "none",
    }
    for key, expected in exact_protocol_values.items():
        for summary in flat:
            actual = summary[key]
            if isinstance(expected, float):
                matches = math.isclose(
                    float(actual),
                    expected,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            else:
                matches = actual == expected
            if not matches:
                raise ValueError(
                    f"protocol setting mismatch for {key}: expected {expected!r}, got {actual!r}"
                )
    for dataset in DATASETS:
        values = {
            summary["pocket_centers_sha256"]
            for (current_dataset, _, _), summaries in grouped.items()
            if current_dataset == dataset
            for _, summary in summaries
        }
        if len(values) != 1:
            raise ValueError(f"{dataset}: pocket_centers_sha256 differs across cells")
        if values != {EXPECTED_POCKET_CENTERS_SHA256[dataset]}:
            raise ValueError(f"{dataset}: protocol pocket_centers_sha256 mismatch")

    guided_parameter_hashes: set[str] = set()
    for (dataset, condition, arm), summaries in grouped.items():
        del dataset, condition
        for path, summary in summaries:
            parameter_set = summary.get("guidance_parameter_set")
            if arm == "guided":
                if not isinstance(parameter_set, dict) or not parameter_set.get("sha256"):
                    raise ValueError(f"{path}: guided shard lacks guidance parameter hash")
                guided_parameter_hashes.add(str(parameter_set["sha256"]))
            elif parameter_set not in (None, {}):
                raise ValueError(f"{path}: unguided shard unexpectedly has guidance parameters")
    if len(guided_parameter_hashes) != 1:
        raise ValueError("guided parameter hash differs across cells")
    if guided_parameter_hashes != {EXPECTED_GUIDANCE_PARAMETER_SHA256}:
        raise ValueError("protocol guidance parameter hash mismatch")


def _validate_nested_prior_pairing(
    rows: dict[tuple[str, str, str], dict[str, dict[str, Any]]],
    dataset: str,
    eligible_ids: tuple[str, ...],
) -> dict[str, Any]:
    ledger = hashlib.sha256()
    ledger.update(b"EFFDOCK_SHARED_PRIOR_PAIRING_V1\0")
    for complex_id in eligible_ids:
        records = [
            rows[(dataset, condition, arm)][complex_id]
            for condition, _, _ in CONDITIONS
            for arm in ARM_SCALES
        ]
        seeds = {int(record["sampling_seed"]) for record in records}
        hashes = {str(record["prior_pool_sha256"]) for record in records}
        if len(seeds) != 1 or len(hashes) != 1:
            raise ValueError(
                f"{dataset}/{complex_id}: guided/unguided or budget cells "
                "do not share the exact 100-pose prior pool"
            )
        seed = next(iter(seeds))
        prior_hash = next(iter(hashes))
        ledger.update(f"{complex_id}\t{seed}\t{prior_hash}\n".encode())
    return {
        "verified": True,
        "contract": "same per-complex seed and EFFDOCK_SHARED_PRIOR_V1 hash in all six cells",
        "complexes": len(eligible_ids),
        "pairing_ledger_sha256": ledger.hexdigest(),
    }


def build_report(
    input_dir: Path,
    eligibility_path: Path,
    *,
    expected_shards: int = DEFAULT_EXPECTED_SHARDS,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
) -> dict[str, Any]:
    """Load every frozen cell, reject drift, and return paired aggregates."""
    if expected_shards < 1:
        raise ValueError("expected_shards must be >= 1")
    if bootstrap_resamples < 1:
        raise ValueError("bootstrap_resamples must be >= 1")
    eligible_by_dataset, manifest_summary = _load_eligibility(eligibility_path)

    grouped: dict[tuple[str, str, str], list[tuple[Path, dict[str, Any]]]] = {}
    matching_paths: list[Path] = []
    for path in sorted(input_dir.rglob("*.summary.json")):
        summary = json.loads(path.read_text())
        if not isinstance(summary, dict) or summary.get("protocol_id") != PROTOCOL_ID:
            continue
        _require_keys(summary, _REQUIRED_SUMMARY_KEYS, label=str(path))
        dataset = str(summary["dataset"])
        if dataset not in DATASETS:
            raise ValueError(f"{path}: unexpected dataset {dataset!r}")
        num_samples = int(summary["num_samples"])
        num_steps = int(summary["num_steps"])
        condition = _cell_key(num_samples, num_steps)
        arm = _arm_for_scale(summary["unified_guidance_scale"])
        expected_name = _expected_run_name(dataset, num_samples, num_steps, arm)
        if summary["run_name"] != expected_name:
            raise ValueError(
                f"{path}: run_name mismatch; expected {expected_name!r}, "
                f"got {summary['run_name']!r}"
            )
        if int(summary["model_pose_step_budget"]) != 1000:
            raise ValueError(f"{path}: model_pose_step_budget must equal 1000")
        if num_samples * num_steps != 1000:
            raise ValueError(f"{path}: num_samples * num_steps must equal 1000")
        grouped.setdefault((dataset, condition, arm), []).append((path, summary))
        matching_paths.append(path)
    if not matching_paths:
        raise FileNotFoundError(f"no {PROTOCOL_ID} summaries found in {input_dir}")

    expected_cells = {
        (dataset, condition, arm)
        for dataset in DATASETS
        for condition, _, _ in CONDITIONS
        for arm in ARM_SCALES
    }
    if set(grouped) != expected_cells:
        missing = sorted(expected_cells - set(grouped))
        extra = sorted(set(grouped) - expected_cells)
        raise ValueError(f"protocol cell mismatch; missing={missing}, extra={extra}")
    eligibility_sha256 = _sha256_file(eligibility_path)
    _validate_cross_cell_metadata(
        grouped,
        eligibility_sha256=eligibility_sha256,
    )

    rows: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = {}
    report: dict[str, Any] = {
        "protocol_id": PROTOCOL_ID,
        "status": "complete_strict_paired",
        "estimand": "frozen eligibility cohort; no survivor-only aggregation",
        "budget": {
            "model_pose_step_budget": 1000,
            "conditions": [
                {"key": key, "num_samples": samples, "num_steps": steps}
                for key, samples, steps in CONDITIONS
            ],
            "prior_pool_size": 100,
        },
        "bootstrap": {
            "method": "paired complex-ID bootstrap, percentile 95% CI",
            "seed": bootstrap_seed,
            "resamples": bootstrap_resamples,
        },
        "eligibility": {
            "path": str(eligibility_path),
            "sha256": eligibility_sha256,
            "datasets": manifest_summary,
        },
        "expected_shards_per_cell": expected_shards,
        "datasets": {},
    }

    for dataset in DATASETS:
        dataset_result: dict[str, Any] = {
            "eligibility_coverage": {
                **manifest_summary[dataset],
                "covered": len(eligible_by_dataset[dataset]),
                "failed": 0,
                "coverage_pct": 100.0,
            },
            "cells": {},
        }
        for condition, _, _ in CONDITIONS:
            dataset_result["cells"][condition] = {}
            for arm in ARM_SCALES:
                cell_rows, cell_result = _aggregate_cell(
                    grouped[(dataset, condition, arm)],
                    eligible_by_dataset[dataset],
                    arm=arm,
                    input_dir=input_dir,
                    expected_shards=expected_shards,
                    manifest_discovered=manifest_summary[dataset]["discovered"],
                )
                rows[(dataset, condition, arm)] = cell_rows
                dataset_result["cells"][condition][arm] = cell_result
            dataset_result["cells"][condition]["guided_vs_unguided"] = _paired_comparison(
                rows[(dataset, condition, "unguided")],
                rows[(dataset, condition, "guided")],
                eligible_by_dataset[dataset],
                baseline_label="unguided",
                comparison_label="guided",
                seed=bootstrap_seed,
                resamples=bootstrap_resamples,
            )

        dataset_result["prior_pairing"] = _validate_nested_prior_pairing(
            rows,
            dataset,
            eligible_by_dataset[dataset],
        )
        guided_common = set(eligible_by_dataset[dataset])
        for condition, _, _ in CONDITIONS:
            guided_common &= set(rows[(dataset, condition, "guided")])
        guided_common_ids = tuple(sorted(guided_common))
        if guided_common_ids != eligible_by_dataset[dataset]:
            raise ValueError(f"{dataset}: guided budget cells lack exact common-ID coverage")
        guided_comparison: dict[str, Any] = {
            "common_ids": len(guided_common_ids),
            "common_ids_sha256": _ids_sha256(guided_common_ids),
            "cell_metrics_on_common_ids": {},
            "pairwise_deltas": {},
        }
        for condition, _, _ in CONDITIONS:
            common_rows = [
                rows[(dataset, condition, "guided")][complex_id] for complex_id in guided_common_ids
            ]
            guided_comparison["cell_metrics_on_common_ids"][condition] = _target_metrics(
                summarize_rows(common_rows)
            )
        for (left, _, _), (right, _, _) in combinations(CONDITIONS, 2):
            key = f"{right}_minus_{left}"
            guided_comparison["pairwise_deltas"][key] = _paired_comparison(
                rows[(dataset, left, "guided")],
                rows[(dataset, right, "guided")],
                guided_common_ids,
                baseline_label=left,
                comparison_label=right,
                seed=bootstrap_seed,
                resamples=bootstrap_resamples,
            )
        dataset_result["guided_budget_comparison"] = guided_comparison
        report["datasets"][dataset] = dataset_result
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--eligibility", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--expected-shards",
        type=int,
        default=DEFAULT_EXPECTED_SHARDS,
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=DEFAULT_BOOTSTRAP_SEED,
    )
    parser.add_argument(
        "--bootstrap-resamples",
        type=int,
        default=DEFAULT_BOOTSTRAP_RESAMPLES,
    )
    args = parser.parse_args(argv)
    result = build_report(
        args.input_dir,
        args.eligibility,
        expected_shards=args.expected_shards,
        bootstrap_seed=args.bootstrap_seed,
        bootstrap_resamples=args.bootstrap_resamples,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
