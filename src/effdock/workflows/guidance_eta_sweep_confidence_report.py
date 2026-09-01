#!/usr/bin/env python3
"""Strict confidence-selected RMSD/PoseBusters report for the eta sweep."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from effdock.workflows import guidance_eta_sweep_confidence_identity
from effdock.workflows.evaluate import sorted_id_sha256
from effdock.workflows.guidance_budget_full_report import EXPECTED_DATASET_COUNTS
from effdock.workflows.guidance_budget_posebusters_report import (
    MODULE_CHECKS,
    POSEBUSTERS_CONFIG,
    POSEBUSTERS_VERSION,
    VALIDITY_CHECKS,
    _read_csv_rows,
    _resolve_cell_csv,
    _summarize_rows,
)
from effdock.workflows.guidance_budget_report import (
    DATASETS,
    DEFAULT_BOOTSTRAP_RESAMPLES,
    DEFAULT_BOOTSTRAP_SEED,
    DEFAULT_EXPECTED_SHARDS,
    EXPECTED_CHECKPOINT_SHA256,
    EXPECTED_CONFIG_SHA256,
    _ids_sha256,
    _paired_metric,
)
from effdock.workflows.guidance_coverage_audit import ID_HASH_CONTRACT
from effdock.workflows.guidance_eta_sweep_confidence_binding import (
    CONFIDENCE_CHECKPOINT_SHA256,
    PROTOCOL_ID,
    SELECTORS,
    build_binding,
)
from effdock.workflows.guidance_eta_sweep_report import (
    CONDITION,
    ETA_TAGS,
    ETA_VALUES,
    NUM_SAMPLES,
    NUM_STEPS,
    expected_run_name,
)
from effdock.workflows.guidance_eta_sweep_report import (
    PROTOCOL_ID as PARENT_PROTOCOL_ID,
)
from effdock.workflows.posebusters_report import file_sha256, load_rows

IDENTITY_AUDIT_CONTRACT = guidance_eta_sweep_confidence_identity.AUDIT_CONTRACT
PRIMARY_SELECTOR = "confidence"
DIAGNOSTIC_SELECTOR = "confidence_filter"
EXPECTED_IDENTITY_ROWS = sum(EXPECTED_DATASET_COUNTS.values()) * len(ETA_VALUES)
EXPECTED_PARENT_AUDIT_SHA256 = "dac7903488ccd36552a9bca134e37e633e3f07166d94f0389837012081ff3048"
EXPECTED_PARENT_IMPLEMENTATION_SHA256 = (
    "d726ddc4cb89b495f0495aa059faf9efdf33ee76c42c4b14e71356068935c0a5"
)
EXPECTED_PARENT_AUDIT_PROTOCOL_ID = "EFFDOCK-UNIFIED-GUIDANCE-BUDGET1000-FULL-V2"
EXPECTED_PARENT_AUDIT_SCHEMA = "effdock.guidance_coverage_audit.v2"

_REQUIRED_OFFICIAL_SUMMARY_KEYS = (
    "posebusters_version",
    "config",
    "selector",
    "input_hashes_verified",
    "num_input_hashes_verified",
    "require_complete_success",
    "num_discovered_total",
    "num_assigned",
    "num_success",
    "num_failed",
    "posebusters_valid_pct",
    "failures",
    "csv",
)


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{label} must be a SHA-256 hex digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{label} must be a SHA-256 hex digest") from exc
    return value


def load_frozen_parent_cohort_audit(path: Path) -> dict[str, dict[str, Any]]:
    """Read the historical cohort without comparing it to replay source bytes.

    The confidence replay intentionally adds output-only fields to ``evaluate.py``.
    Its source identity must therefore differ from the sampling identity.  The
    historical audit is admitted by its exact frozen file hash here; replay source
    and numerical equivalence are checked separately by the execution manifest and
    the versioned numerical replay-equivalence gate.
    """
    observed_sha256 = file_sha256(path)
    if observed_sha256 != EXPECTED_PARENT_AUDIT_SHA256:
        raise ValueError(
            f"{path}: frozen parent audit SHA-256 mismatch; "
            f"expected {EXPECTED_PARENT_AUDIT_SHA256}, got {observed_sha256}"
        )
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: parent audit must be a JSON object")
    exact = {
        "protocol_id": EXPECTED_PARENT_AUDIT_PROTOCOL_ID,
        "schema_version": EXPECTED_PARENT_AUDIT_SCHEMA,
        "receptor_policy": "geometry_only",
        "merged_only": True,
    }
    for key, expected in exact.items():
        if raw.get(key) != expected:
            raise ValueError(f"{path}: parent audit {key} must be {expected!r}")
    implementation = raw.get("implementation")
    if (
        not isinstance(implementation, dict)
        or implementation.get("sha256") != EXPECTED_PARENT_IMPLEMENTATION_SHA256
    ):
        raise ValueError(f"{path}: historical sampling implementation mismatch")
    datasets = raw.get("datasets")
    if not isinstance(datasets, dict) or set(datasets) != set(DATASETS):
        raise ValueError(f"{path}: parent audit must contain exactly {list(DATASETS)}")

    result: dict[str, dict[str, Any]] = {}
    for dataset in DATASETS:
        entry = datasets[dataset]
        if not isinstance(entry, dict):
            raise ValueError(f"{path}: datasets.{dataset} must be an object")
        ids = entry.get("ids")
        expected_count = EXPECTED_DATASET_COUNTS[dataset]
        expected_ids_sha256 = guidance_eta_sweep_confidence_identity.EXPECTED_BENCHMARK_IDENTITIES[
            dataset
        ]["ids_sha256"]
        if (
            not isinstance(ids, list)
            or len(ids) != expected_count
            or any(not isinstance(value, str) or not value for value in ids)
            or len(ids) != len(set(ids))
            or ids != sorted(ids)
        ):
            raise ValueError(f"{path}: datasets.{dataset} has invalid full-cohort IDs")
        ids_digest = sorted_id_sha256(ids)
        if ids_digest != expected_ids_sha256:
            raise ValueError(f"{path}: datasets.{dataset} ID hash mismatch")
        if (
            entry.get("complete") is not True
            or entry.get("discovered") != expected_count
            or entry.get("audited") != expected_count
            or entry.get("success") != expected_count
            or entry.get("failed") != 0
            or entry.get("failed_ids") != []
            or entry.get("ids_sha256") != expected_ids_sha256
            or entry.get("success_ids_sha256") != expected_ids_sha256
            or entry.get("ids_hash_contract") != ID_HASH_CONTRACT
        ):
            raise ValueError(f"{path}: datasets.{dataset} is not the frozen full cohort")
        result[dataset] = {
            "ids": tuple(ids),
            "ids_sha256": ids_digest,
            "source_path": str(path),
            "source_sha256": observed_sha256,
        }
    return result


def validate_identity_audit(path: Path, *, expected_shards: int) -> dict[str, Any]:
    """Require a completed full replay audit before opening external outcomes."""
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: identity audit must be a JSON object")
    expected_top_level = {
        "protocol_id": PROTOCOL_ID,
        "audit_contract": IDENTITY_AUDIT_CONTRACT,
        "parent_sampling_protocol_id": PARENT_PROTOCOL_ID,
        "mode": "full",
        "status": "passed",
        "parent_sentinels_verified": True,
        "candidate_ensemble_hashes_present": True,
        "selector_recomputed": True,
        "summary_contracts_verified": True,
    }
    for key, expected in expected_top_level.items():
        if raw.get(key) != expected:
            raise ValueError(
                f"{path}: identity audit {key} must be {expected!r}, got {raw.get(key)!r}"
            )

    coverage = raw.get("coverage")
    if not isinstance(coverage, dict):
        raise ValueError(f"{path}: identity audit coverage must be an object")
    expected_coverage = {
        "datasets": len(DATASETS),
        "cells": len(DATASETS) * len(ETA_VALUES),
        "shards": len(DATASETS) * len(ETA_VALUES) * expected_shards,
        "rows": EXPECTED_IDENTITY_ROWS,
    }
    for key, expected in expected_coverage.items():
        if coverage.get(key) != expected:
            raise ValueError(
                f"{path}: identity coverage {key} must be {expected}, got {coverage.get(key)!r}"
            )
    expected_per_dataset = {
        dataset: {
            "cells": len(ETA_VALUES),
            "shards": len(ETA_VALUES) * expected_shards,
            "rows": EXPECTED_DATASET_COUNTS[dataset] * len(ETA_VALUES),
            "ids_per_cell": EXPECTED_DATASET_COUNTS[dataset],
        }
        for dataset in DATASETS
    }
    if coverage.get("per_dataset") != expected_per_dataset:
        raise ValueError(f"{path}: identity per-dataset coverage mismatch")

    frozen = raw.get("frozen_hashes")
    expected_hashes = {
        "docking_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "config_sha256": EXPECTED_CONFIG_SHA256,
        "confidence_checkpoint_sha256": CONFIDENCE_CHECKPOINT_SHA256,
    }
    if not isinstance(frozen, dict):
        raise ValueError(f"{path}: identity frozen_hashes must be an object")
    for key, expected in expected_hashes.items():
        if frozen.get(key) != expected:
            raise ValueError(f"{path}: identity frozen hash mismatch for {key}")
    _require_sha256(
        raw.get("global_equivalence_ledger_sha256"),
        label=f"{path}.global_equivalence_ledger_sha256",
    )
    return raw


def revalidate_identity_audit(
    identity: dict[str, Any],
    *,
    sampling_dir: Path,
) -> None:
    """Rebuild the saved identity gate so post-audit mutations fail closed."""
    parent_value = identity.get("parent_dir")
    confidence_value = identity.get("confidence_dir")
    if not isinstance(parent_value, str) or not parent_value:
        raise ValueError("identity audit parent_dir must be a non-empty path")
    if not isinstance(confidence_value, str) or not confidence_value:
        raise ValueError("identity audit confidence_dir must be a non-empty path")
    parent_dir = Path(parent_value)
    confidence_dir = Path(confidence_value)
    if sampling_dir.resolve() != confidence_dir.resolve():
        raise ValueError(
            "sampling_dir must resolve to the confidence_dir bound by the identity audit"
        )
    rebuilt = guidance_eta_sweep_confidence_identity.build_identity_audit(
        parent_dir,
        confidence_dir,
        smoke=False,
    )
    if rebuilt != identity:
        raise ValueError(
            "saved identity audit differs from a fresh full replay audit; "
            "sampling inputs may have changed after audit"
        )


def _require_sampling_inventory(sampling_dir: Path, *, expected_shards: int) -> None:
    expected = {
        sampling_dir
        / (
            f"{expected_run_name(dataset, eta)}."
            f"shard-{shard:03d}-of-{expected_shards:03d}.summary.json"
        )
        for dataset in DATASETS
        for eta in ETA_VALUES
        for shard in range(expected_shards)
    }
    actual = set(sampling_dir.glob("*.summary.json")) if sampling_dir.is_dir() else set()
    if actual != expected:
        missing = sorted(str(path) for path in expected - actual)
        extra = sorted(str(path) for path in actual - expected)
        raise ValueError(
            f"confidence sampling inventory mismatch; missing={missing[:5]}, extra={extra[:5]}"
        )


def aggregate_official_cell(
    cell_dir: Path,
    ids: tuple[str, ...],
    *,
    run_name: str,
    dataset: str,
    eta_tag: str,
    selector: str,
    expected_shards: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any], str]:
    """Aggregate one complete official-PoseBusters cell without survivors."""
    if selector not in SELECTORS:
        raise ValueError(f"selector must be one of {SELECTORS}")
    if not cell_dir.is_dir():
        raise FileNotFoundError(f"missing official PoseBusters run directory: {cell_dir}")
    expected_summaries = {
        cell_dir / f"shard-{index:03d}-of-{expected_shards:03d}.summary.json"
        for index in range(expected_shards)
    }
    actual_summaries = set(cell_dir.glob("*.summary.json"))
    if actual_summaries != expected_summaries:
        missing = sorted(str(path) for path in expected_summaries - actual_summaries)
        extra = sorted(str(path) for path in actual_summaries - expected_summaries)
        raise ValueError(
            f"{run_name}/{selector}: expected exactly {expected_shards} shard summaries; "
            f"missing={missing}, extra={extra}"
        )

    rows_by_id: dict[str, dict[str, Any]] = {}
    rmsd_check: str | None = None
    summary_paths: list[str] = []
    for shard_index in range(expected_shards):
        tag = f"shard-{shard_index:03d}-of-{expected_shards:03d}"
        summary_path = cell_dir / f"{tag}.summary.json"
        summary = json.loads(summary_path.read_text())
        if not isinstance(summary, dict):
            raise ValueError(f"{summary_path}: summary must be a JSON object")
        missing_keys = [key for key in _REQUIRED_OFFICIAL_SUMMARY_KEYS if key not in summary]
        if missing_keys:
            raise ValueError(f"{summary_path}: missing required keys {missing_keys}")
        exact = {
            "posebusters_version": POSEBUSTERS_VERSION,
            "config": POSEBUSTERS_CONFIG,
            "selector": selector,
            "input_hashes_verified": True,
            "require_complete_success": True,
            "num_discovered_total": len(ids),
            "num_failed": 0,
        }
        for key, expected in exact.items():
            if summary.get(key) != expected:
                raise ValueError(f"{summary_path}: {key} must be {expected!r}")

        expected_shard_ids = tuple(ids[shard_index::expected_shards])
        expected_count = len(expected_shard_ids)
        for key in ("num_assigned", "num_success", "num_input_hashes_verified"):
            if int(summary.get(key, -1)) != expected_count:
                raise ValueError(f"{summary_path}: {key} differs from deterministic shard")
        if summary.get("failures") != []:
            raise ValueError(
                f"{run_name}/{selector}: official failures present; "
                "strict report rejects survivor-only aggregation"
            )

        expected_csv = cell_dir / f"{tag}.csv"
        csv_path = _resolve_cell_csv(summary_path, summary["csv"], expected_csv)
        shard_rows, current_rmsd_check = _read_csv_rows(
            csv_path,
            expected_rmsd_check=rmsd_check,
        )
        rmsd_check = current_rmsd_check
        shard_ids = [row["id"] for row in shard_rows]
        if shard_ids != list(expected_shard_ids):
            raise ValueError(f"{summary_path}: official IDs/order differ from cohort shard")
        if len(shard_ids) != len(set(shard_ids)):
            raise ValueError(f"{summary_path}: duplicate success IDs")
        for row in shard_rows:
            row["rmsd_check"] = current_rmsd_check
            if row["id"] in rows_by_id:
                raise ValueError(f"{run_name}/{selector}: duplicate ID {row['id']}")
            rows_by_id[row["id"]] = row
        actual_pct = sum(row["posebusters_valid"] for row in shard_rows) / len(shard_rows) * 100.0
        if summary["posebusters_valid_pct"] is None or not math.isclose(
            float(summary["posebusters_valid_pct"]),
            actual_pct,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"{summary_path}: posebusters_valid_pct does not match CSV")
        summary_paths.append(str(summary_path))

    if tuple(sorted(rows_by_id)) != tuple(ids):
        raise ValueError(f"{run_name}/{selector}: exact full-cohort ID coverage mismatch")
    if rmsd_check is None:
        raise ValueError(f"{run_name}/{selector}: RMSD check column was not observed")
    aggregate = {
        "run_name": run_name,
        "dataset": dataset,
        "condition": CONDITION,
        "eta_tag": eta_tag,
        "num_samples": NUM_SAMPLES,
        "num_steps": NUM_STEPS,
        "model_pose_step_budget": NUM_SAMPLES * NUM_STEPS,
        "selector": selector,
        "selector_role": "primary" if selector == PRIMARY_SELECTOR else "diagnostic",
        "eligible": len(ids),
        "success": len(rows_by_id),
        "failed": 0,
        "eligible_coverage_pct": 100.0,
        "eligible_ids_sha256": _ids_sha256(ids),
        **_summarize_rows(rows_by_id),
        "shard_summaries": summary_paths,
    }
    return rows_by_id, aggregate, rmsd_check


def join_sampling_outcomes(
    *,
    sampling_dir: Path,
    run_name: str,
    selector: str,
    ids: tuple[str, ...],
    official_rows: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Join selection RMSD and official validity on the exact ordered cohort."""
    if selector not in SELECTORS:
        raise ValueError(f"selector must be one of {SELECTORS}")
    sampling_rows = load_rows(sampling_dir, run_name)
    sampling_ids = [row["id"] for row in sampling_rows]
    if sampling_ids != list(ids) or tuple(sorted(official_rows)) != tuple(ids):
        raise ValueError(f"{run_name}/{selector}: sampling/official/cohort ID mismatch")
    outcomes: dict[str, dict[str, Any]] = {}
    rmsd_key = f"{selector}_rmsd"
    index_key = f"{selector}_index"
    for row in sampling_rows:
        complex_id = row["id"]
        try:
            rmsd = float(row[rmsd_key])
            index = int(row[index_key])
            num_samples = int(row["num_samples"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"{run_name}/{complex_id}: invalid confidence selection fields"
            ) from exc
        if not math.isfinite(rmsd) or rmsd < 0.0:
            raise ValueError(f"{run_name}/{complex_id}: {rmsd_key} must be finite/non-negative")
        if num_samples != NUM_SAMPLES or not 0 <= index < num_samples:
            raise ValueError(f"{run_name}/{complex_id}: invalid selected pose index")
        pb_valid = bool(official_rows[complex_id]["posebusters_valid"])
        rmsd_lt2 = rmsd < 2.0
        outcomes[complex_id] = {
            "posebusters_valid": pb_valid,
            "selected_rmsd": rmsd,
            "selected_rmsd_lt2": rmsd_lt2,
            "joint_selected_rmsd_lt2_and_posebusters_valid": rmsd_lt2 and pb_valid,
        }
    return outcomes


def summarize_outcomes(outcomes: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rows = list(outcomes.values())
    if not rows:
        raise ValueError("cannot summarize empty confidence outcomes")
    rmsd = np.asarray([row["selected_rmsd"] for row in rows], dtype=float)
    rmsd_successes = int((rmsd < 2.0).sum())
    pb_successes = sum(bool(row["posebusters_valid"]) for row in rows)
    joint_successes = sum(
        bool(row["joint_selected_rmsd_lt2_and_posebusters_valid"]) for row in rows
    )
    count = len(rows)
    return {
        "count": count,
        "selected_rmsd_lt2_count": rmsd_successes,
        "selected_rmsd_lt2_pct": rmsd_successes / count * 100.0,
        "selected_median_rmsd_A": float(np.median(rmsd)),
        "posebusters_valid_count": pb_successes,
        "posebusters_valid_pct": pb_successes / count * 100.0,
        "joint_selected_rmsd_lt2_and_posebusters_valid_count": joint_successes,
        "joint_selected_rmsd_lt2_and_posebusters_valid_pct": (joint_successes / count * 100.0),
    }


def paired_outcomes(
    baseline: dict[str, dict[str, Any]],
    comparison: dict[str, dict[str, Any]],
    ids: tuple[str, ...],
    *,
    baseline_label: str,
    comparison_label: str,
    seed: int,
    resamples: int,
) -> dict[str, Any]:
    """Compute paired deltas and transition counts on aligned complex IDs."""
    if tuple(baseline) != ids or tuple(comparison) != ids:
        raise ValueError("confidence pairing requires exact ordered full-cohort coverage")

    def values(source: dict[str, dict[str, Any]], key: str) -> np.ndarray:
        return np.asarray([source[complex_id][key] for complex_id in ids], dtype=float)

    metrics = {
        "selected_rmsd_lt2": _paired_metric(
            values(baseline, "selected_rmsd_lt2"),
            values(comparison, "selected_rmsd_lt2"),
            reducer="mean_pp",
            unit="percentage_points",
            seed=seed,
            resamples=resamples,
        ),
        "selected_median_rmsd": _paired_metric(
            values(baseline, "selected_rmsd"),
            values(comparison, "selected_rmsd"),
            reducer="median",
            unit="angstrom",
            seed=seed,
            resamples=resamples,
        ),
        "posebusters_valid": _paired_metric(
            values(baseline, "posebusters_valid"),
            values(comparison, "posebusters_valid"),
            reducer="mean_pp",
            unit="percentage_points",
            seed=seed,
            resamples=resamples,
        ),
        "joint_selected_rmsd_lt2_and_posebusters_valid": _paired_metric(
            values(baseline, "joint_selected_rmsd_lt2_and_posebusters_valid"),
            values(comparison, "joint_selected_rmsd_lt2_and_posebusters_valid"),
            reducer="mean_pp",
            unit="percentage_points",
            seed=seed,
            resamples=resamples,
        ),
    }
    transitions: dict[str, dict[str, int]] = {}
    for key in (
        "selected_rmsd_lt2",
        "posebusters_valid",
        "joint_selected_rmsd_lt2_and_posebusters_valid",
    ):
        left = values(baseline, key).astype(bool)
        right = values(comparison, key).astype(bool)
        transitions[key] = {
            "false_to_true": int((~left & right).sum()),
            "true_to_false": int((left & ~right).sum()),
            "both_true": int((left & right).sum()),
            "both_false": int((~left & ~right).sum()),
        }
    return {
        "direction": f"{comparison_label}_minus_{baseline_label}",
        "common_ids": len(ids),
        "common_ids_sha256": _ids_sha256(ids),
        "metrics": metrics,
        "transitions": transitions,
    }


def _require_bindings(
    *,
    input_dir: Path,
    sampling_dir: Path,
    expected_shards: int,
) -> None:
    expected_selector_dirs = {input_dir / selector for selector in SELECTORS}
    actual_selector_dirs = {
        path for path in input_dir.iterdir() if path.is_dir() and any(path.glob("*/*.summary.json"))
    }
    if actual_selector_dirs != expected_selector_dirs:
        raise ValueError("official confidence selector directory inventory mismatch")
    expected_runs = {expected_run_name(dataset, eta) for dataset in DATASETS for eta in ETA_VALUES}
    for selector in SELECTORS:
        selector_dir = input_dir / selector
        actual_runs = {
            path.name
            for path in selector_dir.iterdir()
            if path.is_dir() and any(path.glob("*.summary.json"))
        }
        if actual_runs != expected_runs:
            raise ValueError(
                f"{selector}: official run-cell mismatch; "
                f"missing={sorted(expected_runs - actual_runs)}, "
                f"extra={sorted(actual_runs - expected_runs)}"
            )
        for dataset in DATASETS:
            for eta in ETA_VALUES:
                run_name = expected_run_name(dataset, eta)
                cell_dir = selector_dir / run_name
                expected_bindings = {
                    cell_dir / f"shard-{index:03d}-of-{expected_shards:03d}.binding.json"
                    for index in range(expected_shards)
                }
                actual_bindings = set(cell_dir.glob("*.binding.json"))
                if actual_bindings != expected_bindings:
                    raise ValueError(f"{run_name}/{selector}: binding inventory mismatch")
                expected_csvs = {
                    cell_dir / f"shard-{index:03d}-of-{expected_shards:03d}.csv"
                    for index in range(expected_shards)
                }
                actual_csvs = set(cell_dir.glob("*.csv"))
                if actual_csvs != expected_csvs:
                    raise ValueError(f"{run_name}/{selector}: official CSV inventory mismatch")
                for shard_index in range(expected_shards):
                    tag = f"shard-{shard_index:03d}-of-{expected_shards:03d}"
                    binding_path = cell_dir / f"{tag}.binding.json"
                    if not binding_path.is_file():
                        raise FileNotFoundError(f"missing confidence binding: {binding_path}")
                    observed = json.loads(binding_path.read_text())
                    expected = build_binding(
                        sampling_dir=sampling_dir,
                        official_dir=cell_dir,
                        run_name=run_name,
                        protocol_id=PROTOCOL_ID,
                        dataset=dataset,
                        eta=eta,
                        selector=selector,
                        shard_index=shard_index,
                        num_shards=expected_shards,
                    )
                    if observed != expected:
                        raise ValueError(f"{binding_path}: official/sampling binding mismatch")


def build_report(
    input_dir: Path,
    sampling_dir: Path,
    cohort_audit: Path,
    identity_audit: Path,
    *,
    expected_shards: int = DEFAULT_EXPECTED_SHARDS,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
) -> dict[str, Any]:
    """Build one strict two-selector report after all integrity gates pass."""
    if expected_shards != DEFAULT_EXPECTED_SHARDS:
        raise ValueError(
            f"full confidence protocol requires exactly {DEFAULT_EXPECTED_SHARDS} shards"
        )
    if bootstrap_resamples < 1:
        raise ValueError("bootstrap_resamples must be >= 1")
    identity = validate_identity_audit(identity_audit, expected_shards=expected_shards)
    revalidate_identity_audit(identity, sampling_dir=sampling_dir)
    audits = load_frozen_parent_cohort_audit(cohort_audit)
    _require_sampling_inventory(sampling_dir, expected_shards=expected_shards)
    _require_bindings(
        input_dir=input_dir,
        sampling_dir=sampling_dir,
        expected_shards=expected_shards,
    )

    outcomes: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = {}
    report: dict[str, Any] = {
        "protocol_id": PROTOCOL_ID,
        "parent_sampling_protocol_id": PARENT_PROTOCOL_ID,
        "status": "complete_strict_full_cohort_paired_confidence_posebusters_eta_sweep",
        "claim_boundary": (
            "post-hoc paired descriptive external benchmark; the primary pure-confidence "
            "selector and diagnostic frozen cluster-free filter are both reported, with "
            "no automatic eta or selector admission"
        ),
        "selectors": {
            PRIMARY_SELECTOR: {
                "role": "primary",
                "definition": "minimum predicted-RMSD confidence head",
                "cluster_dependent": False,
            },
            DIAGNOSTIC_SELECTOR: {
                "role": "diagnostic",
                "definition": "frozen confidence_filter_v1 cluster-free guard",
                "cluster_dependent": False,
                "production_admission": "not admitted by the frozen validation gate",
            },
        },
        "confidence_checkpoint_sha256": CONFIDENCE_CHECKPOINT_SHA256,
        "identity_audit": {
            "path": str(identity_audit),
            "sha256": file_sha256(identity_audit),
            "audit_contract": IDENTITY_AUDIT_CONTRACT,
            "status": identity["status"],
            "mode": identity["mode"],
            "parent_sentinels_verified": identity["parent_sentinels_verified"],
            "candidate_ensemble_hashes_present": identity["candidate_ensemble_hashes_present"],
            "selector_recomputed": identity["selector_recomputed"],
            "summary_contracts_verified": identity["summary_contracts_verified"],
            "global_equivalence_ledger_sha256": identity["global_equivalence_ledger_sha256"],
            "coverage": identity["coverage"],
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
        "official_inventory": {
            "selectors": len(SELECTORS),
            "run_cells_per_selector": len(DATASETS) * len(ETA_VALUES),
            "shards_per_cell": expected_shards,
            "total_shard_tasks": (
                len(SELECTORS) * len(DATASETS) * len(ETA_VALUES) * expected_shards
            ),
        },
        "posebusters": {
            "version": POSEBUSTERS_VERSION,
            "config": POSEBUSTERS_CONFIG,
            "pass_all_definition": "all 27 non-RMSD redock checks",
            "validity_checks": list(VALIDITY_CHECKS),
            "module_checks": {key: list(value) for key, value in MODULE_CHECKS.items()},
        },
        "bootstrap": {
            "method": "paired complex-ID bootstrap, percentile 95% CI",
            "seed": bootstrap_seed,
            "resamples": bootstrap_resamples,
        },
        "datasets": {},
    }

    rmsd_checks: set[str] = set()
    for dataset in DATASETS:
        audit = audits[dataset]
        ids = audit["ids"]
        if len(ids) != EXPECTED_DATASET_COUNTS[dataset]:
            raise ValueError(f"{dataset}: audit is not the frozen full cohort")
        cells: dict[str, Any] = {}
        for eta, tag in zip(ETA_VALUES, ETA_TAGS, strict=True):
            run_name = expected_run_name(dataset, eta)
            selector_cells: dict[str, Any] = {}
            for selector in SELECTORS:
                official_rows, aggregate, rmsd_check = aggregate_official_cell(
                    input_dir / selector / run_name,
                    ids,
                    run_name=run_name,
                    dataset=dataset,
                    eta_tag=tag,
                    selector=selector,
                    expected_shards=expected_shards,
                )
                current = join_sampling_outcomes(
                    sampling_dir=sampling_dir,
                    run_name=run_name,
                    selector=selector,
                    ids=ids,
                    official_rows=official_rows,
                )
                outcomes[(dataset, tag, selector)] = current
                aggregate["eta"] = eta
                aggregate["metrics"] = summarize_outcomes(current)
                aggregate["eligible_ids_sha256"] = sorted_id_sha256(list(ids))
                aggregate["ids_hash_contract"] = ID_HASH_CONTRACT
                selector_cells[selector] = aggregate
                rmsd_checks.add(rmsd_check)
            cells[tag] = {
                "eta": eta,
                "eta_tag": tag,
                "selectors": selector_cells,
                "confidence_filter_minus_confidence": paired_outcomes(
                    outcomes[(dataset, tag, PRIMARY_SELECTOR)],
                    outcomes[(dataset, tag, DIAGNOSTIC_SELECTOR)],
                    ids,
                    baseline_label=PRIMARY_SELECTOR,
                    comparison_label=DIAGNOSTIC_SELECTOR,
                    seed=bootstrap_seed,
                    resamples=bootstrap_resamples,
                ),
            }

        eta_vs_eta0 = {
            selector: {
                tag: paired_outcomes(
                    outcomes[(dataset, ETA_TAGS[0], selector)],
                    outcomes[(dataset, tag, selector)],
                    ids,
                    baseline_label=ETA_TAGS[0],
                    comparison_label=tag,
                    seed=bootstrap_seed,
                    resamples=bootstrap_resamples,
                )
                for tag in ETA_TAGS
            }
            for selector in SELECTORS
        }
        report["datasets"][dataset] = {
            "coverage": {
                "count": len(ids),
                "expected": EXPECTED_DATASET_COUNTS[dataset],
                "ids_sha256": audit["ids_sha256"],
                "ids_hash_contract": ID_HASH_CONTRACT,
                "audit_path": audit["source_path"],
                "audit_sha256": audit["source_sha256"],
            },
            "cells": cells,
            "eta_vs_eta0": eta_vs_eta0,
        }

    if len(rmsd_checks) != 1:
        raise ValueError(f"RMSD check name differs across cells: {sorted(rmsd_checks)}")
    report["posebusters"]["rmsd_check_excluded_from_validity"] = next(iter(rmsd_checks))
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--sampling-dir", type=Path, required=True)
    parser.add_argument("--cohort-audit", type=Path, required=True)
    parser.add_argument("--identity-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-shards", type=int, default=DEFAULT_EXPECTED_SHARDS)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument("--bootstrap-resamples", type=int, default=DEFAULT_BOOTSTRAP_RESAMPLES)
    args = parser.parse_args(argv)
    result = build_report(
        args.input_dir,
        args.sampling_dir,
        args.cohort_audit,
        args.identity_audit,
        expected_shards=args.expected_shards,
        bootstrap_seed=args.bootstrap_seed,
        bootstrap_resamples=args.bootstrap_resamples,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
