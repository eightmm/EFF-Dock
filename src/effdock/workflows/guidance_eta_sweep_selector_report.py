#!/usr/bin/env python3
"""Strict first/vina selected-pose report for the frozen eta-sweep extension."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from effdock.workflows.evaluate import sorted_id_sha256
from effdock.workflows.guidance_budget_full_report import load_full_cohort_audits
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
    _ids_sha256,
    _paired_metric,
)
from effdock.workflows.guidance_coverage_audit import ID_HASH_CONTRACT
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
from effdock.workflows.guidance_eta_sweep_selector_binding import (
    PROTOCOL_ID,
    SELECTORS,
    build_binding,
)
from effdock.workflows.posebusters_report import load_rows

_REQUIRED_SUMMARY_KEYS = (
    "posebusters_version",
    "config",
    "selector",
    "num_discovered_total",
    "num_assigned",
    "num_success",
    "num_failed",
    "posebusters_valid_pct",
    "failures",
    "csv",
)


def aggregate_selector_cell(
    cell_dir: Path,
    eligible_ids: tuple[str, ...],
    *,
    run_name: str,
    dataset: str,
    eta_tag: str,
    selector: str,
    expected_shards: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any], str]:
    """Aggregate one strict full-cohort cell without the oracle-only helper."""
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
        missing_keys = [key for key in _REQUIRED_SUMMARY_KEYS if key not in summary]
        if missing_keys:
            raise ValueError(f"{summary_path}: missing required keys {missing_keys}")
        if summary["posebusters_version"] != POSEBUSTERS_VERSION:
            raise ValueError(f"{summary_path}: PoseBusters version mismatch")
        if summary["config"] != POSEBUSTERS_CONFIG:
            raise ValueError(f"{summary_path}: config must be {POSEBUSTERS_CONFIG!r}")
        if summary["selector"] != selector:
            raise ValueError(f"{summary_path}: selector must be {selector!r}")
        if int(summary["num_discovered_total"]) != len(eligible_ids):
            raise ValueError(f"{summary_path}: discovered count differs from full cohort")

        expected_shard_ids = tuple(eligible_ids[shard_index::expected_shards])
        if int(summary["num_assigned"]) != len(expected_shard_ids):
            raise ValueError(f"{summary_path}: assigned count differs from deterministic shard")
        failures = summary["failures"]
        if not isinstance(failures, list):
            raise ValueError(f"{summary_path}: failures must be a list")
        if int(summary["num_failed"]) != len(failures):
            raise ValueError(f"{summary_path}: num_failed does not match failures")
        if failures:
            raise ValueError(
                f"{run_name}/{selector}: {len(failures)} PoseBusters failures; "
                "strict report rejects survivor-only aggregation"
            )
        if int(summary["num_success"]) != len(expected_shard_ids):
            raise ValueError(f"{summary_path}: strict shard requires complete success")

        expected_csv = cell_dir / f"{tag}.csv"
        csv_path = _resolve_cell_csv(summary_path, summary["csv"], expected_csv)
        shard_rows, current_rmsd_check = _read_csv_rows(
            csv_path,
            expected_rmsd_check=rmsd_check,
        )
        rmsd_check = current_rmsd_check
        shard_ids = [row["id"] for row in shard_rows]
        if shard_ids != list(expected_shard_ids):
            raise ValueError(f"{summary_path}: official IDs/order differ from deterministic shard")
        if len(shard_ids) != len(set(shard_ids)):
            raise ValueError(f"{summary_path}: duplicate success IDs")
        for row in shard_rows:
            row["rmsd_check"] = current_rmsd_check
            if row["id"] in rows_by_id:
                raise ValueError(f"{run_name}/{selector}: duplicate ID {row['id']}")
            rows_by_id[row["id"]] = row
        actual_pct = (
            sum(row["posebusters_valid"] for row in shard_rows) / len(shard_rows) * 100.0
        )
        if summary["posebusters_valid_pct"] is None or not math.isclose(
            float(summary["posebusters_valid_pct"]),
            actual_pct,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"{summary_path}: posebusters_valid_pct does not match CSV")
        summary_paths.append(str(summary_path))

    if set(rows_by_id) != set(eligible_ids):
        missing = sorted(set(eligible_ids) - set(rows_by_id))
        outside = sorted(set(rows_by_id) - set(eligible_ids))
        raise ValueError(
            f"{run_name}/{selector}: full-cohort ID mismatch; "
            f"missing={missing[:5]}, outside={outside[:5]}"
        )
    if rmsd_check is None:
        raise ValueError(f"{run_name}/{selector}: RMSD check column was not observed")

    aggregate = {
        "run_name": run_name,
        "dataset": dataset,
        "condition": CONDITION,
        "arm": eta_tag,
        "num_samples": NUM_SAMPLES,
        "num_steps": NUM_STEPS,
        "model_pose_step_budget": NUM_SAMPLES * NUM_STEPS,
        "selector": selector,
        "eligible": len(eligible_ids),
        "success": len(rows_by_id),
        "failed": 0,
        "eligible_coverage_pct": 100.0,
        "eligible_ids_sha256": _ids_sha256(eligible_ids),
        **_summarize_rows(rows_by_id),
        "shard_summaries": summary_paths,
    }
    return rows_by_id, aggregate, rmsd_check


def _join_sampling_outcomes(
    *,
    sampling_dir: Path,
    run_name: str,
    selector: str,
    ids: tuple[str, ...],
    official_rows: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    sampling_rows = load_rows(sampling_dir, run_name)
    sampling_by_id = {row["id"]: row for row in sampling_rows}
    if set(sampling_by_id) != set(ids) or set(official_rows) != set(ids):
        raise ValueError(f"{run_name}/{selector}: sampling/official/full-cohort ID mismatch")
    outcomes: dict[str, dict[str, Any]] = {}
    key = f"{selector}_rmsd"
    for complex_id in ids:
        try:
            rmsd = float(sampling_by_id[complex_id][key])
        except (KeyError, ValueError) as exc:
            raise ValueError(f"{run_name}/{complex_id}: invalid {key}") from exc
        if not math.isfinite(rmsd) or rmsd < 0.0:
            raise ValueError(f"{run_name}/{complex_id}: {key} must be finite and non-negative")
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
    count = len(rows)
    if not count:
        raise ValueError("cannot summarize empty selector outcomes")
    rmsd = np.asarray([row["selected_rmsd"] for row in rows], dtype=float)
    return {
        "count": count,
        "selected_rmsd_lt2_count": sum(row["selected_rmsd_lt2"] for row in rows),
        "selected_rmsd_lt2_pct": float((rmsd < 2.0).mean() * 100.0),
        "selected_median_rmsd_A": float(np.median(rmsd)),
        "posebusters_valid_count": sum(row["posebusters_valid"] for row in rows),
        "posebusters_valid_pct": float(
            np.mean([row["posebusters_valid"] for row in rows]) * 100.0
        ),
        "joint_selected_rmsd_lt2_and_posebusters_valid_count": sum(
            row["joint_selected_rmsd_lt2_and_posebusters_valid"] for row in rows
        ),
        "joint_selected_rmsd_lt2_and_posebusters_valid_pct": float(
            np.mean(
                [
                    row["joint_selected_rmsd_lt2_and_posebusters_valid"]
                    for row in rows
                ]
            )
            * 100.0
        ),
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
    if set(baseline) != set(ids) or set(comparison) != set(ids):
        raise ValueError("selector pairing requires exact full-cohort ID coverage")

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
    selector: str,
    expected_shards: int,
) -> None:
    for dataset in DATASETS:
        for eta in ETA_VALUES:
            run_name = expected_run_name(dataset, eta)
            cell_dir = input_dir / run_name
            for shard_index in range(expected_shards):
                tag = f"shard-{shard_index:03d}-of-{expected_shards:03d}"
                binding_path = cell_dir / f"{tag}.binding.json"
                if not binding_path.is_file():
                    raise FileNotFoundError(f"missing selector binding: {binding_path}")
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
    *,
    selector: str,
    expected_shards: int = DEFAULT_EXPECTED_SHARDS,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
) -> dict[str, Any]:
    if selector not in SELECTORS:
        raise ValueError(f"selector must be one of {SELECTORS}")
    if expected_shards < 1 or bootstrap_resamples < 1:
        raise ValueError("expected_shards and bootstrap_resamples must be >= 1")
    audits = load_full_cohort_audits(cohort_audit)
    expected_runs = {
        expected_run_name(dataset, eta) for dataset in DATASETS for eta in ETA_VALUES
    }
    actual_runs = (
        {
            path.name
            for path in input_dir.iterdir()
            if path.is_dir() and any(path.glob("*.summary.json"))
        }
        if input_dir.is_dir()
        else set()
    )
    if actual_runs != expected_runs:
        raise ValueError(
            "selector official run-cell mismatch; "
            f"missing={sorted(expected_runs - actual_runs)}, "
            f"extra={sorted(actual_runs - expected_runs)}"
        )
    expected_total = len(DATASETS) * len(ETA_VALUES) * expected_shards
    if len(list(input_dir.glob("*/*.summary.json"))) != expected_total:
        raise ValueError(f"selector report requires exactly {expected_total} shard summaries")
    _require_bindings(
        input_dir=input_dir,
        sampling_dir=sampling_dir,
        selector=selector,
        expected_shards=expected_shards,
    )

    report: dict[str, Any] = {
        "protocol_id": PROTOCOL_ID,
        "parent_sampling_protocol_id": PARENT_PROTOCOL_ID,
        "status": "complete_strict_full_cohort_paired_selector_posebusters_eta_sweep",
        "claim_boundary": (
            "post-hoc paired descriptive extension; no eta or production selector selection"
        ),
        "selector": selector,
        "condition": {
            "name": CONDITION,
            "num_samples": NUM_SAMPLES,
            "num_steps": NUM_STEPS,
            "model_pose_step_budget": NUM_SAMPLES * NUM_STEPS,
        },
        "eta_grid": [
            {"eta": eta, "tag": tag}
            for eta, tag in zip(ETA_VALUES, ETA_TAGS, strict=True)
        ],
        "official_inventory": {
            "run_cells": len(DATASETS) * len(ETA_VALUES),
            "shards_per_cell": expected_shards,
            "total_shard_tasks": expected_total,
        },
        "posebusters": {
            "version": POSEBUSTERS_VERSION,
            "config": POSEBUSTERS_CONFIG,
            "selector": selector,
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
        outcomes_by_eta: dict[str, dict[str, dict[str, Any]]] = {}
        cells: dict[str, Any] = {}
        for eta, eta_tag in zip(ETA_VALUES, ETA_TAGS, strict=True):
            run_name = expected_run_name(dataset, eta)
            official_rows, aggregate, rmsd_check = aggregate_selector_cell(
                input_dir / run_name,
                ids,
                run_name=run_name,
                dataset=dataset,
                eta_tag=eta_tag,
                selector=selector,
                expected_shards=expected_shards,
            )
            outcomes = _join_sampling_outcomes(
                sampling_dir=sampling_dir,
                run_name=run_name,
                selector=selector,
                ids=ids,
                official_rows=official_rows,
            )
            aggregate["eta"] = eta
            aggregate["eta_tag"] = eta_tag
            aggregate["selected_pose_metrics"] = summarize_outcomes(outcomes)
            aggregate["eligible_ids_sha256"] = sorted_id_sha256(list(ids))
            aggregate["ids_hash_contract"] = ID_HASH_CONTRACT
            outcomes_by_eta[eta_tag] = outcomes
            cells[eta_tag] = aggregate
            rmsd_checks.add(rmsd_check)

        baseline = outcomes_by_eta[ETA_TAGS[0]]
        report["datasets"][dataset] = {
            "coverage": {
                "count": len(ids),
                "ids_sha256": audit["ids_sha256"],
                "ids_hash_contract": ID_HASH_CONTRACT,
                "audit_path": audit["source_path"],
                "audit_sha256": audit["source_sha256"],
            },
            "cells": cells,
            "eta_vs_eta0": {
                tag: paired_outcomes(
                    baseline,
                    outcomes_by_eta[tag],
                    ids,
                    baseline_label=ETA_TAGS[0],
                    comparison_label=tag,
                    seed=bootstrap_seed,
                    resamples=bootstrap_resamples,
                )
                for tag in ETA_TAGS
            },
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
    parser.add_argument("--selector", choices=SELECTORS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-shards", type=int, default=DEFAULT_EXPECTED_SHARDS)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument("--bootstrap-resamples", type=int, default=DEFAULT_BOOTSTRAP_RESAMPLES)
    args = parser.parse_args(argv)
    result = build_report(
        args.input_dir,
        args.sampling_dir,
        args.cohort_audit,
        selector=args.selector,
        expected_shards=args.expected_shards,
        bootstrap_seed=args.bootstrap_seed,
        bootstrap_resamples=args.bootstrap_resamples,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
