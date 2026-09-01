#!/usr/bin/env python3
"""Strict official-PoseBusters report for normalized direct guidance drift."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from effdock.workflows.evaluate import sorted_id_sha256
from effdock.workflows.guidance_budget_full_posebusters_report import _paired_pass_all
from effdock.workflows.guidance_budget_full_report import load_full_cohort_audits
from effdock.workflows.guidance_budget_posebusters_report import (
    EXPECTED_SELECTOR,
    MODULE_CHECKS,
    POSEBUSTERS_CONFIG,
    POSEBUSTERS_VERSION,
    VALIDITY_CHECKS,
    _aggregate_cell,
)
from effdock.workflows.guidance_budget_report import (
    CONDITIONS,
    DATASETS,
    DEFAULT_BOOTSTRAP_RESAMPLES,
    DEFAULT_BOOTSTRAP_SEED,
    DEFAULT_EXPECTED_SHARDS,
)
from effdock.workflows.guidance_coverage_audit import ID_HASH_CONTRACT
from effdock.workflows.guidance_direct_drift_report import (
    ARMS,
    PROTOCOL_ID,
    expected_run_name,
)


def _require_input_hash_verification(
    input_dir: Path,
    expected_runs: set[str],
    expected_shards: int,
) -> None:
    """Reject official results not bound to sampling-time file hashes."""
    for run_name in sorted(expected_runs):
        for shard_index in range(expected_shards):
            summary_path = (
                input_dir
                / run_name
                / f"shard-{shard_index:03d}-of-{expected_shards:03d}.summary.json"
            )
            if not summary_path.is_file():
                raise FileNotFoundError(f"missing official summary: {summary_path}")
            summary = json.loads(summary_path.read_text())
            if summary.get("input_hashes_verified") is not True:
                raise ValueError(f"{summary_path}: sampling-time input hashes not verified")
            if int(summary.get("num_input_hashes_verified", -1)) != int(
                summary.get("num_assigned", -2)
            ):
                raise ValueError(f"{summary_path}: input-hash verification count mismatch")


def build_report(
    input_dir: Path,
    cohort_audit: Path,
    *,
    expected_shards: int = DEFAULT_EXPECTED_SHARDS,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
) -> dict[str, Any]:
    audits = load_full_cohort_audits(cohort_audit)
    expected_runs = {
        expected_run_name(dataset, samples, steps, arm)
        for dataset in DATASETS
        for _, samples, steps in CONDITIONS
        for arm in ARMS
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
            f"official run-cell mismatch; missing={sorted(expected_runs - actual_runs)}, "
            f"extra={sorted(actual_runs - expected_runs)}"
        )
    _require_input_hash_verification(input_dir, expected_runs, expected_shards)

    report: dict[str, Any] = {
        "protocol_id": PROTOCOL_ID,
        "status": "complete_strict_full_cohort_paired_official_posebusters",
        "claim_boundary": (
            "paired descriptive reference-pocket redocking; no automatic performance decision"
        ),
        "posebusters": {
            "version": POSEBUSTERS_VERSION,
            "config": POSEBUSTERS_CONFIG,
            "selector": EXPECTED_SELECTOR,
            "pass_all_definition": "all 27 non-RMSD redock checks",
            "validity_checks": list(VALIDITY_CHECKS),
            "module_checks": {key: list(value) for key, value in MODULE_CHECKS.items()},
        },
        "bootstrap": {
            "method": "paired complex-ID bootstrap, percentile 95% CI",
            "seed": bootstrap_seed,
            "resamples": bootstrap_resamples,
        },
        "expected_shards_per_cell": expected_shards,
        "datasets": {},
    }
    for dataset in DATASETS:
        ids = audits[dataset]["ids"]
        dataset_result: dict[str, Any] = {
            "coverage": {
                "count": len(ids),
                "ids_sha256": audits[dataset]["ids_sha256"],
                "ids_hash_contract": ID_HASH_CONTRACT,
                "audit_path": audits[dataset]["source_path"],
                "audit_sha256": audits[dataset]["source_sha256"],
            },
            "cells": {},
        }
        for condition, num_samples, num_steps in CONDITIONS:
            cell: dict[str, Any] = {}
            cell_rows: dict[str, dict[str, dict[str, Any]]] = {}
            rmsd_checks: set[str] = set()
            for arm in ARMS:
                run_name = expected_run_name(dataset, num_samples, num_steps, arm)
                rows, aggregate, rmsd_check = _aggregate_cell(
                    input_dir / run_name,
                    ids,
                    run_name=run_name,
                    dataset=dataset,
                    condition=condition,
                    arm=arm,
                    num_samples=num_samples,
                    num_steps=num_steps,
                    expected_shards=expected_shards,
                )
                cell_rows[arm] = rows
                aggregate["eligible_ids_sha256"] = sorted_id_sha256(list(ids))
                aggregate["ids_hash_contract"] = ID_HASH_CONTRACT
                cell[arm] = aggregate
                rmsd_checks.add(rmsd_check)
            if len(rmsd_checks) != 1:
                raise ValueError(f"{dataset}/{condition}: inconsistent RMSD check names")
            cell["direct_minus_unguided"] = _paired_pass_all(
                cell_rows["unguided"],
                cell_rows["direct"],
                ids,
                baseline_label="unguided",
                comparison_label="direct",
                seed=bootstrap_seed,
                resamples=bootstrap_resamples,
            )
            cell["rmsd_check_excluded_from_validity"] = next(iter(rmsd_checks))
            dataset_result["cells"][condition] = cell
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
