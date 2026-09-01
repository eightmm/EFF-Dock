#!/usr/bin/env python3
"""Strict official-PoseBusters report for the descriptive eta sweep."""

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
    DATASETS,
    DEFAULT_BOOTSTRAP_RESAMPLES,
    DEFAULT_BOOTSTRAP_SEED,
    DEFAULT_EXPECTED_SHARDS,
)
from effdock.workflows.guidance_coverage_audit import ID_HASH_CONTRACT
from effdock.workflows.guidance_eta_sweep_official_binding import build_binding
from effdock.workflows.guidance_eta_sweep_report import (
    CONDITION,
    ETA_TAGS,
    ETA_VALUES,
    NUM_SAMPLES,
    NUM_STEPS,
    PROTOCOL_ID,
    expected_run_name,
)


def _require_input_hash_verification(
    input_dir: Path,
    sampling_dir: Path,
    expected_runs: set[str],
    expected_shards: int,
) -> None:
    run_identity = {
        expected_run_name(dataset, eta): (dataset, eta)
        for dataset in DATASETS
        for eta in ETA_VALUES
    }
    if set(run_identity) != expected_runs:
        raise ValueError("official expected-run identity is inconsistent")
    for run_name in sorted(expected_runs):
        dataset, eta = run_identity[run_name]
        for shard_index in range(expected_shards):
            summary_path = (
                input_dir
                / run_name
                / f"shard-{shard_index:03d}-of-{expected_shards:03d}.summary.json"
            )
            if not summary_path.is_file():
                raise FileNotFoundError(f"missing official summary: {summary_path}")
            summary = json.loads(summary_path.read_text())
            if not isinstance(summary, dict):
                raise ValueError(f"{summary_path}: summary must be a JSON object")
            if summary.get("input_hashes_verified") is not True:
                raise ValueError(f"{summary_path}: sampling-time input hashes not verified")
            if int(summary.get("num_input_hashes_verified", -1)) != int(
                summary.get("num_assigned", -2)
            ):
                raise ValueError(f"{summary_path}: input-hash verification count mismatch")
            tag = f"shard-{shard_index:03d}-of-{expected_shards:03d}"
            binding_path = input_dir / run_name / f"{tag}.binding.json"
            if not binding_path.is_file():
                raise FileNotFoundError(f"missing official binding: {binding_path}")
            observed = json.loads(binding_path.read_text())
            expected = build_binding(
                sampling_dir=sampling_dir,
                official_dir=input_dir / run_name,
                run_name=run_name,
                protocol_id=PROTOCOL_ID,
                dataset=dataset,
                eta=eta,
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
    expected_shards: int = DEFAULT_EXPECTED_SHARDS,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
) -> dict[str, Any]:
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
            "official eta-sweep run-cell mismatch; "
            f"missing={sorted(expected_runs - actual_runs)}, "
            f"extra={sorted(actual_runs - expected_runs)}"
        )
    actual_summaries = list(input_dir.glob("*/*.summary.json"))
    expected_total = len(DATASETS) * len(ETA_VALUES) * expected_shards
    if len(actual_summaries) != expected_total:
        raise ValueError(
            f"official eta-sweep requires exactly {expected_total} shard summaries, "
            f"got {len(actual_summaries)}"
        )
    _require_input_hash_verification(
        input_dir,
        sampling_dir,
        expected_runs,
        expected_shards,
    )

    report: dict[str, Any] = {
        "protocol_id": PROTOCOL_ID,
        "status": "complete_strict_full_cohort_paired_official_posebusters_eta_sweep",
        "claim_boundary": (
            "paired descriptive Astex/PoseBusters reference-pocket redocking; all eta "
            "values are reported without automatic selection or production admission"
        ),
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
        "datasets": {},
    }
    rmsd_checks: set[str] = set()
    for dataset in DATASETS:
        audit = audits[dataset]
        ids = audit["ids"]
        rows_by_eta: dict[str, dict[str, dict[str, Any]]] = {}
        cells: dict[str, Any] = {}
        for eta, tag in zip(ETA_VALUES, ETA_TAGS, strict=True):
            run_name = expected_run_name(dataset, eta)
            rows, aggregate, rmsd_check = _aggregate_cell(
                input_dir / run_name,
                ids,
                run_name=run_name,
                dataset=dataset,
                condition=CONDITION,
                arm=tag,
                num_samples=NUM_SAMPLES,
                num_steps=NUM_STEPS,
                expected_shards=expected_shards,
            )
            rows_by_eta[tag] = rows
            aggregate["eta"] = eta
            aggregate["eta_tag"] = tag
            aggregate["eligible_ids_sha256"] = sorted_id_sha256(list(ids))
            aggregate["ids_hash_contract"] = ID_HASH_CONTRACT
            cells[tag] = aggregate
            rmsd_checks.add(rmsd_check)

        baseline = rows_by_eta[ETA_TAGS[0]]
        eta_vs_eta0 = {
            tag: _paired_pass_all(
                baseline,
                rows_by_eta[tag],
                ids,
                baseline_label=ETA_TAGS[0],
                comparison_label=tag,
                seed=bootstrap_seed,
                resamples=bootstrap_resamples,
            )
            for tag in ETA_TAGS
        }
        report["datasets"][dataset] = {
            "coverage": {
                "count": len(ids),
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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-shards", type=int, default=DEFAULT_EXPECTED_SHARDS)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument("--bootstrap-resamples", type=int, default=DEFAULT_BOOTSTRAP_RESAMPLES)
    args = parser.parse_args(argv)
    result = build_report(
        args.input_dir,
        args.sampling_dir,
        args.cohort_audit,
        expected_shards=args.expected_shards,
        bootstrap_seed=args.bootstrap_seed,
        bootstrap_resamples=args.bootstrap_resamples,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
