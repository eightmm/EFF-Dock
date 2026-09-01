#!/usr/bin/env python3
"""Strict official-PoseBusters report for full-cohort guidance V2."""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Any

from effdock.workflows.evaluate import sorted_id_sha256
from effdock.workflows.guidance_budget_full_report import (
    PROTOCOL_ID,
    _expected_run_name,
    load_full_cohort_audits,
)
from effdock.workflows.guidance_budget_posebusters_report import (
    EXPECTED_SELECTOR,
    MODULE_CHECKS,
    POSEBUSTERS_CONFIG,
    POSEBUSTERS_VERSION,
    VALIDITY_CHECKS,
    _aggregate_cell,
    _sha256_file,
)
from effdock.workflows.guidance_budget_posebusters_report import (
    _paired_pass_all as _legacy_paired_pass_all,
)
from effdock.workflows.guidance_budget_report import (
    CONDITIONS,
    DATASETS,
    DEFAULT_BOOTSTRAP_RESAMPLES,
    DEFAULT_BOOTSTRAP_SEED,
    DEFAULT_EXPECTED_SHARDS,
)
from effdock.workflows.guidance_coverage_audit import ID_HASH_CONTRACT


def _paired_pass_all(
    baseline_rows: dict[str, dict[str, Any]],
    comparison_rows: dict[str, dict[str, Any]],
    ids: tuple[str, ...],
    *,
    baseline_label: str,
    comparison_label: str,
    seed: int,
    resamples: int,
) -> dict[str, Any]:
    """Use the FULL-V2 versioned ID-hash contract in every paired result."""
    result = _legacy_paired_pass_all(
        baseline_rows,
        comparison_rows,
        ids,
        baseline_label=baseline_label,
        comparison_label=comparison_label,
        seed=seed,
        resamples=resamples,
    )
    result["common_ids_sha256"] = sorted_id_sha256(list(ids))
    result["ids_hash_contract"] = ID_HASH_CONTRACT
    return result


def _slice_pass_all(
    baseline_rows: dict[str, dict[str, Any]],
    comparison_rows: dict[str, dict[str, Any]],
    slices: dict[str, tuple[str, ...]],
    *,
    baseline_label: str,
    comparison_label: str,
    seed: int,
    resamples: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, ids in slices.items():
        entry: dict[str, Any] = {
            "count": len(ids),
            "ids_sha256": sorted_id_sha256(list(ids)),
            "ids_hash_contract": ID_HASH_CONTRACT,
        }
        if ids:
            baseline = {complex_id: baseline_rows[complex_id] for complex_id in ids}
            comparison = {complex_id: comparison_rows[complex_id] for complex_id in ids}
            entry["paired_effect"] = _paired_pass_all(
                baseline,
                comparison,
                ids,
                baseline_label=baseline_label,
                comparison_label=comparison_label,
                seed=seed,
                resamples=resamples,
            )
        result[name] = entry
    return result


def build_report(
    input_dir: Path,
    cohort_audits: Path | list[Path] | tuple[Path, ...],
    *,
    expected_shards: int = DEFAULT_EXPECTED_SHARDS,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
) -> dict[str, Any]:
    """Reject incomplete official output and report exact full-cohort validity."""
    if expected_shards < 1 or bootstrap_resamples < 1:
        raise ValueError("expected_shards and bootstrap_resamples must be >= 1")
    audits = load_full_cohort_audits(cohort_audits)
    expected_runs = {
        _expected_run_name(dataset, num_samples, num_steps, arm)
        for dataset in DATASETS
        for _, num_samples, num_steps in CONDITIONS
        for arm in ("unguided", "guided")
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
            f"official PoseBusters run-cell mismatch; "
            f"missing={sorted(expected_runs - actual_runs)}, "
            f"extra={sorted(actual_runs - expected_runs)}"
        )

    report: dict[str, Any] = {
        "protocol_id": PROTOCOL_ID,
        "status": "complete_strict_full_cohort_paired_official_posebusters",
        "claim_boundary": "secondary descriptive V2 result; Astex/PoseBusters were opened in V1",
        "guidance_implementation": audits["astex"]["implementation"],
        "guidance_parameter_set": audits["astex"]["parameter_set"],
        "receptor_policy_identity": audits["astex"]["receptor_policy_identity"],
        "benchmark_input_identities": {
            dataset: audits[dataset]["benchmark_input_identity"] for dataset in DATASETS
        },
        "audit_id_hash_contract": ID_HASH_CONTRACT,
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
    all_rows: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = {}
    rmsd_check_names: set[str] = set()
    for dataset in DATASETS:
        audit = audits[dataset]
        ids = audit["ids"]
        slices = audit["chemistry_slices"]
        representation_slices = audit["ligand_representation_slices"]
        integrity_slices = audit["integrity_slices"]
        dataset_result: dict[str, Any] = {
            "full_cohort_coverage": {
                "discovered": audit["discovered"],
                "official_evaluated_per_cell": audit["discovered"],
                "official_failed": 0,
                "coverage_pct": 100.0,
                "ids_sha256": audit["ids_sha256"],
                "ids_hash_contract": ID_HASH_CONTRACT,
                "audit_path": audit["source_path"],
                "audit_sha256": _sha256_file(Path(audit["source_path"])),
            },
            "chemistry_slices": {
                name: {
                    "count": len(slice_ids),
                    "ids_sha256": sorted_id_sha256(list(slice_ids)),
                    "ids_hash_contract": ID_HASH_CONTRACT,
                }
                for name, slice_ids in slices.items()
            },
            "ligand_representation_slices": {
                name: {
                    "count": len(slice_ids),
                    "ids_sha256": sorted_id_sha256(list(slice_ids)),
                    "ids_hash_contract": ID_HASH_CONTRACT,
                }
                for name, slice_ids in representation_slices.items()
            },
            "checkpoint_integrity_boundary": audit["integrity_boundary"],
            "checkpoint_integrity_slices": {
                name: {
                    "count": len(slice_ids),
                    "ids_sha256": sorted_id_sha256(list(slice_ids)),
                    "ids_hash_contract": ID_HASH_CONTRACT,
                }
                for name, slice_ids in integrity_slices.items()
            },
            "fallback_reasons": audit["fallback_reasons"],
            "cells": {},
        }
        for condition, num_samples, num_steps in CONDITIONS:
            cell: dict[str, Any] = {}
            for arm in ("unguided", "guided"):
                run_name = _expected_run_name(dataset, num_samples, num_steps, arm)
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
                all_rows[(dataset, condition, arm)] = rows
                aggregate["eligible_ids_sha256"] = sorted_id_sha256(list(ids))
                aggregate["ids_hash_contract"] = ID_HASH_CONTRACT
                cell[arm] = aggregate
                rmsd_check_names.add(rmsd_check)
            cell["guided_vs_unguided"] = _paired_pass_all(
                all_rows[(dataset, condition, "unguided")],
                all_rows[(dataset, condition, "guided")],
                ids,
                baseline_label="unguided",
                comparison_label="guided",
                seed=bootstrap_seed,
                resamples=bootstrap_resamples,
            )
            cell["chemistry_slice_guided_vs_unguided"] = _slice_pass_all(
                all_rows[(dataset, condition, "unguided")],
                all_rows[(dataset, condition, "guided")],
                slices,
                baseline_label="unguided",
                comparison_label="guided",
                seed=bootstrap_seed,
                resamples=bootstrap_resamples,
            )
            cell["ligand_representation_slice_guided_vs_unguided"] = _slice_pass_all(
                all_rows[(dataset, condition, "unguided")],
                all_rows[(dataset, condition, "guided")],
                representation_slices,
                baseline_label="unguided",
                comparison_label="guided",
                seed=bootstrap_seed,
                resamples=bootstrap_resamples,
            )
            cell["checkpoint_integrity_slice_guided_vs_unguided"] = _slice_pass_all(
                all_rows[(dataset, condition, "unguided")],
                all_rows[(dataset, condition, "guided")],
                integrity_slices,
                baseline_label="unguided",
                comparison_label="guided",
                seed=bootstrap_seed,
                resamples=bootstrap_resamples,
            )
            dataset_result["cells"][condition] = cell

        guided_budget: dict[str, Any] = {
            "common_ids": len(ids),
            "common_ids_sha256": sorted_id_sha256(list(ids)),
            "ids_hash_contract": ID_HASH_CONTRACT,
            "cell_posebusters_valid_pct": {
                condition: dataset_result["cells"][condition]["guided"]["posebusters_valid_pct"]
                for condition, _, _ in CONDITIONS
            },
            "pairwise_deltas": {},
        }
        for (left, _, _), (right, _, _) in combinations(CONDITIONS, 2):
            guided_budget["pairwise_deltas"][f"{right}_minus_{left}"] = _paired_pass_all(
                all_rows[(dataset, left, "guided")],
                all_rows[(dataset, right, "guided")],
                ids,
                baseline_label=left,
                comparison_label=right,
                seed=bootstrap_seed,
                resamples=bootstrap_resamples,
            )
        dataset_result["guided_budget_comparison"] = guided_budget
        report["datasets"][dataset] = dataset_result

    if len(rmsd_check_names) != 1:
        raise ValueError(f"RMSD check name differs across cells: {sorted(rmsd_check_names)}")
    report["posebusters"]["rmsd_check_excluded_from_validity"] = next(iter(rmsd_check_names))
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--cohort-audit", type=Path, action="append", required=True)
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
