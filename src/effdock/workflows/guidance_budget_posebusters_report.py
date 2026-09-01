#!/usr/bin/env python3
"""Strict official-PoseBusters report for the guidance budget-1000 study."""

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

from effdock.workflows.guidance_budget_report import (
    CONDITIONS,
    DATASETS,
    DEFAULT_BOOTSTRAP_RESAMPLES,
    DEFAULT_BOOTSTRAP_SEED,
    DEFAULT_EXPECTED_SHARDS,
    PROTOCOL_ID,
    _expected_run_name,
    _ids_sha256,
    _load_eligibility,
    _paired_metric,
)

POSEBUSTERS_VERSION = "0.6.5"
POSEBUSTERS_CONFIG = "redock"
EXPECTED_SELECTOR = "oracle"

# PoseBusters 0.6.5 ``redock`` binary outputs that define pass-all validity.
# The separate RMSD check is intentionally not part of this conjunction.
VALIDITY_CHECKS = (
    "mol_pred_loaded",
    "mol_true_loaded",
    "mol_cond_loaded",
    "sanitization",
    "inchi_convertible",
    "all_atoms_connected",
    "no_radicals",
    "molecular_formula",
    "molecular_bonds",
    "double_bond_stereochemistry",
    "tetrahedral_chirality",
    "bond_lengths",
    "bond_angles",
    "internal_steric_clash",
    "aromatic_ring_flatness",
    "non-aromatic_ring_non-flatness",
    "double_bond_flatness",
    "internal_energy",
    "protein-ligand_maximum_distance",
    "minimum_distance_to_protein",
    "minimum_distance_to_organic_cofactors",
    "minimum_distance_to_inorganic_cofactors",
    "minimum_distance_to_waters",
    "volume_overlap_with_protein",
    "volume_overlap_with_organic_cofactors",
    "volume_overlap_with_inorganic_cofactors",
    "volume_overlap_with_waters",
)

# These groups follow the corresponding redock modules while combining the
# repeated Chemistry functions into one interpretable module-level outcome.
MODULE_CHECKS = {
    "loading": ("mol_pred_loaded", "mol_true_loaded", "mol_cond_loaded"),
    "chemistry": (
        "sanitization",
        "inchi_convertible",
        "all_atoms_connected",
        "no_radicals",
        "molecular_formula",
        "molecular_bonds",
        "double_bond_stereochemistry",
        "tetrahedral_chirality",
    ),
    "distance_geometry": ("bond_lengths", "bond_angles", "internal_steric_clash"),
    "ring_flatness": ("aromatic_ring_flatness",),
    "ring_non_flatness": ("non-aromatic_ring_non-flatness",),
    "double_bond_flatness": ("double_bond_flatness",),
    "energy_ratio": ("internal_energy",),
    "distance_to_protein": (
        "protein-ligand_maximum_distance",
        "minimum_distance_to_protein",
    ),
    "distance_to_organic_cofactors": ("minimum_distance_to_organic_cofactors",),
    "distance_to_inorganic_cofactors": ("minimum_distance_to_inorganic_cofactors",),
    "distance_to_waters": ("minimum_distance_to_waters",),
    "volume_overlap_with_protein": ("volume_overlap_with_protein",),
    "volume_overlap_with_organic_cofactors": ("volume_overlap_with_organic_cofactors",),
    "volume_overlap_with_inorganic_cofactors": (
        "volume_overlap_with_inorganic_cofactors",
    ),
    "volume_overlap_with_waters": ("volume_overlap_with_waters",),
}

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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_bool(value: Any, *, label: str) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"{label} must be true or false, got {value!r}")


def _resolve_cell_csv(summary_path: Path, value: Any, expected_path: Path) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{summary_path}: completed shard requires a non-empty csv path")
    raw = Path(value)
    candidates = [raw] if raw.is_absolute() else [Path.cwd() / raw, summary_path.parent / raw]
    hits: list[Path] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.exists() and resolved not in hits:
            hits.append(resolved)
    if not hits:
        raise FileNotFoundError(f"{summary_path}: CSV does not exist: {value}")
    if expected_path.resolve() not in hits:
        raise ValueError(
            f"{summary_path}: csv must resolve to its run-local shard file {expected_path}"
        )
    if len(hits) > 1:
        raise ValueError(f"{summary_path}: ambiguous relative CSV path: {value}")
    return hits[0]


def _read_csv_rows(
    csv_path: Path,
    *,
    expected_rmsd_check: str | None,
) -> tuple[list[dict[str, Any]], str]:
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{csv_path}: missing CSV header")
        if len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError(f"{csv_path}: duplicate CSV columns")
        rmsd_checks = [key for key in reader.fieldnames if key.startswith("rmsd_")]
        if len(rmsd_checks) != 1:
            raise ValueError(f"{csv_path}: expected exactly one separate RMSD check")
        rmsd_check = rmsd_checks[0]
        if expected_rmsd_check is not None and rmsd_check != expected_rmsd_check:
            raise ValueError(f"{csv_path}: inconsistent RMSD check column")
        expected_fields = {"id", "posebusters_valid", rmsd_check, *VALIDITY_CHECKS}
        if set(reader.fieldnames) != expected_fields:
            missing = sorted(expected_fields - set(reader.fieldnames))
            extra = sorted(set(reader.fieldnames) - expected_fields)
            raise ValueError(
                f"{csv_path}: PoseBusters 0.6.5 redock schema mismatch; "
                f"missing={missing}, extra={extra}"
            )
        rows: list[dict[str, Any]] = []
        for line_number, raw in enumerate(reader, start=2):
            complex_id = str(raw["id"]).strip()
            if not complex_id or complex_id != raw["id"]:
                raise ValueError(f"{csv_path}:{line_number}: invalid complex ID")
            checks = {
                key: _parse_bool(raw[key], label=f"{csv_path}:{line_number}.{key}")
                for key in (*VALIDITY_CHECKS, rmsd_check)
            }
            reported_valid = _parse_bool(
                raw["posebusters_valid"],
                label=f"{csv_path}:{line_number}.posebusters_valid",
            )
            recomputed_valid = all(checks[key] for key in VALIDITY_CHECKS)
            if reported_valid != recomputed_valid:
                raise ValueError(
                    f"{csv_path}:{line_number}: posebusters_valid does not equal "
                    "the non-RMSD pass-all conjunction"
                )
            rows.append(
                {
                    "id": complex_id,
                    "posebusters_valid": reported_valid,
                    "checks": checks,
                    "modules": {
                        module: all(checks[key] for key in module_checks)
                        for module, module_checks in MODULE_CHECKS.items()
                    },
                }
            )
    return rows, rmsd_check


def _load_manifest_coverage(
    eligibility_path: Path,
    manifest_summary: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    raw = json.loads(eligibility_path.read_text())
    result: dict[str, dict[str, Any]] = {}
    for dataset in DATASETS:
        entry = raw["datasets"][dataset]
        discovered = manifest_summary[dataset]["discovered"]
        eligible = manifest_summary[dataset]["eligible"]
        if discovered is None:
            raise ValueError(f"eligibility {dataset} requires discovered for full-dataset coverage")
        excluded = discovered - eligible
        if "excluded" in entry and int(entry["excluded"]) != excluded:
            raise ValueError(f"eligibility {dataset} excluded count mismatch")
        eligibility_pct = eligible / discovered * 100.0 if discovered else 0.0
        if "eligibility_pct" in entry and not math.isclose(
            float(entry["eligibility_pct"]),
            eligibility_pct,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"eligibility {dataset} eligibility_pct mismatch")
        failure_codes = entry.get("failure_codes", {})
        if not isinstance(failure_codes, dict):
            raise ValueError(f"eligibility {dataset} failure_codes must be an object")
        if failure_codes and sum(int(value) for value in failure_codes.values()) != excluded:
            raise ValueError(f"eligibility {dataset} failure-code counts do not sum to excluded")
        result[dataset] = {
            "full_dataset_discovered": discovered,
            "chemistry_eligible": eligible,
            "chemistry_excluded": excluded,
            "eligibility_pct": eligibility_pct,
            "eligible_ids_sha256": manifest_summary[dataset]["eligible_ids_sha256"],
            "exclusion_failure_codes": failure_codes,
        }
    return result


def _summarize_rows(rows_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rows = list(rows_by_id.values())
    count = len(rows)
    if not count:
        raise ValueError("cannot summarize an empty official-PoseBusters cell")
    valid_count = sum(row["posebusters_valid"] for row in rows)
    check_pass_count = {
        check: sum(row["checks"][check] for row in rows)
        for check in (*VALIDITY_CHECKS, next(iter(rows))["rmsd_check"])
    }
    module_pass_count = {
        module: sum(row["modules"][module] for row in rows) for module in MODULE_CHECKS
    }
    return {
        "posebusters_valid_count": valid_count,
        "posebusters_valid_pct": valid_count / count * 100.0,
        "check_pass_count": check_pass_count,
        "check_pass_pct": {
            key: value / count * 100.0 for key, value in check_pass_count.items()
        },
        "module_pass_count": module_pass_count,
        "module_pass_pct": {
            key: value / count * 100.0 for key, value in module_pass_count.items()
        },
    }


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
    if set(baseline_rows) != set(ids) or set(comparison_rows) != set(ids):
        raise ValueError("official PoseBusters pairing requires exact eligible-ID coverage")
    baseline = np.asarray(
        [baseline_rows[complex_id]["posebusters_valid"] for complex_id in ids], dtype=float
    )
    comparison = np.asarray(
        [comparison_rows[complex_id]["posebusters_valid"] for complex_id in ids], dtype=float
    )
    metric = _paired_metric(
        baseline,
        comparison,
        reducer="mean_pp",
        unit="percentage_points",
        seed=seed,
        resamples=resamples,
    )
    return {
        "direction": f"{comparison_label}_minus_{baseline_label}",
        "common_ids": len(ids),
        "common_ids_sha256": _ids_sha256(ids),
        "posebusters_valid": metric,
        "transitions": {
            "invalid_to_valid": int(((baseline == 0) & (comparison == 1)).sum()),
            "valid_to_invalid": int(((baseline == 1) & (comparison == 0)).sum()),
            "both_valid": int(((baseline == 1) & (comparison == 1)).sum()),
            "both_invalid": int(((baseline == 0) & (comparison == 0)).sum()),
        },
    }


def _aggregate_cell(
    cell_dir: Path,
    eligible_ids: tuple[str, ...],
    *,
    run_name: str,
    dataset: str,
    condition: str,
    arm: str,
    num_samples: int,
    num_steps: int,
    expected_shards: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any], str]:
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
            f"{run_name}: expected exactly {expected_shards} shard summaries; "
            f"missing={missing}, extra={extra}"
        )

    eligible_set = set(eligible_ids)
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
        if summary["selector"] != EXPECTED_SELECTOR:
            raise ValueError(f"{summary_path}: selector must be {EXPECTED_SELECTOR!r}")
        if int(summary["num_discovered_total"]) != len(eligible_ids):
            raise ValueError(
                f"{summary_path}: num_discovered_total must equal the eligible cohort size"
            )
        expected_shard_ids = tuple(eligible_ids[shard_index::expected_shards])
        if int(summary["num_assigned"]) != len(expected_shard_ids):
            raise ValueError(f"{summary_path}: num_assigned does not match deterministic shard")
        failures = summary["failures"]
        if not isinstance(failures, list):
            raise ValueError(f"{summary_path}: failures must be a list")
        if int(summary["num_failed"]) != len(failures):
            raise ValueError(f"{summary_path}: num_failed does not match failures")
        if int(summary["num_success"]) + int(summary["num_failed"]) != int(
            summary["num_assigned"]
        ):
            raise ValueError(f"{summary_path}: assigned count differs from success plus failure")
        for failure in failures:
            if not isinstance(failure, dict) or not isinstance(failure.get("id"), str):
                raise ValueError(f"{summary_path}: every failure requires a string id")
        if failures:
            shard_failure_ids = [failure["id"] for failure in failures]
            if len(shard_failure_ids) != len(set(shard_failure_ids)):
                raise ValueError(f"{summary_path}: duplicate failure IDs")
            outside_failures = sorted(set(shard_failure_ids) - set(expected_shard_ids))
            if outside_failures:
                raise ValueError(
                    f"{summary_path}: failure IDs outside deterministic eligible shard "
                    f"{outside_failures[:5]}"
                )
            raise ValueError(
                f"{run_name}: {len(failures)} eligible PoseBusters failures; "
                "strict report rejects survivor-only aggregation"
            )

        expected_csv = cell_dir / f"{tag}.csv"
        csv_path = _resolve_cell_csv(summary_path, summary["csv"], expected_csv)
        shard_rows, current_rmsd_check = _read_csv_rows(
            csv_path,
            expected_rmsd_check=rmsd_check,
        )
        rmsd_check = current_rmsd_check
        if int(summary["num_success"]) != len(shard_rows):
            raise ValueError(f"{summary_path}: num_success does not match CSV rows")
        shard_ids = [row["id"] for row in shard_rows]
        if len(shard_ids) != len(set(shard_ids)):
            raise ValueError(f"{run_name}: duplicate success ID within shard {shard_index}")
        for row in shard_rows:
            complex_id = row["id"]
            if complex_id in rows_by_id:
                raise ValueError(f"{run_name}: duplicate success ID {complex_id}")
            row["rmsd_check"] = current_rmsd_check
            rows_by_id[complex_id] = row
        actual_pct = (
            sum(row["posebusters_valid"] for row in shard_rows) / len(shard_rows) * 100.0
            if shard_rows
            else None
        )
        reported_pct = summary["posebusters_valid_pct"]
        if actual_pct is None or reported_pct is None or not math.isclose(
            float(reported_pct), actual_pct, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(f"{summary_path}: posebusters_valid_pct does not match CSV")
        observed_shard_ids = set(shard_ids) | {
            failure["id"] for failure in failures if isinstance(failure.get("id"), str)
        }
        if observed_shard_ids != set(expected_shard_ids):
            missing = sorted(set(expected_shard_ids) - observed_shard_ids)
            outside = sorted(observed_shard_ids - set(expected_shard_ids))
            raise ValueError(
                f"{summary_path}: deterministic shard ID mismatch; "
                f"missing={missing}, outside={outside}"
            )
        summary_paths.append(str(summary_path))

    missing_ids = sorted(eligible_set - set(rows_by_id))
    outside_ids = sorted(set(rows_by_id) - eligible_set)
    if missing_ids or outside_ids:
        raise ValueError(
            f"{run_name}: eligibility coverage mismatch; "
            f"missing={missing_ids[:5]}, outside={outside_ids[:5]}"
        )
    if rmsd_check is None:
        raise ValueError(f"{run_name}: RMSD check column was not observed")

    stats = _summarize_rows(rows_by_id)
    aggregate = {
        "run_name": run_name,
        "dataset": dataset,
        "condition": condition,
        "arm": arm,
        "num_samples": num_samples,
        "num_steps": num_steps,
        "model_pose_step_budget": num_samples * num_steps,
        "selector": EXPECTED_SELECTOR,
        "eligible": len(eligible_ids),
        "success": len(rows_by_id),
        "failed": 0,
        "eligible_coverage_pct": 100.0,
        "eligible_ids_sha256": _ids_sha256(eligible_ids),
        **stats,
        "shard_summaries": summary_paths,
    }
    return rows_by_id, aggregate, rmsd_check


def build_report(
    input_dir: Path,
    eligibility_path: Path,
    *,
    expected_shards: int = DEFAULT_EXPECTED_SHARDS,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
) -> dict[str, Any]:
    """Reject incomplete output and aggregate exact eligible-cohort paired validity."""
    if expected_shards < 1:
        raise ValueError("expected_shards must be >= 1")
    if bootstrap_resamples < 1:
        raise ValueError("bootstrap_resamples must be >= 1")
    eligible_by_dataset, manifest_summary = _load_eligibility(eligibility_path)
    manifest_coverage = _load_manifest_coverage(eligibility_path, manifest_summary)

    expected_runs = {
        _expected_run_name(dataset, num_samples, num_steps, arm)
        for dataset in DATASETS
        for _, num_samples, num_steps in CONDITIONS
        for arm in ("unguided", "guided")
    }
    actual_runs = {
        path.name
        for path in input_dir.iterdir()
        if path.is_dir() and any(path.glob("*.summary.json"))
    } if input_dir.is_dir() else set()
    if actual_runs != expected_runs:
        missing = sorted(expected_runs - actual_runs)
        extra = sorted(actual_runs - expected_runs)
        raise ValueError(f"official PoseBusters run-cell mismatch; missing={missing}, extra={extra}")

    report: dict[str, Any] = {
        "protocol_id": PROTOCOL_ID,
        "status": "complete_strict_paired_official_posebusters",
        "estimand": "frozen chemistry-eligible cohort; no survivor-only aggregation",
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
        "eligibility": {
            "path": str(eligibility_path),
            "sha256": _sha256_file(eligibility_path),
            "scope_warning": (
                "official validity is measured only on the frozen eligible cohort; "
                "eligibility_pct is retained against the full discovered dataset"
            ),
            "datasets": manifest_coverage,
        },
        "expected_shards_per_cell": expected_shards,
        "datasets": {},
    }

    all_rows: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = {}
    rmsd_check_names: set[str] = set()
    for dataset in DATASETS:
        coverage = manifest_coverage[dataset]
        dataset_result: dict[str, Any] = {
            "eligibility_coverage": {
                **coverage,
                "official_evaluated": len(eligible_by_dataset[dataset]),
                "official_failed": 0,
                "eligible_coverage_pct": 100.0,
                "full_dataset_measured_pct": coverage["eligibility_pct"],
            },
            "cells": {},
        }
        for condition, num_samples, num_steps in CONDITIONS:
            dataset_result["cells"][condition] = {}
            for arm in ("unguided", "guided"):
                run_name = _expected_run_name(dataset, num_samples, num_steps, arm)
                rows, aggregate, rmsd_check = _aggregate_cell(
                    input_dir / run_name,
                    eligible_by_dataset[dataset],
                    run_name=run_name,
                    dataset=dataset,
                    condition=condition,
                    arm=arm,
                    num_samples=num_samples,
                    num_steps=num_steps,
                    expected_shards=expected_shards,
                )
                all_rows[(dataset, condition, arm)] = rows
                dataset_result["cells"][condition][arm] = aggregate
                rmsd_check_names.add(rmsd_check)
            dataset_result["cells"][condition]["guided_vs_unguided"] = _paired_pass_all(
                all_rows[(dataset, condition, "unguided")],
                all_rows[(dataset, condition, "guided")],
                eligible_by_dataset[dataset],
                baseline_label="unguided",
                comparison_label="guided",
                seed=bootstrap_seed,
                resamples=bootstrap_resamples,
            )

        guided_budget = {
            "common_ids": len(eligible_by_dataset[dataset]),
            "common_ids_sha256": _ids_sha256(eligible_by_dataset[dataset]),
            "cell_posebusters_valid_pct": {
                condition: dataset_result["cells"][condition]["guided"][
                    "posebusters_valid_pct"
                ]
                for condition, _, _ in CONDITIONS
            },
            "pairwise_deltas": {},
        }
        for (left, _, _), (right, _, _) in combinations(CONDITIONS, 2):
            guided_budget["pairwise_deltas"][f"{right}_minus_{left}"] = _paired_pass_all(
                all_rows[(dataset, left, "guided")],
                all_rows[(dataset, right, "guided")],
                eligible_by_dataset[dataset],
                baseline_label=left,
                comparison_label=right,
                seed=bootstrap_seed,
                resamples=bootstrap_resamples,
            )
        dataset_result["guided_budget_comparison"] = guided_budget
        report["datasets"][dataset] = dataset_result

    if len(rmsd_check_names) != 1:
        raise ValueError(f"RMSD check name differs across cells: {sorted(rmsd_check_names)}")
    report["posebusters"]["rmsd_check_excluded_from_validity"] = next(
        iter(rmsd_check_names)
    )
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--eligibility", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-shards", type=int, default=DEFAULT_EXPECTED_SHARDS)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument(
        "--bootstrap-resamples", type=int, default=DEFAULT_BOOTSTRAP_RESAMPLES
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
