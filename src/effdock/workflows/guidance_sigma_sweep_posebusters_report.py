"""Strict SigmaDock-compatible PoseBusters report for the eta-2 sigma sweep."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from effdock.workflows.guidance_budget_posebusters_report import (
    POSEBUSTERS_CONFIG,
    POSEBUSTERS_VERSION,
    VALIDITY_CHECKS,
    _read_csv_rows,
    _resolve_cell_csv,
)
from effdock.workflows.posebusters_report import file_sha256, load_rows

PROTOCOL_ID = "EFFDOCK-SIGMADOCK-COMPATIBLE-PB-SIGMA-ETA2-V1"
SIGMAS = (0.5, 1.0, 2.0, 3.0, 4.0)
DATASETS = {"astex": 85, "posebusters": 308}
SELECTORS = ("confidence",)
SHARDS = 8
SIGMADOCK_LEGACY_CHECKS = tuple(
    check for check in VALIDITY_CHECKS if check != "no_radicals"
)
SIGMADOCK_STATISTICS_SOURCE = (
    "https://github.com/alvaroprat97/sigmadock/blob/main/"
    "src/sigmadock/chem/statistics.py"
)


def sigma_tag(sigma: float) -> str:
    return f"sigma{int(round(sigma * 1000)):04d}"


def run_name(dataset: str, sigma: float) -> str:
    if sigma == 0.5:
        return f"effdock-guidance-steric-high-eta-v1-{dataset}-n100-s10-eta2000"
    return f"effdock-guidance-sigma-sweep-eta2-v1-{dataset}-n100-s10-{sigma_tag(sigma)}"


def sampling_root(sweep_root: Path, reference_root: Path, sigma: float) -> Path:
    return reference_root / "raw" if sigma == 0.5 else sweep_root / "raw"


def load_expected_ids(manifest_path: Path) -> dict[str, tuple[str, ...]]:
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    datasets = raw.get("datasets")
    if not isinstance(datasets, dict) or set(datasets) != set(DATASETS):
        raise ValueError("benchmark input manifest dataset inventory mismatch")
    result: dict[str, tuple[str, ...]] = {}
    for dataset, expected_count in DATASETS.items():
        ligand_inventory = datasets[dataset].get("ligands")
        ids = (
            list(ligand_inventory)
            if isinstance(ligand_inventory, dict)
            else ligand_inventory
        )
        if (
            not isinstance(ids, list)
            or len(ids) != expected_count
            or len(ids) != len(set(ids))
            or any(not isinstance(value, str) or not value for value in ids)
        ):
            raise ValueError(f"{dataset}: invalid frozen ID inventory")
        result[dataset] = tuple(sorted(ids))
    return result


def load_official_cell(
    official_root: Path,
    *,
    dataset: str,
    sigma: float,
    selector: str,
    expected_ids: tuple[str, ...],
) -> tuple[dict[str, dict[str, Any]], str, list[Path]]:
    name = run_name(dataset, sigma)
    cell_dir = official_root / selector / name
    if not cell_dir.is_dir():
        raise FileNotFoundError(f"missing official PoseBusters cell: {cell_dir}")
    rows_by_id: dict[str, dict[str, Any]] = {}
    rmsd_check: str | None = None
    summaries: list[Path] = []
    for shard_index in range(SHARDS):
        tag = f"shard-{shard_index:03d}-of-{SHARDS:03d}"
        summary_path = cell_dir / f"{tag}.summary.json"
        expected_csv = cell_dir / f"{tag}.csv"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summaries.append(summary_path)
        expected_assigned = len(expected_ids[shard_index::SHARDS])
        required = {
            "posebusters_version": POSEBUSTERS_VERSION,
            "config": POSEBUSTERS_CONFIG,
            "selector": selector,
            "input_hashes_verified": True,
            "num_input_hashes_verified": expected_assigned,
            "expected_discovered_count": DATASETS[dataset],
            "num_discovered_total": DATASETS[dataset],
            "num_assigned": expected_assigned,
            "num_success": expected_assigned,
            "num_failed": 0,
            "failures": [],
        }
        for key, expected in required.items():
            if summary.get(key) != expected:
                raise ValueError(
                    f"{summary_path}: {key} must be {expected!r}, got {summary.get(key)!r}"
                )
        if summary.get("only_id") not in (None, ""):
            raise ValueError(f"{summary_path}: smoke-only output cannot enter full report")
        csv_path = _resolve_cell_csv(summary_path, summary.get("csv"), expected_csv)
        shard_rows, current_rmsd = _read_csv_rows(
            csv_path, expected_rmsd_check=rmsd_check
        )
        rmsd_check = current_rmsd
        shard_expected_ids = set(expected_ids[shard_index::SHARDS])
        if {row["id"] for row in shard_rows} != shard_expected_ids:
            raise ValueError(f"{csv_path}: assigned ID inventory mismatch")
        for row in shard_rows:
            if row["id"] in rows_by_id:
                raise ValueError(f"{cell_dir}: duplicate complex ID {row['id']}")
            row["rmsd_check"] = current_rmsd
            rows_by_id[row["id"]] = row
    if tuple(sorted(rows_by_id)) != expected_ids or rmsd_check is None:
        raise ValueError(f"{cell_dir}: incomplete full-cohort PoseBusters inventory")
    return rows_by_id, rmsd_check, summaries


def summarize_cell(
    sampling_dir: Path,
    *,
    dataset: str,
    sigma: float,
    selector: str,
    expected_ids: tuple[str, ...],
    official_rows: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, dict[str, bool]]]:
    name = run_name(dataset, sigma)
    sampling_rows = load_rows(sampling_dir, name)
    if tuple(row["id"] for row in sampling_rows) != expected_ids:
        raise ValueError(f"{name}: sampling ID inventory mismatch")
    outcomes: dict[str, dict[str, bool]] = {}
    selected_rmsds: list[float] = []
    strict_valid: list[bool] = []
    legacy_valid: list[bool] = []
    rmsd_success: list[bool] = []
    for sampling_row in sampling_rows:
        complex_id = sampling_row["id"]
        rmsd = float(sampling_row[f"{selector}_rmsd"])
        if not math.isfinite(rmsd) or rmsd < 0.0:
            raise ValueError(f"{name}/{complex_id}: invalid selected RMSD")
        official = official_rows[complex_id]
        strict = bool(official["posebusters_valid"])
        legacy = all(official["checks"][key] for key in SIGMADOCK_LEGACY_CHECKS)
        success = rmsd < 2.0
        selected_rmsds.append(rmsd)
        strict_valid.append(strict)
        legacy_valid.append(legacy)
        rmsd_success.append(success)
        outcomes[complex_id] = {
            "rmsd_lt2": success,
            "pb_valid_27": strict,
            "pb_valid_sigmadock_legacy26": legacy,
            "joint_27": success and strict,
            "joint_sigmadock_legacy26": success and legacy,
        }
    count = len(expected_ids)

    def pct(values: list[bool]) -> float:
        return 100.0 * sum(values) / count

    check_pass_count = {
        check: sum(official_rows[complex_id]["checks"][check] for complex_id in expected_ids)
        for check in VALIDITY_CHECKS
    }
    metrics = {
        "count": count,
        "selected_rmsd_lt2_pct": pct(rmsd_success),
        "selected_median_rmsd_A": statistics.median(selected_rmsds),
        "posebusters_valid_27_pct": pct(strict_valid),
        "posebusters_valid_sigmadock_legacy26_pct": pct(legacy_valid),
        "joint_rmsd_lt2_and_posebusters_valid_27_pct": pct(
            [left and right for left, right in zip(rmsd_success, strict_valid)]
        ),
        "joint_rmsd_lt2_and_posebusters_valid_sigmadock_legacy26_pct": pct(
            [left and right for left, right in zip(rmsd_success, legacy_valid)]
        ),
        "check_pass_pct": {
            key: value / count * 100.0 for key, value in check_pass_count.items()
        },
    }
    return metrics, outcomes


def transition_summary(
    baseline: dict[str, dict[str, bool]],
    comparison: dict[str, dict[str, bool]],
    ids: tuple[str, ...],
) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for key in ("rmsd_lt2", "pb_valid_27", "joint_27"):
        left = [baseline[complex_id][key] for complex_id in ids]
        right = [comparison[complex_id][key] for complex_id in ids]
        result[key] = {
            "false_to_true": sum(not a and b for a, b in zip(left, right)),
            "true_to_false": sum(a and not b for a, b in zip(left, right)),
            "both_true": sum(a and b for a, b in zip(left, right)),
            "both_false": sum(not a and not b for a, b in zip(left, right)),
        }
    return result


def build_report(
    *,
    sweep_root: Path,
    reference_root: Path,
    official_root: Path,
    input_manifest: Path,
    full_audit: Path,
    reference_audit: Path,
) -> dict[str, Any]:
    for audit_path, mode in ((full_audit, "full"), (reference_audit, "reference")):
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if audit.get("status") != "passed" or audit.get("mode") != mode:
            raise ValueError(f"{audit_path}: required sampling audit did not pass")
    expected = load_expected_ids(input_manifest)
    cells: dict[tuple[str, float, str], dict[str, dict[str, bool]]] = {}
    result: dict[str, Any] = {
        "schema_version": "effdock.sigmadock_compatible_posebusters_sigma_eta2.v1",
        "protocol_id": PROTOCOL_ID,
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "condition": {"eta": 2.0, "sigmas": list(SIGMAS), "num_samples": 100, "num_steps": 10},
        "selectors": {
            "confidence": "primary minimum predicted-RMSD confidence head",
        },
        "posebusters": {
            "version": POSEBUSTERS_VERSION,
            "config": POSEBUSTERS_CONFIG,
            "primary_definition": "all 27 non-RMSD PoseBusters 0.6.5 redock checks",
            "primary_checks": list(VALIDITY_CHECKS),
            "sigmadock_legacy_compatibility_definition": (
                "the 26 non-RMSD checks listed by the current SigmaDock statistics code; "
                "no_radicals is excluded only from this compatibility view"
            ),
            "sigmadock_legacy_checks": list(SIGMADOCK_LEGACY_CHECKS),
            "sigmadock_statistics_source": SIGMADOCK_STATISTICS_SOURCE,
            "rmsd_check_excluded_from_validity": True,
            "selection_leakage_policy": (
                "PoseBusters outcomes are evaluated after frozen confidence selection and "
                "are never used to choose the primary Top-1 pose"
            ),
        },
        "inputs": {
            "sweep_root": str(sweep_root.resolve()),
            "reference_root": str(reference_root.resolve()),
            "input_manifest": str(input_manifest.resolve()),
            "input_manifest_sha256": file_sha256(input_manifest),
            "full_audit": str(full_audit.resolve()),
            "full_audit_sha256": file_sha256(full_audit),
            "reference_audit": str(reference_audit.resolve()),
            "reference_audit_sha256": file_sha256(reference_audit),
        },
        "datasets": {},
    }
    rmsd_checks: set[str] = set()
    for dataset in DATASETS:
        dataset_cells: dict[str, Any] = {}
        for sigma in SIGMAS:
            selector_cells: dict[str, Any] = {}
            for selector in SELECTORS:
                official_rows, rmsd_check, summaries = load_official_cell(
                    official_root,
                    dataset=dataset,
                    sigma=sigma,
                    selector=selector,
                    expected_ids=expected[dataset],
                )
                rmsd_checks.add(rmsd_check)
                metrics, outcomes = summarize_cell(
                    sampling_root(sweep_root, reference_root, sigma),
                    dataset=dataset,
                    sigma=sigma,
                    selector=selector,
                    expected_ids=expected[dataset],
                    official_rows=official_rows,
                )
                cells[(dataset, sigma, selector)] = outcomes
                selector_cells[selector] = {
                    "metrics": metrics,
                    "shard_summary_sha256": {
                        path.name: file_sha256(path) for path in summaries
                    },
                }
            dataset_cells[sigma_tag(sigma)] = {
                "sigma": sigma,
                "source": "frozen_reference" if sigma == 0.5 else "sigma_sweep",
                "selectors": selector_cells,
            }
        result["datasets"][dataset] = {
            "count": len(expected[dataset]),
            "cells": dataset_cells,
            "vs_sigma_0_5": {
                selector: {
                    sigma_tag(sigma): transition_summary(
                        cells[(dataset, 0.5, selector)],
                        cells[(dataset, sigma, selector)],
                        expected[dataset],
                    )
                    for sigma in SIGMAS
                }
                for selector in SELECTORS
            },
        }
    if len(rmsd_checks) != 1:
        raise ValueError(f"PoseBusters RMSD check column mismatch: {sorted(rmsd_checks)}")
    result["posebusters"]["observed_rmsd_check"] = next(iter(rmsd_checks))
    return result


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# SigmaDock-compatible official PoseBusters: eta=2 sigma sweep",
        "",
        "> Primary PB-valid uses all 27 non-RMSD PoseBusters 0.6.5 redock checks. "
        "The SigmaDock-legacy column excludes only `no_radicals` for comparability.",
        "",
    ]
    for dataset in DATASETS:
        lines.extend(
            [
                f"## {dataset}",
                "",
                "| sigma | selector | RMSD <2A | PB-valid (27) | Joint (27) | PB-valid (SigmaDock 26) | Joint (SigmaDock 26) | median RMSD |",
                "|---:|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for cell in report["datasets"][dataset]["cells"].values():
            for selector in SELECTORS:
                metrics = cell["selectors"][selector]["metrics"]
                lines.append(
                    f"| {cell['sigma']:.1f} | {selector} | "
                    f"{metrics['selected_rmsd_lt2_pct']:.1f}% | "
                    f"{metrics['posebusters_valid_27_pct']:.1f}% | "
                    f"{metrics['joint_rmsd_lt2_and_posebusters_valid_27_pct']:.1f}% | "
                    f"{metrics['posebusters_valid_sigmadock_legacy26_pct']:.1f}% | "
                    f"{metrics['joint_rmsd_lt2_and_posebusters_valid_sigmadock_legacy26_pct']:.1f}% | "
                    f"{metrics['selected_median_rmsd_A']:.2f} |"
                )
        lines.append("")
        lines.extend(
            [
                "Lowest-pass non-RMSD checks (primary confidence):",
                "",
                "| sigma | check 1 | check 2 | check 3 |",
                "|---:|---|---|---|",
            ]
        )
        for cell in report["datasets"][dataset]["cells"].values():
            pass_pct = cell["selectors"]["confidence"]["metrics"]["check_pass_pct"]
            lowest = sorted(pass_pct.items(), key=lambda item: (item[1], item[0]))[:3]
            formatted = [f"{name} ({value:.1f}%)" for name, value in lowest]
            lines.append(f"| {cell['sigma']:.1f} | " + " | ".join(formatted) + " |")
        lines.append("")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--full-audit", type=Path, required=True)
    parser.add_argument("--reference-audit", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args(argv)
    report = build_report(
        sweep_root=args.sweep_root,
        reference_root=args.reference_root,
        official_root=args.official_root,
        input_manifest=args.input_manifest,
        full_audit=args.full_audit,
        reference_audit=args.reference_audit,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    args.output_md.write_text(render_markdown(report), encoding="utf-8")


if __name__ == "__main__":
    main()
