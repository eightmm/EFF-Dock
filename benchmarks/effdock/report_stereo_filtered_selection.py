#!/usr/bin/env python3
"""Aggregate the three-repeat input-stereo-filter selection ablation."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from effdock.workflows.guidance_budget_posebusters_report import VALIDITY_CHECKS

PROTOCOL_ID = "EFFDOCK-INPUT-STEREO-FILTER-SELECTION-V4"
EXPECTED = {"astex": 85, "posebusters": 308}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, default=16)
    return parser.parse_args()


def _bool(value: object) -> bool:
    return str(value).strip().lower() == "true"


def _rate(numerator: int, denominator: int) -> float:
    return 100.0 * numerator / denominator


def _metrics(rows: list[dict[str, str]]) -> dict[str, Any]:
    n = len(rows)
    rmsd_lt2 = [float(row["selected_rmsd"]) < 2.0 for row in rows]
    pb = [_bool(row["posebusters_valid"]) for row in rows]
    joint = [a and b for a, b in zip(rmsd_lt2, pb, strict=True)]
    return {
        "count": n,
        "rmsd_lt2_count": sum(rmsd_lt2),
        "rmsd_lt2_percent": _rate(sum(rmsd_lt2), n),
        "pb_valid_count": sum(pb),
        "pb_valid_percent": _rate(sum(pb), n),
        "joint_count": sum(joint),
        "joint_percent": _rate(sum(joint), n),
    }


def _load_original(condition_root: Path) -> dict[tuple[str, str], dict[str, str]]:
    rows: dict[tuple[str, str], dict[str, str]] = {}
    for path in sorted((condition_root / "full" / "selected_posebusters").glob("shard-*/results.csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                rows[(row["dataset"], row["id"])] = row
    return rows


def main() -> None:
    args = parse_args()
    all_runs: dict[str, list[dict[str, Any]]] = {dataset: [] for dataset in EXPECTED}
    failure_counts: dict[str, Counter[str]] = {dataset: Counter() for dataset in EXPECTED}
    fallback_ids: dict[str, Counter[str]] = {dataset: Counter() for dataset in EXPECTED}
    for repeat in range(3):
        condition_root = args.output_root / "cutoff_10" / f"repeat_{repeat}"
        rows: list[dict[str, str]] = []
        summaries: list[dict[str, Any]] = []
        for shard in range(args.num_shards):
            directory = (
                condition_root / "full" / "stereo_filtered_selection_v4"
                / f"shard-{shard:03d}-of-{args.num_shards:03d}"
            )
            summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
            if summary.get("status") != "complete" or summary.get("protocol_id") != PROTOCOL_ID:
                raise ValueError(f"incomplete or incompatible shard: {directory}")
            summaries.append(summary)
            with (directory / "results.csv").open(newline="", encoding="utf-8") as handle:
                rows.extend(csv.DictReader(handle))
        if len(rows) != sum(EXPECTED.values()) or any(row["error"] for row in rows):
            raise ValueError(f"repeat {repeat}: incomplete result inventory")
        original = _load_original(condition_root)
        if len(original) != len(rows):
            raise ValueError(f"repeat {repeat}: incomplete original PB inventory")

        for dataset, expected_count in EXPECTED.items():
            selected = [row for row in rows if row["dataset"] == dataset]
            if len(selected) != expected_count:
                raise ValueError(f"repeat {repeat}/{dataset}: coverage mismatch")
            baseline_rows = [original[(dataset, row["id"])] for row in selected]
            baseline_lt2 = [_bool(row["selected_rmsd_lt2"]) for row in baseline_rows]
            baseline_pb = [_bool(row["posebusters_valid"]) for row in baseline_rows]
            baseline_joint = [a and b for a, b in zip(baseline_lt2, baseline_pb, strict=True)]
            baseline = {
                "count": expected_count,
                "rmsd_lt2_count": sum(baseline_lt2),
                "rmsd_lt2_percent": _rate(sum(baseline_lt2), expected_count),
                "pb_valid_count": sum(baseline_pb),
                "pb_valid_percent": _rate(sum(baseline_pb), expected_count),
                "joint_count": sum(baseline_joint),
                "joint_percent": _rate(sum(baseline_joint), expected_count),
            }
            filtered = _metrics(selected)
            changed = sum(_bool(row["selection_changed"]) for row in selected)
            fallback = sum(_bool(row["stereo_filter_fallback"]) for row in selected)
            valid_counts = [int(row["stereo_valid_candidate_count"]) for row in selected]
            for row in selected:
                if _bool(row["stereo_filter_fallback"]):
                    fallback_ids[dataset][row["id"]] += 1
                for check in VALIDITY_CHECKS:
                    if not _bool(row[check]):
                        failure_counts[dataset][check] += 1
            all_runs[dataset].append(
                {
                    "repeat_index": repeat,
                    "baseline": baseline,
                    "stereo_filtered": filtered,
                    "delta_percentage_points": {
                        key: filtered[f"{key}_percent"] - baseline[f"{key}_percent"]
                        for key in ("rmsd_lt2", "pb_valid", "joint")
                    },
                    "selection_changed_count": changed,
                    "fallback_count": fallback,
                    "stereo_valid_candidate_count": {
                        "min": min(valid_counts),
                        "median": statistics.median(valid_counts),
                        "mean": statistics.fmean(valid_counts),
                        "max": max(valid_counts),
                    },
                    "official_pb_reused_count": sum(
                        _bool(row["official_pb_reused"]) for row in selected
                    ),
                }
            )

    aggregate: dict[str, Any] = {}
    for dataset, runs in all_runs.items():
        aggregate[dataset] = {
            "runs": runs,
            "mean": {
                label: {
                    metric: statistics.fmean(run[label][metric] for run in runs)
                    for metric in ("rmsd_lt2_percent", "pb_valid_percent", "joint_percent")
                }
                for label in ("baseline", "stereo_filtered")
            },
            "mean_delta_percentage_points": {
                metric: statistics.fmean(
                    run["delta_percentage_points"][metric] for run in runs
                )
                for metric in ("rmsd_lt2", "pb_valid", "joint")
            },
            "top_post_filter_pb_failures_across_repeats": failure_counts[dataset].most_common(),
            "fallback_ids_across_repeats": fallback_ids[dataset].most_common(),
        }

    report = {
        "schema_version": 1,
        "status": "complete",
        "protocol_id": PROTOCOL_ID,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "selection_information_boundary": "frozen input SMILES and U70k predicted RMSD only",
        "external_outcome_policy": "descriptive_only_not_selector_admission",
        "aggregate": aggregate,
    }
    json_path = args.output_root / "stereo_filter_selection_v4_report.json"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Input-SMILES Stereo-filter Selection Ablation",
        "",
        "The hard filter uses only stereochemistry explicitly present in the frozen input SMILES. ",
        "Official PoseBusters and crystal RMSD are evaluated only after selection. External outcomes ",
        "are descriptive and do not admit this selector as a production default.",
        "",
        "| Dataset | Selector | RMSD <2A (%) | PB-valid (%) | Joint (%) |",
        "|---|---|---:|---:|---:|",
    ]
    for dataset in EXPECTED:
        item = aggregate[dataset]
        for label in ("baseline", "stereo_filtered"):
            mean = item["mean"][label]
            lines.append(
                f"| {dataset} | {label} | {mean['rmsd_lt2_percent']:.2f} | "
                f"{mean['pb_valid_percent']:.2f} | {mean['joint_percent']:.2f} |"
            )
        delta = item["mean_delta_percentage_points"]
        lines.append(
            f"| {dataset} | delta (pp) | {delta['rmsd_lt2']:+.2f} | "
            f"{delta['pb_valid']:+.2f} | {delta['joint']:+.2f} |"
        )
    lines.extend(["", "## Per-repeat details", ""])
    for dataset in EXPECTED:
        lines.append(f"### {dataset}")
        lines.append("")
        for run in aggregate[dataset]["runs"]:
            lines.append(
                f"- repeat {run['repeat_index']}: changed={run['selection_changed_count']}, "
                f"fallback={run['fallback_count']}, delta={run['delta_percentage_points']}"
            )
        lines.append("")
    (args.output_root / "stereo_filter_selection_v4_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
