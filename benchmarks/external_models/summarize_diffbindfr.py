#!/usr/bin/env python3
"""Summarize native DiffBindFR outputs against the frozen denominator."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--expected-poses", type=int, required=True)
    parser.add_argument("--results-name", default="results.csv")
    parser.add_argument("--fail-on-incomplete", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.input_csv.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    experiment_dir = args.output_dir / args.experiment_name
    results_csv = experiment_dir / "results" / args.results_name
    result_rows = []
    if results_csv.is_file():
        with results_csv.open(newline="") as handle:
            result_rows = list(csv.DictReader(handle))
    result_rows_by_target: dict[str, list[dict[str, str]]] = {}
    for result_row in result_rows:
        result_rows_by_target.setdefault(result_row["complex_name"], []).append(result_row)

    coverage = {}
    for row in rows:
        target_id = row["complex_name"]
        target_rows = result_rows_by_target.get(target_id, [])
        poses = [
            Path(result_row["docked_lig"])
            for result_row in target_rows
            if result_row.get("docked_lig") and Path(result_row["docked_lig"]).is_file()
        ]
        coverage[target_id] = {
            "pose_count": len(poses),
            "results_row_count": len(target_rows),
            "has_mdn_score": bool(target_rows)
            and all(result_row.get("mdn_score", "") != "" for result_row in target_rows),
        }
    summary = {
        "schema_version": 1,
        "expected_targets": len(rows),
        "targets_with_any_pose": sum(item["pose_count"] > 0 for item in coverage.values()),
        "targets_with_expected_pose_count": sum(
            item["pose_count"] == args.expected_poses for item in coverage.values()
        ),
        "targets_with_mdn_scores": sum(item["has_mdn_score"] for item in coverage.values()),
        "results_csv": str(results_csv.resolve()),
        "coverage": coverage,
    }
    (args.output_dir / "coverage.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        f"DiffBindFR coverage: {summary['targets_with_any_pose']}/{summary['expected_targets']} "
        "targets have poses"
    )
    if (
        args.fail_on_incomplete
        and summary["targets_with_expected_pose_count"] != summary["expected_targets"]
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
