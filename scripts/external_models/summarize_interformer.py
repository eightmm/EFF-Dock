#!/usr/bin/env python3
"""Summarize sampled Interformer poses while excluding the appended input pose."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-poses", type=int, required=True)
    parser.add_argument("--fail-on-incomplete", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.input_csv.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    preprocess_path = args.output_dir / "work" / "preprocess_metadata.json"
    preprocess = json.loads(preprocess_path.read_text())
    target_to_alias = {target_id: alias for alias, target_id in preprocess["aliases"].items()}
    coverage = {}
    for row in rows:
        target_id = row["complex_name"]
        alias = target_to_alias[target_id]
        stat_csv = (
            args.output_dir / "energy" / "ligand_reconstructing" / f"{alias}_docked.sdf_stat.csv"
        )
        sampled_rows = []
        all_rows = []
        if stat_csv.is_file():
            with stat_csv.open(newline="") as handle:
                all_rows = list(csv.DictReader(handle))
            sampled_rows = [
                result_row
                for result_row in all_rows
                if int(float(result_row["pose_rank"])) < args.expected_poses
            ]
        output_sdf = args.output_dir / "energy" / "ligand_reconstructing" / f"{alias}_docked.sdf"
        coverage[target_id] = {
            "pose_count": len(sampled_rows),
            "total_sdf_record_count_from_stat": len(all_rows),
            "appended_input_pose_excluded": len(all_rows) > len(sampled_rows),
            "output_sdf": str(output_sdf.resolve()),
        }
    summary = {
        "schema_version": 1,
        "expected_targets": len(rows),
        "targets_with_any_pose": sum(item["pose_count"] > 0 for item in coverage.values()),
        "targets_with_expected_pose_count": sum(
            item["pose_count"] == args.expected_poses for item in coverage.values()
        ),
        "coverage": coverage,
    }
    (args.output_dir / "coverage.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        f"Interformer coverage: {summary['targets_with_any_pose']}/{summary['expected_targets']} "
        "targets have sampled poses"
    )
    if (
        args.fail_on_incomplete
        and summary["targets_with_expected_pose_count"] != summary["expected_targets"]
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
