#!/usr/bin/env python3
"""Summarize SurfDock outputs against the complete frozen input denominator."""

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
    preprocess_path = args.output_dir / "preprocessed" / "preprocess_metadata.json"
    preprocess = json.loads(preprocess_path.read_text()) if preprocess_path.is_file() else {}
    coverage = {}
    for row in rows:
        target_id = row["complex_name"]
        pocket_stem = f"{target_id}_protein_processed_8A"
        ligand_stem = f"{target_id}_ligand"
        result_dir = args.output_dir / "SurfDock_docking_result" / f"{pocket_stem}_{ligand_stem}"
        poses = sorted(result_dir.glob("*_rank_*_confidence_*.sdf")) if result_dir.is_dir() else []
        coverage[target_id] = {
            "pose_count": len(poses),
            "result_dir": str(result_dir.resolve()),
            "surface_ready": preprocess.get("surface_status", {})
            .get(target_id, {})
            .get("complete", False),
            "embedding_ready": preprocess.get("embedding_status", {})
            .get(target_id, {})
            .get("complete", False),
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
        f"SurfDock coverage: {summary['targets_with_any_pose']}/{summary['expected_targets']} "
        "targets have poses"
    )
    if (
        args.fail_on_incomplete
        and summary["targets_with_expected_pose_count"] != summary["expected_targets"]
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
