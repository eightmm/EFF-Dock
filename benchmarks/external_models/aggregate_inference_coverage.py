#!/usr/bin/env python3
"""Aggregate per-shard external-inference coverage without dropping failures."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-csv", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--allow-native-unsupported", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.expected_csv.open(newline="") as handle:
        expected_ids = [row["complex_name"] for row in csv.DictReader(handle)]
    if len(set(expected_ids)) != len(expected_ids):
        raise ValueError("Expected manifest contains duplicate complex_name values")

    coverage_files = sorted(args.run_root.glob("**/coverage.json"))
    observed: list[str] = []
    targets_with_pose: set[str] = set()
    targets_with_expected_pose_count: set[str] = set()
    targets_native_unsupported: set[str] = set()
    per_target: dict[str, dict] = {}
    for coverage_file in coverage_files:
        payload = json.loads(coverage_file.read_text())
        metadata_file = coverage_file.with_name("run_metadata.json")
        metadata = json.loads(metadata_file.read_text()) if metadata_file.is_file() else {}
        expected_pose_count = (
            metadata.get("samples_per_complex")
            or payload.get("expected_poses_per_target")
            or payload.get("vina_num_modes_requested")
        )
        for target_id, target_payload in payload.get("coverage", {}).items():
            observed.append(target_id)
            if target_id in per_target:
                continue
            per_target[target_id] = {
                **target_payload,
                "coverage_file": str(coverage_file.resolve()),
            }
            pose_count = target_payload.get(
                "selected_pose_count",
                target_payload.get(
                    "pose_count", target_payload.get("pose_file_count", 0)
                ),
            )
            if pose_count > 0:
                targets_with_pose.add(target_id)
            if expected_pose_count is not None and pose_count == expected_pose_count:
                targets_with_expected_pose_count.add(target_id)
            if target_payload.get("status") == "native_unsupported":
                targets_native_unsupported.add(target_id)

    observed_counts = Counter(observed)
    expected_set = set(expected_ids)
    observed_set = set(observed)
    summary = {
        "schema_version": 1,
        "expected_manifest": str(args.expected_csv.resolve()),
        "run_root": str(args.run_root.resolve()),
        "coverage_files": [str(path.resolve()) for path in coverage_files],
        "expected_targets": len(expected_ids),
        "observed_targets": len(observed_set & expected_set),
        "targets_with_any_pose": len(targets_with_pose & expected_set),
        "targets_with_expected_pose_count": len(targets_with_expected_pose_count & expected_set),
        "targets_native_unsupported": len(targets_native_unsupported & expected_set),
        "targets_with_terminal_outcome": len(
            (
                targets_with_expected_pose_count
                | (targets_native_unsupported if args.allow_native_unsupported else set())
            )
            & expected_set
        ),
        "allow_native_unsupported": args.allow_native_unsupported,
        "missing_targets": sorted(expected_set - observed_set),
        "unexpected_targets": sorted(observed_set - expected_set),
        "duplicate_targets": sorted(
            target_id for target_id, count in observed_counts.items() if count > 1
        ),
        "per_target": {target_id: per_target.get(target_id) for target_id in expected_ids},
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2) + "\n")
    print(
        f"Coverage {summary['targets_with_any_pose']}/{summary['expected_targets']} "
        f"from {len(coverage_files)} shard files"
    )

    complete = (
        summary["targets_with_terminal_outcome"] == summary["expected_targets"]
        and not summary["unexpected_targets"]
        and not summary["duplicate_targets"]
    )
    if args.strict and not complete:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
