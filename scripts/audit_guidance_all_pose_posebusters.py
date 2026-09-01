#!/usr/bin/env python3
"""Fail-closed audit for all-pose official PoseBusters shards."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from effdock.workflows.guidance_budget_posebusters_report import VALIDITY_CHECKS

PROTOCOL_ID = "EFFDOCK-GUIDANCE-ALL-POSE-PB-ETA-V1"
SMOKE_IDS = {"astex": "1jje", "posebusters": "7b2c_tp7"}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def truth(value: str, *, label: str) -> bool:
    normalized = value.lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"{label}: expected boolean, got {value!r}")


def selected_records(manifest: dict[str, Any], mode: str) -> list[dict[str, Any]]:
    rows = manifest["records"]
    if mode == "smoke":
        return [row for row in rows if row["id"] == SMOKE_IDS[row["dataset"]]]
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if file_sha256(args.manifest) != args.manifest_sha256:
        raise ValueError("manifest SHA-256 mismatch")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    selected = selected_records(manifest, args.mode)
    expected_keys = {(row["dataset"], float(row["eta"]), row["id"]) for row in selected}
    observed_cells: set[tuple[str, float, str]] = set()
    observed_poses: set[tuple[str, float, str, int]] = set()
    shard_records: list[dict[str, Any]] = []
    valid_poses = 0
    rmsd_check: str | None = None
    for shard in range(args.num_shards):
        shard_dir = args.output_root / args.mode / f"shard-{shard:03d}-of-{args.num_shards:03d}"
        summary_path = shard_dir / "summary.json"
        cells_path = shard_dir / "cells.csv"
        poses_path = shard_dir / "poses.csv.gz"
        for path in (summary_path, cells_path, poses_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        expected_summary = {
            "schema_version": "effdock.guidance_all_pose_pb_shard.v1",
            "protocol_id": PROTOCOL_ID,
            "status": "complete",
            "mode": args.mode,
            "manifest_sha256": args.manifest_sha256,
            "posebusters_version": "0.6.5",
            "posebusters_config": "redock",
            "validity_checks": list(VALIDITY_CHECKS),
            "num_shards": args.num_shards,
            "shard_index": shard,
            "workers": 4,
            "posebusters_chunk_size": 25,
        }
        for key, expected in expected_summary.items():
            if summary.get(key) != expected:
                raise ValueError(f"{summary_path}: {key} mismatch")
        if rmsd_check is None:
            rmsd_check = str(summary["rmsd_check"])
        elif summary["rmsd_check"] != rmsd_check:
            raise ValueError("RMSD check changed across shards")
        runtime = summary.get("runtime") or {}
        if runtime.get("slurm_partition") not in {"test", "cpu_only"}:
            raise ValueError(f"{summary_path}: unexpected partition")
        if int(runtime.get("cpus_per_task", 0)) != 4:
            raise ValueError(f"{summary_path}: expected four CPUs")
        artifacts = summary.get("artifacts") or {}
        expected_artifacts = {
            "poses_csv_gz": str(poses_path),
            "poses_csv_gz_sha256": file_sha256(poses_path),
            "cells_csv": str(cells_path),
            "cells_csv_sha256": file_sha256(cells_path),
            "summary": str(summary_path),
        }
        for key, expected in expected_artifacts.items():
            if artifacts.get(key) != expected:
                raise ValueError(f"{summary_path}: artifact {key} mismatch")
        with cells_path.open(newline="", encoding="utf-8") as handle:
            cell_rows = list(csv.DictReader(handle))
        assigned = selected[shard::args.num_shards]
        assigned_keys = [(row["dataset"], float(row["eta"]), row["id"]) for row in assigned]
        cell_keys = [(row["dataset"], float(row["eta"]), row["id"]) for row in cell_rows]
        if cell_keys != assigned_keys:
            raise ValueError(f"{cells_path}: exact cell assignment mismatch")
        if len(cell_rows) != int(summary["result_cells"]):
            raise ValueError(f"{cells_path}: cell count mismatch")
        for row in cell_rows:
            key = (row["dataset"], float(row["eta"]), row["id"])
            if key in observed_cells:
                raise ValueError(f"duplicate cell {key}")
            observed_cells.add(key)
            if int(row["pose_count"]) != 100 or not 0 <= int(row["valid_count"]) <= 100:
                raise ValueError(f"{key}: invalid cell counts")
            if float(row["valid_pct"]) != int(row["valid_count"]):
                raise ValueError(f"{key}: valid percentage mismatch")
        shard_pose_count = 0
        shard_valid = 0
        with gzip.open(poses_path, "rt", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                key = (row["dataset"], float(row["eta"]), row["id"], int(row["pose_index"]))
                if key in observed_poses:
                    raise ValueError(f"duplicate pose {key}")
                observed_poses.add(key)
                valid = truth(row["posebusters_valid"], label=str(key))
                checks = [truth(row[check], label=f"{key}/{check}") for check in VALIDITY_CHECKS]
                if valid != all(checks):
                    raise ValueError(f"{key}: validity differs from 27 checks")
                truth(row["separate_rmsd_check"], label=f"{key}/rmsd")
                shard_pose_count += 1
                shard_valid += int(valid)
        if shard_pose_count != int(summary["result_poses"]):
            raise ValueError(f"{poses_path}: pose count mismatch")
        if shard_valid != int(summary["valid_poses"]):
            raise ValueError(f"{poses_path}: valid count mismatch")
        valid_poses += shard_valid
        shard_records.append(
            {
                "shard_index": shard,
                "cells": len(cell_rows),
                "poses": shard_pose_count,
                "valid_poses": shard_valid,
                "summary_sha256": file_sha256(summary_path),
            }
        )
    if observed_cells != expected_keys:
        raise ValueError("global cell inventory mismatch")
    expected_pose_count = len(expected_keys) * 100
    if len(observed_poses) != expected_pose_count:
        raise ValueError("global pose inventory mismatch")
    result = {
        "schema_version": "effdock.guidance_all_pose_pb_audit.v1",
        "protocol_id": PROTOCOL_ID,
        "status": "passed",
        "mode": args.mode,
        "manifest": str(args.manifest),
        "manifest_sha256": args.manifest_sha256,
        "posebusters_version": "0.6.5",
        "validity_definition": "all 27 non-RMSD redock checks",
        "validity_checks": list(VALIDITY_CHECKS),
        "rmsd_check": rmsd_check,
        "num_shards": args.num_shards,
        "cells": len(observed_cells),
        "poses": len(observed_poses),
        "valid_poses": valid_poses,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "shards": shard_records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "passed", "cells": len(observed_cells), "poses": len(observed_poses)}, sort_keys=True))


if __name__ == "__main__":
    main()
