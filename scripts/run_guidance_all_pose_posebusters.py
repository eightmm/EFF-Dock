#!/usr/bin/env python3
"""Run official PoseBusters on frozen all-pose eta cells for one shard."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from posebusters import PoseBusters

from effdock.workflows.guidance_budget_posebusters_report import VALIDITY_CHECKS
from effdock.workflows.posebusters_report import require_posebusters_runtime_version

PROTOCOL_ID = "EFFDOCK-GUIDANCE-ALL-POSE-PB-ETA-V1"
SMOKE_IDS = {"astex": "1jje", "posebusters": "7b2c_tp7"}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_frame_row(raw: dict[str, Any], *, label: str) -> tuple[str, dict[str, bool]]:
    rmsd_checks = [key for key in raw if str(key).startswith("rmsd_")]
    if len(rmsd_checks) != 1:
        raise ValueError(f"{label}: expected exactly one separate RMSD check")
    rmsd_check = rmsd_checks[0]
    expected = {*VALIDITY_CHECKS, rmsd_check}
    if set(raw) != expected:
        raise ValueError(
            f"{label}: official redock schema mismatch; "
            f"missing={sorted(expected - set(raw))}, extra={sorted(set(raw) - expected)}"
        )
    return rmsd_check, {
        key: False if pd.isna(value) else bool(value) for key, value in raw.items()
    }


def _selected_records(manifest: dict[str, Any], *, mode: str, num_shards: int, shard_index: int) -> list[dict[str, Any]]:
    records = manifest.get("records")
    if not isinstance(records, list):
        raise ValueError("manifest records are missing")
    if mode == "smoke":
        selected = [row for row in records if row.get("id") == SMOKE_IDS.get(str(row.get("dataset")))]
    elif mode == "full":
        selected = records
    else:
        raise ValueError(f"unknown mode: {mode}")
    assigned = selected[shard_index::num_shards]
    if not assigned:
        raise ValueError("empty PoseBusters shard assignment")
    return assigned


def _write_csv_gz(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with gzip.open(path, "wt", newline="", encoding="utf-8", compresslevel=6) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid shard contract")
    if args.workers < 1:
        raise ValueError("workers must be positive")
    if file_sha256(args.manifest) != args.manifest_sha256:
        raise ValueError("manifest SHA-256 mismatch")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    expected_manifest = {
        "schema_version": "effdock.guidance_all_pose_pb_manifest.v1",
        "protocol_id": PROTOCOL_ID,
        "posebusters_version": "0.6.5",
        "posebusters_config": "redock",
        "poses_per_cell": 100,
        "expected_cells": 2751,
        "expected_poses": 275100,
    }
    for key, expected in expected_manifest.items():
        if manifest.get(key) != expected:
            raise ValueError(f"manifest {key} mismatch")
    records = _selected_records(
        manifest, mode=args.mode, num_shards=args.num_shards, shard_index=args.shard_index
    )
    final_dir = args.output_root / args.mode / f"shard-{args.shard_index:03d}-of-{args.num_shards:03d}"
    if final_dir.exists():
        raise FileExistsError(f"refusing to overwrite completed shard: {final_dir}")
    incomplete = args.output_root / args.mode / ".incomplete"
    incomplete.mkdir(parents=True, exist_ok=True)
    attempt = Path(tempfile.mkdtemp(prefix=f"{final_dir.name}.attempt-", dir=incomplete))
    started = time.monotonic()
    observed_version = require_posebusters_runtime_version()
    if observed_version != "0.6.5":
        raise RuntimeError(f"expected PoseBusters 0.6.5, got {observed_version}")
    # PoseBusters parallelizes over pose chunks.  Its default chunk size of
    # 100 would put this entire 100-pose cell in one process even when four
    # workers were requested.
    buster = PoseBusters(
        config="redock",
        max_workers=args.workers,
        chunk_size=math.ceil(100 / args.workers),
    )
    pose_rows: list[dict[str, Any]] = []
    cell_rows: list[dict[str, Any]] = []
    rmsd_check_name: str | None = None
    for cell_index, record in enumerate(records, start=1):
        dataset = str(record["dataset"])
        eta = float(record["eta"])
        complex_id = str(record["id"])
        pose_path = Path(record["pose_path"])
        protein = Path(record["protein"])
        ligand_ref = Path(record["ligand_ref"])
        assets = (
            (pose_path, str(record["pose_sha256"])),
            (protein, str(record["protein_sha256"])),
            (ligand_ref, str(record["ligand_ref_sha256"])),
        )
        for path, expected_hash in assets:
            if not path.is_file() or file_sha256(path) != expected_hash:
                raise ValueError(f"{dataset}/{eta}/{complex_id}: changed asset {path}")
        frame = buster.bust(pose_path, ligand_ref, protein, full_report=False)
        if len(frame.index) != 100:
            raise ValueError(
                f"{dataset}/{eta}/{complex_id}: expected 100 PB rows, got {len(frame.index)}"
            )
        check_counts = {check: 0 for check in VALIDITY_CHECKS}
        valid_count = 0
        for pose_index, (_, frame_row) in enumerate(frame.iterrows()):
            current_rmsd, checks = _validate_frame_row(
                frame_row.to_dict(), label=f"{dataset}/{eta}/{complex_id}/pose-{pose_index}"
            )
            if rmsd_check_name is None:
                rmsd_check_name = current_rmsd
            elif current_rmsd != rmsd_check_name:
                raise ValueError("PoseBusters RMSD column changed within the run")
            valid = all(checks[check] for check in VALIDITY_CHECKS)
            valid_count += int(valid)
            for check in VALIDITY_CHECKS:
                check_counts[check] += int(checks[check])
            pose_rows.append(
                {
                    "dataset": dataset,
                    "eta": eta,
                    "id": complex_id,
                    "pose_index": pose_index,
                    "posebusters_valid": valid,
                    "separate_rmsd_check": checks[current_rmsd],
                    **{check: checks[check] for check in VALIDITY_CHECKS},
                }
            )
        cell_rows.append(
            {
                "dataset": dataset,
                "eta": eta,
                "id": complex_id,
                "pose_count": 100,
                "valid_count": valid_count,
                "valid_pct": valid_count,
                **{f"{check}_pass_count": check_counts[check] for check in VALIDITY_CHECKS},
            }
        )
        print(
            f"[{cell_index:03d}/{len(records):03d}] {dataset} eta={eta:g} "
            f"{complex_id} PB-valid={valid_count}/100",
            flush=True,
        )
    if rmsd_check_name is None:
        raise RuntimeError("no PoseBusters results were produced")
    pose_fields = [
        "dataset", "eta", "id", "pose_index", "posebusters_valid",
        "separate_rmsd_check", *VALIDITY_CHECKS,
    ]
    cell_fields = [
        "dataset", "eta", "id", "pose_count", "valid_count", "valid_pct",
        *[f"{check}_pass_count" for check in VALIDITY_CHECKS],
    ]
    pose_path = attempt / "poses.csv.gz"
    cell_path = attempt / "cells.csv"
    _write_csv_gz(pose_path, pose_rows, pose_fields)
    with cell_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=cell_fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(cell_rows)
    summary = {
        "schema_version": "effdock.guidance_all_pose_pb_shard.v1",
        "protocol_id": PROTOCOL_ID,
        "status": "complete",
        "mode": args.mode,
        "manifest": str(args.manifest),
        "manifest_sha256": args.manifest_sha256,
        "posebusters_version": observed_version,
        "posebusters_config": "redock",
        "validity_definition": "all 27 non-RMSD redock checks",
        "validity_checks": list(VALIDITY_CHECKS),
        "rmsd_check": rmsd_check_name,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "assigned_cells": len(records),
        "assigned_poses": len(records) * 100,
        "result_cells": len(cell_rows),
        "result_poses": len(pose_rows),
        "valid_poses": sum(int(row["posebusters_valid"]) for row in pose_rows),
        "workers": args.workers,
        "posebusters_chunk_size": math.ceil(100 / args.workers),
        "runtime": {
            "elapsed_seconds": time.monotonic() - started,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
            "cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
            "finished_at_utc": datetime.now(UTC).isoformat(),
        },
        "artifacts": {
            "poses_csv_gz": str(final_dir / "poses.csv.gz"),
            "poses_csv_gz_sha256": file_sha256(pose_path),
            "cells_csv": str(final_dir / "cells.csv"),
            "cells_csv_sha256": file_sha256(cell_path),
            "summary": str(final_dir / "summary.json"),
        },
    }
    (attempt / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    os.rename(attempt, final_dir)
    print(json.dumps({"status": "complete", "cells": len(cell_rows), "poses": len(pose_rows)}, sort_keys=True))


if __name__ == "__main__":
    main()
