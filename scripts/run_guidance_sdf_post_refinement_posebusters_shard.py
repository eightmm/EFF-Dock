#!/usr/bin/env python3
"""Run official PoseBusters 0.6.5 on refined step-100 poses for one shard."""

from __future__ import annotations

import argparse
import csv
import gzip
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

from effdock.workflows.evaluate import file_sha256
from effdock.workflows.guidance_budget_posebusters_report import VALIDITY_CHECKS
from effdock.workflows.guidance_pl_valid import PL_VALIDITY_CHECKS, is_pl_valid
from effdock.workflows.posebusters_report import require_posebusters_runtime_version

PROTOCOL_ID = "EFFDOCK-GUIDANCE-SDF-POST-REFINEMENT-PB-V1"
EXPECTED_COUNTS = {"astex": 85, "posebusters": 308}


def _records(
    manifest: dict[str, Any],
    shard_index: int,
    num_shards: int,
    *,
    eta: float,
) -> list[dict[str, Any]]:
    records = sorted(
        (
            row
            for row in manifest.get("records", [])
            if float(row.get("eta", float("nan"))) == eta
        ),
        key=lambda row: (str(row["dataset"]), str(row["id"])),
    )
    counts = {dataset: sum(row["dataset"] == dataset for row in records) for dataset in EXPECTED_COUNTS}
    if counts != EXPECTED_COUNTS:
        raise ValueError(f"unexpected eta={eta:g} inventory: {counts}")
    assigned = records[shard_index::num_shards]
    if not assigned:
        raise ValueError("empty PoseBusters shard")
    return assigned


def _validated_checks(raw: dict[str, Any], label: str) -> tuple[str, dict[str, bool]]:
    rmsd = [key for key in raw if str(key).startswith("rmsd_")]
    if len(rmsd) != 1 or set(raw) != {*VALIDITY_CHECKS, rmsd[0]}:
        raise ValueError(f"{label}: unexpected PoseBusters redock schema")
    return rmsd[0], {
        key: False if pd.isna(value) else bool(value) for key, value in raw.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--eta", type=float, default=0.0)
    args = parser.parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid shard contract")
    if args.workers < 1:
        raise ValueError("workers must be positive")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    records = _records(
        manifest,
        args.shard_index,
        args.num_shards,
        eta=args.eta,
    )
    final_dir = args.output_root / f"shard-{args.shard_index:03d}-of-{args.num_shards:03d}"
    if final_dir.exists():
        raise FileExistsError(f"refusing to overwrite {final_dir}")
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    incomplete = args.output_root / ".incomplete"
    incomplete.mkdir(parents=True, exist_ok=True)
    attempt = Path(tempfile.mkdtemp(prefix=f"{final_dir.name}.attempt-", dir=incomplete))
    version = require_posebusters_runtime_version()
    if version != "0.6.5":
        raise RuntimeError(f"expected PoseBusters 0.6.5, got {version}")
    buster = PoseBusters(
        config="redock",
        max_workers=args.workers,
        chunk_size=math.ceil(100 / args.workers),
    )
    started = time.monotonic()
    pose_rows: list[dict[str, Any]] = []
    cell_rows: list[dict[str, Any]] = []
    rmsd_name: str | None = None
    for cell_index, record in enumerate(records, start=1):
        dataset = str(record["dataset"])
        complex_id = str(record["id"])
        refinement_summary_path = args.input_root / "refinement" / dataset / complex_id / "summary.json"
        refinement = json.loads(refinement_summary_path.read_text(encoding="utf-8"))
        if (
            refinement.get("protocol_id") != "EFFDOCK-GUIDANCE-SDF-POST-REFINEMENT-V1"
            or refinement.get("status") != "complete_descriptive"
            or int(refinement.get("counts", {}).get("failed", -1)) != 0
        ):
            raise ValueError(f"{dataset}/{complex_id}: invalid refinement summary")
        step_spec = refinement["artifacts"]["step_100_sdf"]
        step_path = Path(step_spec["path"])
        assets = (
            (step_path, step_spec["sha256"]),
            (Path(record["protein"]), record["protein_sha256"]),
            (Path(record["ligand_ref"]), record["ligand_ref_sha256"]),
        )
        for path, expected in assets:
            if not path.is_file() or file_sha256(path) != expected:
                raise ValueError(f"{dataset}/{complex_id}: missing or changed input {path}")
        frame = buster.bust(
            step_path,
            Path(record["ligand_ref"]),
            Path(record["protein"]),
            full_report=False,
        )
        if len(frame.index) != 100:
            raise ValueError(f"{dataset}/{complex_id}: expected 100 PoseBusters rows")
        pl_count = official_count = rmsd_count = 0
        for pose_index, (_, raw_row) in enumerate(frame.iterrows()):
            current_rmsd, checks = _validated_checks(
                raw_row.to_dict(), f"{dataset}/{complex_id}/{pose_index}"
            )
            if rmsd_name is None:
                rmsd_name = current_rmsd
            elif rmsd_name != current_rmsd:
                raise ValueError("PoseBusters RMSD column changed within shard")
            pl_valid = is_pl_valid(checks)
            official_valid = all(checks[key] for key in VALIDITY_CHECKS)
            pl_count += int(pl_valid)
            official_count += int(official_valid)
            rmsd_count += int(checks[current_rmsd])
            pose_rows.append(
                {
                    "dataset": dataset,
                    "id": complex_id,
                    "pose_index": pose_index,
                    "pl_valid": pl_valid,
                    "posebusters_valid": official_valid,
                    "separate_rmsd_check": checks[current_rmsd],
                    **{key: checks[key] for key in VALIDITY_CHECKS},
                }
            )
        cell_rows.append(
            {
                "dataset": dataset,
                "id": complex_id,
                "pose_count": 100,
                "pl_valid_count": pl_count,
                "posebusters_valid_count": official_count,
                "separate_rmsd_count": rmsd_count,
            }
        )
        print(
            f"[{cell_index}/{len(records)}] {dataset}/{complex_id} "
            f"PL={pl_count}/100 official={official_count}/100",
            flush=True,
        )
    assert rmsd_name is not None
    pose_fields = [
        "dataset", "id", "pose_index", "pl_valid", "posebusters_valid",
        "separate_rmsd_check", *VALIDITY_CHECKS,
    ]
    with gzip.open(attempt / "poses.csv.gz", "wt", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=pose_fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(pose_rows)
    with (attempt / "cells.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cell_rows[0]), extrasaction="raise")
        writer.writeheader()
        writer.writerows(cell_rows)
    summary = {
        "protocol_id": PROTOCOL_ID,
        "status": "complete",
        "posebusters_version": version,
        "posebusters_config": "redock",
        "validity_checks": list(VALIDITY_CHECKS),
        "pl_validity_checks": list(PL_VALIDITY_CHECKS),
        "rmsd_check": rmsd_name,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "assigned_cells": len(records),
        "result_poses": len(pose_rows),
        "runtime": {
            "elapsed_seconds": time.monotonic() - started,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "finished_at_utc": datetime.now(UTC).isoformat(),
        },
    }
    (attempt / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.rename(attempt, final_dir)


if __name__ == "__main__":
    main()
