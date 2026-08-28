#!/usr/bin/env python3
"""Compare official PoseBusters redock checks before/after SDF refinement."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import pandas as pd
from posebusters import PoseBusters

from effdock.workflows.posebusters_report import (
    VALIDITY_CHECKS,
    require_posebusters_runtime_version,
)

PROTOCOL_ID = "EFFDOCK-GUIDANCE-SDF-POST-REFINEMENT-PB-V1"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_pb_row(raw: dict[str, Any], *, label: str) -> tuple[str, dict[str, bool]]:
    rmsd_checks = [key for key in raw if str(key).startswith("rmsd_")]
    if len(rmsd_checks) != 1:
        raise ValueError(f"{label}: expected exactly one RMSD check, got {rmsd_checks}")
    rmsd_check = rmsd_checks[0]
    expected = {*VALIDITY_CHECKS, rmsd_check}
    if set(raw) != expected:
        raise ValueError(
            f"{label}: PoseBusters schema mismatch; "
            f"missing={sorted(expected - set(raw))}, extra={sorted(set(raw) - expected)}"
        )
    checks = {key: False if pd.isna(value) else bool(value) for key, value in raw.items()}
    return rmsd_check, checks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refinement-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")

    refinement = json.loads(args.refinement_summary.read_text(encoding="utf-8"))
    if refinement.get("protocol_id") != "EFFDOCK-GUIDANCE-SDF-POST-REFINEMENT-V1":
        raise ValueError("unexpected refinement protocol")
    if refinement.get("status") != "complete_descriptive":
        raise ValueError("refinement is not complete")
    inputs = refinement["inputs"]
    artifacts = refinement["artifacts"]
    stages = {
        "step_000": (Path(artifacts["step_000_sdf"]["path"]), artifacts["step_000_sdf"]["sha256"]),
        "step_100": (Path(artifacts["step_100_sdf"]["path"]), artifacts["step_100_sdf"]["sha256"]),
    }
    ligand_ref = Path(inputs["ligand_reference"])
    protein = Path(inputs["protein"])
    assets = [
        (ligand_ref, inputs["ligand_reference_sha256"]),
        (protein, inputs["protein_sha256"]),
        *stages.values(),
    ]
    for path, expected_hash in assets:
        if not path.is_file() or file_sha256(path) != expected_hash:
            raise ValueError(f"missing or changed input: {path}")

    observed_version = require_posebusters_runtime_version()
    if observed_version != "0.6.5":
        raise RuntimeError(f"expected PoseBusters 0.6.5, got {observed_version}")
    pose_count = int(refinement["counts"]["poses"])
    buster = PoseBusters(
        config="redock",
        max_workers=args.workers,
        chunk_size=math.ceil(pose_count / args.workers),
    )
    started = time.monotonic()
    stage_rows: dict[str, list[dict[str, Any]]] = {}
    rmsd_check_name: str | None = None
    for stage, (sdf_path, _) in stages.items():
        frame = buster.bust(sdf_path, ligand_ref, protein, full_report=False)
        if len(frame.index) != pose_count:
            raise ValueError(f"{stage}: expected {pose_count} rows, got {len(frame.index)}")
        rows: list[dict[str, Any]] = []
        for pose_index, (_, frame_row) in enumerate(frame.iterrows()):
            rmsd_check, checks = validate_pb_row(frame_row.to_dict(), label=f"{stage}/{pose_index}")
            if rmsd_check_name is None:
                rmsd_check_name = rmsd_check
            elif rmsd_check_name != rmsd_check:
                raise ValueError("PoseBusters RMSD check changed between stages")
            rows.append(
                {
                    "pose_index": pose_index,
                    "posebusters_valid": all(checks[key] for key in VALIDITY_CHECKS),
                    "separate_rmsd_check": checks[rmsd_check],
                    **{key: checks[key] for key in VALIDITY_CHECKS},
                }
            )
        stage_rows[stage] = rows
        print(f"{stage}: PB-valid={sum(row['posebusters_valid'] for row in rows)}/{pose_count}", flush=True)

    assert rmsd_check_name is not None
    combined: list[dict[str, Any]] = []
    for pose_index in range(pose_count):
        before = stage_rows["step_000"][pose_index]
        after = stage_rows["step_100"][pose_index]
        combined.append(
            {
                "pose_index": pose_index,
                **{f"before_{key}": value for key, value in before.items() if key != "pose_index"},
                **{f"after_{key}": value for key, value in after.items() if key != "pose_index"},
            }
        )

    before_valid = [row["posebusters_valid"] for row in stage_rows["step_000"]]
    after_valid = [row["posebusters_valid"] for row in stage_rows["step_100"]]
    per_check = {
        check: {
            "before_pass_count": sum(row[check] for row in stage_rows["step_000"]),
            "after_pass_count": sum(row[check] for row in stage_rows["step_100"]),
        }
        for check in VALIDITY_CHECKS
    }
    summary = {
        "schema_version": "effdock.guidance_sdf_post_refinement_pb.v1",
        "protocol_id": PROTOCOL_ID,
        "status": "complete_descriptive",
        "claim_boundary": "official PoseBusters 0.6.5 redock comparison for one frozen smoke complex",
        "complex_id": inputs["complex_id"],
        "dataset": inputs["dataset"],
        "pose_count": pose_count,
        "posebusters_version": observed_version,
        "posebusters_config": "redock",
        "validity_definition": "all 27 non-RMSD redock checks",
        "validity_checks": list(VALIDITY_CHECKS),
        "separate_rmsd_check": rmsd_check_name,
        "before_valid_count": sum(before_valid),
        "after_valid_count": sum(after_valid),
        "invalid_to_valid_count": sum((not b) and a for b, a in zip(before_valid, after_valid)),
        "valid_to_invalid_count": sum(b and (not a) for b, a in zip(before_valid, after_valid)),
        "per_check": per_check,
        "runtime": {
            "elapsed_seconds": time.monotonic() - started,
            "workers": args.workers,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
        },
        "inputs": {
            "refinement_summary": str(args.refinement_summary.resolve()),
            "refinement_summary_sha256": file_sha256(args.refinement_summary),
            **{
                stage: {"path": str(path.resolve()), "sha256": expected_hash}
                for stage, (path, expected_hash) in stages.items()
            },
        },
    }

    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    attempt = Path(tempfile.mkdtemp(prefix=f".{args.output_dir.name}.attempt-", dir=args.output_dir.parent))
    with (attempt / "poses.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(combined[0]), extrasaction="raise")
        writer.writeheader()
        writer.writerows(combined)
    (attempt / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.rename(attempt, args.output_dir)
    print(json.dumps({
        "before_valid": sum(before_valid),
        "after_valid": sum(after_valid),
        "invalid_to_valid": summary["invalid_to_valid_count"],
        "valid_to_invalid": summary["valid_to_invalid_count"],
        "output_dir": str(args.output_dir.resolve()),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
