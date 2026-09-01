#!/usr/bin/env python3
"""Rescore one frozen temporal-external shard with U70k or U100k confidence."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from effdock.workflows.evaluate import file_sha256

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.rescore_external_temporal_u50_shard import (
    DATASETS,
    SCORE_PROTOCOL_ID,
    SOURCE_PROTOCOL_ID,
    read_json,
    score_complete,
    source_shard_path,
)

PROTOCOL_ID = "EFFDOCK-S50-RAW-REFINED-CONFIDENCE-TEMPORAL-EXTERNAL-V1"
ARMS = {"u070000", "u100000"}


def output_shard_path(
    root: Path,
    stage: str,
    arm: str,
    dataset: str,
    shard_index: int,
    num_shards: int,
) -> Path:
    name = f"{dataset}.shard-{shard_index:03d}-of-{num_shards:03d}.json"
    return root / stage / arm / "shards" / name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(DATASETS), required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--stage", choices=("smoke", "full"), required=True)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--external-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--docking-checkpoint", type=Path, required=True)
    parser.add_argument("--confidence-checkpoint", type=Path, required=True)
    parser.add_argument("--arm", choices=tuple(sorted(ARMS)), required=True)
    parser.add_argument("--expected-docking-sha256", required=True)
    parser.add_argument("--expected-confidence-sha256", required=True)
    args = parser.parse_args()

    expected_count, expected_shards = DATASETS[args.dataset]
    if args.num_shards != expected_shards or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard contract mismatch")
    if (args.stage == "smoke") != (args.max_records is not None):
        raise ValueError("smoke requires --max-records and full forbids it")
    if args.max_records is not None and args.max_records < 1:
        raise ValueError("--max-records must be positive")
    if file_sha256(args.docking_checkpoint) != args.expected_docking_sha256:
        raise ValueError("docking checkpoint SHA mismatch")
    if file_sha256(args.confidence_checkpoint) != args.expected_confidence_sha256:
        raise ValueError("confidence checkpoint SHA mismatch")

    source_path = source_shard_path(
        args.source_root.resolve(), args.dataset, args.shard_index, args.num_shards
    )
    source = read_json(source_path)
    if (
        source.get("protocol_id") != SOURCE_PROTOCOL_ID
        or source.get("status") != "complete"
        or source.get("dataset") != args.dataset
        or int(source.get("num_shards", -1)) != args.num_shards
        or int(source.get("shard_index", -1)) != args.shard_index
    ):
        raise ValueError(f"invalid source shard: {source_path}")
    records = list(source.get("records", []))
    assigned = len(range(args.shard_index, expected_count, args.num_shards))
    if len(records) != assigned:
        raise ValueError(f"source shard has {len(records)} records, expected {assigned}")
    if args.max_records is not None:
        records = records[: args.max_records]

    shard_output = output_shard_path(
        args.output_root.resolve(),
        args.stage,
        args.arm,
        args.dataset,
        args.shard_index,
        args.num_shards,
    )
    if shard_output.exists():
        raise FileExistsError(f"refusing to overwrite {shard_output}")

    started = time.monotonic()
    completed: list[dict[str, Any]] = []
    for position, record in enumerate(records, start=1):
        complex_id = str(record["id"])
        refinement_path = Path(str(record["refinement_summary"]))
        refinement_sha256 = str(record["refinement_summary_sha256"])
        if not refinement_path.is_file() or file_sha256(refinement_path) != refinement_sha256:
            raise ValueError(f"{complex_id}: changed refinement summary")

        output_dir = (
            args.output_root.resolve()
            / args.stage
            / args.arm
            / "confidence"
            / args.dataset
            / complex_id
        )
        score_path = output_dir / "summary.json"
        if output_dir.exists() and not score_complete(
            score_path,
            dataset=args.dataset,
            complex_id=complex_id,
            confidence_sha256=args.expected_confidence_sha256,
            docking_sha256=args.expected_docking_sha256,
            refinement_sha256=refinement_sha256,
        ):
            raise RuntimeError(f"{complex_id}: invalid pre-existing score output")
        if not output_dir.exists():
            subprocess.run(
                [
                    sys.executable,
                    "scripts/score_guidance_sdf_post_refinement_confidence.py",
                    "--refinement-summary",
                    str(refinement_path),
                    "--external-dir",
                    str(args.external_dir),
                    "--config",
                    str(args.config),
                    "--docking-checkpoint",
                    str(args.docking_checkpoint),
                    "--confidence-checkpoint",
                    str(args.confidence_checkpoint),
                    "--output-dir",
                    str(output_dir),
                    "--sigma",
                    "2",
                    "--pose-batch-size",
                    "20",
                    "--device",
                    "cuda",
                ],
                check=True,
            )
        if not score_complete(
            score_path,
            dataset=args.dataset,
            complex_id=complex_id,
            confidence_sha256=args.expected_confidence_sha256,
            docking_sha256=args.expected_docking_sha256,
            refinement_sha256=refinement_sha256,
        ):
            raise RuntimeError(f"{complex_id}: score output failed completion gate")
        confidence = read_json(score_path)
        if confidence.get("protocol_id") != SCORE_PROTOCOL_ID:
            raise ValueError(f"{complex_id}: unexpected scoring protocol")
        completed.append(
            {
                "id": complex_id,
                "refinement_summary": str(refinement_path.resolve()),
                "refinement_summary_sha256": refinement_sha256,
                "confidence_summary": str(score_path.resolve()),
                "confidence_summary_sha256": file_sha256(score_path),
                "confidence_checkpoint_sha256": args.expected_confidence_sha256,
                "raw_selected_rmsd": confidence["selected"]["step_000"][
                    "symmetry_rmsd_angstrom"
                ],
                "refined_selected_rmsd": confidence["selected"]["step_100"][
                    "symmetry_rmsd_angstrom"
                ],
            }
        )
        print(
            f"[{position}/{len(records)}] {args.arm} {args.dataset}/{complex_id}",
            flush=True,
        )

    summary = {
        "schema_version": "effdock.s50_raw_refined_confidence_temporal_external_shard.v1",
        "protocol_id": PROTOCOL_ID,
        "source_protocol_id": SOURCE_PROTOCOL_ID,
        "status": "complete",
        "stage": args.stage,
        "arm": args.arm,
        "dataset": args.dataset,
        "expected_dataset_count": expected_count,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "num_completed": len(completed),
        "selector": f"{args.arm} symmetry-confidence stable argmin predicted RMSD",
        "confidence_checkpoint_sha256": args.expected_confidence_sha256,
        "docking_checkpoint_sha256": args.expected_docking_sha256,
        "source_shard": {"path": str(source_path), "sha256": file_sha256(source_path)},
        "records": completed,
        "runtime": {
            "elapsed_seconds": time.monotonic() - started,
            "finished_at_utc": datetime.now(UTC).isoformat(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        },
    }
    shard_output.parent.mkdir(parents=True, exist_ok=True)
    temporary = shard_output.with_suffix(shard_output.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.link(temporary, shard_output)
    temporary.unlink()


if __name__ == "__main__":
    main()
