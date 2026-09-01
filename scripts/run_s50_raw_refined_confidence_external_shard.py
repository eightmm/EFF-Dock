#!/usr/bin/env python3
"""Rescore one frozen raw/refined external shard with U70k or U100k."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from effdock.workflows.evaluate import file_sha256

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_s50_symmetry_confidence_refined_external_shard import (
    _complete,
    _records,
)

PROTOCOL_ID = "EFFDOCK-S50-RAW-REFINED-CONFIDENCE-EXTERNAL-V1"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--refinement-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--benchmark-input-manifest", type=Path, required=True)
    parser.add_argument("--external-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--docking-checkpoint", type=Path, required=True)
    parser.add_argument("--confidence-checkpoint", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--expected-confidence-sha256", required=True)
    parser.add_argument("--expected-docking-sha256", required=True)
    parser.add_argument("--num-shards", type=int, default=32)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.num_shards != 32 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("the frozen full contract is 32 shards")
    if args.arm not in {"u070000", "u100000"}:
        raise ValueError(f"unexpected arm: {args.arm}")
    if file_sha256(args.confidence_checkpoint) != args.expected_confidence_sha256:
        raise ValueError("confidence checkpoint hash mismatch")
    if file_sha256(args.docking_checkpoint) != args.expected_docking_sha256:
        raise ValueError("docking checkpoint hash mismatch")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    records = _records(
        manifest,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
        smoke=args.smoke,
    )
    started = time.monotonic()
    completed: list[dict[str, str]] = []
    for index, record in enumerate(records, start=1):
        dataset, complex_id = str(record["dataset"]), str(record["id"])
        refinement_summary = (
            args.refinement_root / "refinement" / dataset / complex_id / "summary.json"
        )
        output_dir = args.output_root / args.arm / dataset / complex_id
        score_summary = output_dir / "summary.json"
        if output_dir.exists() and not _complete(
            score_summary,
            dataset=dataset,
            complex_id=complex_id,
            confidence_sha256=args.expected_confidence_sha256,
            docking_sha256=args.expected_docking_sha256,
        ):
            raise RuntimeError(f"invalid pre-existing output: {output_dir}")
        if not output_dir.exists():
            subprocess.run(
                [
                    sys.executable,
                    "scripts/score_guidance_sdf_post_refinement_confidence.py",
                    "--refinement-summary",
                    str(refinement_summary),
                    "--benchmark-input-manifest",
                    str(args.benchmark_input_manifest),
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
        completed.append({"dataset": dataset, "id": complex_id})
        print(f"[{index}/{len(records)}] {args.arm} {dataset}/{complex_id}", flush=True)

    shard_dir = args.output_root / args.arm / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    suffix = "smoke" if args.smoke else "full"
    path = shard_dir / f"{suffix}-shard-{args.shard_index:03d}-of-{args.num_shards:03d}.json"
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    summary = {
        "schema_version": "effdock.s50_raw_refined_confidence_external_shard.v1",
        "protocol_id": PROTOCOL_ID,
        "status": "complete",
        "stage": suffix,
        "arm": args.arm,
        "checkpoint_sha256": args.expected_confidence_sha256,
        "docking_checkpoint_sha256": args.expected_docking_sha256,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "assigned": len(completed),
        "completed": completed,
        "runtime": {
            "elapsed_seconds": time.monotonic() - started,
            "finished_at_utc": datetime.now(UTC).isoformat(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        },
    }
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.link(temporary, path)
    temporary.unlink()


if __name__ == "__main__":
    main()
