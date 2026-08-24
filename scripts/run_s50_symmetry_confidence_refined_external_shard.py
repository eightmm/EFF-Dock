#!/usr/bin/env python3
"""Rescore one frozen refined-external shard with one symmetry-confidence arm."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from effdock.workflows.evaluate import file_sha256

PROTOCOL_ID = "EFFDOCK-S50-SYMMETRY-CONFIDENCE-REFINED-EXTERNAL-V1"
SCORE_PROTOCOL_ID = "EFFDOCK-GUIDANCE-SDF-POST-REFINEMENT-CONFIDENCE-V2"
EXPECTED_COUNTS = {"astex": 85, "posebusters": 308}
EXPECTED_POSES = 100
SMOKE_IDS = ("astex/1jje", "posebusters/7b2c_tp7")


def _records(
    manifest: dict,
    *,
    shard_index: int,
    num_shards: int,
    smoke: bool,
) -> list[dict]:
    selected = sorted(
        (row for row in manifest.get("records", []) if float(row.get("eta", -1)) == 2.0),
        key=lambda row: (str(row["dataset"]), str(row["id"])),
    )
    counts = Counter(str(row["dataset"]) for row in selected)
    if dict(counts) != EXPECTED_COUNTS or len(selected) != sum(EXPECTED_COUNTS.values()):
        raise ValueError(f"unexpected eta=2 cohort: {dict(counts)}")
    if smoke:
        by_key = {f"{row['dataset']}/{row['id']}": row for row in selected}
        if any(key not in by_key for key in SMOKE_IDS):
            raise ValueError("smoke IDs are missing from the frozen cohort")
        return [by_key[key] for key in SMOKE_IDS]
    assigned = selected[shard_index::num_shards]
    if not assigned:
        raise ValueError("empty shard")
    return assigned


def _complete(
    path: Path,
    *,
    dataset: str,
    complex_id: str,
    confidence_sha256: str,
    docking_sha256: str,
) -> bool:
    if not path.is_file():
        return False
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    inputs = value.get("inputs", {})
    scores = value.get("artifacts", {}).get("scores_csv", {})
    scores_path = Path(str(scores.get("path", "")))
    scores_complete = (
        scores_path.is_file()
        and isinstance(scores.get("sha256"), str)
        and file_sha256(scores_path) == scores["sha256"]
    )
    return (
        value.get("protocol_id") == SCORE_PROTOCOL_ID
        and value.get("status") == "complete_descriptive"
        and value.get("dataset") == dataset
        and value.get("complex_id") == complex_id
        and int(value.get("pose_count", -1)) == EXPECTED_POSES
        and float(value.get("sigma", -1)) == 2.0
        and int(value.get("pose_batch_size", -1)) == 20
        and inputs.get("confidence_checkpoint_sha256") == confidence_sha256
        and inputs.get("docking_checkpoint_sha256") == docking_sha256
        and scores_complete
    )


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
        "schema_version": "effdock.s50_symmetry_confidence_refined_external_shard.v1",
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
    temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.link(temporary, path)
    temporary.unlink()


if __name__ == "__main__":
    main()
