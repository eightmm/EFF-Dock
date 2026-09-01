#!/usr/bin/env python3
"""Run one resumable N40/S25 refinement or U50-scoring shard."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

MANIFEST_PROTOCOL_ID = "EFFDOCK-FIXED-NFE-STEP-POSE-REFINEMENT-INPUT-V1"
PROTOCOL_ID = "EFFDOCK-FIXED-NFE-STEP-POSE-V1"
REFINEMENT_PROTOCOL_ID = "EFFDOCK-FIXED-NFE-STEP-POSE-REFINEMENT-V1"
CONFIDENCE_PROTOCOL_ID = "EFFDOCK-FIXED-NFE-STEP-POSE-U50-CONFIDENCE-V1"
EXPECTED_POSES = 40


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _complete(path: Path, *, stage: str, dataset: str, complex_id: str) -> bool:
    if not path.is_file():
        return False
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if stage == "refinement":
        return (
            value.get("protocol_id") == REFINEMENT_PROTOCOL_ID
            and value.get("status") == "complete_descriptive"
            and value.get("inputs", {}).get("dataset") == dataset
            and value.get("inputs", {}).get("complex_id") == complex_id
            and int(value.get("counts", {}).get("poses", -1)) == EXPECTED_POSES
            and int(value.get("counts", {}).get("failed", -1)) == 0
        )
    score_spec = value.get("artifacts", {}).get("scores_csv", {})
    score_path = Path(str(score_spec.get("path", "")))
    score_ok = (
        score_path.is_file()
        and isinstance(score_spec.get("sha256"), str)
        and file_sha256(score_path) == score_spec["sha256"]
    )
    return (
        value.get("protocol_id") == CONFIDENCE_PROTOCOL_ID
        and value.get("status") == "complete_descriptive"
        and value.get("dataset") == dataset
        and value.get("complex_id") == complex_id
        and int(value.get("pose_count", -1)) == EXPECTED_POSES
        and float(value.get("sigma", -1)) == 2.0
        and int(value.get("pose_batch_size", -1)) == 20
        and score_ok
    )


def _records(manifest: dict[str, Any], *, shard_index: int, num_shards: int) -> list[dict[str, Any]]:
    if manifest.get("protocol_id") != MANIFEST_PROTOCOL_ID or manifest.get("status") != "complete":
        raise ValueError("unexpected or incomplete N40/S25 manifest")
    records = sorted(
        manifest.get("records", []), key=lambda row: (str(row["dataset"]), str(row["id"]))
    )
    if len(records) != int(manifest.get("expected_complexes", -1)):
        raise ValueError("manifest record-count mismatch")
    if any(int(row.get("pose_count", -1)) != EXPECTED_POSES for row in records):
        raise ValueError("manifest contains a non-40-pose record")
    assigned = records[shard_index::num_shards]
    if not assigned:
        raise ValueError("empty stage shard")
    return assigned


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--benchmark-input-manifest", type=Path, required=True)
    parser.add_argument("--external-dir", type=Path, required=True)
    parser.add_argument("--protocol-file", type=Path, required=True)
    parser.add_argument("--refinement-capsule", type=Path, required=True)
    parser.add_argument("--confidence-capsule", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--docking-checkpoint", type=Path, required=True)
    parser.add_argument("--confidence-checkpoint", type=Path, required=True)
    parser.add_argument("--stage", choices=("refinement", "confidence"), required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    args = parser.parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid stage shard contract")
    repo_root = args.repo_root.resolve()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    records = _records(manifest, shard_index=args.shard_index, num_shards=args.num_shards)
    started = time.monotonic()
    completed: list[dict[str, Any]] = []
    for offset, record in enumerate(records, start=1):
        dataset = str(record["dataset"])
        complex_id = str(record["id"])
        refinement_dir = args.output_root / "refinement" / dataset / complex_id
        confidence_dir = args.output_root / "confidence" / "u050000" / dataset / complex_id
        refinement_summary = refinement_dir / "summary.json"
        confidence_summary = confidence_dir / "summary.json"
        if args.stage == "refinement":
            target_dir = refinement_dir
            target_summary = refinement_summary
        else:
            if not _complete(
                refinement_summary,
                stage="refinement",
                dataset=dataset,
                complex_id=complex_id,
            ):
                raise RuntimeError(f"missing completed refinement: {dataset}/{complex_id}")
            target_dir = confidence_dir
            target_summary = confidence_summary
        if target_dir.exists() and not _complete(
            target_summary, stage=args.stage, dataset=dataset, complex_id=complex_id
        ):
            raise RuntimeError(f"invalid pre-existing stage output: {target_dir}")
        if not target_dir.exists():
            centers = args.external_dir / (
                "astex_reference_pocket_centers.json"
                if dataset == "astex"
                else "posebusters_reference_pocket_centers.json"
            )
            if args.stage == "refinement":
                command = [
                    sys.executable,
                    str(repo_root / "scripts/run_fixed_nfe_step_pose_refinement.py"),
                    "--manifest",
                    str(args.manifest.resolve()),
                    "--benchmark-input-manifest",
                    str(args.benchmark_input_manifest.resolve()),
                    "--external-dir",
                    str(args.external_dir.resolve()),
                    "--pocket-centers",
                    str(centers.resolve()),
                    "--protocol-file",
                    str(args.protocol_file.resolve()),
                    "--dataset",
                    dataset,
                    "--eta",
                    "2",
                    "--complex-id",
                    complex_id,
                    "--output-dir",
                    str(refinement_dir.resolve()),
                    "--device",
                    "cuda",
                    "--steps",
                    "100",
                    "--save-every",
                    "25",
                    "--batch-size",
                    "10",
                    "--energy-convergence-absolute-kcal-mol",
                    "0.02",
                    "--energy-convergence-relative",
                    "0.001",
                    "--energy-convergence-patience",
                    "5",
                    "--energy-convergence-min-steps",
                    "25",
                ]
                pythonpath = args.refinement_capsule.resolve() / "src"
            else:
                command = [
                    sys.executable,
                    str(repo_root / "scripts/run_fixed_nfe_step_pose_confidence.py"),
                    "--refinement-summary",
                    str(refinement_summary.resolve()),
                    "--benchmark-input-manifest",
                    str(args.benchmark_input_manifest.resolve()),
                    "--external-dir",
                    str(args.external_dir.resolve()),
                    "--config",
                    str(args.config.resolve()),
                    "--docking-checkpoint",
                    str(args.docking_checkpoint.resolve()),
                    "--confidence-checkpoint",
                    str(args.confidence_checkpoint.resolve()),
                    "--output-dir",
                    str(confidence_dir.resolve()),
                    "--sigma",
                    "2",
                    "--pose-batch-size",
                    "20",
                    "--device",
                    "cuda",
                ]
                pythonpath = args.confidence_capsule.resolve() / "src"
            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(pythonpath)
            environment["PYTHONDONTWRITEBYTECODE"] = "1"
            subprocess.run(command, check=True, cwd=repo_root, env=environment)
        completed.append({"dataset": dataset, "id": complex_id})
        print(f"[{offset}/{len(records)}] {args.stage} {dataset}/{complex_id}", flush=True)
    shard_dir = args.output_root / "stage_shards" / args.stage
    shard_dir.mkdir(parents=True, exist_ok=True)
    path = shard_dir / f"shard-{args.shard_index:03d}-of-{args.num_shards:03d}.json"
    if path.exists():
        raise FileExistsError(f"refusing to overwrite shard ledger: {path}")
    summary = {
        "schema_version": "effdock.fixed_nfe_step_pose_stage.v1",
        "protocol_id": PROTOCOL_ID,
        "status": "complete",
        "stage": args.stage,
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": file_sha256(args.manifest),
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "assigned": len(completed),
        "completed": completed,
        "runtime": {
            "elapsed_seconds": time.monotonic() - started,
            "finished_at_utc": datetime.now(UTC).isoformat(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
        },
    }
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.link(temporary, path)
    temporary.unlink()


if __name__ == "__main__":
    main()
