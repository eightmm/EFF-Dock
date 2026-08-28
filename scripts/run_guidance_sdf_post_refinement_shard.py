#!/usr/bin/env python3
"""Run one resumable post-refinement or frozen-confidence shard."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

PROTOCOL_ID = "EFFDOCK-GUIDANCE-SDF-POST-REFINEMENT-CONFIDENCE-V2"
EXPECTED_DATASET_COUNTS = {"astex": 85, "posebusters": 308}


def _records(
    manifest: dict,
    *,
    shard_index: int,
    num_shards: int,
    eta: float,
) -> list[dict]:
    selected = sorted(
        (
            row
            for row in manifest.get("records", [])
            if float(row.get("eta", float("nan"))) == eta
        ),
        key=lambda row: (str(row["dataset"]), str(row["id"])),
    )
    counts = {
        dataset: sum(row["dataset"] == dataset for row in selected)
        for dataset in EXPECTED_DATASET_COUNTS
    }
    if counts != EXPECTED_DATASET_COUNTS or len(selected) != 393:
        raise ValueError(f"expected frozen 85+308 eta={eta:g} records, got {counts}")
    assigned = selected[shard_index::num_shards]
    if not assigned:
        raise ValueError("empty refinement shard")
    return assigned


def _complete(path: Path, protocol_id: str) -> bool:
    if not path.is_file():
        return False
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return value.get("protocol_id") == protocol_id and value.get("status") == "complete_descriptive"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--benchmark-input-manifest", type=Path, required=True)
    parser.add_argument("--protocol-file", type=Path, required=True)
    parser.add_argument("--external-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--stage", choices=("refinement", "confidence"), required=True)
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--energy-convergence-absolute-kcal-mol", type=float)
    parser.add_argument("--energy-convergence-relative", type=float)
    parser.add_argument("--energy-convergence-patience", type=int, default=5)
    parser.add_argument("--energy-convergence-min-steps", type=int, default=20)
    parser.add_argument("--confidence-sigma", type=float, default=0.5)
    args = parser.parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid shard contract")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    records = _records(
        manifest,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
        eta=args.eta,
    )
    started = time.monotonic()
    completed: list[dict] = []
    for index, record in enumerate(records, start=1):
        dataset = str(record["dataset"])
        complex_id = str(record["id"])
        refinement_dir = args.output_root / "refinement" / dataset / complex_id
        confidence_dir = args.output_root / "confidence_chunk20_fresh" / dataset / complex_id
        refinement_summary = refinement_dir / "summary.json"
        confidence_summary = confidence_dir / "summary.json"
        centers = args.external_dir / (
            "astex_reference_pocket_centers.json"
            if dataset == "astex"
            else "posebusters_reference_pocket_centers.json"
        )
        refinement_complete = _complete(
            refinement_summary, "EFFDOCK-GUIDANCE-SDF-POST-REFINEMENT-V1"
        )
        if args.stage == "refinement" and not refinement_complete:
            command = [
                sys.executable,
                "scripts/run_guidance_sdf_post_refinement.py",
                "--manifest", str(args.manifest),
                "--benchmark-input-manifest", str(args.benchmark_input_manifest),
                "--external-dir", str(args.external_dir),
                "--pocket-centers", str(centers),
                "--protocol-file", str(args.protocol_file),
                "--dataset", dataset,
                "--eta", str(args.eta),
                "--complex-id", complex_id,
                "--output-dir", str(refinement_dir),
                "--device", "cuda",
                "--steps", "100",
                "--save-every", "25",
                "--batch-size", "10",
            ]
            if args.energy_convergence_absolute_kcal_mol is not None:
                command.extend(
                    (
                        "--energy-convergence-absolute-kcal-mol",
                        str(args.energy_convergence_absolute_kcal_mol),
                    )
                )
            if args.energy_convergence_relative is not None:
                command.extend(
                    (
                        "--energy-convergence-relative",
                        str(args.energy_convergence_relative),
                    )
                )
            command.extend(
                (
                    "--energy-convergence-patience",
                    str(args.energy_convergence_patience),
                    "--energy-convergence-min-steps",
                    str(args.energy_convergence_min_steps),
                )
            )
            subprocess.run(command, check=True)
            refinement_complete = True
        if args.stage == "confidence" and not refinement_complete:
            raise RuntimeError(f"missing completed refinement: {dataset}/{complex_id}")
        if args.stage == "confidence" and not _complete(confidence_summary, PROTOCOL_ID):
            command = [
                sys.executable,
                "scripts/score_guidance_sdf_post_refinement_confidence.py",
                "--refinement-summary", str(refinement_summary),
                "--benchmark-input-manifest", str(args.benchmark_input_manifest),
                "--external-dir", str(args.external_dir),
                "--config", "configs/train.yaml",
                "--docking-checkpoint", "weights/effdock_geometry_ft_100k_best.pt",
                "--confidence-checkpoint", "weights/effdock_confidence_extmatch_n80_s25_step42500.pt",
                "--output-dir", str(confidence_dir),
                "--sigma", str(args.confidence_sigma),
                "--pose-batch-size", "20",
                "--device", "cuda",
            ]
            subprocess.run(command, check=True)
        completed.append({"dataset": dataset, "id": complex_id})
        print(
            f"[{index}/{len(records)}] complete {args.stage} {dataset}/{complex_id}",
            flush=True,
        )
    shard_name = (
        "confidence_chunk20_shards" if args.stage == "confidence" else "refinement_shards"
    )
    shard_dir = args.output_root / shard_name
    shard_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "protocol_id": PROTOCOL_ID,
        "status": "complete",
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "stage": args.stage,
        "eta": args.eta,
        "energy_convergence": {
            "absolute_kcal_mol": args.energy_convergence_absolute_kcal_mol,
            "relative": args.energy_convergence_relative,
            "patience": args.energy_convergence_patience,
            "min_steps": args.energy_convergence_min_steps,
        },
        "confidence_sigma": args.confidence_sigma,
        "assigned": len(records),
        "completed": completed,
        "runtime": {
            "elapsed_seconds": time.monotonic() - started,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "finished_at_utc": datetime.now(UTC).isoformat(),
        },
    }
    path = shard_dir / f"shard-{args.shard_index:03d}-of-{args.num_shards:03d}.json"
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
