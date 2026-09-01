#!/usr/bin/env python3
"""Evaluate one U70k/U100k selected-pose temporal-external shard."""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from posebusters import PoseBusters

from effdock.workflows.evaluate import file_sha256
from effdock.workflows.guidance_budget_posebusters_report import VALIDITY_CHECKS
from effdock.workflows.guidance_pl_valid import PL_VALIDITY_CHECKS, is_pl_valid
from effdock.workflows.posebusters_report import require_posebusters_runtime_version
from scripts.evaluate_external_temporal_posebusters_shard import (
    extract_pose,
    validated_checks,
)
from scripts.rescore_external_temporal_u50_shard import DATASETS, read_json
from scripts.run_s50_raw_refined_confidence_temporal_external_shard import (
    ARMS,
    PROTOCOL_ID,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(DATASETS), required=True)
    parser.add_argument("--arm", choices=tuple(sorted(ARMS)), required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-confidence-sha256", required=True)
    args = parser.parse_args()

    expected_count, expected_shards = DATASETS[args.dataset]
    if args.num_shards != expected_shards or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard contract mismatch")
    shard_name = f"{args.dataset}.shard-{args.shard_index:03d}-of-{args.num_shards:03d}"
    source_summary_path = (
        args.input_root / "full" / args.arm / "shards" / f"{shard_name}.json"
    )
    source = read_json(source_summary_path)
    if (
        source.get("protocol_id") != PROTOCOL_ID
        or source.get("status") != "complete"
        or source.get("stage") != "full"
        or source.get("arm") != args.arm
        or source.get("dataset") != args.dataset
        or source.get("confidence_checkpoint_sha256")
        != args.expected_confidence_sha256
        or int(source.get("num_shards", -1)) != args.num_shards
        or int(source.get("shard_index", -1)) != args.shard_index
    ):
        raise ValueError(f"invalid confidence shard: {source_summary_path}")
    records = source.get("records", [])
    assigned = len(range(args.shard_index, expected_count, args.num_shards))
    if not isinstance(records, list) or len(records) != assigned:
        raise ValueError(f"expected {assigned} confidence records")

    final_dir = args.output_root / "full" / args.arm / "posebusters" / shard_name
    if final_dir.exists():
        raise FileExistsError(final_dir)
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    attempt_parent = args.output_root / "full" / args.arm / "posebusters" / ".incomplete"
    attempt_parent.mkdir(parents=True, exist_ok=True)
    attempt = Path(tempfile.mkdtemp(prefix=f"{shard_name}.", dir=attempt_parent))
    selected_dir = args.output_root / "full" / args.arm / "selected_poses" / args.dataset

    version = require_posebusters_runtime_version()
    buster = PoseBusters(config="redock", max_workers=0)
    started = time.monotonic()
    results: list[dict[str, Any]] = []
    rmsd_check: str | None = None
    for index, record in enumerate(records, start=1):
        complex_id = str(record["id"])
        confidence_path = Path(str(record["confidence_summary"]))
        refinement_path = Path(str(record["refinement_summary"]))
        if file_sha256(confidence_path) != record["confidence_summary_sha256"]:
            raise ValueError(f"{complex_id}: changed confidence summary")
        if file_sha256(refinement_path) != record["refinement_summary_sha256"]:
            raise ValueError(f"{complex_id}: changed refinement summary")
        confidence = read_json(confidence_path)
        refinement = read_json(refinement_path)
        confidence_sha256 = confidence.get("inputs", {}).get(
            "confidence_checkpoint_sha256"
        )
        if confidence_sha256 != args.expected_confidence_sha256:
            raise ValueError(f"{complex_id}: confidence checkpoint SHA mismatch")

        selected = confidence["selected"]["step_100"]
        pose_index = int(selected["pose_index"])
        step_spec = refinement["artifacts"]["step_100_sdf"]
        step_path = Path(step_spec["path"])
        protein = Path(refinement["inputs"]["protein"])
        ligand_ref = Path(refinement["inputs"]["ligand_reference"])
        for path, expected in (
            (step_path, step_spec["sha256"]),
            (protein, refinement["inputs"]["protein_sha256"]),
            (ligand_ref, refinement["inputs"]["ligand_reference_sha256"]),
        ):
            if not path.is_file() or file_sha256(path) != expected:
                raise ValueError(f"{complex_id}: missing or changed input {path}")

        selected_path = selected_dir / f"{complex_id}.sdf"
        if selected_path.exists():
            raise FileExistsError(selected_path)
        extract_pose(step_path, pose_index, selected_path)
        frame = buster.bust(selected_path, ligand_ref, protein, full_report=False)
        if len(frame.index) != 1:
            raise ValueError(f"{complex_id}: expected one PoseBusters row")
        current_rmsd, checks = validated_checks(frame.iloc[0].to_dict(), complex_id)
        if rmsd_check is None:
            rmsd_check = current_rmsd
        elif rmsd_check != current_rmsd:
            raise ValueError("PoseBusters RMSD column changed within shard")
        pl_valid = is_pl_valid(checks)
        official_valid = all(checks[key] for key in VALIDITY_CHECKS)
        raw_rmsd = float(confidence["selected"]["step_000"]["symmetry_rmsd_angstrom"])
        refined_rmsd = float(selected["symmetry_rmsd_angstrom"])
        pose_rows = refinement["poses"]
        results.append(
            {
                "arm": args.arm,
                "dataset": args.dataset,
                "id": complex_id,
                "selected_pose_index": pose_index,
                "raw_selected_rmsd": raw_rmsd,
                "refined_selected_rmsd": refined_rmsd,
                "raw_selected_rmsd_lt2": raw_rmsd < 2.0,
                "refined_selected_rmsd_lt2": refined_rmsd < 2.0,
                "raw_oracle_rmsd": min(
                    float(row["initial_symmetry_rmsd_angstrom"]) for row in pose_rows
                ),
                "refined_oracle_rmsd": min(
                    float(row["final_symmetry_rmsd_angstrom"]) for row in pose_rows
                ),
                "pl_valid": pl_valid,
                "posebusters_valid": official_valid,
                "joint_pl_valid_rmsd_lt2": pl_valid and refined_rmsd < 2.0,
                "joint_posebusters_valid_rmsd_lt2": official_valid
                and refined_rmsd < 2.0,
                "mean_refinement_terminal_step": sum(
                    int(row["terminal_step"]) for row in pose_rows
                )
                / 100.0,
                "selected_pose_sdf": str(selected_path.resolve()),
                "selected_pose_sdf_sha256": file_sha256(selected_path),
                **{key: checks[key] for key in VALIDITY_CHECKS},
            }
        )
        print(
            f"[{index}/{len(records)}] {args.arm} {args.dataset}/{complex_id} "
            f"rmsd={refined_rmsd:.3f} PL={pl_valid} PB={official_valid}",
            flush=True,
        )

    if rmsd_check is None:
        raise AssertionError("empty PoseBusters result")
    csv_path = attempt / "results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]), extrasaction="raise")
        writer.writeheader()
        writer.writerows(results)
    summary = {
        "schema_version": "effdock.s50_raw_refined_confidence_temporal_posebusters_shard.v1",
        "protocol_id": PROTOCOL_ID,
        "status": "complete",
        "arm": args.arm,
        "dataset": args.dataset,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "num_results": len(results),
        "posebusters_version": version,
        "posebusters_config": "redock",
        "official_validity_checks": list(VALIDITY_CHECKS),
        "pl_validity_checks": list(PL_VALIDITY_CHECKS),
        "separate_rmsd_check": rmsd_check,
        "confidence_checkpoint_sha256": args.expected_confidence_sha256,
        "runtime": {
            "elapsed_seconds": time.monotonic() - started,
            "finished_at_utc": datetime.now(UTC).isoformat(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        },
    }
    (attempt / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.rename(attempt, final_dir)


if __name__ == "__main__":
    main()
