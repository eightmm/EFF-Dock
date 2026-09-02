#!/usr/bin/env python3
"""Evaluate confidence-ranked U70k PhiBench Top-5 poses with PoseBusters."""

from __future__ import annotations

import argparse
import csv
import json
import math
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

PROTOCOL_ID = "EFFDOCK-PHIBENCH-U70K-TOP5-V1"
SOURCE_PROTOCOL_ID = "EFFDOCK-S50-RAW-REFINED-CONFIDENCE-TEMPORAL-EXTERNAL-V1"
CONFIDENCE_PROTOCOL_ID = "EFFDOCK-GUIDANCE-SDF-POST-REFINEMENT-CONFIDENCE-V2"
CONFIDENCE_SHA256 = "ce59be42f0ca613871ca079127c3296f5ca9a4ec72e44a9e5cf61878351c2638"
EXPECTED_COMPLEXES = 203
EXPECTED_POSES = 100
TOP_K = 5


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def load_score_rows(summary: dict[str, Any]) -> list[dict[str, float | int]]:
    spec = summary.get("artifacts", {}).get("scores_csv", {})
    path = Path(str(spec.get("path", "")))
    if not path.is_file() or file_sha256(path) != spec.get("sha256"):
        raise ValueError(f"missing or changed score CSV: {path}")
    required = {
        "pose_index",
        "before_confidence_rmsd",
        "after_confidence_rmsd",
        "initial_symmetry_rmsd_angstrom",
        "final_symmetry_rmsd_angstrom",
    }
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError(f"score CSV schema mismatch: {path}")
        raw_rows = list(reader)
    if len(raw_rows) != EXPECTED_POSES:
        raise ValueError(f"expected {EXPECTED_POSES} score rows: {path}")
    rows: list[dict[str, float | int]] = []
    for raw in raw_rows:
        row: dict[str, float | int] = {"pose_index": int(raw["pose_index"])}
        for key in required - {"pose_index"}:
            value = float(raw[key])
            if not math.isfinite(value):
                raise ValueError(f"non-finite {key}: {path}")
            row[key] = value
        rows.append(row)
    if {int(row["pose_index"]) for row in rows} != set(range(EXPECTED_POSES)):
        raise ValueError(f"pose-index inventory mismatch: {path}")
    rows.sort(key=lambda row: int(row["pose_index"]))
    return rows


def rank_indices(rows: list[dict[str, float | int]], stage: str) -> list[int]:
    key = {"raw": "before_confidence_rmsd", "refined": "after_confidence_rmsd"}[stage]
    return sorted(
        (int(row["pose_index"]) for row in rows),
        key=lambda index: (float(rows[index][key]), index),
    )


def _validate_source_shard(
    source_root: Path, shard_index: int, num_shards: int
) -> list[dict[str, Any]]:
    name = f"phibench.shard-{shard_index:03d}-of-{num_shards:03d}.json"
    path = source_root / "full" / "u070000" / "shards" / name
    source = read_json(path)
    if (
        source.get("protocol_id") != SOURCE_PROTOCOL_ID
        or source.get("status") != "complete"
        or source.get("stage") != "full"
        or source.get("arm") != "u070000"
        or source.get("dataset") != "phibench"
        or source.get("confidence_checkpoint_sha256") != CONFIDENCE_SHA256
        or int(source.get("num_shards", -1)) != num_shards
        or int(source.get("shard_index", -1)) != shard_index
    ):
        raise ValueError(f"invalid source shard: {path}")
    records = source.get("records")
    expected = len(range(shard_index, EXPECTED_COMPLEXES, num_shards))
    if not isinstance(records, list) or len(records) != expected:
        raise ValueError(f"source shard record count mismatch: {path}")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, default=13)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--stage", choices=("smoke", "full"), required=True)
    parser.add_argument("--max-records", type=int)
    args = parser.parse_args()
    if args.num_shards != 13 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("PhiBench Top-5 requires 13 shards")
    if (args.stage == "smoke") != (args.max_records is not None):
        raise ValueError("smoke alone requires --max-records")
    if args.max_records is not None and args.max_records < 1:
        raise ValueError("--max-records must be positive")

    records = _validate_source_shard(
        args.source_root.resolve(), args.shard_index, args.num_shards
    )
    if args.max_records is not None:
        records = records[: args.max_records]
    shard_name = f"phibench.shard-{args.shard_index:03d}-of-{args.num_shards:03d}"
    final_dir = args.output_root.resolve() / args.stage / "shards" / shard_name
    if final_dir.exists():
        raise FileExistsError(final_dir)
    incomplete = args.output_root.resolve() / args.stage / "shards" / ".incomplete"
    incomplete.mkdir(parents=True, exist_ok=True)
    attempt = Path(tempfile.mkdtemp(prefix=f"{shard_name}.", dir=incomplete))

    version = require_posebusters_runtime_version()
    if version != "0.6.5":
        raise ValueError(f"expected PoseBusters 0.6.5, got {version}")
    buster = PoseBusters(config="redock", max_workers=0)
    started = time.monotonic()
    completed: list[dict[str, Any]] = []
    rmsd_check: str | None = None
    for position, record in enumerate(records, start=1):
        complex_id = str(record["id"])
        confidence_path = Path(str(record["confidence_summary"]))
        refinement_path = Path(str(record["refinement_summary"]))
        if file_sha256(confidence_path) != record["confidence_summary_sha256"]:
            raise ValueError(f"{complex_id}: changed confidence summary")
        if file_sha256(refinement_path) != record["refinement_summary_sha256"]:
            raise ValueError(f"{complex_id}: changed refinement summary")
        confidence = read_json(confidence_path)
        refinement = read_json(refinement_path)
        if (
            confidence.get("protocol_id") != CONFIDENCE_PROTOCOL_ID
            or confidence.get("status") != "complete_descriptive"
            or confidence.get("dataset") != "phibench"
            or confidence.get("complex_id") != complex_id
            or confidence.get("inputs", {}).get("confidence_checkpoint_sha256")
            != CONFIDENCE_SHA256
            or int(confidence.get("pose_count", -1)) != EXPECTED_POSES
        ):
            raise ValueError(f"{complex_id}: invalid confidence summary")
        rows = load_score_rows(confidence)
        raw_order = rank_indices(rows, "raw")
        refined_order = rank_indices(rows, "refined")
        if raw_order[0] != int(confidence["selected"]["step_000"]["pose_index"]):
            raise ValueError(f"{complex_id}: raw Top-1 reproduction failed")
        if refined_order[0] != int(confidence["selected"]["step_100"]["pose_index"]):
            raise ValueError(f"{complex_id}: refined Top-1 reproduction failed")

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

        evaluated: list[dict[str, Any]] = []
        for rank, pose_index in enumerate(refined_order[:TOP_K], start=1):
            pose_path = attempt / "poses" / complex_id / f"rank-{rank:02d}.sdf"
            extract_pose(step_path, pose_index, pose_path)
            frame = buster.bust(pose_path, ligand_ref, protein, full_report=False)
            if len(frame.index) != 1:
                raise ValueError(f"{complex_id}/rank-{rank}: expected one PoseBusters row")
            current_rmsd, checks = validated_checks(
                frame.iloc[0].to_dict(), f"{complex_id}/rank-{rank}"
            )
            if rmsd_check is None:
                rmsd_check = current_rmsd
            elif rmsd_check != current_rmsd:
                raise ValueError("PoseBusters RMSD column changed within shard")
            pb_valid = all(checks[key] for key in VALIDITY_CHECKS)
            symmetry_rmsd = float(rows[pose_index]["final_symmetry_rmsd_angstrom"])
            evaluated.append(
                {
                    "rank": rank,
                    "pose_index": pose_index,
                    "predicted_rmsd": float(rows[pose_index]["after_confidence_rmsd"]),
                    "symmetry_rmsd_angstrom": symmetry_rmsd,
                    "rmsd_lt2": symmetry_rmsd < 2.0,
                    "pl_valid": is_pl_valid(checks),
                    "posebusters_valid": pb_valid,
                    "joint_posebusters_valid_rmsd_lt2": pb_valid and symmetry_rmsd < 2.0,
                    "pose_sdf": str(pose_path.relative_to(attempt)),
                    "pose_sdf_sha256": file_sha256(pose_path),
                    "checks": checks,
                }
            )
        raw_rmsds = [float(rows[index]["initial_symmetry_rmsd_angstrom"]) for index in raw_order]
        refined_rmsds = [
            float(rows[index]["final_symmetry_rmsd_angstrom"]) for index in refined_order
        ]
        completed.append(
            {
                "id": complex_id,
                "raw_top1_rmsd": raw_rmsds[0],
                "raw_top5_best_rmsd": min(raw_rmsds[:TOP_K]),
                "raw_oracle_rmsd": min(raw_rmsds),
                "refined_top1_rmsd": refined_rmsds[0],
                "refined_top5_best_rmsd": min(refined_rmsds[:TOP_K]),
                "refined_oracle_rmsd": min(refined_rmsds),
                "refined_top1_posebusters_valid": evaluated[0]["posebusters_valid"],
                "refined_top1_joint": evaluated[0][
                    "joint_posebusters_valid_rmsd_lt2"
                ],
                "refined_top5_posebusters_valid": any(
                    bool(row["posebusters_valid"]) for row in evaluated
                ),
                "refined_top5_joint": any(
                    bool(row["joint_posebusters_valid_rmsd_lt2"])
                    for row in evaluated
                ),
                "top5": evaluated,
                "confidence_summary_sha256": record["confidence_summary_sha256"],
                "refinement_summary_sha256": record["refinement_summary_sha256"],
            }
        )
        print(f"[{position}/{len(records)}] {complex_id} Top-5 PB complete", flush=True)

    if rmsd_check is None:
        raise AssertionError("empty Top-5 shard")
    payload = {
        "schema_version": "effdock.phibench_u70k_top5_posebusters_shard.v1",
        "protocol_id": PROTOCOL_ID,
        "status": "complete_descriptive",
        "stage": args.stage,
        "dataset": "phibench",
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "num_complexes": len(completed),
        "num_posebusters_evaluations": len(completed) * TOP_K,
        "posebusters_version": version,
        "posebusters_config": "redock",
        "official_validity_checks": list(VALIDITY_CHECKS),
        "pl_validity_checks": list(PL_VALIDITY_CHECKS),
        "separate_posebusters_rmsd_check": rmsd_check,
        "confidence_checkpoint_sha256": CONFIDENCE_SHA256,
        "records": completed,
        "runtime": {
            "elapsed_seconds": time.monotonic() - started,
            "finished_at_utc": datetime.now(UTC).isoformat(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        },
    }
    (attempt / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.rename(attempt, final_dir)


if __name__ == "__main__":
    main()
