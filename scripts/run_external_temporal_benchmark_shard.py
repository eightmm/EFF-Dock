#!/usr/bin/env python3
"""Run one frozen external benchmark shard through sampling and refinement."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROTOCOL_ID = "EFFDOCK-EXTERNAL-TEMPORAL-GUIDED-REFINED-V1"
REFINEMENT_PROTOCOL_ID = "EFFDOCK-GUIDANCE-SDF-POST-REFINEMENT-V1"
CONFIDENCE_PROTOCOL_ID = "EFFDOCK-GUIDANCE-SDF-POST-REFINEMENT-CONFIDENCE-V2"
EXPECTED_POSES = 100


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def require_complete(path: Path, protocol_id: str) -> dict[str, Any]:
    value = read_json(path)
    if value.get("protocol_id") != protocol_id:
        raise ValueError(f"protocol mismatch in {path}")
    status = value.get("status")
    evaluator_complete = (
        status is None
        and int(value.get("num_assigned", -1)) == int(value.get("num_success", -2))
        and int(value.get("num_failed", -1)) == 0
    )
    if not evaluator_complete and status not in {"complete", "complete_descriptive"}:
        raise ValueError(f"incomplete artifact: {path}")
    return value


def source_tag(
    dataset: str,
    *,
    prefix: str = "effdock-external-temporal-v1",
    sampling_profile: str = "guided_eta2",
) -> str:
    suffix = "eta2" if sampling_profile == "guided_eta2" else "unguided"
    return f"{prefix}-{dataset}-n100-s10-sigma2-{suffix}"


def sampling_paths(
    root: Path,
    dataset: str,
    *,
    num_shards: int,
    shard_index: int,
    prefix: str = "effdock-external-temporal-v1",
    sampling_profile: str = "guided_eta2",
) -> tuple[Path, Path]:
    tag = source_tag(dataset, prefix=prefix, sampling_profile=sampling_profile)
    if num_shards > 1:
        tag += f".shard-{shard_index:03d}-of-{num_shards:03d}"
    return root / f"{tag}.csv", root / f"{tag}.summary.json"


def sampling_command(args: argparse.Namespace, sampling_root: Path) -> list[str]:
    tag = source_tag(
        args.dataset,
        prefix=args.run_name_prefix,
        sampling_profile=args.sampling_profile,
    )
    command = [
        ".venv/bin/eff-dock",
        "evaluate",
        "--dataset",
        args.dataset,
        "--data-dir",
        str(args.data_dir),
        "--external-dir",
        str(args.external_dir),
        "--pocket-centers",
        str(args.pocket_centers),
        "--checkpoint",
        str(args.docking_checkpoint),
        "--confidence-checkpoint",
        str(args.confidence_checkpoint),
        "--selector-profile",
        "confidence_cluster_free",
        "--config",
        str(args.config),
        "--device",
        "cuda",
        "--output-dir",
        str(sampling_root),
        "--protocol-id",
        args.protocol_id,
        "--run-name",
        tag,
        "--num-samples",
        str(EXPECTED_POSES),
        "--num-steps",
        "10",
        "--prior-pool-size",
        str(EXPECTED_POSES),
        "--sigma",
        "2",
        "--time-schedule",
        "late",
        "--schedule-power",
        "3",
        "--pocket-cutoff",
        "10",
        "--center-jitter-sigma",
        "0",
        "--refine",
        "none",
        "--seed",
        "42",
        "--num-shards",
        str(args.num_shards),
        "--shard-index",
        str(args.shard_index),
        "--expected-discovered-count",
        str(args.expected_count),
        "--require-complete-success",
        "--require-full-ligand-atom-mapping",
        "--save-selected-poses",
        "--eligibility-manifest",
        str(args.dataset_manifest),
    ]
    if args.sampling_profile == "guided_eta2":
        command.extend(
            [
                "--unified-guidance-mode",
                "normalized_drift",
                "--unified-guidance-scale",
                "2",
                "--unified-guidance-receptor-policy",
                "geometry_only",
                "--unified-guidance-start-t",
                "0.5",
                "--unified-guidance-ramp-power",
                "1",
                "--unified-guidance-max-force",
                "20",
                "--unified-guidance-max-velocity",
                "5",
                "--unified-guidance-max-angular-velocity",
                "5",
                "--unified-guidance-max-atom-displacement",
                "0.25",
                "--unified-guidance-max-backtracks",
                "8",
                "--unified-guidance-protein-shell",
                "18",
            ]
        )
    if args.only_id is not None:
        command.extend(("--only-id", args.only_id))
    return command


def build_shard_manifest(
    args: argparse.Namespace,
    *,
    csv_path: Path,
    summary_path: Path,
    rows: list[dict[str, str]],
    output: Path,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    expected_guidance_mode = (
        "unified_normalized_drift"
        if args.sampling_profile == "guided_eta2"
        else "none"
    )
    for row in rows:
        if int(row["all_poses_count"]) != EXPECTED_POSES:
            raise ValueError(f"{row['id']}: expected {EXPECTED_POSES} saved poses")
        if row["guidance_mode"] != expected_guidance_mode:
            raise ValueError(f"{row['id']}: unexpected guidance mode")
        for path_key, hash_key in (
            ("all_poses_sdf", "all_poses_sdf_sha256"),
            ("protein", "protein_sha256"),
            ("ligand_ref", "ligand_reference_sha256"),
        ):
            path = Path(row[path_key])
            if not path.is_file() or file_sha256(path) != row[hash_key]:
                raise ValueError(f"{row['id']}: changed sampling asset {path_key}")
        records.append(
            {
                "dataset": args.dataset,
                "id": row["id"].lower(),
                "eta": 2.0 if args.sampling_profile == "guided_eta2" else 0.0,
                "sigma": 2.0,
                "pose_path": row["all_poses_sdf"],
                "pose_sha256": row["all_poses_sdf_sha256"],
                "pose_count": EXPECTED_POSES,
                "protein": row["protein"],
                "protein_sha256": row["protein_sha256"],
                "ligand_ref": row["ligand_ref"],
                "ligand_ref_sha256": row["ligand_reference_sha256"],
                "sampling_seed": int(row["sampling_seed"]),
                "prior_pool_sha256": row["prior_pool_sha256"],
                "guidance_mode": row["guidance_mode"],
                "guidance_parameter_sha256": row["guidance_parameter_sha256"],
            }
        )
    manifest = {
        "schema_version": "effdock.external_temporal_shard_input.v1",
        "protocol_id": args.protocol_id,
        "source_protocol_id": args.protocol_id,
        "dataset": args.dataset,
        "stage": args.stage,
        "sigma": 2.0,
        "eta": 2.0 if args.sampling_profile == "guided_eta2" else 0.0,
        "sampling_profile": args.sampling_profile,
        "num_steps": 10,
        "poses_per_complex": EXPECTED_POSES,
        "expected_dataset_count": args.expected_count,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "source_files": [{"path": str(csv_path), "sha256": file_sha256(csv_path)}],
        "source_summaries": [{"path": str(summary_path), "sha256": file_sha256(summary_path)}],
        "records": sorted(records, key=lambda record: str(record["id"])),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("phibench", "foldbench", "openbind"), required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--external-dir", type=Path, required=True)
    parser.add_argument("--pocket-centers", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--stage", choices=("smoke", "full"), required=True)
    parser.add_argument("--only-id")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--protocol-file", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--docking-checkpoint", type=Path, required=True)
    parser.add_argument("--confidence-checkpoint", type=Path, required=True)
    parser.add_argument("--protocol-id", default=PROTOCOL_ID)
    parser.add_argument("--run-name-prefix", default="effdock-external-temporal-v1")
    parser.add_argument(
        "--sampling-profile",
        choices=("guided_eta2", "unguided"),
        default="guided_eta2",
    )
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid shard index")
    if (args.stage == "smoke") != (args.only_id is not None):
        raise ValueError("smoke requires --only-id and full forbids it")

    started = time.monotonic()
    stage_root = args.output_root.resolve() / args.stage
    sampling_root = stage_root / "sampling"
    sampling_root.mkdir(parents=True, exist_ok=True)
    reservation = stage_root / "reservations" / f"{args.dataset}-{args.shard_index:03d}"
    reservation.parent.mkdir(parents=True, exist_ok=True)
    reservation.mkdir()

    try:
        subprocess.run(sampling_command(args, sampling_root), check=True)
        csv_path, sampling_summary_path = sampling_paths(
            sampling_root,
            args.dataset,
            num_shards=args.num_shards,
            shard_index=args.shard_index,
            prefix=args.run_name_prefix,
            sampling_profile=args.sampling_profile,
        )
        sampling_summary = require_complete(sampling_summary_path, args.protocol_id)
        if int(sampling_summary.get("num_failed", -1)) != 0:
            raise RuntimeError("sampling shard contains failures")
        with csv_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        expected_assigned = (
            1
            if args.stage == "smoke"
            else len(list(range(args.shard_index, args.expected_count, args.num_shards)))
        )
        if len(rows) != expected_assigned:
            raise ValueError(f"expected {expected_assigned} sampled rows, found {len(rows)}")

        manifest_path = (
            stage_root
            / "manifests"
            / (f"{args.dataset}.shard-{args.shard_index:03d}-of-{args.num_shards:03d}.json")
        )
        manifest = build_shard_manifest(
            args,
            csv_path=csv_path,
            summary_path=sampling_summary_path,
            rows=rows,
            output=manifest_path,
        )

        completed: list[dict[str, Any]] = []
        for index, record in enumerate(manifest["records"], start=1):
            complex_id = str(record["id"])
            refinement_dir = stage_root / "refinement" / args.dataset / complex_id
            refinement_summary_path = refinement_dir / "summary.json"
            if refinement_dir.exists():
                require_complete(refinement_summary_path, REFINEMENT_PROTOCOL_ID)
            else:
                subprocess.run(
                    [
                        sys.executable,
                        "scripts/run_guidance_sdf_post_refinement.py",
                        "--manifest",
                        str(manifest_path),
                        "--external-dir",
                        str(args.external_dir),
                        "--pocket-centers",
                        str(args.pocket_centers),
                        "--protocol-file",
                        str(args.protocol_file),
                        "--dataset",
                        args.dataset,
                        "--eta",
                        "2" if args.sampling_profile == "guided_eta2" else "0",
                        "--complex-id",
                        complex_id,
                        "--output-dir",
                        str(refinement_dir),
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
                    ],
                    check=True,
                )
            refinement = require_complete(refinement_summary_path, REFINEMENT_PROTOCOL_ID)

            confidence_dir = stage_root / "confidence" / args.dataset / complex_id
            confidence_summary_path = confidence_dir / "summary.json"
            if confidence_dir.exists():
                require_complete(confidence_summary_path, CONFIDENCE_PROTOCOL_ID)
            else:
                subprocess.run(
                    [
                        sys.executable,
                        "scripts/score_guidance_sdf_post_refinement_confidence.py",
                        "--refinement-summary",
                        str(refinement_summary_path),
                        "--external-dir",
                        str(args.external_dir),
                        "--config",
                        str(args.config),
                        "--docking-checkpoint",
                        str(args.docking_checkpoint),
                        "--confidence-checkpoint",
                        str(args.confidence_checkpoint),
                        "--output-dir",
                        str(confidence_dir),
                        "--sigma",
                        "2",
                        "--pose-batch-size",
                        "20",
                        "--device",
                        "cuda",
                    ],
                    check=True,
                )
            confidence = require_complete(confidence_summary_path, CONFIDENCE_PROTOCOL_ID)
            completed.append(
                {
                    "id": complex_id,
                    "refinement_summary": str(refinement_summary_path),
                    "refinement_summary_sha256": file_sha256(refinement_summary_path),
                    "confidence_summary": str(confidence_summary_path),
                    "confidence_summary_sha256": file_sha256(confidence_summary_path),
                    "raw_selected_rmsd": confidence["selected"]["step_000"][
                        "symmetry_rmsd_angstrom"
                    ],
                    "refined_selected_rmsd": confidence["selected"]["step_100"][
                        "symmetry_rmsd_angstrom"
                    ],
                    "refinement_mean_terminal_step": sum(
                        int(pose["terminal_step"]) for pose in refinement["poses"]
                    )
                    / EXPECTED_POSES,
                }
            )
            print(
                f"[{index}/{len(manifest['records'])}] complete {args.dataset}/{complex_id}",
                flush=True,
            )

        summary = {
            "schema_version": "effdock.external_temporal_shard.v1",
            "protocol_id": args.protocol_id,
            "status": "complete",
            "stage": args.stage,
            "dataset": args.dataset,
            "expected_dataset_count": args.expected_count,
            "num_shards": args.num_shards,
            "shard_index": args.shard_index,
            "num_completed": len(completed),
            "sampling": {
                "num_samples": EXPECTED_POSES,
                "num_steps": 10,
                "sigma": 2.0,
                "guidance_mode": (
                    "normalized_drift"
                    if args.sampling_profile == "guided_eta2"
                    else "none"
                ),
                "guidance_eta": 2.0 if args.sampling_profile == "guided_eta2" else 0.0,
            },
            "refinement": {
                "maximum_steps": 100,
                "energy_convergence_absolute_kcal_mol": 0.02,
                "energy_convergence_relative": 0.001,
                "energy_convergence_patience": 5,
                "energy_convergence_min_steps": 25,
            },
            "records": completed,
            "runtime": {
                "elapsed_seconds": time.monotonic() - started,
                "finished_at_utc": datetime.now(UTC).isoformat(),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            },
        }
        summary_dir = stage_root / "shards"
        summary_dir.mkdir(parents=True, exist_ok=True)
        summary_path = summary_dir / (
            f"{args.dataset}.shard-{args.shard_index:03d}-of-{args.num_shards:03d}.json"
        )
        if summary_path.exists():
            raise FileExistsError(summary_path)
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    except BaseException:
        reservation.rmdir()
        raise


if __name__ == "__main__":
    main()
