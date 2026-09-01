#!/usr/bin/env python3
"""Rescore frozen pre/post guidance pose sets with the frozen confidence model."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from effdock.confidence.runtime import load_pose_confidence_model, score_poses_with_confidence
from effdock.inference.docking import load_model
from effdock.inference.preprocess import preprocess_complex
from effdock.workflows.benchmark_inputs import load_benchmark_inputs, load_benchmark_ligand
from effdock.workflows.evaluate import file_sha256

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_guidance_sdf_post_refinement import (  # noqa: E402
    EXPECTED_POSES,
    _load_source_row,
    _tensor_sha256,
)
from run_guidance_sdf_post_refinement import (
    PROTOCOL_ID as REFINEMENT_PROTOCOL_ID,
)

PROTOCOL_ID = "EFFDOCK-GUIDANCE-SDF-POST-REFINEMENT-CONFIDENCE-V2"
SCHEMA_VERSION = "effdock.guidance_sdf_post_refinement_confidence.v2"
FROZEN_POSE_BATCH_SIZE = 20


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _select_index(scores: list[dict[str, float]]) -> int:
    """Pure-confidence selector: minimum predicted RMSD, stable by pose index."""
    if len(scores) != EXPECTED_POSES:
        raise ValueError(f"expected {EXPECTED_POSES} confidence scores, got {len(scores)}")
    values = [float(row["confidence_rmsd"]) for row in scores]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("confidence scores contain non-finite predicted RMSD")
    return min(range(len(values)), key=lambda index: (values[index], index))


def _chunk_ranges(length: int, batch_size: int) -> list[tuple[int, int]]:
    if length < 1 or batch_size < 1:
        raise ValueError("length and batch_size must be positive")
    return [(start, min(start + batch_size, length)) for start in range(0, length, batch_size)]


def _score_in_batches(
    confidence_model,
    docking_model,
    graph,
    ligand_data,
    meta,
    poses: list[torch.Tensor],
    *,
    sigma: float,
    device: torch.device,
    batch_size: int,
) -> list[dict[str, float]]:
    scores: list[dict[str, float]] = []
    for start, stop in _chunk_ranges(len(poses), batch_size):
        scores.extend(
            score_poses_with_confidence(
                confidence_model,
                docking_model,
                graph,
                ligand_data,
                meta,
                poses[start:stop],
                sigma=sigma,
                device=device,
            )
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if len(scores) != len(poses):
        raise AssertionError("confidence batching changed pose count")
    return scores


def _load_refinement(path: Path) -> dict[str, Any]:
    summary = json.loads(path.read_text(encoding="utf-8"))
    if summary.get("protocol_id") != REFINEMENT_PROTOCOL_ID:
        raise ValueError("unexpected refinement protocol")
    if summary.get("status") != "complete_descriptive":
        raise ValueError("refinement is not complete")
    if int(summary.get("counts", {}).get("poses", -1)) != EXPECTED_POSES:
        raise ValueError("refinement pose count mismatch")
    statuses = {str(row.get("status")) for row in summary.get("poses", [])}
    unusable = statuses - {
        "max_steps",
        "converged_displacement",
        "converged_energy_plateau",
        "line_search_failed",
    }
    if unusable or int(summary.get("counts", {}).get("failed", -1)) != 0:
        raise ValueError(f"refinement contains unusable terminal poses: {sorted(unusable)}")
    return summary


def _trajectory_stages(summary: dict[str, Any], mol_input) -> dict[str, torch.Tensor]:
    spec = summary["artifacts"]["trajectory_pt"]
    path = Path(spec["path"])
    if not path.is_file() or file_sha256(path) != spec["sha256"]:
        raise ValueError(f"missing or changed trajectory tensor: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("protocol_id") != REFINEMENT_PROTOCOL_ID or payload.get(
        "schema_version"
    ) != summary.get("schema_version"):
        raise ValueError("trajectory protocol/schema mismatch")
    steps = payload["saved_steps"].to(torch.long).tolist()
    frames = payload["frames_pocket_centered"].to(torch.float32)
    if steps != [0, 25, 50, 75, 100]:
        raise ValueError(f"unexpected trajectory steps: {steps}")
    expected_shape = (5, EXPECTED_POSES, mol_input.GetNumAtoms(), 3)
    if tuple(frames.shape) != expected_shape or not bool(torch.isfinite(frames).all()):
        raise ValueError(f"invalid trajectory tensor shape/values: {tuple(frames.shape)}")
    selected = {"step_000": frames[0], "step_100": frames[-1]}
    for stage, value in selected.items():
        if _tensor_sha256(value) != summary["coordinate_hashes"][stage]:
            raise ValueError(f"trajectory coordinate hash mismatch: {stage}")
    return selected


def main() -> None:
    pipeline_started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refinement-summary", type=Path, required=True)
    parser.add_argument("--benchmark-input-manifest", type=Path, default=None)
    parser.add_argument("--external-dir", type=Path, default=Path("data/external_test"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--docking-checkpoint", type=Path, required=True)
    parser.add_argument("--confidence-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sigma", type=float, default=0.5)
    parser.add_argument("--pocket-cutoff", type=float)
    parser.add_argument("--pose-batch-size", type=int, default=FROZEN_POSE_BATCH_SIZE)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if not math.isfinite(args.sigma) or args.sigma <= 0:
        raise ValueError("confidence sigma must be finite and positive")
    if args.pose_batch_size != FROZEN_POSE_BATCH_SIZE:
        raise ValueError(f"V2 protocol requires frozen pose chunks of {FROZEN_POSE_BATCH_SIZE}")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but unavailable")

    refinement_path = args.refinement_summary.resolve()
    refinement = _load_refinement(refinement_path)
    inputs = refinement["inputs"]
    dataset = str(inputs["dataset"])
    complex_id = str(inputs["complex_id"])
    refinement_cutoff = float(inputs.get("pocket_cutoff_angstrom", 10.0))
    pocket_cutoff = (
        refinement_cutoff if args.pocket_cutoff is None else args.pocket_cutoff
    )
    if not math.isfinite(pocket_cutoff) or pocket_cutoff <= 0:
        raise ValueError("pocket-cutoff must be finite and positive")
    if not math.isclose(pocket_cutoff, refinement_cutoff, abs_tol=1e-12):
        raise ValueError(
            "confidence pocket-cutoff must match the refinement pocket-cutoff"
        )
    artifacts = refinement["artifacts"]
    stage_specs = {
        "step_000": artifacts["step_000_sdf"],
        "step_100": artifacts["step_100_sdf"],
    }
    for spec in stage_specs.values():
        path = Path(spec["path"])
        if not path.is_file() or file_sha256(path) != spec["sha256"]:
            raise ValueError(f"missing or changed pose input: {path}")
    frozen_files = [
        args.config,
        args.docking_checkpoint,
        args.confidence_checkpoint,
        Path(inputs["manifest"]),
        Path(inputs["protein"]),
        Path(inputs["ligand_reference"]),
    ]
    if args.benchmark_input_manifest is not None:
        frozen_files.append(args.benchmark_input_manifest)
    if any(not path.is_file() for path in frozen_files):
        raise FileNotFoundError("one or more frozen confidence inputs are missing")

    mapping, mapping_identity = load_benchmark_inputs(
        dataset, args.external_dir, args.benchmark_input_manifest
    )
    seed = int(inputs["sampling_seed"])
    mol_input, _ = load_benchmark_ligand(mapping[complex_id], random_seed=seed)
    pocket_center = torch.tensor(inputs["pocket_center_absolute"], dtype=torch.float32)
    graph, ligand_data, meta = preprocess_complex(
        Path(inputs["protein"]),
        mol_input,
        pocket_center=pocket_center,
        pocket_cutoff=pocket_cutoff,
    )
    if not torch.equal(meta["pocket_center"], pocket_center):
        raise AssertionError("preprocessing changed frozen pocket center")

    stage_coordinates = _trajectory_stages(refinement, mol_input)
    input_preparation_seconds = time.perf_counter() - pipeline_started
    _synchronize(device)
    model_load_started = time.perf_counter()
    model, _, docking_ckpt = load_model(args.config, args.docking_checkpoint, device)
    confidence_model, confidence_ckpt = load_pose_confidence_model(
        args.confidence_checkpoint, device
    )
    _synchronize(device)
    model_load_seconds = time.perf_counter() - model_load_started
    stage_scores: dict[str, list[dict[str, float]]] = {}
    confidence_forward_seconds: dict[str, float] = {}
    selector_seconds: dict[str, float] = {}
    selected: dict[str, dict[str, float | int | bool]] = {}
    for stage, spec in stage_specs.items():
        centered = list(stage_coordinates[stage])
        _synchronize(device)
        scoring_started = time.perf_counter()
        scores = _score_in_batches(
            confidence_model,
            model,
            graph,
            ligand_data,
            meta,
            centered,
            sigma=args.sigma,
            device=device,
            batch_size=args.pose_batch_size,
        )
        _synchronize(device)
        confidence_forward_seconds[stage] = time.perf_counter() - scoring_started
        stage_scores[stage] = scores
        selection_started = time.perf_counter()
        index = _select_index(scores)
        selector_seconds[stage] = time.perf_counter() - selection_started
        pose_metrics = refinement["poses"][index]
        rmsd_key = (
            "initial_symmetry_rmsd_angstrom"
            if stage == "step_000"
            else "final_symmetry_rmsd_angstrom"
        )
        selected[stage] = {
            "pose_index": index,
            "confidence_rmsd": scores[index]["confidence_rmsd"],
            "confidence_success": scores[index]["confidence_success"],
            "symmetry_rmsd_angstrom": float(pose_metrics[rmsd_key]),
            "rmsd_lt_2": float(pose_metrics[rmsd_key]) < 2.0,
        }

    source_manifest = json.loads(Path(inputs["manifest"]).read_text(encoding="utf-8"))
    source_record = {
        "id": complex_id,
        "pose_path": inputs["pose_sdf"],
    }
    source_row = _load_source_row(source_manifest, source_record)
    source_score_json = source_row.get("confidence_candidate_scores_json", "")
    if source_score_json:
        expected_scores = json.loads(source_score_json)
        if len(expected_scores) != EXPECTED_POSES:
            raise ValueError("source confidence ledger pose count mismatch")
        baseline_deltas = [
            abs(float(actual["confidence_rmsd"]) - float(expected["confidence_rmsd"]))
            for actual, expected in zip(
                stage_scores["step_000"], expected_scores, strict=True
            )
        ]
        expected_index: int | None = int(source_row["confidence_index"])
        baseline_index_matches: bool | None = (
            selected["step_000"]["pose_index"] == expected_index
        )
    else:
        baseline_deltas = []
        expected_index = None
        baseline_index_matches = None
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    attempt = Path(
        tempfile.mkdtemp(prefix=f".{args.output_dir.name}.attempt-", dir=args.output_dir.parent)
    )
    serialization_started = time.perf_counter()
    score_rows: list[dict[str, Any]] = []
    for pose_index in range(EXPECTED_POSES):
        score_rows.append(
            {
                "pose_index": pose_index,
                **{
                    f"before_{key}": value
                    for key, value in stage_scores["step_000"][pose_index].items()
                },
                **{
                    f"after_{key}": value
                    for key, value in stage_scores["step_100"][pose_index].items()
                },
                "initial_symmetry_rmsd_angstrom": refinement["poses"][pose_index][
                    "initial_symmetry_rmsd_angstrom"
                ],
                "final_symmetry_rmsd_angstrom": refinement["poses"][pose_index][
                    "final_symmetry_rmsd_angstrom"
                ],
            }
        )
    scores_path = attempt / "scores.csv"
    with scores_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(score_rows[0]), extrasaction="raise")
        writer.writeheader()
        writer.writerows(score_rows)
    score_serialization_seconds = time.perf_counter() - serialization_started
    summary = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "complete_descriptive",
        "selector": (
            "fresh within-run argmin confidence_rmsd at both step 0 and step 100; "
            "PL-valid and RMSD are outcomes only"
        ),
        "dataset": dataset,
        "complex_id": complex_id,
        "pose_count": EXPECTED_POSES,
        "sigma": args.sigma,
        "pose_batch_size": args.pose_batch_size,
        "pocket_cutoff_angstrom": pocket_cutoff,
        "selected": selected,
        "selector_changed": selected["step_000"]["pose_index"]
        != selected["step_100"]["pose_index"],
        "historical_baseline_reproduction": {
            "source_selected_index": expected_index,
            "rescored_selected_index": selected["step_000"]["pose_index"],
            "selected_index_matches": baseline_index_matches,
            "max_abs_predicted_rmsd_delta": (
                max(baseline_deltas) if baseline_deltas else None
            ),
            "role": (
                "diagnostic_only_not_a_completion_gate"
                if baseline_deltas
                else "unavailable_source_sampling_skipped_confidence"
            ),
        },
        "inputs": {
            "refinement_summary": str(refinement_path),
            "refinement_summary_sha256": file_sha256(refinement_path),
            "benchmark_input_identity": {
                key: value for key, value in mapping_identity.items() if key != "per_id"
            },
            "config": str(args.config.resolve()),
            "config_sha256": file_sha256(args.config),
            "docking_checkpoint": str(args.docking_checkpoint.resolve()),
            "docking_checkpoint_sha256": file_sha256(args.docking_checkpoint),
            "confidence_checkpoint": str(args.confidence_checkpoint.resolve()),
            "confidence_checkpoint_sha256": file_sha256(args.confidence_checkpoint),
        },
        "checkpoints": {
            "docking_step": docking_ckpt.get("step"),
            "confidence_step": confidence_ckpt.get("step"),
        },
        "artifacts": {
            "scores_csv": {
                "path": str(args.output_dir.resolve() / "scores.csv"),
                "sha256": file_sha256(scores_path),
            }
        },
        "runtime": {
            "stage_seconds": {
                "input_preparation": input_preparation_seconds,
                "model_load": model_load_seconds,
                "confidence_forward_by_pose_stage": confidence_forward_seconds,
                "confidence_forward_total": sum(confidence_forward_seconds.values()),
                "selector_by_pose_stage": selector_seconds,
                "selector_total": sum(selector_seconds.values()),
                "score_serialization": score_serialization_seconds,
            },
            "confidence_forward_seconds_per_scored_pose": (
                sum(confidence_forward_seconds.values()) / (2 * EXPECTED_POSES)
            ),
            "selector_seconds_per_candidate_set": (
                sum(selector_seconds.values()) / len(selector_seconds)
            ),
            "wall_seconds_before_summary_write": time.perf_counter() - pipeline_started,
            "device": str(device),
            "cuda_device_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else None
            ),
            "torch": torch.__version__,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
        },
    }
    (attempt / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.rename(attempt, args.output_dir)
    print(json.dumps({"output_dir": str(args.output_dir.resolve()), **selected}, sort_keys=True))


if __name__ == "__main__":
    main()
