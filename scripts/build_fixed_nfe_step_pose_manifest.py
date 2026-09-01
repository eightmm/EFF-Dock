#!/usr/bin/env python3
"""Validate N40/S25 sampling and freeze its paired refinement manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

SOURCE_PROTOCOL_ID = "EFFDOCK-FIXED-NFE-STEP-POSE-V1"
MANIFEST_PROTOCOL_ID = "EFFDOCK-FIXED-NFE-STEP-POSE-REFINEMENT-INPUT-V1"
BASELINE_MANIFEST_SHA256 = "9e8be4d47dba8e346a6900b6bf02f5b853a93141f571f5c65b4c719de632d695"
DATASETS = {"astex": 85, "posebusters": 308}
SMOKE_IDS = {"astex": "1jje", "posebusters": "7b2c_tp7"}
POSES = 40
STEPS = 25
PRIOR_POOL_SIZE = 100
SIGMA = 2.0
ETA = 2.0


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def _source_stem(dataset: str) -> str:
    return f"effdock-fixed-nfe-step-pose-v1-{dataset}-n40-s25-sigma2000"


def _sampling_artifact_paths(
    root: Path,
    *,
    stem: str,
    mode: str,
    shard: int,
    shards: int,
) -> tuple[Path, Path]:
    """Return evaluator output paths for single- or multi-shard sampling.

    ``eff-dock evaluate`` deliberately omits the shard suffix when
    ``num_shards == 1``.  Full sampling uses eight shards and includes it.
    Keep that evaluator convention explicit here so the smoke gate validates
    the files that sampling actually materialized.
    """
    if mode == "smoke":
        if shard != 0 or shards != 1:
            raise ValueError("smoke sampling must be exactly one shard")
        return root / f"{stem}.csv", root / f"{stem}.summary.json"
    suffix = f"shard-{shard:03d}-of-{shards:03d}"
    return root / f"{stem}.{suffix}.csv", root / f"{stem}.{suffix}.summary.json"


def _require_file(path: Path, expected_sha256: str | None = None) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = file_sha256(path)
    if expected_sha256 is not None and actual != expected_sha256:
        raise ValueError(f"SHA-256 mismatch for {path}: {actual} != {expected_sha256}")
    return actual


def _load_baseline(path: Path) -> tuple[dict[str, Any], dict[tuple[str, str], dict[str, Any]]]:
    _require_file(path, BASELINE_MANIFEST_SHA256)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("protocol_id") != "EFFDOCK-GUIDANCE-SIGMA2-ETA2-REFINEMENT-INPUT-V1"
        or int(payload.get("num_steps", -1)) != 10
        or int(payload.get("poses_per_complex", -1)) != 100
        or float(payload.get("sigma", -1)) != SIGMA
        or float(payload.get("eta", -1)) != ETA
    ):
        raise ValueError("unexpected reused N100/S10 baseline manifest")
    records: dict[tuple[str, str], dict[str, Any]] = {}
    for row in payload.get("records", []):
        key = (str(row["dataset"]), str(row["id"]).lower())
        if key in records:
            raise ValueError(f"duplicate baseline record: {key}")
        records[key] = row
    if len(records) != sum(DATASETS.values()):
        raise ValueError(f"expected 393 baseline records, got {len(records)}")
    return payload, records


def _validate_summary(path: Path, *, dataset: str, shards: int, shard: int) -> dict[str, Any]:
    summary = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "protocol_id": SOURCE_PROTOCOL_ID,
        "dataset": dataset,
        "run_name": _source_stem(dataset),
        "num_samples": POSES,
        "num_steps": STEPS,
        "prior_pool_size": PRIOR_POOL_SIZE,
        "sigma": SIGMA,
        "time_schedule": "late",
        "schedule_power": 3.0,
        "pocket_cutoff": 10.0,
        "center_jitter_sigma": 0.0,
        "unified_guidance_mode": "normalized_drift",
        "unified_guidance_scale": ETA,
        "unified_guidance_start_t": 0.5,
        "unified_guidance_ramp_power": 1.0,
        "unified_guidance_max_force": 20.0,
        "unified_guidance_max_velocity": 5.0,
        "unified_guidance_max_angular_velocity": 5.0,
        "unified_guidance_max_atom_displacement": 0.25,
        "unified_guidance_max_backtracks": 8,
        "unified_guidance_protein_shell": 18.0,
        "seed": 42,
        "num_shards": shards,
        "shard_index": shard,
        "checkpoint": "weights/effdock_geometry_ft_100k_best.pt",
        "checkpoint_step": 100000,
        "confidence_checkpoint": "weights/effdock_confidence_extmatch_n80_s25_step42500.pt",
        "confidence_step": 42500,
    }
    changed = {
        key: {"actual": summary.get(key), "expected": value}
        for key, value in expected.items()
        if summary.get(key) != value
    }
    if changed:
        raise ValueError(f"sampling summary contract mismatch at {path}: {changed}")
    runtime = summary.get("runtime", {})
    if runtime.get("slurm_partition") != "test":
        raise ValueError(f"sampling must run on test for prior pairing: {path}")
    if "RTX A5000" not in str(runtime.get("gpu", "")):
        raise ValueError(f"sampling used the wrong GPU/CPU node family: {path}")
    parameter_sha = str(summary.get("guidance_parameter_set", {}).get("sha256", ""))
    if not parameter_sha:
        raise ValueError(f"missing guidance parameter identity: {path}")
    return summary


def _load_rows(
    root: Path,
    *,
    mode: str,
    baseline: dict[tuple[str, str], dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, str]], list[dict[str, str]]]:
    shards = 1 if mode == "smoke" else 8
    records: list[dict[str, Any]] = []
    source_files: list[dict[str, str]] = []
    source_summaries: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for dataset, expected_full in DATASETS.items():
        stem = _source_stem(dataset)
        for shard in range(shards):
            csv_path, summary_path = _sampling_artifact_paths(
                root,
                stem=stem,
                mode=mode,
                shard=shard,
                shards=shards,
            )
            csv_sha = _require_file(csv_path)
            summary_sha = _require_file(summary_path)
            _validate_summary(summary_path, dataset=dataset, shards=shards, shard=shard)
            source_files.append({"path": str(csv_path.resolve()), "sha256": csv_sha})
            source_summaries.append(
                {"path": str(summary_path.resolve()), "sha256": summary_sha}
            )
            with csv_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            for row in rows:
                complex_id = str(row["id"]).lower()
                key = (dataset, complex_id)
                if key in seen:
                    raise ValueError(f"duplicate sampled complex: {key}")
                seen.add(key)
                baseline_row = baseline.get(key)
                if baseline_row is None:
                    raise ValueError(f"sampled complex absent from baseline: {key}")
                if int(row["all_poses_count"]) != POSES or int(row["num_samples"]) != POSES:
                    raise ValueError(f"{key}: expected exactly {POSES} saved poses")
                if int(row["prior_pool_size"]) != PRIOR_POOL_SIZE:
                    raise ValueError(f"{key}: expected prior pool size {PRIOR_POOL_SIZE}")
                if int(row["sampling_seed"]) != int(baseline_row["sampling_seed"]):
                    raise ValueError(f"{key}: sampling seed is not paired")
                if row["prior_pool_sha256"] != baseline_row["prior_pool_sha256"]:
                    raise ValueError(f"{key}: prior-pool hash is not paired")
                if row["guidance_parameter_sha256"] != baseline_row["guidance_parameter_sha256"]:
                    raise ValueError(f"{key}: guidance parameter identity changed")
                if row["guidance_mode"] != "unified_normalized_drift":
                    raise ValueError(f"{key}: unexpected guidance mode")
                for row_key, baseline_key in (
                    ("protein_sha256", "protein_sha256"),
                    ("ligand_reference_sha256", "ligand_ref_sha256"),
                ):
                    if row[row_key] != baseline_row[baseline_key]:
                        raise ValueError(f"{key}: changed frozen input {row_key}")
                scores = json.loads(row["confidence_candidate_scores_json"])
                if len(scores) != POSES or not all(
                    math.isfinite(float(item["confidence_rmsd"])) for item in scores
                ):
                    raise ValueError(f"{key}: invalid source confidence ledger")
                pose_path = Path(row["all_poses_sdf"])
                protein = Path(row["protein"])
                ligand_ref = Path(row["ligand_ref"])
                _require_file(pose_path, row["all_poses_sdf_sha256"])
                _require_file(protein, row["protein_sha256"])
                _require_file(ligand_ref, row["ligand_reference_sha256"])
                records.append(
                    {
                        "dataset": dataset,
                        "id": complex_id,
                        "eta": ETA,
                        "sigma": SIGMA,
                        "num_steps": STEPS,
                        "pose_path": str(pose_path.resolve()),
                        "pose_sha256": row["all_poses_sdf_sha256"],
                        "pose_count": POSES,
                        "protein": str(protein.resolve()),
                        "protein_sha256": row["protein_sha256"],
                        "ligand_ref": str(ligand_ref.resolve()),
                        "ligand_ref_sha256": row["ligand_reference_sha256"],
                        "sampling_seed": int(row["sampling_seed"]),
                        "prior_pool_size": int(row["prior_pool_size"]),
                        "prior_pool_sha256": row["prior_pool_sha256"],
                        "guidance_mode": row["guidance_mode"],
                        "guidance_parameter_sha256": row["guidance_parameter_sha256"],
                        "source_row_sha256": _canonical_sha256(row),
                    }
                )
        dataset_ids = {complex_id for row_dataset, complex_id in seen if row_dataset == dataset}
        if mode == "smoke":
            if dataset_ids != {SMOKE_IDS[dataset]}:
                raise ValueError(f"{dataset}: smoke inventory mismatch: {dataset_ids}")
        elif len(dataset_ids) != expected_full:
            raise ValueError(
                f"{dataset}: expected {expected_full} sampled complexes, got {len(dataset_ids)}"
            )
    return records, source_files, source_summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sampling-root", type=Path, required=True)
    parser.add_argument("--baseline-manifest", type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    baseline_payload, baseline = _load_baseline(args.baseline_manifest)
    records, source_files, source_summaries = _load_rows(
        args.sampling_root.resolve(), mode=args.mode, baseline=baseline
    )
    records.sort(key=lambda row: (row["dataset"], row["id"]))
    payload = {
        "schema_version": "effdock.fixed_nfe_step_pose_refinement_input.v1",
        "protocol_id": MANIFEST_PROTOCOL_ID,
        "source_protocol_id": SOURCE_PROTOCOL_ID,
        "mode": args.mode,
        "status": "complete",
        "baseline_manifest": str(args.baseline_manifest.resolve()),
        "baseline_manifest_sha256": BASELINE_MANIFEST_SHA256,
        "baseline_protocol_id": baseline_payload["protocol_id"],
        "arms": {
            "s10_n100": {"num_steps": 10, "poses": 100, "learned_pose_steps": 1000},
            "s25_n40": {"num_steps": STEPS, "poses": POSES, "learned_pose_steps": 1000},
        },
        "sigma": SIGMA,
        "eta": ETA,
        "prior_pool_size": PRIOR_POOL_SIZE,
        "datasets": (
            {dataset: 1 for dataset in DATASETS}
            if args.mode == "smoke"
            else DATASETS
        ),
        "expected_complexes": len(records),
        "expected_poses": len(records) * POSES,
        "pairing": "same sampling_seed and exact 100-pose prior_pool_sha256 per complex",
        "source_files": source_files,
        "source_summaries": source_summaries,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.link(temporary, args.output)
    temporary.unlink()
    print(
        json.dumps(
            {
                "status": "complete",
                "mode": args.mode,
                "complexes": len(records),
                "poses": len(records) * POSES,
                "output": str(args.output.resolve()),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
