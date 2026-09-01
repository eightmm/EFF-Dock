#!/usr/bin/env python3
"""Freeze one cutoff/repeat EFF-Dock all-pose bank for refinement."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from effdock.workflows.evaluate import file_sha256

PROTOCOL_ID = "EFFDOCK-POCKET-CUTOFF-ROBUSTNESS-MANIFEST-V1"
SOURCE_PROTOCOL_ID = "EFFDOCK-POCKET-CUTOFF-ROBUSTNESS-V1"
DATASETS = {"astex": 85, "posebusters": 308}
SHARDS = 8
POSES = 100
STEPS = 10
SIGMA = 2.0
ETA = 2.0
DOCKING_SHA256 = "65be44d7dc8f0867eb9fc5d22214b80f93971ea4702679a527c665046e91e6b6"
CONFIDENCE_SHA256 = "ce59be42f0ca613871ca079127c3296f5ca9a4ec72e44a9e5cf61878351c2638"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition-root", type=Path, required=True)
    parser.add_argument("--cutoff", type=float, choices=(6.0, 8.0, 10.0, 12.0), required=True)
    parser.add_argument("--repeat-index", type=int, choices=range(3), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def require_summary(path: Path, dataset: str, cutoff: float, repeat: int) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "protocol_id": SOURCE_PROTOCOL_ID,
        "dataset": dataset,
        "num_samples": POSES,
        "num_steps": STEPS,
        "sigma": SIGMA,
        "pocket_cutoff": cutoff,
        "unified_guidance_scale": ETA,
        "unified_guidance_mode": "normalized_drift",
        "checkpoint_sha256": DOCKING_SHA256,
    }
    changed = {key: (value.get(key), target) for key, target in expected.items() if value.get(key) != target}
    if changed:
        raise ValueError(f"unexpected source contract in {path}: {changed}")
    expected_seed = 42 + repeat * 100_000
    if int(value.get("seed", -1)) != expected_seed:
        raise ValueError(f"unexpected base seed in {path}")
    if int(value.get("num_success", -1)) != int(value.get("num_assigned", -2)):
        raise ValueError(f"incomplete source shard: {path}")
    return value


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    raw_root = args.condition_root / "raw"
    records: list[dict[str, object]] = []
    source_files: list[dict[str, str]] = []
    source_summaries: list[dict[str, str]] = []
    dataset_counts = {"astex": 1} if args.smoke else DATASETS
    shard_count = 1 if args.smoke else SHARDS
    for dataset, expected_count in dataset_counts.items():
        stem = (
            f"effdock-pocket-cutoff-v1-{dataset}-c{int(args.cutoff):02d}-"
            f"r{args.repeat_index}-n100-s10"
        )
        seen: set[str] = set()
        for shard in range(shard_count):
            if args.smoke:
                csv_path = raw_root / f"{stem}.csv"
                summary_path = raw_root / f"{stem}.summary.json"
            else:
                suffix = f"shard-{shard:03d}-of-{shard_count:03d}"
                csv_path = raw_root / f"{stem}.{suffix}.csv"
                summary_path = raw_root / f"{stem}.{suffix}.summary.json"
            if not csv_path.is_file() or not summary_path.is_file():
                raise FileNotFoundError(f"missing source shard {csv_path} / {summary_path}")
            require_summary(summary_path, dataset, args.cutoff, args.repeat_index)
            source_files.append({"path": str(csv_path), "sha256": file_sha256(csv_path)})
            source_summaries.append({"path": str(summary_path), "sha256": file_sha256(summary_path)})
            with csv_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            for row in rows:
                complex_id = row["id"].lower()
                if complex_id in seen:
                    raise ValueError(f"duplicate target {dataset}/{complex_id}")
                seen.add(complex_id)
                if int(row["all_poses_count"]) != POSES:
                    raise ValueError(f"{dataset}/{complex_id}: expected 100 poses")
                if row["guidance_mode"] != "unified_normalized_drift":
                    raise ValueError(f"{dataset}/{complex_id}: guidance mode changed")
                pose = Path(row["all_poses_sdf"])
                protein = Path(row["protein"])
                ligand = Path(row["ligand_ref"])
                for path, expected in (
                    (pose, row["all_poses_sdf_sha256"]),
                    (protein, row["protein_sha256"]),
                    (ligand, row["ligand_reference_sha256"]),
                ):
                    if not path.is_file() or file_sha256(path) != expected:
                        raise ValueError(f"missing or changed source asset {path}")
                records.append(
                    {
                        "dataset": dataset,
                        "id": complex_id,
                        "eta": ETA,
                        "sigma": SIGMA,
                        "pocket_cutoff_angstrom": args.cutoff,
                        "repeat_index": args.repeat_index,
                        "pose_path": str(pose),
                        "pose_sha256": row["all_poses_sdf_sha256"],
                        "pose_count": POSES,
                        "protein": str(protein),
                        "protein_sha256": row["protein_sha256"],
                        "ligand_ref": str(ligand),
                        "ligand_ref_sha256": row["ligand_reference_sha256"],
                        "sampling_seed": int(row["sampling_seed"]),
                        "prior_pool_sha256": row["prior_pool_sha256"],
                    }
                )
        if len(seen) != expected_count:
            raise ValueError(f"{dataset}: {len(seen)} targets != {expected_count}")
    if len(records) != sum(dataset_counts.values()):
        raise AssertionError("combined manifest count mismatch")
    records.sort(key=lambda row: (str(row["dataset"]), str(row["id"])))
    payload = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "source_protocol_id": SOURCE_PROTOCOL_ID,
        "condition": {"pocket_cutoff_angstrom": args.cutoff, "repeat_index": args.repeat_index},
        "eta": ETA,
        "sigma": SIGMA,
        "num_steps": STEPS,
        "poses_per_complex": POSES,
        "datasets": dataset_counts,
        "smoke": args.smoke,
        "docking_checkpoint_sha256": DOCKING_SHA256,
        "confidence_checkpoint_sha256": CONFIDENCE_SHA256,
        "source_files": source_files,
        "source_summaries": source_summaries,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
