#!/usr/bin/env python3
"""Prepare an isolated Interformer PoseScore workspace from native poses."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-shard", type=Path, required=True)
    parser.add_argument("--score-dir", type=Path, required=True)
    parser.add_argument("--expected-poses", type=int, default=20)
    return parser.parse_args()


def replace_symlink(link: Path, target: Path) -> None:
    if link.is_symlink():
        if link.resolve() == target.resolve():
            return
        link.unlink()
    elif link.exists():
        raise RuntimeError(f"refusing to replace non-symlink path: {link}")
    link.symlink_to(target.resolve(), target_is_directory=target.is_dir())


def main() -> None:
    args = parse_args()
    source = args.source_shard.resolve()
    score_dir = args.score_dir.resolve()
    source_work = source / "work"
    reconstructed = source / "energy" / "ligand_reconstructing"
    query_path = source_work / "query.csv"
    stat_path = reconstructed / "stat_concated.csv"

    for required in (query_path, stat_path, source_work / "pocket", source_work / "uff"):
        if not required.exists():
            raise FileNotFoundError(required)

    query = pd.read_csv(query_path)
    stat = pd.read_csv(stat_path)
    stat = stat.rename(columns={"pdb_id": "Target"})
    required_stat = {"Target", "pose_rank", "num_torsions", "energy", "rmsd"}
    missing = required_stat.difference(stat.columns)
    if missing:
        raise ValueError(f"missing stat columns: {sorted(missing)}")

    # Upstream appends the input conformer at pose_rank == expected_poses.
    stat = stat[stat["pose_rank"].astype(int) < args.expected_poses].copy()
    stat["pose_rank"] = stat["pose_rank"].astype(int)
    if "pose_rank" in query.columns:
        query = query.drop(columns="pose_rank")
    merged = query.merge(
        stat[["Target", "pose_rank", "num_torsions", "energy", "rmsd"]],
        on="Target",
        how="inner",
        validate="one_to_many",
    )

    expected_targets = set(query["Target"].astype(str))
    scored_targets = set(merged["Target"].astype(str))
    if scored_targets != expected_targets:
        absent = sorted(expected_targets - scored_targets)
        raise RuntimeError(f"native pose statistics missing targets: {absent}")
    counts = merged.groupby("Target")["pose_rank"].nunique()
    short = counts[counts != args.expected_poses]
    if not short.empty:
        raise RuntimeError(f"expected {args.expected_poses} poses per target: {short.to_dict()}")

    work = score_dir / "work"
    work.mkdir(parents=True, exist_ok=True)
    replace_symlink(work / "pocket", source_work / "pocket")
    replace_symlink(work / "uff", source_work / "uff")
    replace_symlink(work / "infer", reconstructed)
    output_csv = work / "query.round0.csv"
    merged.to_csv(output_csv, index=False)

    metadata = {
        "schema_version": 1,
        "source_shard": str(source),
        "score_dir": str(score_dir),
        "input_csv": str(output_csv),
        "targets": len(expected_targets),
        "poses": len(merged),
        "poses_per_target": args.expected_poses,
        "input_conformer_policy": "excluded_before_pose_scoring",
        "ligand_source": str(reconstructed),
    }
    (score_dir / "prepare_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    # A stale model cache can silently retain a different ligand ordering.
    cache = work / "tmp_beta"
    if cache.exists():
        shutil.rmtree(cache)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
