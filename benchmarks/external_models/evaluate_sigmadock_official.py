#!/usr/bin/env python3
"""Evaluate one N=40 SigmaDock pool with the paper's Vinardo/PB heuristic."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import torch

PAPER_HEURISTIC_CHECKS = (
    "minimum_distance_to_protein",
    "tetrahedral_chirality",
    "internal_energy",
    "internal_steric_clash",
    "double_bond_flatness",
    "bond_lengths",
    "bond_angles",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--pool-size", type=int, default=40)
    parser.add_argument("--expected-targets", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def target_from_key(key: str) -> str:
    return key.split("::", 1)[0]


def main() -> None:
    args = parse_args()
    with args.input_csv.open(newline="") as handle:
        input_rows = list(csv.DictReader(handle))
    if len(input_rows) != args.expected_targets:
        raise ValueError(
            f"expected {args.expected_targets} inputs, found {len(input_rows)}"
        )
    targets = ["_".join(Path(row["PDB"]).stem.split("_")[:2]) for row in input_rows]
    if len(set(targets)) != len(targets):
        raise ValueError("input target IDs are not unique")

    candidates: dict[str, list[dict[str, object]]] = {target: [] for target in targets}
    seeds = range(args.seed_start, args.seed_start + args.pool_size)
    versions: set[str] = set()
    for seed in seeds:
        seed_dir = args.model_root / f"seed_{seed}"
        predictions_path = seed_dir / "predictions.pt"
        rescoring_path = seed_dir / "rescoring.pt"
        posebusters_path = seed_dir / "posebusters.pt"
        if not all(path.is_file() for path in (predictions_path, rescoring_path, posebusters_path)):
            raise FileNotFoundError(f"seed {seed} is incomplete: {seed_dir}")
        predictions = torch.load(predictions_path, map_location="cpu", weights_only=False)["results"]
        rescoring = torch.load(rescoring_path, map_location="cpu", weights_only=False)["scores"]
        pb_payload = torch.load(posebusters_path, map_location="cpu", weights_only=False)
        versions.add(str(pb_payload.get("metadata", {}).get("posebusters_version", "unknown")))
        rmsds = pb_payload["rmsds"]
        pb_checks = pb_payload["pb_checks"]
        pb_dicts = pb_payload["pb_dicts"]
        if not (set(predictions) == set(rescoring) == set(rmsds) == set(pb_checks) == set(pb_dicts)):
            raise ValueError(f"seed {seed} prediction/score/PoseBusters keys differ")
        for key in predictions:
            target = target_from_key(key)
            affinity = float(rescoring[key][0]["Affinity"])
            checks = pb_dicts[key]
            missing = [name for name in PAPER_HEURISTIC_CHECKS if name not in checks]
            if missing:
                raise KeyError(f"seed {seed} target {target} missing checks: {missing}")
            p_value = sum(bool(checks[name]) for name in PAPER_HEURISTIC_CHECKS) / len(
                PAPER_HEURISTIC_CHECKS
            )
            score = -affinity * p_value**4
            full_pb_valid = math.isclose(float(pb_checks[key]), 1.0, abs_tol=1e-12)
            candidates[target].append(
                {
                    "seed": seed,
                    "rmsd": float(rmsds[key]),
                    "affinity": affinity,
                    "paper_pb_fraction": p_value,
                    "paper_score": score,
                    "pb_valid": full_pb_valid,
                }
            )

    rows: list[dict[str, object]] = []
    for target in targets:
        values = candidates[target]
        if len(values) != args.pool_size:
            raise ValueError(f"{target}: expected {args.pool_size} candidates, found {len(values)}")
        selected = max(values, key=lambda row: (float(row["paper_score"]), -int(row["seed"])))
        oracle = min(values, key=lambda row: float(row["rmsd"]))
        rows.append(
            {
                "complex_name": target,
                "selected_seed": selected["seed"],
                "top1_rmsd": selected["rmsd"],
                "top1_pb_valid": selected["pb_valid"],
                "top1_joint": float(selected["rmsd"]) < 2.0 and bool(selected["pb_valid"]),
                "top1_affinity": selected["affinity"],
                "top1_paper_pb_fraction": selected["paper_pb_fraction"],
                "top1_paper_score": selected["paper_score"],
                "oracle_rmsd": oracle["rmsd"],
                "oracle_seed": oracle["seed"],
            }
        )

    denominator = len(rows)
    top1_rmsd_count = sum(float(row["top1_rmsd"]) < 2.0 for row in rows)
    top1_pb_count = sum(bool(row["top1_pb_valid"]) for row in rows)
    top1_joint_count = sum(bool(row["top1_joint"]) for row in rows)
    oracle_count = sum(float(row["oracle_rmsd"]) < 2.0 for row in rows)
    summary = {
        "schema_version": 1,
        "model": "SigmaDock",
        "protocol": "paper official N=40: score_i=-Vinardo_i*p_i^4",
        "input_csv": str(args.input_csv.resolve()),
        "model_root": str(args.model_root.resolve()),
        "seed_start": args.seed_start,
        "seed_end": args.seed_start + args.pool_size - 1,
        "pool_size": args.pool_size,
        "denominator": denominator,
        "posebusters_versions": sorted(versions),
        "heuristic_checks": list(PAPER_HEURISTIC_CHECKS),
        "top1_rmsd_lt2_count": top1_rmsd_count,
        "top1_rmsd_lt2_pct": 100.0 * top1_rmsd_count / denominator,
        "top1_pb_valid_count": top1_pb_count,
        "top1_pb_valid_pct": 100.0 * top1_pb_count / denominator,
        "top1_joint_count": top1_joint_count,
        "top1_joint_pct": 100.0 * top1_joint_count / denominator,
        "oracle_rmsd_lt2_count": oracle_count,
        "oracle_rmsd_lt2_pct": 100.0 * oracle_count / denominator,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "per_target.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
