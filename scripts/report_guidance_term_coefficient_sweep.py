#!/usr/bin/env python3
"""Aggregate paired RMSD, fast-valid, and cap telemetry without selecting an arm."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from datetime import UTC, datetime
from pathlib import Path

ARMS = ("base_r080_c1", "steric_r090_c1", "chiral_r080_c2", "combined_r090_c2")
DATASETS = {"astex": 85, "posebusters": 308}
SHARDS = 8


def _rows(root: Path, dataset: str, arm: str) -> list[dict[str, str]]:
    prefix = f"effdock-guidance-term-coeff-v1-{dataset}-n100-s10-{arm}"
    result: list[dict[str, str]] = []
    for shard in range(SHARDS):
        path = root / "raw" / f"{prefix}.shard-{shard:03d}-of-{SHARDS:03d}.csv"
        with path.open(newline="", encoding="utf-8") as handle:
            result.extend(csv.DictReader(handle))
    if len(result) != DATASETS[dataset] or len({row["id"] for row in result}) != len(result):
        raise ValueError(f"{dataset}/{arm}: incomplete or duplicate inventory")
    return result


def _truth(value: str) -> bool:
    return value.lower() == "true"


def _pct(count: int, total: int) -> float:
    return 100.0 * count / total


def _arm(rows: list[dict[str, str]], dataset: str, arm: str) -> dict:
    confidence = [float(row["confidence_rmsd"]) for row in rows]
    valid = [_truth(row["confidence_fast_valid"]) for row in rows]
    oracle = [float(row["oracle_rmsd"]) for row in rows]
    evals = sum(float(row["guidance_direct_pose_evaluations"]) for row in rows)
    cap = sum(float(row["guidance_direct_any_cap_trigger_count"]) for row in rows)
    return {
        "dataset": dataset,
        "arm": arm,
        "n": len(rows),
        "confidence_rmsd_lt2_pct": _pct(sum(value < 2.0 for value in confidence), len(rows)),
        "confidence_median_rmsd": statistics.median(confidence),
        "confidence_fast_valid_pct": _pct(sum(valid), len(rows)),
        "confidence_joint_lt2_fast_valid_pct": _pct(sum(r < 2.0 and v for r, v in zip(confidence, valid)), len(rows)),
        "oracle_rmsd_lt2_pct": _pct(sum(value < 2.0 for value in oracle), len(rows)),
        "any_cap_pct": 100.0 * cap / evals if evals else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()
    arms = [_arm(_rows(args.output_root, dataset, arm), dataset, arm) for dataset in DATASETS for arm in ARMS]
    result = {
        "schema_version": "effdock.guidance_term_coefficient_sweep_report.v1",
        "status": "complete",
        "created_utc": datetime.now(UTC).isoformat(),
        "eta": 2.0,
        "sigma": 0.5,
        "automatic_selection": False,
        "official_posebusters_validity": False,
        "validity_label": "internal_fast_valid",
        "arms": arms,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = ["# Guidance term-coefficient sweep", "", "> Descriptive only. Validity is internal fast-valid, not official PoseBusters.", ""]
    for dataset in DATASETS:
        lines += [f"## {dataset}", "", "| arm | Confidence <2A | Fast-valid | Joint | Median RMSD | Oracle <2A | Any cap |", "|---|---:|---:|---:|---:|---:|---:|"]
        for value in (item for item in arms if item["dataset"] == dataset):
            lines.append(f"| {value['arm']} | {value['confidence_rmsd_lt2_pct']:.1f}% | {value['confidence_fast_valid_pct']:.1f}% | {value['confidence_joint_lt2_fast_valid_pct']:.1f}% | {value['confidence_median_rmsd']:.3f} | {value['oracle_rmsd_lt2_pct']:.1f}% | {value['any_cap_pct']:.1f}% |")
        lines.append("")
    args.output_md.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": "complete", "arms": len(arms)}, sort_keys=True))


if __name__ == "__main__":
    main()
