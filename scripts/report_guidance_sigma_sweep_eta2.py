#!/usr/bin/env python3
"""Aggregate eta-2 sigma-sweep RMSD and internal fast-valid outcomes."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

SIGMAS = (0.5, 1.0, 2.0, 3.0, 4.0)
DATASETS = {"astex": 85, "posebusters": 308}
SHARDS = 8


def _tag(sigma: float) -> str:
    return f"sigma{int(round(sigma * 1000)):04d}"


def _rows(root: Path, reference: Path, dataset: str, sigma: float) -> list[dict[str, str]]:
    if sigma == 0.5:
        prefix = f"effdock-guidance-steric-high-eta-v1-{dataset}-n100-s10-eta2000"
        base = reference / "raw"
    else:
        prefix = f"effdock-guidance-sigma-sweep-eta2-v1-{dataset}-n100-s10-{_tag(sigma)}"
        base = root / "raw"
    output: list[dict[str, str]] = []
    for shard in range(SHARDS):
        path = base / f"{prefix}.shard-{shard:03d}-of-{SHARDS:03d}.csv"
        with path.open(newline="", encoding="utf-8") as handle:
            output.extend(csv.DictReader(handle))
    if len(output) != DATASETS[dataset] or len({row["id"] for row in output}) != len(output):
        raise ValueError(f"{dataset}/{sigma}: incomplete or duplicate inventory")
    return output


def _truth(value: str) -> bool:
    return value.lower() == "true"


def _pct(values: list[bool]) -> float:
    return 100.0 * sum(values) / len(values)


def _metrics(rows: list[dict[str, str]], selector: str) -> dict[str, float]:
    rmsd = [float(row[f"{selector}_rmsd"]) for row in rows]
    valid = [_truth(row[f"{selector}_fast_valid"]) for row in rows]
    return {
        "rmsd_lt2_pct": _pct([value < 2.0 for value in rmsd]),
        "median_rmsd": statistics.median(rmsd),
        "fast_valid_pct": _pct(valid),
        "joint_rmsd_lt2_fast_valid_pct": _pct([value < 2.0 and ok for value, ok in zip(rmsd, valid)]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()
    arms = []
    for dataset in DATASETS:
        for sigma in SIGMAS:
            rows = _rows(args.output_root, args.reference_root, dataset, sigma)
            oracle = [float(row["oracle_rmsd"]) for row in rows]
            counts = [int(row["num_fast_valid_candidates"]) for row in rows]
            arms.append({
                "dataset": dataset,
                "sigma": sigma,
                "source": "frozen_reference" if sigma == 0.5 else "new_sweep",
                "n": len(rows),
                "confidence": _metrics(rows, "confidence"),
                "confidence_filter": _metrics(rows, "confidence_filter"),
                "oracle_rmsd_lt2_pct": _pct([value < 2.0 for value in oracle]),
                "oracle_median_rmsd": statistics.median(oracle),
                "fast_valid_oracle_rmsd_lt2_pct": _pct([
                    bool(row["fast_valid_oracle_rmsd"]) and float(row["fast_valid_oracle_rmsd"]) < 2.0
                    for row in rows
                ]),
                "any_fast_valid_candidate_pct": _pct([count > 0 for count in counts]),
                "median_fast_valid_candidates": statistics.median(counts),
            })
    result = {
        "schema_version": "effdock.guidance_sigma_sweep_eta2_report.v1",
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "eta": 2.0,
        "sigmas": list(SIGMAS),
        "official_posebusters_validity": False,
        "validity_label": "internal_fast_valid",
        "arms": arms,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# EFF-Dock eta=2 prior-sigma sweep", "",
        "> Validity columns use the internal fast-valid proxy, not official PoseBusters.", "",
    ]
    for dataset in DATASETS:
        lines += [f"## {dataset}", "", "| sigma | Confidence <2A | Confidence median | Filter <2A | Oracle <2A | Valid oracle <2A | <2A & valid | Median valid candidates |", "|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for arm in [item for item in arms if item["dataset"] == dataset]:
            conf = arm["confidence"]
            filt = arm["confidence_filter"]
            lines.append(
                f"| {arm['sigma']:.1f} | {conf['rmsd_lt2_pct']:.1f}% | {conf['median_rmsd']:.2f} | "
                f"{filt['rmsd_lt2_pct']:.1f}% | {arm['oracle_rmsd_lt2_pct']:.1f}% | "
                f"{arm['fast_valid_oracle_rmsd_lt2_pct']:.1f}% | {conf['joint_rmsd_lt2_fast_valid_pct']:.1f}% | "
                f"{arm['median_fast_valid_candidates']:.1f} |"
            )
        lines.append("")
    args.output_md.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": "complete", "arms": len(arms)}, sort_keys=True))


if __name__ == "__main__":
    main()
