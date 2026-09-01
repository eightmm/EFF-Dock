#!/usr/bin/env python3
"""Aggregate the 4-cutoff by 3-repeat EFF-Dock robustness study."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

EXPECTED = {"astex": 85, "posebusters": 308}
METRICS = (
    "selected_rmsd_lt2",
    "posebusters_valid",
    "joint_rmsd_lt2_pb_valid",
    "oracle_rmsd_lt2",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def truth(value: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"invalid boolean {value!r}")


def main() -> None:
    args = parse_args()
    datasets: dict[str, object] = {}
    all_errors = 0
    for dataset, denominator in EXPECTED.items():
        cutoff_rows: list[dict[str, object]] = []
        for cutoff in (6, 8, 10, 12):
            repeats: list[dict[str, object]] = []
            for repeat in range(3):
                condition = args.output_root / f"cutoff_{cutoff:02d}" / f"repeat_{repeat}"
                rows: list[dict[str, str]] = []
                for shard in range(args.num_shards):
                    directory = condition / "full" / "selected_posebusters" / f"shard-{shard:03d}-of-{args.num_shards:03d}"
                    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
                    if summary.get("status") != "complete":
                        raise ValueError(f"incomplete PB shard {directory}")
                    with (directory / "results.csv").open(newline="", encoding="utf-8") as handle:
                        rows.extend(csv.DictReader(handle))
                ids = [row["id"] for row in rows if row["dataset"] == dataset]
                if len(ids) != denominator or len(set(ids)) != denominator:
                    raise ValueError(f"coverage mismatch c{cutoff}/r{repeat}/{dataset}: {len(ids)}")
                subset = [row for row in rows if row["dataset"] == dataset]
                error_count = sum(bool(row["error"]) for row in subset)
                all_errors += error_count
                repeats.append(
                    {
                        "repeat_index": repeat,
                        "denominator": denominator,
                        "error_count": error_count,
                        **{
                            metric: {
                                "count": sum(truth(row[metric]) for row in subset),
                                "pct": 100.0 * sum(truth(row[metric]) for row in subset) / denominator,
                            }
                            for metric in METRICS
                        },
                    }
                )
            aggregate = {}
            for metric in METRICS:
                values = [float(row[metric]["pct"]) for row in repeats]
                aggregate[metric] = {
                    "values": values,
                    "mean": statistics.mean(values),
                    "std": statistics.stdev(values),
                    "std_definition": "sample standard deviation (ddof=1)",
                }
            cutoff_rows.append(
                {
                    "pocket_cutoff_angstrom": cutoff,
                    "repeat_count": 3,
                    "repeats": repeats,
                    "aggregate": aggregate,
                }
            )
        datasets[dataset] = {
            "name": "Astex Diverse" if dataset == "astex" else "PoseBusters v2",
            "n": denominator,
            "cutoffs": cutoff_rows,
        }
    payload = {
        "schema_version": 1,
        "status": "complete" if all_errors == 0 else "complete_with_denominator_failures",
        "protocol_id": "EFFDOCK-POCKET-CUTOFF-ROBUSTNESS-V1",
        "changed_variable": "docking ODE receptor crop only",
        "fixed_refinement_crop_angstrom": 10,
        "fixed_confidence_crop_angstrom": 10,
        "official_posebusters_version": "0.6.5",
        "total_selected_posebusters_errors": all_errors,
        "datasets": datasets,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
