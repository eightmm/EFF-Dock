#!/usr/bin/env python3
"""Aggregate the production-matched cutoff x center-jitter robustness study."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any

from effdock.workflows.evaluate import file_sha256

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
    parser.add_argument("--jitter0-root", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def truth(value: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"invalid boolean {value!r}")


def load_new_repeat(
    root: Path,
    dataset: str,
    denominator: int,
    cutoff: int,
    jitter: int,
    repeat: int,
    num_shards: int,
) -> dict[str, Any]:
    condition = root / f"jitter_{jitter:02d}" / f"cutoff_{cutoff:02d}" / f"repeat_{repeat}"
    rows: list[dict[str, str]] = []
    for shard in range(num_shards):
        directory = condition / "full" / "selected_posebusters" / f"shard-{shard:03d}-of-{num_shards:03d}"
        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        if (
            summary.get("status") != "complete"
            or summary.get("protocol_id") != "EFFDOCK-POCKET-CUTOFF-JITTER-ROBUSTNESS-V1"
            or int(summary.get("cutoff_angstrom", -1)) != cutoff
            or int(summary.get("center_jitter_sigma_angstrom_per_axis", -1)) != jitter
            or int(summary.get("repeat_index", -1)) != repeat
        ):
            raise ValueError(f"invalid PB shard contract {directory}")
        with (directory / "results.csv").open(newline="", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    subset = [row for row in rows if row["dataset"] == dataset]
    ids = [row["id"] for row in subset]
    if len(ids) != denominator or len(set(ids)) != denominator:
        raise ValueError(f"coverage mismatch c{cutoff}/j{jitter}/r{repeat}/{dataset}: {len(ids)}")
    error_count = sum(bool(row["error"]) for row in subset)
    return {
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


def aggregate(repeats: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        metric: {
            "values": [float(row[metric]["pct"]) for row in repeats],
            "mean": statistics.mean(float(row[metric]["pct"]) for row in repeats),
            "std": statistics.stdev(float(row[metric]["pct"]) for row in repeats),
            "std_definition": "sample standard deviation (ddof=1)",
        }
        for metric in METRICS
    }


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    baseline_path = args.jitter0_root / "report.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    if (
        baseline.get("status") != "complete"
        or baseline.get("protocol_id") != "EFFDOCK-POCKET-CUTOFF-ROBUSTNESS-V1"
        or int(baseline.get("total_selected_posebusters_errors", -1)) != 0
    ):
        raise ValueError("jitter=0 baseline is not a complete clean production run")

    datasets: dict[str, Any] = {}
    all_errors = 0
    for dataset, denominator in EXPECTED.items():
        baseline_cutoffs = {
            int(row["pocket_cutoff_angstrom"]): row
            for row in baseline["datasets"][dataset]["cutoffs"]
        }
        cutoff_rows: list[dict[str, Any]] = []
        for cutoff in (6, 8, 10, 12):
            jitter_rows: list[dict[str, Any]] = []
            for jitter in (0, 1, 2):
                if jitter == 0:
                    repeats = baseline_cutoffs[cutoff]["repeats"]
                else:
                    repeats = [
                        load_new_repeat(
                            args.output_root,
                            dataset,
                            denominator,
                            cutoff,
                            jitter,
                            repeat,
                            args.num_shards,
                        )
                        for repeat in range(3)
                    ]
                all_errors += sum(int(row["error_count"]) for row in repeats)
                jitter_rows.append(
                    {
                        "center_jitter_sigma_angstrom_per_axis": jitter,
                        "repeat_count": 3,
                        "repeats": repeats,
                        "aggregate": aggregate(repeats),
                    }
                )
            cutoff_rows.append(
                {"pocket_cutoff_angstrom": cutoff, "jitters": jitter_rows}
            )
        datasets[dataset] = {
            "name": "Astex Diverse" if dataset == "astex" else "PoseBusters v2",
            "n": denominator,
            "cutoffs": cutoff_rows,
        }

    payload = {
        "schema_version": 1,
        "status": "complete" if all_errors == 0 else "complete_with_denominator_failures",
        "protocol_id": "EFFDOCK-POCKET-CUTOFF-JITTER-ROBUSTNESS-V1",
        "jitter0_baseline": {
            "path": str(baseline_path),
            "sha256": file_sha256(baseline_path),
        },
        "center_jitter_definition": "Gaussian sigma in Angstrom per Cartesian axis",
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
