#!/usr/bin/env python3
"""Aggregate the factor-isolated cutoff-14 and prior-sigma extension."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any

from effdock.workflows.evaluate import file_sha256

PROTOCOL_ID = "EFFDOCK-POCKET-PRIOR-ROBUSTNESS-EXTENSION-V1"
EXPECTED = {"astex": 85, "posebusters": 308}
METRICS = (
    "selected_rmsd_lt2",
    "posebusters_valid",
    "joint_rmsd_lt2_pb_valid",
    "oracle_rmsd_lt2",
)
NEW_CONDITIONS = (
    *((14, 2, jitter) for jitter in (0, 1, 2)),
    *((10, sigma, jitter) for sigma in (1, 4) for jitter in (0, 1, 2)),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--baseline-report", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def truth(value: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"invalid boolean {value!r}")


def condition_root(root: Path, cutoff: int, sigma: int, jitter: int, repeat: int) -> Path:
    return (
        root
        / f"cutoff_{cutoff:02d}"
        / f"sigma_{sigma:02d}"
        / f"jitter_{jitter:02d}"
        / f"repeat_{repeat}"
    )


def load_new_repeat(
    root: Path,
    dataset: str,
    denominator: int,
    cutoff: int,
    sigma: int,
    jitter: int,
    repeat: int,
    num_shards: int,
) -> dict[str, Any]:
    condition = condition_root(root, cutoff, sigma, jitter, repeat)
    rows: list[dict[str, str]] = []
    for shard in range(num_shards):
        directory = (
            condition
            / "full"
            / "selected_posebusters"
            / f"shard-{shard:03d}-of-{num_shards:03d}"
        )
        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        expected_summary = {
            "status": "complete",
            "protocol_id": PROTOCOL_ID,
            "cutoff_angstrom": cutoff,
            "prior_sigma_angstrom": sigma,
            "center_jitter_sigma_angstrom_per_axis": jitter,
            "repeat_index": repeat,
            "num_shards": num_shards,
            "shard_index": shard,
        }
        changed = {
            key: (summary.get(key), value)
            for key, value in expected_summary.items()
            if summary.get(key) != value
        }
        if changed:
            raise ValueError(f"invalid selected-pose shard {directory}: {changed}")
        with (directory / "results.csv").open(newline="", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    subset = [row for row in rows if row["dataset"] == dataset]
    ids = [row["id"] for row in subset]
    if len(ids) != denominator or len(set(ids)) != denominator:
        raise ValueError(
            f"coverage mismatch c{cutoff}/s{sigma}/j{jitter}/r{repeat}/{dataset}: "
            f"{len(ids)}"
        )
    errors = [row for row in subset if row["error"]]
    if errors:
        raise ValueError(
            f"selected-pose errors c{cutoff}/s{sigma}/j{jitter}/r{repeat}/"
            f"{dataset}: {len(errors)}"
        )
    return {
        "repeat_index": repeat,
        "denominator": denominator,
        "error_count": 0,
        **{
            metric: {
                "count": sum(truth(row[metric]) for row in subset),
                "pct": 100.0 * sum(truth(row[metric]) for row in subset) / denominator,
            }
            for metric in METRICS
        },
    }


def aggregate(repeats: list[dict[str, Any]]) -> dict[str, Any]:
    if len(repeats) != 3:
        raise ValueError("every condition requires exactly three repeats")
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
    baseline = json.loads(args.baseline_report.read_text(encoding="utf-8"))
    if (
        baseline.get("status") != "complete"
        or baseline.get("protocol_id") != "EFFDOCK-POCKET-CUTOFF-JITTER-ROBUSTNESS-V1"
        or int(baseline.get("total_selected_posebusters_errors", -1)) != 0
    ):
        raise ValueError("baseline cutoff-by-jitter report is not complete and clean")

    loaded: dict[tuple[str, int, int, int], list[dict[str, Any]]] = {}
    for dataset, denominator in EXPECTED.items():
        for cutoff, sigma, jitter in NEW_CONDITIONS:
            loaded[(dataset, cutoff, sigma, jitter)] = [
                load_new_repeat(
                    args.output_root,
                    dataset,
                    denominator,
                    cutoff,
                    sigma,
                    jitter,
                    repeat,
                    args.num_shards,
                )
                for repeat in range(3)
            ]

    datasets: dict[str, Any] = {}
    for dataset, denominator in EXPECTED.items():
        baseline_cutoffs = {
            int(row["pocket_cutoff_angstrom"]): row
            for row in baseline["datasets"][dataset]["cutoffs"]
        }
        cutoff_sweep: list[dict[str, Any]] = []
        for cutoff in (6, 8, 10, 12, 14):
            jitter_rows: list[dict[str, Any]] = []
            for jitter in (0, 1, 2):
                if cutoff == 14:
                    repeats = loaded[(dataset, cutoff, 2, jitter)]
                else:
                    rows = {
                        int(row["center_jitter_sigma_angstrom_per_axis"]): row
                        for row in baseline_cutoffs[cutoff]["jitters"]
                    }
                    repeats = rows[jitter]["repeats"]
                jitter_rows.append(
                    {
                        "center_jitter_sigma_angstrom_per_axis": jitter,
                        "repeat_count": 3,
                        "repeats": repeats,
                        "aggregate": aggregate(repeats),
                    }
                )
            cutoff_sweep.append(
                {
                    "pocket_cutoff_angstrom": cutoff,
                    "prior_sigma_angstrom": 2,
                    "jitters": jitter_rows,
                }
            )

        sigma_sweep: list[dict[str, Any]] = []
        baseline_jitters = {
            int(row["center_jitter_sigma_angstrom_per_axis"]): row
            for row in baseline_cutoffs[10]["jitters"]
        }
        for sigma in (1, 2, 4):
            jitter_rows = []
            for jitter in (0, 1, 2):
                repeats = (
                    baseline_jitters[jitter]["repeats"]
                    if sigma == 2
                    else loaded[(dataset, 10, sigma, jitter)]
                )
                jitter_rows.append(
                    {
                        "center_jitter_sigma_angstrom_per_axis": jitter,
                        "repeat_count": 3,
                        "repeats": repeats,
                        "aggregate": aggregate(repeats),
                    }
                )
            sigma_sweep.append(
                {
                    "prior_sigma_angstrom": sigma,
                    "pocket_cutoff_angstrom": 10,
                    "jitters": jitter_rows,
                }
            )
        datasets[dataset] = {
            "name": "Astex Diverse" if dataset == "astex" else "PoseBusters v2",
            "n": denominator,
            "cutoff_sweep_fixed_sigma_2": cutoff_sweep,
            "prior_sigma_sweep_fixed_cutoff_10": sigma_sweep,
        }

    payload = {
        "schema_version": 1,
        "status": "complete",
        "protocol_id": PROTOCOL_ID,
        "baseline_report": {
            "path": str(args.baseline_report.resolve()),
            "sha256": file_sha256(args.baseline_report),
        },
        "new_condition_count": len(NEW_CONDITIONS),
        "new_repeat_condition_count": len(NEW_CONDITIONS) * 3,
        "center_jitter_definition": "Gaussian sigma in Angstrom per Cartesian axis",
        "fixed_refinement_crop_angstrom": 10,
        "fixed_confidence_crop_angstrom": 10,
        "official_posebusters_version": "0.6.5",
        "total_selected_posebusters_errors": 0,
        "datasets": datasets,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
