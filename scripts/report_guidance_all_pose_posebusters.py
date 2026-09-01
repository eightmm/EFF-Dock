#!/usr/bin/env python3
"""Aggregate all-pose official PB validity and paired eta transitions."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import statistics
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

from effdock.workflows.guidance_budget_posebusters_report import VALIDITY_CHECKS

ETAS = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
DATASETS = ("astex", "posebusters")


def truth(value: str) -> bool:
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    raise ValueError(f"invalid boolean {value!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    if audit.get("status") != "passed" or audit.get("mode") != "full" or int(audit.get("poses", 0)) != 275100:
        raise ValueError("full audit did not pass exact 275,100-pose contract")
    totals: dict[tuple[str, float], int] = defaultdict(int)
    valid_counts: dict[tuple[str, float], int] = defaultdict(int)
    rmsd_counts: dict[tuple[str, float], int] = defaultdict(int)
    check_counts: dict[tuple[str, float, str], int] = defaultdict(int)
    validity: dict[tuple[str, float, str, int], bool] = {}
    macro_values: dict[tuple[str, float], list[float]] = defaultdict(list)
    for shard in range(args.num_shards):
        shard_dir = args.output_root / "full" / f"shard-{shard:03d}-of-{args.num_shards:03d}"
        with (shard_dir / "cells.csv").open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                macro_values[(row["dataset"], float(row["eta"]))].append(float(row["valid_pct"]))
        with gzip.open(shard_dir / "poses.csv.gz", "rt", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                dataset = row["dataset"]
                eta = float(row["eta"])
                complex_id = row["id"]
                pose_index = int(row["pose_index"])
                key = (dataset, eta)
                is_valid = truth(row["posebusters_valid"])
                totals[key] += 1
                valid_counts[key] += int(is_valid)
                rmsd_counts[key] += int(truth(row["separate_rmsd_check"]))
                validity[(dataset, eta, complex_id, pose_index)] = is_valid
                for check in VALIDITY_CHECKS:
                    check_counts[(dataset, eta, check)] += int(truth(row[check]))
    arms: list[dict[str, object]] = []
    per_check: list[dict[str, object]] = []
    transitions: list[dict[str, object]] = []
    for dataset in DATASETS:
        baseline_total = totals[(dataset, 0.0)]
        baseline_valid = valid_counts[(dataset, 0.0)]
        for eta in ETAS:
            key = (dataset, eta)
            total = totals[key]
            if total != baseline_total or total == 0:
                raise ValueError(f"{dataset}/{eta}: unpaired or empty pose inventory")
            pooled = 100.0 * valid_counts[key] / total
            macro = statistics.mean(macro_values[key])
            if abs(pooled - macro) > 1e-12:
                raise ValueError(f"{dataset}/{eta}: pooled and macro validity differ")
            arms.append(
                {
                    "dataset": dataset,
                    "eta": eta,
                    "poses": total,
                    "posebusters_valid_count": valid_counts[key],
                    "posebusters_valid_pct": pooled,
                    "macro_complex_valid_pct": macro,
                    "delta_vs_eta0_pp": pooled - 100.0 * baseline_valid / baseline_total,
                    "separate_rmsd_check_pct": 100.0 * rmsd_counts[key] / total,
                }
            )
            for check in VALIDITY_CHECKS:
                per_check.append(
                    {
                        "dataset": dataset,
                        "eta": eta,
                        "check": check,
                        "pass_count": check_counts[(dataset, eta, check)],
                        "pass_pct": 100.0 * check_counts[(dataset, eta, check)] / total,
                        "delta_vs_eta0_pp": 100.0
                        * (check_counts[(dataset, eta, check)] - check_counts[(dataset, 0.0, check)])
                        / total,
                    }
                )
            gain = loss = stay_valid = stay_invalid = 0
            for complex_id, pose_index in (
                (key_value[2], key_value[3])
                for key_value in validity
                if key_value[0] == dataset and key_value[1] == 0.0
            ):
                base = validity[(dataset, 0.0, complex_id, pose_index)]
                current_key = (dataset, eta, complex_id, pose_index)
                if current_key not in validity:
                    raise ValueError(f"missing paired pose {current_key}")
                current = validity[current_key]
                gain += int(not base and current)
                loss += int(base and not current)
                stay_valid += int(base and current)
                stay_invalid += int(not base and not current)
            transitions.append(
                {
                    "dataset": dataset,
                    "eta": eta,
                    "poses": total,
                    "invalid_to_valid": gain,
                    "valid_to_invalid": loss,
                    "valid_to_valid": stay_valid,
                    "invalid_to_invalid": stay_invalid,
                    "net_valid_poses": gain - loss,
                    "net_delta_pp": 100.0 * (gain - loss) / total,
                }
            )
    result = {
        "schema_version": "effdock.guidance_all_pose_pb_report.v1",
        "protocol_id": "EFFDOCK-GUIDANCE-ALL-POSE-PB-ETA-V1",
        "status": "complete",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "posebusters_version": "0.6.5",
        "validity_definition": "all 27 non-RMSD redock checks",
        "confidence_selection_used": False,
        "automatic_eta_selection": False,
        "arms": arms,
        "paired_transitions_vs_eta0": transitions,
        "per_check": per_check,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# All-pose official PoseBusters validity by eta",
        "",
        "> Every generated pose is counted; confidence selection and joint RMSD/validity are not used.",
        "",
    ]
    for dataset in DATASETS:
        lines += [
            f"## {dataset}",
            "",
            "| eta | poses | PB valid | delta vs 0 | invalid→valid | valid→invalid | net poses | separate RMSD check |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for arm, transition in zip(
            (row for row in arms if row["dataset"] == dataset),
            (row for row in transitions if row["dataset"] == dataset),
            strict=True,
        ):
            lines.append(
                f"| {float(arm['eta']):.1f} | {int(arm['poses']):,} | "
                f"{float(arm['posebusters_valid_pct']):.2f}% | "
                f"{float(arm['delta_vs_eta0_pp']):+.2f} pp | "
                f"{int(transition['invalid_to_valid']):,} | "
                f"{int(transition['valid_to_invalid']):,} | "
                f"{int(transition['net_valid_poses']):+,} | "
                f"{float(arm['separate_rmsd_check_pct']):.2f}% |"
            )
        lines.append("")
    args.output_md.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": "complete", "poses": sum(row["poses"] for row in arms)}, sort_keys=True))


if __name__ == "__main__":
    main()
