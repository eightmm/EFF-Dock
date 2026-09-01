#!/usr/bin/env python3
"""Aggregate paired U70k/U100k temporal-external benchmark results."""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from scripts.report_external_temporal_benchmark import (
    BOOL_FIELDS,
    DATASETS,
    FLOAT_FIELDS,
    parse_bool,
    summarize,
)
from scripts.run_s50_raw_refined_confidence_temporal_external_shard import PROTOCOL_ID

ARMS = {
    "u070000": "ce59be42f0ca613871ca079127c3296f5ca9a4ec72e44a9e5cf61878351c2638",
    "u100000": "2ea1aca4f1c326cd0841e76c3597e3749231854a523d1ba8bd923c6fb5a9bff8",
}
PAIRED_BOOLEAN_FIELDS = (
    "raw_selected_rmsd_lt2",
    "refined_selected_rmsd_lt2",
    "pl_valid",
    "posebusters_valid",
    "joint_pl_valid_rmsd_lt2",
    "joint_posebusters_valid_rmsd_lt2",
)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def paired_comparison(
    rows_by_arm: dict[str, list[dict[str, Any]]], dataset: str
) -> dict[str, Any]:
    by_id = {
        arm: {str(row["id"]): row for row in rows if str(row["dataset"]) == dataset}
        for arm, rows in rows_by_arm.items()
    }
    ids = sorted(by_id["u070000"])
    if set(ids) != set(by_id["u100000"]):
        raise ValueError(f"{dataset}: paired ID mismatch")
    metrics: dict[str, Any] = {}
    for field in PAIRED_BOOLEAN_FIELDS:
        gains = losses = unchanged_pass = unchanged_fail = 0
        for complex_id in ids:
            baseline = bool(by_id["u070000"][complex_id][field])
            candidate = bool(by_id["u100000"][complex_id][field])
            if candidate and not baseline:
                gains += 1
            elif baseline and not candidate:
                losses += 1
            elif baseline:
                unchanged_pass += 1
            else:
                unchanged_fail += 1
        metrics[field] = {
            "gains": gains,
            "losses": losses,
            "net": gains - losses,
            "unchanged_pass": unchanged_pass,
            "unchanged_fail": unchanged_fail,
        }
    refined_deltas = [
        float(by_id["u100000"][complex_id]["refined_selected_rmsd"])
        - float(by_id["u070000"][complex_id]["refined_selected_rmsd"])
        for complex_id in ids
    ]
    return {
        "dataset": dataset,
        "baseline": "u070000",
        "candidate": "u100000",
        "n": len(ids),
        "boolean_metrics": metrics,
        "mean_refined_selected_rmsd_delta": statistics.fmean(refined_deltas),
        "median_refined_selected_rmsd_delta": statistics.median(refined_deltas),
    }


def load_arm(input_root: Path, arm: str, checkpoint_sha256: str) -> tuple[list[dict[str, Any]], float]:
    all_rows: list[dict[str, Any]] = []
    runtime_seconds = 0.0
    schemas: set[tuple[str, ...]] = set()
    for dataset, (expected_count, num_shards) in DATASETS.items():
        dataset_rows: list[dict[str, Any]] = []
        for shard_index in range(num_shards):
            shard_name = f"{dataset}.shard-{shard_index:03d}-of-{num_shards:03d}"
            shard_dir = input_root / "full" / arm / "posebusters" / shard_name
            summary = read_json(shard_dir / "summary.json")
            if (
                summary.get("protocol_id") != PROTOCOL_ID
                or summary.get("status") != "complete"
                or summary.get("arm") != arm
                or summary.get("dataset") != dataset
                or int(summary.get("shard_index", -1)) != shard_index
                or int(summary.get("num_shards", -1)) != num_shards
                or summary.get("confidence_checkpoint_sha256") != checkpoint_sha256
            ):
                raise ValueError(f"invalid result shard: {shard_dir}")
            runtime_seconds += float(summary["runtime"]["elapsed_seconds"])
            with (shard_dir / "results.csv").open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                schemas.add(tuple(reader.fieldnames or ()))
                rows = list(reader)
            if len(rows) != int(summary["num_results"]):
                raise ValueError(f"row count mismatch: {shard_dir}")
            for row in rows:
                if row.get("arm") != arm:
                    raise ValueError(f"arm mismatch: {shard_dir}")
                for field in BOOL_FIELDS:
                    row[field] = parse_bool(str(row[field]))
                for field in FLOAT_FIELDS:
                    row[field] = float(row[field])
            dataset_rows.extend(rows)
        if len(dataset_rows) != expected_count:
            raise ValueError(
                f"{arm}/{dataset}: expected {expected_count} results, got {len(dataset_rows)}"
            )
        ids = [str(row["id"]) for row in dataset_rows]
        if len(ids) != len(set(ids)):
            raise ValueError(f"{arm}/{dataset}: duplicate result IDs")
        all_rows.extend(dataset_rows)
    if len(schemas) != 1:
        raise ValueError(f"{arm}: result CSV schemas differ across shards")
    return all_rows, runtime_seconds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    rows_by_arm: dict[str, list[dict[str, Any]]] = {}
    runtime_seconds = 0.0
    for arm, checkpoint_sha256 in ARMS.items():
        rows, arm_runtime = load_arm(args.input_root, arm, checkpoint_sha256)
        rows_by_arm[arm] = rows
        runtime_seconds += arm_runtime

    aggregates = {
        arm: {
            dataset: summarize(
                dataset, [row for row in rows if str(row["dataset"]) == dataset]
            )
            for dataset in DATASETS
        }
        for arm, rows in rows_by_arm.items()
    }
    comparisons = {
        dataset: paired_comparison(rows_by_arm, dataset) for dataset in DATASETS
    }
    report = {
        "schema_version": "effdock.s50_raw_refined_confidence_temporal_external_report.v1",
        "protocol_id": PROTOCOL_ID,
        "status": "complete_descriptive",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "arms": ARMS,
        "cohort_contract": {dataset: count for dataset, (count, _) in DATASETS.items()},
        "sampling": {
            "reused_immutable": True,
            "num_samples": 100,
            "num_steps": 10,
            "sigma": 2.0,
            "guidance_mode": "normalized_drift",
            "guidance_eta": 2.0,
        },
        "refinement": {"reused_immutable": True, "maximum_steps": 100},
        "claim_boundary": (
            "Repeated-use descriptive pocket-redocking adaptations. PhiBench and "
            "FoldBench are the core temporal checks; OpenBind is an auxiliary dense "
            "single-protease series. External outcomes do not select a checkpoint."
        ),
        "aggregates": aggregates,
        "comparisons_u100000_minus_u070000": comparisons,
        "aggregate_posebusters_cpu_seconds": runtime_seconds,
    }

    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    attempt = Path(tempfile.mkdtemp(prefix=f".{args.output_dir.name}.", dir=args.output_dir.parent))
    with (attempt / "per_complex.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(rows_by_arm["u070000"][0])
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        for arm in ARMS:
            writer.writerows(
                sorted(rows_by_arm[arm], key=lambda row: (str(row["dataset"]), str(row["id"])))
            )
    (attempt / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    lines = [
        "# U70k/U100k temporal external benchmark results",
        "",
        "| Dataset | Arm | N | Raw Top-1 <2A | Refined Top-1 <2A | PB-valid | Joint PB-valid + <2A |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for dataset in DATASETS:
        for arm in ARMS:
            result = aggregates[arm][dataset]
            values = result["percent"]
            lines.append(
                f"| {dataset} | {arm} | {result['n']} | "
                f"{values['raw_top1_rmsd_lt2']:.2f}% | "
                f"{values['refined_top1_rmsd_lt2']:.2f}% | "
                f"{values['refined_top1_posebusters_valid']:.2f}% | "
                f"{values['refined_top1_joint_posebusters_valid_rmsd_lt2']:.2f}% |"
            )
    lines.extend(("", report["claim_boundary"], ""))
    (attempt / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    os.rename(attempt, args.output_dir)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
