#!/usr/bin/env python3
"""Aggregate frozen external temporal benchmark shards."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import statistics
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROTOCOL_ID = "EFFDOCK-EXTERNAL-TEMPORAL-GUIDED-REFINED-V1"
DATASETS = {"phibench": (203, 13), "foldbench": (66, 5), "openbind": (860, 54)}
BOOL_FIELDS = {
    "raw_selected_rmsd_lt2",
    "refined_selected_rmsd_lt2",
    "pl_valid",
    "posebusters_valid",
    "joint_pl_valid_rmsd_lt2",
    "joint_posebusters_valid_rmsd_lt2",
}
FLOAT_FIELDS = {
    "raw_selected_rmsd",
    "refined_selected_rmsd",
    "raw_oracle_rmsd",
    "refined_oracle_rmsd",
    "mean_refinement_terminal_step",
}


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def parse_bool(value: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"invalid boolean CSV value: {value!r}")


def id_hash(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256(b"EFFDOCK_EXTERNAL_TEMPORAL_RESULT_IDS_V1\0")
    for row in sorted(rows, key=lambda item: (str(item["dataset"]), str(item["id"]))):
        digest.update(str(row["dataset"]).encode())
        digest.update(b"/")
        digest.update(str(row["id"]).encode())
        digest.update(b"\0")
    return digest.hexdigest()


def percent(count: int, total: int) -> float:
    return 100.0 * count / total


def summarize(dataset: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    metric_counts = {
        "raw_top1_rmsd_lt2": sum(bool(row["raw_selected_rmsd_lt2"]) for row in rows),
        "refined_top1_rmsd_lt2": sum(bool(row["refined_selected_rmsd_lt2"]) for row in rows),
        "raw_oracle_rmsd_lt2": sum(float(row["raw_oracle_rmsd"]) < 2.0 for row in rows),
        "refined_oracle_rmsd_lt2": sum(float(row["refined_oracle_rmsd"]) < 2.0 for row in rows),
        "refined_top1_pl_valid": sum(bool(row["pl_valid"]) for row in rows),
        "refined_top1_posebusters_valid": sum(bool(row["posebusters_valid"]) for row in rows),
        "refined_top1_joint_pl_valid_rmsd_lt2": sum(
            bool(row["joint_pl_valid_rmsd_lt2"]) for row in rows
        ),
        "refined_top1_joint_posebusters_valid_rmsd_lt2": sum(
            bool(row["joint_posebusters_valid_rmsd_lt2"]) for row in rows
        ),
    }
    return {
        "dataset": dataset,
        "n": count,
        "ids_sha256": id_hash(rows),
        "counts": metric_counts,
        "percent": {key: percent(value, count) for key, value in metric_counts.items()},
        "mean_raw_selected_rmsd": statistics.fmean(float(row["raw_selected_rmsd"]) for row in rows),
        "mean_refined_selected_rmsd": statistics.fmean(
            float(row["refined_selected_rmsd"]) for row in rows
        ),
        "median_refined_selected_rmsd": statistics.median(
            float(row["refined_selected_rmsd"]) for row in rows
        ),
        "mean_refinement_terminal_step": statistics.fmean(
            float(row["mean_refinement_terminal_step"]) for row in rows
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--selector-label",
        default="U25k symmetry-confidence stable argmin predicted RMSD",
    )
    parser.add_argument("--selector-checkpoint-sha256")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    all_rows: list[dict[str, Any]] = []
    runtime_seconds = 0.0
    schemas: set[tuple[str, ...]] = set()
    for dataset, (expected_count, num_shards) in DATASETS.items():
        dataset_rows: list[dict[str, Any]] = []
        for shard_index in range(num_shards):
            shard_name = f"{dataset}.shard-{shard_index:03d}-of-{num_shards:03d}"
            shard_dir = args.input_root / "full" / "posebusters" / shard_name
            summary = read_json(shard_dir / "summary.json")
            if (
                summary.get("protocol_id") != PROTOCOL_ID
                or summary.get("status") != "complete"
                or summary.get("dataset") != dataset
                or int(summary.get("shard_index", -1)) != shard_index
                or int(summary.get("num_shards", -1)) != num_shards
            ):
                raise ValueError(f"invalid result shard: {shard_dir}")
            if (
                args.selector_checkpoint_sha256 is not None
                and summary.get("confidence_checkpoint_sha256")
                != args.selector_checkpoint_sha256
            ):
                raise ValueError(f"selector checkpoint mismatch: {shard_dir}")
            runtime_seconds += float(summary["runtime"]["elapsed_seconds"])
            with (shard_dir / "results.csv").open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                schemas.add(tuple(reader.fieldnames or ()))
                rows = list(reader)
            if len(rows) != int(summary["num_results"]):
                raise ValueError(f"row count mismatch: {shard_dir}")
            for row in rows:
                for field in BOOL_FIELDS:
                    row[field] = parse_bool(str(row[field]))
                for field in FLOAT_FIELDS:
                    row[field] = float(row[field])
            dataset_rows.extend(rows)
        if len(dataset_rows) != expected_count:
            raise ValueError(
                f"{dataset}: expected {expected_count} results, got {len(dataset_rows)}"
            )
        ids = [str(row["id"]) for row in dataset_rows]
        if len(ids) != len(set(ids)):
            raise ValueError(f"{dataset}: duplicate result IDs")
        all_rows.extend(dataset_rows)
    if len(schemas) != 1:
        raise ValueError("result CSV schemas differ across shards")

    dataset_summaries = {
        dataset: summarize(dataset, [row for row in all_rows if str(row["dataset"]) == dataset])
        for dataset in DATASETS
    }
    report = {
        "schema_version": "effdock.external_temporal_report.v1",
        "protocol_id": PROTOCOL_ID,
        "status": "complete_descriptive",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "cohort_contract": {dataset: count for dataset, (count, _) in DATASETS.items()},
        "total_complexes_with_dataset_multiplicity": len(all_rows),
        "sampling": {
            "num_samples": 100,
            "num_steps": 10,
            "sigma": 2.0,
            "guidance_mode": "normalized_drift",
            "guidance_eta": 2.0,
        },
        "refinement": {
            "maximum_steps": 100,
            "adaptive_energy_plateau": "0.02 kcal/mol + 1e-3 * max(1, abs(E))",
            "patience": 5,
            "minimum_steps": 25,
        },
        "selector": args.selector_label,
        "selector_checkpoint_sha256": args.selector_checkpoint_sha256,
        "validity": {
            "pl_valid": "21 PoseBusters checks excluding cofactor and water contacts",
            "posebusters_valid": "all 27 non-RMSD PoseBusters redock checks",
        },
        "claim_boundary": (
            "descriptive EFF-Dock pocket-redocking adaptations; PhiBench cohort is "
            "EFF-Dock-derived, FoldBench is not its native leaderboard contract, and "
            "OpenBind is a dense enterovirus 2A-protease series"
        ),
        "datasets": dataset_summaries,
        "aggregate_posebusters_cpu_seconds": runtime_seconds,
    }

    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    attempt = Path(tempfile.mkdtemp(prefix=f".{args.output_dir.name}.", dir=args.output_dir.parent))
    with (attempt / "per_complex.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(next(iter(schemas)))
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        for row in sorted(all_rows, key=lambda item: (str(item["dataset"]), str(item["id"]))):
            writer.writerow(row)
    (attempt / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    lines = [
        "# EFF-Dock recent external benchmark results",
        "",
        "| Dataset | N | Raw Top-1 <2A | Refined Top-1 <2A | Refined oracle <2A | PL-valid | Joint PL-valid + <2A |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for dataset in DATASETS:
        result = dataset_summaries[dataset]
        values = result["percent"]
        lines.append(
            f"| {dataset} | {result['n']} | {values['raw_top1_rmsd_lt2']:.2f}% | "
            f"{values['refined_top1_rmsd_lt2']:.2f}% | "
            f"{values['refined_oracle_rmsd_lt2']:.2f}% | "
            f"{values['refined_top1_pl_valid']:.2f}% | "
            f"{values['refined_top1_joint_pl_valid_rmsd_lt2']:.2f}% |"
        )
    lines.extend(("", report["claim_boundary"] + ".", ""))
    (attempt / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    os.rename(attempt, args.output_dir)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
