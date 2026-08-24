#!/usr/bin/env python3
"""Aggregate confidence-selected RMSD and PL-validity before/after refinement."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import statistics
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from effdock.workflows.guidance_pl_valid import PL_VALIDITY_CHECKS
from effdock.workflows.posebusters_report import VALIDITY_CHECKS

EXPECTED_COUNTS = {"astex": 85, "posebusters": 308}
CONFIDENCE_PROTOCOL_ID = "EFFDOCK-GUIDANCE-SDF-POST-REFINEMENT-CONFIDENCE-V2"
CONFIDENCE_BATCH_SIZE = 20
STAGES = ("before", "after_fixed", "after_reselected")
STAGE_LABELS = {
    "before": "Before: step-0 confidence Top-1",
    "after_fixed": "After: same pose index",
    "after_reselected": "After: step-100 confidence Top-1",
}
SELECTED_BOOL_METRICS = (
    "selected_rmsd_lt2",
    "selected_pl_valid",
    "selected_joint",
    "selected_official_pb_valid",
)


def _truth(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    raise ValueError(f"invalid boolean {value!r}")


def _stage_metrics(
    rmsds: list[float], checks: list[dict[str, bool]], selected_index: int
) -> dict[str, Any]:
    if len(rmsds) != 100 or len(checks) != 100 or not 0 <= selected_index < 100:
        raise ValueError("invalid per-complex metric inputs")
    pl = [all(row[key] for key in PL_VALIDITY_CHECKS) for row in checks]
    official = [all(row[key] for key in VALIDITY_CHECKS) for row in checks]
    success = [value < 2.0 for value in rmsds]
    return {
        "selected_index": selected_index,
        "selected_rmsd": rmsds[selected_index],
        "selected_rmsd_lt2": success[selected_index],
        "selected_pl_valid": pl[selected_index],
        "selected_joint": pl[selected_index] and success[selected_index],
        "selected_official_pb_valid": official[selected_index],
        "all_pose_pl_valid_count": sum(pl),
        "all_pose_official_pb_valid_count": sum(official),
        "all_pose_rmsd_lt2_count": sum(success),
        "rmsd_oracle_lt2": any(success),
        "pl_valid_oracle_lt2": any(p and s for p, s in zip(pl, success, strict=True)),
        "oracle_rmsd": min(rmsds),
    }


def _decomposition_row(
    *,
    dataset: str,
    metric: str,
    unit: str,
    before: float,
    after_fixed: float,
    after_reselected: float,
) -> dict[str, Any]:
    """Split the full change into same-index refinement and reselection increments."""
    return {
        "dataset": dataset,
        "metric": metric,
        "unit": unit,
        "before": before,
        "after_fixed": after_fixed,
        "after_reselected": after_reselected,
        "refinement_contribution": after_fixed - before,
        "reselection_contribution": after_reselected - after_fixed,
        "total_change": after_reselected - before,
    }


def _load_before(root: Path, num_shards: int) -> dict[tuple[str, str, int], dict[str, bool]]:
    rows: dict[tuple[str, str, int], dict[str, bool]] = {}
    for shard in range(num_shards):
        path = root / "full" / f"shard-{shard:03d}-of-{num_shards:03d}" / "poses.csv.gz"
        with gzip.open(path, "rt", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if float(row["eta"]) != 0.0:
                    continue
                key = (row["dataset"], row["id"], int(row["pose_index"]))
                if key in rows:
                    raise ValueError(f"duplicate step0 PB row {key}")
                rows[key] = {check: _truth(row[check]) for check in VALIDITY_CHECKS}
    if len(rows) != 39300:
        raise ValueError(f"expected 39,300 step0 PB rows, got {len(rows)}")
    return rows


def _load_after(root: Path, num_shards: int) -> dict[tuple[str, str, int], dict[str, bool]]:
    rows: dict[tuple[str, str, int], dict[str, bool]] = {}
    for shard in range(num_shards):
        directory = root / f"shard-{shard:03d}-of-{num_shards:03d}"
        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        if summary.get("status") != "complete" or int(summary.get("result_poses", -1)) <= 0:
            raise ValueError(f"incomplete step100 PB shard {shard}")
        with gzip.open(directory / "poses.csv.gz", "rt", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                key = (row["dataset"], row["id"], int(row["pose_index"]))
                if key in rows:
                    raise ValueError(f"duplicate step100 PB row {key}")
                rows[key] = {check: _truth(row[check]) for check in VALIDITY_CHECKS}
    if len(rows) != 39300:
        raise ValueError(f"expected 39,300 step100 PB rows, got {len(rows)}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--before-pb-root", type=Path, required=True)
    parser.add_argument("--before-pb-audit", type=Path, required=True)
    parser.add_argument("--after-pb-root", type=Path, required=True)
    parser.add_argument("--before-pb-shards", type=int, default=64)
    parser.add_argument("--after-pb-shards", type=int, default=32)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    audit = json.loads(args.before_pb_audit.read_text(encoding="utf-8"))
    if audit.get("status") != "passed" or int(audit.get("poses", -1)) != 275100:
        raise ValueError("frozen all-pose baseline PB audit did not pass")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    records = sorted(
        (row for row in manifest["records"] if float(row["eta"]) == 0.0),
        key=lambda row: (row["dataset"], row["id"]),
    )
    counts = Counter(row["dataset"] for row in records)
    if dict(counts) != EXPECTED_COUNTS:
        raise ValueError(f"unexpected cohort: {dict(counts)}")
    before_rows = _load_before(args.before_pb_root, args.before_pb_shards)
    after_rows = _load_after(args.after_pb_root, args.after_pb_shards)

    complex_rows: list[dict[str, Any]] = []
    for record in records:
        dataset, complex_id = record["dataset"], record["id"]
        refinement_path = args.input_root / "refinement" / dataset / complex_id / "summary.json"
        confidence_path = (
            args.input_root / "confidence_chunk20_fresh" / dataset / complex_id / "summary.json"
        )
        refinement = json.loads(refinement_path.read_text(encoding="utf-8"))
        confidence = json.loads(confidence_path.read_text(encoding="utf-8"))
        if int(refinement["counts"]["failed"]) != 0:
            raise ValueError(f"{dataset}/{complex_id}: unusable refinement poses")
        if confidence.get("protocol_id") != CONFIDENCE_PROTOCOL_ID:
            raise ValueError(f"{dataset}/{complex_id}: unexpected confidence protocol")
        if int(confidence.get("pose_batch_size", -1)) != CONFIDENCE_BATCH_SIZE:
            raise ValueError(f"{dataset}/{complex_id}: confidence batch is not frozen chunk 20")
        initial_rmsds = [float(row["initial_symmetry_rmsd_angstrom"]) for row in refinement["poses"]]
        final_rmsds = [float(row["final_symmetry_rmsd_angstrom"]) for row in refinement["poses"]]
        before_checks = [before_rows[(dataset, complex_id, index)] for index in range(100)]
        after_checks = [after_rows[(dataset, complex_id, index)] for index in range(100)]
        before_index = int(confidence["selected"]["step_000"]["pose_index"])
        after_index = int(confidence["selected"]["step_100"]["pose_index"])
        before = _stage_metrics(initial_rmsds, before_checks, before_index)
        after_fixed = _stage_metrics(final_rmsds, after_checks, before_index)
        after_reselected = _stage_metrics(final_rmsds, after_checks, after_index)
        complex_rows.append(
            {
                "dataset": dataset,
                "id": complex_id,
                **{f"before_{key}": value for key, value in before.items()},
                **{f"after_fixed_{key}": value for key, value in after_fixed.items()},
                **{
                    f"after_reselected_{key}": value
                    for key, value in after_reselected.items()
                },
                "selector_changed": before_index != after_index,
                "historical_selector_mismatch": not bool(
                    confidence["historical_baseline_reproduction"]["selected_index_matches"]
                ),
            }
        )

    aggregate: list[dict[str, Any]] = []
    paired: list[dict[str, Any]] = []
    decomposition: list[dict[str, Any]] = []
    historical_reproduction: list[dict[str, Any]] = []
    bool_metrics = (
        "selected_rmsd_lt2", "selected_pl_valid", "selected_joint",
        "selected_official_pb_valid", "rmsd_oracle_lt2", "pl_valid_oracle_lt2",
    )
    for dataset in EXPECTED_COUNTS:
        subset = [row for row in complex_rows if row["dataset"] == dataset]
        mismatch_count = sum(bool(row["historical_selector_mismatch"]) for row in subset)
        historical_reproduction.append(
            {
                "dataset": dataset,
                "complexes": len(subset),
                "selected_index_mismatches": mismatch_count,
                "selected_index_mismatch_pct": 100.0 * mismatch_count / len(subset),
                "role": "diagnostic_only_not_a_completion_gate",
            }
        )
        for stage in STAGES:
            aggregate.append(
                {
                    "dataset": dataset,
                    "stage": stage,
                    "complexes": len(subset),
                    **{
                        f"{metric}_pct": 100.0 * sum(bool(row[f"{stage}_{metric}"]) for row in subset) / len(subset)
                        for metric in bool_metrics
                    },
                    "selected_median_rmsd": statistics.median(
                        float(row[f"{stage}_selected_rmsd"]) for row in subset
                    ),
                    "all_pose_pl_valid_pct": 100.0 * sum(
                        int(row[f"{stage}_all_pose_pl_valid_count"]) for row in subset
                    ) / (100 * len(subset)),
                    "all_pose_official_pb_valid_pct": 100.0 * sum(
                        int(row[f"{stage}_all_pose_official_pb_valid_count"]) for row in subset
                    ) / (100 * len(subset)),
                    "selector_change_pct": (
                        100.0 * sum(bool(row["selector_changed"]) for row in subset)
                        / len(subset)
                        if stage == "after_reselected"
                        else 0.0
                    ),
                }
            )
        aggregate_by_stage = {
            row["stage"]: row
            for row in aggregate
            if row["dataset"] == dataset
        }
        for metric in SELECTED_BOOL_METRICS:
            field = f"{metric}_pct"
            decomposition.append(
                _decomposition_row(
                    dataset=dataset,
                    metric=metric,
                    unit="percentage_points",
                    before=float(aggregate_by_stage["before"][field]),
                    after_fixed=float(aggregate_by_stage["after_fixed"][field]),
                    after_reselected=float(
                        aggregate_by_stage["after_reselected"][field]
                    ),
                )
            )
        decomposition.append(
            _decomposition_row(
                dataset=dataset,
                metric="selected_median_rmsd",
                unit="angstrom",
                before=float(aggregate_by_stage["before"]["selected_median_rmsd"]),
                after_fixed=float(
                    aggregate_by_stage["after_fixed"]["selected_median_rmsd"]
                ),
                after_reselected=float(
                    aggregate_by_stage["after_reselected"]["selected_median_rmsd"]
                ),
            )
        )
        comparisons = (
            ("same_index_refinement", "before", "after_fixed"),
            ("reselection_increment", "after_fixed", "after_reselected"),
            ("full_pipeline", "before", "after_reselected"),
        )
        for comparison, start, stop in comparisons:
            for metric in SELECTED_BOOL_METRICS:
                paired.append(
                    {
                        "dataset": dataset,
                        "comparison": comparison,
                        "metric": metric,
                        "false_to_true": sum(
                            not bool(row[f"{start}_{metric}"])
                            and bool(row[f"{stop}_{metric}"])
                            for row in subset
                        ),
                        "true_to_false": sum(
                            bool(row[f"{start}_{metric}"])
                            and not bool(row[f"{stop}_{metric}"])
                            for row in subset
                        ),
                    }
                )
    result = {
        "schema_version": "effdock.guidance_sdf_post_refinement_full_report.v3",
        "protocol_id": CONFIDENCE_PROTOCOL_ID,
        "status": "complete_descriptive",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "selector": (
            "fresh chunk-20 argmin confidence_rmsd at both stages; validity and RMSD "
            "are outcomes only"
        ),
        "pl_validity_definition": list(PL_VALIDITY_CHECKS),
        "official_posebusters_validity_definition": list(VALIDITY_CHECKS),
        "aggregate": aggregate,
        "effect_decomposition": decomposition,
        "paired_transitions": paired,
        "historical_baseline_reproduction": historical_reproduction,
    }
    args.output_dir.mkdir(parents=True)
    with (args.output_dir / "complexes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(complex_rows[0]), extrasaction="raise")
        writer.writeheader()
        writer.writerows(complex_rows)
    (args.output_dir / "aggregate.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# Guidance SDF post-refinement effect decomposition", "",
        "> Post-hoc descriptive Astex/PoseBusters result. PL-valid excludes only cofactor/water checks. Same-index refinement follows the fresh step-0 confidence Top-1 through step 100; reselection then changes only the selected index.", "",
        "## Selected-pose results", "",
        "| Dataset | State | Top-1 <2Å | Top-1 PL-valid | Joint | Official PB-valid | Median RMSD | Selector changed |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate:
        lines.append(
            f"| {row['dataset']} | {STAGE_LABELS[row['stage']]} | "
            f"{row['selected_rmsd_lt2_pct']:.2f}% | "
            f"{row['selected_pl_valid_pct']:.2f}% | {row['selected_joint_pct']:.2f}% | "
            f"{row['selected_official_pb_valid_pct']:.2f}% | "
            f"{row['selected_median_rmsd']:.3f}Å | "
            f"{row['selector_change_pct']:.2f}% |"
        )
    lines.extend(
        [
            "",
            "## Effect decomposition",
            "",
            "> Percentage metrics are percentage-point changes. For median RMSD, a negative change is better.",
            "",
            "| Dataset | Metric | Same-index refinement | Reselection | Total |",
            "|---|---|---:|---:|---:|",
        ]
    )
    metric_labels = {
        "selected_rmsd_lt2": "Top-1 <2Å",
        "selected_pl_valid": "Top-1 PL-valid",
        "selected_joint": "Joint",
        "selected_official_pb_valid": "Official PB-valid",
        "selected_median_rmsd": "Median RMSD",
    }
    for row in decomposition:
        suffix = "Å" if row["unit"] == "angstrom" else " pp"
        lines.append(
            f"| {row['dataset']} | {metric_labels[row['metric']]} | "
            f"{row['refinement_contribution']:+.2f}{suffix} | "
            f"{row['reselection_contribution']:+.2f}{suffix} | "
            f"{row['total_change']:+.2f}{suffix} |"
        )
    lines.extend(
        [
            "",
            "## All-pose and oracle results",
            "",
            "| Dataset | Coordinates | All-pose PL-valid | Official PB-valid | RMSD oracle | PL-valid oracle |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in aggregate:
        if row["stage"] == "after_reselected":
            continue
        coordinate_label = "step 0" if row["stage"] == "before" else "step 100"
        lines.append(
            f"| {row['dataset']} | {coordinate_label} | "
            f"{row['all_pose_pl_valid_pct']:.2f}% | "
            f"{row['all_pose_official_pb_valid_pct']:.2f}% | "
            f"{row['rmsd_oracle_lt2_pct']:.2f}% | "
            f"{row['pl_valid_oracle_lt2_pct']:.2f}% |"
        )
    lines.extend(
        [
            "",
            "## Historical selector reproduction (diagnostic only)",
            "",
            "| Dataset | Fresh step-0 Top-1 differs from historical |",
            "|---|---:|",
        ]
    )
    for row in historical_reproduction:
        lines.append(
            f"| {row['dataset']} | {row['selected_index_mismatches']}/{row['complexes']} "
            f"({row['selected_index_mismatch_pct']:.2f}%) |"
        )
    (args.output_dir / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "complexes": len(complex_rows)}, sort_keys=True))


if __name__ == "__main__":
    main()
