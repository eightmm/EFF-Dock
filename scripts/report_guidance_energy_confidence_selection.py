#!/usr/bin/env python3
"""Characterize fixed confidence/GuidanceEnergy pose selectors."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from report_guidance_sdf_post_refinement_full import _load_after, _load_before

from effdock.workflows.guidance_budget_posebusters_report import VALIDITY_CHECKS
from effdock.workflows.guidance_pl_valid import PL_VALIDITY_CHECKS

PROTOCOL_ID = "EFFDOCK-GUIDANCE-ENERGY-CONFIDENCE-SELECTION-V1"
EXPECTED_COUNTS = {"astex": 85, "posebusters": 308}
EXPECTED_POSES = 100
ALPHAS = (0.05, 0.10, 0.25, 0.50, 0.75)
FILTER_FRACTIONS = (0.25, 0.50, 0.75)


def _stable_ordinal_quality(values: list[float]) -> tuple[list[float], list[int]]:
    """Return lower-is-better ordinal quality and rank, stable by pose index."""
    if len(values) != EXPECTED_POSES or not all(math.isfinite(value) for value in values):
        raise ValueError("selector inputs must contain 100 finite values")
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    ranks = [0] * len(values)
    for rank, index in enumerate(order):
        ranks[index] = rank
    quality = [(len(values) - rank) / len(values) for rank in ranks]
    return quality, ranks


def _argmax_stable(values: list[float]) -> int:
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("selection scores must be finite and nonempty")
    return max(range(len(values)), key=lambda index: (values[index], -index))


def _selector_indices(confidence: list[float], energy: list[float]) -> dict[str, int]:
    q_confidence, confidence_ranks = _stable_ordinal_quality(confidence)
    q_energy, energy_ranks = _stable_ordinal_quality(energy)
    selectors = {
        "confidence": min(
            range(EXPECTED_POSES), key=lambda index: (confidence[index], index)
        ),
        "energy": min(range(EXPECTED_POSES), key=lambda index: (energy[index], index)),
    }
    for alpha in ALPHAS:
        tag = f"a{int(round(alpha * 100)):02d}"
        additive = [
            (1.0 - alpha) * q_confidence[index] + alpha * q_energy[index]
            for index in range(EXPECTED_POSES)
        ]
        geometric = [
            q_confidence[index] ** (1.0 - alpha) * q_energy[index] ** alpha
            for index in range(EXPECTED_POSES)
        ]
        selectors[f"rank_add_{tag}"] = _argmax_stable(additive)
        selectors[f"rank_geo_{tag}"] = _argmax_stable(geometric)
    for fraction in FILTER_FRACTIONS:
        keep = int(round(EXPECTED_POSES * fraction))
        eligible = [index for index in range(EXPECTED_POSES) if energy_ranks[index] < keep]
        tag = f"q{int(round(fraction * 100)):02d}"
        selectors[f"energy_filter_{tag}"] = min(
            eligible, key=lambda index: (confidence[index], confidence_ranks[index], index)
        )
    return selectors


def _read_scores(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    indices = [int(row["pose_index"]) for row in rows]
    if indices != list(range(EXPECTED_POSES)):
        raise ValueError(f"{path}: confidence pose order is incomplete or changed")
    return rows


def _pose_outcome(
    *,
    index: int,
    confidence: list[float],
    energy: list[float],
    rmsds: list[float],
    checks: list[dict[str, bool]],
) -> dict[str, Any]:
    pl_valid = all(checks[index][key] for key in PL_VALIDITY_CHECKS)
    official_valid = all(checks[index][key] for key in VALIDITY_CHECKS)
    success = rmsds[index] < 2.0
    return {
        "selected_index": index,
        "confidence_rmsd": confidence[index],
        "guidance_energy": energy[index],
        "rmsd": rmsds[index],
        "rmsd_lt2": success,
        "pl_valid": pl_valid,
        "joint": success and pl_valid,
        "official_pb_valid": official_valid,
    }


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    keys = sorted({(row["dataset"], row["stage"], row["selector"]) for row in rows})
    for dataset, stage, selector in keys:
        subset = [
            row
            for row in rows
            if (row["dataset"], row["stage"], row["selector"])
            == (dataset, stage, selector)
        ]
        expected = EXPECTED_COUNTS[dataset]
        if len(subset) != expected:
            raise ValueError(f"incomplete aggregate cell {dataset}/{stage}/{selector}")
        result.append(
            {
                "dataset": dataset,
                "stage": stage,
                "selector": selector,
                "complexes": len(subset),
                "rmsd_lt2_count": sum(bool(row["rmsd_lt2"]) for row in subset),
                "rmsd_lt2_pct": 100.0
                * sum(bool(row["rmsd_lt2"]) for row in subset)
                / len(subset),
                "pl_valid_count": sum(bool(row["pl_valid"]) for row in subset),
                "pl_valid_pct": 100.0
                * sum(bool(row["pl_valid"]) for row in subset)
                / len(subset),
                "joint_count": sum(bool(row["joint"]) for row in subset),
                "joint_pct": 100.0 * sum(bool(row["joint"]) for row in subset) / len(subset),
                "official_pb_valid_count": sum(
                    bool(row["official_pb_valid"]) for row in subset
                ),
                "official_pb_valid_pct": 100.0
                * sum(bool(row["official_pb_valid"]) for row in subset)
                / len(subset),
                "median_rmsd": statistics.median(float(row["rmsd"]) for row in subset),
                "changed_from_confidence_count": sum(
                    bool(row["changed_from_confidence"]) for row in subset
                ),
                "changed_from_confidence_pct": 100.0
                * sum(bool(row["changed_from_confidence"]) for row in subset)
                / len(subset),
            }
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--before-pb-root", type=Path, required=True)
    parser.add_argument("--before-pb-audit", type=Path, required=True)
    parser.add_argument("--after-pb-root", type=Path, required=True)
    parser.add_argument("--before-pb-shards", type=int, default=64)
    parser.add_argument("--after-pb-shards", type=int, default=32)
    parser.add_argument("--protocol-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    if not args.protocol_file.is_file():
        raise FileNotFoundError(args.protocol_file)
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

    selected_rows: list[dict[str, Any]] = []
    selector_names: list[str] | None = None
    for record in records:
        dataset, complex_id = str(record["dataset"]), str(record["id"])
        refinement = json.loads(
            (
                args.input_root / "refinement" / dataset / complex_id / "summary.json"
            ).read_text(encoding="utf-8")
        )
        if int(refinement["counts"]["failed"]) != 0 or len(refinement["poses"]) != 100:
            raise ValueError(f"{dataset}/{complex_id}: invalid refinement inventory")
        pose_indices = [int(row["pose_index"]) for row in refinement["poses"]]
        if pose_indices != list(range(EXPECTED_POSES)):
            raise ValueError(
                f"{dataset}/{complex_id}: refinement pose order is incomplete or changed"
            )
        score_rows = _read_scores(
            args.input_root
            / "confidence_chunk20_fresh"
            / dataset
            / complex_id
            / "scores.csv"
        )
        stage_inputs = {
            "step_000": {
                "confidence": [float(row["before_confidence_rmsd"]) for row in score_rows],
                "energy": [float(row["initial_total_energy"]) for row in refinement["poses"]],
                "rmsd": [
                    float(row["initial_symmetry_rmsd_angstrom"])
                    for row in refinement["poses"]
                ],
                "checks": [
                    before_rows[(dataset, complex_id, index)]
                    for index in range(EXPECTED_POSES)
                ],
            },
            "step_100": {
                "confidence": [float(row["after_confidence_rmsd"]) for row in score_rows],
                "energy": [float(row["final_total_energy"]) for row in refinement["poses"]],
                "rmsd": [
                    float(row["final_symmetry_rmsd_angstrom"])
                    for row in refinement["poses"]
                ],
                "checks": [
                    after_rows[(dataset, complex_id, index)]
                    for index in range(EXPECTED_POSES)
                ],
            },
        }
        for stage, values in stage_inputs.items():
            selectors = _selector_indices(values["confidence"], values["energy"])
            if selector_names is None:
                selector_names = list(selectors)
            elif list(selectors) != selector_names:
                raise AssertionError("selector inventory changed across complexes")
            baseline_index = selectors["confidence"]
            for selector, index in selectors.items():
                selected_rows.append(
                    {
                        "dataset": dataset,
                        "id": complex_id,
                        "stage": stage,
                        "selector": selector,
                        "changed_from_confidence": index != baseline_index,
                        **_pose_outcome(
                            index=index,
                            confidence=values["confidence"],
                            energy=values["energy"],
                            rmsds=values["rmsd"],
                            checks=values["checks"],
                        ),
                    }
                )

    aggregate = _aggregate(selected_rows)
    baseline = {
        (row["dataset"], row["stage"]): row
        for row in aggregate
        if row["selector"] == "confidence"
    }
    for row in aggregate:
        reference = baseline[(row["dataset"], row["stage"])]
        row["delta_rmsd_lt2_pp"] = row["rmsd_lt2_pct"] - reference["rmsd_lt2_pct"]
        row["delta_pl_valid_pp"] = row["pl_valid_pct"] - reference["pl_valid_pct"]
        row["delta_joint_pp"] = row["joint_pct"] - reference["joint_pct"]
        row["delta_median_rmsd"] = row["median_rmsd"] - reference["median_rmsd"]

    result = {
        "schema_version": "effdock.guidance_energy_confidence_selection.v1",
        "protocol_id": PROTOCOL_ID,
        "status": "complete_descriptive",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "cohort": EXPECTED_COUNTS,
        "poses_per_complex": EXPECTED_POSES,
        "alphas": list(ALPHAS),
        "energy_filter_fractions": list(FILTER_FRACTIONS),
        "selector_order": selector_names,
        "aggregate": aggregate,
    }
    args.output_dir.mkdir(parents=True)
    with (args.output_dir / "selected_poses.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selected_rows[0]))
        writer.writeheader()
        writer.writerows(selected_rows)
    with (args.output_dir / "aggregate.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregate[0]))
        writer.writeheader()
        writer.writerows(aggregate)
    (args.output_dir / "aggregate.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    lines = [
        "# GuidanceEnergy + confidence selector characterization",
        "",
        "> Post-hoc descriptive external-benchmark analysis. No row is selected for production.",
        "",
    ]
    for dataset in EXPECTED_COUNTS:
        for stage in ("step_000", "step_100"):
            lines.extend(
                [
                    f"## {dataset} / {stage}",
                    "",
                    "| Selector | <2A SR | PL-valid | Joint | Official PB-valid | Median RMSD | Changed vs confidence | Delta joint |",
                    "|---|---:|---:|---:|---:|---:|---:|---:|",
                ]
            )
            for row in aggregate:
                if row["dataset"] != dataset or row["stage"] != stage:
                    continue
                lines.append(
                    f"| {row['selector']} | {row['rmsd_lt2_pct']:.2f}% | "
                    f"{row['pl_valid_pct']:.2f}% | {row['joint_pct']:.2f}% | "
                    f"{row['official_pb_valid_pct']:.2f}% | {row['median_rmsd']:.3f}A | "
                    f"{row['changed_from_confidence_pct']:.2f}% | "
                    f"{row['delta_joint_pp']:+.2f} pp |"
                )
            lines.append("")
    (args.output_dir / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "complete",
                "complexes": len(records),
                "selectors": len(selector_names or []),
                "selected_rows": len(selected_rows),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
