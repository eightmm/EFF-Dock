#!/usr/bin/env python3
"""Characterize candidate-count-invariant raw confidence/energy selectors."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from report_guidance_energy_confidence_selection import (
    EXPECTED_COUNTS,
    EXPECTED_POSES,
    _aggregate,
    _pose_outcome,
    _read_scores,
)
from report_guidance_sdf_post_refinement_full import _load_after, _load_before

PROTOCOL_ID = "EFFDOCK-GUIDANCE-ENERGY-CONFIDENCE-RAW-SELECTION-V1"
TOTAL_LAMBDAS = (0.001, 0.0025, 0.005, 0.01, 0.025, 0.05)
PER_ATOM_LAMBDAS = (0.05, 0.10, 0.25, 0.50, 1.00, 2.00)


def _lambda_tag(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def _raw_scores(
    confidence: list[float], energy: list[float], heavy_atoms: int
) -> dict[str, list[float]]:
    if len(confidence) != len(energy) or not confidence:
        raise ValueError("confidence and energy must be nonempty and aligned")
    if heavy_atoms <= 0:
        raise ValueError("heavy atom count must be positive")
    if not all(math.isfinite(value) for value in [*confidence, *energy]):
        raise ValueError("raw selector inputs must be finite")
    scores = {"confidence": list(confidence)}
    for coefficient in TOTAL_LAMBDAS:
        scores[f"raw_total_l{_lambda_tag(coefficient)}"] = [
            confidence[index] + coefficient * energy[index]
            for index in range(len(confidence))
        ]
    for coefficient in PER_ATOM_LAMBDAS:
        scores[f"raw_per_atom_l{_lambda_tag(coefficient)}"] = [
            confidence[index] + coefficient * energy[index] / heavy_atoms
            for index in range(len(confidence))
        ]
    return scores


def _raw_selector_indices(
    confidence: list[float], energy: list[float], heavy_atoms: int
) -> dict[str, int]:
    score_sets = _raw_scores(confidence, energy, heavy_atoms)
    return {
        name: min(range(len(values)), key=lambda index: (values[index], index))
        for name, values in score_sets.items()
    }


def _heavy_atom_count(path: Path) -> int:
    trajectory = torch.load(path, map_location="cpu", weights_only=True)
    frames = trajectory.get("frames_pocket_centered")
    if not isinstance(frames, torch.Tensor) or frames.ndim != 4:
        raise ValueError(f"{path}: invalid saved trajectory frames")
    if int(frames.shape[0]) != 5 or int(frames.shape[1]) != EXPECTED_POSES:
        raise ValueError(f"{path}: unexpected saved trajectory inventory")
    heavy_atoms = int(frames.shape[2])
    fragment_id = trajectory.get("fragment_id")
    if not isinstance(fragment_id, torch.Tensor) or fragment_id.numel() != heavy_atoms:
        raise ValueError(f"{path}: fragment IDs do not match heavy atoms")
    return heavy_atoms


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
    protocol_text = args.protocol_file.read_text(encoding="utf-8")
    if PROTOCOL_ID not in protocol_text:
        raise ValueError("protocol file does not declare the expected identity")
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
        refinement_dir = args.input_root / "refinement" / dataset / complex_id
        refinement = json.loads(
            (refinement_dir / "summary.json").read_text(encoding="utf-8")
        )
        if int(refinement["counts"]["failed"]) != 0 or len(refinement["poses"]) != 100:
            raise ValueError(f"{dataset}/{complex_id}: invalid refinement inventory")
        pose_indices = [int(row["pose_index"]) for row in refinement["poses"]]
        if pose_indices != list(range(EXPECTED_POSES)):
            raise ValueError(
                f"{dataset}/{complex_id}: refinement pose order is incomplete or changed"
            )
        heavy_atoms = _heavy_atom_count(refinement_dir / "trajectory.pt")
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
            selectors = _raw_selector_indices(
                values["confidence"], values["energy"], heavy_atoms
            )
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
                        "heavy_atoms": heavy_atoms,
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
        "schema_version": "effdock.guidance_energy_confidence_raw_selection.v1",
        "protocol_id": PROTOCOL_ID,
        "status": "complete_descriptive",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "cohort": EXPECTED_COUNTS,
        "poses_per_complex": EXPECTED_POSES,
        "total_lambdas": list(TOTAL_LAMBDAS),
        "per_atom_lambdas": list(PER_ATOM_LAMBDAS),
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
        "# Raw GuidanceEnergy + confidence selector characterization",
        "",
        "> Candidate-count-invariant post-hoc external-benchmark analysis. No row is selected for production.",
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
    (args.output_dir / "RESULTS.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
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
