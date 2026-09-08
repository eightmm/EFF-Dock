#!/usr/bin/env python3
"""Post-hoc U70k confidence/GuidanceEnergy ranking on Astex and PoseBusters."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from report_guidance_energy_confidence_selection import (
    _argmax_stable,
    _selector_indices,
    _stable_ordinal_quality,
)
from report_guidance_sdf_post_refinement_full import _load_after

from effdock.workflows.guidance_budget_posebusters_report import VALIDITY_CHECKS
from effdock.workflows.guidance_pl_valid import PL_VALIDITY_CHECKS

PROTOCOL_ID = "EFFDOCK-U70K-ENERGY-CONFIDENCE-EXTERNAL-V1"
EXPECTED_COUNTS = {"astex": 85, "posebusters": 308}
EXPECTED_POSES = 100
U70K_SHA256 = "ce59be42f0ca613871ca079127c3296f5ca9a4ec72e44a9e5cf61878351c2638"


def _clipped_energy_selectors(
    confidence_rmsd: list[float],
    confidence_success: list[float],
    energy: list[float],
) -> dict[str, int]:
    """Return fixed-weight selectors using energy clipped from -20..0 to 1..0."""
    q_confidence_rank, _ = _stable_ordinal_quality(confidence_rmsd)
    q_energy = [min(1.0, max(0.0, -value / 20.0)) for value in energy]
    selectors: dict[str, int] = {}
    for alpha in (0.05, 0.10, 0.25, 0.50):
        tag = f"a{int(round(100 * alpha)):02d}"
        selectors[f"clip20_rankconf_{tag}"] = _argmax_stable(
            [
                (1.0 - alpha) * q_confidence_rank[index]
                + alpha * q_energy[index]
                for index in range(EXPECTED_POSES)
            ]
        )
        selectors[f"clip20_success_{tag}"] = _argmax_stable(
            [
                (1.0 - alpha) * confidence_success[index]
                + alpha * q_energy[index]
                for index in range(EXPECTED_POSES)
            ]
        )
    return selectors


def _read_scores(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if [int(row["pose_index"]) for row in rows] != list(range(EXPECTED_POSES)):
        raise ValueError(f"{path}: expected ordered pose indices 0..99")
    return rows


def _selected_outcome(
    index: int,
    confidence: list[float],
    energy: list[float],
    rmsd: list[float],
    checks: list[dict[str, bool]],
) -> dict[str, Any]:
    pl_valid = all(checks[index][key] for key in PL_VALIDITY_CHECKS)
    official_valid = all(checks[index][key] for key in VALIDITY_CHECKS)
    success = rmsd[index] < 2.0
    return {
        "selected_index": index,
        "confidence_predicted_rmsd": confidence[index],
        "final_total_energy": energy[index],
        "symmetry_rmsd_angstrom": rmsd[index],
        "rmsd_lt2": success,
        "pl_valid": pl_valid,
        "official_valid": official_valid,
        "joint": success and official_valid,
    }


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for dataset in EXPECTED_COUNTS:
        dataset_rows = [row for row in rows if row["dataset"] == dataset]
        selectors = list(dict.fromkeys(row["selector"] for row in dataset_rows))
        baseline_by_id = {
            row["id"]: row for row in dataset_rows if row["selector"] == "confidence"
        }
        for selector in selectors:
            subset = [row for row in dataset_rows if row["selector"] == selector]
            if len(subset) != EXPECTED_COUNTS[dataset]:
                raise ValueError(f"{dataset}/{selector}: incomplete aggregate")
            paired = [(baseline_by_id[row["id"]], row) for row in subset]
            result.append(
                {
                    "dataset": dataset,
                    "selector": selector,
                    "complexes": len(subset),
                    "rmsd_lt2_count": sum(bool(row["rmsd_lt2"]) for row in subset),
                    "official_valid_count": sum(
                        bool(row["official_valid"]) for row in subset
                    ),
                    "joint_count": sum(bool(row["joint"]) for row in subset),
                    "pl_valid_count": sum(bool(row["pl_valid"]) for row in subset),
                    "median_rmsd": statistics.median(
                        float(row["symmetry_rmsd_angstrom"]) for row in subset
                    ),
                    "changed_count": sum(
                        row["selected_index"] != base["selected_index"]
                        for base, row in paired
                    ),
                    "rmsd_gain_count": sum(
                        not base["rmsd_lt2"] and row["rmsd_lt2"]
                        for base, row in paired
                    ),
                    "rmsd_loss_count": sum(
                        base["rmsd_lt2"] and not row["rmsd_lt2"]
                        for base, row in paired
                    ),
                    "joint_gain_count": sum(
                        not base["joint"] and row["joint"] for base, row in paired
                    ),
                    "joint_loss_count": sum(
                        base["joint"] and not row["joint"] for base, row in paired
                    ),
                }
            )
    return result


def _pct(count: int, total: int) -> float:
    return 100.0 * count / total


def _render_markdown(aggregate: list[dict[str, Any]]) -> str:
    lines = [
        "# U70k GuidanceEnergy + confidence external characterization",
        "",
        "> Repeated-use Astex/PoseBusters post-hoc descriptive analysis only; no selector is production-admitted.",
        "",
        "All rows use step-100 U70k predicted RMSD and the paired refinement `final_total_energy`.",
        "Signals are fused only after conversion to stable within-complex ordinal qualities.",
        "",
    ]
    for dataset, total in EXPECTED_COUNTS.items():
        lines.extend(
            [
                f"## {dataset}",
                "",
                "| Selector | RMSD <2A | Official valid | Joint | Median RMSD | Changed | RMSD gain/loss | Joint gain/loss |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in aggregate:
            if row["dataset"] != dataset:
                continue
            lines.append(
                f"| {row['selector']} | {row['rmsd_lt2_count']}/{total} "
                f"({_pct(row['rmsd_lt2_count'], total):.2f}%) | "
                f"{row['official_valid_count']}/{total} "
                f"({_pct(row['official_valid_count'], total):.2f}%) | "
                f"{row['joint_count']}/{total} ({_pct(row['joint_count'], total):.2f}%) | "
                f"{row['median_rmsd']:.3f}A | {row['changed_count']}/{total} | "
                f"{row['rmsd_gain_count']}/{row['rmsd_loss_count']} | "
                f"{row['joint_gain_count']}/{row['joint_loss_count']} |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores-root", type=Path, required=True)
    parser.add_argument("--refinement-root", type=Path, required=True)
    parser.add_argument("--after-pb-root", type=Path, required=True)
    parser.add_argument("--after-pb-shards", type=int, default=32)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")

    checks_by_pose = _load_after(args.after_pb_root, args.after_pb_shards)
    selected_rows: list[dict[str, Any]] = []
    observed = Counter()
    for dataset, expected in EXPECTED_COUNTS.items():
        dataset_root = args.scores_root / dataset
        complex_dirs = sorted(path for path in dataset_root.iterdir() if path.is_dir())
        if len(complex_dirs) != expected:
            raise ValueError(f"{dataset}: expected {expected} complexes, got {len(complex_dirs)}")
        for score_dir in complex_dirs:
            complex_id = score_dir.name
            score_summary = json.loads(
                (score_dir / "summary.json").read_text(encoding="utf-8")
            )
            if (
                score_summary.get("status") != "complete_descriptive"
                or score_summary.get("dataset") != dataset
                or score_summary.get("complex_id") != complex_id
                or score_summary.get("inputs", {}).get("confidence_checkpoint_sha256")
                != U70K_SHA256
                or int(score_summary.get("pose_count", -1)) != EXPECTED_POSES
            ):
                raise ValueError(f"invalid U70k score summary: {score_dir}")
            scores = _read_scores(score_dir / "scores.csv")
            refinement_path = (
                args.refinement_root / dataset / complex_id / "summary.json"
            )
            refinement = json.loads(refinement_path.read_text(encoding="utf-8"))
            poses = refinement["poses"]
            if (
                int(refinement["counts"]["failed"]) != 0
                or [int(row["pose_index"]) for row in poses]
                != list(range(EXPECTED_POSES))
            ):
                raise ValueError(f"invalid refinement inventory: {refinement_path}")

            confidence = [float(row["after_confidence_rmsd"]) for row in scores]
            confidence_success = [
                float(row["after_confidence_success"]) for row in scores
            ]
            energy = [float(row["final_total_energy"]) for row in poses]
            rmsd = [float(row["final_symmetry_rmsd_angstrom"]) for row in poses]
            checks = [
                checks_by_pose[(dataset, complex_id, index)]
                for index in range(EXPECTED_POSES)
            ]
            selectors = _selector_indices(confidence, energy)
            selectors.update(
                _clipped_energy_selectors(confidence, confidence_success, energy)
            )
            for selector, index in selectors.items():
                selected_rows.append(
                    {
                        "dataset": dataset,
                        "id": complex_id,
                        "selector": selector,
                        **_selected_outcome(index, confidence, energy, rmsd, checks),
                    }
                )
            observed[dataset] += 1

    if dict(observed) != EXPECTED_COUNTS:
        raise ValueError(f"unexpected completed cohort: {dict(observed)}")
    aggregate = _aggregate(selected_rows)
    result = {
        "schema_version": "effdock.u70k_energy_confidence_external.v1",
        "protocol_id": PROTOCOL_ID,
        "status": "complete_descriptive",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "claim_boundary": (
            "Repeated-use Astex/PoseBusters post-hoc characterization; external "
            "outcomes do not select or production-admit a selector."
        ),
        "checkpoint": {"name": "U70k", "sha256": U70K_SHA256},
        "cohort": EXPECTED_COUNTS,
        "poses_per_complex": EXPECTED_POSES,
        "aggregate": aggregate,
    }
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "selected_poses.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selected_rows[0]))
        writer.writeheader()
        writer.writerows(selected_rows)
    (args.output_dir / "RESULTS.md").write_text(
        _render_markdown(aggregate), encoding="utf-8"
    )
    print(json.dumps({"status": "complete", "output": str(args.output_dir)}))


if __name__ == "__main__":
    main()
