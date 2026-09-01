#!/usr/bin/env python3
"""Aggregate the paired S10/N100 and S25/N40 EFF-Dock comparison."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROTOCOL_ID = "EFFDOCK-FIXED-NFE-STEP-POSE-V1"
N40_MANIFEST_PROTOCOL_ID = "EFFDOCK-FIXED-NFE-STEP-POSE-REFINEMENT-INPUT-V1"
N40_CONFIDENCE_PROTOCOL_ID = "EFFDOCK-FIXED-NFE-STEP-POSE-U50-CONFIDENCE-V1"
N100_CONFIDENCE_PROTOCOL_ID = "EFFDOCK-GUIDANCE-SDF-POST-REFINEMENT-CONFIDENCE-V2"
N100_MANIFEST_SHA256 = "9e8be4d47dba8e346a6900b6bf02f5b853a93141f571f5c65b4c719de632d695"
BASELINE_REPORT_SHA256 = "501d2010a4df65fb0d9779e66113c7f3f423cd418d4ac683d903ec9b3fe1590a"
DOCKING_SHA256 = "65be44d7dc8f0867eb9fc5d22214b80f93971ea4702679a527c665046e91e6b6"
CONFIDENCE_SHA256 = "fd49fa86f67187bf26d6c1bcf2daf925ba3e3b19dfeae733e57535d183280469"
DATASETS = {"astex": 85, "posebusters": 308}
ARMS = {
    "s10_n100": {"steps": 10, "poses": 100, "label": "10 steps × 100 poses"},
    "s25_n40": {"steps": 25, "poses": 40, "label": "25 steps × 40 poses"},
}
STAGES = {
    "raw": ("initial_symmetry_rmsd_angstrom", "before_confidence_rmsd"),
    "refined": ("final_symmetry_rmsd_angstrom", "after_confidence_rmsd"),
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_manifest(path: Path, *, arm: str) -> dict[tuple[str, str], dict[str, Any]]:
    if arm == "s10_n100" and file_sha256(path) != N100_MANIFEST_SHA256:
        raise ValueError("reused N100/S10 manifest SHA-256 mismatch")
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected_protocol = (
        "EFFDOCK-GUIDANCE-SIGMA2-ETA2-REFINEMENT-INPUT-V1"
        if arm == "s10_n100"
        else N40_MANIFEST_PROTOCOL_ID
    )
    if payload.get("protocol_id") != expected_protocol:
        raise ValueError(f"{arm}: unexpected manifest protocol")
    if arm == "s25_n40" and payload.get("mode") != "full":
        raise ValueError("N40/S25 report requires the full manifest")
    expected_poses = ARMS[arm]["poses"]
    records: dict[tuple[str, str], dict[str, Any]] = {}
    for row in payload.get("records", []):
        key = (str(row["dataset"]), str(row["id"]).lower())
        if key in records or int(row.get("pose_count", -1)) != expected_poses:
            raise ValueError(f"{arm}: duplicate or invalid record {key}")
        records[key] = row
    if len(records) != sum(DATASETS.values()):
        raise ValueError(f"{arm}: expected 393 records, got {len(records)}")
    return records


def _load_scores(path: Path, *, arm: str, dataset: str, complex_id: str) -> list[dict[str, str]]:
    summary_path = path / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected_protocol = (
        N100_CONFIDENCE_PROTOCOL_ID if arm == "s10_n100" else N40_CONFIDENCE_PROTOCOL_ID
    )
    expected_poses = ARMS[arm]["poses"]
    inputs = summary.get("inputs", {})
    if (
        summary.get("protocol_id") != expected_protocol
        or summary.get("status") != "complete_descriptive"
        or summary.get("dataset") != dataset
        or summary.get("complex_id") != complex_id
        or int(summary.get("pose_count", -1)) != expected_poses
        or float(summary.get("sigma", -1)) != 2.0
        or int(summary.get("pose_batch_size", -1)) != 20
        or inputs.get("docking_checkpoint_sha256") != DOCKING_SHA256
        or inputs.get("confidence_checkpoint_sha256") != CONFIDENCE_SHA256
    ):
        raise ValueError(f"invalid confidence summary: {summary_path}")
    score_spec = summary.get("artifacts", {}).get("scores_csv", {})
    score_path = Path(str(score_spec.get("path", "")))
    if not score_path.is_file() or file_sha256(score_path) != score_spec.get("sha256"):
        raise ValueError(f"missing or changed score CSV: {score_path}")
    with score_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != expected_poses or [int(row["pose_index"]) for row in rows] != list(
        range(expected_poses)
    ):
        raise ValueError(f"invalid pose order/inventory: {score_path}")
    required = {
        "initial_symmetry_rmsd_angstrom",
        "final_symmetry_rmsd_angstrom",
        "before_confidence_rmsd",
        "after_confidence_rmsd",
    }
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"missing score columns: {score_path}")
    if not all(
        math.isfinite(float(row[column])) for row in rows for column in required
    ):
        raise ValueError(f"non-finite score/RMSD: {score_path}")
    return rows


def _aggregate_stage(
    complexes: list[list[dict[str, str]]], *, rmsd_field: str, confidence_field: str
) -> tuple[dict[str, Any], list[int]]:
    pose_count = len(complexes[0])
    successes = [0] * pose_count
    selected_rmsds: list[float] = []
    oracle_rmsds: list[float] = []
    for rows in complexes:
        if len(rows) != pose_count:
            raise ValueError("mixed pose counts inside one arm")
        rmsds = [float(row[rmsd_field]) for row in rows]
        scores = [float(row[confidence_field]) for row in rows]
        selected = min(range(pose_count), key=lambda index: (scores[index], index))
        selected_rmsds.append(rmsds[selected])
        oracle_rmsds.append(min(rmsds))
        running = math.inf
        for index, rmsd in enumerate(rmsds):
            running = min(running, rmsd)
            successes[index] += int(running < 2.0)
    count = len(complexes)
    metrics = {
        "complexes": count,
        "selected_lt2_count": sum(value < 2.0 for value in selected_rmsds),
        "selected_lt2_pct": 100.0 * sum(value < 2.0 for value in selected_rmsds) / count,
        "selected_median_rmsd_angstrom": statistics.median(selected_rmsds),
        "oracle_lt2_count": sum(value < 2.0 for value in oracle_rmsds),
        "oracle_lt2_pct": 100.0 * sum(value < 2.0 for value in oracle_rmsds) / count,
        "oracle_median_rmsd_angstrom": statistics.median(oracle_rmsds),
        "oracle_at_40_lt2_pct": 100.0 * successes[39] / count,
    }
    return metrics, successes


def _validate_baseline_report(path: Path, aggregate: dict[tuple[str, str, str], dict[str, Any]]) -> None:
    if file_sha256(path) != BASELINE_REPORT_SHA256:
        raise ValueError("reused U50 baseline report SHA-256 mismatch")
    report = json.loads(path.read_text(encoding="utf-8"))
    for dataset in DATASETS:
        for stage, report_stage in (("raw", "step_000"), ("refined", "step_100")):
            matches = [
                row
                for row in report.get("aggregate", [])
                if row.get("arm") == "u050000"
                and row.get("dataset") == dataset
                and row.get("stage") == report_stage
            ]
            if len(matches) != 1:
                raise ValueError(f"missing reused U50 aggregate: {dataset}/{report_stage}")
            expected = float(matches[0]["selected_lt2_pct"])
            actual = float(aggregate[(dataset, "s10_n100", stage)]["selected_lt2_pct"])
            if not math.isclose(actual, expected, abs_tol=1e-12):
                raise ValueError(
                    f"recomputed U50 baseline differs: {dataset}/{stage} {actual} != {expected}"
                )


def _plot(
    curves: list[dict[str, Any]], *, stage: str, destination: Path, pdf: Path
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "axes.labelsize": 11.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    colors = {"s10_n100": "#7F9BC8", "s25_n40": "#E6A07B"}
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6), sharey=True, constrained_layout=True)
    titles = {"astex": "A  Astex Diverse (N=85)", "posebusters": "B  PoseBusters v2 (N=308)"}
    for axis, dataset in zip(axes, DATASETS, strict=True):
        for arm in ARMS:
            points = [
                row
                for row in curves
                if row["dataset"] == dataset and row["stage"] == stage and row["arm"] == arm
            ]
            x = [int(row["k"]) for row in points]
            y = [float(row["sr_pct"]) for row in points]
            axis.step(
                x,
                y,
                where="post",
                linewidth=2.7,
                color=colors[arm],
                label=ARMS[arm]["label"],
            )
            mark_indices = [index for index, value in enumerate(x) if value in {1, 5, 10, 20, 40, 100}]
            axis.scatter(
                [x[index] for index in mark_indices],
                [y[index] for index in mark_indices],
                s=30,
                color=colors[arm],
                edgecolor="white",
                linewidth=0.8,
                zorder=3,
            )
            axis.annotate(
                f"{y[-1]:.1f}",
                (x[-1], y[-1]),
                xytext=(-4 if arm == "s10_n100" else 5, 7 if arm == "s10_n100" else -13),
                textcoords="offset points",
                ha="right" if arm == "s10_n100" else "left",
                color=colors[arm],
                fontsize=10,
                fontweight="bold",
            )
        axis.set_title(titles[dataset], loc="left", color="#26364A")
        axis.set_xlim(1, 100)
        axis.set_ylim(0, 100)
        axis.set_xticks([1, 10, 20, 40, 60, 80, 100])
        axis.set_yticks([0, 20, 40, 60, 80, 100])
        axis.grid(axis="y", color="#DCE4EC", linewidth=0.9)
        axis.set_axisbelow(True)
        axis.set_xlabel("Number of generated poses (k)")
        axis.tick_params(colors="#526273")
    axes[0].set_ylabel("Cumulative Oracle SR: min RMSD < 2 Å (%)")
    axes[1].legend(frameon=False, loc="lower right", fontsize=10)
    fig.savefig(destination, dpi=240, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n100-manifest", type=Path, required=True)
    parser.add_argument("--n40-manifest", type=Path, required=True)
    parser.add_argument("--n100-scores-root", type=Path, required=True)
    parser.add_argument("--n40-scores-root", type=Path, required=True)
    parser.add_argument("--baseline-report", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite report root: {args.output_root}")
    manifests = {
        "s10_n100": _load_manifest(args.n100_manifest, arm="s10_n100"),
        "s25_n40": _load_manifest(args.n40_manifest, arm="s25_n40"),
    }
    if set(manifests["s10_n100"]) != set(manifests["s25_n40"]):
        raise ValueError("paired arms do not contain identical complex IDs")
    for key, left in manifests["s10_n100"].items():
        right = manifests["s25_n40"][key]
        for field in (
            "sampling_seed",
            "prior_pool_sha256",
            "protein_sha256",
            "ligand_ref_sha256",
            "guidance_parameter_sha256",
        ):
            if left[field] != right[field]:
                raise ValueError(f"paired input mismatch for {key}: {field}")

    score_roots = {"s10_n100": args.n100_scores_root, "s25_n40": args.n40_scores_root}
    aggregate: dict[tuple[str, str, str], dict[str, Any]] = {}
    curves: list[dict[str, Any]] = []
    complex_rows: list[dict[str, Any]] = []
    for dataset, expected_count in DATASETS.items():
        ids = sorted(complex_id for row_dataset, complex_id in manifests["s10_n100"] if row_dataset == dataset)
        if len(ids) != expected_count:
            raise ValueError(f"{dataset}: expected {expected_count} IDs, got {len(ids)}")
        loaded: dict[str, list[list[dict[str, str]]]] = defaultdict(list)
        per_id: dict[tuple[str, str], list[dict[str, str]]] = {}
        for complex_id in ids:
            for arm in ARMS:
                rows = _load_scores(
                    score_roots[arm] / dataset / complex_id,
                    arm=arm,
                    dataset=dataset,
                    complex_id=complex_id,
                )
                loaded[arm].append(rows)
                per_id[(complex_id, arm)] = rows
        for arm in ARMS:
            for stage, (rmsd_field, confidence_field) in STAGES.items():
                metrics, success_counts = _aggregate_stage(
                    loaded[arm], rmsd_field=rmsd_field, confidence_field=confidence_field
                )
                aggregate[(dataset, arm, stage)] = metrics
                for k, success_count in enumerate(success_counts, start=1):
                    curves.append(
                        {
                            "dataset": dataset,
                            "stage": stage,
                            "arm": arm,
                            "k": k,
                            "success_count": success_count,
                            "complexes": expected_count,
                            "sr_pct": 100.0 * success_count / expected_count,
                            "learned_model_evaluations_per_complex": k * ARMS[arm]["steps"],
                        }
                    )
        for complex_id in ids:
            row: dict[str, Any] = {"dataset": dataset, "id": complex_id}
            for arm in ARMS:
                scores = per_id[(complex_id, arm)]
                for stage, (rmsd_field, confidence_field) in STAGES.items():
                    rmsds = [float(item[rmsd_field]) for item in scores]
                    confidences = [float(item[confidence_field]) for item in scores]
                    selected = min(
                        range(len(scores)), key=lambda index: (confidences[index], index)
                    )
                    row[f"{arm}_{stage}_selected_index"] = selected
                    row[f"{arm}_{stage}_selected_rmsd"] = rmsds[selected]
                    row[f"{arm}_{stage}_oracle_rmsd"] = min(rmsds)
            complex_rows.append(row)
    _validate_baseline_report(args.baseline_report, aggregate)

    destination = args.output_root.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    attempt = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    attempt.mkdir()
    curve_csv = attempt / "cumulative_oracle_sr.csv"
    with curve_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(curves[0]), extrasaction="raise")
        writer.writeheader()
        writer.writerows(curves)
    complex_csv = attempt / "complex_metrics.csv"
    with complex_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(complex_rows[0]), extrasaction="raise")
        writer.writeheader()
        writer.writerows(complex_rows)
    _plot(
        curves,
        stage="refined",
        destination=attempt / "cumulative_oracle_sr_refined.png",
        pdf=attempt / "cumulative_oracle_sr_refined.pdf",
    )
    _plot(
        curves,
        stage="raw",
        destination=attempt / "cumulative_oracle_sr_raw.png",
        pdf=attempt / "cumulative_oracle_sr_raw.pdf",
    )
    aggregate_rows = [
        {"dataset": dataset, "arm": arm, "stage": stage, **aggregate[(dataset, arm, stage)]}
        for dataset in DATASETS
        for arm in ARMS
        for stage in STAGES
    ]
    result = {
        "schema_version": "effdock.fixed_nfe_step_pose_report.v1",
        "protocol_id": PROTOCOL_ID,
        "status": "complete_descriptive",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "claim_boundary": "Repeated-use paired Astex/PoseBusters allocation study; single frozen seed; RMSD only.",
        "arms": ARMS,
        "pairing": "identical complex, sampling seed, and exact 100-pose prior-pool hash",
        "metric": "cumulative symmetry-aware no-alignment RMSD <2 A oracle success in sampling order",
        "aggregate": aggregate_rows,
        "artifacts": {},
    }
    for name in (
        "cumulative_oracle_sr.csv",
        "complex_metrics.csv",
        "cumulative_oracle_sr_refined.png",
        "cumulative_oracle_sr_refined.pdf",
        "cumulative_oracle_sr_raw.png",
        "cumulative_oracle_sr_raw.pdf",
    ):
        result["artifacts"][name] = {"path": str(destination / name), "sha256": file_sha256(attempt / name)}
    (attempt / "report.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# Fixed-NFE step/pose allocation results",
        "",
        "> Repeated-use paired descriptive Astex/PoseBusters result; one frozen seed; RMSD endpoint only.",
        "",
        "| Dataset | Arm | Stage | U50 Top-1 <2A | Oracle@40 <2A | Final oracle <2A | Median selected RMSD |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for dataset in DATASETS:
        for arm in ARMS:
            for stage in STAGES:
                row = aggregate[(dataset, arm, stage)]
                lines.append(
                    f"| {dataset} | {ARMS[arm]['label']} | {stage} | "
                    f"{row['selected_lt2_pct']:.2f}% | {row['oracle_at_40_lt2_pct']:.2f}% | "
                    f"{row['oracle_lt2_pct']:.2f}% | {row['selected_median_rmsd_angstrom']:.3f} A |"
                )
    lines.extend(
        [
            "",
            "`Final oracle` is Oracle@100 for S10/N100 and Oracle@40 for S25/N40. "
            "The cumulative curve uses original sampling order, not confidence order.",
            "",
        ]
    )
    (attempt / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    os.rename(attempt, destination)
    print(json.dumps({"status": "complete", "output_root": str(destination)}, sort_keys=True))


if __name__ == "__main__":
    main()
