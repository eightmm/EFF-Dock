#!/usr/bin/env python3
"""Aggregate RMSD, fast-valid, and cap telemetry across eta=0..3."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

REFERENCE_ETAS = (0.0, 0.5, 1.0, 1.5, 2.0)
NEW_ETAS = (2.5, 3.0)
ETAS = REFERENCE_ETAS + NEW_ETAS
DATASETS = {"astex": 85, "posebusters": 308}
SHARDS = 8


def _tag(eta: float) -> str:
    return f"eta{int(round(eta * 1000)):04d}"


def _rows(root: Path, reference: Path, dataset: str, eta: float) -> list[dict[str, str]]:
    if eta in REFERENCE_ETAS:
        base = reference / "raw"
        prefix = f"effdock-guidance-steric-high-eta-v1-{dataset}-n100-s10-{_tag(eta)}"
    else:
        base = root / "raw"
        prefix = f"effdock-guidance-eta-cap-extension-v1-{dataset}-n100-s10-{_tag(eta)}"
    result = []
    for shard in range(SHARDS):
        path = base / f"{prefix}.shard-{shard:03d}-of-{SHARDS:03d}.csv"
        with path.open(newline="", encoding="utf-8") as handle:
            result.extend(csv.DictReader(handle))
    if len(result) != DATASETS[dataset] or len({row["id"] for row in result}) != len(result):
        raise ValueError(f"{dataset}/{eta}: incomplete or duplicate inventory")
    return result


def _sum(rows: list[dict[str, str]], key: str) -> float:
    return sum(float(row[key]) for row in rows)


def _pct(numerator: float, denominator: float) -> float:
    return 100.0 * numerator / denominator if denominator else 0.0


def _truth(value: str) -> bool:
    return value.lower() == "true"


def _arm(rows: list[dict[str, str]], dataset: str, eta: float) -> dict:
    evals = _sum(rows, "guidance_direct_pose_evaluations")
    cap_count = _sum(rows, "guidance_direct_cap_scale_valid_count")
    ratio_count = _sum(rows, "guidance_direct_applied_to_model_rms_ratio_valid_count")
    cosine_count = _sum(rows, "guidance_direct_model_guide_cosine_valid_count")
    model_path = _sum(rows, "guidance_direct_model_rms_path_proxy_sum")
    confidence_rmsd = [float(row["confidence_rmsd"]) for row in rows]
    confidence_valid = [_truth(row["confidence_fast_valid"]) for row in rows]
    oracle = [float(row["oracle_rmsd"]) for row in rows]
    return {
        "dataset": dataset,
        "eta": eta,
        "source": "reference" if eta in REFERENCE_ETAS else "extension",
        "n": len(rows),
        "confidence_rmsd_lt2_pct": _pct(sum(value < 2 for value in confidence_rmsd), len(rows)),
        "confidence_median_rmsd": statistics.median(confidence_rmsd),
        "confidence_fast_valid_pct": _pct(sum(confidence_valid), len(rows)),
        "confidence_joint_lt2_fast_valid_pct": _pct(
            sum(value < 2 and valid for value, valid in zip(confidence_rmsd, confidence_valid)), len(rows)
        ),
        "oracle_rmsd_lt2_pct": _pct(sum(value < 2 for value in oracle), len(rows)),
        "any_cap_pct": _pct(_sum(rows, "guidance_direct_any_cap_trigger_count"), evals),
        "multiple_cap_pct": _pct(_sum(rows, "guidance_direct_multiple_cap_trigger_count"), evals),
        "translation_cap_pct": _pct(_sum(rows, "guidance_direct_translation_cap_trigger_count"), evals),
        "angular_cap_pct": _pct(_sum(rows, "guidance_direct_angular_cap_trigger_count"), evals),
        "displacement_cap_pct": _pct(_sum(rows, "guidance_direct_displacement_cap_trigger_count"), evals),
        "mean_cap_scale": _sum(rows, "guidance_direct_cap_scale_sum") / cap_count if cap_count else 0.0,
        "mean_applied_to_model_ratio": (
            _sum(rows, "guidance_direct_applied_to_model_rms_ratio_sum") / ratio_count if ratio_count else 0.0
        ),
        "applied_path_over_model_path": (
            _sum(rows, "guidance_direct_applied_rms_path_proxy_sum") / model_path if model_path else 0.0
        ),
        "mean_model_guide_cosine": (
            _sum(rows, "guidance_direct_model_guide_cosine_sum") / cosine_count if cosine_count else 0.0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()
    arms = [
        _arm(_rows(args.output_root, args.reference_root, dataset, eta), dataset, eta)
        for dataset in DATASETS
        for eta in ETAS
    ]
    result = {
        "schema_version": "effdock.guidance_eta_cap_extension_report.v1",
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "sigma": 0.5,
        "etas": list(ETAS),
        "official_posebusters_validity": False,
        "validity_label": "internal_fast_valid",
        "arms": arms,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = ["# EFF-Dock eta cap-saturation extension", "", "> Validity is internal fast-valid, not official PoseBusters.", ""]
    for dataset in DATASETS:
        lines += [
            f"## {dataset}", "",
            "| eta | Conf <2A | Conf valid | <2A & valid | Oracle <2A | Any cap | Multi cap | T cap | R cap | Disp cap | Mean cap scale | Applied/model | Path ratio | Cosine |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for arm in [value for value in arms if value["dataset"] == dataset]:
            lines.append(
                f"| {arm['eta']:.1f} | {arm['confidence_rmsd_lt2_pct']:.1f}% | "
                f"{arm['confidence_fast_valid_pct']:.1f}% | {arm['confidence_joint_lt2_fast_valid_pct']:.1f}% | "
                f"{arm['oracle_rmsd_lt2_pct']:.1f}% | {arm['any_cap_pct']:.1f}% | {arm['multiple_cap_pct']:.1f}% | "
                f"{arm['translation_cap_pct']:.1f}% | {arm['angular_cap_pct']:.1f}% | {arm['displacement_cap_pct']:.1f}% | "
                f"{arm['mean_cap_scale']:.3f} | {arm['mean_applied_to_model_ratio']:.3f} | "
                f"{arm['applied_path_over_model_path']:.3f} | {arm['mean_model_guide_cosine']:.3f} |"
            )
        lines.append("")
    args.output_md.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": "complete", "arms": len(arms)}, sort_keys=True))


if __name__ == "__main__":
    main()
