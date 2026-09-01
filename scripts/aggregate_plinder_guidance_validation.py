#!/usr/bin/env python3
"""Strict all-arm PLINDER RMSD/PoseBusters report without automatic eta selection."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from plinder_guidance_audit import build_audit, inspect_incomplete_attempts
from plinder_guidance_common import (
    ETA_TAGS,
    ETA_VALUES,
    EXPECTED_COUNT,
    EXPECTED_POSEBUSTERS_VERSION,
    FULL_SHARDS,
    POSEBUSTERS_SCHEMA,
    PRIMARY_SELECTOR,
    PROTOCOL_ID,
    REPORT_SCHEMA,
    canonical_json_sha256,
    expected_ids,
    file_sha256,
    ids_sha256,
    load_csv,
    load_split_ids,
    parse_bool,
    posebusters_shard_dir,
    sampling_shard_dir,
    validate_passed_audit,
    write_json_noreplace,
)

from effdock.workflows.guidance_budget_posebusters_report import VALIDITY_CHECKS

BOOTSTRAP_SEED = 42
BOOTSTRAP_RESAMPLES = 2_000


def paired_bootstrap_delta(
    baseline: np.ndarray,
    arm: np.ndarray,
    *,
    seed: int,
    statistic: str,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> dict[str, float | int | str]:
    if baseline.shape != arm.shape or baseline.ndim != 1 or baseline.size != EXPECTED_COUNT:
        raise ValueError(f"paired bootstrap requires exact {EXPECTED_COUNT}-complex arrays")
    if resamples < 1:
        raise ValueError("bootstrap resamples must be positive")
    if not np.isfinite(baseline).all() or not np.isfinite(arm).all():
        raise ValueError("paired bootstrap inputs must be finite")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, baseline.size, size=(resamples, baseline.size))
    if statistic == "mean_percentage_points":
        observed = float(np.mean(arm - baseline) * 100.0)
        draws = np.mean(arm[indices] - baseline[indices], axis=1) * 100.0
        units = "percentage_points"
    elif statistic == "median_difference_angstrom":
        observed = float(np.median(arm) - np.median(baseline))
        draws = np.median(arm[indices], axis=1) - np.median(baseline[indices], axis=1)
        units = "angstrom"
    else:
        raise ValueError(f"unknown paired bootstrap statistic: {statistic}")
    low, high = np.percentile(draws, [2.5, 97.5])
    return {
        "statistic": statistic,
        "units": units,
        "delta": observed,
        "ci95_low": float(low),
        "ci95_high": float(high),
        "seed": seed,
        "resamples": resamples,
    }


def _parse_official_csv(
    path: Path, *, expected_ids_for_shard: list[str], expected_rmsd_check: str | None
) -> tuple[list[dict[str, Any]], str]:
    fields, raw_rows = load_csv(path)
    rmsd_checks = [field for field in fields if field.startswith("rmsd_")]
    if len(rmsd_checks) != 1:
        raise ValueError(f"{path}: expected exactly one separate RMSD check")
    rmsd_check = rmsd_checks[0]
    if expected_rmsd_check is not None and rmsd_check != expected_rmsd_check:
        raise ValueError(f"{path}: PoseBusters RMSD check changed across shards")
    expected_fields = {"id", "posebusters_valid", rmsd_check, *VALIDITY_CHECKS}
    if set(fields) != expected_fields:
        raise ValueError(f"{path}: official 27-check redock schema mismatch")
    if [row.get("id") for row in raw_rows] != expected_ids_for_shard:
        raise ValueError(f"{path}: exact PoseBusters ID/order mismatch")
    parsed: list[dict[str, Any]] = []
    for row in raw_rows:
        checks = {
            check: parse_bool(row.get(check), label=f"{row.get('id')}.{check}")
            for check in VALIDITY_CHECKS
        }
        valid = parse_bool(row.get("posebusters_valid"), label="posebusters_valid")
        if valid != all(checks.values()):
            raise ValueError(f"{row.get('id')}: pass-all differs from the 27 checks")
        parsed.append(
            {
                "id": row["id"],
                "posebusters_valid": valid,
                "checks": checks,
                "separate_rmsd_check": parse_bool(
                    row.get(rmsd_check), label=f"{row.get('id')}.{rmsd_check}"
                ),
            }
        )
    return parsed, rmsd_check


def _validate_pb_summary(
    path: Path,
    *,
    eta: float,
    shard_index: int,
    assigned_ids: list[str],
    audit_path: Path,
    sampling_csv: Path,
    sampling_summary: Path,
    official_csv: Path,
) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": POSEBUSTERS_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "status": "complete",
        "mode": "full",
        "eta": eta,
        "eta_tag": ETA_TAGS[eta],
        "primary_selector": PRIMARY_SELECTOR,
        "posebusters_version": EXPECTED_POSEBUSTERS_VERSION,
        "config": "redock",
        "pass_all_definition": "all 27 non-RMSD redock checks",
        "validity_checks": list(VALIDITY_CHECKS),
        "expected_denominator": EXPECTED_COUNT,
        "selected_cohort_count": EXPECTED_COUNT,
        "num_shards": FULL_SHARDS,
        "shard_index": shard_index,
        "assigned_count": len(assigned_ids),
        "assigned_ids": assigned_ids,
        "assigned_ids_sha256": ids_sha256(assigned_ids),
        "success_count": len(assigned_ids),
        "failure_count": 0,
        "failures": [],
        "sampling_audit": str(audit_path.resolve()),
        "sampling_audit_sha256": file_sha256(audit_path),
        "sampling_csv": str(sampling_csv.resolve()),
        "sampling_csv_sha256": file_sha256(sampling_csv),
        "sampling_summary": str(sampling_summary.resolve()),
        "sampling_summary_sha256": file_sha256(sampling_summary),
    }
    for key, value in expected.items():
        if raw.get(key) != value:
            raise ValueError(f"{path}: PoseBusters summary {key} mismatch")
    artifacts = raw.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError(f"{path}: PoseBusters artifacts missing")
    if (
        artifacts.get("csv") != str(official_csv.resolve())
        or artifacts.get("csv_sha256") != file_sha256(official_csv)
        or artifacts.get("summary") != str(path.resolve())
    ):
        raise ValueError(f"{path}: PoseBusters artifact binding mismatch")
    return raw


def _numeric(row: dict[str, str], key: str) -> float:
    value = row.get(key)
    if value in {None, ""}:
        return 0.0
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{row.get('id')}: telemetry {key} is non-finite")
    return result


def _telemetry(rows: list[dict[str, str]], summaries: list[dict[str, Any]]) -> dict[str, Any]:
    pose_evaluations = sum(_numeric(row, "guidance_direct_pose_evaluations") for row in rows)
    ratio_count = sum(
        _numeric(row, "guidance_direct_applied_to_model_rms_ratio_valid_count")
        for row in rows
    )
    ratio_sum = sum(
        _numeric(row, "guidance_direct_applied_to_model_rms_ratio_sum") for row in rows
    )
    cap_count = sum(_numeric(row, "guidance_direct_any_cap_trigger_count") for row in rows)
    multiple_cap_count = sum(
        _numeric(row, "guidance_direct_multiple_cap_trigger_count") for row in rows
    )
    nonfinite = {
        key.removeprefix("guidance_"): int(sum(_numeric(row, key) for row in rows))
        for key in (
            "guidance_nonfinite_base_poses",
            "guidance_nonfinite_trials",
            "guidance_direct_nonfinite_poses",
        )
    }
    cuda_allocated = [
        int(summary["runtime"]["cuda_max_memory_allocated_bytes"]) for summary in summaries
    ]
    cuda_reserved = [
        int(summary["runtime"]["cuda_max_memory_reserved_bytes"]) for summary in summaries
    ]
    return {
        "direct_pose_evaluations": int(pose_evaluations),
        "direct_pose_applied": int(
            sum(_numeric(row, "guidance_direct_pose_applied") for row in rows)
        ),
        "nonfinite_counters": nonfinite,
        "any_cap_trigger_count": int(cap_count),
        "any_cap_trigger_pct": (
            cap_count / pose_evaluations * 100.0 if pose_evaluations else 0.0
        ),
        "multiple_cap_trigger_count": int(multiple_cap_count),
        "multiple_cap_trigger_pct": (
            multiple_cap_count / pose_evaluations * 100.0 if pose_evaluations else 0.0
        ),
        "mean_applied_to_model_rms_ratio": ratio_sum / ratio_count if ratio_count else None,
        "applied_to_model_ratio_valid_count": int(ratio_count),
        "cuda_peak_allocated_bytes": max(cuda_allocated),
        "cuda_peak_reserved_bytes": max(cuda_reserved),
    }


def build_report(
    *,
    sampling_root: Path,
    posebusters_root: Path,
    audit_path: Path,
    split_file: Path,
    raw_manifest: Path,
    raw_gate: Path,
    raw_gate_sidecar: Path,
    raw_root: Path,
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES,
) -> dict[str, Any]:
    sampling_root = sampling_root.resolve()
    posebusters_root = posebusters_root.resolve()
    saved_audit = validate_passed_audit(
        audit_path, mode="full", sampling_root=sampling_root
    )
    rebuilt_audit = build_audit(
        sampling_root=sampling_root,
        split_file=split_file,
        raw_manifest=raw_manifest,
        raw_gate=raw_gate,
        raw_gate_sidecar=raw_gate_sidecar,
        raw_root=raw_root,
        mode="full",
    )
    if rebuilt_audit != saved_audit:
        raise ValueError("saved full audit differs from fresh sampling revalidation")
    split_ids = load_split_ids(split_file)
    selected_ids = expected_ids("full", split_ids)
    expected_eta_dirs = {
        posebusters_root / "full" / ETA_TAGS[eta] for eta in ETA_VALUES
    }
    full_pb_root = posebusters_root / "full"
    actual_eta_dirs = (
        {path for path in full_pb_root.iterdir() if path.is_dir()}
        if full_pb_root.is_dir()
        else set()
    )
    if actual_eta_dirs != expected_eta_dirs:
        raise ValueError("official PoseBusters eta inventory mismatch")

    sampling_by_eta: dict[float, dict[str, dict[str, str]]] = {}
    official_by_eta: dict[float, dict[str, dict[str, Any]]] = {}
    telemetry_by_eta: dict[float, dict[str, Any]] = {}
    official_ledger: list[dict[str, Any]] = []
    posebusters_recovered_attempts: list[dict[str, Any]] = []
    global_rmsd_check: str | None = None
    for eta in ETA_VALUES:
        expected_shard_dirs = {
            posebusters_shard_dir(posebusters_root, "full", eta, shard, FULL_SHARDS)
            for shard in range(FULL_SHARDS)
        }
        arm_root = posebusters_root / "full" / ETA_TAGS[eta]
        eta_recovered = inspect_incomplete_attempts(
            arm_root / ".incomplete", num_shards=FULL_SHARDS
        )
        posebusters_recovered_attempts.extend(
            {"eta": eta, "eta_tag": ETA_TAGS[eta], **record}
            for record in eta_recovered
        )
        actual_shard_dirs = {
            path
            for path in arm_root.iterdir()
            if path.is_dir() and path.name != ".incomplete"
        }
        if actual_shard_dirs != expected_shard_dirs:
            raise ValueError(f"{ETA_TAGS[eta]}: official PoseBusters shard inventory mismatch")
        sampling_rows_by_id: dict[str, dict[str, str]] = {}
        official_rows_by_id: dict[str, dict[str, Any]] = {}
        sampling_summaries: list[dict[str, Any]] = []
        for shard_index in range(FULL_SHARDS):
            assigned_ids = selected_ids[shard_index::FULL_SHARDS]
            sampling_dir = sampling_shard_dir(
                sampling_root, "full", eta, shard_index, FULL_SHARDS
            )
            sampling_csv = sampling_dir / "results.csv"
            sampling_summary = sampling_dir / "summary.json"
            _, sampling_rows = load_csv(sampling_csv)
            sampling_summaries.append(json.loads(sampling_summary.read_text(encoding="utf-8")))
            if [row.get("id") for row in sampling_rows] != assigned_ids:
                raise ValueError(f"{sampling_csv}: sampling ID inventory changed after audit")
            output_dir = posebusters_shard_dir(
                posebusters_root, "full", eta, shard_index, FULL_SHARDS
            )
            if set(output_dir.iterdir()) != {
                output_dir / "results.csv",
                output_dir / "summary.json",
            }:
                raise ValueError(f"{output_dir}: unexpected official artifacts")
            official_csv = output_dir / "results.csv"
            official_summary = output_dir / "summary.json"
            summary = _validate_pb_summary(
                official_summary,
                eta=eta,
                shard_index=shard_index,
                assigned_ids=assigned_ids,
                audit_path=audit_path,
                sampling_csv=sampling_csv,
                sampling_summary=sampling_summary,
                official_csv=official_csv,
            )
            official_rows, rmsd_check = _parse_official_csv(
                official_csv,
                expected_ids_for_shard=assigned_ids,
                expected_rmsd_check=global_rmsd_check,
            )
            if summary.get("rmsd_check") != rmsd_check:
                raise ValueError(f"{official_summary}: RMSD column binding mismatch")
            global_rmsd_check = rmsd_check
            for row in sampling_rows:
                if row["id"] in sampling_rows_by_id:
                    raise ValueError(f"{ETA_TAGS[eta]}: duplicate sampling ID")
                sampling_rows_by_id[row["id"]] = row
            for row in official_rows:
                if row["id"] in official_rows_by_id:
                    raise ValueError(f"{ETA_TAGS[eta]}: duplicate PoseBusters ID")
                official_rows_by_id[row["id"]] = row
            official_ledger.append(
                {
                    "eta": eta,
                    "shard_index": shard_index,
                    "csv_sha256": file_sha256(official_csv),
                    "summary_sha256": file_sha256(official_summary),
                }
            )
        if sorted(sampling_rows_by_id) != selected_ids or sorted(official_rows_by_id) != selected_ids:
            raise ValueError(f"{ETA_TAGS[eta]}: exact {EXPECTED_COUNT} denominator not met")
        sampling_by_eta[eta] = sampling_rows_by_id
        official_by_eta[eta] = official_rows_by_id
        telemetry_by_eta[eta] = _telemetry(
            [sampling_rows_by_id[sample_id] for sample_id in selected_ids],
            sampling_summaries,
        )

    arms: list[dict[str, Any]] = []
    paired_vectors: dict[float, dict[str, np.ndarray]] = {}
    for eta in ETA_VALUES:
        sampling_rows = sampling_by_eta[eta]
        official_rows = official_by_eta[eta]
        confidence_rmsd = np.asarray(
            [float(sampling_rows[sample_id]["confidence_rmsd"]) for sample_id in selected_ids]
        )
        oracle_rmsd = np.asarray(
            [float(sampling_rows[sample_id]["oracle_rmsd"]) for sample_id in selected_ids]
        )
        pb_valid = np.asarray(
            [official_rows[sample_id]["posebusters_valid"] for sample_id in selected_ids],
            dtype=float,
        )
        top1 = (confidence_rmsd < 2.0).astype(float)
        oracle = (oracle_rmsd < 2.0).astype(float)
        joint = top1 * pb_valid
        paired_vectors[eta] = {
            "top1_rmsd_lt2": top1,
            "oracle_rmsd_lt2": oracle,
            "posebusters_valid": pb_valid,
            "joint": joint,
            "confidence_rmsd": confidence_rmsd,
        }
        per_check_failure_pct = {
            check: (
                1.0
                - np.mean(
                    [official_rows[sample_id]["checks"][check] for sample_id in selected_ids]
                )
            )
            * 100.0
            for check in VALIDITY_CHECKS
        }
        arms.append(
            {
                "eta": eta,
                "eta_tag": ETA_TAGS[eta],
                "denominator": EXPECTED_COUNT,
                "confidence_top1_rmsd_lt2_pct": float(np.mean(top1) * 100.0),
                "confidence_top1_median_rmsd_angstrom": float(np.median(confidence_rmsd)),
                "oracle_of_100_rmsd_lt2_pct": float(np.mean(oracle) * 100.0),
                "posebusters_pass_all_27_pct": float(np.mean(pb_valid) * 100.0),
                "joint_top1_rmsd_lt2_and_pb_valid_pct": float(np.mean(joint) * 100.0),
                "per_check_failure_pct": per_check_failure_pct,
                "guidance_telemetry": telemetry_by_eta[eta],
            }
        )

    baseline = paired_vectors[0.0]
    comparisons: list[dict[str, Any]] = []
    binary_metrics = (
        "top1_rmsd_lt2",
        "oracle_rmsd_lt2",
        "posebusters_valid",
        "joint",
    )
    for eta_index, eta in enumerate(ETA_VALUES[1:], start=1):
        arm = paired_vectors[eta]
        metrics: dict[str, Any] = {}
        for metric_index, metric in enumerate(binary_metrics):
            metrics[metric] = paired_bootstrap_delta(
                baseline[metric],
                arm[metric],
                seed=BOOTSTRAP_SEED + eta_index * 100 + metric_index,
                statistic="mean_percentage_points",
                resamples=bootstrap_resamples,
            )
        metrics["confidence_median_rmsd"] = paired_bootstrap_delta(
            baseline["confidence_rmsd"],
            arm["confidence_rmsd"],
            seed=BOOTSTRAP_SEED + eta_index * 100 + len(binary_metrics),
            statistic="median_difference_angstrom",
            resamples=bootstrap_resamples,
        )
        comparisons.append(
            {
                "eta": eta,
                "baseline_eta": 0.0,
                "paired_complex_count": EXPECTED_COUNT,
                "metrics": metrics,
                "predeclared_target_components": {
                    "joint_delta_at_least_2pp": metrics["joint"]["delta"] >= 2.0,
                    "top1_rmsd_lt2_delta_at_least_minus_2pp": (
                        metrics["top1_rmsd_lt2"]["delta"] >= -2.0
                    ),
                },
            }
        )

    return {
        "schema_version": REPORT_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "status": "complete",
        "claim_scope": "guidance_development_not_untouched_confirmation",
        "automatic_eta_selection": False,
        "selected_eta": None,
        "decision_owner": "user",
        "primary_selector": PRIMARY_SELECTOR,
        "posebusters": {
            "version": EXPECTED_POSEBUSTERS_VERSION,
            "config": "redock",
            "pass_all_definition": "all 27 non-RMSD redock checks",
            "validity_checks": list(VALIDITY_CHECKS),
            "separate_rmsd_check_reported_but_not_used_in_pass_all": global_rmsd_check,
        },
        "denominator": EXPECTED_COUNT,
        "ids_sha256": ids_sha256(selected_ids),
        "eta_values": list(ETA_VALUES),
        "sampling_audit": str(audit_path.resolve()),
        "sampling_audit_sha256": file_sha256(audit_path),
        "raw_manifest_sha256": file_sha256(raw_manifest),
        "raw_gate_sha256": file_sha256(raw_gate),
        "raw_gate_sidecar_sha256": file_sha256(raw_gate_sidecar),
        "official_artifact_ledger_sha256": canonical_json_sha256(official_ledger),
        "posebusters_recovered_attempt_count": len(posebusters_recovered_attempts),
        "posebusters_recovered_attempts": posebusters_recovered_attempts,
        "posebusters_recovered_attempts_sha256": canonical_json_sha256(
            posebusters_recovered_attempts
        ),
        "bootstrap": {
            "method": "paired complex-ID percentile bootstrap",
            "base_seed": BOOTSTRAP_SEED,
            "resamples": bootstrap_resamples,
            "confidence_level": 0.95,
        },
        "arms": arms,
        "paired_comparisons_vs_eta0": comparisons,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sampling-root", type=Path, required=True)
    parser.add_argument("--posebusters-root", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, default=Path("data/splits/plinder.json"))
    parser.add_argument("--raw-manifest", type=Path, required=True)
    parser.add_argument("--raw-gate", type=Path, required=True)
    parser.add_argument("--raw-gate-sidecar", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=BOOTSTRAP_RESAMPLES)
    args = parser.parse_args(argv)
    report = build_report(
        sampling_root=args.sampling_root,
        posebusters_root=args.posebusters_root,
        audit_path=args.audit,
        split_file=args.split_file,
        raw_manifest=args.raw_manifest,
        raw_gate=args.raw_gate,
        raw_gate_sidecar=args.raw_gate_sidecar,
        raw_root=args.raw_root,
        bootstrap_resamples=args.bootstrap_resamples,
    )
    write_json_noreplace(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
