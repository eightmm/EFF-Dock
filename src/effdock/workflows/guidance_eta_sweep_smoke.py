#!/usr/bin/env python3
"""Fail-closed numerical gate for one eta-sweep smoke output."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from effdock.workflows.guidance_budget_report import _read_shard_rows
from effdock.workflows.guidance_eta_sweep_report import (
    EXPECTED_GUIDANCE_MODE,
    NUM_SAMPLES,
    NUM_STEPS,
    PROTOCOL_ID,
    _validate_direct_runtime,
    _validate_direct_step_trace,
    _validate_trace_runtime_consistency,
    eta_value,
    expected_run_name,
)


def validate_smoke_summary(
    summary_path: Path,
    *,
    dataset: str,
    eta: float,
    complex_id: str,
) -> dict[str, Any]:
    summary = json.loads(summary_path.read_text())
    if not isinstance(summary, dict):
        raise ValueError(f"{summary_path}: summary must be a JSON object")
    expected_eta = eta_value(eta)
    expected_run = expected_run_name(dataset, expected_eta)
    exact = {
        "protocol_id": PROTOCOL_ID,
        "run_name": expected_run,
        "dataset": dataset,
        "num_samples": NUM_SAMPLES,
        "num_steps": NUM_STEPS,
        "model_pose_step_budget": NUM_SAMPLES * NUM_STEPS,
        "num_shards": 1,
        "shard_index": 0,
        "num_assigned": 1,
        "num_success": 1,
        "num_failed": 0,
        "unified_guidance_mode": EXPECTED_GUIDANCE_MODE,
    }
    for key, expected in exact.items():
        if summary.get(key) != expected:
            raise ValueError(
                f"{summary_path}: {key} must be {expected!r}, got {summary.get(key)!r}"
            )
    if not math.isclose(
        float(summary.get("unified_guidance_scale", math.nan)),
        expected_eta,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(f"{summary_path}: eta mismatch")
    if summary.get("failures") != []:
        raise ValueError(f"{summary_path}: smoke contains recorded failures")

    rows = _read_shard_rows(summary_path, summary, summary_path.parent)
    if len(rows) != 1 or rows[0]["id"] != complex_id:
        raise ValueError(f"{summary_path}: smoke must contain only {complex_id}")
    row = rows[0]
    expected_mode = "unified_normalized_drift" if expected_eta > 0.0 else "none"
    if row.get("guidance_mode") != expected_mode:
        raise ValueError(f"{summary_path}: smoke row guidance mode mismatch")
    trace = _validate_direct_step_trace(row, eta=expected_eta)
    if expected_eta > 0.0:
        _validate_direct_runtime(summary, expected_successes=1)
        _validate_trace_runtime_consistency(summary, [trace])
        if len(trace) != 8:
            raise ValueError(f"{summary_path}: positive eta must trace eight active steps")
    else:
        if summary.get("guidance_runtime_stats") not in (None, {}):
            raise ValueError(f"{summary_path}: eta=0 unexpectedly has guidance telemetry")
        if trace:
            raise ValueError(f"{summary_path}: eta=0 trace must be empty")
    return {
        "status": "passed",
        "protocol_id": PROTOCOL_ID,
        "run_name": expected_run,
        "dataset": dataset,
        "eta": expected_eta,
        "id": complex_id,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--eta", type=float, required=True)
    parser.add_argument("--id", required=True)
    args = parser.parse_args(argv)
    result = validate_smoke_summary(
        args.summary,
        dataset=args.dataset,
        eta=args.eta,
        complex_id=args.id,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
