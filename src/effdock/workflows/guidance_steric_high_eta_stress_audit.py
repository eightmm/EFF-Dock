#!/usr/bin/env python3
"""Fail-closed single-complex CUDA stress audit for the high-eta profile."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from effdock.guidance.provenance import guidance_implementation_identity
from effdock.workflows import guidance_eta_sweep_confidence_identity as replay
from effdock.workflows import guidance_eta_sweep_confidence_standalone_audit as standalone
from effdock.workflows.guidance_eta_sweep_standalone_spec import STERIC_HIGH_ETA_V1

STRESS_ID = "8f4j_pho"
DATASET = "posebusters"
ETA = 2.0
AUDIT_CONTRACT = "EFFDOCK_STERIC_HIGH_ETA_CUDA_STRESS_V1"


def build_stress_audit(
    sampling_dir: Path,
    *,
    cohort_audit: Path,
) -> dict[str, Any]:
    spec = STERIC_HIGH_ETA_V1
    standalone._validate_cohort_audit(cohort_audit)
    sampling_dir = sampling_dir.resolve()
    run_name = spec.expected_run_name(DATASET, ETA)
    summary_path = sampling_dir / f"{run_name}.summary.json"
    observed_summaries = set(sampling_dir.glob("*.summary.json"))
    if observed_summaries != {summary_path}:
        raise ValueError("stress summary inventory must contain exactly the frozen run")

    summary = replay._load_json_object(summary_path)
    current_implementation = guidance_implementation_identity()
    standalone._validate_summary(
        summary,
        path=summary_path,
        dataset=DATASET,
        eta=ETA,
        shard_index=0,
        smoke=True,
        current_implementation=current_implementation,
        spec=spec,
    )
    csv_path = replay._companion_csv(summary_path, summary, sampling_dir)
    fields, rows = replay._read_raw_csv(csv_path)
    if len(rows) != 1 or rows[0].get("id") != STRESS_ID:
        raise ValueError(f"stress CSV must contain exactly {STRESS_ID}")
    row_record, file_record, input_identity, trace = standalone._validate_row(
        rows[0],
        fields=fields,
        sampling_dir=sampling_dir,
        run_name=run_name,
        dataset=DATASET,
        eta=ETA,
        shard_index=0,
        digest_cache={},
        spec=spec,
    )
    replay._validate_trace_runtime_consistency(summary, [trace])

    runtime = summary["runtime"]
    stats = summary["guidance_runtime_stats"]
    return {
        "protocol_id": spec.protocol_id,
        "audit_contract": AUDIT_CONTRACT,
        "status": "passed",
        "selection_basis": (
            "largest historical full-cohort product of standard receptor-source "
            "heavy atoms and ligand input heavy atoms; outcome labels unused"
        ),
        "dataset": DATASET,
        "id": STRESS_ID,
        "eta": ETA,
        "run_name": run_name,
        "sampling_dir": str(sampling_dir),
        "summary": str(summary_path),
        "summary_sha256": replay._file_sha256(summary_path),
        "csv": str(csv_path),
        "csv_sha256": replay._file_sha256(csv_path),
        "guidance_parameter_sha256": spec.guidance_parameter_sha256,
        "physical_parameter_sha256": spec.physical_parameter_sha256,
        "interaction_parameter_sha256": spec.interaction_parameter_sha256,
        "receptor_policy_sha256": spec.receptor_policy_sha256,
        "guidance_implementation_sha256": current_implementation["sha256"],
        "cuda": {
            "gpu": runtime["gpu"],
            "max_memory_allocated_bytes": runtime["cuda_max_memory_allocated_bytes"],
            "max_memory_reserved_bytes": runtime["cuda_max_memory_reserved_bytes"],
            "device_limit_bytes": replay.GPU_MEMORY_LIMIT_BYTES,
            "allocated_headroom_limit_bytes": int(0.9 * replay.GPU_MEMORY_LIMIT_BYTES),
        },
        "numerical": {
            "direct_nonfinite_poses": stats["direct_nonfinite_poses"],
            "direct_zero_raw_direction_poses": stats["direct_zero_raw_direction_poses"],
            "direct_zero_reference_velocity_poses": stats[
                "direct_zero_reference_velocity_poses"
            ],
            "direct_max_translation_velocity": stats[
                "direct_max_translation_velocity"
            ],
            "direct_max_angular_velocity": stats["direct_max_angular_velocity"],
            "direct_max_estimated_atom_displacement": stats[
                "direct_max_estimated_atom_displacement"
            ],
        },
        "row_integrity": row_record,
        "file_binding": file_record,
        "input_identity": input_identity,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sampling-dir", type=Path, required=True)
    parser.add_argument("--cohort-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    audit = build_stress_audit(args.sampling_dir, cohort_audit=args.cohort_audit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(audit, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
