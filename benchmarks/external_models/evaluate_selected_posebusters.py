#!/usr/bin/env python3
"""Run official PoseBusters redock checks on one external-model Top-1 shard."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from posebusters import PoseBusters

from effdock.workflows.guidance_budget_posebusters_report import VALIDITY_CHECKS
from effdock.workflows.posebusters_report import require_posebusters_runtime_version

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RMSD_ROOT = (
    ROOT / "outputs/external_models/evaluation/official_repeat_rmsd_20260901"
)
DEFAULT_OUTPUT_ROOT = (
    ROOT / "outputs/external_models/evaluation/official_selected_posebusters_20260901"
)
DATASETS = {
    "astex_diverse": (
        85,
        ROOT
        / "outputs/external_models/inputs/posebench_native/astex_diverse/"
        "vina_astex_diverse_inputs.csv",
    ),
    "posebusters_benchmark": (
        308,
        ROOT
        / "outputs/external_models/inputs/posebench_native/posebusters_benchmark/"
        "vina_posebusters_benchmark_inputs.csv",
    ),
}
MODELS = ("diffdock_pocket", "rldiff_rlpp", "diffbindfr")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--dataset", choices=tuple(DATASETS), required=True)
    parser.add_argument("--repeat-index", type=int, choices=range(3), required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--rmsd-root", type=Path, default=DEFAULT_RMSD_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def validated_checks(raw: dict[str, Any], label: str) -> tuple[str, dict[str, bool]]:
    rmsd = [key for key in raw if str(key).startswith("rmsd_")]
    if len(rmsd) != 1 or set(raw) != {*VALIDITY_CHECKS, rmsd[0]}:
        raise ValueError(f"{label}: unexpected PoseBusters redock schema")
    checks = {
        key: False if pd.isna(raw[key]) else bool(raw[key])
        for key in VALIDITY_CHECKS
    }
    return rmsd[0], checks


def main() -> None:
    args = parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid shard specification")
    denominator, manifest_path = DATASETS[args.dataset]
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        manifests = {row["complex_name"]: row for row in csv.DictReader(handle)}
    if len(manifests) != denominator:
        raise ValueError(f"manifest denominator mismatch: {len(manifests)} != {denominator}")

    source_path = (
        args.rmsd_root
        / args.model
        / args.dataset
        / f"repeat_{args.repeat_index}"
        / f"{args.model}__{args.dataset}.csv"
    )
    with source_path.open(newline="", encoding="utf-8") as handle:
        source_rows = list(csv.DictReader(handle))
    if len(source_rows) != denominator:
        raise ValueError(f"RMSD result denominator mismatch: {len(source_rows)} != {denominator}")
    if {row["complex_name"] for row in source_rows} != set(manifests):
        raise ValueError("RMSD result and input manifest target IDs differ")
    assigned = source_rows[args.shard_index :: args.num_shards]
    if args.limit is not None:
        assigned = assigned[: args.limit]
    if not assigned:
        raise ValueError("no targets assigned to shard")

    shard = f"shard_{args.shard_index:03d}_of_{args.num_shards:03d}"
    final_dir = (
        args.output_root
        / args.model
        / args.dataset
        / f"repeat_{args.repeat_index}"
        / shard
    )
    if final_dir.exists():
        raise FileExistsError(final_dir)
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    incomplete = final_dir.parent / ".incomplete"
    incomplete.mkdir(parents=True, exist_ok=True)
    attempt = Path(tempfile.mkdtemp(prefix=f"{shard}.", dir=incomplete))

    version = require_posebusters_runtime_version()
    buster = PoseBusters(config="redock", max_workers=0)
    started = time.monotonic()
    results: list[dict[str, Any]] = []
    observed_rmsd_check: str | None = None
    for position, source in enumerate(assigned, start=1):
        target = source["complex_name"]
        manifest = manifests[target]
        rmsd = float(source["top1_rmsd"]) if source["top1_rmsd"] else math.inf
        result: dict[str, Any] = {
            "model": args.model,
            "dataset": args.dataset,
            "repeat_index": args.repeat_index,
            "complex_name": target,
            "top1_pose": source["top1_pose"],
            "top1_rmsd": rmsd,
            "top1_rmsd_lt2": math.isfinite(rmsd) and rmsd < 2.0,
            "posebusters_valid": False,
            "joint_rmsd_lt2_pb_valid": False,
            "posebusters_rmsd": "",
            "error": "",
            **{key: False for key in VALIDITY_CHECKS},
        }
        try:
            pose = Path(source["top1_pose"])
            reference = Path(manifest["reference_ligand"])
            receptor = Path(manifest["holo_protein"])
            for label, path in (("pose", pose), ("reference", reference), ("receptor", receptor)):
                if not path.is_file():
                    raise FileNotFoundError(f"missing {label}: {path}")
            frame = buster.bust(pose, reference, receptor, full_report=False)
            if len(frame.index) != 1:
                raise ValueError(f"expected one PoseBusters row, found {len(frame.index)}")
            rmsd_check, checks = validated_checks(frame.iloc[0].to_dict(), target)
            if observed_rmsd_check is None:
                observed_rmsd_check = rmsd_check
            elif observed_rmsd_check != rmsd_check:
                raise ValueError("PoseBusters RMSD column changed within shard")
            pb_valid = all(checks.values())
            result.update(checks)
            result["posebusters_valid"] = pb_valid
            result["joint_rmsd_lt2_pb_valid"] = bool(result["top1_rmsd_lt2"]) and pb_valid
            raw_pb_rmsd = frame.iloc[0].to_dict()[rmsd_check]
            result["posebusters_rmsd"] = "" if pd.isna(raw_pb_rmsd) else float(raw_pb_rmsd)
        except Exception as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
        results.append(result)
        print(
            f"[{position}/{len(assigned)}] {args.model}/{args.dataset}/{target} "
            f"RMSD<2={result['top1_rmsd_lt2']} PB={result['posebusters_valid']} "
            f"error={bool(result['error'])}",
            flush=True,
        )

    csv_path = attempt / "results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    summary = {
        "schema_version": 1,
        "status": "complete",
        "comparison_scope": "supplied_pocket_only",
        "model": args.model,
        "dataset": args.dataset,
        "repeat_index": args.repeat_index,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "num_results": len(results),
        "num_errors": sum(bool(row["error"]) for row in results),
        "posebusters_version": version,
        "posebusters_config": "redock",
        "receptor_contract": "frozen holo protein from common supplied-pocket manifest",
        "official_validity_checks": list(VALIDITY_CHECKS),
        "separate_rmsd_check": observed_rmsd_check,
        "source_rmsd_csv": str(source_path),
        "runtime": {
            "elapsed_seconds": time.monotonic() - started,
            "finished_at_utc": datetime.now(UTC).isoformat(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        },
    }
    (attempt / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    os.rename(attempt, final_dir)


if __name__ == "__main__":
    main()
