#!/usr/bin/env python3
"""Fail-closed inventory and numerical audit for the eta-2 sigma sweep."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

PROTOCOL_ID = "EFFDOCK-UNIFIED-GUIDANCE-SIGMA-SWEEP-ETA2-V1"
REFERENCE_PROTOCOL_ID = "EFFDOCK-UNIFIED-GUIDANCE-STERIC-HIGH-ETA-CONFIDENCE-PB-V1"
SIGMAS = (1.0, 2.0, 3.0, 4.0)
SIGMA_TAGS = {value: f"sigma{int(round(value * 1000)):04d}" for value in SIGMAS}
DATASETS = {"astex": 85, "posebusters": 308}
SMOKE_IDS = {"astex": "1jje", "posebusters": "7b2c_tp7"}
SHARDS = 8
ALLOCATED_LIMIT_BYTES = 48 * 1024**3


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_name(dataset: str, sigma: float) -> str:
    return f"effdock-guidance-sigma-sweep-eta2-v1-{dataset}-n100-s10-{SIGMA_TAGS[sigma]}"


def _paths(root: Path, mode: str, dataset: str, sigma: float, shard: int) -> tuple[Path, Path]:
    stem = _run_name(dataset, sigma)
    base = root / ("smoke/raw" if mode == "smoke" else "raw")
    if mode == "full":
        stem += f".shard-{shard:03d}-of-{SHARDS:03d}"
    return base / f"{stem}.csv", base / f"{stem}.summary.json"


def _require_equal(actual: object, expected: object, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label}: expected {expected!r}, got {actual!r}")


def _audit_summary(
    path: Path,
    dataset: str,
    sigma: float,
    mode: str,
    shard: int,
    *,
    protocol_id: str = PROTOCOL_ID,
    run_name: str | None = None,
) -> dict:
    summary = json.loads(path.read_text(encoding="utf-8"))
    _require_equal(summary.get("protocol_id"), protocol_id, f"{path}: protocol")
    _require_equal(summary.get("dataset"), dataset, f"{path}: dataset")
    _require_equal(summary.get("run_name"), run_name or _run_name(dataset, sigma), f"{path}: run")
    _require_equal(float(summary.get("sigma")), sigma, f"{path}: sigma")
    _require_equal(float(summary.get("unified_guidance_scale")), 2.0, f"{path}: eta")
    _require_equal(summary.get("unified_guidance_mode"), "normalized_drift", f"{path}: mode")
    _require_equal(summary.get("selector_profile"), "confidence_cluster_free", f"{path}: selector")
    _require_equal(int(summary.get("num_samples")), 100, f"{path}: samples")
    _require_equal(int(summary.get("num_steps")), 10, f"{path}: steps")
    _require_equal(int(summary.get("prior_pool_size")), 100, f"{path}: prior pool")
    _require_equal(int(summary.get("expected_discovered_count")), DATASETS[dataset], f"{path}: count")
    _require_equal(bool(summary.get("require_complete_success")), True, f"{path}: complete flag")
    _require_equal(int(summary.get("num_failed")), 0, f"{path}: failures")
    _require_equal(summary.get("failures"), [], f"{path}: failure records")
    _require_equal(int(summary.get("num_shards")), 1 if mode == "smoke" else SHARDS, f"{path}: shards")
    _require_equal(int(summary.get("shard_index")), 0 if mode == "smoke" else shard, f"{path}: shard")
    runtime = summary.get("runtime") or {}
    _require_equal(runtime.get("device"), "cuda", f"{path}: device")
    allocated = int(runtime.get("cuda_max_memory_allocated_bytes", -1))
    if allocated < 0 or allocated >= ALLOCATED_LIMIT_BYTES:
        raise ValueError(f"{path}: CUDA allocated peak {allocated} violates <48 GiB gate")
    stats = summary.get("guidance_runtime_stats") or {}
    for key in ("direct_nonfinite_poses", "nonfinite_base_poses", "nonfinite_trials"):
        if int(stats.get(key, 0)) != 0:
            raise ValueError(f"{path}: nonzero {key}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("reference", "smoke", "full"), required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.input_manifest.read_text(encoding="utf-8"))
    expected_ids = {
        dataset: set(manifest["datasets"][dataset]["ligands"])
        for dataset in DATASETS
    }
    records: list[dict] = []
    all_rows = 0
    max_allocated = 0
    max_reserved = 0
    active_sigmas = (0.5,) if args.mode == "reference" else SIGMAS
    for dataset, expected_count in DATASETS.items():
        for sigma in active_sigmas:
            ids: set[str] = set()
            shard_range = range(1) if args.mode == "smoke" else range(SHARDS)
            for shard in shard_range:
                if args.mode == "reference":
                    run_name = f"effdock-guidance-steric-high-eta-v1-{dataset}-n100-s10-eta2000"
                    stem = f"{run_name}.shard-{shard:03d}-of-{SHARDS:03d}"
                    csv_path = args.output_root / "raw" / f"{stem}.csv"
                    summary_path = args.output_root / "raw" / f"{stem}.summary.json"
                    protocol_id = REFERENCE_PROTOCOL_ID
                else:
                    csv_path, summary_path = _paths(args.output_root, args.mode, dataset, sigma, shard)
                    run_name = _run_name(dataset, sigma)
                    protocol_id = PROTOCOL_ID
                if not csv_path.is_file() or not summary_path.is_file():
                    raise FileNotFoundError(f"missing artifact: {csv_path} or {summary_path}")
                summary = _audit_summary(
                    summary_path,
                    dataset,
                    sigma,
                    args.mode,
                    shard,
                    protocol_id=protocol_id,
                    run_name=run_name,
                )
                runtime = summary["runtime"]
                max_allocated = max(max_allocated, int(runtime["cuda_max_memory_allocated_bytes"]))
                max_reserved = max(max_reserved, int(runtime.get("cuda_max_memory_reserved_bytes", 0)))
                with csv_path.open(newline="", encoding="utf-8") as handle:
                    rows = list(csv.DictReader(handle))
                _require_equal(len(rows), int(summary["num_success"]), f"{csv_path}: rows")
                for row in rows:
                    complex_id = row["id"]
                    if complex_id in ids:
                        raise ValueError(f"{dataset}/{sigma}: duplicate ID {complex_id}")
                    ids.add(complex_id)
                    _require_equal(int(row["all_poses_count"]), 100, f"{complex_id}: all poses")
                    _require_equal(int(row["prior_pool_size"]), 100, f"{complex_id}: prior pool")
                    for key in ("oracle_rmsd", "confidence_rmsd", "confidence_filter_rmsd", "confidence_pred_rmsd"):
                        if not math.isfinite(float(row[key])):
                            raise ValueError(f"{complex_id}: non-finite {key}")
                    pose_path = Path(row["all_poses_sdf"])
                    if not pose_path.is_file() or _sha256(pose_path) != row["all_poses_sdf_sha256"]:
                        raise ValueError(f"{complex_id}: all-pose artifact hash mismatch")
                records.append({
                    "dataset": dataset,
                    "sigma": sigma,
                    "shard": shard,
                    "rows": len(rows),
                    "summary": str(summary_path),
                    "summary_sha256": _sha256(summary_path),
                    "csv": str(csv_path),
                    "csv_sha256": _sha256(csv_path),
                })
                all_rows += len(rows)
            required = {SMOKE_IDS[dataset]} if args.mode == "smoke" else expected_ids[dataset]
            _require_equal(ids, required, f"{dataset}/{sigma}: exact ID inventory")
            _require_equal(len(ids), 1 if args.mode == "smoke" else expected_count, f"{dataset}/{sigma}: denominator")

    expected_rows = len(active_sigmas) * (
        len(DATASETS) if args.mode == "smoke" else sum(DATASETS.values())
    )
    _require_equal(all_rows, expected_rows, "total rows")
    result = {
        "schema_version": "effdock.guidance_sigma_sweep_eta2_audit.v1",
        "status": "passed",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_id": PROTOCOL_ID,
        "mode": args.mode,
        "eta": 2.0,
        "sigmas": list(SIGMAS),
        "datasets": DATASETS,
        "rows": all_rows,
        "shards": len(records),
        "cuda_max_memory_allocated_bytes": max_allocated,
        "cuda_max_memory_reserved_bytes_record_only": max_reserved,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("status", "mode", "rows", "shards")}, sort_keys=True))


if __name__ == "__main__":
    main()
