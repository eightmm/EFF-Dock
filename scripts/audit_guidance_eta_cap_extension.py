#!/usr/bin/env python3
"""Fail-closed audit for eta={2.5,3.0} cap-saturation sampling."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

PROTOCOL_ID = "EFFDOCK-UNIFIED-GUIDANCE-ETA-CAP-EXTENSION-V1"
ETAS = (2.5, 3.0)
TAGS = {eta: f"eta{int(round(eta * 1000)):04d}" for eta in ETAS}
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


def _run_name(dataset: str, eta: float) -> str:
    return f"effdock-guidance-eta-cap-extension-v1-{dataset}-n100-s10-{TAGS[eta]}"


def _paths(root: Path, mode: str, dataset: str, eta: float, shard: int) -> tuple[Path, Path]:
    stem = _run_name(dataset, eta)
    base = root / ("smoke/raw" if mode == "smoke" else "raw")
    if mode == "full":
        stem += f".shard-{shard:03d}-of-{SHARDS:03d}"
    return base / f"{stem}.csv", base / f"{stem}.summary.json"


def _eq(actual: object, expected: object, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label}: expected {expected!r}, got {actual!r}")


def _summary(path: Path, dataset: str, eta: float, mode: str, shard: int) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    _eq(value.get("protocol_id"), PROTOCOL_ID, f"{path}: protocol")
    _eq(value.get("dataset"), dataset, f"{path}: dataset")
    _eq(value.get("run_name"), _run_name(dataset, eta), f"{path}: run")
    _eq(float(value.get("sigma")), 0.5, f"{path}: sigma")
    _eq(float(value.get("unified_guidance_scale")), eta, f"{path}: eta")
    _eq(value.get("unified_guidance_mode"), "normalized_drift", f"{path}: guidance mode")
    _eq(value.get("selector_profile"), "confidence_cluster_free", f"{path}: selector")
    _eq(int(value.get("num_samples")), 100, f"{path}: samples")
    _eq(int(value.get("num_steps")), 10, f"{path}: steps")
    _eq(int(value.get("prior_pool_size")), 100, f"{path}: prior pool")
    _eq(int(value.get("expected_discovered_count")), DATASETS[dataset], f"{path}: discovered")
    _eq(bool(value.get("require_complete_success")), True, f"{path}: completeness")
    _eq(int(value.get("num_failed")), 0, f"{path}: failed")
    _eq(value.get("failures"), [], f"{path}: failure records")
    _eq(int(value.get("num_shards")), 1 if mode == "smoke" else SHARDS, f"{path}: shards")
    _eq(int(value.get("shard_index")), 0 if mode == "smoke" else shard, f"{path}: shard")
    runtime = value.get("runtime") or {}
    _eq(runtime.get("device"), "cuda", f"{path}: device")
    allocated = int(runtime.get("cuda_max_memory_allocated_bytes", -1))
    if allocated < 0 or allocated >= ALLOCATED_LIMIT_BYTES:
        raise ValueError(f"{path}: allocated CUDA peak violates <48 GiB gate: {allocated}")
    stats = value.get("guidance_runtime_stats") or {}
    if int(stats.get("direct_nonfinite_poses", 0)) != 0:
        raise ValueError(f"{path}: non-finite direct guidance poses")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.input_manifest.read_text(encoding="utf-8"))
    expected_ids = {name: set(manifest["datasets"][name]["ligands"]) for name in DATASETS}
    records = []
    total_rows = 0
    max_allocated = 0
    max_reserved = 0
    for dataset, count in DATASETS.items():
        for eta in ETAS:
            ids: set[str] = set()
            shards = range(1) if args.mode == "smoke" else range(SHARDS)
            for shard in shards:
                csv_path, summary_path = _paths(args.output_root, args.mode, dataset, eta, shard)
                if not csv_path.is_file() or not summary_path.is_file():
                    raise FileNotFoundError(f"missing {csv_path} or {summary_path}")
                summary = _summary(summary_path, dataset, eta, args.mode, shard)
                runtime = summary["runtime"]
                max_allocated = max(max_allocated, int(runtime["cuda_max_memory_allocated_bytes"]))
                max_reserved = max(max_reserved, int(runtime.get("cuda_max_memory_reserved_bytes", 0)))
                with csv_path.open(newline="", encoding="utf-8") as handle:
                    rows = list(csv.DictReader(handle))
                _eq(len(rows), int(summary["num_success"]), f"{csv_path}: rows")
                for row in rows:
                    complex_id = row["id"]
                    if complex_id in ids:
                        raise ValueError(f"{dataset}/{eta}: duplicate {complex_id}")
                    ids.add(complex_id)
                    _eq(int(row["all_poses_count"]), 100, f"{complex_id}: pose count")
                    _eq(int(row["prior_pool_size"]), 100, f"{complex_id}: prior pool")
                    for key in ("oracle_rmsd", "confidence_rmsd", "confidence_filter_rmsd"):
                        if not math.isfinite(float(row[key])):
                            raise ValueError(f"{complex_id}: non-finite {key}")
                    pose_path = Path(row["all_poses_sdf"])
                    if not pose_path.is_file() or _sha256(pose_path) != row["all_poses_sdf_sha256"]:
                        raise ValueError(f"{complex_id}: all-pose hash mismatch")
                records.append({"dataset": dataset, "eta": eta, "shard": shard, "rows": len(rows)})
                total_rows += len(rows)
            required = {SMOKE_IDS[dataset]} if args.mode == "smoke" else expected_ids[dataset]
            _eq(ids, required, f"{dataset}/{eta}: ID inventory")
            _eq(len(ids), 1 if args.mode == "smoke" else count, f"{dataset}/{eta}: denominator")
    expected_rows = len(ETAS) * (len(DATASETS) if args.mode == "smoke" else sum(DATASETS.values()))
    _eq(total_rows, expected_rows, "total rows")
    result = {
        "schema_version": "effdock.guidance_eta_cap_extension_audit.v1",
        "status": "passed",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_id": PROTOCOL_ID,
        "mode": args.mode,
        "etas": list(ETAS),
        "sigma": 0.5,
        "rows": total_rows,
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
