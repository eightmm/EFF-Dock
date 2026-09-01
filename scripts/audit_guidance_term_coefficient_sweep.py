#!/usr/bin/env python3
"""Fail-closed inventory and numerical audit for the term-coefficient sweep."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path

PROTOCOL_ID = "EFFDOCK-UNIFIED-GUIDANCE-TERM-COEFFICIENT-SWEEP-V1"
ARMS = {
    "base_r080_c1": (0.8, 1.0),
    "steric_r090_c1": (0.9, 1.0),
    "chiral_r080_c2": (0.8, 2.0),
    "combined_r090_c2": (0.9, 2.0),
}
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


def _run_name(dataset: str, arm: str) -> str:
    return f"effdock-guidance-term-coeff-v1-{dataset}-n100-s10-{arm}"


def _paths(root: Path, mode: str, dataset: str, arm: str, shard: int) -> tuple[Path, Path]:
    stem = _run_name(dataset, arm)
    base = root / ("smoke/raw" if mode == "smoke" else "raw")
    if mode == "full":
        stem += f".shard-{shard:03d}-of-{SHARDS:03d}"
    return base / f"{stem}.csv", base / f"{stem}.summary.json"


def _eq(actual: object, expected: object, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label}: expected {expected!r}, got {actual!r}")


def _summary(path: Path, dataset: str, arm: str, mode: str, shard: int) -> dict:
    steric, chiral = ARMS[arm]
    value = json.loads(path.read_text(encoding="utf-8"))
    _eq(value.get("protocol_id"), PROTOCOL_ID, f"{path}: protocol")
    _eq(value.get("dataset"), dataset, f"{path}: dataset")
    _eq(value.get("run_name"), _run_name(dataset, arm), f"{path}: run")
    _eq(float(value.get("sigma")), 0.5, f"{path}: sigma")
    _eq(float(value.get("unified_guidance_scale")), 2.0, f"{path}: eta")
    _eq(float(value.get("unified_guidance_steric_radius_scale")), steric, f"{path}: steric")
    _eq(float(value.get("unified_guidance_chiral_improper_scale")), chiral, f"{path}: chiral")
    _eq(value.get("unified_guidance_mode"), "normalized_drift", f"{path}: mode")
    _eq(value.get("selector_profile"), "confidence_cluster_free", f"{path}: selector")
    _eq(int(value.get("num_samples")), 100, f"{path}: samples")
    _eq(int(value.get("num_steps")), 10, f"{path}: steps")
    _eq(int(value.get("prior_pool_size")), 100, f"{path}: prior pool")
    _eq(int(value.get("expected_discovered_count")), DATASETS[dataset], f"{path}: discovered")
    _eq(bool(value.get("require_complete_success")), True, f"{path}: completeness")
    _eq(int(value.get("num_failed")), 0, f"{path}: failed")
    _eq(value.get("failures"), [], f"{path}: failures")
    _eq(int(value.get("num_shards")), 1 if mode == "smoke" else SHARDS, f"{path}: shards")
    _eq(int(value.get("shard_index")), 0 if mode == "smoke" else shard, f"{path}: shard")
    runtime = value.get("runtime") or {}
    _eq(runtime.get("device"), "cuda", f"{path}: device")
    allocated = int(runtime.get("cuda_max_memory_allocated_bytes", -1))
    if allocated < 1 or allocated >= ALLOCATED_LIMIT_BYTES:
        raise ValueError(f"{path}: CUDA allocated memory violates (0,48 GiB): {allocated}")
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
    records: list[dict] = []
    total_rows = 0
    for dataset, count in DATASETS.items():
        for arm in ARMS:
            ids: set[str] = set()
            for shard in (range(1) if args.mode == "smoke" else range(SHARDS)):
                csv_path, summary_path = _paths(args.output_root, args.mode, dataset, arm, shard)
                if not csv_path.is_file() or not summary_path.is_file():
                    raise FileNotFoundError(f"missing {csv_path} or {summary_path}")
                summary = _summary(summary_path, dataset, arm, args.mode, shard)
                with csv_path.open(newline="", encoding="utf-8") as handle:
                    rows = list(csv.DictReader(handle))
                _eq(len(rows), int(summary["num_success"]), f"{csv_path}: rows")
                for row in rows:
                    complex_id = row["id"]
                    if complex_id in ids:
                        raise ValueError(f"{dataset}/{arm}: duplicate {complex_id}")
                    ids.add(complex_id)
                    _eq(int(row["all_poses_count"]), 100, f"{complex_id}: pose count")
                    _eq(int(row["prior_pool_size"]), 100, f"{complex_id}: prior pool")
                    for key in ("oracle_rmsd", "confidence_rmsd", "confidence_filter_rmsd"):
                        if not math.isfinite(float(row[key])):
                            raise ValueError(f"{complex_id}: non-finite {key}")
                    pose_path = Path(row["all_poses_sdf"])
                    if not pose_path.is_file() or _sha256(pose_path) != row["all_poses_sdf_sha256"]:
                        raise ValueError(f"{complex_id}: all-pose hash mismatch")
                records.append({"dataset": dataset, "arm": arm, "shard": shard, "rows": len(rows)})
                total_rows += len(rows)
            required = {SMOKE_IDS[dataset]} if args.mode == "smoke" else expected_ids[dataset]
            _eq(ids, required, f"{dataset}/{arm}: ID inventory")
            _eq(len(ids), 1 if args.mode == "smoke" else count, f"{dataset}/{arm}: denominator")
    expected_rows = len(ARMS) * (len(DATASETS) if args.mode == "smoke" else sum(DATASETS.values()))
    _eq(total_rows, expected_rows, "total rows")
    result = {
        "schema_version": "effdock.guidance_term_coefficient_sweep_audit.v1",
        "status": "passed",
        "created_utc": datetime.now(UTC).isoformat(),
        "protocol_id": PROTOCOL_ID,
        "mode": args.mode,
        "arms": {name: {"steric_radius_scale": values[0], "chiral_improper_scale": values[1]} for name, values in ARMS.items()},
        "eta": 2.0,
        "sigma": 0.5,
        "rows": total_rows,
        "shards": len(records),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("status", "mode", "rows", "shards")}, sort_keys=True))


if __name__ == "__main__":
    main()
