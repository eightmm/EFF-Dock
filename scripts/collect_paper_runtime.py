"""Runtime collector based on GPT-5.6-luna audit; no latency imputation."""

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
B = ROOT / "outputs/benchmarks"
ARMS = {
    "unguided_n100_s10": (
        B / "astex_pb_unguided_r3_hostmatched_v2/n100_s10",
        {"astex": 85, "posebusters": 308},
    ),
    "unguided_n40_s25": (
        B / "astex_pb_unguided_r3_hostmatched_v2/n40_s25",
        {"astex": 85, "posebusters": 308},
    ),
    "guided_n40_s25": (B / "astex_pb_n40_s25_r3_v1", {"astex": 85, "posebusters": 308}),
    "temporal_n100_s10": (
        B / "temporal_full_three_seed_v1",
        {"phibench": 206, "foldbench": 558, "openbind": 925},
    ),
}


def stats(values):
    if not values:
        return dict(count=0, mean=None, median=None, min=None, max=None)
    if not all(math.isfinite(v) and v >= 0 for v in values):
        raise ValueError("invalid time/memory")
    return dict(
        count=len(values),
        mean=statistics.mean(values),
        median=statistics.median(values),
        min=min(values),
        max=max(values),
    )


def collect():
    sources = []
    rows = []
    for arm, (root, cohorts) in ARMS.items():
        for dataset, count in cohorts.items():
            for repeat in range(3):
                base = root / f"{dataset}_repeat_{repeat}/full"
                groups = {
                    "refinement": list((base / f"refinement/{dataset}").glob("*/summary.json")),
                    "confidence": list((base / f"confidence/{dataset}").glob("*/summary.json")),
                    "sampling": list((base / "sampling").glob("*.summary.json")),
                    "full_shard": list((base / "shards").glob("*.json")),
                }
                for stage, paths in groups.items():
                    shard_count = 16 if arm == "temporal_n100_s10" else 8
                    expected = count if stage in ("refinement", "confidence") else shard_count
                    if len(paths) != expected:
                        raise ValueError(
                            f"{arm}/{dataset}/{repeat}/{stage}: {len(paths)} != {expected}"
                        )
                    metrics = {}
                    devices = set()
                    completed = 0
                    elapsed = 0
                    for path in sorted(paths):
                        content = path.read_bytes()
                        d = json.loads(content)
                        rt = d["runtime"]
                        if stage in ("sampling", "full_shard") and d["num_shards"] != shard_count:
                            raise ValueError("registered shard count mismatch")
                        sources.append(
                            dict(
                                path=str(path.relative_to(ROOT)),
                                sha256=hashlib.sha256(content).hexdigest(),
                            )
                        )
                        device = rt.get("cuda_device_name", rt.get("gpu"))
                        if device:
                            devices.add(device)
                        for k, v in rt.items():
                            if k in (
                                "elapsed_seconds",
                                "wall_seconds_before_summary_write",
                                "cuda_max_memory_allocated_bytes",
                                "cuda_max_memory_reserved_bytes",
                                "gpu_total_memory_bytes",
                            ):
                                metrics.setdefault(k, []).append(v)
                        for k, v in rt.get("stage_seconds", {}).items():
                            if isinstance(v, (int, float)):
                                metrics.setdefault("stage_seconds." + k, []).append(v)
                            elif isinstance(v, dict):
                                for sub, value in v.items():
                                    metrics.setdefault("stage_seconds." + k + "." + sub, []).append(
                                        value
                                    )
                        if stage == "full_shard":
                            completed += d["num_completed"]
                            elapsed += rt["elapsed_seconds"]
                        if stage == "sampling" and d["num_failed"] != 0:
                            raise ValueError("sampling failures")
                    row = dict(
                        arm=arm,
                        dataset=dataset,
                        repeat=repeat,
                        stage=stage,
                        records=len(paths),
                        complexes=count,
                        devices=sorted(devices),
                        metrics={k: stats(v) for k, v in metrics.items()},
                    )
                    if stage == "full_shard":
                        if completed != count:
                            raise ValueError("shard complex count mismatch")
                        row["aggregate_shard_wall_seconds_per_complex"] = elapsed / completed
                    rows.append(row)
    return dict(
        rows=rows,
        sources=sources,
        notes=[
            "Sampling-only wall time was not recorded; no subtraction-based estimate is called sampling time.",
            "Full-shard elapsed includes sampling, refinement, both raw/refined confidence, imports and I/O; excludes separate official selected-pose PB jobs.",
            "Per-complex wall is a descriptive throughput normalization, not isolated kernel latency.",
            "Confidence forward totals score BOTH raw and refined banks; per-stage forward fields are separate.",
            "CUDA allocator peaks are per sampling process/shard, not per ligand or whole-process/nvidia-smi peaks.",
            "Host RSS and refinement/confidence GPU peaks are unavailable in these records; requested RAM is not measured usage.",
            "Guided N100 historical pipeline runtime is not mixed with the matched current driver comparison.",
        ],
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=ROOT / "docs/paper/20260919")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    result = collect()
    evidence = B / "paper_analysis_20260919"
    evidence.mkdir(exist_ok=True)
    manifest = evidence / "runtime_sources.json"
    manifest.write_text(json.dumps(result.pop("sources"), indent=2))
    result["source_manifest"] = dict(
        path=str(manifest.relative_to(ROOT)),
        sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
    )
    (args.output / "runtime_metrics.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(dict(status="complete", rows=len(result["rows"]))))


if __name__ == "__main__":
    main()
