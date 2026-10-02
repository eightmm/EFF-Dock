"""Aggregate completed corrected-Rw shard logs without rerunning inference."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

COUNTS = {"astex": 85, "posebusters": 308, "phibench": 206, "foldbench": 558, "openbind": 925}
STUDY = Path("outputs/benchmarks/rw_five_benchmarks_r3_20260929")


def verify(data):
    if data["orientation_injection"] != "rw" or [r["dataset"] for r in data["rows"]] != list(
        COUNTS
    ):
        raise ValueError("Wrong operator or runtime cohort inventory")
    for row in data["rows"]:
        count = COUNTS[row["dataset"]]
        repeats = row["repeats"]
        if (
            row["orientation_injection"] != "rw"
            or row["num_samples"] != 100
            or row["num_steps"] != 10
            or row["count"] != count
            or [r["seed"] for r in repeats] != [42, 100042, 200042]
        ):
            raise ValueError("Runtime denominator or seed drift")
        for r in repeats:
            if (
                r["count"] != count
                or sum(r["device_case_counts"].values()) != count
                or not np.isclose(
                    r["seconds_per_complex"], r["pipeline_seconds"] / count, atol=1e-9, rtol=0
                )
            ):
                raise ValueError("Runtime aggregation mismatch")
        values = np.array([r["seconds_per_complex"] for r in repeats])
        if (
            not np.isfinite(values).all()
            or (values <= 0).any()
            or not np.isfinite(row["generation_process_allocator_peak_gib"])
            or row["generation_process_allocator_peak_gib"] <= 0
            or not np.isclose(row["pipeline_mean_s"], values.mean(), atol=1e-9, rtol=0)
            or not np.isclose(row["pipeline_repeat_sd_s"], values.std(ddof=1), atol=1e-9, rtol=0)
            or row["generation_process_allocator_peak_gib"]
            != max(r["generation_process_allocator_peak_gib"] for r in repeats)
            or sum(row["device_case_counts"].values()) != 3 * count
        ):
            raise ValueError("Runtime mean, sample SD or sampling peak mismatch")


def collect(root):
    sources = {}

    def read(path, expected=None):
        path = Path(path)
        if not path.is_absolute():
            path = root / path
        blob = path.read_bytes()
        sha = hashlib.sha256(blob).hexdigest()
        if expected is not None and sha != expected:
            raise ValueError(f"Source checksum mismatch: {path.name}")
        sources[path.relative_to(root).as_posix()] = sha
        return json.loads(blob)

    rows = []
    for dataset, count in COUNTS.items():
        repeats, device_counts = [], Counter()
        peaks = []
        cohort_ids = None
        profiles = set()
        for repeat in range(3):
            base = STUDY / f"runs/rw/{dataset}_repeat_{repeat}/full"
            elapsed, ids, sampling_ids = 0.0, [], []
            repeat_devices = Counter()
            repeat_peaks = []
            for shard in range(16):
                name = f"{dataset}.shard-{shard:03d}-of-016.json"
                run = read(base / "shards" / name)
                manifest = read(base / "manifests" / name)
                if (
                    run["orientation_injection"] != "rw"
                    or run["expected_dataset_count"] != count
                    or run["num_completed"] != len(run["records"])
                    or run["sampling"]["num_samples"] != 100
                    or run["sampling"]["num_steps"] != 10
                    or run["sampling"]["guidance_eta"] != 0
                    or run["status"] != "complete"
                ):
                    raise ValueError(f"Incomplete or incompatible pipeline shard: {name}")
                seconds = run["runtime"]["elapsed_seconds"]
                if not np.isfinite(seconds) or seconds <= 0:
                    raise ValueError(f"Invalid pipeline timing: {name}")
                elapsed += seconds
                ids.extend(r["id"] for r in run["records"])
                if len(manifest["source_summaries"]) != 1:
                    raise ValueError(f"Ambiguous sampling provenance: {name}")
                source = manifest["source_summaries"][0]
                sample = read(source["path"], source["sha256"])
                profiles.add((sample["selector_profile"], sample["confidence_scoring_enabled"]))
                if (
                    sample["orientation_injection"] != "rw"
                    or sample["num_failed"] != 0
                    or sample["num_success"] != run["num_completed"]
                    or sample["num_assigned"] != run["num_completed"]
                    or sample["num_samples"] != 100
                    or sample["num_steps"] != 10
                    or sample["seed"] != 42 + repeat * 100000
                    or sample["sigma"] != 2.0
                    or sample["pocket_cutoff"] != 10.0
                    or sample["vina_guidance_scale"] != 0.0
                    or sample["checkpoint_sha256"]
                    != "65be44d7dc8f0867eb9fc5d22214b80f93971ea4702679a527c665046e91e6b6"
                ):
                    raise ValueError(f"Incomplete or incompatible sampling shard: {name}")
                sampling_ids.extend(r["id"] for r in manifest["records"])
                peak = sample["runtime"]["cuda_max_memory_allocated_bytes"]
                if not isinstance(peak, int) or peak <= 0:
                    raise ValueError(f"Invalid sampling allocator peak: {name}")
                repeat_peaks.append(peak / 1024**3)
                repeat_devices[sample["runtime"]["gpu"]] += run["num_completed"]
            if len(ids) != count or len(set(ids)) != count or set(ids) != set(sampling_ids):
                raise ValueError(f"Nonunique or incomplete inventory: {dataset} repeat {repeat}")
            if cohort_ids is not None and set(ids) != cohort_ids:
                raise ValueError(f"Runtime cohort membership changed between seeds: {dataset}")
            cohort_ids = set(ids)
            repeats.append(
                dict(
                    repeat=repeat,
                    seed=42 + repeat * 100000,
                    count=count,
                    pipeline_seconds=elapsed,
                    seconds_per_complex=elapsed / count,
                    generation_process_allocator_peak_gib=max(repeat_peaks),
                    device_case_counts=dict(sorted(repeat_devices.items())),
                )
            )
            device_counts.update(repeat_devices)
            peaks.extend(repeat_peaks)
        times = [r["seconds_per_complex"] for r in repeats]
        expected_profile = (
            ("candidate_only", False)
            if dataset in ("astex", "posebusters")
            else ("confidence_cluster_free", True)
        )
        if profiles != {expected_profile}:
            raise ValueError(f"Unexpected initial evaluation-process profile: {dataset}")
        rows.append(
            dict(
                dataset=dataset,
                count=count,
                orientation_injection="rw",
                num_samples=100,
                num_steps=10,
                generation_process_profile=expected_profile[0],
                raw_confidence_scoring_in_generation_process=expected_profile[1],
                pipeline_mean_s=float(np.mean(times)),
                pipeline_repeat_sd_s=float(np.std(times, ddof=1)),
                generation_process_allocator_peak_gib=max(peaks),
                device_case_counts=dict(sorted(device_counts.items())),
                repeats=repeats,
            )
        )
    return dict(
        orientation_injection="rw",
        rows=rows,
        runtime_definition="Sum of complete pipeline shard wall durations per complex; mean and sample SD over three seeds. Includes setup, generation, refinement, confidence and I/O; excludes separately executed official PB.",
        memory_definition="Maximum initial generation/evaluation-process CUDA allocated peak across 16 shards and three seeds. Astex/PB use candidate_only generation; Phi/Fold/Open use confidence_cluster_free and include raw-confidence scoring in this process. Excludes separate refinement/post-refinement scoring processes. Not reserved memory or whole-pipeline peak.",
        hardware_note="Recorded mixed GPU backends. Device case counts are retained; these descriptive costs are not a controlled cross-hardware speed comparison or saturated service throughput.",
        sources=[dict(path=p, sha256=s) for p, s in sorted(sources.items())],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = collect(args.source_root.resolve())
    verify(data)
    args.output.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    for row in data["rows"]:
        print(
            row["dataset"],
            row["count"],
            row["pipeline_mean_s"],
            row["pipeline_repeat_sd_s"],
            row["generation_process_allocator_peak_gib"],
        )


if __name__ == "__main__":
    main()
