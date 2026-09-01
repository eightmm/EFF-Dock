#!/usr/bin/env python3
"""Create isolated Vina shards for targets without a completed SDF."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def completed_targets(run_roots: list[Path]) -> set[str]:
    completed: set[str] = set()
    for root in run_roots:
        if not root.is_dir():
            continue
        for path in root.rglob("*.sdf"):
            if path.parent.name == path.stem and path.stat().st_size > 0:
                completed.add(path.stem)
    return completed


def write_recovery_shards(
    *, input_csv: Path, run_roots: list[Path], output_root: Path, num_shards: int
) -> dict[str, object]:
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    with input_csv.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "complex_name" not in reader.fieldnames:
            raise ValueError("input CSV must contain complex_name")
        fieldnames = reader.fieldnames
        rows = list(reader)

    done = completed_targets(run_roots)
    remaining = [row for row in rows if row["complex_name"] not in done]
    if not remaining:
        raise ValueError("all targets already have a completed SDF")
    num_shards = min(num_shards, len(remaining))

    output_root.mkdir(parents=True, exist_ok=True)
    shard_sizes: dict[str, int] = {}
    for shard_index in range(num_shards):
        shard_rows = remaining[shard_index::num_shards]
        shard_name = f"shard_{shard_index:03d}"
        shard_dir = output_root / shard_name
        shard_dir.mkdir(parents=True, exist_ok=True)
        with (shard_dir / "vina_inputs.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            writer.writerows(shard_rows)
        shard_sizes[shard_name] = len(shard_rows)

    summary: dict[str, object] = {
        "schema_version": 1,
        "input_csv": str(input_csv.resolve()),
        "existing_run_roots": [str(path.resolve()) for path in run_roots],
        "full_denominator": len(rows),
        "completed_at_snapshot": len(done & {row["complex_name"] for row in rows}),
        "remaining_at_snapshot": len(remaining),
        "remaining_target_ids": [row["complex_name"] for row in remaining],
        "num_shards": num_shards,
        "shard_sizes": shard_sizes,
    }
    (output_root / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--existing-run-root", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    args = parser.parse_args()
    summary = write_recovery_shards(
        input_csv=args.input_csv,
        run_roots=args.existing_run_root,
        output_root=args.output_root,
        num_shards=args.num_shards,
    )
    print(
        f"Prepared {summary['remaining_at_snapshot']} remaining targets in "
        f"{summary['num_shards']} shards"
    )


if __name__ == "__main__":
    main()
