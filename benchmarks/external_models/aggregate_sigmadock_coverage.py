#!/usr/bin/env python3
"""Aggregate all independent SigmaDock seeds without dropping failures."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--expected-seeds", type=int, required=True)
    parser.add_argument("--expected-targets", type=int, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = sorted(args.run_root.glob("**/seed_*/coverage.json"))
    by_seed: dict[int, dict] = {}
    duplicates: list[int] = []
    for path in files:
        match = re.fullmatch(r"seed_(\d+)", path.parent.name)
        if match is None:
            continue
        seed = int(match.group(1))
        if seed in by_seed:
            duplicates.append(seed)
            continue
        payload = json.loads(path.read_text())
        by_seed[seed] = {
            "coverage_path": str(path.resolve()),
            "complete_targets": payload.get("complete_targets", 0),
            "expected_targets": payload.get("expected_targets"),
            "vinardo_failures": len(payload.get("vinardo_failures", [])),
        }

    expected_seed_ids = set(range(args.expected_seeds))
    missing_seeds = sorted(expected_seed_ids - set(by_seed))
    incomplete_seeds = sorted(
        seed
        for seed, payload in by_seed.items()
        if payload["complete_targets"] != args.expected_targets
        or payload["expected_targets"] != args.expected_targets
        or payload["vinardo_failures"]
    )
    summary = {
        "schema_version": 1,
        "run_root": str(args.run_root.resolve()),
        "expected_seeds": args.expected_seeds,
        "observed_seeds": len(set(by_seed) & expected_seed_ids),
        "expected_targets_per_seed": args.expected_targets,
        "expected_total_poses": args.expected_seeds * args.expected_targets,
        "complete_total_poses": sum(
            min(payload["complete_targets"], args.expected_targets)
            for seed, payload in by_seed.items()
            if seed in expected_seed_ids
        ),
        "missing_seeds": missing_seeds,
        "unexpected_seeds": sorted(set(by_seed) - expected_seed_ids),
        "duplicate_seeds": sorted(set(duplicates)),
        "incomplete_seeds": incomplete_seeds,
        "seeds": {str(seed): payload for seed, payload in sorted(by_seed.items())},
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2) + "\n")
    print(
        f"SigmaDock aggregate: {summary['observed_seeds']}/{args.expected_seeds} "
        f"seeds, {summary['complete_total_poses']}/{summary['expected_total_poses']} poses"
    )
    if any(
        (
            missing_seeds,
            summary["unexpected_seeds"],
            summary["duplicate_seeds"],
            incomplete_seeds,
        )
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
