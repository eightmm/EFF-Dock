#!/usr/bin/env python3
"""Run SigmaDock's pinned PoseBusters redock checks for one sampled seed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import posebusters
import torch
from sigmadock.chem.statistics import compact_posebusting


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--expected-targets", type=int, required=True)
    parser.add_argument("--config", default="redock", choices=("redock",))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    loaded = torch.load(args.prediction_path, map_location="cpu", weights_only=False)
    results = loaded.get("results")
    if not isinstance(results, dict):
        raise TypeError("predictions.pt has no results dictionary")
    if len(results) != args.expected_targets:
        raise ValueError(
            f"expected {args.expected_targets} targets, found {len(results)}"
        )

    rmsds, pb_checks, pb_dicts = compact_posebusting(results, config=args.config)
    observed = set(rmsds) & set(pb_checks) & set(pb_dicts)
    if len(observed) != args.expected_targets:
        raise ValueError(
            f"PoseBusters completed {len(observed)}/{args.expected_targets} targets"
        )
    payload = {
        "rmsds": rmsds,
        "pb_checks": pb_checks,
        "pb_dicts": pb_dicts,
        "metadata": {
            "schema_version": 1,
            "prediction_path": str(args.prediction_path.resolve()),
            "posebusters_version": posebusters.__version__,
            "config": args.config,
            "expected_targets": args.expected_targets,
        },
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output_path)
    args.output_path.with_suffix(".json").write_text(
        json.dumps(payload["metadata"], indent=2) + "\n"
    )
    print(
        f"SigmaDock PoseBusters: {len(observed)}/{args.expected_targets}; "
        f"version={posebusters.__version__}; config={args.config}"
    )


if __name__ == "__main__":
    main()
