#!/usr/bin/env python3
"""Fail closed on one native SigmaDock seed's pose and Vinardo inventory."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--prediction-path", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--expected-targets", type=int, required=True)
    parser.add_argument("--require-vinardo", action="store_true")
    return parser.parse_args()


def _expected(rows: list[dict[str, str]]) -> dict[str, str]:
    expected: dict[str, str] = {}
    for row in rows:
        protein = Path(row["PDB"])
        ligand = Path(row["SDF"])
        target_id = "_".join(protein.stem.split("_")[:2])
        key = f"{target_id}::{ligand.stem}"
        if key in expected:
            raise ValueError(f"duplicate expected SigmaDock key: {key}")
        expected[key] = target_id
    return expected


def main() -> None:
    args = parse_args()
    with args.input_csv.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != args.expected_targets:
        raise ValueError(
            f"expected {args.expected_targets} input rows, found {len(rows)}"
        )
    expected = _expected(rows)
    if not args.prediction_path.is_file():
        raise FileNotFoundError(args.prediction_path)
    loaded = torch.load(args.prediction_path, map_location="cpu", weights_only=False)
    results = loaded.get("results")
    if not isinstance(results, dict):
        raise TypeError("SigmaDock predictions.pt has no result dictionary")

    observed = set(results)
    expected_keys = set(expected)
    coverage: dict[str, dict[str, int | bool]] = {}
    for key, target_id in expected.items():
        samples = results.get(key, [])
        finite = False
        if len(samples) == 1 and isinstance(samples[0], dict):
            coords = samples[0].get("x0_hat")
            finite = bool(
                torch.is_tensor(coords)
                and coords.ndim == 2
                and coords.shape[-1] == 3
                and coords.numel() > 0
                and torch.isfinite(coords).all().item()
            )
        coverage[target_id] = {
            "pose_count": len(samples),
            "finite_coordinates": finite,
            "vinardo_score_count": 0,
        }

    rescoring_path = args.prediction_path.with_name("rescoring.pt")
    failed_scores: list = []
    if args.require_vinardo:
        if not rescoring_path.is_file():
            raise FileNotFoundError(rescoring_path)
        rescoring = torch.load(rescoring_path, map_location="cpu", weights_only=False)
        scores = rescoring.get("scores")
        failed_scores = rescoring.get("failed", [])
        if not isinstance(scores, dict):
            raise TypeError("SigmaDock rescoring.pt has no score dictionary")
        for key, target_id in expected.items():
            entries = scores.get(key, [])
            valid = [
                item
                for item in entries
                if isinstance(item, dict) and item.get("Affinity") is not None
            ]
            coverage[target_id]["vinardo_score_count"] = len(valid)

    missing = sorted(expected_keys - observed)
    unexpected = sorted(observed - expected_keys)
    complete_targets = sum(
        item["pose_count"] == 1
        and item["finite_coordinates"]
        and (not args.require_vinardo or item["vinardo_score_count"] == 1)
        for item in coverage.values()
    )
    summary = {
        "schema_version": 1,
        "input_csv": str(args.input_csv.resolve()),
        "prediction_path": str(args.prediction_path.resolve()),
        "rescoring_path": str(rescoring_path.resolve()) if args.require_vinardo else None,
        "expected_targets": args.expected_targets,
        "observed_targets": len(observed & expected_keys),
        "complete_targets": complete_targets,
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "vinardo_failures": failed_scores,
        "coverage": coverage,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(
        f"SigmaDock coverage: {complete_targets}/{args.expected_targets}; "
        f"missing={len(missing)} unexpected={len(unexpected)} "
        f"vinardo_failures={len(failed_scores)}"
    )
    if (
        complete_targets != args.expected_targets
        or missing
        or unexpected
        or failed_scores
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
