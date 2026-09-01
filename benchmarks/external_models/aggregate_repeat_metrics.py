#!/usr/bin/env python3
"""Aggregate fixed repeated benchmark summaries as mean and sample standard deviation."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--label", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.input) != 3:
        raise ValueError(f"exactly three fixed repeats are required, found {len(args.input)}")
    repeats = [json.loads(path.read_text()) for path in args.input]
    denominators = {int(item["denominator"]) for item in repeats}
    if len(denominators) != 1:
        raise ValueError(f"repeat denominators differ: {sorted(denominators)}")

    common_keys = set.intersection(*(set(item) for item in repeats))
    metric_keys = sorted(
        key
        for key in common_keys
        if key.endswith("_pct")
        and all(
            isinstance(item[key], (int, float)) and math.isfinite(float(item[key]))
            for item in repeats
        )
    )
    if not metric_keys:
        raise ValueError("no common finite *_pct metrics found")
    metrics = {}
    for key in metric_keys:
        values = [float(item[key]) for item in repeats]
        metrics[key] = {
            "values": values,
            "mean": statistics.mean(values),
            "std": statistics.stdev(values),
            "std_definition": "sample standard deviation (ddof=1)",
        }
    payload = {
        "schema_version": 1,
        "label": args.label,
        "repeat_count": 3,
        "denominator": denominators.pop(),
        "input_summaries": [str(path.resolve()) for path in args.input],
        "metrics": metrics,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n")

    lines = [
        f"# {args.label}",
        "",
        f"Three fixed repeats; N={payload['denominator']}; mean ± sample SD (ddof=1).",
        "",
        "| Metric | Repeat 1 | Repeat 2 | Repeat 3 | Mean ± SD |",
        "|---|---:|---:|---:|---:|",
    ]
    for key, item in metrics.items():
        values = item["values"]
        lines.append(
            f"| `{key}` | {values[0]:.2f} | {values[1]:.2f} | {values[2]:.2f} | "
            f"{item['mean']:.2f} ± {item['std']:.2f} |"
        )
    args.output_md.write_text("\n".join(lines) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
