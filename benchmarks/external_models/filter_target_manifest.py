#!/usr/bin/env python3
"""Create an auditable target-only CSV from a frozen external-model manifest."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--target-id", action="append", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    requested = list(dict.fromkeys(args.target_id))
    with args.input_csv.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "complex_name" not in reader.fieldnames:
            raise ValueError("Input CSV must contain complex_name")
        fieldnames = reader.fieldnames
        rows = list(reader)
    by_id = {row["complex_name"]: row for row in rows}
    missing = [target_id for target_id in requested if target_id not in by_id]
    if missing:
        raise ValueError(f"Targets absent from input manifest: {missing}")

    selected = [by_id[target_id] for target_id in requested]
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(selected)
    provenance = {
        "schema_version": 1,
        "source_csv": str(args.input_csv.resolve()),
        "output_csv": str(args.output_csv.resolve()),
        "target_ids": requested,
        "row_count": len(selected),
        "selection_only": True,
    }
    args.output_csv.with_suffix(".json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    print(f"Wrote {len(selected)} targets to {args.output_csv}")


if __name__ == "__main__":
    main()
