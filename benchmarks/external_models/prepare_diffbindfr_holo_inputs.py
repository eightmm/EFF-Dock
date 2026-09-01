#!/usr/bin/env python3
"""Prepare DiffBindFR paper-style holo redocking manifests and fixed shards."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

FIELDS = (
    "protein",
    "protein_name",
    "ligand",
    "ligand_name",
    "complex_name",
    "crystal_ligand",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-csv", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--expected-size", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    with args.source_csv.open(newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    required = {"complex_name", "experimental_protein", "ligand"}
    if not source_rows or not required.issubset(source_rows[0]):
        raise ValueError(f"source CSV requires {sorted(required)}")
    if len(source_rows) != args.expected_size:
        raise ValueError(f"expected {args.expected_size} rows, found {len(source_rows)}")
    if len({row["complex_name"] for row in source_rows}) != args.expected_size:
        raise ValueError("complex_name values are not unique")

    rows = []
    for row in source_rows:
        target = row["complex_name"]
        protein = Path(row["experimental_protein"]).resolve()
        ligand = Path(row["ligand"]).resolve()
        if not protein.is_file() or not ligand.is_file():
            raise FileNotFoundError(f"missing input for {target}: {protein}, {ligand}")
        rows.append(
            {
                "protein": str(protein),
                "protein_name": target,
                "ligand": str(ligand),
                "ligand_name": target,
                "complex_name": target,
                "crystal_ligand": str(ligand),
            }
        )

    output_root = args.output_root.resolve()
    write_csv(output_root / f"{args.dataset}_full.csv", rows)
    write_csv(output_root / f"{args.dataset}_smoke.csv", rows[:2])
    shard_sizes = {}
    for shard_index in range(args.num_shards):
        shard_rows = [
            row for index, row in enumerate(rows) if index % args.num_shards == shard_index
        ]
        shard_name = f"shard_{shard_index:03d}"
        write_csv(output_root / "shards" / shard_name / "diffbindfr_inputs.csv", shard_rows)
        shard_sizes[shard_name] = len(shard_rows)
    provenance = {
        "schema_version": 1,
        "dataset": args.dataset,
        "source_csv": str(args.source_csv.resolve()),
        "receptor_policy": "fixed holo receptor (DiffBindFR paper redocking protocol)",
        "site_policy": "cognate crystal ligand; upstream default 12 A pocket radius",
        "target_count": len(rows),
        "num_shards": args.num_shards,
        "shard_sizes": shard_sizes,
    }
    (output_root / f"{args.dataset}_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    print(f"Prepared DiffBindFR holo inputs: {args.dataset} {len(rows)}/{args.expected_size}")


if __name__ == "__main__":
    main()
