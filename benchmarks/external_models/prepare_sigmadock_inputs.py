#!/usr/bin/env python3
"""Convert the frozen common pocket benchmark CSV to SigmaDock's datafront."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--expected-size", type=int, required=True)
    parser.add_argument(
        "--receptor-policy",
        default="source CSV experimental_protein",
        help="Human-readable provenance written to the sidecar JSON.",
    )
    parser.add_argument(
        "--site-policy",
        default="cognate crystal ligand defines the supplied pocket",
        help="Human-readable provenance written to the sidecar JSON.",
    )
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.source_csv.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"complex_name", "experimental_protein", "ligand"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"source CSV requires columns {sorted(required)}")
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        rows = rows[: args.limit]
    if len(rows) != args.expected_size:
        raise ValueError(
            f"{args.dataset}: expected {args.expected_size} rows, found {len(rows)}"
        )

    seen: set[str] = set()
    records: list[dict[str, str]] = []
    for row in rows:
        target_id = row["complex_name"].strip()
        if not target_id or target_id in seen:
            raise ValueError(f"missing or duplicate complex_name: {target_id!r}")
        seen.add(target_id)
        protein = Path(row["experimental_protein"]).expanduser().resolve()
        ligand = Path(row["ligand"]).expanduser().resolve()
        if not protein.is_file():
            raise FileNotFoundError(protein)
        if not ligand.is_file():
            raise FileNotFoundError(ligand)
        protein_target = "_".join(protein.stem.split("_")[:2])
        if protein_target.upper() != target_id.upper():
            raise ValueError(
                f"protein/target identity mismatch: {target_id} vs {protein.name}"
            )
        records.append(
            {
                "complex_name": target_id,
                "PDB": str(protein),
                "SDF": str(ligand),
                "protein_sha256": _sha256(protein),
                "ligand_sha256": _sha256(ligand),
            }
        )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["PDB", "SDF"])
        writer.writeheader()
        writer.writerows({"PDB": row["PDB"], "SDF": row["SDF"]} for row in records)

    payload = {
        "schema_version": 1,
        "dataset": args.dataset,
        "source_csv": str(args.source_csv.resolve()),
        "source_csv_sha256": _sha256(args.source_csv),
        "output_csv": str(args.output_csv.resolve()),
        "output_csv_sha256": _sha256(args.output_csv),
        "receptor_policy": args.receptor_policy,
        "site_policy": args.site_policy,
        "target_count": len(records),
        "records": records,
    }
    args.output_csv.with_suffix(".json").write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Prepared SigmaDock inputs: {len(records)}/{args.expected_size}")


if __name__ == "__main__":
    main()
