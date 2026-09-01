#!/usr/bin/env python3
"""Localize PoseBench's released DiffDock input manifests.

The pinned PoseBench checkout contains the exact Astex Diverse and PoseBusters
Benchmark CSV rows used by its DiffDock wrapper, but their protein paths are
absolute paths from the upstream authors' machine.  This script preserves every
other field and rewrites only those paths to the local benchmark mirror.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from rdkit import Chem

DATASET_TO_SIZE = {
    "astex_diverse": 85,
    "posebusters_benchmark": 308,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--posebench-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--dataset", choices=sorted(DATASET_TO_SIZE), required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument(
        "--ligand-description-source",
        choices=("released_smiles", "reference_sdf", "reference_smiles"),
        default="released_smiles",
        help=(
            "Keep PoseBench's released multi-component SMILES, pass the reference "
            "SDF path, or derive an isomeric SMILES from the frozen primary ligand."
        ),
    )
    parser.add_argument("--num-shards", type=int)
    parser.add_argument("--shard-index", type=int)
    parser.add_argument(
        "--target-id",
        action="append",
        default=[],
        help="Emit only this target ID; may be repeated. The default emits the full cohort.",
    )
    return parser.parse_args()


def localize_manifest(
    *,
    posebench_root: Path,
    data_root: Path,
    dataset: str,
    target_ids: list[str],
    ligand_description_source: str = "released_smiles",
) -> tuple[list[dict[str, str]], Path]:
    source_csv = (
        posebench_root / "forks" / "DiffDock" / "inference" / f"diffdock_{dataset}_inputs.csv"
    )
    dataset_root = data_root / f"{dataset}_set"
    protein_root = dataset_root / f"{dataset}_holo_aligned_predicted_structures"
    requested = set(target_ids)

    if not source_csv.is_file():
        raise FileNotFoundError(f"PoseBench input manifest not found: {source_csv}")
    if not protein_root.is_dir():
        raise FileNotFoundError(f"Predicted-receptor directory not found: {protein_root}")

    with source_csv.open(newline="") as handle:
        reader = csv.DictReader(handle)
        expected_fields = [
            "complex_name",
            "protein_path",
            "ligand_description",
            "protein_sequence",
        ]
        if reader.fieldnames != expected_fields:
            raise ValueError(f"Unexpected PoseBench columns in {source_csv}: {reader.fieldnames}")
        rows = list(reader)

    if len(rows) != DATASET_TO_SIZE[dataset]:
        raise ValueError(
            f"Released {dataset} manifest has {len(rows)} rows; expected {DATASET_TO_SIZE[dataset]}"
        )

    source_ids = {row["complex_name"] for row in rows}
    missing_ids = requested - source_ids
    if missing_ids:
        raise ValueError(
            f"Requested target IDs absent from released manifest: {sorted(missing_ids)}"
        )

    localized: list[dict[str, str]] = []
    for row in rows:
        target_id = row["complex_name"]
        if requested and target_id not in requested:
            continue
        protein_path = protein_root / Path(row["protein_path"]).name
        ligand_path = dataset_root / target_id / f"{target_id}_ligand.sdf"
        if not protein_path.is_file():
            raise FileNotFoundError(f"Missing receptor for {target_id}: {protein_path}")
        if not ligand_path.is_file():
            raise FileNotFoundError(f"Missing reference ligand for {target_id}: {ligand_path}")
        localized_row = {**row, "protein_path": str(protein_path.resolve())}
        if ligand_description_source == "reference_sdf":
            localized_row["ligand_description"] = str(ligand_path.resolve())
        elif ligand_description_source == "reference_smiles":
            ligand = Chem.MolFromMolFile(
                str(ligand_path), sanitize=True, removeHs=True, strictParsing=False
            )
            if ligand is None:
                raise ValueError(f"RDKit could not parse reference ligand for {target_id}")
            fragments = Chem.GetMolFrags(ligand)
            if len(fragments) != 1:
                raise ValueError(
                    f"Reference ligand for {target_id} has {len(fragments)} components; "
                    "primary-only SMILES would be ambiguous"
                )
            localized_row["ligand_description"] = Chem.MolToSmiles(
                Chem.RemoveAllHs(ligand), isomericSmiles=True, canonical=True
            )
        localized.append(localized_row)

    if requested and len(localized) != len(requested):
        raise RuntimeError(f"Localized {len(localized)} requested rows, expected {len(requested)}")
    return localized, source_csv


def main() -> None:
    args = parse_args()
    if (args.num_shards is None) != (args.shard_index is None):
        raise ValueError("--num-shards and --shard-index must be provided together")
    if args.num_shards is not None and not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must satisfy 0 <= index < --num-shards")
    posebench_root = args.posebench_root.resolve()
    data_root = args.data_root.resolve()
    rows, source_csv = localize_manifest(
        posebench_root=posebench_root,
        data_root=data_root,
        dataset=args.dataset,
        target_ids=args.target_id,
        ligand_description_source=args.ligand_description_source,
    )
    if args.num_shards is not None:
        rows = [
            row
            for index, row in enumerate(rows)
            if index % args.num_shards == args.shard_index
        ]

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "complex_name",
        "protein_path",
        "ligand_description",
        "protein_sequence",
    ]
    with args.output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    provenance = {
        "schema_version": 1,
        "dataset": args.dataset,
        "expected_full_denominator": DATASET_TO_SIZE[args.dataset],
        "emitted_rows": len(rows),
        "target_ids": [row["complex_name"] for row in rows],
        "released_source_csv": str(source_csv.resolve()),
        "local_data_root": str(data_root),
        "path_rewrite_only": args.ligand_description_source == "released_smiles",
        "ligand_description_source": args.ligand_description_source,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
    }
    provenance_path = args.output_csv.with_suffix(".json")
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Wrote {len(rows)} rows to {args.output_csv}")
    print(f"Wrote provenance to {provenance_path}")


if __name__ == "__main__":
    main()
