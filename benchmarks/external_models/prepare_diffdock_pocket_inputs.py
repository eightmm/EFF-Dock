#!/usr/bin/env python3
"""Prepare explicit, auditable receptor inputs for DiffDock-Pocket.

The final benchmark path uses PoseBench's holo-aligned predicted receptor,
which matches DiffDock-Pocket's documented computational-structure use case.
The ``protein_only`` mode is retained only for compatibility diagnostics: it
keeps holo ATOM coordinates and removes HETATM before the upstream pre-graph
step that otherwise rejects common cofactors.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

DATASET_TO_SIZE = {
    "astex_diverse": 85,
    "posebusters_benchmark": 308,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-csv", type=Path, required=True)
    parser.add_argument("--dataset", choices=sorted(DATASET_TO_SIZE), required=True)
    parser.add_argument(
        "--receptor-mode",
        choices=[
            "protein_only",
            "holo_aligned_predicted",
            "holo_aligned_predicted_pocket_crop",
        ],
        default="protein_only",
    )
    parser.add_argument("--protein-output-dir", type=Path)
    parser.add_argument("--predicted-protein-dir", type=Path)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--num-shards", type=int)
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--target-id", action="append", default=[])
    return parser.parse_args()


def write_protein_only(source: Path, destination: Path) -> tuple[int, int]:
    atom_lines: list[str] = []
    hetero_count = 0
    for line in source.read_text(errors="replace").splitlines():
        record = line[:6].strip()
        if record == "ATOM":
            atom_lines.append(line)
        elif record == "HETATM":
            hetero_count += 1
        elif record == "TER" and atom_lines:
            atom_lines.append(line)
    atom_count = sum(line.startswith("ATOM  ") for line in atom_lines)
    if atom_count == 0:
        raise ValueError(f"Protein has no ATOM records: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join([*atom_lines, "END"]) + "\n")
    return atom_count, hetero_count


def main() -> None:
    args = parse_args()
    if args.receptor_mode == "protein_only" and args.protein_output_dir is None:
        raise ValueError("--protein-output-dir is required for protein_only mode")
    if args.receptor_mode.startswith("holo_aligned_predicted") and args.predicted_protein_dir is None:
        raise ValueError(
            "--predicted-protein-dir is required for predicted-receptor modes"
        )
    if (
        args.receptor_mode == "holo_aligned_predicted_pocket_crop"
        and args.protein_output_dir is None
    ):
        raise ValueError(
            "--protein-output-dir is required for "
            "holo_aligned_predicted_pocket_crop mode"
        )
    if (args.num_shards is None) != (args.shard_index is None):
        raise ValueError("--num-shards and --shard-index must be provided together")
    if args.num_shards is not None and not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must satisfy 0 <= index < --num-shards")

    with args.source_csv.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected_size = DATASET_TO_SIZE[args.dataset]
    if len(rows) != expected_size:
        raise ValueError(
            f"Source {args.dataset} manifest has {len(rows)} rows; expected {expected_size}"
        )
    if len({row["complex_name"] for row in rows}) != expected_size:
        raise ValueError("Source manifest contains duplicate complex_name values")

    requested = set(args.target_id)
    missing = requested - {row["complex_name"] for row in rows}
    if missing:
        raise ValueError(f"Unknown target IDs: {sorted(missing)}")

    selected = [row for row in rows if not requested or row["complex_name"] in requested]
    if args.num_shards is not None:
        selected = [
            row
            for index, row in enumerate(selected)
            if index % args.num_shards == args.shard_index
        ]

    emitted: list[dict[str, str]] = []
    protein_stats: dict[str, dict[str, object]] = {}
    for row in selected:
        target_id = row["complex_name"]
        crop_stats: dict[str, object] | None = None
        if args.receptor_mode == "protein_only":
            source_protein = Path(row["experimental_protein"]).resolve()
            destination = (
                args.protein_output_dir.resolve() / f"{target_id}_protein_only.pdb"
            )
            atom_count, hetero_count = write_protein_only(source_protein, destination)
        else:
            source_predicted = (
                args.predicted_protein_dir.resolve()
                / f"{target_id}_holo_aligned_predicted_protein.pdb"
            )
            if not source_predicted.is_file():
                raise FileNotFoundError(
                    f"Missing predicted receptor: {source_predicted}"
                )
            destination = source_predicted
            lines = source_predicted.read_text(errors="replace").splitlines()
            atom_count = sum(line.startswith("ATOM  ") for line in lines)
            hetero_count = sum(line.startswith("HETATM") for line in lines)
            if atom_count == 0:
                raise ValueError(
                    f"Predicted receptor has no ATOM records: {source_predicted}"
                )
            if args.receptor_mode == "holo_aligned_predicted_pocket_crop":
                from prepare_rldiff_inputs import crop_to_model_native_pocket

                destination = (
                    args.protein_output_dir.resolve()
                    / f"{target_id}_predicted_pocket.pdb"
                )
                crop_stats = crop_to_model_native_pocket(
                    source_predicted,
                    Path(row["ligand"]).resolve(),
                    destination,
                    pocket_cutoff=5.0,
                    pocket_buffer=10.0,
                )
        protein_stats[target_id] = {
            "atom_records": atom_count,
            "hetatm_records_in_source": hetero_count,
            **({"pre_esm_pocket_crop": crop_stats} if crop_stats else {}),
        }
        emitted.append(
            {
                "complex_name": target_id,
                "experimental_protein": str(destination),
                "ligand": str(Path(row["ligand"]).resolve()),
            }
        )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["complex_name", "experimental_protein", "ligand"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(emitted)

    provenance = {
        "schema_version": 1,
        "dataset": args.dataset,
        "expected_full_denominator": expected_size,
        "emitted_rows": len(emitted),
        "target_ids": [row["complex_name"] for row in emitted],
        "source_csv": str(args.source_csv.resolve()),
        "receptor_mode": args.receptor_mode,
        "receptor_policy": (
            "retain holo ATOM coordinates; remove HETATM before DiffDock-Pocket "
            "preprocessing to match checkpoint include_miscellaneous_atoms=false"
            if args.receptor_mode == "protein_only"
            else (
                "PoseBench holo-aligned predicted receptor; crystal SDF supplies the pocket"
                if args.receptor_mode == "holo_aligned_predicted"
                else "PoseBench holo-aligned predicted receptor cropped before ESM with "
                "DiffDock-Pocket pocket_cutoff=5 and pocket_buffer=10"
            )
        ),
        "protein_stats": protein_stats,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
    }
    args.output_csv.with_suffix(".json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Wrote {len(emitted)} rows to {args.output_csv}")


if __name__ == "__main__":
    main()
