#!/usr/bin/env python3
"""Freeze PoseBench FABind/DynamicBind inputs against an explicit cohort."""

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
    parser.add_argument("--dataset", choices=sorted(DATASET_TO_SIZE), required=True)
    parser.add_argument("--source-fabind-csv", type=Path, required=True)
    parser.add_argument("--cohort-manifest", type=Path, required=True)
    parser.add_argument("--predicted-protein-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--smoke-target", required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    return parser.parse_args()


def write_fabind_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["Cleaned_SMILES", "pdb_id"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def write_dynamicbind_inputs(path: Path, rows: list[dict[str, str]]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for row in rows:
        output = path / f"{row['pdb_id']}.csv"
        output.write_text(f"ligand\n{row['Cleaned_SMILES']}\n")


def link_proteins(
    path: Path,
    rows: list[dict[str, str]],
    protein_paths: dict[str, Path],
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for row in rows:
        target = protein_paths[row["pdb_id"]].resolve()
        link = path / target.name
        if link.exists() or link.is_symlink():
            if link.resolve() != target:
                raise FileExistsError(f"Conflicting receptor link: {link}")
        else:
            link.symlink_to(target)


def write_vina_manifest(
    path: Path,
    rows: list[dict[str, str]],
    cohort_by_id: dict[str, dict[str, str]],
    protein_paths: dict[str, Path],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "complex_name",
                "holo_protein",
                "reference_ligand",
                "predicted_receptor",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            target_id = row["pdb_id"]
            cohort_row = cohort_by_id[target_id]
            writer.writerow(
                {
                    "complex_name": target_id,
                    "holo_protein": str(Path(cohort_row["experimental_protein"]).resolve()),
                    "reference_ligand": str(Path(cohort_row["ligand"]).resolve()),
                    "predicted_receptor": str(protein_paths[target_id].resolve()),
                }
            )


def write_diffbindfr_manifest(
    path: Path,
    rows: list[dict[str, str]],
    cohort_by_id: dict[str, dict[str, str]],
    protein_paths: dict[str, Path],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "protein",
                "protein_name",
                "ligand",
                "ligand_name",
                "complex_name",
                "crystal_ligand",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            target_id = row["pdb_id"]
            ligand = Path(cohort_by_id[target_id]["ligand"]).resolve()
            writer.writerow(
                {
                    "protein": str(protein_paths[target_id].resolve()),
                    "protein_name": target_id,
                    "ligand": str(ligand),
                    "ligand_name": target_id,
                    "complex_name": target_id,
                    "crystal_ligand": str(ligand),
                }
            )


def main() -> None:
    args = parse_args()
    if args.num_shards <= 0:
        raise ValueError("--num-shards must be positive")
    expected_size = DATASET_TO_SIZE[args.dataset]
    with args.source_fabind_csv.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    with args.cohort_manifest.open(newline="") as handle:
        cohort_rows = list(csv.DictReader(handle))
    cohort_by_id = {row["complex_name"]: row for row in cohort_rows}

    source_ids = [row["pdb_id"] for row in rows]
    cohort_ids = [row["complex_name"] for row in cohort_rows]
    if len(source_ids) != expected_size or len(set(source_ids)) != expected_size:
        raise ValueError(f"PoseBench {args.dataset} CSV must contain {expected_size} unique IDs")
    if len(cohort_ids) != expected_size or len(set(cohort_ids)) != expected_size:
        raise ValueError(f"Cohort manifest must contain {expected_size} unique complex_name values")
    if set(source_ids) != set(cohort_ids):
        raise ValueError(
            "PoseBench CSV and frozen cohort differ: "
            f"missing={sorted(set(cohort_ids) - set(source_ids))} "
            f"unexpected={sorted(set(source_ids) - set(cohort_ids))}"
        )
    if args.smoke_target not in set(source_ids):
        raise ValueError(f"Unknown smoke target: {args.smoke_target}")

    predicted_protein_dir = args.predicted_protein_dir.resolve()
    protein_paths: dict[str, Path] = {}
    for target_id in source_ids:
        protein = predicted_protein_dir / f"{target_id}_holo_aligned_predicted_protein.pdb"
        if not protein.is_file():
            raise FileNotFoundError(f"Missing predicted receptor: {protein}")
        protein_paths[target_id] = protein

    output_root = args.output_root.resolve()
    fabind_csv = output_root / f"fabind_{args.dataset}_inputs.csv"
    fabind_smoke_csv = output_root / f"fabind_{args.dataset}_inputs_first_1.csv"
    smoke_rows = [row for row in rows if row["pdb_id"] == args.smoke_target]
    write_fabind_csv(fabind_csv, rows)
    write_fabind_csv(fabind_smoke_csv, smoke_rows)

    dynamicbind_dir = output_root / f"dynamicbind_{args.dataset}_inputs"
    dynamicbind_smoke_dir = output_root / f"dynamicbind_{args.dataset}_smoke_inputs"
    write_dynamicbind_inputs(dynamicbind_dir, rows)
    write_dynamicbind_inputs(dynamicbind_smoke_dir, smoke_rows)
    vina_manifest = output_root / f"vina_{args.dataset}_inputs.csv"
    vina_smoke_manifest = output_root / f"vina_{args.dataset}_smoke_inputs.csv"
    write_vina_manifest(vina_manifest, rows, cohort_by_id, protein_paths)
    write_vina_manifest(vina_smoke_manifest, smoke_rows, cohort_by_id, protein_paths)
    diffbindfr_manifest = output_root / f"diffbindfr_{args.dataset}_inputs.csv"
    smoke_target_index = source_ids.index(args.smoke_target)
    diffbindfr_smoke_rows = [
        rows[smoke_target_index],
        rows[(smoke_target_index + 1) % len(rows)],
    ]
    diffbindfr_smoke_manifest = output_root / f"diffbindfr_{args.dataset}_inputs_first_2.csv"
    write_diffbindfr_manifest(
        diffbindfr_manifest,
        rows,
        cohort_by_id,
        protein_paths,
    )
    write_diffbindfr_manifest(
        diffbindfr_smoke_manifest,
        diffbindfr_smoke_rows,
        cohort_by_id,
        protein_paths,
    )

    smoke_protein_dir = output_root / f"{args.dataset}_smoke_proteins"
    link_proteins(smoke_protein_dir, smoke_rows, protein_paths)

    shard_sizes: dict[str, int] = {}
    for shard_index in range(args.num_shards):
        shard_rows = [
            row for row_index, row in enumerate(rows) if row_index % args.num_shards == shard_index
        ]
        shard_name = f"shard_{shard_index:03d}"
        shard_root = output_root / "shards" / shard_name
        write_fabind_csv(shard_root / "fabind_inputs.csv", shard_rows)
        write_dynamicbind_inputs(shard_root / "dynamicbind_inputs", shard_rows)
        link_proteins(shard_root / "proteins", shard_rows, protein_paths)
        write_vina_manifest(
            shard_root / "vina_inputs.csv",
            shard_rows,
            cohort_by_id,
            protein_paths,
        )
        write_diffbindfr_manifest(
            shard_root / "diffbindfr_inputs.csv",
            shard_rows,
            cohort_by_id,
            protein_paths,
        )
        shard_sizes[shard_name] = len(shard_rows)

    provenance = {
        "schema_version": 1,
        "dataset": args.dataset,
        "expected_full_denominator": expected_size,
        "target_ids": source_ids,
        "source_fabind_csv": str(args.source_fabind_csv.resolve()),
        "cohort_manifest": str(args.cohort_manifest.resolve()),
        "predicted_protein_dir": str(predicted_protein_dir),
        "smoke_target": args.smoke_target,
        "fabind_full_csv": str(fabind_csv),
        "fabind_smoke_csv": str(fabind_smoke_csv),
        "dynamicbind_full_input_dir": str(dynamicbind_dir),
        "dynamicbind_smoke_input_dir": str(dynamicbind_smoke_dir),
        "vina_full_manifest": str(vina_manifest),
        "vina_smoke_manifest": str(vina_smoke_manifest),
        "diffbindfr_full_manifest": str(diffbindfr_manifest),
        "diffbindfr_smoke_manifest": str(diffbindfr_smoke_manifest),
        "smoke_protein_dir": str(smoke_protein_dir),
        "num_shards": args.num_shards,
        "shard_sizes": shard_sizes,
    }
    (output_root / f"{args.dataset}_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    print(
        f"Prepared PoseBench native inputs for {args.dataset}: "
        f"full={len(rows)} smoke={args.smoke_target}"
    )


if __name__ == "__main__":
    main()
