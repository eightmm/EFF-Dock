#!/usr/bin/env python3
"""Build strict full-cohort or target-subset CSV inputs for RLDiff.

RLDiff reduces every receptor to a ligand-defined pocket, but its upstream
loader computes full-chain ESM embeddings first.  Chains longer than ESM's
1,022-residue limit can therefore fail before the model-native pocket selector
runs.  ``model_native_pocket_crop`` applies that same center/radius selector to
the input PDB before ESM while preserving every selected ATOM/HETATM record.
"""

from __future__ import annotations

import argparse
import csv
import json
from math import dist
from pathlib import Path

DATASETS = {
    "astex_diverse": ("astex_diverse_85_ids.txt", 85),
    "posebusters_benchmark": ("posebusters_308_ids.txt", 308),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rldiff-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument(
        "--receptor-mode",
        choices=["raw_holo", "model_native_pocket_crop"],
        default="raw_holo",
    )
    parser.add_argument("--protein-output-dir", type=Path)
    parser.add_argument("--pocket-cutoff", type=float, default=5.0)
    parser.add_argument("--pocket-buffer", type=float, default=10.0)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--num-shards", type=int)
    parser.add_argument("--shard-index", type=int)
    parser.add_argument(
        "--cohort-id-list",
        type=Path,
        help="Override the upstream ID list while retaining the declared full denominator.",
    )
    parser.add_argument("--target-id", action="append", default=[])
    return parser.parse_args()


def read_ligand_coordinates(path: Path) -> list[tuple[float, float, float]]:
    from rdkit import Chem

    supplier = Chem.SDMolSupplier(str(path), removeHs=False, sanitize=False)
    molecule = next((mol for mol in supplier if mol is not None), None)
    if molecule is None or molecule.GetNumConformers() == 0:
        raise ValueError(f"Could not read 3D ligand coordinates: {path}")
    conformer = molecule.GetConformer()
    return [
        (
            conformer.GetAtomPosition(index).x,
            conformer.GetAtomPosition(index).y,
            conformer.GetAtomPosition(index).z,
        )
        for index in range(molecule.GetNumAtoms())
    ]


def crop_to_model_native_pocket(
    protein: Path,
    ligand: Path,
    output: Path,
    *,
    pocket_cutoff: float,
    pocket_buffer: float,
) -> dict[str, object]:
    """Apply RLDiff's center-dist PocketSelector before ESM embedding."""

    ligand_coordinates = read_ligand_coordinates(ligand)
    coordinate_lines: list[tuple[str, tuple[str, str, str, str, str], str, tuple[float, float, float]]] = []
    residue_atoms: dict[
        tuple[str, str, str, str, str],
        list[tuple[str, tuple[float, float, float]]],
    ] = {}
    for line in protein.read_text(errors="replace").splitlines():
        if line.startswith("ENDMDL"):
            break
        record = line[:6].strip()
        if record not in {"ATOM", "HETATM"}:
            continue
        if len(line) < 54:
            raise ValueError(f"Malformed coordinate record in {protein}: {line!r}")
        residue_key = (
            line[:6],
            line[21:22],
            line[22:26],
            line[26:27],
            line[17:20],
        )
        atom_name = line[12:16].strip()
        coordinates = (
            float(line[30:38]),
            float(line[38:46]),
            float(line[46:54]),
        )
        residue_atoms.setdefault(residue_key, []).append((atom_name, coordinates))
        coordinate_lines.append((line, residue_key, atom_name, coordinates))

    c_alpha_coordinates = [
        coordinates
        for _, _, atom_name, coordinates in coordinate_lines
        if atom_name == "CA"
    ]
    if not c_alpha_coordinates:
        raise ValueError(f"Receptor has no CA atoms: {protein}")

    nearby_c_alphas = [
        coordinates
        for coordinates in c_alpha_coordinates
        if any(dist(coordinates, ligand_xyz) < pocket_cutoff for ligand_xyz in ligand_coordinates)
    ]
    if not nearby_c_alphas:
        nearby_c_alphas = [
            min(
                c_alpha_coordinates,
                key=lambda coordinates: min(
                    dist(coordinates, ligand_xyz) for ligand_xyz in ligand_coordinates
                ),
            )
        ]
    pocket_center = tuple(
        sum(coordinates[axis] for coordinates in nearby_c_alphas)
        / len(nearby_c_alphas)
        for axis in range(3)
    )
    pocket_radius = (
        max(dist(ligand_xyz, pocket_center) for ligand_xyz in ligand_coordinates)
        + pocket_buffer
    )
    selected_residues = {
        residue_key
        for residue_key, atoms in residue_atoms.items()
        if any(dist(coordinates, pocket_center) < pocket_radius for _, coordinates in atoms)
    }
    selected_lines = [
        line
        for line, residue_key, _, _ in coordinate_lines
        if residue_key in selected_residues
    ]
    if not selected_lines:
        raise ValueError(f"Pocket crop selected no receptor atoms: {protein}")

    amino_like_per_chain: dict[str, int] = {}
    for residue_key in selected_residues:
        _, chain_id, _, _, residue_name = residue_key
        atom_names = {atom_name for atom_name, _ in residue_atoms[residue_key]}
        if residue_name.strip() != "HOH" and {"N", "CA", "C"} <= atom_names:
            amino_like_per_chain[chain_id] = amino_like_per_chain.get(chain_id, 0) + 1
    max_chain_residues = max(amino_like_per_chain.values(), default=0)
    if max_chain_residues > 1022:
        raise ValueError(
            f"Pocket-cropped receptor still exceeds ESM limit: {protein} "
            f"max_chain_residues={max_chain_residues}"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join([*selected_lines, "END"]) + "\n")
    return {
        "source_atom_records": len(coordinate_lines),
        "selected_atom_records": len(selected_lines),
        "selected_residues": len(selected_residues),
        "max_selected_chain_residues": max_chain_residues,
        "nearby_c_alpha_records": len(nearby_c_alphas),
        "pocket_center": [round(value, 6) for value in pocket_center],
        "pocket_radius": round(pocket_radius, 6),
    }


def main() -> None:
    args = parse_args()
    if (
        args.receptor_mode == "model_native_pocket_crop"
        and args.protein_output_dir is None
    ):
        raise ValueError(
            "--protein-output-dir is required for model_native_pocket_crop mode"
        )
    if (args.num_shards is None) != (args.shard_index is None):
        raise ValueError("--num-shards and --shard-index must be provided together")
    if args.num_shards is not None and not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must satisfy 0 <= index < --num-shards")
    id_filename, expected_size = DATASETS[args.dataset]
    id_path = (
        args.cohort_id_list.resolve()
        if args.cohort_id_list is not None
        else args.rldiff_root.resolve() / "data" / id_filename
    )
    dataset_root = args.dataset_root.resolve()

    ids = [line.strip() for line in id_path.read_text().splitlines() if line.strip()]
    if len(ids) != expected_size or len(set(ids)) != expected_size:
        raise ValueError(
            f"Official {args.dataset} ID list must contain {expected_size} unique IDs; "
            f"found {len(ids)} rows and {len(set(ids))} unique IDs"
        )

    requested = set(args.target_id)
    missing_requested = requested - set(ids)
    if missing_requested:
        raise ValueError(f"Unknown target IDs: {sorted(missing_requested)}")

    rows: list[dict[str, str]] = []
    receptor_stats: dict[str, dict[str, object]] = {}
    for target_id in ids:
        if requested and target_id not in requested:
            continue
        target_dir = dataset_root / target_id
        protein = target_dir / f"{target_id}_protein.pdb"
        ligand = target_dir / f"{target_id}_ligand.sdf"
        if not protein.is_file() or not ligand.is_file():
            raise FileNotFoundError(
                f"Incomplete {target_id}: protein={protein.is_file()} ligand={ligand.is_file()}"
            )
        emitted_protein = protein
        if args.receptor_mode == "model_native_pocket_crop":
            emitted_protein = (
                args.protein_output_dir.resolve()
                / f"{target_id}_rldiff_pocket.pdb"
            )
            receptor_stats[target_id] = crop_to_model_native_pocket(
                protein,
                ligand,
                emitted_protein,
                pocket_cutoff=args.pocket_cutoff,
                pocket_buffer=args.pocket_buffer,
            )
        rows.append(
            {
                "complex_name": target_id,
                "experimental_protein": str(emitted_protein.resolve()),
                "ligand": str(ligand.resolve()),
            }
        )

    if args.num_shards is not None:
        rows = [
            row
            for index, row in enumerate(rows)
            if index % args.num_shards == args.shard_index
        ]

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["complex_name", "experimental_protein", "ligand"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)

    provenance = {
        "schema_version": 1,
        "dataset": args.dataset,
        "expected_full_denominator": expected_size,
        "emitted_rows": len(rows),
        "target_ids": [row["complex_name"] for row in rows],
        "official_id_list": str(id_path),
        "dataset_root": str(dataset_root),
        "receptor_mode": args.receptor_mode,
        "receptor": (
            "holo crystal protein"
            if args.receptor_mode == "raw_holo"
            else "holo crystal protein cropped with RLDiff center-dist PocketSelector before ESM"
        ),
        "receptor_stats": receptor_stats,
        "pocket_cutoff": args.pocket_cutoff,
        "pocket_buffer": args.pocket_buffer,
        "ligand_initialization": "reference SDF local geometry",
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
    }
    provenance_path = args.output_csv.with_suffix(".json")
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Wrote {len(rows)} rows to {args.output_csv}")
    print(f"Wrote provenance to {provenance_path}")


if __name__ == "__main__":
    main()
