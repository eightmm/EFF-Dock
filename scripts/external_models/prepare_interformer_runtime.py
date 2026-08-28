#!/usr/bin/env python3
"""Prepare a frozen receptor/ligand shard for native Interformer docking."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

from rdkit import Chem


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--interformer-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reduce-bin", type=Path, required=True)
    return parser.parse_args()


def safe_link(target: Path, link: Path) -> None:
    target = target.resolve()
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.exists() or link.is_symlink():
        if link.resolve() != target:
            raise FileExistsError(f"Conflicting link: {link}")
        return
    link.symlink_to(target)


def write_uff_conformer(
    source_ligand: Path,
    output_ligand: Path,
    interformer_root: Path,
) -> dict[str, object]:
    if output_ligand.is_file():
        return {"complete": True, "cached": True}
    sys.path.insert(0, str(interformer_root / "tools"))
    import rdkit_ETKDG_3d_gen  # type: ignore[import-not-found]

    molecule = Chem.SDMolSupplier(str(source_ligand), sanitize=True)[0]
    result = rdkit_ETKDG_3d_gen.new_conformation(molecule, n_confs=30, num_threads=0)
    output_ligand.parent.mkdir(parents=True, exist_ok=True)
    if result is None:
        shutil.copy2(source_ligand, output_ligand)
        return {
            "complete": True,
            "cached": False,
            "fallback": "reference conformer copied because upstream ETKDG/UFF failed",
        }
    writer = Chem.SDWriter(str(output_ligand))
    writer.write(result["mol"])
    writer.close()
    return {
        "complete": output_ligand.is_file(),
        "cached": False,
        "uff_energy": float(result["energy"]),
        "aligned_rmsd_to_reference": float(result["rmsd"]),
    }


def prepare_receptor(
    source_protein: Path,
    reduced_protein: Path,
    reduce_bin: Path,
) -> dict[str, object]:
    if reduced_protein.is_file():
        return {"complete": True, "cached": True, "fallback": False}
    reduced_protein.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.run(
        [str(reduce_bin), str(source_protein)],
        check=False,
        capture_output=True,
        text=True,
    )
    if process.returncode == 0 and process.stdout.strip():
        reduced_protein.write_text(process.stdout)
        return {"complete": True, "cached": False, "fallback": False}
    shutil.copy2(source_protein, reduced_protein)
    return {
        "complete": True,
        "cached": False,
        "fallback": True,
        "reduce_returncode": process.returncode,
        "reduce_stderr_tail": process.stderr[-1000:],
    }


def extract_pocket(
    reduced_protein: Path,
    ligand_dir: Path,
    raw_protein_dir: Path,
    pocket_link: Path,
    interformer_root: Path,
) -> dict[str, object]:
    if pocket_link.is_file():
        return {"complete": True, "cached": True}
    tools_dir = interformer_root / "tools"
    sys.path.insert(0, str(tools_dir))
    import extract_pocket_by_ligand  # type: ignore[import-not-found]

    extract_pocket_by_ligand.protein_path = str(raw_protein_dir)
    extract_pocket_by_ligand.ligand_path = str(ligand_dir)
    (raw_protein_dir / "output").mkdir(parents=True, exist_ok=True)
    extract_pocket_by_ligand.run_fn(str(reduced_protein), rm_ccd=False)
    generated = raw_protein_dir / "output" / f"{reduced_protein.name[:4]}_pocket.pdb"
    if not generated.is_file():
        return {"complete": False, "error": "upstream pocket output missing"}
    safe_link(generated, pocket_link)
    return {"complete": True, "cached": False, "generated_pocket": str(generated)}


def main() -> None:
    args = parse_args()
    input_csv = args.input_csv.resolve()
    interformer_root = args.interformer_root.resolve()
    output_dir = args.output_dir.resolve()
    reduce_bin = args.reduce_bin.resolve()
    ligand_dir = output_dir / "ligand"
    uff_dir = output_dir / "uff"
    raw_protein_dir = output_dir / "raw_protein"
    pocket_dir = output_dir / "pocket"
    for path in (ligand_dir, uff_dir, raw_protein_dir, pocket_dir):
        path.mkdir(parents=True, exist_ok=True)
    if not reduce_bin.is_file():
        raise FileNotFoundError(f"Reduce executable is unavailable: {reduce_bin}")

    with input_csv.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    required_columns = {"complex_name", "reference_ligand", "predicted_receptor"}
    if not rows or not required_columns.issubset(rows[0]):
        raise ValueError(f"Input CSV must contain {sorted(required_columns)}")

    aliases: dict[str, str] = {}
    statuses: dict[str, dict[str, object]] = {}
    query_rows = []
    for row in rows:
        target_id = row["complex_name"]
        alias = target_id[:4].lower()
        if alias in aliases and aliases[alias] != target_id:
            raise ValueError(f"Interformer 4-character target collision: {alias}")
        aliases[alias] = target_id
        reference_ligand = Path(row["reference_ligand"]).resolve()
        predicted_receptor = Path(row["predicted_receptor"]).resolve()
        ligand = ligand_dir / f"{alias}_docked.sdf"
        uff = uff_dir / f"{alias}_uff.sdf"
        reduced = raw_protein_dir / f"{alias}_reduce.pdb"
        pocket = pocket_dir / f"{alias}_pocket.pdb"
        safe_link(reference_ligand, ligand)
        status: dict[str, object] = {}
        try:
            status["uff"] = write_uff_conformer(reference_ligand, uff, interformer_root)
            status["reduce"] = prepare_receptor(predicted_receptor, reduced, reduce_bin)
            status["pocket"] = extract_pocket(
                reduced,
                ligand_dir,
                raw_protein_dir,
                pocket,
                interformer_root,
            )
            complete = uff.is_file() and pocket.is_file()
        except Exception as error:  # preserve denominator and continue
            complete = False
            status["error"] = f"{type(error).__name__}: {error}"
        status["complete"] = complete
        statuses[target_id] = status
        if complete:
            query_rows.append(
                {
                    "Target": alias,
                    "Uniprot": alias,
                    "Molecule ID": target_id,
                    "pose_rank": 0,
                    "pIC50": 0.0,
                    "complex_name": target_id,
                }
            )

    query_csv = output_dir / "query.csv"
    with query_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "Target",
                "Uniprot",
                "Molecule ID",
                "pose_rank",
                "pIC50",
                "complex_name",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(query_rows)

    metadata = {
        "schema_version": 1,
        "input_csv": str(input_csv),
        "expected_targets": len(rows),
        "prepared_targets": len(query_rows),
        "protein_policy": "holo-aligned predicted receptor",
        "site_policy": "reference-ligand-supplied 10A pocket, then model-native 7A interception",
        "protein_preparation": "Reduce; source receptor fallback recorded per target",
        "ligand_preparation": "upstream ETKDGv3/UFF, seed 42, 30 conformers",
        "aliases": aliases,
        "statuses": statuses,
        "query_csv": str(query_csv),
    }
    (output_dir / "preprocess_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Interformer preprocessing: {len(query_rows)}/{len(rows)} targets ready")
    if not query_rows:
        raise RuntimeError("Interformer preprocessing failed for every target")


if __name__ == "__main__":
    main()
