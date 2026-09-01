"""Prepare immutable local inputs for EFF-Dock external redocking benchmarks."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from effdock.evaluation.benchmark import detect_complex_files, load_ligand
from effdock.inference.preprocess import derive_pocket_center
from effdock.preprocess.protein import parse_pocket_atoms


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pdb_xyz(line: str) -> np.ndarray | None:
    try:
        return np.asarray(
            [float(line[30:38]), float(line[38:46]), float(line[46:54])], dtype=float
        )
    except ValueError:
        return None


def strip_reference_ligand(pdb_text: str, ligand_xyz: np.ndarray) -> tuple[str, list[str]]:
    """Remove receptor residues whose coordinates match a reference ligand."""
    residue_lines: dict[tuple[str, str, str, str], list[str]] = defaultdict(list)
    for line in pdb_text.splitlines(keepends=True):
        if line.startswith(("ATOM", "HETATM")):
            key = (line[21], line[22:26], line[26], line[17:20].strip())
            residue_lines[key].append(line)

    remove: set[tuple[str, str, str, str]] = set()
    for key, lines in residue_lines.items():
        coords = [coord for line in lines if (coord := _pdb_xyz(line)) is not None]
        if len(coords) < 3:
            continue
        distances = np.linalg.norm(
            np.asarray(coords)[:, None, :] - ligand_xyz[None, :, :], axis=2
        )
        matched = int((distances.min(axis=1) < 0.30).sum())
        if matched >= 3 and matched / len(coords) >= 0.5:
            remove.add(key)

    if not remove:
        raise ValueError("no RCSB residue matched the reference ligand")

    kept: list[str] = []
    for line in pdb_text.splitlines(keepends=True):
        if line.startswith(("ATOM", "HETATM")):
            key = (line[21], line[22:26], line[26], line[17:20].strip())
            if key in remove:
                continue
        kept.append(line)
    labels = [
        f"{chain}:{number.strip()}{icode.strip()}:{name}"
        for chain, number, icode, name in sorted(remove)
    ]
    return "".join(kept), labels


def freeze_reference_centers(
    dataset: str,
    data_dir: Path,
    complex_ids: list[str],
    output_path: Path,
    cutoff: float = 8.0,
) -> list[dict]:
    directory_index = {path.name.lower(): path for path in data_dir.iterdir() if path.is_dir()}
    centers: dict[str, dict] = {}
    records: list[dict] = []
    for complex_id in sorted(complex_ids):
        complex_dir = directory_index.get(complex_id.lower())
        if complex_dir is None:
            prefix = complex_id.lower().split("_")[0] + "_"
            complex_dir = next(
                (path for name, path in directory_index.items() if name.startswith(prefix)), None
            )
        if complex_dir is None:
            raise FileNotFoundError(f"missing {dataset} directory for {complex_id}")
        detected = detect_complex_files(complex_dir, complex_dir.name)
        if detected is None:
            raise FileNotFoundError(f"missing receptor/reference ligand in {complex_dir}")
        protein_path, ligand_path, ligand_format = detected
        ligand = load_ligand(ligand_path, ligand_format)
        ligand_xyz = torch.tensor(ligand.GetConformer().GetPositions(), dtype=torch.float32)
        protein = parse_pocket_atoms(protein_path)
        if protein is None:
            raise ValueError(f"failed to parse {protein_path}")
        center = derive_pocket_center(protein, ligand_xyz, cutoff=cutoff)
        entry = {
            "center": [round(float(value), 6) for value in center],
            "definition": "reference_ligand_residue_center",
            "cutoff_angstrom": cutoff,
            "protein_sha256": sha256_file(protein_path),
            "reference_ligand_sha256": sha256_file(ligand_path),
        }
        centers[complex_id.lower()] = entry
        records.append({"id": complex_id.lower(), **entry})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(centers, indent=2, sort_keys=True) + "\n")
    return records


def _mapping_ids(dataset: str, external_dir: Path) -> list[str]:
    name = {"astex": "astex_smiles.json", "posebusters": "pb_smiles.json"}[dataset]
    raw = json.loads((external_dir / name).read_text())
    ids = sorted(raw)
    if dataset == "posebusters":
        keep = {
            line.strip().lower()
            for line in (external_dir / "posebusters_v2_ids.txt").read_text().splitlines()
            if line.strip()
        }
        ids = [complex_id for complex_id in ids if complex_id.lower() in keep]
    return ids


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("astex", "posebusters", "all"), default="all")
    parser.add_argument("--external-dir", type=Path, default=Path("data/external_test"))
    parser.add_argument(
        "--benchmark-root", type=Path, default=Path("data/external_benchmarks/data")
    )
    args = parser.parse_args(argv)

    datasets = ("astex", "posebusters") if args.dataset == "all" else (args.dataset,)
    dataset_dirs = {
        "astex": args.benchmark_root / "astex_diverse_set",
        "posebusters": args.benchmark_root / "posebusters_benchmark_set",
    }
    manifest_path = args.external_dir / "reference_redocking_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    else:
        manifest = {
            "protocol": "reference-defined oracle-pocket redocking diagnostic",
            "pocket_definition": "residue virtual-node centroid within 8A of reference ligand",
            "datasets": {},
        }
    for dataset in datasets:
        center_path = args.external_dir / f"{dataset}_reference_pocket_centers.json"
        records = freeze_reference_centers(
            dataset,
            dataset_dirs[dataset],
            _mapping_ids(dataset, args.external_dir),
            center_path,
        )
        manifest["datasets"][dataset] = {
            "count": len(records),
            "data_dir": str(dataset_dirs[dataset]),
            "centers": str(center_path),
            "centers_sha256": sha256_file(center_path),
        }
        print(f"froze {dataset}: {len(records)} centers")

    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
