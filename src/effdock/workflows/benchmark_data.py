"""Prepare immutable local inputs for EFF-Dock external redocking benchmarks."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import pickle
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import rdMolAlign

from effdock.evaluation.benchmark import detect_complex_files, load_ligand, match_atoms
from effdock.inference.preprocess import derive_pocket_center
from effdock.preprocess.protein import parse_pocket_atoms


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class _RDKitMolUnpickler(pickle.Unpickler):
    """Restricted loader for the CASF archive's ``(ligand, pocket)`` tuples."""

    def find_class(self, module: str, name: str):
        if module == "rdkit.Chem.rdchem" and name == "Mol":
            return Chem.Mol
        raise pickle.UnpicklingError(f"blocked pickle global: {module}.{name}")


def load_casf_pickle(path: Path) -> tuple[Chem.Mol, Chem.Mol]:
    value = _RDKitMolUnpickler(io.BytesIO(path.read_bytes())).load()
    if not (
        isinstance(value, tuple)
        and len(value) == 2
        and all(isinstance(item, Chem.Mol) for item in value)
    ):
        raise ValueError(f"unexpected CASF payload in {path}")
    return value


def _largest_heavy_fragment(mol: Chem.Mol) -> Chem.Mol:
    mol = Chem.RemoveHs(mol)
    fragments = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    if fragments:
        mol = max(fragments, key=lambda fragment: fragment.GetNumAtoms())
    Chem.SanitizeMol(mol)
    if mol.GetNumConformers() != 1:
        raise ValueError("reference ligand must have exactly one conformer")
    return mol


def _pdb_xyz(line: str) -> np.ndarray | None:
    try:
        return np.asarray(
            [float(line[30:38]), float(line[38:46]), float(line[46:54])], dtype=float
        )
    except ValueError:
        return None


def strip_reference_ligand(pdb_text: str, ligand_xyz: np.ndarray) -> tuple[str, list[str]]:
    """Remove PDB residues whose coordinates match the frozen ligand pose.

    CASF contains both small-molecule HETATM ligands and short peptide ligands
    encoded as ordinary ATOM residues, so record type alone is insufficient.
    """
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
        dmat = np.linalg.norm(np.asarray(coords)[:, None, :] - ligand_xyz[None, :, :], axis=2)
        matched = int((dmat.min(axis=1) < 0.30).sum())
        if matched >= 3 and matched / len(coords) >= 0.5:
            remove.add(key)

    if not remove:
        raise ValueError("no RCSB residue matched the CASF reference ligand")

    kept: list[str] = []
    for line in pdb_text.splitlines(keepends=True):
        if line.startswith(("ATOM", "HETATM")):
            key = (line[21], line[22:26], line[26], line[17:20].strip())
            if key in remove:
                continue
        kept.append(line)
    labels = [f"{chain}:{number.strip()}{icode.strip()}:{name}" for chain, number, icode, name in sorted(remove)]
    return "".join(kept), labels


def align_pdb_to_reference(pdb_text: str, reference: Chem.Mol) -> tuple[str, dict]:
    """Rigidly align an RCSB assembly to a CASF/PDBbind ligand frame."""
    residue_lines: dict[tuple[str, str, str, str], list[str]] = defaultdict(list)
    for line in pdb_text.splitlines():
        if line.startswith(("ATOM", "HETATM")):
            key = (line[21], line[22:26], line[26], line[17:20].strip())
            residue_lines[key].append(line)

    best: tuple[int, float, tuple[str, str, str, str], np.ndarray] | None = None
    for key, lines in residue_lines.items():
        if len(lines) < 3:
            continue
        candidate = Chem.MolFromPDBBlock(
            "\n".join(lines) + "\nEND\n",
            sanitize=True,
            removeHs=True,
            proximityBonding=True,
        )
        if candidate is None:
            continue
        dock_indices, ref_indices, _ = match_atoms(reference, candidate)
        if len(dock_indices) < max(3, int(0.8 * reference.GetNumAtoms())):
            continue
        rmsd, transform = rdMolAlign.GetAlignmentTransform(
            candidate,
            reference,
            atomMap=list(zip(dock_indices, ref_indices)),
            reflect=False,
            maxIters=100,
        )
        rank = (len(dock_indices), -float(rmsd))
        if best is None or rank > (best[0], -best[1]):
            best = (len(dock_indices), float(rmsd), key, np.asarray(transform))
    if best is None:
        raise ValueError("could not align RCSB receptor to the CASF ligand frame")

    matched_atoms, rmsd, key, transform = best
    transformed: list[str] = []
    for line in pdb_text.splitlines(keepends=True):
        if line.startswith(("ATOM", "HETATM")):
            xyz = _pdb_xyz(line)
            if xyz is not None:
                aligned = transform @ np.asarray([*xyz, 1.0])
                line = (
                    f"{line[:30]}{aligned[0]:8.3f}{aligned[1]:8.3f}{aligned[2]:8.3f}"
                    f"{line[54:]}"
                )
        transformed.append(line)
    label = f"{key[0]}:{key[1].strip()}{key[2].strip()}:{key[3]}"
    return "".join(transformed), {
        "applied": True,
        "matched_residue": label,
        "matched_atoms": matched_atoms,
        "rmsd_angstrom": rmsd,
        "transform": transform.tolist(),
    }


def prepare_casf(
    pickle_dir: Path,
    output_dir: Path,
    external_dir: Path,
) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_pdb_dir = output_dir / "raw_pdb"
    raw_pdb_dir.mkdir(exist_ok=True)
    records: list[dict] = []
    smiles: dict[str, dict[str, str]] = {}

    for payload in sorted(path for path in pickle_dir.iterdir() if path.is_file()):
        complex_id = payload.name.lower()
        ligand_raw, _ = load_casf_pickle(payload)
        ligand = _largest_heavy_fragment(ligand_raw)
        complex_dir = output_dir / complex_id
        complex_dir.mkdir(exist_ok=True)
        ligand_path = complex_dir / f"{complex_id}_ligand.sdf"
        protein_path = complex_dir / f"{complex_id}_protein.pdb"
        raw_pdb_path = raw_pdb_dir / f"{complex_id}.pdb"

        if not raw_pdb_path.exists():
            url = f"https://files.rcsb.org/download/{complex_id.upper()}.pdb"
            with urllib.request.urlopen(url, timeout=60) as response:
                raw_pdb_path.write_bytes(response.read())

        writer = Chem.SDWriter(str(ligand_path))
        writer.write(ligand)
        writer.close()
        ligand_xyz = np.asarray(ligand.GetConformer().GetPositions(), dtype=float)
        pdb_text = raw_pdb_path.read_text()
        alignment = {"applied": False}
        try:
            cleaned, removed = strip_reference_ligand(pdb_text, ligand_xyz)
        except ValueError:
            aligned_pdb, alignment = align_pdb_to_reference(pdb_text, ligand)
            cleaned, removed = strip_reference_ligand(aligned_pdb, ligand_xyz)
        protein_path.write_text(cleaned)
        canonical = Chem.MolToSmiles(ligand, canonical=True)
        smiles[complex_id] = {"smiles": canonical}
        records.append(
            {
                "id": complex_id,
                "source_pickle_sha256": sha256_file(payload),
                "raw_pdb_sha256": sha256_file(raw_pdb_path),
                "protein_sha256": sha256_file(protein_path),
                "ligand_sha256": sha256_file(ligand_path),
                "removed_reference_residues": removed,
                "rcsb_to_casf_alignment": alignment,
            }
        )
        print(f"prepared CASF {complex_id} ({len(records)}/285)")

    mapping_path = external_dir / "casf_smiles.json"
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    mapping_path.write_text(json.dumps(smiles, indent=2, sort_keys=True) + "\n")
    return records


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
    name = {"astex": "astex_smiles.json", "posebusters": "pb_smiles.json", "casf": "casf_smiles.json"}[dataset]
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
    parser.add_argument("--dataset", choices=("astex", "posebusters", "casf", "all"), default="all")
    parser.add_argument("--external-dir", type=Path, default=Path("data/external_test"))
    parser.add_argument(
        "--benchmark-root", type=Path, default=Path("data/external_benchmarks/data")
    )
    parser.add_argument(
        "--casf-pickle-dir",
        type=Path,
        default=Path("data/external_benchmarks/data/casf2016_scoring_raw/data_5_sdf"),
    )
    args = parser.parse_args(argv)

    datasets = ("astex", "posebusters", "casf") if args.dataset == "all" else (args.dataset,)
    dataset_dirs = {
        "astex": args.benchmark_root / "astex_diverse_set",
        "posebusters": args.benchmark_root / "posebusters_benchmark_set",
        "casf": args.benchmark_root / "casf2016",
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
    if "casf" in datasets:
        manifest["datasets"]["casf_preparation"] = prepare_casf(
            args.casf_pickle_dir,
            dataset_dirs["casf"],
            args.external_dir,
        )

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
