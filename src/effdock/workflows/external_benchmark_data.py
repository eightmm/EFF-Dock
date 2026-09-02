"""Normalize recent external protein-ligand benchmarks for EFF-Dock.

Raw archives stay under ``data/external_benchmarks`` and are intentionally
gitignored.  This module converts selected systems into the canonical
``{id}_protein.pdb`` / ``{id}_ligand.sdf`` layout consumed by
``eff-dock evaluate`` and writes a content-auditable local manifest.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import pickle
import re
import shutil
import urllib.request
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from Bio import Align
from Bio.Data.PDBData import protein_letters_3to1_extended
from rdkit import Chem

from effdock.workflows.benchmark_data import freeze_reference_centers

EXTERNAL_DATASETS = ("phibench", "foldbench", "openbind")
TEMPORAL_CUTOFF = "2024-06-30"
PHIBENCH_DEPOSIT_START = "2024-06-01"
PHIBENCH_DEPOSIT_END = "2024-12-31"
PHIBENCH_SEQUENCE_IDENTITY = 0.995

PHYS_DOCK_ARCHIVE_MD5 = "ad71e631eb439367667a89de8c41892e"
PHYS_DOCK_CCD_SHA256 = "ba30e1cb6e7cc372c325dbb08ae22d81099e133949fa87cbb4824153f6d4a6a7"
FOLDBENCH_CSV_SHA256 = "f0bf964ca1b9699e2036baa9bdfcc231e56181ec2a6df0f2eb24000e23cf3e0a"
FOLDBENCH_ARCHIVE_SHA256 = "69d72dbbddaa4a6b4005220b8eafc09d1a0f7575dcf3783686e7847655f3e1c9"
OPENBIND_ARCHIVE_MD5 = "860a4979d0ba9decaa2bfaa933c1d217"

_AA3 = set(protein_letters_3to1_extended)
_PDB_CHAIN_IDS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"


@dataclass(frozen=True)
class PhiCandidate:
    source_member: str
    complex_id: str
    pdb_id: str
    sequence: str


def file_digest(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_digest(path: Path, expected: str, algorithm: str = "sha256") -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = file_digest(path, algorithm)
    if actual != expected:
        raise ValueError(
            f"{path} {algorithm} mismatch: expected {expected}, got {actual}"
        )


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _stable_ids_sha256(ids: list[str]) -> str:
    digest = hashlib.sha256(b"EFFDOCK_EXTERNAL_BENCHMARK_IDS_V1\0")
    for complex_id in sorted(ids):
        digest.update(complex_id.encode())
        digest.update(b"\0")
    return digest.hexdigest()


def _safe_id(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9._-]+", "_", value.lower()).strip("_.-")
    if not normalized:
        raise ValueError(f"cannot form a benchmark ID from {value!r}")
    return normalized


def _as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if file_digest(source) != file_digest(target):
            raise ValueError(f"refusing to overwrite different normalized file: {target}")
        return
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def _write_sdf(mol: Chem.Mol, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = Chem.SDWriter(str(path))
    try:
        writer.write(mol)
    finally:
        writer.close()
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"failed to write ligand SDF: {path}")


def _largest_heavy_fragment(mol: Chem.Mol) -> Chem.Mol:
    mol = Chem.RemoveAllHs(Chem.Mol(mol), sanitize=False)
    fragments = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    if fragments:
        mol = max(fragments, key=lambda item: item.GetNumAtoms())
    mol.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(mol)
    return mol


def _canonical_smiles(mol: Chem.Mol) -> str:
    return Chem.MolToSmiles(_largest_heavy_fragment(mol), canonical=True, isomericSmiles=True)


def fetch_rcsb_dates(pdb_ids: list[str]) -> dict[str, dict[str, str]]:
    """Fetch deposit/release dates from the official RCSB GraphQL endpoint."""
    query = """
    query Entries($ids: [String!]!) {
      entries(entry_ids: $ids) {
        rcsb_id
        rcsb_accession_info { deposit_date initial_release_date }
      }
    }
    """
    records: dict[str, dict[str, str]] = {}
    for start in range(0, len(pdb_ids), 100):
        batch = pdb_ids[start : start + 100]
        body = json.dumps({"query": query, "variables": {"ids": batch}}).encode()
        request = urllib.request.Request(
            "https://data.rcsb.org/graphql",
            data=body,
            headers={
                "Content-Type": "application/json",
                "User-Agent": "EFF-Dock-benchmark-prep/1",
            },
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = json.load(response)
        if payload.get("errors"):
            raise RuntimeError(payload["errors"])
        for record in payload["data"]["entries"]:
            records[record["rcsb_id"].lower()] = record["rcsb_accession_info"]
    return records


def load_or_fetch_rcsb_dates(
    pdb_ids: list[str],
    path: Path,
    *,
    refresh: bool,
) -> dict[str, dict[str, str]]:
    if refresh or not path.is_file():
        records = fetch_rcsb_dates(sorted(set(pdb_ids)))
        _write_json(path, {"source": "https://data.rcsb.org/graphql", "records": records})
    else:
        records = json.loads(path.read_text()).get("records", {})
    missing = sorted(set(pdb_ids) - set(records))
    if missing:
        print(f"warning: RCSB returned no date record for {len(missing)} IDs: {missing[:8]}")
    return records


def _pdb_atom_name(name: str, element: str) -> str:
    name = name[:4]
    if len(name) < 4 and len(element.strip()) == 1:
        return f" {name:<3}"
    return f"{name:<4}"


def _pdb_line(
    *,
    record: str,
    serial: int,
    atom_name: str,
    residue_name: str,
    chain_id: str,
    residue_number: int,
    xyz: np.ndarray,
    element: str,
    occupancy: float = 1.0,
    bfactor: float = 0.0,
) -> str:
    return (
        f"{record:<6}{serial:>5} {_pdb_atom_name(atom_name, element)} "
        f"{residue_name[:3]:>3} {chain_id}{residue_number:>4}    "
        f"{xyz[0]:>8.3f}{xyz[1]:>8.3f}{xyz[2]:>8.3f}"
        f"{occupancy:>6.2f}{bfactor:>6.2f}          {element[:2]:>2}\n"
    )


def _ccd_ligand(
    ccd_record: dict[str, Any],
    coordinates: object,
    mask: object,
) -> Chem.Mol:
    mol = Chem.RemoveAllHs(Chem.Mol(ccd_record["ref_mol"]), sanitize=False)
    xyz = np.asarray(coordinates, dtype=float)
    atom_mask = np.asarray(mask, dtype=bool)
    if xyz.shape != (mol.GetNumAtoms(), 3) or atom_mask.shape != (mol.GetNumAtoms(),):
        raise ValueError("CCD molecule and coordinate tensor shapes do not match")
    missing_heavy = [
        index
        for index, atom in enumerate(mol.GetAtoms())
        if atom.GetAtomicNum() > 1 and not atom_mask[index]
    ]
    if missing_heavy:
        raise ValueError(f"reference ligand is missing {len(missing_heavy)} heavy atoms")
    mol.RemoveAllConformers()
    conformer = Chem.Conformer(mol.GetNumAtoms())
    conformer.Set3D(True)
    for index, point in enumerate(xyz):
        conformer.SetAtomPosition(index, point.tolist())
    mol.AddConformer(conformer, assignId=True)
    return _largest_heavy_fragment(mol)


def _physdock_protein_pdb(payload: dict[str, Any], ccd_meta: dict[str, Any]) -> str:
    protein_chains = [(str(key), value) for key, value in payload.items() if not str(key).isdigit()]
    if len(protein_chains) > len(_PDB_CHAIN_IDS):
        raise ValueError("too many protein chains for PDB output")
    lines: list[str] = []
    serial = 1
    for output_chain, (_, chain) in zip(_PDB_CHAIN_IDS, protein_chains):
        for residue_number, (ccd_id, coordinates, mask) in enumerate(
            zip(
                chain["ccds"],
                chain["all_atom_positions"],
                chain["all_atom_mask"],
                strict=True,
            ),
            start=1,
        ):
            ccd = ccd_meta[str(ccd_id)]
            mol = ccd["ref_mol"]
            atom_names = ccd["ref_atom_name_chars"]
            xyz = np.asarray(coordinates, dtype=float)
            atom_mask = np.asarray(mask, dtype=bool)
            if len(atom_names) != mol.GetNumAtoms() or xyz.shape != (mol.GetNumAtoms(), 3):
                raise ValueError(f"protein CCD shape mismatch: {ccd_id}")
            record = "ATOM" if str(ccd_id).upper() in _AA3 else "HETATM"
            for index, atom in enumerate(mol.GetAtoms()):
                if not atom_mask[index] or atom.GetAtomicNum() == 1:
                    continue
                lines.append(
                    _pdb_line(
                        record=record,
                        serial=serial,
                        atom_name=str(atom_names[index]),
                        residue_name=str(ccd_id),
                        chain_id=output_chain,
                        residue_number=residue_number,
                        xyz=xyz[index],
                        element=atom.GetSymbol(),
                    )
                )
                serial += 1
        lines.append("TER\n")
    lines.append("END\n")
    if serial == 1:
        raise ValueError("PhysDock payload contains no protein heavy atoms")
    return "".join(lines)


def _physdock_candidates(
    archive: zipfile.ZipFile,
    date_records: dict[str, dict[str, str]],
) -> tuple[list[PhiCandidate], dict[str, int]]:
    candidates: list[PhiCandidate] = []
    exclusions: defaultdict[str, int] = defaultdict(int)
    for member in sorted(archive.namelist()):
        if not (member.startswith("benchmarks/PhiBench/") and member.endswith(".pkl.gz")):
            continue
        filename = Path(member).name
        pdb_id = filename[:4].lower()
        dates = date_records.get(pdb_id)
        if dates is None:
            exclusions["missing_rcsb_date_included_from_official_archive"] += 1
        else:
            deposit_date = str(dates.get("deposit_date", ""))[:10]
            if not (PHIBENCH_DEPOSIT_START <= deposit_date <= PHIBENCH_DEPOSIT_END):
                exclusions["outside_deposit_window"] += 1
                continue
        payload = pickle.loads(gzip.decompress(archive.read(member)))  # noqa: S301
        if not any(str(key).isdigit() for key in payload):
            exclusions["missing_ligand_chain"] += 1
            continue
        sequences = [
            "".join(
                protein_letters_3to1_extended.get(str(ccd).upper(), "X")
                for ccd in chain["ccds"]
            )
            for key, chain in payload.items()
            if not str(key).isdigit()
        ]
        candidates.append(
            PhiCandidate(
                source_member=member,
                complex_id=_safe_id(filename.removesuffix(".pkl.gz")),
                pdb_id=pdb_id,
                sequence="X".join(sequences),
            )
        )
    return candidates, dict(exclusions)


def sequence_diverse_representatives(
    candidates: list[PhiCandidate],
    threshold: float = PHIBENCH_SEQUENCE_IDENTITY,
) -> tuple[list[PhiCandidate], dict[str, int | float]]:
    """Choose one deterministic representative per global-identity component."""
    exact_groups: defaultdict[str, list[PhiCandidate]] = defaultdict(list)
    for candidate in candidates:
        exact_groups[candidate.sequence].append(candidate)
    representatives = [sorted(group, key=lambda item: item.source_member)[0] for group in exact_groups.values()]
    representatives.sort(key=lambda item: item.source_member)

    parent = list(range(len(representatives)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    aligner = Align.PairwiseAligner(scoring="blastp")
    aligner.mode = "global"
    compared = 0
    for left in range(len(representatives)):
        left_seq = representatives[left].sequence
        for right in range(left):
            right_seq = representatives[right].sequence
            if min(len(left_seq), len(right_seq)) / max(len(left_seq), len(right_seq)) < 0.30:
                continue
            compared += 1
            counts = aligner.align(left_seq, right_seq)[0].counts()
            identity = counts.identities / max(len(left_seq), len(right_seq))
            if identity < threshold:
                continue
            left_root, right_root = find(left), find(right)
            if left_root != right_root:
                parent[right_root] = left_root

    components: defaultdict[int, list[PhiCandidate]] = defaultdict(list)
    for index, candidate in enumerate(representatives):
        components[find(index)].append(candidate)
    selected = [sorted(group, key=lambda item: item.source_member)[0] for group in components.values()]
    selected.sort(key=lambda item: item.source_member)
    return selected, {
        "source_candidates": len(candidates),
        "exact_sequence_representatives": len(representatives),
        "global_sequence_identity_threshold": threshold,
        "pairwise_comparisons": compared,
        "selected_components": len(selected),
    }


def _load_verified_ccd(path: Path, *, verify: bool) -> dict[str, Any]:
    if verify:
        require_digest(path, PHYS_DOCK_CCD_SHA256)
    with gzip.open(path, "rb") as handle:
        return pickle.load(handle)  # noqa: S301


def _ccd_from_cif(path: Path) -> dict[str, Any]:
    """Build a heavy-atom RDKit template in the official CCD atom order."""
    import gemmi

    block = gemmi.cif.read_file(str(path)).sole_block()
    atom_table = block.get_mmcif_category("_chem_comp_atom.")
    bond_table = block.get_mmcif_category("_chem_comp_bond.")
    if not atom_table or not bond_table:
        raise ValueError(f"incomplete CCD definition: {path}")
    molecule = Chem.RWMol()
    atom_names: list[str] = []
    index_by_name: dict[str, int] = {}
    for row_index, atom_name_raw in enumerate(atom_table["atom_id"]):
        element = str(atom_table["type_symbol"][row_index]).strip().capitalize()
        if element.upper() in {"H", "D"}:
            continue
        atom_name = str(atom_name_raw)
        atom = Chem.Atom(element)
        try:
            atom.SetFormalCharge(int(atom_table["charge"][row_index]))
        except (TypeError, ValueError):
            atom.SetFormalCharge(0)
        aromatic = str(atom_table["pdbx_aromatic_flag"][row_index]).upper() == "Y"
        atom.SetIsAromatic(aromatic)
        index_by_name[atom_name] = molecule.AddAtom(atom)
        atom_names.append(atom_name)

    bond_types = {
        "SING": Chem.BondType.SINGLE,
        "DOUB": Chem.BondType.DOUBLE,
        "TRIP": Chem.BondType.TRIPLE,
        "AROM": Chem.BondType.AROMATIC,
        "DELO": Chem.BondType.ONEANDAHALF,
    }
    for row_index, left_raw in enumerate(bond_table["atom_id_1"]):
        left, right = str(left_raw), str(bond_table["atom_id_2"][row_index])
        if left not in index_by_name or right not in index_by_name:
            continue
        order = str(bond_table["value_order"][row_index]).upper()
        aromatic = str(bond_table["pdbx_aromatic_flag"][row_index]).upper() == "Y"
        bond_type = Chem.BondType.AROMATIC if aromatic else bond_types.get(order)
        if bond_type is None:
            raise ValueError(f"unsupported CCD bond order {order!r} in {path}")
        molecule.AddBond(index_by_name[left], index_by_name[right], bond_type)
        if bond_type == Chem.BondType.AROMATIC:
            bond = molecule.GetBondBetweenAtoms(index_by_name[left], index_by_name[right])
            bond.SetIsAromatic(True)
            molecule.GetAtomWithIdx(index_by_name[left]).SetIsAromatic(True)
            molecule.GetAtomWithIdx(index_by_name[right]).SetIsAromatic(True)

    mol = molecule.GetMol()
    mol.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(mol)
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        Chem.SanitizeMol(
            mol,
            sanitizeOps=(
                Chem.SanitizeFlags.SANITIZE_FINDRADICALS
                | Chem.SanitizeFlags.SANITIZE_SETAROMATICITY
                | Chem.SanitizeFlags.SANITIZE_SETCONJUGATION
                | Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION
                | Chem.SanitizeFlags.SANITIZE_SYMMRINGS
            ),
        )
    descriptor_table = block.get_mmcif_category("_pdbx_chem_comp_descriptor.")
    canonical_smiles: str | None = None
    candidates: list[tuple[int, str]] = []
    for row_index, descriptor_type in enumerate(descriptor_table.get("type", [])):
        if str(descriptor_type).upper() != "SMILES_CANONICAL":
            continue
        program = str(descriptor_table["program"][row_index]).upper()
        rank = 0 if program == "CACTVS" else 1 if program.startswith("OPENEYE") else 2
        candidates.append((rank, str(descriptor_table["descriptor"][row_index])))
    for _, candidate in sorted(candidates):
        if Chem.MolFromSmiles(candidate) is not None:
            canonical_smiles = candidate
            break
    return {
        "ref_mol": mol,
        "ref_atom_name_chars": atom_names,
        "canonical_smiles": canonical_smiles,
    }


def _ensure_ccd_records(
    ccd_ids: set[str],
    ccd_meta: dict[str, Any],
    cache_dir: Path,
) -> dict[str, dict[str, str]]:
    """Fill gaps in a released CCD pickle from official RCSB definitions."""
    audit: dict[str, dict[str, str]] = {}
    cache_dir.mkdir(parents=True, exist_ok=True)
    for ccd_id in sorted(ccd_ids - set(ccd_meta)):
        path = cache_dir / f"{ccd_id}.cif"
        url = f"https://files.rcsb.org/ligands/download/{ccd_id}.cif"
        if not path.is_file():
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "EFF-Dock-benchmark-prep/1"},
            )
            with urllib.request.urlopen(request, timeout=60) as response:
                path.write_bytes(response.read())
        ccd_meta[ccd_id] = _ccd_from_cif(path)
        audit[ccd_id] = {"url": url, "path": str(path), "sha256": file_digest(path)}
    return audit


def prepare_phibench(
    *,
    archive_path: Path,
    ccd_path: Path,
    dates_path: Path,
    ccd_cache_dir: Path,
    output_root: Path,
    external_dir: Path,
    manifest_dir: Path,
    refresh_dates: bool,
    verify: bool,
) -> dict[str, object]:
    if verify:
        require_digest(archive_path, PHYS_DOCK_ARCHIVE_MD5, "md5")
    ccd_meta = _load_verified_ccd(ccd_path, verify=verify)
    with zipfile.ZipFile(archive_path) as archive:
        pdb_ids = sorted(
            {
                Path(name).name[:4].lower()
                for name in archive.namelist()
                if name.startswith("benchmarks/PhiBench/") and name.endswith(".pkl.gz")
            }
        )
        dates = load_or_fetch_rcsb_dates(pdb_ids, dates_path, refresh=refresh_dates)
        candidates, exclusions = _physdock_candidates(archive, dates)
        required_ccds: set[str] = set()
        for candidate in candidates:
            payload = pickle.loads(gzip.decompress(archive.read(candidate.source_member)))  # noqa: S301
            required_ccds.update(
                str(ccd_id)
                for chain in payload.values()
                for ccd_id in chain["ccds"]
            )
        fallback_ccds = _ensure_ccd_records(required_ccds, ccd_meta, ccd_cache_dir)
        eligible: list[PhiCandidate] = []
        invalid_ligand_examples: list[dict[str, str]] = []
        for candidate in candidates:
            payload = pickle.loads(gzip.decompress(archive.read(candidate.source_member)))  # noqa: S301
            ligand_key = next(key for key in payload if str(key).isdigit())
            ligand_chain = payload[ligand_key]
            try:
                if len(ligand_chain["ccds"]) != 1:
                    raise ValueError("ligand chain contains multiple CCDs")
                ccd_id = str(ligand_chain["ccds"][0])
                _ccd_ligand(
                    ccd_meta[ccd_id],
                    ligand_chain["all_atom_positions"][0],
                    ligand_chain["all_atom_mask"][0],
                )
            except ValueError as exc:
                exclusions["invalid_or_incomplete_ligand_reference"] = (
                    exclusions.get("invalid_or_incomplete_ligand_reference", 0) + 1
                )
                invalid_ligand_examples.append(
                    {"source_member": candidate.source_member, "reason": str(exc)}
                )
                continue
            eligible.append(candidate)
        selected, selection = sequence_diverse_representatives(eligible)
        mapping: dict[str, str] = {}
        records: list[dict[str, object]] = []
        dataset_root = output_root / "phibench"
        for candidate in selected:
            payload = pickle.loads(gzip.decompress(archive.read(candidate.source_member)))  # noqa: S301
            ligand_key = next(key for key in payload if str(key).isdigit())
            ligand_chain = payload[ligand_key]
            if len(ligand_chain["ccds"]) != 1:
                raise ValueError(f"PhiBench ligand is not one CCD: {candidate.source_member}")
            ccd_id = str(ligand_chain["ccds"][0])
            mol = _ccd_ligand(
                ccd_meta[ccd_id],
                ligand_chain["all_atom_positions"][0],
                ligand_chain["all_atom_mask"][0],
            )
            complex_dir = dataset_root / candidate.complex_id
            complex_dir.mkdir(parents=True, exist_ok=True)
            protein_path = complex_dir / f"{candidate.complex_id}_protein.pdb"
            ligand_path = complex_dir / f"{candidate.complex_id}_ligand.sdf"
            protein_path.write_text(_physdock_protein_pdb(payload, ccd_meta))
            _write_sdf(mol, ligand_path)
            mapping[candidate.complex_id] = (
                ccd_meta[ccd_id].get("canonical_smiles") or _canonical_smiles(mol)
            )
            date_record = dates.get(candidate.pdb_id, {})
            records.append(
                {
                    "id": candidate.complex_id,
                    "pdb_id": candidate.pdb_id,
                    "ccd_id": ccd_id,
                    "source_member": candidate.source_member,
                    "deposit_date": date_record.get("deposit_date"),
                    "initial_release_date": date_record.get("initial_release_date"),
                }
            )

    mapping_path = external_dir / "phibench_smiles.json"
    centers_path = external_dir / "phibench_reference_pocket_centers.json"
    _write_json(mapping_path, mapping)
    center_records = freeze_reference_centers(
        "phibench", output_root / "phibench", sorted(mapping), centers_path
    )
    manifest = {
        "schema_version": "effdock.external_benchmark.v1",
        "dataset": "phibench",
        "cohort_label": "EFF-Dock derived temporal sequence-diverse cohort",
        "paper_exact_cohort_claimed": False,
        "source_verification_performed": verify,
        "count": len(records),
        "ids_sha256": _stable_ids_sha256(list(mapping)),
        "selection": selection,
        "exclusions": exclusions,
        "invalid_ligand_examples": invalid_ligand_examples,
        "source": {
            "project": "https://github.com/KexinZhangResearch/PhysDock",
            "archive": "https://zenodo.org/records/15178859/files/physdock_benchmarks.zip",
            "archive_md5": PHYS_DOCK_ARCHIVE_MD5,
            "ccd_sha256": PHYS_DOCK_CCD_SHA256,
            "license": "CC BY 4.0 (Zenodo data)",
        },
        "rcsb_dates": {
            "source": "https://data.rcsb.org/graphql",
            "path": str(dates_path),
            "sha256": file_digest(dates_path),
        },
        "rcsb_ccd_fallbacks": fallback_ccds,
        "normalized_root": str(output_root / "phibench"),
        "mapping": str(mapping_path),
        "pocket_centers": str(centers_path),
        "pocket_center_count": len(center_records),
        "records": records,
    }
    _write_json(manifest_dir / "phibench.json", manifest)
    return manifest


def _foldbench_ligand(
    atom_site: dict[str, list[Any]],
    label_asym_id: str,
    ccd_record: dict[str, Any],
    *,
    impute_missing_heavy: bool = False,
) -> tuple[Chem.Mol, list[str]]:
    rows: dict[str, np.ndarray] = {}
    for index, asym_id in enumerate(atom_site["label_asym_id"]):
        if str(asym_id) != label_asym_id:
            continue
        if str(atom_site["pdbx_PDB_model_num"][index]) != "1":
            continue
        alt = atom_site["label_alt_id"][index]
        if alt not in (False, None, ".", "?", "A", "1"):
            continue
        atom_name = str(atom_site["label_atom_id"][index])
        rows.setdefault(
            atom_name,
            np.asarray(
                [
                    float(atom_site["Cartn_x"][index]),
                    float(atom_site["Cartn_y"][index]),
                    float(atom_site["Cartn_z"][index]),
                ]
            ),
        )
    mol = Chem.RemoveAllHs(Chem.Mol(ccd_record["ref_mol"]), sanitize=False)
    atom_names = [str(name) for name in ccd_record["ref_atom_name_chars"]]
    missing_heavy = [
        name
        for name, atom in zip(atom_names, mol.GetAtoms(), strict=True)
        if atom.GetAtomicNum() > 1 and name not in rows
    ]
    if missing_heavy and not impute_missing_heavy:
        raise ValueError(f"FoldBench ligand is missing heavy atoms: {missing_heavy[:8]}")
    imputed_coordinates: dict[str, np.ndarray] = {}
    if missing_heavy:
        if mol.GetNumConformers() != 1:
            raise ValueError("CCD missing-atom imputation requires one ideal conformer")
        template = np.asarray(mol.GetConformer().GetPositions(), dtype=float)
        name_to_index = {name: index for index, name in enumerate(atom_names)}
        missing_indices = {name_to_index[name] for name in missing_heavy}
        for missing_name in missing_heavy:
            missing_index = name_to_index[missing_name]
            distances = Chem.GetDistanceMatrix(mol)[missing_index]
            local_indices = [
                index
                for index, name in enumerate(atom_names)
                if index not in missing_indices and name in rows and distances[index] <= 2
            ]
            if len(local_indices) < 3:
                local_indices = [
                    index
                    for index, name in enumerate(atom_names)
                    if index not in missing_indices and name in rows
                ]
            if len(local_indices) < 3:
                raise ValueError(
                    f"FoldBench ligand cannot impute {missing_name}: fewer than 3 anchors"
                )
            source = template[local_indices]
            target = np.asarray([rows[atom_names[index]] for index in local_indices])
            source_center = source.mean(axis=0)
            target_center = target.mean(axis=0)
            left, _, right_t = np.linalg.svd(
                (source - source_center).T @ (target - target_center)
            )
            rotation = left @ right_t
            if np.linalg.det(rotation) < 0:
                left[:, -1] *= -1
                rotation = left @ right_t
            imputed_coordinates[missing_name] = (
                (template[missing_index] - source_center) @ rotation + target_center
            )
    mol.RemoveAllConformers()
    conformer = Chem.Conformer(mol.GetNumAtoms())
    conformer.Set3D(True)
    for index, atom_name in enumerate(atom_names):
        coordinate = rows.get(atom_name, imputed_coordinates.get(atom_name))
        if coordinate is None:
            coordinate = np.zeros(3)
        conformer.SetAtomPosition(index, coordinate.tolist())
    mol.AddConformer(conformer, assignId=True)
    return _largest_heavy_fragment(mol), missing_heavy


def _foldbench_protein_pdb(atom_site: dict[str, list[Any]], label_asym_id: str) -> str:
    lines: list[str] = []
    seen: set[tuple[str, str, str]] = set()
    serial = 1
    residue_map: dict[tuple[str, str], int] = {}
    for index, asym_id in enumerate(atom_site["label_asym_id"]):
        if str(asym_id) != label_asym_id or str(atom_site["group_PDB"][index]) != "ATOM":
            continue
        if str(atom_site["pdbx_PDB_model_num"][index]) != "1":
            continue
        element = str(atom_site["type_symbol"][index])
        if element.upper() in {"H", "D"}:
            continue
        alt = atom_site["label_alt_id"][index]
        if alt not in (False, None, ".", "?", "A", "1"):
            continue
        seq_id = str(atom_site["label_seq_id"][index])
        insertion = str(atom_site["pdbx_PDB_ins_code"][index] or "")
        atom_name = str(atom_site["label_atom_id"][index])
        key = (seq_id, insertion, atom_name)
        if key in seen:
            continue
        seen.add(key)
        residue_key = (seq_id, insertion)
        residue_number = residue_map.setdefault(residue_key, len(residue_map) + 1)
        xyz = np.asarray(
            [
                float(atom_site["Cartn_x"][index]),
                float(atom_site["Cartn_y"][index]),
                float(atom_site["Cartn_z"][index]),
            ]
        )
        lines.append(
            _pdb_line(
                record="ATOM",
                serial=serial,
                atom_name=atom_name,
                residue_name=str(atom_site["label_comp_id"][index]),
                chain_id="A",
                residue_number=residue_number,
                xyz=xyz,
                element=element,
                occupancy=float(atom_site["occupancy"][index]),
                bfactor=float(atom_site["B_iso_or_equiv"][index]),
            )
        )
        serial += 1
    if not lines:
        raise ValueError(f"FoldBench protein chain is empty: {label_asym_id}")
    return "".join(lines) + "TER\nEND\n"


def select_foldbench_postcut(
    rows: list[dict[str, str]],
    dates: dict[str, dict[str, str]],
) -> list[dict[str, str]]:
    return [
        row
        for row in rows
        if row["pdb_id"].split("-")[0].lower() in dates
        and str(
            dates[row["pdb_id"].split("-")[0].lower()].get("initial_release_date", "")
        )[:10]
        > TEMPORAL_CUTOFF
    ]


def foldbench_complex_id(row: dict[str, str], *, cohort: str) -> str:
    """Return a stable interface ID without changing the historical subset IDs."""
    if cohort == "postcut":
        return _safe_id(row["pdb_id"])
    if cohort != "full":
        raise ValueError(f"unsupported FoldBench cohort: {cohort}")
    ccd_id = row["ligand_id"].strip("()")
    return _safe_id(
        f"{row['pdb_id']}__protein-{row['native_chain_id_1']}"
        f"__ligand-{row['native_chain_id_2']}__ccd-{ccd_id}"
    )


def prepare_foldbench(
    *,
    csv_path: Path,
    structure_root: Path,
    archive_path: Path,
    ccd_path: Path,
    dates_path: Path,
    ccd_cache_dir: Path,
    output_root: Path,
    external_dir: Path,
    manifest_dir: Path,
    refresh_dates: bool,
    verify: bool,
    cohort: str = "postcut",
) -> dict[str, object]:
    import gemmi

    if verify:
        require_digest(csv_path, FOLDBENCH_CSV_SHA256)
        require_digest(archive_path, FOLDBENCH_ARCHIVE_SHA256)
    ccd_meta = _load_verified_ccd(ccd_path, verify=verify)
    with csv_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    pdb_ids = sorted({row["pdb_id"].split("-")[0].lower() for row in rows})
    dates = load_or_fetch_rcsb_dates(pdb_ids, dates_path, refresh=refresh_dates)
    if cohort == "postcut":
        selected = select_foldbench_postcut(rows, dates)
        cohort_label = "strict post-2024-06-30 protein-ligand interfaces"
    elif cohort == "full":
        selected = rows
        cohort_label = "all official FoldBench protein-ligand interface rows"
    else:
        raise ValueError(f"unsupported FoldBench cohort: {cohort}")
    selected_ids = [foldbench_complex_id(row, cohort=cohort) for row in selected]
    if len(set(selected_ids)) != len(selected_ids):
        raise ValueError(f"FoldBench {cohort} cohort does not have unique interface IDs")

    required_ccds = {row["ligand_id"].strip("()") for row in selected}
    fallback_ccds = _ensure_ccd_records(required_ccds, ccd_meta, ccd_cache_dir)
    mapping: dict[str, str] = {}
    records: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    reference_heavy_atom_imputations: dict[str, list[str]] = {}
    dataset_root = output_root / "foldbench"
    for row, complex_id in zip(selected, selected_ids, strict=True):
        cif_path = structure_root / f"{row['pdb_id']}.cif"
        try:
            atom_site = gemmi.cif.read_file(str(cif_path)).sole_block().get_mmcif_category(
                "_atom_site."
            )
            ccd_id = row["ligand_id"].strip("()")
            mol, imputed_atoms = _foldbench_ligand(
                atom_site,
                row["native_chain_id_2"],
                ccd_meta[ccd_id],
                impute_missing_heavy=cohort == "full",
            )
            protein = _foldbench_protein_pdb(atom_site, row["native_chain_id_1"])
            complex_dir = dataset_root / complex_id
            complex_dir.mkdir(parents=True, exist_ok=True)
            (complex_dir / f"{complex_id}_protein.pdb").write_text(protein)
            _write_sdf(mol, complex_dir / f"{complex_id}_ligand.sdf")
            mapping[complex_id] = ccd_meta[ccd_id].get("canonical_smiles") or _canonical_smiles(mol)
            if imputed_atoms:
                reference_heavy_atom_imputations[complex_id] = imputed_atoms
            pdb_id = row["pdb_id"].split("-")[0].lower()
            records.append(
                {
                    "id": complex_id,
                    "pdb_id": pdb_id,
                    "assembly_id": row["pdb_id"],
                    "protein_label_asym_id": row["native_chain_id_1"],
                    "ligand_label_asym_id": row["native_chain_id_2"],
                    "ccd_id": ccd_id,
                    "initial_release_date": dates[pdb_id]["initial_release_date"],
                }
            )
        except Exception as exc:
            failures.append({"id": complex_id, "error": f"{type(exc).__name__}: {exc}"})

    mapping_path = external_dir / "foldbench_smiles.json"
    centers_path = external_dir / "foldbench_reference_pocket_centers.json"
    _write_json(mapping_path, mapping)
    center_records = freeze_reference_centers(
        "foldbench", dataset_root, sorted(mapping), centers_path
    )
    manifest = {
        "schema_version": "effdock.external_benchmark.v1",
        "dataset": "foldbench",
        "cohort_label": cohort_label,
        "cohort": cohort,
        "protocol_note": "EFF-Dock pocket-redocking adaptation; not the FoldBench leaderboard protocol",
        "source_verification_performed": verify,
        "source_row_count": len(rows),
        "selected_row_count": len(selected),
        "count": len(records),
        "ids_sha256": _stable_ids_sha256(list(mapping)),
        "failures": failures,
        "reference_heavy_atom_imputations": reference_heavy_atom_imputations,
        "reference_heavy_atom_imputation_method": (
            "local ideal-CCD rigid alignment with all observed heavy-atom coordinates preserved"
        ),
        "source": {
            "project": "https://github.com/BEAM-Labs/FoldBench",
            "portal": "https://portal.openfold.omsf.io/benchmarks/fold-bench",
            "csv_sha256": FOLDBENCH_CSV_SHA256,
            "archive_sha256": FOLDBENCH_ARCHIVE_SHA256,
            "license": "CC BY 4.0 (benchmark data)",
        },
        "rcsb_dates": {
            "source": "https://data.rcsb.org/graphql",
            "path": str(dates_path),
            "sha256": file_digest(dates_path),
        },
        "rcsb_ccd_fallbacks": fallback_ccds,
        "normalized_root": str(dataset_root),
        "mapping": str(mapping_path),
        "pocket_centers": str(centers_path),
        "pocket_center_count": len(center_records),
        "records": records,
    }
    _write_json(manifest_dir / "foldbench.json", manifest)
    return manifest


def select_openbind_clean(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        row
        for row in rows
        if not _as_bool(row["covalent"])
        and _as_bool(row["pb_valid_prepared"])
        and _as_bool(row["pb_valid_ref"])
        and not _as_bool(row["suspected_artefact"])
    ]


def prepare_openbind(
    *,
    source_root: Path,
    archive_path: Path,
    output_root: Path,
    external_dir: Path,
    manifest_dir: Path,
    verify: bool,
) -> dict[str, object]:
    if verify:
        require_digest(archive_path, OPENBIND_ARCHIVE_MD5, "md5")
    metadata_path = source_root / "EV-A71_2A_metadata.csv"
    with metadata_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected = select_openbind_clean(rows)
    mapping: dict[str, str] = {}
    records: list[dict[str, object]] = []
    dataset_root = output_root / "openbind"
    for row in selected:
        source_dir = source_root / "structures" / row["compound_group"] / row["complex_name"]
        complex_id = _safe_id(row["complex_name"])
        protein_source = source_dir / f"{row['complex_name']}_prepared.pdb"
        ligand_source = source_dir / f"{row['complex_name']}_ligand_ref.sdf"
        if not protein_source.is_file() or not ligand_source.is_file():
            raise FileNotFoundError(f"missing OpenBind structure for {row['complex_name']}")
        complex_dir = dataset_root / complex_id
        _link_or_copy(protein_source, complex_dir / f"{complex_id}_protein.pdb")
        _link_or_copy(ligand_source, complex_dir / f"{complex_id}_ligand.sdf")
        mol = next(Chem.SDMolSupplier(str(ligand_source), removeHs=True))
        if mol is None:
            raise ValueError(f"RDKit cannot parse {ligand_source}")
        mapping[complex_id] = str(row["smiles"])
        records.append(
            {
                "id": complex_id,
                "compound_group": row["compound_group"],
                "fragment_screen": _as_bool(row["fragment_screen"]),
                "experimental_pKD": row["experimental_pKD"] or None,
                "reference_smiles": _canonical_smiles(mol),
            }
        )

    mapping_path = external_dir / "openbind_smiles.json"
    centers_path = external_dir / "openbind_reference_pocket_centers.json"
    _write_json(mapping_path, mapping)
    center_records = freeze_reference_centers(
        "openbind", dataset_root, sorted(mapping), centers_path
    )
    manifest = {
        "schema_version": "effdock.external_benchmark.v1",
        "dataset": "openbind",
        "cohort_label": "clean non-covalent OpenBind EV-A71/CVA16 2A cohort",
        "source_verification_performed": verify,
        "source_count": len(rows),
        "count": len(records),
        "ids_sha256": _stable_ids_sha256(list(mapping)),
        "filters": {
            "covalent": False,
            "pb_valid_prepared": True,
            "pb_valid_ref": True,
            "suspected_artefact": False,
        },
        "source": {
            "project": "https://github.com/OpenBind-Consortium/EV-A71_2A_benchmark",
            "doi": "https://doi.org/10.5281/zenodo.20026661",
            "archive_md5": OPENBIND_ARCHIVE_MD5,
            "license": "CC0 1.0",
        },
        "normalized_root": str(dataset_root),
        "mapping": str(mapping_path),
        "pocket_centers": str(centers_path),
        "pocket_center_count": len(center_records),
        "records": records,
    }
    _write_json(manifest_dir / "openbind.json", manifest)
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=(*EXTERNAL_DATASETS, "all"), default="all")
    parser.add_argument(
        "--archive-root", type=Path, default=Path("data/external_benchmarks/archives")
    )
    parser.add_argument(
        "--raw-root", type=Path, default=Path("data/external_benchmarks/data")
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/external_benchmarks/normalized"),
    )
    parser.add_argument("--external-dir", type=Path, default=Path("data/external_test"))
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=Path("data/external_benchmarks/manifests"),
    )
    parser.add_argument(
        "--metadata-root",
        type=Path,
        default=Path("data/external_benchmarks/metadata"),
    )
    parser.add_argument("--refresh-rcsb-dates", action="store_true")
    parser.add_argument(
        "--foldbench-cohort",
        choices=("postcut", "full"),
        default="postcut",
        help="Use the historical temporal subset or all 558 official interface rows.",
    )
    parser.add_argument(
        "--skip-source-verification",
        action="store_true",
        help="Skip archive checksums. This also removes the safety boundary around official pickles.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    datasets = EXTERNAL_DATASETS if args.dataset == "all" else (args.dataset,)
    verify = not args.skip_source_verification
    args.manifest_dir.mkdir(parents=True, exist_ok=True)
    args.metadata_root.mkdir(parents=True, exist_ok=True)
    manifests: dict[str, object] = {}
    common = {
        "output_root": args.output_root,
        "external_dir": args.external_dir,
        "manifest_dir": args.manifest_dir,
    }
    if "phibench" in datasets:
        manifests["phibench"] = prepare_phibench(
            archive_path=args.archive_root / "physdock_benchmarks.zip",
            ccd_path=args.archive_root / "physdock_ccd_id_meta_data.pkl.gz",
            dates_path=args.metadata_root / "phibench_rcsb_dates.json",
            ccd_cache_dir=args.metadata_root / "ccd",
            refresh_dates=args.refresh_rcsb_dates,
            verify=verify,
            **common,
        )
    if "foldbench" in datasets:
        manifests["foldbench"] = prepare_foldbench(
            csv_path=args.archive_root / "foldbench_interface_protein_ligand.csv",
            archive_path=args.archive_root / "foldbench_ground_truth_1522.tar",
            structure_root=args.raw_root / "ground_truth_20250520",
            ccd_path=args.archive_root / "physdock_ccd_id_meta_data.pkl.gz",
            dates_path=args.metadata_root / "foldbench_rcsb_dates.json",
            ccd_cache_dir=args.metadata_root / "ccd",
            refresh_dates=args.refresh_rcsb_dates,
            verify=verify,
            cohort=args.foldbench_cohort,
            **common,
        )
    if "openbind" in datasets:
        manifests["openbind"] = prepare_openbind(
            source_root=args.raw_root / "OpenBind_EV-A71_2A",
            archive_path=args.archive_root / "OpenBind_EV-A71_2A.zip",
            verify=verify,
            **common,
        )
    summary_path = args.manifest_dir / "summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text())
        if summary.get("schema_version") != "effdock.external_benchmarks.v1":
            raise ValueError(f"unsupported external benchmark summary: {summary_path}")
    else:
        summary = {"schema_version": "effdock.external_benchmarks.v1", "datasets": {}}
    summary["datasets"].update(
        {
            name: {
                "count": manifest["count"],
                "ids_sha256": manifest["ids_sha256"],
                "manifest": str(args.manifest_dir / f"{name}.json"),
                "manifest_sha256": file_digest(args.manifest_dir / f"{name}.json"),
            }
            for name, manifest in manifests.items()
        }
    )
    _write_json(summary_path, summary)
    for name, manifest in manifests.items():
        print(f"prepared {name}: {manifest['count']} complexes")


if __name__ == "__main__":
    main()
