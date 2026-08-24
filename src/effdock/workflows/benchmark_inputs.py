"""Frozen ligand-input contracts for external redocking workflows.

This module deliberately sits above the generic inference loader.  Historical
benchmark mappings contain a few explicit stereochemical hydrogens, while the
trained EFF-Dock graph is heavy-atom only.  FULL-V2 therefore removes *all*
hydrogens after seeded conformer generation without changing the training or
general inference preprocessing contract.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from rdkit import Chem

from effdock.inference.preprocess import load_ligand as load_generic_ligand

BENCHMARK_INPUT_MANIFEST_SCHEMA = "effdock.benchmark_inputs.v2"
BENCHMARK_INPUT_IDENTITY_SCHEMA = "effdock.benchmark_input_identity.v1"
BENCHMARK_INPUT_PROTOCOL_ID = "EFFDOCK-UNIFIED-GUIDANCE-BUDGET1000-FULL-V2"
HEAVY_ATOM_POLICY = "seeded_generic_loader_then_rdkit_remove_all_hs"


class BenchmarkInputMismatchError(ValueError):
    """Raised when a frozen benchmark input does not match its reference graph."""

    def __init__(self, code: str, message: str, *, details: dict[str, object]) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details

    def as_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "message": self.message,
            "details": self.details,
        }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sorted_id_sha256(ids: list[str]) -> str:
    digest = hashlib.sha256()
    digest.update(b"EFFDOCK_SORTED_COMPLEX_IDS_V1\0")
    for complex_id in ids:
        digest.update(complex_id.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def canonical_heavy_smiles(smiles: str) -> str:
    """Return a canonical isomeric heavy-atom graph or fail explicitly."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise BenchmarkInputMismatchError(
            "invalid_benchmark_smiles",
            "benchmark ligand SMILES cannot be parsed",
            details={"smiles": smiles},
        )
    mol = Chem.RemoveAllHs(mol)
    hydrogen_count = sum(atom.GetAtomicNum() == 1 for atom in mol.GetAtoms())
    if hydrogen_count:
        raise BenchmarkInputMismatchError(
            "benchmark_heavy_atom_normalization_failed",
            "benchmark ligand still contains hydrogen atoms after RemoveAllHs",
            details={"smiles": smiles, "hydrogen_count": hydrogen_count},
        )
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def ligand_input_identity(complex_id: str, smiles: str) -> dict[str, str]:
    canonical = canonical_heavy_smiles(smiles)
    digest = hashlib.sha256()
    digest.update(b"EFFDOCK_BENCHMARK_LIGAND_INPUT_V1\0")
    digest.update(complex_id.encode("utf-8"))
    digest.update(b"\0")
    digest.update(smiles.encode("utf-8"))
    digest.update(b"\0")
    digest.update(canonical.encode("utf-8"))
    return {
        "canonical_heavy_isomeric_smiles": canonical,
        "raw_smiles_sha256": hashlib.sha256(smiles.encode("utf-8")).hexdigest(),
        "sha256": digest.hexdigest(),
    }


def mapping_sha256(dataset: str, smiles_by_id: dict[str, str]) -> str:
    digest = hashlib.sha256()
    digest.update(b"EFFDOCK_BENCHMARK_INPUT_MAPPING_V1\0")
    digest.update(dataset.encode("utf-8"))
    digest.update(b"\0")
    for complex_id in sorted(smiles_by_id):
        digest.update(complex_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(smiles_by_id[complex_id].encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _legacy_mapping(
    dataset: str,
    external_dir: Path,
) -> tuple[dict[str, str], dict[str, object]]:
    names = {
        "astex": "astex_smiles.json",
        "posebusters": "pb_smiles.json",
        "casf": "casf_smiles.json",
    }
    try:
        mapping_path = external_dir / names[dataset]
    except KeyError as exc:
        raise ValueError(f"unsupported benchmark dataset: {dataset}") from exc
    raw = json.loads(mapping_path.read_text())
    membership: dict[str, object] | None = None
    if dataset == "posebusters":
        membership_path = external_dir / "posebusters_v2_ids.txt"
        keep = {
            line.strip().lower()
            for line in membership_path.read_text().splitlines()
            if line.strip()
        }
        raw = {key: value for key, value in raw.items() if key.lower() in keep}
        membership = {
            "path": str(membership_path),
            "sha256": file_sha256(membership_path),
        }
    mapping = {
        str(key).lower(): str(value["smiles"] if isinstance(value, dict) else value)
        for key, value in raw.items()
    }
    sources: dict[str, object] = {
        "mapping": {"path": str(mapping_path), "sha256": file_sha256(mapping_path)}
    }
    if membership is not None:
        sources["membership"] = membership
    return mapping, sources


def _frozen_mapping(
    dataset: str,
    manifest_path: Path,
) -> tuple[dict[str, str], dict[str, object]]:
    raw = json.loads(manifest_path.read_text())
    if raw.get("schema_version") != BENCHMARK_INPUT_MANIFEST_SCHEMA:
        raise ValueError(f"unsupported benchmark input manifest: {manifest_path}")
    if raw.get("protocol_id") != BENCHMARK_INPUT_PROTOCOL_ID:
        raise ValueError(
            "benchmark input manifest protocol mismatch: "
            f"expected {BENCHMARK_INPUT_PROTOCOL_ID!r}, got {raw.get('protocol_id')!r}"
        )
    dataset_raw = raw.get("datasets", {}).get(dataset)
    if not isinstance(dataset_raw, dict):
        raise ValueError(f"benchmark input manifest has no {dataset!r} dataset")
    ligands = dataset_raw.get("ligands")
    if not isinstance(ligands, dict) or not ligands:
        raise ValueError(f"benchmark input manifest {dataset!r} ligands are missing")
    mapping: dict[str, str] = {}
    for complex_id, record in ligands.items():
        if not isinstance(record, dict) or not isinstance(record.get("smiles"), str):
            raise ValueError(f"invalid frozen benchmark ligand record: {dataset}/{complex_id}")
        normalized_id = str(complex_id).lower()
        if normalized_id != complex_id or normalized_id in mapping:
            raise ValueError(f"benchmark IDs must be unique lowercase strings: {complex_id!r}")
        mapping[normalized_id] = record["smiles"]
        actual = ligand_input_identity(normalized_id, record["smiles"])
        declared = record.get("input_identity")
        if declared != actual:
            raise ValueError(
                f"frozen benchmark ligand identity mismatch: {dataset}/{normalized_id}"
            )
    ids = sorted(mapping)
    if int(dataset_raw.get("count", -1)) != len(ids):
        raise ValueError(f"frozen benchmark count mismatch: {dataset}")
    if dataset_raw.get("ids_sha256") != sorted_id_sha256(ids):
        raise ValueError(f"frozen benchmark ID hash mismatch: {dataset}")
    if dataset_raw.get("mapping_sha256") != mapping_sha256(dataset, mapping):
        raise ValueError(f"frozen benchmark mapping hash mismatch: {dataset}")
    sources = {
        "frozen_manifest": {
            "path": str(manifest_path),
            "sha256": file_sha256(manifest_path),
        },
        "source_manifests": dataset_raw.get("source_manifests", {}),
        "integrity_boundary": dataset_raw.get("integrity_boundary", {}),
    }
    return mapping, sources


def load_benchmark_inputs(
    dataset: str,
    external_dir: Path,
    manifest_path: Path | None = None,
) -> tuple[dict[str, str], dict[str, object]]:
    """Load one exact benchmark mapping and its content-addressed identity."""
    if manifest_path is None:
        mapping, sources = _legacy_mapping(dataset, external_dir)
        mode = "legacy_external_mapping"
    else:
        mapping, sources = _frozen_mapping(dataset, manifest_path)
        mode = "frozen_manifest"
    ids = sorted(mapping)
    per_id = {
        complex_id: ligand_input_identity(complex_id, mapping[complex_id])
        for complex_id in ids
    }
    payload: dict[str, Any] = {
        "schema_version": BENCHMARK_INPUT_IDENTITY_SCHEMA,
        "mode": mode,
        "dataset": dataset,
        "heavy_atom_policy": (
            HEAVY_ATOM_POLICY if mode == "frozen_manifest" else "generic_inference_loader"
        ),
        "count": len(ids),
        "ids_sha256": sorted_id_sha256(ids),
        "mapping_sha256": mapping_sha256(dataset, mapping),
        "sources": sources,
        "per_id": per_id,
    }
    stable = dict(payload)
    stable_sources = json.loads(json.dumps(sources))
    if isinstance(stable_sources.get("frozen_manifest"), dict):
        stable_sources["frozen_manifest"].pop("path", None)
    stable["sources"] = stable_sources
    canonical = json.dumps(stable, separators=(",", ":"), sort_keys=True).encode("utf-8")
    payload["sha256"] = hashlib.sha256(
        b"EFFDOCK_BENCHMARK_INPUT_IDENTITY_V1\0" + canonical
    ).hexdigest()
    return mapping, payload


def load_benchmark_ligand(smiles: str, *, random_seed: int) -> tuple[Chem.Mol, bool]:
    """Seeded benchmark-only loader enforcing the frozen heavy-atom contract."""
    mol, has_pose = load_generic_ligand(smiles, random_seed=random_seed)
    mol = Chem.RemoveAllHs(mol)
    hydrogen_count = sum(atom.GetAtomicNum() == 1 for atom in mol.GetAtoms())
    if hydrogen_count:
        raise BenchmarkInputMismatchError(
            "benchmark_heavy_atom_normalization_failed",
            "benchmark ligand contains explicit hydrogens after normalization",
            details={"hydrogen_count": hydrogen_count},
        )
    if mol.GetNumConformers() != 1 or not mol.GetConformer().Is3D():
        raise BenchmarkInputMismatchError(
            "benchmark_conformer_missing",
            "benchmark ligand normalization did not preserve one 3D conformer",
            details={"conformer_count": mol.GetNumConformers()},
        )
    return mol, has_pose


def full_heavy_atom_mapping_metadata(
    mol_ref: Chem.Mol,
    mol_input: Chem.Mol,
    dock_indices: list[int],
    ref_indices: list[int],
    method: str,
) -> dict[str, object]:
    """Validate one complete element/connectivity-preserving atom bijection.

    Formal-charge, aromatic, and bond-order representations may differ across
    benchmark sources (for example, equivalent tautomer/kekulization forms),
    but constitutional connectivity may not.  This is stricter than accepting
    an arbitrary full-atom MCS and broader than requiring identical bond labels.
    """
    input_count = mol_input.GetNumAtoms()
    reference_count = mol_ref.GetNumAtoms()
    full_bijection = bool(
        input_count == reference_count == len(dock_indices) == len(ref_indices)
        and sorted(dock_indices) == list(range(input_count))
        and sorted(ref_indices) == list(range(reference_count))
    )
    atom_elements_match = False
    connectivity_match = False
    bond_orders_match = False
    formal_charges_match = False
    if full_bijection:
        input_to_ref = dict(zip(dock_indices, ref_indices, strict=True))
        atom_elements_match = all(
            mol_input.GetAtomWithIdx(dock_index).GetAtomicNum()
            == mol_ref.GetAtomWithIdx(ref_index).GetAtomicNum()
            for dock_index, ref_index in input_to_ref.items()
        )
        input_edges = {
            tuple(
                sorted(
                    (
                        input_to_ref[bond.GetBeginAtomIdx()],
                        input_to_ref[bond.GetEndAtomIdx()],
                    )
                )
            )
            for bond in mol_input.GetBonds()
        }
        reference_edges = {
            tuple(sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())))
            for bond in mol_ref.GetBonds()
        }
        connectivity_match = input_edges == reference_edges
        if connectivity_match:
            bond_orders_match = all(
                float(bond.GetBondTypeAsDouble())
                == float(
                    mol_ref.GetBondBetweenAtoms(
                        input_to_ref[bond.GetBeginAtomIdx()],
                        input_to_ref[bond.GetEndAtomIdx()],
                    ).GetBondTypeAsDouble()
                )
                for bond in mol_input.GetBonds()
            )
        formal_charges_match = all(
            mol_input.GetAtomWithIdx(dock_index).GetFormalCharge()
            == mol_ref.GetAtomWithIdx(ref_index).GetFormalCharge()
            for dock_index, ref_index in input_to_ref.items()
        )
    accepted = bool(full_bijection and atom_elements_match and connectivity_match)
    if not accepted:
        relation = "incomplete_or_connectivity_mismatch"
    elif bond_orders_match and formal_charges_match:
        relation = "exact_graph"
    else:
        relation = "same_connectivity_representation_mismatch"
    return {
        "accepted": accepted,
        "relation": relation,
        "match_method": method,
        "matched_atoms": len(dock_indices),
        "input_atoms": input_count,
        "reference_atoms": reference_count,
        "full_bijection": full_bijection,
        "atom_elements_match": atom_elements_match,
        "connectivity_match": connectivity_match,
        "bond_orders_match": bond_orders_match,
        "formal_charges_match": formal_charges_match,
    }


__all__ = [
    "BENCHMARK_INPUT_MANIFEST_SCHEMA",
    "BENCHMARK_INPUT_IDENTITY_SCHEMA",
    "BENCHMARK_INPUT_PROTOCOL_ID",
    "HEAVY_ATOM_POLICY",
    "BenchmarkInputMismatchError",
    "canonical_heavy_smiles",
    "file_sha256",
    "full_heavy_atom_mapping_metadata",
    "ligand_input_identity",
    "load_benchmark_inputs",
    "load_benchmark_ligand",
    "mapping_sha256",
    "sorted_id_sha256",
]
