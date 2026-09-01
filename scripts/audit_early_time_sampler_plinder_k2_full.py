#!/usr/bin/env python3
"""Independent coordinate-level audit of the full PLINDER K2 gate.

This auditor intentionally does not import the sampler evaluator or its report
aggregator.  It reads the published full-run artifacts, reparses every retained
multi-record SDF sequentially, and recomputes RMSD, diversity, paired outcomes,
the frozen cluster bootstrap, and the final decision from saved coordinates.

The ordered candidate-ensemble digest was made from pre-serialization float32
coordinates.  SDF stores coordinates at lower precision, so that digest cannot
be regenerated from the retained file.  Instead, the auditor verifies the SDF
file digest and requires every record to carry the same ensemble digest as its
CSV row.  All coordinate-derived quantities are recomputed independently.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import rdDistGeom, rdFMCS, rdMolAlign

PROTOCOL_ID = "EFFDOCK-EARLY-TIME-SAMPLER-PLINDER-K2-GATE-V1"
ELIGIBILITY_SCHEMA = "effdock.plinder_checkpoint_eligibility.v1"
SHARD_SCHEMA = "effdock.plinder_checkpoint_paired_shard.v1"
AUDIT_SCHEMA = "effdock.plinder_checkpoint_full_coordinate_audit.v1"

ARMS = (
    "s25_ema",
    "s50_ema",
    "parent50k_plus10k_t0p10_ema",
)
BASELINE_ARM = "s50_ema"
TREATMENT_ARM = "parent50k_plus10k_t0p10_ema"
CHECKPOINT_SHA256 = {
    "s25_ema": "c343ebc34cea3395762cd82e1c54b8c7b847dc04c4fa9e80b9813a864cafa0e1",
    "s50_ema": "65be44d7dc8f0867eb9fc5d22214b80f93971ea4702679a527c665046e91e6b6",
    "parent50k_plus10k_t0p10_ema": (
        "0a48577379e286c584abd8c652d079b09dd6fff3c06a1a2f433d617ab0cd6074"
    ),
}

EXPECTED_SHARDS = 8
EXPECTED_FULL_COUNT = 1076
EXPECTED_ELIGIBLE_COUNT = 1035
EXPECTED_EXCLUDED_COUNT = 41
EXPECTED_SYSTEM_COUNT = 1020
EXPECTED_NUM_SAMPLES = 100
EXPECTED_NUM_STEPS = 10
EXPECTED_PRIOR_POOL_SIZE = 100
EXPECTED_ELIGIBLE_NEWLINE_SHA256 = (
    "005577bbf2b0c1c1e98bac3092b8e5350a6aa06597442b4c86d05f24e763593f"
)
BASE_SEED = 42
CONFORMER_SEED = 0
POSE_DIVERSITY_CONTRACT = "EFFDOCK_HEAVY_ATOM_RECEPTOR_FRAME_DIVERSITY_V2"
POSE_DIVERSITY_ROUND_DECIMALS = 3
BOOTSTRAP_SEED = 20260815
BOOTSTRAP_RESAMPLES = 20_000
RMSD_SDF_ABS_TOL = 5e-4
DIVERSITY_SDF_ABS_TOL = 5e-4
HEX64_RE = re.compile(r"[0-9a-f]{64}\Z")

_VDW = {
    1: 1.1,
    6: 1.7,
    7: 1.55,
    8: 1.52,
    9: 1.47,
    11: 1.6,
    12: 1.6,
    15: 1.8,
    16: 1.8,
    17: 1.75,
    19: 1.7,
    20: 1.7,
    25: 1.6,
    26: 1.6,
    30: 1.39,
    34: 1.9,
    35: 1.85,
    53: 1.98,
}
_SYMBOL_TO_Z = {
    "C": 6,
    "N": 7,
    "O": 8,
    "S": 16,
    "P": 15,
    "F": 9,
    "CL": 17,
    "BR": 35,
    "I": 53,
    "SE": 34,
    "ZN": 30,
    "MG": 12,
    "MN": 25,
    "FE": 26,
    "CA": 20,
    "NA": 11,
    "K": 19,
}
_WATER = {"HOH", "WAT", "DOD", "H2O"}
_NUCLEIC = {
    "A",
    "C",
    "G",
    "U",
    "DA",
    "DC",
    "DG",
    "DT",
    "ADE",
    "CYT",
    "GUA",
    "THY",
    "URA",
}


class AuditError(RuntimeError):
    """A fail-closed audit contract violation."""


@dataclass(frozen=True)
class RowAudit:
    sample_id: str
    system_id: str
    arm: str
    global_index: int
    sampling_seed: int
    k2: int
    declared_k2: int
    fast_valid_k2: int
    declared_fast_valid_k2: int
    fast_valid_count: int
    first_rmsd: float
    declared_first_rmsd: float
    oracle_rmsd: float
    declared_oracle_rmsd: float
    coordinate_unique_count: int
    nearest_neighbor_rmsd_median: float
    declared_nearest_neighbor_rmsd_median: float
    c2_component_count: int
    declared_c2_component_count: int
    pair_identity: tuple[str, ...]
    all_poses_sha256: str
    fast_valid_recheck_candidates: int
    fast_valid_recheck_mismatches: int
    fast_valid_recheck_mismatch_indices: tuple[int, ...]
    quantization_ambiguous_candidates: tuple[tuple[int, float, float], ...]
    quantization_ambiguous_edges: tuple[tuple[int, int, float], ...]


class FileHashCache:
    def __init__(self) -> None:
        self._values: dict[Path, tuple[int, int, str]] = {}

    def sha256(self, path: Path) -> str:
        path = path.resolve(strict=True)
        stat = path.stat()
        cached = self._values.get(path)
        identity = (int(stat.st_size), int(stat.st_mtime_ns))
        if cached is not None and cached[:2] == identity:
            return cached[2]
        value = file_sha256(path)
        self._values[path] = (*identity, value)
        return value


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise AuditError(f"{label}: missing JSON file: {path}")
    try:
        payload = json.loads(path.read_text())
    except Exception as exc:
        raise AuditError(f"{label}: invalid JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise AuditError(f"{label}: JSON root must be an object: {path}")
    return payload


def _require_hex64(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or HEX64_RE.fullmatch(value) is None:
        raise AuditError(f"{label}: expected lowercase SHA-256")
    return value


def _strict_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool):
        raise AuditError(f"{label}: booleans are not integers")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise AuditError(f"{label}: expected integer, got {value!r}") from exc
    if isinstance(value, float) and not value.is_integer():
        raise AuditError(f"{label}: non-integral float")
    if isinstance(value, str) and value.strip() != str(parsed):
        raise AuditError(f"{label}: non-canonical integer {value!r}")
    return parsed


def _finite_float(value: Any, *, label: str) -> float:
    if isinstance(value, bool):
        raise AuditError(f"{label}: booleans are not numeric")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise AuditError(f"{label}: expected finite float, got {value!r}") from exc
    if not math.isfinite(parsed):
        raise AuditError(f"{label}: non-finite value")
    return parsed


def _strict_bool(value: Any, *, label: str) -> bool:
    if value is True or value == "True" or value == "true":
        return True
    if value is False or value == "False" or value == "false":
        return False
    raise AuditError(f"{label}: expected boolean, got {value!r}")


def _parse_json_list(value: Any, *, label: str) -> list[Any]:
    if not isinstance(value, str):
        raise AuditError(f"{label}: expected serialized JSON list")
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise AuditError(f"{label}: invalid JSON") from exc
    if not isinstance(parsed, list):
        raise AuditError(f"{label}: expected JSON list")
    return parsed


def _parse_json_object(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, str):
        raise AuditError(f"{label}: expected serialized JSON object")
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise AuditError(f"{label}: invalid JSON") from exc
    if not isinstance(parsed, dict):
        raise AuditError(f"{label}: expected JSON object")
    return parsed


def _versioned_ids_sha256(ids: Sequence[str]) -> str:
    digest = hashlib.sha256(b"EFFDOCK_SORTED_COMPLEX_IDS_V1\0")
    for sample_id in ids:
        digest.update(sample_id.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _newline_ids_sha256(ids: Sequence[str]) -> str:
    return hashlib.sha256("".join(f"{sample_id}\n" for sample_id in ids).encode()).hexdigest()


def _canonical_smiles_identity(smiles: str) -> str:
    digest = hashlib.sha256(b"EFFDOCK_PLINDER_CANONICAL_SMILES_V1\0")
    digest.update(smiles.encode("utf-8"))
    return digest.hexdigest()


def _split_sample_key(sample_id: str) -> tuple[str, str]:
    if "__" not in sample_id:
        raise AuditError(f"invalid sample key: {sample_id!r}")
    system_id, chain = sample_id.rsplit("__", 1)
    for label, value in (("system", system_id), ("chain", chain)):
        if not value or value in {".", ".."} or Path(value).name != value or "\\" in value:
            raise AuditError(f"unsafe sample {label}: {value!r}")
    return system_id, chain


def _require_file_identity(
    path_value: Any,
    sha_value: Any,
    *,
    label: str,
    hashes: FileHashCache,
    expected_path: Path | None = None,
    expected_sha256: str | None = None,
) -> Path:
    if not isinstance(path_value, str) or not path_value:
        raise AuditError(f"{label}: missing path")
    path = Path(path_value)
    if not path.is_absolute():
        raise AuditError(f"{label}: path must be absolute: {path}")
    if ".incomplete" in path.parts:
        raise AuditError(f"{label}: published path points into .incomplete")
    if path.is_symlink():
        raise AuditError(f"{label}: symlink artifacts are forbidden: {path}")
    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError as exc:
        raise AuditError(f"{label}: missing file: {path}") from exc
    if not resolved.is_file():
        raise AuditError(f"{label}: not a regular file: {resolved}")
    if path_value != str(resolved):
        raise AuditError(f"{label}: path is not lexical canonical target: {path_value}")
    if expected_path is not None:
        canonical_expected = expected_path.resolve(strict=True)
        if resolved != canonical_expected:
            raise AuditError(f"{label}: non-canonical lexical path: {path_value}")
    declared = _require_hex64(sha_value, label=f"{label}.sha256")
    if expected_sha256 is not None and declared != expected_sha256:
        raise AuditError(f"{label}: wrong frozen SHA-256")
    actual = hashes.sha256(resolved)
    if actual != declared:
        raise AuditError(f"{label}: file SHA-256 mismatch")
    return resolved


def _require_published_path(
    path_value: Any,
    expected_path: Path,
    *,
    label: str,
) -> Path:
    """Require an exact durable published-file path, not a resolving alias."""
    if not isinstance(path_value, str) or not path_value:
        raise AuditError(f"{label}: missing path")
    path = Path(path_value)
    if not path.is_absolute():
        raise AuditError(f"{label}: path must be absolute: {path}")
    if ".incomplete" in path.parts:
        raise AuditError(f"{label}: published path points into .incomplete")
    if path.is_symlink():
        raise AuditError(f"{label}: symlink artifacts are forbidden: {path}")
    try:
        resolved = path.resolve(strict=True)
        canonical_expected = expected_path.resolve(strict=True)
    except FileNotFoundError as exc:
        raise AuditError(f"{label}: missing published file") from exc
    if not resolved.is_file():
        raise AuditError(f"{label}: not a regular file: {resolved}")
    if path_value != str(resolved) or resolved != canonical_expected:
        raise AuditError(f"{label}: path is not the lexical canonical target")
    return resolved


def _verify_asset_record(
    record: Any,
    *,
    label: str,
    hashes: FileHashCache,
) -> None:
    if not isinstance(record, dict):
        raise AuditError(f"{label}: expected identity object")
    path = _require_file_identity(
        record.get("path"), record.get("sha256"), label=label, hashes=hashes
    )
    if "size_bytes" in record and _strict_int(
        record["size_bytes"], label=f"{label}.size_bytes"
    ) != path.stat().st_size:
        raise AuditError(f"{label}: file size mismatch")


def _verify_fixed_identities(payload: Any, hashes: FileHashCache) -> None:
    if not isinstance(payload, dict):
        raise AuditError("eligibility fixed identities are missing")
    for key in (
        "protocol_document",
        "split",
        "pool_parquet",
        "config",
        "raw_gate",
        "conformer_mapping_audit",
    ):
        _verify_asset_record(payload.get(key), label=f"fixed_identities.{key}", hashes=hashes)
    checkpoints = payload.get("checkpoints")
    if not isinstance(checkpoints, dict) or set(checkpoints) != set(ARMS):
        raise AuditError("fixed checkpoint identity inventory mismatch")
    for arm in ARMS:
        record = checkpoints[arm]
        _verify_asset_record(record, label=f"fixed_identities.checkpoints.{arm}", hashes=hashes)
        if record.get("sha256") != CHECKPOINT_SHA256[arm]:
            raise AuditError(f"{arm}: fixed checkpoint digest mismatch")
    code = payload.get("code")
    if not isinstance(code, dict) or code.get("contract") != "EFFDOCK_PLINDER_PAIRED_CODE_INVENTORY_V1":
        raise AuditError("fixed code inventory contract mismatch")
    files = code.get("files")
    if not isinstance(files, dict) or not files:
        raise AuditError("fixed code inventory is empty")
    code_hashes: dict[str, str] = {}
    for name, record in sorted(files.items()):
        _verify_asset_record(record, label=f"fixed_identities.code.{name}", hashes=hashes)
        code_hashes[str(name)] = str(record["sha256"])
    digest = hashlib.sha256(b"EFFDOCK_PLINDER_PAIRED_CODE_INVENTORY_V1\0")
    digest.update(json.dumps(code_hashes, separators=(",", ":"), sort_keys=True).encode())
    if code.get("sha256") != digest.hexdigest():
        raise AuditError("fixed code aggregate digest mismatch")


def _load_eligibility(
    path: Path,
    expected_sha256: str,
    *,
    hashes: FileHashCache,
) -> tuple[dict[str, Any], list[str], list[str], dict[str, dict[str, Any]]]:
    expected_sha256 = _require_hex64(expected_sha256, label="expected eligibility SHA-256")
    actual_sha256 = hashes.sha256(path)
    if actual_sha256 != expected_sha256:
        raise AuditError("eligibility manifest SHA-256 differs from the launch identity")
    payload = _load_json(path, label="eligibility manifest")
    if (
        payload.get("schema_version") != ELIGIBILITY_SCHEMA
        or payload.get("protocol_id") != PROTOCOL_ID
        or payload.get("status") != "complete"
    ):
        raise AuditError("eligibility manifest contract/status mismatch")
    inventory = payload.get("inventory")
    if not isinstance(inventory, dict):
        raise AuditError("eligibility inventory is missing")
    full_ids = inventory.get("full_ids")
    eligible_ids = inventory.get("eligible_ids")
    excluded_ids = inventory.get("excluded_ids")
    if not all(isinstance(values, list) for values in (full_ids, eligible_ids, excluded_ids)):
        raise AuditError("eligibility ID lists are missing")
    full_ids = [str(value) for value in full_ids]
    eligible_ids = [str(value) for value in eligible_ids]
    excluded_ids = [str(value) for value in excluded_ids]
    for label, values in (
        ("full", full_ids),
        ("eligible", eligible_ids),
        ("excluded", excluded_ids),
    ):
        if values != sorted(values) or len(values) != len(set(values)):
            raise AuditError(f"{label} IDs must be sorted and unique")
    if (
        len(full_ids) != EXPECTED_FULL_COUNT
        or len(eligible_ids) != EXPECTED_ELIGIBLE_COUNT
        or len(excluded_ids) != EXPECTED_EXCLUDED_COUNT
        or sorted(eligible_ids + excluded_ids) != full_ids
    ):
        raise AuditError("eligibility exact cohort accounting mismatch")
    if len({_split_sample_key(value)[0] for value in eligible_ids}) != EXPECTED_SYSTEM_COUNT:
        raise AuditError("eligibility system count mismatch")
    declared_counts = {
        "full_count": EXPECTED_FULL_COUNT,
        "eligible_count": EXPECTED_ELIGIBLE_COUNT,
        "excluded_count": EXPECTED_EXCLUDED_COUNT,
        "eligible_system_count": EXPECTED_SYSTEM_COUNT,
        "preflight_error_count": 0,
    }
    for field, expected in declared_counts.items():
        if _strict_int(inventory.get(field), label=f"inventory.{field}") != expected:
            raise AuditError(f"eligibility {field} mismatch")
    if inventory.get("preflight_error_ids") != []:
        raise AuditError("eligibility manifest contains preflight errors")
    if inventory.get("full_ids_sha256") != _versioned_ids_sha256(full_ids):
        raise AuditError("full ID digest mismatch")
    if inventory.get("eligible_ids_sha256") != _versioned_ids_sha256(eligible_ids):
        raise AuditError("eligible versioned ID digest mismatch")
    newline_digest = _newline_ids_sha256(eligible_ids)
    if (
        inventory.get("eligible_ids_newline_sha256") != newline_digest
        or newline_digest != EXPECTED_ELIGIBLE_NEWLINE_SHA256
    ):
        raise AuditError("eligible newline ID digest mismatch")
    if inventory.get("excluded_ids_sha256") != _versioned_ids_sha256(excluded_ids):
        raise AuditError("excluded ID digest mismatch")
    records = payload.get("records")
    if not isinstance(records, list) or len(records) != EXPECTED_FULL_COUNT:
        raise AuditError("eligibility record inventory mismatch")
    if [record.get("sample_key") for record in records] != full_ids:
        raise AuditError("eligibility records are not in full global order")
    by_id: dict[str, dict[str, Any]] = {}
    eligible_set = set(eligible_ids)
    for global_index, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            raise AuditError("eligibility record must be an object")
        sample_id = full_ids[global_index - 1]
        expected_status = "eligible" if sample_id in eligible_set else "excluded"
        if record.get("status") != expected_status:
            raise AuditError(f"{sample_id}: eligibility status mismatch")
        if _strict_int(record.get("global_index"), label=f"{sample_id}.global_index") != global_index:
            raise AuditError(f"{sample_id}: eligibility global index mismatch")
        if _strict_int(record.get("sampling_seed"), label=f"{sample_id}.sampling_seed") != BASE_SEED + global_index:
            raise AuditError(f"{sample_id}: eligibility sampling seed mismatch")
        if _strict_int(record.get("ligand_conformer_seed"), label=f"{sample_id}.conformer_seed") != CONFORMER_SEED:
            raise AuditError(f"{sample_id}: eligibility conformer seed mismatch")
        if sample_id in eligible_set:
            canonical_smiles = record.get("canonical_smiles")
            if not isinstance(canonical_smiles, str) or not canonical_smiles:
                raise AuditError(f"{sample_id}: eligible record lacks canonical SMILES")
            if record.get("canonical_smiles_identity_sha256") != _canonical_smiles_identity(
                canonical_smiles
            ):
                raise AuditError(f"{sample_id}: canonical-SMILES identity mismatch")
            if record.get("audit_mapping_method") != "strict_stereo":
                raise AuditError(f"{sample_id}: eligibility mapping is not strict_stereo")
            if record.get("audit_symmetry_complete") is not True:
                raise AuditError(f"{sample_id}: eligibility symmetry audit is incomplete")
        by_id[sample_id] = record
    inputs = payload.get("inputs")
    if not isinstance(inputs, dict):
        raise AuditError("eligibility inputs are missing")
    _verify_fixed_identities(inputs.get("fixed_identities"), hashes)
    return payload, eligible_ids, excluded_ids, by_id


def _strip_charges(mol: Chem.Mol) -> Chem.Mol:
    copy = Chem.RWMol(mol)
    for atom in copy.GetAtoms():
        atom.SetFormalCharge(0)
        atom.SetNumRadicalElectrons(0)
    try:
        Chem.SanitizeMol(
            copy,
            sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL
            ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE,
        )
    except Exception:
        pass
    return copy


def _match_atoms(
    mol_ref: Chem.Mol, mol_pose: Chem.Mol
) -> tuple[list[int], list[int], str]:
    match = mol_ref.GetSubstructMatch(mol_pose)
    if len(match) == mol_pose.GetNumAtoms():
        return list(range(mol_pose.GetNumAtoms())), list(match), "strict"
    ref_no_charge, pose_no_charge = _strip_charges(mol_ref), _strip_charges(mol_pose)
    match = ref_no_charge.GetSubstructMatch(pose_no_charge)
    if len(match) == pose_no_charge.GetNumAtoms():
        return list(range(mol_pose.GetNumAtoms())), list(match), "nocharges"
    mcs = rdFMCS.FindMCS(
        [ref_no_charge, pose_no_charge],
        timeout=5,
        atomCompare=rdFMCS.AtomCompare.CompareElements,
        bondCompare=rdFMCS.BondCompare.CompareAny,
        matchValences=False,
        ringMatchesRingOnly=False,
    )
    if mcs.numAtoms == 0:
        return [], [], "fail"
    pattern = Chem.MolFromSmarts(mcs.smartsString)
    if pattern is None:
        return [], [], "fail"
    ref_match = ref_no_charge.GetSubstructMatch(pattern)
    pose_match = pose_no_charge.GetSubstructMatch(pattern)
    if len(ref_match) == len(pose_match) == mcs.numAtoms:
        return list(pose_match), list(ref_match), f"mcs({mcs.numAtoms}/{mol_pose.GetNumAtoms()})"
    return [], [], "fail"


def _full_heavy_atom_graph_metadata(
    mol_ref: Chem.Mol,
    mol_pose: Chem.Mol,
    dock_indices: list[int],
    ref_indices: list[int],
    method: str,
) -> dict[str, object]:
    """Independently enforce a complete element/connectivity-preserving map.

    Bond-order and formal-charge representations may differ, but the mapped
    heavy-atom elements and constitutional edge set must be identical.
    """
    pose_count = mol_pose.GetNumAtoms()
    reference_count = mol_ref.GetNumAtoms()
    full_bijection = bool(
        pose_count == reference_count == len(dock_indices) == len(ref_indices)
        and sorted(dock_indices) == list(range(pose_count))
        and sorted(ref_indices) == list(range(reference_count))
    )
    atom_elements_match = False
    connectivity_match = False
    bond_orders_match = False
    formal_charges_match = False
    if full_bijection:
        pose_to_ref = dict(zip(dock_indices, ref_indices, strict=True))
        atom_elements_match = all(
            mol_pose.GetAtomWithIdx(pose_index).GetAtomicNum()
            == mol_ref.GetAtomWithIdx(ref_index).GetAtomicNum()
            for pose_index, ref_index in pose_to_ref.items()
        )
        pose_edges = {
            tuple(
                sorted(
                    (
                        pose_to_ref[bond.GetBeginAtomIdx()],
                        pose_to_ref[bond.GetEndAtomIdx()],
                    )
                )
            )
            for bond in mol_pose.GetBonds()
        }
        reference_edges = {
            tuple(sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())))
            for bond in mol_ref.GetBonds()
        }
        connectivity_match = pose_edges == reference_edges
        if connectivity_match:
            bond_orders_match = all(
                float(bond.GetBondTypeAsDouble())
                == float(
                    mol_ref.GetBondBetweenAtoms(
                        pose_to_ref[bond.GetBeginAtomIdx()],
                        pose_to_ref[bond.GetEndAtomIdx()],
                    ).GetBondTypeAsDouble()
                )
                for bond in mol_pose.GetBonds()
            )
        formal_charges_match = all(
            mol_pose.GetAtomWithIdx(pose_index).GetFormalCharge()
            == mol_ref.GetAtomWithIdx(ref_index).GetFormalCharge()
            for pose_index, ref_index in pose_to_ref.items()
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
        "input_atoms": pose_count,
        "reference_atoms": reference_count,
        "full_bijection": full_bijection,
        "atom_elements_match": atom_elements_match,
        "connectivity_match": connectivity_match,
        "bond_orders_match": bond_orders_match,
        "formal_charges_match": formal_charges_match,
    }


def _matches_frozen_canonical_graph(canonical: Chem.Mol, pose: Chem.Mol) -> bool:
    """Match exact non-stereo chemistry of a generated pose to frozen input.

    Generated 3D coordinates can invert a tetrahedral center or E/Z geometry;
    RDKit then derives different stereo tags when it reads the SDF.  Those tags
    describe a pose outcome, not an atom-order/topology mutation.  Input/reference
    stereo integrity is already frozen by the eligibility strict-stereo audit.
    Here bond order, formal charge, isotope, hydrogen/radical state, and aromatic
    chemistry remain identity-bearing while pose-derived stereo is ignored.
    """
    query = Chem.RemoveAllHs(Chem.Mol(canonical))
    target = Chem.RemoveAllHs(Chem.Mol(pose))
    if query.GetNumAtoms() != target.GetNumAtoms() or query.GetNumBonds() != target.GetNumBonds():
        return False
    matches = target.GetSubstructMatches(
        query,
        uniquify=False,
        useChirality=False,
        maxMatches=100_000,
    )
    for match in matches:
        if len(match) != query.GetNumAtoms():
            continue
        if any(
            query.GetAtomWithIdx(query_index).GetAtomicNum()
            != target.GetAtomWithIdx(target_index).GetAtomicNum()
            or query.GetAtomWithIdx(query_index).GetFormalCharge()
            != target.GetAtomWithIdx(target_index).GetFormalCharge()
            or query.GetAtomWithIdx(query_index).GetIsotope()
            != target.GetAtomWithIdx(target_index).GetIsotope()
            or query.GetAtomWithIdx(query_index).GetTotalNumHs(includeNeighbors=True)
            != target.GetAtomWithIdx(target_index).GetTotalNumHs(includeNeighbors=True)
            or query.GetAtomWithIdx(query_index).GetNumRadicalElectrons()
            != target.GetAtomWithIdx(target_index).GetNumRadicalElectrons()
            for query_index, target_index in enumerate(match)
        ):
            continue
        representation_matches = True
        for query_bond in query.GetBonds():
            target_bond = target.GetBondBetweenAtoms(
                match[query_bond.GetBeginAtomIdx()],
                match[query_bond.GetEndAtomIdx()],
            )
            if target_bond is None or not (
                query_bond.GetIsAromatic() and target_bond.GetIsAromatic()
                or query_bond.GetBondType() == target_bond.GetBondType()
            ):
                representation_matches = False
                break
        if representation_matches:
            return True
    return False


def _load_reference(path: Path) -> Chem.Mol:
    supplier = Chem.SDMolSupplier(str(path), sanitize=True, removeHs=True)
    mol = next(supplier, None)
    if mol is not None:
        fragments = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
        if len(fragments) > 1:
            mol = max(fragments, key=lambda value: value.GetNumAtoms())
        if mol.GetNumConformers() != 1:
            raise AuditError(f"reference must have one conformer: {path}")
        return mol
    supplier = Chem.SDMolSupplier(str(path), sanitize=False, removeHs=False)
    mol = next(supplier, None)
    if mol is None:
        raise AuditError(f"RDKit cannot parse ligand reference: {path}")
    try:
        mol.UpdatePropertyCache(strict=False)
        Chem.FastFindRings(mol)
        relaxed = (
            Chem.SanitizeFlags.SANITIZE_FINDRADICALS
            | Chem.SanitizeFlags.SANITIZE_SETAROMATICITY
            | Chem.SanitizeFlags.SANITIZE_SETCONJUGATION
            | Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION
            | Chem.SanitizeFlags.SANITIZE_SYMMRINGS
        )
        Chem.SanitizeMol(mol, sanitizeOps=relaxed)
        mol = Chem.RemoveHs(mol, sanitize=False)
    except Exception as exc:
        raise AuditError(f"failed relaxed reference parse: {path}: {exc}") from exc
    fragments = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    if len(fragments) > 1:
        mol = max(fragments, key=lambda value: value.GetNumAtoms())
    if mol.GetNumConformers() != 1:
        raise AuditError(f"reference must have one conformer: {path}")
    return mol


def _coordinates(mol: Chem.Mol, *, label: str) -> np.ndarray:
    if mol.GetNumConformers() != 1:
        raise AuditError(f"{label}: expected exactly one conformer")
    coordinates = np.asarray(mol.GetConformer().GetPositions(), dtype=np.float64)
    if coordinates.shape != (mol.GetNumAtoms(), 3) or not np.isfinite(coordinates).all():
        raise AuditError(f"{label}: invalid coordinates")
    return coordinates


def _topology_signature(mol: Chem.Mol) -> tuple[Any, ...]:
    """Ordered non-stereo chemical graph signature for retained pose records."""
    atoms = tuple(
        (
            atom.GetAtomicNum(),
            atom.GetFormalCharge(),
            atom.GetIsAromatic(),
            atom.GetIsotope(),
            atom.GetTotalNumHs(includeNeighbors=True),
            atom.GetNumRadicalElectrons(),
        )
        for atom in mol.GetAtoms()
    )
    bonds = tuple(
        sorted(
            (
                min(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()),
                max(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()),
                str(bond.GetBondType()),
                bond.GetIsAromatic(),
            )
            for bond in mol.GetBonds()
        )
    )
    return atoms, bonds


def _read_sdf_exact(path: Path, *, expected_count: int, label: str) -> list[Chem.Mol]:
    molecules: list[Chem.Mol] = []
    with path.open("rb") as handle:
        supplier = Chem.ForwardSDMolSupplier(
            handle, sanitize=True, removeHs=True, strictParsing=True
        )
        for index, molecule in enumerate(supplier):
            if molecule is None:
                raise AuditError(f"{label}: SDF record {index} did not parse")
            if index >= expected_count:
                raise AuditError(f"{label}: SDF has more than {expected_count} records")
            molecules.append(molecule)
    if len(molecules) != expected_count:
        raise AuditError(f"{label}: SDF record count {len(molecules)} != {expected_count}")
    return molecules


def _pose_rmsd(
    pose_mol: Chem.Mol,
    ref_mol: Chem.Mol,
    pose_coordinates: np.ndarray,
    ref_coordinates: np.ndarray,
    dock_indices: list[int],
    ref_indices: list[int],
) -> tuple[float, str]:
    if len(dock_indices) == pose_mol.GetNumAtoms() == ref_mol.GetNumAtoms():
        try:
            return (
                float(rdMolAlign.CalcRMS(pose_mol, ref_mol)),
                "rdkit_calc_rms_symmetry_no_align",
            )
        except Exception:
            pass
    if not dock_indices or len(dock_indices) != len(ref_indices):
        raise AuditError("mapped-index RMSD has no valid atom mapping")
    delta = pose_coordinates[np.asarray(dock_indices)] - ref_coordinates[np.asarray(ref_indices)]
    return float(np.sqrt(np.mean(np.sum(delta * delta, axis=1)))), "mapped_index_fallback"


def _diversity_metrics(
    coordinates: np.ndarray, atomic_numbers: np.ndarray
) -> dict[str, int | float]:
    if coordinates.ndim != 3 or coordinates.shape[0] < 1 or coordinates.shape[2] != 3:
        raise AuditError("candidate coordinates must have shape [samples, atoms, 3]")
    if coordinates.shape[1] != len(atomic_numbers) or not np.isfinite(coordinates).all():
        raise AuditError("candidate diversity coordinate/topology mismatch")
    heavy_mask = atomic_numbers > 1
    if not bool(heavy_mask.any()):
        raise AuditError("candidate molecule has no heavy atom")
    heavy = coordinates[:, heavy_mask, :]
    scale = 10**POSE_DIVERSITY_ROUND_DECIMALS
    unique_count = len(
        {
            np.rint(pose * scale).astype("<i8", copy=False).tobytes()
            for pose in heavy
        }
    )
    if len(heavy) == 1:
        return {
            "diversity_heavy_atom_count": int(heavy_mask.sum()),
            "coordinate_unique_count": 1,
            "pairwise_heavy_atom_rmsd_mean": 0.0,
            "pairwise_heavy_atom_rmsd_median": 0.0,
            "pairwise_heavy_atom_rmsd_ge2_fraction": 0.0,
            "nearest_neighbor_heavy_atom_rmsd_median": 0.0,
            "c2_connected_component_count": 1,
        }
    pairwise = _heavy_pairwise_rmsd(heavy)
    upper = pairwise[np.triu_indices(len(heavy), k=1)]
    adjacency = pairwise < 2.0
    unseen = set(range(len(heavy)))
    components = 0
    while unseen:
        components += 1
        stack = [unseen.pop()]
        while stack:
            node = stack.pop()
            neighbors = {
                int(index)
                for index in np.flatnonzero(adjacency[node])
                if int(index) in unseen
            }
            unseen.difference_update(neighbors)
            stack.extend(neighbors)
    np.fill_diagonal(pairwise, np.inf)
    return {
        "diversity_heavy_atom_count": int(heavy_mask.sum()),
        "coordinate_unique_count": unique_count,
        "pairwise_heavy_atom_rmsd_mean": float(np.mean(upper)),
        "pairwise_heavy_atom_rmsd_median": float(np.median(upper)),
        "pairwise_heavy_atom_rmsd_ge2_fraction": float(np.mean(upper >= 2.0)),
        "nearest_neighbor_heavy_atom_rmsd_median": float(
            np.median(np.min(pairwise, axis=1))
        ),
        "c2_connected_component_count": components,
    }


def _heavy_pairwise_rmsd(heavy_coordinates: np.ndarray) -> np.ndarray:
    """Return the same-index receptor-frame RMSD matrix for heavy atoms."""
    delta = heavy_coordinates[:, None, :, :] - heavy_coordinates[None, :, :, :]
    return np.sqrt(np.mean(np.sum(delta * delta, axis=-1), axis=-1))


def _component_count(pairwise: np.ndarray, threshold: float) -> int:
    adjacency = pairwise < threshold
    unseen = set(range(len(pairwise)))
    components = 0
    while unseen:
        components += 1
        stack = [unseen.pop()]
        while stack:
            node = stack.pop()
            neighbors = {
                int(index)
                for index in np.flatnonzero(adjacency[node])
                if int(index) in unseen
            }
            unseen.difference_update(neighbors)
            stack.extend(neighbors)
    return components


def _infer_element(atom_name: str, raw_element: str) -> str:
    element = raw_element.strip().upper()
    if element:
        return element
    cleaned = "".join(character for character in atom_name if character.isalpha()).upper()
    if cleaned[:2] in _SYMBOL_TO_Z:
        return cleaned[:2]
    return cleaned[:1] or "C"


def _protein_atoms(path: Path) -> tuple[np.ndarray, np.ndarray]:
    coordinates: list[tuple[float, float, float]] = []
    atomic_numbers: list[int] = []
    with path.open() as handle:
        for line in handle:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            if len(line) < 54 or line[16] not in {" ", "A"}:
                continue
            residue = line[17:20].strip().upper()
            if residue in _WATER or residue in _NUCLEIC:
                continue
            element = _infer_element(line[12:16].strip(), line[76:78] if len(line) >= 78 else "")
            if element == "H":
                continue
            try:
                xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
            except ValueError:
                continue
            coordinates.append(xyz)
            atomic_numbers.append(_SYMBOL_TO_Z.get(element, 6))
    if not coordinates:
        raise AuditError(f"fast-valid secondary parser found no protein atoms: {path}")
    return np.asarray(coordinates, dtype=np.float64), np.asarray(atomic_numbers, dtype=np.int64)


def _distance_bounds(mol: Chem.Mol) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrix = np.asarray(rdDistGeom.GetMoleculeBoundsMatrix(mol), dtype=np.float64)
    upper = np.triu(matrix)
    upper += upper.T
    lower = np.tril(matrix)
    lower += lower.T
    topology = np.asarray(Chem.GetDistanceMatrix(mol), dtype=np.float64)
    return lower, upper, topology


def _secondary_fast_valid(
    coordinates: np.ndarray,
    mol: Chem.Mol,
    protein_path: Path,
) -> list[bool]:
    protein, protein_z = _protein_atoms(protein_path)
    flattened = coordinates.reshape(-1, 3)
    min_distance = np.full(len(protein), np.inf, dtype=np.float64)
    for start in range(0, len(flattened), 512):
        chunk = flattened[start : start + 512]
        distance = np.sqrt(np.sum((protein[:, None, :] - chunk[None, :, :]) ** 2, axis=2))
        min_distance = np.minimum(min_distance, distance.min(axis=1))
    keep = min_distance < 10.0
    protein = protein[keep]
    protein_radii = np.asarray([_VDW.get(int(value), 1.7) for value in protein_z[keep]])
    ligand_z = np.asarray([atom.GetAtomicNum() for atom in mol.GetAtoms()], dtype=np.int64)
    ligand_radii = np.asarray([_VDW.get(int(value), 1.7) for value in ligand_z])
    lower, upper, topology = _distance_bounds(mol)
    triu = np.triu(np.ones((mol.GetNumAtoms(), mol.GetNumAtoms()), dtype=bool), 1)
    mask12 = (topology == 1) & triu
    mask13 = (topology == 2) & triu
    nonbonded = (topology >= 3) & triu

    def within(distance: np.ndarray, mask: np.ndarray) -> bool:
        if not bool(mask.any()):
            return True
        okay = (distance >= lower * 0.75) & (distance <= upper * 1.25)
        return bool(okay[mask].all())

    results: list[bool] = []
    for pose in coordinates:
        ligand_distance = np.sqrt(np.sum((pose[:, None, :] - pose[None, :, :]) ** 2, axis=2))
        bond_ok = within(ligand_distance, mask12)
        angle_ok = within(ligand_distance, mask13)
        internal_ok = (
            True
            if not bool(nonbonded.any())
            else bool((ligand_distance >= lower * 0.70)[nonbonded].all())
        )
        protein_ok = True
        if len(protein):
            cross = np.sqrt(np.sum((pose[:, None, :] - protein[None, :, :]) ** 2, axis=2))
            threshold = 0.75 * (ligand_radii[:, None] + protein_radii[None, :])
            protein_ok = bool((cross >= threshold).all())
        results.append(bond_ok and angle_ok and internal_ok and protein_ok)
    return results


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]], bytes]:
    data = path.read_bytes()
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise AuditError(f"results CSV is not UTF-8: {path}") from exc
    reader = csv.DictReader(text.splitlines())
    if reader.fieldnames is None or len(reader.fieldnames) != len(set(reader.fieldnames)):
        raise AuditError(f"results CSV has missing/duplicate headers: {path}")
    rows = list(reader)
    if any(None in row for row in rows):
        raise AuditError(f"results CSV contains surplus columns: {path}")
    return list(reader.fieldnames), rows, data


def _row_asset(
    row: dict[str, str],
    manifest_record: dict[str, Any],
    *,
    path_field: str,
    hash_field: str,
    manifest_field: str,
    label: str,
    hashes: FileHashCache,
) -> Path:
    frozen = manifest_record.get(manifest_field)
    if not isinstance(frozen, dict):
        raise AuditError(f"{label}: eligibility asset is missing")
    if row.get(hash_field) != frozen.get("sha256"):
        raise AuditError(f"{label}: row asset digest differs from eligibility")
    return _require_file_identity(
        row.get(path_field),
        row.get(hash_field),
        label=label,
        hashes=hashes,
        expected_path=Path(str(frozen.get("path", ""))),
    )


def _audit_row(
    row: dict[str, str],
    *,
    arm: str,
    shard_dir: Path,
    expected_id: str,
    expected_global_index: int,
    manifest_record: dict[str, Any],
    hashes: FileHashCache,
    fast_valid_recheck: bool,
) -> RowAudit:
    sample_id = str(row.get("id", ""))
    label = f"{shard_dir.name}/{arm}/{sample_id or '<missing>'}"
    if sample_id != expected_id or row.get("arm") != arm:
        raise AuditError(f"{label}: row ID/arm/order mismatch")
    system_id, ligand_chain = _split_sample_key(sample_id)
    if row.get("plinder_system_id") != system_id or row.get("plinder_ligand_chain") != ligand_chain:
        raise AuditError(f"{label}: PLINDER system/chain mismatch")
    global_index = _strict_int(row.get("plinder_global_index"), label=f"{label}.global_index")
    if global_index != expected_global_index:
        raise AuditError(f"{label}: global index mismatch")
    sampling_seed = _strict_int(row.get("sampling_seed"), label=f"{label}.sampling_seed")
    if sampling_seed != BASE_SEED + global_index:
        raise AuditError(f"{label}: sampling seed mismatch")
    if _strict_int(row.get("ligand_conformer_seed"), label=f"{label}.conformer_seed") != CONFORMER_SEED:
        raise AuditError(f"{label}: conformer seed mismatch")
    for field, expected in (
        ("num_samples", EXPECTED_NUM_SAMPLES),
        ("all_poses_count", EXPECTED_NUM_SAMPLES),
        ("prior_pool_size", EXPECTED_PRIOR_POOL_SIZE),
    ):
        if _strict_int(row.get(field), label=f"{label}.{field}") != expected:
            raise AuditError(f"{label}: {field} mismatch")
    if (
        row.get("selector_profile") != "candidate_only"
        or row.get("guidance_mode") != "none"
        or row.get("sampling_dynamics") != "deterministic_ode"
        or _finite_float(row.get("translation_sde_base_sigma"), label=f"{label}.sde_sigma")
        != 0.0
        or not _strict_bool(row.get("full_heavy_atom_bijection"), label=f"{label}.bijection")
    ):
        raise AuditError(f"{label}: frozen inference contract mismatch")
    checkpoint = _require_file_identity(
        row.get("checkpoint"),
        row.get("checkpoint_sha256"),
        label=f"{label}.checkpoint",
        hashes=hashes,
        expected_sha256=CHECKPOINT_SHA256[arm],
    )
    del checkpoint
    protein_path = _row_asset(
        row,
        manifest_record,
        path_field="protein",
        hash_field="protein_sha256",
        manifest_field="receptor",
        label=f"{label}.protein",
        hashes=hashes,
    )
    reference_path = _row_asset(
        row,
        manifest_record,
        path_field="ligand_ref",
        hash_field="ligand_reference_sha256",
        manifest_field="ligand_reference",
        label=f"{label}.reference",
        hashes=hashes,
    )
    _row_asset(
        row,
        manifest_record,
        path_field="processed_meta",
        hash_field="processed_meta_sha256",
        manifest_field="processed_meta",
        label=f"{label}.processed_meta",
        hashes=hashes,
    )
    if row.get("ligand_input_identity_sha256") != manifest_record.get(
        "canonical_smiles_identity_sha256"
    ):
        raise AuditError(f"{label}: canonical-SMILES identity mismatch")
    canonical_smiles = manifest_record.get("canonical_smiles")
    if (
        not isinstance(canonical_smiles, str)
        or not canonical_smiles
        or row.get("ligand_input_canonical_smiles") != canonical_smiles
        or manifest_record.get("canonical_smiles_identity_sha256")
        != _canonical_smiles_identity(canonical_smiles)
    ):
        raise AuditError(f"{label}: frozen canonical-SMILES ledger mismatch")
    canonical_molecule = Chem.MolFromSmiles(canonical_smiles)
    if canonical_molecule is None:
        raise AuditError(f"{label}: RDKit cannot parse frozen canonical SMILES")
    canonical_molecule = Chem.RemoveAllHs(canonical_molecule)
    if any(atom.GetAtomicNum() == 1 for atom in canonical_molecule.GetAtoms()):
        raise AuditError(f"{label}: canonical-SMILES graph retains explicit hydrogen")
    prior_hash = _require_hex64(row.get("prior_pool_sha256"), label=f"{label}.prior_hash")
    ensemble_hash = _require_hex64(
        row.get("candidate_ensemble_sha256"), label=f"{label}.ensemble_hash"
    )

    expected_all_poses = (
        shard_dir / "arms" / arm / "poses" / "all_poses" / f"{sample_id}.sdf"
    )
    recorded_all_poses = Path(str(row.get("all_poses_sdf", "")))
    if ".incomplete" in recorded_all_poses.parts:
        raise AuditError(f"{label}: published SDF lexically points into .incomplete")
    all_poses_path = _require_file_identity(
        row.get("all_poses_sdf"),
        row.get("all_poses_sdf_sha256"),
        label=f"{label}.all_poses",
        hashes=hashes,
        expected_path=expected_all_poses,
    )
    selected_hashes = _parse_json_object(
        row.get("saved_pose_sha256_json"), label=f"{label}.saved_pose_sha256_json"
    )
    if set(selected_hashes) != {"selected"}:
        raise AuditError(f"{label}: candidate-only saved pose inventory mismatch")
    selected_path = shard_dir / "arms" / arm / "poses" / "selected" / f"{sample_id}.sdf"
    _require_file_identity(
        str(selected_path.resolve()),
        selected_hashes["selected"],
        label=f"{label}.selected_pose",
        hashes=hashes,
        expected_path=selected_path,
    )

    declared_rmsds_raw = _parse_json_list(
        row.get("candidate_rmsds_json"), label=f"{label}.candidate_rmsds_json"
    )
    declared_methods_raw = _parse_json_list(
        row.get("candidate_rmsd_method_json"), label=f"{label}.candidate_methods_json"
    )
    declared_fast_raw = _parse_json_list(
        row.get("candidate_fast_valid_json"), label=f"{label}.candidate_fast_valid_json"
    )
    if not all(
        len(values) == EXPECTED_NUM_SAMPLES
        for values in (declared_rmsds_raw, declared_methods_raw, declared_fast_raw)
    ):
        raise AuditError(f"{label}: ordered candidate vector length mismatch")
    declared_rmsds = [
        _finite_float(value, label=f"{label}.rmsd[{index}]")
        for index, value in enumerate(declared_rmsds_raw)
    ]
    if any(value < 0 for value in declared_rmsds):
        raise AuditError(f"{label}: negative RMSD")
    allowed_methods = {"rdkit_calc_rms_symmetry_no_align", "mapped_index_fallback"}
    declared_methods = [str(value) for value in declared_methods_raw]
    if any(value not in allowed_methods for value in declared_methods):
        raise AuditError(f"{label}: unrecognized RMSD method")
    if any(not isinstance(value, bool) for value in declared_fast_raw):
        raise AuditError(f"{label}: fast-valid vector must contain booleans")
    declared_fast = [bool(value) for value in declared_fast_raw]

    reference = _load_reference(reference_path)
    reference_coordinates = _coordinates(reference, label=f"{label}.reference")
    pose_molecules: list[Chem.Mol] = []
    pose_coordinates: list[np.ndarray] = []
    observed_rmsds: list[float] = []
    observed_methods: list[str] = []
    observed_fast_props: list[bool] = []
    dock_indices: list[int] | None = None
    ref_indices: list[int] | None = None
    independent_match_method: str | None = None
    first_topology: tuple[Any, ...] | None = None
    with all_poses_path.open("rb") as handle:
        supplier = Chem.ForwardSDMolSupplier(
            handle,
            sanitize=True,
            removeHs=True,
            strictParsing=True,
        )
        for sample_index, molecule in enumerate(supplier):
            if molecule is None:
                raise AuditError(f"{label}: SDF record {sample_index} did not parse")
            if sample_index >= EXPECTED_NUM_SAMPLES:
                raise AuditError(f"{label}: SDF has more than {EXPECTED_NUM_SAMPLES} records")
            properties = {
                "sample_index": sample_index,
                "complex_id": sample_id,
                "dataset": "plinder_val",
                "sampling_seed": sampling_seed,
                "ligand_conformer_seed": CONFORMER_SEED,
                "num_samples": EXPECTED_NUM_SAMPLES,
                "num_steps": EXPECTED_NUM_STEPS,
                "candidate_ensemble_sha256": ensemble_hash,
            }
            for name, expected in properties.items():
                if not molecule.HasProp(name):
                    raise AuditError(f"{label}: SDF record {sample_index} lacks {name}")
                actual = molecule.GetProp(name)
                if isinstance(expected, int):
                    if _strict_int(actual, label=f"{label}.sdf[{sample_index}].{name}") != expected:
                        raise AuditError(f"{label}: SDF {name} mismatch at record {sample_index}")
                elif actual != expected:
                    raise AuditError(f"{label}: SDF {name} mismatch at record {sample_index}")
            if not molecule.HasProp("fast_valid"):
                raise AuditError(f"{label}: SDF record {sample_index} lacks fast_valid")
            fast_prop = _strict_bool(
                molecule.GetProp("fast_valid"),
                label=f"{label}.sdf[{sample_index}].fast_valid",
            )
            if fast_prop != declared_fast[sample_index]:
                raise AuditError(f"{label}: SDF/CSV fast-valid mismatch at record {sample_index}")
            coordinates = _coordinates(molecule, label=f"{label}.sdf[{sample_index}]")
            topology = _topology_signature(molecule)
            if first_topology is None:
                first_topology = topology
            elif topology != first_topology:
                raise AuditError(
                    f"{label}: ordered topology changed at SDF record {sample_index}"
                )
            if dock_indices is None:
                dock_indices, ref_indices, independent_match_method = _match_atoms(
                    reference, molecule
                )
                if not dock_indices or ref_indices is None:
                    raise AuditError(f"{label}: independent atom mapping failed")
                mapping_metadata = _full_heavy_atom_graph_metadata(
                    reference,
                    molecule,
                    dock_indices,
                    ref_indices,
                    independent_match_method,
                )
                if mapping_metadata["accepted"] is not True:
                    raise AuditError(
                        f"{label}: independent full heavy-atom graph/connectivity gate failed"
                    )
                if row.get("match_method") != independent_match_method:
                    raise AuditError(f"{label}: independent/CSV atom mapping method mismatch")
                if _strict_int(row.get("num_match_atoms"), label=f"{label}.num_match_atoms") != len(dock_indices):
                    raise AuditError(f"{label}: mapped atom count mismatch")
                if _strict_int(
                    row.get("num_input_atoms"), label=f"{label}.num_input_atoms"
                ) != molecule.GetNumAtoms() or _strict_int(
                    row.get("num_ref_atoms"), label=f"{label}.num_ref_atoms"
                ) != reference.GetNumAtoms():
                    raise AuditError(f"{label}: input/reference atom count ledger mismatch")
                if row.get("ligand_graph_relation") != mapping_metadata["relation"]:
                    raise AuditError(f"{label}: ligand graph relation mismatch")
                if _strict_bool(
                    row.get("exact_full_heavy_atom_graph"),
                    label=f"{label}.exact_full_heavy_atom_graph",
                ) != (mapping_metadata["relation"] == "exact_graph"):
                    raise AuditError(f"{label}: exact graph flag mismatch")
                declared_mapping = _parse_json_object(
                    row.get("ligand_mapping_metadata_json"),
                    label=f"{label}.ligand_mapping_metadata_json",
                )
                if declared_mapping != mapping_metadata:
                    raise AuditError(f"{label}: ligand mapping metadata mismatch")
                if not _matches_frozen_canonical_graph(canonical_molecule, molecule):
                    raise AuditError(
                        f"{label}: SDF exact non-stereo chemical graph differs from frozen "
                        "canonical SMILES"
                    )
                canonical_dock_indices, canonical_ref_indices, canonical_method = (
                    _match_atoms(canonical_molecule, molecule)
                )
                canonical_metadata = _full_heavy_atom_graph_metadata(
                    canonical_molecule,
                    molecule,
                    canonical_dock_indices,
                    canonical_ref_indices,
                    canonical_method,
                )
                if canonical_metadata["accepted"] is not True:
                    raise AuditError(
                        f"{label}: SDF topology differs from frozen canonical SMILES"
                    )
            assert ref_indices is not None
            rmsd, method = _pose_rmsd(
                molecule,
                reference,
                coordinates,
                reference_coordinates,
                dock_indices,
                ref_indices,
            )
            if method != declared_methods[sample_index]:
                raise AuditError(f"{label}: RMSD method mismatch at record {sample_index}")
            if not math.isclose(
                rmsd,
                declared_rmsds[sample_index],
                rel_tol=0.0,
                abs_tol=RMSD_SDF_ABS_TOL,
            ):
                raise AuditError(
                    f"{label}: coordinate RMSD mismatch at record {sample_index}: "
                    f"saved={rmsd:.8f} csv={declared_rmsds[sample_index]:.8f}"
                )
            pose_molecules.append(molecule)
            pose_coordinates.append(coordinates)
            observed_rmsds.append(rmsd)
            observed_methods.append(method)
            observed_fast_props.append(fast_prop)
    if len(pose_molecules) != EXPECTED_NUM_SAMPLES:
        raise AuditError(
            f"{label}: SDF record count {len(pose_molecules)} != {EXPECTED_NUM_SAMPLES}"
        )
    if observed_methods != declared_methods:
        raise AuditError(f"{label}: ordered RMSD method ledger mismatch")
    selected_molecules = _read_sdf_exact(
        selected_path, expected_count=1, label=f"{label}.selected_pose"
    )
    selected_molecule = selected_molecules[0]
    if _topology_signature(selected_molecule) != first_topology:
        raise AuditError(f"{label}: selected pose topology differs from candidate 0")
    selected_coordinates = _coordinates(
        selected_molecule, label=f"{label}.selected_pose"
    )
    if not np.array_equal(selected_coordinates, pose_coordinates[0]):
        raise AuditError(f"{label}: selected pose coordinates differ from candidate 0")
    ambiguous_candidates = tuple(
        (index, declared, observed)
        for index, (declared, observed) in enumerate(
            zip(declared_rmsds, observed_rmsds, strict=True)
        )
        if min(abs(declared - 2.0), abs(observed - 2.0)) <= RMSD_SDF_ABS_TOL
    )
    ambiguous_candidate_indices = {entry[0] for entry in ambiguous_candidates}
    for index, (declared, observed) in enumerate(
        zip(declared_rmsds, observed_rmsds, strict=True)
    ):
        if (declared < 2.0) != (observed < 2.0) and index not in ambiguous_candidate_indices:
            raise AuditError(
                f"{label}: K2 classification changed outside the frozen quantization band "
                f"at candidate {index}"
            )
    coordinates_array = np.stack(pose_coordinates, axis=0)
    atomic_numbers = np.asarray(
        [atom.GetAtomicNum() for atom in pose_molecules[0].GetAtoms()], dtype=np.int64
    )
    diversity = _diversity_metrics(coordinates_array, atomic_numbers)
    if (
        row.get("pose_diversity_contract") != POSE_DIVERSITY_CONTRACT
        or _strict_int(
            row.get("pose_diversity_round_decimals"),
            label=f"{label}.pose_diversity_round_decimals",
        )
        != POSE_DIVERSITY_ROUND_DECIMALS
    ):
        raise AuditError(f"{label}: diversity contract mismatch")
    for field in ("diversity_heavy_atom_count", "coordinate_unique_count"):
        declared = _strict_int(row.get(field), label=f"{label}.{field}")
        if declared != diversity[field]:
            raise AuditError(f"{label}: coordinate diversity count mismatch for {field}")
    for field in (
        "pairwise_heavy_atom_rmsd_mean",
        "pairwise_heavy_atom_rmsd_median",
        "nearest_neighbor_heavy_atom_rmsd_median",
    ):
        declared = _finite_float(row.get(field), label=f"{label}.{field}")
        if not math.isclose(
            declared,
            float(diversity[field]),
            rel_tol=0.0,
            abs_tol=DIVERSITY_SDF_ABS_TOL,
        ):
            raise AuditError(f"{label}: coordinate diversity mismatch for {field}")
    heavy_mask = atomic_numbers > 1
    pairwise = _heavy_pairwise_rmsd(coordinates_array[:, heavy_mask, :])
    upper_i, upper_j = np.triu_indices(EXPECTED_NUM_SAMPLES, k=1)
    upper = pairwise[upper_i, upper_j]
    ambiguous_edge_mask = np.abs(upper - 2.0) <= DIVERSITY_SDF_ABS_TOL
    ambiguous_edges = tuple(
        (int(left), int(right), float(distance))
        for left, right, distance in zip(
            upper_i[ambiguous_edge_mask],
            upper_j[ambiguous_edge_mask],
            upper[ambiguous_edge_mask],
            strict=True,
        )
    )
    pair_count = len(upper)
    declared_ge2_fraction = _finite_float(
        row.get("pairwise_heavy_atom_rmsd_ge2_fraction"),
        label=f"{label}.pairwise_heavy_atom_rmsd_ge2_fraction",
    )
    declared_ge2_count_float = declared_ge2_fraction * pair_count
    declared_ge2_count = int(round(declared_ge2_count_float))
    if not math.isclose(
        declared_ge2_count_float, declared_ge2_count, rel_tol=0.0, abs_tol=1e-8
    ):
        raise AuditError(f"{label}: declared >=2A pair fraction is not an exact count")
    definite_ge2_count = int((upper >= 2.0 + DIVERSITY_SDF_ABS_TOL).sum())
    possible_ge2_count = int((upper >= 2.0 - DIVERSITY_SDF_ABS_TOL).sum())
    if not definite_ge2_count <= declared_ge2_count <= possible_ge2_count:
        raise AuditError(
            f"{label}: declared >=2A edge count lies outside the quantization band"
        )
    declared_c2 = _strict_int(
        row.get("c2_connected_component_count"),
        label=f"{label}.c2_connected_component_count",
    )
    c2_min = _component_count(pairwise, 2.0 + DIVERSITY_SDF_ABS_TOL)
    c2_max = _component_count(pairwise, 2.0 - DIVERSITY_SDF_ABS_TOL)
    if not c2_min <= declared_c2 <= c2_max:
        raise AuditError(f"{label}: declared C2 lies outside the quantization-edge range")

    observed_k2 = sum(value < 2.0 for value in observed_rmsds)
    declared_k2 = sum(value < 2.0 for value in declared_rmsds)
    observed_fast_k2 = sum(
        rmsd < 2.0 and valid
        for rmsd, valid in zip(observed_rmsds, observed_fast_props, strict=True)
    )
    declared_fast_k2 = sum(
        rmsd < 2.0 and valid
        for rmsd, valid in zip(declared_rmsds, declared_fast, strict=True)
    )
    declared_counts = {
        "num_rmsd_lt2_candidates": declared_k2,
        "num_fast_valid_candidates": sum(observed_fast_props),
        "num_fast_valid_rmsd_lt2_candidates": declared_fast_k2,
        "num_mapped_index_rmsd_fallback_candidates": sum(
            method == "mapped_index_fallback" for method in observed_methods
        ),
    }
    for field, expected in declared_counts.items():
        if _strict_int(row.get(field), label=f"{label}.{field}") != expected:
            raise AuditError(f"{label}: {field} differs from saved candidates")
    if _strict_int(row.get("first_index"), label=f"{label}.first_index") != 0 or _strict_int(
        row.get("selected_index"), label=f"{label}.selected_index"
    ) != 0:
        raise AuditError(f"{label}: candidate-only first/selected index mismatch")
    first = observed_rmsds[0]
    oracle = min(observed_rmsds)
    declared_first = _finite_float(row.get("first_rmsd"), label=f"{label}.first_rmsd")
    declared_selected = _finite_float(
        row.get("selected_rmsd"), label=f"{label}.selected_rmsd"
    )
    declared_oracle = _finite_float(
        row.get("oracle_rmsd"), label=f"{label}.oracle_rmsd"
    )
    declared_mean = _finite_float(
        row.get("mean_sample_rmsd"), label=f"{label}.mean_sample_rmsd"
    )
    for field, declared, expected in (
        ("first_rmsd", declared_first, first),
        ("selected_rmsd", declared_selected, first),
        ("oracle_rmsd", declared_oracle, oracle),
        (
            "mean_sample_rmsd",
            declared_mean,
            math.fsum(observed_rmsds) / EXPECTED_NUM_SAMPLES,
        ),
    ):
        if not math.isclose(declared, expected, rel_tol=0.0, abs_tol=RMSD_SDF_ABS_TOL):
            raise AuditError(f"{label}: {field} differs from saved coordinates")

    recheck_candidates = 0
    recheck_mismatches = 0
    recheck_mismatch_indices: tuple[int, ...] = ()
    if fast_valid_recheck:
        secondary = _secondary_fast_valid(coordinates_array, pose_molecules[0], protein_path)
        recheck_candidates = len(secondary)
        recheck_mismatch_indices = tuple(
            index
            for index, (left, right) in enumerate(
                zip(secondary, observed_fast_props, strict=True)
            )
            if left != right
        )
        recheck_mismatches = len(recheck_mismatch_indices)
    return RowAudit(
        sample_id=sample_id,
        system_id=system_id,
        arm=arm,
        global_index=global_index,
        sampling_seed=sampling_seed,
        k2=observed_k2,
        declared_k2=declared_k2,
        fast_valid_k2=observed_fast_k2,
        declared_fast_valid_k2=declared_fast_k2,
        fast_valid_count=sum(observed_fast_props),
        first_rmsd=first,
        declared_first_rmsd=declared_first,
        oracle_rmsd=oracle,
        declared_oracle_rmsd=declared_oracle,
        coordinate_unique_count=int(diversity["coordinate_unique_count"]),
        nearest_neighbor_rmsd_median=float(
            diversity["nearest_neighbor_heavy_atom_rmsd_median"]
        ),
        declared_nearest_neighbor_rmsd_median=_finite_float(
            row.get("nearest_neighbor_heavy_atom_rmsd_median"),
            label=f"{label}.nearest_neighbor_heavy_atom_rmsd_median",
        ),
        c2_component_count=int(diversity["c2_connected_component_count"]),
        declared_c2_component_count=declared_c2,
        pair_identity=(
            str(sampling_seed),
            str(CONFORMER_SEED),
            prior_hash,
            str(row.get("protein_sha256")),
            str(row.get("ligand_reference_sha256")),
            str(row.get("ligand_input_identity_sha256")),
        ),
        all_poses_sha256=str(row["all_poses_sdf_sha256"]),
        fast_valid_recheck_candidates=recheck_candidates,
        fast_valid_recheck_mismatches=recheck_mismatches,
        fast_valid_recheck_mismatch_indices=recheck_mismatch_indices,
        quantization_ambiguous_candidates=ambiguous_candidates,
        quantization_ambiguous_edges=ambiguous_edges,
    )


def _validate_summary(
    summary: dict[str, Any],
    *,
    shard_dir: Path,
    shard_index: int,
    expected_ids: list[str],
    eligible_ids: list[str],
    excluded_ids: list[str],
    eligibility_path: Path,
    eligibility_sha256: str,
    fixed_identities: dict[str, Any],
) -> None:
    label = shard_dir.name
    if (
        summary.get("schema_version") != SHARD_SCHEMA
        or summary.get("protocol_id") != PROTOCOL_ID
        or summary.get("status") != "complete"
        or summary.get("mode") != "full"
        or summary.get("run_id") != shard_dir.parent.parent.name
        or summary.get("failures") != []
    ):
        raise AuditError(f"{label}: summary contract/status mismatch")
    settings = summary.get("settings")
    expected_settings = {
        "stage": "full",
        "selected_count": None,
        "num_samples": EXPECTED_NUM_SAMPLES,
        "num_steps": EXPECTED_NUM_STEPS,
        "model_pose_step_budget": EXPECTED_NUM_SAMPLES * EXPECTED_NUM_STEPS,
        "sigma": 2.0,
        "prior_pool_size": EXPECTED_PRIOR_POOL_SIZE,
        "time_schedule": "late",
        "schedule_power": 3.0,
        "pocket_cutoff_angstrom": 10.0,
        "center_jitter_sigma": 0.0,
        "confidence": False,
        "vina_selection": False,
        "vina_guidance_scale": 0.0,
        "unified_guidance_scale": 0.0,
        "fk_constraint_beta": 0.0,
        "fk_resample_times": [],
        "translation_sde_base_sigma": 0.0,
        "sampling_dynamics": "deterministic_ode",
        "refine": "none",
        "selector_profile": "candidate_only",
        "ligand_conformer_seed": CONFORMER_SEED,
        "include_s50_replay": False,
    }
    if settings != expected_settings:
        raise AuditError(f"{label}: frozen full settings mismatch")
    eligibility = summary.get("eligibility_manifest")
    if not isinstance(eligibility, dict):
        raise AuditError(f"{label}: eligibility provenance missing")
    _require_published_path(
        eligibility.get("path"),
        eligibility_path,
        label=f"{label}.eligibility_manifest.path",
    )
    if (
        eligibility.get("sha256") != eligibility_sha256
        or _strict_int(eligibility.get("eligible_count"), label=f"{label}.eligible_count")
        != EXPECTED_ELIGIBLE_COUNT
        or _strict_int(eligibility.get("ineligible_count"), label=f"{label}.ineligible_count")
        != EXPECTED_EXCLUDED_COUNT
        or _strict_int(
            eligibility.get("eligible_system_count"), label=f"{label}.eligible_system_count"
        )
        != EXPECTED_SYSTEM_COUNT
        or eligibility.get("eligible_ids_newline_sha256")
        != EXPECTED_ELIGIBLE_NEWLINE_SHA256
    ):
        raise AuditError(f"{label}: eligibility provenance mismatch")
    inventory = summary.get("inventory")
    if not isinstance(inventory, dict):
        raise AuditError(f"{label}: inventory missing")
    declared = {
        "full_count": EXPECTED_FULL_COUNT,
        "eligible_count": EXPECTED_ELIGIBLE_COUNT,
        "selected_count": EXPECTED_ELIGIBLE_COUNT,
        "num_shards": EXPECTED_SHARDS,
        "shard_index": shard_index,
        "assigned_count": len(expected_ids),
    }
    for field, expected in declared.items():
        if _strict_int(inventory.get(field), label=f"{label}.{field}") != expected:
            raise AuditError(f"{label}: {field} mismatch")
    if inventory.get("selected_ids") != eligible_ids or inventory.get("assigned_ids") != expected_ids:
        raise AuditError(f"{label}: selected/assigned ID inventory mismatch")
    if inventory.get("selected_ids_sha256") != _versioned_ids_sha256(eligible_ids) or inventory.get(
        "assigned_ids_sha256"
    ) != _versioned_ids_sha256(expected_ids):
        raise AuditError(f"{label}: selected/assigned ID digest mismatch")
    counts = inventory.get("arm_success_counts")
    if not isinstance(counts, dict) or set(counts) != set(ARMS) or any(
        _strict_int(counts[arm], label=f"{label}.{arm}.count") != len(expected_ids)
        for arm in ARMS
    ):
        raise AuditError(f"{label}: arm success counts mismatch")
    if [entry.get("name") for entry in summary.get("arms", [])] != list(ARMS):
        raise AuditError(f"{label}: arm order/inventory mismatch")
    for entry in summary["arms"]:
        arm = str(entry["name"])
        if entry.get("checkpoint_sha256") != CHECKPOINT_SHA256[arm]:
            raise AuditError(f"{label}: checkpoint digest mismatch for {arm}")
    operational = summary.get("operational_inventory")
    if not isinstance(operational, dict) or (
        _strict_int(operational.get("requested_count"), label=f"{label}.requested")
        != EXPECTED_FULL_COUNT
        or _strict_int(operational.get("evaluable_count"), label=f"{label}.evaluable")
        != EXPECTED_ELIGIBLE_COUNT
        or _strict_int(
            operational.get("common_preprocessing_failure_count"),
            label=f"{label}.preprocessing_failures",
        )
        != EXPECTED_EXCLUDED_COUNT
        or operational.get("common_preprocessing_failure_ids") != excluded_ids
        or operational.get("operational_sensitivity_assignment")
        != "common preprocessing failures have K2=0"
    ):
        raise AuditError(f"{label}: operational accounting mismatch")
    paired = summary.get("paired_identity_gate")
    if not isinstance(paired, dict) or paired.get("passed") is not True or _strict_int(
        paired.get("checked_count"), label=f"{label}.paired_count"
    ) != len(expected_ids):
        raise AuditError(f"{label}: runner paired identity gate mismatch")
    replay = summary.get("replay_integrity_gate")
    if not isinstance(replay, dict) or replay.get("required") is not False or replay.get(
        "passed"
    ) is not True:
        raise AuditError(f"{label}: full replay contract mismatch")
    if summary.get("fixed_identities") != fixed_identities:
        raise AuditError(f"{label}: fixed identity inventory differs from eligibility")


def _arm_metrics(rows: Sequence[RowAudit]) -> dict[str, Any]:
    count = len(rows)
    k2 = np.asarray([row.k2 for row in rows], dtype=np.int64)
    fv2 = np.asarray([row.fast_valid_k2 for row in rows], dtype=np.int64)
    fast = np.asarray([row.fast_valid_count for row in rows], dtype=np.int64)
    unique = np.asarray([row.coordinate_unique_count for row in rows], dtype=np.int64)
    return {
        "sample_count": count,
        "k2_total": int(k2.sum()),
        "k2_mean": float(k2.mean()),
        "k2_median": float(np.median(k2)),
        "k2_ge_1_count": int((k2 >= 1).sum()),
        "k2_ge_1_pct": float(100.0 * (k2 >= 1).mean()),
        "k2_ge_5_count": int((k2 >= 5).sum()),
        "k2_ge_10_count": int((k2 >= 10).sum()),
        "fast_valid_k2_total": int(fv2.sum()),
        "fast_valid_k2_mean": float(fv2.mean()),
        "fast_valid_k2_ge_1_count": int((fv2 >= 1).sum()),
        "fast_valid_candidate_fraction": float(fast.sum() / (count * EXPECTED_NUM_SAMPLES)),
        "coordinate_unique_fraction": float(unique.sum() / (count * EXPECTED_NUM_SAMPLES)),
        "nearest_neighbor_median_mean": float(
            np.mean([row.nearest_neighbor_rmsd_median for row in rows])
        ),
        "c2_component_mean": float(np.mean([row.c2_component_count for row in rows])),
        "first_rmsd_mean": float(np.mean([row.first_rmsd for row in rows])),
        "oracle_rmsd_mean": float(np.mean([row.oracle_rmsd for row in rows])),
    }


def _operational_full_split_metrics(rows: Sequence[RowAudit]) -> dict[str, Any]:
    """Insert the common 41 preprocessing failures as K2=fastK2=0."""
    if len(rows) != EXPECTED_ELIGIBLE_COUNT:
        raise AuditError("operational sensitivity received incomplete eligible rows")
    k2_total = sum(row.k2 for row in rows)
    fv2_total = sum(row.fast_valid_k2 for row in rows)
    return {
        "full_split_count": EXPECTED_FULL_COUNT,
        "eligible_count": EXPECTED_ELIGIBLE_COUNT,
        "common_zero_assigned_count": EXPECTED_EXCLUDED_COUNT,
        "k2_total": int(k2_total),
        "k2_mean": float(k2_total / EXPECTED_FULL_COUNT),
        "k2_ge_1_count": int(sum(row.k2 >= 1 for row in rows)),
        "k2_ge_1_pct": float(
            100.0 * sum(row.k2 >= 1 for row in rows) / EXPECTED_FULL_COUNT
        ),
        "fast_valid_k2_total": int(fv2_total),
        "fast_valid_k2_mean": float(fv2_total / EXPECTED_FULL_COUNT),
        "fast_valid_k2_ge_1_count": int(sum(row.fast_valid_k2 >= 1 for row in rows)),
        "fast_valid_k2_ge_1_pct": float(
            100.0
            * sum(row.fast_valid_k2 >= 1 for row in rows)
            / EXPECTED_FULL_COUNT
        ),
    }


def _operational_full_split_comparison(
    baseline: Sequence[RowAudit], treatment: Sequence[RowAudit]
) -> dict[str, Any]:
    if [row.sample_id for row in baseline] != [row.sample_id for row in treatment]:
        raise AuditError("operational full-split comparison is not paired")
    delta_k2 = sum(
        right.k2 - left.k2 for left, right in zip(baseline, treatment, strict=True)
    )
    delta_fv2 = sum(
        right.fast_valid_k2 - left.fast_valid_k2
        for left, right in zip(baseline, treatment, strict=True)
    )
    coverage_delta = sum(
        (right.k2 >= 1) - (left.k2 >= 1)
        for left, right in zip(baseline, treatment, strict=True)
    )
    fv_coverage_delta = sum(
        (right.fast_valid_k2 >= 1) - (left.fast_valid_k2 >= 1)
        for left, right in zip(baseline, treatment, strict=True)
    )
    return {
        "full_split_count": EXPECTED_FULL_COUNT,
        "common_zero_assigned_count": EXPECTED_EXCLUDED_COUNT,
        "delta_total_k2": int(delta_k2),
        "delta_mean_k2": float(delta_k2 / EXPECTED_FULL_COUNT),
        "k2_ge_1_delta_count": int(coverage_delta),
        "k2_ge_1_delta_pp": float(100.0 * coverage_delta / EXPECTED_FULL_COUNT),
        "delta_total_fast_valid_k2": int(delta_fv2),
        "delta_mean_fast_valid_k2": float(delta_fv2 / EXPECTED_FULL_COUNT),
        "fast_valid_k2_ge_1_delta_count": int(fv_coverage_delta),
        "fast_valid_k2_ge_1_delta_pp": float(
            100.0 * fv_coverage_delta / EXPECTED_FULL_COUNT
        ),
        "note": (
            "The common preprocessing failures are zero in every arm, so totals/deltas are "
            "unchanged while means and percentage-point changes use the full split denominator."
        ),
    }


def _declared_view(row: RowAudit) -> RowAudit:
    """Project a row onto the pre-serialization CSV metrics for sensitivity."""
    return replace(
        row,
        k2=row.declared_k2,
        fast_valid_k2=row.declared_fast_valid_k2,
        first_rmsd=row.declared_first_rmsd,
        oracle_rmsd=row.declared_oracle_rmsd,
        nearest_neighbor_rmsd_median=row.declared_nearest_neighbor_rmsd_median,
        c2_component_count=row.declared_c2_component_count,
    )


def _descriptive_pair(
    left: Sequence[RowAudit],
    right: Sequence[RowAudit],
    *,
    left_arm: str,
    right_arm: str,
) -> dict[str, Any]:
    if [row.sample_id for row in left] != [row.sample_id for row in right]:
        raise AuditError("descriptive paired comparison ID/order mismatch")
    gained = [
        first.sample_id
        for first, second in zip(left, right, strict=True)
        if first.k2 < 1 <= second.k2
    ]
    lost = [
        first.sample_id
        for first, second in zip(left, right, strict=True)
        if second.k2 < 1 <= first.k2
    ]
    return {
        "left_arm": left_arm,
        "right_arm": right_arm,
        "sample_count": len(left),
        "delta_total_k2": int(
            sum(second.k2 - first.k2 for first, second in zip(left, right, strict=True))
        ),
        "delta_mean_k2": float(
            math.fsum(
                second.k2 - first.k2 for first, second in zip(left, right, strict=True)
            )
            / len(left)
        ),
        "k2_ge_1_delta_count": int(len(gained) - len(lost)),
        "k2_ge_1_gained_ids": gained,
        "k2_ge_1_lost_ids": lost,
        "delta_mean_fast_valid_k2": float(
            math.fsum(
                second.fast_valid_k2 - first.fast_valid_k2
                for first, second in zip(left, right, strict=True)
            )
            / len(left)
        ),
        "delta_mean_first_rmsd": float(
            math.fsum(
                second.first_rmsd - first.first_rmsd
                for first, second in zip(left, right, strict=True)
            )
            / len(left)
        ),
        "delta_mean_oracle_rmsd": float(
            math.fsum(
                second.oracle_rmsd - first.oracle_rmsd
                for first, second in zip(left, right, strict=True)
            )
            / len(left)
        ),
    }


def _cluster_arrays(
    baseline: Sequence[RowAudit], treatment: Sequence[RowAudit]
) -> dict[str, np.ndarray]:
    pairs = list(zip(baseline, treatment, strict=True))
    systems = sorted({left.system_id for left, _ in pairs})
    system_index = {system: index for index, system in enumerate(systems)}
    arrays = {
        name: np.zeros(len(systems), dtype=np.float64)
        for name in (
            "count",
            "k2_delta",
            "coverage_delta",
            "fv2_delta",
            "fv_coverage_delta",
            "nn_baseline",
            "nn_treatment",
            "c2_baseline",
            "c2_treatment",
        )
    }
    for left, right in pairs:
        index = system_index[left.system_id]
        arrays["count"][index] += 1
        arrays["k2_delta"][index] += right.k2 - left.k2
        arrays["coverage_delta"][index] += (right.k2 >= 1) - (left.k2 >= 1)
        arrays["fv2_delta"][index] += right.fast_valid_k2 - left.fast_valid_k2
        arrays["fv_coverage_delta"][index] += (right.fast_valid_k2 >= 1) - (
            left.fast_valid_k2 >= 1
        )
        arrays["nn_baseline"][index] += left.nearest_neighbor_rmsd_median
        arrays["nn_treatment"][index] += right.nearest_neighbor_rmsd_median
        arrays["c2_baseline"][index] += left.c2_component_count
        arrays["c2_treatment"][index] += right.c2_component_count
    if np.any(arrays["count"] < 1):
        raise AuditError("cluster bootstrap contains an empty system")
    return arrays


def _interval(values: np.ndarray) -> dict[str, float]:
    if values.ndim != 1 or len(values) != BOOTSTRAP_RESAMPLES or not np.isfinite(values).all():
        raise AuditError("bootstrap statistic is incomplete/non-finite")
    low, high = np.percentile(values, [2.5, 97.5])
    return {
        "bootstrap_mean": float(np.mean(values)),
        "ci95_low": float(low),
        "ci95_high": float(high),
    }


def _cluster_bootstrap(
    baseline: Sequence[RowAudit], treatment: Sequence[RowAudit]
) -> dict[str, Any]:
    arrays = _cluster_arrays(baseline, treatment)
    cluster_count = len(arrays["count"])
    draws = {
        name: np.empty(BOOTSTRAP_RESAMPLES, dtype=np.float64)
        for name in (
            "k2_delta",
            "coverage_delta_pp",
            "fast_valid_k2_delta",
            "fast_valid_coverage_delta_pp",
            "nearest_neighbor_ratio",
            "c2_ratio",
        )
    }
    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    batch_size = 256
    for start in range(0, BOOTSTRAP_RESAMPLES, batch_size):
        stop = min(start + batch_size, BOOTSTRAP_RESAMPLES)
        indices = rng.integers(0, cluster_count, size=(stop - start, cluster_count))
        selected_count = arrays["count"][indices].sum(axis=1)
        k2_delta = arrays["k2_delta"][indices].sum(axis=1)
        coverage_delta = arrays["coverage_delta"][indices].sum(axis=1)
        fv2_delta = arrays["fv2_delta"][indices].sum(axis=1)
        fv_coverage_delta = arrays["fv_coverage_delta"][indices].sum(axis=1)
        nn_baseline = arrays["nn_baseline"][indices].sum(axis=1)
        c2_baseline = arrays["c2_baseline"][indices].sum(axis=1)
        if np.any(nn_baseline <= 0.0) or np.any(c2_baseline <= 0.0):
            raise AuditError("bootstrap diversity ratio has non-positive baseline")
        draws["k2_delta"][start:stop] = k2_delta / selected_count
        draws["coverage_delta_pp"][start:stop] = 100.0 * coverage_delta / selected_count
        draws["fast_valid_k2_delta"][start:stop] = fv2_delta / selected_count
        draws["fast_valid_coverage_delta_pp"][start:stop] = (
            100.0 * fv_coverage_delta / selected_count
        )
        draws["nearest_neighbor_ratio"][start:stop] = (
            arrays["nn_treatment"][indices].sum(axis=1) / nn_baseline
        )
        draws["c2_ratio"][start:stop] = (
            arrays["c2_treatment"][indices].sum(axis=1) / c2_baseline
        )
    return {
        "contract": (
            "NumPy PCG64 percentile bootstrap; resample system_id clusters with replacement "
            "and retain every ligand sample in each drawn system"
        ),
        "seed": BOOTSTRAP_SEED,
        "resamples": BOOTSTRAP_RESAMPLES,
        "cluster_count": cluster_count,
        **{name: _interval(values) for name, values in draws.items()},
    }


def _gate(value: float | int, operator: str, threshold: float | int) -> dict[str, Any]:
    passed = {
        ">=": value >= threshold,
        ">": value > threshold,
    }[operator]
    return {"value": value, "operator": operator, "threshold": threshold, "passed": bool(passed)}


def _reconcile_quantized_decisions(
    coordinate_decision: dict[str, Any],
    csv_decision: dict[str, Any],
    *,
    candidate_flip_count: int,
    c2_changed_row_count: int,
    ambiguous_edge_count: int,
) -> tuple[dict[str, Any], list[str], list[str]]:
    changed_gates = sorted(
        name
        for name in coordinate_decision["gates"]
        if coordinate_decision["gates"][name]["passed"]
        != csv_decision["gates"][name]["passed"]
    )
    allowed_changed_gates: set[str] = set()
    if candidate_flip_count:
        allowed_changed_gates.update(
            name
            for name in coordinate_decision["gates"]
            if name.startswith(("efficacy", "coverage", "fragile", "fast_valid"))
        )
    if c2_changed_row_count and ambiguous_edge_count:
        allowed_changed_gates.update(
            name for name in coordinate_decision["gates"] if name.startswith("c2")
        )
    unexplained_changed_gates = sorted(set(changed_gates) - allowed_changed_gates)
    if (
        coordinate_decision["passed"] != csv_decision["passed"]
        and unexplained_changed_gates
    ):
        raise AuditError(
            "saved-coordinate/CSV-bound decision difference is not explained by a frozen "
            f"2A quantization band: {unexplained_changed_gates}"
        )
    decision_stable = coordinate_decision["passed"] == csv_decision["passed"]
    both_decisions_pass = coordinate_decision["passed"] and csv_decision["passed"]
    effective_failed_gates = sorted(
        set(coordinate_decision.get("failed_gates", []))
        | set(csv_decision.get("failed_gates", []))
    )
    if not decision_stable:
        effective_failed_gates.append("coordinate_quantization_decision_instability")
    effective = {
        "selection_eligible": decision_stable,
        "passed": bool(both_decisions_pass),
        "gates": csv_decision["gates"],
        "failed_gates": effective_failed_gates,
        "saved_coordinate_passed": coordinate_decision["passed"],
        "csv_bound_passed": csv_decision["passed"],
        "action": (
            "promote_existing_parent50k_plus10k_t0p10_ema_and_start_confidence_retraining"
            if both_decisions_pass
            else "keep_s50_ema_and_stop_additional_t0_sampler_continuation"
        ),
        "rationale": (
            "promotion requires both saved-coordinate and CSV-bound independent decisions "
            "to pass; a quantization-instability result is complete but selection-inconclusive"
        ),
    }
    return effective, changed_gates, unexplained_changed_gates


def _paired_comparison(
    baseline: Sequence[RowAudit], treatment: Sequence[RowAudit]
) -> tuple[dict[str, Any], dict[str, Any]]:
    if [row.sample_id for row in baseline] != [row.sample_id for row in treatment]:
        raise AuditError("paired comparison ID/order mismatch")
    count = len(baseline)
    deltas = [right.k2 - left.k2 for left, right in zip(baseline, treatment, strict=True)]
    fv_deltas = [
        right.fast_valid_k2 - left.fast_valid_k2
        for left, right in zip(baseline, treatment, strict=True)
    ]
    baseline_covered = [left.k2 >= 1 for left in baseline]
    treatment_covered = [right.k2 >= 1 for right in treatment]
    baseline_fv_covered = [left.fast_valid_k2 >= 1 for left in baseline]
    treatment_fv_covered = [right.fast_valid_k2 >= 1 for right in treatment]
    fragile = [left.sample_id for left in baseline if 1 <= left.k2 <= 4]
    fragile_retained = [
        left.sample_id
        for left, right in zip(baseline, treatment, strict=True)
        if 1 <= left.k2 <= 4 and right.k2 >= 1
    ]
    fragile_retention_fraction = (
        len(fragile_retained) / len(fragile) if fragile else 1.0
    )
    nn_baseline = math.fsum(row.nearest_neighbor_rmsd_median for row in baseline)
    nn_treatment = math.fsum(row.nearest_neighbor_rmsd_median for row in treatment)
    c2_baseline = math.fsum(row.c2_component_count for row in baseline)
    c2_treatment = math.fsum(row.c2_component_count for row in treatment)
    if nn_baseline <= 0.0 or c2_baseline <= 0.0:
        raise AuditError("diversity ratio has non-positive baseline")
    unique_baseline = math.fsum(row.coordinate_unique_count for row in baseline) / (
        count * EXPECTED_NUM_SAMPLES
    )
    unique_treatment = math.fsum(row.coordinate_unique_count for row in treatment) / (
        count * EXPECTED_NUM_SAMPLES
    )
    bootstrap = _cluster_bootstrap(baseline, treatment)
    comparison = {
        "baseline_arm": BASELINE_ARM,
        "treatment_arm": TREATMENT_ARM,
        "sample_count": count,
        "delta_total_k2": int(sum(deltas)),
        "delta_mean_k2": float(math.fsum(deltas) / count),
        "k2_ge_1_baseline_count": int(sum(baseline_covered)),
        "k2_ge_1_treatment_count": int(sum(treatment_covered)),
        "k2_ge_1_delta_count": int(sum(treatment_covered) - sum(baseline_covered)),
        "k2_ge_1_delta_pp": float(
            100.0 * (sum(treatment_covered) - sum(baseline_covered)) / count
        ),
        "k2_ge_1_gained_ids": [
            left.sample_id
            for left, base, treat in zip(
                baseline, baseline_covered, treatment_covered, strict=True
            )
            if not base and treat
        ],
        "k2_ge_1_lost_ids": [
            left.sample_id
            for left, base, treat in zip(
                baseline, baseline_covered, treatment_covered, strict=True
            )
            if base and not treat
        ],
        "fragile_baseline_count": len(fragile),
        "fragile_retained_count": len(fragile_retained),
        "fragile_retention_fraction": fragile_retention_fraction,
        "fragile_lost_ids": sorted(set(fragile) - set(fragile_retained)),
        "delta_total_fast_valid_k2": int(sum(fv_deltas)),
        "delta_mean_fast_valid_k2": float(math.fsum(fv_deltas) / count),
        "fast_valid_k2_ge_1_delta_count": int(
            sum(treatment_fv_covered) - sum(baseline_fv_covered)
        ),
        "fast_valid_k2_ge_1_delta_pp": float(
            100.0 * (sum(treatment_fv_covered) - sum(baseline_fv_covered)) / count
        ),
        "fast_valid_k2_ge_1_gained_ids": [
            left.sample_id
            for left, base, treat in zip(
                baseline, baseline_fv_covered, treatment_fv_covered, strict=True
            )
            if not base and treat
        ],
        "fast_valid_k2_ge_1_lost_ids": [
            left.sample_id
            for left, base, treat in zip(
                baseline, baseline_fv_covered, treatment_fv_covered, strict=True
            )
            if base and not treat
        ],
        "fast_valid_candidate_delta_pp": float(
            100.0
            * math.fsum(
                right.fast_valid_count - left.fast_valid_count
                for left, right in zip(baseline, treatment, strict=True)
            )
            / (count * EXPECTED_NUM_SAMPLES)
        ),
        "nearest_neighbor_aggregate_ratio": nn_treatment / nn_baseline,
        "c2_aggregate_ratio": c2_treatment / c2_baseline,
        "coordinate_unique_fraction_baseline": unique_baseline,
        "coordinate_unique_fraction_treatment": unique_treatment,
        "coordinate_unique_fraction_delta": unique_treatment - unique_baseline,
        "delta_first_rmsd_mean": float(
            math.fsum(
                right.first_rmsd - left.first_rmsd
                for left, right in zip(baseline, treatment, strict=True)
            )
            / count
        ),
        "delta_oracle_rmsd_mean": float(
            math.fsum(
                right.oracle_rmsd - left.oracle_rmsd
                for left, right in zip(baseline, treatment, strict=True)
            )
            / count
        ),
        "cluster_bootstrap": bootstrap,
    }
    gates = {
        "efficacy_mean_k2": _gate(comparison["delta_mean_k2"], ">=", 1.0),
        "efficacy_k2_ci95_low": _gate(
            bootstrap["k2_delta"]["ci95_low"], ">", 0.0
        ),
        "coverage_count": _gate(comparison["k2_ge_1_delta_count"], ">=", 0),
        "coverage_ci95_low_pp": _gate(
            bootstrap["coverage_delta_pp"]["ci95_low"], ">=", -1.0
        ),
        "fragile_retention": _gate(
            comparison["fragile_retention_fraction"], ">=", 0.95
        ),
        "fast_valid_mean_k2": _gate(
            comparison["delta_mean_fast_valid_k2"], ">=", 0.0
        ),
        "fast_valid_coverage_count": _gate(
            comparison["fast_valid_k2_ge_1_delta_count"], ">=", 0
        ),
        "fast_valid_coverage_ci95_low_pp": _gate(
            bootstrap["fast_valid_coverage_delta_pp"]["ci95_low"], ">=", -1.0
        ),
        "fast_valid_candidate_delta_pp": _gate(
            comparison["fast_valid_candidate_delta_pp"], ">=", -1.0
        ),
        "nearest_neighbor_ratio": _gate(
            comparison["nearest_neighbor_aggregate_ratio"], ">=", 0.95
        ),
        "nearest_neighbor_ratio_ci95_low": _gate(
            bootstrap["nearest_neighbor_ratio"]["ci95_low"], ">=", 0.90
        ),
        "c2_ratio": _gate(comparison["c2_aggregate_ratio"], ">=", 0.95),
        "c2_ratio_ci95_low": _gate(
            bootstrap["c2_ratio"]["ci95_low"], ">=", 0.90
        ),
        "coordinate_unique_fraction": _gate(
            comparison["coordinate_unique_fraction_treatment"], ">=", 0.99
        ),
        "coordinate_unique_delta": _gate(
            comparison["coordinate_unique_fraction_delta"], ">=", -0.005
        ),
    }
    failed = [name for name, gate in gates.items() if not gate["passed"]]
    decision = {
        "selection_eligible": True,
        "passed": not failed,
        "gates": gates,
        "failed_gates": failed,
        "action": (
            "promote_existing_parent50k_plus10k_t0p10_ema_and_start_confidence_retraining"
            if not failed
            else "keep_s50_ema_and_stop_additional_t0_sampler_continuation"
        ),
    }
    return comparison, decision


def audit_full(
    *,
    run_root: Path,
    eligibility_manifest: Path,
    eligibility_manifest_sha256: str,
    fast_valid_mode: str = "sampled",
    fast_valid_sample_count: int = 32,
    progress_every: int = 100,
) -> dict[str, Any]:
    started = time.monotonic()
    run_root = run_root.resolve(strict=True)
    if not run_root.is_dir() or run_root.name != "full":
        raise AuditError(f"run root is not a directory: {run_root}")
    if fast_valid_mode not in {"sampled", "full", "off"}:
        raise AuditError(f"unknown fast-valid mode: {fast_valid_mode}")
    if fast_valid_sample_count < 1 and fast_valid_mode == "sampled":
        raise AuditError("sampled fast-valid recheck requires a positive sample count")
    hashes = FileHashCache()
    eligibility_manifest = eligibility_manifest.resolve(strict=True)
    eligibility, eligible_ids, excluded_ids, manifest_by_id = _load_eligibility(
        eligibility_manifest,
        eligibility_manifest_sha256,
        hashes=hashes,
    )
    fixed_identities = eligibility["inputs"]["fixed_identities"]
    fast_valid_ids = (
        set(eligible_ids)
        if fast_valid_mode == "full"
        else set(eligible_ids[:fast_valid_sample_count])
        if fast_valid_mode == "sampled"
        else set()
    )
    expected_shard_names = {
        f"shard-{index:03d}-of-{EXPECTED_SHARDS:03d}" for index in range(EXPECTED_SHARDS)
    }
    observed_shard_names = {
        path.name
        for path in run_root.glob("shard-*-of-*")
        if path.is_dir()
    }
    if observed_shard_names != expected_shard_names:
        raise AuditError(
            "published shard inventory mismatch: "
            f"missing={sorted(expected_shard_names - observed_shard_names)} "
            f"extra={sorted(observed_shard_names - expected_shard_names)}"
        )
    incomplete = run_root / ".incomplete"
    allowed_root_names = expected_shard_names | ({".incomplete"} if incomplete.exists() else set())
    root_entries = list(run_root.iterdir())
    if {path.name for path in root_entries} != allowed_root_names or any(
        path.is_symlink() for path in root_entries
    ):
        raise AuditError("full-stage root contains an unexpected artifact")
    if incomplete.exists():
        allowed_lock_names = {
            f".shard-{index:03d}-of-{EXPECTED_SHARDS:03d}.publish.lock"
            for index in range(EXPECTED_SHARDS)
        }
        for path in incomplete.iterdir():
            if path.is_dir() or path.name not in allowed_lock_names or path.is_symlink():
                raise AuditError("run root retains an unexpected/incomplete attempt artifact")
    global_index = {
        sample_id: index
        for index, sample_id in enumerate(eligibility["inventory"]["full_ids"], start=1)
    }
    audits_by_arm: dict[str, dict[str, RowAudit]] = {arm: {} for arm in ARMS}
    shard_artifacts: list[dict[str, Any]] = []
    audited_rows = 0
    all_pose_digest = hashlib.sha256(b"EFFDOCK_FULL_COORDINATE_AUDIT_SDF_INVENTORY_V1\0")
    for shard_index in range(EXPECTED_SHARDS):
        shard_dir = run_root / f"shard-{shard_index:03d}-of-{EXPECTED_SHARDS:03d}"
        expected_ids = eligible_ids[shard_index::EXPECTED_SHARDS]
        paired_summary_path = shard_dir / "paired_summary.json"
        shard_entries = list(shard_dir.iterdir())
        if {path.name for path in shard_entries} != {"arms", "paired_summary.json"} or any(
            path.is_symlink() for path in shard_entries
        ):
            raise AuditError(f"{shard_dir.name}: shard artifact inventory mismatch")
        summary = _load_json(paired_summary_path, label=f"shard {shard_index} summary")
        _validate_summary(
            summary,
            shard_dir=shard_dir,
            shard_index=shard_index,
            expected_ids=expected_ids,
            eligible_ids=eligible_ids,
            excluded_ids=excluded_ids,
            eligibility_path=eligibility_manifest,
            eligibility_sha256=eligibility_manifest_sha256,
            fixed_identities=fixed_identities,
        )
        arm_root = shard_dir / "arms"
        arm_root_entries = list(arm_root.iterdir())
        if (
            {path.name for path in arm_root_entries} != set(ARMS)
            or any(path.is_symlink() or not path.is_dir() for path in arm_root_entries)
        ):
            raise AuditError(f"{shard_dir.name}: exact arm directory inventory mismatch")
        artifact_arms = summary.get("artifacts", {}).get("arms")
        if not isinstance(artifact_arms, dict) or set(artifact_arms) != set(ARMS):
            raise AuditError(f"{shard_dir.name}: summary arm artifacts mismatch")
        _require_published_path(
            summary.get("artifacts", {}).get("paired_summary"),
            paired_summary_path,
            label=f"{shard_dir.name}.artifacts.paired_summary",
        )
        per_shard_csv: dict[str, str] = {}
        for arm in ARMS:
            arm_dir = shard_dir / "arms" / arm
            arm_entries = list(arm_dir.iterdir())
            if {path.name for path in arm_entries} != {
                "poses",
                "results.csv",
                "summary.json",
            } or any(path.is_symlink() for path in arm_entries):
                raise AuditError(f"{shard_dir.name}/{arm}: arm artifact inventory mismatch")
            results_path = shard_dir / "arms" / arm / "results.csv"
            fieldnames, rows, csv_bytes = _read_csv(results_path)
            required_fields = {
                "id",
                "arm",
                "candidate_rmsds_json",
                "candidate_rmsd_method_json",
                "candidate_fast_valid_json",
                "all_poses_sdf",
                "all_poses_sdf_sha256",
            }
            if not required_fields.issubset(fieldnames):
                raise AuditError(f"{results_path}: required columns are missing")
            if [row.get("id") for row in rows] != expected_ids:
                raise AuditError(f"{results_path}: row inventory/order mismatch")
            csv_sha = hashlib.sha256(csv_bytes).hexdigest()
            artifact = artifact_arms[arm]
            _require_published_path(
                artifact.get("results_csv"),
                results_path,
                label=f"{shard_dir.name}.{arm}.artifact.results_csv",
            )
            if (
                artifact.get("results_csv_sha256") != csv_sha
                or _strict_int(artifact.get("count"), label=f"{arm}.artifact_count")
                != len(expected_ids)
            ):
                raise AuditError(f"{results_path}: paired-summary artifact identity mismatch")
            arm_summary_path = shard_dir / "arms" / arm / "summary.json"
            arm_summary = _load_json(arm_summary_path, label=f"{arm} summary")
            _require_published_path(
                arm_summary.get("results_csv"),
                results_path,
                label=f"{shard_dir.name}.{arm}.summary.results_csv",
            )
            if (
                arm_summary.get("arm") != arm
                or arm_summary.get("results_csv_sha256") != csv_sha
                or _strict_int(arm_summary.get("count"), label=f"{arm}.summary_count")
                != len(expected_ids)
            ):
                raise AuditError(f"{arm_summary_path}: arm summary identity mismatch")
            _require_published_path(
                artifact.get("summary"),
                arm_summary_path,
                label=f"{shard_dir.name}.{arm}.artifact.summary",
            )
            expected_sdf_names = {f"{sample_id}.sdf" for sample_id in expected_ids}
            poses_dir = shard_dir / "arms" / arm / "poses"
            all_pose_dir = shard_dir / "arms" / arm / "poses" / "all_poses"
            selected_dir = shard_dir / "arms" / arm / "poses" / "selected"
            if (
                {path.name for path in poses_dir.iterdir()} != {"all_poses", "selected"}
                or any(path.is_symlink() for path in poses_dir.iterdir())
                or not all_pose_dir.is_dir()
                or {path.name for path in all_pose_dir.iterdir()} != expected_sdf_names
                or not selected_dir.is_dir()
                or {path.name for path in selected_dir.iterdir()} != expected_sdf_names
                or any(path.is_symlink() or not path.is_file() for path in all_pose_dir.iterdir())
                or any(path.is_symlink() or not path.is_file() for path in selected_dir.iterdir())
            ):
                raise AuditError(f"{shard_dir.name}/{arm}: retained pose inventory mismatch")
            for sample_id, row in zip(expected_ids, rows, strict=True):
                row_audit = _audit_row(
                    row,
                    arm=arm,
                    shard_dir=shard_dir,
                    expected_id=sample_id,
                    expected_global_index=global_index[sample_id],
                    manifest_record=manifest_by_id[sample_id],
                    hashes=hashes,
                    fast_valid_recheck=sample_id in fast_valid_ids,
                )
                if sample_id in audits_by_arm[arm]:
                    raise AuditError(f"{arm}/{sample_id}: duplicate row across shards")
                audits_by_arm[arm][sample_id] = row_audit
                all_pose_digest.update(arm.encode())
                all_pose_digest.update(b"\0")
                all_pose_digest.update(sample_id.encode())
                all_pose_digest.update(b"\0")
                all_pose_digest.update(row_audit.all_poses_sha256.encode())
                all_pose_digest.update(b"\0")
                audited_rows += 1
                if progress_every and audited_rows % progress_every == 0:
                    print(
                        json.dumps(
                            {
                                "audit_progress_rows": audited_rows,
                                "expected_rows": EXPECTED_ELIGIBLE_COUNT * len(ARMS),
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
            per_shard_csv[arm] = csv_sha
        shard_artifacts.append(
            {
                "shard_index": shard_index,
                "assigned_count": len(expected_ids),
                "paired_summary": str(paired_summary_path),
                "paired_summary_sha256": hashes.sha256(paired_summary_path),
                "results_csv_sha256": per_shard_csv,
            }
        )
    expected_id_set = set(eligible_ids)
    for arm in ARMS:
        if set(audits_by_arm[arm]) != expected_id_set:
            raise AuditError(f"{arm}: full 1035-ID union mismatch")
    for sample_id in eligible_ids:
        pair_identities = {audits_by_arm[arm][sample_id].pair_identity for arm in ARMS}
        if len(pair_identities) != 1:
            raise AuditError(f"{sample_id}: independent paired identity mismatch")
    ordered = {
        arm: [audits_by_arm[arm][sample_id] for sample_id in eligible_ids] for arm in ARMS
    }
    declared_ordered = {
        arm: [_declared_view(row) for row in ordered[arm]] for arm in ARMS
    }
    comparison, decision = _paired_comparison(
        ordered[BASELINE_ARM], ordered[TREATMENT_ARM]
    )
    declared_comparison, declared_decision = _paired_comparison(
        declared_ordered[BASELINE_ARM], declared_ordered[TREATMENT_ARM]
    )
    candidate_ledger = [
        {
            "arm": arm,
            "sample_id": row.sample_id,
            "candidate_index": index,
            "csv_rmsd": declared,
            "sdf_rmsd": observed,
            "classification_changed": (declared < 2.0) != (observed < 2.0),
        }
        for arm in ARMS
        for row in ordered[arm]
        for index, declared, observed in row.quantization_ambiguous_candidates
    ]
    edge_ledger = [
        {
            "arm": arm,
            "sample_id": row.sample_id,
            "candidate_i": left,
            "candidate_j": right,
            "sdf_pair_rmsd": distance,
        }
        for arm in ARMS
        for row in ordered[arm]
        for left, right, distance in row.quantization_ambiguous_edges
    ]
    candidate_flip_count = sum(
        bool(entry["classification_changed"]) for entry in candidate_ledger
    )
    c2_changed_rows = [
        {"arm": arm, "sample_id": row.sample_id}
        for arm in ARMS
        for row in ordered[arm]
        if row.c2_component_count != row.declared_c2_component_count
    ]
    effective_decision, changed_gates, unexplained_changed_gates = (
        _reconcile_quantized_decisions(
            decision,
            declared_decision,
            candidate_flip_count=candidate_flip_count,
            c2_changed_row_count=len(c2_changed_rows),
            ambiguous_edge_count=len(edge_ledger),
        )
    )
    decision_stable = decision["passed"] == declared_decision["passed"]
    fast_recheck_candidates = sum(
        row.fast_valid_recheck_candidates for arm in ARMS for row in ordered[arm]
    )
    fast_recheck_mismatches = sum(
        row.fast_valid_recheck_mismatches for arm in ARMS for row in ordered[arm]
    )
    fast_recheck_mismatch_ledger = [
        {"arm": arm, "sample_id": row.sample_id, "candidate_index": index}
        for arm in ARMS
        for row in ordered[arm]
        for index in row.fast_valid_recheck_mismatch_indices
    ]
    auditor_path = Path(__file__).resolve()
    return {
        "schema_version": AUDIT_SCHEMA,
        "status": "complete",
        "protocol_id": PROTOCOL_ID,
        "run_root": str(run_root),
        "run_id": run_root.parent.name,
        "auditor": {
            "path": str(auditor_path),
            "sha256": file_sha256(auditor_path),
            "independence": (
                "does not import sampler evaluator or gate report aggregation; recomputes "
                "RMSD, diversity, cluster bootstrap, and decision from retained SDF coordinates"
            ),
        },
        "eligibility_manifest": {
            "path": str(eligibility_manifest),
            "sha256": eligibility_manifest_sha256,
        },
        "inventory": {
            "shard_count": EXPECTED_SHARDS,
            "arm_order": list(ARMS),
            "eligible_sample_count": EXPECTED_ELIGIBLE_COUNT,
            "excluded_sample_count": EXPECTED_EXCLUDED_COUNT,
            "system_count": EXPECTED_SYSTEM_COUNT,
            "candidate_count_per_sample": EXPECTED_NUM_SAMPLES,
            "audited_csv_rows": audited_rows,
            "parsed_sdf_records": audited_rows * EXPECTED_NUM_SAMPLES,
            "all_pose_sdf_inventory_sha256": all_pose_digest.hexdigest(),
            "shards": shard_artifacts,
        },
        "coordinate_recomputation": {
            "rmsd": (
                "RDKit CalcRMS without alignment when full-topology comparison succeeds; "
                "otherwise independent full mapped-index RMSD"
            ),
            "rmsd_csv_abs_tolerance_angstrom": RMSD_SDF_ABS_TOL,
            "diversity_csv_abs_tolerance_angstrom": DIVERSITY_SDF_ABS_TOL,
            "candidate_ensemble_hash_limitation": (
                "pre-serialization float32 ensemble bytes are not recoverable from SDF; "
                "file SHA-256 and the per-record ensemble-hash binding were verified instead"
            ),
        },
        "coordinate_quantization_sensitivity": {
            "boundary_angstrom": 2.0,
            "frozen_abs_tolerance_angstrom": RMSD_SDF_ABS_TOL,
            "ambiguous_candidate_count": len(candidate_ledger),
            "candidate_classification_flip_count": candidate_flip_count,
            "ambiguous_candidates": candidate_ledger,
            "ambiguous_pair_edge_count": len(edge_ledger),
            "ambiguous_pair_edges": edge_ledger,
            "c2_changed_row_count": len(c2_changed_rows),
            "c2_changed_rows": c2_changed_rows,
            "saved_coordinate_decision_passed": decision["passed"],
            "csv_bound_decision_passed": declared_decision["passed"],
            "decision_stable": decision_stable,
            "changed_gate_classifications": changed_gates,
            "unexplained_changed_gate_classifications": unexplained_changed_gates,
            "interpretation": (
                "K2 or C2 differences are admitted only when every changed candidate/edge "
                "lies inside the frozen SDF quantization band; all larger discrepancies fail."
            ),
        },
        "secondary_fast_valid_coordinate_recheck": {
            "role": "secondary_non_gating_due_to_SDF_coordinate_quantization",
            "mode": fast_valid_mode,
            "selection": (
                "all eligible IDs"
                if fast_valid_mode == "full"
                else f"first {len(fast_valid_ids)} lexicographically sorted eligible IDs"
                if fast_valid_mode == "sampled"
                else "disabled"
            ),
            "sample_ids": sorted(fast_valid_ids),
            "candidate_count": fast_recheck_candidates,
            "mismatch_count": fast_recheck_mismatches,
            "mismatches": fast_recheck_mismatch_ledger,
            "agreement_fraction": (
                1.0 - fast_recheck_mismatches / fast_recheck_candidates
                if fast_recheck_candidates
                else None
            ),
            "note": (
                "All SDF fast_valid properties were still required to match the CSV exactly "
                "for every candidate; this coordinate recheck duplicates the formula on rounded "
                "SDF coordinates and is descriptive rather than a decision gate."
            ),
        },
        "arms": {
            arm: {
                "saved_coordinate_metrics": _arm_metrics(ordered[arm]),
                "csv_bound_metrics": _arm_metrics(declared_ordered[arm]),
            }
            for arm in ARMS
        },
        "operational_full_split_sensitivity": {
            "arms": {
                arm: _operational_full_split_metrics(ordered[arm]) for arm in ARMS
            },
            "primary_comparison": _operational_full_split_comparison(
                ordered[BASELINE_ARM], ordered[TREATMENT_ARM]
            ),
        },
        "primary_comparison": comparison,
        "csv_bound_primary_comparison": declared_comparison,
        "diagnostic_comparisons": {
            "s25_to_s50": _descriptive_pair(
                ordered["s25_ema"],
                ordered["s50_ema"],
                left_arm="s25_ema",
                right_arm="s50_ema",
            ),
            "s25_to_treatment": _descriptive_pair(
                ordered["s25_ema"],
                ordered[TREATMENT_ARM],
                left_arm="s25_ema",
                right_arm=TREATMENT_ARM,
            ),
        },
        "csv_bound_decision": declared_decision,
        "saved_coordinate_decision": decision,
        "decision": effective_decision,
        "runtime": {
            "elapsed_seconds": time.monotonic() - started,
            "python": sys.version.split()[0],
            "rdkit": rdBase.rdkitVersion,
        },
    }


def _write_noreplace(path: Path, payload: dict[str, Any]) -> str:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    data = _canonical_json_bytes(payload)
    try:
        with path.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise AuditError(f"refusing to overwrite audit output: {path}") from exc
    return hashlib.sha256(data).hexdigest()


def _failure_payload(
    *,
    run_root: Path,
    eligibility_manifest: Path,
    error: Exception,
) -> dict[str, Any]:
    auditor_path = Path(__file__).resolve()
    return {
        "schema_version": AUDIT_SCHEMA,
        "status": "failed",
        "protocol_id": PROTOCOL_ID,
        "run_root": str(run_root.resolve()),
        "eligibility_manifest": str(eligibility_manifest.resolve()),
        "auditor": {"path": str(auditor_path), "sha256": file_sha256(auditor_path)},
        "failure": {"error_type": type(error).__name__, "message": str(error)},
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        type=Path,
        required=True,
        help="Published full-stage directory (<output-root>/<run-id>/full).",
    )
    parser.add_argument("--eligibility-manifest", type=Path, required=True)
    parser.add_argument("--eligibility-manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--fast-valid-mode",
        choices=("sampled", "full"),
        default="sampled",
        help="Coordinate-level secondary fast-valid recheck scope (all properties are always checked).",
    )
    parser.add_argument("--fast-valid-sample-count", type=int, default=32)
    parser.add_argument("--progress-every", type=int, default=100)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.output.exists():
        print(f"refusing to overwrite audit output: {args.output}", file=sys.stderr)
        return 2
    try:
        report = audit_full(
            run_root=args.run_root,
            eligibility_manifest=args.eligibility_manifest,
            eligibility_manifest_sha256=args.eligibility_manifest_sha256,
            fast_valid_mode=args.fast_valid_mode,
            fast_valid_sample_count=args.fast_valid_sample_count,
            progress_every=args.progress_every,
        )
        output_sha256 = _write_noreplace(args.output, report)
        print(
            json.dumps(
                {
                    "status": "complete",
                    "passed": report["decision"]["passed"],
                    "action": report["decision"]["action"],
                    "output": str(args.output.resolve()),
                    "sha256": output_sha256,
                },
                sort_keys=True,
            )
        )
        return 0
    except Exception as exc:
        try:
            output_sha256 = _write_noreplace(
                args.output,
                _failure_payload(
                    run_root=args.run_root,
                    eligibility_manifest=args.eligibility_manifest,
                    error=exc,
                ),
            )
            print(
                json.dumps(
                    {
                        "status": "failed",
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                        "output": str(args.output.resolve()),
                        "sha256": output_sha256,
                    },
                    sort_keys=True,
                ),
                file=sys.stderr,
            )
        except Exception as output_exc:
            print(
                f"audit failed ({type(exc).__name__}: {exc}); "
                f"failure output also failed ({type(output_exc).__name__}: {output_exc})",
                file=sys.stderr,
            )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
