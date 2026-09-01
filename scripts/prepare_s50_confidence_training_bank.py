#!/usr/bin/env python3
"""Build a deployment-matched S50 pose bank for confidence training.

The workflow has three fail-closed stages:

``freeze-inputs``
    Freeze the exact train/validation split, canonical PLINDER SMILES, processed
    tensor assets, deterministic seeds, sampler identities, and input-only
    eligibility.  No sampled pose or quality outcome is used for eligibility.

``generate-shard``
    Generate one disjoint shard with the frozen S50 deterministic ODE contract.
    Validation poses may be read byte-for-byte from the already sealed S50 SDF
    bank.  Ligand hidden features are extracted at t=1 in bounded chunks.

``aggregate``
    Verify every expected shard and pose artifact before publishing the sole
    training manifest and its filtered split.  Full aggregation refuses partial
    shards; an explicitly marked smoke subset is never claim eligible.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import rdMolAlign

from effdock.confidence.features import extract_t1_ligand_irreps
from effdock.evaluation.fragment_geometry import (
    enumerate_full_atom_mappings,
    fragment_rigid_fit_floor,
)
from effdock.inference.docking import load_model
from effdock.inference.preprocess import build_inference_bundle
from effdock.inference.sampler import sample_shared_prior_states, sample_unified
from effdock.preprocess.fragments import decompose_fragments
from effdock.preprocess.ligand import (
    BOND_STEREO_MAP,
    BOND_TYPE_MAP,
    CHIRALITY_MAP,
    ELEMENT_VOCAB,
    OTHER_ELEMENT_IDX,
    featurize_ligand,
)
from effdock.workflows.benchmark_inputs import (
    BenchmarkInputMismatchError,
    file_sha256,
    full_heavy_atom_mapping_metadata,
    ligand_input_identity,
    load_benchmark_ligand,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_ID = "EFFDOCK-S50-CONFIDENCE-TRAINING-BANK-V1"
STUDY_PROTOCOL_ID = "EFFDOCK-S50-MATCHED-CONFIDENCE-TRAIN-VAL-V1"
INPUT_SCHEMA = "effdock.s50_confidence_bank.inputs.v1"
SHARD_SCHEMA = "effdock.s50_confidence_bank.shard.v1"
MANIFEST_SCHEMA = "effdock.s50_confidence_bank.manifest.v1"
POSE_STORAGE_VERSION = "effdock_confidence_s50_deployment_pose_v1"
DEFAULT_POSE_TAG = "s50_n100_s10_sig2_latep3_pc10_rdkitseed0"

NUM_SAMPLES = 100
NUM_STEPS = 10
SIGMA = 2.0
TIME_SCHEDULE = "late"
SCHEDULE_POWER = 3.0
POCKET_CUTOFF = 10.0
PRIOR_POOL_SIZE = 100
CONFORMER_SEED = 0
BASE_SEED = 42
HIDDEN_CHUNK_SIZE = 20
MAPPING_FLOAT_TOLERANCE = 1e-12
MAPPING_FLOAT_DIAGNOSTICS = frozenset(
    {"rigid_fragment_floor_rmsd", "pair_distance_rmse"}
)

RUNTIME_FILES = {
    "builder": Path(__file__).resolve(),
    "benchmark_inputs": PROJECT_ROOT / "src/effdock/workflows/benchmark_inputs.py",
    "benchmark_evaluation": PROJECT_ROOT / "src/effdock/evaluation/benchmark.py",
    "fragment_geometry_evaluation": PROJECT_ROOT / "src/effdock/evaluation/fragment_geometry.py",
    "inference_preprocess": PROJECT_ROOT / "src/effdock/inference/preprocess.py",
    "inference_sampler": PROJECT_ROOT / "src/effdock/inference/sampler.py",
    "inference_docking": PROJECT_ROOT / "src/effdock/inference/docking.py",
    "confidence_features": PROJECT_ROOT / "src/effdock/confidence/features.py",
    "checkpoint_loader": PROJECT_ROOT / "src/effdock/checkpoint.py",
    "data_dataset": PROJECT_ROOT / "src/effdock/data/dataset.py",
    "effdock_model": PROJECT_ROOT / "src/effdock/models/effdock.py",
    "equivariant_model": PROJECT_ROOT / "src/effdock/models/equivariant.py",
    "nn_utils": PROJECT_ROOT / "src/effdock/models/nn_utils.py",
    "se3_geometry": PROJECT_ROOT / "src/effdock/geometry/se3.py",
    "fragment_preprocess": PROJECT_ROOT / "src/effdock/preprocess/fragments.py",
    "ligand_preprocess": PROJECT_ROOT / "src/effdock/preprocess/ligand.py",
    "graph_preprocess": PROJECT_ROOT / "src/effdock/preprocess/graph.py",
    "dependency_lock": PROJECT_ROOT / "uv.lock",
}


class BankContractError(RuntimeError):
    """Raised when a frozen identity or bank invariant is violated."""


class InputCompatibilityError(ValueError):
    """A sealed, outcome-independent input cannot enter the inference pipeline."""

    def __init__(self, reason_code: str, message: str) -> None:
        super().__init__(message)
        self.reason_code = reason_code


def _mapping_metadata_matches(observed: Any, expected: Any) -> bool:
    """Compare a frozen atom-map record without weakening its discrete identity."""
    if not isinstance(observed, dict) or not isinstance(expected, dict):
        return False
    if set(observed) != set(expected):
        return False
    for key in observed:
        observed_value = observed[key]
        expected_value = expected[key]
        if key not in MAPPING_FLOAT_DIAGNOSTICS:
            if observed_value != expected_value:
                return False
            continue
        if (
            isinstance(observed_value, bool)
            or isinstance(expected_value, bool)
            or not isinstance(observed_value, (int, float))
            or not isinstance(expected_value, (int, float))
        ):
            return False
        observed_float = float(observed_value)
        expected_float = float(expected_value)
        if (
            not math.isfinite(observed_float)
            or not math.isfinite(expected_float)
            or not math.isclose(
                observed_float,
                expected_float,
                rel_tol=MAPPING_FLOAT_TOLERANCE,
                abs_tol=MAPPING_FLOAT_TOLERANCE,
            )
        ):
            return False
    return True


@dataclass(frozen=True)
class ProcessedPaths:
    sample_key: str
    root: Path
    protein: Path
    ligand: Path
    meta: Path


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _ordered_ids_sha256(ids: Sequence[str]) -> str:
    return hashlib.sha256("".join(f"{sample_id}\n" for sample_id in ids).encode()).hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _asset(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {
        "path": str(resolved),
        "sha256": file_sha256(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _require_asset(path: Path, expected_sha256: str, label: str) -> dict[str, Any]:
    identity = _asset(path)
    if identity["sha256"] != expected_sha256:
        raise BankContractError(
            f"{label} SHA-256 mismatch: expected {expected_sha256}, got {identity['sha256']}"
        )
    return identity


def _verify_asset(identity: dict[str, Any], label: str) -> Path:
    if not isinstance(identity, dict):
        raise BankContractError(f"{label} identity is not an object")
    path = Path(str(identity.get("path", ""))).resolve(strict=True)
    if str(path) != identity.get("path"):
        raise BankContractError(f"{label} path is not canonical")
    current = _asset(path)
    expected_size = identity.get("size_bytes")
    if current["sha256"] != identity.get("sha256") or (
        expected_size is not None and current["size_bytes"] != expected_size
    ):
        raise BankContractError(f"{label} changed after input freeze")
    return path


def _require_unchanged_asset(
    identity_before: dict[str, Any],
    path: Path,
    label: str,
) -> dict[str, Any]:
    """Rehash an asset after reading and reject any concurrent mutation."""
    identity_after = _asset(path)
    if identity_after != identity_before:
        raise BankContractError(f"{label} changed while being validated")
    return identity_after


def _input_compatibility_error(reason_code: str, exc: BaseException) -> InputCompatibilityError:
    return InputCompatibilityError(reason_code, f"{reason_code}: {exc}")


def _atomic_write_noreplace(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite artifact: {path}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_torch_save_noreplace(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite pose artifact: {path}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _runtime_code_identity() -> dict[str, Any]:
    files = {name: _asset(path) for name, path in RUNTIME_FILES.items()}
    stable = {name: value["sha256"] for name, value in files.items()}
    digest = hashlib.sha256(
        b"EFFDOCK_S50_CONFIDENCE_BANK_RUNTIME_V1\0"
        + json.dumps(stable, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()
    return {"aggregate_sha256": digest, "files": files}


def _validate_builder_pin(expected_builder_sha256: str) -> None:
    actual = file_sha256(Path(__file__).resolve())
    if actual != expected_builder_sha256:
        raise BankContractError(
            f"builder SHA-256 mismatch: expected {expected_builder_sha256}, got {actual}"
        )


def _fixed_settings() -> dict[str, Any]:
    return {
        "num_samples": NUM_SAMPLES,
        "num_steps": NUM_STEPS,
        "sample_sigma": SIGMA,
        "time_schedule": TIME_SCHEDULE,
        "schedule_power": SCHEDULE_POWER,
        "pocket_cutoff_angstrom": POCKET_CUTOFF,
        "prior_pool_size": PRIOR_POOL_SIZE,
        "ligand_conformer_seed": CONFORMER_SEED,
        "sampling_dynamics": "deterministic_ode",
        "stochastic_gamma": 0.0,
        "translation_sde_base_sigma": 0.0,
        "guidance": False,
        "refine": "none",
        "fk_resampling": False,
        "particle_resampling": False,
        "eligibility_boundary": "input_only_no_sampled_pose_outcomes",
    }


def _validate_fixed_settings(settings: Any) -> None:
    if settings != _fixed_settings():
        raise BankContractError("sampler settings differ from the frozen S50 contract")


def _processed_paths(processed_root: Path, sample_key: str) -> ProcessedPaths:
    if not sample_key or Path(sample_key).name != sample_key or sample_key in {".", ".."}:
        raise ValueError(f"unsafe PLINDER sample key: {sample_key!r}")
    root = processed_root / sample_key
    return ProcessedPaths(
        sample_key=sample_key,
        root=root,
        protein=root / "protein.pt",
        ligand=root / "ligand.pt",
        meta=root / "meta.pt",
    )


def _load_processed(paths: ProcessedPaths) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    protein = torch.load(paths.protein, map_location="cpu", weights_only=True)
    ligand = torch.load(paths.ligand, map_location="cpu", weights_only=True)
    meta = torch.load(paths.meta, map_location="cpu", weights_only=True)
    if not all(isinstance(value, dict) for value in (protein, ligand, meta)):
        raise TypeError(f"{paths.sample_key}: processed tensors must be mappings")
    center = torch.as_tensor(meta.get("pocket_center"), dtype=torch.float32)
    if center.shape != (3,) or not bool(torch.isfinite(center).all()):
        raise ValueError(f"{paths.sample_key}: invalid processed pocket center")
    return protein, ligand, meta


def _canonical_ligand_data(molecule: Chem.Mol) -> dict[str, torch.Tensor]:
    ligand = featurize_ligand(molecule)
    if ligand is None:
        raise ValueError("canonical-SMILES ligand featurization failed")
    fragments = decompose_fragments(molecule, ligand["atom_coords"])
    if fragments is None:
        raise ValueError("canonical-SMILES fragment decomposition failed")
    for key in (
        "fragment_id",
        "frag_centers",
        "frag_local_coords",
        "frag_sizes",
        "tri_edge_index",
        "tri_edge_ref_dist",
        "fragment_adj_index",
        "cut_bond_index",
    ):
        ligand[key] = fragments[key]
    return ligand


def reconstruct_processed_reference(
    ligand: dict[str, torch.Tensor],
    molecule_input: Chem.Mol | None = None,
) -> Chem.Mol:
    """Reconstruct the ordered processed ligand graph and crystal conformer.

    The reconstruction is exact for the featurizer vocab.  An ``OTHER`` element
    is recovered from the canonical input only when its element multiset leaves
    one unique missing atomic number; ambiguous mixtures remain fail-closed.
    Processed atom order, coordinates, and connectivity remain authoritative.
    ``OTHER`` bond types are never guessed.  Stereo tensors are restored and
    then reassigned from the stored crystal coordinates.
    """
    element_inverse = {value: key for key, value in ELEMENT_VOCAB.items()}
    bond_inverse = {value: key for key, value in BOND_TYPE_MAP.items()}
    chirality_inverse = {value: key for key, value in CHIRALITY_MAP.items()}
    stereo_inverse = {value: key for key, value in BOND_STEREO_MAP.items()}

    elements = torch.as_tensor(ligand["atom_element"]).to(torch.long).tolist()
    charges = torch.as_tensor(ligand["atom_charge"]).to(torch.long).tolist()
    chirality = torch.as_tensor(ligand["atom_chirality"]).to(torch.long).tolist()
    aromatic = torch.as_tensor(ligand["atom_aromatic"]).to(torch.bool).tolist()
    unsupported_codes = {
        int(value)
        for value in elements
        if value not in element_inverse and value != OTHER_ELEMENT_IDX
    }
    if unsupported_codes:
        raise ValueError(
            f"processed ligand contains invalid element token(s): {sorted(unsupported_codes)}"
        )
    other_count = sum(value == OTHER_ELEMENT_IDX for value in elements)
    inferred_other_atomic_number: int | None = None
    if other_count:
        if molecule_input is None or molecule_input.GetNumAtoms() != len(elements):
            raise ValueError(
                "processed OTHER element requires an atom-count-matched canonical input"
            )
        canonical_counts = Counter(
            atom.GetAtomicNum() for atom in molecule_input.GetAtoms()
        )
        known_counts = Counter(
            element_inverse[value] for value in elements if value in element_inverse
        )
        if any(canonical_counts[atomic_number] < count for atomic_number, count in known_counts.items()):
            raise ValueError(
                "canonical/processed known-element multisets are incompatible"
            )
        remaining_counts = canonical_counts.copy()
        remaining_counts.subtract(known_counts)
        positive_remaining = {
            int(atomic_number): int(count)
            for atomic_number, count in remaining_counts.items()
            if count > 0
        }
        if (
            sum(positive_remaining.values()) != other_count
            or len(positive_remaining) != 1
        ):
            raise ValueError(
                "processed OTHER element is ambiguous under the canonical element multiset"
            )
        inferred_other_atomic_number = next(iter(positive_remaining))
        if inferred_other_atomic_number in ELEMENT_VOCAB:
            raise ValueError(
                "processed OTHER element resolves to an in-vocabulary atomic number"
            )

    editable = Chem.RWMol()
    for element, charge, chiral, is_aromatic in zip(
        elements, charges, chirality, aromatic, strict=True
    ):
        atomic_number = (
            inferred_other_atomic_number
            if element == OTHER_ELEMENT_IDX
            else element_inverse[element]
        )
        if atomic_number is None:  # pragma: no cover - guarded above
            raise AssertionError("OTHER element inference was not resolved")
        atom = Chem.Atom(int(atomic_number))
        atom.SetFormalCharge(int(charge))
        atom.SetIsAromatic(bool(is_aromatic))
        if chiral in chirality_inverse:
            atom.SetChiralTag(chirality_inverse[chiral])
        editable.AddAtom(atom)

    bond_index = torch.as_tensor(ligand["bond_index"]).to(torch.long)
    bond_type = torch.as_tensor(ligand["bond_type"]).to(torch.long)
    bond_stereo = torch.as_tensor(ligand["bond_stereo"]).to(torch.long)
    seen: set[tuple[int, int]] = set()
    stereo_by_edge: dict[tuple[int, int], int] = {}
    for column in range(bond_index.shape[1]):
        atom_i = int(bond_index[0, column])
        atom_j = int(bond_index[1, column])
        edge = tuple(sorted((atom_i, atom_j)))
        if edge in seen:
            continue
        seen.add(edge)
        code = int(bond_type[column])
        if code not in bond_inverse:
            raise ValueError("processed ligand contains an unsupported OTHER bond type")
        editable.AddBond(edge[0], edge[1], bond_inverse[code])
        stereo_by_edge[edge] = int(bond_stereo[column])

    molecule = editable.GetMol()
    for edge, code in stereo_by_edge.items():
        if code in stereo_inverse:
            molecule.GetBondBetweenAtoms(*edge).SetStereo(stereo_inverse[code])
    conformer = Chem.Conformer(len(elements))
    coords = torch.as_tensor(ligand["atom_coords"], dtype=torch.float64)
    if coords.shape != (len(elements), 3) or not bool(torch.isfinite(coords).all()):
        raise ValueError("processed crystal ligand coordinates are invalid")
    for atom_index, point in enumerate(coords.tolist()):
        conformer.SetAtomPosition(atom_index, point)
    molecule.AddConformer(conformer, assignId=True)
    molecule.UpdatePropertyCache(strict=False)
    Chem.GetSymmSSSR(molecule)
    try:
        Chem.AssignStereochemistryFrom3D(molecule, confId=0, replaceExistingTags=True)
        Chem.AssignStereochemistry(molecule, cleanIt=True, force=True)
    except Exception:
        pass
    return molecule


def _analyze_mapping_deterministic(
    molecule_reference: Chem.Mol,
    molecule_input: Chem.Mol,
    reference_ligand: dict[str, torch.Tensor],
) -> dict[str, Any]:
    """Select the strict/stereo-aware min-floor map with an explicit tie break."""
    mappings, method, truncated = enumerate_full_atom_mappings(
        molecule_reference, molecule_input, max_matches=1024
    )
    if truncated:
        raise ValueError("full atom mapping enumeration exceeded 1024 matches")
    if not mappings:
        raise ValueError(f"no stereo-compatible full atom mapping ({method})")
    input_coords = torch.tensor(
        molecule_input.GetConformer().GetPositions(), dtype=torch.float64
    )
    input_fragments = decompose_fragments(molecule_input, input_coords)
    if input_fragments is None:
        raise ValueError("canonical input fragment decomposition failed during mapping")
    reference_coords = torch.as_tensor(reference_ligand["atom_coords"], dtype=torch.float64)
    reference_fragments = torch.as_tensor(reference_ligand["fragment_id"], dtype=torch.long)
    candidates: list[tuple[float, tuple[int, ...], dict[str, Any]]] = []
    for mapping in mappings:
        result = fragment_rigid_fit_floor(
            reference_coords,
            input_coords,
            reference_fragments,
            input_fragments["fragment_id"],
            mapping,
        )
        floor = float(result["rigid_fragment_floor_rmsd"])
        pair_rmse = float(result["pair_distance_rmse"])
        if not math.isfinite(floor) or not math.isfinite(pair_rmse):
            raise ValueError("atom mapping diagnostics are non-finite")
        candidates.append((floor, tuple(mapping), result))
    minimum_floor = min(candidate[0] for candidate in candidates)
    tied_candidates = [
        candidate
        for candidate in candidates
        if candidate[0] <= minimum_floor + MAPPING_FLOAT_TOLERANCE
    ]
    _, mapping, result = min(tied_candidates, key=lambda item: item[1])
    return {
        **result,
        "mapping_method": method,
        "mapping_count": len(mappings),
        "mapping_truncated": False,
        "symmetry_complete": method == "strict_stereo",
        "inference_to_crystal": list(mapping),
    }


def _mapping_record(
    molecule_input: Chem.Mol,
    molecule_reference_raw: Chem.Mol,
    reference_ligand: dict[str, torch.Tensor],
) -> tuple[dict[str, Any], list[int], Chem.Mol]:
    # First establish a stereo-compatible full map against the representation
    # reconstructed from processed tensors.  This is the authoritative
    # element/connectivity gate even when aromatic/bond-order encodings differ.
    raw_analysis = _analyze_mapping_deterministic(
        molecule_reference_raw, molecule_input, reference_ligand
    )
    raw_mapping = [int(value) for value in raw_analysis["inference_to_crystal"]]
    raw_metadata = full_heavy_atom_mapping_metadata(
        molecule_reference_raw,
        molecule_input,
        list(range(molecule_input.GetNumAtoms())),
        raw_mapping,
        str(raw_analysis["mapping_method"]),
    )
    if not raw_metadata["accepted"]:
        raise ValueError(f"processed reference connectivity gate failed: {raw_metadata}")

    # Normalize only the RDKit chemistry representation to the canonical input
    # while retaining processed atom order and crystal coordinates.  This makes
    # strict-stereo symmetry enumeration and CalcRMS available without treating
    # the crystal SDF/tensor representation as the sampler input.
    input_index_by_reference = [-1] * molecule_reference_raw.GetNumAtoms()
    for input_index, reference_index in enumerate(raw_mapping):
        input_index_by_reference[reference_index] = input_index
    if sorted(input_index_by_reference) != list(range(molecule_input.GetNumAtoms())):
        raise ValueError("raw processed mapping is not a full bijection")
    molecule_reference = Chem.RenumberAtoms(molecule_input, input_index_by_reference)
    molecule_reference.RemoveAllConformers()
    reference_coords = torch.as_tensor(reference_ligand["atom_coords"], dtype=torch.float64)
    reference_conformer = Chem.Conformer(molecule_reference.GetNumAtoms())
    for atom_index, point in enumerate(reference_coords.tolist()):
        reference_conformer.SetAtomPosition(atom_index, point)
    molecule_reference.AddConformer(reference_conformer, assignId=True)

    analysis = _analyze_mapping_deterministic(
        molecule_reference, molecule_input, reference_ligand
    )
    if not analysis["symmetry_complete"]:
        raise ValueError("chemistry-normalized reference did not yield strict-stereo mapping")
    input_to_reference = [int(value) for value in analysis["inference_to_crystal"]]
    dock_indices = list(range(molecule_input.GetNumAtoms()))
    ref_indices = input_to_reference
    method = str(analysis["mapping_method"])
    metadata = full_heavy_atom_mapping_metadata(
        molecule_reference, molecule_input, dock_indices, ref_indices, method
    )
    if not metadata["accepted"]:
        raise ValueError(f"full element/connectivity mapping failed: {metadata}")
    if sorted(input_to_reference) != list(range(molecule_reference.GetNumAtoms())):
        raise ValueError("atom mapping is not a complete bijection")
    selected_metadata = {
        **metadata,
        "mapping_method": analysis["mapping_method"],
        "mapping_count": int(analysis["mapping_count"]),
        "mapping_truncated": bool(analysis["mapping_truncated"]),
        "symmetry_complete": bool(analysis["symmetry_complete"]),
        "stored_partition_equal": bool(analysis["stored_partition_equal"]),
        "rigid_fragment_floor_rmsd": float(analysis["rigid_fragment_floor_rmsd"]),
        "pair_distance_rmse": float(analysis["pair_distance_rmse"]),
        "processed_representation_relation": raw_metadata["relation"],
        "processed_graph_mapping_method": raw_analysis["mapping_method"],
        "processed_graph_symmetry_complete": bool(raw_analysis["symmetry_complete"]),
    }
    return selected_metadata, input_to_reference, molecule_reference


def _load_split(path: Path) -> dict[str, list[str]]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("split file must contain an object")
    result: dict[str, list[str]] = {}
    for split in ("train", "val"):
        values = payload.get(split)
        if (
            not isinstance(values, list)
            or not values
            or not all(isinstance(value, str) and value for value in values)
            or len(values) != len(set(values))
        ):
            raise ValueError(f"split {split!r} must contain unique non-empty string IDs")
        result[split] = values
    overlap = sorted(set(result["train"]) & set(result["val"]))
    if overlap:
        raise ValueError(f"train/val split overlap detected: {overlap[:5]}")
    return result


def _load_pool_smiles(pool_path: Path, ids: set[str]) -> dict[str, str]:
    frame = pd.read_parquet(
        pool_path,
        columns=("system_id", "ligand_instance_chain", "ligand_rdkit_canonical_smiles"),
    )
    frame = frame.assign(
        sample_key=[
            f"{system_id}__{chain}"
            for system_id, chain in zip(
                frame["system_id"], frame["ligand_instance_chain"], strict=True
            )
        ]
    )
    if bool(frame["sample_key"].duplicated().any()):
        raise ValueError("PLINDER pool contains duplicate sample keys")
    rows = frame.set_index("sample_key")["ligand_rdkit_canonical_smiles"].to_dict()
    missing = sorted(ids - rows.keys())
    if missing:
        raise ValueError(f"{len(missing)} frozen split IDs are absent from PLINDER pool")
    output: dict[str, str] = {}
    for sample_id in ids:
        smiles = rows[sample_id]
        if not isinstance(smiles, str) or not smiles.strip():
            raise ValueError(f"{sample_id}: canonical PLINDER SMILES is missing")
        output[sample_id] = smiles.strip()
    return output


def _load_val_bank(path: Path, expected_sha256: str) -> tuple[dict[str, Any], dict[str, Any]]:
    _require_asset(path, expected_sha256, "reusable validation S50 bank")
    payload = json.loads(path.read_text())
    if (
        payload.get("schema_version") != "effdock.early_time_sampler_s50_confidence_bank.v1"
        or payload.get("status") != "complete_label_free"
    ):
        raise BankContractError("reusable validation bank schema/status mismatch")
    settings = payload.get("fixed_settings", {})
    expected = {
        "pose_count": NUM_SAMPLES,
        "num_steps": NUM_STEPS,
        "sample_sigma": SIGMA,
        "prior_pool_size": PRIOR_POOL_SIZE,
        "ligand_conformer_seed": CONFORMER_SEED,
    }
    if any(settings.get(key) != value for key, value in expected.items()):
        raise BankContractError("reusable validation bank settings differ from S50 contract")
    if payload.get("backbone_arms", {}).get("s50_backbone", {}).get("sha256") is None:
        raise BankContractError("reusable validation bank lacks S50 checkpoint identity")
    records = payload.get("records")
    if not isinstance(records, list):
        raise BankContractError("reusable validation bank lacks records")
    by_id = {str(record.get("sample_key")): record for record in records}
    if len(by_id) != len(records):
        raise BankContractError("reusable validation bank has duplicate IDs")
    return payload, by_id


def _preflight_one(task: dict[str, Any]) -> dict[str, Any]:
    sample_id = str(task["sample_key"])
    split = str(task["split"])
    split_index = int(task["split_index"])
    base = {
        "sample_key": sample_id,
        "split": split,
        "split_index": split_index,
        "global_index": split_index,
        "sampling_seed": BASE_SEED + split_index,
        "ligand_conformer_seed": CONFORMER_SEED,
    }
    try:
        processed_root = Path(task["processed_root"])
        paths = _processed_paths(processed_root, sample_id)
        assets = {
            "processed_protein": _asset(paths.protein),
            "processed_ligand_reference": _asset(paths.ligand),
            "processed_meta": _asset(paths.meta),
        }
        protein, reference_ligand, meta = _load_processed(paths)
        system_id = meta.get("plinder_system_id")
        if not isinstance(system_id, str) or not system_id:
            raise BankContractError(
                f"{sample_id}: processed metadata lacks authoritative plinder_system_id"
            )
        base["system_id"] = system_id
        smiles = str(task["canonical_smiles"])
        try:
            molecule_input, _ = load_benchmark_ligand(
                smiles, random_seed=CONFORMER_SEED
            )
            ligand_data = _canonical_ligand_data(molecule_input)
        except (AssertionError, BenchmarkInputMismatchError, ValueError) as exc:
            raise _input_compatibility_error(
                "canonical_ligand_preparation_failed", exc
            ) from exc
        try:
            molecule_reference_raw = reconstruct_processed_reference(
                reference_ligand, molecule_input
            )
            mapping, input_to_reference, _ = _mapping_record(
                molecule_input, molecule_reference_raw, reference_ligand
            )
        except ValueError as exc:
            raise _input_compatibility_error(
                "processed_reference_mapping_failed", exc
            ) from exc
        bundle = build_inference_bundle(
            protein, ligand_data, meta, pocket_cutoff=POCKET_CUTOFF
        )
        if bundle is None:
            raise InputCompatibilityError(
                "inference_graph_preparation_failed",
                "processed-protein canonical-ligand inference graph failed",
            )
        graph, _, inference_meta = bundle
        center = torch.as_tensor(inference_meta["pocket_center"], dtype=torch.float32)
        if center.shape != (3,) or not bool(torch.isfinite(center).all()):
            raise ValueError("inference pocket center is invalid")
        if int(inference_meta["num_atom"]) != molecule_input.GetNumAtoms():
            raise ValueError("canonical ligand atom count changed during preprocessing")
        if not graph or not all(torch.is_tensor(value) for value in graph.values()):
            raise ValueError("inference graph must be a non-empty tensor mapping")

        record: dict[str, Any] = {
            **base,
            "status": "eligible",
            "system_id": system_id,
            "canonical_smiles": smiles,
            "canonical_smiles_raw_sha256": _sha256_text(smiles),
            "ligand_input_identity": ligand_input_identity(sample_id, smiles),
            "pocket_center": [float(value) for value in center.tolist()],
            "num_input_atoms": molecule_input.GetNumAtoms(),
            "num_fragments": int(inference_meta["num_frag"]),
            "mapping": mapping,
            "input_to_reference": input_to_reference,
            **assets,
        }
        val_source = task.get("val_pose_bank")
        if split == "val":
            if not isinstance(val_source, dict):
                raise InputCompatibilityError(
                    "reusable_validation_pose_unavailable",
                    "eligible validation input is absent from reusable S50 bank",
                )
            source_smiles = val_source.get("canonical_smiles")
            source_center = val_source.get("pocket_center")
            if (
                source_smiles != smiles
                or val_source.get("system_id") != system_id
                or int(val_source.get("sampling_seed", -1)) != base["sampling_seed"]
                or int(val_source.get("ligand_conformer_seed", -1)) != CONFORMER_SEED
                or int(val_source.get("pose_count", -1)) != NUM_SAMPLES
                or int(val_source.get("num_steps", -1)) != NUM_STEPS
                or not math.isclose(
                    float(val_source.get("sample_sigma", float("nan"))),
                    SIGMA,
                    rel_tol=0.0,
                    abs_tol=0.0,
                )
                or source_center != record["pocket_center"]
            ):
                raise BankContractError(
                    f"{sample_id}: reusable validation record identity/settings mismatch"
                )
            source_asset = val_source.get("all_poses_sdf")
            source_path = _verify_asset(source_asset, f"{sample_id}.validation_pose_sdf")
            candidate_sha = val_source.get("candidate_ensemble_sha256")
            prior_sha = val_source.get("prior_pool_sha256")
            if not all(
                isinstance(value, str)
                and len(value) == 64
                and all(character in "0123456789abcdef" for character in value)
                for value in (candidate_sha, prior_sha)
            ):
                raise BankContractError(
                    f"{sample_id}: reusable validation pose/prior identity is invalid"
                )
            record["val_pose_bank"] = {
                "all_poses_sdf": _asset(source_path),
                "candidate_ensemble_sha256": candidate_sha,
                "prior_pool_sha256": prior_sha,
            }
        return record
    except InputCompatibilityError as exc:
        # Only explicitly classified, outcome-independent incompatibilities may
        # alter the derived loader cohort.  Integrity, I/O, and programming
        # failures propagate and fail the complete preflight.
        return {
            **base,
            "status": "input_ineligible",
            "failure_type": type(exc.__cause__ or exc).__name__,
            "failure_reason": exc.reason_code,
            "failure_message": str(exc),
        }


def freeze_inputs(args: argparse.Namespace) -> dict[str, Any]:
    _validate_builder_pin(args.expected_builder_sha256)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite input manifest: {args.output}")
    split_asset = _require_asset(args.split_file, args.expected_split_sha256, "PLINDER split")
    pool_asset = _require_asset(
        args.pool_parquet, args.expected_pool_sha256, "PLINDER canonical-SMILES pool"
    )
    checkpoint_asset = _require_asset(
        args.checkpoint, args.expected_checkpoint_sha256, "S50 sampler checkpoint"
    )
    config_asset = _require_asset(args.config, args.expected_config_sha256, "sampler config")
    checkpoint_payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    expected_checkpoint_fields = {
        "artifact_type": "effdock_ema_inference_checkpoint",
        "inference_only": True,
        "weight_source": "ema",
        "step": 50000,
        "source_checkpoint_step": 50000,
    }
    if any(checkpoint_payload.get(key) != value for key, value in expected_checkpoint_fields.items()):
        raise BankContractError("sampler checkpoint is not the retained S50 EMA artifact")

    split = _load_split(args.split_file)
    smiles_by_id = _load_pool_smiles(args.pool_parquet, set(split["train"] + split["val"]))
    val_payload, val_by_id = _load_val_bank(
        args.val_bank_manifest, args.expected_val_bank_manifest_sha256
    )
    val_checkpoint_sha = val_payload["backbone_arms"]["s50_backbone"]["sha256"]
    if val_checkpoint_sha != checkpoint_asset["sha256"]:
        raise BankContractError("reusable validation bank was not sampled by the pinned S50 EMA")
    unknown_val_ids = sorted(set(val_by_id) - set(split["val"]))
    if unknown_val_ids:
        raise BankContractError("reusable validation bank contains IDs outside frozen validation")

    tasks: list[dict[str, Any]] = []
    for split_name in ("train", "val"):
        for index, sample_id in enumerate(split[split_name], start=1):
            tasks.append(
                {
                    "sample_key": sample_id,
                    "split": split_name,
                    "split_index": index,
                    "canonical_smiles": smiles_by_id[sample_id],
                    "processed_root": str(args.processed_root.resolve()),
                    "val_pose_bank": val_by_id.get(sample_id) if split_name == "val" else None,
                }
            )
    if args.workers == 1:
        records = [_preflight_one(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            records = list(executor.map(_preflight_one, tasks, chunksize=8))
    expected_order = [(name, sample_id) for name in ("train", "val") for sample_id in split[name]]
    observed_order = [(record["split"], record["sample_key"]) for record in records]
    if observed_order != expected_order:
        raise BankContractError("preflight changed frozen split order")

    inventory: dict[str, Any] = {}
    for split_name in ("train", "val"):
        split_records = [record for record in records if record["split"] == split_name]
        eligible_ids = [
            str(record["sample_key"])
            for record in split_records
            if record["status"] == "eligible"
        ]
        excluded_ids = [
            str(record["sample_key"])
            for record in split_records
            if record["status"] == "input_ineligible"
        ]
        inventory[split_name] = {
            "full_count": len(split[split_name]),
            "eligible_count": len(eligible_ids),
            "excluded_count": len(excluded_ids),
            "full_ids_sha256": _ordered_ids_sha256(split[split_name]),
            "eligible_ids_sha256": _ordered_ids_sha256(eligible_ids),
            "excluded_ids_sha256": _ordered_ids_sha256(excluded_ids),
            "eligible_ids": eligible_ids,
            "excluded_ids": excluded_ids,
        }
    if inventory["val"]["eligible_count"] != len(val_by_id):
        raise BankContractError(
            "input-only validation eligibility does not exactly match reusable S50 bank"
        )

    payload = {
        "schema_version": INPUT_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "study_protocol_id": STUDY_PROTOCOL_ID,
        "status": "complete",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "selection_boundary": (
            "canonical-SMILES embedding, processed tensor availability, complete "
            "element/connectivity mapping, graph preprocessing, and reusable-val pose "
            "availability only; sampled pose outcomes are forbidden"
        ),
        "forbidden_eligibility_features": [
            "candidate RMSD",
            "number of poses below any RMSD threshold",
            "confidence score",
            "pose validity",
            "sampler success",
        ],
        "settings": _fixed_settings(),
        "seed_contract": {
            "base_seed": BASE_SEED,
            "formula": "base_seed + one_based_index_within_exact_frozen_split",
            "split_order_preserved": True,
        },
        "inputs": {
            "split": split_asset,
            "pool_parquet": pool_asset,
            "processed_root": str(args.processed_root.resolve()),
            "sampler_checkpoint": {**checkpoint_asset, **expected_checkpoint_fields},
            "sampler_config": config_asset,
            "val_bank_manifest": _asset(args.val_bank_manifest),
            "runtime_code_identity": _runtime_code_identity(),
        },
        "inventory": inventory,
        "records": records,
    }
    _atomic_write_noreplace(args.output.resolve(), _canonical_json_bytes(payload))
    print(
        json.dumps(
            {
                "status": "complete",
                "train_eligible": inventory["train"]["eligible_count"],
                "train_excluded": inventory["train"]["excluded_count"],
                "val_eligible": inventory["val"]["eligible_count"],
                "val_excluded": inventory["val"]["excluded_count"],
                "output": str(args.output.resolve()),
                "sha256": file_sha256(args.output.resolve()),
            },
            sort_keys=True,
        )
    )
    return payload


def _load_input_manifest(
    path: Path,
    expected_sha256: str,
    expected_builder_sha256: str,
) -> dict[str, Any]:
    _validate_builder_pin(expected_builder_sha256)
    _require_asset(path, expected_sha256, "frozen confidence-bank inputs")
    payload = json.loads(path.read_text())
    if (
        payload.get("schema_version") != INPUT_SCHEMA
        or payload.get("protocol_id") != PROTOCOL_ID
        or payload.get("status") != "complete"
    ):
        raise BankContractError("input manifest schema/protocol/status mismatch")
    _validate_fixed_settings(payload.get("settings"))
    if payload.get("inputs", {}).get("runtime_code_identity") != _runtime_code_identity():
        raise BankContractError("runtime code changed after input freeze")
    return payload


def _eligible_records(manifest: dict[str, Any], split: str) -> list[dict[str, Any]]:
    records = [
        record
        for record in manifest.get("records", [])
        if record.get("split") == split and record.get("status") == "eligible"
    ]
    expected_ids = manifest["inventory"][split]["eligible_ids"]
    if [record.get("sample_key") for record in records] != expected_ids:
        raise BankContractError(f"{split}: eligible record order/inventory mismatch")
    return records


def _validate_generation_assets(
    manifest: dict[str, Any],
    *,
    checkpoint: Path,
    expected_checkpoint_sha256: str,
    config: Path,
    expected_config_sha256: str,
) -> None:
    current_checkpoint = _require_asset(
        checkpoint, expected_checkpoint_sha256, "runtime sampler checkpoint"
    )
    current_config = _require_asset(config, expected_config_sha256, "runtime sampler config")
    frozen = manifest["inputs"]
    if current_checkpoint["sha256"] != frozen["sampler_checkpoint"]["sha256"]:
        raise BankContractError("runtime checkpoint differs from frozen input checkpoint")
    if current_config["sha256"] != frozen["sampler_config"]["sha256"]:
        raise BankContractError("runtime config differs from frozen input config")


def _tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(b"\0")
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode())
    digest.update(b"\0")
    digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _topology_signature(molecule: Chem.Mol) -> tuple[Any, Any]:
    atoms = tuple(
        (
            atom.GetAtomicNum(),
            atom.GetFormalCharge(),
            atom.GetIsotope(),
            atom.GetTotalNumHs(includeNeighbors=True),
            atom.GetNumRadicalElectrons(),
            atom.GetIsAromatic(),
        )
        for atom in molecule.GetAtoms()
    )
    edges = tuple(
        sorted(
            (
                min(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()),
                max(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()),
                float(bond.GetBondTypeAsDouble()),
                bond.GetIsAromatic(),
            )
            for bond in molecule.GetBonds()
        )
    )
    return atoms, edges


def _required_sdf_property(molecule: Chem.Mol, name: str, label: str) -> str:
    if not molecule.HasProp(name):
        raise BankContractError(f"{label}: missing SDF property {name!r}")
    return molecule.GetProp(name)


def _read_reused_val_poses(record: dict[str, Any], molecule_input: Chem.Mol) -> torch.Tensor:
    source = record.get("val_pose_bank", {}).get("all_poses_sdf")
    path = _verify_asset(source, f"{record['sample_key']}.reused_val_sdf")
    center = torch.tensor(record["pocket_center"], dtype=torch.float32)
    expected_topology = _topology_signature(molecule_input)
    poses: list[torch.Tensor] = []
    with path.open("rb") as handle:
        supplier = Chem.ForwardSDMolSupplier(
            handle, removeHs=False, sanitize=True, strictParsing=True
        )
        for pose_index, molecule in enumerate(supplier):
            label = f"{record['sample_key']}.pose[{pose_index}]"
            if molecule is None or pose_index >= NUM_SAMPLES:
                raise BankContractError(f"{label}: invalid or excess SDF record")
            if _topology_signature(molecule) != expected_topology:
                raise BankContractError(f"{label}: atom order/connectivity mismatch")
            expected_properties = {
                "_Name": f"docked_pose_{pose_index}",
                "sample_index": str(pose_index),
                "complex_id": str(record["sample_key"]),
                "dataset": "plinder_val",
                "sampling_seed": str(record["sampling_seed"]),
                "ligand_conformer_seed": str(CONFORMER_SEED),
                "num_samples": str(NUM_SAMPLES),
                "num_steps": str(NUM_STEPS),
                "candidate_ensemble_sha256": str(
                    record["val_pose_bank"]["candidate_ensemble_sha256"]
                ),
            }
            for name, expected in expected_properties.items():
                if _required_sdf_property(molecule, name, label) != expected:
                    raise BankContractError(f"{label}: SDF property {name!r} mismatch")
            sample_sigma = float(_required_sdf_property(molecule, "sample_sigma", label))
            if not math.isclose(sample_sigma, SIGMA, rel_tol=0.0, abs_tol=0.0):
                raise BankContractError(f"{label}: SDF sigma mismatch")
            coords = torch.tensor(molecule.GetConformer().GetPositions(), dtype=torch.float32)
            if coords.shape != (molecule_input.GetNumAtoms(), 3):
                raise BankContractError(f"{label}: SDF coordinate shape mismatch")
            poses.append(coords - center)
    if len(poses) != NUM_SAMPLES:
        raise BankContractError(
            f"{record['sample_key']}: reused SDF has {len(poses)} != {NUM_SAMPLES} poses"
        )
    if file_sha256(path) != source["sha256"]:
        raise BankContractError(f"{record['sample_key']}: reused SDF changed while reading")
    return torch.stack(poses)


def _sample_train_poses(
    model: torch.nn.Module,
    graph: dict[str, torch.Tensor],
    ligand_data: dict[str, torch.Tensor],
    inference_meta: dict[str, Any],
    *,
    seed: int,
    model_cfg: dict[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, str]:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    initial_t, initial_q = sample_shared_prior_states(
        PRIOR_POOL_SIZE,
        int(inference_meta["num_frag"]),
        ligand_data["frag_sizes"],
        translation_sigma=SIGMA,
        seed=seed,
    )
    prior_sha = hashlib.sha256(
        b"EFFDOCK_S50_PRIOR_POOL_V1\0"
        + bytes.fromhex(_tensor_sha256(initial_t))
        + bytes.fromhex(_tensor_sha256(initial_q))
    ).hexdigest()
    results = sample_unified(
        model,
        graph,
        ligand_data,
        inference_meta,
        num_samples=NUM_SAMPLES,
        num_steps=NUM_STEPS,
        translation_sigma=SIGMA,
        time_schedule=TIME_SCHEDULE,
        schedule_power=SCHEDULE_POWER,
        device=device,
        stochastic_gamma=0.0,
        translation_sde_base_sigma=0.0,
        pose_objective=model_cfg.get("pose_objective", "linear_fm"),
        score_rot_sigma_max=float(model_cfg.get("score_rot_sigma_max", torch.pi)),
        score_alpha_min=float(model_cfg.get("score_alpha_min", 0.0)),
        initial_T_frag=initial_t[:NUM_SAMPLES],
        initial_q_frag=initial_q[:NUM_SAMPLES],
    )
    poses = torch.stack(
        [result["atom_pos_pred"].detach().cpu().to(torch.float32) for result in results]
    )
    if poses.shape[0] != NUM_SAMPLES or not bool(torch.isfinite(poses).all()):
        raise BankContractError("sampler returned an invalid pose ensemble")
    return poses, prior_sha


def _extract_hidden_chunked(
    model: torch.nn.Module,
    graph: dict[str, torch.Tensor],
    ligand_data: dict[str, torch.Tensor],
    inference_meta: dict[str, Any],
    poses: torch.Tensor,
    *,
    device: torch.device,
    hidden_chunk_size: int,
) -> dict[str, torch.Tensor]:
    if hidden_chunk_size < 1:
        raise ValueError("hidden chunk size must be positive")
    chunks: list[torch.Tensor] = []
    ligand_node_type: torch.Tensor | None = None
    for start in range(0, poses.shape[0], hidden_chunk_size):
        features = extract_t1_ligand_irreps(
            model,
            graph,
            ligand_data,
            inference_meta,
            poses[start : start + hidden_chunk_size],
            sigma=SIGMA,
            device=device,
            hidden_dtype=torch.float16,
        )
        chunks.append(features["h_lig_node"])
        if ligand_node_type is None:
            ligand_node_type = features["lig_node_type"]
        elif not torch.equal(ligand_node_type, features["lig_node_type"]):
            raise BankContractError("ligand node type changed between hidden chunks")
    if ligand_node_type is None:
        raise BankContractError("hidden extraction produced no chunks")
    hidden = torch.cat(chunks, dim=0)
    if hidden.shape[0] != poses.shape[0] or not bool(torch.isfinite(hidden).all()):
        raise BankContractError("hidden extraction produced invalid features")
    return {"h_lig_node": hidden, "lig_node_type": ligand_node_type}


def _center_graph(
    graph: dict[str, torch.Tensor], pocket_center: torch.Tensor
) -> dict[str, torch.Tensor]:
    centered = {key: value.detach().cpu().clone() for key, value in graph.items()}
    centered["node_coords"] = centered["node_coords"].to(torch.float32) - pocket_center
    if not bool(torch.isfinite(centered["node_coords"]).all()):
        raise BankContractError("centered inference graph contains non-finite coordinates")
    return centered


def _fixed_labels(
    poses: torch.Tensor,
    reference_ligand: dict[str, torch.Tensor],
    input_to_reference: Sequence[int],
    pocket_center: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    reference = torch.as_tensor(reference_ligand["atom_coords"], dtype=torch.float32)
    mapping = torch.tensor(input_to_reference, dtype=torch.long)
    aligned = reference.index_select(0, mapping) - pocket_center
    atom_disp = (poses.to(torch.float32) - aligned.unsqueeze(0)).norm(dim=-1)
    pose_rmsd = atom_disp.square().mean(dim=1).sqrt()
    if not bool(torch.isfinite(atom_disp).all()) or not bool(torch.isfinite(pose_rmsd).all()):
        raise BankContractError("fixed-map labels contain non-finite values")
    return aligned, atom_disp, pose_rmsd


def _symmetry_no_align_rmsd(
    poses: torch.Tensor,
    molecule_input: Chem.Mol,
    molecule_reference: Chem.Mol,
    pocket_center: torch.Tensor,
) -> torch.Tensor:
    values: list[float] = []
    for pose in poses:
        molecule_pose = Chem.RWMol(molecule_input)
        conformer = molecule_pose.GetConformer()
        absolute = pose + pocket_center
        for atom_index, point in enumerate(absolute.tolist()):
            conformer.SetAtomPosition(atom_index, point)
        try:
            value = float(rdMolAlign.CalcRMS(molecule_pose, molecule_reference))
        except Exception as exc:
            raise BankContractError(
                "validation symmetry-aware no-align CalcRMS failed"
            ) from exc
        if not math.isfinite(value):
            raise BankContractError("validation symmetry-aware RMSD is non-finite")
        values.append(value)
    return torch.tensor(values, dtype=torch.float32)


def _prepare_runtime_input(
    record: dict[str, Any], processed_root: Path
) -> tuple[
    Chem.Mol,
    Chem.Mol,
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    dict[str, Any],
    dict[str, torch.Tensor],
]:
    sample_id = str(record["sample_key"])
    paths = _processed_paths(processed_root, sample_id)
    for key, path in (
        ("processed_protein", paths.protein),
        ("processed_ligand_reference", paths.ligand),
        ("processed_meta", paths.meta),
    ):
        frozen_path = _verify_asset(record[key], f"{sample_id}.{key}")
        if frozen_path != path.resolve(strict=True):
            raise BankContractError(f"{sample_id}: {key} path differs from processed root")
    protein, reference_ligand, meta = _load_processed(paths)
    smiles = str(record["canonical_smiles"])
    if ligand_input_identity(sample_id, smiles) != record["ligand_input_identity"]:
        raise BankContractError(f"{sample_id}: canonical SMILES identity changed")
    molecule_input, _ = load_benchmark_ligand(smiles, random_seed=CONFORMER_SEED)
    ligand_data = _canonical_ligand_data(molecule_input)
    molecule_reference_raw = reconstruct_processed_reference(
        reference_ligand, molecule_input
    )
    mapping, input_to_reference, molecule_reference = _mapping_record(
        molecule_input, molecule_reference_raw, reference_ligand
    )
    if (
        not _mapping_metadata_matches(mapping, record.get("mapping"))
        or input_to_reference != record["input_to_reference"]
    ):
        raise BankContractError(f"{sample_id}: atom mapping changed after input freeze")
    bundle = build_inference_bundle(protein, ligand_data, meta, pocket_cutoff=POCKET_CUTOFF)
    if bundle is None:
        raise BankContractError(f"{sample_id}: inference graph failed after input freeze")
    graph, ligand_data, inference_meta = bundle
    center = inference_meta["pocket_center"].detach().cpu().to(torch.float32)
    if center.tolist() != record["pocket_center"]:
        raise BankContractError(f"{sample_id}: pocket center changed after input freeze")
    if meta.get("plinder_system_id") != record.get("system_id"):
        raise BankContractError(f"{sample_id}: authoritative system_id changed after input freeze")
    return (
        molecule_input,
        molecule_reference,
        graph,
        ligand_data,
        inference_meta,
        reference_ligand,
    )


def _generate_one(
    record: dict[str, Any],
    *,
    model: torch.nn.Module,
    model_cfg: dict[str, Any],
    device: torch.device,
    processed_root: Path,
    attempt_root: Path,
    final_root: Path,
    pose_tag: str,
    hidden_chunk_size: int,
    checkpoint_identity: dict[str, Any],
    config_identity: dict[str, Any],
) -> dict[str, Any]:
    sample_id = str(record["sample_key"])
    (
        molecule_input,
        molecule_reference,
        graph,
        ligand_data,
        inference_meta,
        reference_ligand,
    ) = _prepare_runtime_input(record, processed_root)
    seed = int(record["sampling_seed"])
    if record["split"] == "val":
        poses = _read_reused_val_poses(record, molecule_input)
        prior_sha = str(record["val_pose_bank"].get("prior_pool_sha256"))
        pose_source = "reused_exact_s50_sdf"
    else:
        poses, prior_sha = _sample_train_poses(
            model,
            graph,
            ligand_data,
            inference_meta,
            seed=seed,
            model_cfg=model_cfg,
            device=device,
        )
        pose_source = "generated_s50_deterministic_ode"
    hidden = _extract_hidden_chunked(
        model,
        graph,
        ligand_data,
        inference_meta,
        poses,
        device=device,
        hidden_chunk_size=hidden_chunk_size,
    )
    pocket_center = inference_meta["pocket_center"].detach().cpu().to(torch.float32)
    reference_aligned, atom_disp, pose_rmsd = _fixed_labels(
        poses,
        reference_ligand,
        record["input_to_reference"],
        pocket_center,
    )
    graph_centered = _center_graph(graph, pocket_center)
    shard: dict[str, Any] = {
        "storage_version": POSE_STORAGE_VERSION,
        "protocol_id": PROTOCOL_ID,
        "study_protocol_id": STUDY_PROTOCOL_ID,
        "pid": sample_id,
        "system_id": record["system_id"],
        "split": record["split"],
        "split_index": int(record["split_index"]),
        "seed": seed,
        "sampling_seed": seed,
        "ligand_conformer_seed": CONFORMER_SEED,
        "pose_tag": pose_tag,
        "pose_source": pose_source,
        "checkpoint": checkpoint_identity["path"],
        "checkpoint_sha256": checkpoint_identity["sha256"],
        "config": config_identity["path"],
        "config_sha256": config_identity["sha256"],
        "sigma": SIGMA,
        "num_steps": NUM_STEPS,
        "num_samples": NUM_SAMPLES,
        "time_schedule": TIME_SCHEDULE,
        "schedule_power": SCHEDULE_POWER,
        "pocket_cutoff": POCKET_CUTOFF,
        "prior_pool_size": PRIOR_POOL_SIZE,
        "prior_pool_sha256": prior_sha,
        "sampling_dynamics": "deterministic_ode",
        "hidden_scope": "ligand",
        "hidden_dtype": "float16",
        "hidden_chunk_size": hidden_chunk_size,
        "pocket_center_used": pocket_center,
        "pose_sigma": torch.full((NUM_SAMPLES,), SIGMA, dtype=torch.float32),
        "pose_num_steps": torch.full((NUM_SAMPLES,), NUM_STEPS, dtype=torch.long),
        "lig_num_atoms": int(inference_meta["num_atom"]),
        "lig_num_frags": int(inference_meta["num_frag"]),
        "lig_atom_coords_crystal_centered": reference_aligned,
        "frag_sizes": ligand_data["frag_sizes"].detach().cpu(),
        "fragment_id": ligand_data["fragment_id"].detach().cpu(),
        "pose_atom_coords": poses,
        "h_lig_node": hidden["h_lig_node"],
        "lig_node_type": hidden["lig_node_type"],
        "atom_disp": atom_disp,
        "pose_rmsd": pose_rmsd,
        "input_to_reference": torch.tensor(record["input_to_reference"], dtype=torch.long),
        "mapping_metadata": record["mapping"],
        "graph_coordinate_frame": "pocket_centered",
        "graph_centered": graph_centered,
        # Alias retained for consumers that use the shorter canonical key.
        "graph": graph_centered,
    }
    if record["split"] == "val":
        shard["pose_rmsd_symmetry_no_align"] = _symmetry_no_align_rmsd(
            poses, molecule_input, molecule_reference, pocket_center
        )
        shard["symmetry_rmsd_method"] = "rdkit_calc_rms_symmetry_no_align"
        shard["source_all_poses_sdf"] = record["val_pose_bank"]["all_poses_sdf"]

    relative = Path(sample_id) / "confidence_poses" / f"confposes_{pose_tag}.pt"
    attempt_path = attempt_root / relative
    final_path = final_root / relative
    _atomic_torch_save_noreplace(attempt_path, shard)
    identity = _asset(attempt_path)
    return {
        "sample_key": sample_id,
        "system_id": record["system_id"],
        "split": record["split"],
        "status": "complete",
        "split_index": int(record["split_index"]),
        "global_index": int(record["global_index"]),
        "sampling_seed": seed,
        "pt_path": str(final_path.resolve()),
        "pt_sha256": identity["sha256"],
        "size_bytes": identity["size_bytes"],
        "pose_count": NUM_SAMPLES,
        "pose_source": pose_source,
        "pose_ensemble_sha256": _tensor_sha256(poses),
    }


def generate_shard(args: argparse.Namespace) -> dict[str, Any]:
    manifest = _load_input_manifest(
        args.input_manifest,
        args.expected_input_manifest_sha256,
        args.expected_builder_sha256,
    )
    _validate_generation_assets(
        manifest,
        checkpoint=args.checkpoint,
        expected_checkpoint_sha256=args.expected_checkpoint_sha256,
        config=args.config,
        expected_config_sha256=args.expected_config_sha256,
    )
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard index must satisfy 0 <= index < num_shards")
    if args.max_records is not None and args.max_records < 1:
        raise ValueError("--max-records must be positive")
    if args.hidden_chunk_size < 1:
        raise ValueError("--hidden-chunk-size must be positive")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("generate-shard requires one allocated CUDA GPU")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("generate-shard requires exactly one visible CUDA GPU")

    eligible = _eligible_records(manifest, args.split)
    full_assigned = eligible[args.shard_index :: args.num_shards]
    assigned = (
        full_assigned
        if args.max_records is None
        else full_assigned[: args.max_records]
    )
    if not assigned:
        raise ValueError("selected shard contains no eligible records")
    shard_name = f"shard-{args.shard_index:03d}-of-{args.num_shards:03d}"
    final_root = (args.output_root / "shards" / args.split / shard_name).resolve()
    if final_root.exists():
        raise FileExistsError(f"refusing to overwrite shard: {final_root}")
    incomplete_parent = (args.output_root / ".incomplete" / args.split).resolve()
    incomplete_parent.mkdir(parents=True, exist_ok=True)
    attempt_root = Path(tempfile.mkdtemp(prefix=f"{shard_name}.", dir=incomplete_parent))

    checkpoint_identity = _asset(args.checkpoint)
    config_identity = _asset(args.config)
    records: list[dict[str, Any]] = []
    try:
        model, cfg, checkpoint = load_model(args.config, args.checkpoint, device)
        if checkpoint.get("step") != 50000:
            raise BankContractError("loaded sampler checkpoint is not S50")
        model_cfg = cfg.get("data", {})
        for index, record in enumerate(assigned, start=1):
            result = _generate_one(
                record,
                model=model,
                model_cfg=model_cfg,
                device=device,
                processed_root=Path(manifest["inputs"]["processed_root"]),
                attempt_root=attempt_root,
                final_root=final_root,
                pose_tag=args.pose_tag,
                hidden_chunk_size=args.hidden_chunk_size,
                checkpoint_identity=checkpoint_identity,
                config_identity=config_identity,
            )
            records.append(result)
            print(f"[{index}/{len(assigned)}] {record['sample_key']} complete")
        summary = {
            "schema_version": SHARD_SCHEMA,
            "protocol_id": PROTOCOL_ID,
            "study_protocol_id": STUDY_PROTOCOL_ID,
            "status": "complete",
            "selection_mode": "full" if args.max_records is None else "smoke_subset",
            "claim_eligible": args.max_records is None,
            "split": args.split,
            "pose_tag": args.pose_tag,
            "settings": _fixed_settings(),
            "input_manifest": {
                "path": str(args.input_manifest.resolve()),
                "sha256": args.expected_input_manifest_sha256,
            },
            "num_shards": args.num_shards,
            "shard_index": args.shard_index,
            "full_assigned_count": len(full_assigned),
            "full_assigned_ids_sha256": _ordered_ids_sha256(
                [str(record["sample_key"]) for record in full_assigned]
            ),
            "record_count": len(records),
            "record_ids_sha256": _ordered_ids_sha256(
                [str(record["sample_key"]) for record in assigned]
            ),
            "max_records": args.max_records,
            "records": records,
        }
        _atomic_write_noreplace(
            attempt_root / "shard_summary.json", _canonical_json_bytes(summary)
        )
        final_root.parent.mkdir(parents=True, exist_ok=True)
        attempt_root.rename(final_root)
    except Exception:
        # Leave the attempt under .incomplete for diagnosis; nothing is published.
        raise
    print(
        json.dumps(
            {
                "status": "complete",
                "split": args.split,
                "records": len(records),
                "output": str(final_root),
                "summary_sha256": file_sha256(final_root / "shard_summary.json"),
            },
            sort_keys=True,
        )
    )
    return summary


def _expected_pose_source(split: str) -> str:
    if split == "train":
        return "generated_s50_deterministic_ode"
    if split == "val":
        return "reused_exact_s50_sdf"
    raise BankContractError(f"unsupported frozen split: {split!r}")


def _validate_summary_record_join(
    record: dict[str, Any],
    frozen_record: dict[str, Any],
    *,
    path: Path,
) -> None:
    """Join one generated record exactly to its frozen input authority."""
    if frozen_record.get("status") != "eligible":
        raise BankContractError(f"{path}: generated record joined a non-eligible input")
    expected = {
        key: frozen_record.get(key)
        for key in (
            "sample_key",
            "system_id",
            "split",
            "split_index",
            "global_index",
            "sampling_seed",
        )
    }
    expected.update(
        {
            "status": "complete",
            "pose_count": NUM_SAMPLES,
            "pose_source": _expected_pose_source(str(frozen_record.get("split"))),
        }
    )
    mismatches = {
        key: {"expected": value, "observed": record.get(key)}
        for key, value in expected.items()
        if record.get(key) != value
    }
    if mismatches:
        raise BankContractError(f"{path}: shard/frozen record join mismatch: {mismatches}")


def _load_expected_aligned_reference(
    frozen_record: dict[str, Any],
    *,
    path: Path,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Rebuild the fixed-map centered crystal target from its sealed tensor asset."""
    asset_identity = frozen_record.get("processed_ligand_reference")
    ligand_path = _verify_asset(
        asset_identity,
        f"{frozen_record.get('sample_key')}.processed_ligand_reference",
    )
    identity_before = _asset(ligand_path)
    ligand = torch.load(ligand_path, map_location="cpu", weights_only=True)
    _require_unchanged_asset(
        identity_before,
        ligand_path,
        f"{frozen_record.get('sample_key')}.processed_ligand_reference",
    )
    if not isinstance(ligand, dict):
        raise BankContractError(f"{path}: processed ligand reference is not a mapping")
    coordinates = torch.as_tensor(ligand.get("atom_coords"), dtype=torch.float32)
    mapping = torch.as_tensor(
        frozen_record.get("input_to_reference"), dtype=torch.long
    )
    center = torch.as_tensor(frozen_record.get("pocket_center"), dtype=torch.float32)
    n_atoms = int(frozen_record.get("num_input_atoms", -1))
    if (
        coordinates.shape != (n_atoms, 3)
        or mapping.shape != (n_atoms,)
        or center.shape != (3,)
        or sorted(mapping.tolist()) != list(range(n_atoms))
        or not bool(torch.isfinite(coordinates).all() and torch.isfinite(center).all())
    ):
        raise BankContractError(f"{path}: frozen crystal-reference contract is invalid")
    return coordinates.index_select(0, mapping) - center, ligand


def _recompute_validation_symmetry_rmsd(
    payload: dict[str, Any],
    frozen_record: dict[str, Any],
    reference_ligand: dict[str, torch.Tensor],
    *,
    path: Path,
) -> torch.Tensor:
    """Independently recover the frozen no-align symmetry target from sources."""
    smiles = frozen_record.get("canonical_smiles")
    sample_id = str(frozen_record.get("sample_key"))
    if not isinstance(smiles, str) or not smiles:
        raise BankContractError(f"{path}: frozen canonical SMILES is missing")
    if ligand_input_identity(sample_id, smiles) != frozen_record.get(
        "ligand_input_identity"
    ):
        raise BankContractError(f"{path}: frozen canonical ligand identity mismatch")
    try:
        molecule_input, _ = load_benchmark_ligand(
            smiles, random_seed=CONFORMER_SEED
        )
        molecule_reference_raw = reconstruct_processed_reference(
            reference_ligand, molecule_input
        )
        mapping_metadata, input_to_reference, molecule_reference = _mapping_record(
            molecule_input, molecule_reference_raw, reference_ligand
        )
    except Exception as exc:
        raise BankContractError(
            f"{path}: validation symmetry source reconstruction failed"
        ) from exc
    if (
        not _mapping_metadata_matches(mapping_metadata, frozen_record.get("mapping"))
        or input_to_reference != frozen_record.get("input_to_reference")
        or mapping_metadata.get("mapping_method") != "strict_stereo"
        or mapping_metadata.get("symmetry_complete") is not True
        or mapping_metadata.get("mapping_truncated") is not False
    ):
        raise BankContractError(
            f"{path}: validation symmetry mapping used a fallback or changed"
        )
    center = torch.as_tensor(frozen_record.get("pocket_center"), dtype=torch.float32)
    try:
        recomputed = _symmetry_no_align_rmsd(
            payload["pose_atom_coords"].to(torch.float32),
            molecule_input,
            molecule_reference,
            center,
        )
    except Exception as exc:
        raise BankContractError(
            f"{path}: validation symmetry no-align recomputation failed"
        ) from exc
    if recomputed.shape != (NUM_SAMPLES,) or not bool(torch.isfinite(recomputed).all()):
        raise BankContractError(
            f"{path}: validation symmetry no-align recomputation is invalid"
        )
    return recomputed


def _load_shard_summary(
    output_root: Path,
    *,
    split: str,
    shard_index: int,
    num_shards: int,
    input_manifest_sha256: str,
    pose_tag: str,
    frozen_records: Sequence[dict[str, Any]],
    frozen_inputs: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = (
        output_root
        / "shards"
        / split
        / f"shard-{shard_index:03d}-of-{num_shards:03d}"
        / "shard_summary.json"
    ).resolve(strict=True)
    identity = _asset(path)
    summary = json.loads(path.read_text())
    expected = {
        "schema_version": SHARD_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "status": "complete",
        "split": split,
        "pose_tag": pose_tag,
        "num_shards": num_shards,
        "shard_index": shard_index,
    }
    if any(summary.get(key) != value for key, value in expected.items()):
        raise BankContractError(f"invalid shard summary identity: {path}")
    if summary.get("input_manifest", {}).get("sha256") != input_manifest_sha256:
        raise BankContractError(f"{path}: input manifest hash mismatch")
    _validate_fixed_settings(summary.get("settings"))
    records = summary.get("records")
    if not isinstance(records, list) or len(records) != int(summary.get("record_count", -1)):
        raise BankContractError(f"{path}: record count mismatch")
    if summary.get("record_ids_sha256") != _ordered_ids_sha256(
        [str(record.get("sample_key")) for record in records]
    ):
        raise BankContractError(f"{path}: record ID ledger mismatch")
    full_assigned = list(frozen_records[shard_index::num_shards])
    full_assigned_ids = [str(record["sample_key"]) for record in full_assigned]
    if (
        int(summary.get("full_assigned_count", -1)) != len(full_assigned)
        or summary.get("full_assigned_ids_sha256")
        != _ordered_ids_sha256(full_assigned_ids)
    ):
        raise BankContractError(f"{path}: frozen shard assignment mismatch")
    selection_mode = summary.get("selection_mode")
    if selection_mode == "full":
        if summary.get("max_records") is not None:
            raise BankContractError(f"{path}: full shard declares a record limit")
        expected_records = full_assigned
    elif selection_mode == "smoke_subset":
        max_records = summary.get("max_records")
        if not isinstance(max_records, int) or max_records < 1:
            raise BankContractError(f"{path}: smoke shard lacks a positive record limit")
        expected_records = full_assigned[:max_records]
    else:
        raise BankContractError(f"{path}: invalid shard selection mode")
    expected_ids = [str(record["sample_key"]) for record in expected_records]
    observed_ids = [str(record.get("sample_key")) for record in records]
    if observed_ids != expected_ids:
        raise BankContractError(f"{path}: shard record order/assignment mismatch")
    frozen_by_id = {str(record["sample_key"]): record for record in expected_records}
    for record in records:
        frozen_record = frozen_by_id[str(record["sample_key"])]
        _validate_summary_record_join(record, frozen_record, path=path)
        pt_path = Path(str(record.get("pt_path", ""))).resolve(strict=True)
        shard_root = path.parent
        try:
            pt_path.relative_to(shard_root)
        except ValueError as exc:
            raise BankContractError(f"{pt_path}: pose artifact escaped shard root") from exc
        identity_pt = _asset(pt_path)
        if (
            identity_pt["sha256"] != record.get("pt_sha256")
            or identity_pt["size_bytes"] != record.get("size_bytes")
            or int(record.get("pose_count", -1)) != NUM_SAMPLES
        ):
            raise BankContractError(f"{pt_path}: pose artifact identity mismatch")
        payload = torch.load(pt_path, map_location="cpu", weights_only=True)
        _validate_pose_payload(
            payload,
            record=record,
            frozen_record=frozen_record,
            frozen_inputs=frozen_inputs,
            pose_tag=pose_tag,
            path=pt_path,
        )
        _require_unchanged_asset(identity_pt, pt_path, f"{pt_path}.pose_artifact")
    identity = _require_unchanged_asset(identity, path, f"{path}.shard_summary")
    return summary, identity


def _validate_pose_payload(
    payload: dict[str, Any],
    *,
    record: dict[str, Any],
    frozen_record: dict[str, Any],
    frozen_inputs: dict[str, Any],
    pose_tag: str,
    path: Path,
) -> None:
    """Fail closed on every tensor/identity required by confidence training."""
    if not isinstance(payload, dict):
        raise BankContractError(f"{path}: pose payload is not a mapping")
    _validate_summary_record_join(record, frozen_record, path=path)
    checkpoint = frozen_inputs.get("sampler_checkpoint")
    config = frozen_inputs.get("sampler_config")
    if not isinstance(checkpoint, dict) or not isinstance(config, dict):
        raise BankContractError(f"{path}: frozen sampler identities are missing")
    frozen_split = str(frozen_record.get("split"))
    identity = {
        "storage_version": POSE_STORAGE_VERSION,
        "protocol_id": PROTOCOL_ID,
        "study_protocol_id": STUDY_PROTOCOL_ID,
        "pid": frozen_record.get("sample_key"),
        "system_id": frozen_record.get("system_id"),
        "split": frozen_split,
        "split_index": frozen_record.get("split_index"),
        "seed": frozen_record.get("sampling_seed"),
        "sampling_seed": frozen_record.get("sampling_seed"),
        "pose_tag": pose_tag,
        "pose_source": _expected_pose_source(frozen_split),
        "checkpoint": checkpoint.get("path"),
        "checkpoint_sha256": checkpoint.get("sha256"),
        "config": config.get("path"),
        "config_sha256": config.get("sha256"),
        "num_samples": NUM_SAMPLES,
        "num_steps": NUM_STEPS,
        "sigma": SIGMA,
        "time_schedule": TIME_SCHEDULE,
        "schedule_power": SCHEDULE_POWER,
        "pocket_cutoff": POCKET_CUTOFF,
        "prior_pool_size": PRIOR_POOL_SIZE,
        "sampling_dynamics": "deterministic_ode",
        "ligand_conformer_seed": CONFORMER_SEED,
        "hidden_scope": "ligand",
        "hidden_dtype": "float16",
        "hidden_chunk_size": HIDDEN_CHUNK_SIZE,
        "graph_coordinate_frame": "pocket_centered",
    }
    if any(payload.get(key) != value for key, value in identity.items()):
        raise BankContractError(f"{path}: pose payload identity/settings mismatch")
    n_atoms = int(payload.get("lig_num_atoms", -1))
    n_fragments = int(payload.get("lig_num_frags", -1))
    if (
        n_atoms < 1
        or n_fragments < 1
        or n_atoms != int(frozen_record.get("num_input_atoms", -1))
        or n_fragments != int(frozen_record.get("num_fragments", -1))
    ):
        raise BankContractError(f"{path}: invalid ligand size metadata")
    expected_shapes = {
        "pocket_center_used": (3,),
        "pose_sigma": (NUM_SAMPLES,),
        "pose_num_steps": (NUM_SAMPLES,),
        "lig_atom_coords_crystal_centered": (n_atoms, 3),
        "frag_sizes": (n_fragments,),
        "fragment_id": (n_atoms,),
        "pose_atom_coords": (NUM_SAMPLES, n_atoms, 3),
        "atom_disp": (NUM_SAMPLES, n_atoms),
        "pose_rmsd": (NUM_SAMPLES,),
        "input_to_reference": (n_atoms,),
    }
    for key, expected_shape in expected_shapes.items():
        value = payload.get(key)
        if not torch.is_tensor(value) or tuple(value.shape) != expected_shape:
            raise BankContractError(
                f"{path}: {key} shape mismatch; expected {expected_shape}, "
                f"got {getattr(value, 'shape', None)}"
            )
    hidden = payload.get("h_lig_node")
    ligand_node_type = payload.get("lig_node_type")
    if (
        not torch.is_tensor(hidden)
        or hidden.ndim != 3
        or hidden.shape[0] != NUM_SAMPLES
        or hidden.shape[1] != n_atoms + n_fragments
        or not torch.is_tensor(ligand_node_type)
        or tuple(ligand_node_type.shape) != (n_atoms + n_fragments,)
    ):
        raise BankContractError(f"{path}: ligand hidden tensor shape mismatch")
    graph = payload.get("graph_centered")
    if not isinstance(graph, dict) or not graph or not all(
        torch.is_tensor(value) for value in graph.values()
    ):
        raise BankContractError(f"{path}: centered graph is missing or invalid")
    if payload.get("graph") is None or set(payload["graph"]) != set(graph):
        raise BankContractError(f"{path}: centered graph alias mismatch")
    if any(
        (torch.is_floating_point(value) or torch.is_complex(value))
        and not bool(torch.isfinite(value).all())
        for value in graph.values()
    ):
        raise BankContractError(f"{path}: pose payload contains non-finite tensors")
    for key, value in graph.items():
        alias = payload["graph"][key]
        if not torch.equal(value, alias):
            raise BankContractError(f"{path}: graph alias tensor {key!r} differs")
    node_coords = graph.get("node_coords")
    if not torch.is_tensor(node_coords) or node_coords.ndim != 2 or node_coords.shape[1] != 3:
        raise BankContractError(f"{path}: centered graph node coordinates are invalid")
    tensor_values = [
        value
        for value in (
            *[payload[key] for key in expected_shapes],
            hidden,
            ligand_node_type,
            *graph.values(),
        )
        if torch.is_floating_point(value) or torch.is_complex(value)
    ]
    if any(not bool(torch.isfinite(value).all()) for value in tensor_values):
        raise BankContractError(f"{path}: pose payload contains non-finite tensors")
    frozen_mapping = torch.as_tensor(
        frozen_record.get("input_to_reference"), dtype=torch.long
    )
    if (
        frozen_mapping.shape != (n_atoms,)
        or not torch.equal(payload["input_to_reference"].to(torch.long), frozen_mapping)
        or not _mapping_metadata_matches(
            payload.get("mapping_metadata"), frozen_record.get("mapping")
        )
    ):
        raise BankContractError(f"{path}: frozen atom-map identity mismatch")
    frozen_center = torch.as_tensor(
        frozen_record.get("pocket_center"), dtype=torch.float32
    )
    if frozen_center.shape != (3,) or not torch.equal(
        payload["pocket_center_used"].to(torch.float32), frozen_center
    ):
        raise BankContractError(f"{path}: frozen pocket center mismatch")
    expected_reference, reference_ligand = _load_expected_aligned_reference(
        frozen_record, path=path
    )
    if not torch.equal(
        payload["lig_atom_coords_crystal_centered"].to(torch.float32),
        expected_reference,
    ):
        raise BankContractError(f"{path}: aligned crystal target differs from frozen input")
    if _tensor_sha256(payload["pose_atom_coords"]) != record.get("pose_ensemble_sha256"):
        raise BankContractError(f"{path}: ordered pose ensemble hash mismatch")
    prior_sha256 = payload.get("prior_pool_sha256")
    if not (
        isinstance(prior_sha256, str)
        and len(prior_sha256) == 64
        and all(character in "0123456789abcdef" for character in prior_sha256)
    ):
        raise BankContractError(f"{path}: prior-pool identity is invalid")
    if frozen_split == "val":
        val_source = frozen_record.get("val_pose_bank")
        if (
            not isinstance(val_source, dict)
            or prior_sha256 != val_source.get("prior_pool_sha256")
            or payload.get("source_all_poses_sdf") != val_source.get("all_poses_sdf")
        ):
            raise BankContractError(f"{path}: reused validation source identity mismatch")
    if (
        not torch.equal(payload["pose_sigma"], torch.full((NUM_SAMPLES,), SIGMA))
        or not torch.equal(
            payload["pose_num_steps"],
            torch.full((NUM_SAMPLES,), NUM_STEPS, dtype=torch.long),
        )
        or int(payload["frag_sizes"].sum()) != n_atoms
        or int(payload["fragment_id"].min()) < 0
        or int(payload["fragment_id"].max()) >= n_fragments
        or sorted(payload["input_to_reference"].tolist()) != list(range(n_atoms))
        or bool((payload["atom_disp"] < 0).any())
    ):
        raise BankContractError(f"{path}: pose tensor value contract mismatch")
    expected_atom_disp = torch.linalg.vector_norm(
        payload["pose_atom_coords"].to(torch.float32)
        - expected_reference.unsqueeze(0),
        dim=-1,
    )
    if not torch.allclose(
        expected_atom_disp,
        payload["atom_disp"].to(torch.float32),
        rtol=1e-6,
        atol=1e-6,
    ):
        raise BankContractError(f"{path}: atom labels disagree with poses/crystal target")
    recomputed_rmsd = expected_atom_disp.square().mean(dim=1).sqrt()
    if not torch.allclose(
        recomputed_rmsd, payload["pose_rmsd"].to(torch.float32), rtol=1e-6, atol=1e-6
    ):
        raise BankContractError(f"{path}: direct RMSD labels disagree with atom labels")
    if payload["split"] == "val":
        symmetry = payload.get("pose_rmsd_symmetry_no_align")
        if (
            not torch.is_tensor(symmetry)
            or tuple(symmetry.shape) != (NUM_SAMPLES,)
            or not bool(torch.isfinite(symmetry).all())
            or payload.get("symmetry_rmsd_method")
            != "rdkit_calc_rms_symmetry_no_align"
        ):
            raise BankContractError(f"{path}: validation symmetry RMSD contract mismatch")
        recomputed_symmetry = _recompute_validation_symmetry_rmsd(
            payload,
            frozen_record,
            reference_ligand,
            path=path,
        )
        if not torch.allclose(
            recomputed_symmetry,
            symmetry.to(torch.float32),
            rtol=1e-6,
            atol=1e-5,
        ):
            max_error = float(
                (recomputed_symmetry - symmetry.to(torch.float32)).abs().max().item()
            )
            raise BankContractError(
                f"{path}: stored validation symmetry RMSD differs from independent "
                f"no-align CalcRMS (max_abs_error={max_error:.3g})"
            )


def aggregate(args: argparse.Namespace) -> dict[str, Any]:
    manifest = _load_input_manifest(
        args.input_manifest,
        args.expected_input_manifest_sha256,
        args.expected_builder_sha256,
    )
    output_manifest_path = args.output_manifest.resolve()
    filtered_path = args.filtered_split_output.resolve()
    if output_manifest_path == filtered_path:
        raise ValueError("aggregate manifest and filtered split paths must differ")
    if output_manifest_path.exists() and not filtered_path.exists():
        raise BankContractError(
            "aggregate commit marker exists without its filtered split dependency"
        )
    shard_counts = {"train": args.num_train_shards, "val": args.num_val_shards}
    if any(value < 1 for value in shard_counts.values()):
        raise ValueError("aggregate shard counts must be positive")

    all_records: list[dict[str, Any]] = []
    shard_assets: list[dict[str, Any]] = []
    filtered: dict[str, list[str]] = {}
    inventory: dict[str, Any] = {}
    for split in ("train", "val"):
        expected_records = _eligible_records(manifest, split)
        expected_ids = [str(record["sample_key"]) for record in expected_records]
        split_records: list[dict[str, Any]] = []
        for shard_index in range(shard_counts[split]):
            summary, identity = _load_shard_summary(
                args.output_root,
                split=split,
                shard_index=shard_index,
                num_shards=shard_counts[split],
                input_manifest_sha256=args.expected_input_manifest_sha256,
                pose_tag=args.pose_tag,
                frozen_records=expected_records,
                frozen_inputs=manifest["inputs"],
            )
            if not args.allow_smoke_subset and (
                summary.get("selection_mode") != "full"
                or summary.get("claim_eligible") is not True
                or summary.get("max_records") is not None
            ):
                raise BankContractError("full aggregate refuses a truncated/smoke shard")
            if args.allow_smoke_subset and summary.get("selection_mode") != "smoke_subset":
                raise BankContractError("smoke aggregate requires explicitly truncated shards")
            split_records.extend(summary["records"])
            shard_assets.append(
                {
                    "split": split,
                    "shard_index": shard_index,
                    "summary": identity,
                    "record_count": summary["record_count"],
                }
            )
        observed_ids = [str(record["sample_key"]) for record in split_records]
        if len(observed_ids) != len(set(observed_ids)):
            raise BankContractError(f"{split}: duplicate IDs across shards")
        if args.allow_smoke_subset:
            if not observed_ids or not set(observed_ids).issubset(expected_ids):
                raise BankContractError(f"{split}: smoke IDs are not an eligible subset")
            observed_set = set(observed_ids)
            observed_ids = [sample_id for sample_id in expected_ids if sample_id in observed_set]
            by_id = {str(record["sample_key"]): record for record in split_records}
            split_records = [by_id[sample_id] for sample_id in observed_ids]
        else:
            by_id = {str(record["sample_key"]): record for record in split_records}
            if set(by_id) != set(expected_ids):
                missing = sorted(set(expected_ids) - set(by_id))
                extra = sorted(set(by_id) - set(expected_ids))
                raise BankContractError(
                    f"{split}: full shard coverage mismatch missing={missing[:5]} extra={extra[:5]}"
                )
            split_records = [by_id[sample_id] for sample_id in expected_ids]
            observed_ids = expected_ids
        filtered[split] = observed_ids
        all_records.extend(split_records)
        source_inventory = manifest["inventory"][split]
        inventory[split] = {
            "full_count": int(source_inventory["full_count"]),
            "eligible_count": int(source_inventory["eligible_count"]),
            "excluded_count": int(source_inventory["excluded_count"]),
            "record_count": len(split_records),
            "full_ids_sha256": source_inventory["full_ids_sha256"],
            "eligible_ids_sha256": source_inventory["eligible_ids_sha256"],
            "excluded_ids_sha256": source_inventory["excluded_ids_sha256"],
            "record_ids_sha256": _ordered_ids_sha256(observed_ids),
        }

    filtered_bytes = _canonical_json_bytes(filtered)
    recovered_filtered = filtered_path.exists()
    if recovered_filtered:
        if not filtered_path.is_file() or filtered_path.read_bytes() != filtered_bytes:
            raise BankContractError(
                "existing uncommitted filtered split differs from verified aggregate"
            )
    else:
        _atomic_write_noreplace(filtered_path, filtered_bytes)
    filtered_identity = _asset(filtered_path)
    status = "smoke_complete" if args.allow_smoke_subset else "complete"
    existing_manifest: dict[str, Any] | None = None
    if output_manifest_path.exists():
        parsed = json.loads(output_manifest_path.read_text())
        if not isinstance(parsed, dict) or not isinstance(parsed.get("created_at_utc"), str):
            raise BankContractError("existing aggregate commit marker is invalid")
        existing_manifest = parsed
    created_at_utc = (
        existing_manifest["created_at_utc"]
        if existing_manifest is not None
        else datetime.now(UTC).isoformat()
    )
    payload = {
        "schema_version": MANIFEST_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "study_protocol_id": STUDY_PROTOCOL_ID,
        "status": status,
        "claim_eligible": not args.allow_smoke_subset,
        "created_at_utc": created_at_utc,
        "pose_tag": args.pose_tag,
        "settings": _fixed_settings(),
        "inputs": manifest["inputs"],
        "input_manifest": {
            "path": str(args.input_manifest.resolve()),
            "sha256": args.expected_input_manifest_sha256,
        },
        "inventory": inventory,
        "filtered_split_path": str(filtered_path),
        "filtered_split_sha256": filtered_identity["sha256"],
        "shard_summaries": shard_assets,
        "records": all_records,
    }
    manifest_bytes = _canonical_json_bytes(payload)
    recovered_manifest = output_manifest_path.exists()
    if recovered_manifest:
        if output_manifest_path.read_bytes() != manifest_bytes:
            raise BankContractError(
                "existing aggregate commit marker differs from verified aggregate"
            )
    else:
        # The manifest is the commit marker.  A crash after publishing the
        # filtered split but before this link is safely recoverable on rerun.
        _atomic_write_noreplace(output_manifest_path, manifest_bytes)
    _require_unchanged_asset(
        filtered_identity, filtered_path, f"{filtered_path}.filtered_split"
    )
    manifest_identity = _asset(output_manifest_path)
    print(
        json.dumps(
            {
                "status": status,
                "train_records": inventory["train"]["record_count"],
                "val_records": inventory["val"]["record_count"],
                "manifest": str(output_manifest_path),
                "manifest_sha256": manifest_identity["sha256"],
                "filtered_split": str(filtered_path),
                "filtered_split_sha256": filtered_identity["sha256"],
                "recovered_filtered_split": recovered_filtered,
                "recovered_manifest": recovered_manifest,
            },
            sort_keys=True,
        )
    )
    return payload


def _add_common_builder_pin(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--expected-builder-sha256", required=True)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)

    freeze = subparsers.add_parser("freeze-inputs")
    freeze.add_argument("--split-file", type=Path, default=Path("data/splits/plinder.json"))
    freeze.add_argument("--expected-split-sha256", required=True)
    freeze.add_argument("--pool-parquet", type=Path, default=Path("data/plinder_pool.parquet"))
    freeze.add_argument("--expected-pool-sha256", required=True)
    freeze.add_argument("--processed-root", type=Path, default=Path("data/plinder_processed"))
    freeze.add_argument("--checkpoint", type=Path, required=True)
    freeze.add_argument("--expected-checkpoint-sha256", required=True)
    freeze.add_argument("--config", type=Path, default=Path("configs/train.yaml"))
    freeze.add_argument("--expected-config-sha256", required=True)
    freeze.add_argument("--val-bank-manifest", type=Path, required=True)
    freeze.add_argument("--expected-val-bank-manifest-sha256", required=True)
    freeze.add_argument("--workers", type=int, default=1)
    freeze.add_argument("--output", type=Path, required=True)
    _add_common_builder_pin(freeze)

    generate = subparsers.add_parser("generate-shard")
    generate.add_argument("--input-manifest", type=Path, required=True)
    generate.add_argument("--expected-input-manifest-sha256", required=True)
    generate.add_argument("--checkpoint", type=Path, required=True)
    generate.add_argument("--expected-checkpoint-sha256", required=True)
    generate.add_argument("--config", type=Path, default=Path("configs/train.yaml"))
    generate.add_argument("--expected-config-sha256", required=True)
    generate.add_argument("--output-root", type=Path, required=True)
    generate.add_argument("--split", choices=("train", "val"), required=True)
    generate.add_argument("--num-shards", type=int, required=True)
    generate.add_argument("--shard-index", type=int, required=True)
    generate.add_argument("--pose-tag", default=DEFAULT_POSE_TAG)
    generate.add_argument("--hidden-chunk-size", type=int, default=HIDDEN_CHUNK_SIZE)
    generate.add_argument("--max-records", type=int, default=None)
    generate.add_argument("--device", default="cuda")
    _add_common_builder_pin(generate)

    combine = subparsers.add_parser("aggregate")
    combine.add_argument("--input-manifest", type=Path, required=True)
    combine.add_argument("--expected-input-manifest-sha256", required=True)
    combine.add_argument("--output-root", type=Path, required=True)
    combine.add_argument("--num-train-shards", type=int, required=True)
    combine.add_argument("--num-val-shards", type=int, required=True)
    combine.add_argument("--pose-tag", default=DEFAULT_POSE_TAG)
    combine.add_argument("--filtered-split-output", type=Path, required=True)
    combine.add_argument("--output-manifest", type=Path, required=True)
    combine.add_argument("--allow-smoke-subset", action="store_true")
    _add_common_builder_pin(combine)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.mode == "freeze-inputs":
        if args.workers < 1:
            raise ValueError("--workers must be positive")
        freeze_inputs(args)
    elif args.mode == "generate-shard":
        generate_shard(args)
    elif args.mode == "aggregate":
        aggregate(args)
    else:  # pragma: no cover
        raise AssertionError(args.mode)


if __name__ == "__main__":
    main()
