#!/usr/bin/env python3
"""Score the frozen S50 PLINDER pose bank without generating new poses.

``freeze-inputs`` is the only stage allowed to read the source sampling CSVs.
It projects an allowlisted, label-free manifest and binds it to all relevant
artifacts.  ``score-shard`` reads only that manifest and the saved multi-record
SDF files.  It never reads RMSD, validity, oracle, or other outcome columns.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Sequence

import torch
from rdkit import Chem, rdBase

from effdock.confidence.runtime import (
    load_pose_confidence_model,
    score_poses_with_confidence,
)
from effdock.inference.docking import load_model
from effdock.inference.preprocess import preprocess_complex
from effdock.workflows.benchmark_inputs import load_benchmark_ligand

PROTOCOL_ID = "EFFDOCK-S50-CONFIDENCE-SCORE-ONLY-PLINDER-V1"
SOURCE_PROTOCOL_ID = "EFFDOCK-EARLY-TIME-SAMPLER-PLINDER-K2-GATE-V1"
BANK_SCHEMA_VERSION = "effdock.early_time_sampler_s50_confidence_bank.v1"
SCORE_SCHEMA_VERSION = "effdock.early_time_sampler_s50_confidence_scores.v1"
SCORE_LEDGER_DOMAIN = b"EFFDOCK_S50_CONFIDENCE_SCORE_LEDGER_V1\0"
ARM_NAMES = ("s50_backbone", "matched_backbone")
SOURCE_ARM = "s50_ema"
EXPECTED_SOURCE_SHARDS = 8
EXPECTED_ELIGIBLE_COUNT = 1035
EXPECTED_POSES = 100
EXPECTED_STEPS = 10
EXPECTED_SIGMA = 2.0
EXPECTED_PRIOR_POOL_SIZE = 100
POSE_BATCH_SIZE = 20
REPLAY_ABS_TOLERANCE = 1e-5
FULL_COUNT = 1076
ELIGIBLE_SYSTEM_COUNT = 1020
EXCLUDED_COUNT = 41

FROZEN_CONFIG_SHA256 = "39aa62e4a48ed6f3aa4ff59345fb43a81220e2baba22edfd5beb0c4981b307ec"
FROZEN_S50_BACKBONE_SHA256 = (
    "65be44d7dc8f0867eb9fc5d22214b80f93971ea4702679a527c665046e91e6b6"
)
FROZEN_MATCHED_BACKBONE_SHA256 = (
    "6932fb3ba6ebac770f714453529656a44b8f33cf15119d23c9e675d2d60b36db"
)
FROZEN_CONFIDENCE_SHA256 = (
    "e31fde6f351284205c78f7a1510002779c43312e94d9f82003d47a14d72bc78f"
)
FROZEN_SOURCE_PROTOCOL_SHA256 = (
    "0250853ae0793db288be2a6a8dc775db391d25aae32835b65b061782f34ab518"
)
FROZEN_ELIGIBILITY_SHA256 = (
    "6ebeb2d165e1def6ebf7b5bba301f82d4a9c3ff9d6c5cd43616dcf09edbd38ac"
)
FROZEN_SOURCE_REPORT_SHA256 = (
    "d4814796a9d274f836888dd614e5b6a4a5fba6b86001da83bea6720fabf02316"
)
FROZEN_SOURCE_AUDIT_SHA256 = (
    "3b6daa4a3d4c74ae384e7c3d2199d3d26f9360fe4b64a33e1c6ab16f4b83eabc"
)

RUNTIME_CODE_FILES = {
    "scorer": "scripts/score_early_time_sampler_plinder_confidence.py",
    "confidence_runtime": "src/effdock/confidence/runtime.py",
    "confidence_features": "src/effdock/confidence/features.py",
    "confidence_model": "src/effdock/confidence/model.py",
    "inference_docking": "src/effdock/inference/docking.py",
    "inference_preprocess": "src/effdock/inference/preprocess.py",
    "benchmark_inputs": "src/effdock/workflows/benchmark_inputs.py",
    "checkpoint": "src/effdock/checkpoint.py",
    "effdock_model": "src/effdock/models/effdock.py",
    "equivariant_model": "src/effdock/models/equivariant.py",
    "nn_utils": "src/effdock/models/nn_utils.py",
    "inference_sampler": "src/effdock/inference/sampler.py",
    "ligand_preprocess": "src/effdock/preprocess/ligand.py",
    "graph_types": "src/effdock/preprocess/graph_types.py",
    "se3_geometry": "src/effdock/geometry/se3.py",
    "dataset": "src/effdock/data/dataset.py",
    "fragment_preprocess": "src/effdock/preprocess/fragments.py",
    "graph_preprocess": "src/effdock/preprocess/graph.py",
    "protein_preprocess": "src/effdock/preprocess/protein.py",
    "dependency_lock": "uv.lock",
}
RUNTIME_CODE_DOMAIN = b"EFFDOCK_S50_CONFIDENCE_RUNTIME_CODE_V1\0"
BANK_INPUT_ASSET_NAMES = (
    "eligibility_manifest",
    "config",
    "s50_backbone_checkpoint",
    "matched_backbone_checkpoint",
    "confidence_checkpoint",
    "source_sampler_protocol",
    "protocol_document",
    "source_sampler_report",
    "source_coordinate_audit",
    "scorer_source",
    "report_source",
)

CONFIDENCE_SCORE_FIELDS = (
    "confidence_rmsd",
    "confidence_success_logit",
    "confidence_success",
    "confidence_atom_rmsd",
    "confidence_atom_q90",
    "confidence_atom_ok",
)

# This is the complete source-CSV information boundary for freeze-inputs.
# Do not add outcome, validity, oracle, selected-pose, or RMSD-label fields.
SOURCE_ROW_ALLOWLIST = (
    "id",
    "arm",
    "plinder_global_index",
    "sampling_seed",
    "ligand_conformer_seed",
    "all_poses_count",
    "all_poses_sdf",
    "all_poses_sdf_sha256",
    "candidate_ensemble_sha256",
    "checkpoint",
    "checkpoint_sha256",
    "ligand_input_canonical_smiles",
    "ligand_input_identity_sha256",
    "num_input_atoms",
    "num_samples",
    "plinder_ligand_chain",
    "plinder_system_id",
    "prior_pool_sha256",
    "prior_pool_size",
    "processed_meta",
    "processed_meta_sha256",
    "protein",
    "protein_sha256",
    "sampling_dynamics",
    "selector_profile",
)

BANK_RECORD_FIELDS = (
    "sample_key",
    "system_id",
    "ligand_chain",
    "plinder_global_index",
    "sampling_seed",
    "ligand_conformer_seed",
    "source_shard_index",
    "pose_count",
    "sample_sigma",
    "num_steps",
    "candidate_ensemble_sha256",
    "prior_pool_sha256",
    "prior_pool_size",
    "all_poses_sdf",
    "receptor",
    "receptor_sha256",
    "processed_meta",
    "processed_meta_sha256",
    "canonical_smiles",
    "ligand_input_identity_sha256",
    "pocket_center",
    "num_input_atoms",
)
SCORE_RECORD_FIELDS = (*BANK_RECORD_FIELDS, "score_arrays", "selected_index", "score_ledger_sha256")


class ScoreContractError(RuntimeError):
    """Raised when a frozen scoring identity or inventory does not match."""


def _require_exact_keys(value: dict[str, Any], expected: Sequence[str], *, label: str) -> None:
    observed = set(value)
    wanted = set(expected)
    if observed != wanted:
        raise ScoreContractError(
            f"{label} field inventory mismatch: missing={sorted(wanted - observed)} "
            f"extra={sorted(observed - wanted)}"
        )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sorted_id_sha256(ids: Sequence[str]) -> str:
    digest = hashlib.sha256()
    digest.update(b"EFFDOCK_SORTED_COMPLEX_IDS_V1\0")
    for sample_id in ids:
        digest.update(sample_id.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def score_ledger_sha256(score_arrays: dict[str, list[float]]) -> str:
    digest = hashlib.sha256(SCORE_LEDGER_DOMAIN)
    digest.update(_canonical_json_bytes(score_arrays))
    return digest.hexdigest()


def _runtime_code_identity() -> dict[str, Any]:
    repository_root = Path(__file__).resolve().parent.parent
    files: dict[str, dict[str, Any]] = {}
    for name, relative_path in RUNTIME_CODE_FILES.items():
        path = repository_root / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"missing runtime code dependency {name}: {path}")
        files[name] = {
            "path": str(path),
            "sha256": file_sha256(path),
            "size_bytes": path.stat().st_size,
        }
    hash_payload = {name: value["sha256"] for name, value in files.items()}
    digest = hashlib.sha256(RUNTIME_CODE_DOMAIN)
    digest.update(_canonical_json_bytes(hash_payload))
    return {"aggregate_sha256": digest.hexdigest(), "files": files}


def _has_symlink_component(path: Path) -> bool:
    current = path
    while True:
        if current.is_symlink():
            return True
        if current.parent == current:
            return False
        current = current.parent


def _canonical_existing_path(
    path: Path, *, label: str, directory: bool = False
) -> Path:
    if not path.is_absolute() or _has_symlink_component(path):
        raise ScoreContractError(
            f"{label} path must be absolute and contain no symlink components: {path}"
        )
    resolved = path.resolve()
    if str(path) != str(resolved):
        raise ScoreContractError(f"{label} path is not lexical-canonical: {path}")
    if directory:
        exists = resolved.is_dir()
        kind = "directory"
    else:
        exists = resolved.is_file()
        kind = "file"
    if not exists:
        raise FileNotFoundError(f"missing {label} {kind}: {resolved}")
    return resolved


def _canonical_new_output_path(path: Path, *, label: str) -> Path:
    if not path.is_absolute() or str(path) != str(path.resolve(strict=False)):
        raise ScoreContractError(f"{label} must be an absolute lexical-canonical path")
    existing = path.parent
    while not existing.exists() and existing.parent != existing:
        existing = existing.parent
    if not existing.is_dir() or _has_symlink_component(existing):
        raise ScoreContractError(f"{label} parent path is missing or contains a symlink")
    return path


def _canonical_smiles_identity(smiles: str) -> str:
    digest = hashlib.sha256()
    digest.update(b"EFFDOCK_PLINDER_CANONICAL_SMILES_V1\0")
    digest.update(smiles.encode("utf-8"))
    return digest.hexdigest()


def _require_sha256(value: str, *, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ScoreContractError(f"{label} must be a SHA-256 hex digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ScoreContractError(f"{label} must be a SHA-256 hex digest") from exc
    return value.lower()


def _asset(path: Path, expected_sha256: str, *, label: str) -> dict[str, Any]:
    expected = _require_sha256(expected_sha256, label=f"{label} expected SHA-256")
    resolved = _canonical_existing_path(path, label=label)
    observed = file_sha256(resolved)
    if observed != expected:
        raise ScoreContractError(
            f"{label} SHA-256 mismatch: expected={expected} observed={observed}"
        )
    return {
        "path": str(resolved),
        "sha256": observed,
        "size_bytes": resolved.stat().st_size,
    }


def _manifest_asset(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ScoreContractError(f"{label} must be an asset object")
    path = value.get("path")
    sha256 = value.get("sha256")
    if not isinstance(path, str) or not path:
        raise ScoreContractError(f"{label}.path must be a non-empty string")
    if not isinstance(sha256, str):
        raise ScoreContractError(f"{label}.sha256 is missing")
    parsed = Path(path)
    resolved = parsed.resolve()
    if not parsed.is_absolute() or str(parsed) != str(resolved):
        raise ScoreContractError(f"{label}.path is not lexical-canonical: {path}")
    return _asset(parsed, sha256, label=label)


def _sealed_manifest_asset(value: object, *, label: str) -> dict[str, str]:
    """Validate a bank-sealed identity without opening its source file."""
    if not isinstance(value, dict):
        raise ScoreContractError(f"{label} must be an asset object")
    path = value.get("path")
    sha256 = value.get("sha256")
    if not isinstance(path, str) or not Path(path).is_absolute():
        raise ScoreContractError(f"{label}.path must be absolute")
    if str(Path(path).resolve()) != path or _has_symlink_component(Path(path)):
        raise ScoreContractError(f"{label}.path is not lexical-canonical")
    return {"path": path, "sha256": _require_sha256(str(sha256), label=label)}


def _path_hash_only(asset: dict[str, Any]) -> dict[str, str]:
    return {"path": str(asset["path"]), "sha256": str(asset["sha256"])}


def _same_path(left: str | Path, right: str | Path) -> bool:
    return Path(left).resolve() == Path(right).resolve()


def _strict_int(value: object, *, label: str) -> int:
    if isinstance(value, bool):
        raise ScoreContractError(f"{label} must be an integer")
    try:
        parsed = int(str(value))
    except (TypeError, ValueError) as exc:
        raise ScoreContractError(f"{label} must be an integer") from exc
    if str(value).strip() not in {str(parsed), f"+{parsed}"}:
        raise ScoreContractError(f"{label} is not a canonical integer: {value!r}")
    return parsed


def _strict_float(value: object, *, label: str) -> float:
    if isinstance(value, bool):
        raise ScoreContractError(f"{label} must be numeric")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ScoreContractError(f"{label} must be numeric") from exc
    if not math.isfinite(parsed):
        raise ScoreContractError(f"{label} must be finite")
    return parsed


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ScoreContractError(f"cannot read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise ScoreContractError(f"{label} must be a JSON object")
    return value


def _atomic_write_new_json(path: Path, value: object) -> None:
    """Publish one complete JSON file atomically and never replace a target."""
    output = _canonical_new_output_path(path, label="output")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output: {output}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        # Hard-link publication is atomic and fails with EEXIST instead of replacing.
        os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def _read_allowlisted_rows(path: Path) -> tuple[list[dict[str, str]], str]:
    """Read a source CSV and immediately project it to the label-free allowlist."""
    data = path.read_bytes()
    sha256 = hashlib.sha256(data).hexdigest()
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ScoreContractError(f"source CSV is not UTF-8: {path}") from exc
    reader = csv.DictReader(io.StringIO(text, newline=""))
    fields = reader.fieldnames
    if fields is None or len(fields) != len(set(fields)):
        raise ScoreContractError(f"source CSV has missing/duplicate headers: {path}")
    missing = sorted(set(SOURCE_ROW_ALLOWLIST) - set(fields))
    if missing:
        raise ScoreContractError(f"source CSV lacks structural fields {missing}: {path}")
    rows: list[dict[str, str]] = []
    for raw in reader:
        if None in raw:
            raise ScoreContractError(f"source CSV has surplus columns: {path}")
        # This projection is the explicit outcome-information boundary.
        rows.append({field: str(raw[field]) for field in SOURCE_ROW_ALLOWLIST})
    return rows, sha256


def _validate_source_settings(settings: object) -> None:
    if not isinstance(settings, dict):
        raise ScoreContractError("paired summary settings must be an object")
    expected: dict[str, object] = {
        "stage": "full",
        "num_samples": EXPECTED_POSES,
        "num_steps": EXPECTED_STEPS,
        "model_pose_step_budget": EXPECTED_POSES * EXPECTED_STEPS,
        "sigma": EXPECTED_SIGMA,
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
        "ligand_conformer_seed": 0,
        "include_s50_replay": False,
    }
    mismatches = {
        key: {"expected": wanted, "observed": settings.get(key)}
        for key, wanted in expected.items()
        if settings.get(key) != wanted
    }
    if mismatches:
        raise ScoreContractError(f"source sampling settings mismatch: {mismatches}")


def _eligible_records(
    eligibility: dict[str, Any], *, expected_count: int
) -> tuple[list[str], dict[str, dict[str, Any]]]:
    if (
        eligibility.get("schema_version") != "effdock.plinder_checkpoint_eligibility.v1"
        or eligibility.get("protocol_id") != SOURCE_PROTOCOL_ID
        or eligibility.get("status") != "complete"
    ):
        raise ScoreContractError("eligibility manifest identity/status mismatch")
    inventory = eligibility.get("inventory")
    records = eligibility.get("records")
    if not isinstance(inventory, dict) or not isinstance(records, list):
        raise ScoreContractError("eligibility manifest lacks inventory/records")
    eligible_ids = inventory.get("eligible_ids")
    if not isinstance(eligible_ids, list) or not all(
        isinstance(value, str) and value for value in eligible_ids
    ):
        raise ScoreContractError("eligibility inventory has invalid eligible_ids")
    if eligible_ids != sorted(eligible_ids) or len(set(eligible_ids)) != len(eligible_ids):
        raise ScoreContractError("eligible IDs must be sorted and unique")
    if len(eligible_ids) != expected_count or inventory.get("eligible_count") != expected_count:
        raise ScoreContractError(
            f"expected {expected_count} eligible IDs, got {len(eligible_ids)}"
        )
    by_id: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("sample_key"), str):
            raise ScoreContractError("eligibility record is malformed")
        sample_id = str(record["sample_key"])
        if sample_id in by_id:
            raise ScoreContractError(f"duplicate eligibility record: {sample_id}")
        by_id[sample_id] = record
    actual_eligible = sorted(
        sample_id for sample_id, record in by_id.items() if record.get("status") == "eligible"
    )
    if actual_eligible != eligible_ids:
        raise ScoreContractError("eligible record set differs from inventory")
    return eligible_ids, by_id


def _validate_eligibility_fixed_inputs(
    eligibility: dict[str, Any],
    *,
    config: dict[str, Any],
    s50_backbone: dict[str, Any],
    source_sampler_protocol: dict[str, Any],
) -> None:
    fixed = eligibility.get("inputs", {}).get("fixed_identities", {})
    if not isinstance(fixed, dict):
        raise ScoreContractError("eligibility fixed identities are missing")
    checks = (
        (fixed.get("config"), config, "config"),
        (
            fixed.get("protocol_document"),
            source_sampler_protocol,
            "source sampler protocol document",
        ),
        (fixed.get("checkpoints", {}).get(SOURCE_ARM), s50_backbone, "S50 checkpoint"),
    )
    for frozen, actual, label in checks:
        if not isinstance(frozen, dict):
            raise ScoreContractError(f"eligibility lacks frozen {label}")
        if (
            frozen.get("sha256") != actual["sha256"]
            or frozen.get("path") != actual["path"]
        ):
            raise ScoreContractError(f"eligibility frozen {label} identity mismatch")


def _project_source_row(
    row: dict[str, str],
    *,
    record: dict[str, Any],
    shard_index: int,
    shard_dir: Path,
    source_output_root: Path,
    s50_backbone: dict[str, Any],
) -> dict[str, Any]:
    sample_id = row["id"]
    label = f"source shard {shard_index} row {sample_id}"
    if row["arm"] != SOURCE_ARM:
        raise ScoreContractError(f"{label}: arm must be {SOURCE_ARM}")
    if row["selector_profile"] != "candidate_only":
        raise ScoreContractError(f"{label}: selector profile mismatch")
    if row["sampling_dynamics"] != "deterministic_ode":
        raise ScoreContractError(f"{label}: sampling dynamics mismatch")
    if (
        row["checkpoint_sha256"] != s50_backbone["sha256"]
        or row["checkpoint"] != s50_backbone["path"]
    ):
        raise ScoreContractError(f"{label}: source checkpoint mismatch")
    integer_fields = {
        "plinder_global_index": int(record["global_index"]),
        "sampling_seed": int(record["sampling_seed"]),
        "ligand_conformer_seed": 0,
        "all_poses_count": EXPECTED_POSES,
        "num_samples": EXPECTED_POSES,
        "prior_pool_size": EXPECTED_PRIOR_POOL_SIZE,
    }
    for field, expected in integer_fields.items():
        if _strict_int(row[field], label=f"{label}.{field}") != expected:
            raise ScoreContractError(f"{label}: {field} mismatch")
    if row["plinder_system_id"] != record.get("system_id"):
        raise ScoreContractError(f"{label}: system ID mismatch")
    if row["plinder_ligand_chain"] != record.get("ligand_chain"):
        raise ScoreContractError(f"{label}: ligand chain mismatch")
    smiles = str(record.get("canonical_smiles", ""))
    smiles_identity = _canonical_smiles_identity(smiles)
    if (
        not smiles
        or row["ligand_input_canonical_smiles"] != smiles
        or row["ligand_input_identity_sha256"] != smiles_identity
        or record.get("canonical_smiles_identity_sha256") != smiles_identity
    ):
        raise ScoreContractError(f"{label}: canonical SMILES identity mismatch")

    receptor = _manifest_asset(
        record.get("receptor"), label=f"{label}.eligibility receptor"
    )
    processed_meta = _manifest_asset(
        record.get("processed_meta"), label=f"{label}.eligibility processed meta"
    )
    asset_rows = (
        (row["protein"], row["protein_sha256"], receptor, "receptor"),
        (
            row["processed_meta"],
            row["processed_meta_sha256"],
            processed_meta,
            "processed meta",
        ),
    )
    for declared_path, declared_hash, frozen, asset_label in asset_rows:
        if (
            declared_hash != frozen.get("sha256")
            or declared_path != frozen.get("path")
            or str(Path(declared_path).resolve()) != declared_path
        ):
            raise ScoreContractError(f"{label}: {asset_label} identity mismatch")

    expected_sdf = (
        shard_dir / "arms" / SOURCE_ARM / "poses" / "all_poses" / f"{sample_id}.sdf"
    ).resolve()
    try:
        expected_sdf.relative_to(source_output_root)
    except ValueError as exc:
        raise ScoreContractError(f"{label}: SDF escaped the frozen source root") from exc
    if row["all_poses_sdf"] != str(expected_sdf):
        raise ScoreContractError(f"{label}: SDF path is not the canonical final path")
    if (
        ".incomplete" in expected_sdf.parts
        or not expected_sdf.is_file()
        or _has_symlink_component(expected_sdf)
    ):
        raise ScoreContractError(f"{label}: final SDF is missing or incomplete")
    sdf_hash = _require_sha256(row["all_poses_sdf_sha256"], label=f"{label}.SDF hash")
    if file_sha256(expected_sdf) != sdf_hash:
        raise ScoreContractError(f"{label}: SDF SHA-256 mismatch")
    candidate_hash = _require_sha256(
        row["candidate_ensemble_sha256"], label=f"{label}.candidate ensemble"
    )
    prior_hash = _require_sha256(row["prior_pool_sha256"], label=f"{label}.prior pool")
    pocket_center = record.get("pocket_center")
    if (
        not isinstance(pocket_center, list)
        or len(pocket_center) != 3
        or not all(math.isfinite(float(value)) for value in pocket_center)
    ):
        raise ScoreContractError(f"{label}: invalid pocket center")
    return {
        "sample_key": sample_id,
        "system_id": str(record["system_id"]),
        "ligand_chain": str(record["ligand_chain"]),
        "plinder_global_index": int(record["global_index"]),
        "sampling_seed": int(record["sampling_seed"]),
        "ligand_conformer_seed": 0,
        "source_shard_index": shard_index,
        "pose_count": EXPECTED_POSES,
        "sample_sigma": EXPECTED_SIGMA,
        "num_steps": EXPECTED_STEPS,
        "candidate_ensemble_sha256": candidate_hash,
        "prior_pool_sha256": prior_hash,
        "prior_pool_size": EXPECTED_PRIOR_POOL_SIZE,
        "all_poses_sdf": {"path": str(expected_sdf), "sha256": sdf_hash},
        "receptor": {
            "path": str(receptor["path"]),
            "sha256": str(receptor["sha256"]),
        },
        "receptor_sha256": str(receptor["sha256"]),
        "processed_meta": {
            "path": str(processed_meta["path"]),
            "sha256": str(processed_meta["sha256"]),
        },
        "processed_meta_sha256": str(processed_meta["sha256"]),
        "canonical_smiles": smiles,
        "ligand_input_identity_sha256": smiles_identity,
        "pocket_center": [float(value) for value in pocket_center],
        "num_input_atoms": _strict_int(
            row["num_input_atoms"], label=f"{label}.num_input_atoms"
        ),
    }


def freeze_label_free_inputs(
    *,
    bank_root: Path,
    eligibility_manifest: Path,
    expected_eligibility_manifest_sha256: str,
    config: Path,
    expected_config_sha256: str,
    s50_backbone_checkpoint: Path,
    expected_s50_backbone_checkpoint_sha256: str,
    matched_backbone_checkpoint: Path,
    expected_matched_backbone_checkpoint_sha256: str,
    confidence_checkpoint: Path,
    expected_confidence_checkpoint_sha256: str,
    source_sampler_protocol: Path,
    expected_source_sampler_protocol_sha256: str,
    protocol_document: Path,
    expected_protocol_sha256: str,
    source_sampler_report: Path,
    expected_source_sampler_report_sha256: str,
    source_coordinate_audit: Path,
    expected_source_coordinate_audit_sha256: str,
    report_source: Path,
    expected_report_source_sha256: str,
    expected_scorer_source_sha256: str,
    expected_runtime_code_sha256: str,
    output: Path,
    expected_source_shards: int = EXPECTED_SOURCE_SHARDS,
    expected_eligible_count: int = EXPECTED_ELIGIBLE_COUNT,
) -> dict[str, Any]:
    if expected_source_shards < 1 or expected_eligible_count < 1:
        raise ValueError("expected source shard/eligible counts must be positive")
    root = _canonical_existing_path(bank_root, label="source output root", directory=True)
    if root.name != "full":
        raise ScoreContractError("bank root must be the completed source run's full directory")
    frozen_exact_hashes = {
        "eligibility manifest": (
            expected_eligibility_manifest_sha256,
            FROZEN_ELIGIBILITY_SHA256,
        ),
        "config": (expected_config_sha256, FROZEN_CONFIG_SHA256),
        "S50 backbone": (
            expected_s50_backbone_checkpoint_sha256,
            FROZEN_S50_BACKBONE_SHA256,
        ),
        "matched backbone": (
            expected_matched_backbone_checkpoint_sha256,
            FROZEN_MATCHED_BACKBONE_SHA256,
        ),
        "confidence": (expected_confidence_checkpoint_sha256, FROZEN_CONFIDENCE_SHA256),
        "source sampler protocol": (
            expected_source_sampler_protocol_sha256,
            FROZEN_SOURCE_PROTOCOL_SHA256,
        ),
        "source sampler report": (
            expected_source_sampler_report_sha256,
            FROZEN_SOURCE_REPORT_SHA256,
        ),
        "source coordinate audit": (
            expected_source_coordinate_audit_sha256,
            FROZEN_SOURCE_AUDIT_SHA256,
        ),
    }
    for label, (observed, expected) in frozen_exact_hashes.items():
        if _require_sha256(observed, label=f"{label} SHA-256") != expected:
            raise ScoreContractError(f"{label} does not match the frozen protocol identity")
    runtime_code_identity = _runtime_code_identity()
    if runtime_code_identity["aggregate_sha256"] != _require_sha256(
        expected_runtime_code_sha256, label="runtime code aggregate SHA-256"
    ):
        raise ScoreContractError("runtime code aggregate SHA-256 mismatch")
    assets = {
        "eligibility_manifest": _asset(
            eligibility_manifest,
            expected_eligibility_manifest_sha256,
            label="eligibility manifest",
        ),
        "config": _asset(config, expected_config_sha256, label="config"),
        "s50_backbone_checkpoint": _asset(
            s50_backbone_checkpoint,
            expected_s50_backbone_checkpoint_sha256,
            label="S50 backbone checkpoint",
        ),
        "matched_backbone_checkpoint": _asset(
            matched_backbone_checkpoint,
            expected_matched_backbone_checkpoint_sha256,
            label="matched backbone checkpoint",
        ),
        "confidence_checkpoint": _asset(
            confidence_checkpoint,
            expected_confidence_checkpoint_sha256,
            label="confidence checkpoint",
        ),
        "source_sampler_protocol": _asset(
            source_sampler_protocol,
            expected_source_sampler_protocol_sha256,
            label="source sampler protocol document",
        ),
        "protocol_document": _asset(
            protocol_document, expected_protocol_sha256, label="protocol document"
        ),
        "source_sampler_report": _asset(
            source_sampler_report,
            expected_source_sampler_report_sha256,
            label="source sampler report",
        ),
        "source_coordinate_audit": _asset(
            source_coordinate_audit,
            expected_source_coordinate_audit_sha256,
            label="source coordinate audit",
        ),
        "scorer_source": _asset(
            Path(__file__).resolve(),
            expected_scorer_source_sha256,
            label="scorer source",
        ),
        "report_source": _asset(
            report_source, expected_report_source_sha256, label="report source"
        ),
        "source_output_root": str(root),
        "runtime_code_identity": runtime_code_identity,
    }
    eligibility = _read_json(Path(assets["eligibility_manifest"]["path"]), label="eligibility")
    eligible_ids, eligibility_by_id = _eligible_records(
        eligibility, expected_count=expected_eligible_count
    )
    _validate_eligibility_fixed_inputs(
        eligibility,
        config=assets["config"],
        s50_backbone=assets["s50_backbone_checkpoint"],
        source_sampler_protocol=assets["source_sampler_protocol"],
    )

    eligibility_inventory = eligibility["inventory"]
    exact_eligibility_inventory = {
        "full_count": FULL_COUNT,
        "eligible_count": expected_eligible_count,
        "eligible_system_count": ELIGIBLE_SYSTEM_COUNT,
        "excluded_count": EXCLUDED_COUNT,
    }
    for key, expected in exact_eligibility_inventory.items():
        if eligibility_inventory.get(key) != expected:
            raise ScoreContractError(f"eligibility inventory {key} mismatch")
    full_ids = eligibility_inventory.get("full_ids")
    excluded_ids = eligibility_inventory.get("excluded_ids")
    if (
        not isinstance(full_ids, list)
        or not isinstance(excluded_ids, list)
        or full_ids != sorted(full_ids)
        or excluded_ids != sorted(excluded_ids)
        or sorted(eligible_ids + excluded_ids) != full_ids
        or eligibility_inventory.get("full_ids_sha256") != sorted_id_sha256(full_ids)
        or eligibility_inventory.get("excluded_ids_sha256")
        != sorted_id_sha256(excluded_ids)
    ):
        raise ScoreContractError("eligibility full/excluded cohort identity mismatch")

    records: list[dict[str, Any]] = []
    source_shards: list[dict[str, Any]] = []
    for shard_index in range(expected_source_shards):
        shard_dir = root / f"shard-{shard_index:03d}-of-{expected_source_shards:03d}"
        paired_path = shard_dir / "paired_summary.json"
        if _canonical_existing_path(
            paired_path, label=f"paired source summary {shard_index}"
        ) != paired_path:
            raise ScoreContractError("paired source summary path is not lexical-canonical")
        paired_hash = file_sha256(paired_path)
        paired = _read_json(paired_path, label=f"paired summary {shard_index}")
        if (
            paired.get("schema_version") != "effdock.plinder_checkpoint_paired_shard.v1"
            or paired.get("protocol_id") != SOURCE_PROTOCOL_ID
            or paired.get("status") != "complete"
            or paired.get("mode") != "full"
        ):
            raise ScoreContractError(f"source shard {shard_index}: paired identity mismatch")
        _validate_source_settings(paired.get("settings"))
        frozen_eligibility = paired.get("eligibility_manifest")
        if (
            not isinstance(frozen_eligibility, dict)
            or frozen_eligibility.get("sha256") != assets["eligibility_manifest"]["sha256"]
        ):
            raise ScoreContractError(f"source shard {shard_index}: eligibility binding mismatch")
        expected_ids = eligible_ids[shard_index::expected_source_shards]
        inventory = paired.get("inventory")
        if (
            not isinstance(inventory, dict)
            or inventory.get("num_shards") != expected_source_shards
            or inventory.get("eligible_count") != expected_eligible_count
            or inventory.get("selected_ids") != eligible_ids
            or inventory.get("assigned_ids") != expected_ids
            or inventory.get("assigned_count") != len(expected_ids)
        ):
            raise ScoreContractError(f"source shard {shard_index}: inventory mismatch")
        if paired.get("failures") != []:
            raise ScoreContractError(f"source shard {shard_index}: source failures are non-empty")
        arms = paired.get("arms")
        if not isinstance(arms, list):
            raise ScoreContractError(f"source shard {shard_index}: arms are missing")
        s50_arms = [arm for arm in arms if isinstance(arm, dict) and arm.get("name") == SOURCE_ARM]
        if len(s50_arms) != 1:
            raise ScoreContractError(f"source shard {shard_index}: S50 arm identity missing")
        s50_arm = s50_arms[0]
        if (
            s50_arm.get("checkpoint_sha256")
            != assets["s50_backbone_checkpoint"]["sha256"]
            or s50_arm.get("checkpoint") != assets["s50_backbone_checkpoint"]["path"]
        ):
            raise ScoreContractError(f"source shard {shard_index}: S50 arm checkpoint mismatch")

        csv_path = shard_dir / "arms" / SOURCE_ARM / "results.csv"
        if _canonical_existing_path(
            csv_path, label=f"source results CSV {shard_index}"
        ) != csv_path:
            raise ScoreContractError("source results CSV path is not lexical-canonical")
        artifact = paired.get("artifacts", {}).get("arms", {}).get(SOURCE_ARM)
        if not isinstance(artifact, dict):
            raise ScoreContractError(f"source shard {shard_index}: S50 artifact missing")
        if artifact.get("results_csv") != str(csv_path):
            raise ScoreContractError(f"source shard {shard_index}: results CSV path mismatch")
        projected_rows, csv_hash = _read_allowlisted_rows(csv_path)
        if artifact.get("results_csv_sha256") != csv_hash:
            raise ScoreContractError(f"source shard {shard_index}: results CSV hash mismatch")
        if artifact.get("count") != len(expected_ids) or len(projected_rows) != len(expected_ids):
            raise ScoreContractError(f"source shard {shard_index}: results row count mismatch")
        observed_ids = [row["id"] for row in projected_rows]
        if observed_ids != expected_ids or len(set(observed_ids)) != len(observed_ids):
            raise ScoreContractError(f"source shard {shard_index}: ordered results IDs mismatch")

        for row in projected_rows:
            projected = _project_source_row(
                row,
                record=eligibility_by_id[row["id"]],
                shard_index=shard_index,
                shard_dir=shard_dir,
                source_output_root=root,
                s50_backbone=assets["s50_backbone_checkpoint"],
            )
            _require_exact_keys(projected, BANK_RECORD_FIELDS, label=row["id"])
            records.append(projected)
        source_shards.append(
            {
                "shard_index": shard_index,
                "paired_summary": {"path": str(paired_path.resolve()), "sha256": paired_hash},
                "results_csv": {"path": str(csv_path.resolve()), "sha256": csv_hash},
                "assigned_count": len(expected_ids),
                "assigned_ids_sha256": sorted_id_sha256(expected_ids),
            }
        )

    if [record["sample_key"] for record in records] != [
        sample_id
        for shard_index in range(expected_source_shards)
        for sample_id in eligible_ids[shard_index::expected_source_shards]
    ]:
        raise ScoreContractError("frozen record order does not match shard-major source order")
    if len(records) != expected_eligible_count or len(
        {record["sample_key"] for record in records}
    ) != expected_eligible_count:
        raise ScoreContractError("frozen bank record union is incomplete or duplicated")
    records_by_id = {str(record["sample_key"]): record for record in records}
    records = [records_by_id[sample_id] for sample_id in eligible_ids]

    manifest_inputs: dict[str, Any] = {
        name: _path_hash_only(assets[name]) for name in BANK_INPUT_ASSET_NAMES
    }
    manifest_inputs["source_output_root"] = str(root)
    manifest_inputs["runtime_code_identity"] = runtime_code_identity
    manifest = {
        "schema_version": BANK_SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "source_protocol_id": SOURCE_PROTOCOL_ID,
        "status": "complete_label_free",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "information_boundary": {
            "source_csv_allowlist": list(SOURCE_ROW_ALLOWLIST),
            "outcome_columns_exported": False,
            "score_stage_reads_source_results_csv": False,
            "crystal_reference_exported": False,
        },
        "bank_root": str(root),
        "inputs": manifest_inputs,
        "backbone_arms": {
            "s50_backbone": {
                **_path_hash_only(assets["s50_backbone_checkpoint"]),
                "role": "primary_deployment_backbone",
            },
            "matched_backbone": {
                **_path_hash_only(assets["matched_backbone_checkpoint"]),
                "role": "diagnostic_training_matched_backbone",
            },
        },
        "fixed_settings": {
            "source_arm": SOURCE_ARM,
            "pose_count": EXPECTED_POSES,
            "sample_sigma": EXPECTED_SIGMA,
            "num_steps": EXPECTED_STEPS,
            "prior_pool_size": EXPECTED_PRIOR_POOL_SIZE,
            "pose_batch_size": POSE_BATCH_SIZE,
            "pocket_cutoff_angstrom": 10.0,
            "ligand_conformer_seed": 0,
            "selector": "stable_argmin_confidence_rmsd",
            "label_blind": True,
            "resampling": False,
        },
        "inventory": {
            "full_count": FULL_COUNT,
            "eligible_count": expected_eligible_count,
            "eligible_system_count": ELIGIBLE_SYSTEM_COUNT,
            "excluded_count": EXCLUDED_COUNT,
            "source_shard_count": expected_source_shards,
            "pose_count": EXPECTED_POSES,
            "full_ids_sha256": eligibility_inventory["full_ids_sha256"],
            "eligible_ids_sha256": sorted_id_sha256(eligible_ids),
            "excluded_ids_sha256": eligibility_inventory["excluded_ids_sha256"],
        },
        "source_shards": source_shards,
        "records": records,
    }
    _atomic_write_new_json(output, manifest)
    return manifest


def _topology_signature(molecule: Chem.Mol) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    """Return an ordered heavy-atom chemical-graph signature.

    V2000 round-tripping may move attached hydrogens between RDKit's explicit-H
    count and implicit valence without changing the heavy-atom molecule.  Use
    the total attached-H count so that representation-only changes compare
    equal while protonation, charge, isotope, radical, atom-order, and bond
    mutations still fail closed.
    """
    atoms = tuple(
        (
            atom.GetIdx(),
            atom.GetAtomicNum(),
            atom.GetFormalCharge(),
            atom.GetIsotope(),
            atom.GetTotalNumHs(includeNeighbors=True),
            atom.GetNumRadicalElectrons(),
            atom.GetIsAromatic(),
        )
        for atom in molecule.GetAtoms()
    )
    bonds = tuple(
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
    return atoms, bonds


def _required_sdf_property(
    molecule: Chem.Mol, name: str, *, label: str
) -> str:
    if not molecule.HasProp(name):
        raise ScoreContractError(f"{label}: SDF property {name!r} is missing")
    return molecule.GetProp(name)


def read_saved_bank_poses(
    record: dict[str, Any],
    *,
    ligand: Chem.Mol,
) -> list[torch.Tensor]:
    sample_id = str(record["sample_key"])
    _require_exact_keys(record, BANK_RECORD_FIELDS, label=f"{sample_id} bank record")
    sdf = _manifest_asset(record.get("all_poses_sdf"), label=f"{sample_id}.all_poses_sdf")
    sdf_path = Path(sdf["path"])
    expected_topology = _topology_signature(ligand)
    center = torch.tensor(record["pocket_center"], dtype=torch.float32)
    poses: list[torch.Tensor] = []
    with sdf_path.open("rb") as handle:
        supplier = Chem.ForwardSDMolSupplier(
            handle, removeHs=False, sanitize=True, strictParsing=True
        )
        for pose_index, molecule in enumerate(supplier):
            label = f"{sample_id}.pose[{pose_index}]"
            if molecule is None:
                raise ScoreContractError(f"{label}: RDKit failed to parse SDF record")
            if pose_index >= EXPECTED_POSES:
                raise ScoreContractError(
                    f"{sample_id}: SDF has more than {EXPECTED_POSES} poses"
                )
            if _topology_signature(molecule) != expected_topology:
                raise ScoreContractError(
                    f"{label}: atom order/constitutional topology mismatch"
                )
            expected_properties: dict[str, str] = {
                "_Name": f"docked_pose_{pose_index}",
                "sample_index": str(pose_index),
                "complex_id": sample_id,
                "dataset": "plinder_val",
                "sampling_seed": str(record["sampling_seed"]),
                "ligand_conformer_seed": str(record["ligand_conformer_seed"]),
                "num_samples": str(EXPECTED_POSES),
                "num_steps": str(EXPECTED_STEPS),
                "candidate_ensemble_sha256": str(record["candidate_ensemble_sha256"]),
            }
            for name, expected in expected_properties.items():
                observed = _required_sdf_property(molecule, name, label=label)
                if observed != expected:
                    raise ScoreContractError(
                        f"{label}: SDF property {name!r} mismatch; "
                        f"expected={expected!r} observed={observed!r}"
                    )
            sigma = _strict_float(
                _required_sdf_property(molecule, "sample_sigma", label=label),
                label=f"{label}.sample_sigma",
            )
            if not math.isclose(sigma, EXPECTED_SIGMA, rel_tol=0.0, abs_tol=1e-12):
                raise ScoreContractError(f"{label}: sample sigma must be {EXPECTED_SIGMA}")
            coordinates = torch.tensor(
                molecule.GetConformer().GetPositions(), dtype=torch.float32
            )
            if tuple(coordinates.shape) != (ligand.GetNumAtoms(), 3) or not bool(
                torch.isfinite(coordinates).all()
            ):
                raise ScoreContractError(f"{label}: coordinate shape/values are invalid")
            poses.append(coordinates - center)
    if len(poses) != EXPECTED_POSES:
        raise ScoreContractError(
            f"{sample_id}: SDF record count {len(poses)} != {EXPECTED_POSES}"
        )
    if file_sha256(sdf_path) != sdf["sha256"]:
        raise ScoreContractError(f"{sample_id}: SDF changed while being read")
    return poses


def _score_in_chunks(
    confidence_model: torch.nn.Module,
    docking_model: torch.nn.Module,
    graph: dict[str, torch.Tensor],
    ligand_data: dict[str, torch.Tensor],
    meta: dict[str, Any],
    poses: list[torch.Tensor],
    *,
    device: torch.device,
    score_fn: Callable[..., list[dict[str, float]]] = score_poses_with_confidence,
) -> list[dict[str, float]]:
    if len(poses) != EXPECTED_POSES:
        raise ScoreContractError(f"scorer requires exactly {EXPECTED_POSES} poses")
    scores: list[dict[str, float]] = []
    for start in range(0, len(poses), POSE_BATCH_SIZE):
        stop = min(start + POSE_BATCH_SIZE, len(poses))
        chunk = score_fn(
            confidence_model,
            docking_model,
            graph,
            ligand_data,
            meta,
            poses[start:stop],
            sigma=EXPECTED_SIGMA,
            device=device,
        )
        if len(chunk) != stop - start:
            raise ScoreContractError("confidence scorer changed the pose count")
        scores.extend(chunk)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if len(scores) != EXPECTED_POSES:
        raise ScoreContractError("confidence score inventory is incomplete")
    for index, score in enumerate(scores):
        if not isinstance(score, dict) or set(score) != set(CONFIDENCE_SCORE_FIELDS):
            raise ScoreContractError(f"pose {index}: confidence score fields mismatch")
        if not all(
            isinstance(score[field], int | float)
            and not isinstance(score[field], bool)
            and math.isfinite(float(score[field]))
            for field in CONFIDENCE_SCORE_FIELDS
        ):
            raise ScoreContractError(f"pose {index}: confidence scores must be finite")
        if any(
            float(score[field]) < 0.0
            for field in (
                "confidence_rmsd",
                "confidence_atom_rmsd",
                "confidence_atom_q90",
            )
        ):
            raise ScoreContractError(f"pose {index}: predicted distances must be non-negative")
        if any(
            not 0.0 <= float(score[field]) <= 1.0
            for field in ("confidence_success", "confidence_atom_ok")
        ):
            raise ScoreContractError(f"pose {index}: confidence probabilities are invalid")
    return scores


def _score_arrays(scores: list[dict[str, float]]) -> dict[str, list[float]]:
    return {
        field: [float(score[field]) for score in scores]
        for field in CONFIDENCE_SCORE_FIELDS
    }


def _selected_index(score_arrays: dict[str, list[float]]) -> int:
    values = score_arrays["confidence_rmsd"]
    if len(values) != EXPECTED_POSES or not all(math.isfinite(value) for value in values):
        raise ScoreContractError("predicted RMSD score inventory is invalid")
    return min(range(len(values)), key=lambda index: (values[index], index))


def _validate_score_bank_manifest(
    manifest: dict[str, Any], *, expected_manifest_sha256: str, manifest_path: Path
) -> None:
    if (
        manifest.get("schema_version") != BANK_SCHEMA_VERSION
        or manifest.get("protocol_id") != PROTOCOL_ID
        or manifest.get("source_protocol_id") != SOURCE_PROTOCOL_ID
        or manifest.get("status") != "complete_label_free"
    ):
        raise ScoreContractError("label-free bank manifest identity/status mismatch")
    if file_sha256(manifest_path) != _require_sha256(
        expected_manifest_sha256, label="expected bank manifest SHA-256"
    ):
        raise ScoreContractError("label-free bank manifest SHA-256 mismatch")
    boundary = manifest.get("information_boundary")
    if not isinstance(boundary, dict) or any(
        (
            boundary.get("outcome_columns_exported") is not False,
            boundary.get("score_stage_reads_source_results_csv") is not False,
            boundary.get("crystal_reference_exported") is not False,
        )
    ):
        raise ScoreContractError("label-free information boundary mismatch")
    fixed = manifest.get("fixed_settings")
    expected_fixed = {
        "source_arm": SOURCE_ARM,
        "pose_count": EXPECTED_POSES,
        "sample_sigma": EXPECTED_SIGMA,
        "num_steps": EXPECTED_STEPS,
        "prior_pool_size": EXPECTED_PRIOR_POOL_SIZE,
        "pose_batch_size": POSE_BATCH_SIZE,
        "pocket_cutoff_angstrom": 10.0,
        "ligand_conformer_seed": 0,
        "selector": "stable_argmin_confidence_rmsd",
        "label_blind": True,
        "resampling": False,
    }
    if not isinstance(fixed, dict) or any(
        fixed.get(key) != value for key, value in expected_fixed.items()
    ):
        raise ScoreContractError("label-free scoring settings mismatch")

    inventory = manifest.get("inventory")
    expected_inventory = {
        "full_count": FULL_COUNT,
        "eligible_count": EXPECTED_ELIGIBLE_COUNT,
        "eligible_system_count": ELIGIBLE_SYSTEM_COUNT,
        "excluded_count": EXCLUDED_COUNT,
        "source_shard_count": EXPECTED_SOURCE_SHARDS,
        "pose_count": EXPECTED_POSES,
    }
    if not isinstance(inventory, dict) or any(
        inventory.get(key) != value for key, value in expected_inventory.items()
    ):
        raise ScoreContractError("label-free bank inventory mismatch")
    for key in ("full_ids_sha256", "eligible_ids_sha256", "excluded_ids_sha256"):
        _require_sha256(str(inventory.get(key, "")), label=f"inventory.{key}")

    inputs = manifest.get("inputs")
    if not isinstance(inputs, dict):
        raise ScoreContractError("label-free bank inputs are missing")
    root_value = inputs.get("source_output_root")
    if not isinstance(root_value, str):
        raise ScoreContractError("source_output_root is missing")
    source_root = _canonical_existing_path(
        Path(root_value), label="frozen source output root", directory=True
    )
    if source_root.name != "full" or manifest.get("bank_root") != str(source_root):
        raise ScoreContractError("frozen source output root identity mismatch")

    records = manifest.get("records")
    if not isinstance(records, list) or len(records) != EXPECTED_ELIGIBLE_COUNT:
        raise ScoreContractError("label-free bank records are incomplete")
    sample_ids: list[str] = []
    for global_order, record in enumerate(records):
        if not isinstance(record, dict):
            raise ScoreContractError(f"bank record {global_order} is not an object")
        _require_exact_keys(record, BANK_RECORD_FIELDS, label=f"bank record {global_order}")
        sample_id = record.get("sample_key")
        if not isinstance(sample_id, str) or not sample_id:
            raise ScoreContractError(f"bank record {global_order} has invalid sample_key")
        sample_ids.append(sample_id)
        expected_shard = global_order % EXPECTED_SOURCE_SHARDS
        if record.get("source_shard_index") != expected_shard:
            raise ScoreContractError(f"{sample_id}: source shard assignment mismatch")
        scalar_exact = {
            "pose_count": EXPECTED_POSES,
            "sample_sigma": EXPECTED_SIGMA,
            "num_steps": EXPECTED_STEPS,
            "prior_pool_size": EXPECTED_PRIOR_POOL_SIZE,
            "ligand_conformer_seed": 0,
        }
        if any(record.get(key) != value for key, value in scalar_exact.items()):
            raise ScoreContractError(f"{sample_id}: frozen scalar settings mismatch")
        for asset_name in ("all_poses_sdf", "receptor", "processed_meta"):
            asset = record.get(asset_name)
            if not isinstance(asset, dict) or set(asset) != {"path", "sha256"}:
                raise ScoreContractError(f"{sample_id}.{asset_name}: invalid path spec")
            path_value = asset.get("path")
            if not isinstance(path_value, str) or not Path(path_value).is_absolute():
                raise ScoreContractError(f"{sample_id}.{asset_name}: path is not absolute")
            if str(Path(path_value).resolve()) != path_value or _has_symlink_component(
                Path(path_value)
            ):
                raise ScoreContractError(
                    f"{sample_id}.{asset_name}: path is not lexical-canonical"
                )
            _require_sha256(str(asset.get("sha256", "")), label=f"{sample_id}.{asset_name}")
        sdf_path = Path(str(record["all_poses_sdf"]["path"]))
        expected_sdf = (
            source_root
            / f"shard-{expected_shard:03d}-of-{EXPECTED_SOURCE_SHARDS:03d}"
            / "arms"
            / SOURCE_ARM
            / "poses"
            / "all_poses"
            / f"{sample_id}.sdf"
        )
        if sdf_path != expected_sdf:
            raise ScoreContractError(f"{sample_id}: SDF escaped canonical source location")
        if record.get("receptor_sha256") != record["receptor"]["sha256"]:
            raise ScoreContractError(f"{sample_id}: receptor hash aliases differ")
        if record.get("processed_meta_sha256") != record["processed_meta"]["sha256"]:
            raise ScoreContractError(f"{sample_id}: processed-meta hash aliases differ")
    if sample_ids != sorted(sample_ids) or len(set(sample_ids)) != len(sample_ids):
        raise ScoreContractError("bank records are not the lexicographic eligible cohort")
    if inventory.get("eligible_ids_sha256") != sorted_id_sha256(sample_ids):
        raise ScoreContractError("bank eligible ID digest mismatch")

    source_shards = manifest.get("source_shards")
    if not isinstance(source_shards, list) or len(source_shards) != EXPECTED_SOURCE_SHARDS:
        raise ScoreContractError("source shard provenance inventory mismatch")
    for shard_index, source in enumerate(source_shards):
        expected_ids = sample_ids[shard_index::EXPECTED_SOURCE_SHARDS]
        if not isinstance(source, dict) or set(source) != {
            "shard_index",
            "paired_summary",
            "results_csv",
            "assigned_count",
            "assigned_ids_sha256",
        }:
            raise ScoreContractError(f"source shard {shard_index} schema mismatch")
        if (
            source.get("shard_index") != shard_index
            or source.get("assigned_count") != len(expected_ids)
            or source.get("assigned_ids_sha256") != sorted_id_sha256(expected_ids)
        ):
            raise ScoreContractError(f"source shard {shard_index} inventory mismatch")
        for asset_name in ("paired_summary", "results_csv"):
            asset = source.get(asset_name)
            if not isinstance(asset, dict) or set(asset) != {"path", "sha256"}:
                raise ScoreContractError(f"source shard {shard_index}.{asset_name} invalid")
            expected_path = (
                source_root
                / f"shard-{shard_index:03d}-of-{EXPECTED_SOURCE_SHARDS:03d}"
                / (
                    "paired_summary.json"
                    if asset_name == "paired_summary"
                    else f"arms/{SOURCE_ARM}/results.csv"
                )
            )
            if asset.get("path") != str(expected_path):
                raise ScoreContractError(
                    f"source shard {shard_index}.{asset_name} path mismatch"
                )
            _require_sha256(
                str(asset.get("sha256", "")),
                label=f"source shard {shard_index}.{asset_name}",
            )


def _verify_score_stage_assets(
    manifest: dict[str, Any],
    *,
    arm: str,
    docking_checkpoint: Path,
    expected_docking_checkpoint_sha256: str,
    confidence_checkpoint: Path,
    expected_confidence_checkpoint_sha256: str,
    expected_protocol_sha256: str,
    expected_scorer_source_sha256: str,
    expected_report_source_sha256: str,
    expected_runtime_code_sha256: str,
) -> dict[str, Any]:
    inputs = manifest.get("inputs")
    backbones = manifest.get("backbone_arms")
    if not isinstance(inputs, dict) or not isinstance(backbones, dict):
        raise ScoreContractError("label-free bank lacks input identities")
    # Deliberately do not open source CSVs, eligibility, sampler report, coordinate
    # audit, source protocol, or processed-meta files here. Their provenance is
    # sealed by the bank manifest hash and none is needed by the GPU scorer.
    verified = {
        name: _manifest_asset(inputs.get(name), label=name)
        for name in (
            "config",
            "protocol_document",
            "scorer_source",
            "report_source",
        )
    }
    for name in (
        "eligibility_manifest",
        "source_sampler_protocol",
        "source_sampler_report",
        "source_coordinate_audit",
    ):
        verified[name] = _sealed_manifest_asset(inputs.get(name), label=name)
    arm_spec = backbones.get(arm)
    frozen_backbone = _sealed_manifest_asset(
        arm_spec, label=f"{arm} frozen backbone checkpoint"
    )
    verified["backbone_checkpoint"] = _asset(
        docking_checkpoint,
        expected_docking_checkpoint_sha256,
        label=f"{arm} backbone checkpoint",
    )
    verified["confidence_checkpoint"] = _asset(
        confidence_checkpoint,
        expected_confidence_checkpoint_sha256,
        label="confidence checkpoint",
    )
    frozen_confidence = _sealed_manifest_asset(
        inputs.get("confidence_checkpoint"), label="frozen confidence checkpoint"
    )
    identity_checks = (
        (verified["backbone_checkpoint"], frozen_backbone, "backbone checkpoint"),
        (verified["confidence_checkpoint"], frozen_confidence, "confidence checkpoint"),
    )
    for actual, frozen, label in identity_checks:
        if actual["path"] != frozen["path"] or actual["sha256"] != frozen["sha256"]:
            raise ScoreContractError(f"CLI/frozen {label} identities differ")
    exact_pins = (
        (verified["protocol_document"]["sha256"], expected_protocol_sha256, "protocol"),
        (
            verified["scorer_source"]["sha256"],
            expected_scorer_source_sha256,
            "scorer source",
        ),
        (
            verified["report_source"]["sha256"],
            expected_report_source_sha256,
            "report source",
        ),
    )
    for observed, expected, label in exact_pins:
        if observed != _require_sha256(expected, label=f"expected {label} SHA-256"):
            raise ScoreContractError(f"{label} SHA-256 differs from CLI pin")
    if verified["scorer_source"]["path"] != str(Path(__file__).resolve()):
        raise ScoreContractError("scorer source path differs from the running scorer")
    runtime_frozen = inputs.get("runtime_code_identity")
    runtime_actual = _runtime_code_identity()
    expected_runtime = _require_sha256(
        expected_runtime_code_sha256, label="expected runtime code SHA-256"
    )
    if (
        not isinstance(runtime_frozen, dict)
        or runtime_frozen != runtime_actual
        or runtime_actual["aggregate_sha256"] != expected_runtime
    ):
        raise ScoreContractError("runtime code inventory changed after bank freeze")
    verified["runtime_code_identity"] = runtime_actual
    return verified


def _prepare_complex(
    record: dict[str, Any],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, Any], list[torch.Tensor]]:
    sample_id = str(record["sample_key"])
    receptor = _manifest_asset(record.get("receptor"), label=f"{sample_id}.receptor")
    processed_meta = _sealed_manifest_asset(
        record.get("processed_meta"), label=f"{sample_id}.processed_meta"
    )
    if receptor["sha256"] != record.get("receptor_sha256"):
        raise ScoreContractError(f"{sample_id}: receptor identity aliases differ")
    if processed_meta["sha256"] != record.get("processed_meta_sha256"):
        raise ScoreContractError(f"{sample_id}: processed-meta identity aliases differ")
    smiles = str(record.get("canonical_smiles", ""))
    if _canonical_smiles_identity(smiles) != record.get("ligand_input_identity_sha256"):
        raise ScoreContractError(f"{sample_id}: canonical SMILES identity changed")
    if _strict_int(
        record.get("ligand_conformer_seed"), label=f"{sample_id}.ligand_conformer_seed"
    ) != 0:
        raise ScoreContractError(f"{sample_id}: ligand conformer seed must be zero")
    ligand, _ = load_benchmark_ligand(smiles, random_seed=0)
    if ligand.GetNumAtoms() != _strict_int(
        record.get("num_input_atoms"), label=f"{sample_id}.num_input_atoms"
    ):
        raise ScoreContractError(f"{sample_id}: regenerated ligand atom count mismatch")
    center = torch.tensor(record["pocket_center"], dtype=torch.float32)
    graph, ligand_data, meta = preprocess_complex(
        Path(receptor["path"]),
        ligand,
        pocket_center=center,
        pocket_cutoff=10.0,
    )
    if not torch.equal(meta["pocket_center"].cpu(), center):
        raise ScoreContractError(f"{sample_id}: preprocessing changed the pocket center")
    poses = read_saved_bank_poses(record, ligand=ligand)
    return graph, ligand_data, meta, poses


def _score_one_record(
    record: dict[str, Any],
    *,
    confidence_model: torch.nn.Module,
    docking_model: torch.nn.Module,
    device: torch.device,
    score_fn: Callable[..., list[dict[str, float]]] = score_poses_with_confidence,
) -> dict[str, Any]:
    graph, ligand_data, meta, poses = _prepare_complex(record)
    scores = _score_in_chunks(
        confidence_model,
        docking_model,
        graph,
        ligand_data,
        meta,
        poses,
        device=device,
        score_fn=score_fn,
    )
    arrays = _score_arrays(scores)
    result = {
        **{key: record[key] for key in BANK_RECORD_FIELDS},
        "score_arrays": arrays,
        "selected_index": _selected_index(arrays),
        "score_ledger_sha256": score_ledger_sha256(arrays),
    }
    _require_exact_keys(result, SCORE_RECORD_FIELDS, label=f"{record['sample_key']} score")
    return result


def _replay_delta(
    first: dict[str, Any], second: dict[str, Any]
) -> dict[str, Any]:
    per_field: dict[str, float] = {}
    for field in CONFIDENCE_SCORE_FIELDS:
        left = first["score_arrays"][field]
        right = second["score_arrays"][field]
        if len(left) != EXPECTED_POSES or len(right) != EXPECTED_POSES:
            raise ScoreContractError("smoke replay score inventory mismatch")
        per_field[field] = max(abs(a - b) for a, b in zip(left, right, strict=True))
    selected_stable = first["selected_index"] == second["selected_index"]
    if not selected_stable:
        raise ScoreContractError("smoke replay changed the selected pose index")
    max_delta = max(per_field.values())
    if max_delta > REPLAY_ABS_TOLERANCE:
        raise ScoreContractError(
            f"smoke replay score delta {max_delta} exceeds {REPLAY_ABS_TOLERANCE}"
        )
    return {
        "passed": True,
        "checked_count": 1,
        "selected_index_mismatches": 0,
        "all_scores_finite": True,
        "per_field_max_abs_score_delta": per_field,
        "max_abs_score_delta": max_delta,
        "absolute_tolerance": REPLAY_ABS_TOLERANCE,
        "records": [
            {
                "sample_key": first["sample_key"],
                "first_selected_index": first["selected_index"],
                "replay_selected_index": second["selected_index"],
                "selected_index_stable": True,
                "replay_score_ledger_sha256": second["score_ledger_sha256"],
            }
        ],
    }


def score_shard(
    *,
    bank_manifest: Path,
    expected_bank_manifest_sha256: str,
    docking_checkpoint: Path,
    expected_docking_checkpoint_sha256: str,
    confidence_checkpoint: Path,
    expected_confidence_checkpoint_sha256: str,
    expected_protocol_sha256: str,
    expected_scorer_source_sha256: str,
    expected_report_source_sha256: str,
    expected_runtime_code_sha256: str,
    scores_root: Path,
    stage: str,
    arm: str,
    shard_index: int,
    num_shards: int,
    device: torch.device,
    smoke_replay_id: str | None = None,
) -> tuple[Path, dict[str, Any]]:
    if stage not in {"smoke", "full"}:
        raise ValueError("stage must be smoke or full")
    if arm not in ARM_NAMES:
        raise ValueError(f"arm must be one of {ARM_NAMES}")
    if num_shards != EXPECTED_SOURCE_SHARDS or not 0 <= shard_index < num_shards:
        raise ScoreContractError(
            f"score-shard requires {EXPECTED_SOURCE_SHARDS} source shards"
        )
    if (stage == "smoke") != (smoke_replay_id is not None):
        raise ScoreContractError("smoke requires --smoke-replay-id; full forbids it")
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ScoreContractError("score-shard requires an allocated CUDA GPU")
    if torch.cuda.device_count() != 1:
        raise ScoreContractError("score-shard requires exactly one visible CUDA GPU")

    bank_path = _canonical_existing_path(bank_manifest, label="label-free bank manifest")
    manifest = _read_json(bank_path, label="label-free bank manifest")
    _validate_score_bank_manifest(
        manifest,
        expected_manifest_sha256=expected_bank_manifest_sha256,
        manifest_path=bank_path,
    )
    assets = _verify_score_stage_assets(
        manifest,
        arm=arm,
        docking_checkpoint=docking_checkpoint,
        expected_docking_checkpoint_sha256=expected_docking_checkpoint_sha256,
        confidence_checkpoint=confidence_checkpoint,
        expected_confidence_checkpoint_sha256=expected_confidence_checkpoint_sha256,
        expected_protocol_sha256=expected_protocol_sha256,
        expected_scorer_source_sha256=expected_scorer_source_sha256,
        expected_report_source_sha256=expected_report_source_sha256,
        expected_runtime_code_sha256=expected_runtime_code_sha256,
    )
    inventory = manifest.get("inventory")
    if (
        not isinstance(inventory, dict)
        or inventory.get("eligible_count") != EXPECTED_ELIGIBLE_COUNT
        or inventory.get("source_shard_count") != EXPECTED_SOURCE_SHARDS
    ):
        raise ScoreContractError("label-free bank inventory mismatch")
    all_records = manifest.get("records")
    if not isinstance(all_records, list) or len(all_records) != EXPECTED_ELIGIBLE_COUNT:
        raise ScoreContractError("label-free bank record count mismatch")
    records = [
        record
        for record in all_records
        if isinstance(record, dict) and record.get("source_shard_index") == shard_index
    ]
    expected_count = len(range(shard_index, EXPECTED_ELIGIBLE_COUNT, num_shards))
    if len(records) != expected_count or len(
        {str(record.get("sample_key")) for record in records}
    ) != expected_count:
        raise ScoreContractError("assigned score-shard inventory mismatch")
    if stage == "smoke":
        matches = [record for record in records if record.get("sample_key") == smoke_replay_id]
        if len(matches) != 1:
            raise ScoreContractError("smoke replay ID must occur once in the assigned shard")
        records = matches

    canonical_scores_root = _canonical_existing_path(
        scores_root, label="scores root", directory=True
    )
    output = (
        canonical_scores_root
        / stage
        / f"shard-{shard_index:03d}-of-{num_shards:03d}"
        / "arms"
        / arm
        / "scores.json"
    )
    if output.exists():
        raise FileExistsError(f"refusing to overwrite score shard: {output}")

    docking_model, _, docking_ckpt = load_model(
        Path(assets["config"]["path"]),
        Path(assets["backbone_checkpoint"]["path"]),
        device,
    )
    confidence_model, confidence_ckpt = load_pose_confidence_model(
        Path(assets["confidence_checkpoint"]["path"]), device
    )
    if arm == "s50_backbone":
        expected_docking = {
            "artifact_type": "effdock_ema_inference_checkpoint",
            "inference_only": True,
            "weight_source": "ema",
            "source_checkpoint_step": 50000,
            "step": 50000,
        }
        if any(docking_ckpt.get(key) != value for key, value in expected_docking.items()):
            raise ScoreContractError("loaded S50 backbone checkpoint metadata mismatch")
    elif int(docking_ckpt.get("step", -1)) != 100000:
        raise ScoreContractError("loaded matched backbone checkpoint step mismatch")
    if (
        confidence_ckpt.get("model_type") != "docking_graph_pose_confidence"
        or int(confidence_ckpt.get("step", -1)) != 42500
    ):
        raise ScoreContractError("loaded confidence checkpoint metadata mismatch")

    started = time.monotonic()
    scored: list[dict[str, Any]] = []
    replay: dict[str, Any] = {}
    for index, record in enumerate(records, start=1):
        result = _score_one_record(
            record,
            confidence_model=confidence_model,
            docking_model=docking_model,
            device=device,
        )
        scored.append(result)
        if stage == "smoke":
            repeated = _score_one_record(
                record,
                confidence_model=confidence_model,
                docking_model=docking_model,
                device=device,
            )
            replay = _replay_delta(result, repeated)
        print(f"[{index}/{len(records)}] scored {arm} {record['sample_key']}", flush=True)

    assigned_ids = [str(record["sample_key"]) for record in records]
    result = {
        "schema_version": SCORE_SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": "complete",
        "mode": "smoke_replay" if stage == "smoke" else "full_shard",
        "stage": stage,
        "arm": arm,
        "arm_role": (
            "primary_deployment_backbone"
            if arm == "s50_backbone"
            else "diagnostic_training_matched_backbone"
        ),
        "selector": "stable_argmin_confidence_rmsd",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "fixed_settings": {
            "saved_pose_bank_only": True,
            "resampling": False,
            "sample_sigma": EXPECTED_SIGMA,
            "pose_count": EXPECTED_POSES,
            "num_steps": EXPECTED_STEPS,
            "prior_pool_size": EXPECTED_PRIOR_POOL_SIZE,
            "pose_batch_size": POSE_BATCH_SIZE,
            "t1_hidden_backbone": arm,
            "pocket_cutoff_angstrom": 10.0,
            "selector": "stable_argmin_confidence_rmsd",
            "label_blind": True,
        },
        "inputs": {
            "label_free_bank_manifest": str(bank_path),
            "label_free_bank_manifest_sha256": file_sha256(bank_path),
            "eligibility_manifest_sha256": assets["eligibility_manifest"]["sha256"],
            "protocol_sha256": assets["protocol_document"]["sha256"],
            "source_sampler_report_sha256": assets["source_sampler_report"]["sha256"],
            "source_coordinate_audit_sha256": assets["source_coordinate_audit"]["sha256"],
            "source_sampler_protocol_sha256": assets["source_sampler_protocol"]["sha256"],
            "scorer_source_sha256": assets["scorer_source"]["sha256"],
            "report_source_sha256": assets["report_source"]["sha256"],
            "config_sha256": assets["config"]["sha256"],
            "backbone_checkpoint": assets["backbone_checkpoint"]["path"],
            "backbone_checkpoint_sha256": assets["backbone_checkpoint"]["sha256"],
            "confidence_checkpoint": assets["confidence_checkpoint"]["path"],
            "confidence_checkpoint_sha256": assets["confidence_checkpoint"]["sha256"],
            "runtime_code_identity_sha256": assets["runtime_code_identity"][
                "aggregate_sha256"
            ],
        },
        "inventory": {
            "eligible_count": EXPECTED_ELIGIBLE_COUNT,
            "source_shard_count": num_shards,
            "source_shard_index": shard_index,
            "assigned_count": len(records),
            "scored_count": len(scored),
            "assigned_ids_sha256": sorted_id_sha256(assigned_ids),
        },
        "records": scored,
        "replay": replay,
        "runtime": {
            "elapsed_seconds": time.monotonic() - started,
            "finished_at_utc": datetime.now(UTC).isoformat(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "cuda_device_name": torch.cuda.get_device_name(device),
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "rdkit_version": rdBase.rdkitVersion,
        },
    }
    _atomic_write_new_json(output, result)
    return output, result


def _add_asset_pin(
    parser: argparse.ArgumentParser,
    name: str,
    *,
    dest: str | None = None,
) -> None:
    destination = dest or name.replace("-", "_")
    parser.add_argument(f"--{name}", dest=destination, type=Path, required=True)
    parser.add_argument(
        f"--expected-{name}-sha256",
        dest=f"expected_{destination}_sha256",
        required=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    freeze = subparsers.add_parser(
        "freeze-inputs", help="project the source bank into a label-free frozen manifest"
    )
    freeze.add_argument("--bank-root", type=Path, required=True)
    _add_asset_pin(freeze, "eligibility-manifest")
    _add_asset_pin(freeze, "config")
    _add_asset_pin(freeze, "s50-backbone-checkpoint")
    _add_asset_pin(freeze, "matched-backbone-checkpoint")
    _add_asset_pin(freeze, "confidence-checkpoint")
    _add_asset_pin(freeze, "source-sampler-protocol")
    _add_asset_pin(freeze, "protocol", dest="protocol_document")
    _add_asset_pin(freeze, "source-sampler-report")
    _add_asset_pin(freeze, "source-coordinate-audit")
    _add_asset_pin(freeze, "report-source")
    freeze.add_argument("--expected-scorer-source-sha256", required=True)
    freeze.add_argument("--expected-runtime-code-sha256", required=True)
    freeze.add_argument("--output", type=Path, required=True)

    score = subparsers.add_parser(
        "score-shard", help="score one frozen source shard without opening outcome CSVs"
    )
    score.add_argument("--bank-manifest", type=Path, required=True)
    score.add_argument("--expected-bank-manifest-sha256", required=True)
    _add_asset_pin(score, "docking-checkpoint")
    _add_asset_pin(score, "confidence-checkpoint")
    score.add_argument("--expected-protocol-sha256", required=True)
    score.add_argument("--expected-scorer-source-sha256", required=True)
    score.add_argument("--expected-report-source-sha256", required=True)
    score.add_argument("--expected-runtime-code-sha256", required=True)
    score.add_argument("--scores-root", type=Path, required=True)
    score.add_argument("--stage", choices=("smoke", "full"), required=True)
    score.add_argument("--arm", choices=ARM_NAMES, required=True)
    score.add_argument("--shard-index", type=int, required=True)
    score.add_argument("--num-shards", type=int, default=EXPECTED_SOURCE_SHARDS)
    score.add_argument("--pose-batch-size", type=int, default=POSE_BATCH_SIZE)
    score.add_argument("--sigma", type=float, default=EXPECTED_SIGMA)
    score.add_argument("--smoke-replay-id")
    score.add_argument("--device", default="cuda")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "freeze-inputs":
        manifest = freeze_label_free_inputs(
            bank_root=args.bank_root,
            eligibility_manifest=args.eligibility_manifest,
            expected_eligibility_manifest_sha256=args.expected_eligibility_manifest_sha256,
            config=args.config,
            expected_config_sha256=args.expected_config_sha256,
            s50_backbone_checkpoint=args.s50_backbone_checkpoint,
            expected_s50_backbone_checkpoint_sha256=(
                args.expected_s50_backbone_checkpoint_sha256
            ),
            matched_backbone_checkpoint=args.matched_backbone_checkpoint,
            expected_matched_backbone_checkpoint_sha256=(
                args.expected_matched_backbone_checkpoint_sha256
            ),
            confidence_checkpoint=args.confidence_checkpoint,
            expected_confidence_checkpoint_sha256=args.expected_confidence_checkpoint_sha256,
            source_sampler_protocol=args.source_sampler_protocol,
            expected_source_sampler_protocol_sha256=(
                args.expected_source_sampler_protocol_sha256
            ),
            protocol_document=args.protocol_document,
            expected_protocol_sha256=args.expected_protocol_document_sha256,
            source_sampler_report=args.source_sampler_report,
            expected_source_sampler_report_sha256=(
                args.expected_source_sampler_report_sha256
            ),
            source_coordinate_audit=args.source_coordinate_audit,
            expected_source_coordinate_audit_sha256=(
                args.expected_source_coordinate_audit_sha256
            ),
            report_source=args.report_source,
            expected_report_source_sha256=args.expected_report_source_sha256,
            expected_scorer_source_sha256=args.expected_scorer_source_sha256,
            expected_runtime_code_sha256=args.expected_runtime_code_sha256,
            output=args.output,
        )
        print(
            json.dumps(
                {
                    "output": str(args.output.resolve()),
                    "sha256": file_sha256(args.output),
                    "records": len(manifest["records"]),
                },
                sort_keys=True,
            )
        )
        return
    if args.command == "score-shard":
        if args.device != "cuda":
            raise ScoreContractError("the frozen scorer requires --device cuda")
        if args.pose_batch_size != POSE_BATCH_SIZE or not math.isclose(
            args.sigma, EXPECTED_SIGMA, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ScoreContractError(
                f"score-shard requires --pose-batch-size {POSE_BATCH_SIZE} "
                f"and --sigma {EXPECTED_SIGMA}"
            )
        output, result = score_shard(
            bank_manifest=args.bank_manifest,
            expected_bank_manifest_sha256=args.expected_bank_manifest_sha256,
            docking_checkpoint=args.docking_checkpoint,
            expected_docking_checkpoint_sha256=(
                args.expected_docking_checkpoint_sha256
            ),
            confidence_checkpoint=args.confidence_checkpoint,
            expected_confidence_checkpoint_sha256=(
                args.expected_confidence_checkpoint_sha256
            ),
            expected_protocol_sha256=args.expected_protocol_sha256,
            expected_scorer_source_sha256=args.expected_scorer_source_sha256,
            expected_report_source_sha256=args.expected_report_source_sha256,
            expected_runtime_code_sha256=args.expected_runtime_code_sha256,
            scores_root=args.scores_root,
            stage=args.stage,
            arm=args.arm,
            shard_index=args.shard_index,
            num_shards=args.num_shards,
            device=torch.device("cuda"),
            smoke_replay_id=args.smoke_replay_id,
        )
        print(
            json.dumps(
                {
                    "output": str(output),
                    "sha256": file_sha256(output),
                    "records": result["inventory"]["scored_count"],
                },
                sort_keys=True,
            )
        )
        return
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    main()
