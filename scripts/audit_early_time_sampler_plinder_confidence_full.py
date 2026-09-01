#!/usr/bin/env python3
"""Independent full-result audit for the frozen S50 confidence experiment.

This program deliberately does not import the production report aggregator.  It
validates the complete score inventory first, then opens the frozen source CSVs
and independently reconstructs the label join, metrics, and clustered
bootstrap.  It is full-stage only; there is no outcome-blind smoke mode here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from rdkit import Chem

PROTOCOL_ID = "EFFDOCK-S50-CONFIDENCE-SCORE-ONLY-PLINDER-V1"
SOURCE_PROTOCOL_ID = "EFFDOCK-EARLY-TIME-SAMPLER-PLINDER-K2-GATE-V1"
AUDIT_SCHEMA_VERSION = "effdock.early_time_sampler_s50_confidence_audit.v1"
STRICT_REPORT_SCHEMA_VERSION = "effdock.early_time_sampler_plinder_confidence_report.v1"
BANK_SCHEMA_VERSION = "effdock.early_time_sampler_s50_confidence_bank.v1"
SCORE_SCHEMA_VERSION = "effdock.early_time_sampler_s50_confidence_scores.v1"
ELIGIBILITY_SCHEMA_VERSION = "effdock.plinder_checkpoint_eligibility.v1"

ARMS = ("s50_backbone", "matched_backbone")
PRIMARY_ARM = "s50_backbone"
DIAGNOSTIC_ARM = "matched_backbone"
SOURCE_ARM = "s50_ema"
SELECTOR = "stable_argmin_confidence_rmsd"
BOOTSTRAP_SEED = 20260815
REPORT_COMPARE_ABS_TOLERANCE = 1e-12


@dataclass(frozen=True)
class AuditContract:
    full_count: int = 1076
    eligible_count: int = 1035
    eligible_system_count: int = 1020
    excluded_count: int = 41
    shard_count: int = 8
    pose_count: int = 100
    sample_sigma: float = 2.0
    num_steps: int = 10
    prior_pool_size: int = 100
    bootstrap_seed: int = BOOTSTRAP_SEED
    bootstrap_resamples: int = 20_000
    bootstrap_batch_size: int = 256


@dataclass(frozen=True)
class FrozenIdentities:
    eligibility_manifest_sha256: str
    config_sha256: str
    s50_backbone_sha256: str
    matched_backbone_sha256: str
    confidence_sha256: str
    source_protocol_sha256: str
    source_report_sha256: str
    source_audit_sha256: str


PRODUCTION_CONTRACT = AuditContract()
PRODUCTION_IDENTITIES = FrozenIdentities(
    eligibility_manifest_sha256=(
        "6ebeb2d165e1def6ebf7b5bba301f82d4a9c3ff9d6c5cd43616dcf09edbd38ac"
    ),
    config_sha256=(
        "39aa62e4a48ed6f3aa4ff59345fb43a81220e2baba22edfd5beb0c4981b307ec"
    ),
    s50_backbone_sha256=(
        "65be44d7dc8f0867eb9fc5d22214b80f93971ea4702679a527c665046e91e6b6"
    ),
    matched_backbone_sha256=(
        "6932fb3ba6ebac770f714453529656a44b8f33cf15119d23c9e675d2d60b36db"
    ),
    confidence_sha256=(
        "e31fde6f351284205c78f7a1510002779c43312e94d9f82003d47a14d72bc78f"
    ),
    source_protocol_sha256=(
        "0250853ae0793db288be2a6a8dc775db391d25aae32835b65b061782f34ab518"
    ),
    source_report_sha256=(
        "d4814796a9d274f836888dd614e5b6a4a5fba6b86001da83bea6720fabf02316"
    ),
    source_audit_sha256=(
        "3b6daa4a3d4c74ae384e7c3d2199d3d26f9360fe4b64a33e1c6ab16f4b83eabc"
    ),
)

SCORE_ARRAY_FIELDS = {
    "confidence_rmsd",
    "confidence_success_logit",
    "confidence_success",
    "confidence_atom_rmsd",
    "confidence_atom_q90",
    "confidence_atom_ok",
}
BANK_RECORD_FIELDS = {
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
}
SCORE_RECORD_FIELDS = BANK_RECORD_FIELDS | {
    "score_arrays",
    "selected_index",
    "score_ledger_sha256",
}
BANK_TOP_LEVEL_FIELDS = {
    "schema_version",
    "protocol_id",
    "source_protocol_id",
    "status",
    "created_at_utc",
    "information_boundary",
    "bank_root",
    "inputs",
    "backbone_arms",
    "fixed_settings",
    "inventory",
    "source_shards",
    "records",
}
BANK_INPUT_FIELDS = {
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
    "source_output_root",
    "runtime_code_identity",
}
SCORE_TOP_LEVEL_FIELDS = {
    "schema_version",
    "protocol_id",
    "status",
    "mode",
    "stage",
    "arm",
    "arm_role",
    "selector",
    "created_at_utc",
    "fixed_settings",
    "inputs",
    "inventory",
    "records",
    "replay",
    "runtime",
}
SCORE_INPUT_FIELDS = {
    "label_free_bank_manifest",
    "label_free_bank_manifest_sha256",
    "eligibility_manifest_sha256",
    "protocol_sha256",
    "source_sampler_report_sha256",
    "source_coordinate_audit_sha256",
    "source_sampler_protocol_sha256",
    "scorer_source_sha256",
    "report_source_sha256",
    "config_sha256",
    "backbone_checkpoint",
    "backbone_checkpoint_sha256",
    "confidence_checkpoint",
    "confidence_checkpoint_sha256",
    "runtime_code_identity_sha256",
}
SCORE_RUNTIME_FIELDS = {
    "elapsed_seconds",
    "finished_at_utc",
    "slurm_job_id",
    "slurm_array_job_id",
    "slurm_array_task_id",
    "cuda_device_name",
    "torch_version",
    "torch_cuda_version",
    "rdkit_version",
}
SOURCE_SHARD_FIELDS = {
    "shard_index",
    "paired_summary",
    "results_csv",
    "assigned_count",
    "assigned_ids_sha256",
}
SOURCE_REQUIRED_FIELDS = {
    "id",
    "arm",
    "plinder_system_id",
    "candidate_ensemble_sha256",
    "all_poses_sdf",
    "all_poses_sdf_sha256",
    "all_poses_count",
    "num_samples",
    "checkpoint_sha256",
    "candidate_rmsds_json",
    "num_rmsd_lt2_candidates",
}
SOURCE_ROW_ALLOWLIST = {
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
}
RUNTIME_CODE_DOMAIN = b"EFFDOCK_S50_CONFIDENCE_RUNTIME_CODE_V1\0"
SCORE_LEDGER_DOMAIN = b"EFFDOCK_S50_CONFIDENCE_SCORE_LEDGER_V1\0"


class AuditError(RuntimeError):
    """Raised when any frozen identity, inventory, or result check fails."""


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _score_ledger_sha256(arrays: Mapping[str, Sequence[float]]) -> str:
    digest = hashlib.sha256(SCORE_LEDGER_DOMAIN)
    digest.update(_canonical_json_bytes(arrays))
    return digest.hexdigest()


def _ids_sha256(ids: Sequence[str]) -> str:
    digest = hashlib.sha256(b"EFFDOCK_SORTED_COMPLEX_IDS_V1\0")
    for sample_id in ids:
        digest.update(sample_id.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _canonical_smiles_identity(smiles: str) -> str:
    digest = hashlib.sha256(b"EFFDOCK_PLINDER_CANONICAL_SMILES_V1\0")
    digest.update(smiles.encode("utf-8"))
    return digest.hexdigest()


def _require_sha256(value: object, *, label: str) -> str:
    if not (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    ):
        raise AuditError(f"{label}: expected a lowercase SHA-256 digest")
    return value


def _require_exact_fields(
    value: Mapping[str, object], expected: set[str], *, label: str
) -> None:
    observed = set(value)
    if observed != expected:
        raise AuditError(
            f"{label}: exact field inventory mismatch; "
            f"missing={sorted(expected - observed)}, extra={sorted(observed - expected)}"
        )


def _strict_int(value: object, *, label: str) -> int:
    if isinstance(value, bool):
        raise AuditError(f"{label}: boolean is not an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        return int(value)
    raise AuditError(f"{label}: expected an integer, got {value!r}")


def _finite_float(value: object, *, label: str) -> float:
    if isinstance(value, bool):
        raise AuditError(f"{label}: boolean is not numeric")
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise AuditError(f"{label}: expected a finite number, got {value!r}") from exc
    if not math.isfinite(result):
        raise AuditError(f"{label}: expected a finite number, got {value!r}")
    return result


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AuditError(f"{label}: failed to load {path}") from exc
    if not isinstance(payload, dict):
        raise AuditError(f"{label}: expected a JSON object")
    return payload


def _canonical_existing(path: Path, *, label: str, directory: bool = False) -> Path:
    if not path.is_absolute():
        raise AuditError(f"{label}: path must be absolute")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise AuditError(f"{label}: missing path {path}") from exc
    if resolved != path:
        raise AuditError(f"{label}: path must be lexical-canonical without symlinks")
    if directory and not path.is_dir():
        raise AuditError(f"{label}: expected a directory")
    if not directory and not path.is_file():
        raise AuditError(f"{label}: expected a file")
    return path


def _path_spec(value: object, *, label: str, rehash: bool = False) -> tuple[Path, str]:
    if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
        raise AuditError(f"{label}: expected exact {{path, sha256}} asset")
    raw_path = value.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise AuditError(f"{label}.path: expected a non-empty string")
    path = _canonical_existing(Path(raw_path), label=label)
    expected = _require_sha256(value.get("sha256"), label=f"{label}.sha256")
    if rehash:
        observed = _file_sha256(path)
        if observed != expected:
            raise AuditError(
                f"{label}: SHA-256 mismatch; expected={expected}, observed={observed}"
            )
    return path, expected


def _require_within(path: Path, root: Path, *, label: str) -> None:
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise AuditError(f"{label}: path escapes frozen root {root}") from exc


def _validate_eligibility(
    path: Path, *, contract: AuditContract
) -> tuple[list[str], list[str], list[str]]:
    payload = _load_json(path, label="eligibility manifest")
    if (
        payload.get("schema_version") != ELIGIBILITY_SCHEMA_VERSION
        or payload.get("protocol_id") != SOURCE_PROTOCOL_ID
        or payload.get("status") != "complete"
    ):
        raise AuditError("eligibility manifest identity/status mismatch")
    inventory = payload.get("inventory")
    if not isinstance(inventory, dict):
        raise AuditError("eligibility manifest lacks inventory")
    expected_counts = {
        "full_count": contract.full_count,
        "eligible_count": contract.eligible_count,
        "eligible_system_count": contract.eligible_system_count,
        "excluded_count": contract.excluded_count,
        "preflight_error_count": 0,
    }
    for key, expected in expected_counts.items():
        if _strict_int(inventory.get(key), label=f"eligibility.{key}") != expected:
            raise AuditError(f"eligibility.{key}: expected {expected}")
    if inventory.get("preflight_error_ids") != []:
        raise AuditError("eligibility manifest contains preflight failures")
    output: list[list[str]] = []
    for key, count in (
        ("full_ids", contract.full_count),
        ("eligible_ids", contract.eligible_count),
        ("excluded_ids", contract.excluded_count),
    ):
        values = inventory.get(key)
        if not (
            isinstance(values, list)
            and len(values) == count
            and all(isinstance(item, str) and item for item in values)
            and values == sorted(values)
            and len(set(values)) == count
        ):
            raise AuditError(f"eligibility.{key}: invalid ordered exact inventory")
        if inventory.get(f"{key}_sha256") != _ids_sha256(values):
            raise AuditError(f"eligibility.{key}_sha256 mismatch")
        output.append(list(values))
    full_ids, eligible_ids, excluded_ids = output
    if sorted(eligible_ids + excluded_ids) != full_ids:
        raise AuditError("eligible/excluded IDs do not partition the full cohort")
    return full_ids, eligible_ids, excluded_ids


def _validate_runtime_identity(value: object) -> str:
    if not isinstance(value, dict):
        raise AuditError("bank runtime_code_identity must be an object")
    _require_exact_fields(
        value, {"aggregate_sha256", "files"}, label="runtime_code_identity"
    )
    declared_aggregate = _require_sha256(
        value.get("aggregate_sha256"), label="runtime aggregate"
    )
    files = value.get("files")
    if not isinstance(files, dict) or not files:
        raise AuditError("runtime code file inventory is empty")
    hashes: dict[str, str] = {}
    for name, raw_spec in files.items():
        if not isinstance(name, str) or not name or not isinstance(raw_spec, dict):
            raise AuditError("runtime code file inventory is malformed")
        _require_exact_fields(
            raw_spec,
            {"path", "sha256", "size_bytes"},
            label=f"runtime file {name}",
        )
        path, expected = _path_spec(
            {"path": raw_spec.get("path"), "sha256": raw_spec.get("sha256")},
            label=f"runtime file {name}",
            rehash=True,
        )
        if _strict_int(raw_spec.get("size_bytes"), label=f"runtime {name}.size") != (
            path.stat().st_size
        ):
            raise AuditError(f"runtime file {name}: size changed")
        hashes[name] = expected
    digest = hashlib.sha256(RUNTIME_CODE_DOMAIN)
    digest.update(_canonical_json_bytes(hashes))
    if digest.hexdigest() != declared_aggregate:
        raise AuditError("runtime aggregate SHA-256 recomputation mismatch")
    return declared_aggregate


def _validate_bank_record(
    record: dict[str, Any],
    *,
    sample_id: str,
    global_order: int,
    source_root: Path,
    contract: AuditContract,
) -> None:
    _require_exact_fields(record, BANK_RECORD_FIELDS, label=f"bank record {sample_id}")
    if record.get("sample_key") != sample_id:
        raise AuditError(f"bank record {global_order}: sample/order mismatch")
    if not isinstance(record.get("system_id"), str) or not record["system_id"]:
        raise AuditError(f"{sample_id}: missing system_id")
    if not isinstance(record.get("ligand_chain"), str) or not record["ligand_chain"]:
        raise AuditError(f"{sample_id}: missing ligand_chain")
    exact_ints = {
        "source_shard_index": global_order % contract.shard_count,
        "pose_count": contract.pose_count,
        "num_steps": contract.num_steps,
        "prior_pool_size": contract.prior_pool_size,
        "ligand_conformer_seed": 0,
    }
    for key, expected in exact_ints.items():
        if _strict_int(record.get(key), label=f"{sample_id}.{key}") != expected:
            raise AuditError(f"{sample_id}.{key}: expected {expected}")
    for key in ("plinder_global_index", "sampling_seed", "num_input_atoms"):
        if _strict_int(record.get(key), label=f"{sample_id}.{key}") < 0:
            raise AuditError(f"{sample_id}.{key}: negative value")
    sigma = _finite_float(record.get("sample_sigma"), label=f"{sample_id}.sample_sigma")
    if not math.isclose(sigma, contract.sample_sigma, rel_tol=0.0, abs_tol=1e-12):
        raise AuditError(f"{sample_id}: sample sigma mismatch")
    for key in (
        "candidate_ensemble_sha256",
        "prior_pool_sha256",
        "receptor_sha256",
        "processed_meta_sha256",
        "ligand_input_identity_sha256",
    ):
        _require_sha256(record.get(key), label=f"{sample_id}.{key}")
    smiles = record.get("canonical_smiles")
    if not isinstance(smiles, str) or not smiles:
        raise AuditError(f"{sample_id}: canonical_smiles is empty")
    if _canonical_smiles_identity(smiles) != record["ligand_input_identity_sha256"]:
        raise AuditError(f"{sample_id}: canonical SMILES identity mismatch")
    center = record.get("pocket_center")
    if not isinstance(center, list) or len(center) != 3:
        raise AuditError(f"{sample_id}: pocket_center must have length three")
    for index, coordinate in enumerate(center):
        _finite_float(coordinate, label=f"{sample_id}.pocket_center[{index}]")
    for asset_name, scalar_name in (
        ("receptor", "receptor_sha256"),
        ("processed_meta", "processed_meta_sha256"),
    ):
        _, asset_sha = _path_spec(
            record.get(asset_name), label=f"{sample_id}.{asset_name}", rehash=True
        )
        if asset_sha != record[scalar_name]:
            raise AuditError(f"{sample_id}: {asset_name} hash aliases differ")
    sdf_path, _ = _path_spec(
        record.get("all_poses_sdf"), label=f"{sample_id}.all_poses_sdf"
    )
    expected_sdf = (
        source_root
        / f"shard-{global_order % contract.shard_count:03d}-of-{contract.shard_count:03d}"
        / "arms"
        / SOURCE_ARM
        / "poses"
        / "all_poses"
        / f"{sample_id}.sdf"
    )
    if sdf_path != expected_sdf:
        raise AuditError(f"{sample_id}: non-canonical frozen SDF path")
    _require_within(sdf_path, source_root, label=f"{sample_id}.SDF")


def _validate_bank(
    payload: dict[str, Any],
    *,
    manifest_path: Path,
    manifest_sha256: str,
    protocol_file: Path,
    protocol_sha256: str,
    contract: AuditContract,
    identities: FrozenIdentities,
) -> tuple[
    list[str],
    dict[str, dict[str, Any]],
    list[dict[str, Any]],
    dict[str, str],
    dict[str, Path],
    Path,
]:
    _require_exact_fields(payload, BANK_TOP_LEVEL_FIELDS, label="bank manifest")
    if (
        payload.get("schema_version") != BANK_SCHEMA_VERSION
        or payload.get("protocol_id") != PROTOCOL_ID
        or payload.get("source_protocol_id") != SOURCE_PROTOCOL_ID
        or payload.get("status") != "complete_label_free"
    ):
        raise AuditError("bank manifest identity/status mismatch")
    if _file_sha256(manifest_path) != manifest_sha256:
        raise AuditError("bank manifest SHA-256 changed during audit")
    inputs = payload.get("inputs")
    if not isinstance(inputs, dict):
        raise AuditError("bank manifest inputs are missing")
    _require_exact_fields(inputs, BANK_INPUT_FIELDS, label="bank inputs")

    asset_names = (
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
    assets: dict[str, tuple[Path, str]] = {
        name: _path_spec(inputs.get(name), label=f"bank.inputs.{name}", rehash=True)
        for name in asset_names
    }
    expected_asset_hashes = {
        "eligibility_manifest": identities.eligibility_manifest_sha256,
        "config": identities.config_sha256,
        "s50_backbone_checkpoint": identities.s50_backbone_sha256,
        "matched_backbone_checkpoint": identities.matched_backbone_sha256,
        "confidence_checkpoint": identities.confidence_sha256,
        "source_sampler_protocol": identities.source_protocol_sha256,
        "source_sampler_report": identities.source_report_sha256,
        "source_coordinate_audit": identities.source_audit_sha256,
        "protocol_document": protocol_sha256,
    }
    for name, expected in expected_asset_hashes.items():
        if assets[name][1] != _require_sha256(expected, label=f"frozen {name}"):
            raise AuditError(f"bank frozen input mismatch: {name}")
    if assets["protocol_document"][0] != protocol_file:
        raise AuditError("bank protocol path differs from explicit audit pin")
    runtime_sha = _validate_runtime_identity(inputs.get("runtime_code_identity"))

    source_root_raw = inputs.get("source_output_root")
    if not isinstance(source_root_raw, str):
        raise AuditError("bank source_output_root is missing")
    source_root = _canonical_existing(
        Path(source_root_raw), label="source output root", directory=True
    )
    if payload.get("bank_root") != str(source_root):
        raise AuditError("bank_root differs from source_output_root")

    fixed = payload.get("fixed_settings")
    expected_fixed: dict[str, object] = {
        "source_arm": SOURCE_ARM,
        "pose_count": contract.pose_count,
        "sample_sigma": contract.sample_sigma,
        "num_steps": contract.num_steps,
        "prior_pool_size": contract.prior_pool_size,
        "pose_batch_size": 20,
        "pocket_cutoff_angstrom": 10.0,
        "ligand_conformer_seed": 0,
        "selector": SELECTOR,
        "label_blind": True,
        "resampling": False,
    }
    if not isinstance(fixed, dict):
        raise AuditError("bank fixed_settings are missing")
    _require_exact_fields(fixed, set(expected_fixed), label="bank fixed_settings")
    if any(fixed.get(key) != expected for key, expected in expected_fixed.items()):
        raise AuditError("bank fixed_settings differ from the audit contract")
    boundary = payload.get("information_boundary")
    if not isinstance(boundary, dict):
        raise AuditError("bank information boundary is missing")
    _require_exact_fields(
        boundary,
        {
            "source_csv_allowlist",
            "outcome_columns_exported",
            "score_stage_reads_source_results_csv",
            "crystal_reference_exported",
        },
        label="bank information boundary",
    )
    for key in (
        "outcome_columns_exported",
        "score_stage_reads_source_results_csv",
        "crystal_reference_exported",
    ):
        if boundary.get(key) is not False:
            raise AuditError(f"bank information boundary is open: {key}")
    declared_allowlist = boundary.get("source_csv_allowlist")
    if not (
        isinstance(declared_allowlist, list)
        and len(declared_allowlist) == len(SOURCE_ROW_ALLOWLIST)
        and set(declared_allowlist) == SOURCE_ROW_ALLOWLIST
    ):
        raise AuditError("bank source CSV allowlist mismatch")

    eligibility_path = assets["eligibility_manifest"][0]
    full_ids, eligible_ids, excluded_ids = _validate_eligibility(
        eligibility_path, contract=contract
    )
    inventory = payload.get("inventory")
    if not isinstance(inventory, dict):
        raise AuditError("bank inventory is missing")
    expected_inventory = {
        "full_count": contract.full_count,
        "eligible_count": contract.eligible_count,
        "eligible_system_count": contract.eligible_system_count,
        "excluded_count": contract.excluded_count,
        "source_shard_count": contract.shard_count,
        "pose_count": contract.pose_count,
        "full_ids_sha256": _ids_sha256(full_ids),
        "eligible_ids_sha256": _ids_sha256(eligible_ids),
        "excluded_ids_sha256": _ids_sha256(excluded_ids),
    }
    _require_exact_fields(inventory, set(expected_inventory), label="bank inventory")
    if any(inventory.get(key) != value for key, value in expected_inventory.items()):
        raise AuditError("bank inventory differs from eligibility/contract")

    backbones = payload.get("backbone_arms")
    if not isinstance(backbones, dict):
        raise AuditError("bank backbone_arms are missing")
    _require_exact_fields(backbones, set(ARMS), label="bank backbone_arms")
    arm_assets: dict[str, Path] = {}
    for arm, input_name, role in (
        (PRIMARY_ARM, "s50_backbone_checkpoint", "primary_deployment_backbone"),
        (
            DIAGNOSTIC_ARM,
            "matched_backbone_checkpoint",
            "diagnostic_training_matched_backbone",
        ),
    ):
        spec = backbones.get(arm)
        if not isinstance(spec, dict):
            raise AuditError(f"bank backbone arm {arm} is missing")
        _require_exact_fields(spec, {"path", "sha256", "role"}, label=f"backbone {arm}")
        path, sha = _path_spec(
            {"path": spec.get("path"), "sha256": spec.get("sha256")},
            label=f"backbone {arm}",
            rehash=True,
        )
        if (path, sha) != assets[input_name] or spec.get("role") != role:
            raise AuditError(f"bank backbone arm binding mismatch: {arm}")
        arm_assets[arm] = path

    records = payload.get("records")
    if not isinstance(records, list) or len(records) != contract.eligible_count:
        raise AuditError("bank does not contain exactly the eligible cohort")
    by_id: dict[str, dict[str, Any]] = {}
    for index, (record, sample_id) in enumerate(zip(records, eligible_ids, strict=True)):
        if not isinstance(record, dict):
            raise AuditError(f"bank record {index} is not an object")
        _validate_bank_record(
            record,
            sample_id=sample_id,
            global_order=index,
            source_root=source_root,
            contract=contract,
        )
        if sample_id in by_id:
            raise AuditError(f"duplicate bank sample {sample_id}")
        by_id[sample_id] = record
    if len({str(row["system_id"]) for row in records}) != contract.eligible_system_count:
        raise AuditError("bank eligible system count mismatch")

    source_shards = payload.get("source_shards")
    if not isinstance(source_shards, list) or len(source_shards) != contract.shard_count:
        raise AuditError("bank source shard inventory is incomplete")
    seen: set[int] = set()
    for raw_shard in source_shards:
        if not isinstance(raw_shard, dict):
            raise AuditError("bank source shard entry is not an object")
        _require_exact_fields(raw_shard, SOURCE_SHARD_FIELDS, label="bank source shard")
        shard = _strict_int(raw_shard.get("shard_index"), label="source shard index")
        if shard in seen or not 0 <= shard < contract.shard_count:
            raise AuditError(f"duplicate/out-of-range source shard {shard}")
        seen.add(shard)
        ids = eligible_ids[shard :: contract.shard_count]
        if (
            _strict_int(raw_shard.get("assigned_count"), label="source assigned_count")
            != len(ids)
            or raw_shard.get("assigned_ids_sha256") != _ids_sha256(ids)
        ):
            raise AuditError(f"source shard {shard}: assigned inventory mismatch")
        paired_path, _ = _path_spec(
            raw_shard.get("paired_summary"),
            label=f"source shard {shard} paired summary",
            rehash=True,
        )
        csv_path, _ = _path_spec(
            raw_shard.get("results_csv"), label=f"source shard {shard} results CSV"
        )
        shard_root = source_root / f"shard-{shard:03d}-of-{contract.shard_count:03d}"
        if paired_path != shard_root / "paired_summary.json":
            raise AuditError(f"source shard {shard}: paired summary path mismatch")
        if csv_path != shard_root / "arms" / SOURCE_ARM / "results.csv":
            raise AuditError(f"source shard {shard}: result CSV path mismatch")
        _require_within(csv_path, source_root, label=f"source shard {shard} CSV")
    if seen != set(range(contract.shard_count)):
        raise AuditError("source shard index inventory mismatch")

    provenance = {
        "eligibility_manifest_sha256": assets["eligibility_manifest"][1],
        "protocol_sha256": assets["protocol_document"][1],
        "source_sampler_report_sha256": assets["source_sampler_report"][1],
        "source_coordinate_audit_sha256": assets["source_coordinate_audit"][1],
        "source_sampler_protocol_sha256": assets["source_sampler_protocol"][1],
        "scorer_source_sha256": assets["scorer_source"][1],
        "report_source_sha256": assets["report_source"][1],
        "config_sha256": assets["config"][1],
        "confidence_checkpoint_sha256": assets["confidence_checkpoint"][1],
        "runtime_code_identity_sha256": runtime_sha,
    }
    all_paths = {
        **arm_assets,
        "confidence_checkpoint": assets["confidence_checkpoint"][0],
    }
    return eligible_ids, by_id, source_shards, provenance, all_paths, source_root


def _score_path(root: Path, shard: int, arm: str, contract: AuditContract) -> Path:
    return (
        root
        / "full"
        / f"shard-{shard:03d}-of-{contract.shard_count:03d}"
        / "arms"
        / arm
        / "scores.json"
    )


def _validate_score_record(
    record: dict[str, Any],
    *,
    bank: dict[str, Any],
    sample_id: str,
    arm: str,
    pose_count: int,
) -> dict[str, Any]:
    _require_exact_fields(record, SCORE_RECORD_FIELDS, label=f"score {sample_id}/{arm}")
    for key in BANK_RECORD_FIELDS:
        if record.get(key) != bank.get(key):
            raise AuditError(f"{sample_id}/{arm}: score/bank identity mismatch for {key}")
    arrays = record.get("score_arrays")
    if not isinstance(arrays, dict):
        raise AuditError(f"{sample_id}/{arm}: score_arrays must be an object")
    _require_exact_fields(arrays, SCORE_ARRAY_FIELDS, label=f"{sample_id}/{arm}.arrays")
    normalized: dict[str, list[float]] = {}
    for field in sorted(SCORE_ARRAY_FIELDS):
        values = arrays.get(field)
        if not isinstance(values, list) or len(values) != pose_count:
            raise AuditError(f"{sample_id}/{arm}.{field}: expected {pose_count} values")
        normalized[field] = [
            _finite_float(value, label=f"{sample_id}/{arm}.{field}[{index}]")
            for index, value in enumerate(values)
        ]
    for field in ("confidence_rmsd", "confidence_atom_rmsd", "confidence_atom_q90"):
        if any(value < 0.0 for value in normalized[field]):
            raise AuditError(f"{sample_id}/{arm}.{field}: negative prediction")
    for field in ("confidence_success", "confidence_atom_ok"):
        if any(not 0.0 <= value <= 1.0 for value in normalized[field]):
            raise AuditError(f"{sample_id}/{arm}.{field}: probability outside [0,1]")
    if record.get("score_ledger_sha256") != _score_ledger_sha256(normalized):
        raise AuditError(f"{sample_id}/{arm}: score ledger SHA-256 mismatch")
    argmin_index = min(
        range(pose_count), key=lambda index: (normalized["confidence_rmsd"][index], index)
    )
    declared = _strict_int(record.get("selected_index"), label=f"{sample_id}.selected_index")
    if declared != argmin_index:
        raise AuditError(f"{sample_id}/{arm}: declared selector index != stable argmin")
    success_argmax = max(
        range(pose_count),
        key=lambda index: (normalized["confidence_success"][index], -index),
    )
    logit_argmax = max(
        range(pose_count),
        key=lambda index: (normalized["confidence_success_logit"][index], -index),
    )
    if success_argmax != logit_argmax:
        raise AuditError(f"{sample_id}/{arm}: stable probability/logit argmax mismatch")
    return {
        **record,
        "score_arrays": normalized,
        "selected_index": argmin_index,
        "success_head_index": success_argmax,
    }


def _load_scores(
    *,
    scores_root: Path,
    manifest_path: Path,
    manifest_sha256: str,
    eligible_ids: list[str],
    bank_by_id: dict[str, dict[str, Any]],
    provenance: dict[str, str],
    checkpoint_paths: dict[str, Path],
    contract: AuditContract,
) -> tuple[dict[str, dict[str, dict[str, Any]]], list[dict[str, Any]]]:
    by_arm: dict[str, dict[str, dict[str, Any]]] = {arm: {} for arm in ARMS}
    artifacts: list[dict[str, Any]] = []
    for shard in range(contract.shard_count):
        ids = eligible_ids[shard :: contract.shard_count]
        for arm in ARMS:
            path = _score_path(scores_root, shard, arm, contract)
            _canonical_existing(path, label=f"score shard {shard}/{arm}")
            _require_within(path, scores_root, label=f"score shard {shard}/{arm}")
            payload = _load_json(path, label=f"score shard {shard}/{arm}")
            _require_exact_fields(payload, SCORE_TOP_LEVEL_FIELDS, label=str(path))
            expected_role = (
                "primary_deployment_backbone"
                if arm == PRIMARY_ARM
                else "diagnostic_training_matched_backbone"
            )
            expected_top = {
                "schema_version": SCORE_SCHEMA_VERSION,
                "protocol_id": PROTOCOL_ID,
                "status": "complete",
                "mode": "full_shard",
                "stage": "full",
                "arm": arm,
                "arm_role": expected_role,
                "selector": SELECTOR,
            }
            if any(payload.get(key) != value for key, value in expected_top.items()):
                raise AuditError(f"score shard {shard}/{arm}: top-level identity mismatch")
            if payload.get("replay") != {}:
                raise AuditError(f"score shard {shard}/{arm}: full artifact contains replay")

            inputs = payload.get("inputs")
            if not isinstance(inputs, dict):
                raise AuditError(f"score shard {shard}/{arm}: inputs missing")
            _require_exact_fields(inputs, SCORE_INPUT_FIELDS, label=f"{path}.inputs")
            expected_inputs: dict[str, object] = {
                "label_free_bank_manifest": str(manifest_path),
                "label_free_bank_manifest_sha256": manifest_sha256,
                **provenance,
                "backbone_checkpoint": str(checkpoint_paths[arm]),
                "backbone_checkpoint_sha256": _file_sha256(checkpoint_paths[arm]),
                "confidence_checkpoint": str(checkpoint_paths["confidence_checkpoint"]),
            }
            for key, expected in expected_inputs.items():
                if inputs.get(key) != expected:
                    raise AuditError(f"score shard {shard}/{arm}: input mismatch for {key}")
            if inputs.get("confidence_checkpoint_sha256") != provenance[
                "confidence_checkpoint_sha256"
            ]:
                raise AuditError(f"score shard {shard}/{arm}: confidence hash mismatch")

            fixed = payload.get("fixed_settings")
            expected_fixed: dict[str, object] = {
                "saved_pose_bank_only": True,
                "resampling": False,
                "sample_sigma": contract.sample_sigma,
                "pose_count": contract.pose_count,
                "num_steps": contract.num_steps,
                "prior_pool_size": contract.prior_pool_size,
                "pose_batch_size": 20,
                "t1_hidden_backbone": arm,
                "pocket_cutoff_angstrom": 10.0,
                "selector": SELECTOR,
                "label_blind": True,
            }
            if not isinstance(fixed, dict):
                raise AuditError(f"score shard {shard}/{arm}: fixed settings missing")
            _require_exact_fields(fixed, set(expected_fixed), label=f"{path}.fixed")
            if any(fixed.get(key) != value for key, value in expected_fixed.items()):
                raise AuditError(f"score shard {shard}/{arm}: fixed settings mismatch")

            inventory = payload.get("inventory")
            expected_inventory = {
                "eligible_count": contract.eligible_count,
                "source_shard_count": contract.shard_count,
                "source_shard_index": shard,
                "assigned_count": len(ids),
                "scored_count": len(ids),
                "assigned_ids_sha256": _ids_sha256(ids),
            }
            if not isinstance(inventory, dict):
                raise AuditError(f"score shard {shard}/{arm}: inventory missing")
            _require_exact_fields(inventory, set(expected_inventory), label=f"{path}.inventory")
            if any(inventory.get(key) != value for key, value in expected_inventory.items()):
                raise AuditError(f"score shard {shard}/{arm}: exact inventory mismatch")

            runtime = payload.get("runtime")
            if not isinstance(runtime, dict):
                raise AuditError(f"score shard {shard}/{arm}: runtime missing")
            _require_exact_fields(runtime, SCORE_RUNTIME_FIELDS, label=f"{path}.runtime")
            if _finite_float(runtime.get("elapsed_seconds"), label=f"{path}.elapsed") < 0:
                raise AuditError(f"score shard {shard}/{arm}: negative runtime")
            if str(runtime.get("slurm_array_task_id")) != str(shard):
                raise AuditError(f"score shard {shard}/{arm}: Slurm task mismatch")

            records = payload.get("records")
            if not isinstance(records, list) or len(records) != len(ids):
                raise AuditError(f"score shard {shard}/{arm}: record count mismatch")
            for raw_record, sample_id in zip(records, ids, strict=True):
                if not isinstance(raw_record, dict):
                    raise AuditError(f"{sample_id}/{arm}: score record is not an object")
                record = _validate_score_record(
                    raw_record,
                    bank=bank_by_id[sample_id],
                    sample_id=sample_id,
                    arm=arm,
                    pose_count=contract.pose_count,
                )
                if sample_id in by_arm[arm]:
                    raise AuditError(f"duplicate score record {sample_id}/{arm}")
                by_arm[arm][sample_id] = record
            artifacts.append(
                {
                    "path": str(path),
                    "sha256": _file_sha256(path),
                    "shard_index": shard,
                    "arm": arm,
                    "record_count": len(records),
                }
            )
    expected = set(eligible_ids)
    for arm in ARMS:
        if set(by_arm[arm]) != expected:
            raise AuditError(f"{arm}: score record coverage is not exact")
    if len(artifacts) != contract.shard_count * len(ARMS):
        raise AuditError("score artifact inventory is not exactly 8 shards x 2 arms")
    return by_arm, artifacts


def _audit_sdf_order(
    sample_id: str, bank: dict[str, Any], *, contract: AuditContract
) -> dict[str, Any]:
    path, expected_sha = _path_spec(
        bank.get("all_poses_sdf"), label=f"{sample_id}.SDF"
    )
    before = _file_sha256(path)
    if before != expected_sha:
        raise AuditError(f"{sample_id}: source SDF hash mismatch")
    count = 0
    with path.open("rb") as handle:
        supplier = Chem.ForwardSDMolSupplier(
            handle, removeHs=False, sanitize=False, strictParsing=True
        )
        for pose_index, molecule in enumerate(supplier):
            if molecule is None:
                raise AuditError(f"{sample_id}: SDF record {pose_index} is unparsable")
            expected_properties = {
                "sample_index": str(pose_index),
                "complex_id": sample_id,
                "candidate_ensemble_sha256": str(bank["candidate_ensemble_sha256"]),
            }
            for name, expected in expected_properties.items():
                if not molecule.HasProp(name) or molecule.GetProp(name) != expected:
                    raise AuditError(
                        f"{sample_id}: SDF record {pose_index} property {name} mismatch"
                    )
            if not molecule.HasProp("sample_sigma") or not math.isclose(
                _finite_float(
                    molecule.GetProp("sample_sigma"),
                    label=f"{sample_id}.SDF[{pose_index}].sigma",
                ),
                contract.sample_sigma,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise AuditError(f"{sample_id}: SDF sample sigma mismatch")
            count += 1
    if count != contract.pose_count:
        raise AuditError(f"{sample_id}: SDF record count {count} != {contract.pose_count}")
    after = _file_sha256(path)
    if after != before:
        raise AuditError(f"{sample_id}: SDF changed during audit")
    return {"path": str(path), "sha256": after, "record_count": count}


def _parse_rmsds(value: object, *, sample_id: str, pose_count: int) -> np.ndarray:
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError as exc:
        raise AuditError(f"{sample_id}: invalid candidate_rmsds_json") from exc
    if not isinstance(parsed, list) or len(parsed) != pose_count:
        raise AuditError(f"{sample_id}: candidate RMSD count mismatch")
    values = np.asarray(
        [
            _finite_float(item, label=f"{sample_id}.candidate_rmsds[{index}]")
            for index, item in enumerate(parsed)
        ],
        dtype=np.float64,
    )
    if bool(np.any(values < 0.0)):
        raise AuditError(f"{sample_id}: negative source RMSD")
    return values


def _read_source_labels(
    *,
    source_shards: list[dict[str, Any]],
    eligible_ids: list[str],
    bank_by_id: dict[str, dict[str, Any]],
    source_root: Path,
    identities: FrozenIdentities,
    contract: AuditContract,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], list[dict[str, Any]]]:
    labels: dict[str, np.ndarray] = {}
    csv_artifacts: list[dict[str, Any]] = []
    sdf_artifacts: list[dict[str, Any]] = []
    by_shard = {
        _strict_int(value.get("shard_index"), label="source shard index"): value
        for value in source_shards
    }
    for shard in range(contract.shard_count):
        spec = by_shard[shard]
        path, expected_sha = _path_spec(
            spec.get("results_csv"), label=f"source shard {shard} CSV"
        )
        expected_path = (
            source_root
            / f"shard-{shard:03d}-of-{contract.shard_count:03d}"
            / "arms"
            / SOURCE_ARM
            / "results.csv"
        )
        if path != expected_path:
            raise AuditError(f"source shard {shard}: CSV path changed")
        before = _file_sha256(path)
        if before != expected_sha:
            raise AuditError(f"source shard {shard}: CSV SHA-256 mismatch")
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or not SOURCE_REQUIRED_FIELDS <= set(reader.fieldnames):
                raise AuditError(f"source shard {shard}: required source columns missing")
            rows = list(reader)
        ids = eligible_ids[shard :: contract.shard_count]
        if [str(row.get("id", "")) for row in rows] != ids:
            raise AuditError(f"source shard {shard}: exact CSV row order mismatch")
        if spec.get("assigned_ids_sha256") != _ids_sha256(ids):
            raise AuditError(f"source shard {shard}: assigned ID hash mismatch")
        for row, sample_id in zip(rows, ids, strict=True):
            bank = bank_by_id[sample_id]
            if row.get("arm") != SOURCE_ARM:
                raise AuditError(f"{sample_id}: wrong frozen source arm")
            if row.get("plinder_system_id") != bank["system_id"]:
                raise AuditError(f"{sample_id}: source system identity mismatch")
            if row.get("candidate_ensemble_sha256") != bank["candidate_ensemble_sha256"]:
                raise AuditError(f"{sample_id}: candidate ensemble identity mismatch")
            bank_sdf, bank_sdf_sha = _path_spec(
                bank.get("all_poses_sdf"), label=f"{sample_id}.bank SDF"
            )
            if (
                row.get("all_poses_sdf") != str(bank_sdf)
                or row.get("all_poses_sdf_sha256") != bank_sdf_sha
            ):
                raise AuditError(f"{sample_id}: source CSV/SDF identity mismatch")
            for field in ("all_poses_count", "num_samples"):
                if _strict_int(row.get(field), label=f"{sample_id}.{field}") != (
                    contract.pose_count
                ):
                    raise AuditError(f"{sample_id}: source pose count mismatch")
            if row.get("checkpoint_sha256") != identities.s50_backbone_sha256:
                raise AuditError(f"{sample_id}: source checkpoint identity mismatch")
            rmsds = _parse_rmsds(
                row.get("candidate_rmsds_json"),
                sample_id=sample_id,
                pose_count=contract.pose_count,
            )
            declared_k2 = _strict_int(
                row.get("num_rmsd_lt2_candidates"), label=f"{sample_id}.declared_k2"
            )
            if declared_k2 != int(np.sum(rmsds < 2.0)):
                raise AuditError(f"{sample_id}: declared K2 disagrees with RMSD ledger")
            if sample_id in labels:
                raise AuditError(f"duplicate source label row {sample_id}")
            labels[sample_id] = rmsds
            sdf_artifacts.append(_audit_sdf_order(sample_id, bank, contract=contract))
        after = _file_sha256(path)
        if after != before:
            raise AuditError(f"source shard {shard}: CSV changed while being read")
        csv_artifacts.append(
            {
                "path": str(path),
                "sha256": after,
                "shard_index": shard,
                "row_count": len(rows),
                "assigned_ids_sha256": _ids_sha256(ids),
            }
        )
    if set(labels) != set(eligible_ids):
        raise AuditError("source labels do not cover the exact eligible cohort")
    return labels, csv_artifacts, sdf_artifacts


def _join_rows(
    scores: dict[str, dict[str, dict[str, Any]]],
    labels: dict[str, np.ndarray],
    eligible_ids: list[str],
    *,
    pose_count: int,
) -> dict[str, list[dict[str, Any]]]:
    joined: dict[str, list[dict[str, Any]]] = {arm: [] for arm in ARMS}
    for arm in ARMS:
        for sample_id in eligible_ids:
            score = scores[arm][sample_id]
            rmsds = labels[sample_id]
            predicted = np.asarray(
                score["score_arrays"]["confidence_rmsd"], dtype=np.float64
            )
            order = np.asarray(
                sorted(range(pose_count), key=lambda index: (predicted[index], index)),
                dtype=np.int64,
            )
            selected = int(score["selected_index"])
            success = rmsds < 2.0
            k2 = int(np.sum(success))
            joined[arm].append(
                {
                    "sample_key": sample_id,
                    "system_id": str(score["system_id"]),
                    "selected_index": selected,
                    "success_head_index": int(score["success_head_index"]),
                    "selected_success": bool(success[selected]),
                    "oracle_success": bool(np.any(success)),
                    "first_success": bool(success[0]),
                    "random_success_expectation": k2 / pose_count,
                    "top5_success": bool(np.any(success[order[: min(5, pose_count)]])),
                }
            )
    return joined


def _pct(value: float, denominator: int) -> float:
    return 100.0 * float(value) / denominator


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    selected = sum(bool(row["selected_success"]) for row in rows)
    oracle = sum(bool(row["oracle_success"]) for row in rows)
    first = sum(bool(row["first_success"]) for row in rows)
    random_sum = sum(float(row["random_success_expectation"]) for row in rows)
    top5 = sum(bool(row["top5_success"]) for row in rows)
    return {
        "sample_count": n,
        "system_count": len({str(row["system_id"]) for row in rows}),
        "top1_success_count": selected,
        "top1_success_pct": _pct(selected, n),
        "oracle_success_count": oracle,
        "oracle_success_pct": _pct(oracle, n),
        "oracle_recovery_fraction": selected / oracle,
        "oracle_gap_count": oracle - selected,
        "oracle_gap_pp": _pct(oracle - selected, n),
        "first_success_count": first,
        "first_success_pct": _pct(first, n),
        "random_expected_success_pct": _pct(random_sum, n),
        "delta_vs_first_pp": _pct(selected - first, n),
        "delta_vs_random_pp": _pct(selected - random_sum, n),
        "top5_success_count": top5,
        "top5_success_pct": _pct(top5, n),
        "top5_rescue_count": top5 - selected,
        "top5_rescue_pp": _pct(top5 - selected, n),
    }


def _percentile(values: np.ndarray) -> dict[str, float]:
    lower, upper = np.percentile(values, [2.5, 97.5])
    return {"lower": float(lower), "upper": float(upper)}


def _cluster_bootstrap(
    rows_by_arm: dict[str, list[dict[str, Any]]], *, contract: AuditContract
) -> tuple[dict[str, dict[str, dict[str, float]]], dict[str, Any]]:
    systems = sorted({str(row["system_id"]) for row in rows_by_arm[PRIMARY_ARM]})
    if len(systems) != contract.eligible_system_count:
        raise AuditError("bootstrap system inventory mismatch")
    system_index = {system: index for index, system in enumerate(systems)}
    matrices: dict[str, np.ndarray] = {}
    for arm in ARMS:
        matrix = np.zeros((len(systems), 6), dtype=np.float64)
        for row in rows_by_arm[arm]:
            index = system_index[str(row["system_id"])]
            matrix[index] += np.asarray(
                (
                    1.0,
                    float(row["selected_success"]),
                    float(row["oracle_success"]),
                    float(row["first_success"]),
                    float(row["random_success_expectation"]),
                    float(row["top5_success"]) - float(row["selected_success"]),
                ),
                dtype=np.float64,
            )
        matrices[arm] = matrix
    collected: dict[str, dict[str, list[np.ndarray]]] = {
        arm: {
            name: []
            for name in (
                "top1_success_pct",
                "oracle_recovery_fraction",
                "oracle_gap_pp",
                "delta_vs_first_pp",
                "delta_vs_random_pp",
                "top5_rescue_pp",
            )
        }
        for arm in ARMS
    }
    paired: list[np.ndarray] = []
    rng = np.random.Generator(np.random.PCG64(contract.bootstrap_seed))
    remaining = contract.bootstrap_resamples
    while remaining:
        batch = min(contract.bootstrap_batch_size, remaining)
        draws = rng.integers(0, len(systems), size=(batch, len(systems)))
        totals = {arm: matrices[arm][draws].sum(axis=1) for arm in ARMS}
        for arm in ARMS:
            values = totals[arm]
            count = values[:, 0]
            selected = values[:, 1]
            oracle = values[:, 2]
            derived = {
                "top1_success_pct": 100.0 * selected / count,
                "oracle_recovery_fraction": selected / oracle,
                "oracle_gap_pp": 100.0 * (oracle - selected) / count,
                "delta_vs_first_pp": 100.0 * (selected - values[:, 3]) / count,
                "delta_vs_random_pp": 100.0 * (selected - values[:, 4]) / count,
                "top5_rescue_pp": 100.0 * values[:, 5] / count,
            }
            if any(not bool(np.all(np.isfinite(value))) for value in derived.values()):
                raise AuditError("bootstrap produced a non-finite statistic")
            for name, value in derived.items():
                collected[arm][name].append(value)
        paired.append(
            100.0
            * (totals[PRIMARY_ARM][:, 1] - totals[DIAGNOSTIC_ARM][:, 1])
            / totals[PRIMARY_ARM][:, 0]
        )
        remaining -= batch
    intervals = {
        arm: {
            name: _percentile(np.concatenate(parts))
            for name, parts in collected[arm].items()
        }
        for arm in ARMS
    }
    primary_selected = sum(
        bool(row["selected_success"]) for row in rows_by_arm[PRIMARY_ARM]
    )
    diagnostic_selected = sum(
        bool(row["selected_success"]) for row in rows_by_arm[DIAGNOSTIC_ARM]
    )
    point = _pct(primary_selected - diagnostic_selected, len(rows_by_arm[PRIMARY_ARM]))
    return intervals, {
        "point_pp": point,
        "ci95": _percentile(np.concatenate(paired)),
    }


def _nested_get(value: Mapping[str, Any], path: Sequence[str]) -> object:
    current: object = value
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            raise AuditError(f"strict report comparison missing {'.'.join(path)}")
        current = current[key]
    return current


def _compare_strict_report(
    strict: dict[str, Any], independent: dict[str, Any]
) -> dict[str, Any]:
    if (
        strict.get("schema_version") != STRICT_REPORT_SCHEMA_VERSION
        or strict.get("protocol_id") != PROTOCOL_ID
        or strict.get("stage") != "full"
        or strict.get("status") != "complete_diagnostic"
    ):
        raise AuditError("strict report identity/status mismatch")
    metric_names = (
        "sample_count",
        "system_count",
        "top1_success_count",
        "top1_success_pct",
        "oracle_success_count",
        "oracle_success_pct",
        "oracle_recovery_fraction",
        "oracle_gap_count",
        "oracle_gap_pp",
        "first_success_count",
        "first_success_pct",
        "random_expected_success_pct",
        "delta_vs_first_pp",
        "delta_vs_random_pp",
        "top5_success_count",
        "top5_success_pct",
        "top5_rescue_count",
        "top5_rescue_pp",
    )
    paths: list[tuple[str, ...]] = []
    for arm in ARMS:
        paths.extend(("arms", arm, name) for name in metric_names)
        for name in (
            "top1_success_pct",
            "oracle_recovery_fraction",
            "oracle_gap_pp",
            "delta_vs_first_pp",
            "delta_vs_random_pp",
            "top5_rescue_pp",
        ):
            paths.extend(
                ("arms", arm, "cluster_bootstrap_ci95", name, bound)
                for bound in ("lower", "upper")
            )
    paths.extend(
        [
            ("paired_backbone", "point_pp"),
            ("paired_backbone", "ci95", "lower"),
            ("paired_backbone", "ci95", "upper"),
            ("paired_backbone", "selected_index_agreement_count"),
            ("paired_backbone", "selected_index_agreement_pct"),
        ]
    )
    max_delta = 0.0
    for path in paths:
        observed = _nested_get(strict, path)
        expected = _nested_get(independent, path)
        if isinstance(expected, bool) or not isinstance(expected, int | float):
            if observed != expected:
                raise AuditError(f"strict report mismatch at {'.'.join(path)}")
            continue
        observed_number = _finite_float(observed, label=f"strict.{'.'.join(path)}")
        delta = abs(observed_number - float(expected))
        max_delta = max(max_delta, delta)
        if delta > REPORT_COMPARE_ABS_TOLERANCE:
            raise AuditError(
                f"strict report numeric mismatch at {'.'.join(path)}: "
                f"delta={delta} > {REPORT_COMPARE_ABS_TOLERANCE}"
            )
    return {
        "passed": True,
        "numeric_absolute_tolerance": REPORT_COMPARE_ABS_TOLERANCE,
        "fields_checked": len(paths),
        "maximum_absolute_delta": max_delta,
    }


def audit_full(
    *,
    scores_root: Path,
    label_free_bank_manifest: Path,
    label_free_bank_manifest_sha256: str,
    protocol_file: Path,
    protocol_sha256: str,
    strict_report: Path | None = None,
    strict_report_sha256: str | None = None,
    contract: AuditContract = PRODUCTION_CONTRACT,
    identities: FrozenIdentities = PRODUCTION_IDENTITIES,
) -> dict[str, Any]:
    scores_root = _canonical_existing(scores_root, label="scores root", directory=True)
    manifest_path = _canonical_existing(
        label_free_bank_manifest, label="label-free bank manifest"
    )
    expected_manifest_sha = _require_sha256(
        label_free_bank_manifest_sha256, label="bank manifest CLI SHA-256"
    )
    if _file_sha256(manifest_path) != expected_manifest_sha:
        raise AuditError("bank manifest does not match explicit CLI SHA-256")
    protocol_file = _canonical_existing(protocol_file, label="protocol file")
    expected_protocol_sha = _require_sha256(
        protocol_sha256, label="protocol CLI SHA-256"
    )
    if _file_sha256(protocol_file) != expected_protocol_sha:
        raise AuditError("protocol does not match explicit CLI SHA-256")
    if (strict_report is None) != (strict_report_sha256 is None):
        raise AuditError("strict report path and SHA-256 must be supplied together")

    bank_payload = _load_json(manifest_path, label="label-free bank manifest")
    (
        eligible_ids,
        bank_by_id,
        source_shards,
        provenance,
        checkpoint_paths,
        source_root,
    ) = _validate_bank(
        bank_payload,
        manifest_path=manifest_path,
        manifest_sha256=expected_manifest_sha,
        protocol_file=protocol_file,
        protocol_sha256=expected_protocol_sha,
        contract=contract,
        identities=identities,
    )

    # Outcome boundary: only after all 16 complete score artifacts validate do
    # we open the eight source results CSVs and their RMSD ledgers.
    scores, score_artifacts = _load_scores(
        scores_root=scores_root,
        manifest_path=manifest_path,
        manifest_sha256=expected_manifest_sha,
        eligible_ids=eligible_ids,
        bank_by_id=bank_by_id,
        provenance=provenance,
        checkpoint_paths=checkpoint_paths,
        contract=contract,
    )
    labels, source_csvs, source_sdfs = _read_source_labels(
        source_shards=source_shards,
        eligible_ids=eligible_ids,
        bank_by_id=bank_by_id,
        source_root=source_root,
        identities=identities,
        contract=contract,
    )
    joined = _join_rows(
        scores, labels, eligible_ids, pose_count=contract.pose_count
    )
    arms = {arm: _metrics(joined[arm]) for arm in ARMS}
    intervals, paired = _cluster_bootstrap(joined, contract=contract)
    for arm in ARMS:
        arms[arm]["cluster_bootstrap_ci95"] = intervals[arm]
    agreement = sum(
        int(scores[PRIMARY_ARM][sample_id]["selected_index"])
        == int(scores[DIAGNOSTIC_ARM][sample_id]["selected_index"])
        for sample_id in eligible_ids
    )
    paired.update(
        {
            "contrast": "s50_backbone_minus_matched_backbone_top1_success_pp",
            "matched_minus_s50_point_pp": -float(paired["point_pp"]),
            "selected_index_agreement_count": agreement,
            "selected_index_agreement_pct": _pct(agreement, contract.eligible_count),
        }
    )
    result: dict[str, Any] = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": "passed",
        "stage": "full",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "auditor_source_sha256": _file_sha256(Path(__file__).resolve()),
        "inputs": {
            "scores_root": str(scores_root),
            "label_free_bank_manifest": {
                "path": str(manifest_path),
                "sha256": expected_manifest_sha,
            },
            "protocol": {
                "path": str(protocol_file),
                "sha256": expected_protocol_sha,
            },
            **provenance,
        },
        "configuration": {
            "arms": list(ARMS),
            "selector": SELECTOR,
            "eligible_count": contract.eligible_count,
            "eligible_system_count": contract.eligible_system_count,
            "pose_count": contract.pose_count,
            "source_shard_count": contract.shard_count,
            "score_artifact_count": contract.shard_count * len(ARMS),
            "bootstrap": {
                "unit": "plinder_system_id",
                "sample_weighted": True,
                "generator": "numpy.random.PCG64",
                "seed": contract.bootstrap_seed,
                "resamples": contract.bootstrap_resamples,
                "interval": "percentile_95",
            },
        },
        "integrity": {
            "exact_8_shards_x_2_arms": len(score_artifacts)
            == contract.shard_count * len(ARMS),
            "score_artifacts": score_artifacts,
            "source_csv_artifacts": source_csvs,
            "source_sdf_count": len(source_sdfs),
            "source_sdf_record_count": sum(
                int(value["record_count"]) for value in source_sdfs
            ),
            "finite_score_arrays": sorted(SCORE_ARRAY_FIELDS),
            "stable_argmin_recomputed": True,
            "stable_argmax_recomputed": True,
            "candidate_identity_join": "sample_key+candidate_ensemble_sha256+SDF_path/hash/order",
        },
        "arms": arms,
        "paired_backbone": paired,
    }
    if strict_report is not None and strict_report_sha256 is not None:
        strict_path = _canonical_existing(strict_report, label="strict report")
        expected_strict_sha = _require_sha256(
            strict_report_sha256, label="strict report CLI SHA-256"
        )
        actual_strict_sha = _file_sha256(strict_path)
        if actual_strict_sha != expected_strict_sha:
            raise AuditError("strict report SHA-256 mismatch")
        comparison = _compare_strict_report(
            _load_json(strict_path, label="strict report"), result
        )
        result["inputs"]["strict_report"] = {
            "path": str(strict_path),
            "sha256": actual_strict_sha,
        }
        result["strict_report_comparison"] = comparison
    return result


def _atomic_write_json_noreplace(path: Path, payload: dict[str, Any]) -> None:
    if not path.is_absolute() or path.resolve(strict=False) != path:
        raise AuditError("output path must be absolute and lexical-canonical")
    if path.exists():
        raise FileExistsError(f"refusing to overwrite audit: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores-root", type=Path, required=True)
    parser.add_argument("--label-free-bank-manifest", type=Path, required=True)
    parser.add_argument("--label-free-bank-manifest-sha256", required=True)
    parser.add_argument("--protocol-file", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--strict-report", type=Path)
    parser.add_argument("--strict-report-sha256")
    parser.add_argument("--output-json", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    result = audit_full(
        scores_root=args.scores_root,
        label_free_bank_manifest=args.label_free_bank_manifest,
        label_free_bank_manifest_sha256=args.label_free_bank_manifest_sha256,
        protocol_file=args.protocol_file,
        protocol_sha256=args.protocol_sha256,
        strict_report=args.strict_report,
        strict_report_sha256=args.strict_report_sha256,
    )
    _atomic_write_json_noreplace(args.output_json, result)
    print(f"wrote {args.output_json} status={result['status']} stage={result['stage']}")


if __name__ == "__main__":
    main()
