#!/usr/bin/env python3
"""Strict score-only report for frozen S50 PLINDER confidence ledgers.

The GPU scorer is label blind.  This reporter first verifies and seals the
complete score inventory, then joins it to the already-frozen S50 candidate
labels.  Smoke mode performs the same identity and artifact checks but emits
no efficacy values.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import tempfile
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from rdkit import Chem

PROTOCOL_ID = "EFFDOCK-S50-CONFIDENCE-SCORE-ONLY-PLINDER-V1"
REPORT_SCHEMA_VERSION = "effdock.early_time_sampler_plinder_confidence_report.v1"
BANK_SCHEMA_VERSION = "effdock.early_time_sampler_s50_confidence_bank.v1"
SCORE_SCHEMA_VERSION = "effdock.early_time_sampler_s50_confidence_scores.v1"

SOURCE_PROTOCOL_ID = "EFFDOCK-EARLY-TIME-SAMPLER-PLINDER-K2-GATE-V1"
SOURCE_REPORT_SCHEMA_VERSION = "effdock.early_time_sampler_plinder_k2_gate_report.v1"
SOURCE_AUDIT_SCHEMA_VERSION = "effdock.plinder_checkpoint_full_coordinate_audit.v1"
ELIGIBILITY_SCHEMA_VERSION = "effdock.plinder_checkpoint_eligibility.v1"

ARMS = ("s50_backbone", "matched_backbone")
PRIMARY_ARM = "s50_backbone"
DIAGNOSTIC_ARM = "matched_backbone"
SOURCE_ARM = "s50_ema"
SELECTOR = "stable_argmin_confidence_rmsd"

FROZEN_CONFIDENCE_SHA256 = (
    "e31fde6f351284205c78f7a1510002779c43312e94d9f82003d47a14d72bc78f"
)
FROZEN_BACKBONE_SHA256 = {
    "s50_backbone": "65be44d7dc8f0867eb9fc5d22214b80f93971ea4702679a527c665046e91e6b6",
    "matched_backbone": "6932fb3ba6ebac770f714453529656a44b8f33cf15119d23c9e675d2d60b36db",
}
FROZEN_CONFIG_SHA256 = "39aa62e4a48ed6f3aa4ff59345fb43a81220e2baba22edfd5beb0c4981b307ec"
FROZEN_SOURCE_PROTOCOL_SHA256 = (
    "0250853ae0793db288be2a6a8dc775db391d25aae32835b65b061782f34ab518"
)
FROZEN_ELIGIBILITY_MANIFEST_SHA256 = (
    "6ebeb2d165e1def6ebf7b5bba301f82d4a9c3ff9d6c5cd43616dcf09edbd38ac"
)
FROZEN_SOURCE_REPORT_SHA256 = (
    "d4814796a9d274f836888dd614e5b6a4a5fba6b86001da83bea6720fabf02316"
)
FROZEN_SOURCE_AUDIT_SHA256 = (
    "3b6daa4a3d4c74ae384e7c3d2199d3d26f9360fe4b64a33e1c6ab16f4b83eabc"
)

FULL_SPLIT_COUNT = 1076
FULL_ELIGIBLE_COUNT = 1035
FULL_SYSTEM_COUNT = 1020
FULL_EXCLUDED_COUNT = 41
EXPECTED_SHARDS = 8
POSE_COUNT = 100
SAMPLE_SIGMA = 2.0
PRIOR_POOL_SIZE = 100
NUM_STEPS = 10
SMOKE_COUNT = 8
BOOTSTRAP_SEED = 20260815
BOOTSTRAP_RESAMPLES = 20_000
BOOTSTRAP_BATCH_SIZE = 256
REPLAY_ABS_TOLERANCE = 1e-5

SCORE_ARRAY_FIELDS = (
    "confidence_rmsd",
    "confidence_success_logit",
    "confidence_success",
    "confidence_atom_rmsd",
    "confidence_atom_q90",
    "confidence_atom_ok",
)
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


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")


def _score_ledger_sha256(score_arrays: dict[str, list[float]]) -> str:
    digest = hashlib.sha256()
    digest.update(b"EFFDOCK_S50_CONFIDENCE_SCORE_LEDGER_V1\0")
    digest.update(_canonical_json_bytes(score_arrays))
    return digest.hexdigest()


def _ids_sha256(ids: Sequence[str]) -> str:
    digest = hashlib.sha256()
    digest.update(b"EFFDOCK_SORTED_COMPLEX_IDS_V1\0")
    for sample_id in ids:
        digest.update(sample_id.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _canonical_smiles_identity(smiles: str) -> str:
    digest = hashlib.sha256()
    digest.update(b"EFFDOCK_PLINDER_CANONICAL_SMILES_V1\0")
    digest.update(smiles.encode("utf-8"))
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_sha256(value: object, *, label: str) -> str:
    if not _is_sha256(value):
        raise ValueError(f"{label}: expected a lowercase SHA-256 digest")
    return str(value)


def _require_exact_fields(
    value: dict[str, Any], expected: set[str], *, label: str
) -> None:
    if set(value) != expected:
        missing = sorted(expected - set(value))
        extra = sorted(set(value) - expected)
        raise ValueError(
            f"{label}: exact field inventory mismatch; missing={missing}, extra={extra}"
        )


def _strict_int(value: object, *, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label}: booleans are not integers")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip() and value.strip().lstrip("-").isdigit():
        return int(value)
    raise ValueError(f"{label}: expected an integer, got {value!r}")


def _finite_float(value: object, *, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label}: booleans are not numeric")
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}: expected a finite number, got {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label}: expected a finite number, got {value!r}")
    return result


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label}: could not read JSON object at {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label}: expected a JSON object")
    return payload


def _pin_file(path: Path, expected_sha256: str, *, label: str) -> dict[str, str]:
    expected_sha256 = _require_sha256(expected_sha256, label=f"{label} expected SHA-256")
    if not path.is_file():
        raise FileNotFoundError(f"{label}: missing file {path}")
    actual = _file_sha256(path)
    if actual != expected_sha256:
        raise ValueError(
            f"{label}: SHA-256 mismatch; expected {expected_sha256}, got {actual}"
        )
    return {"path": str(path.resolve()), "sha256": actual}


def _identity_sha(inputs: dict[str, Any], key: str, *, label: str) -> str:
    value = inputs.get(key)
    if isinstance(value, dict):
        value = value.get("sha256")
    return _require_sha256(value, label=f"{label}.{key}")


def _path_spec(value: object, *, label: str) -> tuple[Path, str]:
    if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{label}: expected {{path, sha256}}")
    path_value = value.get("path")
    if not isinstance(path_value, str) or not path_value:
        raise ValueError(f"{label}.path: expected a non-empty string")
    declared = Path(path_value)
    if not declared.is_absolute():
        raise ValueError(f"{label}.path: must be absolute")
    try:
        path = declared.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"{label}.path: missing declared file") from exc
    if declared != path:
        raise ValueError(f"{label}.path: must be lexical-canonical and contain no symlink")
    return path, _require_sha256(value.get("sha256"), label=f"{label}.sha256")


def _require_within(path: Path, root: Path, *, label: str) -> None:
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label}: path escapes frozen root {root}") from exc


def _canonical_existing_directory(path: Path, *, label: str) -> Path:
    if not path.is_absolute():
        raise ValueError(f"{label}: must be absolute")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"{label}: directory does not exist") from exc
    if resolved != path or not path.is_dir():
        raise ValueError(f"{label}: must be a lexical-canonical directory without symlinks")
    return path


def _validate_eligibility(path: Path) -> dict[str, Any]:
    payload = _load_json_object(path, label="eligibility manifest")
    if payload.get("schema_version") != ELIGIBILITY_SCHEMA_VERSION:
        raise ValueError("eligibility manifest: unexpected schema_version")
    if payload.get("protocol_id") != SOURCE_PROTOCOL_ID or payload.get("status") != "complete":
        raise ValueError("eligibility manifest: wrong protocol or incomplete status")
    inventory = payload.get("inventory")
    if not isinstance(inventory, dict):
        raise ValueError("eligibility manifest.inventory: expected an object")
    counts = {
        "full_count": FULL_SPLIT_COUNT,
        "eligible_count": FULL_ELIGIBLE_COUNT,
        "eligible_system_count": FULL_SYSTEM_COUNT,
        "excluded_count": FULL_EXCLUDED_COUNT,
        "preflight_error_count": 0,
    }
    for key, expected in counts.items():
        if _strict_int(inventory.get(key), label=f"eligibility.inventory.{key}") != expected:
            raise ValueError(f"eligibility.inventory.{key}: expected {expected}")
    if inventory.get("preflight_error_ids") != []:
        raise ValueError("eligibility manifest contains preflight errors")
    lists: dict[str, list[str]] = {}
    for name, expected_count in (
        ("full_ids", FULL_SPLIT_COUNT),
        ("eligible_ids", FULL_ELIGIBLE_COUNT),
        ("excluded_ids", FULL_EXCLUDED_COUNT),
    ):
        values = inventory.get(name)
        if (
            not isinstance(values, list)
            or len(values) != expected_count
            or any(not isinstance(item, str) or not item for item in values)
            or values != sorted(values)
            or len(set(values)) != expected_count
        ):
            raise ValueError(f"eligibility.inventory.{name}: invalid exact ID inventory")
        lists[name] = list(values)
        declared = inventory.get(f"{name}_sha256")
        if declared != _ids_sha256(values):
            raise ValueError(f"eligibility.inventory.{name}_sha256: mismatch")
    if sorted(lists["eligible_ids"] + lists["excluded_ids"]) != lists["full_ids"]:
        raise ValueError("eligibility eligible/excluded IDs do not partition full IDs")
    return {
        **lists,
        "eligible_ids_sha256": _ids_sha256(lists["eligible_ids"]),
        "excluded_ids_sha256": _ids_sha256(lists["excluded_ids"]),
    }


def _validate_source_reports(
    source_report: dict[str, Any],
    source_audit: dict[str, Any],
    *,
    eligibility_sha256: str,
) -> None:
    if (
        source_report.get("schema_version") != SOURCE_REPORT_SCHEMA_VERSION
        or source_report.get("protocol_id") != SOURCE_PROTOCOL_ID
        or source_report.get("stage") != "full"
        or not str(source_report.get("status", "")).startswith("complete_")
    ):
        raise ValueError("source sampler report is not a completed full report")
    configuration = source_report.get("configuration")
    integrity = source_report.get("integrity")
    if not isinstance(configuration, dict) or not isinstance(integrity, dict):
        raise ValueError("source sampler report lacks configuration/integrity")
    expected_report_values = {
        "expected_shards": EXPECTED_SHARDS,
        "num_samples": POSE_COUNT,
        "selected_count": FULL_ELIGIBLE_COUNT,
        "system_count": FULL_SYSTEM_COUNT,
    }
    for key, expected in expected_report_values.items():
        if _strict_int(configuration.get(key), label=f"source report.configuration.{key}") != expected:
            raise ValueError(f"source report.configuration.{key}: expected {expected}")
    if integrity.get("exact_arm_inventory") is not True:
        raise ValueError("source sampler report exact inventory did not pass")
    if integrity.get("zero_runtime_failures") is not True:
        raise ValueError("source sampler report contains runtime failures")
    if integrity.get("eligibility_manifest_sha256") != eligibility_sha256:
        raise ValueError("source sampler report eligibility identity mismatch")
    sdf_audit = integrity.get("all_pose_sdf_artifact_audit")
    if not isinstance(sdf_audit, dict):
        raise ValueError("source sampler report lacks all-pose SDF audit")
    if (
        _strict_int(sdf_audit.get("files_verified"), label="source SDF files")
        != FULL_ELIGIBLE_COUNT * 3
        or _strict_int(sdf_audit.get("records_verified"), label="source SDF records")
        != FULL_ELIGIBLE_COUNT * POSE_COUNT * 3
        or sdf_audit.get("sha256_recomputed") is not True
    ):
        raise ValueError("source sampler report all-pose SDF audit is incomplete")

    if (
        source_audit.get("schema_version") != SOURCE_AUDIT_SCHEMA_VERSION
        or source_audit.get("protocol_id") != SOURCE_PROTOCOL_ID
        or source_audit.get("status") != "complete"
    ):
        raise ValueError("source coordinate audit is not complete")
    audit_eligibility = source_audit.get("eligibility_manifest")
    inventory = source_audit.get("inventory")
    if not isinstance(audit_eligibility, dict) or not isinstance(inventory, dict):
        raise ValueError("source coordinate audit lacks identity/inventory")
    if audit_eligibility.get("sha256") != eligibility_sha256:
        raise ValueError("source coordinate audit eligibility identity mismatch")
    audit_counts = {
        "shard_count": EXPECTED_SHARDS,
        "eligible_sample_count": FULL_ELIGIBLE_COUNT,
        "excluded_sample_count": FULL_EXCLUDED_COUNT,
        "candidate_count_per_sample": POSE_COUNT,
        "audited_csv_rows": FULL_ELIGIBLE_COUNT * 3,
        "parsed_sdf_records": FULL_ELIGIBLE_COUNT * POSE_COUNT * 3,
        "system_count": FULL_SYSTEM_COUNT,
    }
    for key, expected in audit_counts.items():
        if _strict_int(inventory.get(key), label=f"source audit.inventory.{key}") != expected:
            raise ValueError(f"source audit.inventory.{key}: expected {expected}")


def _expected_ids_by_shard(eligible_ids: Sequence[str], stage: str) -> dict[int, list[str]]:
    if stage == "full":
        return {
            shard_index: list(eligible_ids[shard_index::EXPECTED_SHARDS])
            for shard_index in range(EXPECTED_SHARDS)
        }
    if stage == "smoke":
        if len(eligible_ids) < SMOKE_COUNT or SMOKE_COUNT != EXPECTED_SHARDS:
            raise ValueError("smoke requires one of the first eight eligible IDs per shard")
        return {index: [str(eligible_ids[index])] for index in range(EXPECTED_SHARDS)}
    raise ValueError(f"unsupported stage: {stage!r}")


def _validate_bank_manifest(
    payload: dict[str, Any],
    *,
    pins: dict[str, str],
    eligibility: dict[str, Any],
    source_output_root: Path,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], dict[str, str]]:
    _require_exact_fields(
        payload, BANK_TOP_LEVEL_FIELDS, label="label-free bank manifest"
    )
    if payload.get("schema_version") != BANK_SCHEMA_VERSION:
        raise ValueError("label-free bank manifest: unexpected schema_version")
    if (
        payload.get("protocol_id") != PROTOCOL_ID
        or payload.get("source_protocol_id") != SOURCE_PROTOCOL_ID
        or payload.get("status") != "complete_label_free"
    ):
        raise ValueError("label-free bank manifest: wrong protocol or incomplete status")
    inputs = payload.get("inputs")
    inventory = payload.get("inventory")
    fixed_settings = payload.get("fixed_settings")
    information_boundary = payload.get("information_boundary")
    backbone_arms = payload.get("backbone_arms")
    records = payload.get("records")
    source_shards = payload.get("source_shards")
    if (
        not isinstance(inputs, dict)
        or not isinstance(inventory, dict)
        or not isinstance(fixed_settings, dict)
        or not isinstance(information_boundary, dict)
        or not isinstance(backbone_arms, dict)
    ):
        raise ValueError(
            "label-free bank manifest lacks inputs/inventory/fixed_settings/boundary/backbones"
        )
    if not isinstance(records, list) or not isinstance(source_shards, list):
        raise ValueError("label-free bank manifest lacks records/source_shards")
    _require_exact_fields(inputs, BANK_INPUT_FIELDS, label="label-free bank inputs")
    # A manifest cannot contain its own digest without a circular identity.
    # The bank digest is a required CLI pin and is instead checked by every
    # downstream score shard.
    bank_input_names = {
        "protocol_sha256": "protocol_document",
        "eligibility_manifest_sha256": "eligibility_manifest",
        "source_sampler_report_sha256": "source_sampler_report",
        "source_coordinate_audit_sha256": "source_coordinate_audit",
    }
    for pin_name, input_name in bank_input_names.items():
        if _identity_sha(inputs, input_name, label="bank.inputs") != pins[pin_name]:
            raise ValueError(
                f"label-free bank manifest input identity mismatch: {input_name}"
            )
    frozen_bank_inputs = {
        "config": FROZEN_CONFIG_SHA256,
        "s50_backbone_checkpoint": FROZEN_BACKBONE_SHA256[PRIMARY_ARM],
        "matched_backbone_checkpoint": FROZEN_BACKBONE_SHA256[DIAGNOSTIC_ARM],
        "confidence_checkpoint": FROZEN_CONFIDENCE_SHA256,
        "source_sampler_protocol": FROZEN_SOURCE_PROTOCOL_SHA256,
    }
    for input_name, expected_sha in frozen_bank_inputs.items():
        if _identity_sha(inputs, input_name, label="bank.inputs") != expected_sha:
            raise ValueError(f"label-free bank frozen input mismatch: {input_name}")
    asset_input_names = {
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
    }
    bank_assets: dict[str, tuple[Path, str]] = {}
    for input_name in asset_input_names:
        bank_assets[input_name] = _path_spec(
            inputs.get(input_name), label=f"bank.inputs.{input_name}"
        )
    for input_name in ("source_sampler_protocol", "scorer_source", "report_source"):
        asset_path, asset_sha = bank_assets[input_name]
        if _file_sha256(asset_path) != asset_sha:
            raise ValueError(f"label-free bank input changed after freeze: {input_name}")
    declared_source_root = inputs.get("source_output_root")
    if not isinstance(declared_source_root, str):
        raise ValueError("bank.inputs.source_output_root: expected an absolute canonical path")
    declared_source_path = Path(declared_source_root)
    if (
        not declared_source_path.is_absolute()
        or declared_source_path.resolve(strict=True) != declared_source_path
        or declared_source_path != source_output_root
    ):
        raise ValueError("bank.inputs.source_output_root: frozen source root mismatch")
    if payload.get("bank_root") != str(source_output_root):
        raise ValueError("label-free bank bank_root mismatch")
    fixed_values: dict[str, object] = {
        "source_arm": SOURCE_ARM,
        "pose_count": POSE_COUNT,
        "sample_sigma": SAMPLE_SIGMA,
        "num_steps": NUM_STEPS,
        "prior_pool_size": PRIOR_POOL_SIZE,
        "pose_batch_size": 20,
        "pocket_cutoff_angstrom": 10.0,
        "ligand_conformer_seed": 0,
        "selector": SELECTOR,
        "label_blind": True,
        "resampling": False,
    }
    for key, expected in fixed_values.items():
        if fixed_settings.get(key) != expected:
            raise ValueError(f"bank.fixed_settings.{key}: expected {expected!r}")
    _require_exact_fields(
        fixed_settings, set(fixed_values), label="label-free bank fixed_settings"
    )
    expected_boundary_fields = {
        "source_csv_allowlist",
        "outcome_columns_exported",
        "score_stage_reads_source_results_csv",
        "crystal_reference_exported",
    }
    _require_exact_fields(
        information_boundary,
        expected_boundary_fields,
        label="label-free bank information_boundary",
    )
    declared_allowlist = information_boundary.get("source_csv_allowlist")
    if (
        not isinstance(declared_allowlist, list)
        or len(declared_allowlist) != len(SOURCE_ROW_ALLOWLIST)
        or set(declared_allowlist) != SOURCE_ROW_ALLOWLIST
    ):
        raise ValueError("label-free bank source CSV allowlist mismatch")
    if any(
        (
            information_boundary.get("outcome_columns_exported") is not False,
            information_boundary.get("score_stage_reads_source_results_csv") is not False,
            information_boundary.get("crystal_reference_exported") is not False,
        )
    ):
        raise ValueError("label-free bank information boundary is not sealed")
    _require_exact_fields(backbone_arms, set(ARMS), label="label-free bank backbone_arms")
    for arm in ARMS:
        arm_spec = backbone_arms.get(arm)
        if not isinstance(arm_spec, dict):
            raise ValueError(f"label-free bank lacks backbone arm {arm}")
        _require_exact_fields(
            arm_spec, {"path", "sha256", "role"}, label=f"bank.backbone_arms.{arm}"
        )
        if _identity_sha(arm_spec, "sha256", label=f"bank.backbone_arms.{arm}") != (
            FROZEN_BACKBONE_SHA256[arm]
        ):
            raise ValueError(f"label-free bank backbone identity mismatch: {arm}")
        arm_path, arm_sha = _path_spec(
            {"path": arm_spec.get("path"), "sha256": arm_spec.get("sha256")},
            label=f"bank.backbone_arms.{arm}",
        )
        input_name = f"{arm.removesuffix('_backbone')}_backbone_checkpoint"
        if (arm_path, arm_sha) != bank_assets[input_name]:
            raise ValueError(f"label-free bank backbone asset mismatch: {arm}")
    expected_inventory = {
        "full_count": FULL_SPLIT_COUNT,
        "eligible_count": FULL_ELIGIBLE_COUNT,
        "eligible_system_count": FULL_SYSTEM_COUNT,
        "excluded_count": FULL_EXCLUDED_COUNT,
        "source_shard_count": EXPECTED_SHARDS,
        "pose_count": POSE_COUNT,
    }
    for key, expected in expected_inventory.items():
        if _strict_int(inventory.get(key), label=f"bank.inventory.{key}") != expected:
            raise ValueError(f"bank.inventory.{key}: expected {expected}")
    expected_inventory_fields = set(expected_inventory) | {
        "full_ids_sha256",
        "eligible_ids_sha256",
        "excluded_ids_sha256",
    }
    _require_exact_fields(
        inventory, expected_inventory_fields, label="label-free bank inventory"
    )
    if inventory.get("full_ids_sha256") != _ids_sha256(eligibility["full_ids"]):
        raise ValueError("bank full ID identity mismatch")
    if inventory.get("eligible_ids_sha256") != eligibility["eligible_ids_sha256"]:
        raise ValueError("bank eligible ID identity mismatch")
    if inventory.get("excluded_ids_sha256") != eligibility["excluded_ids_sha256"]:
        raise ValueError("bank excluded ID identity mismatch")

    runtime_identity = inputs.get("runtime_code_identity")
    if not isinstance(runtime_identity, dict):
        raise ValueError("bank.inputs.runtime_code_identity: expected an object")
    _require_exact_fields(
        runtime_identity,
        {"aggregate_sha256", "files"},
        label="bank.inputs.runtime_code_identity",
    )
    runtime_sha = _require_sha256(
        runtime_identity.get("aggregate_sha256"),
        label="bank.inputs.runtime_code_identity.aggregate_sha256",
    )
    runtime_files = runtime_identity.get("files")
    if not isinstance(runtime_files, dict) or not runtime_files:
        raise ValueError("bank.inputs.runtime_code_identity.files: expected non-empty object")
    for name, spec in runtime_files.items():
        if not isinstance(name, str) or not name or not isinstance(spec, dict):
            raise ValueError("bank runtime code file inventory is malformed")
        _require_exact_fields(
            spec,
            {"path", "sha256", "size_bytes"},
            label=f"bank.inputs.runtime_code_identity.files.{name}",
        )
        runtime_path, _ = _path_spec(
            {"path": spec.get("path"), "sha256": spec.get("sha256")},
            label=f"bank.inputs.runtime_code_identity.files.{name}",
        )
        if _strict_int(
            spec.get("size_bytes"), label=f"bank runtime file {name} size"
        ) != runtime_path.stat().st_size:
            raise ValueError(f"bank runtime file size changed: {name}")
    if len(records) != FULL_ELIGIBLE_COUNT:
        raise ValueError("bank record count mismatch")
    by_id: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"bank.records[{index}]: expected an object")
        if set(record) != BANK_RECORD_FIELDS:
            missing = sorted(BANK_RECORD_FIELDS - set(record))
            extra = sorted(set(record) - BANK_RECORD_FIELDS)
            raise ValueError(
                f"bank.records[{index}] exact field inventory mismatch; "
                f"missing={missing}, extra={extra}"
            )
        sample_id = record.get("sample_key")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"bank.records[{index}].sample_key: invalid")
        if sample_id in by_id:
            raise ValueError(f"bank contains duplicate sample_key {sample_id}")
        by_id[sample_id] = record
    if list(by_id) != eligibility["eligible_ids"]:
        raise ValueError("bank records are not the exact ordered eligible cohort")
    if len(source_shards) != EXPECTED_SHARDS:
        raise ValueError("bank source_shards count mismatch")
    provenance = {
        "source_sampler_protocol_sha256": FROZEN_SOURCE_PROTOCOL_SHA256,
        "scorer_source_sha256": bank_assets["scorer_source"][1],
        "report_source_sha256": bank_assets["report_source"][1],
        "runtime_code_identity_sha256": runtime_sha,
    }
    return by_id, source_shards, provenance


def _validate_bank_source_shard_metadata(
    source_shards: list[dict[str, Any]],
    *,
    expected_full_by_shard: dict[int, list[str]],
    source_output_root: Path,
) -> None:
    """Validate label-free provenance without hashing or opening outcome files."""
    expected_fields = {
        "shard_index",
        "paired_summary",
        "results_csv",
        "assigned_count",
        "assigned_ids_sha256",
    }
    seen_indices: set[int] = set()
    for source in source_shards:
        if not isinstance(source, dict) or set(source) != expected_fields:
            raise ValueError("bank source_shards exact field inventory mismatch")
        shard_index = _strict_int(source.get("shard_index"), label="source shard_index")
        if shard_index in seen_indices or shard_index not in expected_full_by_shard:
            raise ValueError(f"duplicate/out-of-range source shard {shard_index}")
        seen_indices.add(shard_index)
        expected_ids = expected_full_by_shard[shard_index]
        if _strict_int(source.get("assigned_count"), label="source assigned_count") != len(
            expected_ids
        ):
            raise ValueError(f"source shard {shard_index} assigned_count mismatch")
        if source.get("assigned_ids_sha256") != _ids_sha256(expected_ids):
            raise ValueError(f"source shard {shard_index} assigned_ids_sha256 mismatch")

        paired_path, _ = _path_spec(
            source.get("paired_summary"), label="source paired_summary"
        )
        expected_paired_path = (
            source_output_root
            / f"shard-{shard_index:03d}-of-{EXPECTED_SHARDS:03d}"
            / "paired_summary.json"
        )
        if paired_path != expected_paired_path:
            raise ValueError(f"source shard {shard_index} paired summary path mismatch")
        _require_within(
            paired_path,
            source_output_root,
            label=f"source shard {shard_index} paired summary",
        )

        csv_path, _ = _path_spec(source.get("results_csv"), label="source results_csv")
        expected_csv_path = (
            source_output_root
            / f"shard-{shard_index:03d}-of-{EXPECTED_SHARDS:03d}"
            / "arms"
            / SOURCE_ARM
            / "results.csv"
        )
        if csv_path != expected_csv_path:
            raise ValueError(f"source shard {shard_index} CSV path mismatch")
        _require_within(
            csv_path, source_output_root, label=f"source shard {shard_index} CSV"
        )
    if seen_indices != set(range(EXPECTED_SHARDS)):
        raise ValueError("source shard index inventory mismatch")


def _read_source_rows(
    source_shards: list[dict[str, Any]],
    *,
    expected_full_by_shard: dict[int, list[str]],
    source_output_root: Path,
) -> tuple[dict[str, dict[str, str]], list[dict[str, Any]]]:
    rows_by_id: dict[str, dict[str, str]] = {}
    identities: list[dict[str, Any]] = []
    seen_indices: set[int] = set()
    for source in source_shards:
        if not isinstance(source, dict):
            raise ValueError("bank source_shards entries must be objects")
        expected_fields = {
            "shard_index",
            "paired_summary",
            "results_csv",
            "assigned_count",
            "assigned_ids_sha256",
        }
        if set(source) != expected_fields:
            raise ValueError("bank source_shards exact field inventory mismatch")
        shard_index = _strict_int(source.get("shard_index"), label="source shard_index")
        if shard_index in seen_indices or shard_index not in expected_full_by_shard:
            raise ValueError(f"duplicate/out-of-range source shard {shard_index}")
        seen_indices.add(shard_index)
        paired_path, paired_sha = _path_spec(
            source.get("paired_summary"), label="source paired_summary"
        )
        expected_paired_path = (
            source_output_root
            / f"shard-{shard_index:03d}-of-{EXPECTED_SHARDS:03d}"
            / "paired_summary.json"
        )
        if paired_path != expected_paired_path or _file_sha256(paired_path) != paired_sha:
            raise ValueError(f"source shard {shard_index} paired summary identity mismatch")
        path, expected_sha = _path_spec(source.get("results_csv"), label="source results_csv")
        expected_csv_path = (
            source_output_root
            / f"shard-{shard_index:03d}-of-{EXPECTED_SHARDS:03d}"
            / "arms"
            / SOURCE_ARM
            / "results.csv"
        )
        if path != expected_csv_path:
            raise ValueError(f"source shard {shard_index} CSV is outside its canonical path")
        _require_within(path, source_output_root, label=f"source shard {shard_index} CSV")
        actual_sha = _file_sha256(path)
        if actual_sha != expected_sha:
            raise ValueError(f"source shard {shard_index} CSV SHA-256 mismatch")
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        ids = [str(row.get("id", "")) for row in rows]
        expected_ids = expected_full_by_shard[shard_index]
        if ids != expected_ids:
            raise ValueError(f"source shard {shard_index} exact ordered ID inventory mismatch")
        if _strict_int(source.get("assigned_count"), label="source assigned_count") != len(ids):
            raise ValueError(f"source shard {shard_index} assigned_count mismatch")
        if source.get("assigned_ids_sha256") != _ids_sha256(ids):
            raise ValueError(f"source shard {shard_index} assigned_ids_sha256 mismatch")
        for row in rows:
            sample_id = str(row["id"])
            if sample_id in rows_by_id:
                raise ValueError(f"duplicate source sample {sample_id}")
            if row.get("arm") != SOURCE_ARM:
                raise ValueError(f"{sample_id}: source row is not the frozen S50 arm")
            rows_by_id[sample_id] = row
        identities.append(
            {
                "shard_index": shard_index,
                "results_csv": str(path),
                "results_csv_sha256": actual_sha,
                "assigned_count": len(ids),
                "assigned_ids_sha256": _ids_sha256(ids),
            }
        )
    if seen_indices != set(range(EXPECTED_SHARDS)):
        raise ValueError("source shard index inventory mismatch")
    return rows_by_id, sorted(identities, key=lambda row: int(row["shard_index"]))


def _strict_float_array(value: object, *, label: str, length: int) -> list[float]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{label}: expected {length} values")
    return [_finite_float(item, label=f"{label}[{index}]") for index, item in enumerate(value)]


def _source_float_array(value: object, *, label: str, length: int) -> list[float]:
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label}: invalid JSON array") from exc
    return _strict_float_array(parsed, label=label, length=length)


def _source_bool_array(value: object, *, label: str, length: int) -> list[bool]:
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label}: invalid JSON array") from exc
    if (
        not isinstance(parsed, list)
        or len(parsed) != length
        or any(not isinstance(item, bool) for item in parsed)
    ):
        raise ValueError(f"{label}: expected {length} JSON booleans")
    return list(parsed)


def _validate_bank_record_label_free(
    sample_id: str,
    bank: dict[str, Any],
    *,
    expected_shard_index: int,
    source_output_root: Path,
) -> None:
    if bank.get("sample_key") != sample_id:
        raise ValueError(f"{sample_id}: bank sample key changed")
    if not isinstance(bank.get("system_id"), str) or not bank["system_id"]:
        raise ValueError(f"{sample_id}: invalid bank system_id")
    if not isinstance(bank.get("ligand_chain"), str) or not bank["ligand_chain"]:
        raise ValueError(f"{sample_id}: invalid bank ligand_chain")
    exact_integer_fields = {
        "pose_count": POSE_COUNT,
        "num_steps": NUM_STEPS,
        "prior_pool_size": PRIOR_POOL_SIZE,
    }
    for key, expected in exact_integer_fields.items():
        if _strict_int(bank.get(key), label=f"{sample_id}.{key}") != expected:
            raise ValueError(f"{sample_id}: bank {key} mismatch")
    if _strict_int(
        bank.get("plinder_global_index"), label=f"{sample_id}.plinder_global_index"
    ) < 0:
        raise ValueError(f"{sample_id}: invalid plinder_global_index")
    if _strict_int(
        bank.get("source_shard_index"), label=f"{sample_id}.source_shard_index"
    ) != expected_shard_index:
        raise ValueError(f"{sample_id}: bank source shard mismatch")
    _strict_int(bank.get("sampling_seed"), label=f"{sample_id}.sampling_seed")
    if (
        _strict_int(
            bank.get("ligand_conformer_seed"),
            label=f"{sample_id}.ligand_conformer_seed",
        )
        != 0
    ):
        raise ValueError(f"{sample_id}: ligand conformer seed must be zero")
    sigma = _finite_float(bank.get("sample_sigma"), label=f"{sample_id}.sample_sigma")
    if not math.isclose(sigma, SAMPLE_SIGMA, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{sample_id}: bank sample_sigma mismatch")
    for key in (
        "candidate_ensemble_sha256",
        "prior_pool_sha256",
        "receptor_sha256",
        "processed_meta_sha256",
        "ligand_input_identity_sha256",
    ):
        _require_sha256(bank.get(key), label=f"{sample_id}.{key}")
    canonical_smiles = bank.get("canonical_smiles")
    if not isinstance(canonical_smiles, str) or not canonical_smiles:
        raise ValueError(f"{sample_id}: bank canonical_smiles is empty")
    if bank.get("ligand_input_identity_sha256") != _canonical_smiles_identity(
        canonical_smiles
    ):
        raise ValueError(f"{sample_id}: canonical SMILES identity mismatch")
    if _strict_int(bank.get("num_input_atoms"), label=f"{sample_id}.num_input_atoms") < 1:
        raise ValueError(f"{sample_id}: bank num_input_atoms must be positive")
    pocket_center = bank.get("pocket_center")
    if not isinstance(pocket_center, list) or len(pocket_center) != 3:
        raise ValueError(f"{sample_id}: invalid frozen pocket center")
    for index, value in enumerate(pocket_center):
        _finite_float(value, label=f"{sample_id}.pocket_center[{index}]")

    receptor_path, receptor_sha = _path_spec(
        bank.get("receptor"), label=f"{sample_id}.receptor"
    )
    processed_meta_path, processed_meta_sha = _path_spec(
        bank.get("processed_meta"), label=f"{sample_id}.processed_meta"
    )
    if receptor_sha != bank.get("receptor_sha256"):
        raise ValueError(f"{sample_id}: receptor asset/scalar hash mismatch")
    if processed_meta_sha != bank.get("processed_meta_sha256"):
        raise ValueError(f"{sample_id}: processed-meta asset/scalar hash mismatch")
    # Existence and lexical identity are checked here. Their content is not opened
    # during smoke; the full label join rehashes both assets.
    if not receptor_path.is_file() or not processed_meta_path.is_file():
        raise ValueError(f"{sample_id}: frozen receptor or processed-meta asset is missing")

    sdf_path, _ = _path_spec(
        bank.get("all_poses_sdf"), label=f"{sample_id}.all_poses_sdf"
    )
    expected_sdf = (
        source_output_root
        / f"shard-{expected_shard_index:03d}-of-{EXPECTED_SHARDS:03d}"
        / "arms"
        / SOURCE_ARM
        / "poses"
        / "all_poses"
        / f"{sample_id}.sdf"
    )
    if sdf_path != expected_sdf:
        raise ValueError(f"{sample_id}: source SDF is outside its canonical frozen path")
    _require_within(sdf_path, source_output_root, label=f"{sample_id}.source SDF")


def _validate_bank_source_identity(
    sample_id: str,
    bank: dict[str, Any],
    source: dict[str, str],
    *,
    source_output_root: Path,
) -> None:
    comparisons: tuple[tuple[str, str], ...] = (
        ("system_id", "plinder_system_id"),
        ("ligand_chain", "plinder_ligand_chain"),
        ("plinder_global_index", "plinder_global_index"),
        ("sampling_seed", "sampling_seed"),
        ("ligand_conformer_seed", "ligand_conformer_seed"),
        ("candidate_ensemble_sha256", "candidate_ensemble_sha256"),
        ("prior_pool_sha256", "prior_pool_sha256"),
        ("prior_pool_size", "prior_pool_size"),
        ("receptor_sha256", "protein_sha256"),
        ("processed_meta_sha256", "processed_meta_sha256"),
        ("ligand_input_identity_sha256", "ligand_input_identity_sha256"),
    )
    for bank_key, source_key in comparisons:
        if str(bank.get(bank_key)) != str(source.get(source_key)):
            raise ValueError(f"{sample_id}: bank/source identity mismatch for {bank_key}")
    if _strict_int(source.get("num_samples"), label=f"{sample_id}.num_samples") != POSE_COUNT:
        raise ValueError(f"{sample_id}: source candidate count mismatch")
    if (
        _strict_int(
            source.get("all_poses_count"), label=f"{sample_id}.all_poses_count"
        )
        != POSE_COUNT
    ):
        raise ValueError(f"{sample_id}: source SDF record count mismatch")
    if source.get("checkpoint_sha256") != FROZEN_BACKBONE_SHA256[PRIMARY_ARM]:
        raise ValueError(f"{sample_id}: source S50 checkpoint identity mismatch")
    if source.get("selector_profile") != "candidate_only":
        raise ValueError(f"{sample_id}: source selector was not candidate_only")
    if source.get("sampling_dynamics") != "deterministic_ode":
        raise ValueError(f"{sample_id}: source sampling dynamics changed")
    if _strict_int(bank.get("num_steps"), label=f"{sample_id}.num_steps") != NUM_STEPS:
        raise ValueError(f"{sample_id}: bank num_steps mismatch")
    if _strict_int(bank.get("num_input_atoms"), label=f"{sample_id}.num_input_atoms") < 1:
        raise ValueError(f"{sample_id}: bank num_input_atoms must be positive")
    canonical_smiles = bank.get("canonical_smiles")
    if not isinstance(canonical_smiles, str) or not canonical_smiles:
        raise ValueError(f"{sample_id}: bank canonical_smiles is empty")
    if canonical_smiles != source.get("ligand_input_canonical_smiles"):
        raise ValueError(f"{sample_id}: bank/source canonical SMILES mismatch")
    pocket_center = bank.get("pocket_center")
    if not isinstance(pocket_center, list) or len(pocket_center) != 3:
        raise ValueError(f"{sample_id}: invalid frozen pocket center")
    for index, value in enumerate(pocket_center):
        _finite_float(value, label=f"{sample_id}.pocket_center[{index}]")
    receptor_path, receptor_sha = _path_spec(
        bank.get("receptor"), label=f"{sample_id}.receptor"
    )
    processed_meta_path, processed_meta_sha = _path_spec(
        bank.get("processed_meta"), label=f"{sample_id}.processed_meta"
    )
    if receptor_sha != bank.get("receptor_sha256") or receptor_sha != source.get("protein_sha256"):
        raise ValueError(f"{sample_id}: receptor hash identity mismatch")
    if (
        processed_meta_sha != bank.get("processed_meta_sha256")
        or processed_meta_sha != source.get("processed_meta_sha256")
    ):
        raise ValueError(f"{sample_id}: processed-meta hash identity mismatch")
    if str(receptor_path) != str(source.get("protein", "")):
        raise ValueError(f"{sample_id}: receptor path identity mismatch")
    if str(processed_meta_path) != str(source.get("processed_meta", "")):
        raise ValueError(f"{sample_id}: processed-meta path identity mismatch")
    if _file_sha256(receptor_path) != receptor_sha or _file_sha256(processed_meta_path) != processed_meta_sha:
        raise ValueError(f"{sample_id}: receptor or processed-meta asset changed")
    source_sdf = Path(str(source.get("all_poses_sdf", "")))
    bank_sdf, bank_sdf_sha = _path_spec(bank.get("all_poses_sdf"), label=f"{sample_id}.SDF")
    if source_sdf != bank_sdf or source.get("all_poses_sdf_sha256") != bank_sdf_sha:
        raise ValueError(f"{sample_id}: bank/source SDF identity mismatch")
    source_shard_index = _strict_int(
        bank.get("source_shard_index"), label=f"{sample_id}.source_shard_index"
    )
    expected_sdf = (
        source_output_root
        / f"shard-{source_shard_index:03d}-of-{EXPECTED_SHARDS:03d}"
        / "arms"
        / SOURCE_ARM
        / "poses"
        / "all_poses"
        / f"{sample_id}.sdf"
    )
    if bank_sdf != expected_sdf:
        raise ValueError(f"{sample_id}: source SDF is outside its canonical frozen path")
    _require_within(bank_sdf, source_output_root, label=f"{sample_id}.source SDF")


def _sdf_prop(molecule: Chem.Mol, name: str, *, label: str) -> str:
    if not molecule.HasProp(name):
        raise ValueError(f"{label}: missing SDF property {name!r}")
    return molecule.GetProp(name)


def _audit_source_sdf(sample_id: str, bank: dict[str, Any]) -> dict[str, Any]:
    path, expected_sha = _path_spec(bank.get("all_poses_sdf"), label=f"{sample_id}.all_poses_sdf")
    before_sha = _file_sha256(path)
    if before_sha != expected_sha:
        raise ValueError(f"{sample_id}: source SDF changed after bank freeze")
    record_count = 0
    with path.open("rb") as handle:
        supplier = Chem.ForwardSDMolSupplier(handle)
        for index, molecule in enumerate(supplier):
            label = f"{sample_id}.all_poses_sdf[{index}]"
            if molecule is None:
                raise ValueError(f"{label}: RDKit could not parse record")
            if molecule.GetNumAtoms() != _strict_int(
                bank.get("num_input_atoms"), label=f"{sample_id}.num_input_atoms"
            ):
                raise ValueError(f"{label}: atom-count identity mismatch")
            if _strict_int(_sdf_prop(molecule, "sample_index", label=label), label=label) != index:
                raise ValueError(f"{label}: sample_index order mismatch")
            if (
                _strict_int(
                    _sdf_prop(molecule, "sampling_seed", label=label), label=f"{label}.seed"
                )
                != _strict_int(bank.get("sampling_seed"), label=f"{sample_id}.bank.seed")
            ):
                raise ValueError(f"{label}: sampling seed mismatch")
            if (
                _strict_int(
                    _sdf_prop(molecule, "ligand_conformer_seed", label=label),
                    label=f"{label}.conformer_seed",
                )
                != _strict_int(
                    bank.get("ligand_conformer_seed"), label=f"{sample_id}.bank.conformer_seed"
                )
            ):
                raise ValueError(f"{label}: conformer seed mismatch")
            if _sdf_prop(molecule, "candidate_ensemble_sha256", label=label) != bank.get(
                "candidate_ensemble_sha256"
            ):
                raise ValueError(f"{label}: candidate ensemble identity mismatch")
            sigma = _finite_float(
                _sdf_prop(molecule, "sample_sigma", label=label), label=f"{label}.sample_sigma"
            )
            if not math.isclose(sigma, SAMPLE_SIGMA, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f"{label}: sample sigma is not {SAMPLE_SIGMA}")
            record_count += 1
    if record_count != POSE_COUNT:
        raise ValueError(f"{sample_id}: expected {POSE_COUNT} SDF records, got {record_count}")
    after_sha = _file_sha256(path)
    if after_sha != before_sha:
        raise RuntimeError(f"{sample_id}: source SDF changed while being audited")
    return {"path": str(path), "sha256": after_sha, "record_count": record_count}


def _canonical_score_path(scores_root: Path, stage: str, shard: int, arm: str) -> Path:
    return (
        scores_root
        / stage
        / f"shard-{shard:03d}-of-{EXPECTED_SHARDS:03d}"
        / "arms"
        / arm
        / "scores.json"
    )


def _load_score_artifact(
    path: Path,
    *,
    stage: str,
    arm: str,
    shard_index: int,
    expected_ids: list[str],
    identity_pins: dict[str, str],
    report_source_sha256: str,
    bank_provenance: dict[str, str],
    expected_bank_manifest_path: Path,
    expected_backbone_path: Path,
    expected_confidence_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = _load_json_object(path, label=f"score artifact {arm}/{shard_index}")
    _require_exact_fields(payload, SCORE_TOP_LEVEL_FIELDS, label=f"score artifact {path}")
    expected_mode = "full_shard" if stage == "full" else "smoke_replay"
    expected_arm_role = (
        "primary_deployment_backbone"
        if arm == PRIMARY_ARM
        else "diagnostic_training_matched_backbone"
    )
    exact = {
        "schema_version": SCORE_SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": "complete",
        "mode": expected_mode,
        "stage": stage,
        "arm": arm,
        "arm_role": expected_arm_role,
        "selector": SELECTOR,
    }
    for key, value in exact.items():
        if payload.get(key) != value:
            raise ValueError(f"{path}.{key}: expected {value!r}")
    inputs = payload.get("inputs")
    inventory = payload.get("inventory")
    fixed_settings = payload.get("fixed_settings")
    records = payload.get("records")
    runtime = payload.get("runtime")
    if (
        not isinstance(inputs, dict)
        or not isinstance(inventory, dict)
        or not isinstance(fixed_settings, dict)
        or not isinstance(records, list)
        or not isinstance(runtime, dict)
    ):
        raise ValueError(f"{path}: score artifact lacks inputs/settings/inventory/records/runtime")
    _require_exact_fields(inputs, SCORE_INPUT_FIELDS, label=f"{path}.inputs")
    for key, expected in identity_pins.items():
        if _identity_sha(inputs, key, label=f"{path}.inputs") != expected:
            raise ValueError(f"{path}: input identity mismatch for {key}")
    if _identity_sha(inputs, "report_source_sha256", label=f"{path}.inputs") != report_source_sha256:
        raise ValueError(f"{path}: report source identity mismatch")
    if _identity_sha(inputs, "confidence_checkpoint_sha256", label=f"{path}.inputs") != FROZEN_CONFIDENCE_SHA256:
        raise ValueError(f"{path}: confidence checkpoint identity mismatch")
    if _identity_sha(inputs, "backbone_checkpoint_sha256", label=f"{path}.inputs") != FROZEN_BACKBONE_SHA256[arm]:
        raise ValueError(f"{path}: backbone checkpoint identity mismatch")
    if _identity_sha(inputs, "config_sha256", label=f"{path}.inputs") != FROZEN_CONFIG_SHA256:
        raise ValueError(f"{path}: config identity mismatch")
    scorer_sha = _identity_sha(inputs, "scorer_source_sha256", label=f"{path}.inputs")
    if scorer_sha != bank_provenance["scorer_source_sha256"]:
        raise ValueError(f"{path}: scorer source differs from the frozen bank")
    if (
        _identity_sha(inputs, "source_sampler_protocol_sha256", label=f"{path}.inputs")
        != FROZEN_SOURCE_PROTOCOL_SHA256
    ):
        raise ValueError(f"{path}: source sampler protocol identity mismatch")
    runtime_code_sha = _identity_sha(
        inputs, "runtime_code_identity_sha256", label=f"{path}.inputs"
    )
    if runtime_code_sha != bank_provenance["runtime_code_identity_sha256"]:
        raise ValueError(f"{path}: runtime code identity differs from the frozen bank")
    declared_bank_path = Path(str(inputs.get("label_free_bank_manifest", "")))
    if declared_bank_path != expected_bank_manifest_path:
        raise ValueError(f"{path}: bank manifest path identity mismatch")
    declared_backbone_path = Path(str(inputs.get("backbone_checkpoint", "")))
    if declared_backbone_path != expected_backbone_path:
        raise ValueError(f"{path}: backbone checkpoint path identity mismatch")
    declared_confidence_path = Path(str(inputs.get("confidence_checkpoint", "")))
    if declared_confidence_path != expected_confidence_path:
        raise ValueError(f"{path}: confidence checkpoint path identity mismatch")
    score_fixed_values: dict[str, object] = {
        "saved_pose_bank_only": True,
        "resampling": False,
        "sample_sigma": SAMPLE_SIGMA,
        "pose_count": POSE_COUNT,
        "num_steps": NUM_STEPS,
        "prior_pool_size": PRIOR_POOL_SIZE,
        "pose_batch_size": 20,
        "t1_hidden_backbone": arm,
        "pocket_cutoff_angstrom": 10.0,
        "selector": SELECTOR,
        "label_blind": True,
    }
    _require_exact_fields(
        fixed_settings, set(score_fixed_values), label=f"{path}.fixed_settings"
    )
    for key, expected in score_fixed_values.items():
        if fixed_settings.get(key) != expected:
            raise ValueError(f"{path}.fixed_settings.{key}: expected {expected!r}")
    expected_inventory = {
        "eligible_count": FULL_ELIGIBLE_COUNT,
        "source_shard_count": EXPECTED_SHARDS,
        "source_shard_index": shard_index,
        "assigned_count": len(expected_ids),
        "scored_count": len(expected_ids),
    }
    for key, expected in expected_inventory.items():
        if _strict_int(inventory.get(key), label=f"{path}.inventory.{key}") != expected:
            raise ValueError(f"{path}.inventory.{key}: expected {expected}")
    _require_exact_fields(
        inventory,
        set(expected_inventory) | {"assigned_ids_sha256"},
        label=f"{path}.inventory",
    )
    if inventory.get("assigned_ids_sha256") != _ids_sha256(expected_ids):
        raise ValueError(f"{path}: assigned ID hash mismatch")
    if len(records) != len(expected_ids):
        raise ValueError(f"{path}: scored record count mismatch")
    normalized: list[dict[str, Any]] = []
    for index, (record, expected_id) in enumerate(zip(records, expected_ids, strict=True)):
        if not isinstance(record, dict):
            raise ValueError(f"{path}.records[{index}]: expected an object")
        if set(record) != SCORE_RECORD_FIELDS:
            missing = sorted(SCORE_RECORD_FIELDS - set(record))
            extra = sorted(set(record) - SCORE_RECORD_FIELDS)
            raise ValueError(
                f"{path}.records[{index}] exact field inventory mismatch; "
                f"missing={missing}, extra={extra}"
            )
        if record.get("sample_key") != expected_id:
            raise ValueError(f"{path}.records[{index}]: sample key/order mismatch")
        arrays = record.get("score_arrays")
        if not isinstance(arrays, dict) or set(arrays) != set(SCORE_ARRAY_FIELDS):
            raise ValueError(f"{path}.records[{index}]: score array field inventory mismatch")
        normalized_arrays = {
            key: _strict_float_array(
                arrays[key], label=f"{path}.records[{index}].score_arrays.{key}", length=POSE_COUNT
            )
            for key in SCORE_ARRAY_FIELDS
        }
        for key in ("confidence_rmsd", "confidence_atom_rmsd", "confidence_atom_q90"):
            if any(value < 0.0 for value in normalized_arrays[key]):
                raise ValueError(f"{path}.records[{index}].{key}: negative prediction")
        for key in ("confidence_success", "confidence_atom_ok"):
            if any(not 0.0 <= value <= 1.0 for value in normalized_arrays[key]):
                raise ValueError(f"{path}.records[{index}].{key}: value outside [0,1]")
        if record.get("score_ledger_sha256") != _score_ledger_sha256(normalized_arrays):
            raise ValueError(f"{path}.records[{index}]: score ledger digest mismatch")
        selected_index = min(
            range(POSE_COUNT),
            key=lambda pose_index: (normalized_arrays["confidence_rmsd"][pose_index], pose_index),
        )
        if _strict_int(record.get("selected_index"), label=f"{expected_id}.selected_index") != selected_index:
            raise ValueError(f"{path}.records[{index}]: selector recomputation mismatch")
        normalized.append({**record, "score_arrays": normalized_arrays, "selected_index": selected_index})
    replay = payload.get("replay")
    if stage == "smoke":
        if not isinstance(replay, dict) or replay.get("passed") is not True:
            raise ValueError(f"{path}: smoke replay did not pass")
        replay_fields = {
            "passed",
            "checked_count",
            "selected_index_mismatches",
            "all_scores_finite",
            "per_field_max_abs_score_delta",
            "max_abs_score_delta",
            "absolute_tolerance",
            "records",
        }
        _require_exact_fields(replay, replay_fields, label=f"{path}.replay")
        if _strict_int(replay.get("checked_count"), label=f"{path}.replay.checked_count") != len(expected_ids):
            raise ValueError(f"{path}: smoke replay count mismatch")
        if _strict_int(replay.get("selected_index_mismatches"), label=f"{path}.replay.index_mismatches") != 0:
            raise ValueError(f"{path}: smoke replay selected-index mismatch")
        if replay.get("all_scores_finite") is not True:
            raise ValueError(f"{path}: smoke replay contains non-finite scores")
        if (
            _finite_float(
                replay.get("absolute_tolerance"), label=f"{path}.replay.absolute_tolerance"
            )
            != REPLAY_ABS_TOLERANCE
        ):
            raise ValueError(f"{path}: smoke replay tolerance mismatch")
        per_field_delta = replay.get("per_field_max_abs_score_delta")
        if not isinstance(per_field_delta, dict) or set(per_field_delta) != set(
            SCORE_ARRAY_FIELDS
        ):
            raise ValueError(f"{path}: smoke replay field-delta inventory mismatch")
        normalized_deltas = {
            key: _finite_float(
                per_field_delta[key], label=f"{path}.replay.per_field_delta.{key}"
            )
            for key in SCORE_ARRAY_FIELDS
        }
        if any(value < 0.0 for value in normalized_deltas.values()):
            raise ValueError(f"{path}: smoke replay contains a negative score delta")
        max_delta = _finite_float(replay.get("max_abs_score_delta"), label=f"{path}.replay.max_delta")
        if not math.isclose(
            max_delta, max(normalized_deltas.values()), rel_tol=0.0, abs_tol=1e-15
        ):
            raise ValueError(f"{path}: smoke replay max delta does not match fields")
        if max_delta > REPLAY_ABS_TOLERANCE:
            raise ValueError(f"{path}: smoke replay score delta exceeds tolerance")
        replay_records = replay.get("records")
        if not isinstance(replay_records, list) or len(replay_records) != len(expected_ids):
            raise ValueError(f"{path}: smoke replay record inventory mismatch")
        replay_record_fields = {
            "sample_key",
            "first_selected_index",
            "replay_selected_index",
            "selected_index_stable",
            "replay_score_ledger_sha256",
        }
        for index, (replay_record, expected_id) in enumerate(
            zip(replay_records, expected_ids, strict=True)
        ):
            if not isinstance(replay_record, dict):
                raise ValueError(f"{path}.replay.records[{index}]: expected an object")
            _require_exact_fields(
                replay_record,
                replay_record_fields,
                label=f"{path}.replay.records[{index}]",
            )
            selected = normalized[index]["selected_index"]
            if (
                replay_record.get("sample_key") != expected_id
                or replay_record.get("selected_index_stable") is not True
                or _strict_int(
                    replay_record.get("first_selected_index"),
                    label=f"{path}.replay.records[{index}].first_selected_index",
                )
                != selected
                or _strict_int(
                    replay_record.get("replay_selected_index"),
                    label=f"{path}.replay.records[{index}].replay_selected_index",
                )
                != selected
            ):
                raise ValueError(f"{path}: smoke replay record identity mismatch")
            _require_sha256(
                replay_record.get("replay_score_ledger_sha256"),
                label=f"{path}.replay.records[{index}].replay_score_ledger_sha256",
            )
    elif replay != {}:
        raise ValueError(f"{path}: full score artifact unexpectedly contains replay data")
    runtime_fields = {
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
    _require_exact_fields(runtime, runtime_fields, label=f"{path}.runtime")
    if _finite_float(runtime.get("elapsed_seconds"), label=f"{path}.runtime.elapsed") < 0.0:
        raise ValueError(f"{path}: negative elapsed runtime")
    return normalized, {
        "path": str(path.resolve()),
        "sha256": _file_sha256(path),
        "arm": arm,
        "shard_index": shard_index,
        "record_count": len(normalized),
        "scorer_source_sha256": scorer_sha,
        "runtime_code_identity_sha256": runtime_code_sha,
    }


def _validate_score_bank_identity(
    score: dict[str, Any], bank: dict[str, Any], *, arm: str
) -> None:
    sample_id = str(score["sample_key"])
    scalar_keys = (
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
        "receptor_sha256",
        "processed_meta_sha256",
        "canonical_smiles",
        "ligand_input_identity_sha256",
        "num_input_atoms",
    )
    for key in scalar_keys:
        if str(score.get(key)) != str(bank.get(key)):
            raise ValueError(f"{sample_id}/{arm}: score/bank identity mismatch for {key}")
    if _strict_int(score.get("pose_count"), label=f"{sample_id}.pose_count") != POSE_COUNT:
        raise ValueError(f"{sample_id}/{arm}: pose_count mismatch")
    sigma = _finite_float(score.get("sample_sigma"), label=f"{sample_id}.sample_sigma")
    if not math.isclose(sigma, SAMPLE_SIGMA, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{sample_id}/{arm}: sample_sigma mismatch")
    if score.get("pocket_center") != bank.get("pocket_center"):
        raise ValueError(f"{sample_id}/{arm}: score/bank pocket-center mismatch")
    for asset in ("receptor", "processed_meta"):
        score_path, score_sha = _path_spec(
            score.get(asset), label=f"{sample_id}.score.{asset}"
        )
        bank_path, bank_sha = _path_spec(
            bank.get(asset), label=f"{sample_id}.bank.{asset}"
        )
        if score_path != bank_path or score_sha != bank_sha:
            raise ValueError(f"{sample_id}/{arm}: score/bank {asset} identity mismatch")
    score_sdf, score_sdf_sha = _path_spec(score.get("all_poses_sdf"), label=f"{sample_id}.score.SDF")
    bank_sdf, bank_sdf_sha = _path_spec(bank.get("all_poses_sdf"), label=f"{sample_id}.bank.SDF")
    if score_sdf != bank_sdf or score_sdf_sha != bank_sdf_sha:
        raise ValueError(f"{sample_id}/{arm}: score/bank SDF identity mismatch")


def _binary_auc(predicted_rmsd: np.ndarray, labels: np.ndarray) -> float | None:
    positive = predicted_rmsd[labels]
    negative = predicted_rmsd[~labels]
    if positive.size == 0 or negative.size == 0:
        return None
    comparisons = positive[:, None] - negative[None, :]
    return float((np.sum(comparisons < 0.0) + 0.5 * np.sum(comparisons == 0.0)) / comparisons.size)


def _joined_record(
    score: dict[str, Any],
    source: dict[str, str],
) -> dict[str, Any]:
    sample_id = str(score["sample_key"])
    rmsds = np.asarray(
        _source_float_array(
            source.get("candidate_rmsds_json"),
            label=f"{sample_id}.candidate_rmsds_json",
            length=POSE_COUNT,
        ),
        dtype=np.float64,
    )
    if bool(np.any(rmsds < 0.0)):
        raise ValueError(f"{sample_id}: negative source RMSD")
    fast_valid = np.asarray(
        _source_bool_array(
            source.get("candidate_fast_valid_json"),
            label=f"{sample_id}.candidate_fast_valid_json",
            length=POSE_COUNT,
        ),
        dtype=np.bool_,
    )
    predicted = np.asarray(score["score_arrays"]["confidence_rmsd"], dtype=np.float64)
    success_scores = np.asarray(score["score_arrays"]["confidence_success"], dtype=np.float64)
    order = np.asarray(
        sorted(range(POSE_COUNT), key=lambda index: (predicted[index], index)), dtype=np.int64
    )
    selected_index = int(score["selected_index"])
    success_index = max(
        range(POSE_COUNT), key=lambda index: (success_scores[index], -index)
    )
    success = rmsds < 2.0
    k2 = int(success.sum())
    declared_k2 = _strict_int(
        source.get("num_rmsd_lt2_candidates"), label=f"{sample_id}.declared_k2"
    )
    if declared_k2 != k2:
        raise ValueError(f"{sample_id}: frozen source K2 disagrees with RMSD ledger")
    selected_success = bool(success[selected_index])
    oracle_success = bool(success.any())
    fast_oracle = bool(np.any(success & fast_valid))
    first_correct_rank = None
    if oracle_success:
        first_correct_rank = next(
            rank for rank, pose_index in enumerate(order.tolist(), start=1) if success[pose_index]
        )
    return {
        "sample_key": sample_id,
        "system_id": str(score["system_id"]),
        "rmsds": rmsds,
        "fast_valid": fast_valid,
        "predicted": predicted,
        "success": success,
        "selected_index": selected_index,
        "success_index": success_index,
        "selected_success": selected_success,
        "success_head_success": bool(success[success_index]),
        "oracle_success": oracle_success,
        "first_success": bool(success[0]),
        "random_success_expectation": k2 / POSE_COUNT,
        "top3_success": bool(success[order[:3]].any()),
        "top5_success": bool(success[order[:5]].any()),
        "top10_success": bool(success[order[:10]].any()),
        "selected_rmsd": float(rmsds[selected_index]),
        "oracle_rmsd": float(rmsds.min()),
        "selected_fast_valid": bool(fast_valid[selected_index]),
        "selected_joint": bool(success[selected_index] and fast_valid[selected_index]),
        "fast_oracle": fast_oracle,
        "k2": k2,
        "first_correct_rank": first_correct_rank,
        "auc": _binary_auc(predicted, success),
    }


def _pct(count: float, denominator: int) -> float:
    return 100.0 * float(count) / denominator


def _arm_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    selected_count = sum(bool(row["selected_success"]) for row in rows)
    oracle_count = sum(bool(row["oracle_success"]) for row in rows)
    selection_miss_count = oracle_count - selected_count
    unreachable_count = n - oracle_count
    fast_oracle_count = sum(bool(row["fast_oracle"]) for row in rows)
    joint_count = sum(bool(row["selected_joint"]) for row in rows)
    fast_oracle_gap_count = fast_oracle_count - joint_count
    ranks = [int(row["first_correct_rank"]) for row in rows if row["first_correct_rank"]]
    aucs = [float(row["auc"]) for row in rows if row["auc"] is not None]
    k2_slices: dict[str, dict[str, Any]] = {}
    for name, predicate in (
        ("k2_0", lambda value: value == 0),
        ("k2_1_4", lambda value: 1 <= value <= 4),
        ("k2_5_9", lambda value: 5 <= value <= 9),
        ("k2_ge10", lambda value: value >= 10),
    ):
        subset = [row for row in rows if predicate(int(row["k2"]))]
        successes = sum(bool(row["selected_success"]) for row in subset)
        k2_slices[name] = {
            "count": len(subset),
            "top1_success_count": successes,
            "top1_success_pct": None if not subset else _pct(successes, len(subset)),
        }
    selected_rmsds = [float(row["selected_rmsd"]) for row in rows]
    regrets = [float(row["selected_rmsd"] - row["oracle_rmsd"]) for row in rows]
    return {
        "sample_count": n,
        "system_count": len({str(row["system_id"]) for row in rows}),
        "top1_success_count": selected_count,
        "top1_success_pct": _pct(selected_count, n),
        "oracle_success_count": oracle_count,
        "oracle_success_pct": _pct(oracle_count, n),
        "oracle_recovery_fraction": selected_count / oracle_count,
        "oracle_gap_count": selection_miss_count,
        "oracle_gap_pp": _pct(selection_miss_count, n),
        "selection_miss_count": selection_miss_count,
        "sampler_unreachable_count": unreachable_count,
        "bottleneck_delta_count": selection_miss_count - unreachable_count,
        "bottleneck_delta_pp": _pct(selection_miss_count - unreachable_count, n),
        "first_success_count": sum(bool(row["first_success"]) for row in rows),
        "first_success_pct": _pct(sum(bool(row["first_success"]) for row in rows), n),
        "random_expected_success_pct": _pct(
            sum(float(row["random_success_expectation"]) for row in rows), n
        ),
        "delta_vs_first_pp": _pct(
            sum(float(row["selected_success"]) - float(row["first_success"]) for row in rows), n
        ),
        "delta_vs_random_pp": _pct(
            sum(
                float(row["selected_success"]) - float(row["random_success_expectation"])
                for row in rows
            ),
            n,
        ),
        "top3_success_count": sum(bool(row["top3_success"]) for row in rows),
        "top3_success_pct": _pct(sum(bool(row["top3_success"]) for row in rows), n),
        "top5_success_count": sum(bool(row["top5_success"]) for row in rows),
        "top5_success_pct": _pct(sum(bool(row["top5_success"]) for row in rows), n),
        "top10_success_count": sum(bool(row["top10_success"]) for row in rows),
        "top10_success_pct": _pct(sum(bool(row["top10_success"]) for row in rows), n),
        "top5_rescue_count": sum(
            bool(row["top5_success"]) - bool(row["selected_success"]) for row in rows
        ),
        "top5_rescue_pp": _pct(
            sum(float(row["top5_success"]) - float(row["selected_success"]) for row in rows), n
        ),
        "success_head_top1_success_count": sum(
            bool(row["success_head_success"]) for row in rows
        ),
        "success_head_top1_success_pct": _pct(
            sum(bool(row["success_head_success"]) for row in rows), n
        ),
        "selected_rmsd_mean": statistics.fmean(selected_rmsds),
        "selected_rmsd_median": statistics.median(selected_rmsds),
        "selected_rmsd_lt1_pct": _pct(sum(value < 1.0 for value in selected_rmsds), n),
        "selected_rmsd_lt5_pct": _pct(sum(value < 5.0 for value in selected_rmsds), n),
        "selected_oracle_regret_mean": statistics.fmean(regrets),
        "selected_oracle_regret_median": statistics.median(regrets),
        "selected_fast_valid_pct": _pct(
            sum(bool(row["selected_fast_valid"]) for row in rows), n
        ),
        "selected_joint_lt2_fast_valid_count": joint_count,
        "selected_joint_lt2_fast_valid_pct": _pct(joint_count, n),
        "fast_valid_oracle_count": fast_oracle_count,
        "fast_valid_oracle_pct": _pct(fast_oracle_count, n),
        "fast_valid_oracle_gap_count": fast_oracle_gap_count,
        "fast_valid_oracle_gap_pp": _pct(fast_oracle_gap_count, n),
        "fast_valid_oracle_recovery_fraction": (
            None if fast_oracle_count == 0 else joint_count / fast_oracle_count
        ),
        "correct_pose_rank_median_solvable": statistics.median(ranks),
        "correct_pose_mrr_solvable": statistics.fmean(1.0 / rank for rank in ranks),
        "macro_within_complex_auc": statistics.fmean(aucs),
        "macro_auc_complex_count": len(aucs),
        "k2_slices": k2_slices,
    }


def _percentile_interval(values: np.ndarray) -> dict[str, float]:
    lower, upper = np.percentile(values, [2.5, 97.5])
    return {"lower": float(lower), "upper": float(upper)}


def _cluster_bootstrap(
    rows_by_arm: dict[str, list[dict[str, Any]]]
) -> tuple[dict[str, dict[str, dict[str, float]]], dict[str, float | dict[str, float]]]:
    primary_rows = rows_by_arm[PRIMARY_ARM]
    systems = sorted({str(row["system_id"]) for row in primary_rows})
    system_index = {system: index for index, system in enumerate(systems)}
    feature_names = (
        "count",
        "selected",
        "oracle",
        "first",
        "random",
        "bottleneck",
        "top5_gain",
        "joint",
        "fast_oracle",
        "success_head",
    )
    matrices: dict[str, np.ndarray] = {}
    for arm in ARMS:
        matrix = np.zeros((len(systems), len(feature_names)), dtype=np.float64)
        for row in rows_by_arm[arm]:
            index = system_index[str(row["system_id"])]
            values = (
                1.0,
                float(row["selected_success"]),
                float(row["oracle_success"]),
                float(row["first_success"]),
                float(row["random_success_expectation"]),
                float(row["oracle_success"])
                - float(row["selected_success"])
                - (1.0 - float(row["oracle_success"])),
                float(row["top5_success"]) - float(row["selected_success"]),
                float(row["selected_joint"]),
                float(row["fast_oracle"]),
                float(row["success_head_success"]),
            )
            matrix[index] += np.asarray(values, dtype=np.float64)
        matrices[arm] = matrix
    collected: dict[str, dict[str, list[np.ndarray]]] = {
        arm: defaultdict(list) for arm in ARMS
    }
    paired_delta: list[np.ndarray] = []
    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    remaining = BOOTSTRAP_RESAMPLES
    while remaining:
        batch = min(BOOTSTRAP_BATCH_SIZE, remaining)
        draws = rng.integers(0, len(systems), size=(batch, len(systems)))
        totals = {arm: matrices[arm][draws].sum(axis=1) for arm in ARMS}
        for arm in ARMS:
            values = totals[arm]
            denominator = values[:, 0]
            selected = values[:, 1]
            oracle = values[:, 2]
            fast_oracle = values[:, 8]
            metrics = {
                "top1_success_pct": 100.0 * selected / denominator,
                "oracle_recovery_fraction": selected / oracle,
                "oracle_gap_pp": 100.0 * (oracle - selected) / denominator,
                "delta_vs_first_pp": 100.0 * (selected - values[:, 3]) / denominator,
                "delta_vs_random_pp": 100.0 * (selected - values[:, 4]) / denominator,
                "bottleneck_delta_pp": 100.0 * values[:, 5] / denominator,
                "top5_rescue_pp": 100.0 * values[:, 6] / denominator,
                "selected_joint_lt2_fast_valid_pct": 100.0 * values[:, 7] / denominator,
                "fast_valid_oracle_recovery_fraction": np.divide(
                    values[:, 7],
                    fast_oracle,
                    out=np.zeros_like(fast_oracle),
                    where=fast_oracle > 0,
                ),
                "fast_valid_oracle_gap_pp": (
                    100.0 * (fast_oracle - values[:, 7]) / denominator
                ),
                "success_head_top1_success_pct": 100.0 * values[:, 9] / denominator,
            }
            for name, metric_values in metrics.items():
                collected[arm][name].append(metric_values)
        paired_delta.append(
            100.0
            * (totals[PRIMARY_ARM][:, 1] - totals[DIAGNOSTIC_ARM][:, 1])
            / totals[PRIMARY_ARM][:, 0]
        )
        remaining -= batch
    intervals = {
        arm: {
            name: _percentile_interval(np.concatenate(parts))
            for name, parts in collected[arm].items()
        }
        for arm in ARMS
    }
    paired_values = np.concatenate(paired_delta)
    point = 100.0 * (
        sum(bool(row["selected_success"]) for row in rows_by_arm[PRIMARY_ARM])
        - sum(bool(row["selected_success"]) for row in rows_by_arm[DIAGNOSTIC_ARM])
    ) / len(primary_rows)
    return intervals, {"point_pp": point, "ci95": _percentile_interval(paired_values)}


def _recovery_band(value: float) -> str:
    if value < 0.75:
        return "severe_selection_bottleneck"
    if value < 0.90:
        return "material_mixed_bottleneck"
    return "near_oracle_selection"


def _top5_band(value_pp: float) -> str:
    if value_pp >= 5.0:
        return "actionable_shortlist_headroom"
    if value_pp >= 2.0:
        return "modest_shortlist_headroom"
    return "little_shortlist_headroom"


def _decision(
    metrics: dict[str, Any],
    intervals: dict[str, dict[str, float]],
    paired_backbone: dict[str, Any],
) -> dict[str, Any]:
    bottleneck_ci = intervals["bottleneck_delta_pp"]
    if bottleneck_ci["lower"] > 0.0:
        dominance = "confidence_selection_dominant"
    elif bottleneck_ci["upper"] < 0.0:
        dominance = "sampler_coverage_dominant"
    else:
        dominance = "mixed_or_inconclusive"
    useful = (
        intervals["delta_vs_first_pp"]["lower"] > 0.0
        and intervals["delta_vs_random_pp"]["lower"] > 0.0
    )
    backbone_ci = paired_backbone["ci95"]
    if backbone_ci["lower"] >= -2.0 and backbone_ci["upper"] <= 2.0:
        backbone = "operationally_equivalent_within_2pp"
    elif backbone_ci["upper"] < -2.0:
        backbone = "material_s50_hidden_feature_drift"
    elif backbone_ci["lower"] > 2.0:
        backbone = "material_s50_feature_advantage"
    else:
        backbone = "backbone_effect_inconclusive"
    recovery = float(metrics["oracle_recovery_fraction"])
    if not useful:
        action = "rebuild_or_retrain_confidence_on_s50_distribution"
    elif recovery < 0.90:
        action = "prioritize_s50_matched_confidence_retraining"
    elif dominance == "sampler_coverage_dominant":
        action = "prioritize_sampler_coverage"
    else:
        action = "mixed_followup_required"
    return {
        "role": "diagnostic_only_not_model_or_selector_selection",
        "operational_ranking_signal_demonstrated": useful,
        "oracle_recovery_band": _recovery_band(recovery),
        "bottleneck_dominance": dominance,
        "top5_headroom_band": _top5_band(float(metrics["top5_rescue_pp"])),
        "paired_backbone_interpretation": backbone,
        "recommended_next_development_action": action,
    }


def _atomic_write_json_noreplace(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite report: {path}")
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


def build_report(
    *,
    stage: str,
    scores_root: Path,
    label_free_bank_manifest: Path,
    label_free_bank_manifest_sha256: str,
    eligibility_manifest: Path,
    eligibility_manifest_sha256: str,
    source_sampler_report: Path,
    source_sampler_report_sha256: str,
    source_coordinate_audit: Path,
    source_coordinate_audit_sha256: str,
    protocol_file: Path,
    protocol_sha256: str,
) -> dict[str, Any]:
    if stage not in {"smoke", "full"}:
        raise ValueError("stage must be smoke or full")
    scores_root = _canonical_existing_directory(scores_root, label="scores root")
    pins = {
        "protocol_sha256": _pin_file(protocol_file, protocol_sha256, label="protocol")[
            "sha256"
        ],
        "label_free_bank_manifest_sha256": _pin_file(
            label_free_bank_manifest,
            label_free_bank_manifest_sha256,
            label="label-free bank manifest",
        )["sha256"],
        "eligibility_manifest_sha256": _pin_file(
            eligibility_manifest,
            eligibility_manifest_sha256,
            label="eligibility manifest",
        )["sha256"],
        "source_sampler_report_sha256": _pin_file(
            source_sampler_report,
            source_sampler_report_sha256,
            label="source sampler report",
        )["sha256"],
        "source_coordinate_audit_sha256": _pin_file(
            source_coordinate_audit,
            source_coordinate_audit_sha256,
            label="source coordinate audit",
        )["sha256"],
    }
    if pins["eligibility_manifest_sha256"] != FROZEN_ELIGIBILITY_MANIFEST_SHA256:
        raise ValueError("eligibility manifest is not the frozen source artifact")
    if pins["source_sampler_report_sha256"] != FROZEN_SOURCE_REPORT_SHA256:
        raise ValueError("source sampler report is not the frozen source artifact")
    if pins["source_coordinate_audit_sha256"] != FROZEN_SOURCE_AUDIT_SHA256:
        raise ValueError("source coordinate audit is not the frozen source artifact")
    report_source_sha256 = _file_sha256(Path(__file__).resolve())

    # Eligibility and bank are explicitly label-free. In smoke mode the two
    # source reports above are byte-hash pinned only: their JSON and all source
    # results CSVs remain unopened.
    eligibility = _validate_eligibility(eligibility_manifest)
    bank_payload = _load_json_object(label_free_bank_manifest, label="label-free bank manifest")
    bank_inputs = bank_payload.get("inputs")
    if not isinstance(bank_inputs, dict) or not isinstance(
        bank_inputs.get("source_output_root"), str
    ):
        raise ValueError("label-free bank lacks a frozen source_output_root")
    source_output_root = _canonical_existing_directory(
        Path(bank_inputs["source_output_root"]), label="frozen source output root"
    )
    bank_by_id, source_shards, bank_provenance = _validate_bank_manifest(
        bank_payload,
        pins=pins,
        eligibility=eligibility,
        source_output_root=source_output_root,
    )
    if bank_provenance["report_source_sha256"] != report_source_sha256:
        raise ValueError("label-free bank report source identity mismatch")
    full_ids_by_shard = _expected_ids_by_shard(eligibility["eligible_ids"], "full")
    _validate_bank_source_shard_metadata(
        source_shards,
        expected_full_by_shard=full_ids_by_shard,
        source_output_root=source_output_root,
    )
    for index, sample_id in enumerate(eligibility["eligible_ids"]):
        _validate_bank_record_label_free(
            sample_id,
            bank_by_id[sample_id],
            expected_shard_index=index % EXPECTED_SHARDS,
            source_output_root=source_output_root,
        )
    selected_ids_by_shard = _expected_ids_by_shard(eligibility["eligible_ids"], stage)
    score_identity_pins = dict(pins)
    score_artifacts: list[dict[str, Any]] = []
    score_records_by_arm: dict[str, list[dict[str, Any]]] = {arm: [] for arm in ARMS}
    scorer_shas: set[str] = set()
    runtime_code_shas: set[str] = set()
    expected_backbone_paths = {
        arm: Path(
            str(bank_inputs[f"{arm.removesuffix('_backbone')}_backbone_checkpoint"]["path"])
        )
        for arm in ARMS
    }
    expected_confidence_path = Path(str(bank_inputs["confidence_checkpoint"]["path"]))
    for shard_index in range(EXPECTED_SHARDS):
        expected_ids = selected_ids_by_shard[shard_index]
        for arm in ARMS:
            score_path = _canonical_score_path(scores_root, stage, shard_index, arm)
            if not score_path.is_file():
                raise FileNotFoundError(f"missing required {arm} score shard: {score_path}")
            try:
                resolved_score_path = score_path.resolve(strict=True)
            except OSError as exc:
                raise ValueError(f"score artifact disappeared: {score_path}") from exc
            if resolved_score_path != score_path:
                raise ValueError(
                    f"score artifact path must be lexical-canonical without symlinks: {score_path}"
                )
            _require_within(score_path, scores_root, label="score artifact")
            records, artifact = _load_score_artifact(
                score_path,
                stage=stage,
                arm=arm,
                shard_index=shard_index,
                expected_ids=expected_ids,
                identity_pins=score_identity_pins,
                report_source_sha256=report_source_sha256,
                bank_provenance=bank_provenance,
                expected_bank_manifest_path=label_free_bank_manifest.resolve(strict=True),
                expected_backbone_path=expected_backbone_paths[arm],
                expected_confidence_path=expected_confidence_path,
            )
            score_records_by_arm[arm].extend(records)
            score_artifacts.append(artifact)
            scorer_shas.add(str(artifact["scorer_source_sha256"]))
            runtime_code_shas.add(str(artifact["runtime_code_identity_sha256"]))
    if len(scorer_shas) != 1:
        raise ValueError("score shards do not declare one identical scorer source SHA-256")
    if scorer_shas != {bank_provenance["scorer_source_sha256"]}:
        raise ValueError("score scorer-source identity differs from the frozen bank")
    if runtime_code_shas != {bank_provenance["runtime_code_identity_sha256"]}:
        raise ValueError("score runtime-code identity differs from the frozen bank")
    expected_flat_ids = [sample_id for shard in selected_ids_by_shard.values() for sample_id in shard]
    expected_flat_ids = sorted(expected_flat_ids)
    for arm in ARMS:
        if sorted(str(row["sample_key"]) for row in score_records_by_arm[arm]) != expected_flat_ids:
            raise ValueError(f"{arm}: exact score record inventory mismatch")
    scores_by_arm_id = {
        arm: {str(record["sample_key"]): record for record in score_records_by_arm[arm]}
        for arm in ARMS
    }
    sdf_audit_by_id: dict[str, dict[str, Any]] = {}
    for sample_id in expected_flat_ids:
        bank = bank_by_id[sample_id]
        sdf_audit_by_id[sample_id] = _audit_source_sdf(sample_id, bank)
        for arm in ARMS:
            score = scores_by_arm_id[arm][sample_id]
            _validate_score_bank_identity(score, bank, arm=arm)
            score_path, score_sha = _path_spec(
                score.get("all_poses_sdf"), label=f"{sample_id}/{arm}.all_poses_sdf"
            )
            audit = sdf_audit_by_id[sample_id]
            if score_path != Path(audit["path"]) or score_sha != audit["sha256"]:
                raise ValueError(f"{sample_id}/{arm}: scorer source SDF changed or was rebound")

    common = {
        "schema": REPORT_SCHEMA_VERSION,
        "schema_version": REPORT_SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": "passed" if stage == "smoke" else "complete_diagnostic",
        "stage": stage,
        "integrity_only": stage == "smoke",
        "expected_shards": EXPECTED_SHARDS,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "report_source_sha256": report_source_sha256,
        "inputs": {
            "protocol": {"path": str(protocol_file.resolve()), "sha256": pins["protocol_sha256"]},
            "label_free_bank_manifest": {
                "path": str(label_free_bank_manifest.resolve()),
                "sha256": pins["label_free_bank_manifest_sha256"],
            },
            "eligibility_manifest": {
                "path": str(eligibility_manifest.resolve()),
                "sha256": pins["eligibility_manifest_sha256"],
            },
            "source_sampler_report": {
                "path": str(source_sampler_report.resolve()),
                "sha256": pins["source_sampler_report_sha256"],
            },
            "source_coordinate_audit": {
                "path": str(source_coordinate_audit.resolve()),
                "sha256": pins["source_coordinate_audit_sha256"],
            },
            "scorer_source_sha256": next(iter(scorer_shas)),
            "runtime_code_identity_sha256": next(iter(runtime_code_shas)),
            "source_sampler_protocol_sha256": FROZEN_SOURCE_PROTOCOL_SHA256,
        },
        "configuration": {
            "arms": list(ARMS),
            "primary_arm": PRIMARY_ARM,
            "diagnostic_arm": DIAGNOSTIC_ARM,
            "selector": SELECTOR,
            "pose_count": POSE_COUNT,
            "sample_sigma": SAMPLE_SIGMA,
            "expected_shards": EXPECTED_SHARDS,
            "eligible_sample_count": FULL_ELIGIBLE_COUNT,
            "eligible_system_count": FULL_SYSTEM_COUNT,
            "bootstrap": (
                None
                if stage == "smoke"
                else {
                    "unit": "plinder_system_id",
                    "sample_weighted": True,
                    "generator": "numpy.random.PCG64",
                    "seed": BOOTSTRAP_SEED,
                    "resamples": BOOTSTRAP_RESAMPLES,
                    "interval": "percentile_95",
                }
            ),
        },
        "integrity": {
            "label_blind_score_schema": True,
            "exact_two_arm_inventory": True,
            "score_shard_count": len(score_artifacts),
            "scored_complexes_per_arm": len(expected_flat_ids),
            "scores_per_complex": POSE_COUNT,
            "finite_score_arrays_verified": list(SCORE_ARRAY_FIELDS),
            "selector_recomputed": True,
            "source_label_artifacts_opened": stage == "full",
            "source_sampler_reports_parsed": stage == "full",
            "source_csv_shards": (
                "not_opened_in_outcome_blind_smoke" if stage == "smoke" else None
            ),
            "source_sdf_files_rehashed": len(sdf_audit_by_id),
            "source_sdf_records_parsed": len(sdf_audit_by_id) * POSE_COUNT,
            "source_sdf_unmodified": True,
            "score_artifacts": score_artifacts,
        },
    }
    if stage == "smoke":
        # Keep smoke claim-free: do not join or emit any label-derived aggregate.
        common["integrity"]["smoke_replay_max_abs_tolerance"] = REPLAY_ABS_TOLERANCE
        common["integrity"]["efficacy_emitted"] = False
        return common

    source_report_payload = _load_json_object(source_sampler_report, label="source report")
    source_audit_payload = _load_json_object(source_coordinate_audit, label="source audit")
    _validate_source_reports(
        source_report_payload,
        source_audit_payload,
        eligibility_sha256=pins["eligibility_manifest_sha256"],
    )
    source_rows, source_csv_identities = _read_source_rows(
        source_shards,
        expected_full_by_shard=full_ids_by_shard,
        source_output_root=source_output_root,
    )
    if set(source_rows) != set(eligibility["eligible_ids"]):
        raise ValueError("source CSVs do not cover the exact eligible cohort")
    for sample_id in eligibility["eligible_ids"]:
        _validate_bank_source_identity(
            sample_id,
            bank_by_id[sample_id],
            source_rows[sample_id],
            source_output_root=source_output_root,
        )
    common["integrity"]["source_csv_shards"] = source_csv_identities

    joined_by_arm: dict[str, list[dict[str, Any]]] = {arm: [] for arm in ARMS}
    for arm in ARMS:
        for sample_id in eligibility["eligible_ids"]:
            joined_by_arm[arm].append(
                _joined_record(scores_by_arm_id[arm][sample_id], source_rows[sample_id])
            )
    for primary, diagnostic in zip(
        joined_by_arm[PRIMARY_ARM], joined_by_arm[DIAGNOSTIC_ARM], strict=True
    ):
        if primary["sample_key"] != diagnostic["sample_key"]:
            raise AssertionError("paired backbone row order changed")
        for key in ("rmsds", "fast_valid"):
            if not np.array_equal(primary[key], diagnostic[key]):
                raise ValueError(f"{primary['sample_key']}: paired backbone labels differ")
    metrics_by_arm = {arm: _arm_metrics(joined_by_arm[arm]) for arm in ARMS}
    intervals_by_arm, paired_backbone = _cluster_bootstrap(joined_by_arm)
    for arm in ARMS:
        metrics_by_arm[arm]["cluster_bootstrap_ci95"] = intervals_by_arm[arm]
    selected_index_agreement = sum(
        int(primary["selected_index"]) == int(diagnostic["selected_index"])
        for primary, diagnostic in zip(
            joined_by_arm[PRIMARY_ARM], joined_by_arm[DIAGNOSTIC_ARM], strict=True
        )
    )
    paired_backbone["selected_index_agreement_count"] = selected_index_agreement
    paired_backbone["selected_index_agreement_pct"] = _pct(
        selected_index_agreement, FULL_ELIGIBLE_COUNT
    )
    decisions = _decision(
        metrics_by_arm[PRIMARY_ARM],
        intervals_by_arm[PRIMARY_ARM],
        paired_backbone,
    )
    common.update(
        {
            "arms": metrics_by_arm,
            "paired_backbone": paired_backbone,
            "decision": decisions,
            "operational_full_1076_sensitivity": {
                "full_split_count": FULL_SPLIT_COUNT,
                "common_preprocessing_failure_count": FULL_EXCLUDED_COUNT,
                "assignment": "the same 41 frozen preprocessing failures are failures for both arms",
                "arms": {
                    arm: {
                        "top1_success_count": metrics_by_arm[arm]["top1_success_count"],
                        "top1_success_pct": _pct(
                            metrics_by_arm[arm]["top1_success_count"], FULL_SPLIT_COUNT
                        ),
                        "oracle_success_count": metrics_by_arm[arm]["oracle_success_count"],
                        "oracle_success_pct": _pct(
                            metrics_by_arm[arm]["oracle_success_count"], FULL_SPLIT_COUNT
                        ),
                    }
                    for arm in ARMS
                },
            },
            "claim_boundary": (
                "Repeated-use PLINDER validation distribution-shift diagnostic only; "
                "not independent confirmation and not permission to choose a backbone, "
                "selector, checkpoint, loss, or hyperparameter. External benchmarks remain closed."
            ),
        }
    )
    return common


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("smoke", "full"), required=True)
    parser.add_argument("--scores-root", type=Path, required=True)
    parser.add_argument("--label-free-bank-manifest", type=Path, required=True)
    parser.add_argument("--label-free-bank-manifest-sha256", required=True)
    parser.add_argument("--eligibility-manifest", type=Path, required=True)
    parser.add_argument("--eligibility-manifest-sha256", required=True)
    parser.add_argument("--source-sampler-report", type=Path, required=True)
    parser.add_argument("--source-sampler-report-sha256", required=True)
    parser.add_argument("--source-coordinate-audit", type=Path, required=True)
    parser.add_argument("--source-coordinate-audit-sha256", required=True)
    parser.add_argument("--protocol-file", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    report = build_report(
        stage=args.stage,
        scores_root=args.scores_root,
        label_free_bank_manifest=args.label_free_bank_manifest,
        label_free_bank_manifest_sha256=args.label_free_bank_manifest_sha256,
        eligibility_manifest=args.eligibility_manifest,
        eligibility_manifest_sha256=args.eligibility_manifest_sha256,
        source_sampler_report=args.source_sampler_report,
        source_sampler_report_sha256=args.source_sampler_report_sha256,
        source_coordinate_audit=args.source_coordinate_audit,
        source_coordinate_audit_sha256=args.source_coordinate_audit_sha256,
        protocol_file=args.protocol_file,
        protocol_sha256=args.protocol_sha256,
    )
    _atomic_write_json_noreplace(args.output_json, report)
    print(f"wrote {args.output_json} status={report['status']} stage={report['stage']}")


if __name__ == "__main__":
    main()
