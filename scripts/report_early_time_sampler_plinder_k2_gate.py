#!/usr/bin/env python3
"""Strict audit and gate for the frozen three-arm PLINDER sampler comparison."""

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
from typing import Any, Iterable

import numpy as np
from rdkit import Chem

PROTOCOL_ID = "EFFDOCK-EARLY-TIME-SAMPLER-PLINDER-K2-GATE-V1"
SCHEMA_VERSION = "effdock.early_time_sampler_plinder_k2_gate_report.v1"
ARMS = ("s25_ema", "s50_ema", "parent50k_plus10k_t0p10_ema")
BASELINE_ARM = "s50_ema"
TREATMENT_ARM = "parent50k_plus10k_t0p10_ema"
DIAGNOSTIC_ARM = "s25_ema"
REPLAY_ARM = "s50_ema_replay"

FROZEN_CHECKPOINT_SHA256 = {
    "s25_ema": "c343ebc34cea3395762cd82e1c54b8c7b847dc04c4fa9e80b9813a864cafa0e1",
    "s50_ema": "65be44d7dc8f0867eb9fc5d22214b80f93971ea4702679a527c665046e91e6b6",
    "parent50k_plus10k_t0p10_ema": (
        "0a48577379e286c584abd8c652d079b09dd6fff3c06a1a2f433d617ab0cd6074"
    ),
    "s50_ema_replay": ("65be44d7dc8f0867eb9fc5d22214b80f93971ea4702679a527c665046e91e6b6"),
}
FROZEN_PROTOCOL_SHA256 = "0250853ae0793db288be2a6a8dc775db391d25aae32835b65b061782f34ab518"
FROZEN_CONFIG_SHA256 = "39aa62e4a48ed6f3aa4ff59345fb43a81220e2baba22edfd5beb0c4981b307ec"
FROZEN_SPLIT_SHA256 = "3ac570bf08bced053f1ce040b57efca27c3be616f29a82cd66ef887c08860e6b"
FROZEN_POOL_SHA256 = "0ff455da77ce5540b839918cccb96f45414e91efff6272d7da3a65337ab1fe91"
FROZEN_RAW_GATE_SHA256 = "1ac146cfbec49ebfd1eb4452219320f134b0261bc8dc1bc196bcdab91b60f546"
FROZEN_AUDIT_SHA256 = "d30f7380186d914b60964e120280dd84470b0f67b5a8aa9548e499af0aa942bf"
FROZEN_ELIGIBLE_IDS_SHA256 = "005577bbf2b0c1c1e98bac3092b8e5350a6aa06597442b4c86d05f24e763593f"
FROZEN_EVALUATOR_SHA256 = "0cf1b0e96edfc06467a15cbe2a6f0aaed1ee62d729219caab93b134519ea07dc"
FROZEN_BENCHMARK_SHA256 = "ca98e8fe121ee82b72f1b1a5f72f890ea6a3143d35247026a117365855dad401"
FROZEN_RUNNER_SHA256 = "f003512bb600f342ad0e299185d314dd2e42616374c3003a5a9a6b8316ec5554"
FROZEN_ELIGIBILITY_MANIFEST_SHA256 = (
    "6ebeb2d165e1def6ebf7b5bba301f82d4a9c3ff9d6c5cd43616dcf09edbd38ac"
)

FULL_SPLIT_COUNT = 1_076
FULL_ELIGIBLE_COUNT = 1_035
FULL_INELIGIBLE_COUNT = 41
FULL_SYSTEM_COUNT = 1_020
FULL_SHARDS = 8
FULL_NUM_SAMPLES = 100
FULL_NUM_STEPS = 10
FULL_PRIOR_POOL_SIZE = 100
BOOTSTRAP_SEED = 20_260_815
BOOTSTRAP_RESAMPLES = 20_000
POSE_DIVERSITY_CONTRACT = "EFFDOCK_HEAVY_ATOM_RECEPTOR_FRAME_DIVERSITY_V2"
CANDIDATE_HASH_CONTRACT = "EFFDOCK_CANDIDATE_ENSEMBLE_V1"
PRIOR_HASH_CONTRACT = "EFFDOCK_SHARED_PRIOR_V1"

STAGE_DEFAULTS = {
    "smoke": {"count": 8, "shards": 1, "num_samples": 4, "num_steps": 2, "prior": 4},
    "pilot": {
        "count": 32,
        "shards": 1,
        "num_samples": 100,
        "num_steps": 10,
        "prior": 100,
    },
    "full": {
        "count": FULL_ELIGIBLE_COUNT,
        "shards": FULL_SHARDS,
        "num_samples": FULL_NUM_SAMPLES,
        "num_steps": FULL_NUM_STEPS,
        "prior": FULL_PRIOR_POOL_SIZE,
    },
}

_SHA256_HEX = frozenset("0123456789abcdef")


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path}: invalid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _strict_int(value: object, *, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label}: expected an integer, got {value!r}")
    try:
        parsed = int(str(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}: expected an integer, got {value!r}") from exc
    if str(parsed) != str(value).strip():
        raise ValueError(f"{label}: expected an exact integer, got {value!r}")
    return parsed


def _finite_float(value: object, *, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label}: expected a finite number, got {value!r}")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}: expected a finite number, got {value!r}") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{label}: expected a finite number, got {value!r}")
    return parsed


def _strict_bool(value: object, *, label: str) -> bool:
    if isinstance(value, bool):
        return value
    if value in ("True", "False"):
        return value == "True"
    raise ValueError(f"{label}: expected an exact boolean, got {value!r}")


def _sha256(value: object, *, label: str) -> str:
    parsed = str(value).strip().lower()
    if len(parsed) != 64 or any(char not in _SHA256_HEX for char in parsed):
        raise ValueError(f"{label}: expected a SHA-256 digest, got {value!r}")
    return parsed


def _require_sha(value: object, expected: str, *, label: str) -> str:
    parsed = _sha256(value, label=label)
    if parsed != expected:
        raise ValueError(f"{label}: expected frozen SHA-256 {expected}, got {parsed}")
    return parsed


def _require_float(value: object, expected: float, *, label: str) -> float:
    parsed = _finite_float(value, label=label)
    if not math.isclose(parsed, expected, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{label}: expected {expected}, got {parsed}")
    return parsed


def _parse_json_list(raw: object, *, label: str) -> list[Any]:
    try:
        value = json.loads(str(raw))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label}: invalid JSON") from exc
    if not isinstance(value, list):
        raise ValueError(f"{label}: expected a JSON list")
    return value


def _ids_sha256(ids: Iterable[str]) -> str:
    payload = "".join(f"{item}\n" for item in sorted(ids)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _versioned_ids_sha256(ids: Iterable[str]) -> str:
    digest = hashlib.sha256()
    digest.update(b"EFFDOCK_SORTED_COMPLEX_IDS_V1\0")
    for sample_id in ids:
        digest.update(sample_id.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _split_sample_key(sample_id: str, *, label: str) -> tuple[str, str]:
    if "__" not in sample_id:
        raise ValueError(f"{label}: expected '<system_id>__<ligand_chain>'")
    system_id, ligand_chain = sample_id.rsplit("__", 1)
    if not system_id or not ligand_chain:
        raise ValueError(f"{label}: expected nonempty system and ligand-chain identities")
    return system_id, ligand_chain


def _summary_csv_path(summary_path: Path, summary: dict[str, Any], root: Path) -> Path:
    declared = str(summary.get("csv", "")).strip()
    candidates: list[Path] = []
    if declared:
        path = Path(declared)
        candidates.append(path)
        if not path.is_absolute():
            candidates.extend((summary_path.parent / path, root / path))
    candidates.append(
        summary_path.with_name(summary_path.name.removesuffix(".summary.json") + ".csv")
    )
    for path in candidates:
        if path.is_file():
            return path
    raise ValueError(f"{summary_path}: declared CSV does not exist")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_file(raw: object, *, summary_path: Path, root: Path, label: str) -> Path:
    declared = Path(str(raw))
    candidates = [declared]
    if not declared.is_absolute():
        candidates.extend((summary_path.parent / declared, root / declared, Path.cwd() / declared))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise ValueError(f"{label}: declared file does not exist: {raw!r}")


def _resolve_output_file(raw: object, *, source: Path, root: Path, label: str) -> Path:
    """Resolve a declared output without permitting symlink or ``..`` escape."""
    value = str(raw).strip()
    if not value:
        raise ValueError(f"{label}: output path is empty")
    declared = Path(value)
    candidates = [declared]
    if not declared.is_absolute():
        candidates.extend((source.parent / declared, root / declared, Path.cwd() / declared))
    root_resolved = root.resolve(strict=True)
    for candidate in candidates:
        try:
            resolved = candidate.resolve(strict=True)
        except OSError:
            continue
        if not resolved.is_file():
            continue
        try:
            resolved.relative_to(root_resolved)
        except ValueError as exc:
            raise ValueError(f"{label}: output resolves outside report root: {resolved}") from exc
        return resolved
    raise ValueError(f"{label}: declared output file does not exist: {value!r}")


def _sdf_property(molecule: Chem.Mol, name: str, *, label: str) -> str:
    if not molecule.HasProp(name):
        raise ValueError(f"{label}: SDF record is missing property {name!r}")
    return molecule.GetProp(name)


def _validate_all_poses_sdf(
    path: Path,
    *,
    expected_sha256: str,
    expected_num_samples: int,
    sampling_seed: int,
    ligand_conformer_seed: int,
    candidate_ensemble_sha256: str,
    label: str,
) -> None:
    actual_sha256 = _file_sha256(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"{label}: all-pose SDF SHA-256 mismatch: "
            f"declared={expected_sha256}, actual={actual_sha256}"
        )
    record_count = 0
    with path.open("rb") as handle:
        supplier = Chem.ForwardSDMolSupplier(handle)
        for record_index, molecule in enumerate(supplier):
            record_label = f"{label}.record[{record_index}]"
            if molecule is None:
                raise ValueError(f"{record_label}: RDKit could not parse the SDF record")
            if record_index >= expected_num_samples:
                raise ValueError(
                    f"{label}: expected {expected_num_samples} sequential SDF records, found more"
                )
            sample_index = _strict_int(
                _sdf_property(molecule, "sample_index", label=record_label),
                label=f"{record_label}.sample_index",
            )
            if sample_index != record_index:
                raise ValueError(
                    f"{record_label}.sample_index: expected ordered index {record_index}, "
                    f"got {sample_index}"
                )
            record_sampling_seed = _strict_int(
                _sdf_property(molecule, "sampling_seed", label=record_label),
                label=f"{record_label}.sampling_seed",
            )
            if record_sampling_seed != sampling_seed:
                raise ValueError(
                    f"{record_label}.sampling_seed: expected {sampling_seed}, "
                    f"got {record_sampling_seed}"
                )
            record_conformer_seed = _strict_int(
                _sdf_property(molecule, "ligand_conformer_seed", label=record_label),
                label=f"{record_label}.ligand_conformer_seed",
            )
            if record_conformer_seed != ligand_conformer_seed:
                raise ValueError(
                    f"{record_label}.ligand_conformer_seed: expected {ligand_conformer_seed}, "
                    f"got {record_conformer_seed}"
                )
            record_candidate_hash = _sdf_property(
                molecule, "candidate_ensemble_sha256", label=record_label
            )
            if record_candidate_hash != candidate_ensemble_sha256:
                raise ValueError(f"{record_label}.candidate_ensemble_sha256: differs from CSV row")
            record_count += 1
    if record_count != expected_num_samples:
        raise ValueError(
            f"{label}: expected {expected_num_samples} sequential SDF records, found {record_count}"
        )


def _expected_arms(stage: str) -> tuple[str, ...]:
    return (*ARMS, REPLAY_ARM) if stage in {"smoke", "pilot"} else ARMS


def _validate_eligibility_manifest_file(path: Path, *, label: str) -> dict[str, Any]:
    payload = _load_json_object(path)
    if payload.get("schema_version") != "effdock.plinder_checkpoint_eligibility.v1":
        raise ValueError(f"{label}: wrong eligibility-manifest schema")
    if payload.get("protocol_id") != PROTOCOL_ID or payload.get("status") != "complete":
        raise ValueError(f"{label}: eligibility manifest is not complete for this protocol")
    inventory = payload.get("inventory")
    if not isinstance(inventory, dict):
        raise ValueError(f"{label}.inventory: expected an object")
    count_fields = {
        "full_count": FULL_SPLIT_COUNT,
        "eligible_count": FULL_ELIGIBLE_COUNT,
        "eligible_system_count": FULL_SYSTEM_COUNT,
        "excluded_count": FULL_INELIGIBLE_COUNT,
        "preflight_error_count": 0,
    }
    for field, expected in count_fields.items():
        if _strict_int(inventory.get(field), label=f"{label}.inventory.{field}") != expected:
            raise ValueError(f"{label}.inventory.{field}: expected {expected}")
    if inventory.get("preflight_error_ids") != []:
        raise ValueError(f"{label}.inventory.preflight_error_ids: expected an empty list")

    id_lists: dict[str, list[str]] = {}
    for name, expected_count in (
        ("full_ids", FULL_SPLIT_COUNT),
        ("eligible_ids", FULL_ELIGIBLE_COUNT),
        ("excluded_ids", FULL_INELIGIBLE_COUNT),
    ):
        values = inventory.get(name)
        if (
            not isinstance(values, list)
            or len(values) != expected_count
            or any(not isinstance(value, str) or not value for value in values)
            or values != sorted(values)
            or len(set(values)) != expected_count
        ):
            raise ValueError(f"{label}.inventory.{name}: expected {expected_count} sorted unique IDs")
        id_lists[name] = list(values)
        digest_name = f"{name}_sha256"
        if inventory.get(digest_name) != _versioned_ids_sha256(values):
            raise ValueError(f"{label}.inventory.{digest_name}: mismatch")
    if sorted(id_lists["eligible_ids"] + id_lists["excluded_ids"]) != id_lists["full_ids"]:
        raise ValueError(f"{label}: eligible and excluded IDs do not partition the full cohort")
    if inventory.get("eligible_ids_newline_sha256") != FROZEN_ELIGIBLE_IDS_SHA256:
        raise ValueError(f"{label}: frozen eligible newline digest mismatch")
    eligible_systems = {
        _split_sample_key(sample_id, label=f"{label}.inventory.eligible_ids")[0]
        for sample_id in id_lists["eligible_ids"]
    }
    if len(eligible_systems) != FULL_SYSTEM_COUNT:
        raise ValueError(
            f"{label}: eligible IDs contain {len(eligible_systems)} systems; "
            f"expected {FULL_SYSTEM_COUNT}"
        )
    return {
        "path": str(path.resolve()),
        "full_ids": id_lists["full_ids"],
        "eligible_ids": id_lists["eligible_ids"],
        "excluded_ids": id_lists["excluded_ids"],
        "excluded_ids_sha256": _versioned_ids_sha256(id_lists["excluded_ids"]),
    }


def _validate_paired_summary(
    summary: dict[str, Any],
    *,
    path: Path,
    root: Path,
    stage: str,
    expected_count: int,
    expected_shards: int,
    expected_num_samples: int,
    expected_num_steps: int,
    expected_prior_pool_size: int,
) -> dict[str, Any]:
    label = str(path)
    exact_values = {
        "schema_version": "effdock.plinder_checkpoint_paired_shard.v1",
        "protocol_id": PROTOCOL_ID,
        "status": "complete",
        "mode": stage,
    }
    for key, expected in exact_values.items():
        raw = summary.get(key)
        if raw != expected:
            raise ValueError(f"{label}.{key}: expected {expected!r}, got {raw!r}")

    settings = summary.get("settings")
    if not isinstance(settings, dict):
        raise ValueError(f"{label}.settings: expected an object")
    selected_setting = None if stage == "full" else expected_count
    setting_values: dict[str, object] = {
        "stage": stage,
        "selected_count": selected_setting,
        "num_samples": expected_num_samples,
        "num_steps": expected_num_steps,
        "model_pose_step_budget": expected_num_samples * expected_num_steps,
        "prior_pool_size": expected_prior_pool_size,
        "time_schedule": "late",
        "confidence": False,
        "refine": "none",
        "selector_profile": "candidate_only",
        "ligand_conformer_seed": 0,
        "include_s50_replay": stage in {"smoke", "pilot"},
    }
    for key, expected in setting_values.items():
        if settings.get(key) != expected:
            raise ValueError(
                f"{label}.settings.{key}: expected {expected!r}, got {settings.get(key)!r}"
            )
    _require_float(settings.get("sigma"), 2.0, label=f"{label}.settings.sigma")
    _require_float(settings.get("schedule_power"), 3.0, label=f"{label}.settings.schedule_power")
    _require_float(
        settings.get("pocket_cutoff_angstrom"),
        10.0,
        label=f"{label}.settings.pocket_cutoff_angstrom",
    )
    _require_float(
        settings.get("center_jitter_sigma"),
        0.0,
        label=f"{label}.settings.center_jitter_sigma",
    )
    _require_float(
        settings.get("translation_sde_base_sigma"),
        0.0,
        label=f"{label}.settings.translation_sde_base_sigma",
    )
    for key in ("vina_guidance_scale", "unified_guidance_scale", "fk_constraint_beta"):
        _require_float(settings.get(key), 0.0, label=f"{label}.settings.{key}")

    seed_contract = summary.get("seed_contract")
    expected_seed_contract = {
        "name": "BASE42_PLUS_SORTED_FULL_VAL_GLOBAL_INDEX_1_BASED_V1",
        "base_seed": 42,
        "order": "globally sorted full 1076-ID validation cohort before eligibility",
    }
    if seed_contract != expected_seed_contract:
        raise ValueError(f"{label}.seed_contract: differs from frozen seed policy")
    ligand_contract = summary.get("ligand_input_contract")
    expected_ligand_contract = {
        "source": "data/plinder_pool.parquet:ligand_rdkit_canonical_smiles",
        "conformer_seed": 0,
        "heavy_atom_normalization": "RemoveHs_then_RemoveAllHs",
        "crystal_sdf_role": "RMSD reference and atom-mapping eligibility only",
        "crystal_sdf_input_fallback": False,
    }
    if ligand_contract != expected_ligand_contract:
        raise ValueError(f"{label}.ligand_input_contract: differs from frozen input policy")

    expected_arm_names = list(_expected_arms(stage))
    arms = summary.get("arms")
    if not isinstance(arms, list) or [arm.get("name") for arm in arms] != expected_arm_names:
        raise ValueError(f"{label}.arms: expected exact ordered arms {expected_arm_names}")
    for arm in arms:
        name = str(arm["name"])
        _require_sha(
            arm.get("checkpoint_sha256"),
            FROZEN_CHECKPOINT_SHA256[name],
            label=f"{label}.arms[{name}].checkpoint_sha256",
        )

    fixed = summary.get("fixed_identities")
    if not isinstance(fixed, dict):
        raise ValueError(f"{label}.fixed_identities: expected an object")
    frozen_assets = {
        "protocol_document": FROZEN_PROTOCOL_SHA256,
        "split": FROZEN_SPLIT_SHA256,
        "pool_parquet": FROZEN_POOL_SHA256,
        "config": FROZEN_CONFIG_SHA256,
        "raw_gate": FROZEN_RAW_GATE_SHA256,
        "conformer_mapping_audit": FROZEN_AUDIT_SHA256,
    }
    for key, expected in frozen_assets.items():
        value = fixed.get(key)
        if not isinstance(value, dict):
            raise ValueError(f"{label}.fixed_identities.{key}: expected an object")
        _require_sha(value.get("sha256"), expected, label=f"{label}.fixed_identities.{key}.sha256")
    checkpoints = fixed.get("checkpoints")
    if not isinstance(checkpoints, dict) or set(checkpoints) != set(ARMS):
        raise ValueError(f"{label}.fixed_identities.checkpoints: wrong arm inventory")
    for arm in ARMS:
        value = checkpoints[arm]
        if not isinstance(value, dict):
            raise ValueError(f"{label}.fixed_identities.checkpoints[{arm}]: malformed")
        _require_sha(
            value.get("sha256"),
            FROZEN_CHECKPOINT_SHA256[arm],
            label=f"{label}.fixed_identities.checkpoints[{arm}].sha256",
        )

    code = fixed.get("code")
    if not isinstance(code, dict) or code.get("contract") != (
        "EFFDOCK_PLINDER_PAIRED_CODE_INVENTORY_V1"
    ):
        raise ValueError(f"{label}.fixed_identities.code: wrong code identity contract")
    code_sha256 = _sha256(code.get("sha256"), label=f"{label}.fixed_identities.code.sha256")
    code_files = code.get("files")
    if not isinstance(code_files, dict):
        raise ValueError(f"{label}.fixed_identities.code.files: expected an object")
    expected_code_files = {
        "scripts/run_plinder_checkpoint_paired_validation.py",
        "src/effdock/workflows/evaluate.py",
        "src/effdock/workflows/benchmark_inputs.py",
        "src/effdock/evaluation/benchmark.py",
        "src/effdock/inference/docking.py",
        "src/effdock/inference/preprocess.py",
        "src/effdock/inference/sampler.py",
    }
    if set(code_files) != expected_code_files:
        raise ValueError(f"{label}.fixed_identities.code.files: wrong source inventory")
    declared_file_hashes: dict[str, str] = {}
    for source_name, identity in code_files.items():
        if not isinstance(identity, dict):
            raise ValueError(f"{label}.code.files[{source_name!r}]: malformed")
        declared_file_hashes[source_name] = _sha256(
            identity.get("sha256"), label=f"{label}.code.files[{source_name}].sha256"
        )
    code_digest = hashlib.sha256()
    code_digest.update(b"EFFDOCK_PLINDER_PAIRED_CODE_INVENTORY_V1\0")
    code_digest.update(
        json.dumps(
            declared_file_hashes,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    )
    if code_digest.hexdigest() != code_sha256:
        raise ValueError(f"{label}.fixed_identities.code.sha256: inventory digest mismatch")
    source_hashes = {
        "scripts/run_plinder_checkpoint_paired_validation.py": FROZEN_RUNNER_SHA256,
        "src/effdock/workflows/evaluate.py": FROZEN_EVALUATOR_SHA256,
        "src/effdock/evaluation/benchmark.py": FROZEN_BENCHMARK_SHA256,
    }
    for source_name, expected in source_hashes.items():
        if declared_file_hashes[source_name] != expected:
            raise ValueError(f"{label}.code.files[{source_name}].sha256: not launch-frozen")

    eligibility = summary.get("eligibility_manifest")
    if not isinstance(eligibility, dict):
        raise ValueError(f"{label}.eligibility_manifest: expected an object")
    eligibility_sha256 = _sha256(
        eligibility.get("sha256"), label=f"{label}.eligibility_manifest.sha256"
    )
    if eligibility_sha256 != FROZEN_ELIGIBILITY_MANIFEST_SHA256:
        raise ValueError(f"{label}: eligibility manifest differs from frozen preflight")
    eligibility_path = _resolve_file(
        eligibility.get("path"),
        summary_path=path,
        root=root,
        label=f"{label}.eligibility_manifest.path",
    )
    actual_eligibility_sha256 = _file_sha256(eligibility_path)
    if actual_eligibility_sha256 != eligibility_sha256:
        raise ValueError(
            f"{label}: eligibility manifest SHA-256 mismatch: "
            f"declared={eligibility_sha256}, actual={actual_eligibility_sha256}"
        )
    eligibility_file = _validate_eligibility_manifest_file(
        eligibility_path,
        label=f"{label}.eligibility_manifest.file",
    )
    if (
        _strict_int(
            eligibility.get("eligible_count"), label=f"{label}.eligibility_manifest.eligible_count"
        )
        != FULL_ELIGIBLE_COUNT
    ):
        raise ValueError(f"{label}: expected {FULL_ELIGIBLE_COUNT} eligible inputs")
    if eligibility.get("eligible_ids_newline_sha256") != FROZEN_ELIGIBLE_IDS_SHA256:
        raise ValueError(f"{label}: frozen eligible newline digest mismatch")
    if (
        _strict_int(
            eligibility.get("eligible_system_count"),
            label=f"{label}.eligibility_manifest.eligible_system_count",
        )
        != FULL_SYSTEM_COUNT
    ):
        raise ValueError(f"{label}: expected {FULL_SYSTEM_COUNT} eligible systems")
    if (
        _strict_int(
            eligibility.get("ineligible_count"),
            label=f"{label}.eligibility_manifest.ineligible_count",
        )
        != FULL_INELIGIBLE_COUNT
    ):
        raise ValueError(f"{label}: expected {FULL_INELIGIBLE_COUNT} ineligible inputs")

    inventory = summary.get("inventory")
    if not isinstance(inventory, dict):
        raise ValueError(f"{label}.inventory: expected an object")
    counts = {
        "full_count": FULL_SPLIT_COUNT,
        "eligible_count": FULL_ELIGIBLE_COUNT,
        "selected_count": expected_count,
        "num_shards": expected_shards,
    }
    for key, expected in counts.items():
        if _strict_int(inventory.get(key), label=f"{label}.inventory.{key}") != expected:
            raise ValueError(f"{label}.inventory.{key}: expected {expected}")
    shard_index = _strict_int(inventory.get("shard_index"), label=f"{label}.inventory.shard_index")
    assigned_count = _strict_int(
        inventory.get("assigned_count"), label=f"{label}.inventory.assigned_count"
    )
    assigned_ids = inventory.get("assigned_ids")
    selected_ids = inventory.get("selected_ids")
    if not isinstance(assigned_ids, list) or not all(
        isinstance(value, str) and value for value in assigned_ids
    ):
        raise ValueError(f"{label}.inventory.assigned_ids: malformed")
    if not isinstance(selected_ids, list) or len(selected_ids) != expected_count:
        raise ValueError(f"{label}.inventory.selected_ids: malformed")
    if not all(isinstance(value, str) and value for value in selected_ids):
        raise ValueError(f"{label}.inventory.selected_ids: malformed")
    if len(assigned_ids) != assigned_count or len(set(assigned_ids)) != assigned_count:
        raise ValueError(f"{label}.inventory.assigned_ids: count/uniqueness mismatch")
    if inventory.get("selected_ids_sha256") != _versioned_ids_sha256(selected_ids):
        raise ValueError(f"{label}.inventory.selected_ids_sha256: mismatch")
    if inventory.get("assigned_ids_sha256") != _versioned_ids_sha256(assigned_ids):
        raise ValueError(f"{label}.inventory.assigned_ids_sha256: mismatch")
    if assigned_ids != selected_ids[shard_index::expected_shards]:
        raise ValueError(f"{label}.inventory.assigned_ids: wrong fixed shard assignment")
    expected_selected_ids = eligibility_file["eligible_ids"][:expected_count]
    if selected_ids != expected_selected_ids:
        raise ValueError(f"{label}.inventory.selected_ids: not the frozen stage cohort")
    if not set(selected_ids).isdisjoint(eligibility_file["excluded_ids"]):
        raise ValueError(f"{label}.inventory.selected_ids: includes a preprocessing failure")
    success_counts = inventory.get("arm_success_counts")
    if not isinstance(success_counts, dict) or success_counts != {
        arm: assigned_count for arm in expected_arm_names
    }:
        raise ValueError(f"{label}.inventory.arm_success_counts: incomplete arm output")

    if summary.get("failures") != []:
        raise ValueError(f"{label}: runtime failures are forbidden")
    operational = summary.get("operational_inventory")
    if not isinstance(operational, dict):
        raise ValueError(f"{label}.operational_inventory: expected an object")
    operational_counts = {
        "requested_count": FULL_SPLIT_COUNT,
        "evaluable_count": FULL_ELIGIBLE_COUNT,
        "common_preprocessing_failure_count": FULL_INELIGIBLE_COUNT,
    }
    for field, expected in operational_counts.items():
        if _strict_int(
            operational.get(field), label=f"{label}.operational_inventory.{field}"
        ) != expected:
            raise ValueError(f"{label}.operational_inventory.{field}: expected {expected}")
    excluded_ids = operational.get("common_preprocessing_failure_ids")
    if excluded_ids != eligibility_file["excluded_ids"]:
        raise ValueError(f"{label}.operational_inventory: common failure IDs differ from manifest")
    excluded_ids_sha256 = _sha256(
        operational.get("common_preprocessing_failure_ids_sha256"),
        label=f"{label}.operational_inventory.common_preprocessing_failure_ids_sha256",
    )
    if excluded_ids_sha256 != eligibility_file["excluded_ids_sha256"]:
        raise ValueError(f"{label}.operational_inventory: common failure ID digest mismatch")
    expected_failure_counts = {arm: FULL_INELIGIBLE_COUNT for arm in expected_arm_names}
    if operational.get("per_arm_preprocessing_failure_count") != expected_failure_counts:
        raise ValueError(f"{label}.operational_inventory: wrong per-arm failure accounting")
    if operational.get("operational_sensitivity_assignment") != (
        "common preprocessing failures have K2=0"
    ):
        raise ValueError(f"{label}.operational_inventory: wrong zero-assignment contract")
    paired_gate = summary.get("paired_identity_gate")
    if not isinstance(paired_gate, dict) or paired_gate.get("passed") is not True:
        raise ValueError(f"{label}: runner paired identity gate did not pass")
    if (
        _strict_int(
            paired_gate.get("checked_count"), label=f"{label}.paired_identity_gate.checked_count"
        )
        != assigned_count
    ):
        raise ValueError(f"{label}: paired identity checked count mismatch")
    expected_pair_fields = [
        "sampling_seed",
        "ligand_conformer_seed",
        "prior_pool_sha256",
        "protein_sha256",
        "ligand_reference_sha256",
        "ligand_input_identity_sha256",
    ]
    if paired_gate.get("fields") != expected_pair_fields:
        raise ValueError(f"{label}: paired identity field contract mismatch")
    replay_gate = summary.get("replay_integrity_gate")
    if not isinstance(replay_gate, dict) or replay_gate.get("passed") is not True:
        raise ValueError(f"{label}: runner replay integrity gate did not pass")

    return {
        "shard_index": shard_index,
        "assigned_count": assigned_count,
        "assigned_ids": list(assigned_ids),
        "selected_ids": list(selected_ids),
        "full_ids": eligibility_file["full_ids"],
        "eligibility_manifest_sha256": eligibility_sha256,
        "eligibility_manifest_path": eligibility_file["path"],
        "operational_failure_ids_sha256": excluded_ids_sha256,
        "code_sha256": code_sha256,
        "artifacts": summary.get("artifacts"),
        "replay_integrity_gate": replay_gate,
    }


def _parse_row(
    row: dict[str, str],
    *,
    source: Path,
    root: Path,
    expected_arm: str,
    expected_num_samples: int,
    expected_prior_pool_size: int,
    expected_global_index_by_id: dict[str, int],
) -> dict[str, Any]:
    sample_id = str(row.get("id", "")).strip()
    if not sample_id:
        raise ValueError(f"{source}: row has no sample ID")
    label = f"{source}:{expected_arm}/{sample_id}"
    expected_system_id, expected_ligand_chain = _split_sample_key(
        sample_id, label=f"{label}.id"
    )
    if row.get("arm") != expected_arm:
        raise ValueError(f"{label}.arm: expected {expected_arm!r}")
    system_id = str(row.get("plinder_system_id", "")).strip()
    if system_id != expected_system_id:
        raise ValueError(
            f"{label}.plinder_system_id: expected sample-key system {expected_system_id!r}"
        )
    ligand_chain = str(row.get("plinder_ligand_chain", "")).strip()
    if ligand_chain != expected_ligand_chain:
        raise ValueError(
            f"{label}.plinder_ligand_chain: expected sample-key chain {expected_ligand_chain!r}"
        )
    global_index = _strict_int(
        row.get("plinder_global_index"), label=f"{label}.plinder_global_index"
    )
    expected_global_index = expected_global_index_by_id.get(sample_id)
    if expected_global_index is None:
        raise ValueError(f"{label}.id: absent from the frozen full cohort")
    if global_index != expected_global_index:
        raise ValueError(
            f"{label}.plinder_global_index: expected frozen 1-based index "
            f"{expected_global_index}, got {global_index}"
        )
    sampling_seed = _strict_int(row.get("sampling_seed"), label=f"{label}.sampling_seed")
    if sampling_seed != 42 + global_index:
        raise ValueError(
            f"{label}.sampling_seed: expected 42 + one-based global index, got {sampling_seed}"
        )
    ligand_conformer_seed = _strict_int(
        row.get("ligand_conformer_seed"), label=f"{label}.ligand_conformer_seed"
    )
    if ligand_conformer_seed != 0:
        raise ValueError(f"{label}.ligand_conformer_seed: expected 0")
    if _strict_int(row.get("num_samples"), label=f"{label}.num_samples") != (expected_num_samples):
        raise ValueError(f"{label}.num_samples: wrong candidate count")
    if _strict_int(row.get("prior_pool_size"), label=f"{label}.prior_pool_size") != (
        expected_prior_pool_size
    ):
        raise ValueError(f"{label}.prior_pool_size: wrong prior count")

    raw_rmsds = _parse_json_list(
        row.get("candidate_rmsds_json"), label=f"{label}.candidate_rmsds_json"
    )
    if len(raw_rmsds) != expected_num_samples:
        raise ValueError(f"{label}.candidate_rmsds_json: expected {expected_num_samples} values")
    rmsds = [
        _finite_float(value, label=f"{label}.candidate_rmsds_json[{index}]")
        for index, value in enumerate(raw_rmsds)
    ]
    if any(value < 0.0 for value in rmsds):
        raise ValueError(f"{label}: candidate RMSDs must be nonnegative")
    raw_rmsd_methods = _parse_json_list(
        row.get("candidate_rmsd_method_json"),
        label=f"{label}.candidate_rmsd_method_json",
    )
    allowed_rmsd_methods = {
        "rdkit_calc_rms_symmetry_no_align",
        "mapped_index_fallback",
    }
    if len(raw_rmsd_methods) != expected_num_samples or any(
        method not in allowed_rmsd_methods for method in raw_rmsd_methods
    ):
        raise ValueError(
            f"{label}.candidate_rmsd_method_json: expected one recognized method per candidate"
        )
    rmsd_methods = [str(method) for method in raw_rmsd_methods]
    fallback_count = sum(method == "mapped_index_fallback" for method in rmsd_methods)
    if (
        _strict_int(
            row.get("num_mapped_index_rmsd_fallback_candidates"),
            label=f"{label}.num_mapped_index_rmsd_fallback_candidates",
        )
        != fallback_count
    ):
        raise ValueError(f"{label}: RMSD fallback count differs from ordered method ledger")

    raw_fast_valid = _parse_json_list(
        row.get("candidate_fast_valid_json"),
        label=f"{label}.candidate_fast_valid_json",
    )
    if len(raw_fast_valid) != expected_num_samples or any(
        not isinstance(value, bool) for value in raw_fast_valid
    ):
        raise ValueError(
            f"{label}.candidate_fast_valid_json: expected {expected_num_samples} booleans"
        )
    fast_valid = [bool(value) for value in raw_fast_valid]
    k2 = sum(rmsd < 2.0 for rmsd in rmsds)
    fv2 = sum(rmsd < 2.0 and valid for rmsd, valid in zip(rmsds, fast_valid))
    declared_counts = {
        "num_rmsd_lt2_candidates": k2,
        "num_fast_valid_candidates": sum(fast_valid),
        "num_fast_valid_rmsd_lt2_candidates": fv2,
    }
    for key, expected in declared_counts.items():
        if _strict_int(row.get(key), label=f"{label}.{key}") != expected:
            raise ValueError(f"{label}.{key}: differs from ordered candidate vectors")
    fraction_k2 = _finite_float(
        row.get("fraction_rmsd_lt2_candidates"),
        label=f"{label}.fraction_rmsd_lt2_candidates",
    )
    if not math.isclose(fraction_k2, k2 / expected_num_samples, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{label}: K2 fraction is inconsistent")
    first_index = _strict_int(row.get("first_index"), label=f"{label}.first_index")
    if first_index != 0:
        raise ValueError(f"{label}.first_index: candidate-only contract requires 0")
    selected_index = _strict_int(row.get("selected_index"), label=f"{label}.selected_index")
    if selected_index != 0:
        raise ValueError(f"{label}.selected_index: candidate-only contract requires 0")
    first_rmsd = _finite_float(row.get("first_rmsd"), label=f"{label}.first_rmsd")
    selected_rmsd = _finite_float(row.get("selected_rmsd"), label=f"{label}.selected_rmsd")
    oracle_rmsd = _finite_float(row.get("oracle_rmsd"), label=f"{label}.oracle_rmsd")
    mean_sample_rmsd = _finite_float(row.get("mean_sample_rmsd"), label=f"{label}.mean_sample_rmsd")
    if not math.isclose(first_rmsd, rmsds[0], rel_tol=0.0, abs_tol=1e-10):
        raise ValueError(f"{label}.first_rmsd: differs from ordered RMSD vector")
    if not math.isclose(selected_rmsd, rmsds[0], rel_tol=0.0, abs_tol=1e-10):
        raise ValueError(f"{label}.selected_rmsd: differs from ordered RMSD vector")
    if not math.isclose(oracle_rmsd, min(rmsds), rel_tol=0.0, abs_tol=1e-10):
        raise ValueError(f"{label}.oracle_rmsd: differs from RMSD minimum")
    if not math.isclose(
        mean_sample_rmsd,
        math.fsum(rmsds) / expected_num_samples,
        rel_tol=0.0,
        abs_tol=1e-10,
    ):
        raise ValueError(f"{label}.mean_sample_rmsd: differs from ordered RMSD vector")

    if not _strict_bool(
        row.get("full_heavy_atom_bijection"), label=f"{label}.full_heavy_atom_bijection"
    ):
        raise ValueError(f"{label}: full-heavy-atom mapping is required")
    if row.get("selector_profile") != "candidate_only":
        raise ValueError(f"{label}.selector_profile: expected candidate_only")
    if row.get("guidance_mode") != "none":
        raise ValueError(f"{label}.guidance_mode: expected none")
    if row.get("sampling_dynamics") != "deterministic_ode":
        raise ValueError(f"{label}.sampling_dynamics: expected deterministic_ode")
    _require_float(
        row.get("translation_sde_base_sigma"),
        0.0,
        label=f"{label}.translation_sde_base_sigma",
    )

    if row.get("pose_diversity_contract") != POSE_DIVERSITY_CONTRACT:
        raise ValueError(f"{label}: wrong pose-diversity contract")
    round_decimals = _strict_int(
        row.get("pose_diversity_round_decimals"),
        label=f"{label}.pose_diversity_round_decimals",
    )
    if round_decimals != 3:
        raise ValueError(f"{label}: unique-pose rounding must be 0.001 Angstrom")
    unique_count = _strict_int(
        row.get("coordinate_unique_count"), label=f"{label}.coordinate_unique_count"
    )
    c2 = _strict_int(
        row.get("c2_connected_component_count"),
        label=f"{label}.c2_connected_component_count",
    )
    heavy_atom_count = _strict_int(
        row.get("diversity_heavy_atom_count"),
        label=f"{label}.diversity_heavy_atom_count",
    )
    if not 1 <= unique_count <= expected_num_samples:
        raise ValueError(f"{label}: unique-pose count outside [1,N]")
    if not 1 <= c2 <= expected_num_samples:
        raise ValueError(f"{label}: C2 outside [1,N]")
    if heavy_atom_count < 1:
        raise ValueError(f"{label}: heavy-atom count must be positive")
    nn = _finite_float(
        row.get("nearest_neighbor_heavy_atom_rmsd_median"),
        label=f"{label}.nearest_neighbor_heavy_atom_rmsd_median",
    )
    if nn < 0.0:
        raise ValueError(f"{label}: nearest-neighbor RMSD must be nonnegative")
    for key in (
        "pairwise_heavy_atom_rmsd_mean",
        "pairwise_heavy_atom_rmsd_median",
        "pairwise_heavy_atom_rmsd_ge2_fraction",
    ):
        value = _finite_float(row.get(key), label=f"{label}.{key}")
        if value < 0.0 or (key.endswith("fraction") and value > 1.0):
            raise ValueError(f"{label}.{key}: outside valid range")

    hashes = {
        key: _sha256(row.get(key), label=f"{label}.{key}")
        for key in (
            "prior_pool_sha256",
            "candidate_ensemble_sha256",
            "ligand_input_identity_sha256",
            "protein_sha256",
            "ligand_reference_sha256",
            "processed_meta_sha256",
            "checkpoint_sha256",
            "all_poses_sdf_sha256",
        )
    }
    if hashes["checkpoint_sha256"] != FROZEN_CHECKPOINT_SHA256[expected_arm]:
        raise ValueError(f"{label}: row checkpoint differs from frozen arm")
    if _strict_int(row.get("all_poses_count"), label=f"{label}.all_poses_count") != (
        expected_num_samples
    ):
        raise ValueError(f"{label}: all-pose output is incomplete")
    all_poses_sdf = str(row.get("all_poses_sdf", "")).strip()
    if not all_poses_sdf:
        raise ValueError(f"{label}: all-pose output path is missing")
    all_poses_path = _resolve_output_file(
        all_poses_sdf,
        source=source,
        root=root,
        label=f"{label}.all_poses_sdf",
    )
    _validate_all_poses_sdf(
        all_poses_path,
        expected_sha256=hashes["all_poses_sdf_sha256"],
        expected_num_samples=expected_num_samples,
        sampling_seed=sampling_seed,
        ligand_conformer_seed=ligand_conformer_seed,
        candidate_ensemble_sha256=hashes["candidate_ensemble_sha256"],
        label=f"{label}.all_poses_sdf",
    )
    return {
        "id": sample_id,
        "system_id": system_id,
        "ligand_chain": ligand_chain,
        "global_index": global_index,
        "arm": expected_arm,
        "sampling_seed": sampling_seed,
        "prior_pool_size": expected_prior_pool_size,
        **hashes,
        "rmsds": rmsds,
        "rmsd_methods": rmsd_methods,
        "rmsd_fallback_count": fallback_count,
        "fast_valid": fast_valid,
        "k2": k2,
        "fv2": fv2,
        "fast_valid_count": sum(fast_valid),
        "first_rmsd": first_rmsd,
        "oracle_rmsd": oracle_rmsd,
        "mean_sample_rmsd": mean_sample_rmsd,
        "all_poses_sdf": str(all_poses_path),
        "match_method": str(row.get("match_method", "")),
        "nn": nn,
        "c2": c2,
        "unique_count": unique_count,
        "unique_fraction": unique_count / expected_num_samples,
        "heavy_atom_count": heavy_atom_count,
    }


def _aggregate(rows: list[dict[str, Any]], *, num_samples: int) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot aggregate an empty arm")
    count = len(rows)
    k2 = [int(row["k2"]) for row in rows]
    fv2 = [int(row["fv2"]) for row in rows]
    fast_valid = [int(row["fast_valid_count"]) for row in rows]
    oracle = [float(row["oracle_rmsd"]) for row in rows]
    first = [float(row["first_rmsd"]) for row in rows]
    nn = [float(row["nn"]) for row in rows]
    c2 = [int(row["c2"]) for row in rows]
    unique = [int(row["unique_count"]) for row in rows]
    return {
        "samples": count,
        "systems": len({str(row["system_id"]) for row in rows}),
        "k2_total": sum(k2),
        "k2_mean": math.fsum(k2) / count,
        "k2_median": statistics.median(k2),
        "k2_ge_1_count": sum(value >= 1 for value in k2),
        "k2_ge_1_pct": 100.0 * sum(value >= 1 for value in k2) / count,
        "k2_ge_5_count": sum(value >= 5 for value in k2),
        "k2_ge_5_pct": 100.0 * sum(value >= 5 for value in k2) / count,
        "k2_ge_10_count": sum(value >= 10 for value in k2),
        "k2_ge_10_pct": 100.0 * sum(value >= 10 for value in k2) / count,
        "oracle_rmsd_mean": math.fsum(oracle) / count,
        "oracle_rmsd_median": statistics.median(oracle),
        "first_rmsd_mean": math.fsum(first) / count,
        "first_rmsd_median": statistics.median(first),
        "fast_valid_k2_total": sum(fv2),
        "fast_valid_k2_mean": math.fsum(fv2) / count,
        "fast_valid_k2_ge_1_count": sum(value >= 1 for value in fv2),
        "fast_valid_k2_ge_1_pct": 100.0 * sum(value >= 1 for value in fv2) / count,
        "fast_valid_candidate_count": sum(fast_valid),
        "fast_valid_candidate_fraction": sum(fast_valid) / (count * num_samples),
        "fast_valid_candidate_pct": 100.0 * sum(fast_valid) / (count * num_samples),
        "nearest_neighbor_rmsd_mean": math.fsum(nn) / count,
        "c2_mean": math.fsum(c2) / count,
        "coordinate_unique_count": sum(unique),
        "coordinate_unique_fraction": sum(unique) / (count * num_samples),
        "coordinate_unique_pct": 100.0 * sum(unique) / (count * num_samples),
    }


def _cluster_arrays(
    baseline: list[dict[str, Any]], treatment: list[dict[str, Any]]
) -> tuple[list[str], dict[str, np.ndarray]]:
    by_system: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for base, treat in zip(baseline, treatment):
        if base["id"] != treat["id"] or base["system_id"] != treat["system_id"]:
            raise ValueError("cluster construction received unpaired rows")
        by_system[str(base["system_id"])].append((base, treat))
    systems = sorted(by_system)
    arrays: dict[str, list[float]] = defaultdict(list)
    for system in systems:
        pairs = by_system[system]
        count = len(pairs)
        values = {
            "count": float(count),
            "k2_delta_sum": math.fsum(t["k2"] - b["k2"] for b, t in pairs),
            "coverage_delta_sum": math.fsum((t["k2"] >= 1) - (b["k2"] >= 1) for b, t in pairs),
            "fv2_delta_sum": math.fsum(t["fv2"] - b["fv2"] for b, t in pairs),
            "fv_coverage_delta_sum": math.fsum((t["fv2"] >= 1) - (b["fv2"] >= 1) for b, t in pairs),
            "nn_baseline_sum": math.fsum(b["nn"] for b, _ in pairs),
            "nn_treatment_sum": math.fsum(t["nn"] for _, t in pairs),
            "c2_baseline_sum": math.fsum(b["c2"] for b, _ in pairs),
            "c2_treatment_sum": math.fsum(t["c2"] for _, t in pairs),
        }
        for key, value in values.items():
            arrays[key].append(float(value))
        arrays["k2_system_delta"].append(values["k2_delta_sum"] / count)
        arrays["coverage_system_delta_pp"].append(100.0 * values["coverage_delta_sum"] / count)
        arrays["fv2_system_delta"].append(values["fv2_delta_sum"] / count)
        arrays["fv_coverage_system_delta_pp"].append(
            100.0 * values["fv_coverage_delta_sum"] / count
        )
        arrays["nn_baseline_system_mean"].append(values["nn_baseline_sum"] / count)
        arrays["nn_treatment_system_mean"].append(values["nn_treatment_sum"] / count)
        arrays["c2_baseline_system_mean"].append(values["c2_baseline_sum"] / count)
        arrays["c2_treatment_system_mean"].append(values["c2_treatment_sum"] / count)
    return systems, {key: np.asarray(values, dtype=np.float64) for key, values in arrays.items()}


def _percentile_interval(values: np.ndarray) -> dict[str, float]:
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("bootstrap produced an invalid statistic")
    low, high = np.percentile(values, [2.5, 97.5])
    return {"ci95_low": float(low), "ci95_high": float(high)}


def cluster_bootstrap(
    baseline: list[dict[str, Any]],
    treatment: list[dict[str, Any]],
    *,
    seed: int,
    resamples: int,
) -> dict[str, Any]:
    """PCG64 system-cluster bootstrap, with sample- and system-balanced views."""
    if resamples < 1:
        raise ValueError("bootstrap resamples must be positive")
    systems, arrays = _cluster_arrays(baseline, treatment)
    num_systems = len(systems)
    if num_systems == 0:
        raise ValueError("cluster bootstrap requires at least one system")
    if arrays["nn_baseline_sum"].sum() <= 0 or arrays["c2_baseline_sum"].sum() <= 0:
        raise ValueError("diversity ratios require positive baseline aggregates")

    metric_names = (
        "k2_delta",
        "coverage_delta_pp",
        "fv2_delta",
        "fv_coverage_delta_pp",
        "nn_ratio",
        "c2_ratio",
        "system_balanced_k2_delta",
        "system_balanced_coverage_delta_pp",
        "system_balanced_fv2_delta",
        "system_balanced_fv_coverage_delta_pp",
        "system_balanced_nn_ratio",
        "system_balanced_c2_ratio",
    )
    draws = {name: np.empty(resamples, dtype=np.float64) for name in metric_names}
    rng = np.random.Generator(np.random.PCG64(seed))
    chunk_size = max(1, min(1_000, resamples))
    for start in range(0, resamples, chunk_size):
        stop = min(start + chunk_size, resamples)
        indices = rng.integers(0, num_systems, size=(stop - start, num_systems))
        selected_count = arrays["count"][indices].sum(axis=1)
        draws["k2_delta"][start:stop] = arrays["k2_delta_sum"][indices].sum(axis=1) / selected_count
        draws["coverage_delta_pp"][start:stop] = 100.0 * (
            arrays["coverage_delta_sum"][indices].sum(axis=1) / selected_count
        )
        draws["fv2_delta"][start:stop] = (
            arrays["fv2_delta_sum"][indices].sum(axis=1) / selected_count
        )
        draws["fv_coverage_delta_pp"][start:stop] = 100.0 * (
            arrays["fv_coverage_delta_sum"][indices].sum(axis=1) / selected_count
        )
        nn_base = arrays["nn_baseline_sum"][indices].sum(axis=1)
        c2_base = arrays["c2_baseline_sum"][indices].sum(axis=1)
        draws["nn_ratio"][start:stop] = arrays["nn_treatment_sum"][indices].sum(axis=1) / nn_base
        draws["c2_ratio"][start:stop] = arrays["c2_treatment_sum"][indices].sum(axis=1) / c2_base

        draws["system_balanced_k2_delta"][start:stop] = arrays["k2_system_delta"][indices].mean(
            axis=1
        )
        draws["system_balanced_coverage_delta_pp"][start:stop] = arrays["coverage_system_delta_pp"][
            indices
        ].mean(axis=1)
        draws["system_balanced_fv2_delta"][start:stop] = arrays["fv2_system_delta"][indices].mean(
            axis=1
        )
        draws["system_balanced_fv_coverage_delta_pp"][start:stop] = arrays[
            "fv_coverage_system_delta_pp"
        ][indices].mean(axis=1)
        system_nn_base = arrays["nn_baseline_system_mean"][indices].mean(axis=1)
        system_c2_base = arrays["c2_baseline_system_mean"][indices].mean(axis=1)
        draws["system_balanced_nn_ratio"][start:stop] = (
            arrays["nn_treatment_system_mean"][indices].mean(axis=1) / system_nn_base
        )
        draws["system_balanced_c2_ratio"][start:stop] = (
            arrays["c2_treatment_system_mean"][indices].mean(axis=1) / system_c2_base
        )

    return {
        "method": (
            "PCG64 percentile bootstrap resampling system_id clusters and retaining "
            "all ligand samples in each drawn system"
        ),
        "seed": seed,
        "resamples": resamples,
        "clusters": num_systems,
        **{name: _percentile_interval(values) for name, values in draws.items()},
    }


def _ratio(numerator: float, denominator: float, *, label: str) -> float:
    if denominator <= 0.0:
        raise ValueError(f"{label}: baseline aggregate must be positive")
    value = numerator / denominator
    if not math.isfinite(value):
        raise ValueError(f"{label}: non-finite ratio")
    return value


def _paired_metrics(
    baseline_by_id: dict[str, dict[str, Any]],
    treatment_by_id: dict[str, dict[str, Any]],
    *,
    num_samples: int,
    bootstrap_seed: int,
    bootstrap_resamples: int,
    include_bootstrap: bool = True,
) -> dict[str, Any]:
    if set(baseline_by_id) != set(treatment_by_id):
        missing = sorted(set(baseline_by_id) - set(treatment_by_id))
        extra = sorted(set(treatment_by_id) - set(baseline_by_id))
        raise ValueError(f"paired ID mismatch: missing={missing[:3]}, extra={extra[:3]}")
    ids = sorted(baseline_by_id)
    baseline = [baseline_by_id[item] for item in ids]
    treatment = [treatment_by_id[item] for item in ids]
    for base, treat in zip(baseline, treatment):
        if base["system_id"] != treat["system_id"]:
            raise ValueError(f"{base['id']}: system_id differs across arms")
    count = len(ids)
    k2_deltas = [int(t["k2"]) - int(b["k2"]) for b, t in zip(baseline, treatment)]
    fv2_deltas = [int(t["fv2"]) - int(b["fv2"]) for b, t in zip(baseline, treatment)]
    base_coverage = [int(row["k2"] >= 1) for row in baseline]
    treatment_coverage = [int(row["k2"] >= 1) for row in treatment]
    base_fv_coverage = [int(row["fv2"] >= 1) for row in baseline]
    treatment_fv_coverage = [int(row["fv2"] >= 1) for row in treatment]
    fragile_ids = [sample_id for sample_id, row in zip(ids, baseline) if 1 <= int(row["k2"]) <= 4]
    fragile_retained_ids = [
        sample_id for sample_id in fragile_ids if int(treatment_by_id[sample_id]["k2"]) >= 1
    ]
    fragile_retention = len(fragile_retained_ids) / len(fragile_ids) if fragile_ids else 1.0

    nn_ratio = _ratio(
        math.fsum(row["nn"] for row in treatment),
        math.fsum(row["nn"] for row in baseline),
        label="nearest-neighbor diversity ratio",
    )
    c2_ratio = _ratio(
        math.fsum(row["c2"] for row in treatment),
        math.fsum(row["c2"] for row in baseline),
        label="C2 diversity ratio",
    )
    base_unique = sum(row["unique_count"] for row in baseline) / (count * num_samples)
    treatment_unique = sum(row["unique_count"] for row in treatment) / (count * num_samples)
    result: dict[str, Any] = {
        "samples": count,
        "systems": len({str(row["system_id"]) for row in baseline}),
        "delta_total_k2": sum(k2_deltas),
        "delta_mean_k2": math.fsum(k2_deltas) / count,
        "positive_samples": sum(value > 0 for value in k2_deltas),
        "negative_samples": sum(value < 0 for value in k2_deltas),
        "tied_samples": sum(value == 0 for value in k2_deltas),
        "k2_ge_1_baseline_count": sum(base_coverage),
        "k2_ge_1_treatment_count": sum(treatment_coverage),
        "k2_ge_1_delta_count": sum(treatment_coverage) - sum(base_coverage),
        "k2_ge_1_delta_pp": 100.0 * (sum(treatment_coverage) - sum(base_coverage)) / count,
        "k2_ge_1_gained_ids": [
            sample_id
            for sample_id, base, treat in zip(ids, base_coverage, treatment_coverage)
            if base == 0 and treat == 1
        ],
        "k2_ge_1_lost_ids": [
            sample_id
            for sample_id, base, treat in zip(ids, base_coverage, treatment_coverage)
            if base == 1 and treat == 0
        ],
        "fragile_baseline_count": len(fragile_ids),
        "fragile_retained_count": len(fragile_retained_ids),
        "fragile_retention_fraction": fragile_retention,
        "fragile_retention_pct": 100.0 * fragile_retention,
        "fragile_lost_ids": sorted(set(fragile_ids) - set(fragile_retained_ids)),
        "delta_total_fast_valid_k2": sum(fv2_deltas),
        "delta_mean_fast_valid_k2": math.fsum(fv2_deltas) / count,
        "fast_valid_k2_ge_1_baseline_count": sum(base_fv_coverage),
        "fast_valid_k2_ge_1_treatment_count": sum(treatment_fv_coverage),
        "fast_valid_k2_ge_1_delta_count": (sum(treatment_fv_coverage) - sum(base_fv_coverage)),
        "fast_valid_k2_ge_1_delta_pp": 100.0
        * (sum(treatment_fv_coverage) - sum(base_fv_coverage))
        / count,
        "fast_valid_k2_ge_1_gained_ids": [
            sample_id
            for sample_id, base, treat in zip(ids, base_fv_coverage, treatment_fv_coverage)
            if base == 0 and treat == 1
        ],
        "fast_valid_k2_ge_1_lost_ids": [
            sample_id
            for sample_id, base, treat in zip(ids, base_fv_coverage, treatment_fv_coverage)
            if base == 1 and treat == 0
        ],
        "fast_valid_candidate_fraction_baseline": sum(row["fast_valid_count"] for row in baseline)
        / (count * num_samples),
        "fast_valid_candidate_fraction_treatment": sum(row["fast_valid_count"] for row in treatment)
        / (count * num_samples),
        "fast_valid_candidate_delta_pp": 100.0
        * sum(t["fast_valid_count"] - b["fast_valid_count"] for b, t in zip(baseline, treatment))
        / (count * num_samples),
        "nearest_neighbor_rmsd_ratio": nn_ratio,
        "c2_ratio": c2_ratio,
        "coordinate_unique_fraction_baseline": base_unique,
        "coordinate_unique_fraction_treatment": treatment_unique,
        "coordinate_unique_fraction_delta": treatment_unique - base_unique,
        "delta_oracle_rmsd_mean": math.fsum(
            t["oracle_rmsd"] - b["oracle_rmsd"] for b, t in zip(baseline, treatment)
        )
        / count,
        "delta_first_rmsd_mean": math.fsum(
            t["first_rmsd"] - b["first_rmsd"] for b, t in zip(baseline, treatment)
        )
        / count,
    }
    systems, arrays = _cluster_arrays(baseline, treatment)
    result["system_balanced_sensitivity"] = {
        "systems": len(systems),
        "delta_mean_k2": float(arrays["k2_system_delta"].mean()),
        "k2_ge_1_delta_pp": float(arrays["coverage_system_delta_pp"].mean()),
        "delta_mean_fast_valid_k2": float(arrays["fv2_system_delta"].mean()),
        "fast_valid_k2_ge_1_delta_pp": float(arrays["fv_coverage_system_delta_pp"].mean()),
        "nearest_neighbor_rmsd_ratio": _ratio(
            float(arrays["nn_treatment_system_mean"].mean()),
            float(arrays["nn_baseline_system_mean"].mean()),
            label="system-balanced NN ratio",
        ),
        "c2_ratio": _ratio(
            float(arrays["c2_treatment_system_mean"].mean()),
            float(arrays["c2_baseline_system_mean"].mean()),
            label="system-balanced C2 ratio",
        ),
    }
    result["cluster_bootstrap"] = (
        cluster_bootstrap(
            baseline,
            treatment,
            seed=bootstrap_seed,
            resamples=bootstrap_resamples,
        )
        if include_bootstrap
        else None
    )
    return result


def _gate(observed: float | int, operator: str, threshold: float | int) -> dict[str, Any]:
    operations = {
        ">=": lambda: observed >= threshold,
        ">": lambda: observed > threshold,
        "<=": lambda: observed <= threshold,
    }
    if operator not in operations:
        raise ValueError(f"unsupported gate operator {operator!r}")
    return {
        "observed": observed,
        "operator": operator,
        "threshold": threshold,
        "passed": bool(operations[operator]()),
    }


def _selection_decision(primary: dict[str, Any]) -> dict[str, Any]:
    bootstrap = primary["cluster_bootstrap"]
    assert isinstance(bootstrap, dict)
    gates = {
        "efficacy_mean_k2": _gate(primary["delta_mean_k2"], ">=", 1.0),
        "efficacy_k2_ci95_low": _gate(bootstrap["k2_delta"]["ci95_low"], ">", 0.0),
        "coverage_count": _gate(primary["k2_ge_1_delta_count"], ">=", 0),
        "coverage_ci95_low_pp": _gate(bootstrap["coverage_delta_pp"]["ci95_low"], ">=", -1.0),
        "fragile_retention_fraction": _gate(primary["fragile_retention_fraction"], ">=", 0.95),
        "fast_valid_mean_k2": _gate(primary["delta_mean_fast_valid_k2"], ">=", 0.0),
        "fast_valid_coverage_count": _gate(primary["fast_valid_k2_ge_1_delta_count"], ">=", 0),
        "fast_valid_coverage_ci95_low_pp": _gate(
            bootstrap["fv_coverage_delta_pp"]["ci95_low"], ">=", -1.0
        ),
        "fast_valid_candidate_delta_pp": _gate(
            primary["fast_valid_candidate_delta_pp"], ">=", -1.0
        ),
        "nearest_neighbor_ratio": _gate(primary["nearest_neighbor_rmsd_ratio"], ">=", 0.95),
        "nearest_neighbor_ratio_ci95_low": _gate(bootstrap["nn_ratio"]["ci95_low"], ">=", 0.90),
        "c2_ratio": _gate(primary["c2_ratio"], ">=", 0.95),
        "c2_ratio_ci95_low": _gate(bootstrap["c2_ratio"]["ci95_low"], ">=", 0.90),
        "coordinate_unique_treatment_fraction": _gate(
            primary["coordinate_unique_fraction_treatment"], ">=", 0.99
        ),
        "coordinate_unique_fraction_delta": _gate(
            primary["coordinate_unique_fraction_delta"], ">=", -0.005
        ),
    }
    passed = all(gate["passed"] for gate in gates.values())
    return {
        "selection_eligible": True,
        "passed": passed,
        "action": (
            "promote_parent50k_plus10k_t0p10_ema_and_end_sampler_training"
            if passed
            else "keep_s50_ema_and_stop_additional_t0_continuation"
        ),
        "gates": gates,
        "failed_gates": [name for name, gate in gates.items() if not gate["passed"]],
    }


def _arm_csv_from_summary(
    summary: dict[str, Any],
    *,
    summary_path: Path,
    root: Path,
    arm: str,
    expected_count: int,
) -> Path:
    artifacts = summary.get("artifacts")
    if not isinstance(artifacts, dict) or not isinstance(artifacts.get("arms"), dict):
        raise ValueError(f"{summary_path}.artifacts.arms: missing")
    summary_stage = str(summary.get("mode"))
    if set(artifacts["arms"]) != set(_expected_arms(summary_stage)):
        raise ValueError(f"{summary_path}.artifacts.arms: wrong exact arm inventory")
    record = artifacts["arms"].get(arm)
    if not isinstance(record, dict):
        raise ValueError(f"{summary_path}.artifacts.arms[{arm!r}]: missing")
    if record.get("arm") != arm:
        raise ValueError(f"{summary_path}: arm artifact identity mismatch")
    if _strict_int(record.get("count"), label=f"{summary_path}.{arm}.count") != expected_count:
        raise ValueError(f"{summary_path}: arm artifact row count mismatch")
    path = _resolve_file(
        record.get("results_csv"),
        summary_path=summary_path,
        root=root,
        label=f"{summary_path}.{arm}.results_csv",
    )
    declared = _sha256(
        record.get("results_csv_sha256"),
        label=f"{summary_path}.{arm}.results_csv_sha256",
    )
    actual = _file_sha256(path)
    if declared != actual:
        raise ValueError(
            f"{summary_path}.{arm}: CSV SHA-256 mismatch: declared={declared}, actual={actual}"
        )
    return path


def _load_csv_rows(
    path: Path,
    *,
    root: Path,
    arm: str,
    expected_num_samples: int,
    expected_prior_pool_size: int,
    expected_global_index_by_id: dict[str, int],
) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: CSV has no header")
        rows = [
            _parse_row(
                row,
                source=path,
                root=root,
                expected_arm=arm,
                expected_num_samples=expected_num_samples,
                expected_prior_pool_size=expected_prior_pool_size,
                expected_global_index_by_id=expected_global_index_by_id,
            )
            for row in reader
        ]
    ids = [str(row["id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path}: duplicate sample IDs")
    return rows


def _validate_pairing(cells: dict[str, dict[str, dict[str, Any]]], stage: str) -> None:
    arms = _expected_arms(stage)
    baseline_ids = set(cells[arms[0]])
    for arm in arms[1:]:
        if set(cells[arm]) != baseline_ids:
            raise ValueError(f"{arm}: sample IDs differ across paired arms")
    identity_fields = (
        "system_id",
        "ligand_chain",
        "global_index",
        "sampling_seed",
        "prior_pool_size",
        "prior_pool_sha256",
        "ligand_input_identity_sha256",
        "protein_sha256",
        "ligand_reference_sha256",
        "processed_meta_sha256",
    )
    for sample_id in sorted(baseline_ids):
        rows = [cells[arm][sample_id] for arm in arms]
        for field in identity_fields:
            if len({row[field] for row in rows}) != 1:
                raise ValueError(f"{sample_id}: paired field {field!r} differs across arms")


def _replay_audit(cells: dict[str, dict[str, dict[str, Any]]], *, stage: str) -> dict[str, Any]:
    baseline = [cells[BASELINE_ARM][item] for item in sorted(cells[BASELINE_ARM])]
    replay = [cells[REPLAY_ARM][item] for item in sorted(cells[REPLAY_ARM])]
    coverage_mismatches = sum(
        (base["k2"] >= 1) != (other["k2"] >= 1) for base, other in zip(baseline, replay)
    )
    fast_coverage_mismatches = sum(
        (base["fv2"] >= 1) != (other["fv2"] >= 1) for base, other in zip(baseline, replay)
    )
    mean_abs_k2_difference = math.fsum(
        abs(other["k2"] - base["k2"]) for base, other in zip(baseline, replay)
    ) / len(baseline)
    ratios = {
        "nearest_neighbor_heavy_atom_rmsd_median": _ratio(
            math.fsum(row["nn"] for row in replay),
            math.fsum(row["nn"] for row in baseline),
            label="replay NN ratio",
        ),
        "c2_connected_component_count": _ratio(
            math.fsum(row["c2"] for row in replay),
            math.fsum(row["c2"] for row in baseline),
            label="replay C2 ratio",
        ),
        "coordinate_unique_count": _ratio(
            math.fsum(row["unique_count"] for row in replay),
            math.fsum(row["unique_count"] for row in baseline),
            label="replay unique ratio",
        ),
    }
    if stage == "smoke":
        passed = all(math.isfinite(value) for value in ratios.values())
        rule = "engineering integrity only; efficacy is not inspected"
    elif stage == "pilot":
        passed = (
            coverage_mismatches == 0
            and fast_coverage_mismatches == 0
            and mean_abs_k2_difference <= 0.25
            and all(0.98 <= value <= 1.02 for value in ratios.values())
        )
        rule = (
            "zero coverage classification mismatches, mean absolute K2 difference <=0.25, "
            "and all replay diversity ratios in [0.98,1.02]"
        )
    else:
        raise ValueError("replay audit is only defined for smoke and pilot")
    return {
        "passed": passed,
        "rule": rule,
        "checked_count": len(baseline),
        "k2_coverage_classification_mismatches": coverage_mismatches,
        "fast_valid_k2_coverage_classification_mismatches": fast_coverage_mismatches,
        "mean_abs_k2_difference": mean_abs_k2_difference,
        "diversity_aggregate_ratios": ratios,
    }


def _same_replay_result(left: dict[str, Any], right: dict[str, Any]) -> bool:
    scalar_fields = (
        "passed",
        "checked_count",
        "k2_coverage_classification_mismatches",
        "fast_valid_k2_coverage_classification_mismatches",
    )
    if any(left.get(field) != right.get(field) for field in scalar_fields):
        return False
    if not math.isclose(
        float(left.get("mean_abs_k2_difference")),
        float(right.get("mean_abs_k2_difference")),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        return False
    left_ratios = left.get("diversity_aggregate_ratios")
    right_ratios = right.get("diversity_aggregate_ratios")
    if not isinstance(left_ratios, dict) or not isinstance(right_ratios, dict):
        return False
    return all(
        key in right_ratios
        and math.isclose(float(value), float(right_ratios[key]), rel_tol=0.0, abs_tol=1e-12)
        for key, value in left_ratios.items()
    )


def _operational_sensitivity(
    arm_metrics: dict[str, dict[str, Any]], primary: dict[str, Any]
) -> dict[str, Any]:
    return {
        "denominator": FULL_SPLIT_COUNT,
        "evaluable_count": FULL_ELIGIBLE_COUNT,
        "common_preprocessing_failure_count": FULL_INELIGIBLE_COUNT,
        "common_preprocessing_failure_assignment": "K2=0 and fast-valid K2=0 for every arm",
        "arms": {
            arm: {
                "k2_mean": metrics["k2_total"] / FULL_SPLIT_COUNT,
                "k2_ge_1_pct": 100.0 * metrics["k2_ge_1_count"] / FULL_SPLIT_COUNT,
                "fast_valid_k2_mean": metrics["fast_valid_k2_total"] / FULL_SPLIT_COUNT,
                "fast_valid_k2_ge_1_pct": (
                    100.0 * metrics["fast_valid_k2_ge_1_count"] / FULL_SPLIT_COUNT
                ),
            }
            for arm, metrics in arm_metrics.items()
            if arm in ARMS
        },
        "primary": {
            "delta_mean_k2": primary["delta_total_k2"] / FULL_SPLIT_COUNT,
            "k2_ge_1_delta_pp": (100.0 * primary["k2_ge_1_delta_count"] / FULL_SPLIT_COUNT),
            "delta_mean_fast_valid_k2": (primary["delta_total_fast_valid_k2"] / FULL_SPLIT_COUNT),
            "fast_valid_k2_ge_1_delta_pp": (
                100.0 * primary["fast_valid_k2_ge_1_delta_count"] / FULL_SPLIT_COUNT
            ),
        },
    }


def build_report(
    output_root: Path,
    *,
    stage: str = "full",
    expected_count: int | None = None,
    expected_shards: int | None = None,
    expected_num_samples: int | None = None,
    expected_num_steps: int | None = None,
    expected_prior_pool_size: int | None = None,
    bootstrap_seed: int = BOOTSTRAP_SEED,
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES,
) -> dict[str, Any]:
    if stage not in STAGE_DEFAULTS:
        raise ValueError(f"stage must be one of {tuple(STAGE_DEFAULTS)}")
    defaults = STAGE_DEFAULTS[stage]
    expected_count = defaults["count"] if expected_count is None else expected_count
    expected_shards = defaults["shards"] if expected_shards is None else expected_shards
    expected_num_samples = (
        defaults["num_samples"] if expected_num_samples is None else expected_num_samples
    )
    expected_num_steps = defaults["num_steps"] if expected_num_steps is None else expected_num_steps
    expected_prior_pool_size = (
        defaults["prior"] if expected_prior_pool_size is None else expected_prior_pool_size
    )
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 1
        for value in (
            expected_count,
            expected_shards,
            expected_num_samples,
            expected_num_steps,
            expected_prior_pool_size,
        )
    ):
        raise ValueError("expected counts, shards, samples, steps, and priors must be positive")
    if stage == "full" and (
        expected_count != FULL_ELIGIBLE_COUNT
        or expected_shards != FULL_SHARDS
        or expected_num_samples != FULL_NUM_SAMPLES
        or expected_num_steps != FULL_NUM_STEPS
        or expected_prior_pool_size != FULL_PRIOR_POOL_SIZE
    ):
        raise ValueError("the full selection stage cannot override its frozen dimensions")
    if stage == "full" and (
        bootstrap_seed != BOOTSTRAP_SEED or bootstrap_resamples != BOOTSTRAP_RESAMPLES
    ):
        raise ValueError(
            "the full selection stage cannot override its frozen bootstrap seed or resample count"
        )
    if expected_prior_pool_size < expected_num_samples:
        raise ValueError("prior pool size must be at least the number of candidates")
    if not output_root.is_dir():
        raise ValueError(f"output root is not a directory: {output_root}")

    candidates: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(output_root.rglob("paired_summary.json")):
        summary = _load_json_object(path)
        if summary.get("protocol_id") == PROTOCOL_ID and summary.get("mode") == stage:
            candidates.append((path, summary))
    if len(candidates) != expected_shards:
        raise ValueError(
            f"expected exactly {expected_shards} {stage} paired summaries, found {len(candidates)}"
        )

    summaries: list[tuple[Path, dict[str, Any], dict[str, Any]]] = []
    for path, summary in candidates:
        metadata = _validate_paired_summary(
            summary,
            path=path,
            root=output_root,
            stage=stage,
            expected_count=expected_count,
            expected_shards=expected_shards,
            expected_num_samples=expected_num_samples,
            expected_num_steps=expected_num_steps,
            expected_prior_pool_size=expected_prior_pool_size,
        )
        summaries.append((path, summary, metadata))
    summaries.sort(key=lambda value: int(value[2]["shard_index"]))
    if [metadata["shard_index"] for _, _, metadata in summaries] != list(range(expected_shards)):
        raise ValueError("shard indices are incomplete or duplicated")

    run_ids = {str(summary.get("run_id")) for _, summary, _ in summaries}
    selected_id_lists = {tuple(metadata["selected_ids"]) for _, _, metadata in summaries}
    full_id_lists = {tuple(metadata["full_ids"]) for _, _, metadata in summaries}
    eligibility_hashes = {metadata["eligibility_manifest_sha256"] for _, _, metadata in summaries}
    eligibility_paths = {metadata["eligibility_manifest_path"] for _, _, metadata in summaries}
    operational_failure_hashes = {
        metadata["operational_failure_ids_sha256"] for _, _, metadata in summaries
    }
    code_hashes = {metadata["code_sha256"] for _, _, metadata in summaries}
    if len(run_ids) != 1:
        raise ValueError("run_id differs across shards")
    if len(selected_id_lists) != 1:
        raise ValueError("selected ID inventory differs across shards")
    if len(full_id_lists) != 1:
        raise ValueError("frozen full ID inventory differs across shards")
    if len(eligibility_hashes) != 1:
        raise ValueError("eligibility manifest SHA-256 differs across shards")
    if len(eligibility_paths) != 1:
        raise ValueError("eligibility manifest path differs across shards")
    if len(operational_failure_hashes) != 1:
        raise ValueError("operational preprocessing-failure inventory differs across shards")
    if len(code_hashes) != 1:
        raise ValueError("code inventory SHA-256 differs across shards")
    selected_ids = list(next(iter(selected_id_lists)))
    full_ids = list(next(iter(full_id_lists)))
    expected_global_index_by_id = {
        sample_id: index for index, sample_id in enumerate(full_ids, start=1)
    }
    if len(selected_ids) != expected_count or len(set(selected_ids)) != expected_count:
        raise ValueError("selected cohort count/uniqueness mismatch")
    if selected_ids != sorted(selected_ids):
        raise ValueError("selected cohort must be lexicographically sorted")
    if stage == "full" and _ids_sha256(selected_ids) != FROZEN_ELIGIBLE_IDS_SHA256:
        raise ValueError("full selected cohort differs from the frozen eligible ID digest")

    active_arms = _expected_arms(stage)
    rows_by_arm: dict[str, list[dict[str, Any]]] = {arm: [] for arm in active_arms}
    shard_inventory: list[dict[str, Any]] = []
    for path, summary, metadata in summaries:
        shard_inventory.append(
            {
                "summary": str(path),
                "shard_index": metadata["shard_index"],
                "assigned_count": metadata["assigned_count"],
                "assigned_ids_sha256": _ids_sha256(metadata["assigned_ids"]),
            }
        )
        for arm in active_arms:
            csv_path = _arm_csv_from_summary(
                summary,
                summary_path=path,
                root=output_root,
                arm=arm,
                expected_count=metadata["assigned_count"],
            )
            rows = _load_csv_rows(
                csv_path,
                root=output_root,
                arm=arm,
                expected_num_samples=expected_num_samples,
                expected_prior_pool_size=expected_prior_pool_size,
                expected_global_index_by_id=expected_global_index_by_id,
            )
            if [row["id"] for row in rows] != metadata["assigned_ids"]:
                raise ValueError(f"{path}/{arm}: CSV order differs from shard assignment")
            rows_by_arm[arm].extend(rows)
    for arm, rows in rows_by_arm.items():
        ids = [str(row["id"]) for row in rows]
        if sorted(ids) != selected_ids or len(ids) != expected_count:
            raise ValueError(f"{arm}: merged CSV inventory is incomplete")
    cells = {arm: {str(row["id"]): row for row in rows} for arm, rows in rows_by_arm.items()}
    _validate_pairing(cells, stage)
    system_count = len({row["system_id"] for row in rows_by_arm[BASELINE_ARM]})
    if stage == "full" and system_count != FULL_SYSTEM_COUNT:
        raise ValueError(
            f"full cohort has {system_count} unique systems; expected {FULL_SYSTEM_COUNT}"
        )

    rmsd_fallbacks = {
        arm: [
            {"id": row["id"], "candidate_count": row["rmsd_fallback_count"]}
            for row in sorted(rows_by_arm[arm], key=lambda value: str(value["id"]))
            if int(row["rmsd_fallback_count"]) > 0
        ]
        for arm in active_arms
    }

    replay_audit: dict[str, Any] | None = None
    if stage in {"smoke", "pilot"}:
        replay_audit = _replay_audit(cells, stage=stage)
        if not replay_audit["passed"]:
            raise ValueError(f"{stage}: independently recomputed replay gate failed")
        for _, _, metadata in summaries:
            if not _same_replay_result(replay_audit, metadata["replay_integrity_gate"]):
                raise ValueError(f"{stage}: runner and report replay audits disagree")

    arm_metrics: dict[str, dict[str, Any]] | None = None
    comparisons: dict[str, Any] | None = None
    primary: dict[str, Any] | None = None
    if stage == "full":
        arm_metrics = {
            arm: _aggregate(
                [cells[arm][sample_id] for sample_id in selected_ids],
                num_samples=expected_num_samples,
            )
            for arm in active_arms
        }
        base = cells[BASELINE_ARM]
        treatment = cells[TREATMENT_ARM]
        primary = _paired_metrics(
            base,
            treatment,
            num_samples=expected_num_samples,
            bootstrap_seed=bootstrap_seed,
            bootstrap_resamples=bootstrap_resamples,
        )
        comparisons = {
            "s50_ema_to_parent50k_plus10k_t0p10_ema": {
                "role": "primary_selection_contrast",
                "baseline_arm": BASELINE_ARM,
                "treatment_arm": TREATMENT_ARM,
                **primary,
            },
            "s25_ema_to_s50_ema": {
                "role": "report_only_plateau_diagnostic",
                "baseline_arm": DIAGNOSTIC_ARM,
                "treatment_arm": BASELINE_ARM,
                **_paired_metrics(
                    cells[DIAGNOSTIC_ARM],
                    base,
                    num_samples=expected_num_samples,
                    bootstrap_seed=bootstrap_seed,
                    bootstrap_resamples=bootstrap_resamples,
                    include_bootstrap=False,
                ),
            },
            "s25_ema_to_parent50k_plus10k_t0p10_ema": {
                "role": "report_only_plateau_diagnostic",
                "baseline_arm": DIAGNOSTIC_ARM,
                "treatment_arm": TREATMENT_ARM,
                **_paired_metrics(
                    cells[DIAGNOSTIC_ARM],
                    treatment,
                    num_samples=expected_num_samples,
                    bootstrap_seed=bootstrap_seed,
                    bootstrap_resamples=bootstrap_resamples,
                    include_bootstrap=False,
                ),
            },
        }
        decision = _selection_decision(primary)
    else:
        decision = {
            "selection_eligible": False,
            "passed": replay_audit["passed"] if replay_audit is not None else False,
            "action": (
                "advance_to_next_execution_stage"
                if replay_audit is not None and replay_audit["passed"]
                else "stop_before_selection_sampling"
            ),
            "reason": f"{stage} is an integrity stage; efficacy is not a selection result",
            "gates": {"replay_integrity": replay_audit},
        }
    report = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "created_utc": datetime.now(UTC).isoformat(),
        "status": (
            "complete_selection_pass"
            if stage == "full" and decision["passed"]
            else (
                "complete_selection_fail" if stage == "full" else f"complete_{stage}_integrity_pass"
            )
        ),
        "stage": stage,
        "configuration": {
            "primary_arms": list(ARMS),
            "active_arms": list(active_arms),
            "selection_contrast": [BASELINE_ARM, TREATMENT_ARM],
            "diagnostic_arm": DIAGNOSTIC_ARM,
            "selected_count": expected_count,
            "system_count": system_count,
            "expected_shards": expected_shards,
            "num_samples": expected_num_samples,
            "num_steps": expected_num_steps,
            "prior_pool_size": expected_prior_pool_size,
            "k2_definition": "count(candidate RMSD strictly < 2 Angstrom)",
            "fast_valid_k2_definition": (
                "count(candidate RMSD strictly < 2 Angstrom and ordered fast-valid label true)"
            ),
            "diversity": {
                "contract": POSE_DIVERSITY_CONTRACT,
                "nearest_neighbor": (
                    "within-sample median nearest-neighbor receptor-frame same-index "
                    "heavy-atom RMSD"
                ),
                "c2": "components under pair-RMSD edges strictly < 2 Angstrom",
                "unique_rounding_angstrom": 0.001,
            },
            "bootstrap": {
                "generator": "numpy.random.PCG64",
                "seed": bootstrap_seed,
                "resamples": bootstrap_resamples,
                "cluster": "plinder_system_id",
                "sample_weighted_primary": True,
                "system_balanced_sensitivity": True,
            },
        },
        "integrity": {
            "run_id": next(iter(run_ids)),
            "summary_files_used": len(summaries),
            "zero_runtime_failures": True,
            "exact_arm_inventory": True,
            "paired_identity_fields_equal": True,
            "all_pose_sdf_artifact_audit": {
                "files_verified": expected_count * len(active_arms),
                "records_verified": expected_count * len(active_arms) * expected_num_samples,
                "sha256_recomputed": True,
                "reader": "RDKit Chem.ForwardSDMolSupplier over binary stream",
                "ordered_properties_verified": [
                    "sample_index",
                    "sampling_seed",
                    "ligand_conformer_seed",
                    "candidate_ensemble_sha256",
                ],
                "paths_confined_to_output_root": True,
            },
            "eligibility_manifest_sha256": next(iter(eligibility_hashes)),
            "operational_preprocessing_failure_count": FULL_INELIGIBLE_COUNT,
            "operational_preprocessing_failure_ids_sha256": next(
                iter(operational_failure_hashes)
            ),
            "operational_sensitivity_assignment": (
                "common preprocessing failures have K2=0"
            ),
            "code_inventory_sha256": next(iter(code_hashes)),
            "frozen_hashes": {
                "protocol": FROZEN_PROTOCOL_SHA256,
                "config": FROZEN_CONFIG_SHA256,
                "split": FROZEN_SPLIT_SHA256,
                "pool": FROZEN_POOL_SHA256,
                "raw_gate": FROZEN_RAW_GATE_SHA256,
                "conformer_mapping_audit": FROZEN_AUDIT_SHA256,
                "eligible_ids_newline": FROZEN_ELIGIBLE_IDS_SHA256,
                "runner": FROZEN_RUNNER_SHA256,
                "evaluator": FROZEN_EVALUATOR_SHA256,
                "benchmark": FROZEN_BENCHMARK_SHA256,
                "eligibility_manifest": FROZEN_ELIGIBILITY_MANIFEST_SHA256,
                "checkpoints": FROZEN_CHECKPOINT_SHA256,
            },
            "shards": shard_inventory,
            "rmsd_fallback_ids_by_arm": rmsd_fallbacks,
            "replay": replay_audit,
        },
        "decision": decision,
    }
    if stage == "full":
        assert arm_metrics is not None
        assert comparisons is not None
        assert primary is not None
        report["arms"] = arm_metrics
        report["comparisons"] = comparisons
        report["operational_full_1076_sensitivity"] = _operational_sensitivity(arm_metrics, primary)
    return report


def write_report(report: dict[str, Any], output_json: Path) -> None:
    """Atomically create a report while refusing every overwrite."""
    output_json.parent.mkdir(parents=True, exist_ok=True)
    if output_json.exists():
        raise FileExistsError(f"refusing to overwrite report: {output_json}")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_json.parent,
            prefix=f".{output_json.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if output_json.exists():
            raise FileExistsError(f"refusing to overwrite report: {output_json}")
        os.link(temporary, output_json)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _concise_decision(report: dict[str, Any]) -> dict[str, Any]:
    decision = report["decision"]
    return {
        "status": report["status"],
        "stage": report["stage"],
        "selection_eligible": decision["selection_eligible"],
        "passed": decision["passed"],
        "action": decision["action"],
        "failed_gates": decision.get("failed_gates", []),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--stage", choices=tuple(STAGE_DEFAULTS), default="full")
    parser.add_argument(
        "--expected-count",
        type=int,
        help="Engineering-stage override only; full-stage dimensions are frozen.",
    )
    parser.add_argument("--expected-shards", type=int)
    parser.add_argument("--expected-num-samples", type=int)
    parser.add_argument("--expected-num-steps", type=int)
    parser.add_argument("--expected-prior-pool-size", type=int)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--bootstrap-resamples", type=int, default=BOOTSTRAP_RESAMPLES)
    args = parser.parse_args()
    report = build_report(
        args.output_root,
        stage=args.stage,
        expected_count=args.expected_count,
        expected_shards=args.expected_shards,
        expected_num_samples=args.expected_num_samples,
        expected_num_steps=args.expected_num_steps,
        expected_prior_pool_size=args.expected_prior_pool_size,
        bootstrap_seed=args.bootstrap_seed,
        bootstrap_resamples=args.bootstrap_resamples,
    )
    write_report(report, args.output_json)
    print(json.dumps(_concise_decision(report), sort_keys=True))


if __name__ == "__main__":
    main()
