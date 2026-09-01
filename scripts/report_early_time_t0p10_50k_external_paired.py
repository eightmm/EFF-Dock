#!/usr/bin/env python3
"""Audit and aggregate the three-arm external early-time fine-tune benchmark."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import statistics
import tempfile
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

ARMS = ("current_raw", "parent_ema", "t0p10_50k_ema")
REPLAY_ARM = "current_raw_replay"
KNOWN_ARMS = (REPLAY_ARM, *ARMS)
DATASETS = ("astex", "posebusters")
EXPECTED_COUNTS = {"astex": 85, "posebusters": 308}
EXPECTED_SHARDS = 8
EXPECTED_NUM_SAMPLES = 100
EXPECTED_NUM_STEPS = 10
EXPECTED_PRIOR_POOL_SIZE = 100
BOOTSTRAP_SEED = 20260814
BOOTSTRAP_RESAMPLES = 10_000
FROZEN_PROTOCOL_ID = "EFFDOCK-EARLY-TIME-T0P10-50K-EXTERNAL-PAIRED-V1"
FROZEN_CHECKPOINT_SHA256 = {
    "current_raw": "6932fb3ba6ebac770f714453529656a44b8f33cf15119d23c9e675d2d60b36db",
    "parent_ema": "166d92a7f74015b0011451ad70c71601d72769da00ce1206c8a6a27832e40d97",
    "t0p10_50k_ema": "65be44d7dc8f0867eb9fc5d22214b80f93971ea4702679a527c665046e91e6b6",
    "current_raw_replay": (
        "6932fb3ba6ebac770f714453529656a44b8f33cf15119d23c9e675d2d60b36db"
    ),
}
FROZEN_CONFIG_SHA256 = (
    "39aa62e4a48ed6f3aa4ff59345fb43a81220e2baba22edfd5beb0c4981b307ec"
)
FROZEN_ELIGIBILITY_MANIFEST_SHA256 = (
    "d7321f847c8d6d08950e02d5f41ff42b62fd29ccea78072f27078aa039791c45"
)
FROZEN_BENCHMARK_INPUT_MANIFEST_SHA256 = (
    "99f15f557644cc51c3dd1f559b0dd97dd4259c1de3e1403fb761b7c7e079f668"
)
FROZEN_POCKET_CENTERS_SHA256 = {
    "astex": "1ac4d8629a7ee2adb785173db56fb69ec4140d68e3057631ae10df6ef88d0d85",
    "posebusters": "2d3db55c8cc75650cff85d8e3c12445fb8f45fbe2673d8bbc32045ee8c0f6ad0",
}
FROZEN_BENCHMARK_INPUT_IDENTITY_SCHEMA = "effdock.benchmark_input_identity.v1"
FROZEN_BENCHMARK_INPUT_HEAVY_ATOM_POLICY = (
    "seeded_generic_loader_then_rdkit_remove_all_hs"
)
FROZEN_BASE_SEED = 42
FROZEN_PRIOR_POOL_HASH_CONTRACT = "EFFDOCK_SHARED_PRIOR_V1"
FROZEN_SAMPLING_DYNAMICS_CONTRACT = {"mode": "deterministic_ode"}
COMPARISONS = (
    (
        "parent_ema_to_t0p10_50k_ema",
        "parent_ema",
        "t0p10_50k_ema",
        "primary_causal_finetune_effect",
    ),
    (
        "current_raw_to_t0p10_50k_ema",
        "current_raw",
        "t0p10_50k_ema",
        "practical_replacement_effect",
    ),
    (
        "current_raw_to_parent_ema",
        "current_raw",
        "parent_ema",
        "raw_to_ema_checkpoint_effect",
    ),
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


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
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}: expected a number, got {value!r}") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{label}: metric is non-finite")
    return parsed


def _strict_bool(value: object, *, label: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value in {"True", "False"}:
        return value == "True"
    raise ValueError(f"{label}: expected an exact boolean, got {value!r}")


def _sha256(value: object, *, label: str) -> str:
    parsed = str(value).lower()
    if _SHA256_RE.fullmatch(parsed) is None:
        raise ValueError(f"{label}: expected a SHA-256 digest, got {value!r}")
    return parsed


def _require_frozen_sha256(value: object, expected: str, *, label: str) -> str:
    parsed = _sha256(value, label=label)
    if parsed != expected:
        raise ValueError(f"{label}: expected frozen SHA-256 {expected}, got {parsed}")
    return parsed


def _require_float(value: object, expected: float, *, label: str) -> float:
    parsed = _finite_float(value, label=label)
    if not math.isclose(parsed, expected, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{label}: expected {expected}, got {parsed}")
    return parsed


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_artifact_path(raw: object, *, source: Path, root: Path, label: str) -> Path:
    value = str(raw).strip()
    if not value:
        raise ValueError(f"{label}: artifact path is empty")
    declared = Path(value)
    candidates = [declared] if declared.is_absolute() else [
        Path.cwd() / declared,
        source.parent / declared,
        root / declared,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise ValueError(f"{label}: artifact does not exist: {value}")


def _validate_file_identity(
    raw_path: object,
    raw_sha256: object,
    *,
    source: Path,
    root: Path,
    label: str,
) -> tuple[str, str]:
    path = _resolve_artifact_path(raw_path, source=source, root=root, label=label)
    declared_sha256 = _sha256(raw_sha256, label=f"{label}_sha256")
    actual_sha256 = _file_sha256(path)
    if declared_sha256 != actual_sha256:
        raise ValueError(
            f"{label}_sha256: declared {declared_sha256}, actual {actual_sha256}"
        )
    return str(raw_path), declared_sha256


def _validate_benchmark_input_identity(
    value: object,
    *,
    dataset: str,
    source: Path,
) -> dict[str, Any]:
    label = f"{source}.benchmark_input_identity"
    if not isinstance(value, dict):
        raise ValueError(f"{label}: expected an object")
    expected_scalars = {
        "schema_version": FROZEN_BENCHMARK_INPUT_IDENTITY_SCHEMA,
        "mode": "frozen_manifest",
        "dataset": dataset,
        "heavy_atom_policy": FROZEN_BENCHMARK_INPUT_HEAVY_ATOM_POLICY,
    }
    for key, expected in expected_scalars.items():
        if value.get(key) != expected:
            raise ValueError(
                f"{label}.{key}: expected frozen value {expected!r}, "
                f"got {value.get(key)!r}"
            )
    expected_discovered = EXPECTED_COUNTS[dataset]
    if _strict_int(value.get("count"), label=f"{label}.count") != expected_discovered:
        raise ValueError(f"{label}.count: expected {expected_discovered}")
    for key in ("ids_sha256", "mapping_sha256", "sha256"):
        _sha256(value.get(key), label=f"{label}.{key}")
    sources = value.get("sources")
    if not isinstance(sources, dict):
        raise ValueError(f"{label}.sources: expected an object")
    frozen_manifest = sources.get("frozen_manifest")
    if not isinstance(frozen_manifest, dict):
        raise ValueError(f"{label}.sources.frozen_manifest: expected an object")
    _require_frozen_sha256(
        frozen_manifest.get("sha256"),
        FROZEN_BENCHMARK_INPUT_MANIFEST_SHA256,
        label=f"{label}.sources.frozen_manifest.sha256",
    )
    per_id = value.get("per_id")
    if not isinstance(per_id, dict) or len(per_id) != expected_discovered:
        raise ValueError(
            f"{label}.per_id: expected exactly {expected_discovered} input identities"
        )
    for complex_id, record in per_id.items():
        if not isinstance(complex_id, str) or not isinstance(record, dict):
            raise ValueError(f"{label}.per_id: malformed complex identity record")
        _sha256(record.get("sha256"), label=f"{label}.per_id[{complex_id!r}].sha256")
    return value


def _normalized(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")


def _identify_arm(summary: dict[str, Any], path: Path) -> str:
    explicit = summary.get("arm")
    if explicit in KNOWN_ARMS:
        return str(explicit)
    last_matches: list[str] = []
    for value in (summary.get("run_name", ""), path.name, path.as_posix()):
        haystack = _normalized(value)
        if REPLAY_ARM in haystack:
            return REPLAY_ARM
        matches = [arm for arm in ARMS if arm in haystack]
        if len(matches) == 1:
            return matches[0]
        if matches:
            last_matches = matches
    raise ValueError(
        f"{path}: expected exactly one arm token from {KNOWN_ARMS}, found {last_matches}"
    )


def _resolve_csv_path(summary_path: Path, summary: dict[str, Any], root: Path) -> Path:
    declared = summary.get("csv")
    candidates: list[Path] = []
    if declared:
        declared_path = Path(str(declared))
        candidates.append(declared_path)
        if not declared_path.is_absolute():
            candidates.extend((summary_path.parent / declared_path, root / declared_path))
    candidates.append(
        summary_path.with_name(summary_path.name.removesuffix(".summary.json") + ".csv")
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise ValueError(f"{summary_path}: referenced sampling CSV does not exist")


def _parse_candidate_rmsds(raw: object, *, label: str, expected_num_samples: int) -> list[float]:
    try:
        values = json.loads(str(raw))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label}: candidate_rmsds_json is invalid JSON") from exc
    if not isinstance(values, list) or len(values) != expected_num_samples:
        raise ValueError(
            f"{label}: expected {expected_num_samples} candidate RMSDs, "
            f"got {len(values) if isinstance(values, list) else type(values).__name__}"
        )
    parsed = [
        _finite_float(value, label=f"{label}.candidate_rmsds_json[{index}]")
        for index, value in enumerate(values)
    ]
    if any(value < 0.0 for value in parsed):
        raise ValueError(f"{label}: candidate RMSD cannot be negative")
    return parsed


def _parse_row(
    row: dict[str, str],
    *,
    dataset: str,
    arm: str,
    source: Path,
    root: Path,
    expected_num_samples: int,
    expected_prior_pool_size: int,
    expected_ligand_input_sha256: str,
) -> dict[str, Any]:
    complex_id = str(row.get("id", "")).strip()
    if not complex_id:
        raise ValueError(f"{source}: row is missing id")
    label = f"{source}:{dataset}/{arm}/{complex_id}"
    if _strict_int(row.get("num_samples"), label=f"{label}.num_samples") != expected_num_samples:
        raise ValueError(f"{label}: num_samples differs from {expected_num_samples}")

    rmsds = _parse_candidate_rmsds(
        row.get("candidate_rmsds_json"),
        label=label,
        expected_num_samples=expected_num_samples,
    )
    k2 = _strict_int(
        row.get("num_rmsd_lt2_candidates"),
        label=f"{label}.num_rmsd_lt2_candidates",
    )
    recomputed_k2 = sum(value < 2.0 for value in rmsds)
    if k2 != recomputed_k2:
        raise ValueError(
            f"{label}: strict RMSD <2 count mismatch: row={k2}, recomputed={recomputed_k2}"
        )

    fast_valid_k2 = _strict_int(
        row.get("num_fast_valid_rmsd_lt2_candidates"),
        label=f"{label}.num_fast_valid_rmsd_lt2_candidates",
    )
    fast_valid = _strict_int(
        row.get("num_fast_valid_candidates"),
        label=f"{label}.num_fast_valid_candidates",
    )
    if not 0 <= fast_valid_k2 <= min(k2, fast_valid) or fast_valid > expected_num_samples:
        raise ValueError(f"{label}: invalid fast-valid candidate counts")

    oracle_rmsd = _finite_float(row.get("oracle_rmsd"), label=f"{label}.oracle_rmsd")
    if not math.isclose(oracle_rmsd, min(rmsds), rel_tol=1e-10, abs_tol=1e-10):
        raise ValueError(f"{label}: oracle_rmsd differs from candidate minimum")
    mean_sample_rmsd = _finite_float(
        row.get("mean_sample_rmsd"), label=f"{label}.mean_sample_rmsd"
    )
    if not math.isclose(
        mean_sample_rmsd,
        math.fsum(rmsds) / len(rmsds),
        rel_tol=1e-10,
        abs_tol=1e-10,
    ):
        raise ValueError(f"{label}: mean_sample_rmsd differs from candidate mean")
    fraction_k2 = _finite_float(
        row.get("fraction_rmsd_lt2_candidates"),
        label=f"{label}.fraction_rmsd_lt2_candidates",
    )
    if not math.isclose(
        fraction_k2,
        k2 / expected_num_samples,
        rel_tol=1e-10,
        abs_tol=1e-10,
    ):
        raise ValueError(f"{label}: fraction_rmsd_lt2_candidates is inconsistent")

    first_index = _strict_int(row.get("first_index"), label=f"{label}.first_index")
    if first_index != 0:
        raise ValueError(f"{label}.first_index: expected frozen first index 0")
    first_rmsd = _finite_float(row.get("first_rmsd"), label=f"{label}.first_rmsd")
    if not math.isclose(
        first_rmsd,
        rmsds[first_index],
        rel_tol=1e-10,
        abs_tol=1e-10,
    ):
        raise ValueError(f"{label}: first_rmsd differs from indexed candidate RMSD")

    if (
        _strict_int(row.get("prior_pool_size"), label=f"{label}.prior_pool_size")
        != expected_prior_pool_size
    ):
        raise ValueError(
            f"{label}.prior_pool_size: expected {expected_prior_pool_size}"
        )
    if row.get("guidance_mode") != "none":
        raise ValueError(f"{label}.guidance_mode: expected 'none'")
    if row.get("sampling_dynamics") != "deterministic_ode":
        raise ValueError(f"{label}.sampling_dynamics: expected 'deterministic_ode'")
    _require_float(
        row.get("translation_sde_base_sigma"),
        0.0,
        label=f"{label}.translation_sde_base_sigma",
    )
    if not _strict_bool(
        row.get("full_heavy_atom_bijection"),
        label=f"{label}.full_heavy_atom_bijection",
    ):
        raise ValueError(f"{label}.full_heavy_atom_bijection: expected True")

    protein_path, protein_sha256 = _validate_file_identity(
        row.get("protein"),
        row.get("protein_sha256"),
        source=source,
        root=root,
        label=f"{label}.protein",
    )
    ligand_reference_path, ligand_reference_sha256 = _validate_file_identity(
        row.get("ligand_ref"),
        row.get("ligand_reference_sha256"),
        source=source,
        root=root,
        label=f"{label}.ligand_reference",
    )
    ligand_input_sha256 = _sha256(
        row.get("ligand_input_identity_sha256"),
        label=f"{label}.ligand_input_identity_sha256",
    )
    if ligand_input_sha256 != expected_ligand_input_sha256:
        raise ValueError(
            f"{label}.ligand_input_identity_sha256: differs from frozen summary identity"
        )

    all_poses_count = _strict_int(
        row.get("all_poses_count"), label=f"{label}.all_poses_count"
    )
    if all_poses_count != expected_num_samples:
        raise ValueError(
            f"{label}.all_poses_count: expected {expected_num_samples}, "
            f"got {all_poses_count}"
        )
    all_poses_path, all_poses_sha256 = _validate_file_identity(
        row.get("all_poses_sdf"),
        row.get("all_poses_sdf_sha256"),
        source=source,
        root=root,
        label=f"{label}.all_poses_sdf",
    )

    sampling_seed = _strict_int(row.get("sampling_seed"), label=f"{label}.sampling_seed")
    prior_hash = _sha256(
        row.get("prior_pool_sha256"), label=f"{label}.prior_pool_sha256"
    )
    return {
        "dataset": dataset,
        "id": complex_id,
        "arm": arm,
        "sampling_seed": sampling_seed,
        "prior_pool_sha256": prior_hash,
        "k2": k2,
        "fast_valid_k2": fast_valid_k2,
        "oracle_rmsd": oracle_rmsd,
        "first_index": first_index,
        "first_rmsd": first_rmsd,
        "protein": protein_path,
        "protein_sha256": protein_sha256,
        "ligand_ref": ligand_reference_path,
        "ligand_reference_sha256": ligand_reference_sha256,
        "ligand_input_identity_sha256": ligand_input_sha256,
        "prior_pool_size": expected_prior_pool_size,
        "guidance_mode": "none",
        "sampling_dynamics": "deterministic_ode",
        "all_poses_count": all_poses_count,
        "all_poses_sdf": all_poses_path,
        "all_poses_sdf_sha256": all_poses_sha256,
    }


def _validate_summary_contract(
    summary: dict[str, Any],
    *,
    source: Path,
    dataset: str,
    arm: str,
    expected_num_samples: int,
    expected_num_steps: int,
    expected_prior_pool_size: int,
) -> dict[str, Any]:
    label = str(source)
    if summary.get("protocol_id") != FROZEN_PROTOCOL_ID:
        raise ValueError(
            f"{label}.protocol_id: expected {FROZEN_PROTOCOL_ID!r}, "
            f"got {summary.get('protocol_id')!r}"
        )
    _require_frozen_sha256(
        summary.get("checkpoint_sha256"),
        FROZEN_CHECKPOINT_SHA256[arm],
        label=f"{label}.checkpoint_sha256",
    )
    _require_frozen_sha256(
        summary.get("config_sha256"),
        FROZEN_CONFIG_SHA256,
        label=f"{label}.config_sha256",
    )
    _require_frozen_sha256(
        summary.get("eligibility_manifest_sha256"),
        FROZEN_ELIGIBILITY_MANIFEST_SHA256,
        label=f"{label}.eligibility_manifest_sha256",
    )
    _require_frozen_sha256(
        summary.get("pocket_centers_sha256"),
        FROZEN_POCKET_CENTERS_SHA256[dataset],
        label=f"{label}.pocket_centers_sha256",
    )
    if summary.get("require_full_ligand_atom_mapping") is not True:
        raise ValueError(f"{label}.require_full_ligand_atom_mapping: expected True")
    if summary.get("require_complete_success") is not True:
        raise ValueError(f"{label}.require_complete_success: expected True")

    discovered = EXPECTED_COUNTS[dataset]
    if (
        _strict_int(
            summary.get("expected_discovered_count"),
            label=f"{label}.expected_discovered_count",
        )
        != discovered
    ):
        raise ValueError(f"{label}.expected_discovered_count: expected {discovered}")
    if (
        _strict_int(
            summary.get("num_discovered_total"),
            label=f"{label}.num_discovered_total",
        )
        != discovered
    ):
        raise ValueError(f"{label}.num_discovered_total: expected exactly {discovered}")

    exact_int_settings = {
        "num_samples": expected_num_samples,
        "num_steps": expected_num_steps,
        "model_pose_step_budget": expected_num_samples * expected_num_steps,
        "prior_pool_size": expected_prior_pool_size,
        "seed": FROZEN_BASE_SEED,
    }
    for key, expected in exact_int_settings.items():
        if _strict_int(summary.get(key), label=f"{label}.{key}") != expected:
            raise ValueError(f"{label}.{key}: expected {expected}")
    exact_float_settings = {
        "sigma": 2.0,
        "schedule_power": 3.0,
        "pocket_cutoff": 10.0,
        "center_jitter_sigma": 0.0,
        "translation_sde_base_sigma": 0.0,
        "vina_guidance_scale": 0.0,
        "unified_guidance_scale": 0.0,
    }
    for key, expected in exact_float_settings.items():
        _require_float(summary.get(key), expected, label=f"{label}.{key}")
    if summary.get("sigma_list") != []:
        raise ValueError(f"{label}.sigma_list: expected an empty list")
    if summary.get("sigma_counts") != []:
        raise ValueError(f"{label}.sigma_counts: expected an empty list")
    if summary.get("time_schedule") != "late":
        raise ValueError(f"{label}.time_schedule: expected 'late'")
    if summary.get("refine") != "none":
        raise ValueError(f"{label}.refine: expected 'none'")
    if summary.get("prior_pool_hash_contract") != FROZEN_PRIOR_POOL_HASH_CONTRACT:
        raise ValueError(
            f"{label}.prior_pool_hash_contract: expected "
            f"{FROZEN_PRIOR_POOL_HASH_CONTRACT!r}"
        )
    if summary.get("sampling_dynamics_contract") != FROZEN_SAMPLING_DYNAMICS_CONTRACT:
        raise ValueError(
            f"{label}.sampling_dynamics_contract: expected deterministic ODE"
        )
    for key in (
        "confidence_checkpoint",
        "confidence_checkpoint_sha256",
        "confidence_step",
    ):
        if summary.get(key) is not None:
            raise ValueError(f"{label}.{key}: expected null")

    return _validate_benchmark_input_identity(
        summary.get("benchmark_input_identity"),
        dataset=dataset,
        source=source,
    )


def _load_cell(
    summaries: list[tuple[Path, dict[str, Any]]],
    *,
    root: Path,
    dataset: str,
    arm: str,
    expected_count: int,
    expected_shards: int,
    expected_num_samples: int,
    expected_num_steps: int,
    expected_prior_pool_size: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any], dict[str, Any]]:
    if len(summaries) != expected_shards:
        raise ValueError(
            f"{dataset}/{arm}: expected {expected_shards} shard summaries, "
            f"found {len(summaries)}"
        )
    by_shard: dict[int, tuple[Path, dict[str, Any]]] = {}
    for path, summary in summaries:
        shard = _strict_int(summary.get("shard_index"), label=f"{path}.shard_index")
        num_shards = _strict_int(summary.get("num_shards"), label=f"{path}.num_shards")
        if num_shards != expected_shards:
            raise ValueError(f"{path}: num_shards differs from {expected_shards}")
        if shard in by_shard:
            raise ValueError(f"{dataset}/{arm}: duplicate shard {shard}")
        by_shard[shard] = (path, summary)
    if set(by_shard) != set(range(expected_shards)):
        raise ValueError(f"{dataset}/{arm}: shard indices are incomplete")

    rows_by_id: dict[str, dict[str, Any]] = {}
    discovered_totals: set[int] = set()
    checkpoint_hashes: set[str] = set()
    protocol_ids: set[str] = set()
    benchmark_input_identities: dict[str, dict[str, Any]] = {}
    for shard in range(expected_shards):
        path, summary = by_shard[shard]
        benchmark_input_identity = _validate_summary_contract(
            summary,
            source=path,
            dataset=dataset,
            arm=arm,
            expected_num_samples=expected_num_samples,
            expected_num_steps=expected_num_steps,
            expected_prior_pool_size=expected_prior_pool_size,
        )
        canonical_input_identity = json.dumps(
            benchmark_input_identity, separators=(",", ":"), sort_keys=True
        )
        benchmark_input_identities[canonical_input_identity] = benchmark_input_identity
        num_failed = _strict_int(summary.get("num_failed"), label=f"{path}.num_failed")
        failures = summary.get("failures")
        if num_failed != 0 or failures != []:
            raise ValueError(f"{path}: shard recorded evaluation failures")
        num_assigned = _strict_int(
            summary.get("num_assigned"), label=f"{path}.num_assigned"
        )
        num_success = _strict_int(summary.get("num_success"), label=f"{path}.num_success")
        if num_success != num_assigned:
            raise ValueError(f"{path}: num_success differs from num_assigned")
        if (
            _strict_int(summary.get("num_samples"), label=f"{path}.num_samples")
            != expected_num_samples
        ):
            raise ValueError(f"{path}: num_samples differs from {expected_num_samples}")
        discovered_total = _strict_int(
            summary.get("num_discovered_total"), label=f"{path}.num_discovered_total"
        )
        discovered_totals.add(discovered_total)
        checkpoint_hashes.add(str(summary["checkpoint_sha256"]).lower())
        protocol_ids.add(str(summary["protocol_id"]))

        csv_path = _resolve_csv_path(path, summary, root)
        with csv_path.open(newline="", encoding="utf-8") as handle:
            raw_rows = list(csv.DictReader(handle))
        if len(raw_rows) != num_success:
            raise ValueError(
                f"{path}: CSV row count {len(raw_rows)} differs from num_success {num_success}"
            )
        for raw_row in raw_rows:
            complex_id = str(raw_row.get("id", "")).strip()
            frozen_input_record = benchmark_input_identity["per_id"].get(complex_id)
            if not isinstance(frozen_input_record, dict):
                raise ValueError(
                    f"{csv_path}:{dataset}/{arm}/{complex_id}: "
                    "ID is absent from benchmark_input_identity"
                )
            row = _parse_row(
                raw_row,
                dataset=dataset,
                arm=arm,
                source=csv_path,
                root=root,
                expected_num_samples=expected_num_samples,
                expected_prior_pool_size=expected_prior_pool_size,
                expected_ligand_input_sha256=str(frozen_input_record["sha256"]),
            )
            sorted_full_ids = sorted(benchmark_input_identity["per_id"])
            expected_seed = FROZEN_BASE_SEED + sorted_full_ids.index(row["id"]) + 1
            if row["sampling_seed"] != expected_seed:
                raise ValueError(
                    f"{csv_path}:{dataset}/{arm}/{row['id']}.sampling_seed: "
                    f"expected frozen per-ID seed {expected_seed}, "
                    f"got {row['sampling_seed']}"
                )
            if row["id"] in rows_by_id:
                raise ValueError(f"{dataset}/{arm}: duplicate complex ID {row['id']}")
            rows_by_id[row["id"]] = row

    if len(discovered_totals) != 1:
        raise ValueError(f"{dataset}/{arm}: num_discovered_total changed across shards")
    if len(rows_by_id) != expected_count:
        raise ValueError(
            f"{dataset}/{arm}: expected {expected_count} unique successful IDs, "
            f"found {len(rows_by_id)}"
        )
    if len(checkpoint_hashes) > 1:
        raise ValueError(f"{dataset}/{arm}: checkpoint hash changed across shards")
    if len(protocol_ids) > 1:
        raise ValueError(f"{dataset}/{arm}: protocol ID changed across shards")
    if len(benchmark_input_identities) != 1:
        raise ValueError(f"{dataset}/{arm}: benchmark input identity changed across shards")
    return rows_by_id, {
        "summary_count": len(summaries),
        "successful_complexes": len(rows_by_id),
        "num_discovered_total": next(iter(discovered_totals)),
        "checkpoint_sha256": next(iter(checkpoint_hashes), None),
        "protocol_id": next(iter(protocol_ids), None),
        "benchmark_input_identity_sha256": next(iter(benchmark_input_identities.values()))[
            "sha256"
        ],
    }, next(iter(benchmark_input_identities.values()))


def _aggregate(rows: list[dict[str, Any]], *, expected_num_samples: int) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot aggregate an empty cohort")
    k2 = [int(row["k2"]) for row in rows]
    fast_k2 = [int(row["fast_valid_k2"]) for row in rows]
    oracle = [float(row["oracle_rmsd"]) for row in rows]
    first = [float(row["first_rmsd"]) for row in rows]
    count = len(rows)
    return {
        "complexes": count,
        "candidates": count * expected_num_samples,
        "k2_total": sum(k2),
        "k2_mean": math.fsum(k2) / count,
        "p_k2_ge_1_pct": 100.0 * sum(value >= 1 for value in k2) / count,
        "p_k2_ge_5_pct": 100.0 * sum(value >= 5 for value in k2) / count,
        "p_k2_ge_10_pct": 100.0 * sum(value >= 10 for value in k2) / count,
        "oracle_rmsd_mean": math.fsum(oracle) / count,
        "oracle_rmsd_median": statistics.median(oracle),
        "oracle_rmsd_lt2_pct": 100.0 * sum(value < 2.0 for value in oracle) / count,
        "first_rmsd_mean": math.fsum(first) / count,
        "first_rmsd_median": statistics.median(first),
        "first_rmsd_lt2_pct": 100.0 * sum(value < 2.0 for value in first) / count,
        "fast_valid_k2_total": sum(fast_k2),
        "fast_valid_k2_mean": math.fsum(fast_k2) / count,
        "p_fast_valid_k2_ge_1_pct": (
            100.0 * sum(value >= 1 for value in fast_k2) / count
        ),
    }


def paired_bootstrap_mean_delta(
    deltas: list[float], *, seed: int, resamples: int
) -> dict[str, Any]:
    values = np.asarray(deltas, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("paired bootstrap requires a non-empty finite one-dimensional input")
    if resamples < 1:
        raise ValueError("bootstrap resamples must be positive")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(resamples, values.size))
    draws = values[indices].mean(axis=1)
    low, high = np.percentile(draws, [2.5, 97.5])
    return {
        "statistic": "paired_mean_delta_k2",
        "units": "candidates_per_complex",
        "delta": float(values.mean()),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "seed": seed,
        "resamples": resamples,
    }


def _paired_metrics(
    baseline: dict[tuple[str, str], dict[str, Any]],
    treatment: dict[tuple[str, str], dict[str, Any]],
    *,
    bootstrap_seed: int,
    bootstrap_resamples: int,
) -> dict[str, Any]:
    if set(baseline) != set(treatment):
        missing = sorted(set(baseline) - set(treatment))
        extra = sorted(set(treatment) - set(baseline))
        raise ValueError(f"paired ID mismatch: missing={missing[:3]}, extra={extra[:3]}")
    keys = sorted(baseline)
    deltas = [int(treatment[key]["k2"]) - int(baseline[key]["k2"]) for key in keys]
    fast_deltas = [
        int(treatment[key]["fast_valid_k2"]) - int(baseline[key]["fast_valid_k2"])
        for key in keys
    ]
    oracle_deltas = [
        float(treatment[key]["oracle_rmsd"]) - float(baseline[key]["oracle_rmsd"])
        for key in keys
    ]
    first_deltas = [
        float(treatment[key]["first_rmsd"]) - float(baseline[key]["first_rmsd"])
        for key in keys
    ]
    gained = sum(
        int(baseline[key]["k2"]) == 0 and int(treatment[key]["k2"]) >= 1 for key in keys
    )
    lost = sum(
        int(baseline[key]["k2"]) >= 1 and int(treatment[key]["k2"]) == 0 for key in keys
    )
    return {
        "complexes": len(keys),
        "delta_total_k2": sum(deltas),
        "delta_mean_k2": math.fsum(deltas) / len(keys),
        "delta_fast_valid_k2_total": sum(fast_deltas),
        "delta_fast_valid_k2_mean": math.fsum(fast_deltas) / len(keys),
        "delta_oracle_rmsd_mean": math.fsum(oracle_deltas) / len(keys),
        "delta_first_rmsd_mean": math.fsum(first_deltas) / len(keys),
        "positive_complexes": sum(value > 0 for value in deltas),
        "negative_complexes": sum(value < 0 for value in deltas),
        "tied_complexes": sum(value == 0 for value in deltas),
        "k2_ge_1_gained_complexes": gained,
        "k2_ge_1_lost_complexes": lost,
        "net_k2_ge_1": gained - lost,
        "paired_bootstrap_ci95": paired_bootstrap_mean_delta(
            deltas,
            seed=bootstrap_seed,
            resamples=bootstrap_resamples,
        ),
    }


def _engineering_stage(datasets: tuple[str, ...]) -> dict[str, Any]:
    if set(datasets) == set(DATASETS):
        return {
            "scope": "combined_astex_posebusters",
            "basis": "engineering_integrity_only_no_outcome_gate",
            "status": "complete_all_requested_datasets",
            "next_stage": None,
            "metric_thresholds": None,
        }
    return {
        "scope": "astex",
        "basis": "engineering_integrity_only_no_outcome_gate",
        "next_stage": "run_posebusters_after_integrity_pass",
        "metric_thresholds": None,
    }


def build_report(
    output_root: Path,
    *,
    expected_counts: dict[str, int] | None = None,
    datasets: tuple[str, ...] = DATASETS,
    expected_shards: int = EXPECTED_SHARDS,
    expected_num_samples: int = EXPECTED_NUM_SAMPLES,
    expected_num_steps: int = EXPECTED_NUM_STEPS,
    expected_prior_pool_size: int = EXPECTED_PRIOR_POOL_SIZE,
    bootstrap_seed: int = BOOTSTRAP_SEED,
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES,
) -> dict[str, Any]:
    expected_counts = dict(EXPECTED_COUNTS if expected_counts is None else expected_counts)
    if not datasets or len(set(datasets)) != len(datasets) or any(
        dataset not in DATASETS for dataset in datasets
    ):
        raise ValueError(f"datasets must be a non-empty unique subset of {DATASETS}")
    if any(
        dataset not in expected_counts
        or not isinstance(expected_counts[dataset], int)
        or isinstance(expected_counts[dataset], bool)
        or expected_counts[dataset] < 1
        for dataset in datasets
    ):
        raise ValueError(f"expected_counts must contain positive counts for {datasets}")
    expected_counts = {dataset: expected_counts[dataset] for dataset in datasets}
    if (
        expected_shards < 1
        or expected_num_samples < 1
        or expected_num_steps < 1
        or expected_prior_pool_size < expected_num_samples
    ):
        raise ValueError(
            "expected shards/samples/steps must be positive and prior pool size "
            "must be at least num samples"
        )
    if not output_root.is_dir():
        raise ValueError(f"output root is not a directory: {output_root}")

    discovered: dict[tuple[str, str], list[tuple[Path, dict[str, Any]]]] = defaultdict(list)
    summary_paths = sorted(output_root.rglob("*.summary.json"))
    for path in summary_paths:
        summary = _load_json_object(path)
        dataset = summary.get("dataset")
        if dataset not in datasets:
            continue
        arm = _identify_arm(summary, path)
        discovered[(str(dataset), arm)].append((path, summary))

    replay_present_by_dataset = {
        dataset: bool(discovered[(dataset, REPLAY_ARM)]) for dataset in datasets
    }
    if any(replay_present_by_dataset.values()) and not all(replay_present_by_dataset.values()):
        raise ValueError("current_raw_replay must be present for every selected dataset or none")
    active_arms = (*ARMS, REPLAY_ARM) if all(replay_present_by_dataset.values()) else ARMS

    cells: dict[str, dict[str, dict[str, dict[str, Any]]]] = {
        dataset: {} for dataset in datasets
    }
    inventory: dict[str, dict[str, dict[str, Any]]] = {dataset: {} for dataset in datasets}
    benchmark_input_identities: dict[str, dict[str, dict[str, Any]]] = {
        dataset: {} for dataset in datasets
    }
    for dataset in datasets:
        for arm in active_arms:
            rows, metadata, benchmark_input_identity = _load_cell(
                discovered[(dataset, arm)],
                root=output_root,
                dataset=dataset,
                arm=arm,
                expected_count=expected_counts[dataset],
                expected_shards=expected_shards,
                expected_num_samples=expected_num_samples,
                expected_num_steps=expected_num_steps,
                expected_prior_pool_size=expected_prior_pool_size,
            )
            cells[dataset][arm] = rows
            inventory[dataset][arm] = metadata
            benchmark_input_identities[dataset][arm] = benchmark_input_identity

    protocol_ids = {
        metadata["protocol_id"]
        for by_arm in inventory.values()
        for metadata in by_arm.values()
        if metadata["protocol_id"] is not None
    }
    if len(protocol_ids) > 1:
        raise ValueError(f"protocol ID differs across cells: {sorted(protocol_ids)}")
    checkpoint_hashes_by_arm: dict[str, str | None] = {}
    for arm in active_arms:
        hashes = {
            inventory[dataset][arm]["checkpoint_sha256"]
            for dataset in datasets
            if inventory[dataset][arm]["checkpoint_sha256"] is not None
        }
        if len(hashes) > 1:
            raise ValueError(f"{arm}: checkpoint hash differs across datasets")
        checkpoint_hashes_by_arm[arm] = next(iter(hashes), None)

    pairing: dict[str, Any] = {}
    replay_checks: dict[str, Any] = {}
    for dataset in datasets:
        identities = list(benchmark_input_identities[dataset].values())
        if any(identity != identities[0] for identity in identities[1:]):
            raise ValueError(f"{dataset}: benchmark input identity differs across arms")
        id_sets = {arm: set(cells[dataset][arm]) for arm in active_arms}
        if any(ids != id_sets[ARMS[0]] for ids in id_sets.values()):
            raise ValueError(f"{dataset}: complex IDs differ across arms")
        for complex_id in sorted(id_sets[ARMS[0]]):
            rows = [cells[dataset][arm][complex_id] for arm in active_arms]
            seeds = {int(row["sampling_seed"]) for row in rows}
            prior_hashes = {str(row["prior_pool_sha256"]) for row in rows}
            protein_hashes = {str(row["protein_sha256"]) for row in rows}
            ligand_reference_hashes = {
                str(row["ligand_reference_sha256"]) for row in rows
            }
            ligand_input_hashes = {
                str(row["ligand_input_identity_sha256"]) for row in rows
            }
            if len(seeds) != 1:
                raise ValueError(f"{dataset}/{complex_id}: sampling_seed differs across arms")
            if len(prior_hashes) != 1:
                raise ValueError(f"{dataset}/{complex_id}: prior_pool_sha256 differs across arms")
            if len(protein_hashes) != 1:
                raise ValueError(f"{dataset}/{complex_id}: protein_sha256 differs across arms")
            if len(ligand_reference_hashes) != 1:
                raise ValueError(
                    f"{dataset}/{complex_id}: ligand_reference_sha256 differs across arms"
                )
            if len(ligand_input_hashes) != 1:
                raise ValueError(
                    f"{dataset}/{complex_id}: ligand_input_identity_sha256 "
                    "differs across arms"
                )
        pairing[dataset] = {
            "complexes": len(id_sets[ARMS[0]]),
            "sampling_seed_equal_across_arms": True,
            "prior_pool_sha256_equal_across_arms": True,
            "protein_sha256_equal_across_arms": True,
            "ligand_reference_sha256_equal_across_arms": True,
            "ligand_input_identity_sha256_equal_across_arms": True,
        }
        if REPLAY_ARM in active_arms:
            mismatches = [
                complex_id
                for complex_id in sorted(id_sets[ARMS[0]])
                if (
                    int(cells[dataset]["current_raw"][complex_id]["k2"])
                    != int(cells[dataset][REPLAY_ARM][complex_id]["k2"])
                    or (
                        int(cells[dataset]["current_raw"][complex_id]["k2"]) >= 1
                        != (int(cells[dataset][REPLAY_ARM][complex_id]["k2"]) >= 1)
                    )
                )
            ]
            if mismatches:
                raise ValueError(
                    f"{dataset}: current_raw replay K2 mismatch for {mismatches[:3]}"
                )
            replay_checks[dataset] = {
                "complexes": len(id_sets[ARMS[0]]),
                "exact_k2_equal_per_id": True,
                "k2_ge_1_equal_per_id": True,
            }

    arm_metrics: dict[str, Any] = {}
    for arm in active_arms:
        dataset_metrics = {
            dataset: _aggregate(
                [
                    cells[dataset][arm][complex_id]
                    for complex_id in sorted(cells[dataset][arm])
                ],
                expected_num_samples=expected_num_samples,
            )
            for dataset in datasets
        }
        combined_rows = [
            cells[dataset][arm][complex_id]
            for dataset in datasets
            for complex_id in sorted(cells[dataset][arm])
        ]
        arm_metrics[arm] = {
            "checkpoint_sha256": checkpoint_hashes_by_arm[arm],
            "datasets": dataset_metrics,
            "combined": _aggregate(
                combined_rows, expected_num_samples=expected_num_samples
            ),
        }

    comparisons: dict[str, Any] = {}
    for name, baseline_arm, treatment_arm, role in COMPARISONS:
        scopes: dict[str, Any] = {}
        for dataset in datasets:
            baseline = {
                (dataset, complex_id): row
                for complex_id, row in cells[dataset][baseline_arm].items()
            }
            treatment = {
                (dataset, complex_id): row
                for complex_id, row in cells[dataset][treatment_arm].items()
            }
            scopes[dataset] = _paired_metrics(
                baseline,
                treatment,
                bootstrap_seed=bootstrap_seed,
                bootstrap_resamples=bootstrap_resamples,
            )
        combined_baseline = {
            (dataset, complex_id): row
            for dataset in datasets
            for complex_id, row in cells[dataset][baseline_arm].items()
        }
        combined_treatment = {
            (dataset, complex_id): row
            for dataset in datasets
            for complex_id, row in cells[dataset][treatment_arm].items()
        }
        scopes["combined"] = _paired_metrics(
            combined_baseline,
            combined_treatment,
            bootstrap_seed=bootstrap_seed,
            bootstrap_resamples=bootstrap_resamples,
        )
        comparisons[name] = {
            "role": role,
            "baseline_arm": baseline_arm,
            "treatment_arm": treatment_arm,
            **scopes,
        }

    report = {
        "schema_version": "effdock.early_time_t0p10_50k_external_paired_report.v1",
        "status": "complete_strict_three_arm_paired",
        "created_utc": datetime.now(UTC).isoformat(),
        "protocol_id": next(iter(protocol_ids), None),
        "configuration": {
            "arms": list(active_arms),
            "primary_arms": list(ARMS),
            "optional_replay_arm_present": REPLAY_ARM in active_arms,
            "datasets": list(datasets),
            "expected_counts": expected_counts,
            "expected_shards": expected_shards,
            "expected_num_samples": expected_num_samples,
            "expected_num_steps": expected_num_steps,
            "expected_prior_pool_size": expected_prior_pool_size,
            "k2_definition": "count of candidates with runtime pose RMSD strictly < 2 Angstrom",
            "rmsd_implementation": {
                "preferred": (
                    "RDKit rdMolAlign.CalcRMS symmetry-aware heavy-atom RMSD without "
                    "alignment when the full-topology path succeeds"
                ),
                "fallback": (
                    "index-wise RMSD over the full-heavy-atom mapped subset when "
                    "CalcRMS is unavailable or fails"
                ),
                "metric_caveat": (
                    "K2 uses the stored runtime RMSD values; fallback cases are not "
                    "retrospectively reclassified as symmetry-aware"
                ),
            },
            "bootstrap": {
                "method": "paired complex-ID percentile bootstrap of mean delta K2",
                "seed": bootstrap_seed,
                "resamples": bootstrap_resamples,
                "confidence_interval_pct": 95.0,
            },
        },
        "integrity": {
            "summary_files_used": len(datasets) * len(active_arms) * expected_shards,
            "inventory": inventory,
            "pairing": pairing,
            "checkpoint_sha256_by_arm": checkpoint_hashes_by_arm,
            "current_raw_replay": replay_checks if replay_checks else None,
        },
        "arms": arm_metrics,
        "comparisons": comparisons,
    }
    if "astex" in datasets:
        report["engineering_stage"] = _engineering_stage(datasets)
    return report


def write_report(report: dict[str, Any], output_json: Path, *, overwrite: bool = False) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    if output_json.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {output_json}")
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
        if output_json.exists() and not overwrite:
            raise FileExistsError(f"refusing to overwrite {output_json}")
        os.replace(temporary, output_json)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=DATASETS,
        default=list(DATASETS),
        help="Dataset stages to require and aggregate (Astex-only is the integrity stage).",
    )
    parser.add_argument("--expected-shards", type=int, default=EXPECTED_SHARDS)
    parser.add_argument("--expected-num-samples", type=int, default=EXPECTED_NUM_SAMPLES)
    parser.add_argument("--expected-num-steps", type=int, default=EXPECTED_NUM_STEPS)
    parser.add_argument(
        "--expected-prior-pool-size",
        type=int,
        default=EXPECTED_PRIOR_POOL_SIZE,
    )
    parser.add_argument("--expected-astex-count", type=int, default=EXPECTED_COUNTS["astex"])
    parser.add_argument(
        "--expected-posebusters-count",
        type=int,
        default=EXPECTED_COUNTS["posebusters"],
    )
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--bootstrap-resamples", type=int, default=BOOTSTRAP_RESAMPLES)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    report = build_report(
        args.output_root,
        expected_counts={
            "astex": args.expected_astex_count,
            "posebusters": args.expected_posebusters_count,
        },
        datasets=tuple(args.datasets),
        expected_shards=args.expected_shards,
        expected_num_samples=args.expected_num_samples,
        expected_num_steps=args.expected_num_steps,
        expected_prior_pool_size=args.expected_prior_pool_size,
        bootstrap_seed=args.bootstrap_seed,
        bootstrap_resamples=args.bootstrap_resamples,
    )
    write_report(report, args.output_json, overwrite=args.overwrite)
    print(
        json.dumps(
            {
                "status": report["status"],
                "complexes": sum(report["configuration"]["expected_counts"].values()),
                "output_json": str(args.output_json),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
