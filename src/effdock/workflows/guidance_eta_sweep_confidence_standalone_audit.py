#!/usr/bin/env python3
"""Fail-closed integrity audit for the standalone confidence eta sweep.

This audit describes one fresh CUDA sampling pass.  It deliberately makes no
claim that a second pass would reproduce the generated coordinates.  Instead,
it verifies the frozen scientific settings, complete within-run eta pairing,
confidence score/selector ledgers, and every current input/selected-pose file
binding needed by the downstream selected-pose PoseBusters report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from effdock.confidence.selectors import ConfidenceFilterConfig, select_confidence_filter
from effdock.guidance.provenance import guidance_implementation_identity
from effdock.workflows import guidance_eta_sweep_confidence_identity as replay
from effdock.workflows.evaluate import sorted_id_sha256
from effdock.workflows.guidance_eta_sweep_report import DATASETS
from effdock.workflows.guidance_eta_sweep_standalone_spec import (
    LEGACY_V1,
    PROFILES,
    STERIC_HIGH_ETA_V1,
    StandaloneSweepSpec,
    get_standalone_sweep_spec,
)

PROTOCOL_ID = LEGACY_V1.protocol_id
AUDIT_CONTRACT = LEGACY_V1.audit_contract
SCHEMA_VERSION = LEGACY_V1.audit_schema_version
ETA_VALUES = LEGACY_V1.eta_values
ETA_TAGS = LEGACY_V1.eta_tags
EXPECTED_ELIGIBILITY_MANIFEST_SHA256 = (
    "dac7903488ccd36552a9bca134e37e633e3f07166d94f0389837012081ff3048"
)

EXPECTED_DATASET_COUNTS = replay.EXPECTED_DATASET_COUNTS
EXPECTED_BENCHMARK_IDENTITIES = replay.EXPECTED_BENCHMARK_IDENTITIES
EXPECTED_CHECKPOINT_SHA256 = replay.EXPECTED_CHECKPOINT_SHA256
EXPECTED_CONFIG_SHA256 = replay.EXPECTED_CONFIG_SHA256
EXPECTED_CONFIDENCE_CHECKPOINT_SHA256 = replay.EXPECTED_CONFIDENCE_CHECKPOINT_SHA256
EXPECTED_POCKET_CENTERS_SHA256 = replay.EXPECTED_POCKET_CENTERS_SHA256
EXPECTED_GUIDANCE_PARAMETER_SHA256 = LEGACY_V1.guidance_parameter_sha256
EXPECTED_RECEPTOR_POLICY_SHA256 = LEGACY_V1.receptor_policy_sha256
ALLOWED_GPU_NAME_FRAGMENTS = replay.ALLOWED_GPU_NAME_FRAGMENTS
MIN_GPU_TOTAL_MEMORY_BYTES = replay.MIN_GPU_TOTAL_MEMORY_BYTES
GPU_MEMORY_LIMIT_BYTES = replay.GPU_MEMORY_LIMIT_BYTES
SMOKE_IDS = replay.SMOKE_IDS
SELECTOR_PROFILE = "confidence_cluster_free"
CONFIDENCE_SELECTORS = ("confidence", "confidence_filter")
# Keep the persisted-pose and summary-stat profiles explicit so a future
# protocol revision can narrow either surface without changing audit logic.
SAVED_SELECTORS = CONFIDENCE_SELECTORS
SUMMARY_STATS = ("first", "oracle", "candidate_set", *CONFIDENCE_SELECTORS)
CANDIDATE_ENSEMBLE_HASH_CONTRACT = replay.CANDIDATE_ENSEMBLE_HASH_CONTRACT
CONFIDENCE_SCORE_LEDGER_CONTRACT = replay.CONFIDENCE_SCORE_LEDGER_CONTRACT

_EXPECTED_FILTER_CONFIG = replay._EXPECTED_FILTER_CONFIG
_EXACT_SCORE_FIELDS = {*replay._RUNTIME_SCORE_FIELDS, "pl_clash_1p6"}
_CONFIDENCE_COLUMNS = {
    "candidate_ensemble_sha256",
    "confidence_candidate_scores_json",
    *(
        f"{selector}_fast_{term}"
        for selector in CONFIDENCE_SELECTORS
        for term in replay._FAST_TERMS
    ),
    *(
        f"{selector}_{suffix}"
        for selector in CONFIDENCE_SELECTORS
        for suffix in ("index", "rmsd", "pred_rmsd", "pred_success")
    ),
}
_REQUIRED_ROW_FIELDS = {
    "id",
    "selector_profile",
    "protein",
    "ligand_ref",
    "protein_sha256",
    "ligand_reference_sha256",
    "saved_pose_sha256_json",
    "num_samples",
    "prior_pool_size",
    "sampling_seed",
    "prior_pool_sha256",
    "guidance_direct_step_trace_json",
    *_CONFIDENCE_COLUMNS,
}
_V2_REQUIRED_ROW_FIELDS = {
    "all_poses_sdf",
    "all_poses_sdf_sha256",
    "all_poses_count",
}
_EXPECTED_STATS = set(SUMMARY_STATS)
_FORBIDDEN_SELECTOR_COLUMN_PREFIXES = ("vina_", "confidence_final_")
_BENCHMARK_INPUT_MANIFEST = (
    Path(__file__).resolve().parents[3] / "docs/GUIDANCE_BUDGET1000_FULL_INPUTS.json"
)


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _records_sha256(records: list[dict[str, Any]], *, domain: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(domain)
    for record in records:
        digest.update(_canonical_json(record).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _ordered_sha256(values: list[str], *, domain: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(domain)
    for value in values:
        digest.update(value.encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _validate_cohort_audit(path: Path) -> None:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"missing frozen cohort audit: {path}")
    actual = replay._file_sha256(path)
    if actual != EXPECTED_ELIGIBILITY_MANIFEST_SHA256:
        raise ValueError(
            f"{path}: frozen cohort-audit SHA-256 must be "
            f"{EXPECTED_ELIGIBILITY_MANIFEST_SHA256}, got {actual}"
        )


def _frozen_sampling_seed_by_dataset() -> dict[str, dict[str, int]]:
    if replay._file_sha256(_BENCHMARK_INPUT_MANIFEST) != (
        replay.EXPECTED_BENCHMARK_INPUT_MANIFEST_SHA256
    ):
        raise ValueError("frozen benchmark-input manifest SHA-256 mismatch")
    manifest = replay._load_json_object(_BENCHMARK_INPUT_MANIFEST)
    datasets = manifest.get("datasets")
    if not isinstance(datasets, dict) or set(datasets) != set(DATASETS):
        raise ValueError("frozen benchmark-input manifest dataset inventory mismatch")

    result: dict[str, dict[str, int]] = {}
    for dataset in DATASETS:
        dataset_record = datasets.get(dataset)
        ligands = dataset_record.get("ligands") if isinstance(dataset_record, dict) else None
        if not isinstance(ligands, dict):
            raise ValueError(f"frozen benchmark-input manifest lacks {dataset} ligands")
        ids = sorted(str(complex_id) for complex_id in ligands)
        if (
            len(ids) != EXPECTED_DATASET_COUNTS[dataset]
            or sorted_id_sha256(ids) != EXPECTED_BENCHMARK_IDENTITIES[dataset]["ids_sha256"]
        ):
            raise ValueError(f"frozen benchmark-input manifest {dataset} ID inventory mismatch")
        result[dataset] = {
            complex_id: 42 + global_index
            for global_index, complex_id in enumerate(ids, start=1)
        }
    return result


def _validate_summary(
    summary: dict[str, Any],
    *,
    path: Path,
    dataset: str,
    eta: float,
    shard_index: int,
    smoke: bool,
    current_implementation: dict[str, object],
    spec: StandaloneSweepSpec,
) -> None:
    if summary.get("protocol_id") != spec.protocol_id:
        raise ValueError(f"{path}: protocol_id must be {spec.protocol_id}")
    if summary.get("selector_profile") != SELECTOR_PROFILE:
        raise ValueError(f"{path}: selector_profile must be {SELECTOR_PROFILE}")
    if summary.get("eligibility_manifest_sha256") != EXPECTED_ELIGIBILITY_MANIFEST_SHA256:
        raise ValueError(f"{path}: frozen cohort-audit eligibility hash mismatch")

    # The replay validator already owns the complete frozen sampling contract.
    # Normalize only the newly named protocol field; no parent data are read or
    # compared here.
    normalized = dict(summary)
    normalized["protocol_id"] = replay.PROTOCOL_ID
    replay._validate_summary(
        normalized,
        path=path,
        dataset=dataset,
        eta=eta,
        shard_index=shard_index,
        smoke=smoke,
        confidence=True,
        expected_run_name_fn=spec.expected_run_name,
        expected_guidance_parameter_sha256=spec.guidance_parameter_sha256,
        expected_receptor_policy_sha256=spec.receptor_policy_sha256,
    )

    if eta > 0.0 and spec != LEGACY_V1:
        parameter_set = summary.get("guidance_parameter_set")
        physical = parameter_set.get("physical") if isinstance(parameter_set, dict) else None
        interaction = parameter_set.get("interaction") if isinstance(parameter_set, dict) else None
        expected_physical = {
            "sha256": spec.physical_parameter_sha256,
            "version": spec.physical_parameter_version,
            "formula_version": spec.physical_formula_version,
        }
        if not isinstance(physical, dict) or any(
            physical.get(key) != expected for key, expected in expected_physical.items()
        ):
            raise ValueError(f"{path}: physical guidance identity mismatch")
        if (
            not isinstance(interaction, dict)
            or interaction.get("sha256") != spec.interaction_parameter_sha256
        ):
            raise ValueError(f"{path}: interaction guidance identity mismatch")

    stats = summary.get("stats")
    if not isinstance(stats, dict) or set(stats) != _EXPECTED_STATS:
        raise ValueError(f"{path}: selector statistics are incomplete or unexpected")
    if summary.get("guidance_implementation") != current_implementation:
        raise ValueError(f"{path}: guidance implementation differs from current source/runtime")
    if spec != LEGACY_V1:
        allocated = int(summary["runtime"]["cuda_max_memory_allocated_bytes"])
        if allocated >= int(0.9 * GPU_MEMORY_LIMIT_BYTES):
            raise ValueError(f"{path}: CUDA allocated peak lacks the frozen 10% device headroom")


def _parse_sampling_seed(value: str, *, label: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer") from exc
    if str(parsed) != value or not 0 <= parsed < 2**63:
        raise ValueError(f"{label} must be a canonical non-negative 63-bit integer")
    return parsed


def _validate_score_ledger(row: dict[str, str], *, label: str) -> list[dict[str, float]]:
    try:
        raw = json.loads(row["confidence_candidate_scores_json"])
    except (KeyError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label}: invalid confidence_candidate_scores_json") from exc
    if not isinstance(raw, list) or len(raw) != replay.NUM_SAMPLES:
        raise ValueError(
            f"{label}: confidence score ledger must contain exactly {replay.NUM_SAMPLES} entries"
        )
    for index, entry in enumerate(raw):
        if not isinstance(entry, dict) or set(entry) != _EXACT_SCORE_FIELDS:
            raise ValueError(
                f"{label}: candidate {index} score fields must be exactly "
                f"{sorted(_EXACT_SCORE_FIELDS)}"
            )
    scores = replay._validate_candidate_scores(row, label=label)
    pred = torch.tensor([score["confidence_rmsd"] for score in scores], dtype=torch.float32)
    expected_primary = int(torch.argmin(pred))
    actual_primary = replay._parse_index(row, "confidence_index", label=label)
    if actual_primary != expected_primary:
        raise ValueError(f"{label}: confidence_index does not equal argmin confidence_rmsd")

    clash = torch.tensor([score["pl_clash_1p6"] for score in scores], dtype=torch.float32)
    expected_filter, _ = select_confidence_filter(scores, clash)
    actual_filter = replay._parse_index(row, "confidence_filter_index", label=label)
    if actual_filter != expected_filter:
        raise ValueError(f"{label}: confidence_filter_index does not match frozen filter")

    for selector in CONFIDENCE_SELECTORS:
        index = replay._parse_index(row, f"{selector}_index", label=label)
        for suffix, score_key in (
            ("pred_rmsd", "confidence_rmsd"),
            ("pred_success", "confidence_success"),
        ):
            try:
                recorded = float(row[f"{selector}_{suffix}"])
            except (KeyError, ValueError) as exc:
                raise ValueError(f"{label}: invalid {selector}_{suffix}") from exc
            if not torch.isfinite(torch.tensor(recorded)) or recorded != scores[index][score_key]:
                raise ValueError(f"{label}: {selector}_{suffix} differs from candidate score")
        try:
            rmsd = float(row[f"{selector}_rmsd"])
        except (KeyError, ValueError) as exc:
            raise ValueError(f"{label}: invalid {selector}_rmsd") from exc
        if not torch.isfinite(torch.tensor(rmsd)):
            raise ValueError(f"{label}: {selector}_rmsd must be finite")
        for term in replay._FAST_TERMS:
            if row.get(f"{selector}_fast_{term}") not in {"True", "False"}:
                raise ValueError(f"{label}: invalid {selector}_fast_{term}")
    return scores


def _resolve_current_file(raw_value: str, *, sampling_dir: Path, label: str) -> Path:
    if not isinstance(raw_value, str) or not raw_value:
        raise ValueError(f"{label} path must be non-empty")
    raw = Path(raw_value)
    candidates = [raw] if raw.is_absolute() else [Path.cwd() / raw, sampling_dir / raw]
    existing = {candidate.resolve() for candidate in candidates if candidate.is_file()}
    if len(existing) != 1:
        raise ValueError(
            f"{label} must resolve to exactly one current file, got {sorted(existing)}"
        )
    return next(iter(existing))


def _verify_current_file(
    raw_value: str,
    expected_sha256: str,
    *,
    sampling_dir: Path,
    label: str,
    digest_cache: dict[Path, str],
) -> Path:
    expected = replay._require_sha256(expected_sha256, label=f"{label}.sha256")
    path = _resolve_current_file(raw_value, sampling_dir=sampling_dir, label=label)
    actual = digest_cache.get(path)
    if actual is None:
        actual = replay._file_sha256(path)
        digest_cache[path] = actual
    if actual != expected:
        raise ValueError(f"{label}: current file SHA-256 differs from CSV binding")
    return path


def _validate_row(
    row: dict[str, str],
    *,
    fields: list[str],
    sampling_dir: Path,
    run_name: str,
    dataset: str,
    eta: float,
    shard_index: int,
    digest_cache: dict[Path, str],
    spec: StandaloneSweepSpec,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, str], list[dict[str, Any]]]:
    complex_id = row["id"]
    label = f"{run_name}/shard-{shard_index:03d}/{complex_id}"
    required = _REQUIRED_ROW_FIELDS | (
        _V2_REQUIRED_ROW_FIELDS if spec == STERIC_HIGH_ETA_V1 else set()
    )
    missing = sorted(required - set(fields))
    if missing:
        raise ValueError(f"{label}: CSV lacks required fields {missing}")
    if row["selector_profile"] != SELECTOR_PROFILE:
        raise ValueError(f"{label}: selector_profile must be {SELECTOR_PROFILE}")
    forbidden = sorted(
        field for field in fields if field.startswith(_FORBIDDEN_SELECTOR_COLUMN_PREFIXES)
    )
    if forbidden:
        raise ValueError(f"{label}: CSV contains selectors outside profile {forbidden}")
    try:
        row_num_samples = int(row["num_samples"])
    except ValueError as exc:
        raise ValueError(f"{label}.num_samples must be an integer") from exc
    if row_num_samples != replay.NUM_SAMPLES:
        raise ValueError(f"{label}.num_samples must be {replay.NUM_SAMPLES}")
    if row["prior_pool_size"] != "100":
        raise ValueError(f"{label}.prior_pool_size must be canonical integer 100")

    seed = _parse_sampling_seed(row["sampling_seed"], label=f"{label}.sampling_seed")
    prior_sha256 = replay._require_sha256(
        row["prior_pool_sha256"], label=f"{label}.prior_pool_sha256"
    )
    candidate_sha256 = replay._require_sha256(
        row["candidate_ensemble_sha256"], label=f"{label}.candidate_ensemble_sha256"
    )
    scores = _validate_score_ledger(row, label=label)

    trace = replay._validate_direct_step_trace(row, eta=eta)
    expected_trace_length = 0 if eta == 0.0 else 8
    if len(trace) != expected_trace_length:
        raise ValueError(f"{label}: direct-step trace does not match eta")

    protein_sha256 = replay._require_sha256(row["protein_sha256"], label=f"{label}.protein_sha256")
    reference_sha256 = replay._require_sha256(
        row["ligand_reference_sha256"], label=f"{label}.ligand_reference_sha256"
    )
    protein_path = _verify_current_file(
        row["protein"],
        protein_sha256,
        sampling_dir=sampling_dir,
        label=f"{label}.protein",
        digest_cache=digest_cache,
    )
    reference_path = _verify_current_file(
        row["ligand_ref"],
        reference_sha256,
        sampling_dir=sampling_dir,
        label=f"{label}.ligand_reference",
        digest_cache=digest_cache,
    )

    all_poses_record: dict[str, Any] | None = None
    if spec == STERIC_HIGH_ETA_V1:
        if row["all_poses_count"] != "100":
            raise ValueError(f"{label}.all_poses_count must be canonical integer 100")
        all_poses_sha256 = replay._require_sha256(
            row["all_poses_sdf_sha256"], label=f"{label}.all_poses_sdf_sha256"
        )
        all_poses_path = _verify_current_file(
            row["all_poses_sdf"],
            all_poses_sha256,
            sampling_dir=sampling_dir,
            label=f"{label}.all_poses_sdf",
            digest_cache=digest_cache,
        )
        persisted_count = sum(
            line.strip() == b"$$$$" for line in all_poses_path.read_bytes().splitlines()
        )
        if persisted_count != replay.NUM_SAMPLES:
            raise ValueError(
                f"{label}.all_poses_sdf must contain exactly {replay.NUM_SAMPLES} records"
            )
        all_poses_record = {
            "path": str(all_poses_path),
            "sha256": all_poses_sha256,
            "records": persisted_count,
            "coordinate_precision": "SDF_V2000_4_decimal_angstrom",
        }

    pose_ledger = replay._parse_pose_ledger(
        row["saved_pose_sha256_json"],
        label=f"{label}.saved_pose_sha256_json",
        expected=set(SAVED_SELECTORS),
    )
    pose_files: dict[str, dict[str, str]] = {}
    for selector in SAVED_SELECTORS:
        pose_path = replay._verify_pose_file(
            sampling_dir,
            run_name=run_name,
            dataset=dataset,
            complex_id=complex_id,
            selector=selector,
            expected_sha256=pose_ledger[selector],
        )
        pose_files[selector] = {
            "path": str(pose_path.resolve()),
            "sha256": pose_ledger[selector],
        }

    selector_indices = {
        selector: replay._parse_index(row, f"{selector}_index", label=label)
        for selector in CONFIDENCE_SELECTORS
    }
    score_ledger_sha256 = hashlib.sha256(_canonical_json(scores).encode()).hexdigest()
    row_record = {
        "id": complex_id,
        **({"prior_pool_size": 100} if spec == STERIC_HIGH_ETA_V1 else {}),
        "sampling_seed": seed,
        "prior_pool_sha256": prior_sha256,
        "candidate_ensemble": {
            "sha256": candidate_sha256,
            "status": "digest_present_and_producer_bound",
            "independently_recomputed": False,
        },
        "score_ledger_sha256": score_ledger_sha256,
        "score_entries": len(scores),
        "score_fields": sorted(_EXACT_SCORE_FIELDS),
        "selector_indices": selector_indices,
        **({"all_poses_sdf": all_poses_record} if all_poses_record is not None else {}),
    }
    file_record = {
        "id": complex_id,
        "protein": {"path": str(protein_path), "sha256": protein_sha256},
        "ligand_reference": {"path": str(reference_path), "sha256": reference_sha256},
        "selected_poses": pose_files,
        **({"all_poses_sdf": all_poses_record} if all_poses_record is not None else {}),
    }
    input_identity = {
        "id": complex_id,
        "protein_path": str(protein_path),
        "protein_sha256": protein_sha256,
        "ligand_reference_path": str(reference_path),
        "ligand_reference_sha256": reference_sha256,
    }
    return row_record, file_record, input_identity, trace


def _interleave_shards(shards: list[list[str]]) -> list[str]:
    return [
        values[position]
        for position in range(max((len(values) for values in shards), default=0))
        for values in shards
        if position < len(values)
    ]


def _build_prior_pool_sha256_diagnostics(
    observations_by_id: dict[str, list[dict[str, Any]]],
    *,
    ordered_ids: list[str],
    spec: StandaloneSweepSpec,
) -> dict[str, Any]:
    """Summarize cross-eta prior hash drift without treating it as pose drift."""
    if set(observations_by_id) != set(ordered_ids):
        raise RuntimeError("prior-pool SHA-256 diagnostic inventory is incomplete")

    mismatches: list[dict[str, Any]] = []
    for complex_id in ordered_ids:
        observations = observations_by_id[complex_id]
        observed_tags = [str(item["eta_tag"]) for item in observations]
        if observed_tags != list(spec.eta_tags):
            raise RuntimeError(
                f"{complex_id}: prior-pool SHA-256 observations do not cover the eta grid"
            )
        hash_set = sorted({str(item["prior_pool_sha256"]) for item in observations})
        if len(hash_set) == 1:
            continue
        partitions = sorted(
            {
                str(item["slurm_partition"])
                for item in observations
                if item.get("slurm_partition") is not None
            }
        )
        gpu_names = sorted(
            {str(item["gpu"]) for item in observations if item.get("gpu") is not None}
        )
        gpu_memory = sorted(
            {
                int(item["gpu_total_memory_bytes"])
                for item in observations
                if item.get("gpu_total_memory_bytes") is not None
            }
        )
        mismatches.append(
            {
                "id": complex_id,
                "prior_pool_sha256_set": hash_set,
                "eta_observations": observations,
                "runtime_context": {
                    "slurm_partitions": partitions,
                    "gpu_names": gpu_names,
                    "gpu_total_memory_bytes": gpu_memory,
                    "mixed_slurm_partitions": len(partitions) > 1,
                    "mixed_gpu_names": len(gpu_names) > 1,
                },
            }
        )

    return {
        "complexes": len(ordered_ids),
        "complexes_with_single_hash": len(ordered_ids) - len(mismatches),
        "complexes_with_multiple_hashes": len(mismatches),
        "mismatched_ids": [str(item["id"]) for item in mismatches],
        "mismatches": mismatches,
    }


def validate_v2_prior_pool_sha256_diagnostics(
    audit: dict[str, Any],
    *,
    spec: StandaloneSweepSpec,
) -> dict[str, Any]:
    """Validate the persisted high-eta V2 diagnostic and its count identities."""
    if spec != STERIC_HIGH_ETA_V1:
        raise ValueError("prior-pool SHA-256 diagnostic policy is defined only for high-eta V2")
    if audit.get("schema_version") != spec.audit_schema_version:
        raise ValueError("high-eta audit schema_version mismatch")
    checks = audit.get("checks")
    required_checks = {
        "within_run_sampling_seed_paired_across_eta",
        "sampling_seed_matches_frozen_sorted_id_offset_contract",
        "prior_pool_sha256_cross_eta_differences_recorded",
        "prior_pool_size_100_exact_in_every_csv_row",
        "all_poses_sdf_current_hash_and_100_record_count_exact",
    }
    if not isinstance(checks, dict) or any(checks.get(key) is not True for key in required_checks):
        raise ValueError("high-eta audit V2 prior/seed checks are incomplete")

    diagnostic = audit.get("prior_pool_sha256_diagnostics")
    if not isinstance(diagnostic, dict):
        raise ValueError("high-eta audit lacks prior_pool_sha256_diagnostics")
    expected_policy = {
        "policy": "record_only_across_eta",
        "per_row_sha256_format_verified": True,
        "cross_eta_sha256_equality_required": False,
        "sampling_seed_equality_required": True,
        "sampling_seed_mapping": "base_seed_42_plus_one_based_sorted_dataset_id_index",
        "declared_prior_pool_size": 100,
        "declared_prior_pool_hash_contract": "EFFDOCK_SHARED_PRIOR_V1",
    }
    if any(diagnostic.get(key) != value for key, value in expected_policy.items()):
        raise ValueError("high-eta audit prior-pool SHA-256 policy mismatch")

    coverage = audit.get("coverage")
    per_dataset_coverage = coverage.get("per_dataset") if isinstance(coverage, dict) else None
    datasets = diagnostic.get("datasets")
    if (
        not isinstance(per_dataset_coverage, dict)
        or not isinstance(datasets, dict)
        or set(datasets) != set(DATASETS)
    ):
        raise ValueError("high-eta audit prior-pool diagnostic dataset inventory mismatch")

    def require_count(container: dict[str, Any], key: str, *, label: str) -> int:
        value = container.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{label}.{key} must be a non-negative integer")
        return value

    flattened_ids: list[dict[str, str]] = []
    dataset_complexes = 0
    for dataset in DATASETS:
        current = datasets[dataset]
        current_coverage = per_dataset_coverage.get(dataset)
        if not isinstance(current, dict) or not isinstance(current_coverage, dict):
            raise ValueError(f"{dataset}: invalid prior-pool diagnostic")
        complexes = require_count(current, "complexes", label=dataset)
        single = require_count(current, "complexes_with_single_hash", label=dataset)
        multiple = require_count(current, "complexes_with_multiple_hashes", label=dataset)
        if complexes != current_coverage.get("ids_per_cell") or single + multiple != complexes:
            raise ValueError(f"{dataset}: prior-pool diagnostic counts are inconsistent")
        mismatch_ids = current.get("mismatched_ids")
        mismatches = current.get("mismatches")
        if (
            not isinstance(mismatch_ids, list)
            or any(not isinstance(value, str) or not value for value in mismatch_ids)
            or len(set(mismatch_ids)) != multiple
            or not isinstance(mismatches, list)
            or len(mismatches) != multiple
            or [item.get("id") if isinstance(item, dict) else None for item in mismatches]
            != mismatch_ids
        ):
            raise ValueError(f"{dataset}: prior-pool mismatch ID inventory is inconsistent")

        for mismatch in mismatches:
            complex_id = str(mismatch["id"])
            hash_set = mismatch.get("prior_pool_sha256_set")
            observations = mismatch.get("eta_observations")
            if (
                not isinstance(hash_set, list)
                or hash_set != sorted(set(hash_set))
                or len(hash_set) < 2
                or not isinstance(observations, list)
                or len(observations) != len(spec.eta_values)
            ):
                raise ValueError(f"{dataset}/{complex_id}: invalid prior-pool hash set")
            for digest in hash_set:
                replay._require_sha256(digest, label=f"{dataset}/{complex_id}.prior_pool_sha256")
            observed_hashes: set[str] = set()
            for expected_eta, expected_tag, observation in zip(
                spec.eta_values, spec.eta_tags, observations, strict=True
            ):
                if (
                    not isinstance(observation, dict)
                    or observation.get("eta") != expected_eta
                    or observation.get("eta_tag") != expected_tag
                    or observation.get("run_name") != spec.expected_run_name(dataset, expected_eta)
                ):
                    raise ValueError(
                        f"{dataset}/{complex_id}: invalid prior-pool eta observation"
                    )
                observed_hashes.add(
                    replay._require_sha256(
                        observation.get("prior_pool_sha256"),
                        label=f"{dataset}/{complex_id}/{expected_tag}.prior_pool_sha256",
                    )
                )
            if sorted(observed_hashes) != hash_set:
                raise ValueError(f"{dataset}/{complex_id}: observed prior hash set mismatch")

            runtime = mismatch.get("runtime_context")
            if not isinstance(runtime, dict):
                raise ValueError(f"{dataset}/{complex_id}: runtime_context must be an object")
            partitions = sorted(
                {
                    str(item["slurm_partition"])
                    for item in observations
                    if item.get("slurm_partition") is not None
                }
            )
            gpu_names = sorted(
                {str(item["gpu"]) for item in observations if item.get("gpu") is not None}
            )
            gpu_memory = sorted(
                {
                    int(item["gpu_total_memory_bytes"])
                    for item in observations
                    if item.get("gpu_total_memory_bytes") is not None
                }
            )
            expected_runtime = {
                "slurm_partitions": partitions,
                "gpu_names": gpu_names,
                "gpu_total_memory_bytes": gpu_memory,
                "mixed_slurm_partitions": len(partitions) > 1,
                "mixed_gpu_names": len(gpu_names) > 1,
            }
            if runtime != expected_runtime:
                raise ValueError(f"{dataset}/{complex_id}: runtime_context is inconsistent")

        dataset_complexes += complexes
        flattened_ids.extend({"dataset": dataset, "id": value} for value in mismatch_ids)

    complexes = require_count(diagnostic, "complexes", label="prior_pool_sha256_diagnostics")
    single = require_count(
        diagnostic, "complexes_with_single_hash", label="prior_pool_sha256_diagnostics"
    )
    multiple = require_count(
        diagnostic, "complexes_with_multiple_hashes", label="prior_pool_sha256_diagnostics"
    )
    if (
        complexes != dataset_complexes
        or complexes != coverage.get("unique_complexes")
        or single + multiple != complexes
        or multiple != len(flattened_ids)
        or diagnostic.get("mismatched_ids") != flattened_ids
    ):
        raise ValueError("high-eta audit global prior-pool diagnostic counts are inconsistent")
    return diagnostic


def _validate_companion_csv_inventory(
    sampling_dir: Path,
    summary_paths: dict[tuple[str, str, int], Path],
) -> None:
    required = {
        path.with_name(path.name.removesuffix(".summary.json") + ".csv")
        for path in summary_paths.values()
    }
    observed = set(sampling_dir.glob("*.csv")) if sampling_dir.is_dir() else set()
    if observed != required:
        raise ValueError(
            f"{sampling_dir}: companion CSV inventory mismatch; "
            f"missing={[path.name for path in sorted(required - observed)[:5]]}, "
            f"extra={[path.name for path in sorted(observed - required)[:5]]}"
        )


def build_standalone_audit(
    sampling_dir: Path,
    *,
    smoke: bool = False,
    cohort_audit: Path | None = None,
    spec: StandaloneSweepSpec = LEGACY_V1,
) -> dict[str, Any]:
    """Validate one complete standalone smoke/full run deterministically."""
    if asdict(ConfidenceFilterConfig()) != _EXPECTED_FILTER_CONFIG:
        raise RuntimeError("ConfidenceFilterConfig defaults differ from the frozen contract")
    if cohort_audit is not None:
        _validate_cohort_audit(Path(cohort_audit))

    sampling_dir = Path(sampling_dir).resolve()
    summary_paths = replay._expected_summary_paths(
        sampling_dir,
        smoke=smoke,
        eta_values=spec.eta_values,
        eta_tags=spec.eta_tags,
        expected_run_name_fn=spec.expected_run_name,
    )
    _validate_companion_csv_inventory(sampling_dir, summary_paths)
    shard_count = 1 if smoke else 8
    prior_hash_mismatch_is_diagnostic = spec == STERIC_HIGH_ETA_V1
    frozen_seed_by_dataset = (
        _frozen_sampling_seed_by_dataset() if prior_hash_mismatch_is_diagnostic else {}
    )
    current_implementation = guidance_implementation_identity()
    implementation_sha256 = replay._require_sha256(
        current_implementation.get("sha256"), label="current guidance implementation"
    )

    details: dict[str, Any] = {"datasets": {}}
    coverage_per_dataset: dict[str, dict[str, int]] = {}
    global_records: list[dict[str, Any]] = []
    cuda_runtimes: list[dict[str, Any]] = []
    prior_hash_diagnostics_by_dataset: dict[str, dict[str, Any]] = {}
    digest_cache: dict[Path, str] = {}
    total_rows = 0

    for dataset in DATASETS:
        dataset_cells: dict[str, Any] = {}
        reference_ids: list[str] | None = None
        paired_priors: dict[str, tuple[int, str]] = {}
        prior_hash_observations: dict[str, list[dict[str, Any]]] = {}
        paired_inputs: dict[str, dict[str, str]] = {}
        dataset_rows = 0
        for eta, tag in zip(spec.eta_values, spec.eta_tags, strict=True):
            run_name = spec.expected_run_name(dataset, eta)
            shard_ids: list[list[str]] = []
            row_by_id: dict[str, dict[str, Any]] = {}
            file_by_id: dict[str, dict[str, Any]] = {}
            artifacts: list[dict[str, Any]] = []
            for shard_index in range(shard_count):
                summary_path = summary_paths[(dataset, tag, shard_index)]
                summary = replay._load_json_object(summary_path)
                _validate_summary(
                    summary,
                    path=summary_path,
                    dataset=dataset,
                    eta=eta,
                    shard_index=shard_index,
                    smoke=smoke,
                    current_implementation=current_implementation,
                    spec=spec,
                )
                cuda_runtimes.append(summary["runtime"])
                csv_path = replay._companion_csv(summary_path, summary, sampling_dir)
                fields, rows = replay._read_raw_csv(csv_path)
                if len(rows) != summary["num_success"]:
                    raise ValueError(f"{csv_path}: row count differs from summary")

                traces: list[list[dict[str, Any]]] = []
                ids: list[str] = []
                for row in rows:
                    row_record, file_record, input_identity, trace = _validate_row(
                        row,
                        fields=fields,
                        sampling_dir=sampling_dir,
                        run_name=run_name,
                        dataset=dataset,
                        eta=eta,
                        shard_index=shard_index,
                        digest_cache=digest_cache,
                        spec=spec,
                    )
                    complex_id = row_record["id"]
                    if prior_hash_mismatch_is_diagnostic:
                        expected_seed = frozen_seed_by_dataset[dataset].get(complex_id)
                        if expected_seed is None or row_record["sampling_seed"] != expected_seed:
                            raise ValueError(
                                f"{dataset}/{complex_id}: sampling_seed must match frozen "
                                f"sorted-ID offset {expected_seed}"
                            )
                    ids.append(complex_id)
                    if complex_id in row_by_id:
                        raise ValueError(f"{run_name}: duplicate complex ID across shards")
                    row_by_id[complex_id] = row_record
                    file_by_id[complex_id] = file_record
                    traces.append(trace)

                    prior = (row_record["sampling_seed"], row_record["prior_pool_sha256"])
                    if prior_hash_mismatch_is_diagnostic:
                        runtime = summary["runtime"]
                        prior_hash_observations.setdefault(complex_id, []).append(
                            {
                                "eta": eta,
                                "eta_tag": tag,
                                "run_name": run_name,
                                "shard_index": shard_index,
                                "prior_pool_sha256": row_record["prior_pool_sha256"],
                                "slurm_partition": runtime.get("slurm_partition"),
                                "gpu": runtime.get("gpu"),
                                "gpu_total_memory_bytes": runtime.get(
                                    "gpu_total_memory_bytes"
                                ),
                                "slurm_job_id": runtime.get("slurm_job_id"),
                            }
                        )
                    if tag == spec.eta_tags[0]:
                        paired_priors[complex_id] = prior
                        paired_inputs[complex_id] = input_identity
                    else:
                        baseline_prior = paired_priors.get(complex_id)
                        if baseline_prior is None or baseline_prior[0] != prior[0]:
                            if prior_hash_mismatch_is_diagnostic:
                                raise ValueError(
                                    f"{dataset}/{complex_id}: sampling seed differs across eta"
                                )
                            raise ValueError(
                                f"{dataset}/{complex_id}: sampling seed/prior hash differs across eta"
                            )
                        if not prior_hash_mismatch_is_diagnostic and baseline_prior[1] != prior[1]:
                            raise ValueError(
                                f"{dataset}/{complex_id}: sampling seed/prior hash differs across eta"
                            )
                        if paired_inputs.get(complex_id) != input_identity:
                            raise ValueError(
                                f"{dataset}/{complex_id}: protein/reference binding differs across eta"
                            )

                if eta > 0.0:
                    replay._validate_trace_runtime_consistency(summary, traces)
                shard_ids.append(ids)
                artifacts.append(
                    {
                        "shard_index": shard_index,
                        "rows": len(ids),
                        "ordered_ids_sha256": _ordered_sha256(
                            ids, domain=b"EFFDOCK_STANDALONE_SHARD_IDS_V1\0"
                        ),
                        "summary": str(summary_path),
                        "summary_sha256": replay._file_sha256(summary_path),
                        "csv": str(csv_path),
                        "csv_sha256": replay._file_sha256(csv_path),
                    }
                )

            ordered_ids = _interleave_shards(shard_ids)
            expected_count = 1 if smoke else EXPECTED_DATASET_COUNTS[dataset]
            if len(ordered_ids) != expected_count or len(set(ordered_ids)) != expected_count:
                raise ValueError(f"{run_name}: exact cohort coverage mismatch")
            if smoke and ordered_ids != [SMOKE_IDS[dataset]]:
                raise ValueError(f"{run_name}: smoke ID must be {SMOKE_IDS[dataset]}")
            if (
                not smoke
                and sorted_id_sha256(ordered_ids)
                != EXPECTED_BENCHMARK_IDENTITIES[dataset]["ids_sha256"]
            ):
                raise ValueError(f"{run_name}: IDs differ from the frozen benchmark manifest")
            if reference_ids is None:
                reference_ids = ordered_ids
            elif ordered_ids != reference_ids:
                raise ValueError(f"{dataset}: eta cells do not have identical ID order")

            ordered_rows = [row_by_id[complex_id] for complex_id in ordered_ids]
            ordered_files = [file_by_id[complex_id] for complex_id in ordered_ids]
            row_ledger_sha256 = _records_sha256(
                ordered_rows, domain=b"EFFDOCK_STANDALONE_ROWS_V1\0"
            )
            file_ledger_sha256 = _records_sha256(
                ordered_files, domain=b"EFFDOCK_STANDALONE_FILE_BINDINGS_V1\0"
            )
            artifact_ledger_sha256 = _records_sha256(
                artifacts, domain=b"EFFDOCK_STANDALONE_ARTIFACTS_V1\0"
            )
            cell = {
                "run_name": run_name,
                "eta": eta,
                "eta_tag": spec.eta_tag(eta),
                "shards": shard_count,
                "rows": len(ordered_ids),
                "ordered_ids_sha256": _ordered_sha256(
                    ordered_ids, domain=b"EFFDOCK_STANDALONE_CELL_IDS_V1\0"
                ),
                "row_ledger_sha256": row_ledger_sha256,
                "file_binding_ledger_sha256": file_ledger_sha256,
                "artifact_ledger_sha256": artifact_ledger_sha256,
                "records": ordered_rows,
                "file_bindings": ordered_files,
                "artifacts": artifacts,
            }
            dataset_cells[tag] = cell
            dataset_rows += len(ordered_ids)
            total_rows += len(ordered_ids)
            global_records.append(
                {
                    key: cell[key]
                    for key in (
                        "run_name",
                        "eta",
                        "rows",
                        "ordered_ids_sha256",
                        "row_ledger_sha256",
                        "file_binding_ledger_sha256",
                        "artifact_ledger_sha256",
                    )
                }
            )

        ids_per_cell = 1 if smoke else EXPECTED_DATASET_COUNTS[dataset]
        if set(paired_priors) != set(reference_ids or []):
            raise ValueError(f"{dataset}: eta=0 prior pairing inventory is incomplete")
        if prior_hash_mismatch_is_diagnostic:
            prior_hash_diagnostics_by_dataset[dataset] = _build_prior_pool_sha256_diagnostics(
                prior_hash_observations,
                ordered_ids=reference_ids or [],
                spec=spec,
            )
        coverage_per_dataset[dataset] = {
            "cells": len(spec.eta_values),
            "shards": len(spec.eta_values) * shard_count,
            "rows": dataset_rows,
            "ids_per_cell": ids_per_cell,
        }
        details["datasets"][dataset] = {
            "ids_per_cell": ids_per_cell,
            "ordered_ids_sha256": _ordered_sha256(
                reference_ids or [], domain=b"EFFDOCK_STANDALONE_DATASET_IDS_V1\0"
            ),
            **(
                {"within_run_sampling_seed_pairing_verified": True}
                if prior_hash_mismatch_is_diagnostic
                else {"within_run_prior_pairing_verified": True}
            ),
            "cells": dataset_cells,
        }

    expected_rows = (
        len(DATASETS) * len(spec.eta_values)
        if smoke
        else sum(EXPECTED_DATASET_COUNTS.values()) * len(spec.eta_values)
    )
    if total_rows != expected_rows:
        raise ValueError(f"standalone row total must be {expected_rows}, got {total_rows}")

    prior_hash_diagnostics: dict[str, Any] | None = None
    if prior_hash_mismatch_is_diagnostic:
        mismatched_ids = [
            {"dataset": dataset, "id": complex_id}
            for dataset in DATASETS
            for complex_id in prior_hash_diagnostics_by_dataset[dataset]["mismatched_ids"]
        ]
        complexes = sum(
            int(diagnostic["complexes"])
            for diagnostic in prior_hash_diagnostics_by_dataset.values()
        )
        prior_hash_diagnostics = {
            "policy": "record_only_across_eta",
            "per_row_sha256_format_verified": True,
            "cross_eta_sha256_equality_required": False,
            "sampling_seed_equality_required": True,
            "sampling_seed_mapping": "base_seed_42_plus_one_based_sorted_dataset_id_index",
            "declared_prior_pool_size": 100,
            "declared_prior_pool_hash_contract": "EFFDOCK_SHARED_PRIOR_V1",
            "complexes": complexes,
            "complexes_with_single_hash": complexes - len(mismatched_ids),
            "complexes_with_multiple_hashes": len(mismatched_ids),
            "mismatched_ids": mismatched_ids,
            "datasets": prior_hash_diagnostics_by_dataset,
        }

    frozen_hashes = {
        "docking_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "config_sha256": EXPECTED_CONFIG_SHA256,
        "confidence_checkpoint_sha256": EXPECTED_CONFIDENCE_CHECKPOINT_SHA256,
        "eligibility_cohort_audit_sha256": EXPECTED_ELIGIBILITY_MANIFEST_SHA256,
        "benchmark_input_manifest_sha256": replay.EXPECTED_BENCHMARK_INPUT_MANIFEST_SHA256,
        "pocket_centers_sha256": dict(EXPECTED_POCKET_CENTERS_SHA256),
        "guidance_parameter_sha256": spec.guidance_parameter_sha256,
        "receptor_policy_sha256": spec.receptor_policy_sha256,
        "guidance_implementation_sha256": implementation_sha256,
    }
    if spec != LEGACY_V1:
        frozen_hashes.update(
            {
                "physical_parameter_sha256": spec.physical_parameter_sha256,
                "physical_parameter_version": spec.physical_parameter_version,
                "physical_formula_version": spec.physical_formula_version,
                "interaction_parameter_sha256": spec.interaction_parameter_sha256,
            }
        )
    return {
        "schema_version": spec.audit_schema_version,
        "protocol_id": spec.protocol_id,
        "audit_contract": spec.audit_contract,
        "mode": "fresh_one_pass_characterization",
        "run_scope": "smoke" if smoke else "full",
        "status": "passed",
        "parent_compared": False,
        "deterministic_replay_claim": False,
        "sampling_dir": str(sampling_dir),
        "summary_contracts_verified": True,
        "candidate_ensemble_hashes_present": True,
        "selector_recomputed": True,
        "selector_profile": SELECTOR_PROFILE,
        "frozen_hashes": frozen_hashes,
        **(
            {"prior_pool_sha256_diagnostics": prior_hash_diagnostics}
            if prior_hash_diagnostics is not None
            else {}
        ),
        "coverage": {
            "datasets": len(DATASETS),
            "unique_complexes": len(DATASETS) if smoke else sum(EXPECTED_DATASET_COUNTS.values()),
            "cells": len(DATASETS) * len(spec.eta_values),
            "shards": len(DATASETS) * len(spec.eta_values) * shard_count,
            "rows": total_rows,
            "per_dataset": coverage_per_dataset,
        },
        "cuda_runtime": {
            "slurm_partitions": sorted(
                {str(runtime["slurm_partition"]) for runtime in cuda_runtimes}
            ),
            "gpu_names": sorted({str(runtime["gpu"]) for runtime in cuda_runtimes}),
            "gpu_total_memory_bytes": sorted(
                {int(runtime["gpu_total_memory_bytes"]) for runtime in cuda_runtimes}
            ),
            "minimum_required_total_memory_bytes": MIN_GPU_TOTAL_MEMORY_BYTES,
            "max_memory_allocated_bytes": max(
                int(runtime["cuda_max_memory_allocated_bytes"]) for runtime in cuda_runtimes
            ),
            "max_memory_reserved_bytes": max(
                int(runtime["cuda_max_memory_reserved_bytes"]) for runtime in cuda_runtimes
            ),
            "limit_bytes": GPU_MEMORY_LIMIT_BYTES,
        },
        "candidate_ensemble_verification": {
            "status": "digest_present_and_producer_bound",
            "contract": CANDIDATE_ENSEMBLE_HASH_CONTRACT,
            "digest_present_for_every_row": True,
            "producer_bound_by_hashed_csv_artifacts": True,
            "independently_recomputed_from_all_candidate_coordinates": False,
            "reason": (
                "persisted_decimal_SDF_cannot_reconstruct_original_float32_digest"
                if prior_hash_mismatch_is_diagnostic
                else "all 100 candidate coordinate sets are not persisted"
            ),
            **(
                {
                    "all_poses_sdf_persisted_for_every_row": True,
                    "all_poses_sdf_current_file_hashes_exact": True,
                    "all_poses_sdf_record_counts_exact": True,
                    "persisted_coordinate_precision": "SDF_V2000_4_decimal_angstrom",
                }
                if prior_hash_mismatch_is_diagnostic
                else {}
            ),
        },
        "checks": {
            "summary_inventory_complete": True,
            "companion_csv_inventory_complete": True,
            "protocol_and_frozen_hashes_exact": True,
            "scientific_config_exact": True,
            "complete_success": True,
            "id_order_and_coverage_exact": True,
            **(
                {
                    "within_run_sampling_seed_paired_across_eta": True,
                    "sampling_seed_matches_frozen_sorted_id_offset_contract": True,
                    "prior_pool_sha256_cross_eta_differences_recorded": True,
                    "prior_pool_size_100_exact_in_every_csv_row": True,
                    "all_poses_sdf_current_hash_and_100_record_count_exact": True,
                }
                if prior_hash_mismatch_is_diagnostic
                else {"within_run_sampling_seed_and_prior_hash_paired_across_eta": True}
            ),
            "protein_reference_current_file_hashes_exact": True,
            "selected_pose_current_file_hashes_exact": True,
            "score_ledgers_100_by_7_finite_and_bounded": True,
            "primary_confidence_argmin_recomputed": True,
            "cluster_free_confidence_filter_recomputed": True,
            "artifact_and_file_binding_ledgers_complete": True,
            "cuda_runtime_and_memory_gate_passed": True,
        },
        "details": {
            "frozen_hashes": frozen_hashes,
            "confidence_filter_config": _EXPECTED_FILTER_CONFIG,
            "guidance_implementation": current_implementation,
            **details,
        },
        "global_integrity_ledger_sha256": _records_sha256(
            global_records, domain=b"EFFDOCK_STANDALONE_GLOBAL_V1\0"
        ),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sampling-dir", type=Path, required=True)
    parser.add_argument("--cohort-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--profile", choices=tuple(PROFILES), default=LEGACY_V1.key)
    args = parser.parse_args(argv)
    audit = build_standalone_audit(
        args.sampling_dir,
        smoke=args.smoke,
        cohort_audit=args.cohort_audit,
        spec=get_standalone_sweep_spec(args.profile),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(audit, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
