#!/usr/bin/env python3
"""Strict one-pass confidence RMSD/PoseBusters report for the eta sweep."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from effdock.workflows.evaluate import sorted_id_sha256
from effdock.workflows.guidance_budget_full_report import EXPECTED_DATASET_COUNTS
from effdock.workflows.guidance_budget_posebusters_report import (
    MODULE_CHECKS,
    POSEBUSTERS_CONFIG,
    POSEBUSTERS_VERSION,
    VALIDITY_CHECKS,
)
from effdock.workflows.guidance_budget_report import (
    DATASETS,
    DEFAULT_BOOTSTRAP_RESAMPLES,
    DEFAULT_BOOTSTRAP_SEED,
    DEFAULT_EXPECTED_SHARDS,
    EXPECTED_CHECKPOINT_SHA256,
    EXPECTED_CONFIG_SHA256,
)
from effdock.workflows.guidance_coverage_audit import ID_HASH_CONTRACT
from effdock.workflows.guidance_eta_sweep_confidence_report import (
    aggregate_official_cell,
    join_sampling_outcomes,
    load_frozen_parent_cohort_audit,
    paired_outcomes,
    summarize_outcomes,
)
from effdock.workflows.guidance_eta_sweep_confidence_standalone_audit import (
    AUDIT_CONTRACT as _LEGACY_AUDIT_CONTRACT,
)
from effdock.workflows.guidance_eta_sweep_confidence_standalone_audit import (
    SELECTOR_PROFILE,
    build_standalone_audit,
    validate_v2_prior_pool_sha256_diagnostics,
)
from effdock.workflows.guidance_eta_sweep_confidence_standalone_binding import (
    CONFIDENCE_CHECKPOINT_SHA256,
    SELECTORS,
    build_binding,
)
from effdock.workflows.guidance_eta_sweep_confidence_standalone_binding import (
    PROTOCOL_ID as _LEGACY_PROTOCOL_ID,
)
from effdock.workflows.guidance_eta_sweep_report import (
    CONDITION,
    ETA_TAGS,
    ETA_VALUES,
    NUM_SAMPLES,
    NUM_STEPS,
    expected_run_name,
)
from effdock.workflows.guidance_eta_sweep_standalone_spec import (
    LEGACY_V1,
    PROFILES,
    STERIC_HIGH_ETA_V1,
    StandaloneSweepSpec,
    get_standalone_sweep_spec,
)
from effdock.workflows.posebusters_report import file_sha256

PRIMARY_SELECTOR = "confidence"
DIAGNOSTIC_SELECTOR = "confidence_filter"
EXPECTED_INTEGRITY_ROWS = sum(EXPECTED_DATASET_COUNTS.values()) * len(ETA_VALUES)
# Backward-compatible public aliases used by the completed legacy workflow.
AUDIT_CONTRACT = _LEGACY_AUDIT_CONTRACT
PROTOCOL_ID = _LEGACY_PROTOCOL_ID


def _profile_grid(
    spec: StandaloneSweepSpec,
) -> tuple[tuple[float, ...], tuple[str, ...]]:
    # Preserve the legacy module-level seams used by historical tests/tools.
    if spec == LEGACY_V1:
        return tuple(ETA_VALUES), tuple(ETA_TAGS)
    return spec.eta_values, spec.eta_tags


def _profile_run_name(spec: StandaloneSweepSpec, dataset: str, eta: float) -> str:
    return expected_run_name(dataset, eta) if spec == LEGACY_V1 else spec.expected_run_name(dataset, eta)


def load_frozen_cohort_audit(path: Path) -> dict[str, dict[str, Any]]:
    """Load the content-addressed eligibility cohort without a replay dependency."""
    return load_frozen_parent_cohort_audit(path)


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{label} must be a SHA-256 hex digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{label} must be a SHA-256 hex digest") from exc
    return value


def validate_standalone_audit(
    path: Path,
    *,
    expected_shards: int,
    spec: StandaloneSweepSpec = LEGACY_V1,
) -> dict[str, Any]:
    """Require the completed parent-free full-run integrity audit."""
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: standalone audit must be a JSON object")
    expected_top_level = {
        **(
            {"schema_version": spec.audit_schema_version}
            if spec == STERIC_HIGH_ETA_V1
            else {}
        ),
        "protocol_id": spec.protocol_id,
        "audit_contract": spec.audit_contract,
        "mode": "fresh_one_pass_characterization",
        "run_scope": "full",
        "status": "passed",
        "parent_compared": False,
        "deterministic_replay_claim": False,
        "selector_profile": SELECTOR_PROFILE,
    }
    for key, expected in expected_top_level.items():
        if raw.get(key) != expected:
            raise ValueError(
                f"{path}: standalone audit {key} must be {expected!r}, got {raw.get(key)!r}"
            )

    forbidden = (
        "parent_dir",
        "parent_sampling_protocol_id",
        "parent_sentinels_verified",
        "global_equivalence_ledger_sha256",
    )
    present = [key for key in forbidden if key in raw]
    if present:
        raise ValueError(f"{path}: standalone audit contains replay-only fields {present}")

    coverage = raw.get("coverage")
    if not isinstance(coverage, dict):
        raise ValueError(f"{path}: standalone audit coverage must be an object")
    eta_values, _ = _profile_grid(spec)
    expected_coverage = {
        "datasets": len(DATASETS),
        "cells": len(DATASETS) * len(eta_values),
        "shards": len(DATASETS) * len(eta_values) * expected_shards,
        "rows": sum(EXPECTED_DATASET_COUNTS.values()) * len(eta_values),
    }
    for key, expected in expected_coverage.items():
        if coverage.get(key) != expected:
            raise ValueError(
                f"{path}: standalone coverage {key} must be {expected}, got {coverage.get(key)!r}"
            )
    expected_per_dataset = {
        dataset: {
            "cells": len(eta_values),
            "shards": len(eta_values) * expected_shards,
            "rows": EXPECTED_DATASET_COUNTS[dataset] * len(eta_values),
            "ids_per_cell": EXPECTED_DATASET_COUNTS[dataset],
        }
        for dataset in DATASETS
    }
    if coverage.get("per_dataset") != expected_per_dataset:
        raise ValueError(f"{path}: standalone per-dataset coverage mismatch")

    frozen = raw.get("frozen_hashes")
    expected_hashes = {
        "docking_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "config_sha256": EXPECTED_CONFIG_SHA256,
        "confidence_checkpoint_sha256": CONFIDENCE_CHECKPOINT_SHA256,
    }
    if not isinstance(frozen, dict):
        raise ValueError(f"{path}: standalone frozen_hashes must be an object")
    for key, expected in expected_hashes.items():
        if frozen.get(key) != expected:
            raise ValueError(f"{path}: standalone frozen hash mismatch for {key}")
    if spec != LEGACY_V1:
        expected_profile_hashes = {
            "guidance_parameter_sha256": spec.guidance_parameter_sha256,
            "physical_parameter_sha256": spec.physical_parameter_sha256,
            "physical_parameter_version": spec.physical_parameter_version,
            "physical_formula_version": spec.physical_formula_version,
            "interaction_parameter_sha256": spec.interaction_parameter_sha256,
            "receptor_policy_sha256": spec.receptor_policy_sha256,
        }
        for key, expected in expected_profile_hashes.items():
            if frozen.get(key) != expected:
                raise ValueError(f"{path}: standalone profile hash mismatch for {key}")

    if not isinstance(raw.get("candidate_ensemble_verification"), dict):
        raise ValueError(f"{path}: candidate_ensemble_verification must be an object")
    if not isinstance(raw.get("checks"), dict):
        raise ValueError(f"{path}: checks must be an object")
    if spec == STERIC_HIGH_ETA_V1:
        candidate = raw["candidate_ensemble_verification"]
        expected_candidate = {
            "reason": "persisted_decimal_SDF_cannot_reconstruct_original_float32_digest",
            "all_poses_sdf_persisted_for_every_row": True,
            "all_poses_sdf_current_file_hashes_exact": True,
            "all_poses_sdf_record_counts_exact": True,
            "persisted_coordinate_precision": "SDF_V2000_4_decimal_angstrom",
            "independently_recomputed_from_all_candidate_coordinates": False,
        }
        if any(candidate.get(key) != value for key, value in expected_candidate.items()):
            raise ValueError(f"{path}: high-eta candidate ensemble provenance mismatch")
        validate_v2_prior_pool_sha256_diagnostics(raw, spec=spec)
    _require_sha256(
        raw.get("global_integrity_ledger_sha256"),
        label=f"{path}.global_integrity_ledger_sha256",
    )
    return raw


def revalidate_standalone_audit(
    audit: dict[str, Any],
    *,
    sampling_dir: Path,
    cohort_audit: Path,
    spec: StandaloneSweepSpec = LEGACY_V1,
) -> None:
    """Rebuild the audit before reading official outcomes."""
    sampling_value = audit.get("sampling_dir")
    if not isinstance(sampling_value, str) or not sampling_value:
        raise ValueError("standalone audit sampling_dir must be a non-empty path")
    if sampling_dir.resolve() != Path(sampling_value).resolve():
        raise ValueError("sampling_dir must resolve to the root bound by the standalone audit")
    rebuild_kwargs: dict[str, Any] = {
        "smoke": False,
        "cohort_audit": cohort_audit,
    }
    if spec != LEGACY_V1:
        rebuild_kwargs["spec"] = spec
    rebuilt = build_standalone_audit(sampling_dir, **rebuild_kwargs)
    if rebuilt != audit:
        raise ValueError(
            "saved standalone audit differs from a fresh full audit; "
            "sampling inputs may have changed after audit"
        )


def _require_sampling_inventory(
    sampling_dir: Path,
    *,
    expected_shards: int,
    spec: StandaloneSweepSpec = LEGACY_V1,
) -> None:
    eta_values, _ = _profile_grid(spec)
    expected = {
        sampling_dir
        / (
            f"{_profile_run_name(spec, dataset, eta)}."
            f"shard-{shard:03d}-of-{expected_shards:03d}.summary.json"
        )
        for dataset in DATASETS
        for eta in eta_values
        for shard in range(expected_shards)
    }
    actual = set(sampling_dir.glob("*.summary.json")) if sampling_dir.is_dir() else set()
    if actual != expected:
        missing = sorted(str(path) for path in expected - actual)
        extra = sorted(str(path) for path in actual - expected)
        raise ValueError(
            f"standalone sampling inventory mismatch; missing={missing[:5]}, extra={extra[:5]}"
        )


def _require_bindings(
    *,
    input_dir: Path,
    sampling_dir: Path,
    expected_shards: int,
    integrity_audit: Path | None = None,
    spec: StandaloneSweepSpec = LEGACY_V1,
) -> None:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"missing official PoseBusters root: {input_dir}")
    expected_selector_dirs = {input_dir / selector for selector in SELECTORS}
    actual_selector_dirs = {
        path for path in input_dir.iterdir() if path.is_dir() and any(path.glob("*/*.summary.json"))
    }
    if actual_selector_dirs != expected_selector_dirs:
        raise ValueError("official standalone selector directory inventory mismatch")
    eta_values, _ = _profile_grid(spec)
    expected_runs = {
        _profile_run_name(spec, dataset, eta)
        for dataset in DATASETS
        for eta in eta_values
    }
    for selector in SELECTORS:
        selector_dir = input_dir / selector
        actual_runs = {
            path.name
            for path in selector_dir.iterdir()
            if path.is_dir() and any(path.glob("*.summary.json"))
        }
        if actual_runs != expected_runs:
            raise ValueError(
                f"{selector}: official run-cell mismatch; "
                f"missing={sorted(expected_runs - actual_runs)}, "
                f"extra={sorted(actual_runs - expected_runs)}"
            )
        for dataset in DATASETS:
            for eta in eta_values:
                run_name = _profile_run_name(spec, dataset, eta)
                cell_dir = selector_dir / run_name
                expected_bindings = {
                    cell_dir / f"shard-{index:03d}-of-{expected_shards:03d}.binding.json"
                    for index in range(expected_shards)
                }
                if set(cell_dir.glob("*.binding.json")) != expected_bindings:
                    raise ValueError(f"{run_name}/{selector}: binding inventory mismatch")
                expected_csvs = {
                    cell_dir / f"shard-{index:03d}-of-{expected_shards:03d}.csv"
                    for index in range(expected_shards)
                }
                if set(cell_dir.glob("*.csv")) != expected_csvs:
                    raise ValueError(f"{run_name}/{selector}: official CSV inventory mismatch")
                for shard_index in range(expected_shards):
                    tag = f"shard-{shard_index:03d}-of-{expected_shards:03d}"
                    binding_path = cell_dir / f"{tag}.binding.json"
                    observed = json.loads(binding_path.read_text())
                    expected = build_binding(
                        sampling_dir=sampling_dir,
                        official_dir=cell_dir,
                        run_name=run_name,
                        protocol_id=spec.protocol_id,
                        dataset=dataset,
                        eta=eta,
                        selector=selector,
                        shard_index=shard_index,
                        num_shards=expected_shards,
                        integrity_audit=(
                            integrity_audit if spec == STERIC_HIGH_ETA_V1 else None
                        ),
                        spec=spec,
                    )
                    if observed != expected:
                        raise ValueError(f"{binding_path}: official/sampling binding mismatch")


def build_report(
    input_dir: Path,
    sampling_dir: Path,
    cohort_audit: Path,
    integrity_audit: Path,
    *,
    expected_shards: int = DEFAULT_EXPECTED_SHARDS,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    spec: StandaloneSweepSpec = LEGACY_V1,
) -> dict[str, Any]:
    """Build the strict two-selector report after every integrity gate passes."""
    if expected_shards != DEFAULT_EXPECTED_SHARDS:
        raise ValueError(
            f"full standalone protocol requires exactly {DEFAULT_EXPECTED_SHARDS} shards"
        )
    if bootstrap_resamples < 1:
        raise ValueError("bootstrap_resamples must be >= 1")
    eta_values, eta_tags = _profile_grid(spec)
    integrity = validate_standalone_audit(
        integrity_audit,
        expected_shards=expected_shards,
        spec=spec,
    )
    revalidate_standalone_audit(
        integrity,
        sampling_dir=sampling_dir,
        cohort_audit=cohort_audit,
        spec=spec,
    )
    audits = load_frozen_cohort_audit(cohort_audit)
    _require_sampling_inventory(
        sampling_dir,
        expected_shards=expected_shards,
        spec=spec,
    )
    _require_bindings(
        input_dir=input_dir,
        sampling_dir=sampling_dir,
        expected_shards=expected_shards,
        integrity_audit=integrity_audit,
        spec=spec,
    )

    outcomes: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = {}
    prior_diagnostic = (
        integrity["prior_pool_sha256_diagnostics"]
        if spec == STERIC_HIGH_ETA_V1
        else None
    )
    prior_pairing_claim = (
        {
            "sampling_seed": "exact_base_seed_42_plus_one_based_sorted_dataset_id_index",
            "prior_pool_size": "exact_100",
            "prior_pool_construction_contract": "exact_EFFDOCK_SHARED_PRIOR_V1",
            "prior_pool_sha256_cross_eta": "diagnostic_only",
            "exact_prior_tensor_identity_claim": False,
            "reason": "prior tensors were not persisted at original float32 precision",
            "diagnostic_summary": {
                "complexes": prior_diagnostic["complexes"],
                "complexes_with_single_hash": prior_diagnostic[
                    "complexes_with_single_hash"
                ],
                "complexes_with_multiple_hashes": prior_diagnostic[
                    "complexes_with_multiple_hashes"
                ],
                "mismatched_ids": prior_diagnostic["mismatched_ids"],
            },
        }
        if isinstance(prior_diagnostic, dict)
        else None
    )
    report: dict[str, Any] = {
        "protocol_id": spec.protocol_id,
        "status": "complete_strict_full_cohort_one_pass_confidence_posebusters_eta_sweep",
        "evaluation_mode": "fresh_one_pass_characterization",
        "parent_identity_claim": False,
        "deterministic_replay_claim": False,
        **({"prior_pairing_claim": prior_pairing_claim} if prior_pairing_claim else {}),
        "selector_profile": SELECTOR_PROFILE,
        "winner_selected": False,
        "selection_policy": {
            "eta": "not_performed",
            "selector": "not_performed",
        },
        "claim_boundary": (
            (
                "complex-ID-paired, seed-matched descriptive external benchmark from one "
                "fresh sampling pass; exact prior tensor equality is not claimed; all eta "
                "arms and both frozen selectors are reported without selecting a winner or "
                "admitting guidance"
            )
            if spec == STERIC_HIGH_ETA_V1
            else (
                "paired descriptive external benchmark from one fresh sampling pass; "
                "all eta arms and both frozen selectors are reported without selecting "
                "a winner or admitting guidance"
            )
        ),
        "selectors": {
            PRIMARY_SELECTOR: {
                "role": "primary",
                "definition": "minimum predicted-RMSD confidence head",
                "cluster_dependent": False,
            },
            DIAGNOSTIC_SELECTOR: {
                "role": "diagnostic",
                "definition": "frozen confidence_filter_v1 cluster-free guard",
                "cluster_dependent": False,
                "production_admission": "not admitted by the frozen validation gate",
            },
        },
        "confidence_checkpoint_sha256": CONFIDENCE_CHECKPOINT_SHA256,
        "standalone_audit": {
            "path": str(integrity_audit),
            "sha256": file_sha256(integrity_audit),
            "audit_contract": spec.audit_contract,
            **(
                {
                    "schema_version": spec.audit_schema_version,
                    "prior_pool_sha256_diagnostics": prior_diagnostic,
                }
                if isinstance(prior_diagnostic, dict)
                else {}
            ),
            "status": integrity["status"],
            "mode": integrity["mode"],
            "run_scope": integrity["run_scope"],
            "parent_compared": integrity["parent_compared"],
            "deterministic_replay_claim": integrity["deterministic_replay_claim"],
            "selector_profile": integrity["selector_profile"],
            "candidate_ensemble_verification": integrity["candidate_ensemble_verification"],
            "global_integrity_ledger_sha256": integrity["global_integrity_ledger_sha256"],
            "coverage": integrity["coverage"],
        },
        "condition": {
            "name": CONDITION,
            "num_samples": NUM_SAMPLES,
            "num_steps": NUM_STEPS,
            "model_pose_step_budget": NUM_SAMPLES * NUM_STEPS,
        },
        "eta_grid": [
            {"eta": eta, "tag": tag}
            for eta, tag in zip(eta_values, eta_tags, strict=True)
        ],
        "official_inventory": {
            "selectors": len(SELECTORS),
            "run_cells_per_selector": len(DATASETS) * len(eta_values),
            "shards_per_cell": expected_shards,
            "total_shard_tasks": (
                len(SELECTORS) * len(DATASETS) * len(eta_values) * expected_shards
            ),
        },
        "posebusters": {
            "version": POSEBUSTERS_VERSION,
            "config": POSEBUSTERS_CONFIG,
            "pass_all_definition": "all 27 non-RMSD redock checks",
            "validity_checks": list(VALIDITY_CHECKS),
            "module_checks": {key: list(value) for key, value in MODULE_CHECKS.items()},
        },
        "bootstrap": {
            "method": "paired complex-ID bootstrap, percentile 95% CI",
            "seed": bootstrap_seed,
            "resamples": bootstrap_resamples,
        },
        "datasets": {},
    }

    rmsd_checks: set[str] = set()
    for dataset in DATASETS:
        audit = audits[dataset]
        ids = audit["ids"]
        if len(ids) != EXPECTED_DATASET_COUNTS[dataset]:
            raise ValueError(f"{dataset}: audit is not the frozen full cohort")
        cells: dict[str, Any] = {}
        for eta, tag in zip(eta_values, eta_tags, strict=True):
            run_name = _profile_run_name(spec, dataset, eta)
            selector_cells: dict[str, Any] = {}
            for selector in SELECTORS:
                official_rows, aggregate, rmsd_check = aggregate_official_cell(
                    input_dir / selector / run_name,
                    ids,
                    run_name=run_name,
                    dataset=dataset,
                    eta_tag=tag,
                    selector=selector,
                    expected_shards=expected_shards,
                )
                current = join_sampling_outcomes(
                    sampling_dir=sampling_dir,
                    run_name=run_name,
                    selector=selector,
                    ids=ids,
                    official_rows=official_rows,
                )
                outcomes[(dataset, tag, selector)] = current
                aggregate["eta"] = eta
                aggregate["metrics"] = summarize_outcomes(current)
                aggregate["eligible_ids_sha256"] = sorted_id_sha256(list(ids))
                aggregate["ids_hash_contract"] = ID_HASH_CONTRACT
                selector_cells[selector] = aggregate
                rmsd_checks.add(rmsd_check)
            cells[tag] = {
                "eta": eta,
                "eta_tag": tag,
                "selectors": selector_cells,
                "confidence_filter_minus_confidence": paired_outcomes(
                    outcomes[(dataset, tag, PRIMARY_SELECTOR)],
                    outcomes[(dataset, tag, DIAGNOSTIC_SELECTOR)],
                    ids,
                    baseline_label=PRIMARY_SELECTOR,
                    comparison_label=DIAGNOSTIC_SELECTOR,
                    seed=bootstrap_seed,
                    resamples=bootstrap_resamples,
                ),
            }

        eta_vs_eta0 = {
            selector: {
                tag: paired_outcomes(
                    outcomes[(dataset, eta_tags[0], selector)],
                    outcomes[(dataset, tag, selector)],
                    ids,
                    baseline_label=eta_tags[0],
                    comparison_label=tag,
                    seed=bootstrap_seed,
                    resamples=bootstrap_resamples,
                )
                for tag in eta_tags
            }
            for selector in SELECTORS
        }
        report["datasets"][dataset] = {
            "coverage": {
                "count": len(ids),
                "expected": EXPECTED_DATASET_COUNTS[dataset],
                "ids_sha256": audit["ids_sha256"],
                "ids_hash_contract": ID_HASH_CONTRACT,
                "audit_path": audit["source_path"],
                "audit_sha256": audit["source_sha256"],
            },
            "cells": cells,
            "eta_vs_eta0": eta_vs_eta0,
        }

    if len(rmsd_checks) != 1:
        raise ValueError(f"RMSD check name differs across cells: {sorted(rmsd_checks)}")
    report["posebusters"]["rmsd_check_excluded_from_validity"] = next(iter(rmsd_checks))
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--sampling-dir", type=Path, required=True)
    parser.add_argument("--cohort-audit", type=Path, required=True)
    parser.add_argument("--integrity-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-shards", type=int, default=DEFAULT_EXPECTED_SHARDS)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument("--bootstrap-resamples", type=int, default=DEFAULT_BOOTSTRAP_RESAMPLES)
    parser.add_argument("--profile", choices=tuple(PROFILES), default=LEGACY_V1.key)
    args = parser.parse_args(argv)
    result = build_report(
        args.input_dir,
        args.sampling_dir,
        args.cohort_audit,
        args.integrity_audit,
        expected_shards=args.expected_shards,
        bootstrap_seed=args.bootstrap_seed,
        bootstrap_resamples=args.bootstrap_resamples,
        spec=get_standalone_sweep_spec(args.profile),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
