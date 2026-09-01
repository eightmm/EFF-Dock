from __future__ import annotations

import json
from pathlib import Path

import pytest

from effdock.workflows import guidance_eta_sweep_confidence_identity as replay
from effdock.workflows import guidance_steric_high_eta_stress_audit as stress_audit
from effdock.workflows.guidance_budget_full_report import EXPECTED_DATASET_COUNTS
from effdock.workflows.guidance_budget_report import (
    DATASETS,
    EXPECTED_CHECKPOINT_SHA256,
    EXPECTED_CONFIG_SHA256,
)
from effdock.workflows.guidance_eta_sweep_confidence_standalone_binding import (
    CONFIDENCE_CHECKPOINT_SHA256,
)
from effdock.workflows.guidance_eta_sweep_confidence_standalone_report import (
    validate_standalone_audit,
)
from effdock.workflows.guidance_eta_sweep_standalone_spec import STERIC_HIGH_ETA_V1
from effdock.workflows.guidance_steric_high_eta_preflight import build_preflight_identity


def test_stress_audit_passes_high_eta_spec_to_row_validator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sampling_dir = tmp_path / "stress"
    sampling_dir.mkdir()
    run_name = STERIC_HIGH_ETA_V1.expected_run_name(
        stress_audit.DATASET,
        stress_audit.ETA,
    )
    summary_path = sampling_dir / f"{run_name}.summary.json"
    csv_path = sampling_dir / f"{run_name}.csv"
    summary_path.write_text("{}")
    csv_path.write_text("id\n8f4j_pho\n")
    summary = {
        "runtime": {
            "gpu": "test-gpu",
            "cuda_max_memory_allocated_bytes": 1,
            "cuda_max_memory_reserved_bytes": 2,
        },
        "guidance_runtime_stats": {
            "direct_nonfinite_poses": 0,
            "direct_zero_raw_direction_poses": 0,
            "direct_zero_reference_velocity_poses": 0,
            "direct_max_translation_velocity": 1.0,
            "direct_max_angular_velocity": 1.0,
            "direct_max_estimated_atom_displacement": 0.1,
        },
    }

    monkeypatch.setattr(stress_audit.standalone, "_validate_cohort_audit", lambda _path: None)
    monkeypatch.setattr(stress_audit.replay, "_load_json_object", lambda _path: summary)
    monkeypatch.setattr(stress_audit.standalone, "_validate_summary", lambda *_a, **_k: None)
    monkeypatch.setattr(
        stress_audit.replay,
        "_companion_csv",
        lambda *_a, **_k: csv_path,
    )
    monkeypatch.setattr(
        stress_audit.replay,
        "_read_raw_csv",
        lambda _path: (["id"], [{"id": stress_audit.STRESS_ID}]),
    )
    monkeypatch.setattr(
        stress_audit,
        "guidance_implementation_identity",
        lambda: {"sha256": "a" * 64},
    )
    monkeypatch.setattr(
        stress_audit.replay,
        "_validate_trace_runtime_consistency",
        lambda *_a, **_k: None,
    )
    monkeypatch.setattr(stress_audit.replay, "_file_sha256", lambda _path: "b" * 64)

    def validate_row(*_args, spec, **_kwargs):
        assert spec is STERIC_HIGH_ETA_V1
        return ({"id": stress_audit.STRESS_ID}, {"pose": "bound"}, {"input": "bound"}, [])

    monkeypatch.setattr(stress_audit.standalone, "_validate_row", validate_row)

    audit = stress_audit.build_stress_audit(
        sampling_dir,
        cohort_audit=tmp_path / "cohort.json",
    )

    assert audit["status"] == "passed"
    assert audit["row_integrity"] == {"id": stress_audit.STRESS_ID}


def test_steric_high_eta_profile_is_frozen_and_has_same_run_control() -> None:
    spec = STERIC_HIGH_ETA_V1
    assert spec.audit_contract == "EFFDOCK_STERIC_HIGH_ETA_CONFIDENCE_INTEGRITY_V2"
    assert spec.audit_schema_version == "effdock.guidance_steric_high_eta_confidence_integrity.v2"
    assert spec.binding_contract == "EFFDOCK_STERIC_HIGH_ETA_CONFIDENCE_OFFICIAL_BINDING_V2"
    assert spec.eta_values == (0.0, 0.5, 1.0, 1.5, 2.0)
    assert spec.eta_tags == (
        "eta0000",
        "eta0500",
        "eta1000",
        "eta1500",
        "eta2000",
    )
    assert spec.expected_run_name("posebusters", 2.0) == (
        "effdock-guidance-steric-high-eta-v1-posebusters-n100-s10-eta2000"
    )
    assert spec.guidance_parameter_sha256 == (
        "6621d17c41aeb6c9685075209155850018c5eb9882489ae209c7c30b8070e89f"
    )
    assert spec.physical_parameter_version == "2.1.0"


def test_high_eta_shell_profile_uses_matching_v2_audit_contract() -> None:
    profile = Path("scripts/slurm/guidance_eta_sweep_confidence_standalone_profile.sh")
    text = profile.read_text()
    assert "audit_contract=EFFDOCK_STERIC_HIGH_ETA_CONFIDENCE_INTEGRITY_V2" in text
    assert "audit_schema_version=effdock.guidance_steric_high_eta_confidence_integrity.v2" in text


def test_superseded_high_eta_preflight_fails_closed_on_current_guidance() -> None:
    # The immutable high-eta V1 study used EFF-FF 2.1. The current diagnostic
    # parameter set is 2.2, so replaying V1 against the live runtime must fail
    # rather than silently relabel the historical protocol.
    with pytest.raises(RuntimeError, match="frozen high-eta identity mismatch"):
        build_preflight_identity()


def test_replay_inventory_helper_accepts_explicit_high_eta_grid(tmp_path: Path) -> None:
    spec = STERIC_HIGH_ETA_V1
    for dataset in DATASETS:
        for eta in spec.eta_values:
            run_name = spec.expected_run_name(dataset, eta)
            (tmp_path / f"{run_name}.summary.json").write_text("{}")
    paths = replay._expected_summary_paths(
        tmp_path,
        smoke=True,
        eta_values=spec.eta_values,
        eta_tags=spec.eta_tags,
        expected_run_name_fn=spec.expected_run_name,
    )
    assert len(paths) == 10
    assert set(paths) == {
        (dataset, tag, 0)
        for dataset in DATASETS
        for tag in spec.eta_tags
    }


def test_high_eta_audit_contract_requires_profile_hashes(tmp_path: Path) -> None:
    spec = STERIC_HIGH_ETA_V1
    rows = sum(EXPECTED_DATASET_COUNTS.values()) * len(spec.eta_values)
    audit = {
        "schema_version": spec.audit_schema_version,
        "protocol_id": spec.protocol_id,
        "audit_contract": spec.audit_contract,
        "mode": "fresh_one_pass_characterization",
        "run_scope": "full",
        "status": "passed",
        "parent_compared": False,
        "deterministic_replay_claim": False,
        "selector_profile": "confidence_cluster_free",
        "coverage": {
            "datasets": len(DATASETS),
            "unique_complexes": sum(EXPECTED_DATASET_COUNTS.values()),
            "cells": len(DATASETS) * len(spec.eta_values),
            "shards": len(DATASETS) * len(spec.eta_values) * 8,
            "rows": rows,
            "per_dataset": {
                dataset: {
                    "cells": len(spec.eta_values),
                    "shards": len(spec.eta_values) * 8,
                    "rows": EXPECTED_DATASET_COUNTS[dataset] * len(spec.eta_values),
                    "ids_per_cell": EXPECTED_DATASET_COUNTS[dataset],
                }
                for dataset in DATASETS
            },
        },
        "frozen_hashes": {
            "docking_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
            "config_sha256": EXPECTED_CONFIG_SHA256,
            "confidence_checkpoint_sha256": CONFIDENCE_CHECKPOINT_SHA256,
            "guidance_parameter_sha256": spec.guidance_parameter_sha256,
            "physical_parameter_sha256": spec.physical_parameter_sha256,
            "physical_parameter_version": spec.physical_parameter_version,
            "physical_formula_version": spec.physical_formula_version,
            "interaction_parameter_sha256": spec.interaction_parameter_sha256,
            "receptor_policy_sha256": spec.receptor_policy_sha256,
        },
        "candidate_ensemble_verification": {
            "reason": "persisted_decimal_SDF_cannot_reconstruct_original_float32_digest",
            "all_poses_sdf_persisted_for_every_row": True,
            "all_poses_sdf_current_file_hashes_exact": True,
            "all_poses_sdf_record_counts_exact": True,
            "persisted_coordinate_precision": "SDF_V2000_4_decimal_angstrom",
            "independently_recomputed_from_all_candidate_coordinates": False,
        },
        "checks": {
            "within_run_sampling_seed_paired_across_eta": True,
            "sampling_seed_matches_frozen_sorted_id_offset_contract": True,
            "prior_pool_sha256_cross_eta_differences_recorded": True,
            "prior_pool_size_100_exact_in_every_csv_row": True,
            "all_poses_sdf_current_hash_and_100_record_count_exact": True,
        },
        "prior_pool_sha256_diagnostics": {
            "policy": "record_only_across_eta",
            "per_row_sha256_format_verified": True,
            "cross_eta_sha256_equality_required": False,
            "sampling_seed_equality_required": True,
            "sampling_seed_mapping": (
                "base_seed_42_plus_one_based_sorted_dataset_id_index"
            ),
            "declared_prior_pool_size": 100,
            "declared_prior_pool_hash_contract": "EFFDOCK_SHARED_PRIOR_V1",
            "complexes": sum(EXPECTED_DATASET_COUNTS.values()),
            "complexes_with_single_hash": sum(EXPECTED_DATASET_COUNTS.values()),
            "complexes_with_multiple_hashes": 0,
            "mismatched_ids": [],
            "datasets": {
                dataset: {
                    "complexes": EXPECTED_DATASET_COUNTS[dataset],
                    "complexes_with_single_hash": EXPECTED_DATASET_COUNTS[dataset],
                    "complexes_with_multiple_hashes": 0,
                    "mismatched_ids": [],
                    "mismatches": [],
                }
                for dataset in DATASETS
            },
        },
        "global_integrity_ledger_sha256": "a" * 64,
    }
    path = tmp_path / "audit.json"
    path.write_text(json.dumps(audit))
    assert validate_standalone_audit(path, expected_shards=8, spec=spec) == audit

    audit["prior_pool_sha256_diagnostics"]["complexes_with_multiple_hashes"] = 1
    path.write_text(json.dumps(audit))
    with pytest.raises(ValueError, match="global prior-pool diagnostic counts"):
        validate_standalone_audit(path, expected_shards=8, spec=spec)
