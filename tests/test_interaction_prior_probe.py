from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest
import torch

from effdock.workflows.interaction_prior_probe import (
    ARM_CONTRACTS,
    DEFAULT_SEEDS,
    LEGACY_ALL_TERMS,
    LEGACY_DEFAULT_TERMS,
    PROTOCOL_ID,
    PROTOCOL_NUMERICAL_CONTRACT,
    PROTOCOL_SAMPLE_ID,
    SCHEMA_VERSION,
    _crystal_term_gates,
    _file_sha256,
    _paired_comparisons,
    _require_same_numeric_contract,
    _validate_protocol_arguments,
    aggregate_batches,
)


def _metric(rmsd: float, *, pi: float = 0.0, halogen: float = 0.0) -> dict:
    return {
        "raw_rmsd_angstrom": rmsd,
        "cut_bond_max_abs_error_angstrom": 0.05,
        "minimum_distance_over_uff_x": 0.8,
        "chiral_improper_inversion_count": 0,
        "energies": {
            "interaction_pi_stacking": pi,
            "interaction_halogen_bond": halogen,
        },
    }


def _summary(prior: str, arm: str, final_rmsd: float) -> dict:
    role = (
        "single_new_term_pi_comparison"
        if arm == "guard_pi"
        else "paired_default_interaction_baseline"
    )
    runs = {}
    for index, seed in enumerate(DEFAULT_SEEDS):
        runs[f"{prior}__{arm}__seed-{seed}"] = {
            "seed": seed,
            "status": "max_steps",
            "shell_envelope_valid": True,
            "metrics": [
                _metric(
                    1.0,
                    pi=-0.005 * index,
                    halogen=-0.002 * index,
                ),
                _metric(final_rmsd),
            ],
        }
    return {
        "initialization": {"mode": prior},
        "objective": {
            "arm": arm,
            "admission_role": role,
        },
        "crystal_reference": {
            "energies": {
                "interaction_pi_stacking": -0.05,
                "interaction_halogen_bond": -0.02,
            },
            "contacts": {
                "pi_stacking_weight_sum": 1.0,
                "halogen_bond_weight_sum": 1.0,
            },
        },
        "runs": runs,
    }


def test_frozen_protocol_arguments_reject_changed_seed_or_solver_value() -> None:
    values = {
        "sample_id": PROTOCOL_SAMPLE_ID,
        "arm": "guard_pi",
        **PROTOCOL_NUMERICAL_CONTRACT,
    }
    args = argparse.Namespace(**values)
    _validate_protocol_arguments(args, DEFAULT_SEEDS)

    with pytest.raises(ValueError, match="ordered seeds"):
        _validate_protocol_arguments(args, DEFAULT_SEEDS[:-1])

    changed = argparse.Namespace(**{**values, "steps": 499})
    with pytest.raises(ValueError, match=r"--steps=500"):
        _validate_protocol_arguments(changed, DEFAULT_SEEDS)

    excluded_arm = argparse.Namespace(**{**values, "arm": "guard_metal"})
    with pytest.raises(ValueError, match="relaxation arm"):
        _validate_protocol_arguments(excluded_arm, DEFAULT_SEEDS)


def test_completed_v2_term_sets_are_literal_and_not_runtime_defaults() -> None:
    assert LEGACY_DEFAULT_TERMS == (
        "hydrophobic",
        "hydrogen_bond",
        "screened_formal_charge",
    )
    assert LEGACY_ALL_TERMS == (
        "hydrophobic",
        "hydrogen_bond",
        "screened_formal_charge",
        "pi_stacking",
        "cation_pi",
        "halogen_bond",
        "metal_coordination",
    )
    assert ARM_CONTRACTS["guard_default"].interaction_terms is LEGACY_DEFAULT_TERMS
    assert ARM_CONTRACTS["guard_all"].interaction_terms is LEGACY_ALL_TERMS
    assert ARM_CONTRACTS["interaction_all_raw"].interaction_terms is LEGACY_ALL_TERMS


def test_numeric_contract_allows_only_roundoff_scale_differences() -> None:
    _require_same_numeric_contract(
        {"aligned_rmsd": 1.21e-15, "energy": -3.0},
        {"aligned_rmsd": 1.08e-15, "energy": -3.0},
        "crystal_reference",
    )
    with pytest.raises(ValueError, match="numeric contract mismatch"):
        _require_same_numeric_contract(
            {"energy": -3.0},
            {"energy": -2.99},
            "crystal_reference",
        )


def test_paired_comparison_applies_preregistered_local_admission_gate() -> None:
    baseline = _summary("local", "guard_default", 0.9)
    term = _summary("local", "guard_pi", 0.6)

    comparison = _paired_comparisons([baseline, term])["local__guard_pi_vs_guard_default"]

    assert comparison["median_delta_final_rmsd_angstrom"] == pytest.approx(-0.3)
    assert comparison["improved_seed_count"] == 8
    assert comparison["additional_protocol_failure_count"] == 0
    assert comparison["local_admission_signal"] is True


def test_paired_comparison_counts_new_failures_by_seed() -> None:
    baseline = _summary("local", "guard_default", 0.9)
    term = _summary("local", "guard_pi", 0.6)
    baseline_runs = list(baseline["runs"].values())
    term_runs = list(term["runs"].values())
    baseline_runs[0]["status"] = "line_search_failed"
    term_runs[1]["status"] = "line_search_failed"

    comparison = _paired_comparisons([baseline, term])["local__guard_pi_vs_guard_default"]

    assert comparison["baseline_protocol_failure_count"] == 1
    assert comparison["term_protocol_failure_count"] == 1
    assert comparison["additional_protocol_failure_count"] == 1
    assert comparison["additional_protocol_failure_seeds"] == [DEFAULT_SEEDS[1]]
    assert comparison["local_admission_signal"] is False


def test_crystal_term_gate_counts_paired_local_initials() -> None:
    term = _summary("local", "guard_pi", 0.6)

    gate = _crystal_term_gates([term])["guard_pi"]

    assert gate["crystal_below_local_initial_count"] == 8
    assert gate["contact_nonzero"] is True
    assert gate["gate_pass"] is True


def test_aggregate_rejects_tampered_trajectory_before_loading(tmp_path: Path) -> None:
    trajectory = tmp_path / "trajectory.pt"
    torch.save(
        {
            "schema_version": SCHEMA_VERSION,
            "protocol_id": PROTOCOL_ID,
        },
        trajectory,
    )
    summary_paths = []
    for index in range(2):
        summary_path = tmp_path / f"summary-{index}.json"
        summary_path.write_text(
            json.dumps(
                {
                    "schema_version": SCHEMA_VERSION,
                    "protocol_id": PROTOCOL_ID,
                    "artifacts": {
                        "trajectory_pt": {
                            "path": str(trajectory),
                            "sha256": ("0" * 64 if index == 0 else _file_sha256(trajectory)),
                        }
                    },
                }
            )
        )
        summary_paths.append(summary_path)

    with pytest.raises(ValueError, match="trajectory artifact hash mismatch"):
        aggregate_batches(
            argparse.Namespace(
                inputs=summary_paths,
                output_dir=tmp_path / "aggregate",
            )
        )
