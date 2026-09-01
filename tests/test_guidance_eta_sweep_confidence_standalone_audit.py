from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import pytest

from effdock.guidance.provenance import guidance_implementation_identity
from effdock.workflows import guidance_eta_sweep_confidence_standalone_audit as standalone
from effdock.workflows import guidance_eta_sweep_report as eta_report
from effdock.workflows.guidance_eta_sweep_standalone_spec import (
    LEGACY_V1,
    STERIC_HIGH_ETA_V1,
    StandaloneSweepSpec,
)


def _trace_step(interval: dict[str, float], eta: float) -> dict[str, float | int]:
    step: dict[str, float | int] = {
        **interval,
        "eta": eta,
        "pose_count": 100,
        "finite_count": 100,
        "applied_count": 100,
        "model_atom_speed_rms_sum": 200.0,
        "applied_atom_speed_rms_sum": 20.0,
        "total_atom_speed_rms_sum": 201.0,
        "atom_speed_rms_valid_count": 100,
        "model_rms_path_proxy_sum": interval["dt"] * 200.0,
        "applied_rms_path_proxy_sum": interval["dt"] * 20.0,
        "total_rms_path_proxy_sum": interval["dt"] * 201.0,
        "translation_cap_trigger_count": 0,
        "angular_cap_trigger_count": 0,
        "displacement_cap_trigger_count": 0,
        "any_cap_trigger_count": 0,
        "multiple_cap_trigger_count": 0,
    }
    for name, value in (
        ("applied_to_model_rms_ratio", 0.1),
        ("model_guide_cosine", 0.25),
        ("guide_parallel_to_model_ratio", 0.025),
        ("cap_scale", 1.0),
    ):
        step[f"{name}_sum"] = value * 100
        step[f"{name}_valid_count"] = 100
        for suffix in ("p05", "p50", "p95", "p99"):
            step[f"{name}_{suffix}"] = value
    return step


def _trace_and_runtime(eta: float) -> tuple[str, dict[str, int | float]]:
    trace = [_trace_step(interval, eta) for interval in eta_report._expected_trace_intervals()]
    runtime: dict[str, int | float] = {
        "direct_steps_attempted": 8,
        "direct_pose_evaluations": 800,
        "direct_batched_energy_evaluations": 8,
        "direct_pose_energy_evaluations": 800,
        "direct_nonfinite_poses": 0,
        "direct_pose_applied": 800,
        "direct_max_translation_velocity": 1.0,
        "direct_max_angular_velocity": 1.0,
        "direct_max_estimated_atom_displacement": 0.1,
    }
    mapping = {
        "direct_model_atom_speed_rms_sum": "model_atom_speed_rms_sum",
        "direct_applied_atom_speed_rms_sum": "applied_atom_speed_rms_sum",
        "direct_total_atom_speed_rms_sum": "total_atom_speed_rms_sum",
        "direct_atom_speed_rms_valid_count": "atom_speed_rms_valid_count",
        "direct_applied_to_model_rms_ratio_sum": "applied_to_model_rms_ratio_sum",
        "direct_applied_to_model_rms_ratio_valid_count": ("applied_to_model_rms_ratio_valid_count"),
        "direct_model_guide_cosine_sum": "model_guide_cosine_sum",
        "direct_model_guide_cosine_valid_count": "model_guide_cosine_valid_count",
        "direct_guide_parallel_to_model_ratio_sum": "guide_parallel_to_model_ratio_sum",
        "direct_guide_parallel_to_model_ratio_valid_count": (
            "guide_parallel_to_model_ratio_valid_count"
        ),
        "direct_model_rms_path_proxy_sum": "model_rms_path_proxy_sum",
        "direct_applied_rms_path_proxy_sum": "applied_rms_path_proxy_sum",
        "direct_total_rms_path_proxy_sum": "total_rms_path_proxy_sum",
        "direct_cap_scale_sum": "cap_scale_sum",
        "direct_cap_scale_valid_count": "cap_scale_valid_count",
        "direct_translation_cap_trigger_count": "translation_cap_trigger_count",
        "direct_angular_cap_trigger_count": "angular_cap_trigger_count",
        "direct_displacement_cap_trigger_count": "displacement_cap_trigger_count",
        "direct_any_cap_trigger_count": "any_cap_trigger_count",
        "direct_multiple_cap_trigger_count": "multiple_cap_trigger_count",
    }
    for runtime_key, trace_key in mapping.items():
        value = sum(float(step[trace_key]) for step in trace)
        runtime[runtime_key] = int(value) if trace_key.endswith("count") else value
    return json.dumps(trace, sort_keys=True), runtime


def _scores() -> list[dict[str, float]]:
    return [
        {
            "confidence_rmsd": float(index + 1),
            "confidence_success_logit": -float(index),
            "confidence_success": 1.0 / float(index + 1),
            "confidence_atom_rmsd": float(index + 1),
            "confidence_atom_q90": float(index + 1),
            "confidence_atom_ok": 1.0 / float(index + 1),
            "pl_clash_1p6": 0.0,
        }
        for index in range(100)
    ]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_csv(path: Path, fields: list[str], row: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(row)


def _write_pose(
    root: Path,
    *,
    run_name: str,
    dataset: str,
    complex_id: str,
    selector: str,
) -> str:
    path = root / "poses" / run_name / dataset / selector / f"{complex_id}.sdf"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{run_name}:{complex_id}:{selector}\n")
    return _sha256(path)


def _write_all_poses(
    root: Path,
    *,
    run_name: str,
    dataset: str,
    complex_id: str,
) -> tuple[Path, str]:
    path = root / "poses" / run_name / dataset / "all_poses" / f"{complex_id}.sdf"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n$$$$\n" * 100)
    return path, _sha256(path)


def _summary(
    *,
    dataset: str,
    eta: float,
    csv_path: Path,
    trace_runtime: dict[str, int | float] | None,
    spec: StandaloneSweepSpec = LEGACY_V1,
) -> dict[str, object]:
    value: dict[str, object] = {
        key: None
        for key in set(eta_report._REQUIRED_SUMMARY_KEYS)
        | standalone.replay._EXTRA_REQUIRED_SUMMARY_KEYS
    }
    value.update(
        {
            "protocol_id": spec.protocol_id,
            "selector_profile": standalone.SELECTOR_PROFILE,
            "run_name": spec.expected_run_name(dataset, eta),
            "dataset": dataset,
            "num_samples": 100,
            "num_steps": 10,
            "model_pose_step_budget": 1000,
            "num_shards": 1,
            "shard_index": 0,
            "seed": 42,
            "unified_guidance_scale": eta,
            "unified_guidance_mode": "normalized_drift",
            "prior_pool_size": 100,
            "prior_pool_hash_contract": "EFFDOCK_SHARED_PRIOR_V1",
            "checkpoint_sha256": standalone.EXPECTED_CHECKPOINT_SHA256,
            "checkpoint_step": 100_000,
            "config_sha256": standalone.EXPECTED_CONFIG_SHA256,
            "pocket_centers_sha256": standalone.EXPECTED_POCKET_CENTERS_SHA256[dataset],
            "eligibility_manifest_sha256": standalone.EXPECTED_ELIGIBILITY_MANIFEST_SHA256,
            "num_discovered_total": standalone.EXPECTED_DATASET_COUNTS[dataset],
            "expected_discovered_count": standalone.EXPECTED_DATASET_COUNTS[dataset],
            "num_assigned": 1,
            "num_success": 1,
            "num_failed": 0,
            "failures": [],
            "require_complete_success": True,
            "require_full_ligand_atom_mapping": True,
            "sigma": 0.5,
            "sigma_list": [],
            "sigma_counts": [],
            "pose_objective": "linear_fm",
            "score_rot_sigma_max": 3.141592653589793,
            "score_alpha_min": 0.0,
            "time_schedule": "late",
            "schedule_power": 3.0,
            "pocket_cutoff": 10.0,
            "center_jitter_sigma": 0.0,
            "vina_guidance_scale": 0.0,
            "vina_guidance_start_t": 0.5,
            "vina_guidance_ramp_power": 1.0,
            "vina_guidance_max_force": 10.0,
            "vina_guidance_max_velocity": 5.0,
            "vina_guidance_max_angular_velocity": 5.0,
            "vina_guidance_protein_shell": 18.0,
            "vina_guidance_w_strain": 1.0,
            "unified_guidance_start_t": 0.5,
            "unified_guidance_ramp_power": 1.0,
            "unified_guidance_max_force": 20.0,
            "unified_guidance_max_velocity": 5.0,
            "unified_guidance_max_angular_velocity": 5.0,
            "unified_guidance_max_atom_displacement": 0.25,
            "unified_guidance_max_backtracks": 8,
            "unified_guidance_protein_shell": 18.0,
            "unified_guidance_receptor_policy": "geometry_only",
            "refine": "none",
            "csv": str(csv_path),
            "benchmark_input_identity": {
                "dataset": dataset,
                "count": standalone.EXPECTED_DATASET_COUNTS[dataset],
                "mode": "frozen_manifest",
                **standalone.EXPECTED_BENCHMARK_IDENTITIES[dataset],
                "sources": {
                    "frozen_manifest": {
                        "sha256": standalone.replay.EXPECTED_BENCHMARK_INPUT_MANIFEST_SHA256
                    }
                },
            },
            "guidance_implementation": guidance_implementation_identity(),
            "runtime": {
                "device": "cuda",
                "slurm_partition": "6000ada",
                "gpu": "NVIDIA RTX 6000 Ada Generation",
                "gpu_total_memory_bytes": 48_000 * 1024**2,
                "torch": "test",
                "cuda": "test",
                "cuda_max_memory_allocated_bytes": 1024,
                "cuda_max_memory_reserved_bytes": 2048,
                "slurm_job_id": "1",
            },
            "stats": {
                "first": {"mean_rmsd": 1.0},
                "oracle": {"mean_rmsd": 0.5},
                "candidate_set": {"mean_fast_valid_candidates": 10.0},
                "confidence": {"mean_rmsd": 1.0},
                "confidence_filter": {"mean_rmsd": 1.0},
            },
            "confidence_checkpoint": "weights/confidence.pt",
            "confidence_step": 42_500,
            "confidence_checkpoint_sha256": standalone.EXPECTED_CONFIDENCE_CHECKPOINT_SHA256,
            "candidate_ensemble_hash_contract": standalone.CANDIDATE_ENSEMBLE_HASH_CONTRACT,
            "confidence_score_ledger_contract": standalone.CONFIDENCE_SCORE_LEDGER_CONTRACT,
        }
    )
    if eta > 0.0:
        value["guidance_runtime_stats"] = trace_runtime
        value["guidance_parameter_set"] = {"sha256": spec.guidance_parameter_sha256}
        if spec != LEGACY_V1:
            value["guidance_parameter_set"].update(
                {
                    "physical": {
                        "sha256": spec.physical_parameter_sha256,
                        "version": spec.physical_parameter_version,
                        "formula_version": spec.physical_formula_version,
                    },
                    "interaction": {"sha256": spec.interaction_parameter_sha256},
                }
            )
        value["guidance_receptor_policy_identities"] = {
            spec.receptor_policy_sha256: {"sha256": spec.receptor_policy_sha256}
        }
    return value


def _build_smoke_fixture(
    tmp_path: Path,
    *,
    spec: StandaloneSweepSpec = LEGACY_V1,
) -> Path:
    root = tmp_path / "sampling"
    fields = sorted(
        standalone._REQUIRED_ROW_FIELDS
        | (standalone._V2_REQUIRED_ROW_FIELDS if spec == STERIC_HIGH_ETA_V1 else set())
    )
    frozen_seeds = (
        standalone._frozen_sampling_seed_by_dataset() if spec == STERIC_HIGH_ETA_V1 else {}
    )
    scores = _scores()
    for dataset in standalone.DATASETS:
        complex_id = standalone.SMOKE_IDS[dataset]
        protein = tmp_path / f"{dataset}.pdb"
        reference = tmp_path / f"{dataset}.sdf"
        protein.write_text(f"protein:{dataset}\n")
        reference.write_text(f"reference:{dataset}\n")
        for eta in spec.eta_values:
            run_name = spec.expected_run_name(dataset, eta)
            trace, trace_runtime = _trace_and_runtime(eta) if eta > 0.0 else ("[]", None)
            pose_ledger = {
                selector: _write_pose(
                    root,
                    run_name=run_name,
                    dataset=dataset,
                    complex_id=complex_id,
                    selector=selector,
                )
                for selector in standalone.SAVED_SELECTORS
            }
            all_poses = (
                _write_all_poses(
                    root,
                    run_name=run_name,
                    dataset=dataset,
                    complex_id=complex_id,
                )
                if spec == STERIC_HIGH_ETA_V1
                else None
            )
            row = {
                "id": complex_id,
                "selector_profile": standalone.SELECTOR_PROFILE,
                "protein": str(protein),
                "ligand_ref": str(reference),
                "protein_sha256": _sha256(protein),
                "ligand_reference_sha256": _sha256(reference),
                "saved_pose_sha256_json": json.dumps(pose_ledger, sort_keys=True),
                "num_samples": "100",
                "prior_pool_size": "100",
                "sampling_seed": str(
                    frozen_seeds[dataset][complex_id] if spec == STERIC_HIGH_ETA_V1 else 43
                ),
                "prior_pool_sha256": "d" * 64,
                "guidance_direct_step_trace_json": trace,
                "candidate_ensemble_sha256": "e" * 64,
                "confidence_candidate_scores_json": json.dumps(scores, sort_keys=True),
                **(
                    {
                        "all_poses_sdf": str(all_poses[0]),
                        "all_poses_sdf_sha256": all_poses[1],
                        "all_poses_count": "100",
                    }
                    if all_poses is not None
                    else {}
                ),
            }
            for selector in standalone.CONFIDENCE_SELECTORS:
                row[f"{selector}_index"] = "0"
                row[f"{selector}_rmsd"] = "1.0"
                row[f"{selector}_pred_rmsd"] = "1.0"
                row[f"{selector}_pred_success"] = "1.0"
                for term in standalone.replay._FAST_TERMS:
                    row[f"{selector}_fast_{term}"] = "True"
            csv_path = root / f"{run_name}.csv"
            _write_csv(csv_path, fields, row)
            summary = _summary(
                dataset=dataset,
                eta=eta,
                csv_path=csv_path,
                trace_runtime=trace_runtime,
                spec=spec,
            )
            (root / f"{run_name}.summary.json").write_text(json.dumps(summary))
    return root


def _rewrite_csv(path: Path, mutate: Callable[[dict[str, str]], None]) -> None:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    assert len(rows) == 1
    mutate(rows[0])
    _write_csv(path, fields, rows[0])


def test_smoke_audit_is_parent_free_complete_and_deterministic(tmp_path: Path) -> None:
    root = _build_smoke_fixture(tmp_path)
    assert not list(root.glob("poses/**/vina/*.sdf"))
    assert not list(root.glob("poses/**/confidence_final/*.sdf"))
    first = standalone.build_standalone_audit(root, smoke=True)
    second = standalone.build_standalone_audit(root, smoke=True)

    assert first == second
    assert first["status"] == "passed"
    assert first["protocol_id"] == standalone.PROTOCOL_ID
    assert first["audit_contract"] == "EFFDOCK_CONFIDENCE_STANDALONE_INTEGRITY_V1"
    assert first["mode"] == "fresh_one_pass_characterization"
    assert first["run_scope"] == "smoke"
    assert first["parent_compared"] is False
    assert first["deterministic_replay_claim"] is False
    assert first["cuda_runtime"]["slurm_partitions"] == ["6000ada"]
    assert first["coverage"] == {
        "datasets": 2,
        "unique_complexes": 2,
        "cells": 16,
        "shards": 16,
        "rows": 16,
        "per_dataset": {
            "astex": {"cells": 8, "shards": 8, "rows": 8, "ids_per_cell": 1},
            "posebusters": {"cells": 8, "shards": 8, "rows": 8, "ids_per_cell": 1},
        },
    }
    assert first["candidate_ensemble_verification"]["status"] == (
        "digest_present_and_producer_bound"
    )
    assert (
        first["candidate_ensemble_verification"][
            "independently_recomputed_from_all_candidate_coordinates"
        ]
        is False
    )
    assert "prior_pool_sha256_diagnostics" not in first
    assert first["checks"]["within_run_sampling_seed_and_prior_hash_paired_across_eta"]
    assert "equivalence" not in json.dumps(first, sort_keys=True)
    assert len(first["global_integrity_ledger_sha256"]) == 64


@pytest.mark.parametrize(
    ("field", "value"),
    (("sampling_seed", "44"), ("prior_pool_sha256", "f" * 64)),
)
def test_legacy_audit_rejects_seed_or_prior_change_across_eta(
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    root = _build_smoke_fixture(tmp_path)
    run_name = eta_report.expected_run_name("astex", 0.025)
    _rewrite_csv(
        root / f"{run_name}.csv",
        lambda row: row.__setitem__(field, value),
    )
    with pytest.raises(ValueError, match="sampling seed/prior hash differs across eta"):
        standalone.build_standalone_audit(root, smoke=True)


def test_high_eta_audit_records_prior_hash_drift_with_runtime_context(tmp_path: Path) -> None:
    spec = STERIC_HIGH_ETA_V1
    root = _build_smoke_fixture(tmp_path, spec=spec)
    run_name = spec.expected_run_name("astex", 0.5)
    _rewrite_csv(
        root / f"{run_name}.csv",
        lambda row: row.__setitem__("prior_pool_sha256", "f" * 64),
    )
    summary_path = root / f"{run_name}.summary.json"
    summary: dict[str, Any] = json.loads(summary_path.read_text())
    runtime = summary["runtime"]
    assert isinstance(runtime, dict)
    runtime["slurm_partition"] = "heavy"
    runtime["gpu"] = "NVIDIA RTX PRO 6000 Blackwell Workstation Edition"
    summary_path.write_text(json.dumps(summary))

    audit = standalone.build_standalone_audit(root, smoke=True, spec=spec)

    assert audit["status"] == "passed"
    assert audit["audit_contract"] == "EFFDOCK_STERIC_HIGH_ETA_CONFIDENCE_INTEGRITY_V2"
    assert audit["schema_version"] == "effdock.guidance_steric_high_eta_confidence_integrity.v2"
    assert audit["checks"]["within_run_sampling_seed_paired_across_eta"]
    assert audit["checks"]["sampling_seed_matches_frozen_sorted_id_offset_contract"]
    assert audit["checks"]["all_poses_sdf_current_hash_and_100_record_count_exact"]
    assert audit["candidate_ensemble_verification"]["reason"] == (
        "persisted_decimal_SDF_cannot_reconstruct_original_float32_digest"
    )
    assert audit["checks"]["prior_pool_sha256_cross_eta_differences_recorded"]
    diagnostic = audit["prior_pool_sha256_diagnostics"]
    assert diagnostic["policy"] == "record_only_across_eta"
    assert diagnostic["declared_prior_pool_size"] == 100
    assert diagnostic["declared_prior_pool_hash_contract"] == "EFFDOCK_SHARED_PRIOR_V1"
    assert diagnostic["complexes"] == 2
    assert diagnostic["complexes_with_single_hash"] == 1
    assert diagnostic["complexes_with_multiple_hashes"] == 1
    assert diagnostic["mismatched_ids"] == [
        {"dataset": "astex", "id": standalone.SMOKE_IDS["astex"]}
    ]
    astex = diagnostic["datasets"]["astex"]
    assert astex["mismatched_ids"] == [standalone.SMOKE_IDS["astex"]]
    mismatch = astex["mismatches"][0]
    assert mismatch["prior_pool_sha256_set"] == ["d" * 64, "f" * 64]
    assert mismatch["runtime_context"] == {
        "slurm_partitions": ["6000ada", "heavy"],
        "gpu_names": [
            "NVIDIA RTX 6000 Ada Generation",
            "NVIDIA RTX PRO 6000 Blackwell Workstation Edition",
        ],
        "gpu_total_memory_bytes": [48_000 * 1024**2],
        "mixed_slurm_partitions": True,
        "mixed_gpu_names": True,
    }
    assert mismatch["eta_observations"][1]["eta_tag"] == "eta0500"
    assert mismatch["eta_observations"][1]["slurm_partition"] == "heavy"
    assert mismatch["eta_observations"][1]["prior_pool_sha256"] == "f" * 64


def test_high_eta_audit_still_rejects_sampling_seed_change(tmp_path: Path) -> None:
    spec = STERIC_HIGH_ETA_V1
    root = _build_smoke_fixture(tmp_path, spec=spec)
    run_name = spec.expected_run_name("astex", 0.5)
    _rewrite_csv(
        root / f"{run_name}.csv",
        lambda row: row.__setitem__("sampling_seed", "44"),
    )

    with pytest.raises(ValueError, match="sampling_seed must match frozen sorted-ID offset"):
        standalone.build_standalone_audit(root, smoke=True, spec=spec)


def test_high_eta_audit_rejects_same_wrong_seed_in_every_eta_arm(tmp_path: Path) -> None:
    spec = STERIC_HIGH_ETA_V1
    root = _build_smoke_fixture(tmp_path, spec=spec)
    for eta in spec.eta_values:
        run_name = spec.expected_run_name("astex", eta)
        _rewrite_csv(
            root / f"{run_name}.csv",
            lambda row: row.__setitem__("sampling_seed", "44"),
        )

    with pytest.raises(ValueError, match="sampling_seed must match frozen sorted-ID offset"):
        standalone.build_standalone_audit(root, smoke=True, spec=spec)


def test_audit_requires_canonical_prior_pool_size_in_csv(tmp_path: Path) -> None:
    root = _build_smoke_fixture(tmp_path)
    run_name = LEGACY_V1.expected_run_name("astex", 0.0)
    _rewrite_csv(
        root / f"{run_name}.csv",
        lambda row: row.__setitem__("prior_pool_size", "0100"),
    )

    with pytest.raises(ValueError, match="prior_pool_size must be canonical integer 100"):
        standalone.build_standalone_audit(root, smoke=True)


def test_high_eta_audit_binds_current_all_poses_hash_and_record_count(tmp_path: Path) -> None:
    spec = STERIC_HIGH_ETA_V1
    root = _build_smoke_fixture(tmp_path, spec=spec)
    run_name = spec.expected_run_name("astex", 0.0)
    csv_path = root / f"{run_name}.csv"
    with csv_path.open(newline="") as handle:
        row = next(csv.DictReader(handle))
    all_poses = Path(row["all_poses_sdf"])
    all_poses.write_text("\n$$$$\n" * 99)
    _rewrite_csv(
        csv_path,
        lambda current: current.__setitem__("all_poses_sdf_sha256", _sha256(all_poses)),
    )

    with pytest.raises(ValueError, match="all_poses_sdf must contain exactly 100 records"):
        standalone.build_standalone_audit(root, smoke=True, spec=spec)


@pytest.mark.parametrize(
    ("field", "value"),
    (("prior_pool_size", 99), ("prior_pool_hash_contract", "OTHER")),
)
def test_high_eta_audit_keeps_declared_prior_contract_strict(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    spec = STERIC_HIGH_ETA_V1
    root = _build_smoke_fixture(tmp_path, spec=spec)
    run_name = spec.expected_run_name("astex", 0.0)
    summary_path = root / f"{run_name}.summary.json"
    summary: dict[str, Any] = json.loads(summary_path.read_text())
    summary[field] = value
    summary_path.write_text(json.dumps(summary))

    with pytest.raises(ValueError, match=field):
        standalone.build_standalone_audit(root, smoke=True, spec=spec)


def test_audit_rejects_missing_companion_csv(tmp_path: Path) -> None:
    root = _build_smoke_fixture(tmp_path)
    run_name = eta_report.expected_run_name("astex", 0.0)
    (root / f"{run_name}.csv").unlink()

    with pytest.raises(ValueError, match="companion CSV inventory mismatch.*missing="):
        standalone.build_standalone_audit(root, smoke=True)


def test_audit_rejects_extra_companion_csv(tmp_path: Path) -> None:
    root = _build_smoke_fixture(tmp_path)
    (root / "unexpected.csv").write_text("id\nextra\n")

    with pytest.raises(ValueError, match="companion CSV inventory mismatch.*extra="):
        standalone.build_standalone_audit(root, smoke=True)


@pytest.mark.parametrize("selector", ["vina", "confidence_final"])
def test_audit_rejects_saved_selector_outside_profile(
    tmp_path: Path,
    selector: str,
) -> None:
    root = _build_smoke_fixture(tmp_path)
    run_name = eta_report.expected_run_name("astex", 0.0)

    def mutate(row: dict[str, str]) -> None:
        ledger = json.loads(row["saved_pose_sha256_json"])
        ledger[selector] = "f" * 64
        row["saved_pose_sha256_json"] = json.dumps(ledger, sort_keys=True)

    _rewrite_csv(root / f"{run_name}.csv", mutate)
    with pytest.raises(ValueError, match="saved-pose selectors must be exactly"):
        standalone.build_standalone_audit(root, smoke=True)


@pytest.mark.parametrize("selector", ["vina", "confidence_final"])
def test_audit_rejects_csv_selector_outside_profile(tmp_path: Path, selector: str) -> None:
    root = _build_smoke_fixture(tmp_path)
    run_name = eta_report.expected_run_name("astex", 0.0)
    path = root / f"{run_name}.csv"
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    fields.append(f"{selector}_index")
    rows[0][f"{selector}_index"] = "0"
    _write_csv(path, fields, rows[0])

    with pytest.raises(ValueError, match="CSV contains selectors outside profile"):
        standalone.build_standalone_audit(root, smoke=True)


def test_audit_rejects_wrong_selector_profile(tmp_path: Path) -> None:
    root = _build_smoke_fixture(tmp_path)
    run_name = eta_report.expected_run_name("astex", 0.0)
    path = root / f"{run_name}.summary.json"
    summary: dict[str, Any] = json.loads(path.read_text())
    summary["selector_profile"] = "legacy_all"
    path.write_text(json.dumps(summary))

    with pytest.raises(ValueError, match="selector_profile must be confidence_cluster_free"):
        standalone.build_standalone_audit(root, smoke=True)


def test_audit_requires_exact_100_by_7_score_ledger(tmp_path: Path) -> None:
    root = _build_smoke_fixture(tmp_path)
    run_name = eta_report.expected_run_name("astex", 0.0)

    def mutate(row: dict[str, str]) -> None:
        scores = json.loads(row["confidence_candidate_scores_json"])
        scores[0]["unexpected"] = 0.0
        row["confidence_candidate_scores_json"] = json.dumps(scores)

    _rewrite_csv(root / f"{run_name}.csv", mutate)
    with pytest.raises(ValueError, match="score fields must be exactly"):
        standalone.build_standalone_audit(root, smoke=True)


def test_audit_recomputes_primary_and_cluster_free_filter(tmp_path: Path) -> None:
    root = _build_smoke_fixture(tmp_path)
    run_name = eta_report.expected_run_name("astex", 0.0)
    _rewrite_csv(
        root / f"{run_name}.csv",
        lambda row: row.__setitem__("confidence_filter_index", "1"),
    )
    with pytest.raises(ValueError, match="does not match frozen filter"):
        standalone.build_standalone_audit(root, smoke=True)


def test_audit_rejects_changed_current_pose_file(tmp_path: Path) -> None:
    root = _build_smoke_fixture(tmp_path)
    dataset = "astex"
    complex_id = standalone.SMOKE_IDS[dataset]
    run_name = eta_report.expected_run_name(dataset, 0.0)
    pose = root / "poses" / run_name / dataset / "confidence" / f"{complex_id}.sdf"
    pose.write_text("changed\n")
    with pytest.raises(ValueError, match="saved pose SHA-256 differs from CSV ledger"):
        standalone.build_standalone_audit(root, smoke=True)


def test_audit_rejects_changed_current_protein_file(tmp_path: Path) -> None:
    root = _build_smoke_fixture(tmp_path)
    (tmp_path / "astex.pdb").write_text("changed\n")
    with pytest.raises(ValueError, match="current file SHA-256 differs from CSV binding"):
        standalone.build_standalone_audit(root, smoke=True)


def test_audit_rejects_old_replay_protocol(tmp_path: Path) -> None:
    root = _build_smoke_fixture(tmp_path)
    run_name = eta_report.expected_run_name("astex", 0.0)
    path = root / f"{run_name}.summary.json"
    summary: dict[str, Any] = json.loads(path.read_text())
    summary["protocol_id"] = standalone.replay.PROTOCOL_ID
    path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="protocol_id must be"):
        standalone.build_standalone_audit(root, smoke=True)
