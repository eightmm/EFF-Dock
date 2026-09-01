from __future__ import annotations

import csv
import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest
from rdkit import Chem
from rdkit.Geometry import Point3D

from effdock.workflows import guidance_eta_sweep_confidence_identity as identity
from effdock.workflows import guidance_eta_sweep_report as eta_report


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
        "direct_guide_parallel_to_model_ratio_sum": ("guide_parallel_to_model_ratio_sum"),
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


def _write_csv(path: Path, fields: list[str], row: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(row)


def _pose_digest(
    root: Path,
    run_name: str,
    dataset: str,
    complex_id: str,
    selector: str,
    payload: bytes,
) -> str:
    path = root / "poses" / run_name / dataset / selector / f"{complex_id}.sdf"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _write_test_sdf(
    root: Path,
    run_name: str,
    dataset: str,
    complex_id: str,
    selector: str,
    *,
    first_atom_offset: float,
) -> str:
    molecule = Chem.RWMol()
    molecule.AddAtom(Chem.Atom(6))
    molecule.AddAtom(Chem.Atom(8))
    molecule.AddBond(0, 1, Chem.BondType.SINGLE)
    conformer = Chem.Conformer(2)
    conformer.SetAtomPosition(0, Point3D(first_atom_offset, 0.0, 0.0))
    conformer.SetAtomPosition(1, Point3D(1.25, 0.0, 0.0))
    molecule.AddConformer(conformer)
    path = root / "poses" / run_name / dataset / selector / f"{complex_id}.sdf"
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = Chem.SDWriter(str(path))
    writer.write(molecule)
    writer.close()
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _replace_pose_hash(csv_path: Path, selector: str, digest: str) -> None:
    def mutate(row: dict[str, str]) -> None:
        ledger = json.loads(row["saved_pose_sha256_json"])
        ledger[selector] = digest
        row["saved_pose_sha256_json"] = json.dumps(ledger, sort_keys=True)

    _rewrite_csv(csv_path, mutate)


def _summary(
    *,
    dataset: str,
    eta: float,
    csv_path: Path,
    confidence: bool,
    trace_runtime: dict[str, int | float] | None,
) -> dict[str, object]:
    value: dict[str, object] = {
        key: None
        for key in set(eta_report._REQUIRED_SUMMARY_KEYS) | identity._EXTRA_REQUIRED_SUMMARY_KEYS
    }
    value.update(
        {
            "protocol_id": identity.PROTOCOL_ID if confidence else identity.PARENT_PROTOCOL_ID,
            "run_name": eta_report.expected_run_name(dataset, eta),
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
            "checkpoint_sha256": identity.EXPECTED_CHECKPOINT_SHA256,
            "checkpoint_step": 100_000,
            "config_sha256": identity.EXPECTED_CONFIG_SHA256,
            "pocket_centers_sha256": identity.EXPECTED_POCKET_CENTERS_SHA256[dataset],
            "eligibility_manifest_sha256": "a" * 64,
            "num_discovered_total": identity.EXPECTED_DATASET_COUNTS[dataset],
            "expected_discovered_count": identity.EXPECTED_DATASET_COUNTS[dataset],
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
                "count": identity.EXPECTED_DATASET_COUNTS[dataset],
                "mode": "frozen_manifest",
                **identity.EXPECTED_BENCHMARK_IDENTITIES[dataset],
                "sources": {
                    "frozen_manifest": {"sha256": identity.EXPECTED_BENCHMARK_INPUT_MANIFEST_SHA256}
                },
            },
            "guidance_implementation": {
                "schema_version": "test",
                "files": ["sampler.py", "evaluate.py"],
                "sha256": ("c" if confidence else "b") * 64,
            },
            "runtime": {
                "device": "cuda",
                "slurm_partition": "6000ada",
                "gpu": "NVIDIA RTX 6000 Ada Generation",
                "gpu_total_memory_bytes": 48_000 * 1024**2,
                "torch": "test",
                "cuda": "test",
                "cuda_max_memory_allocated_bytes": 1024,
                "cuda_max_memory_reserved_bytes": 2048,
                "slurm_job_id": "2" if confidence else "1",
            },
            "stats": {
                "first": {"mean_rmsd": 1.0},
                "vina": {"mean_rmsd": 2.0},
                "oracle": {"mean_rmsd": 0.5},
                "candidate_set": {"mean_fast_valid_candidates": 10.0},
            },
            "confidence_checkpoint": "weights/confidence.pt" if confidence else None,
            "confidence_step": 42_500 if confidence else None,
            "confidence_checkpoint_sha256": (
                identity.EXPECTED_CONFIDENCE_CHECKPOINT_SHA256 if confidence else None
            ),
        }
    )
    if confidence:
        value["candidate_ensemble_hash_contract"] = identity.CANDIDATE_ENSEMBLE_HASH_CONTRACT
        value["confidence_score_ledger_contract"] = identity.CONFIDENCE_SCORE_LEDGER_CONTRACT
        stats = value["stats"]
        assert isinstance(stats, dict)
        for selector in identity.CONFIDENCE_SELECTORS:
            stats[selector] = {"mean_rmsd": 1.0}
    if eta > 0.0:
        value["guidance_runtime_stats"] = trace_runtime
        value["guidance_parameter_set"] = {"sha256": identity.EXPECTED_GUIDANCE_PARAMETER_SHA256}
        value["guidance_receptor_policy_identities"] = {
            identity.EXPECTED_RECEPTOR_POLICY_SHA256: {
                "sha256": identity.EXPECTED_RECEPTOR_POLICY_SHA256
            }
        }
    return value


@pytest.mark.parametrize(
    ("partition", "gpu_name"),
    (
        ("6000ada", "NVIDIA RTX 6000 Ada Generation"),
        ("heavy", "NVIDIA H100 80GB HBM3"),
        ("heavy", "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition"),
    ),
)
def test_runtime_accepts_declared_gpu_partition_inventory(partition: str, gpu_name: str) -> None:
    summary = {
        "runtime": {
            "device": "cuda",
            "slurm_partition": partition,
            "gpu": gpu_name,
            "gpu_total_memory_bytes": 48_000 * 1024**2,
            "torch": "test",
            "cuda": "test",
            "cuda_max_memory_allocated_bytes": 1024,
            "cuda_max_memory_reserved_bytes": 2048,
        }
    }
    identity._validate_runtime(summary, label="test")


def test_runtime_rejects_unknown_or_undersized_gpu() -> None:
    runtime = {
        "device": "cuda",
        "slurm_partition": "heavy",
        "gpu": "NVIDIA A5000",
        "gpu_total_memory_bytes": 48_000 * 1024**2,
        "torch": "test",
        "cuda": "test",
        "cuda_max_memory_allocated_bytes": 1024,
        "cuda_max_memory_reserved_bytes": 2048,
    }
    with pytest.raises(ValueError, match="sampling GPU on heavy must match"):
        identity._validate_runtime({"runtime": runtime}, label="test")
    runtime["gpu"] = "NVIDIA H100 80GB HBM3"
    runtime["slurm_partition"] = "test"
    with pytest.raises(ValueError, match="slurm_partition"):
        identity._validate_runtime({"runtime": runtime}, label="test")
    runtime["slurm_partition"] = "6000ada"
    with pytest.raises(ValueError, match="sampling GPU"):
        identity._validate_runtime({"runtime": runtime}, label="test")
    runtime["slurm_partition"] = "heavy"
    runtime["gpu_total_memory_bytes"] = 47_999 * 1024**2
    with pytest.raises(ValueError, match="gpu_total_memory_bytes"):
        identity._validate_runtime({"runtime": runtime}, label="test")


def _build_smoke_fixture(tmp_path: Path) -> tuple[Path, Path]:
    parent_root = tmp_path / "parent"
    confidence_root = tmp_path / "confidence"
    parent_fields = [
        "id",
        "saved_pose_sha256_json",
        "sampling_seed",
        "prior_pool_sha256",
        "first_index",
        "vina_index",
        "oracle_index",
        "first_rmsd",
        "vina_rmsd",
        "oracle_rmsd",
        "guidance_direct_step_trace_json",
    ]
    confidence_fields = parent_fields + sorted(identity._CONFIDENCE_COLUMNS)
    for dataset in identity.DATASETS:
        complex_id = identity.SMOKE_IDS[dataset]
        for eta in identity.ETA_VALUES:
            run_name = eta_report.expected_run_name(dataset, eta)
            trace, trace_runtime = _trace_and_runtime(eta) if eta > 0.0 else ("[]", None)
            parent_ledger: dict[str, str] = {}
            confidence_ledger: dict[str, str] = {}
            for selector in identity.LEGACY_SELECTORS:
                payload = f"{run_name}:{complex_id}:{selector}\n".encode()
                parent_ledger[selector] = _pose_digest(
                    parent_root, run_name, dataset, complex_id, selector, payload
                )
                confidence_ledger[selector] = _pose_digest(
                    confidence_root, run_name, dataset, complex_id, selector, payload
                )
            for selector in identity.CONFIDENCE_SELECTORS:
                payload = f"{run_name}:{complex_id}:{selector}\n".encode()
                confidence_ledger[selector] = _pose_digest(
                    confidence_root, run_name, dataset, complex_id, selector, payload
                )

            parent_row = {
                "id": complex_id,
                "saved_pose_sha256_json": json.dumps(parent_ledger, sort_keys=True),
                "sampling_seed": "43",
                "prior_pool_sha256": "d" * 64,
                "first_index": "0",
                "vina_index": "1",
                "oracle_index": "2",
                "first_rmsd": "1.0",
                "vina_rmsd": "2.0",
                "oracle_rmsd": "0.5",
                "guidance_direct_step_trace_json": trace,
            }
            confidence_row = {
                **parent_row,
                "saved_pose_sha256_json": json.dumps(confidence_ledger, sort_keys=True),
                "candidate_ensemble_sha256": "e" * 64,
                "confidence_candidate_scores_json": json.dumps(_scores(), sort_keys=True),
            }
            for selector, index in (
                ("confidence", 0),
                ("confidence_filter", 0),
                ("confidence_final", 2),
            ):
                score = _scores()[index]
                confidence_row[f"{selector}_index"] = str(index)
                confidence_row[f"{selector}_rmsd"] = str(float(index + 1))
                confidence_row[f"{selector}_pred_rmsd"] = str(score["confidence_rmsd"])
                confidence_row[f"{selector}_pred_success"] = str(score["confidence_success"])
                for term in identity._FAST_TERMS:
                    confidence_row[f"{selector}_fast_{term}"] = "True"

            parent_csv = parent_root / f"{run_name}.csv"
            confidence_csv = confidence_root / f"{run_name}.csv"
            _write_csv(parent_csv, parent_fields, parent_row)
            _write_csv(confidence_csv, confidence_fields, confidence_row)
            parent_summary = _summary(
                dataset=dataset,
                eta=eta,
                csv_path=parent_csv,
                confidence=False,
                trace_runtime=trace_runtime,
            )
            confidence_summary = _summary(
                dataset=dataset,
                eta=eta,
                csv_path=confidence_csv,
                confidence=True,
                trace_runtime=deepcopy(trace_runtime),
            )
            (parent_root / f"{run_name}.summary.json").write_text(json.dumps(parent_summary))
            (confidence_root / f"{run_name}.summary.json").write_text(
                json.dumps(confidence_summary)
            )
    return parent_root, confidence_root


def _rewrite_csv(path: Path, mutate: object) -> None:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    assert len(rows) == 1
    mutate(rows[0])  # type: ignore[operator]
    _write_csv(path, fields, rows[0])


def test_smoke_identity_audit_is_complete_and_auditable(tmp_path: Path) -> None:
    parent, confidence = _build_smoke_fixture(tmp_path)
    audit = identity.build_identity_audit(parent, confidence, smoke=True)

    assert audit["status"] == "passed"
    assert audit["audit_contract"] == "EFFDOCK_CONFIDENCE_REPLAY_IDENTITY_V2"
    assert audit["schema_version"] == "effdock.guidance_eta_sweep_confidence_identity.v2"
    assert audit["parent_sentinels_verified"] is True
    assert audit["summary_contracts_verified"] is True
    assert audit["candidate_ensemble_hashes_present"] is True
    assert audit["selector_recomputed"] is True
    assert audit["cuda_runtime"]["slurm_partitions"] == ["6000ada"]
    assert audit["coverage"] == {
        "datasets": 2,
        "cells": 16,
        "shards": 16,
        "rows": 16,
        "per_dataset": {
            "astex": {"cells": 8, "shards": 8, "rows": 8, "ids_per_cell": 1},
            "posebusters": {
                "cells": 8,
                "shards": 8,
                "rows": 8,
                "ids_per_cell": 1,
            },
        },
    }
    assert audit["frozen_hashes"]["confidence_checkpoint_sha256"] == (
        identity.EXPECTED_CONFIDENCE_CHECKPOINT_SHA256
    )
    assert audit["numerical_replay_contract"]["direct_step_and_summary_guidance_telemetry"] == {
        "rel_tol": 2e-4,
        "abs_tol": 2e-4,
        "finite_required": True,
        "counters": "exact",
    }
    assert len(audit["global_equivalence_ledger_sha256"]) == 64


def test_identity_accepts_bounded_numeric_and_pose_replay_noise(tmp_path: Path) -> None:
    parent, confidence = _build_smoke_fixture(tmp_path)
    dataset = "astex"
    complex_id = identity.SMOKE_IDS[dataset]
    run_name = eta_report.expected_run_name(dataset, 0.0)
    parent_csv = parent / f"{run_name}.csv"
    confidence_csv = confidence / f"{run_name}.csv"
    _rewrite_csv(
        confidence_csv,
        lambda row: row.__setitem__("oracle_rmsd", "0.50005"),
    )
    parent_digest = _write_test_sdf(
        parent,
        run_name,
        dataset,
        complex_id,
        "oracle",
        first_atom_offset=0.0,
    )
    replay_digest = _write_test_sdf(
        confidence,
        run_name,
        dataset,
        complex_id,
        "oracle",
        first_atom_offset=0.0001,
    )
    _replace_pose_hash(parent_csv, "oracle", parent_digest)
    _replace_pose_hash(confidence_csv, "oracle", replay_digest)
    summary_path = confidence / f"{run_name}.summary.json"
    summary = json.loads(summary_path.read_text())
    summary["stats"]["oracle"]["mean_rmsd"] = 0.50005
    summary_path.write_text(json.dumps(summary))

    audit = identity.build_identity_audit(parent, confidence, smoke=True)
    observed = audit["maximum_observed_deltas"]
    assert observed["legacy_csv_scalars"]["max_absolute_delta"] == pytest.approx(5e-5)
    assert observed["legacy_summary_stats"]["max_absolute_delta"] == pytest.approx(5e-5)
    assert observed["legacy_selected_poses"]["hash_mismatch_but_numerically_equivalent"] == 1
    assert observed["legacy_selected_poses"]["max_atom_displacement_angstrom"] == (
        pytest.approx(1e-4)
    )


def test_identity_rejects_numeric_delta_beyond_contract(tmp_path: Path) -> None:
    parent, confidence = _build_smoke_fixture(tmp_path)
    run_name = eta_report.expected_run_name("astex", 0.0)
    _rewrite_csv(
        confidence / f"{run_name}.csv",
        lambda row: row.__setitem__("oracle_rmsd", "0.5002"),
    )
    with pytest.raises(ValueError, match="numerical replay mismatch"):
        identity.build_identity_audit(parent, confidence, smoke=True)


def test_identity_rejects_categorical_or_discrete_replay_change(tmp_path: Path) -> None:
    parent, confidence = _build_smoke_fixture(tmp_path)
    run_name = eta_report.expected_run_name("astex", 0.0)
    _rewrite_csv(
        confidence / f"{run_name}.csv",
        lambda row: row.__setitem__("sampling_seed", "44"),
    )
    with pytest.raises(ValueError, match="categorical/discrete replay value differs"):
        identity.build_identity_audit(parent, confidence, smoke=True)


def test_identity_rejects_changed_saved_pose_sentinel(tmp_path: Path) -> None:
    parent, confidence = _build_smoke_fixture(tmp_path)
    run_name = eta_report.expected_run_name("astex", 0.0)

    def mutate(row: dict[str, str]) -> None:
        ledger = json.loads(row["saved_pose_sha256_json"])
        ledger["oracle"] = "f" * 64
        row["saved_pose_sha256_json"] = json.dumps(ledger, sort_keys=True)

    _rewrite_csv(confidence / f"{run_name}.csv", mutate)
    with pytest.raises(ValueError, match="saved pose SHA-256 differs from CSV ledger"):
        identity.build_identity_audit(parent, confidence, smoke=True)


def test_identity_rejects_legacy_pose_outside_coordinate_tolerance(tmp_path: Path) -> None:
    parent, confidence = _build_smoke_fixture(tmp_path)
    dataset = "astex"
    complex_id = identity.SMOKE_IDS[dataset]
    run_name = eta_report.expected_run_name(dataset, 0.0)
    parent_digest = _write_test_sdf(
        parent,
        run_name,
        dataset,
        complex_id,
        "oracle",
        first_atom_offset=0.0,
    )
    replay_digest = _write_test_sdf(
        confidence,
        run_name,
        dataset,
        complex_id,
        "oracle",
        first_atom_offset=0.002,
    )
    _replace_pose_hash(parent / f"{run_name}.csv", "oracle", parent_digest)
    _replace_pose_hash(confidence / f"{run_name}.csv", "oracle", replay_digest)

    with pytest.raises(ValueError, match="coordinate RMSD .* exceeds"):
        identity.build_identity_audit(parent, confidence, smoke=True)


def test_nested_telemetry_uses_wider_tolerance_but_exact_counts() -> None:
    observations = identity._new_delta_observations()
    identity._compare_numeric_tree(
        {"model_atom_speed_rms_sum": 1.0, "pose_count": 100},
        {"model_atom_speed_rms_sum": 1.00015, "pose_count": 100},
        label="trace",
        rel_tol=identity.TELEMETRY_REL_TOL,
        abs_tol=identity.TELEMETRY_ABS_TOL,
        bucket=observations["direct_step_telemetry"],
    )
    identity._compare_numeric_tree(
        {"model_guide_cosine_p99": 0.6770693737268451},
        {"model_guide_cosine_p99": 0.676925359964371},
        label="trace",
        rel_tol=identity.TELEMETRY_REL_TOL,
        abs_tol=identity.TELEMETRY_ABS_TOL,
        bucket=observations["direct_step_telemetry"],
    )
    with pytest.raises(ValueError, match="numerical replay mismatch"):
        identity._compare_numeric_tree(
            {"model_atom_speed_rms_sum": 1.0},
            {"model_atom_speed_rms_sum": 1.0003},
            label="trace",
            rel_tol=identity.TELEMETRY_REL_TOL,
            abs_tol=identity.TELEMETRY_ABS_TOL,
            bucket=observations["direct_step_telemetry"],
        )
    with pytest.raises(ValueError, match="discrete replay value differs"):
        identity._compare_numeric_tree(
            {"pose_count": 100},
            {"pose_count": 99},
            label="trace",
            rel_tol=identity.TELEMETRY_REL_TOL,
            abs_tol=identity.TELEMETRY_ABS_TOL,
            bucket=observations["direct_step_telemetry"],
        )


def test_identity_recomputes_primary_and_cluster_free_filter(tmp_path: Path) -> None:
    parent, confidence = _build_smoke_fixture(tmp_path)
    run_name = eta_report.expected_run_name("astex", 0.0)
    _rewrite_csv(
        confidence / f"{run_name}.csv",
        lambda row: row.__setitem__("confidence_filter_index", "1"),
    )
    with pytest.raises(ValueError, match="does not match frozen filter"):
        identity.build_identity_audit(parent, confidence, smoke=True)


def test_identity_requires_finite_full_candidate_score_ledger(tmp_path: Path) -> None:
    parent, confidence = _build_smoke_fixture(tmp_path)
    run_name = eta_report.expected_run_name("astex", 0.0)

    def mutate(row: dict[str, str]) -> None:
        scores = json.loads(row["confidence_candidate_scores_json"])
        scores[1]["confidence_rmsd"] = float("nan")
        row["confidence_candidate_scores_json"] = json.dumps(scores)

    _rewrite_csv(confidence / f"{run_name}.csv", mutate)
    with pytest.raises(ValueError, match="must be finite"):
        identity.build_identity_audit(parent, confidence, smoke=True)


def test_identity_requires_frozen_child_summary_contracts(tmp_path: Path) -> None:
    parent, confidence = _build_smoke_fixture(tmp_path)
    run_name = eta_report.expected_run_name("astex", 0.0)
    path = confidence / f"{run_name}.summary.json"
    summary = json.loads(path.read_text())
    summary["confidence_score_ledger_contract"] = "wrong"
    path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="confidence_score_ledger_contract must be"):
        identity.build_identity_audit(parent, confidence, smoke=True)
