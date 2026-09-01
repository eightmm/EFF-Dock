from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem

sys.path.insert(0, str(Path(__file__).parents[1]))

from scripts import report_early_time_sampler_plinder_k2_gate as gate_report

NUM_SAMPLES = 4
NUM_STEPS = 2
PRIOR_POOL_SIZE = 4


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


_PRIMARY_ELIGIBLE_IDS = ["sys0__L", "sys1__L", "sys2__L", "sys2__M"]
_SYNTHETIC_ELIGIBLE_IDS = [
    sample_id
    for system_index in range(1_017)
    for sample_id in (
        [f"zzsys{system_index:04d}__L", f"zzsys{system_index:04d}__M"]
        if system_index < 14
        else [f"zzsys{system_index:04d}__L"]
    )
]
_TEST_ELIGIBLE_IDS = sorted(_PRIMARY_ELIGIBLE_IDS + _SYNTHETIC_ELIGIBLE_IDS)
_TEST_EXCLUDED_IDS = [f"zzzz_excluded{index:02d}__L" for index in range(41)]
_TEST_FULL_IDS = sorted(_TEST_ELIGIBLE_IDS + _TEST_EXCLUDED_IDS)
_TEST_ELIGIBLE_NEWLINE_SHA256 = gate_report._ids_sha256(_TEST_ELIGIBLE_IDS)
_TEST_ELIGIBILITY_PAYLOAD = {
    "schema_version": "effdock.plinder_checkpoint_eligibility.v1",
    "protocol_id": gate_report.PROTOCOL_ID,
    "status": "complete",
    "inventory": {
        "full_count": len(_TEST_FULL_IDS),
        "full_ids": _TEST_FULL_IDS,
        "full_ids_sha256": gate_report._versioned_ids_sha256(_TEST_FULL_IDS),
        "eligible_count": len(_TEST_ELIGIBLE_IDS),
        "eligible_ids": _TEST_ELIGIBLE_IDS,
        "eligible_ids_sha256": gate_report._versioned_ids_sha256(_TEST_ELIGIBLE_IDS),
        "eligible_ids_newline_sha256": _TEST_ELIGIBLE_NEWLINE_SHA256,
        "eligible_system_count": 1_020,
        "excluded_count": len(_TEST_EXCLUDED_IDS),
        "excluded_ids": _TEST_EXCLUDED_IDS,
        "excluded_ids_sha256": gate_report._versioned_ids_sha256(_TEST_EXCLUDED_IDS),
        "preflight_error_count": 0,
        "preflight_error_ids": [],
    },
}
_TEST_ELIGIBILITY_BYTES = (
    json.dumps(_TEST_ELIGIBILITY_PAYLOAD, separators=(",", ":"), sort_keys=True) + "\n"
).encode("utf-8")
_TEST_ELIGIBILITY_SHA256 = hashlib.sha256(_TEST_ELIGIBILITY_BYTES).hexdigest()


@pytest.fixture(autouse=True)
def _use_synthetic_frozen_eligibility(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        gate_report,
        "FROZEN_ELIGIBILITY_MANIFEST_SHA256",
        _TEST_ELIGIBILITY_SHA256,
    )
    monkeypatch.setattr(
        gate_report,
        "FROZEN_ELIGIBLE_IDS_SHA256",
        _TEST_ELIGIBLE_NEWLINE_SHA256,
    )


def _candidate_rmsds(k2: int) -> list[float]:
    return [1.0 + 0.01 * index for index in range(k2)] + [
        2.0 + 0.01 * index for index in range(NUM_SAMPLES - k2)
    ]


def _row(
    *,
    sample_id: str,
    system_id: str,
    global_index: int,
    arm: str,
    k2: int,
    fallback_count: int = 0,
) -> dict[str, object]:
    rmsds = _candidate_rmsds(k2)
    valid = [True] * NUM_SAMPLES
    methods = ["mapped_index_fallback"] * fallback_count + ["rdkit_calc_rms_symmetry_no_align"] * (
        NUM_SAMPLES - fallback_count
    )
    shared = sample_id
    return {
        "id": sample_id,
        "arm": arm,
        "plinder_system_id": system_id,
        "plinder_ligand_chain": sample_id.rsplit("__", 1)[1],
        "plinder_global_index": global_index,
        "sampling_seed": 42 + global_index,
        "ligand_conformer_seed": 0,
        "num_samples": NUM_SAMPLES,
        "prior_pool_size": PRIOR_POOL_SIZE,
        "prior_pool_sha256": _digest(f"prior:{shared}"),
        "candidate_ensemble_sha256": _digest(f"candidates:{arm}:{shared}"),
        "candidate_rmsds_json": json.dumps(rmsds, separators=(",", ":")),
        "candidate_rmsd_method_json": json.dumps(methods, separators=(",", ":")),
        "num_mapped_index_rmsd_fallback_candidates": fallback_count,
        "candidate_fast_valid_json": json.dumps(valid, separators=(",", ":")),
        "num_rmsd_lt2_candidates": k2,
        "fraction_rmsd_lt2_candidates": k2 / NUM_SAMPLES,
        "num_fast_valid_candidates": NUM_SAMPLES,
        "num_fast_valid_rmsd_lt2_candidates": k2,
        "first_index": 0,
        "selected_index": 0,
        "first_rmsd": rmsds[0],
        "selected_rmsd": rmsds[0],
        "oracle_rmsd": min(rmsds),
        "mean_sample_rmsd": sum(rmsds) / NUM_SAMPLES,
        "all_poses_count": NUM_SAMPLES,
        "all_poses_sdf": f"poses/{arm}/{sample_id}.sdf",
        "all_poses_sdf_sha256": _digest(f"all-poses:{arm}:{sample_id}"),
        "selector_profile": "candidate_only",
        "guidance_mode": "none",
        "sampling_dynamics": "deterministic_ode",
        "translation_sde_base_sigma": 0.0,
        "full_heavy_atom_bijection": True,
        "match_method": "strict",
        "pose_diversity_contract": gate_report.POSE_DIVERSITY_CONTRACT,
        "pose_diversity_round_decimals": 3,
        "diversity_heavy_atom_count": 12,
        "coordinate_unique_count": NUM_SAMPLES,
        "pairwise_heavy_atom_rmsd_mean": 3.0,
        "pairwise_heavy_atom_rmsd_median": 3.0,
        "pairwise_heavy_atom_rmsd_ge2_fraction": 1.0,
        "nearest_neighbor_heavy_atom_rmsd_median": 2.5,
        "c2_connected_component_count": NUM_SAMPLES,
        "ligand_input_identity_sha256": _digest(f"smiles:{shared}"),
        "protein_sha256": _digest(f"protein:{shared}"),
        "ligand_reference_sha256": _digest(f"reference:{shared}"),
        "processed_meta_sha256": _digest(f"meta:{shared}"),
        "checkpoint_sha256": gate_report.FROZEN_CHECKPOINT_SHA256[arm],
    }


def _write_all_poses_sdf(
    root: Path,
    row: dict[str, object],
    *,
    record_count: int = NUM_SAMPLES,
    property_override: tuple[int, str, str] | None = None,
) -> Path:
    path = root / str(row["all_poses_sdf"])
    path.parent.mkdir(parents=True, exist_ok=True)
    molecule = Chem.MolFromSmiles("CC")
    assert molecule is not None
    conformer = Chem.Conformer(molecule.GetNumAtoms())
    for atom_index in range(molecule.GetNumAtoms()):
        conformer.SetAtomPosition(atom_index, (float(atom_index), 0.0, 0.0))
    molecule.AddConformer(conformer)
    writer = Chem.SDWriter(str(path))
    for sample_index in range(record_count):
        pose = Chem.Mol(molecule)
        props = {
            "sample_index": str(sample_index),
            "sampling_seed": str(row["sampling_seed"]),
            "ligand_conformer_seed": str(row["ligand_conformer_seed"]),
            "candidate_ensemble_sha256": str(row["candidate_ensemble_sha256"]),
        }
        if property_override is not None and property_override[0] == sample_index:
            props[property_override[1]] = property_override[2]
        for name, value in props.items():
            pose.SetProp(name, value)
        writer.write(pose)
    writer.close()
    return path


def _fixed_identities() -> dict[str, object]:
    assets = {
        "protocol_document": gate_report.FROZEN_PROTOCOL_SHA256,
        "split": gate_report.FROZEN_SPLIT_SHA256,
        "pool_parquet": gate_report.FROZEN_POOL_SHA256,
        "config": gate_report.FROZEN_CONFIG_SHA256,
        "raw_gate": gate_report.FROZEN_RAW_GATE_SHA256,
        "conformer_mapping_audit": gate_report.FROZEN_AUDIT_SHA256,
    }
    code_hashes = {
        "scripts/run_plinder_checkpoint_paired_validation.py": gate_report.FROZEN_RUNNER_SHA256,
        "src/effdock/workflows/evaluate.py": gate_report.FROZEN_EVALUATOR_SHA256,
        "src/effdock/evaluation/benchmark.py": gate_report.FROZEN_BENCHMARK_SHA256,
        "src/effdock/workflows/benchmark_inputs.py": _digest("benchmark-inputs"),
        "src/effdock/inference/docking.py": _digest("docking"),
        "src/effdock/inference/preprocess.py": _digest("preprocess"),
        "src/effdock/inference/sampler.py": _digest("sampler"),
    }
    code_digest = hashlib.sha256()
    code_digest.update(b"EFFDOCK_PLINDER_PAIRED_CODE_INVENTORY_V1\0")
    code_digest.update(
        json.dumps(code_hashes, separators=(",", ":"), sort_keys=True).encode("utf-8")
    )
    return {
        **{name: {"sha256": digest} for name, digest in assets.items()},
        "checkpoints": {
            arm: {"sha256": gate_report.FROZEN_CHECKPOINT_SHA256[arm]} for arm in gate_report.ARMS
        },
        "code": {
            "contract": "EFFDOCK_PLINDER_PAIRED_CODE_INVENTORY_V1",
            "sha256": code_digest.hexdigest(),
            "files": {name: {"sha256": digest} for name, digest in code_hashes.items()},
        },
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return gate_report._file_sha256(path)


def _write_smoke_fixture(root: Path) -> dict[str, Path]:
    sample_ids = _PRIMARY_ELIGIBLE_IDS
    systems = ["sys0", "sys1", "sys2", "sys2"]
    k2_by_arm = {
        "s25_ema": [1, 1, 0, 2],
        "s50_ema": [1, 2, 0, 2],
        "parent50k_plus10k_t0p10_ema": [2, 3, 1, 3],
        "s50_ema_replay": [1, 2, 0, 2],
    }
    stage_root = root / "smoke" / "shard-000-of-001"
    artifacts: dict[str, dict[str, object]] = {}
    csv_paths: dict[str, Path] = {}
    for arm in (*gate_report.ARMS, gate_report.REPLAY_ARM):
        rows = [
            _row(
                sample_id=sample_id,
                system_id=system_id,
                global_index=index,
                arm=arm,
                k2=k2_by_arm[arm][index - 1],
                fallback_count=(1 if arm == "parent50k_plus10k_t0p10_ema" and index == 1 else 0),
            )
            for index, (sample_id, system_id) in enumerate(
                zip(sample_ids, systems, strict=True), start=1
            )
        ]
        for row in rows:
            pose_path = _write_all_poses_sdf(root, row)
            row["all_poses_sdf_sha256"] = gate_report._file_sha256(pose_path)
        csv_path = stage_root / "arms" / arm / "results.csv"
        csv_sha256 = _write_csv(csv_path, rows)
        csv_paths[arm] = csv_path
        artifacts[arm] = {
            "arm": arm,
            "count": len(rows),
            "results_csv": str(csv_path),
            "results_csv_sha256": csv_sha256,
        }
    replay_gate = {
        "required": True,
        "passed": True,
        "stage": "smoke",
        "checked_count": len(sample_ids),
        "k2_coverage_classification_mismatches": 0,
        "fast_valid_k2_coverage_classification_mismatches": 0,
        "mean_abs_k2_difference": 0.0,
        "diversity_aggregate_ratios": {
            "nearest_neighbor_heavy_atom_rmsd_median": 1.0,
            "c2_connected_component_count": 1.0,
            "coordinate_unique_count": 1.0,
        },
    }
    summary = {
        "schema_version": "effdock.plinder_checkpoint_paired_shard.v1",
        "protocol_id": gate_report.PROTOCOL_ID,
        "status": "complete",
        "run_id": "unit-smoke",
        "mode": "smoke",
        "settings": {
            "stage": "smoke",
            "selected_count": len(sample_ids),
            "num_samples": NUM_SAMPLES,
            "num_steps": NUM_STEPS,
            "model_pose_step_budget": NUM_SAMPLES * NUM_STEPS,
            "sigma": 2.0,
            "prior_pool_size": PRIOR_POOL_SIZE,
            "time_schedule": "late",
            "schedule_power": 3.0,
            "pocket_cutoff_angstrom": 10.0,
            "center_jitter_sigma": 0.0,
            "confidence": False,
            "vina_guidance_scale": 0.0,
            "unified_guidance_scale": 0.0,
            "fk_constraint_beta": 0.0,
            "translation_sde_base_sigma": 0.0,
            "refine": "none",
            "selector_profile": "candidate_only",
            "ligand_conformer_seed": 0,
            "include_s50_replay": True,
        },
        "eligibility_manifest": {
            "path": "eligibility.json",
            "sha256": gate_report.FROZEN_ELIGIBILITY_MANIFEST_SHA256,
            "eligible_count": gate_report.FULL_ELIGIBLE_COUNT,
            "eligible_ids_sha256": gate_report._versioned_ids_sha256(
                _TEST_ELIGIBLE_IDS
            ),
            "eligible_ids_newline_sha256": gate_report.FROZEN_ELIGIBLE_IDS_SHA256,
            "eligible_system_count": gate_report.FULL_SYSTEM_COUNT,
            "ineligible_count": gate_report.FULL_INELIGIBLE_COUNT,
        },
        "inventory": {
            "full_count": gate_report.FULL_SPLIT_COUNT,
            "eligible_count": gate_report.FULL_ELIGIBLE_COUNT,
            "selected_count": len(sample_ids),
            "selected_ids": sample_ids,
            "selected_ids_sha256": gate_report._versioned_ids_sha256(sample_ids),
            "num_shards": 1,
            "shard_index": 0,
            "assigned_count": len(sample_ids),
            "assigned_ids": sample_ids,
            "assigned_ids_sha256": gate_report._versioned_ids_sha256(sample_ids),
            "arm_success_counts": {
                arm: len(sample_ids) for arm in (*gate_report.ARMS, gate_report.REPLAY_ARM)
            },
        },
        "operational_inventory": {
            "requested_count": gate_report.FULL_SPLIT_COUNT,
            "evaluable_count": gate_report.FULL_ELIGIBLE_COUNT,
            "common_preprocessing_failure_count": gate_report.FULL_INELIGIBLE_COUNT,
            "common_preprocessing_failure_ids": _TEST_EXCLUDED_IDS,
            "common_preprocessing_failure_ids_sha256": gate_report._versioned_ids_sha256(
                _TEST_EXCLUDED_IDS
            ),
            "per_arm_preprocessing_failure_count": {
                arm: gate_report.FULL_INELIGIBLE_COUNT
                for arm in (*gate_report.ARMS, gate_report.REPLAY_ARM)
            },
            "operational_sensitivity_assignment": "common preprocessing failures have K2=0",
        },
        "arms": [
            {
                "name": arm,
                "checkpoint_sha256": gate_report.FROZEN_CHECKPOINT_SHA256[arm],
            }
            for arm in (*gate_report.ARMS, gate_report.REPLAY_ARM)
        ],
        "fixed_identities": _fixed_identities(),
        "seed_contract": {
            "name": "BASE42_PLUS_SORTED_FULL_VAL_GLOBAL_INDEX_1_BASED_V1",
            "base_seed": 42,
            "order": "globally sorted full 1076-ID validation cohort before eligibility",
        },
        "ligand_input_contract": {
            "source": "data/plinder_pool.parquet:ligand_rdkit_canonical_smiles",
            "conformer_seed": 0,
            "heavy_atom_normalization": "RemoveHs_then_RemoveAllHs",
            "crystal_sdf_role": "RMSD reference and atom-mapping eligibility only",
            "crystal_sdf_input_fallback": False,
        },
        "paired_identity_gate": {
            "passed": True,
            "checked_count": len(sample_ids),
            "fields": [
                "sampling_seed",
                "ligand_conformer_seed",
                "prior_pool_sha256",
                "protein_sha256",
                "ligand_reference_sha256",
                "ligand_input_identity_sha256",
            ],
        },
        "replay_integrity_gate": replay_gate,
        "failures": [],
        "artifacts": {"arms": artifacts},
    }
    (root / "eligibility.json").write_bytes(_TEST_ELIGIBILITY_BYTES)
    stage_root.mkdir(parents=True, exist_ok=True)
    (stage_root / "paired_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return csv_paths


def _rewrite_csv(path: Path, mutation) -> None:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    mutation(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary_path = path.parents[2] / "paired_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    arm = path.parent.name
    summary["artifacts"]["arms"][arm]["results_csv_sha256"] = gate_report._file_sha256(path)
    summary_path.write_text(json.dumps(summary), encoding="utf-8")


def _rewrite_summary(root: Path, mutation) -> None:
    summary_path = root / "smoke/shard-000-of-001/paired_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    mutation(summary)
    summary_path.write_text(json.dumps(summary), encoding="utf-8")


def _first_csv_row(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        return next(csv.DictReader(handle))


def _build_smoke_report(root: Path) -> dict[str, object]:
    return gate_report.build_report(
        root,
        stage="smoke",
        expected_count=4,
        expected_num_samples=NUM_SAMPLES,
        expected_num_steps=NUM_STEPS,
        expected_prior_pool_size=PRIOR_POOL_SIZE,
        bootstrap_resamples=20,
    )


def test_smoke_report_recomputes_replay_and_never_emits_efficacy(tmp_path: Path) -> None:
    paths = _write_smoke_fixture(tmp_path)
    report = _build_smoke_report(tmp_path)

    assert report["status"] == "complete_smoke_integrity_pass"
    assert report["decision"]["selection_eligible"] is False
    assert report["decision"]["action"] == "advance_to_next_execution_stage"
    assert "arms" not in report
    assert "comparisons" not in report
    assert "operational_full_1076_sensitivity" not in report
    serialized = json.dumps(report, sort_keys=True)
    assert "k2_total" not in serialized
    assert "gained_ids" not in serialized
    assert "lost_ids" not in serialized
    assert report["integrity"]["replay"]["passed"] is True
    assert report["integrity"]["all_pose_sdf_artifact_audit"] == {
        "files_verified": 16,
        "records_verified": 64,
        "sha256_recomputed": True,
        "reader": "RDKit Chem.ForwardSDMolSupplier over binary stream",
        "ordered_properties_verified": [
            "sample_index",
            "sampling_seed",
            "ligand_conformer_seed",
            "candidate_ensemble_sha256",
        ],
        "paths_confined_to_output_root": True,
    }
    assert report["integrity"]["rmsd_fallback_ids_by_arm"]["parent50k_plus10k_t0p10_ema"] == [
        {"id": "sys0__L", "candidate_count": 1}
    ]
    assert paths["s50_ema"].is_file()


def test_report_rejects_all_pose_sdf_byte_tamper(tmp_path: Path) -> None:
    paths = _write_smoke_fixture(tmp_path)
    row = _first_csv_row(paths["s50_ema"])
    pose_path = tmp_path / row["all_poses_sdf"]
    with pose_path.open("ab") as handle:
        handle.write(b"\nTAMPERED\n")

    with pytest.raises(ValueError, match="all-pose SDF SHA-256 mismatch"):
        _build_smoke_report(tmp_path)


def test_report_rejects_missing_all_pose_sdf(tmp_path: Path) -> None:
    paths = _write_smoke_fixture(tmp_path)
    row = _first_csv_row(paths["s50_ema"])
    pose_path = tmp_path / row["all_poses_sdf"]
    pose_path.unlink()

    with pytest.raises(ValueError, match="declared output file does not exist"):
        _build_smoke_report(tmp_path)


def test_report_rejects_eligibility_manifest_byte_tamper(tmp_path: Path) -> None:
    _write_smoke_fixture(tmp_path)
    with (tmp_path / "eligibility.json").open("ab") as handle:
        handle.write(b"\n")

    with pytest.raises(ValueError, match="eligibility manifest SHA-256 mismatch"):
        _build_smoke_report(tmp_path)


def test_report_rejects_operational_failure_inventory_drift(tmp_path: Path) -> None:
    _write_smoke_fixture(tmp_path)
    _rewrite_summary(
        tmp_path,
        lambda summary: summary["operational_inventory"].__setitem__(
            "common_preprocessing_failure_ids",
            list(reversed(_TEST_EXCLUDED_IDS)),
        ),
    )

    with pytest.raises(ValueError, match="common failure IDs differ from manifest"):
        _build_smoke_report(tmp_path)


def test_report_rejects_wrong_per_arm_preprocessing_failure_count(tmp_path: Path) -> None:
    _write_smoke_fixture(tmp_path)
    _rewrite_summary(
        tmp_path,
        lambda summary: summary["operational_inventory"][
            "per_arm_preprocessing_failure_count"
        ].__setitem__("s50_ema", 40),
    )

    with pytest.raises(ValueError, match="wrong per-arm failure accounting"):
        _build_smoke_report(tmp_path)


@pytest.mark.parametrize(
    ("field", "bad_value", "message"),
    [
        ("plinder_system_id", "wrong_system", "expected sample-key system"),
        ("plinder_ligand_chain", "wrong_chain", "expected sample-key chain"),
    ],
)
def test_report_rejects_row_cluster_identity_not_derived_from_sample_key(
    tmp_path: Path,
    field: str,
    bad_value: str,
    message: str,
) -> None:
    paths = _write_smoke_fixture(tmp_path)
    _rewrite_csv(
        paths["s50_ema"],
        lambda row: row.__setitem__(field, bad_value),
    )

    with pytest.raises(ValueError, match=message):
        _build_smoke_report(tmp_path)


def test_report_rejects_self_consistent_seed_with_wrong_frozen_global_index(
    tmp_path: Path,
) -> None:
    paths = _write_smoke_fixture(tmp_path)

    def change_index_and_seed(row: dict[str, str]) -> None:
        row["plinder_global_index"] = "2"
        row["sampling_seed"] = "44"

    _rewrite_csv(paths["s50_ema"], change_index_and_seed)
    with pytest.raises(ValueError, match="expected frozen 1-based index 1, got 2"):
        _build_smoke_report(tmp_path)


def test_report_rejects_all_pose_sdf_outside_report_root(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    paths = _write_smoke_fixture(run_root)
    csv_path = paths["s50_ema"]
    row = _first_csv_row(csv_path)
    source_path = run_root / row["all_poses_sdf"]
    outside_path = tmp_path / "outside.sdf"
    outside_path.write_bytes(source_path.read_bytes())
    _rewrite_csv(
        csv_path,
        lambda first: first.__setitem__("all_poses_sdf", str(outside_path)),
    )

    with pytest.raises(ValueError, match="output resolves outside report root"):
        _build_smoke_report(run_root)


def test_report_rejects_stale_atomic_publish_attempt_path(tmp_path: Path) -> None:
    paths = _write_smoke_fixture(tmp_path)
    csv_path = paths["s50_ema"]
    row = _first_csv_row(csv_path)
    source_path = tmp_path / row["all_poses_sdf"]
    published_path = csv_path.parent / "poses" / "all_poses" / source_path.name
    published_path.parent.mkdir(parents=True, exist_ok=True)
    published_path.write_bytes(source_path.read_bytes())
    source_path.unlink()
    stale_attempt_path = (
        tmp_path
        / "smoke/.incomplete/shard.attempt-dead/arms/s50_ema/poses/all_poses"
        / source_path.name
    )
    _rewrite_csv(
        csv_path,
        lambda first: first.__setitem__("all_poses_sdf", str(stale_attempt_path)),
    )

    with pytest.raises(ValueError, match="declared output file does not exist"):
        _build_smoke_report(tmp_path)


def test_report_rejects_all_pose_sdf_record_count_mismatch(tmp_path: Path) -> None:
    paths = _write_smoke_fixture(tmp_path)
    csv_path = paths["s50_ema"]
    row = _first_csv_row(csv_path)
    pose_path = _write_all_poses_sdf(tmp_path, row, record_count=NUM_SAMPLES - 1)
    actual_sha256 = gate_report._file_sha256(pose_path)
    _rewrite_csv(
        csv_path,
        lambda first: first.__setitem__("all_poses_sdf_sha256", actual_sha256),
    )

    with pytest.raises(ValueError, match="expected 4 sequential SDF records, found 3"):
        _build_smoke_report(tmp_path)


@pytest.mark.parametrize(
    ("property_name", "bad_value", "message"),
    [
        ("sample_index", "3", "sample_index: expected ordered index 1"),
        ("sampling_seed", "999", "sampling_seed: expected 43"),
        ("ligand_conformer_seed", "7", "ligand_conformer_seed: expected 0"),
        (
            "candidate_ensemble_sha256",
            _digest("wrong-candidate-ensemble"),
            "candidate_ensemble_sha256: differs from CSV row",
        ),
    ],
)
def test_report_rejects_all_pose_sdf_property_mismatch(
    tmp_path: Path,
    property_name: str,
    bad_value: str,
    message: str,
) -> None:
    paths = _write_smoke_fixture(tmp_path)
    csv_path = paths["s50_ema"]
    row = _first_csv_row(csv_path)
    pose_path = _write_all_poses_sdf(
        tmp_path,
        row,
        property_override=(1, property_name, bad_value),
    )
    actual_sha256 = gate_report._file_sha256(pose_path)
    _rewrite_csv(
        csv_path,
        lambda first: first.__setitem__("all_poses_sdf_sha256", actual_sha256),
    )

    with pytest.raises(ValueError, match=message):
        _build_smoke_report(tmp_path)


def test_report_rejects_recomputed_fast_valid_k2_mismatch(tmp_path: Path) -> None:
    paths = _write_smoke_fixture(tmp_path)
    _rewrite_csv(
        paths["s25_ema"],
        lambda row: row.__setitem__("num_fast_valid_rmsd_lt2_candidates", "0"),
    )
    with pytest.raises(ValueError, match="ordered candidate vectors"):
        gate_report.build_report(
            tmp_path,
            stage="smoke",
            expected_count=4,
            expected_num_samples=NUM_SAMPLES,
            expected_num_steps=NUM_STEPS,
            expected_prior_pool_size=PRIOR_POOL_SIZE,
        )


def test_report_rejects_unpaired_prior_hash(tmp_path: Path) -> None:
    paths = _write_smoke_fixture(tmp_path)
    _rewrite_csv(
        paths["parent50k_plus10k_t0p10_ema"],
        lambda row: row.__setitem__("prior_pool_sha256", _digest("wrong-prior")),
    )
    with pytest.raises(ValueError, match="prior_pool_sha256.*differs"):
        gate_report.build_report(
            tmp_path,
            stage="smoke",
            expected_count=4,
            expected_num_samples=NUM_SAMPLES,
            expected_num_steps=NUM_STEPS,
            expected_prior_pool_size=PRIOR_POOL_SIZE,
        )


def _metric_row(sample_id: str, system_id: str, *, k2: int) -> dict[str, object]:
    return {
        "id": sample_id,
        "system_id": system_id,
        "k2": k2,
        "fv2": k2,
        "fast_valid_count": 100,
        "oracle_rmsd": 1.0,
        "first_rmsd": 2.5,
        "nn": 2.0,
        "c2": 100,
        "unique_count": 100,
    }


def test_cluster_bootstrap_and_selection_gate_are_deterministic() -> None:
    baseline = {
        f"id{index}": _metric_row(f"id{index}", f"sys{index // 2}", k2=2) for index in range(6)
    }
    treatment = {sample_id: {**row, "k2": 4, "fv2": 4} for sample_id, row in baseline.items()}
    first = gate_report._paired_metrics(
        baseline,
        treatment,
        num_samples=100,
        bootstrap_seed=gate_report.BOOTSTRAP_SEED,
        bootstrap_resamples=200,
    )
    second = gate_report._paired_metrics(
        baseline,
        treatment,
        num_samples=100,
        bootstrap_seed=gate_report.BOOTSTRAP_SEED,
        bootstrap_resamples=200,
    )

    assert first["delta_mean_k2"] == 2.0
    assert first["cluster_bootstrap"] == second["cluster_bootstrap"]
    assert first["cluster_bootstrap"]["clusters"] == 3
    assert first["cluster_bootstrap"]["k2_delta"]["ci95_low"] == 2.0
    decision = gate_report._selection_decision(first)
    assert decision["passed"] is True
    assert decision["failed_gates"] == []


def test_cluster_bootstrap_retains_duplicate_system_samples_and_reports_both_weightings() -> None:
    baseline = {
        "a1": _metric_row("a1", "sys_a", k2=1),
        "a2": _metric_row("a2", "sys_a", k2=1),
        "b1": _metric_row("b1", "sys_b", k2=1),
    }
    treatment_k2 = {"a1": 11, "a2": 1, "b1": 0}
    treatment = {
        sample_id: {**row, "k2": treatment_k2[sample_id], "fv2": treatment_k2[sample_id]}
        for sample_id, row in baseline.items()
    }
    seed = 17
    resamples = 25
    metrics = gate_report._paired_metrics(
        baseline,
        treatment,
        num_samples=100,
        bootstrap_seed=seed,
        bootstrap_resamples=resamples,
    )

    # Sample-weighted: (10 + 0 - 1) / 3. System-balanced: ((10 + 0) / 2 - 1) / 2.
    assert metrics["delta_mean_k2"] == 3.0
    assert metrics["system_balanced_sensitivity"]["delta_mean_k2"] == 2.0

    rng = np.random.Generator(np.random.PCG64(seed))
    indices = rng.integers(0, 2, size=(resamples, 2))
    cluster_counts = np.asarray([2.0, 1.0])
    cluster_delta_sums = np.asarray([10.0, -1.0])
    expected_sample_draws = cluster_delta_sums[indices].sum(axis=1) / cluster_counts[
        indices
    ].sum(axis=1)
    expected_system_draws = np.asarray([5.0, -1.0])[indices].mean(axis=1)
    expected_sample_interval = np.percentile(expected_sample_draws, [2.5, 97.5])
    expected_system_interval = np.percentile(expected_system_draws, [2.5, 97.5])
    bootstrap = metrics["cluster_bootstrap"]
    assert bootstrap["clusters"] == 2
    assert bootstrap["k2_delta"]["ci95_low"] == pytest.approx(expected_sample_interval[0])
    assert bootstrap["k2_delta"]["ci95_high"] == pytest.approx(expected_sample_interval[1])
    assert bootstrap["system_balanced_k2_delta"]["ci95_low"] == pytest.approx(
        expected_system_interval[0]
    )
    assert bootstrap["system_balanced_k2_delta"]["ci95_high"] == pytest.approx(
        expected_system_interval[1]
    )


def test_selection_gate_uses_frozen_exact_inclusive_and_strict_thresholds() -> None:
    primary = {
        "delta_mean_k2": 1.0,
        "k2_ge_1_delta_count": 0,
        "fragile_retention_fraction": 0.95,
        "delta_mean_fast_valid_k2": 0.0,
        "fast_valid_k2_ge_1_delta_count": 0,
        "fast_valid_candidate_delta_pp": -1.0,
        "nearest_neighbor_rmsd_ratio": 0.95,
        "c2_ratio": 0.95,
        "coordinate_unique_fraction_treatment": 0.99,
        "coordinate_unique_fraction_delta": -0.005,
        "cluster_bootstrap": {
            "k2_delta": {"ci95_low": np.nextafter(0.0, 1.0)},
            "coverage_delta_pp": {"ci95_low": -1.0},
            "fv_coverage_delta_pp": {"ci95_low": -1.0},
            "nn_ratio": {"ci95_low": 0.90},
            "c2_ratio": {"ci95_low": 0.90},
        },
    }
    decision = gate_report._selection_decision(primary)
    assert decision["passed"] is True
    assert decision["failed_gates"] == []

    primary["cluster_bootstrap"]["k2_delta"]["ci95_low"] = 0.0
    strict_boundary = gate_report._selection_decision(primary)
    assert strict_boundary["passed"] is False
    assert strict_boundary["failed_gates"] == ["efficacy_k2_ci95_low"]


def test_operational_sensitivity_assigns_41_common_failures_zero() -> None:
    arm_metrics = {
        arm: {
            "k2_total": 100 + index,
            "k2_ge_1_count": 80 + index,
            "fast_valid_k2_total": 60 + index,
            "fast_valid_k2_ge_1_count": 40 + index,
        }
        for index, arm in enumerate(gate_report.ARMS)
    }
    primary = {
        "delta_total_k2": 12,
        "k2_ge_1_delta_count": 3,
        "delta_total_fast_valid_k2": 8,
        "fast_valid_k2_ge_1_delta_count": 2,
    }
    sensitivity = gate_report._operational_sensitivity(arm_metrics, primary)

    assert sensitivity["denominator"] == 1_076
    assert sensitivity["evaluable_count"] == 1_035
    assert sensitivity["common_preprocessing_failure_count"] == 41
    assert sensitivity["primary"]["delta_mean_k2"] == pytest.approx(12 / 1_076)
    assert sensitivity["primary"]["k2_ge_1_delta_pp"] == pytest.approx(300 / 1_076)


def test_full_dimensions_are_not_overridable(tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)
    with pytest.raises(ValueError, match="cannot override"):
        gate_report.build_report(tmp_path, stage="full", expected_count=4)


@pytest.mark.parametrize(
    ("overrides"),
    [
        {"bootstrap_seed": gate_report.BOOTSTRAP_SEED + 1},
        {"bootstrap_resamples": gate_report.BOOTSTRAP_RESAMPLES - 1},
    ],
)
def test_full_bootstrap_contract_is_not_overridable(
    tmp_path: Path, overrides: dict[str, int]
) -> None:
    with pytest.raises(ValueError, match="frozen bootstrap seed or resample count"):
        gate_report.build_report(tmp_path, stage="full", **overrides)


def test_write_report_is_atomic_and_no_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "report.json"
    gate_report.write_report({"status": "ok"}, path)
    assert json.loads(path.read_text(encoding="utf-8")) == {"status": "ok"}
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        gate_report.write_report({"status": "different"}, path)
