from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/run_plinder_checkpoint_paired_validation.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "run_plinder_checkpoint_paired_validation", _SCRIPT
)
assert _SPEC is not None and _SPEC.loader is not None
driver = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = driver
_SPEC.loader.exec_module(driver)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_global_seed_is_based_on_full_split_before_eligibility_and_sharding() -> None:
    full = ["sys-d__L", "sys-b__L", "sys-a__L", "sys-c__L"]
    eligible = ["sys-d__L", "sys-a__L", "sys-c__L"]
    shard_zero = driver.plan_assignments(
        full, eligible, num_shards=2, shard_index=0
    )
    shard_one = driver.plan_assignments(
        full, eligible, num_shards=2, shard_index=1
    )

    assert shard_zero.full_keys == ("sys-a__L", "sys-b__L", "sys-c__L", "sys-d__L")
    assert [
        (item.sample_key, item.global_index, item.sampling_seed)
        for item in shard_zero.assigned
    ] == [("sys-a__L", 1, 43), ("sys-d__L", 4, 46)]
    assert [
        (item.sample_key, item.global_index, item.sampling_seed)
        for item in shard_one.assigned
    ] == [("sys-c__L", 3, 45)]

    smoke = driver.plan_assignments(
        full, eligible, num_shards=1, shard_index=0, smoke_count=2
    )
    assert [item.sample_key for item in smoke.assigned] == ["sys-a__L", "sys-c__L"]
    assert [item.sampling_seed for item in smoke.assigned] == [43, 45]


def test_frozen_audit_defines_exact_cohort_not_a_fresh_mapping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    full = ["sys-a__L", "sys-b__L", "sys-c__L"]
    audit = tmp_path / "audit.json"
    records = [
        {
            "sample_key": "sys-a__L",
            "system_id": "sys-a",
            "status": "ok",
            "mapping_method": "strict_stereo",
            "symmetry_complete": True,
        },
        {
            "sample_key": "sys-b__L",
            "system_id": "sys-b",
            "status": "failed",
            "failure_code": "generate_smiles_conformer:AssertionError",
        },
        {
            "sample_key": "sys-c__L",
            "system_id": "sys-c",
            "status": "ok",
            "mapping_method": "strict_stereo",
            "symmetry_complete": True,
        },
    ]
    audit.write_text(
        json.dumps(
            {
                "protocol_id": "EFFDOCK-RDKIT-FRAGMENT-GEOMETRY-AUDIT-V2",
                "config": {
                    "requested_complexes": 3,
                    "rdkit_seed": 0,
                    "hydrogen_policy": "remove_all_hs",
                },
                "inputs": {
                    "split_file_sha256": "split",
                    "pool_parquet_sha256": "pool",
                },
                "records": records,
            }
        )
    )
    eligible = ["sys-a__L", "sys-c__L"]
    monkeypatch.setattr(driver, "EXPECTED_CONFORMER_AUDIT_SHA256", _sha(audit))
    monkeypatch.setattr(driver, "EXPECTED_SPLIT_SHA256", "split")
    monkeypatch.setattr(driver, "EXPECTED_POOL_SHA256", "pool")
    monkeypatch.setattr(driver, "EXPECTED_VAL_COUNT", 3)
    monkeypatch.setattr(driver, "EXPECTED_ELIGIBLE_COUNT", 2)
    monkeypatch.setattr(driver, "EXPECTED_ELIGIBLE_SYSTEM_COUNT", 2)
    monkeypatch.setattr(
        driver, "EXPECTED_ELIGIBLE_NEWLINE_SHA256", driver._newline_id_sha256(eligible)
    )

    by_id, actual = driver.load_frozen_conformer_audit(audit, full)

    assert actual == eligible
    assert by_id["sys-b__L"]["status"] == "failed"


def test_preflight_record_uses_smiles_and_retains_audit_failure(
    tmp_path: Path,
) -> None:
    raw_root = tmp_path / "raw"
    processed_root = tmp_path / "processed"
    key = "sys-a__L"
    paths = driver.input_paths_for_sample(
        key, raw_root=raw_root, processed_root=processed_root
    )
    paths.receptor.parent.mkdir(parents=True)
    paths.ligand_reference.parent.mkdir(parents=True)
    paths.receptor.write_text("ATOM\n")
    paths.ligand_reference.write_text("reference only\n")
    paths.processed_meta.parent.mkdir(parents=True)
    torch.save(
        {
            "pdb_id": key,
            "plinder_system_id": "sys-a",
            "plinder_ligand_chain": "L",
            "pocket_center": torch.tensor([1.0, 2.0, 3.0]),
        },
        paths.processed_meta,
    )
    smiles = "CCO"
    raw_asset = {
        "receptor": driver._asset_identity(paths.receptor),
        "ligand": driver._asset_identity(paths.ligand_reference),
    }
    audit_record = {
        "status": "failed",
        "system_id": "sys-a",
        "ligand_instance_chain": "L",
        "failure_stage": "generate_smiles_conformer",
        "failure_code": "generate_smiles_conformer:AssertionError",
        "error_type": "AssertionError",
        "error": "embedding failed",
        "input_sha256": {
            "raw_ligand_sdf": _sha(paths.ligand_reference),
            "processed_meta_pt": _sha(paths.processed_meta),
        },
    }

    record = driver._preflight_one(
        {
            "sample_key": key,
            "global_index": 7,
            "smiles": smiles,
            "raw_root": str(raw_root),
            "processed_root": str(processed_root),
            "raw_asset": raw_asset,
            "audit_record": audit_record,
            "audit_eligible": False,
        }
    )

    assert record["status"] == "excluded"
    assert record["canonical_smiles"] == smiles
    assert record["ligand_reference"]["path"] == str(paths.ligand_reference.resolve())
    assert record["audit_failure_code"] == "generate_smiles_conformer:AssertionError"
    assert record["sampling_seed"] == 49


def _fake_evaluator_row(item, kwargs: dict[str, object]) -> dict[str, object]:
    num_samples = int(kwargs["num_samples"])
    pose_dir = Path(kwargs["pose_dir"])
    all_poses = pose_dir / "all_poses" / f"{item.complex_id}.sdf"
    selected = pose_dir / "selected" / f"{item.complex_id}.sdf"
    all_poses.parent.mkdir(parents=True, exist_ok=True)
    selected.parent.mkdir(parents=True, exist_ok=True)
    all_poses.write_text("all poses\n")
    selected.write_text("selected\n")
    rmsds = [1.0] + [3.0] * (num_samples - 1)
    fast = [True] * num_samples
    return {
        "id": item.complex_id,
        "selector_profile": "candidate_only",
        "num_samples": num_samples,
        "sampling_seed": kwargs["seed"],
        "ligand_conformer_seed": kwargs["ligand_conformer_seed"],
        "prior_pool_size": kwargs["prior_pool_size"],
        "prior_pool_sha256": f"prior-{kwargs['seed']}-{kwargs['prior_pool_size']}",
        "candidate_ensemble_sha256": "candidate",
        "guidance_mode": "none",
        "sampling_dynamics": "deterministic_ode",
        "full_heavy_atom_bijection": True,
        "ligand_input_identity_sha256": "smiles",
        "selected_index": 0,
        "candidate_rmsds_json": json.dumps(rmsds),
        "candidate_rmsd_method_json": json.dumps(
            ["rdkit_calc_rms_symmetry_no_align"] * num_samples
        ),
        "num_mapped_index_rmsd_fallback_candidates": 0,
        "candidate_fast_valid_json": json.dumps(fast),
        "num_fast_valid_candidates": num_samples,
        "num_rmsd_lt2_candidates": 1,
        "num_fast_valid_rmsd_lt2_candidates": 1,
        "all_poses_count": num_samples,
        "all_poses_sdf": str(all_poses),
        "all_poses_sdf_sha256": _sha(all_poses),
        "saved_pose_sha256_json": json.dumps({"selected": _sha(selected)}),
        "first_rmsd": 1.0,
        "first_fast_valid": True,
        "oracle_rmsd": 1.0,
        "oracle_fast_valid": True,
        "coordinate_unique_count": num_samples,
        "pairwise_heavy_atom_rmsd_mean": 2.0,
        "pairwise_heavy_atom_rmsd_median": 2.0,
        "pairwise_heavy_atom_rmsd_ge2_fraction": 1.0,
        "nearest_neighbor_heavy_atom_rmsd_median": 2.0,
        "c2_connected_component_count": num_samples,
        "diversity_heavy_atom_count": 3,
    }


def test_evaluate_prepared_freezes_smoke_sampling_and_disables_experimental_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protein = tmp_path / "protein.pdb"
    reference = tmp_path / "reference.sdf"
    meta = tmp_path / "meta.pt"
    protein.write_text("ATOM\n")
    reference.write_text("reference\n")
    meta.write_bytes(b"meta")
    assignment = driver.Assignment("sys-a__L", 1, 43)
    paths = driver.InputPaths("sys-a__L", "sys-a", "L", protein, reference, meta)
    item = driver.ComplexInput("sys-a__L", protein, reference, "sdf", "CCO", (0, 0, 0))
    prepared = driver.PreparedInput(
        assignment,
        paths,
        item,
        {
            "processed_meta_sha256": _sha(meta),
            "canonical_smiles_identity_sha256": "smiles",
        },
    )
    calls: list[dict[str, object]] = []

    def fake_evaluate_one(_model, ligand_item, **kwargs):
        calls.append(kwargs)
        return _fake_evaluator_row(ligand_item, kwargs)

    monkeypatch.setattr(driver, "evaluate_one", fake_evaluate_one)
    row = driver.evaluate_prepared(
        prepared,
        model=torch.nn.Identity(),
        cfg={"data": {}},
        device=torch.device("cpu"),
        pose_dir=tmp_path / "poses",
        stage="smoke",
    )

    assert row["num_rmsd_lt2_candidates"] == 1
    assert len(calls) == 1
    call = calls[0]
    assert call["num_samples"] == 4
    assert call["num_steps"] == 2
    assert call["prior_pool_size"] == 4
    assert call["sigma"] == 2.0
    assert call["selector_profile"] == "candidate_only"
    assert call["confidence_model"] is None
    assert call["ligand_conformer_seed"] == 0
    assert call["fk_constraint_beta"] == 0.0
    assert call["translation_sde_base_sigma"] == 0.0
    assert call["refine"] == "none"


def _all_pose_row(source: Path, *, sample_id: str = "sys-a__L") -> dict[str, object]:
    return {
        "id": sample_id,
        "all_poses_sdf": str(source),
        "all_poses_sdf_sha256": _sha(source),
    }


def test_all_pose_csv_path_is_rebased_to_final_visible_tree_without_mutating_row(
    tmp_path: Path,
) -> None:
    attempt = tmp_path / ".incomplete" / "shard.attempt-abc"
    final = tmp_path / "shard-000-of-001"
    source = attempt / "arms" / "s50_ema" / "poses/all_poses/sys-a__L.sdf"
    source.parent.mkdir(parents=True)
    source.write_text("poses\n")
    row = _all_pose_row(source)

    serialized = driver._canonicalize_all_pose_paths_for_csv(
        [row],
        arm="s50_ema",
        attempt_dir=attempt,
        visible_root=final,
    )

    expected = final.resolve() / source.relative_to(attempt)
    assert serialized[0]["all_poses_sdf"] == str(expected)
    assert row["all_poses_sdf"] == str(source)
    assert serialized[0]["all_poses_sdf_sha256"] == _sha(source)


def test_all_pose_csv_path_stays_in_attempt_tree_for_failed_attempt(
    tmp_path: Path,
) -> None:
    attempt = tmp_path / ".incomplete" / "shard.attempt-abc"
    source = attempt / "arms" / "s50_ema" / "poses/all_poses/sys-a__L.sdf"
    source.parent.mkdir(parents=True)
    source.write_text("poses\n")

    serialized = driver._canonicalize_all_pose_paths_for_csv(
        [_all_pose_row(source)],
        arm="s50_ema",
        attempt_dir=attempt,
        visible_root=attempt,
    )

    assert serialized[0]["all_poses_sdf"] == str(source.resolve())


def test_all_pose_csv_path_rejects_missing_source(tmp_path: Path) -> None:
    attempt = tmp_path / ".incomplete" / "shard.attempt-abc"
    source = attempt / "arms" / "s50_ema" / "poses/all_poses/sys-a__L.sdf"
    row = {
        "id": "sys-a__L",
        "all_poses_sdf": str(source),
        "all_poses_sdf_sha256": "0" * 64,
    }

    with pytest.raises(FileNotFoundError, match="missing source all-pose SDF"):
        driver._canonicalize_all_pose_paths_for_csv(
            [row],
            arm="s50_ema",
            attempt_dir=attempt,
            visible_root=tmp_path / "final",
        )


def test_all_pose_csv_path_rejects_hash_mismatch(tmp_path: Path) -> None:
    attempt = tmp_path / ".incomplete" / "shard.attempt-abc"
    source = attempt / "arms" / "s50_ema" / "poses/all_poses/sys-a__L.sdf"
    source.parent.mkdir(parents=True)
    source.write_text("poses\n")
    row = _all_pose_row(source)
    row["all_poses_sdf_sha256"] = "0" * 64

    with pytest.raises(RuntimeError, match="source all-pose SDF SHA-256 mismatch"):
        driver._canonicalize_all_pose_paths_for_csv(
            [row],
            arm="s50_ema",
            attempt_dir=attempt,
            visible_root=tmp_path / "final",
        )


def test_all_pose_csv_path_rejects_noncanonical_recorded_path(tmp_path: Path) -> None:
    attempt = tmp_path / ".incomplete" / "shard.attempt-abc"
    expected = attempt / "arms" / "s50_ema" / "poses/all_poses/sys-a__L.sdf"
    recorded = attempt / "arms" / "s50_ema" / "poses/all_poses/other.sdf"
    expected.parent.mkdir(parents=True)
    expected.write_text("expected\n")
    recorded.write_text("recorded\n")
    row = _all_pose_row(recorded)
    row["id"] = "sys-a__L"

    with pytest.raises(ValueError, match="outside its canonical attempt path"):
        driver._canonicalize_all_pose_paths_for_csv(
            [row],
            arm="s50_ema",
            attempt_dir=attempt,
            visible_root=tmp_path / "final",
        )


def test_full_shard_runs_three_arms_sequentially_and_refuses_reuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    full = [f"sys-{index:02d}__L" for index in range(8)]
    manifest = {
        "inventory": {
            "full_ids": full,
            "eligible_ids": full,
            "eligible_ids_sha256": driver.sorted_id_sha256(full),
            "eligible_ids_newline_sha256": driver._newline_id_sha256(full),
            "eligible_system_count": 8,
            "excluded_count": 0,
            "excluded_ids": [],
            "excluded_ids_sha256": driver.sorted_id_sha256([]),
        },
        "inputs": {"fixed_identities": {"fixture": True}},
        "seed_contract": {"name": "fixture"},
        "ligand_input_contract": {"source": "canonical SMILES"},
        "records": [
            {"sample_key": key, "status": "eligible"} for key in full
        ],
    }
    manifest_path = tmp_path / "eligible.json"
    manifest_path.write_text(json.dumps(manifest))
    fake_arms = tuple(
        driver.ArmSpec(name, tmp_path / f"{name}.pt", name, step, step)
        for name, step in (("s25_ema", 25), ("s50_ema", 50), ("plus10k", 10))
    )
    for arm in fake_arms:
        arm.checkpoint.write_bytes(b"checkpoint")
    monkeypatch.setattr(driver, "ARMS", fake_arms)
    monkeypatch.setattr(driver, "validate_eligibility_manifest", lambda *_a, **_k: manifest)
    monkeypatch.setattr(driver, "resolve_runtime_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(
        driver,
        "prepare_input",
        lambda assignment, _record, **_kwargs: driver.PreparedInput(
            assignment,
            driver.InputPaths(
                assignment.sample_key,
                assignment.sample_key.rsplit("__", 1)[0],
                "L",
                tmp_path / "protein",
                tmp_path / "reference",
                tmp_path / "meta",
            ),
            driver.ComplexInput(
                assignment.sample_key,
                tmp_path / "protein",
                tmp_path / "reference",
                "sdf",
                "CCO",
                (0, 0, 0),
            ),
            {},
        ),
    )
    load_order: list[str] = []

    def fake_load_model(_config, checkpoint, _device):
        arm = next(arm for arm in fake_arms if arm.checkpoint == checkpoint)
        load_order.append(arm.name)
        return torch.nn.Identity(), {"data": {}}, {
            "artifact_type": "effdock_ema_inference_checkpoint",
            "inference_only": True,
            "weight_source": "ema",
            "step": arm.step,
            "source_checkpoint_step": arm.source_checkpoint_step,
        }

    monkeypatch.setattr(driver, "load_model", fake_load_model)

    def fake_evaluate_prepared(prepared, *, pose_dir, stage, **_kwargs):
        arm = pose_dir.parent.name
        all_poses = pose_dir / "all_poses" / f"{prepared.assignment.sample_key}.sdf"
        all_poses.parent.mkdir(parents=True, exist_ok=True)
        all_poses.write_text(f"{arm} {prepared.assignment.sample_key}\n")
        return {
            "id": prepared.assignment.sample_key,
            "arm": arm,
            "sampling_seed": prepared.assignment.sampling_seed,
            "ligand_conformer_seed": 0,
            "prior_pool_sha256": f"prior-{prepared.assignment.sampling_seed}",
            "protein_sha256": "protein",
            "ligand_reference_sha256": "reference",
            "ligand_input_identity_sha256": "smiles",
            "num_rmsd_lt2_candidates": 1,
            "num_fast_valid_rmsd_lt2_candidates": 1,
            "first_rmsd": 1.0,
            "first_fast_valid": True,
            "oracle_rmsd": 1.0,
            "oracle_fast_valid": True,
            "nearest_neighbor_heavy_atom_rmsd_median": 2.0,
            "c2_connected_component_count": 100,
            "coordinate_unique_count": 100,
            "all_poses_count": 100,
            "all_poses_sdf": str(all_poses),
            "all_poses_sdf_sha256": _sha(all_poses),
        }

    monkeypatch.setattr(driver, "evaluate_prepared", fake_evaluate_prepared)
    args = argparse.Namespace(
        stage="full",
        raw_root=tmp_path,
        processed_root=tmp_path,
        output_root=tmp_path / "outputs",
        eligibility_manifest=manifest_path,
        eligibility_manifest_sha256=_sha(manifest_path),
        run_id="fixture-run",
        num_shards=8,
        shard_index=0,
    )

    summary = driver.execute_shard(args)

    assert summary["status"] == "complete"
    assert summary["mode"] == "full"
    assert summary["replay_integrity_gate"] == {"required": False, "passed": True}
    assert load_order == [arm.name for arm in fake_arms]
    final = tmp_path / "outputs/fixture-run/full/shard-000-of-008"
    assert (final / "paired_summary.json").is_file()
    assert all((final / "arms" / arm.name / "results.csv").is_file() for arm in fake_arms)
    for arm in fake_arms:
        results_path = final / "arms" / arm.name / "results.csv"
        with results_path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 1
        expected_sdf = (
            final
            / "arms"
            / arm.name
            / "poses/all_poses"
            / f"{rows[0]['id']}.sdf"
        ).resolve()
        assert rows[0]["all_poses_sdf"] == str(expected_sdf)
        assert ".incomplete" not in rows[0]["all_poses_sdf"]
        assert expected_sdf.is_file()
        assert rows[0]["all_poses_sdf_sha256"] == _sha(expected_sdf)
    with pytest.raises(FileExistsError, match="refusing to rerun or overwrite"):
        driver.execute_shard(args)


def test_pilot_replay_gate_enforces_frozen_thresholds() -> None:
    def row(k2: int, fast_k2: int, value: float = 1.0) -> dict[str, object]:
        return {
            "num_rmsd_lt2_candidates": k2,
            "num_fast_valid_rmsd_lt2_candidates": fast_k2,
            "nearest_neighbor_heavy_atom_rmsd_median": value,
            "c2_connected_component_count": value,
            "coordinate_unique_count": value,
        }

    rows = {
        "s50_ema": [row(1, 1), row(0, 0)],
        "s50_ema_replay": [row(1, 1), row(0, 0)],
    }
    assert driver._replay_integrity_gate(rows, stage="pilot")["passed"] is True
    rows["s50_ema_replay"][0] = row(0, 0)
    assert driver._replay_integrity_gate(rows, stage="pilot")["passed"] is False
