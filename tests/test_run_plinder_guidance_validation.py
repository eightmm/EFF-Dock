from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_plinder_guidance_validation.py"
_SPEC = importlib.util.spec_from_file_location("run_plinder_guidance_validation", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
driver = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = driver
_SPEC.loader.exec_module(driver)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_parse_sample_key_uses_final_delimiter() -> None:
    key = "1a5s__1__1.A__1.C__1.C"
    assert driver.parse_sample_key(key) == ("1a5s__1__1.A__1.C", "1.C")

    with pytest.raises(ValueError, match="invalid PLINDER sample key"):
        driver.parse_sample_key("not-a-sample-key")
    with pytest.raises(ValueError, match="unsafe ligand_chain"):
        driver.parse_sample_key("system__../escape")


def test_seed_and_shard_assignment_is_global_and_invariant() -> None:
    keys = ["sys-c__L", "sys-a__L", "sys-b__L", "sys-d__L"]
    shard_zero = driver.plan_assignments(keys, num_shards=2, shard_index=0)
    shard_one = driver.plan_assignments(keys, num_shards=2, shard_index=1)
    repeat = driver.plan_assignments(list(reversed(keys)), num_shards=2, shard_index=0)

    assert shard_zero.full_keys == ("sys-a__L", "sys-b__L", "sys-c__L", "sys-d__L")
    assert [
        (item.sample_key, item.global_index, item.sampling_seed) for item in shard_zero.assigned
    ] == [
        ("sys-a__L", 1, 43),
        ("sys-c__L", 3, 45),
    ]
    assert [
        (item.sample_key, item.global_index, item.sampling_seed) for item in shard_one.assigned
    ] == [
        ("sys-b__L", 2, 44),
        ("sys-d__L", 4, 46),
    ]
    assert repeat == shard_zero

    smoke = driver.plan_assignments(keys, num_shards=1, shard_index=0, smoke_count=2)
    only = driver.plan_assignments(keys, num_shards=1, shard_index=0, only_ids=["sys-c__L"])
    assert smoke.assigned[1].sampling_seed == 44
    assert only.assigned[0].sampling_seed == 45


@pytest.mark.parametrize("value", ["0", "0.5", "1", "1.5", "2", 0.5])
def test_validate_eta_accepts_only_frozen_values(value: str | float) -> None:
    assert driver.validate_eta(value) in driver.ETA_VALUES


@pytest.mark.parametrize("value", ["-0.5", "0.25", "2.5", "nan", "inf", True])
def test_validate_eta_rejects_non_protocol_values(value: object) -> None:
    with pytest.raises(ValueError, match="eta must be one of"):
        driver.validate_eta(value)  # type: ignore[arg-type]


def test_raw_path_mapping_uses_system_and_ligand_chain(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    processed_root = tmp_path / "processed"
    paths = driver.input_paths_for_sample(
        "1a5s__1__1.A__1.C__1.C",
        raw_root=raw_root,
        processed_root=processed_root,
    )
    system_root = raw_root / "systems" / "1a5s__1__1.A__1.C"
    assert paths.receptor == system_root / "receptor.pdb"
    assert paths.ligand == system_root / "ligand_files" / "1.C.sdf"
    assert paths.meta == processed_root / "1a5s__1__1.A__1.C__1.C" / "meta.pt"


def test_failed_attempt_is_retained_and_does_not_block_retry(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs"
    first = driver.reserve_shard_directory(
        output_root,
        run_id="retry-run",
        eta=1.0,
        num_shards=4,
        shard_index=2,
    )
    assert not first.final_dir.exists()
    (first.attempt_dir / "partial-pose.sdf").write_text("interrupted\n")
    failed_summary = {
        "status": "failed",
        "failures": [{"id": "case", "stage": "evaluate_one"}],
        "inventory": {"failure_count": 1},
    }
    driver._publish_artifacts(
        first.attempt_dir,
        first.attempt_dir,
        [],
        failed_summary,
    )

    retry = driver.reserve_shard_directory(
        output_root,
        run_id="retry-run",
        eta=1.0,
        num_shards=4,
        shard_index=2,
    )
    assert retry.attempt_dir != first.attempt_dir
    assert retry.final_dir == first.final_dir
    assert first.attempt_dir.is_dir()
    completed_summary = {
        "status": "complete",
        "failures": [],
        "inventory": {"failure_count": 0},
    }
    published = driver.publish_complete_attempt(retry, [], completed_summary)

    assert published == retry.final_dir
    assert published.is_dir()
    assert not retry.attempt_dir.exists()
    assert first.attempt_dir.is_dir()
    assert completed_summary["artifacts"]["summary"] == str(published / "summary.json")
    with pytest.raises(FileExistsError, match="refusing to rerun or overwrite"):
        driver.reserve_shard_directory(
            output_root,
            run_id="retry-run",
            eta=1.0,
            num_shards=4,
            shard_index=2,
        )


def _write_input_fixture(raw_root: Path, processed_root: Path, key: str) -> None:
    system_id, ligand_chain = driver.parse_sample_key(key)
    system_root = raw_root / "systems" / system_id
    (system_root / "ligand_files").mkdir(parents=True)
    (system_root / "receptor.pdb").write_text("ATOM\n")
    (system_root / "ligand_files" / f"{ligand_chain}.sdf").write_text("fake sdf\n")
    meta_root = processed_root / key
    meta_root.mkdir(parents=True)
    torch.save(
        {
            "pdb_id": key,
            "plinder_system_id": system_id,
            "plinder_ligand_chain": ligand_chain,
            "pocket_center": torch.tensor([1.0, 2.0, 3.0]),
        },
        meta_root / "meta.pt",
    )


def test_atomic_shard_output_refuses_overwrite_with_mocked_evaluation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    keys = ["1abc__1__1.A__1.C__1.C", "2def__1__1.A__1.D__1.D"]
    split_file = tmp_path / "plinder.json"
    split_file.write_text(json.dumps({"train": [], "val": list(reversed(keys))}))
    raw_root = tmp_path / "raw"
    processed_root = tmp_path / "processed"
    for key in keys:
        _write_input_fixture(raw_root, processed_root, key)

    config = tmp_path / "train.yaml"
    docking = tmp_path / "dock.pt"
    confidence = tmp_path / "confidence.pt"
    protocol = tmp_path / "protocol.md"
    config.write_text("model: {}\ndata: {}\n")
    docking.write_bytes(b"docking")
    confidence.write_bytes(b"confidence")
    protocol.write_text("frozen\n")

    monkeypatch.setattr(driver, "SPLIT_FILE", split_file)
    monkeypatch.setattr(driver, "EXPECTED_SPLIT_SHA256", _sha(split_file))
    monkeypatch.setattr(driver, "EXPECTED_VAL_COUNT", len(keys))
    monkeypatch.setattr(driver, "CONFIG", config)
    monkeypatch.setattr(driver, "DOCKING_CHECKPOINT", docking)
    monkeypatch.setattr(driver, "CONFIDENCE_CHECKPOINT", confidence)
    monkeypatch.setattr(driver, "PROTOCOL_DOCUMENT", protocol)
    monkeypatch.setattr(driver, "EXPECTED_CONFIG_SHA256", _sha(config))
    monkeypatch.setattr(driver, "EXPECTED_DOCKING_SHA256", _sha(docking))
    monkeypatch.setattr(driver, "EXPECTED_CONFIDENCE_SHA256", _sha(confidence))
    monkeypatch.setattr(driver, "EXPECTED_GUIDANCE_PARAMETER_SHA256", "parameters")
    monkeypatch.setattr(driver, "EXPECTED_GUIDANCE_IMPLEMENTATION_SHA256", "implementation")
    monkeypatch.setattr(driver, "guidance_parameter_identity", lambda: {"sha256": "parameters"})
    monkeypatch.setattr(
        driver, "guidance_implementation_identity", lambda: {"sha256": "implementation"}
    )
    monkeypatch.setattr(driver, "resolve_runtime_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(
        driver,
        "load_model",
        lambda *_args: (torch.nn.Identity(), {"data": {}}, {"step": 100_000}),
    )
    monkeypatch.setattr(
        driver,
        "load_pose_confidence_model",
        lambda *_args: (torch.nn.Identity(), {"step": 42_500}),
    )

    calls: list[dict[str, object]] = []

    def fake_evaluate_one(_model, item, **kwargs):
        calls.append(kwargs)
        pose_path = kwargs["pose_dir"] / "confidence" / f"{item.complex_id}.sdf"
        pose_path.parent.mkdir(parents=True, exist_ok=True)
        pose_path.write_text(f"pose {item.complex_id}\n")
        pose_hash = _sha(pose_path)
        seed = kwargs["seed"]
        return {
            "id": item.complex_id,
            "protein": str(item.protein),
            "ligand_ref": str(item.ligand_ref),
            "protein_sha256": _sha(item.protein),
            "ligand_reference_sha256": _sha(item.ligand_ref),
            "saved_pose_sha256_json": json.dumps({"confidence": pose_hash}),
            "sampling_seed": seed,
            "prior_pool_size": kwargs["prior_pool_size"],
            "prior_pool_sha256": f"prior-{seed}",
            "first_rmsd": 3.0,
            "first_fast_valid": True,
            "confidence_index": 1,
            "confidence_rmsd": 1.0,
            "confidence_pred_rmsd": 0.9,
            "confidence_fast_valid": True,
            "oracle_rmsd": 0.5,
            "oracle_fast_valid": True,
        }

    monkeypatch.setattr(driver, "evaluate_one", fake_evaluate_one)
    output_root = tmp_path / "outputs"
    args = argparse.Namespace(
        raw_root=raw_root,
        processed_root=processed_root,
        output_root=output_root,
        run_id="fixture-run",
        eta=0.5,
        num_shards=1,
        shard_index=0,
        only_id=[],
        smoke=None,
    )

    summary = driver.execute(args)
    shard_dir = output_root / "fixture-run" / "eta0500" / "shard-000-of-001"
    assert summary["status"] == "complete"
    assert (shard_dir / "results.csv").is_file()
    assert (shard_dir / "summary.json").is_file()
    assert summary["artifacts"]["csv"] == str(shard_dir / "results.csv")
    assert summary["artifacts"]["summary"] == str(shard_dir / "summary.json")
    assert summary["artifacts"]["primary_pose_dir"] == str(shard_dir / "poses/confidence")
    assert len(calls) == len(keys)
    assert all(call["num_samples"] == 100 for call in calls)
    assert all(call["num_steps"] == 10 for call in calls)
    assert all(call["sigma"] == 0.5 for call in calls)
    assert all(call["prior_pool_size"] == 100 for call in calls)
    assert all(call["unified_guidance_mode"] == "normalized_drift" for call in calls)
    assert all(call["unified_guidance_scale"] == 0.5 for call in calls)
    assert all(call["selector_profile"] == "confidence_cluster_free" for call in calls)
    assert [call["seed"] for call in calls] == [43, 44]

    with pytest.raises(FileExistsError, match="refusing to rerun or overwrite"):
        driver.execute(args)
    assert len(calls) == len(keys)
