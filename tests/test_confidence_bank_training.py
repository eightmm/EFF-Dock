from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from effdock.confidence.dataset import (
    LigandPoseConfidenceDataset,
    PairedLigandPoseConfidenceDataset,
)
from effdock.confidence.losses import pose_confidence_loss
from effdock.preprocess.graph_types import NTYPE_FRAGMENT, NTYPE_LIG_ATOM, NTYPE_PROT_ATOM
from effdock.training.trainer import configure_optimizers
from effdock.workflows.train_confidence import (
    BANK_SETTINGS,
    REFINED_BANK_SETTINGS,
    _ordered_ids_sha256,
    _release_eval_cuda_cache,
    _restore_training_states,
    _ResumableDistributedSampler,
    _resume_eval_state,
    _stable_pose_order,
    _summarize_eval_target,
    _write_eval_ledger,
    evaluate,
    validate_bank_manifest,
)

POSE_TAG = "s50-test"


def test_release_eval_cuda_cache_is_cuda_only(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        "effdock.workflows.train_confidence.gc.collect",
        lambda: calls.append("gc"),
    )
    monkeypatch.setattr(
        "effdock.workflows.train_confidence.torch.cuda.empty_cache",
        lambda: calls.append("empty_cache"),
    )

    _release_eval_cuda_cache(torch.device("cpu"))
    assert calls == []

    _release_eval_cuda_cache(torch.device("cuda"))
    assert calls == ["gc", "empty_cache"]


def _saved_graph() -> dict[str, torch.Tensor]:
    return {
        "node_coords": torch.tensor(
            [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.5, 0.0, 0.0]]
        ),
        "node_type": torch.tensor([NTYPE_LIG_ATOM, NTYPE_LIG_ATOM, NTYPE_FRAGMENT]),
        "edge_index": torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        "lig_atom_slice": torch.tensor([0, 2], dtype=torch.long),
        "lig_frag_slice": torch.tensor([2, 3], dtype=torch.long),
    }


def _shard(*, bad_atom_disp: bool = False) -> dict[str, object]:
    graph = _saved_graph()
    return {
        "system_id": "system-1",
        "pocket_center_used": torch.zeros(3),
        "pose_atom_coords": torch.zeros(3, 2, 3),
        "h_lig_node": torch.zeros(3, 3, 8),
        "lig_node_type": torch.tensor(
            [NTYPE_LIG_ATOM, NTYPE_LIG_ATOM, NTYPE_FRAGMENT]
        ),
        "fragment_id": torch.tensor([0, 0]),
        "frag_sizes": torch.tensor([2]),
        "atom_disp": torch.zeros(3, 1 if bad_atom_disp else 2),
        "pose_rmsd": torch.tensor([3.0, 2.0, 1.0]),
        "pose_rmsd_symmetry_no_align": torch.tensor([2.5, 1.5, 0.5]),
        "graph_centered": graph,
        "graph": graph,
    }


def _dataset_fixture(tmp_path: Path, *, bad_atom_disp: bool = False) -> tuple[Path, Path]:
    split_path = tmp_path / "split.json"
    split_path.write_text(json.dumps({"train": ["sample"], "val": []}) + "\n")
    shard_path = (
        tmp_path
        / "nested"
        / "sample"
        / "confidence_poses"
        / f"confposes_{POSE_TAG}.pt"
    )
    shard_path.parent.mkdir(parents=True)
    torch.save(_shard(bad_atom_disp=bad_atom_disp), shard_path)
    return split_path, shard_path


def test_dataset_uses_sealed_centered_graph_and_manifest_shard_path(tmp_path: Path) -> None:
    split_path, shard_path = _dataset_fixture(tmp_path)
    dataset = LigandPoseConfidenceDataset(
        split_file=split_path,
        split="train",
        processed_dir=tmp_path / "does-not-contain-processed-tensors",
        pose_tag=POSE_TAG,
        eval_target_key="pose_rmsd_symmetry_no_align",
        shard_paths={"sample": shard_path},
        system_ids={"sample": "system-1"},
    )

    item = dataset[0]

    assert torch.equal(item["graph"]["node_coords"], _saved_graph()["node_coords"])
    assert torch.equal(
        item["pose_rmsd_symmetry_no_align"], torch.tensor([2.5, 1.5, 0.5])
    )
    assert item["shard_path"] == str(shard_path)
    assert item["system_id"] == "system-1"


def test_dataset_rejects_inconsistent_pose_atom_shape(tmp_path: Path) -> None:
    split_path, shard_path = _dataset_fixture(tmp_path, bad_atom_disp=True)
    dataset = LigandPoseConfidenceDataset(
        split_file=split_path,
        split="train",
        processed_dir=tmp_path,
        pose_tag=POSE_TAG,
        shard_paths={"sample": shard_path},
    )

    with pytest.raises(ValueError, match="atom_disp shape"):
        dataset[0]


def test_paired_dataset_balances_raw_refined_and_adds_one_crystal_anchor(
    tmp_path: Path,
) -> None:
    primary_root = tmp_path / "primary"
    auxiliary_root = tmp_path / "auxiliary"
    primary_root.mkdir()
    auxiliary_root.mkdir()
    primary_split, primary_path = _dataset_fixture(primary_root)
    auxiliary_split, auxiliary_path = _dataset_fixture(auxiliary_root)
    primary_shard = torch.load(primary_path, map_location="cpu", weights_only=False)
    primary_shard.update(
        {
            "crystal_anchor_pose_atom_coords": torch.zeros(1, 2, 3),
            "crystal_anchor_h_lig_node": torch.zeros(1, 3, 8),
            "crystal_anchor_atom_disp": torch.zeros(1, 2),
            "crystal_anchor_pose_rmsd": torch.zeros(1),
        }
    )
    torch.save(primary_shard, primary_path)
    auxiliary_shard = torch.load(auxiliary_path, map_location="cpu", weights_only=False)
    auxiliary_shard["pose_atom_coords"] = torch.ones(3, 2, 3)
    torch.save(auxiliary_shard, auxiliary_path)

    primary = LigandPoseConfidenceDataset(
        split_file=primary_split,
        split="train",
        processed_dir=primary_root,
        pose_tag=POSE_TAG,
        shard_paths={"sample": primary_path},
        system_ids={"sample": "system-1"},
        max_poses_per_complex=2,
        pose_sample_strategy="stratified",
    )
    auxiliary = LigandPoseConfidenceDataset(
        split_file=auxiliary_split,
        split="train",
        processed_dir=auxiliary_root,
        pose_tag=POSE_TAG,
        shard_paths={"sample": auxiliary_path},
        system_ids={"sample": "system-1"},
        max_poses_per_complex=2,
        pose_sample_strategy="stratified",
    )

    item = PairedLigandPoseConfidenceDataset(primary, auxiliary)[0]

    assert item["pose_atom_coords"].shape == (5, 2, 3)
    assert sorted(item["pose_bank_component"].tolist()) == [0, 0, 1, 1, 2]
    crystal_index = int((item["pose_bank_component"] == 2).nonzero().item())
    assert float(item["pose_rmsd"][crystal_index]) == 0.0
    assert bool((item["atom_disp"][crystal_index] == 0).all())

    pose_count, atom_count = item["atom_disp"].shape
    out = {
        "atom_disp_log1p": torch.zeros(
            pose_count, atom_count, requires_grad=True
        ),
        "atom_ok_logit": torch.zeros(pose_count, atom_count, requires_grad=True),
        "pose_rmsd_log1p": torch.zeros(pose_count, requires_grad=True),
        "pose_success_logit": torch.zeros(pose_count, requires_grad=True),
    }
    losses = pose_confidence_loss(out, item, success_listwise_weight=1.0)
    assert bool(torch.isfinite(losses["loss"]))
    losses["loss"].backward()
    assert all(value.grad is not None for value in out.values())


def test_paired_dataset_rejects_cross_bank_graph_drift(tmp_path: Path) -> None:
    primary_root = tmp_path / "primary"
    auxiliary_root = tmp_path / "auxiliary"
    primary_root.mkdir()
    auxiliary_root.mkdir()
    primary_split, primary_path = _dataset_fixture(primary_root)
    auxiliary_split, auxiliary_path = _dataset_fixture(auxiliary_root)
    auxiliary_shard = torch.load(auxiliary_path, map_location="cpu", weights_only=False)
    auxiliary_shard["graph_centered"]["node_coords"][0, 0] += 0.1
    torch.save(auxiliary_shard, auxiliary_path)
    primary = LigandPoseConfidenceDataset(
        split_file=primary_split,
        split="train",
        processed_dir=primary_root,
        pose_tag=POSE_TAG,
        shard_paths={"sample": primary_path},
    )
    auxiliary = LigandPoseConfidenceDataset(
        split_file=auxiliary_split,
        split="train",
        processed_dir=auxiliary_root,
        pose_tag=POSE_TAG,
        shard_paths={"sample": auxiliary_path},
    )

    with pytest.raises(ValueError, match="graph field 'node_coords' differs"):
        PairedLigandPoseConfidenceDataset(primary, auxiliary)[0]


def test_dataset_caps_train_poses_by_saved_graph_node_product(tmp_path: Path) -> None:
    split_path, shard_path = _dataset_fixture(tmp_path)
    dataset = LigandPoseConfidenceDataset(
        split_file=split_path,
        split="train",
        processed_dir=tmp_path,
        pose_tag=POSE_TAG,
        shard_paths={"sample": shard_path},
        max_poses_per_complex=3,
        max_pose_node_product=6,
    )

    item = dataset[0]

    assert item["pose_atom_coords"].shape[0] == 2
    assert item["pose_rmsd"].shape == (2,)
    assert 1.0 in item["pose_rmsd"].tolist()


def test_dataset_applies_stronger_cap_to_large_saved_graph(tmp_path: Path) -> None:
    split_path, shard_path = _dataset_fixture(tmp_path)
    dataset = LigandPoseConfidenceDataset(
        split_file=split_path,
        split="train",
        processed_dir=tmp_path,
        pose_tag=POSE_TAG,
        shard_paths={"sample": shard_path},
        max_poses_per_complex=3,
        max_pose_node_product=100,
        large_graph_node_threshold=2,
        large_graph_max_poses=1,
    )

    item = dataset[0]

    assert item["pose_atom_coords"].shape[0] == 1
    assert item["pose_rmsd"].tolist() == [1.0]


def _symmetry_digest(sample_key: str, labels: torch.Tensor) -> str:
    values = labels.contiguous().to(torch.float32)
    return hashlib.sha256(
        b"EFFDOCK_SYMMETRY_RMSD_LABEL_V1\0"
        + sample_key.encode()
        + b"\0"
        + values.numpy().tobytes(order="C")
    ).hexdigest()


def test_dataset_uses_external_symmetry_target_for_training_only(tmp_path: Path) -> None:
    split_path, shard_path = _dataset_fixture(tmp_path)
    source = _shard()
    source["pose_atom_coords"] = torch.zeros(100, 2, 3)
    source["h_lig_node"] = torch.zeros(100, 3, 8)
    source["atom_disp"] = torch.full((100, 2), 7.0)
    source["pose_rmsd"] = torch.full((100,), 9.0)
    source["pose_rmsd_symmetry_no_align"] = torch.full((100,), 8.0)
    torch.save(source, shard_path)
    labels = torch.linspace(0.25, 6.0, 100)
    label_sha = _symmetry_digest("sample", labels)
    sidecar = {
        "schema_version": "EFFDOCK_S50_SYMMETRY_RMSD_SIDECAR_V1",
        "status": "complete",
        "method": "rdkit_calc_rms_symmetry_no_align",
        "split": "train",
        "bank_manifest_sha256": "c" * 64,
        "input_manifest_sha256": "d" * 64,
        "sample_keys": ["sample"],
        "system_ids": ["system-1"],
        "split_indices": [0],
        "source_pt_sha256": ["a" * 64],
        "pose_ensemble_sha256": ["b" * 64],
        "label_sha256": [label_sha],
        "pose_rmsd_symmetry_no_align": labels.unsqueeze(0),
    }
    sidecar_path = tmp_path / "sidecar.pt"
    torch.save(sidecar, sidecar_path)
    record = {
        "sample_key": "sample",
        "system_id": "system-1",
        "split_index": 0,
        "source_pt_sha256": "a" * 64,
        "pose_ensemble_sha256": "b" * 64,
        "label_sha256": label_sha,
        "row_index": 0,
        "sidecar_path": str(sidecar_path),
        "sidecar_sha256": hashlib.sha256(sidecar_path.read_bytes()).hexdigest(),
        "bank_manifest_sha256": "c" * 64,
        "input_manifest_sha256": "d" * 64,
    }
    dataset = LigandPoseConfidenceDataset(
        split_file=split_path,
        split="train",
        processed_dir=tmp_path,
        pose_tag=POSE_TAG,
        pose_target_key="pose_rmsd_symmetry_no_align",
        eval_target_key="pose_rmsd_symmetry_no_align",
        external_pose_targets={"sample": record},
        shard_paths={"sample": shard_path},
        system_ids={"sample": "system-1"},
        max_poses_per_complex=100,
    )

    item = dataset[0]

    assert torch.equal(item["pose_rmsd"], labels)
    assert torch.equal(item["pose_rmsd_symmetry_no_align"], labels)
    assert torch.equal(item["atom_disp"], torch.full((100, 2), 7.0))


def _manifest_fixture(tmp_path: Path, *, status: str = "complete") -> tuple[Path, Path, Path]:
    split_ids = {"train": ["train-id"], "val": ["val-id"]}
    split_path = tmp_path / "filtered_split.json"
    split_path.write_text(json.dumps(split_ids, indent=2, sort_keys=True) + "\n")
    records = []
    for split, ids in split_ids.items():
        for index, pid in enumerate(ids):
            pt_path = (
                tmp_path
                / "shards"
                / split
                / "shard-0000"
                / pid
                / "confidence_poses"
                / f"confposes_{POSE_TAG}.pt"
            )
            pt_path.parent.mkdir(parents=True)
            pt_path.write_bytes(f"{split}-{pid}".encode())
            records.append(
                {
                    "sample_key": pid,
                    "split": split,
                    "split_index": index,
                    "global_index": index,
                    "sampling_seed": index,
                    "system_id": f"system-{split}",
                    "pt_path": str(pt_path.resolve()),
                    "pt_sha256": hashlib.sha256(pt_path.read_bytes()).hexdigest(),
                    "size_bytes": pt_path.stat().st_size,
                    "pose_count": 100,
                }
            )
    inventory = {}
    empty_sha = hashlib.sha256(b"").hexdigest()
    for split, ids in split_ids.items():
        eligible_ids = ids if status == "complete" else [*ids, f"unused-{split}-id"]
        inventory[split] = {
            "full_count": len(eligible_ids),
            "eligible_count": len(eligible_ids),
            "excluded_count": 0,
            "record_count": len(ids),
            "full_ids_sha256": _ordered_ids_sha256(eligible_ids),
            "eligible_ids_sha256": _ordered_ids_sha256(eligible_ids),
            "excluded_ids_sha256": empty_sha,
            "record_ids_sha256": _ordered_ids_sha256(ids),
        }
    manifest = {
        "schema_version": "effdock.s50_confidence_bank.manifest.v1",
        "protocol_id": "EFFDOCK-S50-CONFIDENCE-TRAINING-BANK-V1",
        "status": status,
        "claim_eligible": status == "complete",
        "pose_tag": POSE_TAG,
        "settings": BANK_SETTINGS,
        "filtered_split_path": str(split_path.resolve()),
        "filtered_split_sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(),
        "inventory": inventory,
        "records": records,
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest_path, split_path, records[0]["pt_path"]


def test_bank_manifest_seals_filtered_inventory_and_nested_paths(tmp_path: Path) -> None:
    manifest_path, split_path, train_pt = _manifest_fixture(tmp_path)
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()

    provenance = validate_bank_manifest(
        manifest_path,
        expected_sha256=manifest_sha,
        split_file=split_path,
        pose_tag=POSE_TAG,
    )

    assert provenance["sha256"] == manifest_sha
    assert provenance["claim_eligible"] is True
    assert provenance["_shard_paths"]["train"]["train-id"] == train_pt


def test_bank_manifest_accepts_frozen_refined_pose_contract(tmp_path: Path) -> None:
    manifest_path, split_path, _ = _manifest_fixture(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    manifest["settings"] = REFINED_BANK_SETTINGS
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    provenance = validate_bank_manifest(
        manifest_path,
        expected_sha256=None,
        split_file=split_path,
        pose_tag=POSE_TAG,
    )

    assert provenance["protocol_id"] == "EFFDOCK-S50-CONFIDENCE-TRAINING-BANK-V1"


def test_bank_manifest_rejects_smoke_for_full_training(tmp_path: Path) -> None:
    manifest_path, split_path, _ = _manifest_fixture(tmp_path, status="smoke_complete")

    with pytest.raises(ValueError, match="status must be 'complete'"):
        validate_bank_manifest(
            manifest_path,
            expected_sha256=None,
            split_file=split_path,
            pose_tag=POSE_TAG,
        )

    provenance = validate_bank_manifest(
        manifest_path,
        expected_sha256=None,
        split_file=split_path,
        pose_tag=POSE_TAG,
        allow_smoke=True,
    )
    assert provenance["claim_eligible"] is False


def test_bank_manifest_requires_exact_frozen_pose_count(tmp_path: Path) -> None:
    manifest_path, split_path, _ = _manifest_fixture(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    manifest["records"][0]["pose_count"] = BANK_SETTINGS["num_samples"] - 1
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match="pose_count 99 != frozen sampler count 100"):
        validate_bank_manifest(
            manifest_path,
            expected_sha256=None,
            split_file=split_path,
            pose_tag=POSE_TAG,
        )


def test_eval_target_metrics_use_stable_head_order_and_k2_slices() -> None:
    assert _stable_pose_order(torch.tensor([1.0, 0.5, 0.5, 2.0])) == [1, 2, 0, 3]

    metrics = _summarize_eval_target(
        selected=[3.0, 1.0, 2.5, 0.5],
        top5_selected=[3.0, 0.8, 1.5, 0.4],
        oracle=[3.0, 0.5, 0.2, 0.1],
        oracle_k2=[0, 3, 7, 12],
    )

    assert metrics["eval_top1_lt2"] == pytest.approx(50.0)
    assert metrics["eval_top5_lt2"] == pytest.approx(75.0)
    for name in ("0", "1_4", "5_9", "ge10"):
        assert metrics[f"eval_oracle_k2_{name}_n"] == 1.0


def test_resumable_sampler_applies_offset_once() -> None:
    sampler = _ResumableDistributedSampler(
        list(range(10)),
        num_replicas=1,
        rank=0,
        shuffle=True,
        seed=43,
    )
    sampler.set_epoch(2)
    full_order = list(iter(sampler))
    sampler.set_epoch(2)
    sampler.set_start_index(3)

    assert list(iter(sampler)) == full_order[3:]
    assert list(iter(sampler)) == full_order


def test_configure_optimizers_applies_requested_weight_decay_to_muon() -> None:
    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.LayerNorm(4))

    optimizers = configure_optimizers(
        model,
        lr=3.0e-5,
        muon_lr=2.0e-3,
        weight_decay=0.01,
        use_muon=True,
    )

    assert {type(optimizer).__name__ for optimizer in optimizers} == {"Muon", "AdamW"}
    for optimizer in optimizers:
        assert {group["weight_decay"] for group in optimizer.param_groups} == {0.01}


def test_resume_state_restore_rejects_partial_optimizer_inventory() -> None:
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.ones(()))])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    checkpoint = {
        "optimizer_state_dicts": [],
        "scheduler_state_dicts": [scheduler.state_dict()],
    }

    with pytest.raises(ValueError, match="optimizer state count mismatch"):
        _restore_training_states(
            checkpoint,
            optimizers=[optimizer],
            schedulers=[scheduler],
        )


def test_evaluate_ledger_record_carries_system_id_and_stable_top1() -> None:
    class FakeModel:
        def eval(self) -> None:
            return None

        def forward_complex(self, item: dict[str, object]) -> dict[str, torch.Tensor]:
            del item
            return {
                "pose_rmsd": torch.tensor([0.5, 0.5, 1.0]),
                "pose_rmsd_log1p": torch.log1p(torch.tensor([0.5, 0.5, 1.0])),
                "pose_success_logit": torch.tensor([0.0, 1.0, -1.0]),
                "atom_disp_log1p": torch.zeros(3, 2),
                "atom_ok_logit": torch.zeros(3, 2),
            }

    item = {
        "pid": "sample",
        "system_id": "system-cluster",
        "pose_atom_coords": torch.stack(
            [torch.zeros(2, 3), torch.ones(2, 3), torch.full((2, 3), 2.0)]
        ),
        "pose_rmsd": torch.tensor([3.0, 1.0, 4.0]),
        "pose_rmsd_symmetry_no_align": torch.tensor([2.5, 0.5, 3.5]),
        "graph": {
            "node_type": torch.tensor([NTYPE_PROT_ATOM]),
            "node_coords": torch.tensor([[10.0, 10.0, 10.0]]),
        },
    }
    records: list[dict[str, object]] = []

    metrics = evaluate(
        FakeModel(),  # type: ignore[arg-type]
        [[item]],  # type: ignore[arg-type]
        torch.device("cpu"),
        max_complexes=None,
        eval_target_key="pose_rmsd_symmetry_no_align",
        eval_records=records,
    )

    assert metrics["eval_top1_lt2"] == 0.0
    assert records[0]["system_id"] == "system-cluster"
    assert records[0]["top1_index"] == 0


def test_eval_ledger_is_idempotent_but_never_overwritten(tmp_path: Path) -> None:
    path = tmp_path / "eval_u000000.json"
    records = [{"pid": "x", "oracle_k2": 1}]
    first = _write_eval_ledger(
        path,
        step=0,
        eval_target_key="pose_rmsd_symmetry_no_align",
        records=records,
        bank_provenance={"sha256": "a" * 64},
    )
    second = _write_eval_ledger(
        path,
        step=0,
        eval_target_key="pose_rmsd_symmetry_no_align",
        records=records,
        bank_provenance={"sha256": "a" * 64},
    )
    assert first == second

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        _write_eval_ledger(
            path,
            step=1,
            eval_target_key="pose_rmsd_symmetry_no_align",
            records=records,
            bank_provenance={"sha256": "a" * 64},
        )


def test_scheduled_resume_repairs_only_the_uncommitted_evaluation_phase(
    tmp_path: Path,
) -> None:
    checkpoint: dict[str, object] = {"step": 5000, "args": {"eval_on_start": True}}
    provenance = {"sha256": "a" * 64}

    state, ledger, metrics = _resume_eval_state(
        checkpoint,
        out_dir=tmp_path,
        step=5000,
        eval_every=5000,
        eval_target_key="pose_rmsd_symmetry_no_align",
        bank_provenance=provenance,
    )
    assert (state, ledger, metrics) == ("evaluate_missing", None, None)

    records = [
        {
            "pid": "x",
            "top1_rmsd": 1.0,
            "top5_best_rmsd": 0.8,
            "oracle_rmsd": 0.5,
            "oracle_k2": 3,
        }
    ]
    ledger = _write_eval_ledger(
        tmp_path / "eval_u005000.json",
        step=5000,
        eval_target_key="pose_rmsd_symmetry_no_align",
        records=records,
        bank_provenance=provenance,
    )
    state, observed_ledger, metrics = _resume_eval_state(
        checkpoint,
        out_dir=tmp_path,
        step=5000,
        eval_every=5000,
        eval_target_key="pose_rmsd_symmetry_no_align",
        bank_provenance=provenance,
    )
    assert state == "commit_existing"
    assert observed_ledger == ledger
    assert metrics is not None and metrics["eval_top1_lt2"] == 100.0

    checkpoint["evaluation_ledger"] = ledger
    state, observed_ledger, _ = _resume_eval_state(
        checkpoint,
        out_dir=tmp_path,
        step=5000,
        eval_every=5000,
        eval_target_key="pose_rmsd_symmetry_no_align",
        bank_provenance=provenance,
    )
    assert state == "committed"
    assert observed_ledger == ledger

    (tmp_path / "eval_u005000.json").unlink()
    with pytest.raises(FileNotFoundError, match="commits a missing"):
        _resume_eval_state(
            checkpoint,
            out_dir=tmp_path,
            step=5000,
            eval_every=5000,
            eval_target_key="pose_rmsd_symmetry_no_align",
            bank_provenance=provenance,
        )
