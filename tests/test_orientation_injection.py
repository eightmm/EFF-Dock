"""Operator covariance and checkpoint semantics, with nonzero learned mixing weights."""

from __future__ import annotations

import json

import pytest
import torch
import yaml
from torch import nn

from effdock.checkpoint import checkpoint_orientation_injection
from effdock.geometry.se3 import quaternion_to_matrix
from effdock.inference.docking import build_arg_parser, load_model, options_from_args
from effdock.models.effdock import mix_fragment_orientation
from effdock.training.trainer import Trainer
from effdock.workflows.benchmark_report import aggregate_dataset


def test_nonzero_orientation_mix_covariance_and_legacy_counterexample():
    rng = torch.Generator().manual_seed(37)
    R = quaternion_to_matrix(torch.randn(5, 4, generator=rng, dtype=torch.float64))
    Q = quaternion_to_matrix(torch.randn(4, generator=rng, dtype=torch.float64))
    W = torch.randn(7, 3, generator=rng, dtype=torch.float64)
    h = mix_fragment_orientation(R, W, "rw")
    torch.testing.assert_close(
        mix_fragment_orientation(Q @ R, W, "rw"), h @ Q.T, atol=1e-12, rtol=1e-12
    )
    legacy = mix_fragment_orientation(R, W, "legacy_rt_w")
    assert torch.equal(legacy, torch.einsum("nki,ck->nci", R, W))
    assert (mix_fragment_orientation(Q @ R, W, "legacy_rt_w") - legacy @ Q.T).abs().max() > 0.1


def test_orientation_mix_direction_and_gradients():
    R = torch.tensor([[[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]], dtype=torch.float64)
    W = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
    torch.testing.assert_close(
        mix_fragment_orientation(R, W, "rw"), torch.tensor([[[0.0, 1.0, 0.0]]], dtype=torch.float64)
    )
    torch.testing.assert_close(
        mix_fragment_orientation(R, W, "legacy_rt_w"),
        torch.tensor([[[0.0, -1.0, 0.0]]], dtype=torch.float64),
    )
    assert torch.autograd.gradcheck(
        lambda r, w: mix_fragment_orientation(r, w, "rw"), (R.requires_grad_(), W.requires_grad_())
    )
    with pytest.raises(ValueError, match="orientation_injection"):
        mix_fragment_orientation(R, W, "typo")


@pytest.mark.parametrize(
    "payload, expected",
    [
        ({}, "legacy_rt_w"),
        ({"config": {"model": {}}}, "legacy_rt_w"),
        ({"config": {"model": {"orientation_injection": "rw"}}}, "rw"),
    ],
)
def test_checkpoint_operator_metadata(payload, expected):
    assert checkpoint_orientation_injection(payload) == expected


@pytest.mark.parametrize(
    "payload",
    [
        {"config": None},
        {"config": {"model": None}},
        {"config": {"model": {"orientation_injection": "unknown"}}},
    ],
)
def test_malformed_checkpoint_operator_fails(payload):
    with pytest.raises(ValueError):
        checkpoint_orientation_injection(payload)


@pytest.mark.parametrize(
    "saved, override, expected",
    [
        (None, None, "legacy_rt_w"),
        (None, "rw", "rw"),
        ("rw", None, "rw"),
        ("rw", "legacy_rt_w", "legacy_rt_w"),
    ],
)
def test_loader_uses_checkpoint_or_explicit_override_not_yaml(
    tmp_path, monkeypatch, saved, override, expected
):
    import effdock.models.effdock as module

    class Toy(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.orientation_injection = kwargs["orientation_injection"]
            self.weight = nn.Parameter(torch.zeros(2))

    monkeypatch.setattr(module, "EFFDock", Toy)
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump({"model": {"model_type": "unified", "orientation_injection": "rw"}})
    )
    path = tmp_path / "weights.pt"
    payload = {"model_state_dict": {"weight": torch.ones(2)}}
    if saved is not None:
        payload["config"] = {"model": {"orientation_injection": saved}}
    torch.save(payload, path)
    model, cfg, ckpt = load_model(config, path, torch.device("cpu"), orientation_injection=override)
    assert model.orientation_injection == cfg["model"]["orientation_injection"] == expected
    assert torch.equal(model.weight, torch.ones(2)) and not model.training
    assert checkpoint_orientation_injection(ckpt) == (saved or "legacy_rt_w")
    assert yaml.safe_load(config.read_text())["model"]["orientation_injection"] == "rw"


def test_resume_rejects_operator_change_even_without_strict_config(tmp_path):
    checkpoint = tmp_path / "old.pt"
    torch.save({"config": {"model": {}}}, checkpoint)
    trainer = Trainer.__new__(Trainer)
    trainer.cfg = {
        "model": {"orientation_injection": "rw"},
        "training": {"strict_resume_config": False},
    }
    with pytest.raises(RuntimeError, match="resume orientation_injection differs"):
        trainer.load_checkpoint(str(checkpoint))


def test_strict_resume_accepts_explicit_legacy_metadata_only():
    saved = {"model": {"hidden_dim": 8}, "training": {"seed": 42}}
    current = {
        "model": {"hidden_dim": 8, "orientation_injection": "legacy_rt_w"},
        "training": {"seed": 42},
    }
    Trainer._validate_resume_config(current, saved)
    assert "orientation_injection" not in saved["model"]
    current["model"]["orientation_injection"] = "rw"
    with pytest.raises(RuntimeError, match="resume checkpoint config differs"):
        Trainer._validate_resume_config(current, saved)


def test_dock_cli_operator_override_is_explicit():
    parser = build_arg_parser()
    args = ["--protein", "receptor.pdb", "--ligand", "CC", "--pocket-center", "0,0,0"]
    assert options_from_args(parser.parse_args(args)).orientation_injection is None
    assert (
        options_from_args(
            parser.parse_args(args + ["--orientation-injection", "rw"])
        ).orientation_injection
        == "rw"
    )


def test_shard_aggregation_rejects_mixed_operators_before_reading_rows(tmp_path):
    # A missing field denotes historical output, not an unknown wildcard.
    (tmp_path / "run0.summary.json").write_text(json.dumps({}))
    (tmp_path / "run1.summary.json").write_text(json.dumps({"orientation_injection": "rw"}))
    with pytest.raises(ValueError, match="inconsistent run shards:.*orientation_injection"):
        aggregate_dataset(tmp_path, "run", 2)


@pytest.mark.parametrize("stored, expected", [("rw", "legacy_rt_w"), (None, "rw")])
def test_confidence_dataset_rejects_cross_operator_banks(tmp_path, stored, expected):
    from effdock.confidence.dataset import LigandPoseConfidenceDataset

    split = tmp_path / "split.json"
    split.write_text(json.dumps({"train": ["case"]}))
    path = tmp_path / "bank.pt"
    torch.save({} if stored is None else {"docking_orientation_injection": stored}, path)
    ds = LigandPoseConfidenceDataset(
        split_file=split,
        split="train",
        processed_dir=tmp_path,
        shard_paths={"case": path},
        docking_orientation_injection=expected,
    )
    with pytest.raises(ValueError, match="confidence shard docking orientation mismatch"):
        ds[0]


def test_confidence_runtime_discloses_cross_operator_features(monkeypatch):
    import effdock.confidence.runtime as runtime

    docking = nn.Module()
    docking.orientation_injection = "rw"
    confidence = nn.Module()
    confidence.docking_orientation_injection = "legacy_rt_w"

    def stop(*args, **kwargs):
        raise RuntimeError("feature extraction reached")

    monkeypatch.setattr(runtime, "extract_t1_ligand_irreps", stop)
    with pytest.warns(UserWarning, match="cross-operator frozen-weight"):
        with pytest.raises(RuntimeError, match="feature extraction reached"):
            runtime.score_poses_with_confidence(
                confidence,
                docking,
                {},
                {},
                {},
                [torch.zeros(1, 3)],
                sigma=2.0,
                device=torch.device("cpu"),
            )


def test_weights_only_migration_records_source_operator(tmp_path):
    source = nn.Linear(2, 1)
    path = tmp_path / "old.pt"
    torch.save({"model_state_dict": source.state_dict(), "step": 12}, path)
    trainer = Trainer.__new__(Trainer)
    trainer.model = nn.Linear(2, 1)
    trainer.ema_model = None
    trainer.is_main = False
    trainer.load_model_weights(str(path))
    assert trainer.initialization_provenance["orientation_injection"] == "legacy_rt_w"
    assert len(trainer.initialization_provenance["checkpoint_sha256"]) == 64
    assert trainer.initialization_provenance["step"] == 12
    assert trainer.global_step == 0
    torch.testing.assert_close(trainer.model.weight, source.weight)
