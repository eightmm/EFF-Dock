from __future__ import annotations

import hashlib
import random

import numpy as np
import pytest
import torch
from torch import nn

from effdock.checkpoint import (
    atomic_torch_save,
    export_ema_inference_checkpoint,
    extract_ema_model_state,
    load_checkpoint_file,
    load_portable_model_state,
)
from effdock.training.trainer import Trainer


class _PortableToy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(2))
        self.graph = nn.Module()
        self.graph.register_buffer("c0", torch.ones(1))


def test_portable_loader_allows_only_runtime_graph_buffers() -> None:
    model = _PortableToy()
    load_portable_model_state(model, {"weight": torch.zeros(2)})
    assert torch.equal(model.weight, torch.zeros(2))

    with pytest.raises(RuntimeError, match="missing=.*weight"):
        load_portable_model_state(model, {})


def test_rng_state_is_safe_loadable_and_restorable(tmp_path) -> None:
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    state = Trainer._capture_rng_state()
    expected = (random.random(), np.random.random(), torch.rand(1))

    path = tmp_path / "rng.pt"
    torch.save({"rng_state": state}, path)
    loaded = load_checkpoint_file(path)
    Trainer._restore_rng_state(loaded["rng_state"])
    actual = (random.random(), np.random.random(), torch.rand(1))

    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    assert torch.equal(actual[2], expected[2])


def test_atomic_torch_save_replaces_complete_artifact(tmp_path) -> None:
    path = tmp_path / "checkpoint.pt"
    atomic_torch_save({"step": 1, "value": torch.tensor([1.0])}, path)
    atomic_torch_save({"step": 2, "value": torch.tensor([2.0])}, path)

    loaded = load_checkpoint_file(path)
    assert loaded["step"] == 2
    assert torch.equal(loaded["value"], torch.tensor([2.0]))
    assert not list(tmp_path.glob(".checkpoint.pt.*.tmp"))


def test_strict_resume_config_reports_changed_sections() -> None:
    current = {"model": {"width": 4}, "training": {"max_steps": 50_000}}
    Trainer._validate_resume_config(current, current.copy())

    with pytest.raises(RuntimeError, match=r"changed top-level sections=\['training'\]"):
        Trainer._validate_resume_config(
            current, {"model": {"width": 4}, "training": {"max_steps": 2_000}}
        )


def test_save_rollout_serializes_updated_best_rmsd(tmp_path, monkeypatch) -> None:
    trainer = Trainer.__new__(Trainer)
    trainer.is_main = True
    trainer.rank = 0
    trainer.world_size = 1
    trainer.ckpt_dir = tmp_path
    trainer.global_step = 500
    trainer._best_rmsd = 5.0
    trainer.rollout_selection_metric = "rollout/rmsd_median"
    trainer.rollout_selection_mode = "min"
    trainer._best_selection_value = 5.0

    def build_state(
        epoch: int, metrics: dict, *, rank_rng_states: list[dict] | None = None
    ) -> dict:
        assert rank_rng_states is not None and len(rank_rng_states) == 1
        return {"epoch": epoch, "metrics": metrics, "best_rmsd": trainer._best_rmsd}

    monkeypatch.setattr(trainer, "_build_checkpoint_state", build_state)
    trainer._save_rollout(0, {"rollout/rmsd_median": 4.25})

    first_rollout = load_checkpoint_file(tmp_path / "rollout_step0000500.pt")
    first_best = load_checkpoint_file(tmp_path / "best.pt")
    assert first_rollout["best_rmsd"] == 4.25
    assert first_best["best_rmsd"] == 4.25

    trainer.global_step = 1000
    trainer._save_rollout(1, {"rollout/rmsd_median": 4.5})
    second_rollout = load_checkpoint_file(tmp_path / "rollout_step0001000.pt")
    retained_best = load_checkpoint_file(tmp_path / "best.pt")
    assert second_rollout["best_rmsd"] == 4.25
    assert retained_best["metrics"]["rollout/rmsd_median"] == 4.25


def test_save_rollout_can_select_best_by_success_rate(tmp_path, monkeypatch) -> None:
    trainer = Trainer.__new__(Trainer)
    trainer.is_main = True
    trainer.rank = 0
    trainer.world_size = 1
    trainer.ckpt_dir = tmp_path
    trainer.global_step = 5_000
    trainer._best_rmsd = float("inf")
    trainer.rollout_selection_metric = "rollout/success_2A"
    trainer.rollout_selection_mode = "max"
    trainer._best_selection_value = -float("inf")

    def build_state(
        epoch: int, metrics: dict, *, rank_rng_states: list[dict] | None = None
    ) -> dict:
        assert rank_rng_states is not None and len(rank_rng_states) == 1
        return {
            "epoch": epoch,
            "metrics": metrics,
            "best_rmsd": trainer._best_rmsd,
            "best_selection_value": trainer._best_selection_value,
        }

    monkeypatch.setattr(trainer, "_build_checkpoint_state", build_state)
    trainer._save_rollout(1, {"rollout/rmsd_median": 4.4, "rollout/success_2A": 0.17})
    trainer.global_step = 10_000
    trainer._save_rollout(2, {"rollout/rmsd_median": 4.2, "rollout/success_2A": 0.16})

    best = load_checkpoint_file(tmp_path / "best.pt")
    latest_named = load_checkpoint_file(tmp_path / "rollout_step0010000.pt")
    assert best["metrics"]["rollout/success_2A"] == 0.17
    assert latest_named["best_selection_value"] == 0.17
    assert latest_named["best_rmsd"] == 4.2


def test_export_ema_inference_checkpoint_is_atomic_and_inference_only(tmp_path) -> None:
    source = tmp_path / "source.pt"
    output = tmp_path / "exported" / "model_ema.pt"
    raw_state = {"weight": torch.ones(2), "graph.c0": torch.ones(1)}
    ema_state = {
        "n_averaged": torch.tensor(12),
        "module.weight": torch.full((2,), 3.0),
        "module.graph.c0": torch.full((1,), 4.0),
    }
    torch.save(
        {
            "format_version": 1,
            "step": 17,
            "epoch": 2,
            "model_state_dict": raw_state,
            "ema_state_dict": ema_state,
            "optimizer_state_dicts": [{"state": {}}],
            "scheduler_state_dicts": [{"last_epoch": 17}],
            "rng_state": {"torch": torch.random.get_rng_state()},
            "metrics": {"rollout/rmsd_median": 4.0},
            "config": {"model": {"hidden_dim": 8}},
            "best_rmsd": 4.5,
        },
        source,
    )
    source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()

    assert extract_ema_model_state(load_checkpoint_file(source))["weight"].tolist() == [3.0, 3.0]
    assert export_ema_inference_checkpoint(source, output) == output

    exported = load_checkpoint_file(output)
    assert exported["artifact_type"] == "effdock_ema_inference_checkpoint"
    assert exported["inference_only"] is True
    assert exported["weight_source"] == "ema"
    assert exported["source_checkpoint_sha256"] == source_sha256
    assert exported["source_checkpoint_step"] == 17
    assert exported["ema_n_averaged"] == 12
    assert torch.equal(exported["model_state_dict"]["weight"], torch.full((2,), 3.0))
    assert torch.equal(exported["ema_state_dict"]["module.weight"], torch.full((2,), 3.0))
    assert "optimizer_state_dicts" not in exported
    assert "scheduler_state_dicts" not in exported
    assert "rng_state" not in exported
    assert exported["source_best_rmsd"] == 4.5
    assert hashlib.sha256(source.read_bytes()).hexdigest() == source_sha256

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        export_ema_inference_checkpoint(source, output)
    with pytest.raises(ValueError, match="must differ"):
        export_ema_inference_checkpoint(source, source)
