"""Validation and exact checkpoint resume on CPU and Apple Silicon."""

from copy import deepcopy
from pathlib import Path
from shutil import copyfile

import cuequivariance as cue
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from effdock.checkpoint import load_checkpoint_file
from effdock.models.equivariant import EquivariantDropout
from effdock.training.trainer import Trainer, _require_finite_metrics


class _DropoutHead(nn.Module):
    def __init__(self, dropout: float, **kwargs) -> None:
        super().__init__()
        self.dropout = EquivariantDropout(cue.Irreps("O3", "4x0e + 1x1o"), p=dropout)
        self.projection = nn.Linear(7, 6)

    def forward(self, batch: dict) -> dict:
        output = self.projection(self.dropout(batch["features"]))
        return {"v_pred": output[:, :3], "omega_pred": output[:, 3:]}


def _build_smoke_loaders(trainer: Trainer) -> None:
    sample = {
        "features": torch.arange(7, dtype=torch.float32) / 7,
        "t": torch.tensor([0.5]),
        "frag_batch": torch.tensor(0),
        "frag_sizes": torch.tensor(2),
        "atom_pos_t": torch.zeros(2, 3),
        "v_target": torch.ones(3),
        "omega_target": torch.ones(3),
    }
    trainer.train_sampler = None
    trainer.val_sampler = None
    trainer.train_loader_generator = torch.Generator().manual_seed(19)
    trainer.train_loader = DataLoader(
        [sample] * 12, batch_size=4, generator=trainer.train_loader_generator
    )
    trainer.val_loader = DataLoader([sample] * 4, batch_size=4)


@pytest.mark.parametrize("value", [1e100, float("nan"), float("inf"), -float("inf")])
def test_mps_metric_check_preserves_float64_range_and_rejects_nonfinite(value) -> None:
    # This path accepts Python floats and needs no MPS hardware allocation.
    kwargs = {"context": "MPS validation", "device": torch.device("mps"), "world_size": 1}
    if value == 1e100:
        _require_finite_metrics({"loss": value}, **kwargs)
    else:
        with pytest.raises(FloatingPointError, match="MPS validation"):
            _require_finite_metrics({"loss": value}, **kwargs)


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "mps",
            marks=pytest.mark.skipif(
                not torch.backends.mps.is_available(), reason="MPS is unavailable"
            ),
        ),
    ],
)
@pytest.mark.parametrize("val_every", [0, 1])
def test_training_save_resume_reproduces_next_dropout_step(
    tmp_path: Path, monkeypatch, device: str, val_every: int
) -> None:
    # Keep the real Trainer, loss, optimizer, scheduler, EMA and checkpoint
    # paths; replace only dataset loading and the large docking network.
    monkeypatch.setattr("effdock.training.trainer.EFFDock", _DropoutHead)
    monkeypatch.setattr(Trainer, "_build_dataloaders", _build_smoke_loaders)
    config = {
        "model": {"dropout": 0.4},
        "data": {},
        "training": {
            "device": device,
            "seed": 19,
            "lr": 0.01,
            "batch_size": 4,
            "max_steps": 2,
            "warmup_ratio": 0.0,
            "strict_resume_config": True,
            "fail_on_nonfinite": True,
        },
        "logging": {
            "output_dir": str(tmp_path),
            "val_every": val_every,
            "require_exact_eval_coverage": True,
        },
    }
    original_save = Trainer._save_latest
    resume_path = tmp_path / "resume.pt"

    def save_first_step(trainer, epoch, metrics):
        original_save(trainer, epoch, metrics)
        if trainer.global_step == 1:
            copyfile(trainer.ckpt_dir / "latest.pt", resume_path)

    # A save after step one is needed even when validation is disabled.
    original_step = Trainer._total_steps
    trainer = Trainer(deepcopy(config))
    monkeypatch.setattr(trainer, "_total_steps", lambda: 1)
    monkeypatch.setattr(Trainer, "_save_latest", save_first_step)
    trainer.train()
    first = load_checkpoint_file(resume_path)
    if device == "mps":
        assert "mps" in first["rank_rng_states"][0]
    if val_every:
        assert all(
            torch.isfinite(torch.tensor(first["metrics"][key]))
            for key in ("val_loss", "val_loss_v", "val_loss_omega")
        )

    # Continue in memory, then repeat that step through the public resume path.
    monkeypatch.setattr(trainer, "_total_steps", lambda: original_step(trainer))
    trainer.start_epoch = first["data_pass_epoch"]
    trainer.resume_batch_idx = first["next_batch_idx"]
    trainer._resume_loader_rng_state = first["epoch_loader_rng_state"]
    trainer.train()
    expected = load_checkpoint_file(trainer.ckpt_dir / "latest.pt")
    expected_draw = torch.rand(8, device=device).cpu()

    resumed = Trainer(deepcopy(config))
    resumed.load_checkpoint(str(resume_path))
    resumed.train()
    actual = load_checkpoint_file(resumed.ckpt_dir / "latest.pt")
    actual_draw = torch.rand(8, device=device).cpu()
    assert actual["step"] == expected["step"] == 2
    assert actual["metrics"] == expected["metrics"]
    assert actual["scheduler_state_dicts"] == expected["scheduler_state_dicts"]
    for key in ("model_state_dict", "ema_state_dict"):
        for name, value in expected[key].items():
            assert torch.equal(actual[key][name], value), (key, name)
    for expected_opt, actual_opt in zip(
        expected["optimizer_state_dicts"], actual["optimizer_state_dicts"], strict=True
    ):
        for index, values in expected_opt["state"].items():
            for name, value in values.items():
                assert torch.equal(actual_opt["state"][index][name], value), name
    assert torch.equal(actual_draw, expected_draw)


def test_legacy_checkpoint_rng_without_mps_is_restored() -> None:
    state = Trainer._capture_rng_state()
    state.pop("mps", None)
    Trainer._restore_rng_state(state)


def test_checkpoint_rng_from_mps_can_restore_without_mps(monkeypatch) -> None:
    state = Trainer._capture_rng_state()
    state["mps"] = torch.zeros(16, dtype=torch.uint8)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

    def unexpected_restore(*args):
        pytest.fail("MPS RNG must not be restored on a host without MPS")

    monkeypatch.setattr(torch.mps, "set_rng_state", unexpected_restore)
    Trainer._restore_rng_state(state)
