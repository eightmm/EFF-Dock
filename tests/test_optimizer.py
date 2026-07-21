from __future__ import annotations

import torch
from torch import nn

from effdock.training.trainer import configure_optimizers


class _ToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(8, 4)
        self.linear = nn.Linear(4, 4)
        self.norm = nn.LayerNorm(4)


def _parameter_ids(optimizer) -> set[int]:
    return {id(param) for group in optimizer.param_groups for param in group["params"]}


def test_adamw_is_the_default_and_owns_every_parameter() -> None:
    model = _ToyModel()
    optimizers = configure_optimizers(model)
    assert len(optimizers) == 1
    assert isinstance(optimizers[0], torch.optim.AdamW)
    assert _parameter_ids(optimizers[0]) == {id(param) for param in model.parameters()}


def test_muon_is_opt_in_and_only_owns_linear_weights() -> None:
    model = _ToyModel()
    optimizers = configure_optimizers(model, use_muon=True)
    assert [type(opt).__name__ for opt in optimizers] == ["Muon", "AdamW"]
    assert _parameter_ids(optimizers[0]) == {id(model.linear.weight)}
    assert _parameter_ids(optimizers[0]).isdisjoint(_parameter_ids(optimizers[1]))
    assert _parameter_ids(optimizers[0]) | _parameter_ids(optimizers[1]) == {
        id(param) for param in model.parameters()
    }
