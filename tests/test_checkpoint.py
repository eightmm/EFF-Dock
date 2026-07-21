from __future__ import annotations

import random

import numpy as np
import pytest
import torch
from torch import nn

from effdock.checkpoint import load_checkpoint_file, load_portable_model_state
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
