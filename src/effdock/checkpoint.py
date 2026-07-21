"""Safe, backend-portable checkpoint helpers for EFF-Dock."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn


def load_checkpoint_file(path: str | Path) -> dict[str, Any]:
    """Load a tensor/basic-container checkpoint on CPU with the safe unpickler."""
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"checkpoint must contain a mapping, got {type(checkpoint).__name__}")
    return checkpoint


def is_runtime_graph_buffer(key: str) -> bool:
    """Return whether a state key is a derived cuEquivariance graph constant."""
    return ".graph.c" in f".{key}"


def load_portable_model_state(model: nn.Module, state: dict[str, Any]) -> None:
    """Load learned state while tolerating only backend-derived graph buffers."""
    incompatible = model.load_state_dict(state, strict=False)
    bad_missing = [key for key in incompatible.missing_keys if not is_runtime_graph_buffer(key)]
    bad_unexpected = [
        key for key in incompatible.unexpected_keys if not is_runtime_graph_buffer(key)
    ]
    if bad_missing or bad_unexpected:
        raise RuntimeError(
            "checkpoint is incompatible with the configured EFF-Dock model: "
            f"missing={bad_missing}, unexpected={bad_unexpected}"
        )
