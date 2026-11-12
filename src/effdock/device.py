"""Device selection helpers shared by inference, training and evaluation."""

from __future__ import annotations

import torch
from torch import nn


def resolve_device(device: str | torch.device | None = None) -> torch.device:
    """Resolve an explicit device or prefer CUDA, then MPS, then CPU."""
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def move_module_to_device(module: nn.Module, device: str | torch.device) -> nn.Module:
    """Move a model, converting floating buffers to float32 for MPS support."""
    target = torch.device(device)
    if target.type == "mps":
        return module.to(device=target, dtype=torch.float32)
    return module.to(target)
