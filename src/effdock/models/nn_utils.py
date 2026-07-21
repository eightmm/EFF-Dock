"""Shared utility layers for the unified model."""

from __future__ import annotations

import math

import torch
from torch import Tensor


def rbf_encode(
    dist: Tensor,
    n_rbf: int = 16,
    d_min: float = 0.0,
    d_max: float = 16.0,
) -> Tensor:
    """Radial basis function encoding of distances.

    Args:
        dist: Tensor of shape ``[E]`` or ``[E, 1]``.
        n_rbf: Number of RBF centers.
        d_min: Minimum distance.
        d_max: Maximum distance.

    Returns:
        Tensor of shape ``[E, n_rbf]``.
    """
    dist = dist.view(-1, 1)
    centers = torch.linspace(d_min, d_max, n_rbf, device=dist.device, dtype=dist.dtype)
    width = (d_max - d_min) / n_rbf
    return torch.exp(-((dist - centers) ** 2) / (2 * width**2))


def sinusoidal_embedding(t: Tensor, dim: int = 32) -> Tensor:
    """Sinusoidal time embedding.

    Args:
        t: Scalar time values ``[B]`` or ``[B, 1]``.
        dim: Embedding dimension (must be even).

    Returns:
        ``[B, dim]``.
    """
    t = t.view(-1)
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, device=t.device, dtype=t.dtype) / (half - 1)
    )
    args = t.unsqueeze(-1) * freqs.unsqueeze(0)
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


def scatter_mean(src: Tensor, idx: Tensor, n: int) -> Tensor:
    """Scatter-mean aggregation over per-element segment ids.

    Args:
        src: Source features ``[E, D]``.
        idx: Target segment index per row ``[E]``.
        n: Number of target segments.

    Returns:
        ``[n, D]`` mean over segments (zero for empty segments).
    """
    out = torch.zeros(n, src.shape[-1], device=src.device, dtype=src.dtype)
    count = torch.zeros(n, 1, device=src.device, dtype=src.dtype)
    out.scatter_add_(0, idx.unsqueeze(-1).expand_as(src), src)
    count.scatter_add_(
        0,
        idx.unsqueeze(-1),
        torch.ones(idx.shape[0], 1, device=src.device, dtype=src.dtype),
    )
    return out / count.clamp_min(1.0)


def segment_mean_csr(x: Tensor, ptr: Tensor) -> Tensor:
    """Mean over CSR-style segments.  ``x: [N] or [N, D]``, ``ptr: [P+1]`` → ``[P]`` or ``[P, D]``."""
    sizes = ptr[1:] - ptr[:-1]
    P = ptr.shape[0] - 1
    seg_id = torch.repeat_interleave(torch.arange(P, device=x.device), sizes)
    if x.ndim == 1:
        out = torch.zeros(P, device=x.device, dtype=x.dtype)
        out.index_add_(0, seg_id, x)
        return out / sizes.clamp_min(1).to(x.dtype)
    out = torch.zeros(P, x.shape[-1], device=x.device, dtype=x.dtype)
    out.index_add_(0, seg_id, x)
    return out / sizes.clamp_min(1).unsqueeze(-1).to(x.dtype)


def segment_max_csr(x: Tensor, ptr: Tensor) -> Tensor:
    """Max over CSR-style segments.  ``x: [N, D]``, ``ptr: [P+1]`` → ``[P, D]`` (empty segments → 0)."""
    sizes = ptr[1:] - ptr[:-1]
    P = ptr.shape[0] - 1
    seg_id = torch.repeat_interleave(torch.arange(P, device=x.device), sizes)
    out = torch.full((P, x.shape[-1]), float("-inf"), device=x.device, dtype=x.dtype)
    idx = seg_id.unsqueeze(-1).expand_as(x)
    out.scatter_reduce_(0, idx, x, reduce="amax", include_self=False)
    empty = (sizes == 0).unsqueeze(-1)
    return torch.where(empty, torch.zeros_like(out), out)
