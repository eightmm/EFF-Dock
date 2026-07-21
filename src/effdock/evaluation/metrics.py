"""Evaluation metrics for protein-ligand docking."""

from __future__ import annotations

import torch
from torch import Tensor


def ligand_rmsd(pred_pos: Tensor, true_pos: Tensor) -> Tensor:
    """Compute ligand heavy-atom RMSD between predicted and true positions.

    Args:
        pred_pos: Predicted atom coordinates ``[N_atom, 3]``.
        true_pos: Ground-truth atom coordinates ``[N_atom, 3]``.

    Returns:
        Scalar RMSD in Angstroms.
    """
    assert pred_pos.shape == true_pos.shape, f"Shape mismatch: {pred_pos.shape} vs {true_pos.shape}"
    diff = pred_pos - true_pos
    return torch.sqrt((diff**2).sum(dim=-1).mean())


def centroid_distance(pred_pos: Tensor, true_pos: Tensor) -> Tensor:
    """Compute distance between predicted and true ligand centroids.

    Args:
        pred_pos: Predicted atom coordinates ``[N_atom, 3]``.
        true_pos: Ground-truth atom coordinates ``[N_atom, 3]``.

    Returns:
        Scalar centroid distance in Angstroms.
    """
    pred_center = pred_pos.mean(dim=0)
    true_center = true_pos.mean(dim=0)
    return torch.linalg.vector_norm(pred_center - true_center)


def frag_centroid_rmsd(pred_T: Tensor, true_T: Tensor) -> Tensor:
    """Compute RMSD of fragment centroid positions.

    Args:
        pred_T: Predicted fragment translations ``[N_frag, 3]``.
        true_T: Ground-truth fragment translations ``[N_frag, 3]``.

    Returns:
        Scalar RMSD in Angstroms.
    """
    diff = pred_T - true_T
    return torch.sqrt((diff**2).sum(dim=-1).mean())


def success_rate(rmsds: Tensor, threshold: float = 2.0) -> Tensor:
    """Compute fraction of predictions below an RMSD threshold.

    Args:
        rmsds: Tensor of RMSD values (any shape).
        threshold: Success threshold in Angstroms (default 2.0).

    Returns:
        Scalar success rate in [0, 1].
    """
    return (rmsds < threshold).float().mean()


__all__ = ["ligand_rmsd", "centroid_distance", "frag_centroid_rmsd", "success_rate"]
