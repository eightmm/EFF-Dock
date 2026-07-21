"""Loss functions for pose-confidence training."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def pose_confidence_loss(
    out: dict[str, Tensor],
    item: dict[str, Tensor],
    *,
    atom_weight: float = 0.2,
    atom_bce_weight: float = 0.2,
    pose_weight: float = 1.0,
    pose_bce_weight: float = 0.4,
    rank_weight: float = 0.5,
    listwise_weight: float = 0.0,
    listwise_tau: float = 1.0,
    listwise_pred_temp: float = 1.0,
    success_listwise_weight: float = 0.0,
    success_listwise_margin: float = 0.5,
    setwise_success_weight: float = 0.0,
    setwise_success_temperature: float = 1.0,
    pairwise_success_weight: float = 0.0,
    pairwise_success_temperature: float = 1.0,
    hard_pair_weight: float = 0.0,
    hard_pair_margin: float = 0.2,
    hard_pair_positive_threshold: float = 2.0,
    hard_pair_negative_threshold: float = 2.0,
    hard_pair_min_gap: float = 0.5,
    success_threshold: float = 2.0,
    rank_margin: float = 0.05,
    min_rank_gap: float = 0.3,
) -> dict[str, Tensor]:
    device = out["pose_rmsd_log1p"].device
    atom_disp = item["atom_disp"].to(device=device)
    pose_rmsd = item["pose_rmsd"].to(device=device)
    atom_target = torch.log1p(atom_disp.clamp_min(0.0))
    pose_target = torch.log1p(pose_rmsd.clamp_min(0.0))

    loss_atom = F.huber_loss(out["atom_disp_log1p"], atom_target)
    loss_atom_bce = F.binary_cross_entropy_with_logits(
        out["atom_ok_logit"],
        (atom_disp < success_threshold).float(),
    )
    loss_pose = F.huber_loss(out["pose_rmsd_log1p"], pose_target)
    loss_pose_bce = F.binary_cross_entropy_with_logits(
        out["pose_success_logit"],
        (pose_rmsd < success_threshold).float(),
    )

    true_gap = pose_target.unsqueeze(0) - pose_target.unsqueeze(1)
    pred_gap = out["pose_rmsd_log1p"].unsqueeze(0) - out["pose_rmsd_log1p"].unsqueeze(1)
    keep = true_gap > min_rank_gap
    if bool(keep.any()):
        loss_rank = F.relu(rank_margin - pred_gap[keep]).mean()
    else:
        loss_rank = loss_pose.new_zeros(())

    if pose_rmsd.numel() > 1 and listwise_weight > 0.0:
        target = F.softmax(-pose_rmsd.clamp_min(0.0) / max(float(listwise_tau), 1e-6), dim=0)
        pred_scores = -out["pose_rmsd_log1p"] / max(float(listwise_pred_temp), 1e-6)
        loss_listwise = -(target.detach() * F.log_softmax(pred_scores, dim=0)).sum()
    else:
        loss_listwise = loss_pose.new_zeros(())

    if pose_rmsd.numel() > 1 and success_listwise_weight > 0.0:
        quality = torch.sigmoid(
            (success_threshold - pose_rmsd.clamp_min(0.0))
            / max(float(success_listwise_margin), 1e-6)
        )
        if bool((quality.sum() <= 0).item()):
            target_success = torch.full_like(quality, 1.0 / max(1, quality.numel()))
        else:
            target_success = quality / quality.sum()
        loss_success_listwise = -(
            target_success.detach() * F.log_softmax(out["pose_success_logit"], dim=0)
        ).sum()
    else:
        loss_success_listwise = loss_pose.new_zeros(())

    success_mask = pose_rmsd < success_threshold
    failure_mask = ~success_mask
    if (
        pose_rmsd.numel() > 1
        and setwise_success_weight > 0.0
        and bool(success_mask.any().item())
        and bool(failure_mask.any().item())
    ):
        temperature = max(float(setwise_success_temperature), 1e-6)
        scaled_logits = out["pose_success_logit"] / temperature
        # Negative log probability that a categorical draw from the predicted
        # pose distribution lands anywhere in the successful (<2 A) set.
        loss_setwise_success = torch.logsumexp(scaled_logits, dim=0) - torch.logsumexp(
            scaled_logits[success_mask], dim=0
        )
    else:
        loss_setwise_success = loss_pose.new_zeros(())

    if (
        pose_rmsd.numel() > 1
        and pairwise_success_weight > 0.0
        and bool(success_mask.any().item())
        and bool(failure_mask.any().item())
    ):
        temperature = max(float(pairwise_success_temperature), 1e-6)
        positive = out["pose_success_logit"][success_mask]
        negative = out["pose_success_logit"][failure_mask]
        # Logistic ranking loss over all positive/negative pose pairs.  Unlike
        # hard-negative-only mining this keeps a stable gradient over the full
        # candidate set and directly matches the top-1 <2 A decision boundary.
        pairwise_gap = (negative[:, None] - positive[None, :]) / temperature
        loss_pairwise_success = F.softplus(pairwise_gap).mean()
    else:
        loss_pairwise_success = loss_pose.new_zeros(())

    if pose_rmsd.numel() > 1 and hard_pair_weight > 0.0:
        pos_pool = pose_rmsd < hard_pair_positive_threshold
        if bool(pos_pool.any().item()):
            pos_candidates = torch.where(pos_pool)[0]
            pos_idx = pos_candidates[pose_rmsd[pos_candidates].argmin()]
            neg_pool = (pose_rmsd >= hard_pair_negative_threshold) & (
                (pose_rmsd - pose_rmsd[pos_idx]) >= hard_pair_min_gap
            )
            if bool(neg_pool.any().item()):
                neg_candidates = torch.where(neg_pool)[0]
                # Mine the current model's most dangerous bad pose in-pool.
                neg_idx = neg_candidates[out["pose_rmsd_log1p"][neg_candidates].argmin()]
                pred_gap_hard = out["pose_rmsd_log1p"][neg_idx] - out["pose_rmsd_log1p"][pos_idx]
                loss_hard_pair = F.relu(hard_pair_margin - pred_gap_hard)
            else:
                loss_hard_pair = loss_pose.new_zeros(())
        else:
            loss_hard_pair = loss_pose.new_zeros(())
    else:
        loss_hard_pair = loss_pose.new_zeros(())

    total = (
        atom_weight * loss_atom
        + atom_bce_weight * loss_atom_bce
        + pose_weight * loss_pose
        + pose_bce_weight * loss_pose_bce
        + rank_weight * loss_rank
        + listwise_weight * loss_listwise
        + success_listwise_weight * loss_success_listwise
        + setwise_success_weight * loss_setwise_success
        + pairwise_success_weight * loss_pairwise_success
        + hard_pair_weight * loss_hard_pair
    )
    return {
        "loss": total,
        "loss_atom": loss_atom.detach(),
        "loss_atom_bce": loss_atom_bce.detach(),
        "loss_pose": loss_pose.detach(),
        "loss_pose_bce": loss_pose_bce.detach(),
        "loss_rank": loss_rank.detach(),
        "loss_listwise": loss_listwise.detach(),
        "loss_success_listwise": loss_success_listwise.detach(),
        "loss_setwise_success": loss_setwise_success.detach(),
        "loss_pairwise_success": loss_pairwise_success.detach(),
        "loss_hard_pair": loss_hard_pair.detach(),
    }


__all__ = ["pose_confidence_loss"]
