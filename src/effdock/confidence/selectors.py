"""Pose selectors used with the extmatch confidence checkpoint."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from effdock.preprocess.graph_types import NTYPE_PROT_ATOM


def _ranks(values: torch.Tensor, *, low_is_good: bool = True) -> torch.Tensor:
    values = values.detach().cpu().to(torch.float32)
    if not low_is_good:
        values = -values
    order = torch.argsort(values, stable=True)
    ranks = torch.empty_like(values)
    ranks[order] = torch.arange(values.numel(), dtype=torch.float32)
    return ranks


def _pairwise_pose_rmsd(poses: list[torch.Tensor]) -> torch.Tensor:
    pose = torch.stack([p.detach().cpu().to(torch.float32) for p in poses])
    flat = pose.reshape(pose.shape[0], -1)
    return torch.cdist(flat, flat) / float(pose.shape[1]) ** 0.5


def _cluster_stats(pairwise: torch.Tensor, cutoff: float) -> tuple[torch.Tensor, torch.Tensor]:
    within = pairwise <= float(cutoff)
    sizes = within.sum(dim=1).to(torch.float32)
    local_mean = torch.stack(
        [
            pairwise[i, within[i]].mean()
            if bool(within[i].any())
            else pairwise.new_tensor(float("inf"))
            for i in range(pairwise.shape[0])
        ]
    )
    return sizes, local_mean


def _pair_gate(
    base: int,
    alt: int,
    pred_rmsd: torch.Tensor,
    pairwise: torch.Tensor,
    *,
    margin: float,
    pair_min: float,
    pair_max: float,
) -> int:
    pair = float(pairwise[base, alt])
    if float(pred_rmsd[alt]) <= float(pred_rmsd[base]) + margin and pair_min <= pair <= pair_max:
        return int(alt)
    return int(base)


def _pair_delta_gate(
    base: int,
    alt: int,
    pred_rmsd: torch.Tensor,
    pairwise: torch.Tensor,
    *,
    delta_min: float,
    delta_max: float,
    pair_min: float,
    pair_max: float,
) -> int:
    delta = float(pred_rmsd[alt]) - float(pred_rmsd[base])
    pair = float(pairwise[base, alt])
    if delta_min <= delta <= delta_max and pair_min <= pair <= pair_max:
        return int(alt)
    return int(base)


@dataclass(frozen=True)
class ConfidenceFilterConfig:
    """Cardinality-independent thresholds for the deployable confidence filter."""

    pred_rmsd_margin: float = 0.03
    success_gain: float = 0.0
    atom_ok_gain: float = 0.0
    clash_limit: float | None = 0.0
    fallback_head_tolerance: float = 0.0
    mode: str = "strict_both"
    atom_rmsd_gain: float = 0.0
    success_tolerance: float = 0.05
    atom_ok_tolerance: float = 0.02


def protein_ligand_clash_rates(
    poses: list[torch.Tensor],
    graph: dict[str, torch.Tensor],
    pocket_center: torch.Tensor,
    *,
    cutoff: float = 1.6,
) -> torch.Tensor:
    """Return <=cutoff protein-ligand contacts per ligand atom for each pose."""
    if not poses:
        raise ValueError("poses must be non-empty")
    node_type = graph["node_type"].detach().cpu().to(torch.long)
    prot = graph["node_coords"].detach().cpu().to(torch.float32)[node_type == NTYPE_PROT_ATOM]
    prot = prot - pocket_center.detach().cpu().to(torch.float32)
    pose = torch.stack([p.detach().cpu().to(torch.float32) for p in poses])
    if not prot.numel():
        return torch.zeros(len(poses), dtype=torch.float32)
    return (
        (torch.cdist(pose, prot) <= float(cutoff)).float().sum(dim=(1, 2))
        / max(pose.shape[1], 1)
    )


def select_confidence_filter(
    scores: list[dict[str, float]],
    clash: torch.Tensor,
    *,
    config: ConfidenceFilterConfig = ConfidenceFilterConfig(),
) -> tuple[int, str]:
    """Select using fixed base-relative guards, without ranks or pose clustering."""
    if not scores or clash.numel() != len(scores):
        raise ValueError("scores and clash must be non-empty and have equal length")
    pred = torch.tensor([score["confidence_rmsd"] for score in scores], dtype=torch.float32)
    success = torch.tensor(
        [score["confidence_success"] for score in scores], dtype=torch.float32
    )
    atom_ok = torch.tensor(
        [score["confidence_atom_ok"] for score in scores], dtype=torch.float32
    )
    clash = clash.detach().cpu().to(torch.float32)
    if not all(torch.isfinite(value).all() for value in (pred, success, atom_ok, clash)):
        raise ValueError("confidence filter inputs must be finite")

    base = int(torch.argmin(pred))
    within_margin = pred <= pred[base] + float(config.pred_rmsd_margin)

    if config.clash_limit is not None and float(clash[base]) > float(config.clash_limit):
        tolerance = float(config.fallback_head_tolerance)
        physical = (
            within_margin
            & (clash <= float(config.clash_limit))
            & (success >= success[base] - tolerance)
            & (atom_ok >= atom_ok[base] - tolerance)
        )
        if bool(physical.any()):
            masked = pred.masked_fill(~physical, float("inf"))
            return int(torch.argmin(masked)), "physical_fallback"

    if config.mode == "strict_both":
        consensus = (
            within_margin
            & (clash <= clash[base])
            & (success >= success[base] + float(config.success_gain))
            & (atom_ok >= atom_ok[base] + float(config.atom_ok_gain))
        )
    elif config.mode == "atom_rmsd_guard":
        atom_rmsd = torch.tensor(
            [score["confidence_atom_rmsd"] for score in scores], dtype=torch.float32
        )
        if not torch.isfinite(atom_rmsd).all():
            raise ValueError("confidence atom RMSD inputs must be finite")
        consensus = (
            within_margin
            & (clash <= clash[base])
            & (atom_rmsd <= atom_rmsd[base] - float(config.atom_rmsd_gain))
            & (success >= success[base] - float(config.success_tolerance))
            & (atom_ok >= atom_ok[base] - float(config.atom_ok_tolerance))
        )
    else:
        raise ValueError(f"unknown confidence filter mode: {config.mode!r}")
    consensus[base] = False
    if bool(consensus.any()):
        masked = pred.masked_fill(~consensus, float("inf"))
        return int(torch.argmin(masked)), "head_consensus"
    return base, "base"


def select_confidence_poses(
    poses: list[torch.Tensor],
    scores: list[dict[str, float]],
    graph: dict[str, torch.Tensor],
    pocket_center: torch.Tensor,
) -> dict[str, int]:
    """Return frozen selector indices, including the deployed single-run selector.

    ``pair_gate_density_rank_vote_plclash_ambig`` is the exact post-hoc-frozen
    selector used for the recorded N80/S25 Astex and PoseBusters results.
    """
    if not poses or len(poses) != len(scores):
        raise ValueError("poses and scores must be non-empty and have equal length")
    required = {
        "confidence_rmsd",
        "confidence_success",
        "confidence_atom_rmsd",
        "confidence_atom_q90",
        "confidence_atom_ok",
    }
    missing = required - scores[0].keys()
    if missing:
        raise ValueError(f"confidence score fields missing: {sorted(missing)}")

    pred = torch.tensor([s["confidence_rmsd"] for s in scores])
    success = torch.tensor([s["confidence_success"] for s in scores])
    atom_rmsd = torch.tensor([s["confidence_atom_rmsd"] for s in scores])
    atom_q90 = torch.tensor([s["confidence_atom_q90"] for s in scores])
    atom_ok = torch.tensor([s["confidence_atom_ok"] for s in scores])
    pairwise = _pairwise_pose_rmsd(poses)

    sizes2, _ = _cluster_stats(pairwise, 2.0)
    density = pred - 0.5 * sizes2 / sizes2.max().clamp_min(1.0)
    density_i = int(torch.argmin(density))

    conf_q90 = 0.75 * _ranks(pred) + _ranks(success, low_is_good=False) + 0.5 * _ranks(atom_q90)
    conf_q90_i = int(torch.argmin(conf_q90))
    rank_vote = (
        _ranks(pred)
        + _ranks(atom_rmsd)
        + _ranks(success, low_is_good=False)
        + _ranks(atom_ok, low_is_good=False)
    )
    rank_vote_i = int(torch.argmin(rank_vote))

    selected = _pair_gate(
        conf_q90_i, density_i, pred, pairwise, margin=0.1, pair_min=2.5, pair_max=4.0
    )
    if selected == conf_q90_i:
        selected = _pair_gate(
            conf_q90_i, rank_vote_i, pred, pairwise, margin=0.05, pair_min=3.0, pair_max=4.5
        )

    clash = protein_ligand_clash_rates(poses, graph, pocket_center)
    top8_score = _ranks(pred) + 0.5 * _ranks(clash)
    top8_masked = torch.full_like(top8_score, float("inf"))
    top8 = torch.argsort(pred, stable=True)[: min(8, len(poses))]
    top8_masked[top8] = top8_score[top8]
    clash_i = int(torch.argmin(top8_masked))

    selected = _pair_gate(selected, clash_i, pred, pairwise, margin=0.0, pair_min=2.5, pair_max=3.5)
    selected = _pair_delta_gate(
        selected,
        clash_i,
        pred,
        pairwise,
        delta_min=0.0,
        delta_max=0.05,
        pair_min=4.0,
        pair_max=4.8,
    )

    for i, score in enumerate(scores):
        score["confidence_rank_vote"] = float(rank_vote[i])
        score["confidence_top8_pl_clash1p6_w0.5"] = float(top8_masked[i])
        score["pl_clash_1p6"] = float(clash[i])
    filter_i, _ = select_confidence_filter(scores, clash)
    return {
        "confidence": int(torch.argmin(pred)),
        "success": int(torch.argmax(success)),
        "atom_success": int(torch.argmax(atom_ok)),
        "rank_vote": rank_vote_i,
        "confidence_filter_v1": filter_i,
        "pair_gate_density_rank_vote_plclash_ambig": selected,
    }


__all__ = [
    "ConfidenceFilterConfig",
    "protein_ligand_clash_rates",
    "select_confidence_filter",
    "select_confidence_poses",
]
