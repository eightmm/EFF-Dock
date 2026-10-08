from __future__ import annotations

import pytest
import torch

from effdock.confidence.features import recover_frag_state
from effdock.confidence.losses import pose_confidence_loss
from effdock.confidence.selectors import (
    ConfidenceFilterConfig,
    select_confidence_filter,
    select_confidence_poses,
)
from effdock.geometry.se3 import quaternion_to_matrix
from effdock.preprocess.graph_types import NTYPE_PROT_ATOM
from effdock.workflows.tune_confidence_filter import _select_batch


def test_frozen_confidence_selector_returns_declared_indices() -> None:
    poses = [
        torch.zeros(2, 3),
        torch.ones(2, 3),
        torch.full((2, 3), 4.0),
    ]
    scores = [
        {
            "confidence_rmsd": 1.5,
            "confidence_success": 0.4,
            "confidence_atom_rmsd": 1.5,
            "confidence_atom_q90": 1.5,
            "confidence_atom_ok": 0.4,
        },
        {
            "confidence_rmsd": 0.8,
            "confidence_success": 0.9,
            "confidence_atom_rmsd": 0.8,
            "confidence_atom_q90": 0.8,
            "confidence_atom_ok": 0.9,
        },
        {
            "confidence_rmsd": 2.5,
            "confidence_success": 0.1,
            "confidence_atom_rmsd": 2.5,
            "confidence_atom_q90": 2.5,
            "confidence_atom_ok": 0.1,
        },
    ]
    graph = {
        "node_type": torch.tensor([NTYPE_PROT_ATOM]),
        "node_coords": torch.tensor([[10.0, 10.0, 10.0]]),
    }
    indices = select_confidence_poses(poses, scores, graph, torch.zeros(3))

    assert indices["confidence"] == 1
    assert indices["success"] == 1
    assert indices["atom_success"] == 1
    assert indices["rank_vote"] == 1
    assert indices["confidence_filter_v1"] == 1
    assert indices["pair_gate_density_rank_vote_plclash_ambig"] == 1
    assert all("confidence_rank_vote" in score for score in scores)


def test_confidence_filter_switches_only_within_fixed_guards() -> None:
    scores = [
        {"confidence_rmsd": 0.80, "confidence_success": 0.40, "confidence_atom_ok": 0.45},
        {"confidence_rmsd": 0.87, "confidence_success": 0.52, "confidence_atom_ok": 0.58},
        {"confidence_rmsd": 0.95, "confidence_success": 0.80, "confidence_atom_ok": 0.80},
    ]
    selected, reason = select_confidence_filter(
        scores,
        torch.tensor([0.0, 0.0, 0.0]),
        config=ConfidenceFilterConfig(
            pred_rmsd_margin=0.10,
            success_gain=0.05,
            atom_ok_gain=0.05,
            clash_limit=None,
        ),
    )
    assert selected == 1
    assert reason == "head_consensus"


def test_confidence_filter_uses_physical_fallback_without_pose_clustering() -> None:
    scores = [
        {"confidence_rmsd": 0.80, "confidence_success": 0.60, "confidence_atom_ok": 0.60},
        {"confidence_rmsd": 0.86, "confidence_success": 0.59, "confidence_atom_ok": 0.59},
    ]
    selected, reason = select_confidence_filter(
        scores,
        torch.tensor([0.25, 0.0]),
        config=ConfidenceFilterConfig(
            pred_rmsd_margin=0.10,
            success_gain=0.10,
            atom_ok_gain=0.10,
            clash_limit=0.05,
            fallback_head_tolerance=0.02,
        ),
    )
    assert selected == 1
    assert reason == "physical_fallback"


def test_tuning_batch_selector_matches_runtime_filter() -> None:
    config = {
        "pred_rmsd_margin": 0.10,
        "success_gain": 0.05,
        "atom_ok_gain": 0.05,
        "clash_limit": 0.05,
        "fallback_head_tolerance": 0.02,
    }
    pred = torch.tensor([[0.80, 0.86, 1.20], [0.70, 0.76, 0.90]])
    success = torch.tensor([[0.60, 0.59, 0.90], [0.40, 0.50, 0.60]])
    atom_ok = torch.tensor([[0.60, 0.59, 0.90], [0.40, 0.50, 0.60]])
    clash = torch.tensor([[0.25, 0.00, 0.00], [0.00, 0.00, 0.00]])
    cache = {
        "records": [
            {
                "pred_rmsd": pred[i],
                "success": success[i],
                "atom_ok": atom_ok[i],
                "clash": clash[i],
            }
            for i in range(2)
        ]
    }
    batch_selected = _select_batch(cache, config)
    runtime_selected = []
    for i in range(2):
        scores = [
            {
                "confidence_rmsd": float(pred[i, j]),
                "confidence_success": float(success[i, j]),
                "confidence_atom_ok": float(atom_ok[i, j]),
            }
            for j in range(3)
        ]
        selected, _ = select_confidence_filter(
            scores,
            clash[i],
            config=ConfidenceFilterConfig(**config),
        )
        runtime_selected.append(selected)
    assert batch_selected.tolist() == runtime_selected


def test_confidence_filter_is_unchanged_by_irrelevant_duplicate_poses() -> None:
    scores = [
        {"confidence_rmsd": 0.80, "confidence_success": 0.40, "confidence_atom_ok": 0.40},
        {"confidence_rmsd": 0.86, "confidence_success": 0.50, "confidence_atom_ok": 0.50},
    ]
    config = ConfidenceFilterConfig(
        pred_rmsd_margin=0.10,
        success_gain=0.05,
        atom_ok_gain=0.05,
        clash_limit=None,
    )
    selected, _ = select_confidence_filter(scores, torch.zeros(2), config=config)
    duplicate = {
        "confidence_rmsd": 2.0,
        "confidence_success": 0.10,
        "confidence_atom_ok": 0.10,
    }
    expanded, _ = select_confidence_filter(
        [*scores, *([duplicate] * 20)],
        torch.zeros(22),
        config=config,
    )
    assert selected == expanded == 1


def test_atom_rmsd_guard_filters_a_near_tie() -> None:
    scores = [
        {
            "confidence_rmsd": 0.80,
            "confidence_success": 0.60,
            "confidence_atom_rmsd": 0.90,
            "confidence_atom_ok": 0.60,
        },
        {
            "confidence_rmsd": 0.92,
            "confidence_success": 0.56,
            "confidence_atom_rmsd": 0.80,
            "confidence_atom_ok": 0.59,
        },
    ]
    config = ConfidenceFilterConfig(
        mode="atom_rmsd_guard",
        pred_rmsd_margin=0.20,
        atom_rmsd_gain=0.0,
        success_tolerance=0.05,
        atom_ok_tolerance=0.02,
        clash_limit=None,
    )
    selected, reason = select_confidence_filter(scores, torch.zeros(2), config=config)
    assert selected == 1
    assert reason == "head_consensus"


def test_confidence_multitask_loss_backpropagates() -> None:
    out = {
        "pose_rmsd_log1p": torch.zeros(3, requires_grad=True),
        "pose_success_logit": torch.zeros(3, requires_grad=True),
        "atom_disp_log1p": torch.zeros(3, 2, requires_grad=True),
        "atom_ok_logit": torch.zeros(3, 2, requires_grad=True),
    }
    item = {
        "pose_rmsd": torch.tensor([0.8, 2.5, 4.0]),
        "atom_disp": torch.tensor([[0.5, 0.8], [2.0, 2.5], [3.0, 4.0]]),
    }
    losses = pose_confidence_loss(
        out,
        item,
        success_listwise_weight=1.0,
        rank_weight=0.1,
    )
    losses["loss"].backward()

    assert torch.isfinite(losses["loss"])
    assert out["pose_rmsd_log1p"].grad is not None
    assert out["pose_success_logit"].grad is not None


def test_selection_losses_reward_successful_pose_ranking() -> None:
    item = {
        "pose_rmsd": torch.tensor([0.8, 1.4, 2.5, 4.0]),
        "atom_disp": torch.ones(4, 2),
    }

    def selection_loss(logits: torch.Tensor) -> torch.Tensor:
        out = {
            "pose_rmsd_log1p": torch.zeros(4, requires_grad=True),
            "pose_success_logit": logits,
            "atom_disp_log1p": torch.zeros(4, 2, requires_grad=True),
            "atom_ok_logit": torch.zeros(4, 2, requires_grad=True),
        }
        losses = pose_confidence_loss(
            out,
            item,
            atom_weight=0.0,
            atom_bce_weight=0.0,
            pose_weight=0.0,
            pose_bce_weight=0.0,
            rank_weight=0.0,
            success_listwise_weight=0.0,
            setwise_success_weight=1.0,
            pairwise_success_weight=1.0,
        )
        return losses["loss"]

    correctly_ranked = selection_loss(torch.tensor([2.0, 1.0, -1.0, -2.0]))
    incorrectly_ranked = selection_loss(torch.tensor([-2.0, -1.0, 1.0, 2.0]))
    assert correctly_ranked < incorrectly_ranked


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("first", ["line", "plane", "full"])
def test_contextual_recovery_is_se3_covariant_and_preserves_geometry(dtype, first):
    fragments = {
        "line": [[-1.0, -0.3, -0.2], [0.0, 0.0, 0.0], [1.0, 0.3, 0.2]],
        "plane": [[0.0, 0.0, 0.0], [1.0, 0.2, 0.0], [-0.2, 0.8, 0.0]],
        "full": [[0.0, 0.0, 0.0], [1.0, 0.2, 0.0], [-0.2, 0.8, 0.1], [0.1, -0.3, 1.0]],
    }
    left = torch.tensor(fragments[first], dtype=dtype)
    right = torch.tensor(fragments["full"], dtype=dtype) + torch.tensor(
        [3.0, 2.0, 1.0], dtype=dtype
    )
    template = torch.cat((left, right))
    ids = torch.tensor([0] * len(left) + [1] * len(right))
    centers = torch.stack((left.mean(0), right.mean(0)))
    local = template - centers[ids]
    rotation = quaternion_to_matrix(torch.tensor([0.7, 0.2, -0.4, 0.5], dtype=dtype))
    change = quaternion_to_matrix(torch.tensor([0.1, -0.7, 0.2, 0.3], dtype=dtype))
    translation = torch.tensor([2.3, -1.7, 0.5], dtype=dtype)
    posed = template @ rotation.T + translation
    T, q = recover_frag_state(
        posed, local, ids, 2, frame_policy="contextual_v1", template_atom_pos=template
    )
    R = quaternion_to_matrix(q)
    changed = posed @ change.T + translation
    new_T, new_q = recover_frag_state(
        changed, local, ids, 2, frame_policy="contextual_v1", template_atom_pos=template
    )
    new_R = quaternion_to_matrix(new_q)
    tolerance = 3e-5 if dtype == torch.float32 else 1e-10
    assert torch.allclose(new_T, T @ change.T + translation, atol=tolerance, rtol=0)
    assert torch.allclose(new_R, change @ R, atol=tolerance, rtol=0)
    assert torch.allclose(R, rotation.expand_as(R), atol=tolerance, rtol=0)
    assert torch.allclose(torch.linalg.det(R), torch.ones(2, dtype=dtype), atol=tolerance, rtol=0)
    reconstructed = torch.einsum("nij,nj->ni", R[ids], local) + T[ids]
    assert torch.allclose(reconstructed, posed, atol=tolerance, rtol=0)
    if first != "line":
        old_T, old_q = recover_frag_state(posed, local, ids, 2)
        assert torch.equal(T, old_T) and torch.equal(q, old_q)


def test_contextual_rank_one_recovery_uses_receptor_when_ligand_is_collinear():
    local = torch.tensor([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float64)
    ids = torch.zeros(2, dtype=torch.long)
    pose = local + torch.tensor([1.0, 3.0, -2.0], dtype=torch.float64)
    context = torch.tensor([[0.0, 1.0, 2.0], [2.0, -1.0, 3.0]], dtype=torch.float64)
    Q = quaternion_to_matrix(torch.tensor([0.7, 0.2, 0.3, -0.4], dtype=torch.float64))
    shift = torch.tensor([0.8, -1.1, 0.2], dtype=torch.float64)
    T, q = recover_frag_state(
        pose, local, ids, 1, frame_policy="contextual_v1", receptor_atom_pos=context
    )
    new_T, new_q = recover_frag_state(
        pose @ Q.T + shift,
        local,
        ids,
        1,
        frame_policy="contextual_v1",
        receptor_atom_pos=context @ Q.T + shift,
    )
    assert torch.allclose(
        quaternion_to_matrix(new_q), Q @ quaternion_to_matrix(q), atol=1e-10, rtol=0
    )
    assert torch.allclose(new_T, T @ Q.T + shift, atol=1e-10, rtol=0)


def test_contextual_singleton_frame_is_covariant():
    pose = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float64)
    context = torch.tensor([[2.0, 3.0, 5.0], [0.0, 5.0, 1.0]], dtype=torch.float64)
    ids = torch.zeros(1, dtype=torch.long)
    Q = quaternion_to_matrix(torch.tensor([0.7, 0.2, 0.3, -0.4], dtype=torch.float64))
    _, q = recover_frag_state(
        pose,
        torch.zeros_like(pose),
        ids,
        1,
        frame_policy="contextual_v1",
        receptor_atom_pos=context,
    )
    _, changed_q = recover_frag_state(
        pose @ Q.T,
        torch.zeros_like(pose),
        ids,
        1,
        frame_policy="contextual_v1",
        receptor_atom_pos=context @ Q.T,
    )
    assert torch.allclose(
        quaternion_to_matrix(changed_q), Q @ quaternion_to_matrix(q), atol=1e-10, rtol=0
    )


@pytest.mark.parametrize("height", [1e-7, 1e-3])
def test_contextual_nearly_collinear_template_has_covariant_frame(height):
    template = torch.tensor(
        [[-1.0, 0.0, 0.0], [0.0, height, 0.0], [1.0, 0.0, 0.0], [0.2, 2.0, 1.0], [0.3, 2.5, -1.0]],
        dtype=torch.float64,
    )
    ids = torch.tensor([0, 0, 0, 1, 1])
    centers = torch.stack((template[:3].mean(0), template[3:].mean(0)))
    local = template - centers[ids]
    Q = quaternion_to_matrix(torch.tensor([0.7, 0.2, 0.3, -0.4], dtype=torch.float64))
    _, q = recover_frag_state(
        template, local, ids, 2, frame_policy="contextual_v1", template_atom_pos=template
    )
    _, new_q = recover_frag_state(
        template @ Q.T, local, ids, 2, frame_policy="contextual_v1", template_atom_pos=template
    )
    assert torch.allclose(
        quaternion_to_matrix(new_q), Q @ quaternion_to_matrix(q), atol=1e-9, rtol=0
    )


def test_contextual_recovery_rejects_unobservable_scene_and_collapsed_fragment():
    local = torch.tensor([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    ids = torch.zeros(2, dtype=torch.long)
    with pytest.raises(ValueError, match="no context perpendicular"):
        recover_frag_state(local, local, ids, 1, frame_policy="contextual_v1")
    with pytest.raises(ValueError, match="observed fragment is collapsed"):
        recover_frag_state(torch.zeros_like(local), local, ids, 1, frame_policy="contextual_v1")
    with pytest.raises(ValueError, match="collapsed multi-atom template"):
        recover_frag_state(local, torch.zeros_like(local), ids, 1, frame_policy="contextual_v1")
    with pytest.raises(ValueError, match="nonfinite"):
        recover_frag_state(local * float("nan"), local, ids, 1, frame_policy="contextual_v1")


def test_contextual_recovery_uses_each_pose_independently():
    template = torch.tensor([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 2.0, 1.0]])
    ids = torch.tensor([0, 0, 1, 1])
    local = template - torch.stack((template[:2].mean(0), template[2:].mean(0)))[ids]
    _, q = recover_frag_state(
        template, local, ids, 2, frame_policy="contextual_v1", template_atom_pos=template
    )
    jitter = template.clone()
    jitter[2:, 2] += 1e-6
    _, perturbed_q = recover_frag_state(
        jitter, local, ids, 2, frame_policy="contextual_v1", template_atom_pos=template
    )
    _, repeated_q = recover_frag_state(
        template, local, ids, 2, frame_policy="contextual_v1", template_atom_pos=template
    )
    assert torch.equal(q, repeated_q)
    assert float((quaternion_to_matrix(q) - quaternion_to_matrix(perturbed_q)).abs().max()) < 3e-6
