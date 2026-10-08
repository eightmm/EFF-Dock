"""Fragment pose utilities used by confidence-data generation and inference."""

from __future__ import annotations

from typing import Any

import torch

from effdock.geometry.se3 import matrix_to_quaternion
from effdock.inference.sampler import build_batched_graph

FRAME_RECOVERY_POLICIES = ("historical_kabsch", "contextual_v1")
_FRAME_RANK_RELATIVE_TOLERANCE = 1e-6
_FRAME_CONTEXT_RELATIVE_TOLERANCE = 1e-5


def _context_perpendicular(
    vectors: torch.Tensor, axis: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    perpendicular = vectors - (vectors @ axis).unsqueeze(-1) * axis
    length = torch.linalg.vector_norm(perpendicular, dim=-1)
    scale = torch.linalg.vector_norm(vectors, dim=-1).clamp_min(1.0)
    valid = length > _FRAME_CONTEXT_RELATIVE_TOLERANCE * scale
    return perpendicular, valid


def _context_basis(vectors: torch.Tensor) -> torch.Tensor:
    length = torch.linalg.vector_norm(vectors, dim=-1)
    nonzero = (length > _FRAME_CONTEXT_RELATIVE_TOLERANCE).nonzero(as_tuple=True)[0]
    if nonzero.numel() == 0:
        raise ValueError("contextual_v1 frame recovery: no nonzero context direction")
    axis = vectors[nonzero[0]] / length[nonzero[0]]
    perpendicular, valid = _context_perpendicular(vectors, axis)
    candidates = valid.nonzero(as_tuple=True)[0]
    if candidates.numel() == 0:
        raise ValueError("contextual_v1 frame recovery: context is collinear")
    second = perpendicular[candidates[0]]
    second = second / torch.linalg.vector_norm(second)
    return torch.stack((axis, second, torch.linalg.cross(axis, second)), dim=-1)


def _rank_one_rotation(
    local: torch.Tensor,
    covariance: torch.Tensor,
    world_context: torch.Tensor,
    template_context: torch.Tensor | None,
    receptor_context: torch.Tensor | None,
) -> torch.Tensor:
    _, _, Vh = torch.linalg.svd((local - local.mean(0)).double())
    axis = Vh[0].to(local)
    # A template-only sign convention is unaffected by global scene rotations.
    direction = local - local[0]
    sign = (direction @ axis).abs() > _FRAME_CONTEXT_RELATIVE_TOLERANCE
    index = sign.nonzero(as_tuple=True)[0]
    if index.numel() == 0:
        raise ValueError("contextual_v1 frame recovery: rank-one template has no axis")
    axis = torch.where((direction[index[0]] @ axis) < 0, -axis, axis)
    target = covariance.T @ axis
    magnitude = torch.linalg.vector_norm(target)
    if not bool(magnitude > torch.finfo(target.dtype).eps):
        raise ValueError("contextual_v1 frame recovery: observed fragment is collapsed")
    target = target / magnitude
    world_perp, world_valid = _context_perpendicular(world_context, target)
    local_second = None
    world_second = None
    if template_context is not None:
        local_perp, local_valid = _context_perpendicular(template_context, axis)
        paired = (local_valid & world_valid).nonzero(as_tuple=True)[0]
        if paired.numel():
            local_second, world_second = local_perp[paired[0]], world_perp[paired[0]]
    if world_second is None:
        # Cartesian axes complete only the fixed template/body basis. The world
        # basis must always come from scene vectors, never from global xyz axes.
        body_axis = torch.eye(3, device=axis.device, dtype=axis.dtype)[axis.abs().argmin()]
        local_second = body_axis - (body_axis @ axis) * axis
        candidates = world_valid.nonzero(as_tuple=True)[0]
        if candidates.numel():
            world_second = world_perp[candidates[0]]
        elif receptor_context is not None:
            perpendicular, valid = _context_perpendicular(receptor_context, target)
            candidates = valid.nonzero(as_tuple=True)[0]
            if candidates.numel():
                world_second = perpendicular[candidates[0]]
        if world_second is None:
            raise ValueError(
                "contextual_v1 frame recovery: no context perpendicular to fragment axis"
            )
    local_second = local_second / torch.linalg.vector_norm(local_second)
    world_second = world_second / torch.linalg.vector_norm(world_second)
    body = torch.stack((axis, local_second, torch.linalg.cross(axis, local_second)), dim=-1)
    world = torch.stack((target, world_second, torch.linalg.cross(target, world_second)), dim=-1)
    return world @ body.T


# ---------------------------------------------------------------------------
# Kabsch per-fragment rigid fit
# ---------------------------------------------------------------------------
def recover_frag_state(
    atom_pos: torch.Tensor,  # [N_atom, 3] global coords (pocket-centered OK)
    local_pos: torch.Tensor,  # [N_atom, 3] fragment-local (centroid-subtracted)
    frag_id: torch.Tensor,  # [N_atom]
    n_frag: int,
    *,
    frame_policy: str = "historical_kabsch",
    template_atom_pos: torch.Tensor | None = None,
    receptor_atom_pos: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Recover fragment poses; contextual_v1 fixes low-rank frames from scene context.

    Historical Kabsch reproduces checkpoint training features. The opt-in
    contextual convention is a frozen-feature change, not a weight conversion.
    Template coordinates remain in their fixed body convention, while receptor
    context must share the current atom-coordinate frame. A degenerate complete
    scene has no equivariant full frame and is rejected by contextual_v1.
    """
    if frame_policy not in FRAME_RECOVERY_POLICIES:
        raise ValueError(f"unknown confidence frame policy: {frame_policy!r}")
    if frame_policy == "contextual_v1":
        if atom_pos.shape != local_pos.shape or atom_pos.ndim != 2 or atom_pos.shape[-1] != 3:
            raise ValueError("contextual_v1 frame recovery: invalid atom/local coordinate shape")
        if frag_id.shape != atom_pos.shape[:1] or frag_id.dtype != torch.long:
            raise ValueError("contextual_v1 frame recovery: invalid fragment IDs")
        if (
            n_frag < 1
            or atom_pos.shape[0] == 0
            or not bool(((frag_id >= 0) & (frag_id < n_frag)).all())
        ):
            raise ValueError("contextual_v1 frame recovery: invalid fragment partition")
        for value in (atom_pos, local_pos, template_atom_pos, receptor_atom_pos):
            if value is not None and (
                value.ndim != 2 or value.shape[-1] != 3 or not bool(torch.isfinite(value).all())
            ):
                raise ValueError("contextual_v1 frame recovery: invalid/nonfinite coordinates")
            if value is not None and (
                value.dtype != atom_pos.dtype
                or value.device != atom_pos.device
                or not value.is_floating_point()
            ):
                raise ValueError("contextual_v1 frame recovery: coordinate dtype/device mismatch")
        if frag_id.device != atom_pos.device:
            raise ValueError("contextual_v1 frame recovery: fragment-ID device mismatch")
        if template_atom_pos is not None and template_atom_pos.shape != atom_pos.shape:
            raise ValueError("contextual_v1 frame recovery: template atom-order/shape mismatch")
    T = torch.zeros(n_frag, 3, device=atom_pos.device, dtype=atom_pos.dtype)
    R = torch.eye(3, device=atom_pos.device, dtype=atom_pos.dtype).expand(n_frag, 3, 3).clone()
    for f in range(n_frag):
        mask = frag_id == f
        if mask.sum() == 0:
            if frame_policy == "contextual_v1":
                raise ValueError("contextual_v1 frame recovery: empty fragment")
            continue
        y = atom_pos[mask]
        x = local_pos[mask]
        T[f] = y.mean(dim=0)
        if frame_policy == "contextual_v1":
            singular = torch.linalg.svdvals((x - x.mean(0)).double())
            threshold = max(1e-12, float(singular.max()) * _FRAME_RANK_RELATIVE_TOLERANCE)
            rank = int((singular > threshold).sum())
            world_context = atom_pos - T[f]
            template_context = None
            if template_atom_pos is not None:
                template_context = template_atom_pos - template_atom_pos[mask].mean(0)
            receptor_context = None if receptor_atom_pos is None else receptor_atom_pos - T[f]
            if rank == 0:
                if int(mask.sum()) != 1:
                    raise ValueError("contextual_v1 frame recovery: collapsed multi-atom template")
                context = (
                    world_context
                    if receptor_context is None
                    else torch.cat((world_context, receptor_context))
                )
                R[f] = _context_basis(context)
                continue
            if rank == 1:
                R[f] = _rank_one_rotation(
                    x, x.T @ (y - T[f]), world_context, template_context, receptor_context
                )
                continue
        if mask.sum() == 1:
            continue
        y_c = y - T[f]
        H = x.T @ y_c
        U, singular, Vh = torch.linalg.svd(H)
        if frame_policy == "contextual_v1" and not bool(
            singular[1] > max(1e-12, float(singular[0]) * 1e-12)
        ):
            raise ValueError("contextual_v1 frame recovery: observed fragment lost template rank")
        d = torch.sign(torch.linalg.det(Vh.T @ U.T))
        D = torch.eye(3, device=atom_pos.device, dtype=atom_pos.dtype)
        D[2, 2] = d
        R[f] = Vh.T @ D @ U.T
    q = matrix_to_quaternion(R)
    return T, q


def extract_t1_ligand_irreps(
    model: torch.nn.Module,
    graph: dict[str, torch.Tensor],
    lig: dict[str, torch.Tensor],
    meta: dict[str, Any],
    poses: torch.Tensor,
    *,
    sigma: float | torch.Tensor,
    device: torch.device,
    hidden_dtype: torch.dtype,
    frame_policy: str = "historical_kabsch",
) -> dict[str, torch.Tensor]:
    """Run one batched t=1 forward and keep selected ligand-node features."""
    B, n_atoms, _ = poses.shape
    n_frags = int(meta["num_frag"])
    pocket_center = meta["pocket_center"].to(device)
    batch = build_batched_graph(graph, B, n_frags, device)
    batch["node_coords"] = batch["node_coords"] - pocket_center

    frag_id = lig["fragment_id"].to(device)
    local_pos = lig["frag_local_coords"].to(device)
    frag_sizes = lig["frag_sizes"].to(device)
    template = lig["atom_coords"].to(device) if frame_policy == "contextual_v1" else None
    receptor = None
    if frame_policy == "contextual_v1":
        from effdock.preprocess.graph_types import NTYPE_PROT_ATOM

        receptor = (
            graph["node_coords"][graph["node_type"] == NTYPE_PROT_ATOM].to(device) - pocket_center
        )

    T_list: list[torch.Tensor] = []
    q_list: list[torch.Tensor] = []
    for pose in poses:
        if frame_policy == "historical_kabsch":
            T_f, q_f = recover_frag_state(pose.to(device), local_pos, frag_id, n_frags)
        else:
            T_f, q_f = recover_frag_state(
                pose.to(device),
                local_pos,
                frag_id,
                n_frags,
                frame_policy=frame_policy,
                template_atom_pos=template,
                receptor_atom_pos=receptor,
            )
        T_list.append(T_f)
        q_list.append(q_f)
    T_flat = torch.cat(T_list, dim=0)
    q_flat = torch.cat(q_list, dim=0)
    atom_pos_flat = poses.to(device).reshape(B * n_atoms, 3)

    n_nodes = int(graph["node_coords"].shape[0])
    frag_start, frag_end = int(graph["lig_frag_slice"][0]), int(graph["lig_frag_slice"][1])
    atom_start = int(graph["lig_atom_slice"][0])
    frag_slots = torch.cat(
        [torch.arange(frag_start, frag_end, device=device) + i * n_nodes for i in range(B)]
    )
    atom_slots = torch.cat(
        [
            torch.arange(atom_start, atom_start + n_atoms, device=device) + i * n_nodes
            for i in range(B)
        ]
    )

    node_coords = batch["node_coords"].clone()
    node_coords[frag_slots] = T_flat
    node_coords[atom_slots] = atom_pos_flat

    frag_id_flat = (
        frag_id.repeat(B) + torch.arange(B, device=device).repeat_interleave(n_atoms) * n_frags
    )
    batch["node_coords"] = node_coords
    batch["T_frag"] = T_flat
    batch["q_frag"] = q_flat
    batch["frag_sizes"] = frag_sizes.repeat(B)
    batch["t"] = torch.ones(B, 1, device=device, dtype=torch.float32)
    batch["frag_id_for_atoms"] = frag_id_flat
    if isinstance(sigma, torch.Tensor):
        prior_sigma = sigma.view(-1).to(device=device, dtype=torch.float32)
        if prior_sigma.numel() != B:
            raise ValueError(f"sigma tensor must have {B} entries, got {prior_sigma.numel()}")
    else:
        prior_sigma = torch.full((B,), float(sigma), device=device, dtype=torch.float32)
    batch["prior_sigma"] = prior_sigma

    with torch.no_grad():
        out = model(batch, return_hidden=True)

    h = out["h"].view(B, n_nodes, -1)
    node_indices = torch.cat(
        [
            torch.arange(frag_start, frag_end, dtype=torch.long),
            torch.arange(atom_start, atom_start + n_atoms, dtype=torch.long),
        ]
    )
    return {
        "h_lig_node": h.index_select(1, node_indices.to(device)).to(hidden_dtype).cpu(),
        "lig_node_type": graph["node_type"].index_select(0, node_indices).cpu(),
    }


__all__ = ["FRAME_RECOVERY_POLICIES", "extract_t1_ligand_irreps", "recover_frag_state"]
