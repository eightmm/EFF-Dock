"""Fragment pose utilities used by confidence-data generation and inference."""

from __future__ import annotations

from typing import Any

import torch

from effdock.geometry.se3 import matrix_to_quaternion
from effdock.inference.sampler import build_batched_graph


# ---------------------------------------------------------------------------
# Kabsch per-fragment rigid fit
# ---------------------------------------------------------------------------
def recover_frag_state(
    atom_pos: torch.Tensor,  # [N_atom, 3] global coords (pocket-centered OK)
    local_pos: torch.Tensor,  # [N_atom, 3] fragment-local (centroid-subtracted)
    frag_id: torch.Tensor,  # [N_atom]
    n_frag: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return T_frag [N_frag, 3] and q_frag [N_frag, 4] (w,x,y,z)."""
    T = torch.zeros(n_frag, 3, device=atom_pos.device, dtype=atom_pos.dtype)
    R = torch.eye(3, device=atom_pos.device, dtype=atom_pos.dtype).expand(n_frag, 3, 3).clone()
    for f in range(n_frag):
        mask = frag_id == f
        if mask.sum() == 0:
            continue
        y = atom_pos[mask]
        x = local_pos[mask]
        T[f] = y.mean(dim=0)
        if mask.sum() == 1:
            continue
        y_c = y - T[f]
        H = x.T @ y_c
        U, _, Vh = torch.linalg.svd(H)
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

    T_list: list[torch.Tensor] = []
    q_list: list[torch.Tensor] = []
    for pose in poses:
        T_f, q_f = recover_frag_state(pose.to(device), local_pos, frag_id, n_frags)
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


__all__ = ["extract_t1_ligand_irreps", "recover_frag_state"]
