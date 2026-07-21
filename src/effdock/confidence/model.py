"""Docking-graph confidence model for generated ligand poses.

This scorer keeps the confidence objective, but upgrades the representation to
the same heterogeneous graph vocabulary used by EFF-Dock: ligand atoms,
ligand fragments, protein atoms, protein residue virtual nodes, static
topology edges, residue-fragment edges, and dynamic protein-ligand contacts.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from effdock.inference.sampler import build_batched_graph
from effdock.models.effdock import (
    EFFDockInteractionLayer,
    EFFDockNodeEmbedding,
    _build_node_irreps,
    build_pair_contact_features,
)
from effdock.models.equivariant import IrrepsLayout
from effdock.models.nn_utils import rbf_encode, scatter_mean
from effdock.preprocess.graph_types import (
    ETYPE_DYNAMIC_CONTACT,
    NTYPE_FRAGMENT,
    NTYPE_LIG_ATOM,
    NTYPE_PROT_ATOM,
    NTYPE_PROT_RES,
    NUM_EDGE_TYPES,
)

PROTEIN_FLAG_KEYS = [
    "patom_is_backbone",
    "patom_is_metal",
    "patom_is_donor",
    "patom_is_acceptor",
    "patom_is_positive",
    "patom_is_negative",
    "patom_is_hydrophobic",
]


def _mlp(in_dim: int, hidden: int, out_dim: int, depth: int, dropout: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    dim = in_dim
    for _ in range(max(depth - 1, 0)):
        layers += [nn.Linear(dim, hidden), nn.SiLU(), nn.Dropout(dropout)]
        dim = hidden
    layers.append(nn.Linear(dim, out_dim))
    return nn.Sequential(*layers)


def _pool(z: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
    if not bool(mask.any()):
        empty = z.new_zeros(z.shape[0], z.shape[-1])
        return empty, empty
    selected = z[:, mask]
    return selected.mean(dim=1), selected.amax(dim=1)


class DockingGraphPoseConfidence(nn.Module):
    """Predict pose quality with a EFF-Dock-style full protein-ligand graph."""

    def __init__(
        self,
        *,
        flow_hidden_dim: int = 384,
        flow_hidden_vec_dim: int = 32,
        flow_l2_dim: int = 16,
        flow_l2o_dim: int = 16,
        n_rbf: int = 32,
        contact_cutoff: float = 5.0,
        hidden: int = 512,
        num_layers: int = 4,
        sh_lmax: int = 2,
        cond_dim: int = 128,
        dropout: float = 0.1,
        max_frag_size: int = 120,
        use_saved_ligand_hidden: bool = True,
        pose_readout: str = "global_pool",
    ) -> None:
        super().__init__()
        self.flow_hidden_dim = int(flow_hidden_dim)
        self.flow_hidden_vec_dim = int(flow_hidden_vec_dim)
        self.flow_l2_dim = int(flow_l2_dim)
        self.flow_l2o_dim = int(flow_l2o_dim)
        self.n_rbf = int(n_rbf)
        self.contact_cutoff = float(contact_cutoff)
        self.hidden = int(hidden)
        self.cond_dim = int(cond_dim)
        self.use_saved_ligand_hidden = bool(use_saved_ligand_hidden)
        self.pose_readout = str(pose_readout)
        if self.pose_readout not in {
            "global_pool",
            "contact_attention",
            "global_contact_attention",
        }:
            raise ValueError(f"unknown pose_readout={self.pose_readout!r}")

        self.node_irreps = _build_node_irreps(
            self.flow_hidden_dim,
            self.flow_hidden_vec_dim,
            self.flow_l2_dim,
            self.flow_l2o_dim,
        )
        self.layout = IrrepsLayout(self.node_irreps)
        self.feat_dim = self.layout.feat_dim
        if self.feat_dim != self._expected_flow_dim:
            raise ValueError(
                f"EFF-Dock hidden dim mismatch: irreps={self.feat_dim}, expected={self._expected_flow_dim}"
            )

        self.node_emb = EFFDockNodeEmbedding(self.flow_hidden_dim)
        self.frag_size_emb = nn.Embedding(max_frag_size + 1, 16)
        self.frag_init_mlp = nn.Sequential(
            nn.Linear(self.flow_hidden_dim + 16, self.flow_hidden_dim),
            nn.SiLU(),
            nn.Linear(self.flow_hidden_dim, self.flow_hidden_dim),
        )
        self.vec_gate = nn.Sequential(
            nn.Linear(self.flow_hidden_dim, self.flow_hidden_vec_dim),
            nn.Tanh(),
        )
        self.lig_hidden_gate = nn.Parameter(
            torch.tensor(0.25),
            requires_grad=self.use_saved_ligand_hidden,
        )

        self.layers = nn.ModuleList(
            [
                EFFDockInteractionLayer(
                    scalar_dim=self.flow_hidden_dim,
                    vec_dim=self.flow_hidden_vec_dim,
                    l2_dim=self.flow_l2_dim,
                    l2o_dim=self.flow_l2o_dim,
                    t_emb_dim=cond_dim,
                    n_edge_types=NUM_EDGE_TYPES,
                    n_rbf=n_rbf,
                    sh_lmax=sh_lmax,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        node_inv_dim = self.layout.n_scalar_channels + self.layout.n_nonscalar_channels
        contact_dim = self.n_rbf + 1 + 4 + len(PROTEIN_FLAG_KEYS)
        self.atom_norm = nn.LayerNorm(node_inv_dim + contact_dim)
        self.atom_trunk = _mlp(node_inv_dim + contact_dim, hidden, hidden, 2, dropout)
        self.atom_head = _mlp(hidden, hidden, 2, 2, dropout)
        if self.pose_readout in {"contact_attention", "global_contact_attention"}:
            self.contact_readout_norm = nn.LayerNorm(hidden + contact_dim)
            self.contact_readout_proj = _mlp(hidden + contact_dim, hidden, hidden, 2, dropout)
            self.contact_readout_score = _mlp(hidden, hidden, 1, 2, dropout)
        if self.pose_readout == "contact_attention":
            self.contact_pose_norm = nn.LayerNorm(hidden * 4)
            self.contact_pose_head = _mlp(hidden * 4, hidden, 2, 3, dropout)
        if self.pose_readout in {"global_pool", "global_contact_attention"}:
            self.node_norm = nn.LayerNorm(node_inv_dim)
            self.node_proj = _mlp(node_inv_dim, hidden, hidden, 2, dropout)
        if self.pose_readout == "global_pool":
            self.pose_norm = nn.LayerNorm(hidden * 8)
            self.pose_head = _mlp(hidden * 8, hidden, 2, 3, dropout)
        if self.pose_readout == "global_contact_attention":
            self.hybrid_pose_norm = nn.LayerNorm(hidden * 12)
            self.hybrid_pose_head = _mlp(hidden * 12, hidden, 2, 3, dropout)

    def forward(self, item: dict[str, Tensor]) -> dict[str, Tensor]:
        return self.forward_complex(item)

    @property
    def _expected_flow_dim(self) -> int:
        return (
            self.flow_hidden_dim
            + 2 * self.flow_hidden_vec_dim * 3
            + self.flow_l2_dim * 5
            + self.flow_l2o_dim * 5
        )

    def hidden_invariants(self, h: Tensor) -> Tensor:
        scalars = torch.cat(
            [h[..., b.offset : b.end] for b in self.layout.scalar_blocks],
            dim=-1,
        )
        return torch.cat([scalars, self.layout.gather_nonscalar_norms(h)], dim=-1)

    def _append_dynamic_contacts(
        self,
        batch: dict[str, Tensor],
        coords: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        edge_index = batch["edge_index"]
        edge_type = batch["edge_type"]
        edge_bond_type = batch["edge_bond_type"]
        edge_bond_conjugated = batch["edge_bond_conjugated"]
        edge_bond_in_ring = batch["edge_bond_in_ring"]
        edge_bond_stereo = batch["edge_bond_stereo"]
        edge_ref_dist = batch["edge_ref_dist"]
        edge_frag_hop = batch["edge_frag_hop"]
        if self.contact_cutoff <= 0:
            return (
                edge_index,
                edge_type,
                edge_bond_type,
                edge_bond_conjugated,
                edge_bond_in_ring,
                edge_bond_stereo,
                edge_ref_dist,
                edge_frag_hop,
            )

        node_type = batch["node_type"]
        prot_idx = (node_type == NTYPE_PROT_ATOM).nonzero(as_tuple=True)[0]
        lig_idx = (node_type == NTYPE_LIG_ATOM).nonzero(as_tuple=True)[0]
        if prot_idx.numel() == 0 or lig_idx.numel() == 0:
            return (
                edge_index,
                edge_type,
                edge_bond_type,
                edge_bond_conjugated,
                edge_bond_in_ring,
                edge_bond_stereo,
                edge_ref_dist,
                edge_frag_hop,
            )

        dists = torch.cdist(coords.index_select(0, prot_idx), coords.index_select(0, lig_idx))
        same_sample = batch["batch"][prot_idx].unsqueeze(1) == batch["batch"][lig_idx].unsqueeze(0)
        pi, li = ((dists <= self.contact_cutoff) & same_sample).nonzero(as_tuple=True)
        if pi.numel() == 0:
            return (
                edge_index,
                edge_type,
                edge_bond_type,
                edge_bond_conjugated,
                edge_bond_in_ring,
                edge_bond_stereo,
                edge_ref_dist,
                edge_frag_hop,
            )

        c_src = torch.cat([prot_idx[pi], lig_idx[li]])
        c_dst = torch.cat([lig_idx[li], prot_idx[pi]])
        nc = c_src.numel()
        return (
            torch.cat([edge_index, torch.stack([c_src, c_dst])], dim=1),
            torch.cat([edge_type, edge_type.new_full((nc,), ETYPE_DYNAMIC_CONTACT)]),
            torch.cat([edge_bond_type, edge_bond_type.new_full((nc,), -1)]),
            torch.cat([edge_bond_conjugated, edge_bond_conjugated.new_zeros(nc)]),
            torch.cat([edge_bond_in_ring, edge_bond_in_ring.new_zeros(nc)]),
            torch.cat([edge_bond_stereo, edge_bond_stereo.new_full((nc,), -1)]),
            torch.cat([edge_ref_dist, edge_ref_dist.new_zeros(nc)]),
            torch.cat([edge_frag_hop, edge_frag_hop.new_full((nc,), -1)]),
        )

    def _build_pose_batch(
        self, item: dict[str, Tensor]
    ) -> tuple[dict[str, Tensor], Tensor, Tensor]:
        param = next(self.parameters())
        device = param.device
        dtype = param.dtype
        graph = {k: v.to(device=device) for k, v in item["graph"].items() if torch.is_tensor(v)}
        poses = item["pose_atom_coords"].to(device=device, dtype=dtype)
        K, n_atoms, _ = poses.shape
        n_frags = int(item["frag_sizes"].numel())
        n_nodes = int(graph["node_coords"].shape[0])

        batch = build_batched_graph(graph, K, n_frags, device)
        node_coords = batch["node_coords"].to(dtype=dtype).clone()
        frag_start, frag_end = int(graph["lig_frag_slice"][0]), int(graph["lig_frag_slice"][1])
        atom_start = int(graph["lig_atom_slice"][0])
        frag_slots = torch.cat(
            [torch.arange(frag_start, frag_end, device=device) + k * n_nodes for k in range(K)]
        )
        atom_slots = torch.cat(
            [
                torch.arange(atom_start, atom_start + n_atoms, device=device) + k * n_nodes
                for k in range(K)
            ]
        )
        frag_id = item["fragment_id"].to(device=device).long()
        pose_frag_T = poses.new_zeros(K, n_frags, 3)
        pose_frag_T.scatter_add_(1, frag_id.view(1, n_atoms, 1).expand(K, n_atoms, 3), poses)
        frag_counts = (
            torch.bincount(frag_id, minlength=n_frags).clamp_min(1).to(device=device, dtype=dtype)
        )
        pose_frag_T = pose_frag_T / frag_counts.view(1, n_frags, 1)
        node_coords[frag_slots] = pose_frag_T.reshape(K * n_frags, 3)
        node_coords[atom_slots] = poses.reshape(K * n_atoms, 3)
        batch["node_coords"] = node_coords
        batch["T_frag"] = pose_frag_T.reshape(K * n_frags, 3)
        batch["frag_sizes"] = item["frag_sizes"].to(device=device).repeat(K)
        batch["frag_id_for_atoms"] = (
            frag_id.repeat(K) + torch.arange(K, device=device).repeat_interleave(n_atoms) * n_frags
        )
        return batch, poses, graph["node_type"].to(device=device)

    def _initial_hidden(self, item: dict[str, Tensor], batch: dict[str, Tensor]) -> Tensor:
        param = next(self.parameters())
        device = param.device
        dtype = param.dtype
        coords = batch["node_coords"].to(dtype=dtype)
        K = int(item["pose_atom_coords"].shape[0])
        n_nodes = coords.shape[0]
        cond_pose = coords.new_zeros(K, self.cond_dim)
        cond_nodes = cond_pose.index_select(0, batch["batch"])

        h_scalar = self.node_emb(batch)
        frag_idx = (batch["node_type"] == NTYPE_FRAGMENT).nonzero(as_tuple=True)[0]
        size_feat = self.frag_size_emb(
            batch["frag_sizes"].clamp(max=self.frag_size_emb.num_embeddings - 1).long()
        )
        h_scalar = h_scalar.clone()
        h_scalar[frag_idx] = self.frag_init_mlp(torch.cat([h_scalar[frag_idx], size_feat], dim=-1))

        center = scatter_mean(coords, batch["batch"], K)
        r = coords - center.index_select(0, batch["batch"])
        gate = self.vec_gate(h_scalar)
        h_1o = (gate.unsqueeze(-1) * r.unsqueeze(1)).reshape(n_nodes, self.flow_hidden_vec_dim * 3)

        h_1e = torch.zeros(n_nodes, self.flow_hidden_vec_dim * 3, device=device, dtype=dtype)
        parts = [h_scalar, h_1o, h_1e]
        if self.flow_l2_dim > 0:
            parts.append(torch.zeros(n_nodes, self.flow_l2_dim * 5, device=device, dtype=dtype))
        if self.flow_l2o_dim > 0:
            parts.append(torch.zeros(n_nodes, self.flow_l2o_dim * 5, device=device, dtype=dtype))
        h = torch.cat(parts, dim=-1)

        if self.use_saved_ligand_hidden and "h_lig_node" in item:
            graph = item["graph"]
            n_base = int(graph["node_coords"].shape[0])
            frag_start, frag_end = int(graph["lig_frag_slice"][0]), int(graph["lig_frag_slice"][1])
            atom_start, atom_end = int(graph["lig_atom_slice"][0]), int(graph["lig_atom_slice"][1])
            frag_slots = torch.cat(
                [torch.arange(frag_start, frag_end, device=device) + k * n_base for k in range(K)]
            )
            atom_slots = torch.cat(
                [torch.arange(atom_start, atom_end, device=device) + k * n_base for k in range(K)]
            )
            h_lig = item["h_lig_node"].to(device=device, dtype=dtype)
            lig_type = item["lig_node_type"].to(device=device)
            h = h.clone()
            h[frag_slots] = h[frag_slots] + self.lig_hidden_gate * h_lig[
                :, lig_type == NTYPE_FRAGMENT
            ].reshape(-1, self.feat_dim)
            h[atom_slots] = h[atom_slots] + self.lig_hidden_gate * h_lig[
                :, lig_type == NTYPE_LIG_ATOM
            ].reshape(-1, self.feat_dim)
        return h, cond_nodes

    def contact_features(
        self,
        pose_atom_coords: Tensor,
        protein_coords: Tensor,
        protein_flags: Tensor,
    ) -> Tensor:
        K, A, _ = pose_atom_coords.shape
        if protein_coords.numel() == 0:
            atom = pose_atom_coords.new_zeros(K, A, self.n_rbf + 1 + 4 + len(PROTEIN_FLAG_KEYS))
            return atom
        d = torch.cdist(pose_atom_coords, protein_coords)
        min_d = d.amin(dim=-1)
        rbf = rbf_encode(min_d.reshape(-1), self.n_rbf, d_min=0.0, d_max=10.0).view(K, A, -1)
        count_2 = (d < 2.0).float().sum(dim=-1, keepdim=True)
        count_35 = (d < 3.5).float().sum(dim=-1, keepdim=True)
        count_5 = (d < self.contact_cutoff).float().sum(dim=-1, keepdim=True)
        soft = torch.exp(-d / 2.0)
        soft_sum = soft.sum(dim=-1, keepdim=True)
        weighted_flags = (soft @ protein_flags) / soft_sum.clamp_min(1e-6)
        atom = torch.cat(
            [
                min_d.unsqueeze(-1) / 10.0,
                rbf,
                torch.log1p(torch.cat([count_2, count_35, count_5, soft_sum], dim=-1)),
                weighted_flags,
            ],
            dim=-1,
        )

        return atom

    def forward_complex(self, item: dict[str, Tensor]) -> dict[str, Tensor]:
        batch, poses, base_node_type = self._build_pose_batch(item)
        h, cond_nodes = self._initial_hidden(item, batch)
        coords = batch["node_coords"].to(dtype=h.dtype)
        (
            edge_index,
            edge_type,
            edge_bond_type,
            edge_bond_conjugated,
            edge_bond_in_ring,
            edge_bond_stereo,
            edge_ref_dist,
            edge_frag_hop,
        ) = self._append_dynamic_contacts(batch, coords)
        pair_contact = build_pair_contact_features({**batch, "node_coords": coords}, edge_index)

        for layer in self.layers:
            h = layer(
                h,
                coords,
                edge_index,
                edge_type,
                edge_bond_type,
                edge_bond_conjugated,
                edge_bond_in_ring,
                edge_bond_stereo,
                edge_ref_dist,
                edge_frag_hop,
                cond_nodes,
                R_frag=None,
                node_frag_id=batch["node_fragment_id"],
                pair_contact_features=pair_contact,
            )

        K = poses.shape[0]
        n_base = base_node_type.shape[0]
        h_b = h.view(K, n_base, self.feat_dim)
        coords_b = coords.view(K, n_base, 3)
        inv = self.hidden_invariants(h_b)

        atom_mask = base_node_type == NTYPE_LIG_ATOM
        frag_mask = base_node_type == NTYPE_FRAGMENT
        prot_atom_mask = base_node_type == NTYPE_PROT_ATOM
        prot_res_mask = base_node_type == NTYPE_PROT_RES
        protein_flags_base = torch.stack(
            [
                batch["node_patom_is_backbone"][:n_base].float(),
                batch["node_patom_is_metal"][:n_base].float(),
                batch["node_patom_is_donor"][:n_base].float(),
                batch["node_patom_is_acceptor"][:n_base].float(),
                batch["node_patom_is_positive"][:n_base].float(),
                batch["node_patom_is_negative"][:n_base].float(),
                batch["node_patom_is_hydrophobic"][:n_base].float(),
            ],
            dim=-1,
        ).to(device=coords.device, dtype=coords.dtype)
        protein_flags = protein_flags_base[prot_atom_mask].unsqueeze(0).expand(K, -1, -1)
        protein_coords = coords_b[:, prot_atom_mask]
        contact_atom = self.contact_features(poses, protein_coords, protein_flags)

        atom_inv = inv[:, atom_mask]
        atom_in = self.atom_norm(torch.cat([atom_inv, contact_atom], dim=-1))
        atom_z = self.atom_trunk(atom_in)
        atom_out = self.atom_head(atom_z)
        atom_disp_log1p = atom_out[..., 0]
        atom_ok_logit = atom_out[..., 1]

        if self.pose_readout in {"contact_attention", "global_contact_attention"}:
            contact_in = self.contact_readout_norm(torch.cat([atom_z, contact_atom], dim=-1))
            contact_z = self.contact_readout_proj(contact_in)
            attn = torch.softmax(self.contact_readout_score(contact_z).squeeze(-1), dim=-1)
            attn_pool = (attn.unsqueeze(-1) * contact_z).sum(dim=1)
            contact_max = contact_z.amax(dim=1)
            atom_z_mean = atom_z.mean(dim=1)
            atom_z_max = atom_z.amax(dim=1)
            contact_pose_in = torch.cat([attn_pool, contact_max, atom_z_mean, atom_z_max], dim=-1)
        if self.pose_readout in {"global_pool", "global_contact_attention"}:
            node_z = self.node_proj(self.node_norm(inv))
            atom_mean, atom_max = _pool(node_z, atom_mask)
            frag_mean, frag_max = _pool(node_z, frag_mask)
            prot_atom_mean, prot_atom_max = _pool(node_z, prot_atom_mask)
            prot_res_mean, prot_res_max = _pool(node_z, prot_res_mask)
            global_pose_in = torch.cat(
                [
                    atom_mean,
                    atom_max,
                    frag_mean,
                    frag_max,
                    prot_atom_mean,
                    prot_atom_max,
                    prot_res_mean,
                    prot_res_max,
                ],
                dim=-1,
            )
        if self.pose_readout == "contact_attention":
            pose_out = self.contact_pose_head(self.contact_pose_norm(contact_pose_in))
        elif self.pose_readout == "global_contact_attention":
            pose_in = torch.cat([global_pose_in, contact_pose_in], dim=-1)
            pose_out = self.hybrid_pose_head(self.hybrid_pose_norm(pose_in))
        else:
            pose_out = self.pose_head(self.pose_norm(global_pose_in))
        pose_rmsd_log1p = pose_out[:, 0]
        pose_success_logit = pose_out[:, 1]
        return {
            "atom_disp_log1p": atom_disp_log1p,
            "atom_ok_logit": atom_ok_logit,
            "pose_rmsd_log1p": pose_rmsd_log1p,
            "pose_success_logit": pose_success_logit,
            "pose_rmsd": torch.expm1(pose_rmsd_log1p.clamp(-2.0, 5.0)).clamp_min(0.0),
        }


__all__ = ["DockingGraphPoseConfidence"]
