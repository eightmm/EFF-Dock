"""EFFDock equivariant model: single GNN over all node/edge types.

All nodes (ligand atoms, fragments, protein atoms, residue virtuals)
live in the same irreps space and exchange messages through edge-type-specific
tensor product convolutions.
"""

from __future__ import annotations

import cuequivariance as cue
import cuequivariance_torch as cuet
import torch
import torch.nn as nn
from torch import Tensor

from ..geometry.se3 import quaternion_to_matrix
from ..preprocess.protein import NUM_ATOM_TOKENS, NUM_RES_TYPES
from .equivariant import (
    EquivariantActivation,
    EquivariantAdaLN,
    EquivariantBlock,
    EquivariantRMSNorm,
    GatedEquivariantConv,
)
from .nn_utils import rbf_encode, scatter_mean, sinusoidal_embedding


def _graph_bool(
    graph: dict[str, Tensor],
    key: str,
    n_nodes: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    value = graph.get(key)
    if value is None:
        return torch.zeros(n_nodes, device=device, dtype=dtype)
    return value.to(device=device, dtype=dtype)


def _graph_int(
    graph: dict[str, Tensor],
    key: str,
    n_nodes: int,
    device: torch.device,
) -> Tensor:
    value = graph.get(key)
    if value is None:
        return torch.zeros(n_nodes, device=device, dtype=torch.long)
    return value.to(device=device, dtype=torch.long)


def build_pair_contact_features(graph: dict[str, Tensor], edge_index: Tensor) -> Tensor:
    """Return protein-ligand pair chemistry flags for each edge.

    Features are symmetric and non-zero only when one endpoint carries ligand
    atom flags and the other carries protein atom flags.
    """
    n_nodes = graph["node_coords"].shape[0]
    device = edge_index.device
    dtype = graph["node_coords"].dtype

    lig_donor = _graph_bool(graph, "node_is_donor", n_nodes, device, dtype)
    lig_acceptor = _graph_bool(graph, "node_is_acceptor", n_nodes, device, dtype)
    lig_pos = _graph_bool(graph, "node_is_positive", n_nodes, device, dtype)
    lig_neg = _graph_bool(graph, "node_is_negative", n_nodes, device, dtype)
    lig_hydro = _graph_bool(graph, "node_is_hydrophobe", n_nodes, device, dtype)
    lig_halogen = _graph_bool(graph, "node_is_halogen", n_nodes, device, dtype)

    prot_donor = _graph_bool(graph, "node_patom_is_donor", n_nodes, device, dtype)
    prot_acceptor = _graph_bool(graph, "node_patom_is_acceptor", n_nodes, device, dtype)
    prot_pos = _graph_bool(graph, "node_patom_is_positive", n_nodes, device, dtype)
    prot_neg = _graph_bool(graph, "node_patom_is_negative", n_nodes, device, dtype)
    prot_hydro = _graph_bool(graph, "node_patom_is_hydrophobic", n_nodes, device, dtype)
    prot_metal = _graph_bool(graph, "node_patom_is_metal", n_nodes, device, dtype)

    src, dst = edge_index

    def pair(a: Tensor, b: Tensor) -> Tensor:
        return (a[src] * b[dst] + a[dst] * b[src]).clamp(max=1.0)

    hbond = (pair(lig_donor, prot_acceptor) + pair(lig_acceptor, prot_donor)).clamp(max=1.0)
    salt_bridge = (pair(lig_pos, prot_neg) + pair(lig_neg, prot_pos)).clamp(max=1.0)
    charge_clash = (pair(lig_pos, prot_pos) + pair(lig_neg, prot_neg)).clamp(max=1.0)
    hydrophobe = pair(lig_hydro, prot_hydro)
    halogen_acceptor = pair(lig_halogen, prot_acceptor)
    metal_ligand = (pair(lig_acceptor, prot_metal) + pair(lig_neg, prot_metal)).clamp(max=1.0)

    return torch.stack(
        [
            hbond,
            salt_bridge,
            charge_clash,
            hydrophobe,
            halogen_acceptor,
            metal_ligand,
        ],
        dim=-1,
    )


def _build_node_irreps(scalar_dim: int, vec_dim: int, l2_dim: int, l2o_dim: int = 0) -> cue.Irreps:
    """Build node irreps string, omitting 2e/2o if dims are 0."""
    parts = [f"{scalar_dim}x0e", f"{vec_dim}x1o", f"{vec_dim}x1e"]
    if l2_dim > 0:
        parts.append(f"{l2_dim}x2e")
    if l2o_dim > 0:
        parts.append(f"{l2o_dim}x2o")
    return cue.Irreps("O3", " + ".join(parts))


# ---------------------------------------------------------------------------
# Newton-Euler aggregation: atom forces → fragment (v, omega)
# ---------------------------------------------------------------------------


def _stable_symmetric_eigh(matrix: Tensor) -> tuple[Tensor, Tensor]:
    """Eigendecompose small symmetric matrices robustly in float64.

    CUDA's batched symmetric eigensolver can occasionally fail to converge on
    singular or nearly degenerate 3x3 inertia tensors.  Preserve the same
    mathematical operation while making its numerical contract explicit:
    validate finite inputs, remove round-off asymmetry, use float64, and retry
    convergence failures on the CPU.  Other runtime errors are not hidden.
    """
    if matrix.shape[-2:] != (3, 3):
        raise ValueError(f"expected [..., 3, 3] symmetric matrices, got {matrix.shape}")

    work = matrix.detach().to(dtype=torch.float64)
    work = 0.5 * (work + work.transpose(-1, -2))
    if not bool(torch.isfinite(work).all()):
        bad = int((~torch.isfinite(work)).sum().item())
        raise FloatingPointError(f"inertia tensor contains {bad} non-finite values")

    try:
        return torch.linalg.eigh(work)
    except RuntimeError as exc:
        message = str(exc).lower()
        convergence_failure = "failed to converge" in message or "ill-conditioned" in message
        if work.device.type != "cuda" or not convergence_failure:
            raise
        eigenvalues_cpu, eigenvectors_cpu = torch.linalg.eigh(work.cpu())
        return eigenvalues_cpu.to(work.device), eigenvectors_cpu.to(work.device)


def newton_euler_aggregate(
    f_atom: Tensor,
    atom_pos: Tensor,
    T_frag: Tensor,
    frag_id: Tensor,
    n_frag: int,
    frag_sizes: Tensor,
    eig_threshold: float = 0.01,
) -> tuple[Tensor, Tensor, Tensor]:
    """Aggregate per-atom forces to fragment translation/angular velocities.

    Uses eigendecomposition of the inertia tensor instead of direct solve,
    so rank-deficient fragments (2-atom, collinear) get zero angular velocity
    on their unobservable rotation axes. Returns a projection matrix ``P``
    that the loss should apply to ``omega_target`` to match.

    Args:
        f_atom: Per-atom force vectors [N_atom, 3].
        atom_pos: Atom positions at time t [N_atom, 3].
        T_frag: Fragment centers [N_frag, 3].
        frag_id: Fragment assignment per atom [N_atom].
        n_frag: Number of fragments.
        frag_sizes: Fragment sizes [N_frag].
        eig_threshold: Relative threshold for observable eigenvalues.
            An eigenvalue below ``eig_threshold * max_eigenvalue`` is
            considered unobservable (rotation around that axis is
            undefined or nearly so).

    Returns:
        v_frag: Translation velocity [N_frag, 3].
        omega_frag: Angular velocity [N_frag, 3] — zero on unobservable axes.
        P_observable: Projection matrix [N_frag, 3, 3] into the observable
            rotation subspace. Apply to omega_target before computing loss.
    """
    device, dtype = f_atom.device, f_atom.dtype

    # Translation: mean force per fragment
    v_frag = scatter_mean(f_atom, frag_id, n_frag)

    # Lever arm: atom position relative to fragment center
    r_arm = atom_pos - T_frag[frag_id]  # [N_atom, 3]

    # Torque: sum of cross(r, f) per fragment
    torque_per_atom = torch.cross(r_arm, f_atom, dim=-1)  # [N_atom, 3]
    torque = torch.zeros(n_frag, 3, device=device, dtype=dtype)
    torque.scatter_add_(0, frag_id.unsqueeze(-1).expand_as(torque_per_atom), torque_per_atom)

    # Inertia tensor per fragment: I = sum(|r|^2 * I_3 - r ⊗ r).
    # Construct it in float64: nearly collinear and symmetric fragments can
    # lose the small eigenvalue entirely in float32 before eigh even starts.
    r_arm_work = r_arm.to(dtype=torch.float64)
    r2 = (r_arm_work * r_arm_work).sum(-1, keepdim=True)  # [N_atom, 1]
    I_diag = torch.zeros(n_frag, 1, device=device, dtype=torch.float64)
    I_diag.scatter_add_(0, frag_id.unsqueeze(-1), r2)
    I_diag = I_diag.expand(-1, 3)  # [N_frag, 3]

    rr = r_arm_work.unsqueeze(-1) * r_arm_work.unsqueeze(-2)  # [N_atom, 3, 3]
    I_off = torch.zeros(n_frag, 3, 3, device=device, dtype=torch.float64)
    I_off.scatter_add_(0, frag_id.view(-1, 1, 1).expand_as(rr), rr)

    I_tensor = torch.diag_embed(I_diag) - I_off  # [N_frag, 3, 3]

    # Single-atom fragments have a zero inertia tensor and no observable
    # rotation.  Exclude them from the eigensolver rather than relying on a
    # backend to choose an arbitrary basis for a triply-degenerate matrix.
    active = frag_sizes > 1
    active_idx = active.nonzero(as_tuple=True)[0]
    eigenvalues = torch.zeros(n_frag, 3, device=device, dtype=torch.float64)
    eigenvectors = torch.eye(3, device=device, dtype=torch.float64).expand(n_frag, -1, -1).clone()
    if active_idx.numel() > 0:
        active_values, active_vectors = _stable_symmetric_eigh(
            I_tensor.index_select(0, active_idx)
        )
        eigenvalues.index_copy_(0, active_idx, active_values)
        eigenvectors.index_copy_(0, active_idx, active_vectors)

    # Observable mask: eigenvalue > threshold * max_eigenvalue_per_fragment
    max_eig = eigenvalues.max(dim=-1, keepdim=True).values.clamp(min=1e-8)
    observable = (eigenvalues > eig_threshold * max_eig).to(
        dtype=eigenvalues.dtype
    )  # [N_frag, 3]

    # Single-atom fragments: all axes unobservable
    single_mask = (~active).unsqueeze(-1)  # [N_frag, 1]
    observable = observable.masked_fill(single_mask, 0.0)

    # Pseudo-inverse solve in eigenspace: ω_k = τ_k / λ_k on observable axes
    torque_work = torque.to(dtype=torch.float64)
    torque_eig = torch.einsum("nji,nj->ni", eigenvectors, torque_work)  # [N_frag, 3]
    safe_eig = eigenvalues.clamp(min=1e-6)
    omega_eig = torque_eig / safe_eig * observable  # [N_frag, 3]

    # Reconstruct in global frame
    omega_frag = torch.einsum("nij,nj->ni", eigenvectors, omega_eig).to(dtype=dtype)

    # Projection matrix: P = V @ diag(mask) @ V^T
    # Used by loss to project omega_target into the same observable subspace.
    P_observable = torch.einsum(
        "nij,nj,nkj->nik",
        eigenvectors,
        observable,
        eigenvectors,
    ).to(dtype=dtype)  # [N_frag, 3, 3]

    return v_frag, omega_frag, P_observable


# ---------------------------------------------------------------------------
# Node / edge type constants — single source of truth in graph_types.
# ---------------------------------------------------------------------------
from ..preprocess.graph_types import (  # noqa: E402
    ETYPE_DYNAMIC_CONTACT,
    NTYPE_FRAGMENT,
    NTYPE_LIG_ATOM,
    NTYPE_PROT_ATOM,
    NUM_EDGE_TYPES,
    NUM_NODE_TYPES,
)

# ---------------------------------------------------------------------------
# Unified node embedding
# ---------------------------------------------------------------------------


class EFFDockNodeEmbedding(nn.Module):
    """Embed all node types into a shared scalar space.

    Ligand nodes use chemistry/topology features. Protein atom nodes use
    (residue, atom_name) token plus coarse structural/pharmacophore flags.
    Both share node_type and residue_type embeddings. Non-applicable slots are
    padded with sentinels in graph construction.
    """

    def __init__(self, hidden_dim: int = 256) -> None:
        super().__init__()
        # Ligand chemistry (padded with sentinels for non-ligand nodes)
        self.elem_emb = nn.Embedding(13, 32)  # C..Se,OTHER
        self.charge_proj = nn.Linear(1, 8)
        self.aromatic_emb = nn.Embedding(2, 8)
        self.hybrid_emb = nn.Embedding(7, 16)  # SP..SP3D2,UNSPEC,OTHER
        self.num_rings_emb = nn.Embedding(5, 8)  # 0..4 rings
        self.degree_emb = nn.Embedding(7, 8)  # 0..6
        self.impl_valence_emb = nn.Embedding(8, 6)  # 0..7
        self.expl_valence_emb = nn.Embedding(8, 6)  # 0..7
        self.chirality_emb = nn.Embedding(5, 8)  # none/CW/CCW/other
        self.bool_proj = nn.Linear(6, 16)  # donor/acceptor/pos/neg/hydro/halogen
        # subtotal: 32+8+8+16+8+8+6+6+8+16 = 116

        # Protein atom identity token (padded with UNK for non-protein)
        self.patom_token_emb = nn.Embedding(NUM_ATOM_TOKENS, 32)  # 32
        self.patom_struct_proj = nn.Linear(2, 8)  # backbone/metal

        # Shared across all node types
        self.type_emb = nn.Embedding(NUM_NODE_TYPES, 16)  # 16
        self.res_type_emb = nn.Embedding(NUM_RES_TYPES, 16)  # 16
        self.pres_struct_proj = nn.Linear(1, 4)  # pseudo-CB flag

        # Protein-atom pharmacophore (schema_v2). Mirrors ligand bool_proj —
        # zero for non-protein nodes (lig_atom / frag / pres). Initialized
        # near zero so old checkpoints can warm-start without disturbing
        # the existing patom token embedding.
        self.patom_bool_proj = nn.Linear(5, 16)
        nn.init.zeros_(self.patom_bool_proj.weight)
        nn.init.zeros_(self.patom_bool_proj.bias)

        # Total: 116 + 32 + 8 + 16 + 16 + 4 + 16 = 208
        self.proj = nn.Sequential(
            nn.Linear(208, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, graph: dict[str, Tensor]) -> Tensor:
        n_nodes = graph["node_coords"].shape[0]
        device = graph["node_coords"].device
        dtype = graph["node_coords"].dtype

        h_elem = self.elem_emb(graph["node_element"].clamp(max=12))
        h_charge = self.charge_proj(graph["node_charge"].float().unsqueeze(-1))
        h_aromatic = self.aromatic_emb(graph["node_aromatic"].long())
        h_hybrid = self.hybrid_emb(graph["node_hybridization"].clamp(0, 6).long())
        h_rings = self.num_rings_emb(graph["node_num_rings"].clamp(0, 4).long())
        h_degree = self.degree_emb(
            _graph_int(graph, "node_atom_degree", n_nodes, device).clamp(0, 6)
        )
        h_impl_valence = self.impl_valence_emb(
            _graph_int(graph, "node_atom_implicit_valence", n_nodes, device).clamp(0, 7)
        )
        h_expl_valence = self.expl_valence_emb(
            _graph_int(graph, "node_atom_explicit_valence", n_nodes, device).clamp(0, 7)
        )
        h_chirality = self.chirality_emb(
            _graph_int(graph, "node_atom_chirality", n_nodes, device).clamp(0, 4)
        )

        bools = torch.stack(
            [
                graph["node_is_donor"].float(),
                graph["node_is_acceptor"].float(),
                graph["node_is_positive"].float(),
                graph["node_is_negative"].float(),
                graph["node_is_hydrophobe"].float(),
                _graph_bool(graph, "node_is_halogen", n_nodes, device, dtype),
            ],
            dim=-1,
        )
        h_bool = self.bool_proj(bools)

        h_patom = self.patom_token_emb(graph["node_patom_token"].clamp(max=NUM_ATOM_TOKENS - 1))
        patom_struct = torch.stack(
            [
                _graph_bool(graph, "node_patom_is_backbone", n_nodes, device, dtype),
                _graph_bool(graph, "node_patom_is_metal", n_nodes, device, dtype),
            ],
            dim=-1,
        )
        h_patom_struct = self.patom_struct_proj(patom_struct)
        h_type = self.type_emb(graph["node_type"])
        h_res = self.res_type_emb(graph["node_pres_residue_type"].clamp(max=NUM_RES_TYPES - 1))
        pres_struct = _graph_bool(
            graph,
            "node_pres_is_pseudo",
            n_nodes,
            device,
            dtype,
        ).unsqueeze(-1)
        h_pres_struct = self.pres_struct_proj(pres_struct)

        # Protein-atom pharmacophore (schema_v2). Falls back to all-zero when
        # an older protein.pt without these fields is loaded — keeps backward
        # compatibility with PDBbind-era checkpoints.
        if "node_patom_is_donor" in graph:
            patom_bools = torch.stack(
                [
                    graph["node_patom_is_donor"].float(),
                    graph["node_patom_is_acceptor"].float(),
                    graph["node_patom_is_positive"].float(),
                    graph["node_patom_is_negative"].float(),
                    graph["node_patom_is_hydrophobic"].float(),
                ],
                dim=-1,
            )
        else:
            patom_bools = torch.zeros(
                h_elem.shape[0],
                5,
                dtype=h_elem.dtype,
                device=h_elem.device,
            )
        h_patom_bool = self.patom_bool_proj(patom_bools)

        h = torch.cat(
            [
                h_elem,
                h_charge,
                h_aromatic,
                h_hybrid,
                h_rings,
                h_degree,
                h_impl_valence,
                h_expl_valence,
                h_chirality,
                h_bool,
                h_patom,
                h_patom_struct,
                h_type,
                h_res,
                h_pres_struct,
                h_patom_bool,
            ],
            dim=-1,
        )
        return self.proj(h)


# ---------------------------------------------------------------------------
# Unified interaction layer
# ---------------------------------------------------------------------------


class EFFDockInteractionLayer(nn.Module):
    """Single equivariant interaction layer over the unified graph.

    Uses ONE TP convolution for all edge types.  Edge-type specialization
    comes from the edge scalar features: edge_type embedding + bond-specific
    features (type, conjugation, ring, stereo) + RBF distance + node scalars.
    """

    NUM_BOND_TYPES = 5
    # Matches BOND_STEREO_MAP in src/preprocess/ligand.py (STEREONONE, STEREOANY,
    # STEREOZ, STEREOE, STEREOCIS, STEREOTRANS = 0..5). Previously 4 — truncated
    # E/cis/trans stereos onto wrap-around indices and silently corrupted the
    # embedding lookup for any bond with CIS/TRANS designation.
    NUM_BOND_STEREO = 6

    def __init__(
        self,
        scalar_dim: int,
        vec_dim: int,
        l2_dim: int,
        t_emb_dim: int,
        n_edge_types: int = NUM_EDGE_TYPES,
        n_rbf: int = 16,
        sh_lmax: int = 2,
        dropout: float = 0.0,
        l2o_dim: int = 0,
    ) -> None:
        super().__init__()
        self.scalar_dim = scalar_dim
        self.n_rbf = n_rbf

        node_irreps = _build_node_irreps(scalar_dim, vec_dim, l2_dim, l2o_dim)

        # Edge type embedding
        self.edge_type_emb = nn.Embedding(n_edge_types, 16)

        # Bond feature embeddings (only active for bond-type edges)
        self.bond_type_emb = nn.Embedding(self.NUM_BOND_TYPES + 1, 8)  # +1 for "no bond"
        self.bond_conj_emb = nn.Embedding(3, 4)  # 0=no, 1=yes, 2=N/A
        self.bond_ring_emb = nn.Embedding(3, 4)  # 0=no, 1=yes, 2=N/A
        self.bond_stereo_emb = nn.Embedding(self.NUM_BOND_STEREO + 1, 4)  # +1 for "N/A"

        # Pair distance evolution: (d_t, d_t - d_ref, log(d_t / d_ref), has_ref)
        self.dist_evolve_proj = nn.Linear(4, 16)

        # Fragment topological hop distance (frag_frag edges only; -1 → 0 for others)
        self.frag_hop_emb = nn.Embedding(17, 8)  # 0=N/A, 1..16=hop count

        # Local-frame projection: R_src^T @ Δx → 3 SO(3)-invariant scalars → 16-dim embed
        self.local_frame_proj = nn.Linear(3, 16)

        # Protein-ligand pair chemistry on contact edges:
        # H-bond, salt bridge, charge clash, hydrophobe, halogen, metal.
        self.pair_contact_proj = nn.Linear(6, 16)

        # Edge scalar dim:
        #   rbf(n_rbf) + edge_type(16) + bond(8+4+4+4) + dist_evolve(16) + frag_hop(8)
        #   + local_frame(16) + pair_contact(16) + src_scalar + dst_scalar + t_emb
        edge_scalar_dim = n_rbf + 16 + 20 + 16 + 8 + 16 + 16 + scalar_dim + scalar_dim + t_emb_dim

        # Pre-norm: normalize h before conv (stable-scale messages)
        self.pre_norm = EquivariantRMSNorm(node_irreps)

        # Single unified TP conv
        self.conv = GatedEquivariantConv(
            node_irreps,
            sh_lmax=sh_lmax,
            edge_scalar_dim=edge_scalar_dim,
            n_edge_types=n_edge_types,
        )

        # Post-aggregation: Linear → Activation → Dropout (residual is in forward)
        self.post_block = EquivariantBlock(node_irreps, node_irreps, dropout=dropout)
        # Post-norm + time conditioning (AdaLN includes EquivariantRMSNorm internally)
        self.ada_ln = EquivariantAdaLN(node_irreps, t_emb_dim)

    def forward(
        self,
        h: Tensor,
        coords: Tensor,
        edge_index: Tensor,
        edge_type: Tensor,
        edge_bond_type: Tensor,
        edge_bond_conjugated: Tensor,
        edge_bond_in_ring: Tensor,
        edge_bond_stereo: Tensor,
        edge_ref_dist: Tensor,
        edge_frag_hop: Tensor,
        t_emb: Tensor,
        R_frag: Tensor | None = None,
        node_frag_id: Tensor | None = None,
        pair_contact_features: Tensor | None = None,
    ) -> Tensor:
        n = h.shape[0]
        S = self.scalar_dim

        # Pre-norm: conv and edge scalars see normalized features (stable scale)
        h_normed = self.pre_norm(h)
        h_s = h_normed[:, :S]

        src = edge_index[0]
        dst = edge_index[1]

        diff = coords[dst] - coords[src]
        dist = torch.linalg.vector_norm(diff, dim=-1)

        # Build edge scalar features
        # Preprocessing convention: -1 = "not a bond edge" sentinel.
        # Model convention: index 0 = "no bond / N/A", so shift -1 → 0.
        # For conjugated/in_ring (stored as bool), non-bond edges get 2 (N/A).
        e_type = self.edge_type_emb(edge_type.long())
        is_bond_edge = edge_bond_type >= 0  # True for actual bond edges
        conj_idx = edge_bond_conjugated.long()  # 0=no, 1=yes
        conj_idx = torch.where(
            is_bond_edge, conj_idx, torch.full_like(conj_idx, 2)
        )  # non-bond → 2 (N/A)
        ring_idx = edge_bond_in_ring.long()
        ring_idx = torch.where(is_bond_edge, ring_idx, torch.full_like(ring_idx, 2))
        e_bond = torch.cat(
            [
                self.bond_type_emb(edge_bond_type.long() + 1),  # -1→0(no bond), 0→1, ...
                self.bond_conj_emb(conj_idx),  # 0=no, 1=yes, 2=N/A
                self.bond_ring_emb(ring_idx),  # 0=no, 1=yes, 2=N/A
                self.bond_stereo_emb(edge_bond_stereo.long() + 1),  # -1→0(N/A), 0→1, ...
            ],
            dim=-1,
        )  # [E, 20]

        # Pair distance evolution: (d_t, d_t - d_ref, log(d_t / d_ref), has_ref)
        has_ref = (edge_ref_dist > 0).float()
        delta_d = dist - edge_ref_dist
        safe_ratio = dist / (edge_ref_dist + 1e-6)
        log_ratio = torch.log(safe_ratio.clamp(min=1e-3)) * has_ref
        e_dist_evolve = self.dist_evolve_proj(
            torch.stack(
                [
                    dist,
                    delta_d,
                    log_ratio,
                    has_ref,
                ],
                dim=-1,
            )
        )  # [E, 16]

        # Fragment topological hop: -1 → 0 (N/A), 1..N → clamp to 16
        e_hop = self.frag_hop_emb((edge_frag_hop + 1).clamp(0, 16).long())  # [E, 8]

        # Local-frame projection: R_src^T @ Δx gives 3 SO(3)-invariant
        # scalars describing the neighbor displacement in the source
        # fragment's body frame. We mask non-fragment edges (where R_src
        # would be ill-defined) but no longer apply a separate t-gate —
        # EquivariantAdaLN already provides time-conditioned scaling on
        # all features, so the previous ``× t`` linear gate was redundant.
        if R_frag is not None and node_frag_id is not None:
            src_frag_id = node_frag_id[src]  # [E]
            has_frag = (src_frag_id >= 0).float().unsqueeze(-1)  # [E, 1]
            R_src = R_frag[src_frag_id.clamp(min=0)]  # [E, 3, 3]
            local_disp = torch.einsum("eji,ej->ei", R_src, diff)  # R^T @ diff: [E, 3]
            local_disp = local_disp * has_frag
        else:
            local_disp = torch.zeros(src.shape[0], 3, device=h.device, dtype=h.dtype)
        e_local_frame = self.local_frame_proj(local_disp)  # [E, 16]
        if pair_contact_features is None:
            pair_contact_features = torch.zeros(src.shape[0], 6, device=h.device, dtype=h.dtype)
        e_pair_contact = self.pair_contact_proj(pair_contact_features.to(dtype=h.dtype))

        edge_scalars = torch.cat(
            [
                rbf_encode(dist, self.n_rbf),
                e_type,
                e_bond,
                e_dist_evolve,
                e_hop,
                e_local_frame,
                e_pair_contact,
                h_s[src],
                h_s[dst],
                t_emb[dst],
            ],
            dim=-1,
        )

        msg = self.conv(
            h_normed, diff, edge_scalars, src, dst, n, edge_dist=dist, edge_type=edge_type
        )

        # Residual on ORIGINAL h (pre-norm pattern) + post-norm with time conditioning
        h = h + self.post_block(msg)
        h = self.ada_ln(h, t_emb)

        return h


# ---------------------------------------------------------------------------
# Unified EFFDock model
# ---------------------------------------------------------------------------


class EFFDock(nn.Module):
    """EFFDock equivariant model for fragment-based flow matching.

    All node types share the same irreps space and exchange messages through
    edge-type-specific tensor product convolutions. Output: per-atom forces
    aggregated to fragment (v, omega) via Newton-Euler mechanics.

    Args:
        hidden_dim: Scalar dimension for node embedding.
        hidden_vec_dim: Vector (1o, 1e) channel count.
        l2_dim: Quadrupole (2e) channel count.
        num_layers: Number of interaction layers.
        n_rbf: RBF basis count.
        t_emb_dim: Time embedding dimension.
        sh_lmax: Maximum spherical harmonics degree.
        max_frag_size: Maximum fragment size for size embedding.
        dropout: Dropout probability.
        contact_cutoff: Dynamic protein-ligand contact edge cutoff (0 = disabled).
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        hidden_vec_dim: int = 32,
        l2_dim: int = 16,
        l2o_dim: int = 0,
        num_layers: int = 4,
        n_rbf: int = 16,
        t_emb_dim: int = 128,
        sh_lmax: int = 2,
        max_frag_size: int = 50,
        dropout: float = 0.0,
        contact_cutoff: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.hidden_vec_dim = hidden_vec_dim
        self.l2_dim = l2_dim
        self.l2o_dim = l2o_dim
        self.n_rbf = n_rbf
        self.contact_cutoff = contact_cutoff

        # Node embedding (scalar)
        self.node_emb = EFFDockNodeEmbedding(hidden_dim)

        # Time embedding
        self.t_emb_mlp = nn.Sequential(
            nn.Linear(32, t_emb_dim),
            nn.SiLU(),
            nn.Linear(t_emb_dim, t_emb_dim),
        )
        # Conditional prior-σ embedding (schema_v2). Multi-σ training trains
        # an *unconditional* mixture vector field by default; feeding log(σ)
        # to the same time-conditioning path lets the model learn distinct
        # behaviour per σ instead of averaging across [σ_min, σ_max]. The
        # final Linear is zero-initialised so the contribution is exactly
        # zero at init — older checkpoints without σ in the batch keep
        # working unchanged, and the path activates as it trains.
        self.sigma_emb_mlp = nn.Sequential(
            nn.Linear(32, t_emb_dim),
            nn.SiLU(),
            nn.Linear(t_emb_dim, t_emb_dim),
        )
        nn.init.zeros_(self.sigma_emb_mlp[-1].weight)
        nn.init.zeros_(self.sigma_emb_mlp[-1].bias)

        # Fragment-specific: size embedding
        self.frag_size_emb = nn.Embedding(max_frag_size + 1, 16)
        self.frag_init_mlp = nn.Sequential(
            nn.Linear(hidden_dim + 16 + t_emb_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Vector init: gated displacement → 1o
        self.vec_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_vec_dim),
            nn.Tanh(),
        )

        # R_frag injection: mix 3 R_t columns (1o vectors) into vec_dim
        # channels via a scalar linear combination (preserves equivariance).
        # The earlier `R_frag_gate` (per-channel t-conditional Tanh) was
        # removed — EquivariantAdaLN inside every interaction layer already
        # provides t-conditional scaling on all features, so an additional
        # init-time gate is redundant. The mix weight is zero-initialised
        # so the contribution starts at exactly 0 and warms up only through
        # gradient updates (preserves backward compatibility with older
        # checkpoints that did not see this path).
        self.R_frag_mix = nn.Linear(3, hidden_vec_dim, bias=False)
        nn.init.zeros_(self.R_frag_mix.weight)

        # Interaction layers
        self.layers = nn.ModuleList(
            [
                EFFDockInteractionLayer(
                    scalar_dim=hidden_dim,
                    vec_dim=hidden_vec_dim,
                    l2_dim=l2_dim,
                    t_emb_dim=t_emb_dim,
                    n_rbf=n_rbf,
                    sh_lmax=sh_lmax,
                    dropout=dropout,
                    l2o_dim=l2o_dim,
                )
                for _ in range(num_layers)
            ]
        )

        # Output: atom-level force head → Newton-Euler aggregation
        node_irreps = _build_node_irreps(hidden_dim, hidden_vec_dim, l2_dim, l2o_dim)
        head_irreps = _build_node_irreps(hidden_dim // 2, hidden_vec_dim, l2_dim, l2o_dim)

        self.atom_head_pre = cuet.Linear(
            node_irreps, head_irreps, layout=cue.mul_ir, method="fused_tp"
        )
        self.atom_head_act = EquivariantActivation(head_irreps)
        self.f_atom_linear = cuet.Linear(
            head_irreps, cue.Irreps("O3", "1x1o"), layout=cue.mul_ir, method="fused_tp"
        )

        # Self-TP head: transfers l=2 (and other non-l=1) directional info into
        # the 1o force output via paths Linear cannot express:
        #   1o ⊗ 2e → 1o,  1e ⊗ 2o → 1o,  1o ⊗ 0e → 1o (redundant w/ linear).
        # Without this, l=2 channels only leak into f_atom via norm-gated
        # magnitude modulation in EquivariantActivation (no direction transfer).
        # Zero-init means at step 0 this branch contributes 0, so training
        # starts identical to the linear-only head and gradually picks up the
        # l=2 → 1o contribution as it becomes useful.
        self.f_atom_self_tp = cuet.FullyConnectedTensorProduct(
            head_irreps,
            head_irreps,
            cue.Irreps("O3", "1x1o"),
            layout_in1=cue.mul_ir,
            layout_in2=cue.mul_ir,
            layout_out=cue.mul_ir,
            internal_weights=True,
            shared_weights=True,
        )
        with torch.no_grad():
            for p in self.f_atom_self_tp.parameters():
                p.zero_()

    def forward(self, batch: dict[str, Tensor], return_hidden: bool = False) -> dict[str, Tensor]:
        """Forward pass on a collated batch from EFFDockDataset.

        If ``return_hidden`` is True, also returns the final per-node irrep
        tensor ``h`` (concatenation of scalar + non-scalar blocks, layout
        consistent with ``node_irreps``).
        """
        device = batch["node_coords"].device
        n_nodes = batch["node_coords"].shape[0]
        coords = batch["node_coords"]
        node_type = batch["node_type"]
        edge_index = batch["edge_index"]
        edge_type = batch["edge_type"]

        # Edge features
        edge_bond_type = batch["edge_bond_type"]
        edge_bond_conjugated = batch["edge_bond_conjugated"]
        edge_bond_in_ring = batch["edge_bond_in_ring"]
        edge_bond_stereo = batch["edge_bond_stereo"]
        edge_ref_dist = batch["edge_ref_dist"]
        edge_frag_hop = batch["edge_frag_hop"]

        # Dynamic protein_atom ↔ ligand_atom contact edges (batch-safe)
        if self.contact_cutoff > 0:
            node_batch_idx = batch["batch"]
            prot_idx = (node_type == NTYPE_PROT_ATOM).nonzero(as_tuple=True)[0]
            lig_idx = (node_type == NTYPE_LIG_ATOM).nonzero(as_tuple=True)[0]

            if prot_idx.shape[0] > 0 and lig_idx.shape[0] > 0:
                dists = torch.cdist(coords[prot_idx], coords[lig_idx])
                # Mask cross-sample pairs: only same-sample atoms can form contact edges
                same_sample = node_batch_idx[prot_idx].unsqueeze(1) == node_batch_idx[
                    lig_idx
                ].unsqueeze(0)
                pi, li = ((dists <= self.contact_cutoff) & same_sample).nonzero(as_tuple=True)
                if pi.shape[0] > 0:
                    c_src = torch.cat([prot_idx[pi], lig_idx[li]])
                    c_dst = torch.cat([lig_idx[li], prot_idx[pi]])
                    nc = c_src.shape[0]

                    edge_index = torch.cat([edge_index, torch.stack([c_src, c_dst])], dim=1)
                    edge_type = torch.cat(
                        [edge_type, edge_type.new_full((nc,), ETYPE_DYNAMIC_CONTACT)]
                    )
                    edge_bond_type = torch.cat([edge_bond_type, edge_bond_type.new_full((nc,), -1)])
                    edge_bond_conjugated = torch.cat(
                        [edge_bond_conjugated, edge_bond_conjugated.new_zeros(nc)]
                    )
                    edge_bond_in_ring = torch.cat(
                        [edge_bond_in_ring, edge_bond_in_ring.new_zeros(nc)]
                    )
                    edge_bond_stereo = torch.cat(
                        [edge_bond_stereo, edge_bond_stereo.new_full((nc,), -1)]
                    )
                    edge_ref_dist = torch.cat([edge_ref_dist, edge_ref_dist.new_zeros(nc)])
                    edge_frag_hop = torch.cat([edge_frag_hop, edge_frag_hop.new_full((nc,), -1)])

        frag_mask = node_type == NTYPE_FRAGMENT
        frag_idx = frag_mask.nonzero(as_tuple=True)[0]
        n_frag = frag_idx.shape[0]
        frag_sizes = batch["frag_sizes"]

        # --- Time embedding ---
        t_per_sample = batch["t"].view(-1)
        t_sin = sinusoidal_embedding(t_per_sample, dim=32)
        t_emb = self.t_emb_mlp(t_sin)
        # Add log(prior σ) conditioning when σ is supplied in the batch.
        # Zero-init σ MLP means this contribution starts at 0 and the model
        # warm-starts identical to the non-conditional version. When σ is
        # absent (older preprocess / inference without σ), we silently skip.
        if "prior_sigma" in batch and batch["prior_sigma"] is not None:
            log_sigma = torch.log(batch["prior_sigma"].clamp(min=0.1)).view(-1, 1)
            sigma_sin = sinusoidal_embedding(log_sigma, dim=32)
            t_emb = t_emb + self.sigma_emb_mlp(sigma_sin)
        node_batch_idx = batch["batch"]
        t_emb_nodes = t_emb[node_batch_idx]

        frag_batch_idx = batch["frag_batch"]
        t_emb_frags = t_emb[frag_batch_idx]

        # --- Node scalar embedding ---
        h_scalar = self.node_emb(batch)

        # Fragment-specific init: add size + time.
        # Clamp guards against fragments larger than the embedding table
        # (single aromatic / fused-ring / steroid bases can exceed 30 atoms).
        size_feat = self.frag_size_emb(
            frag_sizes.clamp(max=self.frag_size_emb.num_embeddings - 1).long()
        )
        h_frag_init = self.frag_init_mlp(
            torch.cat([h_scalar[frag_idx], size_feat, t_emb_frags], dim=-1)
        )
        h_scalar = h_scalar.clone()
        h_scalar[frag_idx] = h_frag_init

        # --- Build equivariant state: 0e + 1o + 1e [+ 2e] [+ 2o] ---
        vec_dim = self.hidden_vec_dim
        n_samples = t_per_sample.shape[0]

        # 1o init: gated displacement from per-sample center
        center = scatter_mean(coords, node_batch_idx, n_samples)
        r = coords - center[node_batch_idx]
        gate = self.vec_gate(h_scalar)
        h_1o = (gate.unsqueeze(-1) * r.unsqueeze(1)).reshape(n_nodes, vec_dim * 3)

        # R_frag injection: mix R_t columns into fragment nodes' 1o channels.
        # R_t columns are 1o vectors; scalar linear combination preserves
        # equivariance. AdaLN inside the layers handles the time-dependent
        # scaling of this contribution; no separate init-time gate needed.
        if "q_frag" in batch:
            R_t = quaternion_to_matrix(batch["q_frag"])  # [N_frag, 3, 3]
            # einsum: R_t[:, :, k] = k-th column (1o), mix weight[c, k]
            #   → [N_frag, vec_dim, 3]
            h_R = torch.einsum("nki,ck->nci", R_t, self.R_frag_mix.weight)
            h_R = h_R.reshape(-1, vec_dim * 3)
            h_1o = h_1o.clone()
            h_1o[frag_idx] = h_1o[frag_idx] + h_R

        # 1e zero init
        h_1e = torch.zeros(n_nodes, vec_dim * 3, device=device, dtype=coords.dtype)
        parts = [h_scalar, h_1o, h_1e]
        if self.l2_dim > 0:
            parts.append(torch.zeros(n_nodes, self.l2_dim * 5, device=device, dtype=coords.dtype))
        if self.l2o_dim > 0:
            parts.append(torch.zeros(n_nodes, self.l2o_dim * 5, device=device, dtype=coords.dtype))
        h = torch.cat(parts, dim=-1)

        # --- Precompute local-frame data for interaction layers ---
        R_frag = quaternion_to_matrix(batch["q_frag"]) if "q_frag" in batch else None
        node_frag_id = batch.get("node_fragment_id")
        pair_contact_features = build_pair_contact_features(batch, edge_index)

        # --- Interaction layers ---
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
                t_emb_nodes,
                R_frag=R_frag,
                node_frag_id=node_frag_id,
                pair_contact_features=pair_contact_features,
            )

        # --- Output: atom forces → Newton-Euler aggregation ---
        atom_idx = (node_type == NTYPE_LIG_ATOM).nonzero(as_tuple=True)[0]

        h_atom_head = self.atom_head_pre(h[atom_idx])
        h_atom_head = self.atom_head_act(h_atom_head)
        # f_atom = linear path (1o ⊗ 0e → 1o) + self-TP path (1o ⊗ 2e → 1o, …).
        # Self-TP starts at 0 via zero-init weights (see __init__).
        f_atom = self.f_atom_linear(h_atom_head) + self.f_atom_self_tp(h_atom_head, h_atom_head)

        v_pred, omega_pred, P_observable = newton_euler_aggregate(
            f_atom,
            coords[atom_idx],
            batch["T_frag"],
            batch["frag_id_for_atoms"],
            n_frag,
            frag_sizes,
        )

        result = {
            "v_pred": v_pred,
            "omega_pred": omega_pred,
            "P_observable": P_observable,
        }
        if return_hidden:
            result["h"] = h
        return result


__all__ = ["EFFDock"]
