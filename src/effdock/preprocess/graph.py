"""Static unified graph construction for protein-ligand complexes.

Node schema (four types):
    ligand_atom, ligand_fragment, protein_atom, protein_res

Protein chemistry comes from a single ``patom_token`` (the ``(residue, atom)``
identity); ligand atoms keep their multi-feature chemistry. All per-node
tensors stay concatenated in a fixed order; non-applicable slots are padded
with sentinel values so downstream code can safely index the whole tensor and
gate by ``node_type``.
"""

from collections import deque

import torch

from .graph_types import (  # noqa: F401  (re-exported for callers)
    DYNAMIC_EDGE_TYPES,
    EDGE_TYPES,
    NODE_TYPES,
)
from .ligand import OTHER_ELEMENT_IDX, OTHER_HYBRID_IDX
from .protein import UNK_ATOM_TOKEN, UNK_RES_IDX


def _frag_hop_distances(n_frag: int, adj_index: torch.Tensor) -> torch.Tensor:
    """BFS shortest path between all fragment pairs.

    Returns [n_frag, n_frag] int tensor. Self-distance = 0,
    unreachable = n_frag (sentinel).
    """
    adj: list[list[int]] = [[] for _ in range(n_frag)]
    for k in range(adj_index.shape[1]):
        s, d = int(adj_index[0, k]), int(adj_index[1, k])
        adj[s].append(d)

    dist = torch.full((n_frag, n_frag), n_frag, dtype=torch.int64)
    for src in range(n_frag):
        dist[src, src] = 0
        q: deque[int] = deque([src])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[src, v] > dist[src, u] + 1:
                    dist[src, v] = dist[src, u] + 1
                    q.append(v)
    return dist


def build_static_complex_graph(
    lig_data: dict[str, torch.Tensor],
    patom_data: dict[str, torch.Tensor],
    pres_cutoff: float = 10.0,
) -> dict[str, torch.Tensor]:
    """Build a unified static graph for one complex.

    Protein-ligand interaction edges are intentionally left out so they can be
    rebuilt dynamically from the current flow-matching state.
    """
    lig_atom_coords = lig_data["atom_coords"]
    frag_coords = lig_data["frag_centers"]
    prot_atom_coords = patom_data["patom_coords"]
    pres_coords = patom_data["pres_coords"]

    n_lig_atom = lig_atom_coords.shape[0]
    n_frag = frag_coords.shape[0]
    n_prot_atom = prot_atom_coords.shape[0]
    n_pres = pres_coords.shape[0]

    # Residue virtual nodes inherit their residue_id from the atom they refer
    # to (CB, pseudo-CB with CA sentinel, or metal atom).
    pres_residue_id = patom_data["patom_residue_id"][patom_data["pres_atom_index"]]

    frag_offset = n_lig_atom
    prot_atom_offset = frag_offset + n_frag
    pres_offset = prot_atom_offset + n_prot_atom

    node_coords = torch.cat([lig_atom_coords, frag_coords, prot_atom_coords, pres_coords], dim=0)
    total_nodes = node_coords.shape[0]

    # --- Node type -----------------------------------------------------------
    node_type = torch.cat(
        [
            torch.full((n_lig_atom,), NODE_TYPES["ligand_atom"], dtype=torch.int64),
            torch.full((n_frag,), NODE_TYPES["ligand_fragment"], dtype=torch.int64),
            torch.full((n_prot_atom,), NODE_TYPES["protein_atom"], dtype=torch.int64),
            torch.full((n_pres,), NODE_TYPES["protein_res"], dtype=torch.int64),
        ],
        dim=0,
    )

    # --- Ligand chemistry features (padded for non-ligand slots) -------------
    def _ligand_pad_int64(key: str, pad: int) -> torch.Tensor:
        return torch.cat(
            [
                lig_data[key].to(torch.int64),
                torch.full((n_frag,), pad, dtype=torch.int64),
                torch.full((n_prot_atom,), pad, dtype=torch.int64),
                torch.full((n_pres,), pad, dtype=torch.int64),
            ]
        )

    def _ligand_pad_int8(key: str, pad: int) -> torch.Tensor:
        return torch.cat(
            [
                lig_data[key].to(torch.int8),
                torch.full((n_frag,), pad, dtype=torch.int8),
                torch.full((n_prot_atom,), pad, dtype=torch.int8),
                torch.full((n_pres,), pad, dtype=torch.int8),
            ]
        )

    def _ligand_pad_bool(key: str) -> torch.Tensor:
        return torch.cat(
            [
                lig_data[key].to(torch.bool),
                torch.zeros(n_frag, dtype=torch.bool),
                torch.zeros(n_prot_atom, dtype=torch.bool),
                torch.zeros(n_pres, dtype=torch.bool),
            ]
        )

    node_element = _ligand_pad_int64("atom_element", OTHER_ELEMENT_IDX)
    node_charge = _ligand_pad_int8("atom_charge", 0)
    node_aromatic = _ligand_pad_bool("atom_aromatic")
    node_hybridization = _ligand_pad_int8("atom_hybridization", OTHER_HYBRID_IDX)
    node_num_rings = _ligand_pad_int8("atom_num_rings", 0)
    node_atom_degree = _ligand_pad_int8("atom_degree", 0)
    node_atom_implicit_valence = _ligand_pad_int8("atom_implicit_valence", 0)
    node_atom_explicit_valence = _ligand_pad_int8("atom_explicit_valence", 0)
    node_atom_chirality = _ligand_pad_int8("atom_chirality", 0)
    node_is_donor = _ligand_pad_bool("atom_is_donor")
    node_is_acceptor = _ligand_pad_bool("atom_is_acceptor")
    node_is_positive = _ligand_pad_bool("atom_is_positive")
    node_is_negative = _ligand_pad_bool("atom_is_negative")
    node_is_hydrophobe = _ligand_pad_bool("atom_is_hydrophobe")
    node_is_halogen = _ligand_pad_bool("atom_is_halogen")

    # --- Protein-atom features (padded for non-protein slots) ----------------
    node_patom_token = torch.cat(
        [
            torch.full((n_lig_atom,), UNK_ATOM_TOKEN, dtype=torch.int64),
            torch.full((n_frag,), UNK_ATOM_TOKEN, dtype=torch.int64),
            patom_data["patom_token"],
            torch.full((n_pres,), UNK_ATOM_TOKEN, dtype=torch.int64),
        ]
    )

    def _patom_pad_bool(key: str) -> torch.Tensor:
        if key not in patom_data:
            return torch.zeros(total_nodes, dtype=torch.bool)
        return torch.cat(
            [
                torch.zeros(n_lig_atom, dtype=torch.bool),
                torch.zeros(n_frag, dtype=torch.bool),
                patom_data[key],
                torch.zeros(n_pres, dtype=torch.bool),
            ]
        )

    node_patom_is_backbone = _patom_pad_bool("patom_is_backbone")
    node_patom_is_metal = _patom_pad_bool("patom_is_metal")

    # schema_v2 protein pharmacophore (donor / acceptor / +/-/ hydrophobic).
    # Mirrors the ligand-side flags so the model can pick complementary pairs
    # symmetrically. Zero-padded for non-protein-atom slots.
    node_patom_is_donor = _patom_pad_bool("patom_is_donor")
    node_patom_is_acceptor = _patom_pad_bool("patom_is_acceptor")
    node_patom_is_positive = _patom_pad_bool("patom_is_positive")
    node_patom_is_negative = _patom_pad_bool("patom_is_negative")
    node_patom_is_hydrophobic = _patom_pad_bool("patom_is_hydrophobic")

    # --- Residue virtual node features (padded for non-virtual slots) --------
    node_pres_residue_type = torch.cat(
        [
            torch.full((n_lig_atom,), UNK_RES_IDX, dtype=torch.int64),
            torch.full((n_frag,), UNK_RES_IDX, dtype=torch.int64),
            torch.full((n_prot_atom,), UNK_RES_IDX, dtype=torch.int64),
            patom_data["pres_residue_type"],
        ]
    )
    node_pres_is_pseudo = torch.cat(
        [
            torch.zeros(n_lig_atom, dtype=torch.bool),
            torch.zeros(n_frag, dtype=torch.bool),
            torch.zeros(n_prot_atom, dtype=torch.bool),
            patom_data.get("pres_is_pseudo", torch.zeros(n_pres, dtype=torch.bool)),
        ]
    )

    # --- Shared structural fields --------------------------------------------
    node_fragment_id = torch.cat(
        [
            lig_data["fragment_id"],
            torch.arange(n_frag, dtype=torch.int64),
            torch.full((n_prot_atom,), -1, dtype=torch.int64),
            torch.full((n_pres,), -1, dtype=torch.int64),
        ]
    )
    # --- Edge assembly -------------------------------------------------------
    # Precompute fragment hop distances for frag_frag edge features
    frag_hop_mat = _frag_hop_distances(n_frag, lig_data["fragment_adj_index"])

    edge_src: list[int] = []
    edge_dst: list[int] = []
    edge_type: list[int] = []
    edge_ref_dist: list[float] = []
    edge_bond_type: list[int] = []
    edge_bond_conjugated: list[bool] = []
    edge_bond_in_ring: list[bool] = []
    edge_bond_stereo: list[int] = []
    edge_frag_hop: list[int] = []

    def add_edges(
        src: torch.Tensor,
        dst: torch.Tensor,
        etype: int,
        ref_dist: torch.Tensor | None = None,
        bond_type: torch.Tensor | None = None,
        bond_conj: torch.Tensor | None = None,
        bond_ring: torch.Tensor | None = None,
        bond_stereo: torch.Tensor | None = None,
        frag_hop: torch.Tensor | None = None,
    ) -> None:
        if src.numel() == 0:
            return
        if etype in DYNAMIC_EDGE_TYPES:
            ref_dist = torch.full((src.shape[0],), -1.0)
        elif ref_dist is None:
            ref_dist = torch.linalg.vector_norm(node_coords[src] - node_coords[dst], dim=-1)
        if bond_type is None:
            bond_type = torch.full((src.shape[0],), -1, dtype=torch.int8)
        if bond_conj is None:
            bond_conj = torch.zeros(src.shape[0], dtype=torch.bool)
        if bond_ring is None:
            bond_ring = torch.zeros(src.shape[0], dtype=torch.bool)
        if bond_stereo is None:
            bond_stereo = torch.full((src.shape[0],), -1, dtype=torch.int8)
        if frag_hop is None:
            frag_hop = torch.full((src.shape[0],), -1, dtype=torch.int8)

        edge_src.extend(src.tolist())
        edge_dst.extend(dst.tolist())
        edge_type.extend([etype] * src.shape[0])
        edge_ref_dist.extend(ref_dist.tolist())
        edge_bond_type.extend(bond_type.tolist())
        edge_bond_conjugated.extend(bond_conj.tolist())
        edge_bond_in_ring.extend(bond_ring.tolist())
        edge_bond_stereo.extend(bond_stereo.tolist())
        edge_frag_hop.extend(frag_hop.tolist())

    # Ligand covalent bonds (keeps bond-type features)
    add_edges(
        lig_data["bond_index"][0],
        lig_data["bond_index"][1],
        EDGE_TYPES["ligand_bond"],
        bond_type=lig_data["bond_type"],
        bond_conj=lig_data["bond_conjugated"],
        bond_ring=lig_data["bond_in_ring"],
        bond_stereo=lig_data["bond_stereo"],
    )

    # Ligand triangulation edges (distance-only)
    add_edges(
        lig_data["tri_edge_index"][0],
        lig_data["tri_edge_index"][1],
        EDGE_TYPES["ligand_tri"],
        ref_dist=lig_data["tri_edge_ref_dist"],
    )

    # Explicit cut-bond edges
    cut_index = lig_data["cut_bond_index"]
    if cut_index.numel() > 0:
        cut_src = torch.cat([cut_index[0], cut_index[1]])
        cut_dst = torch.cat([cut_index[1], cut_index[0]])
        add_edges(cut_src, cut_dst, EDGE_TYPES["ligand_cut"])

    # Atom ↔ fragment ownership edges (bidirectional)
    atom_idx = torch.arange(n_lig_atom, dtype=torch.int64)
    frag_idx = frag_offset + lig_data["fragment_id"].to(torch.int64)
    add_edges(atom_idx, frag_idx, EDGE_TYPES["ligand_atom_frag"])
    add_edges(frag_idx, atom_idx, EDGE_TYPES["ligand_atom_frag"])

    # Fragment-fragment dense graph (with topological hop distance)
    if n_frag > 1:
        ii, jj = torch.meshgrid(torch.arange(n_frag), torch.arange(n_frag), indexing="ij")
        mask = ii != jj
        ff_src = ii[mask].flatten()
        ff_dst = jj[mask].flatten()
        ff_hop = frag_hop_mat[ff_src, ff_dst].to(torch.int8)
        add_edges(
            frag_offset + ff_src,
            frag_offset + ff_dst,
            EDGE_TYPES["ligand_frag_frag"],
            frag_hop=ff_hop,
        )

    # Protein covalent bonds (canonical topology; no chemistry features)
    pbond_index = patom_data["pbond_index"]
    if pbond_index.numel() > 0:
        add_edges(
            prot_atom_offset + pbond_index[0],
            prot_atom_offset + pbond_index[1],
            EDGE_TYPES["protein_bond"],
        )

    # Protein atom ↔ residue virtual node hierarchy (bidirectional)
    residue_lookup = {
        residue_id: local_pres for local_pres, residue_id in enumerate(pres_residue_id.tolist())
    }
    atom_src: list[int] = []
    atom_dst: list[int] = []
    for local_atom, residue_id in enumerate(patom_data["patom_residue_id"].tolist()):
        local_pres = residue_lookup.get(residue_id)
        if local_pres is None:
            continue
        atom_src.append(prot_atom_offset + local_atom)
        atom_dst.append(pres_offset + local_pres)
    if atom_src:
        prot_src = torch.tensor(atom_src, dtype=torch.int64)
        prot_dst = torch.tensor(atom_dst, dtype=torch.int64)
        add_edges(prot_src, prot_dst, EDGE_TYPES["protein_atom_res"])
        add_edges(prot_dst, prot_src, EDGE_TYPES["protein_atom_res"])

    # Protein residue virtual-node neighborhood (distance cutoff)
    if n_pres > 1:
        res_dmat = torch.cdist(pres_coords, pres_coords)
        mask = (res_dmat <= pres_cutoff) & (~torch.eye(n_pres, dtype=torch.bool))
        rs, rd = mask.nonzero(as_tuple=True)
        add_edges(
            pres_offset + rs,
            pres_offset + rd,
            EDGE_TYPES["protein_res_res"],
            ref_dist=res_dmat[rs, rd],
        )

    # Protein residue ↔ ligand fragment global edges (bidirectional)
    if n_pres > 0 and n_frag > 0:
        ii, jj = torch.meshgrid(torch.arange(n_pres), torch.arange(n_frag), indexing="ij")
        cf_src = pres_offset + ii.flatten()
        cf_dst = frag_offset + jj.flatten()
        add_edges(cf_src, cf_dst, EDGE_TYPES["protein_res_frag"])
        add_edges(cf_dst, cf_src, EDGE_TYPES["protein_res_frag"])

    return {
        "node_coords": node_coords,
        "node_type": node_type,
        # Ligand chemistry
        "node_element": node_element,
        "node_charge": node_charge,
        "node_aromatic": node_aromatic,
        "node_hybridization": node_hybridization,
        "node_num_rings": node_num_rings,
        "node_atom_degree": node_atom_degree,
        "node_atom_implicit_valence": node_atom_implicit_valence,
        "node_atom_explicit_valence": node_atom_explicit_valence,
        "node_atom_chirality": node_atom_chirality,
        "node_is_donor": node_is_donor,
        "node_is_acceptor": node_is_acceptor,
        "node_is_positive": node_is_positive,
        "node_is_negative": node_is_negative,
        "node_is_hydrophobe": node_is_hydrophobe,
        "node_is_halogen": node_is_halogen,
        # Protein atom token features
        "node_patom_token": node_patom_token,
        "node_patom_is_backbone": node_patom_is_backbone,
        "node_patom_is_metal": node_patom_is_metal,
        # schema_v2 protein-atom pharmacophore (mirrors ligand-side bools)
        "node_patom_is_donor": node_patom_is_donor,
        "node_patom_is_acceptor": node_patom_is_acceptor,
        "node_patom_is_positive": node_patom_is_positive,
        "node_patom_is_negative": node_patom_is_negative,
        "node_patom_is_hydrophobic": node_patom_is_hydrophobic,
        # Protein residue virtual-node features
        "node_pres_residue_type": node_pres_residue_type,
        "node_pres_is_pseudo": node_pres_is_pseudo,
        # Shared structural
        "node_fragment_id": node_fragment_id,
        # Edges
        "edge_index": torch.tensor([edge_src, edge_dst], dtype=torch.int64),
        "edge_type": torch.tensor(edge_type, dtype=torch.int8),
        "edge_ref_dist": torch.tensor(edge_ref_dist, dtype=torch.float32),
        "edge_bond_type": torch.tensor(edge_bond_type, dtype=torch.int8),
        "edge_bond_conjugated": torch.tensor(edge_bond_conjugated, dtype=torch.bool),
        "edge_bond_in_ring": torch.tensor(edge_bond_in_ring, dtype=torch.bool),
        "edge_bond_stereo": torch.tensor(edge_bond_stereo, dtype=torch.int8),
        "edge_frag_hop": torch.tensor(edge_frag_hop, dtype=torch.int8),
        # Counts + slice metadata
        "num_nodes": torch.tensor(total_nodes, dtype=torch.int64),
        "num_lig_atom": torch.tensor(n_lig_atom, dtype=torch.int64),
        "num_lig_frag": torch.tensor(n_frag, dtype=torch.int64),
        "num_prot_atom": torch.tensor(n_prot_atom, dtype=torch.int64),
        "num_prot_res": torch.tensor(n_pres, dtype=torch.int64),
        "lig_atom_slice": torch.tensor([0, n_lig_atom], dtype=torch.int64),
        "lig_frag_slice": torch.tensor([frag_offset, prot_atom_offset], dtype=torch.int64),
        "prot_atom_slice": torch.tensor([prot_atom_offset, pres_offset], dtype=torch.int64),
        "prot_res_slice": torch.tensor([pres_offset, total_nodes], dtype=torch.int64),
    }
