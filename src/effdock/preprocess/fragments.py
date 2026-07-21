"""Fragment decomposition: rotatable bond cuts → rigid fragments.

Each fragment gets:
- T_frag (centroid) and R_frag = I (identity, by construction)
- local_coords relative to centroid
- Single-atom fragments are avoided by merging back into a neighbor.
"""

import torch
from rdkit import Chem


def decompose_fragments(mol: Chem.Mol, atom_coords: torch.Tensor) -> dict | None:
    """Decompose molecule into rigid fragments via rotatable bond cuts.

    Args:
        mol: RDKit molecule (heavy atoms only, sanitized)
        atom_coords: [N_atom, 3] crystal coordinates

    Returns dict with:
        fragment_id: [N_atom] int64 — fragment assignment per atom
        frag_centers: [N_frag, 3] float32 — geometric centroids
        frag_local_coords: [N_atom, 3] float32 — coords in fragment frame
        frag_sizes: [N_frag] int64 — number of atoms per fragment (real only)
        n_frags: int
        rot_bonds: cut bonds AFTER singleton merge (only cross-fragment)
        cut_bond_index, tri_edge_index, tri_edge_ref_dist, fragment_adj_index

    Returns None on failure.
    """
    n_atoms = mol.GetNumAtoms()
    if n_atoms < 2:
        return None

    rot_bonds = _get_rotatable_bonds(mol)
    fragment_id = _assign_fragments(mol, rot_bonds, n_atoms)

    # Merge any size-1 fragments into an adjacent fragment (via any heavy bond).
    # This avoids degenerate single-atom fragments that arise when both bonds of
    # a degree-2 middle atom are cut (e.g., alkyl chains).
    fragment_id = _merge_singleton_fragments(mol, fragment_id, n_atoms)

    _, fragment_id = torch.unique(fragment_id, return_inverse=True)
    n_frags = int(fragment_id.max().item() + 1)

    # After merge, some originally-cut bonds may now lie inside a single fragment.
    # Filter rot_bonds to keep only cross-fragment ones so downstream edges and
    # dummy-atom insertion stay consistent.
    rot_bonds = [(a, b) for a, b in rot_bonds if fragment_id[a].item() != fragment_id[b].item()]

    frag_sizes = torch.bincount(fragment_id, minlength=n_frags)

    frag_centers = torch.zeros(n_frags, 3, dtype=torch.float32)
    frag_centers.scatter_add_(0, fragment_id[:, None].expand(-1, 3), atom_coords.to(torch.float32))
    frag_centers /= frag_sizes[:, None].clamp(min=1).to(torch.float32)

    # Local coords: x_global = R_frag @ x_local + T_frag; at crystal pose R = I.
    frag_local_coords = atom_coords - frag_centers[fragment_id]

    if not torch.isfinite(frag_centers).all():
        return None
    if not torch.isfinite(frag_local_coords).all():
        return None

    tri_data = _build_triangulation_edges(mol, rot_bonds, fragment_id, atom_coords)

    return {
        "fragment_id": fragment_id,
        "frag_centers": frag_centers,
        "frag_local_coords": frag_local_coords,
        "frag_sizes": frag_sizes,
        "n_frags": n_frags,
        "rot_bonds": rot_bonds,
        **tri_data,
    }


def _merge_singleton_fragments(
    mol: Chem.Mol,
    fragment_id: torch.Tensor,
    n_atoms: int,
) -> torch.Tensor:
    """Absorb size-1 fragments into their LARGEST neighboring fragment.

    When a singleton atom has multiple neighboring fragments, it joins the one
    with the most atoms. Tie-break: first neighbor in RDKit's deterministic
    order. Iterates until no singletons remain.
    """
    for _ in range(n_atoms):
        counts = torch.bincount(fragment_id)
        singletons = (counts == 1).nonzero(as_tuple=True)[0]
        if singletons.numel() == 0:
            return fragment_id
        changed = False
        for f in singletons.tolist():
            idxs = (fragment_id == f).nonzero(as_tuple=True)[0]
            if idxs.numel() != 1:
                continue
            a = int(idxs.item())
            best_target: int | None = None
            best_size = -1
            for nbr in mol.GetAtomWithIdx(a).GetNeighbors():
                target = int(fragment_id[nbr.GetIdx()].item())
                if target == f:
                    continue
                size = int((fragment_id == target).sum().item())
                if size > best_size:
                    best_size = size
                    best_target = target
            if best_target is not None:
                fragment_id[a] = best_target
                changed = True
        if not changed:
            return fragment_id
    return fragment_id


def _build_triangulation_edges(
    mol: Chem.Mol,
    rot_bonds: list[tuple[int, int]],
    fragment_id: torch.Tensor,
    atom_coords: torch.Tensor,
) -> dict:
    """Build cross-fragment triangulation edges around cut bonds.

    For each cut bond (a, b) with a in frag_A, b in frag_B:
    - Collect 1-hop neighbors of a within frag_A (including a)
    - Collect 1-hop neighbors of b within frag_B (including b)
    - Add cross-fragment pairs where at least one endpoint IS a cut-bond atom
      (invariant under torsion around the cut bond axis)
    - Store crystal reference distances for each pair

    Also builds fragment adjacency from cut bond topology. Both edge sets are
    deduplicated.
    """
    tri_src: list[int] = []
    tri_dst: list[int] = []
    ref_dists: list[float] = []
    tri_seen: set[tuple[int, int]] = set()

    adj_seen: set[tuple[int, int]] = set()
    frag_adj_src: list[int] = []
    frag_adj_dst: list[int] = []

    for a, b in rot_bonds:
        fa = int(fragment_id[a].item())
        fb = int(fragment_id[b].item())
        if fa == fb:
            continue

        for s, d in ((fa, fb), (fb, fa)):
            if (s, d) not in adj_seen:
                adj_seen.add((s, d))
                frag_adj_src.append(s)
                frag_adj_dst.append(d)

        left = {a}
        for nbr in mol.GetAtomWithIdx(a).GetNeighbors():
            ni = nbr.GetIdx()
            if int(fragment_id[ni].item()) == fa:
                left.add(ni)

        right = {b}
        for nbr in mol.GetAtomWithIdx(b).GetNeighbors():
            ni = nbr.GetIdx()
            if int(fragment_id[ni].item()) == fb:
                right.add(ni)

        for i in left:
            for j in right:
                if i != a and j != b:
                    continue
                key = (min(i, j), max(i, j))
                if key in tri_seen:
                    continue
                tri_seen.add(key)
                d_val = torch.linalg.vector_norm(atom_coords[i] - atom_coords[j]).item()
                tri_src.extend([i, j])
                tri_dst.extend([j, i])
                ref_dists.extend([d_val, d_val])

    if tri_src:
        tri_edge_index = torch.tensor([tri_src, tri_dst], dtype=torch.int64)
        tri_edge_ref_dist = torch.tensor(ref_dists, dtype=torch.float32)
    else:
        tri_edge_index = torch.zeros(2, 0, dtype=torch.int64)
        tri_edge_ref_dist = torch.zeros(0, dtype=torch.float32)

    if frag_adj_src:
        fragment_adj_index = torch.tensor([frag_adj_src, frag_adj_dst], dtype=torch.int64)
    else:
        fragment_adj_index = torch.zeros(2, 0, dtype=torch.int64)

    return {
        "cut_bond_index": (
            torch.tensor(rot_bonds, dtype=torch.int64).T
            if rot_bonds
            else torch.zeros(2, 0, dtype=torch.int64)
        ),
        "tri_edge_index": tri_edge_index,
        "tri_edge_ref_dist": tri_edge_ref_dist,
        "fragment_adj_index": fragment_adj_index,
    }


def _get_rotatable_bonds(mol: Chem.Mol) -> list[tuple[int, int]]:
    """Return bonds to cut. Greedy selection that avoids singleton fragments.

    Per-bond rules (``_is_cuttable_bond``): single, not in ring, not amide-like,
    both endpoints heavy-degree > 1.

    Greedy pass: iterates cuttable candidates in deterministic order and only
    accepts a cut if BOTH endpoints would still have >=1 remaining non-cut
    heavy bond afterwards. This guarantees every atom stays connected to at
    least one neighbor in the fragment graph (no size-1 fragments) and gives
    alternating cuts for long alkyl chains.
    """
    candidates: list[tuple[int, int]] = []
    for bond in mol.GetBonds():
        if not _is_cuttable_bond(bond):
            continue
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        candidates.append((min(i, j), max(i, j)))
    candidates.sort()

    remaining = [a.GetDegree() for a in mol.GetAtoms()]
    cut_bonds: list[tuple[int, int]] = []
    for i, j in candidates:
        if remaining[i] > 1 and remaining[j] > 1:
            cut_bonds.append((i, j))
            remaining[i] -= 1
            remaining[j] -= 1
    return cut_bonds


def _is_planar_conjugated_bond(bond: Chem.Bond) -> bool:
    """Return True for amide/ester/urea/carbamate/sulfonamide single bonds.

    These are the conjugated single bonds we want to keep rigid. Biaryl
    Caromatic-Caromatic single bonds are *not* included — they are physically
    rotatable and should become fragment boundaries.
    """
    a1, a2 = bond.GetBeginAtom(), bond.GetEndAtom()
    for heavy, partner in ((a1, a2), (a2, a1)):
        z = heavy.GetAtomicNum()
        if z == 6 and partner.GetAtomicNum() in (7, 8):
            for b in heavy.GetBonds():
                if (
                    b.GetBondType() == Chem.rdchem.BondType.DOUBLE
                    and b.GetOtherAtom(heavy).GetAtomicNum() == 8
                ):
                    return True
        elif z == 16 and partner.GetAtomicNum() == 7:
            n_double_o = sum(
                1
                for b in heavy.GetBonds()
                if b.GetBondType() == Chem.rdchem.BondType.DOUBLE
                and b.GetOtherAtom(heavy).GetAtomicNum() == 8
            )
            if n_double_o >= 2:
                return True
    return False


def _is_cuttable_bond(bond: Chem.Bond) -> bool:
    if bond.GetBondType() != Chem.rdchem.BondType.SINGLE:
        return False
    if bond.IsInRing():
        return False
    if _is_planar_conjugated_bond(bond):
        return False
    if bond.GetBeginAtom().GetDegree() <= 1 or bond.GetEndAtom().GetDegree() <= 1:
        return False
    return True


def _assign_fragments(
    mol: Chem.Mol,
    rot_bonds: list[tuple[int, int]],
    n_atoms: int,
) -> torch.Tensor:
    """Assign atoms to fragments by connected components after cutting bonds."""
    cut_set = set()
    for i, j in rot_bonds:
        cut_set.add((min(i, j), max(i, j)))

    parent = list(range(n_atoms))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        key = (min(i, j), max(i, j))
        if key not in cut_set:
            union(i, j)

    return torch.tensor([find(i) for i in range(n_atoms)], dtype=torch.int64)
