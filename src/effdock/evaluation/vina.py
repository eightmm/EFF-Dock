"""Torch-differentiable AutoDock Vina scoring.

Reimplements the AutoDock Vina empirical scoring function (Trott & Olson,
*J. Comput. Chem.* 2010) in pure PyTorch so it can be used as a docking score
**or** as a differentiable loss / guidance signal during sampling.

The free energy is a weighted sum of five pairwise terms evaluated on the
*surface distance* ``d = r_ij - (R_i + R_j)`` (``R`` = XS van-der-Waals radius)::

    gauss1      = exp(-(d / 0.5)^2)
    gauss2      = exp(-((d - 3.0) / 2.0)^2)
    repulsion   = d^2            if d < 0 else 0
    hydrophobic = clamp linear in [0.5, 1.5]      (hydrophobe-hydrophobe pairs)
    h_bond      = clamp linear in [-0.7, 0.0]     (donor-acceptor pairs)

    e_inter = sum_pairs (w1*g1 + w2*g2 + w3*rep + w4*hphob + w5*hbond)
    score   = e_inter / (1 + w_rot * N_rot)

Only pairs with atom-center distance below ``CUTOFF`` (8 Å) contribute. This
detail follows AutoDock Vina v1.2.7's ``potentials.h``; the five potentials
themselves are evaluated on surface distance.

Atom inputs are atomic numbers (int tensors) plus three boolean flags
(hydrophobic / donor / acceptor) per atom.

For the **ligand**, derive faithful Vina XS flags from the RDKit ``Mol`` with
:func:`vina_atom_types` — do NOT reuse the project's ``atom_is_hydrophobe``
(RDKit ``Hydrophobe`` family ≠ Vina ``C_H``; halogens are also a separate flag
there). For the **protein**, the curated ``patom_is_{donor,acceptor,
hydrophobic}`` flags from ``src/preprocess/protein.py`` are residue-table based
and Vina-compatible; pass them directly.

NOTE: ligand ``atom_element`` from preprocessing is a *vocab index* (C=0, N=1,
…), NOT an atomic number — feed real Z (e.g. ``vina_atom_types``'s
``atomic_nums`` or ``mol.GetAtomicNum()``) to :func:`vina_atom_radii`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

__all__ = [
    "VINA_WEIGHTS",
    "VinaWeights",
    "VINA_XS_TYPE_NAMES",
    "vina_atom_radii",
    "vina_xs_inputs",
    "vina_xs_types_from_ad",
    "read_vina_pdbqt",
    "vina_atom_types",
    "count_rotatable_bonds",
    "vina_pair_terms",
    "vina_score",
    "vina_score_batched",
    "vina_score_from_xs",
    "ligand_dg_reference",
    "ligand_dg_penalty",
    "ligand_dg_penalty_batched",
    "vina_score_with_strain",
    "vina_score_with_strain_batched",
    "VinaScorer",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# XS (AutoDock Vina) van-der-Waals radii, keyed by atomic number (Å). This is a
# convenience lookup; explicit XS types below are the exact public interface.
_XS_RADII_BY_Z: dict[int, float] = {
    1: 1.0,  # H  (usually absent — heavy-atom scoring)
    5: 1.9,  # B  (no XS type; treat as carbon-like)
    6: 1.9,  # C
    7: 1.8,  # N
    8: 1.7,  # O
    9: 1.5,  # F
    12: 1.2,  # Mg (metal)
    14: 2.2,  # Si
    15: 2.1,  # P
    16: 2.0,  # S
    17: 1.8,  # Cl
    20: 1.2,  # Ca (metal)
    25: 1.2,  # Mn (metal)
    26: 1.2,  # Fe (metal)
    30: 1.2,  # Zn (metal)
    34: 2.0,  # Se (treat as S-like)
    35: 2.0,  # Br
    53: 2.2,  # I
    85: 2.3,  # At
}
_DEFAULT_RADIUS = 1.9
_METAL_RADIUS = 1.2

CUTOFF = 8.0  # Å, atom-center cutoff above which pairs are ignored.

# Exact order from AutoDock Vina v1.2.7 src/lib/atom_constants.h. Dummy G types
# are included because they may appear in macrocycle PDBQT files. W is rejected
# by Vina's XS assignment and is consequently not exposed as a scoreable type.
VINA_XS_TYPE_NAMES: tuple[str, ...] = (
    "C_H", "C_P", "N_P", "N_D", "N_A", "N_DA", "O_P", "O_D",
    "O_A", "O_DA", "S_P", "P_P", "F_H", "Cl_H", "Br_H", "I_H",
    "Si", "At", "Met_D", "C_H_CG0", "C_P_CG0", "G0", "C_H_CG1",
    "C_P_CG1", "G1", "C_H_CG2", "C_P_CG2", "G2", "C_H_CG3",
    "C_P_CG3", "G3",
)
_XS_RADII = (
    1.9, 1.9, 1.8, 1.8, 1.8, 1.8, 1.7, 1.7, 1.7, 1.7, 2.0,
    2.1, 1.5, 1.8, 2.0, 2.2, 2.2, 2.3, 1.2, 1.9, 1.9, 1.9,
    1.9, 1.9, 1.9, 1.9, 1.9, 0.0, 0.0, 0.0, 0.0,
)
_XS_HYDROPHOBIC = frozenset({0, 12, 13, 14, 15})
_XS_DONOR = frozenset({3, 5, 7, 9, 18})
_XS_ACCEPTOR = frozenset({4, 5, 8, 9})

_AD_COVALENT_RADIUS = {
    "C": 0.77, "A": 0.77, "N": 0.75, "O": 0.73, "P": 1.06,
    "S": 1.02, "H": 0.37, "F": 0.71, "I": 1.33, "NA": 0.75,
    "OA": 0.73, "SA": 1.02, "HD": 0.37, "MG": 1.30, "MN": 1.39,
    "ZN": 1.31, "CA": 1.74, "FE": 1.25, "CL": 0.99, "BR": 1.14,
    "SI": 1.11, "AT": 1.44, "G0": 0.77, "G1": 0.77, "G2": 0.77,
    "G3": 0.77, "CG0": 0.77, "CG1": 0.77, "CG2": 0.77, "CG3": 0.77,
}
_NON_AD_METALS = frozenset({"CU", "FE", "NA", "K", "HG", "CO", "U", "CD", "NI"})


@dataclass(frozen=True)
class VinaWeights:
    """Term weights of the Vina free-energy function (kcal/mol)."""

    gauss1: float = -0.035579
    gauss2: float = -0.005156
    repulsion: float = 0.840245
    hydrophobic: float = -0.035069
    hydrogen_bond: float = -0.587439
    rot: float = 0.05846  # N_rot penalty in the denominator


VINA_WEIGHTS = VinaWeights()


# ---------------------------------------------------------------------------
# Radii lookup
# ---------------------------------------------------------------------------
def vina_atom_radii(
    atomic_nums: Tensor,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Map atomic numbers ``[N]`` -> XS van-der-Waals radii ``[N]`` (Å).

    Unknown elements default to carbon's radius (1.9 Å).
    """
    device = device or atomic_nums.device
    radii = torch.full((int(atomic_nums.numel()),), _DEFAULT_RADIUS, device=device, dtype=dtype)
    z_cpu = atomic_nums.reshape(-1).tolist()
    for i, z in enumerate(z_cpu):
        radii[i] = _XS_RADII_BY_Z.get(int(z), _DEFAULT_RADIUS)
    return radii


def vina_xs_inputs(
    xs_types: Tensor,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> dict[str, Tensor]:
    """Return exact Vina radii and interaction flags for explicit XS types.

    ``xs_types`` uses :data:`VINA_XS_TYPE_NAMES` indices. Unlike the atomic-
    number convenience route, this preserves donor/acceptor and polar carbon
    variants exactly. Invalid types fail explicitly instead of being guessed.
    """
    flat = xs_types.reshape(-1).to(dtype=torch.long)
    if flat.numel() and (int(flat.min()) < 0 or int(flat.max()) >= len(_XS_RADII)):
        raise ValueError(
            f"XS type indices must be in [0, {len(_XS_RADII) - 1}], "
            f"got [{int(flat.min())}, {int(flat.max())}]"
        )
    device = device or xs_types.device
    table = torch.tensor(_XS_RADII, device=device, dtype=dtype)
    idx = flat.to(device)
    return {
        "xs_types": idx,
        "radii": table.index_select(0, idx),
        "is_hydrophobic": torch.tensor(
            [int(x) in _XS_HYDROPHOBIC for x in flat.tolist()], device=device
        ),
        "is_donor": torch.tensor([int(x) in _XS_DONOR for x in flat.tolist()], device=device),
        "is_acceptor": torch.tensor(
            [int(x) in _XS_ACCEPTOR for x in flat.tolist()], device=device
        ),
    }


def vina_xs_types_from_ad(ad_types: list[str], bond_index: Tensor) -> tuple[Tensor, Tensor]:
    """Apply Vina v1.2.7 ``model::assign_types`` to AutoDock atom types.

    Returns ``(heavy_atom_indices, xs_types)``. Polar hydrogens are omitted from
    the score but retained while determining whether bonded N/O atoms donate.
    """
    ad = [value.strip().upper() for value in ad_types]
    neighbors: list[list[int]] = [[] for _ in ad]
    for i, j in bond_index.T.tolist():
        neighbors[i].append(j)
        neighbors[j].append(i)
    heavy: list[int] = []
    xs: list[int] = []
    carbon_closure = {
        "CG0": (19, 20), "CG1": (22, 23), "CG2": (25, 26), "CG3": (28, 29)
    }
    direct = {
        "S": 10, "SA": 10, "P": 11, "F": 12, "CL": 13, "BR": 14,
        "I": 15, "SI": 16, "AT": 17, "G0": 21, "G1": 24, "G2": 27,
        "G3": 30,
    }
    for i, atom_type in enumerate(ad):
        if atom_type in {"H", "HD"}:
            continue
        if atom_type == "W":
            continue  # official assign_types sets W to XS_TYPE_SIZE (unscoreable)
        if atom_type in _NON_AD_METALS or atom_type in {"MG", "MN", "ZN", "CA", "FE"}:
            value = 18
        elif atom_type in {"C", "A"} or atom_type in carbon_closure:
            hetero = any(ad[j] not in {"C", "A", "H", "HD"} for j in neighbors[i])
            value = carbon_closure.get(atom_type, (0, 1))[int(hetero)]
        elif atom_type in {"N", "NA", "O", "OA"}:
            acceptor = atom_type in {"NA", "OA"}
            donor = any(ad[j] == "HD" for j in neighbors[i])
            if atom_type.startswith("N"):
                value = 5 if donor and acceptor else 4 if acceptor else 3 if donor else 2
            else:
                value = 9 if donor and acceptor else 8 if acceptor else 7 if donor else 6
        elif atom_type in direct:
            value = direct[atom_type]
        else:
            raise ValueError(f"unsupported AutoDock atom type {ad_types[i]!r}")
        heavy.append(i)
        xs.append(value)
    return torch.tensor(heavy, dtype=torch.long), torch.tensor(xs, dtype=torch.long)


def read_vina_pdbqt(path: str | Path) -> dict[str, Tensor | int]:
    """Read one PDBQT model and reproduce Vina XS typing.

    Bonds are inferred with Vina's 1.1x covalent-radius criterion, including its
    intervening-atom rejection. This route is intended for numerical parity and
    exact scoring of already prepared PDBQT inputs.
    """
    coords: list[list[float]] = []
    ad_types: list[str] = []
    torsdof = 0
    seen_model = False
    for line in Path(path).read_text().splitlines():
        if line.startswith("MODEL"):
            if seen_model:
                break
            seen_model = True
        if line.startswith("ENDMDL"):
            break
        if line.startswith(("ATOM  ", "HETATM")):
            coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            ad_types.append(line.split()[-1])
        elif line.startswith("TORSDOF"):
            torsdof = int(line.split()[1])
    if not coords:
        raise ValueError(f"no PDBQT atoms found in {path}")
    xyz = torch.tensor(coords, dtype=torch.float32)
    radii = torch.tensor(
        [_AD_COVALENT_RADIUS.get(atom_type.upper(), 1.75) for atom_type in ad_types]
    )
    distance = torch.cdist(xyz, xyz)
    candidate = distance < (1.1 * (radii[:, None] + radii[None, :]))
    pairs: list[tuple[int, int]] = []
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            if not candidate[i, j]:
                continue
            dij = distance[i, j]
            between = ((distance[i] < dij) & (distance[j] < dij)).clone()
            between[i] = False
            between[j] = False
            if not bool(between.any()):
                pairs.append((i, j))
    bond_index = (
        torch.tensor(pairs, dtype=torch.long).T.contiguous()
        if pairs
        else torch.empty(2, 0, dtype=torch.long)
    )
    heavy, xs_types = vina_xs_types_from_ad(ad_types, bond_index)
    return {
        "coords": xyz.index_select(0, heavy),
        "xs_types": xs_types,
        "num_rotatable_bonds": torsdof,
        "bond_index_all": bond_index,
        "heavy_atom_indices": heavy,
    }


# ---------------------------------------------------------------------------
# Faithful XS atom typing (ligand, from an RDKit Mol)
# ---------------------------------------------------------------------------
_HALOGEN_Z: frozenset[int] = frozenset({9, 17, 35, 53})


def _nitrogen_is_acceptor(atom) -> bool:
    """Whether a nitrogen has a lone pair free for H-bond acceptance (AutoDock NA).

    * Positively charged N (ammonium / quaternary) → no lone pair → not acceptor.
    * Aromatic N → pyridine-type (no H) accepts; pyrrole-type (N-H, lone pair in
      the ring π-system) does not.
    * Aliphatic N → acceptor unless its lone pair is delocalised into an adjacent
      π-acceptor (amide / amidine / sulfonamide: a neighbour carbon or sulfur
      double-bonded to O/N/S).
    """
    if atom.GetFormalCharge() > 0:
        return False
    if atom.GetIsAromatic():
        return atom.GetTotalNumHs(includeNeighbors=True) == 0
    for nb in atom.GetNeighbors():
        if nb.GetAtomicNum() not in (6, 16):  # conjugating centre: C or S
            continue
        for bond in nb.GetBonds():
            if bond.GetBondTypeAsDouble() >= 2.0:
                other = bond.GetOtherAtom(nb)
                if other.GetIdx() != atom.GetIdx() and other.GetAtomicNum() in (7, 8, 16):
                    return False
    return True


def count_rotatable_bonds(mol, strict: bool = True) -> int:
    """Active rotatable-bond count for the Vina ``N_rot`` penalty.

    Uses RDKit's rotatable-bond descriptor (``Strict`` excludes amides, conjugated
    and other non-rotating single bonds), matching AutoDock's torsion count
    closely. Pass the same ``mol`` used for typing / docking.
    """
    from rdkit.Chem import rdMolDescriptors

    opt = (
        rdMolDescriptors.NumRotatableBondsOptions.Strict
        if strict
        else rdMolDescriptors.NumRotatableBondsOptions.Default
    )
    return int(rdMolDescriptors.CalcNumRotatableBonds(mol, opt))


def vina_atom_types(mol) -> dict[str, Tensor | int]:
    """Compute Vina XS-faithful per-atom inputs from an RDKit ``Mol``.

    Returns everything :func:`vina_score` needs for one molecule — atomic
    numbers, the three boolean flags driving the hydrophobic / H-bond terms, and
    the rotatable-bond count for the ``N_rot`` penalty. Use this for the
    **ligand**, where the RDKit graph is available, instead of the project's
    pharmacophore flags (whose ``Hydrophobe`` family does not match Vina's XS
    definition).

    XS rules applied:
      * **hydrophobic** = halogen (F/Cl/Br/I), or a carbon whose every *heavy*
        neighbour is also carbon (Vina ``C_H``; a carbon bonded to any
        heteroatom becomes polar ``C_P`` and is excluded).
      * **donor** = N/O carrying at least one hydrogen (``GetTotalNumHs`` counts
        implicit + explicit), or a metal.
      * **acceptor** = every O; nitrogen per :func:`_nitrogen_is_acceptor`
        (charge / aromaticity / amide-delocalisation aware — AutoDock NA).

    Returns:
        dict with ``atomic_nums`` ``[N] int64``; ``is_hydrophobic`` / ``is_donor``
        / ``is_acceptor`` (``[N] bool`` each); ``num_rotatable_bonds`` (int).
    """
    n = mol.GetNumAtoms()
    z = torch.empty(n, dtype=torch.int64)
    hphob = torch.zeros(n, dtype=torch.bool)
    donor = torch.zeros(n, dtype=torch.bool)
    accept = torch.zeros(n, dtype=torch.bool)

    for i in range(n):
        atom = mol.GetAtomWithIdx(i)
        num = atom.GetAtomicNum()
        z[i] = num
        n_h = atom.GetTotalNumHs(includeNeighbors=True)
        heavy_nbr_z = [b.GetAtomicNum() for b in atom.GetNeighbors() if b.GetAtomicNum() != 1]

        if num in _HALOGEN_Z:
            hphob[i] = True
        elif num == 6:  # carbon: hydrophobic only if no heteroatom neighbour
            hphob[i] = all(zz == 6 for zz in heavy_nbr_z)
        elif num == 7:  # nitrogen
            donor[i] = n_h > 0
            accept[i] = _nitrogen_is_acceptor(atom)
        elif num == 8:  # oxygen
            donor[i] = n_h > 0
            accept[i] = True
        elif num in _XS_RADII_BY_Z and _XS_RADII_BY_Z[num] == _METAL_RADIUS:
            donor[i] = True  # metal -> Met_D

    return {
        "atomic_nums": z,
        "is_hydrophobic": hphob,
        "is_donor": donor,
        "is_acceptor": accept,
        "num_rotatable_bonds": count_rotatable_bonds(mol),
    }


# ---------------------------------------------------------------------------
# Pairwise terms
# ---------------------------------------------------------------------------
def vina_pair_terms(
    surf_dist: Tensor,
    hphob_pair: Tensor,
    hbond_pair: Tensor,
) -> Tensor:
    """Per-pair Vina energy on surface distance ``d`` (already cutoff-masked).

    Args:
        surf_dist:  ``[...]`` surface distance ``d = r - R_i - R_j``.
        hphob_pair: ``[...]`` bool — both atoms hydrophobic.
        hbond_pair: ``[...]`` bool — donor/acceptor complementary pair.

    Returns:
        ``[...]`` weighted pairwise energy.
    """
    w = VINA_WEIGHTS
    d = surf_dist

    gauss1 = torch.exp(-((d / 0.5) ** 2))
    gauss2 = torch.exp(-(((d - 3.0) / 2.0) ** 2))
    repulsion = torch.where(d < 0.0, d * d, torch.zeros_like(d))

    # Hydrophobic: 1 for d<0.5, ramps to 0 at d=1.5.
    hydrophobic = (1.5 - d).clamp(0.0, 1.0)
    hydrophobic = hydrophobic * hphob_pair.to(d.dtype)

    # H-bond: 1 for d<-0.7, ramps to 0 at d=0.
    h_bond = (-d / 0.7).clamp(0.0, 1.0)
    h_bond = h_bond * hbond_pair.to(d.dtype)

    return (
        w.gauss1 * gauss1
        + w.gauss2 * gauss2
        + w.repulsion * repulsion
        + w.hydrophobic * hydrophobic
        + w.hydrogen_bond * h_bond
    )


# ---------------------------------------------------------------------------
# Full intermolecular score
# ---------------------------------------------------------------------------
def vina_score(
    lig_coords: Tensor,
    prot_coords: Tensor,
    lig_radii: Tensor,
    prot_radii: Tensor,
    *,
    lig_is_hydrophobic: Tensor,
    prot_is_hydrophobic: Tensor,
    lig_is_donor: Tensor,
    prot_is_donor: Tensor,
    lig_is_acceptor: Tensor,
    prot_is_acceptor: Tensor,
    num_rotatable_bonds: int | Tensor = 0,
    lig_batch: Tensor | None = None,
    prot_batch: Tensor | None = None,
    cutoff: float = CUTOFF,
    eps: float = 1e-6,
) -> Tensor:
    """Vina intermolecular free energy between a ligand and a protein pocket.

    All ligand/protein tensors are aligned: index ``i`` of every ``lig_*`` arg
    describes ligand atom ``i``. Coordinates are differentiable, so gradients
    flow back to ``lig_coords`` (and ``prot_coords``) — usable as a loss.

    Pass ``lig_batch`` / ``prot_batch`` (per-atom sample ids, like the project's
    ``node_batch_idx``) to score a **batch of complexes** packed into one tensor:
    only same-sample atom pairs interact (pitfall #3 — no cross-sample leakage),
    and a ``[B]`` vector of per-complex energies is returned. Heavy atoms only —
    drop explicit hydrogens before calling (Vina is a heavy-atom function).

    Args:
        lig_coords:  ``[N_l, 3]`` ligand coordinates (Å).
        prot_coords: ``[N_p, 3]`` protein coordinates (Å).
        lig_radii / prot_radii: ``[N_l] / [N_p]`` XS radii (see
            :func:`vina_atom_radii`).
        lig_is_* / prot_is_*: ``[N_l] / [N_p]`` bool flags (see
            :func:`vina_atom_types`).
        num_rotatable_bonds: scalar, or ``[B]`` per-complex when batched.
        lig_batch / prot_batch: ``[N_l] / [N_p]`` int sample ids, or ``None``
            for a single complex.
        cutoff: atom-center cutoff (Å), matching official Vina.
        eps: floor on squared distance — keeps the distance gradient finite when
            two atoms coincide (important for guidance / loss).

    Returns:
        Scalar free energy for a single complex, or ``[B]`` when batched
        (kcal/mol; lower = better binding).
    """
    # NaN-safe pairwise distance [N_l, N_p]: manual cdist with a squared-distance
    # floor so d(sqrt)/dx stays finite at coincident atoms (torch.cdist -> NaN).
    diff = lig_coords[:, None, :] - prot_coords[None, :, :]
    dist = diff.pow(2).sum(-1).clamp_min(eps * eps).sqrt()
    surf = dist - (lig_radii[:, None] + prot_radii[None, :])

    within = dist < cutoff  # [N_l, N_p] bool; official cutoff precedes surface distance
    if (lig_batch is None) != (prot_batch is None):
        raise ValueError("pass both lig_batch and prot_batch, or neither")
    if lig_batch is not None:
        within = within & (lig_batch[:, None] == prot_batch[None, :])

    hphob_pair = lig_is_hydrophobic[:, None] & prot_is_hydrophobic[None, :]
    # Donor on one side complemented by acceptor on the other (both orders).
    hbond_pair = (lig_is_donor[:, None] & prot_is_acceptor[None, :]) | (
        lig_is_acceptor[:, None] & prot_is_donor[None, :]
    )

    pair_e = vina_pair_terms(surf, hphob_pair, hbond_pair)
    pair_e = pair_e * within.to(pair_e.dtype)

    def _denom(n_rot: int | Tensor) -> Tensor:
        if not torch.is_tensor(n_rot):
            n_rot = torch.tensor(float(n_rot), device=pair_e.device, dtype=pair_e.dtype)
        return 1.0 + VINA_WEIGHTS.rot * n_rot.to(pair_e.dtype)

    if lig_batch is None:
        return pair_e.sum() / _denom(num_rotatable_bonds)

    # Batched: scatter each pair's energy onto its (masked) sample id.
    n_samples = int(torch.maximum(lig_batch.max(), prot_batch.max()).item()) + 1
    sample_of_pair = lig_batch[:, None].expand_as(pair_e).reshape(-1)
    e_inter = torch.zeros(n_samples, device=pair_e.device, dtype=pair_e.dtype)
    e_inter = e_inter.index_add(0, sample_of_pair, pair_e.reshape(-1))
    return e_inter / _denom(num_rotatable_bonds)


def vina_score_batched(
    lig_coords: Tensor,
    prot_coords: Tensor,
    lig_radii: Tensor,
    prot_radii: Tensor,
    *,
    lig_is_hydrophobic: Tensor,
    prot_is_hydrophobic: Tensor,
    lig_is_donor: Tensor,
    prot_is_donor: Tensor,
    lig_is_acceptor: Tensor,
    prot_is_acceptor: Tensor,
    num_rotatable_bonds: int | Tensor = 0,
    cutoff: float = CUTOFF,
    eps: float = 1e-6,
) -> Tensor:
    """Efficiently score many poses of one ligand/receptor pair.

    Args:
        lig_coords: ``[B, N_l, 3]`` current ligand poses.
        prot_coords: ``[N_p, 3]`` shared receptor or ``[B, N_p, 3]``.
        Remaining atom properties are unbatched ``[N_l]`` / ``[N_p]`` because
        all poses have identical chemistry.

    Returns:
        ``[B]`` energies. This avoids the quadratic-in-batch cross-product of a
        generic packed representation and is the intended guidance kernel.
    """
    if lig_coords.ndim != 3 or lig_coords.shape[-1] != 3:
        raise ValueError(f"lig_coords must have shape [B,N,3], got {tuple(lig_coords.shape)}")
    if prot_coords.ndim == 2:
        prot_coords = prot_coords.unsqueeze(0)
    if prot_coords.ndim != 3 or prot_coords.shape[-1] != 3:
        raise ValueError(f"prot_coords must have shape [P,3] or [B,P,3], got {tuple(prot_coords.shape)}")
    if prot_coords.shape[0] not in (1, lig_coords.shape[0]):
        raise ValueError("protein batch dimension must be 1 or match ligand batch")

    diff = lig_coords[:, :, None, :] - prot_coords[:, None, :, :]
    dist = diff.pow(2).sum(-1).clamp_min(eps * eps).sqrt()
    surf = dist - (lig_radii[None, :, None] + prot_radii[None, None, :])
    within = dist < cutoff

    hphob_pair = lig_is_hydrophobic[:, None] & prot_is_hydrophobic[None, :]
    hbond_pair = (lig_is_donor[:, None] & prot_is_acceptor[None, :]) | (
        lig_is_acceptor[:, None] & prot_is_donor[None, :]
    )
    pair_e = vina_pair_terms(surf, hphob_pair[None], hbond_pair[None])
    e_inter = (pair_e * within.to(pair_e.dtype)).sum(dim=(1, 2))
    n_rot = torch.as_tensor(num_rotatable_bonds, device=e_inter.device, dtype=e_inter.dtype)
    return e_inter / (1.0 + VINA_WEIGHTS.rot * n_rot)


def vina_score_from_xs(
    lig_coords: Tensor,
    prot_coords: Tensor,
    lig_xs_types: Tensor,
    prot_xs_types: Tensor,
    *,
    num_rotatable_bonds: int | Tensor = 0,
    cutoff: float = CUTOFF,
    eps: float = 1e-6,
) -> Tensor:
    """Score explicit official Vina XS types without chemistry inference."""
    lig = vina_xs_inputs(lig_xs_types, device=lig_coords.device, dtype=lig_coords.dtype)
    prot = vina_xs_inputs(prot_xs_types, device=prot_coords.device, dtype=prot_coords.dtype)
    kwargs = {
        "lig_is_hydrophobic": lig["is_hydrophobic"],
        "prot_is_hydrophobic": prot["is_hydrophobic"],
        "lig_is_donor": lig["is_donor"],
        "prot_is_donor": prot["is_donor"],
        "lig_is_acceptor": lig["is_acceptor"],
        "prot_is_acceptor": prot["is_acceptor"],
        "num_rotatable_bonds": num_rotatable_bonds,
        "cutoff": cutoff,
        "eps": eps,
    }
    if lig_coords.ndim == 3:
        return vina_score_batched(
            lig_coords, prot_coords, lig["radii"], prot["radii"], **kwargs
        )
    return vina_score(lig_coords, prot_coords, lig["radii"], prot["radii"], **kwargs)


# ---------------------------------------------------------------------------
# Ligand intramolecular distance-geometry strain penalty
# ---------------------------------------------------------------------------
def ligand_dg_reference(mol, conf_id: int = -1) -> dict[str, Tensor]:
    """Ideal bonded distances for a ligand, from its RDKit conformer.

    The conformer (the same one whose rigid fragments the docker transports)
    defines chemically valid bond lengths. If the molecule has no conformer, an
    ETKDG one is generated.

    Returns:
        dict with ``bond_index`` ``[2, Nb] int64`` (undirected, i<j) and
        ``bond_ref_len`` ``[Nb] float32`` — ideal length of each bond.
    """
    if mol.GetNumConformers() == 0:
        from rdkit.Chem import AllChem

        mol = type(mol)(mol)  # copy; don't mutate caller's mol
        if AllChem.EmbedMolecule(mol, randomSeed=0) != 0:
            raise ValueError("ligand_dg_reference: ETKDG embedding failed")
        conf_id = -1

    conf = mol.GetConformer(conf_id)
    pos = torch.tensor(conf.GetPositions(), dtype=torch.float32)
    bi, bj, ref = [], [], []
    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if a > b:
            a, b = b, a
        bi.append(a)
        bj.append(b)
        ref.append(float((pos[a] - pos[b]).norm()))
    return {
        "bond_index": torch.tensor([bi, bj], dtype=torch.int64),
        "bond_ref_len": torch.tensor(ref, dtype=torch.float32),
    }


def ligand_dg_penalty(
    coords: Tensor,
    bond_index: Tensor,
    bond_ref_len: Tensor,
    *,
    frag_id: Tensor | None = None,
    radii: Tensor | None = None,
    clash_scale: float = 0.75,
    lig_batch: Tensor | None = None,
    eps: float = 1e-6,
) -> Tensor:
    """Distance-geometry strain penalty for a docked ligand pose (>= 0).

    Catches intramolecular geometry the rigid-fragment docker can break — the
    only two failure modes, since within-fragment geometry is transported
    rigidly (pitfall #4):

      * **bond stretch** — every bond's length vs its ideal (``bond_ref_len``);
        ``(d - d_ref)^2``. Cut bonds joining two fragments are where this bites.
      * **inter-fragment clash** *(optional, needs ``frag_id`` + ``radii``)* —
        atoms in *different* fragments overlapping below
        ``clash_scale * (R_i + R_j)``; ``(thr - d)^2``. Within-fragment contacts
        (rigid) and 1-2 / 1-3 bonded neighbours (whose short separations are
        normal geometry, e.g. across a cut bond) are skipped.

    Differentiable in ``coords`` and NaN-safe at coincident atoms. Add it to the
    Vina free energy (both "lower = better"): ``vina_score(...) + w * penalty``.

    Args:
        coords: ``[N, 3]`` ligand coordinates (Å).
        bond_index: ``[2, Nb]`` undirected bond endpoints (see
            :func:`ligand_dg_reference`).
        bond_ref_len: ``[Nb]`` ideal bond lengths.
        frag_id: ``[N]`` fragment id per atom (enables the clash term).
        radii: ``[N]`` XS radii per atom (enables the clash term).
        clash_scale: fraction of summed radii below which overlap is penalised.
        lig_batch: ``[N]`` sample ids for a packed batch → returns ``[B]``.
        eps: distance floor for finite gradients.

    Returns:
        Scalar penalty, or ``[B]`` when ``lig_batch`` is given.
    """
    i, j = bond_index[0], bond_index[1]
    bond_d = (coords[i] - coords[j]).pow(2).sum(-1).clamp_min(eps * eps).sqrt()
    bond_pen = (bond_d - bond_ref_len) ** 2  # [Nb]

    if lig_batch is None:
        total = bond_pen.sum()
    else:
        n_samples = int(lig_batch.max().item()) + 1
        total = torch.zeros(n_samples, device=coords.device, dtype=bond_pen.dtype)
        total = total.index_add(0, lig_batch[i], bond_pen)

    if frag_id is not None and radii is not None:
        n = coords.shape[0]
        diff = coords[:, None, :] - coords[None, :, :]
        dist = diff.pow(2).sum(-1).clamp_min(eps * eps).sqrt()  # [N, N]
        thr = clash_scale * (radii[:, None] + radii[None, :])
        overlap = (thr - dist).clamp_min(0.0) ** 2  # [N, N]

        # Exclude 1-2 (bonded) and 1-3 (share a neighbour) pairs: their short
        # separations are normal bonded geometry, not clashes.
        adj = torch.zeros(n, n, dtype=torch.float32, device=coords.device)
        adj[i, j] = 1.0
        adj[j, i] = 1.0
        near = (adj + adj @ adj) > 0  # 1-2 and 1-3 neighbours
        triu = torch.triu(torch.ones(n, n, dtype=torch.bool, device=coords.device), 1)
        mask = triu & (~near) & (frag_id[:, None] != frag_id[None, :])
        if lig_batch is not None:
            mask = mask & (lig_batch[:, None] == lig_batch[None, :])

        clash = overlap * mask.to(overlap.dtype)
        if lig_batch is None:
            total = total + clash.sum()
        else:
            rows = torch.arange(n, device=coords.device)[:, None].expand(n, n)
            total = total.index_add(0, lig_batch[rows.reshape(-1)], clash.reshape(-1))

    return total


def ligand_dg_penalty_batched(
    coords: Tensor,
    bond_index: Tensor,
    bond_ref_len: Tensor,
    *,
    frag_id: Tensor | None = None,
    radii: Tensor | None = None,
    clash_scale: float = 0.75,
    eps: float = 1e-6,
) -> Tensor:
    """Vectorized :func:`ligand_dg_penalty` for ``[B,N,3]`` poses."""
    if coords.ndim != 3 or coords.shape[-1] != 3:
        raise ValueError(f"coords must have shape [B,N,3], got {tuple(coords.shape)}")
    i, j = bond_index[0], bond_index[1]
    bond_d = (coords[:, i] - coords[:, j]).pow(2).sum(-1).clamp_min(eps * eps).sqrt()
    total = ((bond_d - bond_ref_len[None, :]) ** 2).sum(dim=1)

    if frag_id is not None and radii is not None:
        n = coords.shape[1]
        diff = coords[:, :, None, :] - coords[:, None, :, :]
        dist = diff.pow(2).sum(-1).clamp_min(eps * eps).sqrt()
        threshold = clash_scale * (radii[:, None] + radii[None, :])

        adj = torch.zeros(n, n, dtype=torch.float32, device=coords.device)
        adj[i, j] = 1.0
        adj[j, i] = 1.0
        near = (adj + adj @ adj) > 0
        upper = torch.triu(torch.ones(n, n, dtype=torch.bool, device=coords.device), 1)
        mask = upper & (~near) & (frag_id[:, None] != frag_id[None, :])
        overlap = (threshold[None] - dist).clamp_min(0.0).square()
        total = total + (overlap * mask[None].to(overlap.dtype)).sum(dim=(1, 2))
    return total


# ---------------------------------------------------------------------------
# Combined score: Vina free energy + ligand strain
# ---------------------------------------------------------------------------
def vina_score_with_strain(
    lig_coords: Tensor,
    prot_coords: Tensor,
    lig_radii: Tensor,
    prot_radii: Tensor,
    *,
    lig_is_hydrophobic: Tensor,
    prot_is_hydrophobic: Tensor,
    lig_is_donor: Tensor,
    prot_is_donor: Tensor,
    lig_is_acceptor: Tensor,
    prot_is_acceptor: Tensor,
    bond_index: Tensor,
    bond_ref_len: Tensor,
    num_rotatable_bonds: int | Tensor = 0,
    frag_id: Tensor | None = None,
    lig_batch: Tensor | None = None,
    prot_batch: Tensor | None = None,
    w_strain: float = 1.0,
    clash_scale: float = 0.75,
    cutoff: float = CUTOFF,
    eps: float = 1e-6,
    return_components: bool = False,
) -> Tensor | dict[str, Tensor]:
    """Vina free energy plus the ligand distance-geometry strain penalty.

    ``total = vina_score + w_strain * ligand_dg_penalty`` — both terms are
    "lower = better", so a strained or clashing pose is pushed up (worse). Use
    this as the docking objective / guidance loss when you want pose validity
    enforced alongside binding. See :func:`vina_score` and
    :func:`ligand_dg_penalty` for the per-term arguments.

    The strain clash term activates when ``frag_id`` is given (uses ``lig_radii``
    for atom sizes). ``lig_batch`` / ``prot_batch`` enable batched scoring and
    make the result a ``[B]`` vector.

    Args:
        bond_index / bond_ref_len: ligand bonds + ideal lengths (see
            :func:`ligand_dg_reference`).
        w_strain: weight on the strain penalty (kcal/mol per Å² of violation).
        return_components: if True, return a dict ``{"total", "vina", "strain"}``
            instead of just the total.

    Returns:
        Total score (scalar or ``[B]``), or a dict of components.
    """
    vina = vina_score(
        lig_coords,
        prot_coords,
        lig_radii,
        prot_radii,
        lig_is_hydrophobic=lig_is_hydrophobic,
        prot_is_hydrophobic=prot_is_hydrophobic,
        lig_is_donor=lig_is_donor,
        prot_is_donor=prot_is_donor,
        lig_is_acceptor=lig_is_acceptor,
        prot_is_acceptor=prot_is_acceptor,
        num_rotatable_bonds=num_rotatable_bonds,
        lig_batch=lig_batch,
        prot_batch=prot_batch,
        cutoff=cutoff,
        eps=eps,
    )
    strain = ligand_dg_penalty(
        lig_coords,
        bond_index,
        bond_ref_len,
        frag_id=frag_id,
        radii=lig_radii if frag_id is not None else None,
        clash_scale=clash_scale,
        lig_batch=lig_batch,
        eps=eps,
    )
    total = vina + w_strain * strain
    if return_components:
        return {"total": total, "vina": vina, "strain": strain}
    return total


def vina_score_with_strain_batched(
    lig_coords: Tensor,
    prot_coords: Tensor,
    lig_radii: Tensor,
    prot_radii: Tensor,
    *,
    lig_is_hydrophobic: Tensor,
    prot_is_hydrophobic: Tensor,
    lig_is_donor: Tensor,
    prot_is_donor: Tensor,
    lig_is_acceptor: Tensor,
    prot_is_acceptor: Tensor,
    bond_index: Tensor,
    bond_ref_len: Tensor,
    num_rotatable_bonds: int | Tensor = 0,
    frag_id: Tensor | None = None,
    w_strain: float = 1.0,
    clash_scale: float = 0.75,
    cutoff: float = CUTOFF,
    eps: float = 1e-6,
    return_components: bool = False,
) -> Tensor | dict[str, Tensor]:
    """Dense batched Vina+DG objective used for differentiable guidance."""
    vina = vina_score_batched(
        lig_coords,
        prot_coords,
        lig_radii,
        prot_radii,
        lig_is_hydrophobic=lig_is_hydrophobic,
        prot_is_hydrophobic=prot_is_hydrophobic,
        lig_is_donor=lig_is_donor,
        prot_is_donor=prot_is_donor,
        lig_is_acceptor=lig_is_acceptor,
        prot_is_acceptor=prot_is_acceptor,
        num_rotatable_bonds=num_rotatable_bonds,
        cutoff=cutoff,
        eps=eps,
    )
    strain = ligand_dg_penalty_batched(
        lig_coords,
        bond_index,
        bond_ref_len,
        frag_id=frag_id,
        radii=lig_radii if frag_id is not None else None,
        clash_scale=clash_scale,
        eps=eps,
    )
    total = vina + w_strain * strain
    if return_components:
        return {"total": total, "vina": vina, "strain": strain}
    return total


# ---------------------------------------------------------------------------
# Convenience module
# ---------------------------------------------------------------------------
class VinaScorer(torch.nn.Module):
    """Stateless ``nn.Module`` wrapper around :func:`vina_score`.

    Caches per-atom radii (derived from atomic numbers) so they need not be
    recomputed each call. Pharmacophore flags and coordinates are passed at
    call time.
    """

    def __init__(
        self,
        lig_atomic_nums: Tensor,
        prot_atomic_nums: Tensor,
        cutoff: float = CUTOFF,
    ) -> None:
        super().__init__()
        self.cutoff = cutoff
        self.register_buffer("lig_radii", vina_atom_radii(lig_atomic_nums))
        self.register_buffer("prot_radii", vina_atom_radii(prot_atomic_nums))

    def forward(
        self,
        lig_coords: Tensor,
        prot_coords: Tensor,
        *,
        lig_is_hydrophobic: Tensor,
        prot_is_hydrophobic: Tensor,
        lig_is_donor: Tensor,
        prot_is_donor: Tensor,
        lig_is_acceptor: Tensor,
        prot_is_acceptor: Tensor,
        num_rotatable_bonds: int | Tensor = 0,
        lig_batch: Tensor | None = None,
        prot_batch: Tensor | None = None,
    ) -> Tensor:
        return vina_score(
            lig_coords,
            prot_coords,
            self.lig_radii,
            self.prot_radii,
            lig_is_hydrophobic=lig_is_hydrophobic,
            prot_is_hydrophobic=prot_is_hydrophobic,
            lig_is_donor=lig_is_donor,
            prot_is_donor=prot_is_donor,
            lig_is_acceptor=lig_is_acceptor,
            prot_is_acceptor=prot_is_acceptor,
            num_rotatable_bonds=num_rotatable_bonds,
            lig_batch=lig_batch,
            prot_batch=prot_batch,
            cutoff=self.cutoff,
        )
