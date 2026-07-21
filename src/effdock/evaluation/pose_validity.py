"""Self-contained, fast PoseBusters-style pose validity (no posebusters dep).

Implements the cheap, high-signal subset of PoseBusters checks in torch + RDKit:

  * bond lengths   — 1-2 pairs within the RDKit distance-geometry bounds
  * bond angles    — 1-3 pairs within the DG bounds
  * internal clash — non-bonded (>=1-4) intra pairs not below the DG lower bound
  * protein clash  — ligand-protein heavy-atom pairs not overlapping below
                     ``prot_clash_scale * (vdw_i + vdw_j)``

Deliberately omits PoseBusters' expensive ``volume_overlap`` and the
``energy_ratio`` (UFF) test. The DG bounds matrix is topology-derived and
captures bond/angle/clash geometry the way PoseBusters' ``distance_geometry``
module does, but vectorised and computed once per molecule.
"""

from __future__ import annotations

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import rdDistGeom

# Covalent/vdw-ish radii (Å) by atomic number for the protein-clash check.
_VDW = {
    1: 1.1,
    6: 1.7,
    7: 1.55,
    8: 1.52,
    9: 1.47,
    15: 1.8,
    16: 1.8,
    17: 1.75,
    35: 1.85,
    53: 1.98,
    12: 1.6,
    20: 1.7,
    25: 1.6,
    26: 1.6,
    30: 1.39,
    11: 1.6,
    19: 1.7,
    34: 1.9,
}


def vdw_radii(atomic_nums: torch.Tensor) -> torch.Tensor:
    return torch.tensor([_VDW.get(int(z), 1.7) for z in atomic_nums.tolist()], dtype=torch.float32)


def ligand_bounds(mol: Chem.Mol) -> dict:
    """Per-molecule geometry references: DG bounds matrix + topology distances.

    Returns a dict with ``lower`` / ``upper`` ``[N,N]`` distance bounds and the
    integer ``topo`` bond-count matrix (all torch tensors).
    """
    bm = rdDistGeom.GetMoleculeBoundsMatrix(mol)  # upper=triu, lower=tril
    upper_np = np.triu(bm)
    upper_np = upper_np + upper_np.T
    lower_np = np.tril(bm)
    lower_np = lower_np + lower_np.T
    return {
        "lower": torch.tensor(lower_np, dtype=torch.float32),
        "upper": torch.tensor(upper_np, dtype=torch.float32),
        "topo": torch.tensor(Chem.GetDistanceMatrix(mol), dtype=torch.float32),
    }


def check_validity(
    lig_xyz: torch.Tensor,
    bounds: dict,
    *,
    prot_xyz: torch.Tensor | None = None,
    prot_r: torch.Tensor | None = None,
    lig_r: torch.Tensor | None = None,
    bond_tol: float = 0.25,
    clash_tol: float = 0.30,
    prot_clash_scale: float = 0.75,
    return_terms: bool = False,
):
    """Validate one ligand pose. Returns bool valid (or per-check dict)."""
    n = lig_xyz.shape[0]
    d = torch.cdist(lig_xyz, lig_xyz)
    lo, up, topo = bounds["lower"], bounds["upper"], bounds["topo"]
    triu = torch.triu(torch.ones(n, n, dtype=torch.bool, device=lig_xyz.device), 1)

    m12 = (topo == 1) & triu
    m13 = (topo == 2) & triu
    mnb = (topo >= 3) & triu

    def within(mask):
        if not mask.any():
            return True
        ok = (d >= lo * (1 - bond_tol)) & (d <= up * (1 + bond_tol))
        return bool(ok[mask].all())

    bond_ok = within(m12)
    angle_ok = within(m13)
    # internal clash: non-bonded pairs must not fall below (widened) lower bound
    clash_ok = True
    if mnb.any():
        clash_ok = bool((d >= lo * (1 - clash_tol))[mnb].all())

    prot_ok = True
    if prot_xyz is not None and prot_r is not None and lig_r is not None and prot_xyz.numel():
        dp = torch.cdist(lig_xyz, prot_xyz)
        thr = prot_clash_scale * (lig_r[:, None] + prot_r[None, :])
        prot_ok = bool((dp >= thr).all())

    valid = bond_ok and angle_ok and clash_ok and prot_ok
    if return_terms:
        return {
            "valid": valid,
            "bond": bond_ok,
            "angle": angle_ok,
            "internal_clash": clash_ok,
            "protein_clash": prot_ok,
        }
    return valid


def validity_penalty(
    lig_xyz: torch.Tensor,
    bounds: dict,
    *,
    prot_xyz: torch.Tensor | None = None,
    prot_r: torch.Tensor | None = None,
    lig_r: torch.Tensor | None = None,
    bond_tol: float = 0.25,
    clash_tol: float = 0.30,
    prot_clash_scale: float = 0.75,
    w_bond: float = 1.0,
    w_angle: float = 1.0,
    w_clash: float = 1.0,
    w_prot: float = 1.0,
    return_terms: bool = False,
):
    """Continuous (>=0) version of :func:`check_validity` for use as a penalty.

    Sums squared violations: how far 1-2/1-3 distances fall outside the DG
    bounds, how far non-bonded intra pairs dip below the lower bound (internal
    clash), and how far ligand-protein pairs overlap below the vdW clash
    threshold. Differentiable in ``lig_xyz``. Returns a scalar (or per-term dict).
    """
    n = lig_xyz.shape[0]
    d = torch.cdist(lig_xyz, lig_xyz)
    lo, up, topo = bounds["lower"], bounds["upper"], bounds["topo"]
    triu = torch.triu(torch.ones(n, n, dtype=torch.bool, device=lig_xyz.device), 1)

    def out_of_bounds_sq(mask):
        if not mask.any():
            return lig_xyz.new_zeros(())
        over = (d - up * (1 + bond_tol)).clamp_min(0.0)
        under = (lo * (1 - bond_tol) - d).clamp_min(0.0)
        return ((over**2 + under**2) * mask).sum()

    bond = out_of_bounds_sq((topo == 1) & triu)
    angle = out_of_bounds_sq((topo == 2) & triu)
    mnb = (topo >= 3) & triu
    clash = (
        (((lo * (1 - clash_tol) - d).clamp_min(0.0) ** 2) * mnb).sum()
        if mnb.any()
        else lig_xyz.new_zeros(())
    )

    prot = lig_xyz.new_zeros(())
    if prot_xyz is not None and prot_r is not None and lig_r is not None and prot_xyz.numel():
        dp = torch.cdist(lig_xyz, prot_xyz)
        thr = prot_clash_scale * (lig_r[:, None] + prot_r[None, :])
        prot = (thr - dp).clamp_min(0.0).pow(2).sum()

    total = w_bond * bond + w_angle * angle + w_clash * clash + w_prot * prot
    if return_terms:
        return {
            "total": total,
            "bond": bond,
            "angle": angle,
            "internal_clash": clash,
            "protein_clash": prot,
        }
    return total


__all__ = ["vdw_radii", "ligand_bounds", "check_validity", "validity_penalty"]
