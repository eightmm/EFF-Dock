"""Score docked ligand poses with the torch Vina function + DG strain.

Bridges protein-PDB parsing (elements + AutoDock-style pharmacophore flags) and
the differentiable scoring in :mod:`effdock.evaluation.vina`. Used by the docking
pipeline to rank generated poses by binding free energy and pose validity.
"""

from __future__ import annotations

from pathlib import Path

import torch
from rdkit import Chem

from ..preprocess.protein import _parse_pdb_lines, _patom_pharmacophore
from .vina import (
    ligand_dg_reference,
    vina_atom_radii,
    vina_atom_types,
    vina_score_with_strain,
)

# Element symbol -> atomic number for protein heavy atoms (radii lookup).
_SYM2Z: dict[str, int] = {
    "C": 6,
    "N": 7,
    "O": 8,
    "S": 16,
    "P": 15,
    "F": 9,
    "CL": 17,
    "BR": 35,
    "I": 53,
    "SE": 34,
    "ZN": 30,
    "MG": 12,
    "MN": 25,
    "FE": 26,
    "CA": 20,
    "NA": 11,
    "K": 19,
}


def build_protein_vina_inputs(
    protein_pdb: str | Path,
    near_coords: torch.Tensor,
    cutoff: float = 10.0,
) -> dict[str, torch.Tensor]:
    """Parse protein heavy atoms within ``cutoff`` Å of ``near_coords``.

    Args:
        protein_pdb: protein PDB path.
        near_coords: ``[M, 3]`` ligand coordinates the pocket is built around
            (absolute frame, same as the PDB).
        cutoff: keep protein heavy atoms within this distance of any
            ``near_coords`` atom.

    Returns:
        dict with ``coords`` ``[N, 3]``, ``radii`` ``[N]`` and bool
        ``is_donor`` / ``is_acceptor`` / ``is_hydrophobic`` ``[N]``.
    """
    atoms = [a for a in _parse_pdb_lines(Path(protein_pdb)) if a.element != "H"]
    all_xyz = torch.tensor([a.coords for a in atoms], dtype=torch.float32)
    keep = torch.cdist(all_xyz, near_coords).min(dim=1).values < cutoff
    idx = keep.nonzero().flatten().tolist()

    z = torch.tensor([_SYM2Z.get(atoms[i].element.upper(), 6) for i in idx])
    donor, acceptor, hydro = [], [], []
    for i in idx:
        d, a, _, _, h = _patom_pharmacophore(atoms[i].res_name, atoms[i].atom_name)
        donor.append(d)
        acceptor.append(a)
        hydro.append(h)
    return {
        "coords": all_xyz[keep],
        "radii": vina_atom_radii(z),
        "atomic_nums": z,
        "is_donor": torch.tensor(donor),
        "is_acceptor": torch.tensor(acceptor),
        "is_hydrophobic": torch.tensor(hydro),
    }


def score_poses(
    mol: Chem.Mol,
    poses: list[torch.Tensor],
    protein_pdb: str | Path,
    *,
    pocket_center: torch.Tensor,
    frag_id: torch.Tensor | None = None,
    pocket_cutoff: float = 10.0,
    w_strain: float = 1.0,
) -> list[dict[str, float]]:
    """Score each docked pose with Vina free energy + ligand DG strain.

    Args:
        mol: ligand RDKit mol (heavy atoms, with the conformer used for docking —
            its bond lengths define the DG reference).
        poses: list of ``[N_atom, 3]`` predicted positions, **pocket-centered**
            (the docker's ``atom_pos_pred``); ``pocket_center`` is added back.
        protein_pdb: protein PDB path.
        pocket_center: ``[3]`` center to lift poses into the absolute frame.
        frag_id: ``[N_atom]`` fragment id per ligand atom — enables the
            inter-fragment clash term (skipped if ``None``).
        pocket_cutoff: protein-atom inclusion radius around the poses.
        w_strain: weight on the strain penalty added to the Vina energy.

    Returns:
        One dict per pose with ``vina``, ``strain``, ``total`` (kcal/mol;
        lower = better), in the input order.
    """
    lig_t = vina_atom_types(mol)
    lig_r = vina_atom_radii(lig_t["atomic_nums"])
    dg = ligand_dg_reference(mol)

    abs_poses = [p + pocket_center for p in poses]
    prot = build_protein_vina_inputs(protein_pdb, torch.cat(abs_poses, dim=0), cutoff=pocket_cutoff)

    results: list[dict[str, float]] = []
    for lig_xyz in abs_poses:
        out = vina_score_with_strain(
            lig_xyz,
            prot["coords"],
            lig_r,
            prot["radii"],
            lig_is_hydrophobic=lig_t["is_hydrophobic"],
            prot_is_hydrophobic=prot["is_hydrophobic"],
            lig_is_donor=lig_t["is_donor"],
            prot_is_donor=prot["is_donor"],
            lig_is_acceptor=lig_t["is_acceptor"],
            prot_is_acceptor=prot["is_acceptor"],
            bond_index=dg["bond_index"],
            bond_ref_len=dg["bond_ref_len"],
            num_rotatable_bonds=lig_t["num_rotatable_bonds"],
            frag_id=frag_id,
            w_strain=w_strain,
            return_components=True,
        )
        results.append({k: float(out[k]) for k in ("vina", "strain", "total")})
    return results
