"""Distance-based recovery of native protein-ligand interaction fingerprints.

The first version reports hydrogen-bond, salt-bridge, hydrophobic, and
metal-donor contact recovery. Hydrogen bonds are distance-only because the
standard receptor input omits hydrogens; water-mediated contacts are excluded
because the current protein parser drops waters.
"""

from __future__ import annotations

from pathlib import Path
from typing import TypeAlias

import torch
from rdkit import Chem

from effdock.preprocess.ligand import compute_atom_pharmacophore_features
from effdock.preprocess.protein import ParsedAtom, _parse_pdb_lines, _patom_pharmacophore

ProteinAtomKey: TypeAlias = tuple[str, int, str, str, str, str]
ContactKey: TypeAlias = tuple[int, ProteinAtomKey]

_INTERACTION_TYPES = ("hbond", "salt_bridge", "hydrophobic", "metal_coordination")
_DISTANCE_RANGES = {
    "hbond": (2.2, 3.5),
    "salt_bridge": (2.0, 4.0),
    "hydrophobic": (3.0, 4.5),
    "metal_coordination": (1.8, 2.8),
}


def _coordinates(value: torch.Tensor, n_atoms: int, name: str) -> torch.Tensor:
    coords = torch.as_tensor(value, dtype=torch.float32, device="cpu")
    if coords.shape != (n_atoms, 3):
        raise ValueError(f"{name} must have shape ({n_atoms}, 3); got {tuple(coords.shape)}")
    if not torch.isfinite(coords).all():
        raise ValueError(f"{name} contains non-finite coordinates")
    return coords


def _ligand_fingerprint(
    mol: Chem.Mol,
    ligand_xyz: torch.Tensor,
    protein_atoms: list[ParsedAtom],
) -> dict[str, set[ContactKey]]:
    n_lig = mol.GetNumAtoms()
    if any(atom.GetAtomicNum() == 1 for atom in mol.GetAtoms()):
        raise ValueError("interaction recovery expects a heavy-atom ligand")
    lig_xyz = _coordinates(ligand_xyz, n_lig, "ligand_xyz")
    result = {name: set() for name in _INTERACTION_TYPES}
    if not protein_atoms:
        return result

    prot_xyz = torch.tensor([atom.coords for atom in protein_atoms], dtype=torch.float32)
    distances = torch.cdist(lig_xyz, prot_xyz)
    ligand_features = compute_atom_pharmacophore_features(mol)
    ranks = list(Chem.CanonicalRankAtoms(mol, breakTies=False, includeChirality=True))

    prot_features = [_patom_pharmacophore(atom.res_name, atom.atom_name) for atom in protein_atoms]
    prot_donor = torch.tensor([flags[0] for flags in prot_features])
    prot_acceptor = torch.tensor([flags[1] for flags in prot_features])
    prot_positive = torch.tensor([flags[2] for flags in prot_features])
    prot_negative = torch.tensor([flags[3] for flags in prot_features])
    prot_hydrophobic = torch.tensor([flags[4] for flags in prot_features])
    prot_metal = torch.tensor([atom.is_metal for atom in protein_atoms])

    lig_donor = ligand_features["atom_is_donor"]
    lig_acceptor = ligand_features["atom_is_acceptor"]
    lig_positive = ligand_features["atom_is_positive"]
    lig_negative = ligand_features["atom_is_negative"]
    lig_hydrophobic = ligand_features["atom_is_hydrophobe"]

    compatible_pairs = {
        "hbond": (lig_donor[:, None] & prot_acceptor[None, :])
        | (lig_acceptor[:, None] & prot_donor[None, :]),
        "salt_bridge": (lig_positive[:, None] & prot_negative[None, :])
        | (lig_negative[:, None] & prot_positive[None, :]),
        "hydrophobic": lig_hydrophobic[:, None] & prot_hydrophobic[None, :],
        "metal_coordination": (lig_acceptor | lig_negative)[:, None] & prot_metal[None, :],
    }

    for interaction_type, compatible in compatible_pairs.items():
        lower, upper = _DISTANCE_RANGES[interaction_type]
        atom_pairs = (compatible & (distances >= lower) & (distances <= upper)).nonzero()
        for ligand_idx, protein_idx in atom_pairs.tolist():
            atom = protein_atoms[protein_idx]
            protein_key = (
                atom.chain.strip(),
                int(atom.res_num),
                atom.icode,
                atom.res_name,
                atom.atom_name,
                atom.element,
            )
            result[interaction_type].add((int(ranks[ligand_idx]), protein_key))
    return result


def _f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None:
        return None
    return 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0


def _score_fingerprints(
    predicted: dict[str, set[ContactKey]],
    reference: dict[str, set[ContactKey]],
) -> dict[str, float | int | None]:
    output: dict[str, float | int | None] = {}
    total_predicted = total_reference = total_recovered = 0
    reference_class_f1: list[float] = []

    for interaction_type in _INTERACTION_TYPES:
        pred_contacts = predicted[interaction_type]
        ref_contacts = reference[interaction_type]
        recovered = len(pred_contacts & ref_contacts)
        n_pred, n_ref = len(pred_contacts), len(ref_contacts)
        precision = recovered / n_pred if n_pred else (0.0 if n_ref else None)
        recall = recovered / n_ref if n_ref else None
        f1 = _f1(precision, recall)
        output[f"{interaction_type}_precision"] = precision
        output[f"{interaction_type}_recall"] = recall
        output[f"{interaction_type}_f1"] = f1
        output[f"{interaction_type}_predicted_count"] = n_pred
        output[f"{interaction_type}_reference_count"] = n_ref
        output[f"{interaction_type}_recovered_count"] = recovered
        total_predicted += n_pred
        total_reference += n_ref
        total_recovered += recovered
        if n_ref and f1 is not None:
            reference_class_f1.append(f1)

    micro_precision = (
        total_recovered / total_predicted if total_predicted else (0.0 if total_reference else None)
    )
    micro_recall = total_recovered / total_reference if total_reference else None
    output.update(
        {
            "interaction_micro_precision": micro_precision,
            "interaction_micro_recall": micro_recall,
            "interaction_micro_f1": _f1(micro_precision, micro_recall),
            "interaction_macro_f1": (
                sum(reference_class_f1) / len(reference_class_f1) if reference_class_f1 else None
            ),
            "interaction_predicted_count": total_predicted,
            "interaction_reference_count": total_reference,
            "interaction_recovered_count": total_recovered,
        }
    )
    return output


def interaction_recovery(
    mol: Chem.Mol,
    predicted_xyz: torch.Tensor,
    reference_xyz: torch.Tensor,
    protein_pdb: str | Path,
) -> dict[str, float | int | None]:
    """Compare a pose's typed contacts with the crystal-pose contacts.

    ``predicted_xyz`` and ``reference_xyz`` must both use ``mol``'s heavy-atom
    order and the receptor PDB's absolute coordinate frame. Ligand symmetry is
    handled by grouping equivalent atoms using RDKit canonical ranks. Contact
    identities retain the protein chain, residue, and atom identity.

    The returned precision/recall/F1 values are fractions. A class with no
    reference contacts has ``None`` recall/F1 and is omitted from macro-F1.
    The interaction classes and distance cutoffs are intentionally explicit
    so downstream reports can state exactly what was measured.
    """
    n_atoms = mol.GetNumAtoms()
    pred_xyz = _coordinates(predicted_xyz, n_atoms, "predicted_xyz")
    ref_xyz = _coordinates(reference_xyz, n_atoms, "reference_xyz")
    protein_atoms = _parse_pdb_lines(Path(protein_pdb))
    if not protein_atoms:
        raise ValueError(f"no usable protein atoms in {protein_pdb}")
    predicted = _ligand_fingerprint(mol, pred_xyz, protein_atoms)
    reference = _ligand_fingerprint(mol, ref_xyz, protein_atoms)
    return _score_fingerprints(predicted, reference)


__all__ = ["interaction_recovery"]
