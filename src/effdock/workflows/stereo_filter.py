"""Input-only stereochemistry checks for candidate pose selection."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Sequence

from rdkit import Chem
from rdkit.Chem.rdchem import Mol
from rdkit.Chem.rdmolops import (
    AddHs,
    AssignStereochemistry,
    AssignStereochemistryFrom3D,
    RemoveStereochemistry,
    SanitizeMol,
)


@dataclass(frozen=True)
class ExplicitStereoConstraints:
    """Atom-indexed stereo explicitly declared by the input ligand."""

    atomic_numbers: tuple[int, ...]
    bonds: tuple[tuple[int, int], ...]
    tetrahedral: tuple[tuple[int, str], ...]
    double_bond: tuple[tuple[int, int, str], ...]


@dataclass(frozen=True)
class StereoCompatibility:
    tetrahedral: bool
    double_bond: bool

    @property
    def valid(self) -> bool:
        return self.tetrahedral and self.double_bond


def _normalized_double_bond_stereo(stereo: Chem.BondStereo) -> str | None:
    if stereo in (Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOTRANS):
        return "E"
    if stereo in (Chem.BondStereo.STEREOZ, Chem.BondStereo.STEREOCIS):
        return "Z"
    return None


def _heavy_copy(mol: Mol) -> Mol:
    normalized = deepcopy(mol)
    if SanitizeMol(normalized, catchErrors=True):
        raise ValueError("molecule cannot be sanitized for stereochemistry filtering")
    for atom in normalized.GetAtoms():
        atom.SetIsotope(0)
    normalized = Chem.RemoveAllHs(normalized)
    SanitizeMol(normalized)
    return normalized


def explicit_stereo_constraints(
    mol: Mol,
    *,
    heavy_atom_observable_only: bool = True,
) -> ExplicitStereoConstraints:
    """Extract explicit stereo while preserving the input heavy-atom order."""
    declared = deepcopy(mol)
    if SanitizeMol(declared, catchErrors=True):
        raise ValueError("input molecule cannot be sanitized for stereo extraction")
    for atom in declared.GetAtoms():
        atom.SetIsotope(0)
    AssignStereochemistry(declared, cleanIt=True, force=True)
    old_to_heavy: dict[int, int] = {}
    for atom in declared.GetAtoms():
        if atom.GetAtomicNum() != 1:
            old_to_heavy[atom.GetIdx()] = len(old_to_heavy)
    tetrahedral = tuple(
        sorted(
            (old_to_heavy[atom_index], label)
            for atom_index, label in Chem.FindMolChiralCenters(
                declared,
                includeUnassigned=False,
                includeCIP=True,
                useLegacyImplementation=False,
            )
            if atom_index in old_to_heavy
        )
    )
    double_bond: list[tuple[int, int, str]] = []
    for bond in declared.GetBonds():
        stereo = _normalized_double_bond_stereo(bond.GetStereo())
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        stereo_atoms = tuple(bond.GetStereoAtoms())
        explicit_hydrogen_defined = any(
            declared.GetAtomWithIdx(atom_index).GetAtomicNum() == 1
            for atom_index in stereo_atoms
        )
        if stereo is not None and begin in old_to_heavy and end in old_to_heavy:
            if heavy_atom_observable_only and explicit_hydrogen_defined:
                continue
            double_bond.append(
                (
                    min(old_to_heavy[begin], old_to_heavy[end]),
                    max(old_to_heavy[begin], old_to_heavy[end]),
                    stereo,
                )
            )
    normalized = Chem.RemoveAllHs(declared)
    SanitizeMol(normalized)
    bonds = tuple(
        sorted(
            (
                min(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()),
                max(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()),
            )
            for bond in normalized.GetBonds()
        )
    )
    return ExplicitStereoConstraints(
        atomic_numbers=tuple(atom.GetAtomicNum() for atom in normalized.GetAtoms()),
        bonds=bonds,
        tetrahedral=tetrahedral,
        double_bond=tuple(sorted(double_bond)),
    )


def _stereo_from_coordinates(mol: Mol) -> Mol:
    normalized = _heavy_copy(mol)
    if normalized.GetNumConformers() != 1 or not normalized.GetConformer().Is3D():
        raise ValueError("candidate must contain exactly one 3D conformer")
    RemoveStereochemistry(normalized)
    normalized = AddHs(normalized, addCoords=True)
    AssignStereochemistryFrom3D(normalized, replaceExistingTags=True)
    AssignStereochemistry(normalized, cleanIt=True, force=True)
    return normalized


def stereo_compatibility(
    expected: ExplicitStereoConstraints,
    candidate_mol: Mol,
) -> StereoCompatibility:
    """Check input-declared stereo after deriving candidate stereo from its 3D pose."""
    candidate = _stereo_from_coordinates(candidate_mol)
    heavy_atoms = tuple(
        atom.GetAtomicNum() for atom in candidate.GetAtoms() if atom.GetAtomicNum() != 1
    )
    if heavy_atoms != expected.atomic_numbers:
        raise ValueError("candidate atom order/elements differ from the input ligand")
    candidate_bonds = tuple(
        sorted(
            (
                min(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()),
                max(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()),
            )
            for bond in candidate.GetBonds()
            if bond.GetBeginAtom().GetAtomicNum() != 1
            and bond.GetEndAtom().GetAtomicNum() != 1
        )
    )
    if candidate_bonds != expected.bonds:
        raise ValueError("candidate connectivity differs from the input ligand")

    candidate_centers = dict(
        Chem.FindMolChiralCenters(
            candidate,
            includeUnassigned=False,
            includeCIP=True,
            useLegacyImplementation=False,
        )
    )
    tetrahedral = all(
        candidate_centers.get(atom_index) == label
        for atom_index, label in expected.tetrahedral
    )

    candidate_double_bond: dict[tuple[int, int], str] = {}
    for bond in candidate.GetBonds():
        stereo = _normalized_double_bond_stereo(bond.GetStereo())
        if stereo is not None:
            candidate_double_bond[
                (
                    min(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()),
                    max(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()),
                )
            ] = stereo
    double_bond = all(
        candidate_double_bond.get((begin, end)) == stereo
        for begin, end, stereo in expected.double_bond
    )
    return StereoCompatibility(tetrahedral=tetrahedral, double_bond=double_bond)


def select_stereo_filtered_index(
    predicted_rmsd: Sequence[float],
    stereo_valid: Sequence[bool],
    *,
    fallback_index: int,
) -> tuple[int, bool]:
    """Return the lowest predicted-RMSD valid pose and whether fallback was required."""
    if len(predicted_rmsd) != len(stereo_valid) or not predicted_rmsd:
        raise ValueError("score and stereochemistry masks must have equal non-zero length")
    if not 0 <= fallback_index < len(predicted_rmsd):
        raise ValueError("fallback index is out of range")
    valid = [index for index, keep in enumerate(stereo_valid) if keep]
    if not valid:
        return fallback_index, True
    return min(valid, key=lambda index: (float(predicted_rmsd[index]), index)), False


__all__ = [
    "ExplicitStereoConstraints",
    "StereoCompatibility",
    "explicit_stereo_constraints",
    "select_stereo_filtered_index",
    "stereo_compatibility",
]
