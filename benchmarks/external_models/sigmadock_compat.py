#!/usr/bin/env python3
"""Narrow runtime compatibility fixes for pinned SigmaDock inference."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from rdkit import Chem


def _residue_id(atom: Chem.Atom) -> tuple[str, int, str] | None:
    info = atom.GetPDBResidueInfo()
    if info is None:
        return None
    return (
        info.GetChainId(),
        info.GetResidueNumber(),
        info.GetInsertionCode(),
    )


def _atom_name(atom: Chem.Atom) -> str | None:
    info = atom.GetPDBResidueInfo()
    return info.GetName().strip() if info is not None else None


def remove_residues_without_ca(mol: Chem.Mol) -> tuple[Chem.Mol, int, int]:
    """Remove truncated protein residues that cannot enter SigmaDock's CA graph.

    SigmaDock indexes every parsed protein residue through its alpha carbon.
    PoseBench occasionally retains a terminal/incomplete residue with backbone
    atoms but no ``CA`` record.  Inventing a coordinate would change the
    receptor; retaining it crashes graph construction.  Remove only the whole
    residue groups that have PDB residue identity but no CA atom.
    """

    residue_atoms: dict[tuple[str, int, str], list[int]] = {}
    residues_with_ca: set[tuple[str, int, str]] = set()
    for atom in mol.GetAtoms():
        info = atom.GetPDBResidueInfo()
        if info is None or info.GetIsHeteroAtom():
            # Cofactors, waters, and metal ions are not protein residues and
            # are not expected to have alpha carbons. Preserve them exactly.
            continue
        residue = _residue_id(atom)
        assert residue is not None
        residue_atoms.setdefault(residue, []).append(atom.GetIdx())
        if _atom_name(atom) == "CA":
            residues_with_ca.add(residue)

    incomplete = set(residue_atoms).difference(residues_with_ca)
    if not incomplete:
        return mol, 0, 0

    removed_atoms = sorted(
        (index for residue in incomplete for index in residue_atoms[residue]),
        reverse=True,
    )
    editable = Chem.RWMol(mol)
    for atom_index in removed_atoms:
        editable.RemoveAtom(atom_index)
    cleaned = editable.GetMol()
    cleaned.UpdatePropertyCache(strict=False)
    Chem.SanitizeMol(cleaned)
    return cleaned, len(incomplete), len(removed_atoms)


def _is_peptide_bond(begin: Chem.Atom, end: Chem.Atom) -> bool:
    begin_residue = _residue_id(begin)
    end_residue = _residue_id(end)
    if begin_residue is None or end_residue is None:
        return False
    begin_chain, begin_number, _ = begin_residue
    end_chain, end_number, _ = end_residue
    return (
        begin_chain == end_chain
        and abs(begin_number - end_number) == 1
        and {_atom_name(begin), _atom_name(end)} == {"C", "N"}
    )


def recover_nonstandard_inter_residue_bonds(
    pdb_string: str,
) -> tuple[Chem.Mol | None, int]:
    """Retry a failed PDB parse after removing impossible inferred residue links.

    RDKit infers protein bonds from distance when a PDB block has no CONECT
    records. Independently aligned predicted chains can clash closely enough to
    create a spurious inter-chain bond. We preserve all intra-chain bonds and
    protein.  Keep bonds within a residue, sequential peptide C--N bonds, and
    disulfides.  Remove only other inferred inter-residue links.  The fallback
    runs only after the normal sanitized RDKit parser has failed.
    """

    raw = Chem.MolFromPDBBlock(
        pdb_string,
        removeHs=True,
        sanitize=False,
        proximityBonding=True,
    )
    if raw is None:
        return None, 0

    editable = Chem.RWMol(raw)
    removed = 0
    for bond in list(raw.GetBonds()):
        begin = bond.GetBeginAtom()
        end = bond.GetEndAtom()
        begin_residue = _residue_id(begin)
        end_residue = _residue_id(end)
        different_residues = (
            begin_residue is not None
            and end_residue is not None
            and begin_residue != end_residue
        )
        disulfide = begin.GetSymbol() == "S" and end.GetSymbol() == "S"
        if different_residues and not _is_peptide_bond(begin, end) and not disulfide:
            editable.RemoveBond(begin.GetIdx(), end.GetIdx())
            removed += 1

    if removed == 0:
        return None, 0
    recovered = editable.GetMol()
    try:
        Chem.SanitizeMol(recovered)
    except Exception:
        return None, removed
    return recovered, removed


def assign_missing_stereochemistry_from_3d(mol: Chem.Mol) -> int:
    """Restore stereochemistry normally perceived by a sanitized SDF reader.

    SigmaDock's defensive SDF path reads with ``sanitize=False`` and then
    sanitizes the molecule in-place.  RDKit does not infer double-bond stereo
    in that sequence, which makes ETKDG pathologically slow for polyenes.  The
    SDF already contains the experimental 3D coordinates, so perceive the
    chemical stereochemistry from those coordinates before generating the
    independent inference conformer.
    """

    before = sum(
        bond.GetStereo() != Chem.BondStereo.STEREONONE
        for bond in mol.GetBonds()
        if bond.GetBondType() == Chem.BondType.DOUBLE
    )
    if mol.GetNumConformers() > 0:
        Chem.AssignStereochemistryFrom3D(mol)
    after = sum(
        bond.GetStereo() != Chem.BondStereo.STEREONONE
        for bond in mol.GetBonds()
        if bond.GetBondType() == Chem.BondType.DOUBLE
    )
    return max(0, after - before)


def install_sigmadock_parser_compat() -> None:
    """Patch the parser alias used by SigmaDataset, without editing upstream."""

    import sigmadock.chem.parsing as parsing
    import sigmadock.data as data

    if getattr(parsing, "_effdock_compat_installed", False):
        return

    original: Callable[..., Any] = parsing.read_pdb_from_string
    original_ligand_reader: Callable[..., list[Chem.Mol]] = (
        parsing.read_ligands_from_sdf
    )

    def compatible_read_pdb_from_string(
        pdb_string: str,
        as_biopython: bool = False,
    ) -> Any:
        parsed = original(pdb_string, as_biopython=as_biopython)
        if as_biopython:
            return parsed
        if parsed is None:
            parsed, removed = recover_nonstandard_inter_residue_bonds(pdb_string)
        else:
            removed = 0
        if parsed is None:
            return None
        if removed:
            print(
                "[SigmaDock compatibility] recovered predicted pocket after "
                f"removing {removed} nonstandard inter-residue proximity bond(s)."
            )
        cleaned, removed_residues, removed_atoms = remove_residues_without_ca(parsed)
        if removed_residues:
            print(
                "[SigmaDock compatibility] removed "
                f"{removed_residues} incomplete residue(s) / {removed_atoms} atom(s) "
                "without CA before pocket graph construction."
            )
        return cleaned

    def compatible_read_ligands_from_sdf(
        ligand_sdf: str | Path,
        remove_hs: bool = True,
    ) -> list[Chem.Mol]:
        mols = original_ligand_reader(ligand_sdf, remove_hs=remove_hs)
        assigned = sum(assign_missing_stereochemistry_from_3d(mol) for mol in mols)
        if assigned:
            print(
                "[SigmaDock compatibility] restored "
                f"{assigned} missing double-bond stereochemistry assignment(s) "
                f"from 3D SDF coordinates in {ligand_sdf}."
            )
        return mols

    parsing.read_pdb_from_string = compatible_read_pdb_from_string
    parsing.read_ligands_from_sdf = compatible_read_ligands_from_sdf
    data.read_pdb_from_string = compatible_read_pdb_from_string
    data.read_ligands_from_sdf = compatible_read_ligands_from_sdf
    parsing._effdock_compat_installed = True
