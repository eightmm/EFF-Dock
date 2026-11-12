from __future__ import annotations

from pathlib import Path

import pytest
import torch
from rdkit import Chem
from rdkit.Geometry import Point3D

from effdock.evaluation.interaction_recovery import interaction_recovery


def _protein_atom(
    atom_name: str,
    residue: str,
    element: str,
    xyz: tuple[float, float, float],
    *,
    residue_number: int = 1,
) -> str:
    x, y, z = xyz
    return (
        f"ATOM  {1:5d} {atom_name:>4s} {residue:>3s} A{residue_number:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}{1.0:6.2f}{20.0:6.2f}          {element:>2s}\n"
    )


def _write_protein(path: Path, *atoms: str) -> Path:
    path.write_text("".join(atoms) + "TER\nEND\n", encoding="ascii")
    return path


def _mol_with_coords(smiles: str, coords: list[tuple[float, float, float]]) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None
    assert len(coords) == mol.GetNumAtoms()
    conf = Chem.Conformer(mol.GetNumAtoms())
    for idx, xyz in enumerate(coords):
        conf.SetAtomPosition(idx, Point3D(*xyz))
    mol.AddConformer(conf)
    return mol


def test_native_contacts_recover_hbond_and_salt_bridge(tmp_path: Path) -> None:
    mol = _mol_with_coords("C[NH3+]", [(1.5, 0.0, 0.0), (0.0, 0.0, 0.0)])
    protein = _write_protein(
        tmp_path / "protein.pdb",
        _protein_atom("OE1", "GLU", "O", (2.8, 0.0, 0.0)),
    )

    scores = interaction_recovery(
        mol,
        torch.tensor([[1.5, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        torch.tensor([[1.5, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        protein,
    )

    assert scores["hbond_reference_count"] == 1
    assert scores["hbond_f1"] == 1.0
    assert scores["salt_bridge_reference_count"] == 1
    assert scores["salt_bridge_f1"] == 1.0
    assert scores["interaction_micro_f1"] == 1.0
    assert scores["interaction_macro_f1"] == 1.0


def test_displaced_pose_loses_native_contacts(tmp_path: Path) -> None:
    mol = _mol_with_coords("C[NH3+]", [(1.5, 0.0, 0.0), (0.0, 0.0, 0.0)])
    protein = _write_protein(
        tmp_path / "protein.pdb",
        _protein_atom("OE1", "GLU", "O", (2.8, 0.0, 0.0)),
    )

    scores = interaction_recovery(
        mol,
        torch.tensor([[11.5, 0.0, 0.0], [10.0, 0.0, 0.0]]),
        torch.tensor([[1.5, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        protein,
    )

    assert scores["interaction_micro_precision"] == 0.0
    assert scores["interaction_micro_recall"] == 0.0
    assert scores["interaction_micro_f1"] == 0.0


def test_ligand_symmetry_equivalent_contacts_are_matched(tmp_path: Path) -> None:
    mol = _mol_with_coords(
        "CC(=O)[O-]",
        [(0.0, 0.0, 0.0), (1.5, 0.0, 0.0), (2.1, 1.0, 0.0), (2.1, -1.0, 0.0)],
    )
    protein = _write_protein(
        tmp_path / "protein.pdb",
        _protein_atom("OG", "SER", "O", (4.9, 1.0, 0.0)),
    )
    ref = torch.tensor(
        [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [2.1, 1.0, 0.0], [2.1, -1.0, 0.0]]
    )
    pred = ref.clone()
    pred[[2, 3]] = pred[[3, 2]]

    scores = interaction_recovery(mol, pred, ref, protein)

    assert scores["hbond_reference_count"] == 1
    assert scores["hbond_recovered_count"] == 1
    assert scores["interaction_micro_f1"] == 1.0


def test_interaction_recovery_marks_absent_reference_class_undefined(tmp_path: Path) -> None:
    mol = _mol_with_coords("CC", [(0.0, 0.0, 0.0), (1.5, 0.0, 0.0)])
    protein = _write_protein(
        tmp_path / "protein.pdb",
        _protein_atom("CA", "ALA", "C", (10.0, 0.0, 0.0)),
    )

    coords = torch.tensor([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
    scores = interaction_recovery(mol, coords, coords, protein)

    assert scores["interaction_reference_count"] == 0
    assert scores["interaction_micro_f1"] is None
    assert scores["hbond_f1"] is None


def test_interaction_recovery_requires_matched_heavy_atom_coordinates(tmp_path: Path) -> None:
    mol = _mol_with_coords("CC", [(0.0, 0.0, 0.0), (1.5, 0.0, 0.0)])
    protein = _write_protein(
        tmp_path / "protein.pdb",
        _protein_atom("CA", "ALA", "C", (10.0, 0.0, 0.0)),
    )

    with pytest.raises(ValueError, match="predicted_xyz must have shape"):
        interaction_recovery(mol, torch.zeros((1, 3)), torch.zeros((2, 3)), protein)
