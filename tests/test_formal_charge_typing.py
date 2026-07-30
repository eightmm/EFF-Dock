from __future__ import annotations

import pytest
import torch
from rdkit import Chem

from effdock.guidance.parameterization import (
    interaction_parameter_identity,
    load_interaction_v1,
)
from effdock.guidance.system import (
    InteractionTopology,
    _protein_charge_sites,
    type_ligand_interactions,
)
from effdock.preprocess.protein import ParsedAtom


def _mol(smiles: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None
    return mol


def test_ligand_charge_sites_conserve_input_formal_charge() -> None:
    for smiles in (
        "CC(=O)[O-]",
        "NC(=[NH2+])N",
        "CC(=[NH2+])N",
        "C[N+](C)(C)C",
        "[NH3+]CC(=O)[O-]",
        "CC",
    ):
        mol = _mol(smiles)
        typed = type_ligand_interactions(mol)
        membership = typed["charge_site_membership"]
        charges = typed["charge_site_charge"]
        expected_charge = sum(atom.GetFormalCharge() for atom in mol.GetAtoms())

        assert membership.shape == (
            charges.numel(),
            mol.GetNumAtoms(),
        )
        torch.testing.assert_close(
            membership.sum(dim=1),
            torch.ones(charges.numel(), dtype=torch.float64),
        )
        assert float(charges.sum()) == expected_charge
        assert bool((membership > 0).sum(dim=0).le(1).all())


def test_carboxylate_uses_one_resonance_equivalent_centroid_site() -> None:
    mol = _mol("CC(=O)[O-]")
    typed = type_ligand_interactions(mol)
    membership = typed["charge_site_membership"]
    charges = typed["charge_site_charge"]

    assert charges.tolist() == [-1.0]
    assert "carboxylate" in typed["charge_site_labels"][0]
    members = membership[0].nonzero(as_tuple=False).flatten()
    assert members.numel() == 2
    assert {mol.GetAtomWithIdx(int(index)).GetSymbol() for index in members} == {"O"}
    torch.testing.assert_close(
        membership[0, members],
        torch.full((2,), 0.5, dtype=torch.float64),
    )

    coords = torch.arange(
        mol.GetNumAtoms() * 3,
        dtype=torch.float64,
    ).reshape(-1, 3)
    site_coord = membership @ coords
    torch.testing.assert_close(
        site_coord[0],
        coords[members].mean(dim=0),
    )


def test_zwitterion_retains_both_opposite_charge_sites() -> None:
    typed = type_ligand_interactions(_mol("[NH3+]CC(=O)[O-]"))
    charges = typed["charge_site_charge"]
    assert sorted(charges.tolist()) == [-1.0, 1.0]
    assert typed["charge_site_membership"].shape[0] == 2
    assert any("carboxylate" in label for label in typed["charge_site_labels"])
    assert any("ligand:atom" in label for label in typed["charge_site_labels"])


def test_delocalized_positive_charge_groups_use_equal_nitrogen_weights() -> None:
    for smiles, expected_name, expected_members in (
        ("NC(=[NH2+])N", "guanidinium", 3),
        ("CC(=[NH2+])N", "amidinium", 2),
    ):
        mol = _mol(smiles)
        typed = type_ligand_interactions(mol)
        labels = typed["charge_site_labels"]
        site_index = next(index for index, label in enumerate(labels) if expected_name in label)
        membership = typed["charge_site_membership"][site_index]
        members = membership.nonzero(as_tuple=False).flatten()
        assert members.numel() == expected_members
        assert {mol.GetAtomWithIdx(int(index)).GetSymbol() for index in members} == {"N"}
        torch.testing.assert_close(
            membership[members],
            torch.full(
                (expected_members,),
                1.0 / expected_members,
                dtype=torch.float64,
            ),
        )
        assert typed["charge_site_charge"][site_index] == 1


def test_remaining_formal_charge_becomes_atom_centered_site() -> None:
    mol = _mol("C[N+](C)(C)C")
    typed = type_ligand_interactions(mol)
    membership = typed["charge_site_membership"]

    assert typed["charge_site_charge"].tolist() == [1.0]
    charged_atom = next(atom.GetIdx() for atom in mol.GetAtoms() if atom.GetFormalCharge() == 1)
    assert membership[0, charged_atom] == 1
    assert int(torch.count_nonzero(membership[0])) == 1
    assert "ligand:atom" in typed["charge_site_labels"][0]


def _protein_atom(
    normalized_residue: str,
    raw_residue: str,
    residue_number: int,
    atom_name: str,
    atom_offset: int,
) -> ParsedAtom:
    return ParsedAtom(
        record_type="ATOM",
        atom_name=atom_name,
        res_name=normalized_residue,
        chain="A",
        res_num=residue_number,
        icode="",
        coords=(
            float(residue_number),
            float(atom_offset),
            0.0,
        ),
        element=atom_name[0],
        is_metal=False,
        raw_res_name=raw_residue,
    )


def test_protein_charge_groups_are_complete_variant_aware_and_shell_safe() -> None:
    residue_specs = (
        ("ARG", "ARG", 1, ("NE", "NH1", "NH2")),
        ("LYS", "LYS", 2, ("NZ",)),
        ("ASP", "ASP", 3, ("OD1", "OD2")),
        ("GLU", "GLU", 4, ("OE1", "OE2")),
        ("HIS", "HIP", 5, ("ND1", "NE2")),
        ("HIS", "HID", 6, ("ND1", "NE2")),
        ("HIS", "HIE", 7, ("ND1", "NE2")),
        ("HIS", "HIS", 8, ("ND1", "NE2")),
        ("ARG", "ARG", 9, ("NE", "NH1")),
        ("ASP", "ASH", 10, ("OD1", "OD2")),
        ("GLU", "GLU", 11, ("OE1", "OE2")),
    )
    protein_atoms = [
        _protein_atom(
            normalized,
            raw,
            residue_number,
            atom_name,
            atom_offset,
        )
        for normalized, raw, residue_number, atom_names in residue_specs
        for atom_offset, atom_name in enumerate(atom_names)
    ]
    keep = torch.ones(len(protein_atoms), dtype=torch.bool)
    for atom_index, atom in enumerate(protein_atoms):
        if atom.res_num == 11 and atom.atom_name == "OE2":
            keep[atom_index] = False

    typed = _protein_charge_sites(
        protein_atoms,
        keep,
        load_interaction_v1()["protein_charge_groups"],
    )
    membership = typed["membership"]

    assert typed["charge"].tolist() == [
        1.0,
        1.0,
        -1.0,
        -1.0,
        1.0,
    ]
    assert membership.shape == (5, int(keep.sum()))
    torch.testing.assert_close(
        membership.sum(dim=1),
        torch.ones(5, dtype=torch.float64),
    )
    assert bool((membership > 0).sum(dim=0).le(1).all())
    assert any("HIP5" in label for label in typed["labels"])

    exclusions = typed["exclusion_labels"]
    for expected in (
        "HID6:HIS:charge_state_not_admitted",
        "HIE7:HIS:charge_state_not_admitted",
        "HIS8:HIS:charge_state_not_admitted",
        "ARG9:ARG:missing_or_duplicate_members=NH2",
        "ASH10:ASP:charge_state_not_admitted",
        "GLU11:GLU:members_outside_shell=OE2",
    ):
        assert any(expected in label for label in exclusions)


def test_new_charge_fields_have_safe_empty_defaults_and_hash() -> None:
    topology = InteractionTopology(
        ligand_neighbor_index=torch.empty((2, 0), dtype=torch.long),
        ligand_direction_target_cosine=torch.zeros(2),
        ligand_direction_geometry_valid=torch.zeros(2, dtype=torch.bool),
        ligand_is_donor=torch.zeros(2, dtype=torch.bool),
        ligand_is_acceptor=torch.zeros(2, dtype=torch.bool),
        ligand_is_hydrophobe=torch.zeros(2, dtype=torch.bool),
        ligand_is_geometry_excluded_hbond_site=torch.zeros(
            2,
            dtype=torch.bool,
        ),
        protein_is_donor=torch.zeros(3, dtype=torch.bool),
        protein_is_acceptor=torch.zeros(3, dtype=torch.bool),
        protein_is_hydrophobe=torch.zeros(3, dtype=torch.bool),
        protein_outward_direction=torch.zeros((3, 3)),
        protein_direction_target_cosine=torch.zeros(3),
        protein_direction_quality=torch.zeros(3),
        protein_direction_valid=torch.zeros(3, dtype=torch.bool),
        protein_is_ambiguous_histidine=torch.zeros(
            3,
            dtype=torch.bool,
        ),
        protein_is_unsupported_variant=torch.zeros(
            3,
            dtype=torch.bool,
        ),
        protein_is_geometry_excluded_hbond_site=torch.zeros(
            3,
            dtype=torch.bool,
        ),
        ligand_atom_labels=("0:C", "1:N"),
        protein_atom_labels=("A:ALA1:N", "A:ALA1:CA", "A:ALA1:C"),
    )

    assert topology.ligand_charge_site_membership.shape == (0, 0)
    assert topology.protein_charge_site_membership.shape == (0, 0)
    moved = topology.to(torch.device("cpu"), torch.float64)
    assert moved.ligand_charge_site_charge.dtype == torch.float64
    assert (
        moved.reference_sha256()
        == topology.to(
            torch.device("cpu"),
            torch.float64,
        ).reference_sha256()
    )
    counts = moved.term_counts()
    assert counts["ligand_charge_sites"] == 0
    assert counts["protein_charge_sites"] == 0
    assert counts["formal_charge_candidate_pairs"] == 0

    parameter_identity = interaction_parameter_identity()
    assert parameter_identity["version"] == "1.3.0"
    assert parameter_identity["formula_version"] == "effdock-interaction-diagnostic-4"
    assert len(parameter_identity["sha256"]) == 64


def test_malformed_charge_topology_fails_closed() -> None:
    base = {
        "ligand_neighbor_index": torch.empty((2, 0), dtype=torch.long),
        "ligand_direction_target_cosine": torch.zeros(2),
        "ligand_direction_geometry_valid": torch.zeros(2, dtype=torch.bool),
        "ligand_is_donor": torch.zeros(2, dtype=torch.bool),
        "ligand_is_acceptor": torch.zeros(2, dtype=torch.bool),
        "ligand_is_hydrophobe": torch.zeros(2, dtype=torch.bool),
        "ligand_is_geometry_excluded_hbond_site": torch.zeros(2, dtype=torch.bool),
        "protein_is_donor": torch.zeros(1, dtype=torch.bool),
        "protein_is_acceptor": torch.zeros(1, dtype=torch.bool),
        "protein_is_hydrophobe": torch.zeros(1, dtype=torch.bool),
        "protein_outward_direction": torch.zeros((1, 3)),
        "protein_direction_target_cosine": torch.zeros(1),
        "protein_direction_quality": torch.zeros(1),
        "protein_direction_valid": torch.zeros(1, dtype=torch.bool),
        "protein_is_ambiguous_histidine": torch.zeros(1, dtype=torch.bool),
        "protein_is_unsupported_variant": torch.zeros(1, dtype=torch.bool),
        "protein_is_geometry_excluded_hbond_site": torch.zeros(1, dtype=torch.bool),
        "ligand_atom_labels": ("0:C", "1:N"),
        "protein_atom_labels": ("A:ASP1:OD1",),
        "ligand_charge_site_charge": torch.tensor([1.0]),
        "ligand_charge_site_labels": ("ligand:N+",),
    }
    for membership in (
        torch.tensor([[0.2, 0.2]]),
        torch.tensor([[-0.5, 1.5]]),
        torch.tensor([[1.0, 0.0], [0.5, 0.5]]),
    ):
        kwargs = dict(base)
        kwargs["ligand_charge_site_membership"] = membership
        if membership.shape[0] == 2:
            kwargs["ligand_charge_site_charge"] = torch.tensor([1.0, -1.0])
            kwargs["ligand_charge_site_labels"] = ("first", "second")
        with pytest.raises(ValueError):
            InteractionTopology(**kwargs)
