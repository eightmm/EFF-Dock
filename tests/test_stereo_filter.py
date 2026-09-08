from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import AllChem

from effdock.workflows.stereo_filter import (
    explicit_stereo_constraints,
    select_stereo_filtered_index,
    stereo_compatibility,
)


def _embedded(smiles: str, seed: int = 7) -> Chem.Mol:
    molecule = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert AllChem.EmbedMolecule(molecule, randomSeed=seed) == 0
    return molecule


def test_tetrahedral_coordinate_inversion_is_rejected() -> None:
    expected_molecule = Chem.MolFromSmiles("F[C@](Cl)(Br)I")
    assert expected_molecule is not None
    expected = explicit_stereo_constraints(expected_molecule)
    candidate = _embedded("F[C@](Cl)(Br)I")
    assert stereo_compatibility(expected, candidate).valid

    conformer = candidate.GetConformer()
    for index in range(candidate.GetNumAtoms()):
        position = conformer.GetAtomPosition(index)
        conformer.SetAtomPosition(index, (-position.x, position.y, position.z))
    compatibility = stereo_compatibility(expected, candidate)
    assert not compatibility.tetrahedral
    assert compatibility.double_bond


def test_double_bond_stereochemistry_is_rejected() -> None:
    expected_molecule = Chem.MolFromSmiles("F/C=C/Cl")
    assert expected_molecule is not None
    expected = explicit_stereo_constraints(expected_molecule)
    matching = _embedded("F/C=C/Cl")
    inverted = _embedded("F/C=C\\Cl")

    assert stereo_compatibility(expected, matching).valid
    compatibility = stereo_compatibility(expected, inverted)
    assert compatibility.tetrahedral
    assert not compatibility.double_bond


def test_explicit_hydrogen_double_bond_is_ignored_by_heavy_atom_policy() -> None:
    expected_molecule = Chem.MolFromSmiles("[H]/N=C\\1/CCCN1")
    assert expected_molecule is not None
    expected = explicit_stereo_constraints(expected_molecule)
    assert len(expected.double_bond) == 0
    candidate = Chem.RemoveAllHs(_embedded("[H]/N=C\\1/CCCN1"))

    assert stereo_compatibility(expected, candidate).valid


def test_explicit_hydrogen_double_bond_can_be_requested_for_full_atom_use() -> None:
    expected_molecule = Chem.MolFromSmiles("[H]/N=C\\1/CCCN1")
    assert expected_molecule is not None
    expected = explicit_stereo_constraints(
        expected_molecule,
        heavy_atom_observable_only=False,
    )

    assert len(expected.double_bond) == 1


def test_unspecified_input_stereo_does_not_create_a_constraint() -> None:
    expected_molecule = Chem.MolFromSmiles("FC(Cl)(Br)I")
    assert expected_molecule is not None
    expected = explicit_stereo_constraints(expected_molecule)
    candidate = _embedded("F[C@](Cl)(Br)I")

    assert not expected.tetrahedral
    assert stereo_compatibility(expected, candidate).valid


def test_unassigned_center_is_ignored_when_an_explicit_center_is_present() -> None:
    expected_molecule = Chem.MolFromSmiles("N[C@@H](C)C(O)(F)Cl")
    assert expected_molecule is not None
    expected = explicit_stereo_constraints(expected_molecule)
    assert len(expected.tetrahedral) == 1
    candidate_r = _embedded("N[C@@H](C)[C@](O)(F)Cl")
    candidate_s = _embedded("N[C@@H](C)[C@@](O)(F)Cl")

    assert stereo_compatibility(expected, candidate_r).valid
    assert stereo_compatibility(expected, candidate_s).valid


def test_selector_filters_before_predicted_rmsd_ranking() -> None:
    selected, fallback = select_stereo_filtered_index(
        [0.2, 0.4, 0.3],
        [False, True, True],
        fallback_index=0,
    )
    assert selected == 2
    assert fallback is False


def test_selector_fallback_is_explicit_when_no_candidate_passes() -> None:
    selected, fallback = select_stereo_filtered_index(
        [0.2, 0.4],
        [False, False],
        fallback_index=0,
    )
    assert selected == 0
    assert fallback is True
