from rdkit import Chem
from rdkit.Chem import AllChem

from scripts.external_models.evaluate_native_outputs import no_align_rmsd


def _molecule(smiles: str, seed: int) -> Chem.Mol:
    molecule = Chem.AddHs(Chem.MolFromSmiles(smiles))
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    assert AllChem.EmbedMolecule(molecule, params) == 0
    return Chem.RemoveHs(molecule)


def test_no_align_rmsd_uses_full_topology_and_symmetry() -> None:
    reference = _molecule("CCO", 1)
    pose = Chem.Mol(reference)

    value, method = no_align_rmsd(pose, reference)

    assert value == 0.0
    assert method == "rdkit_calc_rms"


def test_no_align_rmsd_rejects_partial_atom_mapping() -> None:
    reference = _molecule("CCO", 1)
    pose = _molecule("CC", 2)

    try:
        no_align_rmsd(pose, reference)
    except ValueError as exc:
        assert "heavy_atom_count_mismatch" in str(exc)
    else:
        raise AssertionError("partial mapping must fail closed")
