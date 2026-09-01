from __future__ import annotations

import torch
from rdkit import Chem

from effdock.inference.io import write_multi_sdf


def _single_atom_mol() -> Chem.Mol:
    mol = Chem.MolFromSmiles("C")
    assert mol is not None
    conformer = Chem.Conformer(1)
    conformer.SetAtomPosition(0, (0.0, 0.0, 0.0))
    mol.AddConformer(conformer)
    return mol


def test_write_multi_sdf_preserves_pose_order_and_confidence_molprops(tmp_path) -> None:
    path = tmp_path / "all_poses.sdf"
    write_multi_sdf(
        _single_atom_mol(),
        [torch.tensor([[1.0, 0.0, 0.0]]), torch.tensor([[2.0, 0.0, 0.0]])],
        torch.tensor([10.0, 0.0, 0.0]),
        path,
        scores=[
            {"confidence_rmsd": 0.75, "confidence_success": 0.8},
            {"confidence_rmsd": 1.25, "confidence_success": 0.4},
        ],
        props={"complex_id": "case"},
    )

    poses = [mol for mol in Chem.SDMolSupplier(str(path), removeHs=False)]
    assert len(poses) == 2
    assert [mol.GetIntProp("sample_index") for mol in poses] == [0, 1]
    assert [mol.GetDoubleProp("confidence_rmsd") for mol in poses] == [0.75, 1.25]
    assert [mol.GetDoubleProp("confidence_pred_rmsd") for mol in poses] == [0.75, 1.25]
    assert [mol.GetDoubleProp("confidence_success") for mol in poses] == [0.8, 0.4]
    assert [mol.GetDoubleProp("confidence_pred_success") for mol in poses] == [0.8, 0.4]
    assert [mol.GetProp("complex_id") for mol in poses] == ["case", "case"]
    assert [mol.GetConformer().GetAtomPosition(0).x for mol in poses] == [11.0, 12.0]


def test_write_multi_sdf_rejects_unpaired_scores(tmp_path) -> None:
    path = tmp_path / "all_poses.sdf"
    try:
        write_multi_sdf(
            _single_atom_mol(),
            [torch.zeros((1, 3)), torch.ones((1, 3))],
            torch.zeros(3),
            path,
            scores=[{"confidence_rmsd": 1.0}],
        )
    except ValueError as exc:
        assert str(exc) == "scores must have one entry per pose"
    else:
        raise AssertionError("mismatched score count must fail")
