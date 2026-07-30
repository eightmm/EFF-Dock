import importlib.util
from pathlib import Path

import torch
from rdkit import Chem

_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "analyze_charge_guidance_benchmarks.py"
)
_SPEC = importlib.util.spec_from_file_location("analyze_charge_guidance_benchmarks", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
_formal_charge_summary = _MODULE._formal_charge_summary
_pose_coords_in_reference_order = _MODULE._pose_coords_in_reference_order


def test_formal_charge_eligibility_includes_net_neutral_zwitterion():
    mol = Chem.MolFromSmiles("[NH3+]CC(=O)[O-]")

    summary = _formal_charge_summary(mol)

    assert summary["net_formal_charge_e"] == 0
    assert summary["nonzero_formal_charge_atoms"] == 2
    assert summary["formal_charge_sites"] == 2
    assert summary["has_nonzero_formal_charge_site"] is True


def test_pose_coordinates_are_mapped_to_reference_atom_order():
    reference = Chem.MolFromSmiles("CCO")
    conformer = Chem.Conformer(reference.GetNumAtoms())
    conformer.SetAtomPosition(0, (0.0, 0.0, 0.0))
    conformer.SetAtomPosition(1, (1.0, 0.0, 0.0))
    conformer.SetAtomPosition(2, (2.0, 1.0, 0.0))
    reference.AddConformer(conformer)
    pose = Chem.RenumberAtoms(reference, [2, 0, 1])

    mapped = _pose_coords_in_reference_order(reference, pose)

    expected = torch.tensor(reference.GetConformer().GetPositions(), dtype=torch.float64)
    torch.testing.assert_close(mapped, expected)
