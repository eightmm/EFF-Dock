from __future__ import annotations

import json

import pytest
from rdkit import Chem

from benchmarks.external_models import recover_surfdock_minimization as recovery


def saved_sdf(path, at):
    mol = Chem.MolFromSmiles("CC")
    conformer = Chem.Conformer(2)
    conformer.Set3D(True)
    conformer.SetAtomPosition(0, (at, 0.0, 1.0))
    conformer.SetAtomPosition(1, (at + 1.5, 0.0, 1.0))
    mol.AddConformer(conformer)
    with Chem.SDWriter(str(path)) as writer:
        writer.write(mol)


def test_recovery_reads_exact_overflowed_coordinates_and_rejects_the_invalid_frame(tmp_path):
    source = tmp_path / "overflow.sdf"
    saved_sdf(source, 100001.0)
    original = source.read_bytes()
    mol = recovery.read_saved_pose(source)
    x = mol.GetConformer().GetPositions()[:, 0].tolist()
    assert x == [100001.0, 100002.5]
    with pytest.raises(ValueError, match="outside the receptor PDB frame"):
        recovery.validate_pose(mol)
    assert source.read_bytes() == original


def test_recovery_retries_parameter_failure_and_records_success(tmp_path, monkeypatch):
    inputs = tmp_path / "source.sdf"
    saved_sdf(inputs, 8.0)
    original = inputs.read_bytes()
    called = []

    def mock_minimizer(mol, receptor, forcefield):
        called.append(forcefield)
        if forcefield == "openff-2.0.0":
            raise ValueError("unassigned angle")
        return mol, 0.0, -2.0

    monkeypatch.setattr(recovery, "minimize_pose", mock_minimizer)
    row = {"pose_id": "rank001", "pose": str(inputs), "receptor": "receptor.pdb"}
    status = recovery.recover(row, tmp_path, recovery.FORCE_FIELDS)
    assert status == "minimized"
    assert called == ["openff-2.0.0", "gaff-2.11"]
    receipt = json.loads((tmp_path / "rank001.json").read_text())
    assert receipt["forcefield"] == "gaff-2.11"
    assert receipt["final_kj"] < receipt["initial_kj"]
    assert "unassigned angle" in receipt["failed_attempts"]["openff-2.0.0"]
    assert recovery.ordered_graph(recovery.read_saved_pose(tmp_path / "rank001.sdf")) == (
        recovery.ordered_graph(recovery.read_saved_pose(inputs))
    )
    assert inputs.read_bytes() == original
