from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
from rdkit import Chem

from effdock.cli import main
from effdock.confidence import runtime, selectors
from effdock.inference import MinimizationOptions, docking, minimize, saved_poses
from effdock.inference.preprocess import load_ligand


@pytest.fixture
def pose_inputs(tmp_path):
    center = torch.tensor([11.0, -3.0, 9.0])
    protein = tmp_path / "receptor.pdb"
    atoms = [
        ("N", "N", (-4.0, 0.0, 0.0)),
        ("CA", "C", (-2.55, 0.0, 0.0)),
        ("C", "C", (-1.5, 1.0, 0.0)),
        ("O", "O", (-1.8, 2.2, 0.0)),
    ]
    lines = []
    for index, (name, element, offset) in enumerate(atoms, 1):
        x, y, z = (center + torch.tensor(offset)).tolist()
        lines.append(
            f"ATOM  {index:5d} {name:^4s} GLY A   1    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00          {element:>2s}"
        )
    protein.write_text("\n".join([*lines, "TER", "END", ""]))
    ligand, _ = load_ligand("CCO")
    poses = tmp_path / "poses.sdf"
    with Chem.SDWriter(str(poses)) as writer:
        xyz = torch.tensor(ligand.GetConformer().GetPositions(), dtype=torch.float32)
        for index in range(3):
            mol = Chem.Mol(ligand)
            mol.SetIntProp("sample_index", 14 + index)
            mol.SetDoubleProp("confidence_rmsd", 9.0)
            for atom, pos in enumerate(xyz + center + torch.tensor([0.2 * index, 0, 0])):
                mol.GetConformer().SetAtomPosition(atom, pos.tolist())
            writer.write(mol)
    return protein, poses, center


def test_confidence_cli_scores_in_saved_order_across_batches(pose_inputs, tmp_path, monkeypatch):
    protein, poses, center = pose_inputs
    original = poses.read_bytes()
    output = tmp_path / "confidence.json"
    docking_model = SimpleNamespace(orientation_injection="rw")
    monkeypatch.setattr(saved_poses, "load_model", lambda *args, **kw: (docking_model, {}, {}))
    monkeypatch.setattr(runtime, "load_pose_confidence_model", lambda *args: (object(), {}))
    actual = []
    values = iter((2.0, 1.0, 1.0))

    def score(model, docking, graph, ligand, meta, batch, **options):
        assert docking is docking_model
        assert options["frame_policy"] == "contextual_v1"
        assert options["sigma"] == 2.0
        torch.testing.assert_close(meta["pocket_center"], center)
        actual.append(torch.stack(batch))
        return [{"confidence_rmsd": next(values)} for _ in batch]

    monkeypatch.setattr(runtime, "score_poses_with_confidence", score)
    main([
        "confidence", "score", "--protein", str(protein), "--poses", str(poses),
        "--pocket-center", "11,-3,9", "--output", str(output), "--batch-size", "2",
        "--confidence-frame-policy", "contextual_v1", "--ligand-template", "CCO",
        "--device", "cpu",
    ])
    records = json.loads(output.read_text())["poses"]
    assert [r["input_index"] for r in records] == [0, 1, 2]
    assert [r["sample_index"] for r in records] == ["14", "15", "16"]
    assert [r["rank"] for r in records] == [3, 1, 2]
    assert [len(batch) for batch in actual] == [2, 1]
    source = list(Chem.SDMolSupplier(str(poses)))
    coordinates = torch.stack([
        torch.tensor(mol.GetConformer().GetPositions(), dtype=torch.float32) for mol in source
    ])
    torch.testing.assert_close(torch.cat(actual), coordinates - center)
    assert poses.read_bytes() == original


def test_minimize_cli_preserves_order_and_handles_a_singleton_tail(pose_inputs, tmp_path):
    protein, poses, _ = pose_inputs
    original = poses.read_bytes()
    output = tmp_path / "minimized.sdf"
    main([
        "minimize", "--protein", str(protein), "--poses", str(poses), "--pocket-center", "11,-3,9",
        "--output", str(output), "--steps", "3", "--batch-size", "2", "--device", "cpu",
    ])
    source = list(Chem.SDMolSupplier(str(poses)))
    refined = list(Chem.SDMolSupplier(str(output)))
    assert len(refined) == 3
    for index, (raw, mol) in enumerate(zip(source, refined, strict=True)):
        assert mol is not None and Chem.MolToSmiles(mol) == Chem.MolToSmiles(raw)
        assert mol.GetIntProp("minimization_input_index") == index
        assert mol.GetIntProp("sample_index") == 14 + index
        assert mol.GetIntProp("minimization_steps") <= 3
        assert mol.GetDoubleProp("minimization_final_energy") <= mol.GetDoubleProp(
            "minimization_initial_energy"
        )
        assert mol.GetBoolProp("minimization_shell_envelope_valid")
        assert not mol.HasProp("confidence_rmsd")
        before = torch.tensor(raw.GetConformer().GetPositions())
        after = torch.tensor(mol.GetConformer().GetPositions())
        assert not torch.equal(before, after)
        torch.testing.assert_close(torch.pdist(before), torch.pdist(after), atol=2e-4, rtol=0)
    assert poses.read_bytes() == original


def test_incompatible_template_cannot_replace_existing_output(pose_inputs, tmp_path):
    protein, poses, center = pose_inputs
    output = tmp_path / "minimized.sdf"
    output.write_text("existing output")
    with pytest.raises(ValueError, match="ordered ligand graph"):
        minimize(MinimizationOptions(
            protein, poses, center, output, ligand_template="CCN", device="cpu",
        ))
    assert output.read_text() == "existing output"


def test_minimize_rejects_input_as_output(pose_inputs):
    protein, poses, center = pose_inputs
    with pytest.raises(ValueError, match="separate from the input"):
        minimize(MinimizationOptions(protein, poses, center, poses, device="cpu"))


@pytest.mark.parametrize("count", [1, 2])
def test_dock_writes_matched_raw_and_minimized_confidence(pose_inputs, tmp_path, monkeypatch, count):
    protein, _, center = pose_inputs
    model = SimpleNamespace(
        orientation_injection="legacy_rt_w", docking_orientation_injection="legacy_rt_w",
        confidence_frame_policy="historical_kabsch",
    )
    monkeypatch.setattr(docking, "load_model", lambda *args, **kw: (model, {"data": {}}, {}))
    monkeypatch.setattr(runtime, "load_pose_confidence_model", lambda *args: (model, {}))
    ligand, _ = load_ligand("CCO")
    coordinates = torch.tensor(ligand.GetConformer().GetPositions(), dtype=torch.float32)
    first = coordinates - coordinates[0]
    candidates = [
        {"atom_pos_pred": first + torch.tensor([float(index), 0.0, 0.0]), "sigma": index + 1.0}
        for index in range(count)
    ]
    monkeypatch.setattr(docking, "sample_docked_poses", lambda *args, **kw: candidates)
    calls = []

    def score(confidence, base, graph, lig_data, meta, poses, *, sigma, **kwargs):
        calls.append(sigma.tolist())
        if len(calls) == 1:
            estimates = [3.0 - index for index in range(len(poses))]
        else:
            assert all(torch.allclose(pose[0, 1], torch.tensor(0.5)) for pose in poses)
            estimates = [1.0 + (count - index) for index in range(len(poses))]
        return [dict(confidence_rmsd=v, confidence_success=0.5, confidence_atom_ok=0.4)
                for v in estimates]

    def optimize(mol, receptor, poses, pocket_center, ids, device):
        assert poses[0, 0, 0].item() == count - 1
        return ([pose + torch.tensor([0.0, 0.5, 0.0]) for pose in poses],
                [{"minimization_status": "max_steps"} for _ in poses])

    monkeypatch.setattr(runtime, "score_poses_with_confidence", score)
    monkeypatch.setattr(selectors, "select_confidence_poses",
                        lambda *args: {"confidence": count - 1})
    monkeypatch.setattr(docking, "minimize_poses", optimize)
    output = tmp_path / "docked"
    docking.dock(docking.DockingOptions(
        protein=protein, ligand="CCO", pocket_center=center, out_dir=output,
        num_samples=count, score=False, device="cpu",
    ))
    stem = "docked" if count == 1 else "docked_poses"
    raw = list(Chem.SDMolSupplier(str(output / f"{stem}.sdf")))
    minimized = list(Chem.SDMolSupplier(str(output / f"{stem}_minimized.sdf")))
    assert len(raw) == len(minimized) == count
    assert calls == [list(range(1, count + 1)), list(range(count, 0, -1))]
    for index, (original, refined) in enumerate(zip(raw, minimized, strict=True)):
        assert original.GetIntProp("sample_index") == refined.GetIntProp("sample_index") == index
        assert original.GetIntProp("confidence_rank") == index + 1
        assert refined.GetIntProp("confidence_rank") == count - index
        assert refined.GetDoubleProp("confidence_pred_rmsd") == 1 + count - index
        assert original.GetDoubleProp("confidence_pred_rmsd") == 4 - count + index
        assert refined.GetDoubleProp("sample_sigma") == original.GetDoubleProp("sample_sigma") == count - index
        assert refined.GetProp("minimization_status") == "max_steps"
    tensors = torch.load(output / "results.pt", weights_only=True)
    assert len(tensors["poses"]) == count
    assert tensors["poses"][0]["score"]["confidence_rmsd"] == raw[0].GetDoubleProp("confidence_rmsd")
