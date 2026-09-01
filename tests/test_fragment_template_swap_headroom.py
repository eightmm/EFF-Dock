"""Focused tests for the saved-pose fragment-template headroom probe."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pytest
import torch
from rdkit import Chem

from effdock.inference.io import write_multi_sdf
from effdock.workflows.benchmark_inputs import (
    BENCHMARK_INPUT_MANIFEST_SCHEMA,
    BENCHMARK_INPUT_PROTOCOL_ID,
    ligand_input_identity,
    load_benchmark_ligand,
    mapping_sha256,
    sorted_id_sha256,
)
from scripts.analyze_fragment_template_swap_headroom import (
    _load_all_poses,
    _resolve_sampling_seed,
    _validate_saved_ensemble,
    benchmark_symmetry_aware_rmsd,
    enumerate_symmetry_mappings,
    replace_fragment_internal_geometry,
    run,
)


def _rotation_z(angle: float) -> torch.Tensor:
    return torch.tensor(
        [
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float64,
    )


def test_template_swap_preserves_independent_fragment_frames() -> None:
    tetrahedron = torch.tensor(
        [
            [1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
        ],
        dtype=torch.float64,
    )
    crystal = torch.cat((tetrahedron, tetrahedron + torch.tensor([8.0, 0.0, 0.0])))
    rotation_a = _rotation_z(0.7)
    rotation_b = _rotation_z(-1.1)
    expected_a = tetrahedron @ rotation_a.T + torch.tensor([12.0, -3.0, 2.0])
    expected_b = tetrahedron @ rotation_b.T + torch.tensor([-5.0, 7.0, -4.0])
    pose = torch.cat(
        (
            1.4 * tetrahedron @ rotation_a.T + torch.tensor([12.0, -3.0, 2.0]),
            0.7 * tetrahedron @ rotation_b.T + torch.tensor([-5.0, 7.0, -4.0]),
        )
    )

    swapped = replace_fragment_internal_geometry(
        pose,
        crystal,
        torch.tensor([0, 0, 0, 0, 1, 1, 1, 1]),
        list(range(8)),
    )

    torch.testing.assert_close(swapped[:4], expected_a, rtol=0.0, atol=1e-12)
    torch.testing.assert_close(swapped[4:], expected_b, rtol=0.0, atol=1e-12)
    torch.testing.assert_close(swapped[:4].mean(0), pose[:4].mean(0))
    torch.testing.assert_close(swapped[4:].mean(0), pose[4:].mean(0))
    assert not torch.allclose(swapped, crystal)


def test_symmetry_aware_rmsd_accepts_equivalent_atom_permutation() -> None:
    crystal = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    pose = crystal.flip(0)
    mol_ref = Chem.MolFromSmiles("CC")
    assert mol_ref is not None
    conformer = Chem.Conformer(2)
    for index, xyz in enumerate(crystal.tolist()):
        conformer.SetAtomPosition(index, xyz)
    mol_ref.AddConformer(conformer)
    mol_input = Chem.Mol(mol_ref)

    rmsd = benchmark_symmetry_aware_rmsd(
        pose,
        crystal,
        (0, 1),
        mol_input,
        mol_ref,
    )

    assert rmsd == pytest.approx(0.0)


def test_sampling_seed_must_be_explicit_and_consistent(tmp_path: Path) -> None:
    mol = Chem.MolFromSmiles("CC")
    assert mol is not None
    poses = [Chem.Mol(mol), Chem.Mol(mol)]
    pose_path = tmp_path / "case.sdf"

    with pytest.raises(ValueError, match="sampling seed is absent"):
        _resolve_sampling_seed("case", pose_path, poses, None)

    for pose in poses:
        pose.SetProp("sampling_seed", "7")
    with pytest.raises(ValueError, match="SDF and summary sampling seeds disagree"):
        _resolve_sampling_seed(
            "case",
            pose_path,
            poses,
            {
                "sampling_seed": "8",
                "all_poses_sdf": "",
                "all_poses_sdf_sha256": "",
            },
        )


def test_saved_atom_order_mismatch_fails_closed() -> None:
    template, _ = load_benchmark_ligand("CCO", random_seed=7)
    reordered = Chem.RenumberAtoms(template, [2, 1, 0])
    for key, value in {
        "_Name": "docked_pose_0",
        "dataset": "astex",
        "complex_id": "case",
        "num_samples": "1",
        "sample_sigma": "2.0",
        "sample_index": "0",
        "num_steps": "10",
        "candidate_ensemble_sha256": "0" * 64,
    }.items():
        reordered.SetProp(key, value)

    with pytest.raises(ValueError, match="atom order/graph"):
        _validate_saved_ensemble(
            dataset="astex",
            complex_id="case",
            poses=[reordered],
            template=template,
            expected_poses=1,
            expected_sigma=2.0,
        )


def test_all_pose_loader_uses_forward_supplier(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mol = Chem.MolFromSmiles("CC")
    assert mol is not None
    mol.SetProp("sample_index", "0")
    path = tmp_path / "poses.sdf"
    writer = Chem.SDWriter(str(path))
    writer.write(mol)
    writer.close()

    def reject_indexed_supplier(*args: object, **kwargs: object) -> None:
        raise AssertionError("indexed SDMolSupplier must not be used")

    monkeypatch.setattr(Chem, "SDMolSupplier", reject_indexed_supplier)

    poses = _load_all_poses(path)

    assert len(poses) == 1
    assert poses[0].GetProp("sample_index") == "0"


def test_connectivity_sensitivity_is_explicit_for_enantiomers() -> None:
    mol_ref = Chem.MolFromSmiles("F[C@](Cl)(Br)I")
    mol_input = Chem.MolFromSmiles("F[C@@](Cl)(Br)I")
    assert mol_ref is not None and mol_input is not None

    with pytest.raises(ValueError, match="no stereo-preserving full atom mapping"):
        enumerate_symmetry_mappings(
            mol_ref,
            mol_input,
            max_matches=64,
            stereo_policy="require",
        )

    mappings, metadata = enumerate_symmetry_mappings(
        mol_ref,
        mol_input,
        max_matches=64,
        stereo_policy="connectivity_sensitivity",
    )

    assert mappings == [(0, 1, 2, 3, 4)]
    assert metadata["stereo_preserving"] is False
    assert metadata["sensitivity_only"] is True
    assert metadata["symmetry_complete"] is True


def _write_frozen_manifest(path: Path, smiles: str) -> None:
    complex_id = "case"
    mapping = {complex_id: smiles}
    payload = {
        "schema_version": BENCHMARK_INPUT_MANIFEST_SCHEMA,
        "protocol_id": BENCHMARK_INPUT_PROTOCOL_ID,
        "datasets": {
            "astex": {
                "count": 1,
                "ids_sha256": sorted_id_sha256([complex_id]),
                "mapping_sha256": mapping_sha256("astex", mapping),
                "ligands": {
                    complex_id: {
                        "smiles": smiles,
                        "input_identity": ligand_input_identity(complex_id, smiles),
                    }
                },
                "source_manifests": {},
                "integrity_boundary": {},
            }
        },
    }
    path.write_text(json.dumps(payload))


def test_end_to_end_two_pose_fixture(tmp_path: Path) -> None:
    smiles = "c1ccccc1O"
    sampling_seed = 43
    template, _ = load_benchmark_ligand(smiles, random_seed=sampling_seed)
    template_coords = torch.as_tensor(
        template.GetConformer().GetPositions(),
        dtype=torch.float32,
    )

    dataset_root = tmp_path / "astex"
    reference_dir = dataset_root / "case"
    reference_dir.mkdir(parents=True)
    reference_path = reference_dir / "CASE_ligand.sdf"
    writer = Chem.SDWriter(str(reference_path))
    writer.write(template)
    writer.close()

    all_poses_dir = tmp_path / "all_poses"
    all_poses_dir.mkdir()
    all_poses_path = all_poses_dir / "case.sdf"
    write_multi_sdf(
        template,
        [template_coords, template_coords + torch.tensor([3.0, 0.0, 0.0])],
        torch.zeros(3),
        all_poses_path,
        props={
            "dataset": "astex",
            "complex_id": "case",
            "sampling_seed": sampling_seed,
            "num_samples": 2,
            "num_steps": 10,
            "candidate_ensemble_sha256": "0" * 64,
        },
        per_pose_props=[{"sample_sigma": 2.0}, {"sample_sigma": 2.0}],
    )

    manifest_path = tmp_path / "inputs.json"
    _write_frozen_manifest(manifest_path, smiles)
    output_path = tmp_path / "result.json"
    args = argparse.Namespace(
        dataset="astex",
        dataset_root=dataset_root,
        all_poses_dir=all_poses_dir,
        benchmark_input_manifest=manifest_path,
        output=output_path,
        summary_csv=[],
        expected_poses=2,
        expected_sigma=2.0,
        max_matches=64,
        only_id=[],
    )

    result = run(args)

    assert output_path.is_file()
    assert result["aggregate"]["successful_complexes"] == 1
    assert result["aggregate"]["failed_complexes"] == 0
    record = result["records"][0]
    assert record["sampling_seed"] == sampling_seed
    assert record["sampling_seed_source"] == "all_poses_sdf.sampling_seed"
    assert record["before"]["k2"] == 1
    assert record["after"]["k2"] == 1
    assert record["mapping"]["symmetry_complete"] is True
    assert result["aggregate"]["before"]["macro_k2_per_100"] == pytest.approx(50.0)
    assert result["aggregate"]["mapping_subgroups"]["stereo_preserving"] == {
        "complex_count": 1,
        "complex_ids": ["case"],
        "before_k2_total": 1,
        "after_k2_total": 1,
        "delta_k2_total": 0,
        "mean_delta_k2": 0.0,
    }
    assert result["aggregate"]["mapping_subgroups"][
        "connectivity_sensitivity_only"
    ]["complex_count"] == 0
