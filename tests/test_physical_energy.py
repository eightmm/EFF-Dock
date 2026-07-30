from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path

import pytest
import torch

Chem = pytest.importorskip("rdkit.Chem")
from rdkit.Chem import AllChem  # noqa: E402

from effdock.guidance.parameterization import element_parameters  # noqa: E402
from effdock.guidance.physical import PhysicalEnergyConfig, physical_energy  # noqa: E402
from effdock.guidance.system import PhysicalSystem  # noqa: E402
from effdock.guidance.topology import build_physical_topology  # noqa: E402
from effdock.preprocess.fragments import decompose_fragments  # noqa: E402


def _embedded_butane() -> "Chem.Mol":
    mol = Chem.AddHs(Chem.MolFromSmiles("CCCC"))
    assert AllChem.EmbedMolecule(mol, randomSeed=0xEFF) == 0
    return Chem.RemoveHs(mol)


@pytest.fixture
def butane_system() -> tuple["Chem.Mol", torch.Tensor, PhysicalSystem]:
    mol = _embedded_butane()
    fragment_id = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    topology = build_physical_topology(mol, fragment_id).to(
        torch.device("cpu"),
        torch.float64,
    )
    coords = torch.as_tensor(
        mol.GetConformer().GetPositions(),
        dtype=torch.float64,
    )

    protein_atomic_numbers = torch.tensor([6, 8], dtype=torch.long)
    protein_parameters = element_parameters(
        protein_atomic_numbers,
        dtype=torch.float64,
    )
    protein_coords = torch.stack(
        (
            coords.mean(dim=0) + torch.tensor([4.0, 0.2, -0.1], dtype=torch.float64),
            coords.mean(dim=0) + torch.tensor([-0.5, 4.5, 0.8], dtype=torch.float64),
        )
    )
    system = PhysicalSystem(
        topology=topology,
        protein_coords=protein_coords,
        protein_atomic_numbers=protein_atomic_numbers,
        protein_uff_x=protein_parameters.uff_x,
        protein_uff_d=protein_parameters.uff_d,
        parameter_set={"name": "test", "version": "test"},
        protein_source_atoms=2,
    )
    return mol, coords, system


def _proper_rotation() -> torch.Tensor:
    rotation = torch.tensor(
        [
            [0.36, -0.48, 0.80],
            [0.80, 0.60, 0.00],
            [-0.48, 0.64, 0.60],
        ],
        dtype=torch.float64,
    )
    torch.testing.assert_close(
        rotation @ rotation.T,
        torch.eye(3, dtype=torch.float64),
    )
    torch.testing.assert_close(
        torch.det(rotation),
        torch.tensor(1.0, dtype=torch.float64),
    )
    return rotation


def _fragment_perturbation(coords: torch.Tensor) -> torch.Tensor:
    perturbed = coords.clone()
    perturbed[2:] += torch.tensor([0.31, -0.12, 0.08], dtype=coords.dtype)
    return perturbed


def test_butane_fragment_topology_counts_and_one_four_exclusion(
    butane_system: tuple["Chem.Mol", torch.Tensor, PhysicalSystem],
) -> None:
    _, _, system = butane_system
    topology = system.topology

    assert topology.term_counts() == {
        "atoms": 4,
        "cut_bonds": 1,
        "cross_fragment_angles": 2,
        "torsion_cut_bonds": 1,
        "torsion_quads": 1,
        "cross_fragment_impropers": 0,
        "interfragment_nonbonded_pairs": 1,
    }
    assert topology.bond_index.T.tolist() == [[1, 2]]
    assert topology.ligand_pair_index.T.tolist() == [[0, 3]]
    torch.testing.assert_close(
        topology.ligand_pair_scale,
        torch.tensor([0.5], dtype=torch.float64),
    )
    torch.testing.assert_close(
        topology.proper_weight,
        torch.tensor([1.0], dtype=torch.float64),
    )


def test_input_reference_bond_and_angle_start_at_zero(
    butane_system: tuple["Chem.Mol", torch.Tensor, PhysicalSystem],
) -> None:
    _, coords, system = butane_system
    energy = physical_energy(coords, system)

    torch.testing.assert_close(
        energy["ligand_intra_bond"],
        torch.tensor(0.0, dtype=torch.float64),
        atol=1e-12,
        rtol=0.0,
    )
    torch.testing.assert_close(
        energy["ligand_intra_angle"],
        torch.tensor(0.0, dtype=torch.float64),
        atol=1e-12,
        rtol=0.0,
    )
    perturbed = physical_energy(_fragment_perturbation(coords), system)
    assert float(perturbed["ligand_intra_bond"]) > 0.0
    assert float(perturbed["ligand_intra_angle"]) > 0.0


def test_topology_reference_hash_tracks_input_geometry(
    butane_system: tuple["Chem.Mol", torch.Tensor, PhysicalSystem],
) -> None:
    mol, _, system = butane_system
    rebuilt = build_physical_topology(
        mol,
        system.topology.fragment_id,
    )
    assert rebuilt.reference_sha256() == system.topology.reference_sha256()

    changed = Chem.Mol(mol)
    conformer = changed.GetConformer()
    position = conformer.GetAtomPosition(2)
    position.x += 0.1
    conformer.SetAtomPosition(2, position)
    changed_topology = build_physical_topology(
        changed,
        system.topology.fragment_id,
    )
    assert changed_topology.reference_sha256() != system.topology.reference_sha256()


def test_proper_weights_sum_to_one_per_cut_bond() -> None:
    mol = Chem.AddHs(Chem.MolFromSmiles("CC(C)C(C)C"))
    assert AllChem.EmbedMolecule(mol, randomSeed=0xE11) == 0
    mol = Chem.RemoveHs(mol)
    coords = torch.as_tensor(mol.GetConformer().GetPositions(), dtype=torch.float64)
    fragments = decompose_fragments(mol, coords)
    assert fragments is not None
    topology = build_physical_topology(mol, fragments["fragment_id"])
    counts = torch.bincount(topology.proper_cut_bond_id)
    assert int(counts.max()) > 1
    for cut_bond_id in torch.unique(topology.proper_cut_bond_id).tolist():
        weight = topology.proper_weight[topology.proper_cut_bond_id == cut_bond_id]
        torch.testing.assert_close(
            weight.sum(),
            torch.tensor(1.0, dtype=torch.float64),
        )


@pytest.mark.parametrize("smiles", ["CC(=O)NC", "C1CCCCC1", "CC=CC"])
def test_noncuttable_ligand_geometry_stays_inside_rigid_fragment(smiles: str) -> None:
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert AllChem.EmbedMolecule(mol, randomSeed=0xE12) == 0
    mol = Chem.RemoveHs(mol)
    coords = torch.as_tensor(mol.GetConformer().GetPositions(), dtype=torch.float64)
    fragments = decompose_fragments(mol, coords)
    assert fragments is not None
    topology = build_physical_topology(mol, fragments["fragment_id"])

    assert topology.bond_index.shape[1] == 0
    assert topology.angle_index.shape[1] == 0
    assert topology.proper_index.shape[1] == 0


def test_components_sum_to_total_and_batch_matches_pose_loop(
    butane_system: tuple["Chem.Mol", torch.Tensor, PhysicalSystem],
) -> None:
    _, coords, system = butane_system
    batch = torch.stack((coords, _fragment_perturbation(coords)))
    config = PhysicalEnergyConfig(
        softcore=0.75,
        switch_distance=6.0,
        cutoff=8.0,
        protein_chunk_size=1,
    )

    batched = physical_energy(batch, system, config)
    component_total = sum(
        value for name, value in batched.items() if name != "total"
    )
    torch.testing.assert_close(batched["total"], component_total)

    for pose_index, pose in enumerate(batch):
        single = physical_energy(pose, system, config)
        assert set(single) == set(batched)
        for name in batched:
            torch.testing.assert_close(
                batched[name][pose_index],
                single[name],
                atol=1e-10,
                rtol=1e-10,
            )


def test_force_is_finite_and_matches_negative_directional_derivative(
    butane_system: tuple["Chem.Mol", torch.Tensor, PhysicalSystem],
) -> None:
    _, coords, system = butane_system
    coords = _fragment_perturbation(coords).requires_grad_(True)
    energy = physical_energy(coords, system)["total"]
    gradient = torch.autograd.grad(energy, coords)[0]
    force = -gradient

    assert torch.isfinite(energy)
    assert torch.isfinite(force).all()
    assert float(force.norm()) > 0.0

    direction = force.detach() / force.detach().norm()
    step = 1e-5
    with torch.no_grad():
        energy_plus = physical_energy(coords + step * direction, system)["total"]
        energy_minus = physical_energy(coords - step * direction, system)["total"]
    directional_derivative = (energy_plus - energy_minus) / (2.0 * step)

    assert float(energy_plus) < float(energy.detach())
    torch.testing.assert_close(
        directional_derivative,
        -force.norm(),
        atol=2e-5,
        rtol=2e-5,
    )


def test_energy_is_rigid_transform_invariant_and_force_is_equivariant(
    butane_system: tuple["Chem.Mol", torch.Tensor, PhysicalSystem],
) -> None:
    _, coords, system = butane_system
    coords = _fragment_perturbation(coords).requires_grad_(True)
    energy = physical_energy(coords, system)["total"]
    force = -torch.autograd.grad(energy, coords)[0]

    rotation = _proper_rotation()
    translation = torch.tensor([2.3, -1.7, 0.6], dtype=torch.float64)
    transformed_coords = (coords.detach() @ rotation.T + translation).requires_grad_(True)
    transformed_system = replace(
        system,
        protein_coords=system.protein_coords @ rotation.T + translation,
    )
    transformed_energy = physical_energy(transformed_coords, transformed_system)["total"]
    transformed_force = -torch.autograd.grad(transformed_energy, transformed_coords)[0]

    torch.testing.assert_close(transformed_energy, energy, atol=1e-9, rtol=1e-9)
    torch.testing.assert_close(
        transformed_force,
        force @ rotation.T,
        atol=1e-8,
        rtol=1e-8,
    )


def test_unsupported_element_fails_explicitly() -> None:
    with pytest.raises(
        ValueError,
        match=r"EFF-FF-v2 unsupported atomic numbers: \[2\]",
    ):
        element_parameters(torch.tensor([2], dtype=torch.long))


def test_guidance_runtime_does_not_import_external_engines() -> None:
    guidance_dir = Path(__file__).parents[1] / "src" / "effdock" / "guidance"
    forbidden_roots = {
        "MDAnalysis",
        "amber",
        "mdtraj",
        "meeko",
        "openff",
        "openforcefield",
        "openmm",
        "parmed",
        "simtk",
        "smina",
        "vina",
    }
    imported_roots: set[str] = set()
    source_paths = [
        *guidance_dir.rglob("*.py"),
        guidance_dir.parent / "workflows" / "trace_physical.py",
    ]
    for source_path in source_paths:
        tree = ast.parse(source_path.read_text(), filename=str(source_path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imported_roots.add(node.module.split(".", 1)[0])

    assert imported_roots.isdisjoint(forbidden_roots), (
        f"external engine imports found: {sorted(imported_roots & forbidden_roots)}"
    )
