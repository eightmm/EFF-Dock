from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

Chem = pytest.importorskip("rdkit.Chem")
from rdkit.Chem import AllChem  # noqa: E402

from effdock.guidance.chemical import (  # noqa: E402
    ChemicalConstraintEnergyConfig,
    chemical_constraint_diagnostics,
    chemical_constraint_energy,
)
from effdock.guidance.runtime import (  # noqa: E402
    UnifiedGuidance,
    UnifiedGuidanceConfig,
    project_atom_forces,
)
from effdock.guidance.topology import build_physical_topology  # noqa: E402
from effdock.preprocess.fragments import decompose_fragments  # noqa: E402


def _embedded(smiles: str) -> "Chem.Mol":
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert AllChem.EmbedMolecule(mol, randomSeed=0xEFF) == 0
    return Chem.RemoveHs(mol)


def _system(smiles: str) -> tuple[torch.Tensor, SimpleNamespace]:
    mol = _embedded(smiles)
    coords = torch.as_tensor(mol.GetConformer().GetPositions(), dtype=torch.float64)
    topology = build_physical_topology(
        mol,
        torch.arange(mol.GetNumAtoms(), dtype=torch.long),
    ).to(torch.device("cpu"), torch.float64)
    return coords, SimpleNamespace(topology=topology)


def _active() -> ChemicalConstraintEnergyConfig:
    return ChemicalConstraintEnergyConfig(scale=1.0)


def test_tetrahedral_barrier_is_reference_quiet_and_mirror_sensitive() -> None:
    coords, system = _system("C[C@H](O)F")
    assert system.topology.stereo_term_counts() == {
        "tetrahedral": 1,
        "tetrahedral_cross_fragment": 1,
        "double_bond": 0,
        "double_bond_cross_fragment": 0,
    }
    mirrored = coords.clone()
    mirrored[:, 0] *= -1.0

    reference = chemical_constraint_energy(coords, system, _active())[
        "chemical_stereo_tetrahedral_barrier"
    ]
    inverted = chemical_constraint_energy(mirrored, system, _active())[
        "chemical_stereo_tetrahedral_barrier"
    ]
    assert float(reference) < 1e-8
    assert float(inverted) > 1.0
    assert int(
        chemical_constraint_diagnostics(coords, system)["tetrahedral_inversion_count"]
    ) == 0
    assert int(
        chemical_constraint_diagnostics(mirrored, system)["tetrahedral_inversion_count"]
    ) == 1


def test_tetrahedral_barrier_force_moves_an_inversion_toward_valid_alignment() -> None:
    coords, system = _system("C[C@H](O)F")
    variable = coords.clone()
    variable[:, 0] *= -1.0
    variable.requires_grad_(True)
    before = chemical_constraint_diagnostics(variable, system)[
        "tetrahedral_min_alignment"
    ]
    energy = chemical_constraint_energy(variable, system, _active())["total"]
    force = -torch.autograd.grad(energy, variable)[0]
    after = chemical_constraint_diagnostics(variable.detach() + 1e-3 * force, system)[
        "tetrahedral_min_alignment"
    ]
    assert torch.isfinite(force).all()
    assert float(after.detach()) > float(before.detach())


def test_tetrahedral_barrier_is_invariant_to_proper_global_rotation() -> None:
    coords, system = _system("C[C@H](O)F")
    rotation = torch.tensor(
        [[0.36, -0.48, 0.80], [0.80, 0.60, 0.00], [-0.48, 0.64, 0.60]],
        dtype=torch.float64,
    )
    torch.testing.assert_close(torch.det(rotation), torch.tensor(1.0, dtype=torch.float64))
    rotated = coords @ rotation.T
    torch.testing.assert_close(
        chemical_constraint_energy(rotated, system, _active())["total"],
        chemical_constraint_energy(coords, system, _active())["total"],
    )


def test_double_bond_barrier_rejects_opposite_ez_alignment() -> None:
    coords, system = _system("F/C=C/F")
    first, begin, end, fourth = system.topology.stereo_double_index[:, 0].tolist()
    del first
    axis = coords[end] - coords[begin]
    axis = axis / axis.norm()
    offset = coords[fourth] - coords[end]
    parallel = torch.dot(offset, axis) * axis
    inverted = coords.clone()
    inverted[fourth] = coords[end] + 2.0 * parallel - offset

    reference = chemical_constraint_energy(coords, system, _active())[
        "chemical_stereo_double_bond_barrier"
    ]
    opposite = chemical_constraint_energy(inverted, system, _active())[
        "chemical_stereo_double_bond_barrier"
    ]
    assert float(reference) < 1e-8
    assert float(opposite) > 1.0
    assert int(
        chemical_constraint_diagnostics(inverted, system)["double_bond_inversion_count"]
    ) == 1


def test_zero_scale_is_an_exact_noop_and_config_is_guarded() -> None:
    coords, system = _system("C[C@H](O)F")
    mirrored = coords.clone()
    mirrored[:, 0] *= -1.0
    components = chemical_constraint_energy(mirrored, system)
    assert all(float(value) == 0.0 for value in components.values())
    with pytest.raises(ValueError, match="scale"):
        ChemicalConstraintEnergyConfig(scale=-0.1)


def test_fragment_projected_stereo_force_points_out_of_inverted_basin() -> None:
    mol = _embedded("CCC[C@H](F)CC")
    reference = torch.as_tensor(mol.GetConformer().GetPositions(), dtype=torch.float64)
    fragments = decompose_fragments(mol, reference)
    assert fragments is not None
    fragment_id = fragments["fragment_id"]
    topology = build_physical_topology(mol, fragment_id).to(
        torch.device("cpu"),
        torch.float64,
    )
    system = SimpleNamespace(topology=topology)
    center, first, second, third = topology.stereo_tetra_index[:, 0].tolist()
    normal = torch.linalg.cross(
        reference[first] - reference[center],
        reference[second] - reference[center],
        dim=-1,
    )
    normal = normal / normal.norm()
    displacement = -2.0 * torch.dot(reference[third] - reference[center], normal) * normal
    moved_fragment = int(fragment_id[third])
    inverted = reference.clone()
    inverted[fragment_id == moved_fragment] += displacement
    centers = fragments["frag_centers"].to(torch.float64)
    centers[moved_fragment] += displacement

    variable = inverted.unsqueeze(0).requires_grad_(True)
    before = chemical_constraint_diagnostics(variable, system)[
        "tetrahedral_min_alignment"
    ]
    energy = chemical_constraint_energy(variable, system, _active())["total"]
    atom_force = -torch.autograd.grad(energy.sum(), variable)[0]
    translation, angular = project_atom_forces(
        atom_force,
        variable.detach(),
        centers.unsqueeze(0),
        fragment_id,
        topology.mass,
    )
    lever = variable.detach() - centers.unsqueeze(0)[:, fragment_id]
    atom_velocity = translation[:, fragment_id] + torch.linalg.cross(
        angular[:, fragment_id],
        lever,
        dim=-1,
    )
    after = chemical_constraint_diagnostics(variable.detach() + 1e-3 * atom_velocity, system)[
        "tetrahedral_min_alignment"
    ]
    assert float(before.detach()) < 0.0
    assert float(after.detach()) > float(before.detach())


def test_chemical_constraint_direct_drift_is_an_independent_normalized_channel(
    monkeypatch,
) -> None:
    mol = _embedded("CCC[C@H](F)CC")
    reference = torch.as_tensor(mol.GetConformer().GetPositions(), dtype=torch.float64)
    fragments = decompose_fragments(mol, reference)
    assert fragments is not None
    fragment_id = fragments["fragment_id"]
    topology = build_physical_topology(mol, fragment_id).to(
        torch.device("cpu"),
        torch.float64,
    )
    system = SimpleNamespace(topology=topology)
    center, first, second, third = topology.stereo_tetra_index[:, 0].tolist()
    normal = torch.linalg.cross(
        reference[first] - reference[center],
        reference[second] - reference[center],
        dim=-1,
    )
    normal = normal / normal.norm()
    displacement = -2.0 * torch.dot(reference[third] - reference[center], normal) * normal
    moved_fragment = int(fragment_id[third])
    inverted = reference.clone()
    inverted[fragment_id == moved_fragment] += displacement
    centers = fragments["frag_centers"].to(torch.float64)
    centers[moved_fragment] += displacement

    guidance = UnifiedGuidance(
        system,
        UnifiedGuidanceConfig(
            start_t=0.5,
            chemical_constraint_strength=0.25,
            chemical_constraint_ramp_power=1.0,
            chemical_constraint_max_atom_displacement=0.05,
            max_atom_force=1e6,
            max_translation_velocity=1e6,
            max_angular_velocity=1e6,
            max_atom_displacement=1e6,
        ),
    )

    def zero_main_direction(coords, centers, *, progress, apply_schedule_and_caps):
        del progress, apply_schedule_and_caps
        batch_size = coords.shape[0]
        zeros = coords.new_zeros(batch_size, guidance.n_fragments, 3)
        return (
            zeros,
            zeros,
            coords.new_zeros(batch_size),
            torch.ones(batch_size, dtype=torch.bool),
        )

    monkeypatch.setattr(guidance, "_direction", zero_main_direction)
    learned_translation = torch.ones_like(centers).unsqueeze(0)
    learned_angular = torch.zeros_like(learned_translation)
    gated_translation, gated_angular = guidance.direct_velocity(
        inverted,
        centers,
        learned_translation.reshape(-1, 3),
        learned_angular.reshape(-1, 3),
        fragments["frag_sizes"],
        t_start=0.0,
        t_end=0.5,
        strength=0.0,
    )
    assert torch.equal(gated_translation, torch.zeros_like(gated_translation))
    assert torch.equal(gated_angular, torch.zeros_like(gated_angular))

    translation, angular = guidance.direct_velocity(
        inverted,
        centers,
        learned_translation.reshape(-1, 3),
        learned_angular.reshape(-1, 3),
        fragments["frag_sizes"],
        t_start=0.5,
        t_end=1.0,
        strength=0.0,
    )
    translation = translation.view(1, guidance.n_fragments, 3)
    angular = angular.view(1, guidance.n_fragments, 3)
    lever = inverted.unsqueeze(0) - centers.unsqueeze(0)[:, fragment_id]
    atom_velocity = translation[:, fragment_id] + torch.linalg.cross(
        angular[:, fragment_id],
        lever,
        dim=-1,
    )
    before = chemical_constraint_diagnostics(inverted, system)[
        "tetrahedral_min_alignment"
    ]
    after = chemical_constraint_diagnostics(
        inverted.unsqueeze(0) + 0.5 * atom_velocity,
        system,
    )["tetrahedral_min_alignment"]

    assert float(before) < 0.0
    assert float(after) > float(before)
    stats = guidance.diagnostics()
    assert stats["direct_chemical_constraint_pose_applied"] == 1
    assert stats["direct_chemical_constraint_max_estimated_atom_displacement"] <= 0.05
