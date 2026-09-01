from __future__ import annotations

import math
from dataclasses import replace

import pytest
import torch

Chem = pytest.importorskip("rdkit.Chem")

from effdock.guidance.errors import UnsupportedPhysicalChemistryError  # noqa: E402
from effdock.guidance.interaction import (  # noqa: E402
    InteractionEnergyConfig,
    _aggregate_ring_system_pairs,
    _aggregate_ring_to_system,
    interaction_contact_stats,
    interaction_energy,
)
from effdock.guidance.parameterization import element_parameters  # noqa: E402
from effdock.guidance.system import (  # noqa: E402
    InteractionTopology,
    PhysicalSystem,
    build_physical_system,
    type_ligand_interactions,
)
from effdock.guidance.topology import build_physical_topology  # noqa: E402

DTYPE = torch.float64


def _reference_soft_or(values: torch.Tensor, dim: int | tuple[int, ...]) -> torch.Tensor:
    clamped = values.clamp(0.0, 1.0)
    return -torch.expm1(torch.log1p(-(1.0 - 1e-7) * clamped).sum(dim=dim))


def _reference_ring_system_pairs(
    weights: torch.Tensor,
    left_system: torch.Tensor,
    right_system: torch.Tensor,
) -> torch.Tensor:
    rows = []
    for left_index in range(int(left_system.max().item()) + 1):
        columns = []
        left_mask = left_system == left_index
        for right_index in range(int(right_system.max().item()) + 1):
            selected = weights[:, left_mask][:, :, right_system == right_index]
            columns.append(_reference_soft_or(selected, dim=(1, 2)))
        rows.append(torch.stack(columns, dim=1))
    return torch.stack(rows, dim=1)


def _reference_ring_to_system(
    weights: torch.Tensor,
    ring_system: torch.Tensor,
) -> torch.Tensor:
    return torch.stack(
        [
            _reference_soft_or(weights[:, ring_system == system_index], dim=1)
            for system_index in range(int(ring_system.max().item()) + 1)
        ],
        dim=1,
    )


def test_batched_ring_system_pair_soft_or_matches_reference_and_gradient() -> None:
    left_system = torch.tensor([0, 0, 2, 3, 3], dtype=torch.long)
    right_system = torch.tensor([0, 1, 1, 3], dtype=torch.long)
    values = torch.linspace(0.02, 0.94, 3 * 5 * 4, dtype=DTYPE).reshape(3, 5, 4)
    scatter_values = values.clone().requires_grad_(True)
    reference_values = values.clone().requires_grad_(True)

    actual = _aggregate_ring_system_pairs(
        scatter_values,
        left_system,
        right_system,
    )
    expected = _reference_ring_system_pairs(
        reference_values,
        left_system,
        right_system,
    )
    probe = torch.linspace(0.3, 1.7, actual.numel(), dtype=DTYPE).reshape_as(actual)
    actual_gradient = torch.autograd.grad((actual * probe).sum(), scatter_values)[0]
    expected_gradient = torch.autograd.grad((expected * probe).sum(), reference_values)[0]

    assert actual.shape == (3, 4, 4)
    assert bool(torch.isfinite(actual).all())
    assert bool(torch.isfinite(actual_gradient).all())
    torch.testing.assert_close(actual, expected, atol=2e-15, rtol=2e-15)
    torch.testing.assert_close(actual_gradient, expected_gradient, atol=2e-14, rtol=2e-14)


def test_batched_ring_to_system_soft_or_matches_reference_and_gradient() -> None:
    ring_system = torch.tensor([0, 0, 2, 3, 3], dtype=torch.long)
    values = torch.linspace(0.03, 0.91, 3 * 5 * 4, dtype=DTYPE).reshape(3, 5, 4)
    scatter_values = values.clone().requires_grad_(True)
    reference_values = values.clone().requires_grad_(True)

    actual = _aggregate_ring_to_system(scatter_values, ring_system)
    expected = _reference_ring_to_system(reference_values, ring_system)
    probe = torch.linspace(0.4, 1.6, actual.numel(), dtype=DTYPE).reshape_as(actual)
    actual_gradient = torch.autograd.grad((actual * probe).sum(), scatter_values)[0]
    expected_gradient = torch.autograd.grad((expected * probe).sum(), reference_values)[0]

    assert actual.shape == (3, 4, 4)
    assert bool(torch.isfinite(actual).all())
    assert bool(torch.isfinite(actual_gradient).all())
    torch.testing.assert_close(actual, expected, atol=2e-15, rtol=2e-15)
    torch.testing.assert_close(actual_gradient, expected_gradient, atol=2e-14, rtol=2e-14)


@pytest.mark.parametrize(
    ("weights", "left_system", "right_system", "expected_shape"),
    (
        (
            torch.empty((2, 0, 3), dtype=DTYPE),
            torch.empty(0, dtype=torch.long),
            torch.tensor([0, 2, 2], dtype=torch.long),
            (2, 0, 3),
        ),
        (
            torch.empty((2, 2, 0), dtype=DTYPE),
            torch.tensor([0, 2], dtype=torch.long),
            torch.empty(0, dtype=torch.long),
            (2, 3, 0),
        ),
    ),
)
def test_batched_ring_system_pair_soft_or_preserves_empty_shapes(
    weights: torch.Tensor,
    left_system: torch.Tensor,
    right_system: torch.Tensor,
    expected_shape: tuple[int, int, int],
) -> None:
    assert (
        _aggregate_ring_system_pairs(weights, left_system, right_system).shape
        == expected_shape
    )


def test_batched_ring_to_system_soft_or_preserves_empty_shape() -> None:
    weights = torch.empty((2, 0, 4), dtype=DTYPE)
    ring_system = torch.empty(0, dtype=torch.long)
    assert _aggregate_ring_to_system(weights, ring_system).shape == (2, 0, 4)


def _mol(smiles: str, coords: torch.Tensor) -> "Chem.Mol":
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None
    assert mol.GetNumAtoms() == coords.shape[0]
    conformer = Chem.Conformer(mol.GetNumAtoms())
    for index, point in enumerate(coords.tolist()):
        conformer.SetAtomPosition(index, point)
    conformer.Set3D(True)
    mol.AddConformer(conformer)
    return mol


def _hexagon(
    *,
    radius: float = 1.4,
    center: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> torch.Tensor:
    angles = torch.arange(6, dtype=DTYPE) * (math.pi / 3.0)
    coords = torch.stack(
        (
            radius * torch.cos(angles),
            radius * torch.sin(angles),
            torch.zeros_like(angles),
        ),
        dim=1,
    )
    return coords + torch.tensor(center, dtype=DTYPE)


def _area(coords: torch.Tensor, triplet: tuple[int, int, int] = (0, 2, 4)) -> float:
    a, b, c = (coords[index] for index in triplet)
    return float(torch.linalg.cross(b - a, c - a, dim=-1).norm())


def _topology(
    mol: "Chem.Mol",
    protein_coords: torch.Tensor,
    *,
    protein_atomic_numbers: torch.Tensor,
    **overrides: object,
) -> InteractionTopology:
    typed = type_ligand_interactions(mol)
    protein_count = int(protein_coords.shape[0])
    values: dict[str, object] = {
        "ligand_neighbor_index": typed["neighbor_index"],
        "ligand_direction_target_cosine": typed["direction_target_cosine"],
        "ligand_direction_geometry_valid": typed["direction_geometry_valid"],
        "ligand_is_donor": typed["is_donor"],
        "ligand_is_acceptor": typed["is_acceptor"],
        "ligand_is_hydrophobe": typed["is_hydrophobe"],
        "ligand_is_geometry_excluded_hbond_site": typed["is_geometry_excluded_hbond_site"],
        "protein_is_donor": torch.zeros(protein_count, dtype=torch.bool),
        "protein_is_acceptor": torch.zeros(protein_count, dtype=torch.bool),
        "protein_is_hydrophobe": torch.zeros(protein_count, dtype=torch.bool),
        "protein_outward_direction": torch.tensor(
            [[1.0, 0.0, 0.0]] * protein_count,
            dtype=DTYPE,
        ),
        "protein_direction_target_cosine": torch.ones(protein_count, dtype=DTYPE),
        "protein_direction_quality": torch.ones(protein_count, dtype=DTYPE),
        "protein_direction_valid": torch.ones(protein_count, dtype=torch.bool),
        "protein_is_ambiguous_histidine": torch.zeros(protein_count, dtype=torch.bool),
        "protein_is_unsupported_variant": torch.zeros(protein_count, dtype=torch.bool),
        "protein_is_geometry_excluded_hbond_site": torch.zeros(
            protein_count,
            dtype=torch.bool,
        ),
        "ligand_atom_labels": typed["atom_labels"],
        "protein_atom_labels": tuple(f"A:RES1:{index}" for index in range(protein_count)),
        "ligand_charge_site_membership": typed["charge_site_membership"],
        "ligand_charge_site_charge": typed["charge_site_charge"],
        "ligand_charge_site_labels": typed["charge_site_labels"],
        "ligand_charge_site_exclusion_labels": typed["charge_site_exclusion_labels"],
        "protein_charge_site_membership": torch.empty(
            (0, protein_count),
            dtype=DTYPE,
        ),
        "protein_charge_site_charge": torch.empty(0, dtype=DTYPE),
        "ligand_aromatic_ring_membership": typed["aromatic_ring_membership"],
        "ligand_aromatic_ring_triplet": typed["aromatic_ring_triplet"],
        "ligand_aromatic_ring_system": typed["aromatic_ring_system"],
        "ligand_aromatic_ring_reference_area": typed["aromatic_ring_reference_area"],
        "ligand_aromatic_ring_is_cation_pi_acceptor": typed["aromatic_ring_is_cation_pi_acceptor"],
        "ligand_aromatic_ring_labels": typed["aromatic_ring_labels"],
        "ligand_aromatic_ring_exclusion_labels": typed["aromatic_ring_exclusion_labels"],
        "protein_aromatic_ring_membership": torch.empty(
            (0, protein_count),
            dtype=DTYPE,
        ),
        "protein_aromatic_ring_triplet": torch.empty((0, 3), dtype=torch.long),
        "protein_aromatic_ring_system": torch.empty(0, dtype=torch.long),
        "protein_aromatic_ring_reference_area": torch.empty(0, dtype=DTYPE),
        "protein_aromatic_ring_is_cation_pi_acceptor": torch.empty(
            0,
            dtype=torch.bool,
        ),
        "ligand_halogen_donor_index": typed["halogen_donor_index"],
        "ligand_halogen_parent_index": typed["halogen_parent_index"],
        "ligand_halogen_exclusion_labels": typed["halogen_exclusion_labels"],
        "ligand_zinc_donor_index": typed["zinc_donor_index"],
        "ligand_zinc_donor_element": typed["zinc_donor_element"],
        "ligand_zinc_donor_exclusion_labels": typed["zinc_donor_exclusion_labels"],
    }
    values.update(overrides)
    topology = InteractionTopology(**values)
    assert topology.ligand_atom_labels == typed["atom_labels"]
    assert protein_atomic_numbers.shape == (protein_count,)
    return topology


def _system(
    mol: "Chem.Mol",
    protein_coords: torch.Tensor,
    protein_atomic_numbers: torch.Tensor,
    **topology_overrides: object,
) -> PhysicalSystem:
    interaction = _topology(
        mol,
        protein_coords,
        protein_atomic_numbers=protein_atomic_numbers,
        **topology_overrides,
    )
    physical = build_physical_topology(
        mol,
        torch.zeros(mol.GetNumAtoms(), dtype=torch.long),
    ).to(torch.device("cpu"), DTYPE)
    parameters = element_parameters(protein_atomic_numbers, dtype=DTYPE)
    return PhysicalSystem(
        topology=physical,
        protein_coords=protein_coords,
        protein_atomic_numbers=protein_atomic_numbers,
        protein_uff_x=parameters.uff_x,
        protein_uff_d=parameters.uff_d,
        protein_vdw_radius=parameters.vdw_radius,
        parameter_set={"name": "test", "version": "test"},
        protein_source_atoms=int(protein_coords.shape[0]),
        interaction_topology=interaction,
        interaction_parameter_set={"name": "test", "version": "test"},
    )


def _pi_system(protein_coords: torch.Tensor) -> tuple[torch.Tensor, PhysicalSystem]:
    ligand_coords = _hexagon()
    mol = _mol("c1ccccc1", ligand_coords)
    membership = torch.full((1, 6), 1.0 / 6.0, dtype=DTYPE)
    system = _system(
        mol,
        protein_coords,
        torch.full((6,), 6, dtype=torch.long),
        protein_is_hydrophobe=torch.ones(6, dtype=torch.bool),
        protein_aromatic_ring_membership=membership,
        protein_aromatic_ring_triplet=torch.tensor([[0, 2, 4]], dtype=torch.long),
        protein_aromatic_ring_system=torch.tensor([0], dtype=torch.long),
        protein_aromatic_ring_reference_area=torch.tensor(
            [_area(protein_coords)],
            dtype=DTYPE,
        ),
        protein_aromatic_ring_is_cation_pi_acceptor=torch.tensor([True]),
        protein_aromatic_ring_labels=("protein:PHE:ring",),
    )
    return ligand_coords, system


def test_pi_stacking_geometry_cutoff_collapse_batch_and_se3() -> None:
    config = InteractionEnergyConfig(active_terms=("pi_stacking",))
    ligand, parallel_system = _pi_system(_hexagon(center=(0.0, 0.0, 4.0)))

    rotation_90 = torch.tensor(
        [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]],
        dtype=DTYPE,
    )
    t_ring = _hexagon() @ rotation_90.T + torch.tensor([0.0, 0.0, 4.0], dtype=DTYPE)
    _, t_system = _pi_system(t_ring)

    angle = math.pi / 4.0
    rotation_45 = torch.tensor(
        [
            [math.cos(angle), 0.0, math.sin(angle)],
            [0.0, 1.0, 0.0],
            [-math.sin(angle), 0.0, math.cos(angle)],
        ],
        dtype=DTYPE,
    )
    oblique_ring = _hexagon() @ rotation_45.T + torch.tensor(
        [0.0, 0.0, 4.0],
        dtype=DTYPE,
    )
    _, oblique_system = _pi_system(oblique_ring)
    _, cutoff_system = _pi_system(_hexagon(center=(0.0, 0.0, 5.6)))

    parallel = interaction_energy(ligand, parallel_system, config)["interaction_pi_stacking"]
    t_shaped = interaction_energy(ligand, t_system, config)["interaction_pi_stacking"]
    oblique = interaction_energy(ligand, oblique_system, config)["interaction_pi_stacking"]
    cutoff = interaction_energy(ligand, cutoff_system, config)["interaction_pi_stacking"]
    assert parallel < -0.2
    assert t_shaped < -0.2
    assert oblique == pytest.approx(0.0, abs=1e-12)
    assert cutoff == pytest.approx(0.0, abs=1e-12)

    collapsed = torch.zeros_like(ligand, requires_grad=True)
    collapsed_energy = interaction_energy(collapsed, parallel_system, config)[
        "interaction_pi_stacking"
    ]
    collapsed_energy.backward()
    assert torch.isfinite(collapsed_energy)
    assert collapsed.grad is not None and bool(torch.isfinite(collapsed.grad).all())

    batch = torch.stack((ligand, ligand))
    batched = interaction_energy(batch, parallel_system, config)["interaction_pi_stacking"]
    assert torch.allclose(batched, torch.stack((parallel, parallel)))

    rigid_rotation = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=DTYPE,
    )
    translation = torch.tensor([2.0, -3.0, 1.5], dtype=DTYPE)
    transformed_ligand = ligand @ rigid_rotation.T + translation
    transformed_topology = replace(
        parallel_system.interaction_topology,
        protein_outward_direction=(
            parallel_system.interaction_topology.protein_outward_direction @ rigid_rotation.T
        ),
    )
    transformed_system = replace(
        parallel_system,
        protein_coords=parallel_system.protein_coords @ rigid_rotation.T + translation,
        interaction_topology=transformed_topology,
    )
    transformed = interaction_energy(transformed_ligand, transformed_system, config)[
        "interaction_pi_stacking"
    ]
    assert transformed == pytest.approx(float(parallel), abs=1e-12)


def test_pi_stacking_is_a_favorable_correction_to_hydrophobic_contact() -> None:
    ligand, system = _pi_system(_hexagon(center=(0.0, 0.0, 4.0)))
    hydrophobic = interaction_energy(
        ligand,
        system,
        InteractionEnergyConfig(active_terms=("hydrophobic",)),
    )["total"]
    combined_config = InteractionEnergyConfig(
        active_terms=("hydrophobic", "pi_stacking"),
    )
    combined = interaction_energy(ligand, system, combined_config)["total"]
    stats = interaction_contact_stats(ligand, system, combined_config)
    default_stats = interaction_contact_stats(
        ligand,
        system,
        InteractionEnergyConfig(),
    )

    assert combined < hydrophobic
    assert stats["hydrophobic"]["pi_overlap_weight_sum"] > 0
    assert default_stats["pi_stacking"]["weight_sum"] > 0
    assert default_stats["hydrophobic"]["pi_overlap_weight_sum"] > 0


def test_cation_pi_is_callable_in_both_directions() -> None:
    ring = _hexagon()
    ligand_coords = torch.cat(
        (ring, torch.tensor([[0.0, 0.0, 4.0]], dtype=DTYPE)),
        dim=0,
    )
    mol = _mol("c1ccccc1.[NH4+]", ligand_coords)
    protein_coords = ligand_coords.clone()
    ring_membership = torch.zeros((1, 7), dtype=DTYPE)
    ring_membership[0, :6] = 1.0 / 6.0
    protein_charge_membership = torch.zeros((1, 7), dtype=DTYPE)
    protein_charge_membership[0, 6] = 1.0
    system = _system(
        mol,
        protein_coords,
        torch.tensor([6, 6, 6, 6, 6, 6, 7], dtype=torch.long),
        protein_aromatic_ring_membership=ring_membership,
        protein_aromatic_ring_triplet=torch.tensor([[0, 2, 4]], dtype=torch.long),
        protein_aromatic_ring_system=torch.tensor([0], dtype=torch.long),
        protein_aromatic_ring_reference_area=torch.tensor([_area(ring)], dtype=DTYPE),
        protein_aromatic_ring_is_cation_pi_acceptor=torch.tensor([True]),
        protein_aromatic_ring_labels=("protein:PHE:ring",),
        protein_charge_site_membership=protein_charge_membership,
        protein_charge_site_charge=torch.tensor([1.0], dtype=DTYPE),
        protein_charge_site_labels=("protein:LYS:NZ",),
    )
    config = InteractionEnergyConfig(active_terms=("cation_pi",))
    energy = interaction_energy(ligand_coords, system, config)
    stats = interaction_contact_stats(ligand_coords, system, config)["cation_pi"]

    assert energy["interaction_cation_pi"] < -0.9
    assert stats["ligand_ring_to_protein_cation"]["nonzero_pairs"] == 1
    assert stats["protein_ring_to_ligand_cation"]["nonzero_pairs"] == 1


def test_halogen_bond_geometry_cutoff_and_typing() -> None:
    ligand = torch.tensor([[0.0, 0.0, 0.0], [1.8, 0.0, 0.0]], dtype=DTYPE)
    mol = _mol("CCl", ligand)
    protein = torch.tensor([[4.9, 0.0, 0.0]], dtype=DTYPE)
    system = _system(
        mol,
        protein,
        torch.tensor([8], dtype=torch.long),
        protein_outward_direction=torch.tensor([[-1.0, 0.0, 0.0]], dtype=DTYPE),
        protein_halogen_acceptor_index=torch.tensor([0], dtype=torch.long),
    )
    config = InteractionEnergyConfig(active_terms=("halogen_bond",))
    aligned = interaction_energy(ligand, system, config)["interaction_halogen_bond"]

    misaligned_ligand = ligand.clone()
    misaligned_ligand[0] = torch.tensor([1.8, -1.8, 0.0], dtype=DTYPE)
    misaligned = interaction_energy(misaligned_ligand, system, config)["interaction_halogen_bond"]
    cutoff_system = replace(
        system,
        protein_coords=torch.tensor([[5.7, 0.0, 0.0]], dtype=DTYPE),
    )
    cutoff = interaction_energy(ligand, cutoff_system, config)["interaction_halogen_bond"]

    typed = type_ligand_interactions(mol)
    assert typed["halogen_donor_index"].tolist() == [1]
    assert typed["halogen_parent_index"].tolist() == [0]
    assert aligned < -0.4
    assert misaligned == pytest.approx(0.0, abs=1e-12)
    assert cutoff == pytest.approx(0.0, abs=1e-12)


def _metal_system(
    ligand: torch.Tensor,
    *,
    donor_indices: tuple[int, ...] = (0,),
) -> tuple[torch.Tensor, PhysicalSystem]:
    mol = _mol("OCO", ligand)
    protein = torch.tensor(
        [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=DTYPE,
    )
    system = _system(
        mol,
        protein,
        torch.tensor([7, 8, 16], dtype=torch.long),
        ligand_zinc_donor_index=torch.tensor(donor_indices, dtype=torch.long),
        ligand_zinc_donor_element=torch.full(
            (len(donor_indices),),
            8,
            dtype=torch.long,
        ),
        zinc_coords=torch.zeros((1, 3), dtype=DTYPE),
        zinc_vacant_direction=torch.tensor([[1.0, 0.0, 0.0]], dtype=DTYPE),
        zinc_receptor_donor_index=torch.tensor([[0, 1, 2]], dtype=torch.long),
        zinc_receptor_donor_element=torch.tensor([[7, 8, 16]], dtype=torch.long),
        zinc_site_labels=("protein:ZN1",),
    )
    return ligand, system


def _extended_term_directional_derivative_case(
    term: str,
) -> tuple[torch.Tensor, PhysicalSystem, torch.Tensor]:
    if term == "pi_stacking":
        coords, system = _pi_system(_hexagon(center=(0.0, 0.0, 5.0)))
        direction = torch.zeros_like(coords)
        direction[:, 2] = 1.0
        return coords, system, direction

    if term == "cation_pi":
        ring = _hexagon()
        coords = torch.cat(
            (ring, torch.tensor([[0.0, 0.0, 5.2]], dtype=DTYPE)),
            dim=0,
        )
        mol = _mol("c1ccccc1.[NH4+]", coords)
        protein_charge_membership = torch.zeros((1, 7), dtype=DTYPE)
        protein_charge_membership[0, 6] = 1.0
        ring_membership = torch.zeros((1, 7), dtype=DTYPE)
        ring_membership[0, :6] = 1.0 / 6.0
        system = _system(
            mol,
            coords.clone(),
            torch.tensor([6, 6, 6, 6, 6, 6, 7], dtype=torch.long),
            protein_aromatic_ring_membership=ring_membership,
            protein_aromatic_ring_triplet=torch.tensor([[0, 2, 4]], dtype=torch.long),
            protein_aromatic_ring_system=torch.tensor([0], dtype=torch.long),
            protein_aromatic_ring_reference_area=torch.tensor(
                [_area(ring)],
                dtype=DTYPE,
            ),
            protein_aromatic_ring_is_cation_pi_acceptor=torch.tensor(
                [True],
            ),
            protein_aromatic_ring_labels=("protein:PHE:ring",),
            protein_charge_site_membership=protein_charge_membership,
            protein_charge_site_charge=torch.tensor([1.0], dtype=DTYPE),
            protein_charge_site_labels=("protein:LYS:NZ",),
        )
        direction = torch.zeros_like(coords)
        direction[6, 2] = 1.0
        return coords, system, direction

    if term == "halogen_bond":
        coords = torch.tensor(
            [[0.0, 0.0, 0.0], [1.8, 0.0, 0.0]],
            dtype=DTYPE,
        )
        mol = _mol("CCl", coords)
        system = _system(
            mol,
            torch.tensor([[5.315, 0.0, 0.0]], dtype=DTYPE),
            torch.tensor([8], dtype=torch.long),
            protein_outward_direction=torch.tensor(
                [[-1.0, 0.0, 0.0]],
                dtype=DTYPE,
            ),
            protein_halogen_acceptor_index=torch.tensor([0], dtype=torch.long),
        )
        direction = torch.zeros_like(coords)
        direction[:, 0] = 1.0
        return coords, system, direction

    if term == "metal_coordination":
        coords = torch.tensor(
            [[2.55, 0.0, 0.0], [6.0, 5.0, 5.0], [6.0, 6.0, 5.0]],
            dtype=DTYPE,
        )
        coords, system = _metal_system(coords)
        direction = torch.zeros_like(coords)
        direction[0, 0] = 1.0
        return coords, system, direction

    raise AssertionError(f"unknown extended interaction term: {term}")


@pytest.mark.parametrize(
    "term",
    ("pi_stacking", "cation_pi", "halogen_bond", "metal_coordination"),
)
def test_extended_term_autograd_matches_central_directional_difference(
    term: str,
) -> None:
    coords, system, direction = _extended_term_directional_derivative_case(term)
    assert coords.dtype == DTYPE
    direction = direction / direction.norm()
    config = InteractionEnergyConfig(active_terms=(term,))
    component = f"interaction_{term}"

    work = coords.clone().requires_grad_(True)
    energy = interaction_energy(work, system, config)[component]
    gradient = torch.autograd.grad(energy, work)[0]
    autograd_directional = (gradient * direction).sum()

    step = 1e-5
    energy_plus = interaction_energy(
        coords + step * direction,
        system,
        config,
    )[component]
    energy_minus = interaction_energy(
        coords - step * direction,
        system,
        config,
    )[component]
    finite_difference = (energy_plus - energy_minus) / (2.0 * step)

    assert torch.isfinite(autograd_directional)
    assert torch.isfinite(finite_difference)
    assert abs(float(finite_difference)) > 1e-5
    torch.testing.assert_close(
        autograd_directional,
        finite_difference,
        atol=2e-8,
        rtol=2e-6,
    )


def test_metal_coordination_direction_r0_slot_non_donor_and_finiteness() -> None:
    config = InteractionEnergyConfig(active_terms=("metal_coordination",))
    base = torch.tensor(
        [[2.1, 0.0, 0.0], [6.0, 5.0, 5.0], [6.0, 6.0, 5.0]],
        dtype=DTYPE,
    )
    aligned_coords, aligned_system = _metal_system(base)
    aligned = interaction_energy(aligned_coords, aligned_system, config)[
        "interaction_metal_coordination"
    ]
    aligned_stats = interaction_contact_stats(
        aligned_coords,
        aligned_system,
        config,
    )["metal_coordination"]

    shifted = base.clone()
    shifted[0, 0] = 2.7
    shifted_energy = interaction_energy(shifted, aligned_system, config)[
        "interaction_metal_coordination"
    ]
    opposite = base.clone()
    opposite[0, 0] = -2.1
    opposite_energy = interaction_energy(opposite, aligned_system, config)[
        "interaction_metal_coordination"
    ]
    assert aligned < shifted_energy
    assert aligned < opposite_energy
    assert aligned < 0
    assert aligned_stats["top_donor_pairs"][0]["distance_angstrom"] == pytest.approx(2.1)
    assert aligned_stats["top_donor_pairs"][0]["occupancy"] > 0.9

    two_donors = base.clone()
    two_donors[2] = torch.tensor([2.1, 0.05, 0.0], dtype=DTYPE)
    _, two_donor_system = _metal_system(two_donors, donor_indices=(0, 2))
    slot_stats = interaction_contact_stats(two_donors, two_donor_system, config)[
        "metal_coordination"
    ]
    assert slot_stats["slot_energy_kcal_mol"] > 0
    assert slot_stats["overcoordination_energy_kcal_mol"] > 0

    crowded = base.clone()
    crowded[1] = torch.tensor([0.1, 0.0, 0.0], dtype=DTYPE)
    crowded_energy = interaction_energy(crowded, aligned_system, config)[
        "interaction_metal_coordination"
    ]
    crowded_stats = interaction_contact_stats(crowded, aligned_system, config)["metal_coordination"]
    assert crowded_stats["non_donor_repulsion_kcal_mol"] > 0
    assert crowded_stats["top_non_donor_repulsions"]
    assert crowded_energy > aligned

    collapsed = base.clone()
    collapsed[0] = 0
    collapsed.requires_grad_(True)
    collapsed_energy = interaction_energy(collapsed, aligned_system, config)[
        "interaction_metal_coordination"
    ]
    collapsed_energy.backward()
    assert torch.isfinite(collapsed_energy)
    assert collapsed.grad is not None and bool(torch.isfinite(collapsed.grad).all())


def test_polar_unsatisfied_proxy_is_trace_only() -> None:
    ligand = torch.tensor([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]], dtype=DTYPE)
    mol = _mol("CO", ligand)
    system = _system(
        mol,
        torch.tensor([[2.4, 0.0, 0.0]], dtype=DTYPE),
        torch.tensor([6], dtype=torch.long),
    )
    config = InteractionEnergyConfig(active_terms=("hydrophobic",))
    components = interaction_energy(ligand, system, config)
    proxy = interaction_contact_stats(ligand, system, config)["polar_unsatisfied_proxy"]

    assert proxy["status"] == "trace_only_unitless_no_force"
    assert proxy["site_count"] >= 1
    assert proxy["sum"] > 0
    assert not any("polar" in name for name in components)


def test_extended_ligand_typing_masks() -> None:
    benzene = type_ligand_interactions(Chem.MolFromSmiles("c1ccccc1"))
    pyridine = type_ligand_interactions(Chem.MolFromSmiles("n1ccccc1"))
    thiocarbonyl = type_ligand_interactions(Chem.MolFromSmiles("C=S"))
    thiolate = type_ligand_interactions(Chem.MolFromSmiles("C[S-]"))
    chloromethane = type_ligand_interactions(Chem.MolFromSmiles("CCl"))
    fluoromethane = type_ligand_interactions(Chem.MolFromSmiles("CF"))
    chloride = type_ligand_interactions(Chem.MolFromSmiles("[Cl-]"))
    chloramine = type_ligand_interactions(Chem.MolFromSmiles("NCl"))
    methylamine = type_ligand_interactions(Chem.MolFromSmiles("CN"))
    thioether = type_ligand_interactions(Chem.MolFromSmiles("CSC"))

    assert benzene["aromatic_ring_membership"].shape[0] == 1
    assert benzene["aromatic_ring_is_cation_pi_acceptor"].tolist() == [True]
    assert pyridine["aromatic_ring_membership"].shape[0] == 1
    assert pyridine["aromatic_ring_is_cation_pi_acceptor"].tolist() == [False]
    assert 7 in pyridine["zinc_donor_element"].tolist()
    assert chloromethane["halogen_donor_index"].tolist() == [1]
    assert chloromethane["halogen_parent_index"].tolist() == [0]
    assert fluoromethane["halogen_donor_index"].numel() == 0
    assert chloride["halogen_donor_index"].numel() == 0
    assert chloride["halogen_exclusion_labels"]
    assert chloramine["halogen_donor_index"].numel() == 0
    assert chloramine["halogen_exclusion_labels"]
    assert methylamine["zinc_donor_index"].numel() == 0
    assert methylamine["zinc_donor_exclusion_labels"]
    assert thiocarbonyl["zinc_donor_element"].tolist() == [16]
    assert thiolate["zinc_donor_element"].tolist() == [16]
    assert 16 in thioether["zinc_donor_element"].tolist()


def _pdb_line(
    record: str,
    serial: int,
    atom_name: str,
    residue: str,
    residue_number: int,
    coord: torch.Tensor,
    element: str,
    altloc: str = " ",
) -> str:
    return (
        f"{record:<6}{serial:>5} {atom_name:^4}{altloc:1}{residue:>3} A"
        f"{residue_number:>4}    {float(coord[0]):>8.3f}"
        f"{float(coord[1]):>8.3f}{float(coord[2]):>8.3f}"
        f"  1.00 20.00          {element:>2}\n"
    )


def test_trp_only_six_member_ring_is_cation_pi_acceptor(tmp_path) -> None:
    ligand = torch.tensor([[0.0, 0.0, 0.0]], dtype=DTYPE)
    mol = _mol("C", ligand)
    ring_coords = {
        "CG": (-1.2, 0.0, 0.0),
        "CD1": (-0.6, 1.2, 0.0),
        "NE1": (0.7, 1.1, 0.0),
        "CE2": (1.1, -0.1, 0.0),
        "CD2": (-0.1, -0.7, 0.0),
        "CZ2": (2.2, -0.9, 0.0),
        "CH2": (1.9, -2.2, 0.0),
        "CZ3": (0.6, -2.7, 0.0),
        "CE3": (-0.4, -1.8, 0.0),
    }
    offset = torch.tensor([4.0, 4.0, 0.0], dtype=DTYPE)
    receptor = tmp_path / "trp.pdb"
    receptor.write_text(
        "".join(
            _pdb_line(
                "ATOM",
                serial,
                atom_name,
                "TRP",
                1,
                torch.tensor(coord, dtype=DTYPE) + offset,
                "N" if atom_name == "NE1" else "C",
            )
            for serial, (atom_name, coord) in enumerate(ring_coords.items(), start=1)
        )
        + "END\n"
    )
    system = build_physical_system(
        mol,
        receptor,
        fragment_id=torch.zeros(1, dtype=torch.long),
        near_coords=ligand,
    ).to(torch.device("cpu"), DTYPE)
    topology = system.interaction_topology
    assert topology is not None

    flags = {
        "pyrrole" if ":pyrrole:" in label else "benzene": bool(flag)
        for label, flag in zip(
            topology.protein_aromatic_ring_labels,
            topology.protein_aromatic_ring_is_cation_pi_acceptor,
            strict=True,
        )
    }
    assert flags == {"pyrrole": False, "benzene": True}
    assert topology.protein_aromatic_ring_system.tolist() == [0, 0]


def test_strict_pdb_zinc_site_is_admitted_and_coordination_water_fails(
    tmp_path,
) -> None:
    tetrahedral = torch.tensor(
        [
            [1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
        ],
        dtype=DTYPE,
    ) / math.sqrt(3.0)
    ligand = torch.stack(
        (
            3.7 * tetrahedral[0],
            2.25 * tetrahedral[0],
            3.7 * tetrahedral[0] + torch.tensor([0.0, 1.2, -1.2], dtype=DTYPE),
        )
    )
    mol = _mol("CSC", ligand)
    lines = [
        _pdb_line("HETATM", 1, "ZN", "ZN", 1, torch.zeros(3), "ZN"),
        *[
            _pdb_line(
                "ATOM",
                donor + 2,
                "SG",
                "CYM",
                donor + 2,
                2.25 * tetrahedral[donor + 1],
                "S",
            )
            for donor in range(3)
        ],
        "END\n",
    ]
    receptor = tmp_path / "zinc.pdb"
    receptor.write_text("".join(lines))
    system = build_physical_system(
        mol,
        receptor,
        fragment_id=torch.zeros(3, dtype=torch.long),
        near_coords=ligand,
    ).to(torch.device("cpu"), DTYPE)
    topology = system.interaction_topology
    assert topology is not None
    assert topology.zinc_coords.shape == (1, 3)
    assert topology.zinc_receptor_donor_index.shape == (1, 3)
    torch.testing.assert_close(
        topology.zinc_vacant_direction[0],
        tetrahedral[0],
        atol=2e-3,
        rtol=0.0,
    )
    default_energy = interaction_energy(ligand, system)["interaction_metal_coordination"]
    assert default_energy < 0
    with pytest.raises(
        UnsupportedPhysicalChemistryError,
        match="requires explicit metal_coordination activation",
    ) as inactive_exc:
        interaction_energy(
            ligand,
            system,
            InteractionEnergyConfig(
                active_terms=(
                    "hydrophobic",
                    "hydrogen_bond",
                    "screened_formal_charge",
                )
            ),
        )
    assert inactive_exc.value.code == "required_interaction_term_inactive"

    energy = interaction_energy(
        ligand,
        system,
        InteractionEnergyConfig(active_terms=("metal_coordination",)),
    )["interaction_metal_coordination"]
    assert energy < 0

    with_water = tmp_path / "zinc_water.pdb"
    with_water.write_text(
        "".join(
            lines[:-1]
            + [
                _pdb_line(
                    "HETATM",
                    10,
                    "O",
                    "HOH",
                    10,
                    2.0 * tetrahedral[0],
                    "O",
                ),
                "END\n",
            ]
        )
    )
    with pytest.raises(
        UnsupportedPhysicalChemistryError,
        match="coordination water",
    ) as exc_info:
        build_physical_system(
            mol,
            with_water,
            fragment_id=torch.zeros(3, dtype=torch.long),
            near_coords=ligand,
        )
    assert exc_info.value.code == "unsupported_zinc_site"

    alternate_records = {
        "donor": _pdb_line(
            "ATOM",
            11,
            "ND1",
            "HIS",
            11,
            2.4 * tetrahedral[0],
            "N",
            altloc="B",
        ),
        "water": _pdb_line(
            "HETATM",
            12,
            "O",
            "HOH",
            12,
            2.0 * tetrahedral[0],
            "O",
            altloc="C",
        ),
    }
    for name, alternate_record in alternate_records.items():
        alternate_path = tmp_path / f"zinc_altloc_{name}.pdb"
        alternate_path.write_text("".join(lines[:-1] + [alternate_record, "END\n"]))
        with pytest.raises(
            UnsupportedPhysicalChemistryError,
            match="alternate-location",
        ) as alternate_exc:
            build_physical_system(
                mol,
                alternate_path,
                fragment_id=torch.zeros(3, dtype=torch.long),
                near_coords=ligand,
            )
        assert alternate_exc.value.code == "unsupported_zinc_site"
