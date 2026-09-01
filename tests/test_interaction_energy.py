from __future__ import annotations

import math
from dataclasses import fields, replace

import pytest
import torch

Chem = pytest.importorskip("rdkit.Chem")

import effdock.guidance.interaction as interaction_module  # noqa: E402
from effdock.guidance.interaction import (  # noqa: E402
    InteractionEnergyConfig,
    _cone_gate,
    _site_saturated_energy,
    _squared_decreasing_switch,
    interaction_contact_stats,
    interaction_energy,
)
from effdock.guidance.parameterization import element_parameters  # noqa: E402
from effdock.guidance.physical import physical_energy  # noqa: E402
from effdock.guidance.runtime import guidance_energy  # noqa: E402
from effdock.guidance.system import (  # noqa: E402
    InteractionTopology,
    PhysicalSystem,
    type_ligand_interactions,
)
from effdock.guidance.topology import build_physical_topology  # noqa: E402


def _molecule(
    smiles: str,
    positions: torch.Tensor,
) -> "Chem.Mol":
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None
    assert mol.GetNumAtoms() == positions.shape[0]
    conformer = Chem.Conformer(mol.GetNumAtoms())
    for index, position in enumerate(positions.tolist()):
        conformer.SetAtomPosition(index, position)
    conformer.Set3D(True)
    mol.AddConformer(conformer)
    return mol


def _system(
    mol: "Chem.Mol",
    ligand_positions: torch.Tensor,
    *,
    protein_coords: torch.Tensor,
    protein_atomic_numbers: torch.Tensor,
    interaction: InteractionTopology,
) -> PhysicalSystem:
    topology = build_physical_topology(
        mol,
        torch.zeros(mol.GetNumAtoms(), dtype=torch.long),
    ).to(torch.device("cpu"), torch.float64)
    parameters = element_parameters(
        protein_atomic_numbers,
        dtype=torch.float64,
    )
    return PhysicalSystem(
        topology=topology,
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


def _hbond_fixture(
    protein_distance: float = 2.9,
) -> tuple[torch.Tensor, PhysicalSystem]:
    ligand_positions = torch.tensor(
        [[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        dtype=torch.float64,
    )
    mol = _molecule("CN", ligand_positions)
    interaction = InteractionTopology(
        ligand_neighbor_index=torch.tensor(
            [[0, 1], [1, 0]],
            dtype=torch.long,
        ),
        ligand_direction_target_cosine=torch.tensor(
            [1.0, 1.0],
            dtype=torch.float64,
        ),
        ligand_direction_geometry_valid=torch.ones(
            2,
            dtype=torch.bool,
        ),
        ligand_is_donor=torch.tensor([False, True]),
        ligand_is_acceptor=torch.zeros(2, dtype=torch.bool),
        ligand_is_hydrophobe=torch.zeros(2, dtype=torch.bool),
        ligand_is_geometry_excluded_hbond_site=torch.zeros(
            2,
            dtype=torch.bool,
        ),
        protein_is_donor=torch.tensor([False]),
        protein_is_acceptor=torch.tensor([True]),
        protein_is_hydrophobe=torch.tensor([False]),
        protein_outward_direction=torch.tensor(
            [[-1.0, 0.0, 0.0]],
            dtype=torch.float64,
        ),
        protein_direction_target_cosine=torch.tensor(
            [1.0],
            dtype=torch.float64,
        ),
        protein_direction_quality=torch.tensor(
            [1.0],
            dtype=torch.float64,
        ),
        protein_direction_valid=torch.tensor([True]),
        protein_is_ambiguous_histidine=torch.tensor([False]),
        protein_is_unsupported_variant=torch.tensor([False]),
        protein_is_geometry_excluded_hbond_site=torch.tensor([False]),
        ligand_atom_labels=("0:C", "1:N"),
        protein_atom_labels=("A:ASN1:OD1",),
    )
    system = _system(
        mol,
        ligand_positions,
        protein_coords=torch.tensor(
            [[protein_distance, 0.0, 0.0]],
            dtype=torch.float64,
        ),
        protein_atomic_numbers=torch.tensor([8]),
        interaction=interaction,
    )
    return ligand_positions, system


def _charge_fixture(
    *,
    protein_distance: float = 4.0,
    ligand_charge: float = 1.0,
    protein_charge: float = -1.0,
) -> tuple[torch.Tensor, PhysicalSystem]:
    coords, system = _hbond_fixture(protein_distance=protein_distance)
    interaction = replace(
        system.interaction_topology,
        ligand_charge_site_membership=torch.tensor(
            [[0.0, 1.0]],
            dtype=torch.float64,
        ),
        ligand_charge_site_charge=torch.tensor(
            [ligand_charge],
            dtype=torch.float64,
        ),
        ligand_charge_site_labels=("ligand:N+",),
        protein_charge_site_membership=torch.tensor(
            [[1.0]],
            dtype=torch.float64,
        ),
        protein_charge_site_charge=torch.tensor(
            [protein_charge],
            dtype=torch.float64,
        ),
        protein_charge_site_labels=("A:ASP1:OD1+OD2",),
    )
    return coords, replace(system, interaction_topology=interaction)


def test_versioned_smarts_type_key_chemistry() -> None:
    amide = Chem.MolFromSmiles("CC(=O)N")
    assert amide is not None
    typed = type_ligand_interactions(amide)
    assert bool(typed["is_hydrophobe"][0])
    assert not bool(typed["is_hydrophobe"][1])
    assert bool(typed["is_acceptor"][2])
    assert bool(typed["is_donor"][3])
    assert not bool(typed["is_acceptor"][3])
    assert float(typed["direction_target_cosine"][2]) == pytest.approx(0.5)
    assert float(typed["direction_target_cosine"][3]) == pytest.approx(0.5)

    alcohol = Chem.MolFromSmiles("CO")
    ether = Chem.MolFromSmiles("COC")
    assert alcohol is not None and ether is not None
    alcohol_oxygen = next(atom.GetIdx() for atom in alcohol.GetAtoms() if atom.GetAtomicNum() == 8)
    ether_oxygen = next(atom.GetIdx() for atom in ether.GetAtoms() if atom.GetAtomicNum() == 8)
    assert float(
        type_ligand_interactions(alcohol)["direction_target_cosine"][alcohol_oxygen]
    ) == pytest.approx(1.0 / 3.0)
    assert float(
        type_ligand_interactions(ether)["direction_target_cosine"][ether_oxygen]
    ) == pytest.approx(1.0 / (3.0**0.5))

    quaternary = Chem.MolFromSmiles("[N+](C)(C)(C)C")
    assert quaternary is not None
    typed_quaternary = type_ligand_interactions(quaternary)
    nitrogen = next(atom.GetIdx() for atom in quaternary.GetAtoms() if atom.GetAtomicNum() == 7)
    assert not bool(typed_quaternary["is_donor"][nitrogen])
    assert not bool(typed_quaternary["is_acceptor"][nitrogen])

    pyridine = Chem.MolFromSmiles("n1ccccc1")
    pyrrole = Chem.MolFromSmiles("[nH]1cccc1")
    assert pyridine is not None and pyrrole is not None
    pyridine_n = next(atom.GetIdx() for atom in pyridine.GetAtoms() if atom.GetAtomicNum() == 7)
    pyrrole_n = next(atom.GetIdx() for atom in pyrrole.GetAtoms() if atom.GetAtomicNum() == 7)
    assert bool(type_ligand_interactions(pyridine)["is_acceptor"][pyridine_n])
    pyrrole_typed = type_ligand_interactions(pyrrole)
    assert bool(pyrrole_typed["is_donor"][pyrrole_n])
    assert not bool(pyrrole_typed["is_acceptor"][pyrrole_n])

    fluorobenzene = Chem.MolFromSmiles("Fc1ccccc1")
    assert fluorobenzene is not None
    fluorobenzene_typed = type_ligand_interactions(fluorobenzene)
    fluorine = next(atom.GetIdx() for atom in fluorobenzene.GetAtoms() if atom.GetAtomicNum() == 9)
    assert not bool(fluorobenzene_typed["is_hydrophobe"][fluorine])
    assert int(fluorobenzene_typed["is_hydrophobe"].sum()) > 0

    ammonia = Chem.MolFromSmiles("N")
    assert ammonia is not None
    ammonia_typed = type_ligand_interactions(ammonia)
    assert bool(ammonia_typed["is_geometry_excluded_hbond_site"][0])
    assert not bool(ammonia_typed["is_donor"][0])


def test_hydrogen_bond_prefers_aligned_geometry_and_smoothly_cuts_off() -> None:
    coords, system = _hbond_fixture()
    aligned = interaction_energy(coords, system)["interaction_hydrogen_bond"]
    misaligned_coords = coords.clone()
    misaligned_coords[0] = torch.tensor(
        [1.0, 0.0, 0.0],
        dtype=torch.float64,
    )
    misaligned = interaction_energy(
        misaligned_coords,
        system,
    )["interaction_hydrogen_bond"]
    acceptor_misaligned_system = replace(
        system,
        interaction_topology=replace(
            system.interaction_topology,
            protein_outward_direction=torch.tensor(
                [[1.0, 0.0, 0.0]],
                dtype=torch.float64,
            ),
        ),
    )
    acceptor_misaligned = interaction_energy(
        coords,
        acceptor_misaligned_system,
    )["interaction_hydrogen_bond"]
    assert float(aligned) < -0.99
    assert abs(float(misaligned)) < 1e-12
    assert abs(float(acceptor_misaligned)) < 1e-12

    _, outside_system = _hbond_fixture(protein_distance=4.1001)
    outside = interaction_energy(
        coords,
        outside_system,
    )["interaction_hydrogen_bond"]
    assert float(outside) == pytest.approx(0.0, abs=1e-12)

    _, boundary_system = _hbond_fixture(protein_distance=4.1)
    boundary_coords = coords.clone().requires_grad_(True)
    boundary = interaction_energy(
        boundary_coords,
        boundary_system,
    )["interaction_hydrogen_bond"]
    boundary_force = -torch.autograd.grad(boundary, boundary_coords)[0]
    assert float(boundary.detach()) == pytest.approx(0.0, abs=1e-12)
    torch.testing.assert_close(
        boundary_force,
        torch.zeros_like(boundary_force),
        atol=1e-10,
        rtol=0.0,
    )


def test_one_neighbor_cone_prefers_sp2_missing_valence_angle() -> None:
    coords, system = _hbond_fixture()
    target = 0.5
    distance = 2.9
    cone_coords = torch.tensor(
        [[distance * target, distance * (3.0**0.5) / 2.0, 0.0]],
        dtype=torch.float64,
    )
    donor_to_acceptor = cone_coords[0] / distance
    topology = replace(
        system.interaction_topology,
        ligand_direction_target_cosine=torch.tensor(
            [1.0, target],
            dtype=torch.float64,
        ),
        protein_outward_direction=(-donor_to_acceptor).view(1, 3),
        protein_direction_target_cosine=torch.tensor(
            [1.0],
            dtype=torch.float64,
        ),
    )
    cone_system = replace(
        system,
        protein_coords=cone_coords,
        interaction_topology=topology,
    )
    cone_energy = interaction_energy(
        coords,
        cone_system,
    )["interaction_hydrogen_bond"]

    opposite_axis_system = replace(
        cone_system,
        protein_coords=torch.tensor(
            [[distance, 0.0, 0.0]],
            dtype=torch.float64,
        ),
        interaction_topology=replace(
            topology,
            protein_outward_direction=torch.tensor(
                [[-1.0, 0.0, 0.0]],
                dtype=torch.float64,
            ),
        ),
    )
    opposite_axis = interaction_energy(
        coords,
        opposite_axis_system,
    )["interaction_hydrogen_bond"]
    assert float(cone_energy) < -0.99
    assert float(opposite_axis) == pytest.approx(0.0, abs=1e-12)


def test_cone_gate_is_c2_at_support_and_finite_at_axis_target() -> None:
    target = torch.tensor(0.5, dtype=torch.float64)
    lower_and_upper = torch.tensor(
        [-0.5, 1.0],
        dtype=torch.float64,
        requires_grad=True,
    )
    gate = _cone_gate(lower_and_upper, target, 60.0)
    first = torch.autograd.grad(
        gate.sum(),
        lower_and_upper,
        create_graph=True,
    )[0]
    second = torch.autograd.grad(
        first.sum(),
        lower_and_upper,
    )[0]
    torch.testing.assert_close(
        gate,
        torch.zeros_like(gate),
        atol=1e-12,
        rtol=0.0,
    )
    torch.testing.assert_close(
        first,
        torch.zeros_like(first),
        atol=1e-10,
        rtol=0.0,
    )
    torch.testing.assert_close(
        second,
        torch.zeros_like(second),
        atol=1e-8,
        rtol=0.0,
    )

    axis = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    axis_gate = _cone_gate(
        axis,
        torch.tensor(1.0, dtype=torch.float64),
        45.0,
    )
    axis_gradient = torch.autograd.grad(axis_gate, axis)[0]
    assert bool(torch.isfinite(axis_gate))
    assert bool(torch.isfinite(axis_gradient))
    assert float(axis_gate.detach()) == pytest.approx(1.0)
    assert float(axis_gradient.detach()) == pytest.approx(
        0.0,
        abs=1e-10,
    )


def test_near_cancelled_ligand_axis_has_finite_suppressed_force() -> None:
    positions = torch.tensor(
        [[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.11, 0.0]],
        dtype=torch.float64,
    )
    mol = _molecule("CNC", positions)
    neighbor_index = torch.tensor(
        [[0, 1, 1, 2], [1, 0, 2, 1]],
        dtype=torch.long,
    )
    first_axis = positions[1] - positions[0]
    second_axis = positions[1] - positions[2]
    direction = first_axis / first_axis.norm() + second_axis / second_axis.norm()
    direction = direction / direction.norm()
    interaction = InteractionTopology(
        ligand_neighbor_index=neighbor_index,
        ligand_direction_target_cosine=torch.tensor(
            [1.0, 1.0, 1.0],
            dtype=torch.float64,
        ),
        ligand_direction_geometry_valid=torch.ones(
            3,
            dtype=torch.bool,
        ),
        ligand_is_donor=torch.tensor([False, True, False]),
        ligand_is_acceptor=torch.zeros(3, dtype=torch.bool),
        ligand_is_hydrophobe=torch.zeros(3, dtype=torch.bool),
        ligand_is_geometry_excluded_hbond_site=torch.zeros(
            3,
            dtype=torch.bool,
        ),
        protein_is_donor=torch.tensor([False]),
        protein_is_acceptor=torch.tensor([True]),
        protein_is_hydrophobe=torch.tensor([False]),
        protein_outward_direction=(-direction).view(1, 3),
        protein_direction_target_cosine=torch.tensor(
            [1.0],
            dtype=torch.float64,
        ),
        protein_direction_quality=torch.tensor(
            [1.0],
            dtype=torch.float64,
        ),
        protein_direction_valid=torch.tensor([True]),
        protein_is_ambiguous_histidine=torch.tensor([False]),
        protein_is_unsupported_variant=torch.tensor([False]),
        protein_is_geometry_excluded_hbond_site=torch.tensor([False]),
        ligand_atom_labels=("0:C", "1:N", "2:C"),
        protein_atom_labels=("A:ASN1:OD1",),
    )
    system = _system(
        mol,
        positions,
        protein_coords=(direction * 2.9).view(1, 3),
        protein_atomic_numbers=torch.tensor([8]),
        interaction=interaction,
    )
    work = positions.clone().requires_grad_(True)
    energy = interaction_energy(work, system)["interaction_hydrogen_bond"]
    force = -torch.autograd.grad(energy, work)[0]
    assert bool(torch.isfinite(energy))
    assert bool(torch.isfinite(force).all())

    cancelled = positions.clone()
    cancelled[2] = torch.tensor(
        [1.0, 0.0, 0.0],
        dtype=torch.float64,
    )
    cancelled_energy = interaction_energy(cancelled, system)["interaction_hydrogen_bond"]
    assert float(cancelled_energy) == pytest.approx(0.0, abs=1e-12)

    collapsed = positions.clone()
    collapsed[0] = collapsed[1]
    collapsed_work = collapsed.requires_grad_(True)
    collapsed_energy = interaction_energy(collapsed_work, system)["interaction_hydrogen_bond"]
    collapsed_force = -torch.autograd.grad(
        collapsed_energy,
        collapsed_work,
    )[0]
    assert float(collapsed_energy.detach()) == pytest.approx(
        0.0,
        abs=1e-12,
    )
    assert bool(torch.isfinite(collapsed_force).all())


def test_soft_or_clamps_float32_roundoff_to_probability_domain() -> None:
    weights = torch.tensor(
        [[[1.000001]]],
        dtype=torch.float32,
        requires_grad=True,
    )
    energy = _site_saturated_energy(weights, epsilon=1.0)
    gradient = torch.autograd.grad(energy.sum(), weights)[0]
    assert bool(torch.isfinite(energy).all())
    assert bool(torch.isfinite(gradient).all())
    assert float(energy.detach()[0]) == pytest.approx(
        -1.0,
        abs=2e-6,
    )


def test_hydrogen_bond_trace_records_auditable_pair_geometry() -> None:
    coords, system = _hbond_fixture()
    stats = interaction_contact_stats(coords, system)
    pairs = stats["hydrogen_bond"]["ligand_donor_to_protein_acceptor"]["top_pairs"]
    assert len(pairs) == 1
    pair = pairs[0]
    assert pair["donor"]["label"] == "1:N"
    assert pair["acceptor"]["label"] == "A:ASN1:OD1"
    assert pair["distance_angstrom"] == pytest.approx(2.9)
    assert pair["radial_gate"] == pytest.approx(1.0)
    assert pair["donor_cone_gate"] == pytest.approx(1.0)
    assert pair["acceptor_cone_gate"] == pytest.approx(1.0)
    assert pair["weight"] == pytest.approx(1.0)

    misaligned_system = replace(
        system,
        interaction_topology=replace(
            system.interaction_topology,
            protein_outward_direction=torch.tensor(
                [[1.0, 0.0, 0.0]],
                dtype=torch.float64,
            ),
        ),
    )
    misaligned_stats = interaction_contact_stats(
        coords,
        misaligned_system,
    )
    direction = misaligned_stats["hydrogen_bond"]["ligand_donor_to_protein_acceptor"]
    assert direction["top_pairs"] == []
    assert len(direction["top_radial_candidates"]) == 1
    assert direction["top_radial_candidates"][0]["weight"] == pytest.approx(
        0.0,
        abs=1e-12,
    )


def test_interaction_batch_matches_pose_loop() -> None:
    coords, system = _hbond_fixture(protein_distance=3.2)
    second = coords.clone()
    second[0] = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    batch = torch.stack((coords, second))
    batched = interaction_energy(batch, system)
    loop = [interaction_energy(pose, system) for pose in batch]
    for name in batched:
        torch.testing.assert_close(
            batched[name],
            torch.stack([pose[name] for pose in loop]),
        )


def test_hydrophobic_soft_or_saturates_duplicate_receptor_atoms() -> None:
    coords = torch.tensor(
        [[0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]],
        dtype=torch.float64,
    )
    mol = _molecule("CC", coords)

    def make_system(protein_atoms: int) -> PhysicalSystem:
        interaction = InteractionTopology(
            ligand_neighbor_index=torch.tensor(
                [[0, 1], [1, 0]],
                dtype=torch.long,
            ),
            ligand_direction_target_cosine=torch.ones(
                2,
                dtype=torch.float64,
            ),
            ligand_direction_geometry_valid=torch.ones(
                2,
                dtype=torch.bool,
            ),
            ligand_is_donor=torch.zeros(2, dtype=torch.bool),
            ligand_is_acceptor=torch.zeros(2, dtype=torch.bool),
            ligand_is_hydrophobe=torch.tensor([True, False]),
            ligand_is_geometry_excluded_hbond_site=torch.zeros(
                2,
                dtype=torch.bool,
            ),
            protein_is_donor=torch.zeros(protein_atoms, dtype=torch.bool),
            protein_is_acceptor=torch.zeros(protein_atoms, dtype=torch.bool),
            protein_is_hydrophobe=torch.ones(
                protein_atoms,
                dtype=torch.bool,
            ),
            protein_outward_direction=torch.zeros(
                (protein_atoms, 3),
                dtype=torch.float64,
            ),
            protein_direction_target_cosine=torch.zeros(
                protein_atoms,
                dtype=torch.float64,
            ),
            protein_direction_quality=torch.zeros(
                protein_atoms,
                dtype=torch.float64,
            ),
            protein_direction_valid=torch.zeros(
                protein_atoms,
                dtype=torch.bool,
            ),
            protein_is_ambiguous_histidine=torch.zeros(
                protein_atoms,
                dtype=torch.bool,
            ),
            protein_is_unsupported_variant=torch.zeros(
                protein_atoms,
                dtype=torch.bool,
            ),
            protein_is_geometry_excluded_hbond_site=torch.zeros(
                protein_atoms,
                dtype=torch.bool,
            ),
            ligand_atom_labels=("0:C", "1:C"),
            protein_atom_labels=tuple(f"A:ALA1:C{index}" for index in range(protein_atoms)),
        )
        return _system(
            mol,
            coords,
            protein_coords=torch.tensor(
                [[3.0, 0.0, 0.0]] * protein_atoms,
                dtype=torch.float64,
            ),
            protein_atomic_numbers=torch.full(
                (protein_atoms,),
                6,
                dtype=torch.long,
            ),
            interaction=interaction,
        )

    single = interaction_energy(
        coords,
        make_system(1),
    )["interaction_hydrophobic"]
    duplicated = interaction_energy(
        coords,
        make_system(8),
    )["interaction_hydrophobic"]
    assert float(single) == pytest.approx(-0.25, abs=1e-6)
    assert float(duplicated) == pytest.approx(float(single), abs=1e-6)

    boundary_system = replace(
        make_system(1),
        protein_coords=torch.tensor(
            [[4.5, 0.0, 0.0]],
            dtype=torch.float64,
        ),
    )
    boundary_coords = coords.clone().requires_grad_(True)
    boundary = interaction_energy(
        boundary_coords,
        boundary_system,
    )["interaction_hydrophobic"]
    boundary_force = -torch.autograd.grad(boundary, boundary_coords)[0]
    assert float(boundary.detach()) == pytest.approx(0.0, abs=1e-12)
    torch.testing.assert_close(
        boundary_force,
        torch.zeros_like(boundary_force),
        atol=1e-10,
        rtol=0.0,
    )


def test_screened_formal_charge_has_declared_sign_and_value() -> None:
    coords, system = _charge_fixture()
    config = InteractionEnergyConfig(
        active_terms=("screened_formal_charge",),
    )
    attractive = interaction_energy(
        coords,
        system,
        config,
    )["interaction_screened_formal_charge"]
    rho = (4.0**2 + config.formal_charge_softcore**2) ** 0.5
    expected = (
        -config.formal_charge_coulomb_constant
        / config.formal_charge_relative_dielectric
        * math.exp(-config.formal_charge_screening_kappa * rho)
        / rho
    )
    assert float(attractive) == pytest.approx(expected, rel=1e-12)

    repulsive_system = replace(
        system,
        interaction_topology=replace(
            system.interaction_topology,
            protein_charge_site_charge=torch.tensor(
                [1.0],
                dtype=torch.float64,
            ),
        ),
    )
    repulsive = interaction_energy(
        coords,
        repulsive_system,
        config,
    )["interaction_screened_formal_charge"]
    assert float(repulsive) == pytest.approx(-expected, rel=1e-12)


def test_screened_formal_charge_is_zero_for_neutral_ligand_and_at_cutoff() -> None:
    coords, system = _charge_fixture()
    config = InteractionEnergyConfig(
        active_terms=("screened_formal_charge",),
    )
    neutral_topology = replace(
        system.interaction_topology,
        ligand_charge_site_membership=torch.empty(
            (0, coords.shape[0]),
            dtype=torch.float64,
        ),
        ligand_charge_site_charge=torch.empty(0, dtype=torch.float64),
        ligand_charge_site_labels=(),
    )
    neutral_system = replace(
        system,
        interaction_topology=neutral_topology,
    )
    neutral_coords = coords.clone().requires_grad_(True)
    neutral_energy = interaction_energy(
        neutral_coords,
        neutral_system,
        config,
    )["interaction_screened_formal_charge"]
    neutral_force = -torch.autograd.grad(
        neutral_energy,
        neutral_coords,
    )[0]
    assert float(neutral_energy.detach()) == pytest.approx(0.0, abs=0.0)
    torch.testing.assert_close(
        neutral_force,
        torch.zeros_like(neutral_force),
        atol=0.0,
        rtol=0.0,
    )

    cutoff_coords, cutoff_system = _charge_fixture(
        protein_distance=config.formal_charge_cutoff,
    )
    cutoff_coords = cutoff_coords.clone().requires_grad_(True)
    cutoff_energy = interaction_energy(
        cutoff_coords,
        cutoff_system,
        config,
    )["interaction_screened_formal_charge"]
    cutoff_gradient = torch.autograd.grad(
        cutoff_energy,
        cutoff_coords,
        create_graph=True,
    )[0]
    cutoff_curvature = torch.autograd.grad(
        cutoff_gradient[1, 0],
        cutoff_coords,
    )[0]
    cutoff_force = -cutoff_gradient
    assert float(cutoff_energy.detach()) == pytest.approx(0.0, abs=1e-14)
    torch.testing.assert_close(
        cutoff_force,
        torch.zeros_like(cutoff_force),
        atol=1e-12,
        rtol=0.0,
    )
    torch.testing.assert_close(
        cutoff_curvature,
        torch.zeros_like(cutoff_curvature),
        atol=1e-11,
        rtol=0.0,
    )


def test_screened_formal_charge_switch_is_c2_at_both_boundaries() -> None:
    config = InteractionEnergyConfig(
        active_terms=("screened_formal_charge",),
    )
    for distance, expected in (
        (config.formal_charge_switch_distance, 1.0),
        (config.formal_charge_cutoff, 0.0),
    ):
        radius = torch.tensor(
            distance,
            dtype=torch.float64,
            requires_grad=True,
        )
        value = _squared_decreasing_switch(
            radius.square(),
            config.formal_charge_switch_distance,
            config.formal_charge_cutoff,
        )
        first = torch.autograd.grad(
            value,
            radius,
            create_graph=True,
        )[0]
        second = torch.autograd.grad(
            first,
            radius,
        )[0]
        assert float(value.detach()) == pytest.approx(expected, abs=1e-14)
        assert float(first.detach()) == pytest.approx(0.0, abs=1e-13)
        assert float(second.detach()) == pytest.approx(0.0, abs=1e-12)


def test_screened_formal_charge_overlap_and_group_gradient_are_finite() -> None:
    coords, system = _charge_fixture(protein_distance=0.0)
    config = InteractionEnergyConfig(
        active_terms=("screened_formal_charge",),
    )
    work = coords.clone().requires_grad_(True)
    energy = interaction_energy(
        work,
        system,
        config,
    )["interaction_screened_formal_charge"]
    force = -torch.autograd.grad(energy, work)[0]
    assert bool(torch.isfinite(energy))
    assert bool(torch.isfinite(force).all())

    coords, system = _charge_fixture(protein_distance=4.0)
    topology = replace(
        system.interaction_topology,
        ligand_charge_site_membership=torch.tensor(
            [[0.25, 0.75]],
            dtype=torch.float64,
        ),
    )
    system = replace(system, interaction_topology=topology)
    work = coords.clone().requires_grad_(True)
    energy = interaction_energy(
        work,
        system,
        config,
    )["interaction_screened_formal_charge"]
    force = -torch.autograd.grad(energy, work)[0]
    total_force = force.sum(dim=0)
    torch.testing.assert_close(
        force[0],
        0.25 * total_force,
        atol=1e-12,
        rtol=1e-12,
    )
    torch.testing.assert_close(
        force[1],
        0.75 * total_force,
        atol=1e-12,
        rtol=1e-12,
    )


def test_screened_formal_charge_trace_and_se3_equivariance() -> None:
    coords, system = _charge_fixture(protein_distance=5.0)
    config = InteractionEnergyConfig(
        active_terms=("screened_formal_charge",),
    )
    work = coords.clone().requires_grad_(True)
    energy = interaction_energy(
        work,
        system,
        config,
    )["interaction_screened_formal_charge"]
    force = -torch.autograd.grad(energy, work)[0]

    rotation = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float64,
    )
    offset = torch.tensor([1.0, -7.0, 3.0], dtype=torch.float64)
    transformed_system = replace(
        system,
        protein_coords=system.protein_coords @ rotation.T + offset,
    )
    transformed_coords = (coords @ rotation.T + offset).clone().requires_grad_(True)
    transformed_energy = interaction_energy(
        transformed_coords,
        transformed_system,
        config,
    )["interaction_screened_formal_charge"]
    transformed_force = -torch.autograd.grad(
        transformed_energy,
        transformed_coords,
    )[0]
    torch.testing.assert_close(
        transformed_energy,
        energy,
        atol=1e-12,
        rtol=1e-12,
    )
    torch.testing.assert_close(
        transformed_force,
        force @ rotation.T,
        atol=1e-11,
        rtol=1e-11,
    )

    stats = interaction_contact_stats(coords, system, config)["screened_formal_charge"]
    assert stats["eligibility"] == "eligible"
    assert stats["attractive_pairs"] == 1
    assert stats["repulsive_pairs"] == 0
    assert stats["ligand_sites"] == [
        {
            "index": 0,
            "label": "ligand:N+",
            "charge_e": 1.0,
            "members": [
                {
                    "atom_index": 1,
                    "atom_label": "1:N",
                    "weight": 1.0,
                }
            ],
        }
    ]
    assert stats["protein_sites"][0]["charge_e"] == -1.0
    pair = stats["top_attractive_pairs"][0]
    assert pair["charge_product_e2"] == pytest.approx(-1.0)
    assert pair["distance_angstrom"] == pytest.approx(5.0)
    assert pair["energy_kcal_mol"] < 0
    assert stats["total_energy_kcal_mol"] == pytest.approx(
        stats["attractive_energy_kcal_mol"] + stats["repulsive_energy_kcal_mol"]
    )

    repulsive_system = replace(
        system,
        interaction_topology=replace(
            system.interaction_topology,
            protein_charge_site_charge=torch.tensor(
                [1.0],
                dtype=torch.float64,
            ),
        ),
    )
    repulsive_stats = interaction_contact_stats(
        coords,
        repulsive_system,
        config,
    )["screened_formal_charge"]
    assert repulsive_stats["attractive_pairs"] == 0
    assert repulsive_stats["repulsive_pairs"] == 1
    assert repulsive_stats["repulsive_energy_kcal_mol"] > 0
    assert repulsive_stats["top_repulsive_pairs"][0]["energy_kcal_mol"] > 0
    assert (
        system.interaction_topology.reference_sha256()
        != replace(
            system.interaction_topology,
            ligand_charge_site_charge=-system.interaction_topology.ligand_charge_site_charge,
        ).reference_sha256()
    )


def test_screened_formal_charge_matches_finite_difference_and_batch_loop() -> None:
    coords, system = _charge_fixture(protein_distance=4.0)
    config = InteractionEnergyConfig(
        active_terms=("screened_formal_charge",),
    )
    work = coords.clone().requires_grad_(True)
    energy = interaction_energy(
        work,
        system,
        config,
    )["interaction_screened_formal_charge"]
    gradient = torch.autograd.grad(energy, work)[0]

    step = 1e-5
    plus = coords.clone()
    minus = coords.clone()
    plus[1, 0] += step
    minus[1, 0] -= step
    finite_difference = (
        interaction_energy(plus, system, config)["interaction_screened_formal_charge"]
        - interaction_energy(minus, system, config)["interaction_screened_formal_charge"]
    ) / (2.0 * step)
    torch.testing.assert_close(
        gradient[1, 0],
        finite_difference,
        atol=1e-10,
        rtol=2e-9,
    )

    shifted = coords + torch.tensor(
        [0.35, -0.2, 0.1],
        dtype=torch.float64,
    )
    batch = torch.stack((coords, shifted))
    batched_energy = interaction_energy(
        batch,
        system,
        config,
    )["interaction_screened_formal_charge"]
    loop_energy = torch.stack(
        [
            interaction_energy(pose, system, config)["interaction_screened_formal_charge"]
            for pose in batch
        ]
    )
    torch.testing.assert_close(
        batched_energy,
        loop_energy,
        atol=1e-13,
        rtol=1e-13,
    )


def test_screened_formal_charge_large_offset_and_float32_agree() -> None:
    coords, system = _charge_fixture(protein_distance=5.0)
    config = InteractionEnergyConfig(
        active_terms=("screened_formal_charge",),
    )
    work64 = coords.clone().requires_grad_(True)
    energy64 = interaction_energy(
        work64,
        system,
        config,
    )["interaction_screened_formal_charge"]
    force64 = -torch.autograd.grad(energy64, work64)[0]

    offset = torch.tensor(
        [1.0e6, -2.0e6, 3.0e6],
        dtype=torch.float64,
    )
    offset_system = replace(
        system,
        protein_coords=system.protein_coords + offset,
    )
    offset_coords = (coords + offset).requires_grad_(True)
    offset_energy = interaction_energy(
        offset_coords,
        offset_system,
        config,
    )["interaction_screened_formal_charge"]
    offset_force = -torch.autograd.grad(offset_energy, offset_coords)[0]
    assert bool(torch.isfinite(offset_energy))
    assert bool(torch.isfinite(offset_force).all())
    torch.testing.assert_close(
        offset_energy,
        energy64,
        atol=1e-12,
        rtol=1e-12,
    )
    torch.testing.assert_close(
        offset_force,
        force64,
        atol=1e-12,
        rtol=1e-12,
    )

    system32 = system.to(torch.device("cpu"), torch.float32)
    work32 = coords.to(torch.float32).requires_grad_(True)
    energy32 = interaction_energy(
        work32,
        system32,
        config,
    )["interaction_screened_formal_charge"]
    force32 = -torch.autograd.grad(energy32, work32)[0]
    assert energy32.dtype == torch.float32
    assert force32.dtype == torch.float32
    torch.testing.assert_close(
        energy32.to(torch.float64),
        energy64,
        atol=2e-7,
        rtol=2e-6,
    )
    torch.testing.assert_close(
        force32.to(torch.float64),
        force64,
        atol=2e-7,
        rtol=2e-5,
    )


def test_interaction_force_is_energy_gradient_and_se3_equivariant() -> None:
    coords, system = _hbond_fixture(protein_distance=3.2)
    work = coords.clone().requires_grad_(True)
    energy = interaction_energy(work, system)["total"]
    force = -torch.autograd.grad(energy, work)[0]
    assert bool(torch.isfinite(force).all())

    direction = torch.tensor(
        [[0.2, -0.4, 0.3], [-0.1, 0.5, -0.2]],
        dtype=torch.float64,
    )
    direction = direction / direction.norm()
    step = 1e-5
    energy_plus = interaction_energy(
        coords + step * direction,
        system,
    )["total"]
    energy_minus = interaction_energy(
        coords - step * direction,
        system,
    )["total"]
    finite_difference = (energy_plus - energy_minus) / (2.0 * step)
    torch.testing.assert_close(
        finite_difference,
        -(force * direction).sum(),
        atol=2e-6,
        rtol=2e-5,
    )

    rotation = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float64,
    )
    offset = torch.tensor([4.0, -3.0, 2.0], dtype=torch.float64)
    transformed_topology = replace(
        system.interaction_topology,
        protein_outward_direction=(
            system.interaction_topology.protein_outward_direction @ rotation.T
        ),
    )
    transformed_system = replace(
        system,
        protein_coords=system.protein_coords @ rotation.T + offset,
        interaction_topology=transformed_topology,
    )
    transformed_coords = (coords @ rotation.T + offset).clone().requires_grad_(True)
    transformed_energy = interaction_energy(
        transformed_coords,
        transformed_system,
    )["total"]
    transformed_force = -torch.autograd.grad(
        transformed_energy,
        transformed_coords,
    )[0]
    torch.testing.assert_close(
        transformed_energy,
        energy.detach(),
        atol=1e-12,
        rtol=1e-12,
    )
    torch.testing.assert_close(
        transformed_force,
        force @ rotation.T,
        atol=1e-10,
        rtol=1e-10,
    )


def test_guidance_total_is_physical_plus_interaction() -> None:
    coords, system = _hbond_fixture(protein_distance=3.2)
    physical = physical_energy(coords, system)
    interaction = interaction_energy(coords, system)
    combined = guidance_energy(coords, system)
    torch.testing.assert_close(
        combined["total"],
        physical["total"] + interaction["total"],
    )
    torch.testing.assert_close(
        combined["interaction_hydrogen_bond"],
        interaction["interaction_hydrogen_bond"],
    )
    torch.testing.assert_close(
        combined["interaction_screened_formal_charge"],
        interaction["interaction_screened_formal_charge"],
    )


def test_active_interaction_rejects_missing_typing() -> None:
    coords, system = _hbond_fixture()
    missing = replace(
        system,
        interaction_topology=None,
        interaction_parameter_set=None,
    )
    with pytest.raises(ValueError, match="require system interaction typing"):
        interaction_energy(
            coords,
            missing,
            InteractionEnergyConfig(),
        )
    disabled = interaction_energy(
        coords,
        missing,
        InteractionEnergyConfig(active_terms=()),
    )
    assert set(disabled) == {"total"}
    assert float(disabled["total"]) == pytest.approx(0.0)


def test_custom_baseline_path_does_not_build_inactive_interaction_grids(
    monkeypatch,
) -> None:
    coords, system = _hbond_fixture()

    def inactive_called(*_args, **_kwargs):
        raise AssertionError("inactive interaction family was evaluated")

    monkeypatch.setattr(interaction_module, "_ring_geometry", inactive_called)
    monkeypatch.setattr(interaction_module, "_halogen_bond_components", inactive_called)
    monkeypatch.setattr(interaction_module, "_metal_components", inactive_called)

    baseline = InteractionEnergyConfig(
        active_terms=(
            "hydrophobic",
            "hydrogen_bond",
            "screened_formal_charge",
        )
    )
    components = interaction_energy(coords, system, baseline)
    assert set(components) == {
        "interaction_hydrophobic",
        "interaction_hydrogen_bond",
        "interaction_screened_formal_charge",
        "total",
    }


@pytest.mark.parametrize("bad_value", [math.nan, math.inf, -math.inf])
def test_every_numeric_interaction_config_field_must_be_finite(
    bad_value: float,
) -> None:
    for item in fields(InteractionEnergyConfig):
        if item.name == "active_terms":
            continue
        with pytest.raises(ValueError, match=f"{item.name} must be finite"):
            InteractionEnergyConfig(**{item.name: bad_value})
