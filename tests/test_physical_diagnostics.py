from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

Chem = pytest.importorskip("rdkit.Chem")

from effdock.geometry.flow_matching import sample_prior_poses  # noqa: E402
from effdock.geometry.se3 import quaternion_to_matrix  # noqa: E402
from effdock.guidance.diagnostics import (  # noqa: E402
    fragment_centers,
    make_crystal_perturbations,
    trace_physical_pose,
)
from effdock.guidance.errors import UnsupportedPhysicalChemistryError  # noqa: E402
from effdock.guidance.interaction import (  # noqa: E402
    InteractionEnergyConfig,
    interaction_energy,
    interaction_profile_metadata,
    metal_coordination_v0_contract,
    metal_coordination_v1_contract,
)
from effdock.guidance.parameterization import (  # noqa: E402
    element_parameters,
    guidance_parameter_identity,
)
from effdock.guidance.physical import PhysicalEnergyConfig, physical_energy  # noqa: E402
from effdock.guidance.runtime import project_atom_forces  # noqa: E402
from effdock.guidance.system import (  # noqa: E402
    InteractionTopology,
    PhysicalSystem,
    _protein_valence_geometry,
    build_physical_system,
)
from effdock.guidance.topology import build_physical_topology  # noqa: E402
from effdock.preprocess.ligand import ligand_graph_identity  # noqa: E402
from effdock.workflows import trace_physical  # noqa: E402
from effdock.workflows.relax_guidance import (  # noqa: E402
    RelaxationRun,
    RigidRelaxationConfig,
    _relaxation_energy,
    _success_assessment,
    load_explicit_pocket_center,
    make_crystal_local_fragment_batch,
    make_pocket_prior_fragment_batch,
    make_pocket_prior_fragment_pose,
    make_torn_fragment_pose,
    relax_rigid_fragments,
    relax_rigid_fragments_batch,
)


def _butane_molecule() -> tuple["Chem.Mol", torch.Tensor]:
    mol = Chem.MolFromSmiles("CCCC")
    conformer = Chem.Conformer(4)
    positions = torch.tensor(
        [
            [0.00, 0.00, 0.00],
            [1.54, 0.00, 0.00],
            [2.10, 1.43, 0.00],
            [3.45, 1.70, 0.75],
        ],
        dtype=torch.float64,
    )
    for atom_index, position in enumerate(positions.tolist()):
        conformer.SetAtomPosition(atom_index, position)
    conformer.Set3D(True)
    mol.AddConformer(conformer)
    return mol, positions


def _butane_system() -> tuple[torch.Tensor, PhysicalSystem]:
    mol, positions = _butane_molecule()
    topology = build_physical_topology(
        mol,
        torch.tensor([0, 0, 1, 1]),
    ).to(torch.device("cpu"), torch.float64)
    protein_atomic_numbers = torch.tensor([6, 8])
    protein_parameters = element_parameters(
        protein_atomic_numbers,
        dtype=torch.float64,
    )
    protein_coords = torch.tensor(
        [[0.0, 4.0, 0.0], [4.0, 4.0, 0.0]],
        dtype=torch.float64,
    )
    system = PhysicalSystem(
        topology=topology,
        protein_coords=protein_coords,
        protein_atomic_numbers=protein_atomic_numbers,
        protein_uff_x=protein_parameters.uff_x,
        protein_uff_d=protein_parameters.uff_d,
        protein_vdw_radius=protein_parameters.vdw_radius,
        parameter_set={"name": "test", "version": "test"},
        protein_source_atoms=2,
        interaction_topology=InteractionTopology(
            ligand_neighbor_index=torch.tensor(
                [[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]],
                dtype=torch.long,
            ),
            ligand_direction_target_cosine=torch.ones(
                4,
                dtype=torch.float64,
            ),
            ligand_direction_geometry_valid=torch.ones(
                4,
                dtype=torch.bool,
            ),
            ligand_is_donor=torch.zeros(4, dtype=torch.bool),
            ligand_is_acceptor=torch.zeros(4, dtype=torch.bool),
            ligand_is_hydrophobe=torch.zeros(4, dtype=torch.bool),
            ligand_is_geometry_excluded_hbond_site=torch.zeros(
                4,
                dtype=torch.bool,
            ),
            protein_is_donor=torch.zeros(2, dtype=torch.bool),
            protein_is_acceptor=torch.zeros(2, dtype=torch.bool),
            protein_is_hydrophobe=torch.zeros(2, dtype=torch.bool),
            protein_outward_direction=torch.zeros(
                (2, 3),
                dtype=torch.float64,
            ),
            protein_direction_target_cosine=torch.zeros(
                2,
                dtype=torch.float64,
            ),
            protein_direction_quality=torch.zeros(
                2,
                dtype=torch.float64,
            ),
            protein_direction_valid=torch.zeros(2, dtype=torch.bool),
            protein_is_ambiguous_histidine=torch.zeros(
                2,
                dtype=torch.bool,
            ),
            protein_is_unsupported_variant=torch.zeros(
                2,
                dtype=torch.bool,
            ),
            protein_is_geometry_excluded_hbond_site=torch.zeros(
                2,
                dtype=torch.bool,
            ),
            ligand_atom_labels=("0:C", "1:C", "2:C", "3:C"),
            protein_atom_labels=("A:ALA1:CA", "A:ALA1:O"),
        ),
        interaction_parameter_set={"name": "test", "version": "test"},
    )
    return positions, system


def test_fragment_tear_is_deterministic_rigid_and_mass_com_preserving() -> None:
    crystal, system = _butane_system()
    fragment_id = system.topology.fragment_id
    torn, displacement = make_torn_fragment_pose(
        crystal,
        fragment_id,
        system.topology.mass,
        distance_angstrom=3.0,
        seed=20260730,
    )
    repeated, repeated_displacement = make_torn_fragment_pose(
        crystal,
        fragment_id,
        system.topology.mass,
        distance_angstrom=3.0,
        seed=20260730,
    )

    torch.testing.assert_close(torn, repeated)
    torch.testing.assert_close(displacement, repeated_displacement)
    torch.testing.assert_close(
        displacement.norm(dim=-1).max(),
        torch.tensor(3.0, dtype=torch.float64),
    )
    fragment_mass = torch.zeros(
        displacement.shape[0],
        dtype=torch.float64,
    )
    fragment_mass.index_add_(
        0,
        fragment_id,
        system.topology.mass,
    )
    torch.testing.assert_close(
        (fragment_mass[:, None] * displacement).sum(dim=0),
        torch.zeros(3, dtype=torch.float64),
        atol=1e-12,
        rtol=0.0,
    )
    for fragment in range(displacement.shape[0]):
        mask = fragment_id == fragment
        torch.testing.assert_close(
            torch.pdist(torn[mask]),
            torch.pdist(crystal[mask]),
            atol=1e-12,
            rtol=1e-12,
        )


def _butane_fragment_local_coordinates(
    crystal: torch.Tensor,
    fragment_id: torch.Tensor,
) -> torch.Tensor:
    centers = fragment_centers(crystal, fragment_id)[0]
    return crystal - centers[fragment_id]


def test_model_prior_matches_sampler_float32_draw_order_and_t0_pose() -> None:
    crystal, system = _butane_system()
    fragment_id = system.topology.fragment_id
    local = _butane_fragment_local_coordinates(crystal, fragment_id)
    pocket_center = torch.tensor([4.25, -1.5, 8.0])
    sigma = 0.5
    seed = 731

    initial, metadata = make_pocket_prior_fragment_pose(
        local,
        fragment_id,
        pocket_center,
        sigma_angstrom=sigma,
        seed=seed,
        rotation_mode="uniform",
    )

    frag_sizes = torch.bincount(fragment_id)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    expected_translation, expected_quaternion = sample_prior_poses(
        int(frag_sizes.numel()),
        torch.zeros(3, dtype=torch.float32),
        sigma,
        frag_sizes=frag_sizes,
        generator=generator,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    expected_centered = (
        torch.einsum(
            "nij,nj->ni",
            quaternion_to_matrix(expected_quaternion)[fragment_id],
            local.to(torch.float32),
        )
        + expected_translation[fragment_id]
    )
    expected_absolute = expected_centered + pocket_center

    assert torch.equal(
        metadata["translation_relative_to_pocket_angstrom"],
        expected_translation,
    )
    assert torch.equal(
        metadata["quaternion_scalar_first"],
        expected_quaternion,
    )
    assert torch.equal(initial, expected_absolute)
    assert metadata["exact_active_sampler_model_prior"] is True


def test_model_prior_pairs_share_eps_and_quaternion_across_sigma() -> None:
    crystal, system = _butane_system()
    fragment_id = system.topology.fragment_id
    local = _butane_fragment_local_coordinates(crystal, fragment_id)
    center = torch.tensor([1.0, 2.0, 3.0])

    _, narrow = make_pocket_prior_fragment_pose(
        local,
        fragment_id,
        center,
        sigma_angstrom=0.5,
        seed=20260730,
        rotation_mode="uniform",
    )
    _, wide = make_pocket_prior_fragment_pose(
        local,
        fragment_id,
        center,
        sigma_angstrom=3.0,
        seed=20260730,
        rotation_mode="uniform",
    )

    assert torch.equal(
        narrow["standard_normal_translation_eps"],
        wide["standard_normal_translation_eps"],
    )
    assert torch.equal(
        narrow["quaternion_scalar_first"],
        wide["quaternion_scalar_first"],
    )
    torch.testing.assert_close(
        wide["translation_relative_to_pocket_angstrom"],
        6.0 * narrow["translation_relative_to_pocket_angstrom"],
        atol=0.0,
        rtol=0.0,
    )


def test_pocket_gaussian_identity_control_and_single_atom_mask() -> None:
    fragment_id = torch.tensor([0, 0, 1])
    local = torch.tensor([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.0, 0.0]])
    center = torch.tensor([2.0, -3.0, 1.0])

    identity_pose, identity = make_pocket_prior_fragment_pose(
        local,
        fragment_id,
        center,
        sigma_angstrom=0.5,
        seed=19,
        rotation_mode="identity",
    )
    _, model_prior = make_pocket_prior_fragment_pose(
        local,
        fragment_id,
        center,
        sigma_angstrom=0.5,
        seed=19,
        rotation_mode="uniform",
    )

    expected_identity = torch.zeros(2, 4)
    expected_identity[:, 0] = 1.0
    assert torch.equal(
        identity["quaternion_scalar_first"],
        expected_identity,
    )
    assert torch.equal(
        identity["translation_relative_to_pocket_angstrom"],
        model_prior["translation_relative_to_pocket_angstrom"],
    )
    assert torch.equal(
        model_prior["quaternion_scalar_first"][1],
        expected_identity[1],
    )
    assert identity["exact_active_sampler_model_prior"] is False
    expected_pose = (
        local + identity["translation_relative_to_pocket_angstrom"][fragment_id] + center
    )
    assert torch.equal(identity_pose, expected_pose)


def test_explicit_pocket_center_missing_key_fails(tmp_path: Path) -> None:
    path = tmp_path / "centers.json"
    path.write_text(json.dumps({"other": [1.0, 2.0, 3.0]}))

    with pytest.raises(KeyError, match="1g9v"):
        load_explicit_pocket_center(
            path,
            sample_id="1G9V_RQ3",
        )


def test_explicit_pocket_center_records_source_and_default_key(
    tmp_path: Path,
) -> None:
    path = tmp_path / "centers.json"
    path.write_text(
        json.dumps(
            {
                "1g9v": {
                    "pocket_center": [1.25, -2.5, 4.0],
                    "definition": "reference_ligand_residue_center",
                    "reference_ligand_sha256": "abc123",
                }
            }
        )
    )

    center, provenance = load_explicit_pocket_center(
        path,
        sample_id="1G9V_RQ3",
    )

    torch.testing.assert_close(
        center,
        torch.tensor([1.25, -2.5, 4.0]),
    )
    assert provenance["source_path"] == str(path.resolve())
    assert len(provenance["source_sha256"]) == 64
    assert provenance["selected_key"] == "1g9v"
    assert provenance["definition"] == "reference_ligand_residue_center"
    assert provenance["reference_ligand_derived"] is True
    assert provenance["derived_from_crystal"] is True
    assert provenance["derived_at_runtime_from_input_ligand"] is False
    assert provenance["source_entry_metadata"]["reference_ligand_sha256"] == ("abc123")


def test_pocket_prior_requires_center_and_full_shell() -> None:
    crystal, system = _butane_system()
    with pytest.raises(ValueError, match="at least 18 A"):
        RigidRelaxationConfig(
            initialization_mode="model_prior",
            protein_shell_cutoff_angstrom=17.9,
        )

    with pytest.raises(ValueError, match="pocket_center is required"):
        relax_rigid_fragments(
            crystal,
            crystal,
            system,
            config=RigidRelaxationConfig(
                initialization_mode="model_prior",
                max_steps=1,
                save_every=1,
                protein_shell_cutoff_angstrom=18.0,
            ),
            mode="physical_only",
        )


def test_pocket_prior_shell_records_radius_and_invalidates_run() -> None:
    crystal, system = _butane_system()
    initial = crystal + torch.tensor([11.0, 0.0, 0.0])
    run = relax_rigid_fragments(
        crystal,
        initial,
        system,
        config=RigidRelaxationConfig(
            initialization_mode="pocket_gaussian",
            max_steps=1,
            save_every=1,
            protein_shell_cutoff_angstrom=18.0,
        ),
        mode="physical_only",
        pocket_center=torch.zeros(3),
    )

    assert run.shell_envelope_valid is False
    for row in run.metrics:
        assert row["maximum_ligand_atom_radius_from_pocket_center_angstrom"] > 10.0
        assert row["fixed_shell_valid_radius_angstrom"] == pytest.approx(8.0)
        assert row["within_fixed_shell_valid_radius"] is False
        assert "ligand_mass_com_distance_to_pocket_center_angstrom" in row
        assert "ligand_mass_com_distance_to_crystal_mass_com_angstrom" in row


def test_pocket_prior_success_uses_exact_frozen_joint_gate() -> None:
    run = RelaxationRun(
        mode="unified",
        status="max_steps",
        metrics=[
            {
                "step": 0,
                "raw_rmsd_angstrom": 2.0,
                "cut_bond_max_abs_error_angstrom": 1.0,
                "minimum_distance_over_uff_x": 0.4,
            },
            {
                "step": 1,
                "raw_rmsd_angstrom": 1.5,
                "cut_bond_max_abs_error_angstrom": 0.2,
                "minimum_distance_over_uff_x": 0.65,
            },
        ],
        frames=[],
        saved_steps=[],
        total_backtracks=0,
        shell_envelope_valid=True,
    )

    assessment = _success_assessment(
        run,
        crystal_minimum_distance_over_uff_x=1.0,
        initialization_mode="model_prior",
    )

    assert assessment["raw_rmsd_reduction_fraction"] == pytest.approx(0.25)
    assert assessment["gates"]["raw_rmsd_reduction_ge_70_percent"] is False
    assert assessment["gates"]["final_clash_not_worse_than_crystal_minus_0_05"] is False
    assert assessment["primary_joint_success"] is True
    assert assessment["classification"] == "success"

    run.metrics[-1]["raw_rmsd_angstrom"] = 2.0
    boundary = _success_assessment(
        run,
        crystal_minimum_distance_over_uff_x=1.0,
        initialization_mode="model_prior",
    )
    assert boundary["primary_joint_success"] is False
    assert boundary["pose_recovery"] is False

    run.status = "line_search_failed"
    numerical_failure = _success_assessment(
        run,
        crystal_minimum_distance_over_uff_x=1.0,
        initialization_mode="model_prior",
    )
    assert numerical_failure["gates"]["finite_and_completed"] is False
    assert numerical_failure["primary_joint_success"] is False


def test_local_prior_success_uses_one_angstrom_threshold() -> None:
    run = RelaxationRun(
        mode="guarded_interaction",
        status="max_steps",
        metrics=[
            {
                "step": 0,
                "raw_rmsd_angstrom": 1.5,
                "cut_bond_max_abs_error_angstrom": 0.0,
                "minimum_distance_over_uff_x": 0.8,
            },
            {
                "step": 1,
                "raw_rmsd_angstrom": 1.1,
                "cut_bond_max_abs_error_angstrom": 0.0,
                "minimum_distance_over_uff_x": 0.8,
            },
        ],
        frames=[],
        saved_steps=[],
        total_backtracks=0,
        shell_envelope_valid=True,
    )
    assessment = _success_assessment(
        run,
        crystal_minimum_distance_over_uff_x=0.8,
        initialization_mode="model_prior",
        pose_threshold_angstrom=1.0,
    )

    assert assessment["pose_recovery_threshold_angstrom"] == 1.0
    assert assessment["gates"]["final_raw_rmsd_lt_1_angstrom"] is False
    assert assessment["primary_joint_success"] is False


def test_nonfinite_initial_energy_keeps_failure_metric_and_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    crystal, system = _butane_system()

    def nonfinite_energy(
        coords: torch.Tensor,
        *_args: object,
        **_kwargs: object,
    ) -> dict[str, torch.Tensor]:
        return {"total": coords.sum() * 0.0 + torch.tensor(float("nan"))}

    monkeypatch.setattr(
        "effdock.workflows.relax_guidance.guidance_energy",
        nonfinite_energy,
    )
    run = relax_rigid_fragments(
        crystal,
        crystal,
        system,
        config=RigidRelaxationConfig(max_steps=1, save_every=1),
        mode="physical_only",
    )

    assert run.status == "nonfinite_energy"
    assert len(run.metrics) == 1
    assert len(run.frames) == 1
    assert run.saved_steps == [0]
    assessment = _success_assessment(
        run,
        crystal_minimum_distance_over_uff_x=0.0,
        initialization_mode="crystal_tear",
    )
    assert assessment["gates"]["finite_and_completed"] is False
    assert assessment["classification"] == "failure"


def test_rigid_fragment_relaxation_descends_energy_without_deformation() -> None:
    crystal, system = _butane_system()
    torn, _ = make_torn_fragment_pose(
        crystal,
        system.topology.fragment_id,
        system.topology.mass,
        distance_angstrom=1.0,
        seed=20260730,
    )
    run = relax_rigid_fragments(
        crystal,
        torn,
        system,
        config=RigidRelaxationConfig(
            tear_distance_angstrom=1.0,
            max_steps=80,
            save_every=7,
            protein_shell_cutoff_angstrom=11.0,
        ),
        mode="physical_only",
    )

    energies = [float(row["energy_groups"]["combined"]) for row in run.metrics]
    assert run.status not in {"nonfinite_energy", "nonfinite_force"}
    assert all(right <= left + 1e-9 for left, right in zip(energies, energies[1:]))
    assert (
        run.metrics[-1]["cut_bond_max_abs_error_angstrom"]
        < run.metrics[0]["cut_bond_max_abs_error_angstrom"]
    )
    assert run.saved_steps[0] == 0
    assert run.saved_steps[-1] == run.metrics[-1]["step"]
    for row in run.metrics[1:]:
        accepted_step = row["accepted_max_atom_step_angstrom"]
        if accepted_step is not None:
            assert accepted_step <= 0.1000001
    fragment_id = system.topology.fragment_id
    for frame in run.frames:
        for fragment in range(int(fragment_id.max()) + 1):
            mask = fragment_id == fragment
            torch.testing.assert_close(
                torch.pdist(frame[mask]),
                torch.pdist(crystal[mask]),
                atol=1e-10,
                rtol=1e-10,
            )


def test_pocket_prior_batch_preserves_per_seed_sampler_draws() -> None:
    crystal, system = _butane_system()
    fragment_id = system.topology.fragment_id
    local = crystal - fragment_centers(crystal, fragment_id)[0, fragment_id]
    seeds = [20260731, 20260732]
    batched, metadata = make_pocket_prior_fragment_batch(
        local.to(torch.float32),
        fragment_id,
        torch.tensor([1.0, 2.0, 3.0]),
        sigma_angstrom=0.5,
        seeds=seeds,
        rotation_mode="uniform",
    )

    assert batched.shape == (2, 4, 3)
    assert len(metadata) == 2
    for batch_index, seed in enumerate(seeds):
        expected, expected_metadata = make_pocket_prior_fragment_pose(
            local.to(torch.float32),
            fragment_id,
            torch.tensor([1.0, 2.0, 3.0]),
            sigma_angstrom=0.5,
            seed=seed,
            rotation_mode="uniform",
        )
        torch.testing.assert_close(batched[batch_index], expected)
        torch.testing.assert_close(
            metadata[batch_index]["standard_normal_translation_eps"],
            expected_metadata["standard_normal_translation_eps"],
        )


def test_crystal_local_prior_batch_is_seeded_and_fragment_rigid() -> None:
    crystal, system = _butane_system()
    fragment_id = system.topology.fragment_id
    seeds = [20260731, 20260732]
    batched, metadata = make_crystal_local_fragment_batch(
        crystal,
        fragment_id,
        translation_sigma_angstrom=0.5,
        rotation_sigma_degrees=15.0,
        seeds=seeds,
    )
    repeated, repeated_metadata = make_crystal_local_fragment_batch(
        crystal,
        fragment_id,
        translation_sigma_angstrom=0.5,
        rotation_sigma_degrees=15.0,
        seeds=seeds,
    )

    torch.testing.assert_close(batched, repeated)
    assert len(metadata) == len(repeated_metadata) == 2
    for pose in batched:
        for fragment in range(int(fragment_id.max()) + 1):
            mask = fragment_id == fragment
            torch.testing.assert_close(
                torch.pdist(pose[mask]),
                torch.pdist(crystal[mask]),
                atol=1e-6,
                rtol=1e-6,
            )


def test_guard_objective_excludes_protein_ligand_attraction() -> None:
    crystal, system = _butane_system()
    config = PhysicalEnergyConfig()
    components = _relaxation_energy(
        crystal,
        system,
        mode="guard_only",
        physical_config=config,
        interaction_config=InteractionEnergyConfig(active_terms=()),
    )

    assert "protein_ligand_lj_repulsive" in components
    assert "protein_ligand_lj_attractive" not in components
    assert all(
        name.startswith("ligand_intra_") or name == "protein_ligand_lj_repulsive" or name == "total"
        for name in components
    )
    expected = sum(
        (value for name, value in components.items() if name != "total"),
        start=components["total"].new_zeros(()),
    )
    torch.testing.assert_close(components["total"], expected)


def test_batched_relaxation_matches_independent_pose_loops() -> None:
    crystal, system = _butane_system()
    first, _ = make_torn_fragment_pose(
        crystal,
        system.topology.fragment_id,
        system.topology.mass,
        distance_angstrom=0.75,
        seed=20260731,
    )
    second, _ = make_torn_fragment_pose(
        crystal,
        system.topology.fragment_id,
        system.topology.mass,
        distance_angstrom=0.75,
        seed=20260732,
    )
    initial = torch.stack((first, second))
    config = RigidRelaxationConfig(
        initialization_mode="crystal_tear",
        tear_distance_angstrom=0.75,
        max_steps=3,
        save_every=1,
        protein_shell_cutoff_angstrom=11.0,
    )
    batched = relax_rigid_fragments_batch(
        crystal,
        initial,
        system,
        config=config,
        mode="physical_only",
    )
    independent = [
        relax_rigid_fragments(
            crystal,
            pose,
            system,
            config=config,
            mode="physical_only",
        )
        for pose in initial
    ]

    assert batched.statuses == [run.status for run in independent]
    assert batched.total_backtracks == [run.total_backtracks for run in independent]
    assert batched.saved_steps == independent[0].saved_steps
    assert len(batched.frames) == len(independent[0].frames)
    for frame_index, frame in enumerate(batched.frames):
        for pose_index, run in enumerate(independent):
            torch.testing.assert_close(
                frame[pose_index],
                run.frames[frame_index],
                atol=1e-9,
                rtol=1e-9,
            )
    for pose_metrics in batched.metrics:
        energies = [float(row["energy_groups"]["combined"]) for row in pose_metrics]
        assert all(right <= left + 1e-9 for left, right in zip(energies, energies[1:]))


def test_sparse_batched_diagnostics_preserve_coordinates_and_solver_outcomes() -> None:
    crystal, system = _butane_system()
    poses = [
        make_torn_fragment_pose(
            crystal,
            system.topology.fragment_id,
            system.topology.mass,
            distance_angstrom=0.75,
            seed=seed,
        )[0]
        for seed in (20260731, 20260732)
    ]
    initial = torch.stack(poses)
    config = RigidRelaxationConfig(
        initialization_mode="crystal_tear",
        tear_distance_angstrom=0.75,
        max_steps=3,
        save_every=2,
        protein_shell_cutoff_angstrom=11.0,
    )
    dense = relax_rigid_fragments_batch(
        crystal,
        initial,
        system,
        config=config,
        mode="physical_only",
    )
    sparse = relax_rigid_fragments_batch(
        crystal,
        initial,
        system,
        config=config,
        mode="physical_only",
        collect_every_step_metrics=False,
        collect_contact_stats=False,
    )

    assert sparse.statuses == dense.statuses
    assert sparse.total_backtracks == dense.total_backtracks
    assert sparse.terminal_steps == dense.terminal_steps
    assert sparse.shell_envelope_valid == dense.shell_envelope_valid
    assert sparse.saved_steps == dense.saved_steps == [0, 2, 3]
    assert [[row["step"] for row in rows] for rows in sparse.metrics] == [
        [0, 2, 3],
        [0, 2, 3],
    ]
    assert [[row["step"] for row in rows] for rows in dense.metrics] == [
        [0, 1, 2, 3],
        [0, 1, 2, 3],
    ]
    for sparse_frame, dense_frame in zip(sparse.frames, dense.frames, strict=True):
        torch.testing.assert_close(sparse_frame, dense_frame, atol=1e-9, rtol=1e-9)
    for sparse_rows, dense_rows in zip(sparse.metrics, dense.metrics, strict=True):
        for sparse_row, dense_row in zip(
            sparse_rows,
            (dense_rows[0], dense_rows[2], dense_rows[3]),
            strict=True,
        ):
            assert "contacts" not in sparse_row
            assert "fragment_centers_angstrom" not in sparse_row
            assert sparse_row["energy_groups"] == dense_row["energy_groups"]
            assert sparse_row["raw_rmsd_angstrom"] == dense_row["raw_rmsd_angstrom"]


def test_batched_energy_plateau_stops_each_pose_after_minimum_and_patience() -> None:
    crystal, system = _butane_system()
    initial = torch.stack(
        [
            make_torn_fragment_pose(
                crystal,
                system.topology.fragment_id,
                system.topology.mass,
                distance_angstrom=0.75,
                seed=seed,
            )[0]
            for seed in (20260731, 20260732)
        ]
    )
    config = RigidRelaxationConfig(
        initialization_mode="crystal_tear",
        tear_distance_angstrom=0.75,
        max_steps=6,
        save_every=2,
        convergence_displacement_angstrom=1e-30,
        convergence_patience=100,
        convergence_energy_absolute_kcal_mol=1e12,
        convergence_energy_relative=0.0,
        convergence_energy_patience=2,
        convergence_energy_min_steps=2,
        protein_shell_cutoff_angstrom=11.0,
    )
    run = relax_rigid_fragments_batch(
        crystal,
        initial,
        system,
        config=config,
        mode="physical_only",
        collect_every_step_metrics=False,
        collect_contact_stats=False,
    )

    assert run.statuses == ["converged_energy_plateau"] * 2
    assert run.terminal_steps == [3, 3]
    assert run.saved_steps == [0, 2, 3]
    assert run.shell_envelope_valid == [True, True]


def _pdb_atom_line(
    record: str,
    serial: int,
    atom_name: str,
    residue: str,
    xyz: tuple[float, float, float],
    element: str,
) -> str:
    return (
        f"{record:<6}{serial:5d} {atom_name:>4s} {residue:>3s} A{1:4d}    "
        f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}"
        f"{1.0:6.2f}{20.0:6.2f}          {element:>2s}\n"
    )


def test_crystal_perturbations_target_expected_terms() -> None:
    coords, system = _butane_system()
    states = make_crystal_perturbations(
        coords,
        system,
        stretch_angstrom=0.5,
        torsion_degrees=30.0,
        overlap_distance_angstrom=0.5,
    )
    assert [state.name for state in states] == [
        "crystal",
        "cut_bond_stretch",
        "cut_bond_torsion",
        "protein_ligand_overlap",
    ]
    energies = {state.name: physical_energy(state.coords, system) for state in states}

    assert float(energies["cut_bond_stretch"]["ligand_intra_bond"]) > float(
        energies["crystal"]["ligand_intra_bond"]
    )
    torch.testing.assert_close(
        energies["cut_bond_torsion"]["ligand_intra_bond"],
        energies["crystal"]["ligand_intra_bond"],
        atol=1e-10,
        rtol=1e-10,
    )
    torch.testing.assert_close(
        energies["cut_bond_torsion"]["ligand_intra_angle"],
        energies["crystal"]["ligand_intra_angle"],
        atol=1e-10,
        rtol=1e-10,
    )
    assert not math.isclose(
        float(energies["cut_bond_torsion"]["ligand_intra_proper"]),
        float(energies["crystal"]["ligand_intra_proper"]),
        abs_tol=1e-8,
    )
    assert float(energies["protein_ligand_overlap"]["protein_ligand_lj_repulsive"]) > float(
        energies["crystal"]["protein_ligand_lj_repulsive"]
    )
    for term in (
        "ligand_intra_bond",
        "ligand_intra_angle",
        "ligand_intra_proper",
        "ligand_intra_improper",
        "ligand_intra_lj_repulsive",
        "ligand_intra_lj_attractive",
    ):
        torch.testing.assert_close(
            energies["protein_ligand_overlap"][term],
            energies["crystal"][term],
            atol=1e-9,
            rtol=1e-9,
        )


def test_trace_reports_per_term_force_and_normalized_contact() -> None:
    coords, system = _butane_system()
    row = trace_physical_pose(
        coords,
        system,
        pose_kind="crystal",
    )

    energies = row["energies"]
    component_sum = sum(value for name, value in energies.items() if name != "total")
    assert energies["total"] == pytest.approx(component_sum)
    assert row["energy_groups"]["physical"] == pytest.approx(energies["total"])
    assert row["energy_groups"]["interaction"] == pytest.approx(0.0)
    assert row["energy_groups"]["combined"] == pytest.approx(energies["total"])
    assert set(row["force_by_term"]) == set(energies)
    assert row["force_by_term"]["total"]["max_norm"] > 0
    assert row["fragment_projection"]["mass_preconditioned_translation"]["max_norm"] > 0
    assert row["protein_ligand_contacts"]["pair_count"] == 8
    assert row["protein_ligand_contacts"]["minimum_distance_over_uff_x"] > 0


def test_interaction_layer_default_enables_every_implemented_term() -> None:
    coords, system = _butane_system()
    combined_identity = guidance_parameter_identity()
    assert combined_identity["version"] == "1.6.0"
    assert combined_identity["formula_version"] == "physical-v2.2_plus_interaction-v1.6"
    components = interaction_energy(coords, system)
    assert set(components) == {
        "interaction_hydrophobic",
        "interaction_hydrogen_bond",
        "interaction_screened_formal_charge",
        "interaction_pi_stacking",
        "interaction_cation_pi",
        "interaction_halogen_bond",
        "interaction_metal_coordination",
        "total",
    }
    assert float(components["total"]) == pytest.approx(0.0)
    metadata = interaction_profile_metadata()
    assert metadata["status"] == "active_diagnostic"
    assert metadata["active_terms"] == [
        "hydrophobic",
        "hydrogen_bond",
        "screened_formal_charge",
        "pi_stacking",
        "cation_pi",
        "halogen_bond",
        "metal_coordination",
    ]
    assert metadata["inactive_terms"] == []
    assert metadata["vina"] == "excluded_from_guidance"
    metal = metal_coordination_v0_contract()
    assert metal["status"] == "superseded_by_profile_dispatched_v1"
    assert metal["supported_scope"]["metal"] == "Zn(II)"
    assert metal["supported_scope"]["fixed_receptor_donors"] == 3
    assert "directional_attraction" in metal["pair_energy"]["trace_components"][1]
    assert metal["pair_masking"]["active_metal_donor_pair"].startswith("replace")
    current_metal = metal_coordination_v1_contract()
    assert current_metal["status"] == "user_requested_default_on_diagnostic"
    assert current_metal["attraction_enabled_profiles"] == ["ZN", "MG"]
    assert {"CA", "MN", "FE", "CO", "NI", "CU"} == set(
        current_metal["repulsion_only_profiles"]
    )
    metal_only = InteractionEnergyConfig(active_terms=("metal_coordination",))
    metal_components = interaction_energy(coords, system, metal_only)
    assert set(metal_components) == {"interaction_metal_coordination", "total"}
    assert float(metal_components["total"]) == pytest.approx(0.0)


def test_fragment_force_projection_is_se3_equivariant() -> None:
    coords, system = _butane_system()
    coords = coords.unsqueeze(0)
    centers = fragment_centers(coords, system.topology.fragment_id)
    force = torch.tensor(
        [[[1.0, -0.3, 0.2], [-0.2, 0.8, 0.4], [0.1, -0.7, 0.6], [0.5, 0.2, -0.9]]],
        dtype=torch.float64,
    )
    translation, angular = project_atom_forces(
        force,
        coords,
        centers,
        system.topology.fragment_id,
        system.topology.mass,
    )
    rotation = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float64,
    )
    offset = torch.tensor([3.0, -2.0, 1.0], dtype=torch.float64)
    transformed_coords = coords @ rotation.T + offset
    transformed_centers = centers @ rotation.T + offset
    transformed_force = force @ rotation.T
    transformed_translation, transformed_angular = project_atom_forces(
        transformed_force,
        transformed_coords,
        transformed_centers,
        system.topology.fragment_id,
        system.topology.mass,
    )
    torch.testing.assert_close(
        transformed_translation,
        translation @ rotation.T,
        atol=1e-10,
        rtol=1e-10,
    )
    torch.testing.assert_close(
        transformed_angular,
        angular @ rotation.T,
        atol=1e-10,
        rtol=1e-10,
    )


def test_exact_protein_ligand_overlap_has_finite_energy_and_force() -> None:
    coords, system = _butane_system()
    overlapped = coords.clone()
    overlapped[0] = system.protein_coords[0]
    row = trace_physical_pose(
        overlapped,
        system,
        pose_kind="coincident_atom",
    )
    assert math.isfinite(row["energies"]["total"])
    assert math.isfinite(row["force_by_term"]["total"]["max_norm"])


def test_collinear_torsion_has_finite_energy_and_force() -> None:
    _, system = _butane_system()
    collinear = torch.tensor(
        [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [3.0, 0.0, 0.0], [4.5, 0.0, 0.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    energy = physical_energy(collinear, system)["total"]
    force = -torch.autograd.grad(energy, collinear)[0]
    assert torch.isfinite(energy)
    assert torch.isfinite(force).all()


def test_absolute_and_centered_frames_have_identical_energy_and_force(tmp_path) -> None:
    mol, coords = _butane_molecule()
    origin = torch.tensor([10.0, -4.0, 2.0], dtype=torch.float64)
    absolute_coords = coords + origin
    protein = tmp_path / "protein.pdb"
    protein.write_text(
        _pdb_atom_line("ATOM", 1, "CA", "ALA", (10.0, 0.0, 2.0), "C")
        + _pdb_atom_line("ATOM", 2, "O", "ALA", (14.0, 0.0, 2.0), "O")
        + "END\n"
    )
    fragment_id = torch.tensor([0, 0, 1, 1])
    absolute_system = build_physical_system(
        mol,
        protein,
        fragment_id=fragment_id,
        near_coords=absolute_coords,
    ).to(torch.device("cpu"), torch.float64)
    centered_system = build_physical_system(
        mol,
        protein,
        fragment_id=fragment_id,
        near_coords=absolute_coords,
        coordinate_origin=origin,
    ).to(torch.device("cpu"), torch.float64)

    absolute = absolute_coords.clone().requires_grad_(True)
    centered = coords.clone().requires_grad_(True)
    energy_absolute = physical_energy(absolute, absolute_system)["total"]
    energy_centered = physical_energy(centered, centered_system)["total"]
    force_absolute = -torch.autograd.grad(energy_absolute, absolute)[0]
    force_centered = -torch.autograd.grad(energy_centered, centered)[0]
    torch.testing.assert_close(energy_absolute, energy_centered, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(force_absolute, force_centered, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize(
    ("residue", "expected"),
    [
        ("HIS", {"ND1": (False, False), "NE2": (False, False)}),
        ("HID", {"ND1": (True, False), "NE2": (False, True)}),
        ("HIE", {"ND1": (False, True), "NE2": (True, False)}),
        ("HIP", {"ND1": (True, False), "NE2": (True, False)}),
    ],
)
def test_protein_histidine_typing_preserves_explicit_state_and_excludes_plain_his(
    tmp_path,
    residue: str,
    expected: dict[str, tuple[bool, bool]],
) -> None:
    mol, coords = _butane_molecule()
    atom_positions = {
        "N": (0.0, 4.0, 0.0),
        "CA": (1.0, 4.0, 0.0),
        "C": (2.0, 4.0, 0.0),
        "O": (3.0, 4.0, 0.0),
        "CB": (1.0, 5.0, 0.0),
        "CG": (2.0, 5.0, 0.0),
        "ND1": (3.0, 5.0, 0.0),
        "CD2": (2.0, 6.0, 0.0),
        "CE1": (4.0, 6.0, 0.0),
        "NE2": (3.0, 7.0, 0.0),
    }
    protein = tmp_path / f"{residue}.pdb"
    protein.write_text(
        "".join(
            _pdb_atom_line(
                "ATOM",
                serial,
                atom_name,
                residue,
                xyz,
                ("N" if atom_name in {"N", "ND1", "NE2"} else "O" if atom_name == "O" else "C"),
            )
            for serial, (atom_name, xyz) in enumerate(
                atom_positions.items(),
                start=1,
            )
        )
        + "END\n"
    )
    system = build_physical_system(
        mol,
        protein,
        fragment_id=torch.tensor([0, 0, 1, 1]),
        near_coords=coords,
    ).to(torch.device("cpu"), torch.float64)
    topology = system.interaction_topology
    assert topology is not None
    shell_names = list(atom_positions)
    for atom_name, (donor, acceptor) in expected.items():
        index = shell_names.index(atom_name)
        assert bool(topology.protein_is_donor[index]) is donor
        assert bool(topology.protein_is_acceptor[index]) is acceptor
    expected_ambiguous = 2 if residue == "HIS" else 0
    assert int(topology.protein_is_ambiguous_histidine.sum()) == expected_ambiguous


def test_protein_hbond_geometry_declares_exact_expected_heavy_degree() -> None:
    expected = {
        ("ALA", "N", "N"): (3, 2),
        ("ALA", "O", "O"): (3, 1),
        ("ARG", "NE", "N"): (3, 2),
        ("ARG", "NH1", "N"): (3, 1),
        ("ARG", "NH2", "N"): (3, 1),
        ("ASN", "ND2", "N"): (3, 1),
        ("ASN", "OD1", "O"): (3, 1),
        ("GLN", "NE2", "N"): (3, 1),
        ("GLN", "OE1", "O"): (3, 1),
        ("ASP", "OD1", "O"): (3, 1),
        ("GLU", "OE2", "O"): (3, 1),
        ("HIS", "ND1", "N"): (3, 2),
        ("TRP", "NE1", "N"): (3, 2),
        ("LYS", "NZ", "N"): (4, 1),
        ("SER", "OG", "O"): (4, 1),
        ("THR", "OG1", "O"): (4, 1),
        ("TYR", "OH", "O"): (3, 1),
    }
    for site, geometry in expected.items():
        assert _protein_valence_geometry(*site) == geometry
    assert _protein_valence_geometry("ALA", "OXT", "O") is None
    assert _protein_valence_geometry("ALA", "CA", "C") is None


def test_incomplete_protein_hbond_sites_fail_closed(tmp_path) -> None:
    mol, coords = _butane_molecule()
    protein = tmp_path / "incomplete_sites.pdb"
    protein.write_text(
        _pdb_atom_line("ATOM", 1, "N", "ARG", (0.0, 4.0, 0.0), "N")
        + _pdb_atom_line("ATOM", 2, "CA", "ARG", (1.0, 4.0, 0.0), "C")
        + _pdb_atom_line("ATOM", 3, "CD", "ARG", (0.0, 6.0, 0.0), "C")
        + _pdb_atom_line("ATOM", 4, "NE", "ARG", (1.0, 6.0, 0.0), "N")
        + "END\n"
    )
    system = build_physical_system(
        mol,
        protein,
        fragment_id=torch.tensor([0, 0, 1, 1]),
        near_coords=coords,
    ).to(torch.device("cpu"), torch.float64)
    topology = system.interaction_topology
    assert topology is not None
    labels = list(topology.protein_atom_labels)
    terminal_n = labels.index("A:ARG1:N")
    incomplete_ne = labels.index("A:ARG1:NE")
    assert not bool(topology.protein_is_donor[terminal_n])
    assert not bool(topology.protein_is_donor[incomplete_ne])
    assert bool(topology.protein_is_geometry_excluded_hbond_site[terminal_n])
    assert bool(topology.protein_is_geometry_excluded_hbond_site[incomplete_ne])
    assert float(topology.protein_direction_target_cosine[terminal_n]) == pytest.approx(0.0)
    assert float(topology.protein_direction_target_cosine[incomplete_ne]) == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("residue", "atom_names"),
    [
        ("ASH", ("N", "CA", "C", "O", "CB", "CG", "OD1", "OD2")),
        (
            "GLH",
            ("N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "OE2"),
        ),
        ("CYM", ("N", "CA", "C", "O", "CB", "SG")),
        ("CYX", ("N", "CA", "C", "O", "CB", "SG")),
        ("SEP", ("N", "CA", "C", "O", "CB", "OG")),
    ],
)
def test_unsupported_explicit_residue_variants_fail_closed_for_interactions(
    tmp_path,
    residue: str,
    atom_names: tuple[str, ...],
) -> None:
    mol, coords = _butane_molecule()
    protein = tmp_path / f"{residue}.pdb"
    protein.write_text(
        "".join(
            _pdb_atom_line(
                "ATOM",
                serial,
                atom_name,
                residue,
                (float(serial % 3), 4.0 + float(serial // 3), 0.0),
                (
                    "N"
                    if atom_name.startswith("N")
                    else "O"
                    if atom_name.startswith("O")
                    else "S"
                    if atom_name.startswith("S")
                    else "C"
                ),
            )
            for serial, atom_name in enumerate(atom_names, start=1)
        )
        + "END\n"
    )
    system = build_physical_system(
        mol,
        protein,
        fragment_id=torch.tensor([0, 0, 1, 1]),
        near_coords=coords,
    ).to(torch.device("cpu"), torch.float64)
    topology = system.interaction_topology
    assert topology is not None
    assert int(topology.protein_is_unsupported_variant.sum()) == len(atom_names)
    assert not bool(topology.protein_is_donor.any())
    assert not bool(topology.protein_is_acceptor.any())
    assert not bool(topology.protein_is_hydrophobe.any())


@pytest.mark.parametrize("record", ["ATOM", "HETATM"])
def test_active_nonprotein_residue_fails_regardless_of_record_type(
    tmp_path,
    record: str,
) -> None:
    mol, coords = _butane_molecule()
    protein = tmp_path / "protein_with_ligand.pdb"
    protein.write_text(
        _pdb_atom_line("ATOM", 1, "CA", "ALA", (0.0, 4.0, 0.0), "C")
        + _pdb_atom_line(record, 2, "C1", "LIG", (0.0, 0.0, 0.0), "C")
        + "END\n"
    )
    with pytest.raises(
        UnsupportedPhysicalChemistryError,
        match="does not parameterize non-protein residues",
    ) as exc_info:
        build_physical_system(
            mol,
            protein,
            fragment_id=torch.tensor([0, 0, 1, 1]),
            near_coords=coords,
        )
    assert exc_info.value.code == "active_nonprotein_residue"
    assert exc_info.value.details["record_types"] == [record]


def test_trace_ligand_loader_rejects_failed_sanitization(
    monkeypatch,
    tmp_path,
) -> None:
    mol, _ = _butane_molecule()
    monkeypatch.setattr(
        trace_physical,
        "load_molecule",
        lambda *_args, **_kwargs: (mol, False, False),
    )
    with pytest.raises(ValueError, match="requires successful ligand sanitization"):
        trace_physical._load_trace_ligand(tmp_path / "ligand.sdf")


def test_trace_shell_includes_crystal_perturbation_coordinates(tmp_path) -> None:
    mol, _ = _butane_molecule()
    ligand = tmp_path / "ligand.sdf"
    writer = Chem.SDWriter(str(ligand))
    writer.write(mol)
    writer.close()
    protein = tmp_path / "protein.pdb"
    protein.write_text(
        _pdb_atom_line("ATOM", 1, "CA", "ALA", (0.0, 6.0, 0.0), "C")
        + _pdb_atom_line("ATOM", 2, "CB", "ALA", (0.0, 13.0, 0.0), "C")
        + "END\n"
    )
    args = trace_physical.build_arg_parser().parse_args(
        ["--protein", str(protein), "--ligand", str(ligand)]
    )
    report = trace_physical.build_trace_report(args)

    assert report["schema_version"] == "effdock.guidance_trace.v6"
    assert report["protocol_id"] == "EFFDOCK-GUIDANCE-DIAGNOSTIC-V5"
    assert report["system"]["protein_shell_heavy_atoms"] == 2
    assert report["system"]["term_counts"]["protein_ligand_pairs"] == 8
    assert report["system"]["topology_reference_sha256"]
    assert report["guidance_layers"]["physical"]["status"] == "active_diagnostic"
    assert report["guidance_layers"]["interaction"]["status"] == "active_diagnostic"
    assert report["system"]["interaction_reference_sha256"]
    crystal = report["rows"][0]
    assert "interaction_hydrophobic" in crystal["energies"]
    assert "interaction_hydrogen_bond" in crystal["energies"]
    assert "interaction_screened_formal_charge" in crystal["energies"]
    assert crystal["energy_groups"]["combined"] == pytest.approx(
        crystal["energy_groups"]["physical"] + crystal["energy_groups"]["interaction"]
    )
    assert (
        crystal["interaction_contacts"]["typing_counts"]
        == report["system"]["interaction_term_counts"]
    )

    narrow_shell_args = trace_physical.build_arg_parser().parse_args(
        [
            "--protein",
            str(protein),
            "--ligand",
            str(ligand),
            "--protein-cutoff",
            "9.9",
        ]
    )
    with pytest.raises(
        ValueError,
        match="must cover every active physical/interaction cutoff",
    ):
        trace_physical.build_trace_report(narrow_shell_args)


def test_trace_saved_results_emits_final_and_trajectory_rows(tmp_path) -> None:
    mol, coords = _butane_molecule()
    ligand = tmp_path / "ligand.sdf"
    writer = Chem.SDWriter(str(ligand))
    writer.write(mol)
    writer.close()
    protein = tmp_path / "protein.pdb"
    protein.write_text(
        _pdb_atom_line("ATOM", 1, "CA", "ALA", (0.0, 4.0, 0.0), "C")
        + _pdb_atom_line("ATOM", 2, "O", "ALA", (4.0, 4.0, 0.0), "O")
        + "END\n"
    )
    results = tmp_path / "results.pt"
    torch.save(
        {
            "schema_version": "effdock.docking_results.v2",
            "ligand_identity": {
                **ligand_graph_identity(
                    mol,
                    torch.tensor([0, 0, 1, 1]),
                ),
                "source": {
                    "kind": "file",
                    "sha256": trace_physical._file_sha256(ligand),
                },
            },
            "poses": [{"atom_pos_pred": coords + 0.1}],
            "trajectories": [
                {
                    "traj": [coords + 0.2, coords + 0.1],
                    "traj_times": [0.0, 1.0],
                }
            ],
        },
        results,
    )
    args = trace_physical.build_arg_parser().parse_args(
        [
            "--protein",
            str(protein),
            "--ligand",
            str(ligand),
            "--results",
            str(results),
            "--results-frame",
            "absolute",
        ]
    )
    report = trace_physical.build_trace_report(args)

    assert [row["pose_kind"] for row in report["rows"]] == [
        "crystal",
        "cut_bond_stretch",
        "cut_bond_torsion",
        "protein_ligand_overlap",
        "sampled_final",
        "trajectory",
        "trajectory",
    ]
    assert [row["step"] for row in report["rows"][-2:]] == [0, 1]
    assert [row["t"] for row in report["rows"][-2:]] == [0.0, 1.0]
    for row in report["rows"]:
        assert set(row["energy_delta_from_crystal"]) == set(row["energies"])
    assert (
        report["inputs"]["results_ligand_identity_sha256"]
        == ligand_graph_identity(
            mol,
            torch.tensor([0, 0, 1, 1]),
        )["sha256"]
    )


def test_trace_rejects_saved_results_for_different_ligand_identity(tmp_path) -> None:
    mol, coords = _butane_molecule()
    ligand = tmp_path / "ligand.sdf"
    writer = Chem.SDWriter(str(ligand))
    writer.write(mol)
    writer.close()
    protein = tmp_path / "protein.pdb"
    protein.write_text(_pdb_atom_line("ATOM", 1, "CA", "ALA", (0.0, 4.0, 0.0), "C") + "END\n")
    identity = ligand_graph_identity(
        mol,
        torch.tensor([0, 0, 1, 1]),
    )
    identity["atoms"][0]["atomic_number"] = 7
    results = tmp_path / "results.pt"
    torch.save(
        {
            "schema_version": "effdock.docking_results.v2",
            "ligand_identity": {
                **identity,
                "source": {
                    "kind": "file",
                    "sha256": trace_physical._file_sha256(ligand),
                },
            },
            "poses": [{"atom_pos_pred": coords}],
        },
        results,
    )
    args = trace_physical.build_arg_parser().parse_args(
        [
            "--protein",
            str(protein),
            "--ligand",
            str(ligand),
            "--results",
            str(results),
            "--results-frame",
            "absolute",
        ]
    )
    with pytest.raises(ValueError, match="identity/order does not match"):
        trace_physical.build_trace_report(args)

    source_mismatch = tmp_path / "results_source_mismatch.pt"
    torch.save(
        {
            "schema_version": "effdock.docking_results.v2",
            "ligand_identity": {
                **ligand_graph_identity(
                    mol,
                    torch.tensor([0, 0, 1, 1]),
                ),
                "source": {"kind": "file", "sha256": "0" * 64},
            },
            "poses": [{"atom_pos_pred": coords}],
        },
        source_mismatch,
    )
    source_args = trace_physical.build_arg_parser().parse_args(
        [
            "--protein",
            str(protein),
            "--ligand",
            str(ligand),
            "--results",
            str(source_mismatch),
            "--results-frame",
            "absolute",
        ]
    )
    with pytest.raises(ValueError, match="source hash does not match"):
        trace_physical.build_trace_report(source_args)


def test_protein_ligand_energy_is_independent_of_shell_and_chunk_layout() -> None:
    coords, shell_system = _butane_system()
    far_coords = torch.tensor(
        [[100.0 + float(index), 100.0, 100.0] for index in range(40)],
        dtype=torch.float64,
    )
    far_atomic_numbers = torch.full((far_coords.shape[0],), 6, dtype=torch.long)
    far_parameters = element_parameters(far_atomic_numbers, dtype=torch.float64)
    expanded_system = PhysicalSystem(
        topology=shell_system.topology,
        protein_coords=torch.cat((far_coords[:17], shell_system.protein_coords, far_coords[17:])),
        protein_atomic_numbers=torch.cat(
            (
                far_atomic_numbers[:17],
                shell_system.protein_atomic_numbers,
                far_atomic_numbers[17:],
            )
        ),
        protein_uff_x=torch.cat(
            (
                far_parameters.uff_x[:17],
                shell_system.protein_uff_x,
                far_parameters.uff_x[17:],
            )
        ),
        protein_uff_d=torch.cat(
            (
                far_parameters.uff_d[:17],
                shell_system.protein_uff_d,
                far_parameters.uff_d[17:],
            )
        ),
        protein_vdw_radius=torch.cat(
            (
                far_parameters.vdw_radius[:17],
                shell_system.protein_vdw_radius,
                far_parameters.vdw_radius[17:],
            )
        ),
        parameter_set=shell_system.parameter_set,
        protein_source_atoms=42,
    )
    shell_energy = physical_energy(
        coords,
        shell_system,
        PhysicalEnergyConfig(protein_chunk_size=1),
    )
    expanded_energy = physical_energy(
        coords,
        expanded_system,
        PhysicalEnergyConfig(protein_chunk_size=19),
    )

    for term in (
        "protein_ligand_lj_repulsive",
        "protein_ligand_lj_attractive",
        "protein_ligand_steric_barrier",
        "total",
    ):
        torch.testing.assert_close(
            shell_energy[term],
            expanded_energy[term],
            atol=1e-12,
            rtol=1e-12,
        )


def test_frozen_v3_diagnostic_results_remain_historical() -> None:
    results_path = Path(__file__).parents[1] / "docs" / "GUIDANCE_DIAGNOSTIC_RESULTS.json"
    results = json.loads(results_path.read_text())
    assert results["schema_version"] == "effdock.guidance_diagnostic_results.v3"
    assert results["protocol_id"] == "EFFDOCK-GUIDANCE-DIAGNOSTIC-V3"
    assert results["guidance_layers"]["interaction"]["active_terms"] == [
        "hydrophobic",
        "hydrogen_bond",
    ]
    assert "ligand_intra_bond" in results["guidance_layers"]["physical"]["active_terms"]
    assert results["parameter_set"]["sha256"] != guidance_parameter_identity()["sha256"]
    assert (
        results["implementation"]["sha256"] != trace_physical._implementation_identity()["sha256"]
    )
