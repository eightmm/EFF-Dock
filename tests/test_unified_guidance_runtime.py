from __future__ import annotations

import torch
from rdkit import Chem

import effdock.guidance.runtime as runtime_module
from effdock.guidance import (
    UnifiedGuidance,
    UnifiedGuidanceConfig,
    build_physical_system,
    guidance_energy,
)
from effdock.inference.sampler import build_time_grid


def _ethane() -> Chem.Mol:
    mol = Chem.MolFromSmiles("CC")
    conf = Chem.Conformer(mol.GetNumAtoms())
    conf.SetAtomPosition(0, (0.0, 0.0, 0.0))
    conf.SetAtomPosition(1, (1.5, 0.0, 0.0))
    mol.AddConformer(conf)
    return mol


def _pdb_atom_line(x: float, y: float, z: float) -> str:
    return (
        f"{'ATOM':<6}{1:5d} {'CB':^4s} {'ALA':>3s} A{1:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00           C  \n"
    )


def _system(tmp_path):
    protein = tmp_path / "protein.pdb"
    protein.write_text(_pdb_atom_line(4.0, 0.0, 0.0) + "END\n")
    mol = _ethane()
    fragment_id = torch.tensor([0, 1], dtype=torch.long)
    system = build_physical_system(
        mol,
        protein,
        fragment_id=fragment_id,
        near_coords=torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64),
        protein_cutoff=10.0,
    ).to(torch.device("cpu"), torch.float64)
    return system, fragment_id


def _single_fragment_system(tmp_path):
    protein = tmp_path / "protein_single_fragment.pdb"
    protein.write_text(_pdb_atom_line(4.0, 0.0, 0.0) + "END\n")
    fragment_id = torch.tensor([0, 0], dtype=torch.long)
    system = build_physical_system(
        _ethane(),
        protein,
        fragment_id=fragment_id,
        near_coords=torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64),
        protein_cutoff=10.0,
    ).to(torch.device("cpu"), torch.float64)
    return system


def test_unified_operator_split_descends_and_traces_all_interactions(tmp_path) -> None:
    system, fragment_id = _system(tmp_path)
    corrector = UnifiedGuidance(
        system,
        UnifiedGuidanceConfig(
            start_t=0.0,
            softcore_start=0.75,
            softcore_end=0.75,
            max_atom_displacement=0.2,
            max_backtracks=8,
        ),
    )
    T = torch.tensor(
        [[[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]]],
        dtype=torch.float64,
    )
    q = torch.tensor(
        [[[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]],
        dtype=torch.float64,
    )
    local = torch.zeros(2, 3, dtype=torch.float64)
    before = guidance_energy(T[0], system)["total"]

    corrected_T, corrected_q = corrector.correct(
        T.view(-1, 3),
        q.view(-1, 4),
        local,
        fragment_id,
        torch.tensor([1, 1], dtype=torch.long),
        dt=0.25,
        t=1.0,
        scale=0.2,
    )
    after = guidance_energy(corrected_T.view(2, 3), system)["total"]

    assert torch.isfinite(corrected_T).all()
    assert torch.isfinite(corrected_q).all()
    assert after <= before + 1e-8
    assert (corrected_T.view(1, 2, 3) - T).norm(dim=-1).max() <= 0.2 + 1e-8
    assert corrector.last_components is not None
    assert {
        "protein_ligand_steric_barrier",
        "interaction_hydrophobic",
        "interaction_hydrogen_bond",
        "interaction_screened_formal_charge",
        "interaction_pi_stacking",
        "interaction_cation_pi",
        "interaction_halogen_bond",
        "interaction_metal_coordination",
    }.issubset(corrector.last_components)
    stats = corrector.diagnostics()
    assert stats["pose_corrections_attempted"] == 1
    assert stats["pose_corrections_accepted"] == 1
    assert stats["pose_corrections_rejected"] == 0


def test_unified_operator_split_is_exact_noop_before_start(tmp_path) -> None:
    system, fragment_id = _system(tmp_path)
    corrector = UnifiedGuidance(
        system,
        UnifiedGuidanceConfig(start_t=0.5),
    )
    T = torch.tensor([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]], dtype=torch.float64)
    q = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
        dtype=torch.float64,
    )
    corrected_T, corrected_q = corrector.correct(
        T,
        q,
        torch.zeros(2, 3, dtype=torch.float64),
        fragment_id,
        torch.tensor([1, 1], dtype=torch.long),
        dt=0.25,
        t=0.4,
        scale=0.2,
    )
    torch.testing.assert_close(corrected_T, T)
    torch.testing.assert_close(corrected_q, q)
    assert corrector.diagnostics()["steps_attempted"] == 0


def test_unified_operator_split_is_exact_noop_at_zero_scale(tmp_path) -> None:
    system, fragment_id = _system(tmp_path)
    corrector = UnifiedGuidance(
        system,
        UnifiedGuidanceConfig(start_t=0.0),
    )
    T = torch.tensor([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]], dtype=torch.float64)
    q = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
        dtype=torch.float64,
    )
    corrected_T, corrected_q = corrector.correct(
        T,
        q,
        torch.zeros(2, 3, dtype=torch.float64),
        fragment_id,
        torch.tensor([1, 1], dtype=torch.long),
        dt=0.25,
        t=1.0,
        scale=0.0,
    )
    torch.testing.assert_close(corrected_T, T)
    torch.testing.assert_close(corrected_q, q)
    assert corrector.diagnostics()["steps_attempted"] == 0


def test_unified_operator_split_rejects_without_changing_state(tmp_path) -> None:
    system, fragment_id = _system(tmp_path)
    corrector = UnifiedGuidance(
        system,
        UnifiedGuidanceConfig(
            start_t=0.0,
            softcore_start=0.75,
            softcore_end=0.75,
            max_atom_displacement=1e-12,
            max_backtracks=0,
        ),
    )
    T = torch.tensor([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]], dtype=torch.float64)
    q = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
        dtype=torch.float64,
    )
    corrected_T, corrected_q = corrector.correct(
        T,
        q,
        torch.zeros(2, 3, dtype=torch.float64),
        fragment_id,
        torch.tensor([1, 1], dtype=torch.long),
        dt=0.25,
        t=1.0,
        scale=0.2,
    )
    torch.testing.assert_close(corrected_T, T)
    torch.testing.assert_close(corrected_q, q)
    stats = corrector.diagnostics()
    assert stats["pose_corrections_accepted"] == 0
    assert stats["pose_corrections_rejected"] == 1
    assert stats["total_backtracks"] == 0


def test_unified_operator_split_rejects_only_nonfinite_pose(
    tmp_path,
    monkeypatch,
) -> None:
    system, fragment_id = _system(tmp_path)

    def mixed_finite_energy(coords, _system, _config):
        total = coords.square().sum(dim=(1, 2))
        denominator = torch.where(
            coords[:, 0, 0] < 1.0,
            torch.zeros_like(total),
            torch.ones_like(total),
        )
        return {"total": total / denominator}

    monkeypatch.setattr(runtime_module, "guidance_energy", mixed_finite_energy)
    corrector = UnifiedGuidance(
        system,
        UnifiedGuidanceConfig(
            start_t=0.0,
            max_atom_displacement=0.25,
            max_backtracks=2,
        ),
    )
    one_T = torch.tensor([[0.5, 0.0, 0.0], [1.5, 0.0, 0.0]], dtype=torch.float64)
    T = torch.cat((one_T, one_T + torch.tensor([2.0, 0.0, 0.0])), dim=0)
    one_q = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
        dtype=torch.float64,
    )
    q = one_q.repeat(2, 1)
    corrected_T, corrected_q = corrector.correct(
        T,
        q,
        torch.zeros(2, 3, dtype=torch.float64),
        fragment_id,
        torch.tensor([1, 1, 1, 1], dtype=torch.long),
        dt=0.25,
        t=1.0,
        scale=0.2,
    )

    torch.testing.assert_close(corrected_T[:2], T[:2])
    torch.testing.assert_close(corrected_q[:2], q[:2])
    assert torch.isfinite(corrected_T[2:]).all()
    assert not torch.equal(corrected_T[2:], T[2:])
    stats = corrector.diagnostics()
    assert stats["nonfinite_base_poses"] == 1
    assert stats["pose_corrections_accepted"] == 1
    assert stats["pose_corrections_rejected"] == 1


def test_unified_direct_velocity_matches_pose_atom_speed_and_preserves_singletons(
    tmp_path,
    monkeypatch,
) -> None:
    system, _ = _system(tmp_path)

    def quadratic_energy(coords, _system, _config):
        return {"total": coords.square().sum(dim=(1, 2))}

    monkeypatch.setattr(runtime_module, "guidance_energy", quadratic_energy)
    guidance = UnifiedGuidance(
        system,
        UnifiedGuidanceConfig(
            start_t=0.5,
            ramp_power=1.0,
            max_atom_force=1e6,
            max_translation_velocity=1e6,
            max_angular_velocity=1e6,
            max_atom_displacement=1e6,
        ),
    )
    coords = torch.tensor(
        [
            [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[2.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        ],
        dtype=torch.float64,
    )
    centers = coords.clone()
    learned_translation = torch.tensor(
        [
            [[2.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            [[4.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
        ],
        dtype=torch.float64,
    )
    learned_angular = torch.full_like(learned_translation, 7.0)

    direct_translation, direct_angular = guidance.direct_velocity(
        coords.reshape(-1, 3),
        centers.reshape(-1, 3),
        learned_translation.reshape(-1, 3),
        learned_angular.reshape(-1, 3),
        torch.ones(4, dtype=torch.long),
        t_start=0.5,
        t_end=1.0,
        strength=0.5,
    )

    direct_translation = direct_translation.view(2, 2, 3)
    direct_rms = direct_translation.square().sum(dim=-1).mean(dim=1).sqrt()
    reference_rms = learned_translation.square().sum(dim=-1).mean(dim=1).sqrt()
    # Average linear ramp on [0.5,1] is 0.5, so eta=0.5 gives ratio 0.25.
    torch.testing.assert_close(direct_rms, 0.25 * reference_rms)
    assert torch.all(direct_translation[..., 0] < 0)
    torch.testing.assert_close(direct_angular, torch.zeros_like(direct_angular))
    stats = guidance.diagnostics()
    assert stats["direct_steps_attempted"] == 1
    assert stats["direct_pose_evaluations"] == 2
    assert stats["direct_pose_applied"] == 2
    assert stats["direct_nonfinite_poses"] == 0
    assert stats["direct_model_atom_speed_rms_sum"] == 6.0
    assert stats["direct_applied_atom_speed_rms_sum"] == 1.5
    assert stats["direct_total_atom_speed_rms_sum"] == 4.5
    assert stats["direct_atom_speed_rms_valid_count"] == 2
    assert stats["direct_applied_to_model_rms_ratio_sum"] == 0.5
    assert stats["direct_applied_to_model_rms_ratio_valid_count"] == 2
    assert stats["direct_model_guide_cosine_sum"] == -2.0
    assert stats["direct_model_guide_cosine_valid_count"] == 2
    assert stats["direct_guide_parallel_to_model_ratio_sum"] == -0.5
    assert stats["direct_guide_parallel_to_model_ratio_valid_count"] == 2
    assert stats["direct_model_rms_path_proxy_sum"] == 3.0
    assert stats["direct_applied_rms_path_proxy_sum"] == 0.75
    assert stats["direct_total_rms_path_proxy_sum"] == 2.25
    assert stats["direct_cap_scale_valid_count"] == 2
    assert stats["direct_any_cap_trigger_count"] == 0

    trace = guidance.direct_step_trace()
    assert len(trace) == 1
    step = trace[0]
    assert step["t"] == 0.5
    assert step["dt"] == 0.5
    assert step["ramp"] == 0.5
    assert step["eta"] == 0.5
    assert step["applied_to_model_rms_ratio_valid_count"] == 2
    assert step["applied_to_model_rms_ratio_p05"] == 0.25
    assert step["applied_to_model_rms_ratio_p99"] == 0.25
    assert step["model_guide_cosine_p50"] == -1.0
    assert step["guide_parallel_to_model_ratio_p95"] == -0.25
    assert step["cap_scale_p99"] == 1.0


def test_unified_direct_velocity_traces_each_cap_trigger_exactly(
    tmp_path,
    monkeypatch,
) -> None:
    system = _single_fragment_system(tmp_path)
    guidance = UnifiedGuidance(
        system,
        UnifiedGuidanceConfig(
            start_t=0.5,
            ramp_power=1.0,
            max_atom_force=1e6,
            max_translation_velocity=1.0,
            max_angular_velocity=1.0,
            max_atom_displacement=1.5,
        ),
    )

    def fixed_direction(coords, centers, *, progress, apply_schedule_and_caps):
        del centers, progress
        assert apply_schedule_and_caps is False
        translation = coords.new_tensor(
            [
                [[1.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0]],
                [[1.0, 0.0, 0.0]],
            ]
        )
        angular = coords.new_tensor(
            [
                [[0.0, 0.0, 0.0]],
                [[0.0, 0.0, 1.0]],
                [[0.0, 0.0, 1.0]],
                [[0.0, 0.0, 1.0]],
            ]
        )
        total = coords.new_zeros(coords.shape[0])
        finite = torch.ones(coords.shape[0], dtype=torch.bool, device=coords.device)
        return translation, angular, total, finite

    monkeypatch.setattr(guidance, "_direction", fixed_direction)
    coords = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[-0.1, 0.0, 0.0], [0.1, 0.0, 0.0]],
            [[-20.0, 0.0, 0.0], [20.0, 0.0, 0.0]],
            [[-2.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        ],
        dtype=torch.float64,
    )
    centers = torch.zeros(4, 1, 3, dtype=torch.float64)
    learned_translation = torch.tensor(
        [[[8.0, 0.0, 0.0]], [[8.0, 0.0, 0.0]], [[16.0, 0.0, 0.0]], [[20.0, 0.0, 0.0]]],
        dtype=torch.float64,
    )
    zeros = torch.zeros_like(learned_translation)

    direct_translation, direct_angular = guidance.direct_velocity(
        coords.reshape(-1, 3),
        centers.reshape(-1, 3),
        learned_translation.reshape(-1, 3),
        zeros.reshape(-1, 3),
        torch.full((4,), 2, dtype=torch.long),
        t_start=0.5,
        t_end=1.0,
        strength=0.5,
    )

    assert direct_translation.view(4, 1, 3).norm(dim=-1).max() <= 1.0 + 1e-12
    assert direct_angular.view(4, 1, 3).norm(dim=-1).max() <= 1.0 + 1e-12
    stats = guidance.diagnostics()
    assert stats["direct_translation_cap_trigger_count"] == 2
    assert stats["direct_angular_cap_trigger_count"] == 2
    assert stats["direct_displacement_cap_trigger_count"] == 2
    assert stats["direct_any_cap_trigger_count"] == 4
    assert stats["direct_multiple_cap_trigger_count"] == 1
    assert stats["direct_cap_scale_valid_count"] == 4
    assert stats["direct_max_estimated_atom_displacement"] <= 1.5 + 1e-12

    step = guidance.direct_step_trace()[0]
    assert step["eta"] == 0.5
    assert step["translation_cap_trigger_count"] == 2
    assert step["angular_cap_trigger_count"] == 2
    assert step["displacement_cap_trigger_count"] == 2
    assert step["any_cap_trigger_count"] == 4
    assert step["multiple_cap_trigger_count"] == 1
    assert step["cap_scale_valid_count"] == 4


def test_unified_direct_interval_average_has_same_integrated_ramp_across_grids(
    tmp_path,
    monkeypatch,
) -> None:
    system, _ = _system(tmp_path)

    def quadratic_energy(coords, _system, _config):
        return {"total": coords.square().sum(dim=(1, 2))}

    monkeypatch.setattr(runtime_module, "guidance_energy", quadratic_energy)
    coords = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float64)
    learned = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float64)
    zero_angular = torch.zeros_like(learned)
    integrated: list[float] = []

    for steps in (10, 20, 25):
        guidance = UnifiedGuidance(
            system,
            UnifiedGuidanceConfig(
                start_t=0.5,
                ramp_power=1.0,
                max_atom_force=1e6,
                max_translation_velocity=1e6,
                max_angular_velocity=1e6,
                max_atom_displacement=1e6,
            ),
        )
        grid = build_time_grid(steps, schedule="late", power=3.0, dtype=torch.float64)
        displacement = 0.0
        for left, right in zip(grid[:-1], grid[1:]):
            velocity, _ = guidance.direct_velocity(
                coords,
                coords,
                learned,
                zero_angular,
                torch.ones(2, dtype=torch.long),
                t_start=float(left),
                t_end=float(right),
                strength=0.2,
            )
            displacement += float((right - left) * velocity[0, 0])
        integrated.append(displacement)

    torch.testing.assert_close(
        torch.tensor(integrated, dtype=torch.float64),
        torch.full((3,), -0.05, dtype=torch.float64),
        atol=1e-12,
        rtol=1e-12,
    )
