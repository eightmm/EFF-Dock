"""Tests for inference time discretization."""

import pytest
import torch

from effdock.inference.sampler import (
    build_time_grid,
    linear_fm_gaussian_translation_score,
    parse_sigma_list,
    sample_shared_prior_states,
    sample_unified,
    sample_unified_multi_sigma,
    score_corrected_translation_drift,
)


def test_explicit_sigma_counts_must_match_num_samples() -> None:
    sigmas, counts = parse_sigma_list("2:25,3:25,4:50", 100)
    assert sigmas == [2.0, 3.0, 4.0]
    assert counts == [25, 25, 50]

    with pytest.raises(ValueError, match="must sum to num_samples"):
        parse_sigma_list("2:20,3:20,4:40", 100)


class _ZeroSamplerModel(torch.nn.Module):
    def forward(self, batch):
        return {
            "v_pred": torch.zeros_like(batch["T_frag"]),
            "omega_pred": torch.zeros_like(batch["T_frag"]),
        }


class _UnitTranslationModel(torch.nn.Module):
    def forward(self, batch):
        return {
            "v_pred": torch.ones_like(batch["T_frag"]) * 2.0,
            "omega_pred": torch.zeros_like(batch["T_frag"]),
        }


def _tiny_sampler_inputs():
    graph = {
        "node_coords": torch.zeros(2, 3),
        "edge_index": torch.empty(2, 0, dtype=torch.long),
        "lig_frag_slice": torch.tensor([0, 1]),
        "lig_atom_slice": torch.tensor([1, 2]),
    }
    lig_data = {
        "frag_sizes": torch.tensor([1], dtype=torch.long),
        "fragment_id": torch.tensor([0], dtype=torch.long),
        "frag_local_coords": torch.zeros(1, 3),
    }
    meta = {
        "num_frag": 1,
        "pocket_center": torch.zeros(3),
    }
    return graph, lig_data, meta


def test_build_time_grid_uniform_matches_linspace():
    grid = build_time_grid(5, schedule="uniform", power=3.0)
    expected = torch.linspace(0.0, 1.0, 6)
    torch.testing.assert_close(grid, expected)


def test_build_time_grid_late_is_denser_near_t1():
    grid = build_time_grid(8, schedule="late", power=3.0)
    diffs = grid[1:] - grid[:-1]
    assert torch.isclose(grid[0], torch.tensor(0.0))
    assert torch.isclose(grid[-1], torch.tensor(1.0))
    assert torch.all(diffs > 0)
    assert diffs[0] > diffs[-1]


def test_build_time_grid_early_is_denser_near_t0():
    grid = build_time_grid(8, schedule="early", power=3.0)
    diffs = grid[1:] - grid[:-1]
    assert torch.isclose(grid[0], torch.tensor(0.0))
    assert torch.isclose(grid[-1], torch.tensor(1.0))
    assert torch.all(diffs > 0)
    assert diffs[0] < diffs[-1]


def test_linear_fm_translation_score_accounts_for_prior_variance() -> None:
    position = torch.tensor([[0.5, -0.25, 1.0]])
    velocity = torch.tensor([[2.0, 1.0, -1.0]])

    score_t0 = linear_fm_gaussian_translation_score(
        position,
        velocity,
        0.0,
        0.5,
    )
    torch.testing.assert_close(score_t0, -position / 0.25)

    score = linear_fm_gaussian_translation_score(
        position,
        velocity,
        0.4,
        0.5,
    )
    torch.testing.assert_close(score, (0.4 * velocity - position) / (0.6 * 0.25))


def test_score_corrected_translation_drift_matches_fokker_planck_correction() -> None:
    position = torch.tensor([[1.0, 2.0, 3.0]])
    velocity = torch.full_like(position, 2.0)
    score = (0.5 * velocity - position) / (0.5 * 0.5**2)

    corrected = score_corrected_translation_drift(
        position,
        velocity,
        0.5,
        0.5,
        0.3,
    )
    torch.testing.assert_close(corrected, velocity + 0.5 * 0.3**2 * score)


def test_sample_unified_particle_resample_copies_selected_states():
    graph, lig_data, meta = _tiny_sampler_inputs()
    initial_T = torch.arange(4, dtype=torch.float32).view(4, 1, 1).expand(4, 1, 3)
    initial_q = torch.tensor([1.0, 0.0, 0.0, 0.0]).view(1, 1, 4).expand(4, 1, 4)

    def reverse_particles(atom_pos, T_frag, q_frag, prior_sigma, time):
        assert atom_pos.shape == (4, 1, 3)
        assert T_frag.shape == (4, 1, 3)
        assert q_frag.shape == (4, 1, 4)
        assert prior_sigma.shape == (4,)
        assert time == 0.5
        return torch.tensor([3, 2, 1, 0])

    out = sample_unified(
        _ZeroSamplerModel(),
        graph,
        lig_data,
        meta,
        num_samples=4,
        num_steps=2,
        translation_sigma=1.0,
        time_schedule="uniform",
        device=torch.device("cpu"),
        initial_T_frag=initial_T,
        initial_q_frag=initial_q,
        particle_resample_times=[0.5],
        particle_resample_fn=reverse_particles,
    )

    final_x = torch.tensor([item["T_frag"][0, 0].item() for item in out])
    torch.testing.assert_close(final_x, torch.tensor([3.0, 2.0, 1.0, 0.0]))


def test_sample_unified_multi_sigma_reports_resampled_source_sigma():
    graph, lig_data, meta = _tiny_sampler_inputs()
    seen_prior_sigmas = []

    def duplicate_cross_sigma(atom_pos, T_frag, q_frag, prior_sigma, time):
        seen_prior_sigmas.append(prior_sigma.cpu())
        return torch.tensor([2, 2, 0, 0])

    out = sample_unified_multi_sigma(
        _ZeroSamplerModel(),
        graph,
        lig_data,
        meta,
        sigma_list=[0.5, 1.0],
        samples_per_sigma=[2, 2],
        num_steps=2,
        time_schedule="uniform",
        device=torch.device("cpu"),
        particle_resample_times=[0.5],
        particle_resample_fn=duplicate_cross_sigma,
    )

    torch.testing.assert_close(
        seen_prior_sigmas[0],
        torch.tensor([0.5, 0.5, 1.0, 1.0]),
    )
    assert [item["sigma"] for item in out] == [1.0, 1.0, 0.5, 0.5]


def test_fk_resampling_scores_constant_velocity_endpoint_without_extra_model_call():
    graph, lig_data, meta = _tiny_sampler_inputs()
    initial_T = torch.tensor([0.0, 10.0]).view(2, 1, 1).expand(2, 1, 3)
    initial_q = torch.tensor([1.0, 0.0, 0.0, 0.0]).view(1, 1, 4).expand(2, 1, 4)

    class CountingModel(_UnitTranslationModel):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, batch):
            self.calls += 1
            return super().forward(batch)

    class DuplicateSecondEndpoint:
        def __init__(self):
            self.seen = None
            self.reset_calls = 0

        def reset(self):
            self.reset_calls += 1

        def resample(
            self,
            terminal_atom_pos,
            *,
            prior_sigma,
            requested_time,
            actual_time,
        ):
            self.seen = (
                terminal_atom_pos.clone(),
                prior_sigma.clone(),
                requested_time,
                actual_time,
            )
            return torch.tensor([1, 1])

    model = CountingModel()
    resampler = DuplicateSecondEndpoint()
    out = sample_unified(
        model,
        graph,
        lig_data,
        meta,
        num_samples=2,
        num_steps=2,
        translation_sigma=1.0,
        time_schedule="uniform",
        device=torch.device("cpu"),
        save_traj=True,
        initial_T_frag=initial_T,
        initial_q_frag=initial_q,
        fk_resample_times=[0.5],
        fk_resampler=resampler,
    )

    assert model.calls == 2
    assert resampler.reset_calls == 1
    assert resampler.seen is not None
    torch.testing.assert_close(
        resampler.seen[0][:, 0, 0],
        torch.tensor([2.0, 12.0]),
    )
    assert resampler.seen[2:] == (0.5, 0.5)
    for result in out:
        torch.testing.assert_close(result["T_frag"], torch.full((1, 3), 12.0))
        assert int(result["initial_sample_index"]) == 1
        torch.testing.assert_close(
            torch.tensor([frame[0, 0].item() for frame in result["traj"]]),
            torch.tensor([10.0, 11.0, 12.0]),
        )


def test_fk_translation_sde_diversifies_resampled_clones() -> None:
    graph, lig_data, meta = _tiny_sampler_inputs()
    initial_T = torch.zeros(2, 1, 3)
    initial_q = torch.tensor([1.0, 0.0, 0.0, 0.0]).view(1, 1, 4).expand(2, 1, 4)

    class DuplicateSecondParticle:
        def resample(self, terminal_atom_pos, **kwargs):
            return torch.tensor([1, 1])

    sde_generator = torch.Generator(device="cpu")
    sde_generator.manual_seed(7)
    out = sample_unified(
        _ZeroSamplerModel(),
        graph,
        lig_data,
        meta,
        num_samples=2,
        num_steps=2,
        translation_sigma=0.5,
        time_schedule="uniform",
        device=torch.device("cpu"),
        initial_T_frag=initial_T,
        initial_q_frag=initial_q,
        fk_resample_times=[0.5],
        fk_resampler=DuplicateSecondParticle(),
        translation_sde_base_sigma=0.3,
        translation_sde_generator=sde_generator,
    )

    assert [int(result["initial_sample_index"]) for result in out] == [1, 1]
    assert not torch.equal(out[0]["T_frag"], out[1]["T_frag"])


def test_fk_resample_times_must_map_to_distinct_grid_states():
    graph, lig_data, meta = _tiny_sampler_inputs()

    class IdentityResampler:
        def resample(self, terminal_atom_pos, **kwargs):
            return torch.arange(terminal_atom_pos.shape[0])

    with pytest.raises(ValueError, match="same integration step"):
        sample_unified(
            _ZeroSamplerModel(),
            graph,
            lig_data,
            meta,
            num_samples=2,
            num_steps=2,
            time_schedule="uniform",
            fk_resample_times=[0.25, 0.4],
            fk_resampler=IdentityResampler(),
        )


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        ({"pose_objective": "vp_flow"}, "requires linear_fm"),
        ({"stochastic_gamma": 0.1}, "requires deterministic dynamics"),
        ({"guidance_scale": 0.1}, "gradient guidance are mutually exclusive"),
        (
            {"particle_resample_fn": lambda *args: torch.arange(2)},
            "callback particle resampling are mutually exclusive",
        ),
    ],
)
def test_fk_first_pilot_rejects_unimplemented_couplings(extra, message):
    graph, lig_data, meta = _tiny_sampler_inputs()

    class IdentityResampler:
        def resample(self, terminal_atom_pos, **kwargs):
            return torch.arange(terminal_atom_pos.shape[0])

    with pytest.raises(ValueError, match=message):
        sample_unified(
            _ZeroSamplerModel(),
            graph,
            lig_data,
            meta,
            num_samples=2,
            num_steps=2,
            time_schedule="uniform",
            fk_resample_times=[0.5],
            fk_resampler=IdentityResampler(),
            **extra,
        )


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        ({"pose_objective": "vp_flow"}, "requires linear_fm"),
        ({"stochastic_gamma": 0.1}, "mutually exclusive"),
        ({"guidance_scale": 0.1}, "gradient guidance"),
        ({"translation_sigma": 0.0}, "positive translation priors"),
    ],
)
def test_score_corrected_translation_sde_rejects_invalid_couplings(extra, message):
    graph, lig_data, meta = _tiny_sampler_inputs()

    with pytest.raises(ValueError, match=message):
        sample_unified(
            _ZeroSamplerModel(),
            graph,
            lig_data,
            meta,
            num_samples=2,
            num_steps=2,
            time_schedule="uniform",
            translation_sde_base_sigma=0.3,
            **extra,
        )


def test_shared_prior_pool_has_stable_nested_prefix() -> None:
    frag_sizes = torch.tensor([2, 1], dtype=torch.long)
    full_T, full_q = sample_shared_prior_states(
        100,
        2,
        frag_sizes,
        translation_sigma=0.5,
        seed=43,
    )
    repeat_T, repeat_q = sample_shared_prior_states(
        100,
        2,
        frag_sizes,
        translation_sigma=0.5,
        seed=43,
    )
    torch.testing.assert_close(full_T, repeat_T)
    torch.testing.assert_close(full_q, repeat_q)
    torch.testing.assert_close(full_T[:40], repeat_T[:40])
    torch.testing.assert_close(full_q[:50], repeat_q[:50])
    torch.testing.assert_close(
        full_q[:, 1],
        torch.tensor([1.0, 0.0, 0.0, 0.0]).expand(100, 4),
    )


def test_operator_split_guidance_runs_after_learned_proposal() -> None:
    graph, lig_data, meta = _tiny_sampler_inputs()
    initial_T = torch.zeros(1, 1, 3)
    initial_q = torch.tensor([1.0, 0.0, 0.0, 0.0]).view(1, 1, 4)

    class Corrector:
        def __init__(self):
            self.seen = None

        def correct(
            self,
            T_flat,
            q_flat,
            local_pos,
            fragment_id,
            frag_sizes_flat,
            *,
            dt,
            t,
            scale,
        ):
            self.seen = (T_flat.clone(), float(t), float(scale))
            return T_flat + 3.0, q_flat

    corrector = Corrector()
    out = sample_unified(
        _UnitTranslationModel(),
        graph,
        lig_data,
        meta,
        num_samples=1,
        num_steps=1,
        translation_sigma=1.0,
        time_schedule="uniform",
        device=torch.device("cpu"),
        initial_T_frag=initial_T,
        initial_q_frag=initial_q,
        guidance_fn=corrector,
        guidance_scale=0.1,
        guidance_min_t=0.5,
        guidance_operator_split=True,
    )

    assert corrector.seen is not None
    torch.testing.assert_close(corrector.seen[0], torch.full((1, 3), 2.0))
    assert corrector.seen[1:] == (1.0, 0.1)
    torch.testing.assert_close(out[0]["T_frag"], torch.full((1, 3), 5.0))


def test_operator_split_zero_scale_is_exact_sampler_noop() -> None:
    graph, lig_data, meta = _tiny_sampler_inputs()
    initial_T = torch.zeros(2, 1, 3)
    initial_q = torch.tensor([1.0, 0.0, 0.0, 0.0]).view(1, 1, 4).expand(2, 1, 4)

    class MustNotRun:
        def correct(self, *args, **kwargs):
            raise AssertionError("zero-scale corrector must not run")

    baseline = sample_unified(
        _UnitTranslationModel(),
        graph,
        lig_data,
        meta,
        num_samples=2,
        num_steps=2,
        translation_sigma=1.0,
        time_schedule="uniform",
        device=torch.device("cpu"),
        initial_T_frag=initial_T,
        initial_q_frag=initial_q,
    )
    zero_scale = sample_unified(
        _UnitTranslationModel(),
        graph,
        lig_data,
        meta,
        num_samples=2,
        num_steps=2,
        translation_sigma=1.0,
        time_schedule="uniform",
        device=torch.device("cpu"),
        initial_T_frag=initial_T,
        initial_q_frag=initial_q,
        guidance_fn=MustNotRun(),
        guidance_scale=0.0,
        guidance_operator_split=True,
    )
    for expected, actual in zip(baseline, zero_scale):
        torch.testing.assert_close(actual["T_frag"], expected["T_frag"])
        torch.testing.assert_close(actual["q_frag"], expected["q_frag"])


def test_nonzero_operator_split_requires_callback() -> None:
    graph, lig_data, meta = _tiny_sampler_inputs()
    with pytest.raises(ValueError, match="requires a guidance callback"):
        sample_unified(
            _ZeroSamplerModel(),
            graph,
            lig_data,
            meta,
            num_samples=1,
            num_steps=1,
            translation_sigma=1.0,
            device=torch.device("cpu"),
            guidance_scale=0.1,
            guidance_operator_split=True,
        )


def test_multi_sigma_operator_split_invokes_corrector() -> None:
    graph, lig_data, meta = _tiny_sampler_inputs()

    class CountingCorrector:
        def __init__(self):
            self.calls = 0

        def correct(
            self,
            T_flat,
            q_flat,
            local_pos,
            fragment_id,
            frag_sizes_flat,
            *,
            dt,
            t,
            scale,
        ):
            self.calls += 1
            assert T_flat.shape == (2, 3)
            assert float(scale) == 0.1
            return T_flat, q_flat

    corrector = CountingCorrector()
    result = sample_unified_multi_sigma(
        _ZeroSamplerModel(),
        graph,
        lig_data,
        meta,
        sigma_list=[0.5, 1.0],
        samples_per_sigma=[1, 1],
        num_steps=1,
        time_schedule="uniform",
        device=torch.device("cpu"),
        guidance_fn=corrector,
        guidance_scale=0.1,
        guidance_min_t=0.5,
        guidance_operator_split=True,
    )
    assert len(result) == 2
    assert corrector.calls == 1


def test_normalized_direct_drift_is_added_before_one_ode_integration() -> None:
    graph, lig_data, meta = _tiny_sampler_inputs()
    initial_T = torch.zeros(1, 1, 3)
    initial_q = torch.tensor([1.0, 0.0, 0.0, 0.0]).view(1, 1, 4)

    class DirectDrift:
        def __init__(self):
            self.seen = None

        def direct_velocity(
            self,
            atom_pos_flat,
            centers_flat,
            learned_translation_flat,
            learned_angular_flat,
            frag_sizes_flat,
            *,
            t_start,
            t_end,
            strength,
        ):
            self.seen = (
                atom_pos_flat.clone(),
                centers_flat.clone(),
                learned_translation_flat.clone(),
                learned_angular_flat.clone(),
                frag_sizes_flat.clone(),
                t_start,
                t_end,
                strength,
            )
            return torch.full_like(centers_flat, 0.3), torch.zeros_like(centers_flat)

        def correct(self, *args, **kwargs):
            raise AssertionError("direct drift must not invoke the post-step corrector")

    guidance = DirectDrift()
    out = sample_unified(
        _UnitTranslationModel(),
        graph,
        lig_data,
        meta,
        num_samples=1,
        num_steps=1,
        translation_sigma=1.0,
        time_schedule="uniform",
        device=torch.device("cpu"),
        initial_T_frag=initial_T,
        initial_q_frag=initial_q,
        guidance_fn=guidance,
        guidance_scale=0.1,
        guidance_min_t=0.0,
        guidance_direct_drift=True,
    )

    assert guidance.seen is not None
    torch.testing.assert_close(guidance.seen[0], torch.zeros(1, 3))
    torch.testing.assert_close(guidance.seen[1], torch.zeros(1, 3))
    torch.testing.assert_close(guidance.seen[2], torch.full((1, 3), 2.0))
    torch.testing.assert_close(guidance.seen[3], torch.zeros(1, 3))
    torch.testing.assert_close(guidance.seen[4], torch.ones(1, dtype=torch.long))
    assert guidance.seen[5:] == (0.0, 1.0, 0.1)
    torch.testing.assert_close(out[0]["T_frag"], torch.full((1, 3), 2.3))


def test_direct_and_operator_split_modes_are_mutually_exclusive() -> None:
    graph, lig_data, meta = _tiny_sampler_inputs()
    with pytest.raises(ValueError, match="mutually exclusive"):
        sample_unified(
            _ZeroSamplerModel(),
            graph,
            lig_data,
            meta,
            num_samples=1,
            num_steps=1,
            guidance_fn=object(),
            guidance_scale=0.1,
            guidance_operator_split=True,
            guidance_direct_drift=True,
        )
