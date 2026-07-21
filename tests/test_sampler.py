"""Tests for inference time discretization."""

import torch

from effdock.inference.sampler import build_time_grid, sample_unified, sample_unified_multi_sigma


class _ZeroSamplerModel(torch.nn.Module):
    def forward(self, batch):
        return {
            "v_pred": torch.zeros_like(batch["T_frag"]),
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
