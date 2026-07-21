from __future__ import annotations

import torch

from effdock.workflows.evaluate import _sample_center_jitter


def test_center_jitter_is_paired_by_seed_and_scale() -> None:
    one_a = _sample_center_jitter(seed=43, sigma=1.0)
    two_a = _sample_center_jitter(seed=43, sigma=2.0)

    torch.testing.assert_close(two_a, 2.0 * one_a)


def test_center_jitter_does_not_consume_global_sampling_rng() -> None:
    torch.manual_seed(43)
    expected = torch.randn(8)

    torch.manual_seed(43)
    _sample_center_jitter(seed=43, sigma=2.0)
    observed = torch.randn(8)

    torch.testing.assert_close(observed, expected)
