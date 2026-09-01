import pytest
import torch

from effdock.data.dataset import sample_flow_time


def _draw(distribution: str, count: int = 20_000) -> torch.Tensor:
    generator = torch.Generator().manual_seed(20260813)
    return torch.tensor([sample_flow_time(distribution, generator=generator) for _ in range(count)])


def test_simplefold_early_replay_matches_registered_mixture() -> None:
    samples = _draw("simplefold_early_replay")

    assert bool(((samples >= 0.0) & (samples <= 1.0)).all())
    assert float((samples == 0.0).float().mean()) == pytest.approx(0.10, abs=0.01)
    assert float((samples <= 0.30).float().mean()) == pytest.approx(0.335, abs=0.02)


def test_t0_dose_treatment_reallocates_early_mass_without_increasing_it() -> None:
    generator = torch.Generator().manual_seed(20260814)
    samples = torch.tensor(
        [
            sample_flow_time(
                "simplefold_early_replay",
                generator=generator,
                early_replay_weights=(0.80, 0.05, 0.15),
            )
            for _ in range(20_000)
        ]
    )

    assert float((samples == 0.0).float().mean()) == pytest.approx(0.15, abs=0.01)
    # The explicit exact-zero + U(0, 0.3) budget remains 20%; only its
    # allocation changes relative to the 80/10/10 control.
    assert float((samples <= 0.30).float().mean()) == pytest.approx(0.335, abs=0.02)


def test_simplefold_control_has_no_exact_zero_mass() -> None:
    samples = _draw("simplefold", count=5_000)
    assert not bool((samples == 0.0).any())


@pytest.mark.parametrize(
    ("weights", "max_time", "match"),
    [
        ((0.8, 0.2), 0.3, "must contain"),
        ((0.8, 0.15, 0.10), 0.3, "sum to 1"),
        ((0.8, 0.15, 0.05), 0.0, "must lie in"),
    ],
)
def test_simplefold_early_replay_rejects_invalid_parameters(
    weights: tuple[float, ...], max_time: float, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        sample_flow_time(
            "simplefold_early_replay",
            early_replay_weights=weights,  # type: ignore[arg-type]
            early_replay_max_time=max_time,
        )
