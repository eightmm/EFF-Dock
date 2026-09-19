import pytest

from scripts.collect_paper_runtime import stats
from scripts.paper_runtime_figures import normalize_cost


def test_missing_is_not_zero():
    result = stats([])
    assert result["count"] == 0 and result["mean"] is None


def test_measured_statistics():
    result = stats([1.0, 3.0, 8.0])
    assert result["mean"] == 4.0 and result["median"] == 3.0 and result["max"] == 8.0


@pytest.mark.parametrize("values", [[-1], [float("nan")], [float("inf")]])
def test_invalid_measurements(values):
    with pytest.raises(ValueError):
        stats(values)


@pytest.mark.parametrize("arm,n", [("unguided_n100_s10", 100), ("unguided_n40_s25", 40)])
def test_cost_denominators_and_memory_not_divided(arm, n):
    entry = dict(
        arm=arm,
        pipeline_mean_s=100.0,
        pipeline_repeat_sd_s=2.0,
        refinement_mean_s=70.0,
        confidence_both_banks_mean_s=10.0,
        refinement_compute_mean_s=60.0,
        confidence_forward_both_mean_s=4.0,
        sampling_allocator_peak_gib=12.0,
    )
    result = normalize_cost(entry)
    assert result["pipeline_amortized_s_per_generated_pose"] == 100 / n
    assert result["pipeline_amortized_repeat_sd_s_per_generated_pose"] == 2 / n
    assert result["refinement_amortized_s_per_pose"] == 70 / n
    assert result["confidence_amortized_s_per_pose_evaluation"] == 10 / (2 * n)
    assert result["confidence_forward_amortized_s_per_pose_evaluation"] == 4 / (2 * n)
    assert result["sampling_allocator_peak_gib"] == 12
    assert "poses_per_complex" not in entry
