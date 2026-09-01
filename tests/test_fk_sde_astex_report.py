from __future__ import annotations

import pytest

from effdock.workflows.fk_sde_astex_report import paired_metrics


def _rows(selected: list[float], oracle: list[float], unique: list[int]) -> dict[str, dict]:
    return {
        f"id-{index}": {
            "selected_rmsd": selected[index],
            "oracle_rmsd": oracle[index],
            "coordinate_unique_count": unique[index],
        }
        for index in range(len(selected))
    }


def test_paired_metrics_preserve_pairing_and_threshold_direction() -> None:
    baseline = _rows([1.0, 2.5, 2.2], [1.0, 1.5, 2.2], [3, 10, 20])
    comparison = _rows([2.1, 1.5, 1.8], [1.1, 2.5, 1.8], [40, 40, 40])

    result = paired_metrics(baseline, comparison)

    assert result["confidence_selected_rmsd_lt2_count_delta"] == 1
    assert result["confidence_selected_rmsd_lt2_pct_delta"] == pytest.approx(100.0 / 3.0)
    assert result["confidence_selected_gained_complexes"] == 2
    assert result["confidence_selected_lost_complexes"] == 1
    assert result["confidence_selected_paired_median_rmsd_delta"] == pytest.approx(-0.4)
    assert result["oracle_rmsd_lt2_count_delta"] == 0
    assert result["terminal_unique_coordinate_fraction_delta"] == pytest.approx(87.0 / 120.0)


def test_paired_metrics_reject_id_mismatch() -> None:
    with pytest.raises(ValueError, match="different ID sets"):
        paired_metrics(
            _rows([1.0], [1.0], [40]),
            {"different": {"selected_rmsd": 1.0, "oracle_rmsd": 1.0}},
        )
