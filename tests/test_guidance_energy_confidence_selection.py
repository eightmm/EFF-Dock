from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from report_guidance_energy_confidence_selection import (  # noqa: E402
    _selector_indices,
    _stable_ordinal_quality,
)


def _values() -> list[float]:
    return [float(index) for index in range(100)]


def test_ordinal_quality_is_stable_and_lower_is_better() -> None:
    values = _values()
    quality, ranks = _stable_ordinal_quality(values)
    assert ranks == list(range(100))
    assert quality[0] == 1.0
    assert quality[-1] == 0.01


def test_selector_inventory_and_extremes() -> None:
    confidence = _values()
    energy = list(reversed(_values()))
    selectors = _selector_indices(confidence, energy)
    assert selectors["confidence"] == 0
    assert selectors["energy"] == 99
    assert len(selectors) == 15
    assert list(selectors) == [
        "confidence",
        "energy",
        "rank_add_a05",
        "rank_geo_a05",
        "rank_add_a10",
        "rank_geo_a10",
        "rank_add_a25",
        "rank_geo_a25",
        "rank_add_a50",
        "rank_geo_a50",
        "rank_add_a75",
        "rank_geo_a75",
        "energy_filter_q25",
        "energy_filter_q50",
        "energy_filter_q75",
    ]


def test_energy_filter_keeps_energy_eligible_then_uses_confidence() -> None:
    confidence = _values()
    energy = list(reversed(_values()))
    selectors = _selector_indices(confidence, energy)
    assert selectors["energy_filter_q25"] == 75
    assert selectors["energy_filter_q50"] == 50
    assert selectors["energy_filter_q75"] == 25
