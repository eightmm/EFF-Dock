from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from report_guidance_energy_confidence_raw_selection import (  # noqa: E402
    _raw_scores,
    _raw_selector_indices,
)


def test_raw_scores_are_candidate_count_invariant() -> None:
    short = _raw_scores([1.0, 2.0], [-2.0, 1.0], 10)
    extended = _raw_scores([1.0, 2.0, 100.0], [-2.0, 1.0, 1.0e6], 10)
    assert list(short) == list(extended)
    for name in short:
        assert extended[name][:2] == short[name]


def test_raw_selector_inventory_and_stable_tie_break() -> None:
    selectors = _raw_selector_indices([1.0, 1.0], [0.0, 0.0], 10)
    assert len(selectors) == 13
    assert set(selectors.values()) == {0}


def test_per_atom_score_has_declared_formula() -> None:
    scores = _raw_scores([2.0], [-20.0], 10)
    assert scores["raw_total_l0p05"] == pytest.approx([1.0])
    assert scores["raw_per_atom_l0p5"] == pytest.approx([1.0])


@pytest.mark.parametrize("heavy_atoms", [0, -1])
def test_raw_scores_reject_invalid_heavy_atom_count(heavy_atoms: int) -> None:
    with pytest.raises(ValueError, match="heavy atom"):
        _raw_scores([1.0], [0.0], heavy_atoms)
