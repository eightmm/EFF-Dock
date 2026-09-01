from __future__ import annotations

from pathlib import Path

import pytest

from scripts.report_s50_raw_refined_confidence_temporal_external import (
    paired_comparison,
)
from scripts.run_s50_raw_refined_confidence_temporal_external_shard import (
    output_shard_path,
)


def _row(complex_id: str, **values: object) -> dict[str, object]:
    return {
        "dataset": "phibench",
        "id": complex_id,
        "raw_selected_rmsd_lt2": False,
        "refined_selected_rmsd_lt2": False,
        "pl_valid": False,
        "posebusters_valid": False,
        "joint_pl_valid_rmsd_lt2": False,
        "joint_posebusters_valid_rmsd_lt2": False,
        "refined_selected_rmsd": 3.0,
        **values,
    }


def test_output_shard_path_separates_checkpoint_arms() -> None:
    root = Path("/tmp/results")
    u70 = output_shard_path(root, "full", "u070000", "phibench", 2, 13)
    u100 = output_shard_path(root, "full", "u100000", "phibench", 2, 13)
    assert u70 != u100
    assert u70 == root / "full/u070000/shards/phibench.shard-002-of-013.json"


def test_paired_comparison_counts_gains_and_losses() -> None:
    rows = {
        "u070000": [
            _row("a", refined_selected_rmsd_lt2=False, refined_selected_rmsd=2.1),
            _row("b", refined_selected_rmsd_lt2=True, refined_selected_rmsd=1.8),
        ],
        "u100000": [
            _row("a", refined_selected_rmsd_lt2=True, refined_selected_rmsd=1.9),
            _row("b", refined_selected_rmsd_lt2=False, refined_selected_rmsd=2.2),
        ],
    }
    result = paired_comparison(rows, "phibench")
    paired = result["boolean_metrics"]["refined_selected_rmsd_lt2"]
    assert paired == {
        "gains": 1,
        "losses": 1,
        "net": 0,
        "unchanged_pass": 0,
        "unchanged_fail": 0,
    }
    assert result["mean_refined_selected_rmsd_delta"] == pytest.approx(0.1)


def test_paired_comparison_rejects_id_mismatch() -> None:
    rows = {
        "u070000": [_row("a")],
        "u100000": [_row("b")],
    }
    with pytest.raises(ValueError, match="paired ID mismatch"):
        paired_comparison(rows, "phibench")
