from __future__ import annotations

import numpy as np
import pytest

from effdock.workflows.fk_sde_posebusters_report import (
    NUM_SAMPLES,
    _paired_metrics,
    _sampling_tag,
    pose_diversity_metrics,
)


def test_sampling_tag_matches_evaluator_single_and_multi_shard_names() -> None:
    assert _sampling_tag("smoke", shard_index=0, num_shards=1) == "smoke"
    assert (
        _sampling_tag("full", shard_index=3, num_shards=8)
        == "full.shard-003-of-008"
    )


def test_pose_diversity_metrics_separate_clone_and_pose_spread() -> None:
    coordinates = np.zeros((NUM_SAMPLES, 3, 3), dtype=np.float64)
    heavy_mask = np.asarray([True, True, False])

    collapsed = pose_diversity_metrics(coordinates, heavy_mask)

    assert collapsed["coordinate_unique_count"] == 1
    assert collapsed["pairwise_heavy_atom_rmsd_median"] == 0.0
    assert collapsed["nearest_neighbor_heavy_atom_rmsd_median"] == 0.0

    coordinates[:, :, 0] = np.arange(NUM_SAMPLES)[:, None] * 0.1
    spread = pose_diversity_metrics(coordinates, heavy_mask)

    assert spread["coordinate_unique_count"] == NUM_SAMPLES
    assert spread["pairwise_heavy_atom_rmsd_median"] > 1.0
    assert spread["pairwise_heavy_atom_rmsd_ge2_fraction"] > 0.25
    assert spread["nearest_neighbor_heavy_atom_rmsd_median"] == pytest.approx(0.1)


def _cell(
    *,
    selected_joint: bool,
    selected_valid: bool,
    selected_rmsd_success: bool,
    oracle_joint: bool,
    oracle_rmsd: bool,
    selected_rmsd: float,
    valid_candidates: int,
    joint_candidates: int,
    unique: int,
    pairwise: float,
    nearest: float,
) -> dict:
    return {
        "selected_joint": selected_joint,
        "selected_pb_valid": selected_valid,
        "selected_rmsd_success": selected_rmsd_success,
        "oracle_joint": oracle_joint,
        "oracle_rmsd_success": oracle_rmsd,
        "selected_rmsd": selected_rmsd,
        "valid_candidates": valid_candidates,
        "joint_candidates": joint_candidates,
        "diversity": {
            "coordinate_unique_count": unique,
            "pairwise_heavy_atom_rmsd_median": pairwise,
            "nearest_neighbor_heavy_atom_rmsd_median": nearest,
        },
    }


def test_paired_metrics_preserve_complex_pairing() -> None:
    baseline = {
        "a": _cell(
            selected_joint=True,
            selected_valid=True,
            selected_rmsd_success=True,
            oracle_joint=True,
            oracle_rmsd=True,
            selected_rmsd=1.0,
            valid_candidates=10,
            joint_candidates=5,
            unique=10,
            pairwise=0.5,
            nearest=0.1,
        ),
        "b": _cell(
            selected_joint=False,
            selected_valid=False,
            selected_rmsd_success=False,
            oracle_joint=False,
            oracle_rmsd=True,
            selected_rmsd=3.0,
            valid_candidates=5,
            joint_candidates=1,
            unique=20,
            pairwise=1.0,
            nearest=0.2,
        ),
    }
    comparison = {
        "a": _cell(
            selected_joint=False,
            selected_valid=True,
            selected_rmsd_success=False,
            oracle_joint=True,
            oracle_rmsd=True,
            selected_rmsd=2.2,
            valid_candidates=12,
            joint_candidates=4,
            unique=40,
            pairwise=1.5,
            nearest=0.4,
        ),
        "b": _cell(
            selected_joint=True,
            selected_valid=True,
            selected_rmsd_success=True,
            oracle_joint=True,
            oracle_rmsd=True,
            selected_rmsd=1.5,
            valid_candidates=9,
            joint_candidates=3,
            unique=40,
            pairwise=2.0,
            nearest=0.5,
        ),
    }

    result = _paired_metrics(baseline, comparison)

    assert result["selected_joint_valid_rmsd_le2_count_delta"] == 0
    assert result["selected_joint_gained_complexes"] == 1
    assert result["selected_joint_lost_complexes"] == 1
    assert result["selected_posebusters_valid_pct_delta"] == pytest.approx(50.0)
    assert result["candidate_posebusters_valid_pct_delta"] == pytest.approx(7.5)
    assert result["terminal_unique_coordinate_fraction_delta"] == pytest.approx(0.625)
    assert result["pairwise_heavy_atom_rmsd_median_paired_delta"] == pytest.approx(1.0)
