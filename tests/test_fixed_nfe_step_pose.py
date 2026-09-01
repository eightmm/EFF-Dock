from __future__ import annotations

from pathlib import Path

import pytest

from scripts.build_fixed_nfe_step_pose_manifest import _sampling_artifact_paths
from scripts.evaluate_fixed_nfe_step_pose_pb import (
    PL_VALIDITY_CHECKS,
    VALIDITY_CHECKS,
    _validated_checks,
)
from scripts.evaluate_fixed_nfe_step_pose_pb import (
    _metric as _pb_metric,
)
from scripts.evaluate_fixed_nfe_step_pose_pb import (
    _transition as _pb_transition,
)
from scripts.report_fixed_nfe_step_pose import _aggregate_stage
from scripts.run_fixed_nfe_step_pose_stage import _records


def _score_rows(rmsds: list[float], scores: list[float]) -> list[dict[str, str]]:
    return [
        {
            "final_symmetry_rmsd_angstrom": str(rmsd),
            "after_confidence_rmsd": str(score),
        }
        for rmsd, score in zip(rmsds, scores, strict=True)
    ]


def test_cumulative_oracle_uses_sampling_order_and_stable_confidence() -> None:
    first_rmsd = [3.0] * 40
    first_rmsd[4] = 1.5
    first_scores = [4.0] * 40
    first_scores[4] = 0.5
    second_rmsd = [3.0] * 40
    second_rmsd[39] = 1.0
    second_scores = [4.0] * 40
    second_scores[0] = 0.5
    metrics, successes = _aggregate_stage(
        [
            _score_rows(first_rmsd, first_scores),
            _score_rows(second_rmsd, second_scores),
        ],
        rmsd_field="final_symmetry_rmsd_angstrom",
        confidence_field="after_confidence_rmsd",
    )
    assert successes[:4] == [0, 0, 0, 0]
    assert successes[4:39] == [1] * 35
    assert successes[39] == 2
    assert metrics["selected_lt2_count"] == 1
    assert metrics["oracle_lt2_count"] == 2
    assert metrics["oracle_at_40_lt2_pct"] == 100.0


def test_stage_manifest_requires_exact_40_pose_records() -> None:
    manifest = {
        "protocol_id": "EFFDOCK-FIXED-NFE-STEP-POSE-REFINEMENT-INPUT-V1",
        "status": "complete",
        "expected_complexes": 2,
        "records": [
            {"dataset": "astex", "id": "a", "pose_count": 40},
            {"dataset": "posebusters", "id": "p", "pose_count": 40},
        ],
    }
    assert _records(manifest, shard_index=0, num_shards=2)[0]["id"] == "a"
    manifest["records"][1]["pose_count"] = 39
    with pytest.raises(ValueError, match="non-40-pose"):
        _records(manifest, shard_index=0, num_shards=2)


def test_sampling_artifact_paths_follow_evaluator_shard_convention() -> None:
    root = Path("sampling")
    smoke_csv, smoke_summary = _sampling_artifact_paths(
        root,
        stem="run",
        mode="smoke",
        shard=0,
        shards=1,
    )
    assert smoke_csv == root / "run.csv"
    assert smoke_summary == root / "run.summary.json"

    full_csv, full_summary = _sampling_artifact_paths(
        root,
        stem="run",
        mode="full",
        shard=3,
        shards=8,
    )
    assert full_csv == root / "run.shard-003-of-008.csv"
    assert full_summary == root / "run.shard-003-of-008.summary.json"


def test_smoke_sampling_artifact_paths_reject_non_single_shard() -> None:
    with pytest.raises(ValueError, match="exactly one shard"):
        _sampling_artifact_paths(
            Path("sampling"),
            stem="run",
            mode="smoke",
            shard=1,
            shards=2,
        )


def test_fixed_nfe_pb_schema_keeps_21_pl_checks_and_27_official_checks() -> None:
    raw = {name: True for name in VALIDITY_CHECKS}
    raw["rmsd_≤_2å"] = True
    rmsd_name, checks = _validated_checks(raw, "synthetic")
    assert rmsd_name == "rmsd_≤_2å"
    assert all(checks.values())
    assert len(VALIDITY_CHECKS) == 27
    assert len(PL_VALIDITY_CHECKS) == 21


def test_fixed_nfe_pb_metrics_and_paired_transitions() -> None:
    rows = [
        {
            "selected_rmsd_angstrom": 1.0,
            "rmsd_lt2": True,
            "pl_valid": True,
            "joint_pl_valid_rmsd_lt2": True,
            "posebusters_valid": False,
            "joint_posebusters_valid_rmsd_lt2": False,
            **{name: name != "minimum_distance_to_waters" for name in VALIDITY_CHECKS},
        },
        {
            "selected_rmsd_angstrom": 3.0,
            "rmsd_lt2": False,
            "pl_valid": True,
            "joint_pl_valid_rmsd_lt2": False,
            "posebusters_valid": True,
            "joint_posebusters_valid_rmsd_lt2": False,
            **{name: True for name in VALIDITY_CHECKS},
        },
    ]
    metrics = _pb_metric(rows)
    assert metrics["rmsd_lt2_pct"] == 50.0
    assert metrics["pl_valid_pct"] == 100.0
    assert metrics["posebusters_valid_pct"] == 50.0
    assert metrics["joint_pl_valid_rmsd_lt2_pct"] == 50.0

    keyed = {
        ("astex", "a", "s10_n100", "raw"): {**rows[0], "rmsd_lt2": False},
        ("astex", "a", "s25_n40", "raw"): rows[0],
    }
    transition = _pb_transition(
        keyed,
        [("astex", "a")],
        ("s10_n100", "raw"),
        ("s25_n40", "raw"),
    )
    assert transition["rmsd_lt2"]["false_to_true"] == 1
    assert transition["rmsd_lt2"]["delta_percentage_points"] == 100.0
