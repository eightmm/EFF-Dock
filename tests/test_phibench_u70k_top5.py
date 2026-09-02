from __future__ import annotations

from scripts.evaluate_phibench_u70k_top5_posebusters_shard import rank_indices
from scripts.report_phibench_u70k_top5 import aggregate


def test_rank_indices_uses_stage_score_and_stable_pose_index() -> None:
    rows = [
        {
            "pose_index": index,
            "before_confidence_rmsd": before,
            "after_confidence_rmsd": after,
        }
        for index, (before, after) in enumerate(
            ((2.0, 0.5), (1.0, 0.5), (1.0, 3.0), (4.0, 1.0))
        )
    ]
    assert rank_indices(rows, "raw") == [1, 2, 0, 3]
    assert rank_indices(rows, "refined") == [0, 1, 3, 2]


def test_aggregate_reproduces_top1_and_orders_top5() -> None:
    rows = []
    for index in range(203):
        rows.append(
            {
                "raw_top1_rmsd": 1.0 if index < 128 else 3.0,
                "raw_top5_best_rmsd": 1.0 if index < 160 else 3.0,
                "raw_oracle_rmsd": 1.0 if index < 180 else 3.0,
                "refined_top1_rmsd": 1.0 if index < 131 else 3.0,
                "refined_top5_best_rmsd": 1.0 if index < 170 else 3.0,
                "refined_oracle_rmsd": 1.0 if index < 179 else 3.0,
                "refined_top1_posebusters_valid": index < 184,
                "refined_top5_posebusters_valid": index < 195,
                "refined_top1_joint": index < 120,
                "refined_top5_joint": index < 160,
            }
        )
    result = aggregate(rows)
    assert result["counts"]["raw_top5_rmsd_lt2"] == 160
    assert result["counts"]["refined_top5_rmsd_lt2"] == 170
    assert result["counts"]["refined_top5_posebusters_valid"] == 195
    assert result["counts"]["refined_top5_joint_posebusters_valid_rmsd_lt2"] == 160
    assert result["top5_rescue"] == {
        "raw_rmsd_count": 32,
        "refined_rmsd_count": 39,
        "refined_joint_count": 40,
    }
