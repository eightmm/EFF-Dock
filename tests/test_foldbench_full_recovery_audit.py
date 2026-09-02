from __future__ import annotations

import pytest

from scripts.audit_foldbench_full_recovery import audit_replayed_row


def _row() -> dict[str, str]:
    return {
        "id": "complex-a",
        "confidence_index": "7",
        "confidence_fast_valid": "True",
        "first_rmsd": "1.250000",
        "candidate_rmsds_json": "[1.25,2.5]",
        "confidence_candidate_scores_json": '[{"confidence_success":0.75}]',
        "all_poses_sdf_sha256": "old",
    }


def test_recovery_audit_accepts_bounded_gpu_replay_drift() -> None:
    previous = _row()
    current = dict(previous)
    current.update(
        {
            "first_rmsd": "1.250010",
            "candidate_rmsds_json": "[1.2501,2.5001]",
            "confidence_candidate_scores_json": '[{"confidence_success":0.78}]',
            "all_poses_sdf_sha256": "new",
        }
    )

    deltas, changed_hashes = audit_replayed_row(previous, current)

    assert deltas["candidate_rmsds_json"] == pytest.approx(1e-4)
    assert changed_hashes == ["all_poses_sdf_sha256"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("confidence_index", "8"),
        ("confidence_fast_valid", "False"),
        ("first_rmsd", "1.251"),
        ("candidate_rmsds_json", "[1.26,2.5]"),
    ],
)
def test_recovery_audit_rejects_semantic_or_large_numeric_change(
    field: str, value: str
) -> None:
    previous = _row()
    current = dict(previous)
    current[field] = value

    with pytest.raises(ValueError):
        audit_replayed_row(previous, current)
