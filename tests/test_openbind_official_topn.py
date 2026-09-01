from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from scripts.evaluate_openbind_official_topn import (  # noqa: E402
    assigned_ids,
    load_official_cohort,
    load_ranked_scores,
    split_sdf_records,
)


def test_official_cohort_reproduces_filtered_scaffold_contract(tmp_path: Path) -> None:
    path = tmp_path / "metadata.csv"
    rows = [
        ["a", "False", "False", "True", "False"],
        ["b", "False", "True", "True", "False"],
        ["c", "False", "False", "False", "False"],
        ["d", "False", "False", "True", "True"],
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "complex_name",
                "covalent",
                "fragment_screen",
                "pb_valid_prepared",
                "suspected_artefact",
            ]
        )
        writer.writerows(rows)
    assert load_official_cohort(path, expected_source_count=4, expected_cohort_count=1) == ["a"]


def test_confidence_ranking_is_stable_and_zero_based(tmp_path: Path) -> None:
    path = tmp_path / "scores.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "pose_index",
                "after_confidence_rmsd",
                "final_symmetry_rmsd_angstrom",
            ],
        )
        writer.writeheader()
        for pose_index in range(100):
            writer.writerow(
                {
                    "pose_index": pose_index,
                    "after_confidence_rmsd": 0.0 if pose_index in {3, 7} else pose_index + 1,
                    "final_symmetry_rmsd_angstrom": pose_index / 10,
                }
            )
    ranked = load_ranked_scores(path, top_n=5)
    assert [row["pose_index"] for row in ranked[:2]] == [3, 7]
    assert [row["rank"] for row in ranked] == list(range(5))


def test_assignment_is_disjoint_and_complete() -> None:
    cohort = [f"x{i:03d}" for i in range(17)]
    shards = [
        assigned_ids(cohort, num_shards=4, shard_index=i, max_complexes=None) for i in range(4)
    ]
    assert sorted(item for shard in shards for item in shard) == cohort
    assert sum(len(shard) for shard in shards) == len(cohort)


def test_sdf_record_split_preserves_delimiters(tmp_path: Path) -> None:
    path = tmp_path / "poses.sdf"
    path.write_bytes(b"\n$$$$\n".join([f"pose-{i}".encode() for i in range(100)]) + b"\n$$$$\n")
    records = split_sdf_records(path)
    assert len(records) == 100
    assert records[7].startswith(b"pose-7")
    assert records[7].endswith(b"$$$$\n")
