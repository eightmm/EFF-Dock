from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from scripts.report_external_temporal_benchmark import (  # noqa: E402
    parse_bool,
    summarize,
)
from scripts.run_external_temporal_benchmark_shard import (  # noqa: E402
    sampling_paths,
    source_tag,
)


def test_sampling_paths_bind_dataset_and_shard() -> None:
    root = Path("/tmp/results")
    assert source_tag("openbind").endswith("openbind-n100-s10-sigma2-eta2")
    csv_path, summary_path = sampling_paths(
        root, "openbind", num_shards=54, shard_index=7
    )
    assert csv_path.name.endswith(".shard-007-of-054.csv")
    assert summary_path.name.endswith(".shard-007-of-054.summary.json")
    smoke_csv, smoke_summary = sampling_paths(
        root, "openbind", num_shards=1, shard_index=0
    )
    assert ".shard-" not in smoke_csv.name
    assert smoke_summary.name == smoke_csv.with_suffix(".summary.json").name


def test_report_summary_keeps_raw_refined_validity_endpoints_separate() -> None:
    rows = [
        {
            "dataset": "openbind",
            "id": "a",
            "raw_selected_rmsd": 3.0,
            "refined_selected_rmsd": 1.0,
            "raw_selected_rmsd_lt2": False,
            "refined_selected_rmsd_lt2": True,
            "raw_oracle_rmsd": 1.5,
            "refined_oracle_rmsd": 0.8,
            "pl_valid": True,
            "posebusters_valid": False,
            "joint_pl_valid_rmsd_lt2": True,
            "joint_posebusters_valid_rmsd_lt2": False,
            "mean_refinement_terminal_step": 75.0,
        },
        {
            "dataset": "openbind",
            "id": "b",
            "raw_selected_rmsd": 1.5,
            "refined_selected_rmsd": 2.5,
            "raw_selected_rmsd_lt2": True,
            "refined_selected_rmsd_lt2": False,
            "raw_oracle_rmsd": 2.5,
            "refined_oracle_rmsd": 1.5,
            "pl_valid": False,
            "posebusters_valid": False,
            "joint_pl_valid_rmsd_lt2": False,
            "joint_posebusters_valid_rmsd_lt2": False,
            "mean_refinement_terminal_step": 100.0,
        },
    ]
    result = summarize("openbind", rows)
    assert result["counts"]["raw_top1_rmsd_lt2"] == 1
    assert result["counts"]["refined_top1_rmsd_lt2"] == 1
    assert result["counts"]["refined_oracle_rmsd_lt2"] == 2
    assert result["counts"]["refined_top1_pl_valid"] == 1
    assert result["percent"]["refined_top1_joint_pl_valid_rmsd_lt2"] == 50.0


def test_csv_boolean_parser_is_strict() -> None:
    assert parse_bool("True") is True
    assert parse_bool("False") is False
    with pytest.raises(ValueError, match="invalid boolean"):
        parse_bool("1")
