from __future__ import annotations

import csv
import json

from effdock.workflows.guidance_budget_posebusters_report import VALIDITY_CHECKS
from effdock.workflows.guidance_sigma_sweep_posebusters_report import (
    DATASETS,
    SIGMADOCK_LEGACY_CHECKS,
    load_expected_ids,
    run_name,
    summarize_cell,
    transition_summary,
)


def test_sigmadock_legacy_profile_excludes_only_no_radicals() -> None:
    assert set(VALIDITY_CHECKS) - set(SIGMADOCK_LEGACY_CHECKS) == {"no_radicals"}
    assert len(SIGMADOCK_LEGACY_CHECKS) == 26


def test_load_expected_ids_accepts_manifest_mapping(tmp_path) -> None:
    manifest = {
        "datasets": {
            dataset: {"ligands": {f"{dataset}-{index}": {} for index in range(count)}}
            for dataset, count in DATASETS.items()
        }
    }
    path = tmp_path / "inputs.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    ids = load_expected_ids(path)
    assert {dataset: len(values) for dataset, values in ids.items()} == DATASETS


def test_summarize_cell_reports_strict_and_legacy_views(tmp_path) -> None:
    name = run_name("astex", 1.0)
    path = tmp_path / f"{name}.csv"
    rows = [
        {"id": "a", "confidence_rmsd": "1.0"},
        {"id": "b", "confidence_rmsd": "3.0"},
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    checks_a = {key: True for key in VALIDITY_CHECKS}
    checks_a["no_radicals"] = False
    checks_b = {key: True for key in VALIDITY_CHECKS}
    official = {
        "a": {"posebusters_valid": False, "checks": checks_a},
        "b": {"posebusters_valid": True, "checks": checks_b},
    }
    metrics, outcomes = summarize_cell(
        tmp_path,
        dataset="astex",
        sigma=1.0,
        selector="confidence",
        expected_ids=("a", "b"),
        official_rows=official,
    )
    assert metrics["selected_rmsd_lt2_pct"] == 50.0
    assert metrics["posebusters_valid_27_pct"] == 50.0
    assert metrics["posebusters_valid_sigmadock_legacy26_pct"] == 100.0
    assert metrics["joint_rmsd_lt2_and_posebusters_valid_27_pct"] == 0.0
    assert metrics[
        "joint_rmsd_lt2_and_posebusters_valid_sigmadock_legacy26_pct"
    ] == 50.0
    assert outcomes["a"]["joint_sigmadock_legacy26"] is True


def test_transition_summary_is_paired_by_complex_id() -> None:
    baseline = {
        "a": {"rmsd_lt2": False, "pb_valid_27": True, "joint_27": False},
        "b": {"rmsd_lt2": True, "pb_valid_27": True, "joint_27": True},
    }
    comparison = {
        "a": {"rmsd_lt2": True, "pb_valid_27": True, "joint_27": True},
        "b": {"rmsd_lt2": False, "pb_valid_27": False, "joint_27": False},
    }
    result = transition_summary(baseline, comparison, ("a", "b"))
    assert result["joint_27"] == {
        "false_to_true": 1,
        "true_to_false": 1,
        "both_true": 0,
        "both_false": 0,
    }
