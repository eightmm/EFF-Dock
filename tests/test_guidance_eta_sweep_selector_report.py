from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from effdock.workflows.guidance_budget_posebusters_report import VALIDITY_CHECKS
from effdock.workflows.guidance_eta_sweep_report import PROTOCOL_ID as PARENT_PROTOCOL_ID
from effdock.workflows.guidance_eta_sweep_selector_binding import (
    PROTOCOL_ID,
    build_binding,
)
from effdock.workflows.guidance_eta_sweep_selector_report import (
    aggregate_selector_cell,
    paired_outcomes,
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _binding_fixture(tmp_path: Path, *, selector: str) -> tuple[Path, Path, str]:
    sampling_dir = tmp_path / "raw"
    official_dir = tmp_path / "official" / selector
    run_name = "selector-test-run"
    row = {
        "id": "case",
        "protein_sha256": "1" * 64,
        "ligand_reference_sha256": "2" * 64,
        "saved_pose_sha256_json": json.dumps(
            {"first": "3" * 64, "vina": "4" * 64, "oracle": "5" * 64}
        ),
    }
    _write_csv(sampling_dir / f"{run_name}.csv", [row])
    tag = "shard-000-of-001"
    (sampling_dir / f"{run_name}.{tag}.summary.json").write_text(
        json.dumps(
            {
                "protocol_id": PARENT_PROTOCOL_ID,
                "run_name": run_name,
                "dataset": "astex",
                "shard_index": 0,
                "num_shards": 1,
                "unified_guidance_scale": 0.1,
            }
        )
    )
    official_dir.mkdir(parents=True)
    official_csv = official_dir / f"{tag}.csv"
    _write_csv(official_csv, [{"id": "case"}])
    (official_dir / f"{tag}.summary.json").write_text(
        json.dumps(
            {
                "selector": selector,
                "input_hashes_verified": True,
                "num_input_hashes_verified": 1,
                "num_discovered_total": 1,
                "num_assigned": 1,
                "num_success": 1,
                "num_failed": 0,
                "failures": [],
            }
        )
    )
    return sampling_dir, official_dir, run_name


def test_selector_binding_is_selector_specific_and_rejects_summary_mismatch(
    tmp_path: Path,
) -> None:
    ledgers: dict[str, str] = {}
    for selector in ("first", "vina"):
        sampling_dir, official_dir, run_name = _binding_fixture(
            tmp_path / selector,
            selector=selector,
        )
        binding = build_binding(
            sampling_dir=sampling_dir,
            official_dir=official_dir,
            run_name=run_name,
            protocol_id=PROTOCOL_ID,
            dataset="astex",
            eta=0.1,
            selector=selector,
            shard_index=0,
            num_shards=1,
        )
        assert binding["selector"] == selector
        ledgers[selector] = binding["sampling_input_pose_ledger_sha256"]
    assert ledgers["first"] != ledgers["vina"]

    sampling_dir, official_dir, run_name = _binding_fixture(
        tmp_path / "mismatch",
        selector="vina",
    )
    with pytest.raises(ValueError, match="selector mismatch"):
        build_binding(
            sampling_dir=sampling_dir,
            official_dir=official_dir,
            run_name=run_name,
            protocol_id=PROTOCOL_ID,
            dataset="astex",
            eta=0.1,
            selector="first",
            shard_index=0,
            num_shards=1,
        )


def _official_cell(tmp_path: Path, *, selector: str = "first") -> Path:
    cell_dir = tmp_path / "cell"
    tag = "shard-000-of-001"
    csv_path = cell_dir / f"{tag}.csv"
    row: dict[str, object] = {
        "id": "case",
        "posebusters_valid": True,
        "rmsd_≤_2å": True,
    }
    row.update({check: True for check in VALIDITY_CHECKS})
    _write_csv(csv_path, [row])
    (cell_dir / f"{tag}.summary.json").write_text(
        json.dumps(
            {
                "posebusters_version": "0.6.5",
                "config": "redock",
                "selector": selector,
                "num_discovered_total": 1,
                "num_assigned": 1,
                "num_success": 1,
                "num_failed": 0,
                "posebusters_valid_pct": 100.0,
                "failures": [],
                "csv": str(csv_path),
            }
        )
    )
    return cell_dir


def test_selector_cell_rejects_survivor_only_and_selector_mismatch(tmp_path: Path) -> None:
    cell_dir = _official_cell(tmp_path)
    rows, aggregate, _ = aggregate_selector_cell(
        cell_dir,
        ("case",),
        run_name="run",
        dataset="astex",
        eta_tag="eta0000",
        selector="first",
        expected_shards=1,
    )
    assert set(rows) == {"case"}
    assert aggregate["eligible_coverage_pct"] == 100.0

    summary_path = cell_dir / "shard-000-of-001.summary.json"
    summary = json.loads(summary_path.read_text())
    summary.update(
        {
            "num_success": 0,
            "num_failed": 1,
            "failures": [{"id": "case", "error": "boom"}],
        }
    )
    summary_path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="survivor-only"):
        aggregate_selector_cell(
            cell_dir,
            ("case",),
            run_name="run",
            dataset="astex",
            eta_tag="eta0000",
            selector="first",
            expected_shards=1,
        )


def test_paired_selector_metrics_keep_complex_alignment() -> None:
    baseline = {
        "a": {
            "posebusters_valid": False,
            "selected_rmsd": 3.0,
            "selected_rmsd_lt2": False,
            "joint_selected_rmsd_lt2_and_posebusters_valid": False,
        },
        "b": {
            "posebusters_valid": True,
            "selected_rmsd": 1.0,
            "selected_rmsd_lt2": True,
            "joint_selected_rmsd_lt2_and_posebusters_valid": True,
        },
    }
    comparison = {
        "a": {
            "posebusters_valid": True,
            "selected_rmsd": 1.5,
            "selected_rmsd_lt2": True,
            "joint_selected_rmsd_lt2_and_posebusters_valid": True,
        },
        "b": baseline["b"],
    }
    report = paired_outcomes(
        baseline,
        comparison,
        ("a", "b"),
        baseline_label="first",
        comparison_label="vina",
        seed=7,
        resamples=100,
    )
    assert report["metrics"]["posebusters_valid"]["delta"] == 50.0
    assert report["metrics"]["selected_rmsd_lt2"]["delta"] == 50.0
    assert report["transitions"]["posebusters_valid"]["false_to_true"] == 1
