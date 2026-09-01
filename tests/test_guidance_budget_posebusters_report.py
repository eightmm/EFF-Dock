from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from effdock.workflows.guidance_budget_posebusters_report import (
    EXPECTED_SELECTOR,
    MODULE_CHECKS,
    POSEBUSTERS_CONFIG,
    POSEBUSTERS_VERSION,
    VALIDITY_CHECKS,
    build_report,
    main,
)
from effdock.workflows.guidance_budget_report import (
    CONDITIONS,
    DATASETS,
    PROTOCOL_ID,
)

_RMSD_CHECK = "rmsd_≤_2å"


def _ids_sha256(ids: list[str]) -> str:
    payload = "".join(f"{complex_id}\n" for complex_id in sorted(ids))
    return hashlib.sha256(payload.encode()).hexdigest()


def _run_name(dataset: str, num_samples: int, num_steps: int, arm: str) -> str:
    return f"effdock-guidance-budget1000-v1-{dataset}-n{num_samples}-s{num_steps}-{arm}"


def _write_fixture(root: Path) -> tuple[Path, Path, dict[str, list[str]]]:
    input_dir = root / "official"
    input_dir.mkdir()
    ids_by_dataset = {"astex": ["a1", "a2"], "posebusters": ["p1", "p2"]}
    eligibility = {
        "protocol_id": PROTOCOL_ID,
        "datasets": {
            dataset: {
                "discovered": 3,
                "eligible": 2,
                "excluded": 1,
                "eligibility_pct": 2 / 3 * 100.0,
                "eligible_ids": ids,
                "eligible_ids_sha256": _ids_sha256(ids),
                "failure_codes": {"unsupported": 1},
            }
            for dataset, ids in ids_by_dataset.items()
        },
    }
    eligibility_path = root / "eligibility.json"
    eligibility_path.write_text(json.dumps(eligibility))

    for dataset in DATASETS:
        for condition_index, (_, num_samples, num_steps) in enumerate(CONDITIONS):
            for arm in ("unguided", "guided"):
                run_name = _run_name(dataset, num_samples, num_steps, arm)
                run_dir = input_dir / run_name
                run_dir.mkdir()
                for shard_index, complex_id in enumerate(ids_by_dataset[dataset]):
                    if arm == "unguided":
                        valid = shard_index == 1
                    elif condition_index == 0:
                        valid = True
                    elif condition_index == 1:
                        valid = shard_index == 1
                    else:
                        valid = False
                    checks = {key: True for key in VALIDITY_CHECKS}
                    if not valid:
                        checks["minimum_distance_to_protein"] = False
                    row = {
                        "id": complex_id,
                        "posebusters_valid": valid,
                        **checks,
                        # This deliberately differs from pass-all for one valid pose.
                        _RMSD_CHECK: not (condition_index == 0 and shard_index == 0),
                    }
                    tag = f"shard-{shard_index:03d}-of-002"
                    csv_path = run_dir / f"{tag}.csv"
                    with csv_path.open("w", newline="") as handle:
                        writer = csv.DictWriter(handle, fieldnames=list(row))
                        writer.writeheader()
                        writer.writerow(row)
                    summary = {
                        "posebusters_version": POSEBUSTERS_VERSION,
                        "config": POSEBUSTERS_CONFIG,
                        "selector": EXPECTED_SELECTOR,
                        "num_discovered_total": 2,
                        "num_assigned": 1,
                        "num_success": 1,
                        "num_failed": 0,
                        "posebusters_valid_pct": 100.0 if valid else 0.0,
                        "failures": [],
                        "csv": str(csv_path),
                    }
                    (run_dir / f"{tag}.summary.json").write_text(json.dumps(summary))
    return input_dir, eligibility_path, ids_by_dataset


def test_build_report_strict_success_modules_and_budget_comparisons(tmp_path: Path) -> None:
    input_dir, eligibility_path, _ = _write_fixture(tmp_path)
    report = build_report(
        input_dir,
        eligibility_path,
        expected_shards=2,
        bootstrap_resamples=200,
    )

    assert report["status"] == "complete_strict_paired_official_posebusters"
    coverage = report["datasets"]["astex"]["eligibility_coverage"]
    assert coverage["full_dataset_discovered"] == 3
    assert coverage["chemistry_eligible"] == 2
    assert coverage["chemistry_excluded"] == 1
    assert coverage["eligible_coverage_pct"] == 100.0
    cell = report["datasets"]["astex"]["cells"]["n100_s10"]
    assert cell["unguided"]["posebusters_valid_pct"] == 50.0
    assert cell["guided"]["posebusters_valid_pct"] == 100.0
    assert cell["guided"]["check_pass_pct"][_RMSD_CHECK] == 50.0
    assert cell["guided"]["module_pass_pct"]["distance_to_protein"] == 100.0
    assert set(cell["guided"]["module_pass_pct"]) == set(MODULE_CHECKS)
    effect = cell["guided_vs_unguided"]
    assert effect["posebusters_valid"]["delta"] == 50.0
    assert effect["transitions"]["invalid_to_valid"] == 1
    budget = report["datasets"]["posebusters"]["guided_budget_comparison"]
    assert budget["cell_posebusters_valid_pct"] == {
        "n100_s10": 100.0,
        "n50_s20": 50.0,
        "n40_s25": 0.0,
    }
    assert set(budget["pairwise_deltas"]) == {
        "n50_s20_minus_n100_s10",
        "n40_s25_minus_n100_s10",
        "n40_s25_minus_n50_s20",
    }


def test_paired_bootstrap_ci_is_deterministic(tmp_path: Path) -> None:
    input_dir, eligibility_path, _ = _write_fixture(tmp_path)
    kwargs = {
        "expected_shards": 2,
        "bootstrap_seed": 123,
        "bootstrap_resamples": 137,
    }
    first = build_report(input_dir, eligibility_path, **kwargs)
    second = build_report(input_dir, eligibility_path, **kwargs)
    first_metric = first["datasets"]["astex"]["cells"]["n100_s10"][
        "guided_vs_unguided"
    ]["posebusters_valid"]
    second_metric = second["datasets"]["astex"]["cells"]["n100_s10"][
        "guided_vs_unguided"
    ]["posebusters_valid"]
    assert first_metric["ci95"] == second_metric["ci95"]


def test_cli_writes_strict_json(tmp_path: Path) -> None:
    input_dir, eligibility_path, _ = _write_fixture(tmp_path)
    output_path = tmp_path / "report.json"
    main(
        [
            "--input-dir",
            str(input_dir),
            "--eligibility",
            str(eligibility_path),
            "--output",
            str(output_path),
            "--expected-shards",
            "2",
            "--bootstrap-resamples",
            "20",
        ]
    )

    result = json.loads(output_path.read_text())
    assert result["status"] == "complete_strict_paired_official_posebusters"
    assert result["bootstrap"]["resamples"] == 20


@pytest.mark.parametrize("mismatch", ["missing_shard", "selector", "validity"])
def test_build_report_rejects_incomplete_or_drifted_official_output(
    tmp_path: Path,
    mismatch: str,
) -> None:
    input_dir, eligibility_path, _ = _write_fixture(tmp_path)
    run_dir = input_dir / _run_name("astex", 100, 10, "guided")
    summary_path = run_dir / "shard-001-of-002.summary.json"
    if mismatch == "missing_shard":
        summary_path.rename(run_dir / "missing.summary.disabled")
        match = "expected exactly 2 shard summaries"
    elif mismatch == "selector":
        payload = json.loads(summary_path.read_text())
        payload["selector"] = "not-oracle"
        summary_path.write_text(json.dumps(payload))
        match = "selector must be 'oracle'"
    else:
        payload = json.loads(summary_path.read_text())
        csv_path = Path(payload["csv"])
        with csv_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames
            rows = list(reader)
        assert fieldnames is not None
        rows[0]["posebusters_valid"] = "False"
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        match = "does not equal the non-RMSD pass-all conjunction"

    with pytest.raises((ValueError, FileNotFoundError), match=match):
        build_report(
            input_dir,
            eligibility_path,
            expected_shards=2,
            bootstrap_resamples=20,
        )


@pytest.mark.parametrize("mismatch", ["duplicate", "outside", "failure"])
def test_build_report_rejects_nonexact_eligibility_and_failures(
    tmp_path: Path,
    mismatch: str,
) -> None:
    input_dir, eligibility_path, ids_by_dataset = _write_fixture(tmp_path)
    run_dir = input_dir / _run_name("posebusters", 50, 20, "unguided")
    summary_path = run_dir / "shard-001-of-002.summary.json"
    payload = json.loads(summary_path.read_text())
    csv_path = Path(payload["csv"])
    if mismatch in {"duplicate", "outside"}:
        with csv_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames
            rows = list(reader)
        assert fieldnames is not None
        rows[0]["id"] = ids_by_dataset["posebusters"][0] if mismatch == "duplicate" else "x"
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        match = "duplicate success ID" if mismatch == "duplicate" else "shard ID mismatch"
    else:
        payload["num_success"] = 0
        payload["num_failed"] = 1
        payload["failures"] = [{"id": ids_by_dataset["posebusters"][1], "error": "boom"}]
        payload["posebusters_valid_pct"] = None
        payload["csv"] = None
        summary_path.write_text(json.dumps(payload))
        match = "strict report rejects survivor-only"

    with pytest.raises(ValueError, match=match):
        build_report(
            input_dir,
            eligibility_path,
            expected_shards=2,
            bootstrap_resamples=20,
        )
