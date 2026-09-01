from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from effdock.workflows.guidance_budget_report import (
    CONDITIONS,
    DATASETS,
    EXPECTED_CHECKPOINT_SHA256,
    EXPECTED_CONFIG_SHA256,
    EXPECTED_GUIDANCE_PARAMETER_SHA256,
    EXPECTED_POCKET_CENTERS_SHA256,
    PROTOCOL_ID,
    build_report,
)


def _ids_sha256(ids: list[str]) -> str:
    payload = "".join(f"{complex_id}\n" for complex_id in sorted(ids))
    return hashlib.sha256(payload.encode()).hexdigest()


def _write_fixture(root: Path) -> tuple[Path, Path]:
    input_dir = root / "raw"
    input_dir.mkdir()
    ids_by_dataset = {
        "astex": ["a1", "a2"],
        "posebusters": ["p1", "p2"],
    }
    eligibility = {
        "protocol_id": PROTOCOL_ID,
        "datasets": {
            dataset: {
                "discovered": 2,
                "eligible": 2,
                "eligible_ids": ids,
                "eligible_ids_sha256": _ids_sha256(ids),
            }
            for dataset, ids in ids_by_dataset.items()
        },
    }
    eligibility_path = root / "eligibility.json"
    eligibility_path.write_text(json.dumps(eligibility))
    eligibility_sha256 = hashlib.sha256(eligibility_path.read_bytes()).hexdigest()

    for dataset in DATASETS:
        for condition_index, (_, num_samples, num_steps) in enumerate(CONDITIONS):
            for arm in ("unguided", "guided"):
                scale = 0.0 if arm == "unguided" else 0.1
                run_name = (
                    f"effdock-guidance-budget1000-v1-{dataset}-n{num_samples}-s{num_steps}-{arm}"
                )
                for shard_index, complex_id in enumerate(ids_by_dataset[dataset]):
                    csv_path = input_dir / f"{run_name}.shard-{shard_index}.csv"
                    base_rmsd = 2.4 - 0.4 * condition_index
                    rmsd = base_rmsd - (0.6 if arm == "guided" else 0.0)
                    joint = rmsd < 2.0
                    with csv_path.open("w", newline="") as handle:
                        writer = csv.DictWriter(
                            handle,
                            fieldnames=[
                                "id",
                                "oracle_rmsd",
                                "oracle_fast_valid",
                                "num_fast_valid_candidates",
                                "fast_valid_oracle_rmsd",
                                "joint_fast_valid_and_rmsd_lt2",
                                "prior_pool_size",
                                "sampling_seed",
                                "prior_pool_sha256",
                                "guidance_mode",
                                "guidance_parameter_sha256",
                            ],
                        )
                        writer.writeheader()
                        writer.writerow(
                            {
                                "id": complex_id,
                                "oracle_rmsd": rmsd + 0.1 * shard_index,
                                "oracle_fast_valid": joint,
                                "num_fast_valid_candidates": int(joint),
                                "fast_valid_oracle_rmsd": (
                                    rmsd + 0.1 * shard_index if joint else "inf"
                                ),
                                "joint_fast_valid_and_rmsd_lt2": joint,
                                "prior_pool_size": 100,
                                "sampling_seed": 100 + shard_index,
                                "prior_pool_sha256": hashlib.sha256(
                                    f"prior-{complex_id}".encode()
                                ).hexdigest(),
                                "guidance_mode": (
                                    "unified_operator_split"
                                    if arm == "guided"
                                    else "none"
                                ),
                                "guidance_parameter_sha256": (
                                    EXPECTED_GUIDANCE_PARAMETER_SHA256
                                    if arm == "guided"
                                    else ""
                                ),
                            }
                        )
                    summary = {
                        "run_name": run_name,
                        "protocol_id": PROTOCOL_ID,
                        "dataset": dataset,
                        "num_discovered_total": 2,
                        "num_assigned": 1,
                        "num_success": 1,
                        "num_failed": 0,
                        "num_samples": num_samples,
                        "num_steps": num_steps,
                        "model_pose_step_budget": 1000,
                        "num_shards": 2,
                        "shard_index": shard_index,
                        "seed": 7,
                        "unified_guidance_scale": scale,
                        "prior_pool_size": 100,
                        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
                        "confidence_checkpoint_sha256": None,
                        "config_sha256": EXPECTED_CONFIG_SHA256,
                        "pocket_centers_sha256": EXPECTED_POCKET_CENTERS_SHA256[
                            dataset
                        ],
                        "eligibility_manifest_sha256": eligibility_sha256,
                        "sigma": 0.5,
                        "time_schedule": "late",
                        "schedule_power": 3.0,
                        "pocket_cutoff": 10.0,
                        "center_jitter_sigma": 0.0,
                        "vina_guidance_scale": 0.0,
                        "unified_guidance_start_t": 0.5,
                        "unified_guidance_ramp_power": 1.0,
                        "unified_guidance_max_force": 20.0,
                        "unified_guidance_max_velocity": 5.0,
                        "unified_guidance_max_angular_velocity": 5.0,
                        "unified_guidance_max_atom_displacement": 0.25,
                        "unified_guidance_max_backtracks": 8,
                        "unified_guidance_protein_shell": 18.0,
                        "refine": "none",
                        "csv": str(csv_path),
                        "failures": [],
                    }
                    if arm == "guided":
                        summary["guidance_parameter_set"] = {
                            "sha256": EXPECTED_GUIDANCE_PARAMETER_SHA256
                        }
                        summary["guidance_operator_stats"] = {
                            "steps_attempted": 1,
                            "pose_corrections_attempted": 1,
                            "pose_corrections_accepted": 1,
                            "pose_corrections_rejected": 0,
                            "nonfinite_base_poses": 0,
                            "nonfinite_trials": 0,
                            "max_accepted_atom_displacement": 0.1,
                        }
                    summary_path = input_dir / f"{run_name}.shard-{shard_index}.summary.json"
                    summary_path.write_text(json.dumps(summary))
    return input_dir, eligibility_path


def test_build_report_strict_success_and_guided_budget_comparison(
    tmp_path: Path,
) -> None:
    input_dir, eligibility_path = _write_fixture(tmp_path)
    report = build_report(
        input_dir,
        eligibility_path,
        expected_shards=2,
        bootstrap_resamples=200,
    )

    assert report["status"] == "complete_strict_paired"
    assert report["datasets"]["astex"]["eligibility_coverage"]["coverage_pct"] == 100.0
    assert report["datasets"]["astex"]["prior_pairing"]["verified"]
    comparison = report["datasets"]["astex"]["cells"]["n100_s10"]["guided_vs_unguided"]
    assert comparison["common_ids"] == 2
    assert comparison["metrics"]["oracle_lt2"]["delta"] == 100.0
    guided_budget = report["datasets"]["posebusters"]["guided_budget_comparison"]
    assert guided_budget["common_ids"] == 2
    assert set(guided_budget["pairwise_deltas"]) == {
        "n50_s20_minus_n100_s10",
        "n40_s25_minus_n100_s10",
        "n40_s25_minus_n50_s20",
    }


@pytest.mark.parametrize("mismatch", ["metadata", "duplicate", "failure"])
def test_build_report_rejects_incomplete_or_unpaired_cells(
    tmp_path: Path,
    mismatch: str,
) -> None:
    input_dir, eligibility_path = _write_fixture(tmp_path)
    summaries = sorted(input_dir.glob("*astex*n100-s10-unguided*.summary.json"))
    if mismatch == "metadata":
        payload = json.loads(summaries[1].read_text())
        payload["seed"] = 99
        summaries[1].write_text(json.dumps(payload))
        match = "metadata mismatch"
    elif mismatch == "duplicate":
        payload = json.loads(summaries[1].read_text())
        with Path(payload["csv"]).open(newline="") as handle:
            row = next(csv.DictReader(handle))
        row["id"] = "a1"
        with Path(payload["csv"]).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
        match = "duplicate success row"
    else:
        payload = json.loads(summaries[1].read_text())
        payload["failures"] = [{"id": "a2", "error": "synthetic"}]
        payload["num_failed"] = 1
        payload["num_success"] = 0
        with Path(payload["csv"]).open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "id",
                    "oracle_rmsd",
                    "oracle_fast_valid",
                    "num_fast_valid_candidates",
                    "fast_valid_oracle_rmsd",
                    "joint_fast_valid_and_rmsd_lt2",
                    "prior_pool_size",
                    "sampling_seed",
                    "prior_pool_sha256",
                    "guidance_mode",
                    "guidance_parameter_sha256",
                ],
            )
            writer.writeheader()
        summaries[1].write_text(json.dumps(payload))
        match = "strict paired report rejects survivor-only"

    with pytest.raises(ValueError, match=match):
        build_report(
            input_dir,
            eligibility_path,
            expected_shards=2,
            bootstrap_resamples=20,
        )


def test_paired_bootstrap_ci_is_deterministic(tmp_path: Path) -> None:
    input_dir, eligibility_path = _write_fixture(tmp_path)
    first = build_report(
        input_dir,
        eligibility_path,
        expected_shards=2,
        bootstrap_seed=123,
        bootstrap_resamples=137,
    )
    second = build_report(
        input_dir,
        eligibility_path,
        expected_shards=2,
        bootstrap_seed=123,
        bootstrap_resamples=137,
    )
    first_ci = first["datasets"]["astex"]["cells"]["n100_s10"]["guided_vs_unguided"]["metrics"][
        "oracle_median_rmsd"
    ]["ci95"]
    second_ci = second["datasets"]["astex"]["cells"]["n100_s10"]["guided_vs_unguided"]["metrics"][
        "oracle_median_rmsd"
    ]["ci95"]
    assert first_ci == second_ci


def test_build_report_rejects_consistent_pocket_center_drift(tmp_path: Path) -> None:
    input_dir, eligibility_path = _write_fixture(tmp_path)
    for summary_path in input_dir.glob("*astex*.summary.json"):
        payload = json.loads(summary_path.read_text())
        payload["pocket_centers_sha256"] = "drifted-astex-centers"
        summary_path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="astex: protocol pocket_centers_sha256 mismatch"):
        build_report(
            input_dir,
            eligibility_path,
            expected_shards=2,
            bootstrap_resamples=20,
        )


def test_build_report_rejects_consistent_guidance_parameter_drift(
    tmp_path: Path,
) -> None:
    input_dir, eligibility_path = _write_fixture(tmp_path)
    for summary_path in input_dir.glob("*-guided*.summary.json"):
        payload = json.loads(summary_path.read_text())
        payload["guidance_parameter_set"]["sha256"] = "drifted-guidance-parameters"
        csv_path = Path(payload["csv"])
        with csv_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames
            rows = list(reader)
        assert fieldnames is not None
        for row in rows:
            row["guidance_parameter_sha256"] = "drifted-guidance-parameters"
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        summary_path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="protocol guidance parameter hash mismatch"):
        build_report(
            input_dir,
            eligibility_path,
            expected_shards=2,
            bootstrap_resamples=20,
        )
