from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from scripts import report_early_time_t0p10_50k_external_paired as paired_report
from scripts.report_early_time_t0p10_50k_external_paired import (
    ARMS,
    EXPECTED_COUNTS,
    FROZEN_BENCHMARK_INPUT_HEAVY_ATOM_POLICY,
    FROZEN_BENCHMARK_INPUT_IDENTITY_SCHEMA,
    FROZEN_BENCHMARK_INPUT_MANIFEST_SHA256,
    FROZEN_CHECKPOINT_SHA256,
    FROZEN_CONFIG_SHA256,
    FROZEN_ELIGIBILITY_MANIFEST_SHA256,
    FROZEN_POCKET_CENTERS_SHA256,
    FROZEN_PRIOR_POOL_HASH_CONTRACT,
    FROZEN_PROTOCOL_ID,
    FROZEN_SAMPLING_DYNAMICS_CONTRACT,
    REPLAY_ARM,
    build_report,
    write_report,
)

NUM_SAMPLES = 10
NUM_STEPS = 2
PRIOR_POOL_SIZE = 10
K2_BY_ARM = {
    "current_raw": (0, 1, 4, 9),
    "parent_ema": (0, 1, 5, 10),
    "t0p10_50k_ema": (1, 0, 7, 10),
}


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _candidate_rmsds(k2: int) -> list[float]:
    return [1.0 + 0.01 * index for index in range(k2)] + [
        2.0 + 0.01 * index for index in range(NUM_SAMPLES - k2)
    ]


def _benchmark_input_identity(dataset: str, ids: list[str]) -> dict[str, object]:
    discovered = EXPECTED_COUNTS[dataset]
    filler_count = discovered - len(ids)
    all_ids = [*ids, *(f"zz_{dataset}_{index:04d}" for index in range(filler_count))]
    return {
        "schema_version": FROZEN_BENCHMARK_INPUT_IDENTITY_SCHEMA,
        "mode": "frozen_manifest",
        "dataset": dataset,
        "heavy_atom_policy": FROZEN_BENCHMARK_INPUT_HEAVY_ATOM_POLICY,
        "count": discovered,
        "ids_sha256": _digest(f"{dataset}:ids"),
        "mapping_sha256": _digest(f"{dataset}:mapping"),
        "sha256": _digest(f"{dataset}:identity"),
        "sources": {
            "frozen_manifest": {
                "path": "docs/GUIDANCE_BUDGET1000_FULL_INPUTS.json",
                "sha256": FROZEN_BENCHMARK_INPUT_MANIFEST_SHA256,
            }
        },
        "per_id": {
            complex_id: {"sha256": _digest(f"{dataset}:{complex_id}:ligand-input")}
            for complex_id in all_ids
        },
    }


def _write_fixture(
    root: Path,
    *,
    datasets: tuple[str, ...] = ("astex", "posebusters"),
    count: int = 4,
    shards: int = 2,
    include_replay: bool = False,
) -> dict[tuple[str, str, int], Path]:
    root.mkdir(parents=True)
    paths: dict[tuple[str, str, int], Path] = {}
    arms = (*ARMS, REPLAY_ARM) if include_replay else ARMS
    for dataset in datasets:
        ids = [f"{dataset[0]}{index}" for index in range(count)]
        benchmark_input_identity = _benchmark_input_identity(dataset, ids)
        input_files: dict[str, tuple[Path, Path]] = {}
        for complex_id in ids:
            protein_path = root / "inputs" / dataset / f"{complex_id}_protein.pdb"
            ligand_ref_path = root / "inputs" / dataset / f"{complex_id}_ligand.sdf"
            protein_path.parent.mkdir(parents=True, exist_ok=True)
            protein_path.write_text(f"protein:{dataset}:{complex_id}\n", encoding="utf-8")
            ligand_ref_path.write_text(
                f"ligand-reference:{dataset}:{complex_id}\n", encoding="utf-8"
            )
            input_files[complex_id] = (protein_path, ligand_ref_path)
        for arm in arms:
            source_arm = "current_raw" if arm == REPLAY_ARM else arm
            for shard in range(shards):
                assigned = ids[shard::shards]
                rows = []
                for complex_id in assigned:
                    index = ids.index(complex_id)
                    k2 = K2_BY_ARM[source_arm][index % len(K2_BY_ARM[source_arm])]
                    rmsds = _candidate_rmsds(k2)
                    fast_k2 = max(0, k2 - 1)
                    protein_path, ligand_ref_path = input_files[complex_id]
                    all_poses_path = (
                        root / "poses" / dataset / arm / f"{complex_id}_all_poses.sdf"
                    )
                    all_poses_path.parent.mkdir(parents=True, exist_ok=True)
                    all_poses_path.write_text(
                        f"all-poses:{dataset}:{source_arm}:{complex_id}\n",
                        encoding="utf-8",
                    )
                    rows.append(
                        {
                            "id": complex_id,
                            "protein": str(protein_path),
                            "ligand_ref": str(ligand_ref_path),
                            "protein_sha256": _digest(
                                f"protein:{dataset}:{complex_id}\n"
                            ),
                            "ligand_reference_sha256": _digest(
                                f"ligand-reference:{dataset}:{complex_id}\n"
                            ),
                            "ligand_input_identity_sha256": (
                                benchmark_input_identity["per_id"][complex_id]["sha256"]
                            ),
                            "all_poses_sdf": str(all_poses_path),
                            "all_poses_sdf_sha256": _digest(
                                f"all-poses:{dataset}:{source_arm}:{complex_id}\n"
                            ),
                            "all_poses_count": NUM_SAMPLES,
                            "num_samples": NUM_SAMPLES,
                            "first_index": 0,
                            "first_rmsd": rmsds[0],
                            "candidate_rmsds_json": json.dumps(rmsds, separators=(",", ":")),
                            "num_rmsd_lt2_candidates": k2,
                            "fraction_rmsd_lt2_candidates": k2 / NUM_SAMPLES,
                            "num_fast_valid_candidates": fast_k2,
                            "num_fast_valid_rmsd_lt2_candidates": fast_k2,
                            "oracle_rmsd": min(rmsds),
                            "mean_sample_rmsd": sum(rmsds) / len(rmsds),
                            "prior_pool_size": PRIOR_POOL_SIZE,
                            "sampling_seed": 43 + index,
                            "prior_pool_sha256": _digest(f"{dataset}:{complex_id}:prior"),
                            "guidance_mode": "none",
                            "sampling_dynamics": "deterministic_ode",
                            "translation_sde_base_sigma": 0.0,
                            "full_heavy_atom_bijection": True,
                        }
                    )
                run_name = f"early-time-{dataset}-{arm.replace('_', '-')}"
                tag = f"{run_name}.shard-{shard:03d}-of-{shards:03d}"
                csv_path = root / f"{tag}.csv"
                with csv_path.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
                summary_path = root / f"{tag}.summary.json"
                summary_path.write_text(
                    json.dumps(
                        {
                            "dataset": dataset,
                            "run_name": run_name,
                            "protocol_id": FROZEN_PROTOCOL_ID,
                            "num_shards": shards,
                            "shard_index": shard,
                            "expected_discovered_count": EXPECTED_COUNTS[dataset],
                            "num_discovered_total": EXPECTED_COUNTS[dataset],
                            "num_assigned": len(rows),
                            "num_success": len(rows),
                            "num_failed": 0,
                            "failures": [],
                            "num_samples": NUM_SAMPLES,
                            "num_steps": NUM_STEPS,
                            "model_pose_step_budget": NUM_SAMPLES * NUM_STEPS,
                            "sigma": 2.0,
                            "sigma_list": [],
                            "sigma_counts": [],
                            "time_schedule": "late",
                            "schedule_power": 3.0,
                            "pocket_cutoff": 10.0,
                            "center_jitter_sigma": 0.0,
                            "refine": "none",
                            "seed": 42,
                            "confidence_checkpoint": None,
                            "confidence_checkpoint_sha256": None,
                            "confidence_step": None,
                            "sampling_dynamics_contract": (
                                FROZEN_SAMPLING_DYNAMICS_CONTRACT
                            ),
                            "translation_sde_base_sigma": 0.0,
                            "vina_guidance_scale": 0.0,
                            "unified_guidance_scale": 0.0,
                            "prior_pool_size": PRIOR_POOL_SIZE,
                            "prior_pool_hash_contract": FROZEN_PRIOR_POOL_HASH_CONTRACT,
                            "require_full_ligand_atom_mapping": True,
                            "require_complete_success": True,
                            "benchmark_input_identity": benchmark_input_identity,
                            "checkpoint_sha256": FROZEN_CHECKPOINT_SHA256[arm],
                            "config_sha256": FROZEN_CONFIG_SHA256,
                            "eligibility_manifest_sha256": (
                                FROZEN_ELIGIBILITY_MANIFEST_SHA256
                            ),
                            "pocket_centers_sha256": FROZEN_POCKET_CENTERS_SHA256[
                                dataset
                            ],
                            "csv": str(csv_path),
                        }
                    ),
                    encoding="utf-8",
                )
                paths[(dataset, arm, shard)] = csv_path
    return paths


def _rewrite_csv(path: Path, mutation) -> None:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    mutation(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _rewrite_summary(csv_path: Path, mutation) -> None:
    path = csv_path.with_suffix(".summary.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutation(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_report_aggregates_k2_and_paired_effects_deterministically(tmp_path: Path) -> None:
    _write_fixture(tmp_path / "raw")
    kwargs = {
        "expected_counts": {"astex": 4, "posebusters": 4},
        "expected_shards": 2,
        "expected_num_samples": NUM_SAMPLES,
        "expected_num_steps": NUM_STEPS,
        "expected_prior_pool_size": PRIOR_POOL_SIZE,
        "bootstrap_seed": 7,
        "bootstrap_resamples": 200,
    }
    first = build_report(tmp_path / "raw", **kwargs)
    second = build_report(tmp_path / "raw", **kwargs)

    assert first["status"] == "complete_strict_three_arm_paired"
    assert first["configuration"]["k2_definition"] == (
        "count of candidates with runtime pose RMSD strictly < 2 Angstrom"
    )
    assert first["configuration"]["rmsd_implementation"] == {
        "preferred": (
            "RDKit rdMolAlign.CalcRMS symmetry-aware heavy-atom RMSD without "
            "alignment when the full-topology path succeeds"
        ),
        "fallback": (
            "index-wise RMSD over the full-heavy-atom mapped subset when "
            "CalcRMS is unavailable or fails"
        ),
        "metric_caveat": (
            "K2 uses the stored runtime RMSD values; fallback cases are not "
            "retrospectively reclassified as symmetry-aware"
        ),
    }
    assert first["integrity"]["summary_files_used"] == 12
    assert first["integrity"]["pairing"]["astex"] == {
        "complexes": 4,
        "sampling_seed_equal_across_arms": True,
        "prior_pool_sha256_equal_across_arms": True,
        "protein_sha256_equal_across_arms": True,
        "ligand_reference_sha256_equal_across_arms": True,
        "ligand_input_identity_sha256_equal_across_arms": True,
    }
    t0_astex = first["arms"]["t0p10_50k_ema"]["datasets"]["astex"]
    assert t0_astex["k2_total"] == 18
    assert t0_astex["k2_mean"] == 4.5
    assert t0_astex["p_k2_ge_1_pct"] == 75.0
    assert t0_astex["p_k2_ge_5_pct"] == 50.0
    assert t0_astex["p_k2_ge_10_pct"] == 25.0
    assert t0_astex["first_rmsd_mean"] == 1.25
    assert t0_astex["first_rmsd_lt2_pct"] == 75.0
    assert first["arms"]["t0p10_50k_ema"]["combined"]["k2_total"] == 36

    primary = first["comparisons"]["parent_ema_to_t0p10_50k_ema"]["combined"]
    assert primary["delta_total_k2"] == 4
    assert primary["delta_mean_k2"] == 0.5
    assert primary["positive_complexes"] == 4
    assert primary["negative_complexes"] == 2
    assert primary["tied_complexes"] == 2
    assert primary["k2_ge_1_gained_complexes"] == 2
    assert primary["k2_ge_1_lost_complexes"] == 2
    assert primary["net_k2_ge_1"] == 0
    assert primary["delta_first_rmsd_mean"] == 0.0
    assert (
        primary["paired_bootstrap_ci95"]
        == second["comparisons"]["parent_ema_to_t0p10_50k_ema"]["combined"][
            "paired_bootstrap_ci95"
        ]
    )
    assert first["engineering_stage"] == {
        "scope": "combined_astex_posebusters",
        "basis": "engineering_integrity_only_no_outcome_gate",
        "status": "complete_all_requested_datasets",
        "next_stage": None,
        "metric_thresholds": None,
    }


def test_astex_only_smoke_accepts_replay_and_cli_count_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = tmp_path / "raw"
    _write_fixture(
        raw,
        datasets=("astex",),
        count=1,
        shards=1,
        include_replay=True,
    )
    output = tmp_path / "report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "report",
            "--output-root",
            str(raw),
            "--output-json",
            str(output),
            "--datasets",
            "astex",
            "--expected-astex-count",
            "1",
            "--expected-shards",
            "1",
            "--expected-num-samples",
            str(NUM_SAMPLES),
            "--expected-num-steps",
            str(NUM_STEPS),
            "--expected-prior-pool-size",
            str(PRIOR_POOL_SIZE),
            "--bootstrap-resamples",
            "20",
        ],
    )
    paired_report.main()
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["configuration"]["datasets"] == ["astex"]
    assert result["configuration"]["optional_replay_arm_present"] is True
    assert result["integrity"]["summary_files_used"] == 4
    assert result["integrity"]["current_raw_replay"]["astex"] == {
        "complexes": 1,
        "exact_k2_equal_per_id": True,
        "k2_ge_1_equal_per_id": True,
    }
    assert "posebusters" not in result["arms"]["current_raw"]["datasets"]
    assert result["engineering_stage"]["next_stage"] == (
        "run_posebusters_after_integrity_pass"
    )


def test_report_rejects_cross_arm_seed_or_prior_mismatch(tmp_path: Path) -> None:
    paths = _write_fixture(tmp_path / "raw")
    _rewrite_csv(
        paths[("astex", "t0p10_50k_ema", 0)],
        lambda row: row.__setitem__("sampling_seed", "999"),
    )
    with pytest.raises(ValueError, match="expected frozen per-ID seed"):
        build_report(
            tmp_path / "raw",
            expected_counts={"astex": 4, "posebusters": 4},
            expected_shards=2,
            expected_num_samples=NUM_SAMPLES,
            expected_num_steps=NUM_STEPS,
            expected_prior_pool_size=PRIOR_POOL_SIZE,
            bootstrap_resamples=10,
        )

    _rewrite_csv(
        paths[("astex", "t0p10_50k_ema", 0)],
        lambda row: (
            row.__setitem__("sampling_seed", "43"),
            row.__setitem__("prior_pool_sha256", "f" * 64),
        ),
    )
    with pytest.raises(ValueError, match="prior_pool_sha256 differs across arms"):
        build_report(
            tmp_path / "raw",
            expected_counts={"astex": 4, "posebusters": 4},
            expected_shards=2,
            expected_num_samples=NUM_SAMPLES,
            expected_num_steps=NUM_STEPS,
            expected_prior_pool_size=PRIOR_POOL_SIZE,
            bootstrap_resamples=10,
        )


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("protocol_id", "WRONG-PROTOCOL", "protocol_id: expected"),
        ("checkpoint_sha256", "f" * 64, "checkpoint_sha256: expected frozen"),
        ("sigma", 1.5, "sigma: expected 2.0"),
        ("num_steps", 99, "num_steps: expected"),
        ("prior_pool_size", 99, "prior_pool_size: expected"),
    ],
)
def test_report_rejects_wrong_frozen_protocol_checkpoint_or_setting(
    tmp_path: Path, field: str, value: object, error: str
) -> None:
    paths = _write_fixture(tmp_path / "raw")
    _rewrite_summary(
        paths[("astex", "current_raw", 0)],
        lambda summary: summary.__setitem__(field, value),
    )
    with pytest.raises(ValueError, match=error):
        build_report(
            tmp_path / "raw",
            expected_counts={"astex": 4, "posebusters": 4},
            expected_shards=2,
            expected_num_samples=NUM_SAMPLES,
            expected_num_steps=NUM_STEPS,
            expected_prior_pool_size=PRIOR_POOL_SIZE,
            bootstrap_resamples=10,
        )


def test_smoke_still_requires_full_discovery_count(tmp_path: Path) -> None:
    paths = _write_fixture(
        tmp_path / "raw",
        datasets=("astex",),
        count=1,
        shards=1,
        include_replay=True,
    )
    _rewrite_summary(
        paths[("astex", "current_raw", 0)],
        lambda summary: summary.__setitem__("num_discovered_total", 1),
    )
    with pytest.raises(ValueError, match="num_discovered_total: expected exactly 85"):
        build_report(
            tmp_path / "raw",
            datasets=("astex",),
            expected_counts={"astex": 1},
            expected_shards=1,
            expected_num_samples=NUM_SAMPLES,
            expected_num_steps=NUM_STEPS,
            expected_prior_pool_size=PRIOR_POOL_SIZE,
            bootstrap_resamples=10,
        )

@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("num_rmsd_lt2_candidates", "9", "strict RMSD <2 count mismatch"),
        (
            "candidate_rmsds_json",
            json.dumps([float("inf")] + [3.0] * (NUM_SAMPLES - 1)),
            "metric is non-finite",
        ),
    ],
)
def test_report_recomputes_k2_and_rejects_nonfinite_metrics(
    tmp_path: Path, field: str, value: str, error: str
) -> None:
    paths = _write_fixture(tmp_path / "raw")
    _rewrite_csv(
        paths[("astex", "current_raw", 0)],
        lambda row: row.__setitem__(field, value),
    )
    with pytest.raises(ValueError, match=error):
        build_report(
            tmp_path / "raw",
            expected_counts={"astex": 4, "posebusters": 4},
            expected_shards=2,
            expected_num_samples=NUM_SAMPLES,
            expected_num_steps=NUM_STEPS,
            expected_prior_pool_size=PRIOR_POOL_SIZE,
            bootstrap_resamples=10,
        )


def test_report_rejects_failed_or_incomplete_shards(tmp_path: Path) -> None:
    _write_fixture(tmp_path / "raw")
    summary = next((tmp_path / "raw").glob("*astex-current-raw*.summary.json"))
    payload = json.loads(summary.read_text(encoding="utf-8"))
    payload["num_failed"] = 1
    payload["failures"] = [{"id": "a0"}]
    summary.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="recorded evaluation failures"):
        build_report(
            tmp_path / "raw",
            expected_counts={"astex": 4, "posebusters": 4},
            expected_shards=2,
            expected_num_samples=NUM_SAMPLES,
            expected_num_steps=NUM_STEPS,
            expected_prior_pool_size=PRIOR_POOL_SIZE,
            bootstrap_resamples=10,
        )


def test_report_rejects_replay_k2_drift(tmp_path: Path) -> None:
    paths = _write_fixture(
        tmp_path / "raw",
        datasets=("astex",),
        count=1,
        shards=1,
        include_replay=True,
    )
    path = paths[("astex", REPLAY_ARM, 0)]

    def mutate(row: dict[str, str]) -> None:
        rmsds = _candidate_rmsds(1)
        row["candidate_rmsds_json"] = json.dumps(rmsds)
        row["num_rmsd_lt2_candidates"] = "1"
        row["fraction_rmsd_lt2_candidates"] = "0.1"
        row["oracle_rmsd"] = str(min(rmsds))
        row["mean_sample_rmsd"] = str(sum(rmsds) / len(rmsds))
        row["first_rmsd"] = str(rmsds[0])

    _rewrite_csv(path, mutate)
    with pytest.raises(ValueError, match="current_raw replay K2 mismatch"):
        build_report(
            tmp_path / "raw",
            datasets=("astex",),
            expected_counts={"astex": 1},
            expected_shards=1,
            expected_num_samples=NUM_SAMPLES,
            expected_num_steps=NUM_STEPS,
            expected_prior_pool_size=PRIOR_POOL_SIZE,
            bootstrap_resamples=10,
        )


def test_atomic_writer_refuses_overwrite_unless_explicit(tmp_path: Path) -> None:
    output = tmp_path / "report.json"
    write_report({"value": 1}, output)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_report({"value": 2}, output)
    assert json.loads(output.read_text(encoding="utf-8")) == {"value": 1}
    write_report({"value": 2}, output, overwrite=True)
    assert json.loads(output.read_text(encoding="utf-8")) == {"value": 2}
    assert not list(tmp_path.glob("*.tmp"))
