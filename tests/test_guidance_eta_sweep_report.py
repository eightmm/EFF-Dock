from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from effdock.workflows import guidance_eta_sweep_official_binding as official_binding
from effdock.workflows import guidance_eta_sweep_posebusters_report as official_report
from effdock.workflows import guidance_eta_sweep_report as report


def _trace_step(interval: dict[str, float], eta: float) -> dict[str, float | int | None]:
    model_sum = 200.0
    applied_sum = 20.0
    total_sum = 201.0
    step: dict[str, float | int | None] = {
        **interval,
        "eta": eta,
        "pose_count": 100,
        "finite_count": 100,
        "applied_count": 100,
        "model_atom_speed_rms_sum": model_sum,
        "applied_atom_speed_rms_sum": applied_sum,
        "total_atom_speed_rms_sum": total_sum,
        "atom_speed_rms_valid_count": 100,
        "model_rms_path_proxy_sum": interval["dt"] * model_sum,
        "applied_rms_path_proxy_sum": interval["dt"] * applied_sum,
        "total_rms_path_proxy_sum": interval["dt"] * total_sum,
        "translation_cap_trigger_count": 0,
        "angular_cap_trigger_count": 0,
        "displacement_cap_trigger_count": 0,
        "any_cap_trigger_count": 0,
        "multiple_cap_trigger_count": 0,
    }
    for name, value in (
        ("applied_to_model_rms_ratio", 0.1),
        ("model_guide_cosine", 0.25),
        ("guide_parallel_to_model_ratio", 0.025),
        ("cap_scale", 1.0),
    ):
        step[f"{name}_sum"] = value * 100
        step[f"{name}_valid_count"] = 100
        for suffix in ("p05", "p50", "p95", "p99"):
            step[f"{name}_{suffix}"] = value
    return step


def test_eta_tags_are_frozen_and_filesystem_safe() -> None:
    assert [report.eta_tag(value) for value in report.ETA_VALUES] == list(report.ETA_TAGS)
    assert report.expected_run_name("astex", 0.025).endswith("-eta0025")
    with pytest.raises(ValueError, match="eta must be one of"):
        report.eta_tag(0.15)


def test_runtime_telemetry_uses_metric_specific_denominators() -> None:
    stats = {
        "direct_pose_evaluations": 800,
        "direct_nonfinite_poses": 0,
        "direct_atom_speed_rms_valid_count": 800,
        "direct_model_atom_speed_rms_sum": 4000.0,
        "direct_applied_to_model_rms_ratio_sum": 80.0,
        "direct_applied_to_model_rms_ratio_valid_count": 800,
        "direct_model_guide_cosine_sum": 200.0,
        "direct_model_guide_cosine_valid_count": 400,
        "direct_guide_parallel_to_model_ratio_sum": 20.0,
        "direct_guide_parallel_to_model_ratio_valid_count": 800,
        "direct_cap_scale_sum": 760.0,
        "direct_cap_scale_valid_count": 800,
        "direct_model_rms_path_proxy_sum": 250.0,
        "direct_applied_rms_path_proxy_sum": 25.0,
        "direct_total_rms_path_proxy_sum": 252.0,
        "direct_new_scalar_counter": 3,
        "direct_max_translation_velocity": 2.0,
    }
    combined = report._combine_runtime_stats(
        [{"guidance_runtime_stats": stats}, {"guidance_runtime_stats": stats}]
    )
    assert combined["means"]["direct_model_atom_speed_rms_mean"] == 5.0
    assert combined["means"]["direct_applied_to_model_rms_ratio_mean"] == 0.1
    assert combined["means"]["direct_model_guide_cosine_mean"] == 0.5
    assert combined["means"]["direct_cap_scale_mean"] == 0.95
    assert combined["trajectory_count"] == 200
    assert combined["means"]["direct_model_rms_path_proxy_mean_per_trajectory"] == 2.5
    assert combined["scalars"]["direct_new_scalar_counter"] == 6


def test_direct_step_trace_requires_exact_grid_eta_and_finite_fields() -> None:
    eta = 0.2
    trace = [_trace_step(interval, eta) for interval in report._expected_trace_intervals()]
    row = {"id": "case", "guidance_direct_step_trace_json": json.dumps(trace)}
    assert len(report._validate_direct_step_trace(row, eta=eta)) == 8

    trace[0]["eta"] = 0.3
    row["guidance_direct_step_trace_json"] = json.dumps(trace)
    with pytest.raises(ValueError, match="eta must be"):
        report._validate_direct_step_trace(row, eta=eta)


def test_direct_step_trace_rejects_negative_and_inconsistent_cap_telemetry() -> None:
    eta = 0.2
    trace = [_trace_step(interval, eta) for interval in report._expected_trace_intervals()]
    trace[0]["applied_atom_speed_rms_sum"] = -1.0
    row = {"id": "case", "guidance_direct_step_trace_json": json.dumps(trace)}
    with pytest.raises(ValueError, match="must be non-negative"):
        report._validate_direct_step_trace(row, eta=eta)

    trace = [_trace_step(interval, eta) for interval in report._expected_trace_intervals()]
    trace[0]["translation_cap_trigger_count"] = 2
    trace[0]["any_cap_trigger_count"] = 1
    trace[0]["multiple_cap_trigger_count"] = 0
    row["guidance_direct_step_trace_json"] = json.dumps(trace)
    with pytest.raises(ValueError, match="cap trigger counters are inconsistent"):
        report._validate_direct_step_trace(row, eta=eta)


def test_interval_aggregate_preserves_per_complex_pose_quantile_summaries() -> None:
    eta = 0.2
    first = [_trace_step(interval, eta) for interval in report._expected_trace_intervals()]
    second = [_trace_step(interval, eta) for interval in report._expected_trace_intervals()]
    second[0]["applied_to_model_rms_ratio_p50"] = 0.3
    aggregate = report._aggregate_interval_telemetry([first, second], eta=eta)
    first_interval = aggregate[0]
    assert first_interval["complexes"] == 2
    assert first_interval["pose_trajectories"] == 200
    quantiles = first_interval["per_complex_pose_quantiles"]
    assert "not pooled-pose quantiles" in quantiles["semantics"]
    assert quantiles["metrics"]["applied_to_model_rms_ratio"]["p50"]["p50"] == 0.2


def test_eta_zero_requires_empty_direct_step_trace() -> None:
    row = {"id": "case", "guidance_direct_step_trace_json": "[]"}
    assert report._validate_direct_step_trace(row, eta=0.0) == []
    row["guidance_direct_step_trace_json"] = "[{}]"
    with pytest.raises(ValueError, match=r"must be \[\]"):
        report._validate_direct_step_trace(row, eta=0.0)


def test_sampling_inventory_requires_all_128_shards(tmp_path) -> None:
    for dataset in report.DATASETS:
        for eta, tag in zip(report.ETA_VALUES, report.ETA_TAGS, strict=True):
            for shard in range(8):
                summary = {key: None for key in report._REQUIRED_SUMMARY_KEYS}
                summary.update(
                    {
                        "protocol_id": report.PROTOCOL_ID,
                        "run_name": report.expected_run_name(dataset, eta),
                        "dataset": dataset,
                        "unified_guidance_scale": eta,
                        "num_samples": 100,
                        "num_steps": 10,
                        "model_pose_step_budget": 1000,
                        "shard_index": shard,
                    }
                )
                (tmp_path / f"{dataset}-{tag}-{shard}.summary.json").write_text(json.dumps(summary))
    grouped = report._load_sampling_inventory(tmp_path, expected_shards=8)
    assert len(grouped) == 16
    assert sum(len(value) for value in grouped.values()) == 128

    next(tmp_path.glob("*.summary.json")).unlink()
    with pytest.raises(ValueError, match="exactly 128"):
        report._load_sampling_inventory(tmp_path, expected_shards=8)


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _official_binding_fixture(tmp_path: Path) -> tuple[Path, Path, set[str]]:
    sampling_dir = tmp_path / "raw"
    official_dir = tmp_path / "official"
    expected_runs: set[str] = set()
    for dataset in report.DATASETS:
        for eta in report.ETA_VALUES:
            run_name = report.expected_run_name(dataset, eta)
            expected_runs.add(run_name)
            complex_id = f"{dataset}-{report.eta_tag(eta)}"
            row = {
                "id": complex_id,
                "protein_sha256": "1" * 64,
                "ligand_reference_sha256": "2" * 64,
                "saved_pose_sha256_json": json.dumps({"oracle": "3" * 64}),
            }
            _write_csv(sampling_dir / f"{run_name}.csv", [row])
            tag = "shard-000-of-001"
            (sampling_dir / f"{run_name}.{tag}.summary.json").write_text(
                json.dumps(
                    {
                        "protocol_id": report.PROTOCOL_ID,
                        "run_name": run_name,
                        "dataset": dataset,
                        "shard_index": 0,
                        "num_shards": 1,
                        "unified_guidance_scale": eta,
                    }
                )
            )
            cell_dir = official_dir / run_name
            cell_dir.mkdir(parents=True)
            (cell_dir / f"{tag}.summary.json").write_text(
                json.dumps(
                    {
                        "input_hashes_verified": True,
                        "num_input_hashes_verified": 1,
                        "num_assigned": 1,
                    }
                )
            )
            _write_csv(cell_dir / f"{tag}.csv", [{"id": complex_id}])
            binding = official_binding.build_binding(
                sampling_dir=sampling_dir,
                official_dir=cell_dir,
                run_name=run_name,
                protocol_id=report.PROTOCOL_ID,
                dataset=dataset,
                eta=eta,
                shard_index=0,
                num_shards=1,
            )
            (cell_dir / f"{tag}.binding.json").write_text(json.dumps(binding))
    return sampling_dir, official_dir, expected_runs


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("swapped_eta", "eta mismatch"),
        ("sampling_row", "official/sampling binding mismatch"),
        ("official_csv", "official IDs/order differ"),
    ),
)
def test_official_binding_rejects_modified_inputs(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    sampling_dir, official_dir, expected_runs = _official_binding_fixture(tmp_path)
    dataset, eta = "astex", 0.1
    run_name = report.expected_run_name(dataset, eta)
    tag = "shard-000-of-001"
    if mutation == "swapped_eta":
        path = sampling_dir / f"{run_name}.{tag}.summary.json"
        payload = json.loads(path.read_text())
        payload["unified_guidance_scale"] = 0.2
        path.write_text(json.dumps(payload))
    elif mutation == "sampling_row":
        _write_csv(
            sampling_dir / f"{run_name}.csv",
            [
                {
                    "id": f"{dataset}-{report.eta_tag(eta)}",
                    "protein_sha256": "4" * 64,
                    "ligand_reference_sha256": "2" * 64,
                    "saved_pose_sha256_json": json.dumps({"oracle": "3" * 64}),
                }
            ],
        )
    else:
        _write_csv(
            official_dir / run_name / f"{tag}.csv",
            [{"id": "modified-id"}],
        )

    with pytest.raises(ValueError, match=message):
        official_report._require_input_hash_verification(
            official_dir,
            sampling_dir,
            expected_runs,
            1,
        )
