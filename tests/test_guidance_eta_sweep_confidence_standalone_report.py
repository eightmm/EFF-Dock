from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from effdock.workflows.guidance_budget_report import (
    EXPECTED_CHECKPOINT_SHA256,
    EXPECTED_CONFIG_SHA256,
)
from effdock.workflows.guidance_eta_sweep_confidence_standalone_binding import (
    CONFIDENCE_CHECKPOINT_SHA256,
    PROTOCOL_ID,
    build_binding,
)
from effdock.workflows.guidance_eta_sweep_confidence_standalone_report import (
    AUDIT_CONTRACT,
    _require_bindings,
    build_report,
    join_sampling_outcomes,
    paired_outcomes,
    revalidate_standalone_audit,
    validate_standalone_audit,
)
from effdock.workflows.guidance_eta_sweep_report import (
    DATASETS,
    ETA_VALUES,
    expected_run_name,
)
from effdock.workflows.guidance_eta_sweep_standalone_spec import (
    LEGACY_V1,
    STERIC_HIGH_ETA_V1,
    StandaloneSweepSpec,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _binding_fixture(
    tmp_path: Path,
    *,
    selector: str = "confidence",
) -> tuple[Path, Path, str, Path]:
    sampling_dir = tmp_path / "raw"
    official_dir = tmp_path / "official" / selector
    run_name = "standalone-binding-test"
    protein = tmp_path / "protein.pdb"
    ligand = tmp_path / "ligand.sdf"
    protein.write_text("protein\n")
    ligand.write_text("ligand\n")
    pose_hashes: dict[str, str] = {}
    for current in ("confidence", "confidence_filter"):
        pose = sampling_dir / "poses" / run_name / "astex" / current / "case.sdf"
        pose.parent.mkdir(parents=True, exist_ok=True)
        pose.write_text(f"{current}\n")
        pose_hashes[current] = _sha256(pose)
    row = {
        "id": "case",
        "protein": str(protein),
        "ligand_ref": str(ligand),
        "protein_sha256": _sha256(protein),
        "ligand_reference_sha256": _sha256(ligand),
        "saved_pose_sha256_json": json.dumps(pose_hashes, sort_keys=True),
    }
    tag = "shard-000-of-001"
    sampling_csv = sampling_dir / f"{run_name}.csv"
    _write_csv(sampling_csv, [row])
    (sampling_dir / f"{run_name}.summary.json").write_text(
        json.dumps(
            {
                "protocol_id": PROTOCOL_ID,
                "selector_profile": "confidence_cluster_free",
                "run_name": run_name,
                "dataset": "astex",
                "shard_index": 0,
                "num_shards": 1,
                "confidence_checkpoint_sha256": CONFIDENCE_CHECKPOINT_SHA256,
                "unified_guidance_scale": 0.1,
                "expected_discovered_count": 1,
                "num_discovered_total": 1,
                "num_assigned": 1,
                "num_success": 1,
                "num_failed": 0,
                "num_samples": 100,
                "num_steps": 10,
                "model_pose_step_budget": 1000,
                "require_complete_success": True,
                "failures": [],
                "csv": str(sampling_csv),
            }
        )
    )
    official_dir.mkdir(parents=True)
    official_csv = official_dir / f"{tag}.csv"
    _write_csv(official_csv, [{"id": "case"}])
    (official_dir / f"{tag}.summary.json").write_text(
        json.dumps(
            {
                "posebusters_version": "0.6.5",
                "config": "redock",
                "selector": selector,
                "input_hashes_verified": True,
                "num_input_hashes_verified": 1,
                "expected_discovered_count": 1,
                "require_complete_success": True,
                "num_discovered_total": 1,
                "num_assigned": 1,
                "num_success": 1,
                "num_failed": 0,
                "failures": [],
                "csv": str(official_csv),
            }
        )
    )
    pose = sampling_dir / "poses" / run_name / "astex" / selector / "case.sdf"
    return sampling_dir, official_dir, run_name, pose


def test_standalone_binding_verifies_selected_pose_and_protocol(tmp_path: Path) -> None:
    sampling_dir, official_dir, run_name, pose = _binding_fixture(tmp_path)
    binding = build_binding(
        sampling_dir=sampling_dir,
        official_dir=official_dir,
        run_name=run_name,
        protocol_id=PROTOCOL_ID,
        dataset="astex",
        eta=0.1,
        selector="confidence",
        shard_index=0,
        num_shards=1,
    )
    assert binding["protocol_id"] == PROTOCOL_ID
    assert binding["selector_role"] == "primary"
    assert len(binding["sampling_input_file_ledger_sha256"]) == 64

    pose.write_text("tampered\n")
    with pytest.raises(ValueError, match="selected_pose_sha256 differs"):
        build_binding(
            sampling_dir=sampling_dir,
            official_dir=official_dir,
            run_name=run_name,
            protocol_id=PROTOCOL_ID,
            dataset="astex",
            eta=0.1,
            selector="confidence",
            shard_index=0,
            num_shards=1,
        )


def test_high_eta_binding_binds_v2_integrity_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = STERIC_HIGH_ETA_V1
    sampling_dir, official_dir, run_name, _pose = _binding_fixture(tmp_path)
    sampling_summary = sampling_dir / f"{run_name}.summary.json"
    summary = json.loads(sampling_summary.read_text())
    summary["protocol_id"] = spec.protocol_id
    sampling_summary.write_text(json.dumps(summary))
    integrity_path = tmp_path / "full-integrity.json"
    integrity_path.write_text(
        json.dumps(
            {
                "schema_version": spec.audit_schema_version,
                "protocol_id": spec.protocol_id,
                "audit_contract": spec.audit_contract,
                "mode": "fresh_one_pass_characterization",
                "run_scope": "full",
                "status": "passed",
                "parent_compared": False,
                "deterministic_replay_claim": False,
                "global_integrity_ledger_sha256": "a" * 64,
            }
        )
    )
    monkeypatch.setattr(
        "effdock.workflows.guidance_eta_sweep_confidence_standalone_binding."
        "validate_v2_prior_pool_sha256_diagnostics",
        lambda *_args, **_kwargs: {
            "policy": "record_only_across_eta",
            "complexes_with_multiple_hashes": 1,
        },
    )

    binding = build_binding(
        sampling_dir=sampling_dir,
        official_dir=official_dir,
        run_name=run_name,
        protocol_id=spec.protocol_id,
        dataset="astex",
        eta=0.1,
        selector="confidence",
        shard_index=0,
        num_shards=1,
        integrity_audit=integrity_path,
        spec=spec,
    )

    assert binding["integrity_audit"]["schema_version"] == spec.audit_schema_version
    assert binding["integrity_audit"]["prior_pool_sha256_cross_eta_identity_claim"] is False
    assert binding["integrity_audit"]["complexes_with_multiple_prior_pool_sha256"] == 1


def _standalone_audit(sampling_dir: Path) -> dict[str, object]:
    return {
        "protocol_id": PROTOCOL_ID,
        "audit_contract": AUDIT_CONTRACT,
        "mode": "fresh_one_pass_characterization",
        "run_scope": "full",
        "status": "passed",
        "parent_compared": False,
        "deterministic_replay_claim": False,
        "selector_profile": "confidence_cluster_free",
        "sampling_dir": str(sampling_dir),
        "coverage": {
            "datasets": 2,
            "cells": 16,
            "shards": 128,
            "rows": 3144,
            "per_dataset": {
                "astex": {"cells": 8, "shards": 64, "rows": 680, "ids_per_cell": 85},
                "posebusters": {
                    "cells": 8,
                    "shards": 64,
                    "rows": 2464,
                    "ids_per_cell": 308,
                },
            },
        },
        "frozen_hashes": {
            "docking_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
            "config_sha256": EXPECTED_CONFIG_SHA256,
            "confidence_checkpoint_sha256": CONFIDENCE_CHECKPOINT_SHA256,
        },
        "candidate_ensemble_verification": {
            "claim": "digest_present_and_producer_bound",
        },
        "checks": {"summary_contracts_verified": True},
        "global_integrity_ledger_sha256": "a" * 64,
    }


def test_standalone_audit_is_full_parent_free_and_rebuilt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sampling_dir = tmp_path / "raw"
    sampling_dir.mkdir()
    cohort = tmp_path / "cohort.json"
    cohort.write_text("{}")
    audit = _standalone_audit(sampling_dir)
    path = tmp_path / "integrity.json"
    path.write_text(json.dumps(audit))
    assert validate_standalone_audit(path, expected_shards=8)["status"] == "passed"

    calls: list[tuple[Path, bool, Path | None]] = []

    def rebuild(
        root: Path,
        *,
        smoke: bool,
        cohort_audit: Path | None,
    ) -> dict[str, object]:
        calls.append((root, smoke, cohort_audit))
        return audit

    monkeypatch.setattr(
        "effdock.workflows.guidance_eta_sweep_confidence_standalone_report.build_standalone_audit",
        rebuild,
    )
    revalidate_standalone_audit(
        audit,
        sampling_dir=sampling_dir,
        cohort_audit=cohort,
    )
    assert calls == [(sampling_dir, False, cohort)]

    changed = dict(audit)
    changed["parent_sentinels_verified"] = True
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="replay-only fields"):
        validate_standalone_audit(path, expected_shards=8)


def test_official_inventory_requires_exactly_256_bound_shards(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_dir = tmp_path / "official"
    expected_binding = {"binding_contract": "test"}
    for selector in ("confidence", "confidence_filter"):
        for dataset in DATASETS:
            for eta in ETA_VALUES:
                cell = input_dir / selector / expected_run_name(dataset, eta)
                cell.mkdir(parents=True)
                for shard in range(8):
                    tag = f"shard-{shard:03d}-of-008"
                    (cell / f"{tag}.summary.json").write_text("{}")
                    (cell / f"{tag}.csv").write_text("id\n")
                    (cell / f"{tag}.binding.json").write_text(json.dumps(expected_binding))

    calls = 0

    def rebuild_binding(**_: object) -> dict[str, str]:
        nonlocal calls
        calls += 1
        return expected_binding

    monkeypatch.setattr(
        "effdock.workflows.guidance_eta_sweep_confidence_standalone_report.build_binding",
        rebuild_binding,
    )
    _require_bindings(input_dir=input_dir, sampling_dir=tmp_path / "raw", expected_shards=8)
    assert calls == 256

    missing = (
        input_dir
        / "confidence"
        / expected_run_name("astex", ETA_VALUES[0])
        / "shard-000-of-008.binding.json"
    )
    missing.unlink()
    with pytest.raises(ValueError, match="binding inventory mismatch"):
        _require_bindings(input_dir=input_dir, sampling_dir=tmp_path / "raw", expected_shards=8)


def test_join_and_paired_metrics_use_both_frozen_selectors(tmp_path: Path) -> None:
    sampling_dir = tmp_path / "raw"
    run_name = "run"
    _write_csv(
        sampling_dir / f"{run_name}.csv",
        [
            {
                "id": "a",
                "num_samples": 100,
                "confidence_index": 3,
                "confidence_rmsd": 3.0,
                "confidence_filter_index": 4,
                "confidence_filter_rmsd": 1.5,
            },
            {
                "id": "b",
                "num_samples": 100,
                "confidence_index": 5,
                "confidence_rmsd": 1.0,
                "confidence_filter_index": 5,
                "confidence_filter_rmsd": 1.0,
            },
        ],
    )
    ids = ("a", "b")
    confidence = join_sampling_outcomes(
        sampling_dir=sampling_dir,
        run_name=run_name,
        selector="confidence",
        ids=ids,
        official_rows={
            "a": {"posebusters_valid": False},
            "b": {"posebusters_valid": True},
        },
    )
    confidence_filter = join_sampling_outcomes(
        sampling_dir=sampling_dir,
        run_name=run_name,
        selector="confidence_filter",
        ids=ids,
        official_rows={
            "a": {"posebusters_valid": True},
            "b": {"posebusters_valid": True},
        },
    )
    comparison = paired_outcomes(
        confidence,
        confidence_filter,
        ids,
        baseline_label="confidence",
        comparison_label="confidence_filter",
        seed=7,
        resamples=100,
    )
    assert comparison["metrics"]["selected_rmsd_lt2"]["delta"] == 50.0
    assert comparison["metrics"]["posebusters_valid"]["delta"] == 50.0
    assert comparison["transitions"]["posebusters_valid"]["false_to_true"] == 1


@pytest.mark.parametrize("spec", (LEGACY_V1, STERIC_HIGH_ETA_V1))
def test_report_metadata_is_one_pass_parent_free_and_selects_no_winner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    spec: StandaloneSweepSpec,
) -> None:
    module = "effdock.workflows.guidance_eta_sweep_confidence_standalone_report"
    monkeypatch.setattr(f"{module}.ETA_VALUES", (0.0,))
    monkeypatch.setattr(f"{module}.ETA_TAGS", ("eta0000",))
    monkeypatch.setattr(f"{module}.DEFAULT_EXPECTED_SHARDS", 1)
    monkeypatch.setattr(f"{module}.EXPECTED_DATASET_COUNTS", {"astex": 1, "posebusters": 1})
    monkeypatch.setattr(f"{module}._profile_grid", lambda _spec: ((0.0,), ("eta0000",)))
    monkeypatch.setattr(
        f"{module}._profile_run_name",
        lambda _spec, dataset, _eta: f"run-{dataset}",
    )

    audit = {
        "status": "passed",
        "mode": "fresh_one_pass_characterization",
        "run_scope": "full",
        "parent_compared": False,
        "deterministic_replay_claim": False,
        "selector_profile": "confidence_cluster_free",
        "candidate_ensemble_verification": {"status": "digest_present_and_producer_bound"},
        "global_integrity_ledger_sha256": "a" * 64,
        "coverage": {"datasets": 2, "cells": 2, "shards": 2, "rows": 2},
        **(
            {
                "prior_pool_sha256_diagnostics": {
                    "policy": "record_only_across_eta",
                    "complexes": 2,
                    "complexes_with_single_hash": 1,
                    "complexes_with_multiple_hashes": 1,
                    "mismatched_ids": [{"dataset": "astex", "id": "case-astex"}],
                }
            }
            if spec == STERIC_HIGH_ETA_V1
            else {}
        ),
    }
    ids = {dataset: (f"case-{dataset}",) for dataset in DATASETS}
    monkeypatch.setattr(f"{module}.validate_standalone_audit", lambda *_args, **_kw: audit)
    monkeypatch.setattr(f"{module}.revalidate_standalone_audit", lambda *_args, **_kw: None)
    monkeypatch.setattr(
        f"{module}.load_frozen_cohort_audit",
        lambda _path: {
            dataset: {
                "ids": dataset_ids,
                "ids_sha256": "b" * 64,
                "source_path": "cohort.json",
                "source_sha256": "c" * 64,
            }
            for dataset, dataset_ids in ids.items()
        },
    )
    monkeypatch.setattr(f"{module}._require_sampling_inventory", lambda *_args, **_kw: None)
    monkeypatch.setattr(f"{module}._require_bindings", lambda *_args, **_kw: None)
    monkeypatch.setattr(f"{module}.file_sha256", lambda _path: "d" * 64)

    def aggregate(
        _cell: Path,
        dataset_ids: tuple[str, ...],
        **kwargs: object,
    ) -> tuple[dict[str, dict[str, bool]], dict[str, object], str]:
        return (
            {dataset_ids[0]: {"posebusters_valid": True}},
            {"selector": kwargs["selector"]},
            "rmsd_≤_2å",
        )

    def join(
        *,
        selector: str,
        ids: tuple[str, ...],
        **_kwargs: object,
    ) -> dict[str, dict[str, object]]:
        rmsd = 1.0 if selector == "confidence" else 3.0
        return {
            ids[0]: {
                "posebusters_valid": True,
                "selected_rmsd": rmsd,
                "selected_rmsd_lt2": rmsd < 2.0,
                "joint_selected_rmsd_lt2_and_posebusters_valid": rmsd < 2.0,
            }
        }

    monkeypatch.setattr(f"{module}.aggregate_official_cell", aggregate)
    monkeypatch.setattr(f"{module}.join_sampling_outcomes", join)
    integrity_path = tmp_path / "integrity.json"
    integrity_path.write_text("{}")
    report = build_report(
        tmp_path / "official",
        tmp_path / "raw",
        tmp_path / "cohort.json",
        integrity_path,
        expected_shards=1,
        bootstrap_resamples=10,
        spec=spec,
    )

    assert report["evaluation_mode"] == "fresh_one_pass_characterization"
    assert report["parent_identity_claim"] is False
    assert report["deterministic_replay_claim"] is False
    assert report["winner_selected"] is False
    assert report["official_inventory"]["total_shard_tasks"] == 4
    if spec == STERIC_HIGH_ETA_V1:
        assert "complex-ID-paired, seed-matched descriptive" in report["claim_boundary"]
        assert "exact prior tensor equality is not claimed" in report["claim_boundary"]
        assert report["prior_pairing_claim"] == {
            "sampling_seed": "exact_base_seed_42_plus_one_based_sorted_dataset_id_index",
            "prior_pool_size": "exact_100",
            "prior_pool_construction_contract": "exact_EFFDOCK_SHARED_PRIOR_V1",
            "prior_pool_sha256_cross_eta": "diagnostic_only",
            "exact_prior_tensor_identity_claim": False,
            "reason": "prior tensors were not persisted at original float32 precision",
            "diagnostic_summary": {
                "complexes": 2,
                "complexes_with_single_hash": 1,
                "complexes_with_multiple_hashes": 1,
                "mismatched_ids": [{"dataset": "astex", "id": "case-astex"}],
            },
        }
        assert report["standalone_audit"]["schema_version"] == spec.audit_schema_version
        assert report["standalone_audit"]["prior_pool_sha256_diagnostics"] == audit[
            "prior_pool_sha256_diagnostics"
        ]
    else:
        assert "prior_pairing_claim" not in report
        assert report["claim_boundary"].startswith(
            "paired descriptive external benchmark from one fresh sampling pass"
        )
    assert (
        report["datasets"]["astex"]["cells"]["eta0000"]["selectors"]["confidence"]["metrics"][
            "selected_rmsd_lt2_count"
        ]
        == 1
    )
    assert "parent_sampling_protocol_id" not in report
