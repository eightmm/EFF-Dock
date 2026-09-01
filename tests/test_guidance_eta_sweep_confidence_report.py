from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from effdock.workflows.guidance_budget_posebusters_report import VALIDITY_CHECKS
from effdock.workflows.guidance_budget_report import (
    EXPECTED_CHECKPOINT_SHA256,
    EXPECTED_CONFIG_SHA256,
)
from effdock.workflows.guidance_eta_sweep_confidence_binding import (
    CONFIDENCE_CHECKPOINT_SHA256,
    PROTOCOL_ID,
    build_binding,
)
from effdock.workflows.guidance_eta_sweep_confidence_report import (
    EXPECTED_PARENT_AUDIT_SHA256,
    IDENTITY_AUDIT_CONTRACT,
    aggregate_official_cell,
    join_sampling_outcomes,
    load_frozen_parent_cohort_audit,
    paired_outcomes,
    revalidate_identity_audit,
    validate_identity_audit,
)
from effdock.workflows.guidance_eta_sweep_report import PROTOCOL_ID as PARENT_PROTOCOL_ID


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
    run_name = "confidence-binding-test"
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
    sampling_csv = sampling_dir / f"{run_name}.{tag}.csv"
    _write_csv(sampling_csv, [row])
    (sampling_dir / f"{run_name}.{tag}.summary.json").write_text(
        json.dumps(
            {
                "protocol_id": PROTOCOL_ID,
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


def test_confidence_binding_verifies_selected_pose_and_complete_summaries(
    tmp_path: Path,
) -> None:
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
    assert binding["selector_role"] == "primary"
    assert binding["confidence_checkpoint_sha256"] == CONFIDENCE_CHECKPOINT_SHA256
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


def _identity_audit() -> dict[str, object]:
    return {
        "protocol_id": PROTOCOL_ID,
        "audit_contract": IDENTITY_AUDIT_CONTRACT,
        "parent_sampling_protocol_id": PARENT_PROTOCOL_ID,
        "mode": "full",
        "status": "passed",
        "parent_sentinels_verified": True,
        "candidate_ensemble_hashes_present": True,
        "selector_recomputed": True,
        "summary_contracts_verified": True,
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
        "global_equivalence_ledger_sha256": "a" * 64,
    }


def test_identity_audit_requires_full_parent_verified_replay(tmp_path: Path) -> None:
    path = tmp_path / "identity.json"
    audit = _identity_audit()
    path.write_text(json.dumps(audit))
    assert validate_identity_audit(path, expected_shards=8)["status"] == "passed"

    audit["parent_sentinels_verified"] = False
    path.write_text(json.dumps(audit))
    with pytest.raises(ValueError, match="parent_sentinels_verified"):
        validate_identity_audit(path, expected_shards=8)

    audit = _identity_audit()
    audit["mode"] = "smoke"
    path.write_text(json.dumps(audit))
    with pytest.raises(ValueError, match="mode"):
        validate_identity_audit(path, expected_shards=8)


def test_frozen_parent_cohort_audit_loads_exact_production_manifest(tmp_path: Path) -> None:
    path = Path(
        "outputs/benchmarks/guidance_eta_sweep_v2_runs/20260801T102903Z/audit/combined.json"
    )
    if not path.is_file():
        pytest.skip("production eta-sweep audit is not available")
    assert _sha256(path) == EXPECTED_PARENT_AUDIT_SHA256
    audits = load_frozen_parent_cohort_audit(path)
    assert len(audits["astex"]["ids"]) == 85
    assert len(audits["posebusters"]["ids"]) == 308

    changed = json.loads(path.read_text())
    changed["created_utc"] = "tampered"
    tampered = tmp_path / "combined.tampered.json"
    tampered.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="frozen parent audit SHA-256 mismatch"):
        load_frozen_parent_cohort_audit(tampered)


def test_identity_audit_is_rebuilt_before_posebusters_outcomes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent_dir = tmp_path / "parent"
    confidence_dir = tmp_path / "confidence"
    parent_dir.mkdir()
    confidence_dir.mkdir()
    identity = _identity_audit()
    identity.update(
        {
            "parent_dir": str(parent_dir),
            "confidence_dir": str(confidence_dir),
        }
    )
    calls: list[tuple[Path, Path, bool]] = []

    def rebuild(parent: Path, confidence: Path, *, smoke: bool) -> dict[str, object]:
        calls.append((parent, confidence, smoke))
        return identity

    monkeypatch.setattr(
        "effdock.workflows.guidance_eta_sweep_confidence_report."
        "guidance_eta_sweep_confidence_identity.build_identity_audit",
        rebuild,
    )
    revalidate_identity_audit(identity, sampling_dir=confidence_dir)
    assert calls == [(parent_dir, confidence_dir, False)]

    with pytest.raises(ValueError, match="sampling_dir must resolve"):
        revalidate_identity_audit(identity, sampling_dir=tmp_path / "other")

    def changed_rebuild(parent: Path, confidence: Path, *, smoke: bool) -> dict[str, object]:
        changed = dict(identity)
        changed["global_equivalence_ledger_sha256"] = "b" * 64
        return changed

    monkeypatch.setattr(
        "effdock.workflows.guidance_eta_sweep_confidence_report."
        "guidance_eta_sweep_confidence_identity.build_identity_audit",
        changed_rebuild,
    )
    with pytest.raises(ValueError, match="fresh full replay audit"):
        revalidate_identity_audit(identity, sampling_dir=confidence_dir)


def _official_cell(tmp_path: Path, *, selector: str = "confidence") -> Path:
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
                "input_hashes_verified": True,
                "num_input_hashes_verified": 1,
                "require_complete_success": True,
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


def test_official_cell_rejects_unverified_or_survivor_only_outputs(tmp_path: Path) -> None:
    cell_dir = _official_cell(tmp_path)
    rows, aggregate, _ = aggregate_official_cell(
        cell_dir,
        ("case",),
        run_name="run",
        dataset="astex",
        eta_tag="eta0000",
        selector="confidence",
        expected_shards=1,
    )
    assert set(rows) == {"case"}
    assert aggregate["posebusters_valid_count"] == 1

    summary_path = cell_dir / "shard-000-of-001.summary.json"
    summary = json.loads(summary_path.read_text())
    summary["input_hashes_verified"] = False
    summary_path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="input_hashes_verified"):
        aggregate_official_cell(
            cell_dir,
            ("case",),
            run_name="run",
            dataset="astex",
            eta_tag="eta0000",
            selector="confidence",
            expected_shards=1,
        )


def test_join_and_paired_metrics_use_confidence_outputs(tmp_path: Path) -> None:
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
    official_confidence = {
        "a": {"posebusters_valid": False},
        "b": {"posebusters_valid": True},
    }
    official_filter = {
        "a": {"posebusters_valid": True},
        "b": {"posebusters_valid": True},
    }
    ids = ("a", "b")
    confidence = join_sampling_outcomes(
        sampling_dir=sampling_dir,
        run_name=run_name,
        selector="confidence",
        ids=ids,
        official_rows=official_confidence,
    )
    confidence_filter = join_sampling_outcomes(
        sampling_dir=sampling_dir,
        run_name=run_name,
        selector="confidence_filter",
        ids=ids,
        official_rows=official_filter,
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
    assert comparison["direction"] == "confidence_filter_minus_confidence"
    assert comparison["metrics"]["selected_rmsd_lt2"]["delta"] == 50.0
    assert comparison["metrics"]["posebusters_valid"]["delta"] == 50.0
    assert comparison["transitions"]["posebusters_valid"]["false_to_true"] == 1
