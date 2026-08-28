from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "report_s50_confidence_training.py"
SPEC = importlib.util.spec_from_file_location("report_s50_confidence_training", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
reporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reporter)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _eval_record(pid: str, system_id: str, *, success: bool) -> dict[str, object]:
    top1 = 1.0 if success else 3.0
    return {
        "pid": pid,
        "system_id": system_id,
        "pose_count": 100,
        "oracle_k2": 1,
        "oracle_k2_slice": "1_4",
        "top1_index": 0,
        "top5_indices": [0, 1, 2, 3, 4],
        "top1_rmsd": top1,
        "top5_best_rmsd": 1.0,
        "oracle_rmsd": 0.5,
        "top1_lt2": success,
        "top5_lt2": True,
        "oracle_lt2": True,
    }


def _fixture(
    tmp_path: Path,
    *,
    looks: dict[int, list[bool]] | None = None,
) -> tuple[Path, Path, str, Path, str]:
    training = tmp_path / "training"
    training.mkdir()
    ids = ["sysa__lig1", "sysa__lig2", "sysb__lig1", "sysb__lig2"]
    systems = {
        "sysa__lig1": "authoritative-system-one",
        "sysa__lig2": "authoritative-system-one",
        "sysb__lig1": "authoritative-system-two",
        "sysb__lig2": "authoritative-system-two",
    }
    baseline = [True, False, False, False]
    if looks is None:
        looks = {
            5_000: [True, True, True, False],
            10_000: [True, True, True, True],
        }

    excluded_id = "excluded__lig"
    full_ids = [ids[0], ids[1], excluded_id, ids[2], ids[3]]
    eligible_sha = reporter.ordered_ids_sha256(ids)
    excluded_sha = reporter.ordered_ids_sha256([excluded_id])
    full_sha = reporter.ordered_ids_sha256(full_ids)
    input_records = []
    for split_index, pid in enumerate(full_ids, start=1):
        if pid == excluded_id:
            input_records.append(
                {
                    "sample_key": pid,
                    "split": "val",
                    "split_index": split_index,
                    "status": "input_ineligible",
                    "failure": "fixed mapping unavailable",
                }
            )
        else:
            input_records.append(
                {
                    "sample_key": pid,
                    "system_id": systems[pid],
                    "split": "val",
                    "split_index": split_index,
                    "status": "eligible",
                }
            )
    input_manifest = tmp_path / "input_manifest.json"
    _write_json(
        input_manifest,
        {
            "schema_version": reporter.INPUT_SCHEMA,
            "protocol_id": reporter.BANK_PROTOCOL_ID,
            "study_protocol_id": reporter.PROTOCOL_ID,
            "status": "complete",
            "inputs": {"val_bank_manifest": {"sha256": reporter.SOURCE_VAL_BANK_SHA256}},
            "inventory": {
                "val": {
                    "full_count": 5,
                    "eligible_count": 4,
                    "excluded_count": 1,
                    "full_ids_sha256": full_sha,
                    "eligible_ids_sha256": eligible_sha,
                    "excluded_ids_sha256": excluded_sha,
                    "eligible_ids": ids,
                    "excluded_ids": [excluded_id],
                }
            },
            "records": input_records,
        },
    )
    input_sha = _sha(input_manifest)

    records = []
    original_split_indices = [1, 2, 4, 5]
    for split_index, pid in zip(original_split_indices, ids):
        pt = tmp_path / f"{pid}.pt"
        pt.write_bytes(pid.encode("utf-8"))
        records.append(
            {
                "sample_key": pid,
                # Deliberately differs from the sample-key prefix: the sealed
                # system_id is authoritative for clustered uncertainty.
                "system_id": systems[pid],
                "split": "val",
                "split_index": split_index,
                "status": "complete",
                "pose_count": 100,
                "pt_sha256": _sha(pt),
            }
        )
    bank_manifest = tmp_path / "bank_manifest.json"
    _write_json(
        bank_manifest,
        {
            "schema_version": reporter.BANK_SCHEMA,
            "protocol_id": reporter.BANK_PROTOCOL_ID,
            "study_protocol_id": reporter.PROTOCOL_ID,
            "status": "complete",
            "claim_eligible": True,
            "pose_tag": reporter.POSE_TAG,
            "input_manifest": {"path": str(input_manifest), "sha256": input_sha},
            "inventory": {
                "val": {
                    "eligible_count": 4,
                    "full_count": 5,
                    "excluded_count": 1,
                    "record_count": 4,
                    "full_ids_sha256": full_sha,
                    "eligible_ids_sha256": eligible_sha,
                    "excluded_ids_sha256": excluded_sha,
                    "record_ids_sha256": eligible_sha,
                }
            },
            "records": records,
        },
    )
    bank_sha = _sha(bank_manifest)

    all_steps = {0: baseline, **looks}
    for step, outcomes in all_steps.items():
        _write_json(
            training / f"eval_u{step:06d}.json",
            {
                "schema_version": reporter.EVAL_SCHEMA,
                "status": "complete",
                "step": step,
                "eval_target": reporter.EVAL_TARGET,
                "bank_manifest_sha256": bank_sha,
                "record_count": len(ids),
                "records": [
                    _eval_record(pid, systems[pid], success=success)
                    for pid, success in zip(ids, outcomes)
                ],
            },
        )

    scores = {step: 100.0 * sum(outcomes) / len(outcomes) for step, outcomes in all_steps.items()}
    best_step = max(scores, key=lambda step: (scores[step], -step))
    best_score = scores[best_step]
    torch.save(
        {
            "step": best_step,
            "metrics": {"eval_top1_lt2": best_score},
            "bank_provenance": {"sha256": bank_sha},
            "initialization_provenance": {"sha256": reporter.WARM_START_SHA256},
            "effective_global_batch_complexes": 4,
            "state_dict": {},
        },
        training / "best.pt",
    )
    _write_json(
        training / "metrics.json",
        {
            "final_step": 50_000,
            "best_metric": "eval_top1_lt2",
            "best_score": best_score,
            "eval_target": reporter.EVAL_TARGET,
            "effective_global_batch_complexes": 4,
            "bank_provenance": {"sha256": bank_sha},
            "initialization_provenance": {"sha256": reporter.WARM_START_SHA256},
        },
    )
    return training, bank_manifest, bank_sha, input_manifest, input_sha


def test_report_selects_best_5k_look_and_passes_paired_gate(tmp_path: Path) -> None:
    training, bank, bank_sha, inputs, input_sha = _fixture(tmp_path)

    report = reporter.build_report(
        training_dir=training,
        bank_manifest=bank,
        expected_bank_manifest_sha256=bank_sha,
        frozen_input_manifest=inputs,
        expected_frozen_input_manifest_sha256=input_sha,
        expected_val_count=4,
        expected_full_val_count=5,
        expected_excluded_val_count=1,
        scheduled_steps=(5_000, 10_000),
        bootstrap_draws=2_000,
    )

    assert report["selection"]["step"] == 10_000
    assert report["selection"]["paired_delta_pp"] == pytest.approx(75.0)
    assert report["selection"]["paired_ci95_pp"][0] > 0.0
    assert report["gate"]["new_checkpoint_admitted"] is True
    assert report["best_checkpoint"]["step"] == 10_000
    assert report["inventory"]["system_clusters"] == 2
    assert report["baseline_u0"]["diagnostics"]["selected_rmsd_median"] == 3.0
    assert report["selection"]["diagnostics"]["selected_rmsd_median"] == 1.0
    assert report["selection"]["diagnostics"]["k2_strata"]["1_4"] == {
        "n": 4,
        "top1_successes": 4,
        "eval_top1_lt2": 100.0,
        "selected_rmsd_mean": 1.0,
        "selected_rmsd_median": 1.0,
    }


def test_report_rejects_null_effect_without_changing_the_retained_default(
    tmp_path: Path,
) -> None:
    baseline = [True, False, False, False]
    training, bank, bank_sha, inputs, input_sha = _fixture(
        tmp_path,
        looks={5_000: baseline, 10_000: baseline},
    )

    report = reporter.build_report(
        training_dir=training,
        bank_manifest=bank,
        expected_bank_manifest_sha256=bank_sha,
        frozen_input_manifest=inputs,
        expected_frozen_input_manifest_sha256=input_sha,
        expected_val_count=4,
        expected_full_val_count=5,
        expected_excluded_val_count=1,
        scheduled_steps=(5_000, 10_000),
        bootstrap_draws=500,
    )

    assert report["selection"]["step"] == 5_000
    assert report["gate"]["new_checkpoint_admitted"] is False
    assert report["best_checkpoint"]["step"] == 0


def test_report_requires_the_complete_nonadaptive_schedule(tmp_path: Path) -> None:
    training, bank, bank_sha, inputs, input_sha = _fixture(tmp_path)
    (training / "eval_u010000.json").unlink()

    with pytest.raises(ValueError, match="incomplete or adaptive"):
        reporter.build_report(
            training_dir=training,
            bank_manifest=bank,
            expected_bank_manifest_sha256=bank_sha,
            frozen_input_manifest=inputs,
            expected_frozen_input_manifest_sha256=input_sha,
            expected_val_count=4,
            expected_full_val_count=5,
            expected_excluded_val_count=1,
            scheduled_steps=(5_000, 10_000),
            bootstrap_draws=100,
        )


def test_report_rejects_sample_key_prefix_as_system_id(tmp_path: Path) -> None:
    training, bank, bank_sha, inputs, input_sha = _fixture(tmp_path)
    ledger_path = training / "eval_u005000.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    ledger["records"][0]["system_id"] = "sysa"
    _write_json(ledger_path, ledger)

    with pytest.raises(ValueError, match="system_id disagrees"):
        reporter.build_report(
            training_dir=training,
            bank_manifest=bank,
            expected_bank_manifest_sha256=bank_sha,
            frozen_input_manifest=inputs,
            expected_frozen_input_manifest_sha256=input_sha,
            expected_val_count=4,
            expected_full_val_count=5,
            expected_excluded_val_count=1,
            scheduled_steps=(5_000, 10_000),
            bootstrap_draws=100,
        )


def test_report_rejects_changed_gapped_original_split_index(tmp_path: Path) -> None:
    training, bank, _, inputs, input_sha = _fixture(tmp_path)
    manifest = json.loads(bank.read_text(encoding="utf-8"))
    # Still strictly increasing (1, 2, 3, 5), but ID 3 was originally index 4
    # because index 3 belongs to the sealed ineligible validation record.
    manifest["records"][2]["split_index"] = 3
    _write_json(bank, manifest)

    with pytest.raises(ValueError, match="frozen/final split_index mismatch"):
        reporter.build_report(
            training_dir=training,
            bank_manifest=bank,
            expected_bank_manifest_sha256=_sha(bank),
            frozen_input_manifest=inputs,
            expected_frozen_input_manifest_sha256=input_sha,
            expected_val_count=4,
            expected_full_val_count=5,
            expected_excluded_val_count=1,
            scheduled_steps=(5_000, 10_000),
            bootstrap_draws=100,
        )


def test_clustered_interval_is_reproducible() -> None:
    deltas = np.asarray([1.0, 0.0, -1.0, 1.0], dtype=np.float64)
    systems = ["a", "a", "b", "c"]
    first = reporter.clustered_paired_interval(deltas, systems, draws=1_000, seed=7)
    second = reporter.clustered_paired_interval(deltas, systems, draws=1_000, seed=7)
    assert first == second
