#!/usr/bin/env python3
"""Seal the pre-registered U0-paired S50 confidence checkpoint decision."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

PROTOCOL_ID = "EFFDOCK-S50-MATCHED-CONFIDENCE-TRAIN-VAL-V1"
BANK_SCHEMA = "effdock.s50_confidence_bank.manifest.v1"
INPUT_SCHEMA = "effdock.s50_confidence_bank.inputs.v1"
BANK_PROTOCOL_ID = "EFFDOCK-S50-CONFIDENCE-TRAINING-BANK-V1"
EVAL_SCHEMA = "effdock.confidence_eval_ledger.v1"
EVAL_TARGET = "pose_rmsd_symmetry_no_align"
POSE_TAG = "s50_n100_s10_sig2_latep3_pc10_rdkitseed0"
SCHEDULED_STEPS = tuple(range(5_000, 50_001, 5_000))
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 20_260_816
MIN_DELTA_PP = 3.0
WARM_START_SHA256 = "e31fde6f351284205c78f7a1510002779c43312e94d9f82003d47a14d72bc78f"
SOURCE_VAL_BANK_SHA256 = "928b7219ed1ef8375c1ee52470f6ef606b8fca4d5bf4ea5c51355e8332e29a4b"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ordered_ids_sha256(ids: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for sample_id in ids:
        digest.update(sample_id.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def require_sha256(value: Any, *, label: str) -> str:
    normalized = str(value).lower()
    if len(normalized) != 64 or any(ch not in "0123456789abcdef" for ch in normalized):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return normalized


def verify_file(path: Path, expected_sha256: str, *, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    expected = require_sha256(expected_sha256, label=f"{label} expected SHA-256")
    actual = file_sha256(path)
    if actual != expected:
        raise ValueError(
            f"{label} SHA-256 mismatch: expected={expected} actual={actual} path={path}"
        )
    return actual


def load_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return payload


def _system_id(sample_key: str, declared: Any) -> str:
    if not isinstance(declared, str) or not declared:
        raise ValueError(f"record {sample_key!r} requires an authoritative system_id")
    return declared


def _validate_bank_inventory(
    manifest: dict[str, Any],
    *,
    expected_val_count: int,
    expected_full_val_count: int,
    expected_excluded_val_count: int,
) -> tuple[list[str], dict[str, str]]:
    if manifest.get("schema_version") != BANK_SCHEMA:
        raise ValueError(f"bank schema must be {BANK_SCHEMA!r}")
    if manifest.get("protocol_id") != BANK_PROTOCOL_ID:
        raise ValueError("bank protocol identity mismatch")
    if manifest.get("study_protocol_id") != PROTOCOL_ID:
        raise ValueError("bank study protocol identity mismatch")
    if manifest.get("status") != "complete" or manifest.get("claim_eligible") is not True:
        raise ValueError("final bank manifest must be complete and claim_eligible")
    if manifest.get("pose_tag") != POSE_TAG:
        raise ValueError("final bank manifest pose tag mismatch")
    inventory = manifest.get("inventory")
    records = manifest.get("records")
    if not isinstance(inventory, dict) or not isinstance(records, list):
        raise ValueError("bank manifest requires inventory and records")
    val_inventory = inventory.get("val")
    if not isinstance(val_inventory, dict):
        raise ValueError("bank manifest requires validation inventory")
    observed_counts = (
        int(val_inventory.get("eligible_count", -1)),
        int(val_inventory.get("full_count", -1)),
        int(val_inventory.get("excluded_count", -1)),
        int(val_inventory.get("record_count", -1)),
    )
    expected_counts = (
        expected_val_count,
        expected_full_val_count,
        expected_excluded_val_count,
        expected_val_count,
    )
    if observed_counts != expected_counts:
        raise ValueError(
            f"validation inventory mismatch: observed={observed_counts} expected={expected_counts}"
        )

    val_records = [row for row in records if isinstance(row, dict) and row.get("split") == "val"]
    if len(val_records) != expected_val_count:
        raise ValueError("validation record count does not match the frozen eligible inventory")
    try:
        split_indices = [int(row["split_index"]) for row in val_records]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("validation bank records require integer split_index") from exc
    if any(left >= right for left, right in zip(split_indices, split_indices[1:])):
        raise ValueError("validation split indices must preserve strict original-split order")

    ids: list[str] = []
    systems: dict[str, str] = {}
    for row in val_records:
        sample_key = row.get("sample_key")
        if not isinstance(sample_key, str) or not sample_key:
            raise ValueError("validation bank record has invalid sample_key")
        if sample_key in systems:
            raise ValueError(f"duplicate validation sample_key: {sample_key}")
        if row.get("status") != "complete":
            raise ValueError(f"validation record {sample_key} is not complete")
        if int(row.get("pose_count", -1)) != 100:
            raise ValueError(f"validation record {sample_key} does not contain 100 poses")
        require_sha256(row.get("pt_sha256"), label=f"record {sample_key} pt_sha256")
        systems[sample_key] = _system_id(sample_key, row.get("system_id"))
        ids.append(sample_key)
    return ids, systems


def _validate_frozen_input_join(
    frozen_inputs: dict[str, Any],
    final_manifest: dict[str, Any],
    *,
    expected_ids: Sequence[str],
    expected_systems: dict[str, str],
    expected_full_val_count: int,
    expected_excluded_val_count: int,
) -> None:
    if (
        frozen_inputs.get("schema_version") != INPUT_SCHEMA
        or frozen_inputs.get("protocol_id") != BANK_PROTOCOL_ID
        or frozen_inputs.get("study_protocol_id") != PROTOCOL_ID
        or frozen_inputs.get("status") != "complete"
    ):
        raise ValueError("frozen input manifest schema/protocol/status mismatch")
    inventory = frozen_inputs.get("inventory")
    records = frozen_inputs.get("records")
    if not isinstance(inventory, dict) or not isinstance(records, list):
        raise ValueError("frozen input manifest requires inventory and records")
    val_inventory = inventory.get("val")
    final_inventory = final_manifest.get("inventory", {}).get("val")
    if not isinstance(val_inventory, dict) or not isinstance(final_inventory, dict):
        raise ValueError("missing frozen/final validation inventories")
    eligible_ids = val_inventory.get("eligible_ids")
    excluded_ids = val_inventory.get("excluded_ids")
    if (
        not isinstance(eligible_ids, list)
        or not all(isinstance(value, str) and value for value in eligible_ids)
        or not isinstance(excluded_ids, list)
        or not all(isinstance(value, str) and value for value in excluded_ids)
    ):
        raise ValueError("frozen validation eligible/excluded ID ledgers are invalid")
    if eligible_ids != list(expected_ids):
        raise ValueError("final validation records do not exactly match frozen eligible order")
    if len(excluded_ids) != expected_excluded_val_count or set(eligible_ids) & set(excluded_ids):
        raise ValueError("frozen validation exclusions are incomplete or overlap eligible IDs")

    val_records = [row for row in records if isinstance(row, dict) and row.get("split") == "val"]
    if len(val_records) != expected_full_val_count:
        raise ValueError("frozen input manifest does not account for every validation key")
    full_ids = [row.get("sample_key") for row in val_records]
    if not all(isinstance(value, str) and value for value in full_ids):
        raise ValueError("frozen validation records contain invalid sample keys")
    if len(full_ids) != len(set(full_ids)):
        raise ValueError("frozen validation records contain duplicate sample keys")
    try:
        split_indices = [int(row["split_index"]) for row in val_records]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("frozen validation records require original split_index") from exc
    if split_indices != list(range(1, expected_full_val_count + 1)):
        raise ValueError("frozen validation records do not preserve the complete original order")

    eligible_rows = [row for row in val_records if row.get("status") == "eligible"]
    excluded_rows = [row for row in val_records if row.get("status") == "input_ineligible"]
    if [row.get("sample_key") for row in eligible_rows] != list(expected_ids):
        raise ValueError("frozen eligible record states disagree with the final bank")
    if [row.get("sample_key") for row in excluded_rows] != excluded_ids:
        raise ValueError("frozen ineligible record states disagree with the exclusion ledger")
    if len(eligible_rows) + len(excluded_rows) != expected_full_val_count:
        raise ValueError("frozen validation records contain an unrecognized terminal state")
    final_val_rows = [
        row
        for row in final_manifest.get("records", [])
        if isinstance(row, dict) and row.get("split") == "val"
    ]
    final_by_id = {str(row.get("sample_key")): row for row in final_val_rows}
    if list(final_by_id) != list(expected_ids):
        raise ValueError("final validation record order disagrees with frozen eligible IDs")
    for row in eligible_rows:
        sample_key = str(row["sample_key"])
        if row.get("system_id") != expected_systems[sample_key]:
            raise ValueError(f"{sample_key}: frozen/final system_id mismatch")
        if int(row["split_index"]) != int(final_by_id[sample_key]["split_index"]):
            raise ValueError(f"{sample_key}: frozen/final split_index mismatch")

    expected_hashes = {
        "full_ids_sha256": ordered_ids_sha256([str(value) for value in full_ids]),
        "eligible_ids_sha256": ordered_ids_sha256(list(expected_ids)),
        "excluded_ids_sha256": ordered_ids_sha256(excluded_ids),
    }
    for key, expected in expected_hashes.items():
        if val_inventory.get(key) != expected or final_inventory.get(key) != expected:
            raise ValueError(f"frozen/final validation {key} mismatch")


def _find_source_val_bank_sha(payload: Any) -> set[str]:
    """Collect explicitly named validation-bank hashes from a frozen input manifest."""
    found: set[str] = set()
    if isinstance(payload, dict):
        for key, value in payload.items():
            normalized = str(key).lower()
            if "val" in normalized and "bank" in normalized and isinstance(value, dict):
                sha = value.get("sha256")
                if isinstance(sha, str) and len(sha) == 64:
                    found.add(sha.lower())
            found.update(_find_source_val_bank_sha(value))
    elif isinstance(payload, list):
        for value in payload:
            found.update(_find_source_val_bank_sha(value))
    return found


def _k2_slice(k2: int) -> str:
    if k2 == 0:
        return "0"
    if k2 <= 4:
        return "1_4"
    if k2 <= 9:
        return "5_9"
    return "ge10"


def _finite_number(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _validate_eval_record(
    record: dict[str, Any],
    *,
    expected_pid: str,
    expected_system_id: str,
) -> dict[str, Any]:
    if record.get("pid") != expected_pid:
        raise ValueError(
            f"evaluation ID/order mismatch: expected={expected_pid!r} got={record.get('pid')!r}"
        )
    if record.get("system_id") != expected_system_id:
        raise ValueError(f"{expected_pid}: evaluation system_id disagrees with bank manifest")
    pose_count = int(record.get("pose_count", -1))
    if pose_count != 100:
        raise ValueError(f"{expected_pid}: evaluation pose_count must be 100")
    top1_index = record.get("top1_index")
    if isinstance(top1_index, bool) or not isinstance(top1_index, int):
        raise ValueError(f"{expected_pid}: top1_index must be an integer")
    if not 0 <= top1_index < pose_count:
        raise ValueError(f"{expected_pid}: top1_index is out of range")
    top5_indices = record.get("top5_indices")
    if (
        not isinstance(top5_indices, list)
        or not 1 <= len(top5_indices) <= 5
        or any(isinstance(index, bool) or not isinstance(index, int) for index in top5_indices)
        or len(set(top5_indices)) != len(top5_indices)
        or any(index < 0 or index >= pose_count for index in top5_indices)
        or top5_indices[0] != top1_index
    ):
        raise ValueError(f"{expected_pid}: invalid stable Top-5 index inventory")

    top1 = _finite_number(record.get("top1_rmsd"), label=f"{expected_pid} top1_rmsd")
    top5 = _finite_number(record.get("top5_best_rmsd"), label=f"{expected_pid} top5_best_rmsd")
    oracle = _finite_number(record.get("oracle_rmsd"), label=f"{expected_pid} oracle_rmsd")
    if oracle > top5 + 1e-6 or top5 > top1 + 1e-6:
        raise ValueError(f"{expected_pid}: oracle/Top-5/Top-1 RMSDs are inconsistent")
    k2 = record.get("oracle_k2")
    if isinstance(k2, bool) or not isinstance(k2, int) or not 0 <= k2 <= pose_count:
        raise ValueError(f"{expected_pid}: oracle_k2 is invalid")
    if record.get("oracle_k2_slice") != _k2_slice(k2):
        raise ValueError(f"{expected_pid}: oracle_k2_slice is inconsistent")

    expected_bools = {
        "top1_lt2": top1 < 2.0,
        "top5_lt2": top5 < 2.0,
        "oracle_lt2": oracle < 2.0,
    }
    for key, expected in expected_bools.items():
        if record.get(key) is not expected:
            raise ValueError(f"{expected_pid}: {key} is inconsistent with its RMSD")
    if expected_bools["oracle_lt2"] != (k2 > 0):
        raise ValueError(f"{expected_pid}: oracle_k2 and oracle_lt2 disagree")
    return {
        "top1_index": top1_index,
        "top1_rmsd": top1,
        "top5_best_rmsd": top5,
        "oracle_rmsd": oracle,
        "oracle_k2": k2,
        **expected_bools,
    }


def _load_eval_ledger(
    path: Path,
    *,
    expected_step: int,
    expected_bank_sha256: str,
    expected_ids: Sequence[str],
    expected_system_ids: dict[str, str],
) -> tuple[dict[str, dict[str, Any]], str]:
    payload = load_json(path, label=f"evaluation ledger U{expected_step}")
    if payload.get("schema_version") != EVAL_SCHEMA or payload.get("status") != "complete":
        raise ValueError(f"evaluation ledger U{expected_step} schema/status mismatch")
    if int(payload.get("step", -1)) != expected_step:
        raise ValueError(f"evaluation ledger step mismatch at U{expected_step}")
    if payload.get("eval_target") != EVAL_TARGET:
        raise ValueError(f"evaluation ledger U{expected_step} target mismatch")
    if payload.get("bank_manifest_sha256") != expected_bank_sha256:
        raise ValueError(f"evaluation ledger U{expected_step} bank identity mismatch")
    records = payload.get("records")
    if not isinstance(records, list) or int(payload.get("record_count", -1)) != len(expected_ids):
        raise ValueError(f"evaluation ledger U{expected_step} record count mismatch")
    if len(records) != len(expected_ids):
        raise ValueError(f"evaluation ledger U{expected_step} is incomplete")
    validated: dict[str, dict[str, Any]] = {}
    for expected_pid, raw in zip(expected_ids, records):
        if not isinstance(raw, dict):
            raise ValueError(f"evaluation ledger U{expected_step} contains a non-object record")
        if expected_pid in validated:
            raise ValueError(f"evaluation ledger U{expected_step} contains duplicate IDs")
        validated[expected_pid] = _validate_eval_record(
            raw,
            expected_pid=expected_pid,
            expected_system_id=expected_system_ids[expected_pid],
        )
    return validated, file_sha256(path)


def _evaluation_diagnostics(
    records: dict[str, dict[str, Any]], expected_ids: Sequence[str]
) -> dict[str, Any]:
    """Summarize sealed targets without affecting checkpoint selection or the gate."""

    ordered = [records[sample_id] for sample_id in expected_ids]
    selected = np.asarray([row["top1_rmsd"] for row in ordered], dtype=np.float64)
    top5 = np.asarray([row["top5_best_rmsd"] for row in ordered], dtype=np.float64)
    oracle = np.asarray([row["oracle_rmsd"] for row in ordered], dtype=np.float64)
    oracle_k2 = np.asarray([row["oracle_k2"] for row in ordered], dtype=np.int64)
    strata: dict[str, dict[str, Any]] = {}
    for name in ("0", "1_4", "5_9", "ge10"):
        indices = [
            index
            for index, row in enumerate(ordered)
            if _k2_slice(int(row["oracle_k2"])) == name
        ]
        successes = sum(int(ordered[index]["top1_lt2"]) for index in indices)
        stratum_selected = selected[indices]
        strata[name] = {
            "n": len(indices),
            "top1_successes": successes,
            "eval_top1_lt2": (
                100.0 * successes / len(indices) if indices else None
            ),
            "selected_rmsd_mean": (
                float(stratum_selected.mean()) if indices else None
            ),
            "selected_rmsd_median": (
                float(np.median(stratum_selected)) if indices else None
            ),
        }
    return {
        "selected_rmsd_mean": float(selected.mean()),
        "selected_rmsd_median": float(np.median(selected)),
        "top5_best_rmsd_mean": float(top5.mean()),
        "top5_best_rmsd_median": float(np.median(top5)),
        "oracle_rmsd_mean": float(oracle.mean()),
        "oracle_rmsd_median": float(np.median(oracle)),
        "oracle_k2_total": int(oracle_k2.sum()),
        "oracle_k2_mean": float(oracle_k2.mean()),
        "oracle_k2_median": float(np.median(oracle_k2)),
        "k2_strata": strata,
    }


def clustered_paired_interval(
    deltas: np.ndarray,
    system_ids: Sequence[str],
    *,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    if deltas.ndim != 1 or len(deltas) != len(system_ids) or len(deltas) == 0:
        raise ValueError("paired deltas and system IDs must be non-empty aligned vectors")
    unique_systems = sorted(set(system_ids))
    if not unique_systems:
        raise ValueError("no PLINDER system clusters")
    cluster_index = {system_id: index for index, system_id in enumerate(unique_systems)}
    cluster_sums = np.zeros(len(unique_systems), dtype=np.float64)
    cluster_sizes = np.zeros(len(unique_systems), dtype=np.int64)
    for delta, system_id in zip(deltas, system_ids):
        index = cluster_index[system_id]
        cluster_sums[index] += float(delta)
        cluster_sizes[index] += 1
    rng = np.random.Generator(np.random.PCG64(seed))
    estimates = np.empty(draws, dtype=np.float64)
    batch_size = 256
    for start in range(0, draws, batch_size):
        stop = min(draws, start + batch_size)
        sampled = rng.integers(
            0,
            len(unique_systems),
            size=(stop - start, len(unique_systems)),
            endpoint=False,
        )
        numerator = cluster_sums[sampled].sum(axis=1)
        denominator = cluster_sizes[sampled].sum(axis=1)
        estimates[start:stop] = 100.0 * numerator / denominator
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def _load_confidence_checkpoint(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing selected confidence checkpoint: {path}")
    with torch.serialization.safe_globals([type(Path())]):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("selected confidence checkpoint must be a mapping")
    return checkpoint


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("xb") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build_report(
    *,
    training_dir: Path,
    bank_manifest: Path,
    expected_bank_manifest_sha256: str,
    frozen_input_manifest: Path,
    expected_frozen_input_manifest_sha256: str,
    expected_val_count: int = 1_035,
    expected_full_val_count: int = 1_076,
    expected_excluded_val_count: int = 41,
    scheduled_steps: Sequence[int] = SCHEDULED_STEPS,
    bootstrap_draws: int = BOOTSTRAP_DRAWS,
    bootstrap_seed: int = BOOTSTRAP_SEED,
    min_delta_pp: float = MIN_DELTA_PP,
) -> dict[str, Any]:
    bank_sha256 = verify_file(
        bank_manifest,
        expected_bank_manifest_sha256,
        label="bank manifest",
    )
    frozen_input_sha256 = verify_file(
        frozen_input_manifest,
        expected_frozen_input_manifest_sha256,
        label="frozen input manifest",
    )
    frozen_inputs = load_json(frozen_input_manifest, label="frozen input manifest")
    source_val_shas = _find_source_val_bank_sha(frozen_inputs)
    if SOURCE_VAL_BANK_SHA256 not in source_val_shas:
        raise ValueError("frozen input manifest does not pin the exact source validation bank")

    manifest = load_json(bank_manifest, label="bank manifest")
    expected_ids, systems = _validate_bank_inventory(
        manifest,
        expected_val_count=expected_val_count,
        expected_full_val_count=expected_full_val_count,
        expected_excluded_val_count=expected_excluded_val_count,
    )
    declared_input = manifest.get("input_manifest")
    if not isinstance(declared_input, dict) or declared_input.get("sha256") != frozen_input_sha256:
        raise ValueError("bank manifest input-manifest provenance mismatch")
    _validate_frozen_input_join(
        frozen_inputs,
        manifest,
        expected_ids=expected_ids,
        expected_systems=systems,
        expected_full_val_count=expected_full_val_count,
        expected_excluded_val_count=expected_excluded_val_count,
    )

    expected_eval_steps = (0, *tuple(scheduled_steps))
    actual_ledger_names = sorted(path.name for path in training_dir.glob("eval_u*.json"))
    required_ledger_names = sorted(f"eval_u{step:06d}.json" for step in expected_eval_steps)
    if actual_ledger_names != required_ledger_names:
        raise ValueError(
            "evaluation schedule is incomplete or adaptive: "
            f"observed={actual_ledger_names} expected={required_ledger_names}"
        )

    evaluations: dict[int, dict[str, dict[str, Any]]] = {}
    ledger_hashes: dict[str, str] = {}
    for step in expected_eval_steps:
        path = training_dir / f"eval_u{step:06d}.json"
        records, digest = _load_eval_ledger(
            path,
            expected_step=step,
            expected_bank_sha256=bank_sha256,
            expected_ids=expected_ids,
            expected_system_ids=systems,
        )
        evaluations[step] = records
        ledger_hashes[str(step)] = digest

    baseline = evaluations[0]
    baseline_oracle = {
        pid: (
            row["oracle_k2"],
            row["oracle_rmsd"],
            row["oracle_lt2"],
        )
        for pid, row in baseline.items()
    }
    looks: list[dict[str, Any]] = []
    for step in scheduled_steps:
        records = evaluations[step]
        current_oracle = {
            pid: (
                row["oracle_k2"],
                row["oracle_rmsd"],
                row["oracle_lt2"],
            )
            for pid, row in records.items()
        }
        if current_oracle != baseline_oracle:
            raise ValueError(f"validation target/oracle inventory changed at U{step}")
        top1_successes = sum(int(row["top1_lt2"]) for row in records.values())
        top5_successes = sum(int(row["top5_lt2"]) for row in records.values())
        oracle_successes = sum(int(row["oracle_lt2"]) for row in records.values())
        if not top1_successes <= top5_successes <= oracle_successes:
            raise ValueError(f"Top-1/Top-5/oracle guard failed at U{step}")
        looks.append(
            {
                "step": int(step),
                "top1_successes": top1_successes,
                "eval_top1_lt2": 100.0 * top1_successes / expected_val_count,
                "top5_successes": top5_successes,
                "eval_top5_lt2": 100.0 * top5_successes / expected_val_count,
                "oracle_successes": oracle_successes,
                "oracle_lt2": 100.0 * oracle_successes / expected_val_count,
                "diagnostics": _evaluation_diagnostics(records, expected_ids),
                "eval_ledger_sha256": ledger_hashes[str(step)],
            }
        )
    if not looks:
        raise ValueError("at least one non-U0 scheduled evaluation is required")
    chosen = max(looks, key=lambda row: (float(row["eval_top1_lt2"]), -int(row["step"])))
    chosen_step = int(chosen["step"])

    baseline_successes = sum(int(row["top1_lt2"]) for row in baseline.values())
    baseline_top1 = 100.0 * baseline_successes / expected_val_count
    baseline_diagnostics = _evaluation_diagnostics(baseline, expected_ids)
    deltas = np.asarray(
        [
            int(evaluations[chosen_step][pid]["top1_lt2"]) - int(baseline[pid]["top1_lt2"])
            for pid in expected_ids
        ],
        dtype=np.float64,
    )
    delta_pp = 100.0 * float(deltas.mean())
    ci_low, ci_high = clustered_paired_interval(
        deltas,
        [systems[pid] for pid in expected_ids],
        draws=bootstrap_draws,
        seed=bootstrap_seed,
    )

    metrics_path = training_dir / "metrics.json"
    metrics = load_json(metrics_path, label="training metrics")
    metrics_sha256 = file_sha256(metrics_path)
    if int(metrics.get("final_step", -1)) != 50_000:
        raise ValueError("training did not complete exactly 50,000 updates")
    if metrics.get("best_metric") != "eval_top1_lt2":
        raise ValueError("training best metric was not pure predicted-RMSD Top-1")
    if metrics.get("eval_target") != EVAL_TARGET:
        raise ValueError("training evaluation target mismatch")
    if int(metrics.get("effective_global_batch_complexes", -1)) != 4:
        raise ValueError("training effective global batch was not four")
    provenance = metrics.get("bank_provenance")
    if not isinstance(provenance, dict) or provenance.get("sha256") != bank_sha256:
        raise ValueError("training metrics bank provenance mismatch")
    initialization = metrics.get("initialization_provenance")
    if not isinstance(initialization, dict) or initialization.get("sha256") != WARM_START_SHA256:
        raise ValueError("training did not preserve the frozen warm-start provenance")

    best_overall_score = max(baseline_top1, float(chosen["eval_top1_lt2"]))
    expected_best_step = 0 if baseline_top1 >= float(chosen["eval_top1_lt2"]) else chosen_step
    if not math.isclose(
        float(metrics.get("best_score", math.nan)), best_overall_score, abs_tol=1e-4
    ):
        raise ValueError("training best score disagrees with the sealed evaluation ledgers")
    best_checkpoint = training_dir / "best.pt"
    checkpoint = _load_confidence_checkpoint(best_checkpoint)
    if int(checkpoint.get("step", -1)) != expected_best_step:
        raise ValueError("best.pt step disagrees with the frozen strict-improvement rule")
    checkpoint_metrics = checkpoint.get("metrics")
    if not isinstance(checkpoint_metrics, dict) or not math.isclose(
        float(checkpoint_metrics.get("eval_top1_lt2", math.nan)),
        best_overall_score,
        abs_tol=1e-4,
    ):
        raise ValueError("best.pt metric disagrees with the evaluation ledger")
    checkpoint_bank = checkpoint.get("bank_provenance")
    if not isinstance(checkpoint_bank, dict) or checkpoint_bank.get("sha256") != bank_sha256:
        raise ValueError("best.pt bank provenance mismatch")
    if int(checkpoint.get("effective_global_batch_complexes", -1)) != 4:
        raise ValueError("best.pt effective global batch provenance mismatch")
    best_checkpoint_sha256 = file_sha256(best_checkpoint)

    effect_gate = delta_pp >= min_delta_pp
    interval_gate = ci_low > 0.0
    admitted = effect_gate and interval_gate
    report = {
        "schema_version": "effdock.s50_matched_confidence_training_report.v1",
        "protocol_id": PROTOCOL_ID,
        "status": "complete",
        "claim_boundary": "internal_repeated_validation_checkpoint_decision",
        "inputs": {
            "bank_manifest": {
                "path": str(bank_manifest.resolve()),
                "sha256": bank_sha256,
            },
            "frozen_input_manifest": {
                "path": str(frozen_input_manifest.resolve()),
                "sha256": frozen_input_sha256,
            },
            "source_validation_bank_sha256": SOURCE_VAL_BANK_SHA256,
            "training_metrics": {
                "path": str(metrics_path.resolve()),
                "sha256": metrics_sha256,
            },
            "warm_start_sha256": WARM_START_SHA256,
            "evaluation_ledgers": ledger_hashes,
        },
        "configuration": {
            "selector": "stable_ascending_predicted_rmsd",
            "eval_target": EVAL_TARGET,
            "threshold_angstrom": 2.0,
            "scheduled_steps": list(scheduled_steps),
            "checkpoint_tie_break": "earliest_step",
            "bootstrap": {
                "unit": "PLINDER_system_id_cluster",
                "draws": bootstrap_draws,
                "rng": "numpy.PCG64",
                "seed": bootstrap_seed,
                "interval": "percentile_95",
            },
            "minimum_delta_pp": min_delta_pp,
        },
        "inventory": {
            "eligible_validation_samples": expected_val_count,
            "full_validation_samples": expected_full_val_count,
            "predeclared_ineligible": expected_excluded_val_count,
            "system_clusters": len(set(systems.values())),
            "poses_per_sample": 100,
        },
        "baseline_u0": {
            "top1_successes": baseline_successes,
            "eval_top1_lt2": baseline_top1,
            "diagnostics": baseline_diagnostics,
            "eval_ledger_sha256": ledger_hashes["0"],
        },
        "scheduled_looks": looks,
        "selection": {
            "step": chosen_step,
            "eval_top1_lt2": float(chosen["eval_top1_lt2"]),
            "paired_delta_pp": delta_pp,
            "paired_ci95_pp": [ci_low, ci_high],
            "diagnostics": chosen["diagnostics"],
            "selected_only_from_5k_looks": True,
        },
        "guards": {
            "exact_fixed_schedule_complete": True,
            "no_adaptive_early_stop": True,
            "exact_val_inventory": True,
            "oracle_inventory_invariant": True,
            "pure_predicted_rmsd_selector": True,
            "k2_and_selected_rmsd_diagnostics_nonselecting": True,
            "full_50000_updates": True,
            "effective_global_batch_four": True,
            "best_checkpoint_matches_ledgers": True,
        },
        "gate": {
            "delta_at_least_3pp": effect_gate,
            "paired_ci_lower_above_zero": interval_gate,
            "new_checkpoint_admitted": admitted,
        },
        "best_checkpoint": {
            "path": str(best_checkpoint.resolve()),
            "sha256": best_checkpoint_sha256,
            "step": expected_best_step,
            "eval_top1_lt2": best_overall_score,
            "admitted_for_frozen_external_check": admitted,
        },
        "decision": (
            "freeze best.pt for one later Astex/PoseBusters check"
            if admitted
            else "retain the existing confidence checkpoint; do not open an external tuning loop"
        ),
    }
    return report


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-dir", type=Path, required=True)
    parser.add_argument("--bank-manifest", type=Path, required=True)
    parser.add_argument("--expected-bank-manifest-sha256", required=True)
    parser.add_argument("--frozen-input-manifest", type=Path, required=True)
    parser.add_argument("--expected-frozen-input-manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = build_report(
        training_dir=args.training_dir.resolve(),
        bank_manifest=args.bank_manifest.resolve(),
        expected_bank_manifest_sha256=args.expected_bank_manifest_sha256,
        frozen_input_manifest=args.frozen_input_manifest.resolve(),
        expected_frozen_input_manifest_sha256=args.expected_frozen_input_manifest_sha256,
    )
    _atomic_write_json(args.output.resolve(), report)
    print(json.dumps(report["gate"], sort_keys=True))


if __name__ == "__main__":
    main()
