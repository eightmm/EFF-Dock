#!/usr/bin/env python3
"""Fail-closed inventory, pairing, selector, and numerical audit for PLINDER."""

from __future__ import annotations

import argparse
import json
import math
import re
import stat
from pathlib import Path
from typing import Any

from plinder_guidance_common import (
    AUDIT_SCHEMA,
    BASE_SEED,
    ETA_TAGS,
    ETA_VALUES,
    EXPECTED_CONFIDENCE_SHA256,
    EXPECTED_CONFIG_SHA256,
    EXPECTED_COUNT,
    EXPECTED_DOCKING_SHA256,
    EXPECTED_GUIDANCE_IMPLEMENTATION_SHA256,
    EXPECTED_GUIDANCE_PARAMETER_SHA256,
    EXPECTED_SPLIT_SHA256,
    PRIMARY_SELECTOR,
    PROTOCOL_ID,
    SAMPLING_SCHEMA,
    canonical_json_sha256,
    exact_int,
    expected_ids,
    expected_num_shards,
    file_sha256,
    finite_float,
    ids_sha256,
    load_csv,
    load_split_ids,
    parse_bool,
    require_sha256,
    sampling_shard_dir,
    validate_raw_gate,
    verify_raw_manifest,
    write_json_noreplace,
)

EXPECTED_SETTINGS = {
    "num_samples": 100,
    "num_steps": 10,
    "model_pose_step_budget": 1000,
    "sigma": 0.5,
    "prior_pool_size": 100,
    "time_schedule": "late",
    "schedule_power": 3.0,
    "pocket_cutoff_angstrom": 10.0,
    "center_jitter_sigma": 0.0,
    "coupling": "normalized_drift",
    "receptor_policy": "geometry_only",
    "guidance_start_t": 0.5,
    "guidance_ramp_power": 1.0,
    "max_atom_force": 20.0,
    "max_translation_velocity": 5.0,
    "max_angular_velocity": 5.0,
    "max_atom_displacement_angstrom": 0.25,
    "max_backtracks": 8,
    "protein_shell_angstrom": 18.0,
    "vina_guidance_scale": 0.0,
    "refine": "none",
    "selector_profile": "confidence_cluster_free",
    "saved_primary_selector": PRIMARY_SELECTOR,
}

_REQUIRED_ROW_FIELDS = {
    "id",
    "plinder_system_id",
    "plinder_ligand_chain",
    "plinder_global_index",
    "processed_meta",
    "processed_meta_sha256",
    "protein",
    "ligand_ref",
    "protein_sha256",
    "ligand_reference_sha256",
    "saved_pose_sha256_json",
    "selector_profile",
    "num_samples",
    "oracle_index",
    "oracle_rmsd",
    "confidence_index",
    "confidence_rmsd",
    "confidence_pred_rmsd",
    "confidence_pred_success",
    "confidence_candidate_scores_json",
    "candidate_ensemble_sha256",
    "prior_pool_size",
    "sampling_seed",
    "prior_pool_sha256",
    "guidance_mode",
    "guidance_parameter_sha256",
    "guidance_direct_step_trace_json",
    "full_heavy_atom_bijection",
}
_ATTEMPT_RE = re.compile(
    r"shard-(?P<shard>[0-9]{3})-of-(?P<count>[0-9]{3})\.attempt-[a-z0-9_]{8}\Z"
)
GPU_NAME_FRAGMENTS_BY_PARTITION = {
    "6000ada": ("RTX 6000 Ada",),
    "heavy": ("H100", "RTX PRO 6000"),
}
ALLOWED_GPU_NAME_FRAGMENTS = tuple(
    fragment for fragments in GPU_NAME_FRAGMENTS_BY_PARTITION.values() for fragment in fragments
)
MIN_GPU_TOTAL_MEMORY_BYTES = 48_000 * 1024**2


def _stale_attempt_identity(path: Path, *, shard_index: int) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    total_bytes = 0
    for child in sorted(path.rglob("*")):
        relative = child.relative_to(path).as_posix()
        mode = child.lstat().st_mode
        if stat.S_ISLNK(mode):
            raise ValueError(f"stale attempt contains a symlink: {child}")
        if stat.S_ISDIR(mode):
            entries.append({"path": relative, "type": "directory"})
        elif stat.S_ISREG(mode):
            size = child.stat().st_size
            total_bytes += size
            entries.append(
                {
                    "path": relative,
                    "type": "file",
                    "size_bytes": size,
                    "sha256": file_sha256(child),
                }
            )
        else:
            raise ValueError(f"stale attempt contains a special filesystem entry: {child}")
    return {
        "name": path.name,
        "shard_index": shard_index,
        "file_count": sum(entry["type"] == "file" for entry in entries),
        "total_size_bytes": total_bytes,
        "tree_sha256": canonical_json_sha256(entries),
    }


def inspect_incomplete_attempts(incomplete_root: Path, *, num_shards: int) -> list[dict[str, Any]]:
    """Allow exact publish locks and named stale attempts, rejecting all else."""
    if not incomplete_root.is_dir() or incomplete_root.is_symlink():
        raise ValueError(f"missing regular incomplete-attempt directory: {incomplete_root}")
    expected_locks = {
        incomplete_root / f".shard-{shard:03d}-of-{num_shards:03d}.publish.lock"
        for shard in range(num_shards)
    }
    actual_entries = set(incomplete_root.iterdir())
    for lock in expected_locks:
        if lock not in actual_entries:
            raise ValueError(f"missing publish lock for completed shard: {lock}")
        if lock.is_symlink() or not lock.is_file() or lock.stat().st_size != 0:
            raise ValueError(f"publish lock must be a zero-byte regular file: {lock}")
    recovered: list[dict[str, Any]] = []
    for entry in sorted(actual_entries - expected_locks):
        match = _ATTEMPT_RE.fullmatch(entry.name)
        if match is None or entry.is_symlink() or not entry.is_dir():
            raise ValueError(f"unexpected incomplete-attempt entry: {entry}")
        shard_index = int(match.group("shard"))
        count = int(match.group("count"))
        if count != num_shards or not 0 <= shard_index < num_shards:
            raise ValueError(f"stale attempt has an invalid shard identity: {entry}")
        recovered.append(_stale_attempt_identity(entry, shard_index=shard_index))
    return recovered


def _require_path_hash(raw_path: object, raw_sha256: object, *, label: str) -> Path:
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"{label} path is missing")
    path = Path(raw_path)
    expected = require_sha256(raw_sha256, label=f"{label} SHA-256")
    if not path.is_file() or file_sha256(path) != expected:
        raise ValueError(f"{label} changed or is missing: {path}")
    return path.resolve()


def _validate_cuda_runtime(runtime: object, *, label: str) -> dict[str, Any]:
    if not isinstance(runtime, dict) or runtime.get("device") != "cuda":
        raise ValueError(f"{label}: sampling did not report CUDA")
    partition = runtime.get("slurm_partition")
    expected_gpu_fragments = (
        GPU_NAME_FRAGMENTS_BY_PARTITION.get(partition) if isinstance(partition, str) else None
    )
    if expected_gpu_fragments is None:
        allowed = ", ".join(GPU_NAME_FRAGMENTS_BY_PARTITION)
        raise ValueError(f"{label}: runtime.slurm_partition must be one of {allowed}")
    gpu = runtime.get("gpu")
    if not isinstance(gpu, str) or not any(fragment in gpu for fragment in expected_gpu_fragments):
        allowed = ", ".join(expected_gpu_fragments)
        raise ValueError(
            f"{label}: sampling GPU is not allowed on {partition}; expected one of {allowed}"
        )
    total = exact_int(runtime.get("gpu_total_memory_bytes"), label=f"{label}.gpu memory")
    allocated = exact_int(
        runtime.get("cuda_max_memory_allocated_bytes"), label=f"{label}.CUDA allocated"
    )
    if total < MIN_GPU_TOTAL_MEMORY_BYTES or allocated < 1 or allocated >= int(0.9 * total):
        raise ValueError(f"{label}: CUDA memory headroom gate failed")
    return runtime


def _validate_summary(
    path: Path,
    *,
    mode: str,
    eta: float,
    shard_index: int,
    num_shards: int,
    split_ids: list[str],
    selected_ids: list[str],
    raw_root: Path,
) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    expected_top = {
        "schema_version": SAMPLING_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "status": "complete",
        "run_id": mode,
        "eta_tag": ETA_TAGS[eta],
        "failures": [],
    }
    for key, expected in expected_top.items():
        if raw.get(key) != expected:
            raise ValueError(f"{path}: {key} mismatch")
    settings = raw.get("settings")
    expected_settings = {"eta": eta, **EXPECTED_SETTINGS}
    if settings != expected_settings:
        raise ValueError(f"{path}: frozen settings mismatch")
    split = raw.get("split")
    if not isinstance(split, dict):
        raise ValueError(f"{path}: split metadata missing")
    expected_split = {
        "sha256": EXPECTED_SPLIT_SHA256,
        "expected_unique_val_count": EXPECTED_COUNT,
        "full_val_ids_sha256": ids_sha256(split_ids),
    }
    for key, expected in expected_split.items():
        if split.get(key) != expected:
            raise ValueError(f"{path}: split.{key} mismatch")
    inventory = raw.get("inventory")
    if not isinstance(inventory, dict):
        raise ValueError(f"{path}: inventory missing")
    assigned_ids = selected_ids[shard_index::num_shards]
    expected_inventory = {
        "full_val_count": EXPECTED_COUNT,
        "full_val_ids": split_ids,
        "selected_count": len(selected_ids),
        "selected_ids": selected_ids,
        "selected_ids_sha256": ids_sha256(selected_ids),
        "num_shards": num_shards,
        "shard_index": shard_index,
        "assigned_count": len(assigned_ids),
        "assigned_ids": assigned_ids,
        "assigned_ids_sha256": ids_sha256(assigned_ids),
        "attempted_ids": assigned_ids,
        "success_count": len(assigned_ids),
        "success_ids": assigned_ids,
        "not_attempted_ids": [],
        "failure_count": 0,
    }
    for key, expected in expected_inventory.items():
        if inventory.get(key) != expected:
            raise ValueError(f"{path}: inventory.{key} mismatch")
    coverage = raw.get("processed_coverage")
    if not isinstance(coverage, dict):
        raise ValueError(f"{path}: processed_coverage missing")
    if (
        coverage.get("expected") != EXPECTED_COUNT
        or coverage.get("present") != EXPECTED_COUNT
        or coverage.get("missing_ids") != []
    ):
        raise ValueError(f"{path}: processed validation coverage is incomplete")
    source = raw.get("source")
    if not isinstance(source, dict):
        raise ValueError(f"{path}: source metadata missing")
    if source.get("release") != "2024-06/v2":
        raise ValueError(f"{path}: PLINDER release mismatch")
    source_root = source.get("raw_root")
    if not isinstance(source_root, str) or Path(source_root).resolve() != raw_root.resolve():
        raise ValueError(f"{path}: raw root mismatch")
    identities = raw.get("fixed_identities")
    if not isinstance(identities, dict):
        raise ValueError(f"{path}: fixed identities missing")
    expected_hashes = {
        "docking_checkpoint": EXPECTED_DOCKING_SHA256,
        "confidence_checkpoint": EXPECTED_CONFIDENCE_SHA256,
        "config": EXPECTED_CONFIG_SHA256,
        "guidance_parameters": EXPECTED_GUIDANCE_PARAMETER_SHA256,
        "guidance_implementation": EXPECTED_GUIDANCE_IMPLEMENTATION_SHA256,
    }
    for key, expected in expected_hashes.items():
        identity = identities.get(key)
        if not isinstance(identity, dict) or identity.get("sha256") != expected:
            raise ValueError(f"{path}: fixed identity mismatch for {key}")
    seed_contract = raw.get("seed_contract")
    if not isinstance(seed_contract, dict) or seed_contract.get("base_seed") != BASE_SEED:
        raise ValueError(f"{path}: seed contract mismatch")
    _validate_cuda_runtime(raw.get("runtime"), label=str(path))
    artifacts = raw.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError(f"{path}: artifacts missing")
    csv_path = path.parent / "results.csv"
    if artifacts.get("csv") != str(csv_path):
        raise ValueError(f"{path}: results.csv path binding mismatch")
    if artifacts.get("csv_sha256") != file_sha256(csv_path):
        raise ValueError(f"{path}: results.csv hash binding mismatch")
    if artifacts.get("summary") != str(path):
        raise ValueError(f"{path}: summary path binding mismatch")
    return raw


def _validate_row(
    row: dict[str, str],
    fields: list[str],
    *,
    sample_id: str,
    global_index: int,
    eta: float,
    pose_dir: Path,
    raw_gate_asset: dict[str, Any],
) -> dict[str, Any]:
    missing = sorted(_REQUIRED_ROW_FIELDS - set(fields))
    if missing:
        raise ValueError(f"{sample_id}: sampling CSV lacks fields {missing}")
    forbidden = [
        field
        for field in fields
        if field.startswith(("vina_", "confidence_final_", "cluster_", "density_"))
    ]
    if forbidden:
        raise ValueError(f"{sample_id}: forbidden selector columns are present: {forbidden}")
    if row.get("id") != sample_id:
        raise ValueError(f"sampling row ID/order mismatch for {sample_id}")
    if row.get("selector_profile") != "confidence_cluster_free":
        raise ValueError(f"{sample_id}: selector profile mismatch")
    if exact_int(row.get("plinder_global_index"), label="global index") != global_index:
        raise ValueError(f"{sample_id}: global index mismatch")
    if exact_int(row.get("sampling_seed"), label="sampling seed") != BASE_SEED + global_index:
        raise ValueError(f"{sample_id}: sampling seed mismatch")
    if exact_int(row.get("num_samples"), label="num_samples") != 100:
        raise ValueError(f"{sample_id}: candidate count mismatch")
    if exact_int(row.get("prior_pool_size"), label="prior_pool_size") != 100:
        raise ValueError(f"{sample_id}: prior pool size mismatch")
    prior_hash = require_sha256(row.get("prior_pool_sha256"), label="prior pool")
    require_sha256(row.get("candidate_ensemble_sha256"), label="candidate ensemble")
    confidence_rmsd = finite_float(row.get("confidence_rmsd"), label="confidence RMSD")
    oracle_rmsd = finite_float(row.get("oracle_rmsd"), label="oracle RMSD")
    if confidence_rmsd < 0.0 or oracle_rmsd < 0.0:
        raise ValueError(f"{sample_id}: RMSD cannot be negative")
    confidence_index = exact_int(row.get("confidence_index"), label="confidence index")
    oracle_index = exact_int(row.get("oracle_index"), label="oracle index")
    if not 0 <= confidence_index < 100 or not 0 <= oracle_index < 100:
        raise ValueError(f"{sample_id}: selector index is out of range")
    try:
        scores = json.loads(row["confidence_candidate_scores_json"])
    except json.JSONDecodeError as exc:
        raise ValueError(f"{sample_id}: invalid confidence score ledger") from exc
    if not isinstance(scores, list) or len(scores) != 100:
        raise ValueError(f"{sample_id}: confidence score ledger must have 100 entries")
    predicted = [
        finite_float(score.get("confidence_rmsd"), label="candidate predicted RMSD")
        if isinstance(score, dict)
        else (_ for _ in ()).throw(ValueError("candidate score must be an object"))
        for score in scores
    ]
    expected_index = min(range(100), key=predicted.__getitem__)
    if confidence_index != expected_index:
        raise ValueError(f"{sample_id}: confidence selector is not argmin predicted RMSD")
    if (
        finite_float(row.get("confidence_pred_rmsd"), label="selected predicted RMSD")
        != predicted[confidence_index]
    ):
        raise ValueError(f"{sample_id}: selected predicted RMSD ledger mismatch")
    finite_float(row.get("confidence_pred_success"), label="selected predicted success")
    if not parse_bool(row.get("full_heavy_atom_bijection"), label="atom bijection"):
        raise ValueError(f"{sample_id}: full ligand atom mapping is required")
    protein = _require_path_hash(
        row.get("protein"), row.get("protein_sha256"), label=f"{sample_id} receptor"
    )
    ligand = _require_path_hash(
        row.get("ligand_ref"),
        row.get("ligand_reference_sha256"),
        label=f"{sample_id} reference ligand",
    )
    for label, observed_path, row_hash_key in (
        ("receptor", protein, "protein_sha256"),
        ("ligand", ligand, "ligand_reference_sha256"),
    ):
        frozen = raw_gate_asset.get(label)
        if not isinstance(frozen, dict):
            raise ValueError(f"{sample_id}: raw gate lacks {label} binding")
        frozen_path = frozen.get("path")
        if (
            not isinstance(frozen_path, str)
            or Path(frozen_path).resolve() != observed_path
            or frozen.get("sha256") != row.get(row_hash_key)
        ):
            raise ValueError(f"{sample_id}: sampling {label} differs from the raw gate")
    _require_path_hash(
        row.get("processed_meta"),
        row.get("processed_meta_sha256"),
        label=f"{sample_id} processed meta",
    )
    try:
        saved_hashes = json.loads(row["saved_pose_sha256_json"])
    except json.JSONDecodeError as exc:
        raise ValueError(f"{sample_id}: invalid saved-pose hash ledger") from exc
    expected_pose_hash = (
        saved_hashes.get(PRIMARY_SELECTOR) if isinstance(saved_hashes, dict) else None
    )
    expected_pose_hash = require_sha256(expected_pose_hash, label="confidence pose")
    pose_path = pose_dir / PRIMARY_SELECTOR / f"{sample_id}.sdf"
    if not pose_path.is_file() or file_sha256(pose_path) != expected_pose_hash:
        raise ValueError(f"{sample_id}: confidence-selected pose hash mismatch")
    try:
        trace = json.loads(row["guidance_direct_step_trace_json"])
    except json.JSONDecodeError as exc:
        raise ValueError(f"{sample_id}: invalid direct-step trace") from exc
    if eta == 0.0:
        if row.get("guidance_mode") != "none" or trace != []:
            raise ValueError(f"{sample_id}: eta-zero must be a newly generated unguided baseline")
        if row.get("guidance_parameter_sha256") not in {None, ""}:
            raise ValueError(f"{sample_id}: eta-zero unexpectedly reports active guidance")
    else:
        if row.get("guidance_mode") != "unified_normalized_drift":
            raise ValueError(f"{sample_id}: normalized direct guidance was not active")
        if row.get("guidance_parameter_sha256") != EXPECTED_GUIDANCE_PARAMETER_SHA256:
            raise ValueError(f"{sample_id}: guidance parameter identity mismatch")
        if not isinstance(trace, list) or not trace:
            raise ValueError(f"{sample_id}: guided arm lacks a direct-step trace")
        if any(not _json_numbers_are_finite(step) for step in trace):
            raise ValueError(f"{sample_id}: direct-step trace contains non-finite numbers")
        nonfinite = sum(
            exact_int(row.get(key, "0"), label=key)
            for key in (
                "guidance_nonfinite_base_poses",
                "guidance_nonfinite_trials",
                "guidance_direct_nonfinite_poses",
            )
        )
        if nonfinite != 0:
            raise ValueError(f"{sample_id}: guided sampling recorded non-finite poses")
    return {
        "id": sample_id,
        "eta": eta,
        "sampling_seed": BASE_SEED + global_index,
        "prior_pool_sha256": prior_hash,
        "confidence_rmsd": confidence_rmsd,
        "oracle_rmsd": oracle_rmsd,
        "protein": str(protein),
        "ligand_ref": str(ligand),
        "pose": str(pose_path.resolve()),
        "pose_sha256": expected_pose_hash,
    }


def _json_numbers_are_finite(value: Any) -> bool:
    if isinstance(value, bool | str) or value is None:
        return True
    if isinstance(value, int):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(_json_numbers_are_finite(item) for item in value)
    if isinstance(value, dict):
        return all(_json_numbers_are_finite(item) for item in value.values())
    return False


def build_audit(
    *,
    sampling_root: Path,
    split_file: Path,
    raw_manifest: Path,
    raw_gate: Path,
    raw_gate_sidecar: Path,
    raw_root: Path,
    mode: str,
) -> dict[str, Any]:
    sampling_root = sampling_root.resolve()
    split_ids = load_split_ids(split_file)
    verify_raw_manifest(raw_manifest, split_ids=split_ids)
    raw_gate_payload = validate_raw_gate(
        raw_gate,
        raw_gate_sidecar,
        raw_manifest=raw_manifest,
        raw_root=raw_root,
        split_ids=split_ids,
    )
    raw_assets = raw_gate_payload.get("assets")
    if not isinstance(raw_assets, list) or len(raw_assets) != EXPECTED_COUNT:
        raise ValueError("raw-gate sample asset ledger has the wrong length")
    if not all(
        isinstance(asset, dict) and isinstance(asset.get("sample_id"), str) for asset in raw_assets
    ):
        raise ValueError("raw-gate sample asset ledger is malformed")
    raw_gate_assets = {asset["sample_id"]: asset for asset in raw_assets}
    if len(raw_gate_assets) != len(raw_assets) or sorted(raw_gate_assets) != split_ids:
        raise ValueError("raw-gate sample asset map is incomplete")
    selected_ids = expected_ids(mode, split_ids)
    num_shards = expected_num_shards(mode)
    mode_root = sampling_root / mode
    expected_eta_dirs = {mode_root / ETA_TAGS[eta] for eta in ETA_VALUES}
    actual_eta_dirs = (
        {path for path in mode_root.iterdir() if path.is_dir()} if mode_root.is_dir() else set()
    )
    if actual_eta_dirs != expected_eta_dirs:
        raise ValueError(
            f"sampling eta inventory mismatch; missing={sorted(expected_eta_dirs - actual_eta_dirs)}, "
            f"extra={sorted(actual_eta_dirs - expected_eta_dirs)}"
        )

    records_by_eta: dict[float, dict[str, dict[str, Any]]] = {}
    cells: list[dict[str, Any]] = []
    global_artifacts: list[dict[str, Any]] = []
    cuda_runtimes: list[dict[str, Any]] = []
    recovered_attempts: list[dict[str, Any]] = []
    implementation_identity: dict[str, Any] | None = None
    for eta in ETA_VALUES:
        expected_shard_dirs = {
            sampling_shard_dir(sampling_root, mode, eta, shard, num_shards)
            for shard in range(num_shards)
        }
        arm_root = sampling_root / mode / ETA_TAGS[eta]
        incomplete_root = arm_root / ".incomplete"
        eta_recovered = inspect_incomplete_attempts(incomplete_root, num_shards=num_shards)
        recovered_attempts.extend(
            {"eta": eta, "eta_tag": ETA_TAGS[eta], **record} for record in eta_recovered
        )
        actual_shard_dirs = {
            path for path in arm_root.iterdir() if path.is_dir() and path.name != ".incomplete"
        }
        if actual_shard_dirs != expected_shard_dirs:
            raise ValueError(f"{ETA_TAGS[eta]}: exact shard inventory mismatch")
        arm_records: dict[str, dict[str, Any]] = {}
        shard_records: list[dict[str, Any]] = []
        for shard_index in range(num_shards):
            shard_dir = sampling_shard_dir(sampling_root, mode, eta, shard_index, num_shards)
            expected_root_entries = {
                shard_dir / "results.csv",
                shard_dir / "summary.json",
                shard_dir / "poses",
            }
            if set(shard_dir.iterdir()) != expected_root_entries:
                raise ValueError(f"{shard_dir}: unexpected or missing shard-root artifacts")
            summary_path = shard_dir / "summary.json"
            csv_path = shard_dir / "results.csv"
            summary = _validate_summary(
                summary_path,
                mode=mode,
                eta=eta,
                shard_index=shard_index,
                num_shards=num_shards,
                split_ids=split_ids,
                selected_ids=selected_ids,
                raw_root=raw_root,
            )
            cuda_runtimes.append(summary["runtime"])
            current_implementation = summary["fixed_identities"]["guidance_implementation"]
            if implementation_identity is None:
                implementation_identity = current_implementation
            elif current_implementation != implementation_identity:
                raise ValueError("guidance implementation identity differs across sampling shards")
            fields, rows = load_csv(csv_path)
            assigned_ids = selected_ids[shard_index::num_shards]
            if [row.get("id") for row in rows] != assigned_ids:
                raise ValueError(f"{csv_path}: exact assigned row inventory/order mismatch")
            for sample_id, row in zip(assigned_ids, rows, strict=True):
                global_index = split_ids.index(sample_id) + 1
                record = _validate_row(
                    row,
                    fields,
                    sample_id=sample_id,
                    global_index=global_index,
                    eta=eta,
                    pose_dir=shard_dir / "poses",
                    raw_gate_asset=raw_gate_assets[sample_id],
                )
                if sample_id in arm_records:
                    raise ValueError(f"{ETA_TAGS[eta]}: duplicate sample {sample_id}")
                arm_records[sample_id] = record
            shard_record = {
                "shard_index": shard_index,
                "assigned_count": len(assigned_ids),
                "assigned_ids_sha256": ids_sha256(assigned_ids),
                "csv": str(csv_path.resolve()),
                "csv_sha256": file_sha256(csv_path),
                "summary": str(summary_path.resolve()),
                "summary_sha256": file_sha256(summary_path),
            }
            shard_records.append(shard_record)
            global_artifacts.append({"eta": eta, **shard_record})
        if sorted(arm_records) != selected_ids:
            raise ValueError(f"{ETA_TAGS[eta]}: complete selected ID coverage mismatch")
        records_by_eta[eta] = arm_records
        cells.append(
            {
                "eta": eta,
                "eta_tag": ETA_TAGS[eta],
                "row_count": len(arm_records),
                "ids_sha256": ids_sha256(arm_records),
                "shards": shard_records,
            }
        )

    pairing_ledger: list[dict[str, Any]] = []
    for sample_id in selected_ids:
        records = [records_by_eta[eta][sample_id] for eta in ETA_VALUES]
        seeds = {record["sampling_seed"] for record in records}
        prior_hashes = {record["prior_pool_sha256"] for record in records}
        proteins = {record["protein"] for record in records}
        ligands = {record["ligand_ref"] for record in records}
        if len(seeds) != 1 or len(prior_hashes) != 1:
            raise ValueError(f"{sample_id}: eta arms do not share the exact seed/prior")
        if len(proteins) != 1 or len(ligands) != 1:
            raise ValueError(f"{sample_id}: eta arms do not share exact raw inputs")
        pairing_ledger.append(
            {
                "id": sample_id,
                "sampling_seed": records[0]["sampling_seed"],
                "prior_pool_sha256": records[0]["prior_pool_sha256"],
            }
        )
    ledger_payload = {
        "mode": mode,
        "selected_ids": selected_ids,
        "pairing": pairing_ledger,
        "artifacts": global_artifacts,
        "recovered_attempts": recovered_attempts,
    }
    return {
        "schema_version": AUDIT_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "status": "passed",
        "mode": mode,
        "claim_scope": "guidance_development_not_untouched_confirmation",
        "primary_selector": PRIMARY_SELECTOR,
        "automatic_eta_selection": False,
        "expected_denominator": EXPECTED_COUNT,
        "selected_count": len(selected_ids),
        "selected_ids_sha256": ids_sha256(selected_ids),
        "num_shards": num_shards,
        "eta_values": list(ETA_VALUES),
        "sampling_root": str(sampling_root),
        "raw_root": str(raw_root.resolve()),
        "split_sha256": EXPECTED_SPLIT_SHA256,
        "raw_manifest": str(raw_manifest.resolve()),
        "raw_manifest_sha256": file_sha256(raw_manifest),
        "raw_gate": str(raw_gate.resolve()),
        "raw_gate_sha256": file_sha256(raw_gate),
        "raw_gate_sidecar": str(raw_gate_sidecar.resolve()),
        "raw_gate_sidecar_sha256": file_sha256(raw_gate_sidecar),
        "fixed_hashes": {
            "docking_checkpoint_sha256": EXPECTED_DOCKING_SHA256,
            "confidence_checkpoint_sha256": EXPECTED_CONFIDENCE_SHA256,
            "config_sha256": EXPECTED_CONFIG_SHA256,
            "guidance_parameter_sha256": EXPECTED_GUIDANCE_PARAMETER_SHA256,
            "guidance_implementation_sha256": EXPECTED_GUIDANCE_IMPLEMENTATION_SHA256,
        },
        "cuda_runtime": {
            "slurm_partitions": sorted(
                {str(runtime["slurm_partition"]) for runtime in cuda_runtimes}
            ),
            "gpu_names": sorted({str(runtime["gpu"]) for runtime in cuda_runtimes}),
            "gpu_total_memory_bytes": sorted(
                {int(runtime["gpu_total_memory_bytes"]) for runtime in cuda_runtimes}
            ),
            "minimum_required_total_memory_bytes": MIN_GPU_TOTAL_MEMORY_BYTES,
        },
        "cells": cells,
        "paired_complex_count": len(pairing_ledger),
        "recovered_attempt_count": len(recovered_attempts),
        "recovered_attempts": recovered_attempts,
        "recovered_attempts_sha256": canonical_json_sha256(recovered_attempts),
        "global_sampling_ledger_sha256": canonical_json_sha256(ledger_payload),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sampling-root", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, default=Path("data/splits/plinder.json"))
    parser.add_argument("--raw-manifest", type=Path, required=True)
    parser.add_argument("--raw-gate", type=Path, required=True)
    parser.add_argument("--raw-gate-sidecar", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    audit = build_audit(
        sampling_root=args.sampling_root,
        split_file=args.split_file,
        raw_manifest=args.raw_manifest,
        raw_gate=args.raw_gate,
        raw_gate_sidecar=args.raw_gate_sidecar,
        raw_root=args.raw_root,
        mode=args.mode,
    )
    write_json_noreplace(args.output, audit)
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
