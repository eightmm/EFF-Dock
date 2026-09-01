#!/usr/bin/env python3
"""Run one frozen EFF-Dock guidance arm on the PLINDER validation split.

This driver deliberately exposes only data locations, run identity, eta, and
mechanical subsetting/sharding.  Every scientifically meaningful sampling and
selection setting is fixed by ``EFFDOCK-PLINDER-GUIDANCE-DEV-V1``.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import io
import json
import math
import os
import re
import tempfile
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Sequence

import torch

from effdock.confidence.runtime import load_pose_confidence_model
from effdock.guidance.parameterization import guidance_parameter_identity
from effdock.guidance.provenance import guidance_implementation_identity
from effdock.inference.docking import load_model
from effdock.workflows.evaluate import (
    ComplexInput,
    evaluate_one,
    file_sha256,
    serialize_evaluation_failure,
    sorted_id_sha256,
    summarize_rows,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_ID = "EFFDOCK-PLINDER-GUIDANCE-DEV-V1"
SCHEMA_VERSION = "effdock.plinder_guidance_validation_shard.v1"
PLINDER_RELEASE = "2024-06/v2"

SPLIT_FILE = PROJECT_ROOT / "data/splits/plinder.json"
PROCESSED_ROOT = PROJECT_ROOT / "data/plinder_processed"
CONFIG = PROJECT_ROOT / "configs/train.yaml"
DOCKING_CHECKPOINT = PROJECT_ROOT / "weights/effdock_geometry_ft_100k_best.pt"
CONFIDENCE_CHECKPOINT = PROJECT_ROOT / "weights/effdock_confidence_extmatch_n80_s25_step42500.pt"
PROTOCOL_DOCUMENT = PROJECT_ROOT / "docs/PLINDER_GUIDANCE_VALIDATION_PROTOCOL.md"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs/benchmarks/plinder_guidance_validation_runs"

EXPECTED_SPLIT_SHA256 = "3ac570bf08bced053f1ce040b57efca27c3be616f29a82cd66ef887c08860e6b"
EXPECTED_VAL_COUNT = 1076
EXPECTED_DOCKING_SHA256 = "6932fb3ba6ebac770f714453529656a44b8f33cf15119d23c9e675d2d60b36db"
EXPECTED_CONFIDENCE_SHA256 = "e31fde6f351284205c78f7a1510002779c43312e94d9f82003d47a14d72bc78f"
EXPECTED_CONFIG_SHA256 = "39aa62e4a48ed6f3aa4ff59345fb43a81220e2baba22edfd5beb0c4981b307ec"
EXPECTED_GUIDANCE_PARAMETER_SHA256 = (
    "6621d17c41aeb6c9685075209155850018c5eb9882489ae209c7c30b8070e89f"
)
EXPECTED_GUIDANCE_IMPLEMENTATION_SHA256 = (
    "04271077bfc9fe255e370cb5b985efe4df7242ba700abc6f81c50ec12aff6b25"
)

BASE_SEED = 42
ETA_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0)
ETA_DECIMALS = {Decimal(str(value)): value for value in ETA_VALUES}
ETA_TAGS = {0.0: "eta0000", 0.5: "eta0500", 1.0: "eta1000", 1.5: "eta1500", 2.0: "eta2000"}
RUN_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")


@dataclass(frozen=True)
class Assignment:
    sample_key: str
    global_index: int
    sampling_seed: int


@dataclass(frozen=True)
class AssignmentPlan:
    full_keys: tuple[str, ...]
    selected_keys: tuple[str, ...]
    assigned: tuple[Assignment, ...]


@dataclass(frozen=True)
class InputPaths:
    sample_key: str
    system_id: str
    ligand_chain: str
    receptor: Path
    ligand: Path
    meta: Path


@dataclass(frozen=True)
class PreparedInput:
    assignment: Assignment
    paths: InputPaths
    item: ComplexInput
    source_identity: dict[str, Any]


@dataclass(frozen=True)
class ShardAttempt:
    attempt_dir: Path
    final_dir: Path
    publish_lock: Path


def validate_eta(value: str | float | Decimal) -> float:
    """Return the canonical frozen eta value or reject it exactly."""
    if isinstance(value, bool):
        raise ValueError(f"eta must be one of {ETA_VALUES}, got {value!r}")
    try:
        decimal = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"eta must be one of {ETA_VALUES}, got {value!r}") from exc
    if not decimal.is_finite() or decimal not in ETA_DECIMALS:
        raise ValueError(f"eta must be one of {ETA_VALUES}, got {value!r}")
    return ETA_DECIMALS[decimal]


def _eta_arg(value: str) -> float:
    try:
        return validate_eta(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return parsed


def _safe_component(value: str, label: str) -> str:
    if not value or value in {".", ".."} or Path(value).name != value or "\\" in value:
        raise ValueError(f"unsafe {label}: {value!r}")
    return value


def parse_sample_key(sample_key: str) -> tuple[str, str]:
    """Split ``<system_id>__<ligand_chain>`` at its final delimiter."""
    if not isinstance(sample_key, str) or sample_key.count("__") < 1:
        raise ValueError(f"invalid PLINDER sample key: {sample_key!r}")
    system_id, ligand_chain = sample_key.rsplit("__", 1)
    return _safe_component(system_id, "system_id"), _safe_component(ligand_chain, "ligand_chain")


def load_frozen_val_keys(
    split_file: Path,
    *,
    expected_sha256: str | None = None,
    expected_count: int | None = None,
) -> list[str]:
    expected_sha256 = EXPECTED_SPLIT_SHA256 if expected_sha256 is None else expected_sha256
    expected_count = EXPECTED_VAL_COUNT if expected_count is None else expected_count
    actual_sha256 = file_sha256(split_file)
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"PLINDER split SHA-256 mismatch: expected {expected_sha256}, got {actual_sha256}"
        )
    payload = json.loads(split_file.read_text())
    raw_keys = payload.get("val") if isinstance(payload, dict) else None
    if not isinstance(raw_keys, list) or not all(isinstance(key, str) and key for key in raw_keys):
        raise ValueError("PLINDER split must contain a non-empty string list at key 'val'")
    if len(raw_keys) != expected_count or len(set(raw_keys)) != expected_count:
        raise ValueError(
            "PLINDER validation inventory mismatch: "
            f"expected {expected_count} unique keys, got {len(raw_keys)} rows/"
            f"{len(set(raw_keys))} unique"
        )
    for key in raw_keys:
        parse_sample_key(key)
    return sorted(raw_keys)


def plan_assignments(
    full_keys: Sequence[str],
    *,
    num_shards: int,
    shard_index: int,
    only_ids: Sequence[str] = (),
    smoke_count: int | None = None,
) -> AssignmentPlan:
    """Assign global seeds before any subset or shard operation."""
    ordered = tuple(sorted(full_keys))
    if len(ordered) != len(set(ordered)):
        raise ValueError("full validation keys must be unique")
    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError("shard_index must satisfy 0 <= shard_index < num_shards")
    if only_ids and smoke_count is not None:
        raise ValueError("--only-id and --smoke are mutually exclusive")
    if smoke_count is not None and smoke_count < 1:
        raise ValueError("smoke_count must be at least 1")

    seed_by_key = {
        key: BASE_SEED + global_index for global_index, key in enumerate(ordered, start=1)
    }
    index_by_key = {key: index for index, key in enumerate(ordered, start=1)}
    if only_ids:
        requested = tuple(only_ids)
        if len(requested) != len(set(requested)):
            raise ValueError("--only-id values must be unique")
        missing = sorted(set(requested) - set(ordered))
        if missing:
            raise ValueError(f"requested PLINDER validation IDs not found: {missing}")
        requested_set = set(requested)
        selected = tuple(key for key in ordered if key in requested_set)
    elif smoke_count is not None:
        selected = ordered[:smoke_count]
    else:
        selected = ordered

    assigned_keys = selected[shard_index::num_shards]
    if not assigned_keys:
        raise ValueError("the requested subset/shard has no assigned PLINDER validation IDs")
    assigned = tuple(
        Assignment(
            sample_key=key,
            global_index=index_by_key[key],
            sampling_seed=seed_by_key[key],
        )
        for key in assigned_keys
    )
    return AssignmentPlan(ordered, selected, assigned)


def input_paths_for_sample(
    sample_key: str,
    *,
    raw_root: Path,
    processed_root: Path,
) -> InputPaths:
    system_id, ligand_chain = parse_sample_key(sample_key)
    system_root = raw_root / "systems" / system_id
    return InputPaths(
        sample_key=sample_key,
        system_id=system_id,
        ligand_chain=ligand_chain,
        receptor=system_root / "receptor.pdb",
        ligand=system_root / "ligand_files" / f"{ligand_chain}.sdf",
        meta=processed_root / sample_key / "meta.pt",
    )


def _asset_identity(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _describe_paths(paths: InputPaths) -> dict[str, Any]:
    def describe(path: Path) -> dict[str, Any]:
        result: dict[str, Any] = {"path": str(path), "exists": path.is_file()}
        if path.is_file():
            result.update(sha256=file_sha256(path), size_bytes=path.stat().st_size)
        return result

    return {
        "sample_key": paths.sample_key,
        "system_id": paths.system_id,
        "ligand_chain": paths.ligand_chain,
        "receptor": describe(paths.receptor),
        "ligand": describe(paths.ligand),
        "processed_meta": describe(paths.meta),
    }


def prepare_input(
    assignment: Assignment,
    *,
    raw_root: Path,
    processed_root: Path,
) -> PreparedInput:
    paths = input_paths_for_sample(
        assignment.sample_key, raw_root=raw_root, processed_root=processed_root
    )
    missing = [
        str(path) for path in (paths.receptor, paths.ligand, paths.meta) if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(f"missing frozen PLINDER input files: {missing}")
    meta = torch.load(paths.meta, map_location="cpu", weights_only=True)
    if not isinstance(meta, dict):
        raise TypeError(f"processed meta.pt is not a mapping: {paths.meta}")
    expected_meta = {
        "pdb_id": paths.sample_key,
        "plinder_system_id": paths.system_id,
        "plinder_ligand_chain": paths.ligand_chain,
    }
    for field, expected in expected_meta.items():
        if meta.get(field) != expected:
            raise ValueError(
                f"{paths.sample_key}: meta.pt {field} mismatch: "
                f"expected {expected!r}, got {meta.get(field)!r}"
            )
    if "pocket_center" not in meta:
        raise ValueError(f"{paths.sample_key}: meta.pt is missing pocket_center")
    center = torch.as_tensor(meta["pocket_center"], dtype=torch.float32).detach().cpu()
    if center.shape != (3,) or not bool(torch.isfinite(center).all()):
        raise ValueError(f"{paths.sample_key}: invalid pocket_center {center!r}")

    item = ComplexInput(
        complex_id=paths.sample_key,
        protein=paths.receptor,
        ligand_ref=paths.ligand,
        ligand_format="sdf",
        smiles=None,
        pocket_center=tuple(float(value) for value in center.tolist()),
    )
    source_identity = {
        "sample_key": paths.sample_key,
        "system_id": paths.system_id,
        "ligand_chain": paths.ligand_chain,
        "global_index": assignment.global_index,
        "sampling_seed": assignment.sampling_seed,
        "pocket_center": center.tolist(),
        "receptor": _asset_identity(paths.receptor),
        "ligand": _asset_identity(paths.ligand),
        "processed_meta": _asset_identity(paths.meta),
    }
    return PreparedInput(assignment, paths, item, source_identity)


def fixed_settings(eta: float) -> dict[str, Any]:
    return {
        "eta": validate_eta(eta),
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
        "saved_primary_selector": "confidence",
    }


def _require_identity(path: Path, expected_sha256: str, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing fixed {label}: {path}")
    identity = _asset_identity(path)
    if identity["sha256"] != expected_sha256:
        raise RuntimeError(
            f"{label} SHA-256 mismatch: expected {expected_sha256}, got {identity['sha256']}"
        )
    return identity


def collect_fixed_identities() -> dict[str, Any]:
    guidance_parameters = guidance_parameter_identity()
    if guidance_parameters["sha256"] != EXPECTED_GUIDANCE_PARAMETER_SHA256:
        raise RuntimeError(
            "guidance parameter SHA-256 mismatch: "
            f"expected {EXPECTED_GUIDANCE_PARAMETER_SHA256}, "
            f"got {guidance_parameters['sha256']}"
        )
    guidance_implementation = guidance_implementation_identity()
    if guidance_implementation["sha256"] != EXPECTED_GUIDANCE_IMPLEMENTATION_SHA256:
        raise RuntimeError(
            "guidance implementation SHA-256 mismatch: "
            f"expected {EXPECTED_GUIDANCE_IMPLEMENTATION_SHA256}, "
            f"got {guidance_implementation['sha256']}"
        )
    return {
        "docking_checkpoint": _require_identity(
            DOCKING_CHECKPOINT, EXPECTED_DOCKING_SHA256, "docking checkpoint"
        ),
        "confidence_checkpoint": _require_identity(
            CONFIDENCE_CHECKPOINT, EXPECTED_CONFIDENCE_SHA256, "confidence checkpoint"
        ),
        "config": _require_identity(CONFIG, EXPECTED_CONFIG_SHA256, "config"),
        "guidance_parameters": guidance_parameters,
        "guidance_implementation": guidance_implementation,
        "driver": _asset_identity(Path(__file__).resolve()),
        "protocol_document": (
            _asset_identity(PROTOCOL_DOCUMENT) if PROTOCOL_DOCUMENT.is_file() else None
        ),
    }


def resolve_runtime_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("frozen PLINDER guidance sampling requires a visible CUDA GPU")
    return torch.device("cuda")


def evaluate_prepared(
    prepared: PreparedInput,
    *,
    eta: float,
    model: torch.nn.Module,
    confidence_model: torch.nn.Module,
    cfg: dict[str, Any],
    device: torch.device,
    pose_dir: Path,
) -> dict[str, Any]:
    data_cfg = cfg.get("data", {})
    row = evaluate_one(
        model,
        prepared.item,
        dataset="plinder_val",
        confidence_model=confidence_model,
        device=device,
        num_samples=100,
        num_steps=10,
        sigma=0.5,
        sigma_list=[],
        sigma_counts=[],
        center_jitter_sigma=0.0,
        pocket_cutoff=10.0,
        pose_objective=data_cfg.get("pose_objective", "linear_fm"),
        score_rot_sigma_max=float(data_cfg.get("score_rot_sigma_max", torch.pi)),
        score_alpha_min=float(data_cfg.get("score_alpha_min", 0.0)),
        time_schedule="late",
        schedule_power=3.0,
        vina_guidance_scale=0.0,
        vina_guidance_start_t=0.5,
        vina_guidance_ramp_power=1.0,
        vina_guidance_max_force=10.0,
        vina_guidance_max_velocity=5.0,
        vina_guidance_max_angular_velocity=5.0,
        vina_guidance_protein_shell=18.0,
        vina_guidance_w_strain=1.0,
        unified_guidance_scale=eta,
        unified_guidance_start_t=0.5,
        unified_guidance_ramp_power=1.0,
        unified_guidance_max_force=20.0,
        unified_guidance_max_velocity=5.0,
        unified_guidance_max_angular_velocity=5.0,
        unified_guidance_max_atom_displacement=0.25,
        unified_guidance_max_backtracks=8,
        unified_guidance_protein_shell=18.0,
        unified_guidance_receptor_policy="geometry_only",
        unified_guidance_mode="normalized_drift",
        prior_pool_size=100,
        seed=prepared.assignment.sampling_seed,
        refine="none",
        pose_dir=pose_dir,
        trajectory_dir=None,
        require_full_ligand_atom_mapping=True,
        selector_profile="confidence_cluster_free",
    )
    if row.get("id") != prepared.assignment.sample_key:
        raise RuntimeError("evaluate_one returned the wrong sample ID")
    if int(row.get("sampling_seed", -1)) != prepared.assignment.sampling_seed:
        raise RuntimeError("evaluate_one changed the frozen sampling seed")
    if int(row.get("prior_pool_size", -1)) != 100 or not row.get("prior_pool_sha256"):
        raise RuntimeError("evaluate_one did not record the frozen shared prior pool")
    for field in ("confidence_index", "confidence_rmsd", "confidence_pred_rmsd"):
        if field not in row:
            raise RuntimeError(f"evaluate_one omitted required confidence field {field}")
    saved_hashes = json.loads(str(row.get("saved_pose_sha256_json", "{}")))
    pose_path = pose_dir / "confidence" / f"{prepared.assignment.sample_key}.sdf"
    if not pose_path.is_file() or saved_hashes.get("confidence") != file_sha256(pose_path):
        raise RuntimeError("primary confidence-selected pose was not saved with its exact hash")
    row.update(
        plinder_system_id=prepared.paths.system_id,
        plinder_ligand_chain=prepared.paths.ligand_chain,
        plinder_global_index=prepared.assignment.global_index,
        processed_meta=str(prepared.paths.meta),
        processed_meta_sha256=prepared.source_identity["processed_meta"]["sha256"],
    )
    return row


def _validate_run_id(run_id: str) -> str:
    if not RUN_ID_RE.fullmatch(run_id):
        raise ValueError(
            "run ID must start with an alphanumeric character and contain only "
            "letters, digits, '.', '_' or '-'"
        )
    return run_id


def reserve_shard_directory(
    output_root: Path,
    *,
    run_id: str,
    eta: float,
    num_shards: int,
    shard_index: int,
) -> ShardAttempt:
    run_id = _validate_run_id(run_id)
    arm_dir = output_root / run_id / ETA_TAGS[validate_eta(eta)]
    arm_dir.mkdir(parents=True, exist_ok=True)
    shard_name = f"shard-{shard_index:03d}-of-{num_shards:03d}"
    final_dir = arm_dir / shard_name
    if final_dir.exists():
        raise FileExistsError(f"refusing to rerun or overwrite shard output: {final_dir}")
    incomplete_root = arm_dir / ".incomplete"
    incomplete_root.mkdir(exist_ok=True)
    attempt_dir = Path(tempfile.mkdtemp(prefix=f"{shard_name}.attempt-", dir=incomplete_root))
    return ShardAttempt(
        attempt_dir=attempt_dir,
        final_dir=final_dir,
        publish_lock=incomplete_root / f".{shard_name}.publish.lock",
    )


def _atomic_write_noreplace(path: Path, data: bytes) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite artifact: {path}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _csv_bytes(rows: list[dict[str, Any]]) -> bytes:
    core_fields = [
        "id",
        "plinder_system_id",
        "plinder_ligand_chain",
        "plinder_global_index",
        "sampling_seed",
    ]
    field_set = {key for row in rows for key in row}
    fieldnames = core_fields + sorted(field_set - set(core_fields))
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, extrasaction="raise")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, str | int | bool):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return "NaN" if math.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        return _json_safe(value.item() if value.ndim == 0 else value.tolist())
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple | set):
        return [_json_safe(item) for item in value]
    return str(value)


def _runtime_snapshot(
    device: torch.device | None,
    *,
    started_at: str,
    elapsed_seconds: float,
) -> dict[str, Any]:
    runtime: dict[str, Any] = {
        "started_at_utc": started_at,
        "finished_at_utc": datetime.now(UTC).isoformat(),
        "elapsed_seconds": elapsed_seconds,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "device": str(device) if device is not None else None,
        "gpu": None,
        "cuda_max_memory_allocated_bytes": None,
        "cuda_max_memory_reserved_bytes": None,
    }
    if device is not None and device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        runtime.update(
            gpu=torch.cuda.get_device_name(device),
            gpu_total_memory_bytes=int(properties.total_memory),
            cuda_max_memory_allocated_bytes=int(torch.cuda.max_memory_allocated(device)),
            cuda_max_memory_reserved_bytes=int(torch.cuda.max_memory_reserved(device)),
        )
    return runtime


def _failure(sample_id: str, exc: Exception, stage: str) -> dict[str, Any]:
    result = serialize_evaluation_failure(sample_id, exc)
    result["stage"] = stage
    return result


def _publish_artifacts(
    storage_dir: Path,
    visible_dir: Path,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    csv_data = _csv_bytes(rows)
    summary["artifacts"] = {
        "csv": str(visible_dir / "results.csv"),
        "csv_sha256": hashlib.sha256(csv_data).hexdigest(),
        "summary": str(visible_dir / "summary.json"),
        "primary_pose_dir": str(visible_dir / "poses/confidence"),
    }
    summary_data = (
        json.dumps(_json_safe(summary), indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    _atomic_write_noreplace(storage_dir / "results.csv", csv_data)
    _atomic_write_noreplace(storage_dir / "summary.json", summary_data)


def publish_complete_attempt(
    attempt: ShardAttempt,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> Path:
    """Publish one complete attempt by a locked, atomic directory rename."""
    with attempt.publish_lock.open("a+b") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        if attempt.final_dir.exists():
            summary["status"] = "publish_conflict"
            summary["failures"].append(
                {
                    "id": "__publish__",
                    "stage": "atomic_publish",
                    "error_type": "FinalShardAlreadyExists",
                    "message": f"refusing to overwrite completed shard {attempt.final_dir}",
                }
            )
            summary["inventory"]["failure_count"] = len(summary["failures"])
            _publish_artifacts(attempt.attempt_dir, attempt.attempt_dir, rows, summary)
            raise FileExistsError(
                f"refusing duplicate publish for completed shard: {attempt.final_dir}"
            )
        _publish_artifacts(attempt.attempt_dir, attempt.final_dir, rows, summary)
        try:
            os.rename(attempt.attempt_dir, attempt.final_dir)
        except OSError as exc:
            raise RuntimeError(
                f"atomic shard publish failed; attempt retained at {attempt.attempt_dir}"
            ) from exc
    return attempt.final_dir


def execute(args: argparse.Namespace) -> dict[str, Any]:
    eta = validate_eta(args.eta)
    raw_root = args.raw_root.resolve()
    processed_root = args.processed_root.resolve()
    output_root = args.output_root.resolve()
    full_keys = load_frozen_val_keys(SPLIT_FILE)
    plan = plan_assignments(
        full_keys,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
        only_ids=args.only_id,
        smoke_count=args.smoke,
    )
    attempt = reserve_shard_directory(
        output_root,
        run_id=args.run_id,
        eta=eta,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
    )

    started_at = datetime.now(UTC).isoformat()
    started_clock = time.monotonic()
    failures: list[dict[str, Any]] = []
    prepared_inputs: list[PreparedInput] = []
    source_records: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    attempted_ids: list[str] = []
    identities: dict[str, Any] | None = None
    docking_step: Any = None
    confidence_step: Any = None
    device: torch.device | None = None

    processed_missing = [
        key for key in plan.full_keys if not (processed_root / key / "meta.pt").is_file()
    ]
    if processed_missing:
        failures.append(
            {
                "id": "__cohort__",
                "stage": "processed_coverage",
                "error_type": "MissingProcessedValidationData",
                "message": f"missing meta.pt for {len(processed_missing)} validation samples",
            }
        )

    try:
        identities = collect_fixed_identities()
    except Exception as exc:
        failures.append(_failure("__run__", exc, "fixed_identity"))

    if not failures:
        for assignment in plan.assigned:
            paths = input_paths_for_sample(
                assignment.sample_key, raw_root=raw_root, processed_root=processed_root
            )
            try:
                prepared = prepare_input(
                    assignment, raw_root=raw_root, processed_root=processed_root
                )
                prepared_inputs.append(prepared)
                source_records.append(prepared.source_identity)
            except Exception as exc:
                source_records.append(_describe_paths(paths))
                failures.append(_failure(assignment.sample_key, exc, "input_preflight"))

    model: torch.nn.Module | None = None
    confidence_model: torch.nn.Module | None = None
    cfg: dict[str, Any] = {}
    if not failures:
        try:
            device = resolve_runtime_device()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            model, cfg, docking_ckpt = load_model(CONFIG, DOCKING_CHECKPOINT, device)
            confidence_model, confidence_ckpt = load_pose_confidence_model(
                CONFIDENCE_CHECKPOINT, device
            )
            docking_step = docking_ckpt.get("step")
            confidence_step = confidence_ckpt.get("step")
        except Exception as exc:
            failures.append(_failure("__run__", exc, "model_load"))

    if not failures:
        assert model is not None and confidence_model is not None and device is not None
        pose_dir = attempt.attempt_dir / "poses"
        for index, prepared in enumerate(prepared_inputs, start=1):
            attempted_ids.append(prepared.assignment.sample_key)
            try:
                row = evaluate_prepared(
                    prepared,
                    eta=eta,
                    model=model,
                    confidence_model=confidence_model,
                    cfg=cfg,
                    device=device,
                    pose_dir=pose_dir,
                )
                rows.append(row)
                print(
                    f"[{index:04d}/{len(prepared_inputs)}] {prepared.assignment.sample_key} "
                    f"confidence={row['confidence_rmsd']:.3f} oracle={row['oracle_rmsd']:.3f}"
                )
            except Exception as exc:
                failures.append(_failure(prepared.assignment.sample_key, exc, "evaluate_one"))
                print(
                    f"[{index:04d}/{len(prepared_inputs)}] "
                    f"{prepared.assignment.sample_key} FAIL {exc!r}"
                )

    if device is not None and device.type == "cuda":
        try:
            torch.cuda.synchronize(device)
        except Exception as exc:
            failures.append(_failure("__runtime__", exc, "cuda_synchronize"))

    assigned_ids = [assignment.sample_key for assignment in plan.assigned]
    success_ids = [str(row["id"]) for row in rows]
    not_attempted_ids = sorted(set(assigned_ids) - set(attempted_ids))
    row_inventory_ok = (
        len(rows) == len(assigned_ids)
        and success_ids == assigned_ids
        and not failures
        and not not_attempted_ids
    )
    status = "complete" if row_inventory_ok else "failed"
    elapsed = time.monotonic() - started_clock
    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": status,
        "run_id": args.run_id,
        "eta_tag": ETA_TAGS[eta],
        "settings": fixed_settings(eta),
        "split": {
            "path": str(SPLIT_FILE),
            "sha256": EXPECTED_SPLIT_SHA256,
            "expected_unique_val_count": EXPECTED_VAL_COUNT,
            "full_val_ids_sha256": sorted_id_sha256(list(plan.full_keys)),
        },
        "inventory": {
            "full_val_count": len(plan.full_keys),
            "full_val_ids": list(plan.full_keys),
            "selected_count": len(plan.selected_keys),
            "selected_ids": list(plan.selected_keys),
            "selected_ids_sha256": sorted_id_sha256(list(plan.selected_keys)),
            "num_shards": args.num_shards,
            "shard_index": args.shard_index,
            "assigned_count": len(assigned_ids),
            "assigned_ids": assigned_ids,
            "assigned_ids_sha256": sorted_id_sha256(assigned_ids),
            "attempted_ids": attempted_ids,
            "success_count": len(rows),
            "success_ids": success_ids,
            "not_attempted_ids": not_attempted_ids,
            "failure_count": len(failures),
        },
        "processed_coverage": {
            "root": str(processed_root),
            "expected": len(plan.full_keys),
            "present": len(plan.full_keys) - len(processed_missing),
            "missing_ids": processed_missing,
        },
        "source": {
            "name": "PLINDER",
            "release": PLINDER_RELEASE,
            "raw_root": str(raw_root),
            "raw_layout": "systems/<system_id>/receptor.pdb + ligand_files/<ligand_chain>.sdf",
            "assigned_input_identities": source_records,
        },
        "fixed_identities": identities,
        "checkpoint_steps": {
            "docking": docking_step,
            "confidence": confidence_step,
        },
        "seed_contract": {
            "name": "BASE42_PLUS_SORTED_GLOBAL_INDEX_1_BASED_V1",
            "base_seed": BASE_SEED,
            "assignment_order": "globally sorted full validation sample keys",
            "subset_and_shard_order": "after global seed assignment",
        },
        "statistics": summarize_rows(rows),
        "failures": failures,
        "runtime": _runtime_snapshot(device, started_at=started_at, elapsed_seconds=elapsed),
    }
    if row_inventory_ok:
        output_dir = publish_complete_attempt(attempt, rows, summary)
    else:
        _publish_artifacts(
            attempt.attempt_dir,
            attempt.attempt_dir,
            rows,
            summary,
        )
        output_dir = attempt.attempt_dir
    print(
        json.dumps(
            {
                "status": status,
                "run_id": args.run_id,
                "eta": eta,
                "shard": f"{args.shard_index}/{args.num_shards}",
                "assigned": len(assigned_ids),
                "success": len(rows),
                "failures": len(failures),
                "output": str(output_dir),
            },
            sort_keys=True,
        )
    )
    if not row_inventory_ok:
        raise RuntimeError(
            "PLINDER guidance shard incomplete: "
            f"assigned={len(assigned_ids)} success={len(rows)} "
            f"failures={len(failures)} not_attempted={len(not_attempted_ids)}"
        )
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--processed-root", type=Path, default=PROCESSED_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--eta", type=_eta_arg, required=True)
    parser.add_argument("--num-shards", type=_positive_int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--only-id", action="append", default=[])
    parser.add_argument(
        "--smoke",
        nargs="?",
        const=1,
        type=_positive_int,
        default=None,
        metavar="N",
        help="Use the first N frozen validation IDs (default: 1).",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    execute(args)


if __name__ == "__main__":
    main()
