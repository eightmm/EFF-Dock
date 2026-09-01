#!/usr/bin/env python3
"""Preflight and run the frozen paired PLINDER checkpoint comparison.

The CPU preflight fixes an outcome-independent cohort from canonical PLINDER
SMILES.  The GPU command then evaluates all three EMA checkpoints sequentially
on one shard and refuses partial or unpaired publication.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import gc
import hashlib
import io
import json
import math
import os
import re
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import torch

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
PROTOCOL_ID = "EFFDOCK-EARLY-TIME-SAMPLER-PLINDER-K2-GATE-V1"
ELIGIBILITY_SCHEMA = "effdock.plinder_checkpoint_eligibility.v1"
SHARD_SCHEMA = "effdock.plinder_checkpoint_paired_shard.v1"
PLINDER_RELEASE = "2024-06/v2"

SPLIT_FILE = PROJECT_ROOT / "data/splits/plinder.json"
POOL_PARQUET = PROJECT_ROOT / "data/plinder_pool.parquet"
PROCESSED_ROOT = PROJECT_ROOT / "data/plinder_processed"
CONFIG = PROJECT_ROOT / "configs/train.yaml"
PROTOCOL_DOCUMENT = (
    PROJECT_ROOT / "docs/EARLY_TIME_SAMPLER_PLINDER_K2_GATE_PROTOCOL.md"
)
CONFORMER_AUDIT = (
    PROJECT_ROOT
    / "outputs/analysis/rdkit_fragment_geometry_v2/val1076_seed0_heavy_only.json"
)
RAW_GATE_MANIFEST = (
    PROJECT_ROOT
    / "outputs/benchmarks/plinder_guidance_validation_runs/20260804T042517Z"
    / "raw_gate/verified.json"
)
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "outputs/benchmarks/early_time_sampler_plinder_k2_paired_runs"
)

EXPECTED_SPLIT_SHA256 = "3ac570bf08bced053f1ce040b57efca27c3be616f29a82cd66ef887c08860e6b"
EXPECTED_POOL_SHA256 = "0ff455da77ce5540b839918cccb96f45414e91efff6272d7da3a65337ab1fe91"
EXPECTED_CONFIG_SHA256 = "39aa62e4a48ed6f3aa4ff59345fb43a81220e2baba22edfd5beb0c4981b307ec"
EXPECTED_RAW_GATE_SHA256 = "1ac146cfbec49ebfd1eb4452219320f134b0261bc8dc1bc196bcdab91b60f546"
EXPECTED_PROTOCOL_SHA256 = "0250853ae0793db288be2a6a8dc775db391d25aae32835b65b061782f34ab518"
EXPECTED_EVALUATOR_SHA256 = "0cf1b0e96edfc06467a15cbe2a6f0aaed1ee62d729219caab93b134519ea07dc"
EXPECTED_CONFORMER_AUDIT_SHA256 = (
    "d30f7380186d914b60964e120280dd84470b0f67b5a8aa9548e499af0aa942bf"
)
EXPECTED_ELIGIBLE_NEWLINE_SHA256 = (
    "005577bbf2b0c1c1e98bac3092b8e5350a6aa06597442b4c86d05f24e763593f"
)
EXPECTED_VAL_COUNT = 1076
EXPECTED_ELIGIBLE_COUNT = 1035
EXPECTED_ELIGIBLE_SYSTEM_COUNT = 1020
MIN_ELIGIBLE_FRACTION = 0.95
BASE_SEED = 42
CONFORMER_SEED = 0
RUN_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")


@dataclass(frozen=True)
class ArmSpec:
    name: str
    checkpoint: Path
    sha256: str
    step: int
    source_checkpoint_step: int


ARMS = (
    ArmSpec(
        name="s25_ema",
        checkpoint=(
            PROJECT_ROOT
            / "outputs/eff-dock/early-time-t0p10-50k-v1-20260813/checkpoints"
            / "step25000_ema_inference.pt"
        ),
        sha256="c343ebc34cea3395762cd82e1c54b8c7b847dc04c4fa9e80b9813a864cafa0e1",
        step=25000,
        source_checkpoint_step=25000,
    ),
    ArmSpec(
        name="s50_ema",
        checkpoint=(
            PROJECT_ROOT
            / "outputs/eff-dock/early-time-t0p10-50k-v1-20260813/checkpoints"
            / "step50000_ema_common_init.pt"
        ),
        sha256="65be44d7dc8f0867eb9fc5d22214b80f93971ea4702679a527c665046e91e6b6",
        step=50000,
        source_checkpoint_step=50000,
    ),
    ArmSpec(
        name="parent50k_plus10k_t0p10_ema",
        checkpoint=(
            PROJECT_ROOT
            / "outputs/eff-dock/early-time-t0-dose-control-t0p10-10k-v1-20260814"
            / "checkpoints/parent50k_plus10k_t0p10_ema_inference.pt"
        ),
        sha256="0a48577379e286c584abd8c652d079b09dd6fff3c06a1a2f433d617ab0cd6074",
        step=10000,
        source_checkpoint_step=10000,
    ),
)

REPLAY_ARM = ArmSpec(
    name="s50_ema_replay",
    checkpoint=ARMS[1].checkpoint,
    sha256=ARMS[1].sha256,
    step=ARMS[1].step,
    source_checkpoint_step=ARMS[1].source_checkpoint_step,
)

STAGE_SETTINGS = {
    "smoke": {
        "selected_count": 8,
        "num_samples": 4,
        "num_steps": 2,
        "prior_pool_size": 4,
        "include_s50_replay": True,
    },
    "pilot": {
        "selected_count": 32,
        "num_samples": 100,
        "num_steps": 10,
        "prior_pool_size": 100,
        "include_s50_replay": True,
    },
    "full": {
        "selected_count": None,
        "num_samples": 100,
        "num_steps": 10,
        "prior_pool_size": 100,
        "include_s50_replay": False,
    },
}

CODE_IDENTITY_FILES = (
    Path(__file__).resolve(),
    PROJECT_ROOT / "src/effdock/workflows/evaluate.py",
    PROJECT_ROOT / "src/effdock/workflows/benchmark_inputs.py",
    PROJECT_ROOT / "src/effdock/evaluation/benchmark.py",
    PROJECT_ROOT / "src/effdock/inference/docking.py",
    PROJECT_ROOT / "src/effdock/inference/preprocess.py",
    PROJECT_ROOT / "src/effdock/inference/sampler.py",
)


@dataclass(frozen=True)
class Assignment:
    sample_key: str
    global_index: int
    sampling_seed: int


@dataclass(frozen=True)
class AssignmentPlan:
    full_keys: tuple[str, ...]
    eligible_keys: tuple[str, ...]
    selected_keys: tuple[str, ...]
    assigned: tuple[Assignment, ...]


@dataclass(frozen=True)
class InputPaths:
    sample_key: str
    system_id: str
    ligand_chain: str
    receptor: Path
    ligand_reference: Path
    processed_meta: Path


@dataclass(frozen=True)
class PreparedInput:
    assignment: Assignment
    paths: InputPaths
    item: ComplexInput
    identity: dict[str, Any]


@dataclass(frozen=True)
class ShardAttempt:
    attempt_dir: Path
    final_dir: Path
    publish_lock: Path


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
    if not isinstance(sample_key, str) or "__" not in sample_key:
        raise ValueError(f"invalid PLINDER sample key: {sample_key!r}")
    system_id, ligand_chain = sample_key.rsplit("__", 1)
    return _safe_component(system_id, "system_id"), _safe_component(
        ligand_chain, "ligand_chain"
    )


def _asset_identity(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _require_identity(path: Path, expected_sha256: str, label: str) -> dict[str, Any]:
    identity = _asset_identity(path)
    if identity["sha256"] != expected_sha256:
        raise RuntimeError(
            f"{label} SHA-256 mismatch: expected {expected_sha256}, got {identity['sha256']}"
        )
    return identity


def _smiles_identity(smiles: str) -> str:
    digest = hashlib.sha256()
    digest.update(b"EFFDOCK_PLINDER_CANONICAL_SMILES_V1\0")
    digest.update(smiles.encode("utf-8"))
    return digest.hexdigest()


def _newline_id_sha256(ids: Sequence[str]) -> str:
    payload = "".join(f"{sample_id}\n" for sample_id in ids).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


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


def _atomic_write_noreplace(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def load_frozen_val_keys(
    split_file: Path = SPLIT_FILE,
    *,
    expected_sha256: str = EXPECTED_SPLIT_SHA256,
    expected_count: int = EXPECTED_VAL_COUNT,
) -> list[str]:
    _require_identity(split_file, expected_sha256, "PLINDER split")
    payload = json.loads(split_file.read_text())
    values = payload.get("val") if isinstance(payload, dict) else None
    if not isinstance(values, list) or not all(isinstance(key, str) and key for key in values):
        raise ValueError("PLINDER split must contain a non-empty string list at key 'val'")
    if len(values) != expected_count or len(set(values)) != expected_count:
        raise ValueError(
            f"expected {expected_count} unique PLINDER val IDs, got "
            f"{len(values)} rows/{len(set(values))} unique"
        )
    for key in values:
        parse_sample_key(key)
    return sorted(values)


def load_pool_smiles(
    pool_path: Path,
    full_keys: Sequence[str],
    *,
    expected_sha256: str = EXPECTED_POOL_SHA256,
) -> dict[str, str]:
    _require_identity(pool_path, expected_sha256, "PLINDER pool parquet")
    frame = pd.read_parquet(
        pool_path,
        columns=("system_id", "ligand_instance_chain", "ligand_rdkit_canonical_smiles"),
    )
    frame = frame.assign(
        sample_key=[
            f"{system_id}__{chain}"
            for system_id, chain in zip(
                frame["system_id"], frame["ligand_instance_chain"], strict=True
            )
        ]
    )
    if bool(frame["sample_key"].duplicated().any()):
        raise ValueError("PLINDER pool parquet contains duplicate sample keys")
    rows = frame.set_index("sample_key")["ligand_rdkit_canonical_smiles"].to_dict()
    missing = [key for key in full_keys if key not in rows]
    if missing:
        raise ValueError(f"{len(missing)} frozen val IDs are missing from the PLINDER pool")
    result: dict[str, str] = {}
    invalid: list[str] = []
    for key in full_keys:
        value = rows[key]
        if not isinstance(value, str) or not value.strip():
            invalid.append(key)
        else:
            result[key] = value.strip()
    if invalid:
        raise ValueError(f"{len(invalid)} frozen val IDs lack canonical SMILES")
    return result


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
        ligand_reference=system_root / "ligand_files" / f"{ligand_chain}.sdf",
        processed_meta=processed_root / sample_key / "meta.pt",
    )


def load_raw_gate(raw_gate_path: Path, raw_root: Path, full_keys: Sequence[str]) -> dict[str, Any]:
    _require_identity(raw_gate_path, EXPECTED_RAW_GATE_SHA256, "frozen PLINDER raw gate")
    payload = json.loads(raw_gate_path.read_text())
    expected_ids = list(full_keys)
    if payload.get("status") != "passed" or payload.get("mismatches") != []:
        raise ValueError("frozen PLINDER raw gate is not a passed zero-mismatch inventory")
    if payload.get("split_sha256") != EXPECTED_SPLIT_SHA256:
        raise ValueError("frozen PLINDER raw gate split identity mismatch")
    if int(payload.get("sample_count", -1)) != len(expected_ids):
        raise ValueError("frozen PLINDER raw gate sample count mismatch")
    if payload.get("sample_ids_sha256") != sorted_id_sha256(expected_ids):
        raise ValueError("frozen PLINDER raw gate sample ID hash mismatch")
    if Path(str(payload.get("raw_root", ""))).resolve() != raw_root.resolve():
        raise ValueError("requested raw root differs from the frozen PLINDER raw gate")
    assets = payload.get("assets")
    if not isinstance(assets, list):
        raise ValueError("frozen PLINDER raw gate lacks its asset ledger")
    by_id = {str(record.get("sample_id")): record for record in assets}
    if len(by_id) != len(assets) or sorted(by_id) != expected_ids:
        raise ValueError("frozen PLINDER raw gate asset IDs mismatch")
    return by_id


def load_frozen_conformer_audit(
    audit_path: Path,
    full_keys: Sequence[str],
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    _require_identity(
        audit_path,
        EXPECTED_CONFORMER_AUDIT_SHA256,
        "outcome-independent conformer/mapping audit",
    )
    payload = json.loads(audit_path.read_text())
    config = payload.get("config", {})
    inputs = payload.get("inputs", {})
    if payload.get("protocol_id") != "EFFDOCK-RDKIT-FRAGMENT-GEOMETRY-AUDIT-V2":
        raise ValueError("unexpected conformer/mapping audit protocol")
    if (
        int(config.get("requested_complexes", -1)) != EXPECTED_VAL_COUNT
        or int(config.get("rdkit_seed", -1)) != CONFORMER_SEED
        or config.get("hydrogen_policy") != "remove_all_hs"
        or inputs.get("split_file_sha256") != EXPECTED_SPLIT_SHA256
        or inputs.get("pool_parquet_sha256") != EXPECTED_POOL_SHA256
    ):
        raise ValueError("conformer/mapping audit input contract mismatch")
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError("conformer/mapping audit lacks records")
    by_id = {str(record.get("sample_key")): record for record in records}
    if len(by_id) != len(records) or sorted(by_id) != list(full_keys):
        raise ValueError("conformer/mapping audit does not cover the exact frozen val cohort")
    eligible = sorted(
        sample_id
        for sample_id, record in by_id.items()
        if record.get("status") == "ok"
        and record.get("mapping_method") == "strict_stereo"
        and record.get("symmetry_complete") is True
    )
    if len(eligible) != EXPECTED_ELIGIBLE_COUNT:
        raise ValueError(
            f"expected {EXPECTED_ELIGIBLE_COUNT} audit-eligible IDs, got {len(eligible)}"
        )
    if _newline_id_sha256(eligible) != EXPECTED_ELIGIBLE_NEWLINE_SHA256:
        raise ValueError("audit-eligible newline ID digest mismatch")
    systems = {str(by_id[sample_id].get("system_id")) for sample_id in eligible}
    if len(systems) != EXPECTED_ELIGIBLE_SYSTEM_COUNT:
        raise ValueError(
            f"expected {EXPECTED_ELIGIBLE_SYSTEM_COUNT} eligible systems, got {len(systems)}"
        )
    return by_id, eligible


def _load_meta(paths: InputPaths) -> tuple[torch.Tensor, dict[str, Any]]:
    if not paths.processed_meta.is_file():
        raise FileNotFoundError(paths.processed_meta)
    meta = torch.load(paths.processed_meta, map_location="cpu", weights_only=True)
    if not isinstance(meta, dict):
        raise TypeError(f"processed meta is not a mapping: {paths.processed_meta}")
    expected = {
        "pdb_id": paths.sample_key,
        "plinder_system_id": paths.system_id,
        "plinder_ligand_chain": paths.ligand_chain,
    }
    for field, value in expected.items():
        if meta.get(field) != value:
            raise ValueError(
                f"{paths.sample_key}: meta {field} mismatch: expected {value!r}, "
                f"got {meta.get(field)!r}"
            )
    center = torch.as_tensor(meta.get("pocket_center"), dtype=torch.float32).detach().cpu()
    if center.shape != (3,) or not bool(torch.isfinite(center).all()):
        raise ValueError(f"{paths.sample_key}: invalid pocket_center")
    return center, meta


def _assert_asset_matches(path: Path, expected: dict[str, Any], label: str) -> dict[str, Any]:
    current = _asset_identity(path)
    if current["sha256"] != expected.get("sha256"):
        raise RuntimeError(f"{label} differs from frozen raw-gate asset")
    if int(current["size_bytes"]) != int(expected.get("size_bytes", -1)):
        raise RuntimeError(f"{label} size differs from frozen raw-gate asset")
    return current


def _preflight_one(task: dict[str, Any]) -> dict[str, Any]:
    sample_key = str(task["sample_key"])
    try:
        paths = input_paths_for_sample(
            sample_key,
            raw_root=Path(task["raw_root"]),
            processed_root=Path(task["processed_root"]),
        )
        raw_asset = task["raw_asset"]
        receptor = _assert_asset_matches(paths.receptor, raw_asset["receptor"], "receptor")
        reference = _assert_asset_matches(
            paths.ligand_reference, raw_asset["ligand"], "ligand reference"
        )
        center, _ = _load_meta(paths)
        meta_identity = _asset_identity(paths.processed_meta)
        smiles = str(task["smiles"])
        if not smiles:
            raise ValueError("empty canonical SMILES")
        audit_record = task["audit_record"]
        audit_inputs = audit_record.get("input_sha256", {})
        if audit_record.get("system_id") != paths.system_id:
            raise ValueError("conformer audit system_id mismatch")
        if audit_record.get("ligand_instance_chain") != paths.ligand_chain:
            raise ValueError("conformer audit ligand chain mismatch")
        if audit_inputs.get("raw_ligand_sdf") != reference["sha256"]:
            raise ValueError("conformer audit crystal-reference identity mismatch")
        if audit_inputs.get("processed_meta_pt") != meta_identity["sha256"]:
            raise ValueError("conformer audit processed-meta identity mismatch")
        raw_smiles_sha256 = hashlib.sha256(smiles.encode("utf-8")).hexdigest()
        if task["audit_eligible"] and audit_record.get("smiles_sha256") != raw_smiles_sha256:
            raise ValueError("conformer audit canonical-SMILES identity mismatch")

        common = {
            "sample_key": sample_key,
            "global_index": int(task["global_index"]),
            "sampling_seed": BASE_SEED + int(task["global_index"]),
            "ligand_conformer_seed": CONFORMER_SEED,
            "system_id": paths.system_id,
            "ligand_chain": paths.ligand_chain,
            "canonical_smiles": smiles,
            "canonical_smiles_identity_sha256": _smiles_identity(smiles),
            "canonical_smiles_raw_sha256": raw_smiles_sha256,
            "pocket_center": [float(value) for value in center.tolist()],
            "receptor": receptor,
            "ligand_reference": reference,
            "processed_meta": meta_identity,
            "audit_record_status": audit_record.get("status"),
        }
        if not task["audit_eligible"]:
            return {
                **common,
                "status": "excluded",
                "audit_failure_stage": audit_record.get("failure_stage"),
                "audit_failure_code": audit_record.get("failure_code"),
                "audit_error_type": audit_record.get("error_type"),
                "audit_error": audit_record.get("error"),
            }
        return {
            **common,
            "status": "eligible",
            "audit_mapping_method": audit_record.get("mapping_method"),
            "audit_symmetry_complete": audit_record.get("symmetry_complete"),
        }
    except Exception as exc:
        return {
            "sample_key": sample_key,
            "status": "preflight_error",
            "global_index": int(task["global_index"]),
            "sampling_seed": BASE_SEED + int(task["global_index"]),
            "error_type": type(exc).__name__,
            "message": str(exc),
        }


def _code_identities() -> dict[str, Any]:
    files = {
        str(path.relative_to(PROJECT_ROOT)): _asset_identity(path) for path in CODE_IDENTITY_FILES
    }
    evaluator = files["src/effdock/workflows/evaluate.py"]
    if evaluator["sha256"] != EXPECTED_EVALUATOR_SHA256:
        raise RuntimeError(
            "evaluator SHA-256 mismatch: "
            f"expected {EXPECTED_EVALUATOR_SHA256}, got {evaluator['sha256']}"
        )
    digest = hashlib.sha256()
    digest.update(b"EFFDOCK_PLINDER_PAIRED_CODE_INVENTORY_V1\0")
    digest.update(
        json.dumps(
            {name: identity["sha256"] for name, identity in files.items()},
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    )
    return {"contract": "EFFDOCK_PLINDER_PAIRED_CODE_INVENTORY_V1", "sha256": digest.hexdigest(), "files": files}


def _checkpoint_identity(arm: ArmSpec) -> dict[str, Any]:
    identity = _require_identity(arm.checkpoint, arm.sha256, f"{arm.name} checkpoint")
    payload = torch.load(arm.checkpoint, map_location="cpu", weights_only=True)
    expected = {
        "artifact_type": "effdock_ema_inference_checkpoint",
        "inference_only": True,
        "weight_source": "ema",
        "step": arm.step,
        "source_checkpoint_step": arm.source_checkpoint_step,
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            raise ValueError(
                f"{arm.name}: checkpoint {field} mismatch: expected {value!r}, "
                f"got {payload.get(field)!r}"
            )
    return {**identity, **expected, "ema_n_averaged": int(payload["ema_n_averaged"])}


def collect_fixed_identities() -> dict[str, Any]:
    return {
        "protocol_document": _require_identity(
            PROTOCOL_DOCUMENT, EXPECTED_PROTOCOL_SHA256, "frozen protocol document"
        ),
        "split": _require_identity(SPLIT_FILE, EXPECTED_SPLIT_SHA256, "PLINDER split"),
        "pool_parquet": _require_identity(
            POOL_PARQUET, EXPECTED_POOL_SHA256, "PLINDER pool parquet"
        ),
        "config": _require_identity(CONFIG, EXPECTED_CONFIG_SHA256, "training config"),
        "raw_gate": _require_identity(
            RAW_GATE_MANIFEST, EXPECTED_RAW_GATE_SHA256, "PLINDER raw gate"
        ),
        "conformer_mapping_audit": _require_identity(
            CONFORMER_AUDIT,
            EXPECTED_CONFORMER_AUDIT_SHA256,
            "outcome-independent conformer/mapping audit",
        ),
        "checkpoints": {arm.name: _checkpoint_identity(arm) for arm in ARMS},
        "code": _code_identities(),
    }


def build_eligibility_manifest(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite eligibility manifest: {args.output}")
    full_keys = load_frozen_val_keys()
    smiles_by_key = load_pool_smiles(POOL_PARQUET, full_keys)
    raw_root = args.raw_root.resolve()
    processed_root = args.processed_root.resolve()
    raw_assets = load_raw_gate(RAW_GATE_MANIFEST, raw_root, full_keys)
    audit_by_id, frozen_eligible_ids = load_frozen_conformer_audit(
        CONFORMER_AUDIT, full_keys
    )
    frozen_eligible_set = set(frozen_eligible_ids)
    fixed_identities = collect_fixed_identities()

    tasks = [
        {
            "sample_key": key,
            "global_index": index,
            "smiles": smiles_by_key[key],
            "raw_root": str(raw_root),
            "processed_root": str(processed_root),
            "raw_asset": raw_assets[key],
            "audit_record": audit_by_id[key],
            "audit_eligible": key in frozen_eligible_set,
        }
        for index, key in enumerate(full_keys, start=1)
    ]
    if args.workers == 1:
        records = [_preflight_one(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            records = list(executor.map(_preflight_one, tasks, chunksize=4))
    records.sort(key=lambda record: str(record["sample_key"]))
    preflight_errors = [record for record in records if record["status"] == "preflight_error"]
    eligible = [record for record in records if record["status"] == "eligible"]
    excluded = [record for record in records if record["status"] == "excluded"]
    eligible_ids = [str(record["sample_key"]) for record in eligible]
    excluded_ids = [str(record["sample_key"]) for record in excluded]
    fraction = len(frozen_eligible_ids) / len(full_keys)
    integrity_ok = (
        not preflight_errors
        and eligible_ids == frozen_eligible_ids
        and len(excluded_ids) == EXPECTED_VAL_COUNT - EXPECTED_ELIGIBLE_COUNT
        and sorted(eligible_ids + excluded_ids) == full_keys
    )
    status = (
        "complete"
        if integrity_ok and fraction >= args.min_eligible_fraction
        else "failed"
    )
    payload = {
        "schema_version": ELIGIBILITY_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "status": status,
        "selection_boundary": (
            "outcome-independent input availability, canonical-SMILES conformer embedding, "
            "and complete connectivity-preserving heavy-atom reference mapping only"
        ),
        "forbidden_eligibility_features": [
            "model checkpoint output",
            "candidate RMSD",
            "K2 count",
            "validity outcome",
            "confidence score",
        ],
        "ligand_input_contract": {
            "source": "data/plinder_pool.parquet:ligand_rdkit_canonical_smiles",
            "conformer_seed": CONFORMER_SEED,
            "heavy_atom_normalization": "RemoveHs_then_RemoveAllHs",
            "crystal_sdf_role": "RMSD reference and atom-mapping eligibility only",
            "crystal_sdf_input_fallback": False,
        },
        "seed_contract": {
            "name": "BASE42_PLUS_SORTED_FULL_VAL_GLOBAL_INDEX_1_BASED_V1",
            "base_seed": BASE_SEED,
            "order": "globally sorted full 1076-ID validation cohort before eligibility",
        },
        "inputs": {
            "plinder_release": PLINDER_RELEASE,
            "raw_root": str(raw_root),
            "processed_root": str(processed_root),
            "fixed_identities": fixed_identities,
        },
        "inventory": {
            "full_count": len(full_keys),
            "full_ids": full_keys,
            "full_ids_sha256": sorted_id_sha256(full_keys),
            "eligible_count": len(frozen_eligible_ids),
            "eligible_fraction": fraction,
            "minimum_eligible_fraction": args.min_eligible_fraction,
            "eligible_ids": frozen_eligible_ids,
            "eligible_ids_sha256": sorted_id_sha256(frozen_eligible_ids),
            "eligible_ids_newline_sha256": _newline_id_sha256(frozen_eligible_ids),
            "eligible_system_count": EXPECTED_ELIGIBLE_SYSTEM_COUNT,
            "excluded_count": EXPECTED_VAL_COUNT - EXPECTED_ELIGIBLE_COUNT,
            "excluded_ids": sorted(set(full_keys) - frozen_eligible_set),
            "excluded_ids_sha256": sorted_id_sha256(
                sorted(set(full_keys) - frozen_eligible_set)
            ),
            "preflight_error_count": len(preflight_errors),
            "preflight_error_ids": [record["sample_key"] for record in preflight_errors],
        },
        "records": records,
    }
    _atomic_write_noreplace(args.output.resolve(), _canonical_json_bytes(payload))
    print(
        json.dumps(
            {
                "status": status,
                "full": len(full_keys),
                "eligible": len(frozen_eligible_ids),
                "excluded": EXPECTED_VAL_COUNT - EXPECTED_ELIGIBLE_COUNT,
                "preflight_errors": len(preflight_errors),
                "coverage": fraction,
                "output": str(args.output.resolve()),
                "sha256": file_sha256(args.output.resolve()),
            },
            sort_keys=True,
        )
    )
    if status != "complete":
        raise RuntimeError(
            "frozen eligibility preflight failed: "
            f"coverage={fraction:.6f} minimum={args.min_eligible_fraction:.6f} "
            f"integrity_errors={len(preflight_errors)}"
        )
    return payload


def _validate_fixed_identities(frozen: dict[str, Any]) -> None:
    current = collect_fixed_identities()
    if frozen != current:
        raise RuntimeError("current protocol/checkpoint/input/code identities differ from preflight")


def validate_eligibility_manifest(
    path: Path,
    expected_sha256: str,
    *,
    raw_root: Path,
    processed_root: Path,
) -> dict[str, Any]:
    _require_identity(path, expected_sha256, "eligibility manifest")
    payload = json.loads(path.read_text())
    if payload.get("schema_version") != ELIGIBILITY_SCHEMA:
        raise ValueError("eligibility manifest schema mismatch")
    if payload.get("protocol_id") != PROTOCOL_ID or payload.get("status") != "complete":
        raise ValueError("eligibility manifest is not complete for this protocol")
    ligand_contract = payload.get("ligand_input_contract", {})
    if (
        ligand_contract.get("source")
        != "data/plinder_pool.parquet:ligand_rdkit_canonical_smiles"
        or ligand_contract.get("conformer_seed") != CONFORMER_SEED
        or ligand_contract.get("crystal_sdf_input_fallback") is not False
    ):
        raise ValueError("eligibility manifest violates the canonical-SMILES input contract")
    inputs = payload.get("inputs", {})
    if Path(str(inputs.get("raw_root", ""))).resolve() != raw_root.resolve():
        raise ValueError("runtime raw root differs from eligibility preflight")
    if Path(str(inputs.get("processed_root", ""))).resolve() != processed_root.resolve():
        raise ValueError("runtime processed root differs from eligibility preflight")
    _validate_fixed_identities(inputs.get("fixed_identities", {}))

    inventory = payload.get("inventory", {})
    full_keys = load_frozen_val_keys()
    eligible_ids = inventory.get("eligible_ids")
    excluded_ids = inventory.get("excluded_ids")
    if not isinstance(eligible_ids, list) or not isinstance(excluded_ids, list):
        raise ValueError("eligibility manifest lacks cohort ID lists")
    if (
        inventory.get("full_ids") != full_keys
        or inventory.get("full_ids_sha256") != sorted_id_sha256(full_keys)
        or int(inventory.get("full_count", -1)) != len(full_keys)
    ):
        raise ValueError("eligibility manifest full cohort differs from frozen split")
    if eligible_ids != sorted(eligible_ids) or len(eligible_ids) != len(set(eligible_ids)):
        raise ValueError("eligible IDs must be unique and sorted")
    if excluded_ids != sorted(excluded_ids) or len(excluded_ids) != len(set(excluded_ids)):
        raise ValueError("excluded IDs must be unique and sorted")
    if sorted(eligible_ids + excluded_ids) != full_keys:
        raise ValueError("eligible and excluded IDs do not exactly partition the full cohort")
    if inventory.get("eligible_ids_sha256") != sorted_id_sha256(eligible_ids):
        raise ValueError("eligible ID hash mismatch")
    if inventory.get("eligible_ids_newline_sha256") != EXPECTED_ELIGIBLE_NEWLINE_SHA256:
        raise ValueError("eligible newline ID digest mismatch")
    if inventory.get("excluded_ids_sha256") != sorted_id_sha256(excluded_ids):
        raise ValueError("excluded ID hash mismatch")
    if (
        int(inventory.get("eligible_count", -1)) != EXPECTED_ELIGIBLE_COUNT
        or len(eligible_ids) != EXPECTED_ELIGIBLE_COUNT
        or int(inventory.get("excluded_count", -1))
        != EXPECTED_VAL_COUNT - EXPECTED_ELIGIBLE_COUNT
        or int(inventory.get("eligible_system_count", -1)) != EXPECTED_ELIGIBLE_SYSTEM_COUNT
        or int(inventory.get("preflight_error_count", -1)) != 0
        or inventory.get("preflight_error_ids") != []
    ):
        raise ValueError("eligible count mismatch")
    if len(eligible_ids) / len(full_keys) < MIN_ELIGIBLE_FRACTION:
        raise ValueError("eligibility manifest is below the frozen minimum coverage")
    records = payload.get("records")
    if not isinstance(records, list) or len(records) != len(full_keys):
        raise ValueError("eligibility record inventory mismatch")
    if [record.get("sample_key") for record in records] != full_keys:
        raise ValueError("eligibility records are not in frozen global order")
    status_by_id = {str(record["sample_key"]): record.get("status") for record in records}
    if any(status_by_id[sample_id] != "eligible" for sample_id in eligible_ids):
        raise ValueError("an audit-eligible ID did not pass CPU input integrity preflight")
    if any(status_by_id[sample_id] != "excluded" for sample_id in excluded_ids):
        raise ValueError("the 41 frozen preprocessing failures were not retained exactly")
    return payload


def plan_assignments(
    full_keys: Sequence[str],
    eligible_keys: Sequence[str],
    *,
    num_shards: int,
    shard_index: int,
    only_ids: Sequence[str] = (),
    smoke_count: int | None = None,
) -> AssignmentPlan:
    full = tuple(sorted(full_keys))
    eligible = tuple(sorted(eligible_keys))
    if len(full) != len(set(full)) or len(eligible) != len(set(eligible)):
        raise ValueError("full and eligible ID inventories must be unique")
    if not set(eligible).issubset(full):
        raise ValueError("eligible IDs must be a subset of the full validation IDs")
    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError("shard_index must satisfy 0 <= shard_index < num_shards")
    if only_ids and smoke_count is not None:
        raise ValueError("--only-id and --smoke are mutually exclusive")
    if only_ids:
        requested = tuple(only_ids)
        if len(requested) != len(set(requested)):
            raise ValueError("--only-id values must be unique")
        missing = sorted(set(requested) - set(eligible))
        if missing:
            raise ValueError(f"requested IDs are not in the eligible cohort: {missing}")
        requested_set = set(requested)
        selected = tuple(key for key in eligible if key in requested_set)
    elif smoke_count is not None:
        if smoke_count < 1:
            raise ValueError("smoke_count must be positive")
        selected = eligible[:smoke_count]
    else:
        selected = eligible
    assigned_keys = selected[shard_index::num_shards]
    if not assigned_keys:
        raise ValueError("requested subset/shard has no assigned eligible IDs")
    global_index = {key: index for index, key in enumerate(full, start=1)}
    assigned = tuple(
        Assignment(key, global_index[key], BASE_SEED + global_index[key])
        for key in assigned_keys
    )
    return AssignmentPlan(full, eligible, selected, assigned)


def _record_by_id(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(record["sample_key"]): record
        for record in manifest["records"]
        if record.get("status") == "eligible"
    }


def prepare_input(
    assignment: Assignment,
    record: dict[str, Any],
    *,
    raw_root: Path,
    processed_root: Path,
) -> PreparedInput:
    paths = input_paths_for_sample(
        assignment.sample_key, raw_root=raw_root, processed_root=processed_root
    )
    if record.get("canonical_smiles") in {None, ""}:
        raise ValueError(f"{assignment.sample_key}: eligibility record has no canonical SMILES")
    if int(record.get("global_index", -1)) != assignment.global_index:
        raise ValueError(f"{assignment.sample_key}: frozen global index mismatch")
    if int(record.get("sampling_seed", -1)) != assignment.sampling_seed:
        raise ValueError(f"{assignment.sample_key}: frozen sampling seed mismatch")
    if int(record.get("ligand_conformer_seed", -1)) != CONFORMER_SEED:
        raise ValueError(f"{assignment.sample_key}: frozen conformer seed mismatch")
    current_receptor = _asset_identity(paths.receptor)
    current_reference = _asset_identity(paths.ligand_reference)
    current_meta = _asset_identity(paths.processed_meta)
    for label, current, frozen in (
        ("receptor", current_receptor, record["receptor"]),
        ("ligand reference", current_reference, record["ligand_reference"]),
        ("processed meta", current_meta, record["processed_meta"]),
    ):
        if current["sha256"] != frozen.get("sha256"):
            raise RuntimeError(f"{assignment.sample_key}: {label} changed after preflight")
    center, _ = _load_meta(paths)
    frozen_center = torch.tensor(record["pocket_center"], dtype=torch.float32)
    if not torch.equal(center, frozen_center):
        raise ValueError(f"{assignment.sample_key}: pocket center changed after preflight")
    smiles = str(record["canonical_smiles"])
    smiles_sha256 = _smiles_identity(smiles)
    if smiles_sha256 != record.get("canonical_smiles_identity_sha256"):
        raise ValueError(f"{assignment.sample_key}: canonical SMILES identity mismatch")
    item = ComplexInput(
        complex_id=assignment.sample_key,
        protein=paths.receptor,
        ligand_ref=paths.ligand_reference,
        ligand_format="sdf",
        smiles=smiles,
        pocket_center=tuple(float(value) for value in center.tolist()),
        ligand_input_identity_sha256=smiles_sha256,
        ligand_input_canonical_smiles=smiles,
        # Reuse the benchmark loader's strict RemoveAllHs normalization.  The
        # source is still the PLINDER canonical SMILES, never the reference SDF.
        enforce_benchmark_heavy_atom_policy=True,
    )
    if item.smiles is None:
        raise RuntimeError("crystal-input fallback is forbidden")
    identity = {
        "sample_key": assignment.sample_key,
        "global_index": assignment.global_index,
        "sampling_seed": assignment.sampling_seed,
        "ligand_conformer_seed": CONFORMER_SEED,
        "canonical_smiles_identity_sha256": smiles_sha256,
        "receptor_sha256": current_receptor["sha256"],
        "ligand_reference_sha256": current_reference["sha256"],
        "processed_meta_sha256": current_meta["sha256"],
    }
    return PreparedInput(assignment, paths, item, identity)


def execution_arms(stage: str) -> tuple[ArmSpec, ...]:
    if stage not in STAGE_SETTINGS:
        raise ValueError(f"unknown execution stage: {stage!r}")
    return ARMS + ((REPLAY_ARM,) if STAGE_SETTINGS[stage]["include_s50_replay"] else ())


def fixed_settings(stage: str) -> dict[str, Any]:
    if stage not in STAGE_SETTINGS:
        raise ValueError(f"unknown execution stage: {stage!r}")
    stage_settings = STAGE_SETTINGS[stage]
    return {
        "stage": stage,
        "selected_count": stage_settings["selected_count"],
        "num_samples": stage_settings["num_samples"],
        "num_steps": stage_settings["num_steps"],
        "model_pose_step_budget": (
            int(stage_settings["num_samples"]) * int(stage_settings["num_steps"])
        ),
        "sigma": 2.0,
        "prior_pool_size": stage_settings["prior_pool_size"],
        "time_schedule": "late",
        "schedule_power": 3.0,
        "pocket_cutoff_angstrom": 10.0,
        "center_jitter_sigma": 0.0,
        "confidence": False,
        "vina_selection": False,
        "vina_guidance_scale": 0.0,
        "unified_guidance_scale": 0.0,
        "fk_constraint_beta": 0.0,
        "fk_resample_times": [],
        "translation_sde_base_sigma": 0.0,
        "sampling_dynamics": "deterministic_ode",
        "refine": "none",
        "selector_profile": "candidate_only",
        "ligand_conformer_seed": CONFORMER_SEED,
        "include_s50_replay": stage_settings["include_s50_replay"],
    }


def resolve_runtime_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("paired PLINDER sampling requires one visible CUDA GPU")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("paired PLINDER sampling requires exactly one visible CUDA GPU")
    return torch.device("cuda")


def _validate_loaded_checkpoint(arm: ArmSpec, checkpoint: dict[str, Any]) -> None:
    expected = {
        "artifact_type": "effdock_ema_inference_checkpoint",
        "inference_only": True,
        "weight_source": "ema",
        "step": arm.step,
        "source_checkpoint_step": arm.source_checkpoint_step,
    }
    for field, value in expected.items():
        if checkpoint.get(field) != value:
            raise ValueError(f"{arm.name}: loaded checkpoint {field} mismatch")


def evaluate_prepared(
    prepared: PreparedInput,
    *,
    model: torch.nn.Module,
    cfg: dict[str, Any],
    device: torch.device,
    pose_dir: Path,
    stage: str,
) -> dict[str, Any]:
    data_cfg = cfg.get("data", {})
    settings = fixed_settings(stage)
    num_samples = int(settings["num_samples"])
    num_steps = int(settings["num_steps"])
    prior_pool_size = int(settings["prior_pool_size"])
    row = evaluate_one(
        model,
        prepared.item,
        dataset="plinder_val",
        confidence_model=None,
        device=device,
        num_samples=num_samples,
        num_steps=num_steps,
        sigma=2.0,
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
        unified_guidance_scale=0.0,
        unified_guidance_start_t=0.5,
        unified_guidance_ramp_power=1.0,
        unified_guidance_max_force=20.0,
        unified_guidance_max_velocity=5.0,
        unified_guidance_max_angular_velocity=5.0,
        unified_guidance_max_atom_displacement=0.25,
        unified_guidance_max_backtracks=8,
        unified_guidance_protein_shell=18.0,
        unified_guidance_receptor_policy="fail_closed",
        unified_guidance_mode="operator_split",
        prior_pool_size=prior_pool_size,
        seed=prepared.assignment.sampling_seed,
        refine="none",
        pose_dir=pose_dir,
        trajectory_dir=None,
        require_full_ligand_atom_mapping=True,
        selector_profile="candidate_only",
        fk_constraint_beta=0.0,
        fk_resample_times=(),
        fk_resample_translation_jitter=0.0,
        fk_resample_rotation_jitter=0.0,
        translation_sde_base_sigma=0.0,
        ligand_conformer_seed=CONFORMER_SEED,
    )
    if row.get("id") != prepared.assignment.sample_key:
        raise RuntimeError("evaluate_one returned the wrong sample ID")
    if row.get("selector_profile") != "candidate_only":
        raise RuntimeError("evaluate_one did not retain candidate_only selector provenance")
    if int(row.get("num_samples", -1)) != num_samples:
        raise RuntimeError("evaluate_one changed the frozen candidate count")
    if int(row.get("sampling_seed", -1)) != prepared.assignment.sampling_seed:
        raise RuntimeError("evaluate_one changed the frozen sampling seed")
    if int(row.get("ligand_conformer_seed", -1)) != CONFORMER_SEED:
        raise RuntimeError("evaluate_one changed the frozen ligand conformer seed")
    if (
        int(row.get("prior_pool_size", -1)) != prior_pool_size
        or not row.get("prior_pool_sha256")
    ):
        raise RuntimeError("evaluate_one did not record the frozen shared prior pool")
    if not row.get("candidate_ensemble_sha256"):
        raise RuntimeError("evaluate_one omitted the ordered candidate-ensemble hash")
    if row.get("guidance_mode") != "none" or row.get("sampling_dynamics") != "deterministic_ode":
        raise RuntimeError("evaluate_one activated non-protocol inference dynamics")
    if row.get("full_heavy_atom_bijection") is not True:
        raise RuntimeError("evaluate_one did not retain a complete heavy-atom mapping")
    if row.get("ligand_input_identity_sha256") != prepared.identity.get(
        "canonical_smiles_identity_sha256"
    ):
        raise RuntimeError("evaluate_one changed the canonical-SMILES input identity")
    if int(row.get("selected_index", -1)) != 0:
        raise RuntimeError("candidate_only selected index is not the first candidate")
    candidate_rmsds = json.loads(str(row.get("candidate_rmsds_json", "[]")))
    if len(candidate_rmsds) != num_samples or not all(
        math.isfinite(float(value)) for value in candidate_rmsds
    ):
        raise RuntimeError("evaluate_one did not return all finite candidate RMSDs")
    rmsd_methods = json.loads(str(row.get("candidate_rmsd_method_json", "[]")))
    allowed_rmsd_methods = {
        "rdkit_calc_rms_symmetry_no_align",
        "mapped_index_fallback",
    }
    if len(rmsd_methods) != num_samples or not all(
        method in allowed_rmsd_methods for method in rmsd_methods
    ):
        raise RuntimeError("evaluate_one did not retain one recognized RMSD method per candidate")
    fallback_count = sum(method == "mapped_index_fallback" for method in rmsd_methods)
    if int(row.get("num_mapped_index_rmsd_fallback_candidates", -1)) != fallback_count:
        raise RuntimeError("evaluate_one mapped-index fallback count disagrees with method ledger")
    candidate_fast_valid = json.loads(str(row.get("candidate_fast_valid_json", "[]")))
    if len(candidate_fast_valid) != num_samples or not all(
        isinstance(value, bool) for value in candidate_fast_valid
    ):
        raise RuntimeError("evaluate_one did not retain all ordered fast-valid labels")
    if int(row.get("num_fast_valid_candidates", -1)) != sum(candidate_fast_valid):
        raise RuntimeError("evaluate_one fast-valid count disagrees with ordered labels")
    expected_k2 = sum(float(value) < 2.0 for value in candidate_rmsds)
    if int(row.get("num_rmsd_lt2_candidates", -1)) != expected_k2:
        raise RuntimeError("evaluate_one K2 count disagrees with candidate RMSDs")
    expected_fast_k2 = sum(
        float(rmsd) < 2.0 and is_valid
        for rmsd, is_valid in zip(candidate_rmsds, candidate_fast_valid, strict=True)
    )
    if int(row.get("num_fast_valid_rmsd_lt2_candidates", -1)) != expected_fast_k2:
        raise RuntimeError("evaluate_one fast-valid K2 disagrees with ordered candidate labels")
    if int(row.get("all_poses_count", -1)) != num_samples:
        raise RuntimeError("evaluate_one did not save the complete candidate ensemble")
    all_poses = Path(str(row.get("all_poses_sdf", "")))
    if not all_poses.is_file() or row.get("all_poses_sdf_sha256") != file_sha256(all_poses):
        raise RuntimeError("all-pose SDF is missing or has the wrong hash")
    selected_path = pose_dir / "selected" / f"{prepared.assignment.sample_key}.sdf"
    saved_hashes = json.loads(str(row.get("saved_pose_sha256_json", "{}")))
    if not selected_path.is_file() or saved_hashes.get("selected") != file_sha256(selected_path):
        raise RuntimeError("candidate_only selected-pose artifact is missing or unhashed")
    diversity_fields = (
        "coordinate_unique_count",
        "pairwise_heavy_atom_rmsd_mean",
        "pairwise_heavy_atom_rmsd_median",
        "pairwise_heavy_atom_rmsd_ge2_fraction",
        "nearest_neighbor_heavy_atom_rmsd_median",
        "c2_connected_component_count",
        "diversity_heavy_atom_count",
    )
    if any(field not in row for field in diversity_fields):
        raise RuntimeError("evaluate_one omitted frozen candidate-diversity fields")
    if not all(math.isfinite(float(row[field])) for field in diversity_fields):
        raise RuntimeError("evaluate_one returned non-finite candidate diversity")
    if not 1 <= int(row["coordinate_unique_count"]) <= num_samples:
        raise RuntimeError("evaluate_one coordinate-unique count is out of range")
    if not 1 <= int(row["c2_connected_component_count"]) <= num_samples:
        raise RuntimeError("evaluate_one C2 component count is out of range")
    row.update(
        plinder_system_id=prepared.paths.system_id,
        plinder_ligand_chain=prepared.paths.ligand_chain,
        plinder_global_index=prepared.assignment.global_index,
        processed_meta=str(prepared.paths.processed_meta),
        processed_meta_sha256=prepared.identity["processed_meta_sha256"],
    )
    return row


def _validate_run_id(run_id: str) -> str:
    if not RUN_ID_RE.fullmatch(run_id):
        raise ValueError(
            "run ID must start with an alphanumeric character and contain only letters, "
            "digits, '.', '_' or '-'"
        )
    return run_id


def reserve_shard_directory(
    output_root: Path,
    *,
    run_id: str,
    mode: str,
    num_shards: int,
    shard_index: int,
) -> ShardAttempt:
    run_id = _validate_run_id(run_id)
    run_root = output_root / run_id / mode
    run_root.mkdir(parents=True, exist_ok=True)
    shard_name = f"shard-{shard_index:03d}-of-{num_shards:03d}"
    final_dir = run_root / shard_name
    if final_dir.exists():
        raise FileExistsError(f"refusing to rerun or overwrite shard output: {final_dir}")
    incomplete_root = run_root / ".incomplete"
    incomplete_root.mkdir(exist_ok=True)
    attempt_dir = Path(tempfile.mkdtemp(prefix=f"{shard_name}.attempt-", dir=incomplete_root))
    return ShardAttempt(
        attempt_dir=attempt_dir,
        final_dir=final_dir,
        publish_lock=incomplete_root / f".{shard_name}.publish.lock",
    )


def _csv_bytes(rows: list[dict[str, Any]]) -> bytes:
    core_fields = [
        "id",
        "arm",
        "plinder_global_index",
        "sampling_seed",
        "ligand_conformer_seed",
        "num_rmsd_lt2_candidates",
        "num_fast_valid_rmsd_lt2_candidates",
    ]
    field_set = {key for row in rows for key in row}
    fieldnames = core_fields + sorted(field_set - set(core_fields))
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, extrasaction="raise")
    writer.writeheader()
    writer.writerows(_json_safe(rows))
    return buffer.getvalue().encode("utf-8")


def _canonicalize_all_pose_paths_for_csv(
    rows: list[dict[str, Any]],
    *,
    arm: str,
    attempt_dir: Path,
    visible_root: Path,
) -> list[dict[str, Any]]:
    """Verify attempt SDFs and point serialized rows at their visible location."""
    canonical_rows: list[dict[str, Any]] = []
    attempt_dir = attempt_dir.resolve()
    visible_root = visible_root.resolve()
    arm = _safe_component(arm, "arm")
    for row in rows:
        sample_id = _safe_component(str(row.get("id", "")), "sample ID")
        expected_source = (
            attempt_dir / "arms" / arm / "poses" / "all_poses" / f"{sample_id}.sdf"
        )
        recorded_value = row.get("all_poses_sdf")
        if not isinstance(recorded_value, str) or not recorded_value:
            raise ValueError(f"{arm}/{sample_id}: missing all_poses_sdf path")
        recorded_source = Path(recorded_value)
        if recorded_source.resolve() != expected_source:
            raise ValueError(
                f"{arm}/{sample_id}: all_poses_sdf is outside its canonical attempt path"
            )
        if not expected_source.is_file():
            raise FileNotFoundError(
                f"{arm}/{sample_id}: missing source all-pose SDF: {expected_source}"
            )
        expected_sha256 = row.get("all_poses_sdf_sha256")
        actual_sha256 = file_sha256(expected_source)
        if not isinstance(expected_sha256, str) or actual_sha256 != expected_sha256:
            raise RuntimeError(
                f"{arm}/{sample_id}: source all-pose SDF SHA-256 mismatch: "
                f"expected={expected_sha256!r} actual={actual_sha256}"
            )
        relative_path = expected_source.relative_to(attempt_dir)
        canonical_row = dict(row)
        canonical_row["all_poses_sdf"] = str(visible_root / relative_path)
        canonical_rows.append(canonical_row)
    return canonical_rows


def _runtime_snapshot(device: torch.device, started_at: str, elapsed: float) -> dict[str, Any]:
    runtime: dict[str, Any] = {
        "started_at_utc": started_at,
        "finished_at_utc": datetime.now(UTC).isoformat(),
        "elapsed_seconds": elapsed,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "device": str(device),
        "gpu": None,
    }
    if device.type == "cuda":
        runtime.update(
            gpu=torch.cuda.get_device_name(device),
            gpu_total_memory_bytes=int(torch.cuda.get_device_properties(device).total_memory),
            cuda_max_memory_allocated_bytes=int(torch.cuda.max_memory_allocated(device)),
            cuda_max_memory_reserved_bytes=int(torch.cuda.max_memory_reserved(device)),
        )
    return runtime


def _write_attempt_artifacts(
    attempt: ShardAttempt,
    *,
    rows_by_arm: dict[str, list[dict[str, Any]]],
    summary: dict[str, Any],
    visible_root: Path,
) -> None:
    arm_artifacts: dict[str, Any] = {}
    for arm, rows in rows_by_arm.items():
        arm_dir = attempt.attempt_dir / "arms" / arm
        arm_dir.mkdir(parents=True, exist_ok=True)
        serialized_rows = _canonicalize_all_pose_paths_for_csv(
            rows,
            arm=arm,
            attempt_dir=attempt.attempt_dir,
            visible_root=visible_root,
        )
        data = _csv_bytes(serialized_rows)
        _atomic_write_noreplace(arm_dir / "results.csv", data)
        arm_summary = {
            "arm": arm,
            "count": len(rows),
            "operational_requested_count": summary["operational_inventory"][
                "requested_count"
            ],
            "common_preprocessing_failure_count": summary["operational_inventory"][
                "common_preprocessing_failure_count"
            ],
            "statistics": summarize_rows(rows),
            "results_csv": str(visible_root / "arms" / arm / "results.csv"),
            "results_csv_sha256": hashlib.sha256(data).hexdigest(),
        }
        _atomic_write_noreplace(
            arm_dir / "summary.json", _canonical_json_bytes(_json_safe(arm_summary))
        )
        arm_artifacts[arm] = {
            **arm_summary,
            "summary": str(visible_root / "arms" / arm / "summary.json"),
        }
    summary["artifacts"] = {
        "paired_summary": str(visible_root / "paired_summary.json"),
        "arms": arm_artifacts,
    }
    _atomic_write_noreplace(
        attempt.attempt_dir / "paired_summary.json",
        _canonical_json_bytes(_json_safe(summary)),
    )


def publish_complete_attempt(attempt: ShardAttempt) -> Path:
    with attempt.publish_lock.open("a+b") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        if attempt.final_dir.exists():
            raise FileExistsError(f"refusing duplicate shard publish: {attempt.final_dir}")
        os.rename(attempt.attempt_dir, attempt.final_dir)
    return attempt.final_dir


def _paired_identity_gate(
    rows_by_arm: dict[str, list[dict[str, Any]]],
    arms: Sequence[ArmSpec],
) -> dict[str, Any]:
    expected_arms = [arm.name for arm in arms]
    if list(rows_by_arm) != expected_arms:
        raise RuntimeError("paired arm execution order changed")
    baseline_ids = [str(row["id"]) for row in rows_by_arm[expected_arms[0]]]
    failures: list[dict[str, Any]] = []
    prior_hashes: dict[str, str] = {}
    for index, sample_id in enumerate(baseline_ids):
        rows = [rows_by_arm[arm][index] for arm in expected_arms]
        if [str(row["id"]) for row in rows] != [sample_id] * len(expected_arms):
            failures.append({"id": sample_id, "reason": "arm ID/order mismatch"})
            continue
        fields = (
            "sampling_seed",
            "ligand_conformer_seed",
            "prior_pool_sha256",
            "protein_sha256",
            "ligand_reference_sha256",
            "ligand_input_identity_sha256",
        )
        mismatched = [field for field in fields if len({str(row[field]) for row in rows}) != 1]
        if mismatched:
            failures.append({"id": sample_id, "reason": "paired identity mismatch", "fields": mismatched})
        else:
            prior_hashes[sample_id] = str(rows[0]["prior_pool_sha256"])
    if failures:
        raise RuntimeError(f"paired identity gate failed for {len(failures)} complexes")
    return {
        "passed": True,
        "arm_order": expected_arms,
        "checked_count": len(baseline_ids),
        "fields": [
            "sampling_seed",
            "ligand_conformer_seed",
            "prior_pool_sha256",
            "protein_sha256",
            "ligand_reference_sha256",
            "ligand_input_identity_sha256",
        ],
        "prior_pool_sha256_by_id": prior_hashes,
    }


def _aggregate_ratio(
    baseline: Sequence[dict[str, Any]],
    replay: Sequence[dict[str, Any]],
    field: str,
) -> float:
    baseline_mean = sum(float(row[field]) for row in baseline) / len(baseline)
    replay_mean = sum(float(row[field]) for row in replay) / len(replay)
    if baseline_mean == 0.0:
        return 1.0 if replay_mean == 0.0 else float("inf")
    return replay_mean / baseline_mean


def _replay_integrity_gate(
    rows_by_arm: dict[str, list[dict[str, Any]]],
    *,
    stage: str,
) -> dict[str, Any]:
    if stage == "full":
        return {"required": False, "passed": True}
    baseline = rows_by_arm["s50_ema"]
    replay = rows_by_arm["s50_ema_replay"]
    if len(baseline) != len(replay) or not baseline:
        raise RuntimeError("s50 replay inventory mismatch")
    k2_coverage_mismatches = sum(
        (int(left["num_rmsd_lt2_candidates"]) >= 1)
        != (int(right["num_rmsd_lt2_candidates"]) >= 1)
        for left, right in zip(baseline, replay, strict=True)
    )
    fast_k2_coverage_mismatches = sum(
        (int(left["num_fast_valid_rmsd_lt2_candidates"]) >= 1)
        != (int(right["num_fast_valid_rmsd_lt2_candidates"]) >= 1)
        for left, right in zip(baseline, replay, strict=True)
    )
    mean_abs_k2_difference = sum(
        abs(
            int(left["num_rmsd_lt2_candidates"])
            - int(right["num_rmsd_lt2_candidates"])
        )
        for left, right in zip(baseline, replay, strict=True)
    ) / len(baseline)
    diversity_ratios = {
        field: _aggregate_ratio(baseline, replay, field)
        for field in (
            "nearest_neighbor_heavy_atom_rmsd_median",
            "c2_connected_component_count",
            "coordinate_unique_count",
        )
    }
    finite = all(math.isfinite(value) for value in diversity_ratios.values())
    if stage == "smoke":
        passed = finite
        rule = "finite/count/prior/replay engineering integrity only"
    elif stage == "pilot":
        passed = (
            k2_coverage_mismatches == 0
            and fast_k2_coverage_mismatches == 0
            and mean_abs_k2_difference <= 0.25
            and finite
            and all(0.98 <= value <= 1.02 for value in diversity_ratios.values())
        )
        rule = (
            "zero K2>=1 and fast-valid K2>=1 classification mismatches; "
            "mean abs K2 difference <=0.25; diversity ratios in [0.98,1.02]"
        )
    else:  # pragma: no cover - caller validates the stage
        raise AssertionError(stage)
    result = {
        "required": True,
        "passed": passed,
        "stage": stage,
        "rule": rule,
        "checked_count": len(baseline),
        "k2_coverage_classification_mismatches": k2_coverage_mismatches,
        "fast_valid_k2_coverage_classification_mismatches": fast_k2_coverage_mismatches,
        "mean_abs_k2_difference": mean_abs_k2_difference,
        "diversity_aggregate_ratios": diversity_ratios,
    }
    return result


def execute_shard(args: argparse.Namespace) -> dict[str, Any]:
    stage = args.stage
    if stage not in STAGE_SETTINGS:
        raise ValueError(f"unknown stage: {stage!r}")
    stage_settings = STAGE_SETTINGS[stage]
    if stage in {"smoke", "pilot"} and (args.num_shards != 1 or args.shard_index != 0):
        raise ValueError(f"{stage} requires the unsharded cohort (shard 0 of 1)")
    if stage == "full" and args.num_shards != 8:
        raise ValueError("the frozen full stage requires exactly 8 shards")
    raw_root = args.raw_root.resolve()
    processed_root = args.processed_root.resolve()
    output_root = args.output_root.resolve()
    eligibility_path = args.eligibility_manifest.resolve()
    manifest = validate_eligibility_manifest(
        eligibility_path,
        args.eligibility_manifest_sha256,
        raw_root=raw_root,
        processed_root=processed_root,
    )
    inventory = manifest["inventory"]
    plan = plan_assignments(
        inventory["full_ids"],
        inventory["eligible_ids"],
        num_shards=args.num_shards,
        shard_index=args.shard_index,
        smoke_count=stage_settings["selected_count"],
    )
    attempt = reserve_shard_directory(
        output_root,
        run_id=args.run_id,
        mode=stage,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
    )
    started_at = datetime.now(UTC).isoformat()
    started_clock = time.monotonic()
    arms = execution_arms(stage)
    rows_by_arm: dict[str, list[dict[str, Any]]] = {arm.name: [] for arm in arms}
    failures: list[dict[str, Any]] = []
    records = _record_by_id(manifest)
    device: torch.device | None = None
    prepared: list[PreparedInput] = []
    paired_gate: dict[str, Any] = {"passed": False}
    replay_gate: dict[str, Any] = {"required": stage != "full", "passed": False}
    try:
        for assignment in plan.assigned:
            prepared.append(
                prepare_input(
                    assignment,
                    records[assignment.sample_key],
                    raw_root=raw_root,
                    processed_root=processed_root,
                )
            )
        device = resolve_runtime_device()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        for arm_index, arm in enumerate(arms, start=1):
            print(
                f"arm {arm_index}/{len(arms)} {arm.name} checkpoint={arm.checkpoint} "
                f"sha256={arm.sha256}"
            )
            model, cfg, checkpoint = load_model(CONFIG, arm.checkpoint, device)
            _validate_loaded_checkpoint(arm, checkpoint)
            pose_dir = attempt.attempt_dir / "arms" / arm.name / "poses"
            try:
                for item_index, item in enumerate(prepared, start=1):
                    row = evaluate_prepared(
                        item,
                        model=model,
                        cfg=cfg,
                        device=device,
                        pose_dir=pose_dir,
                        stage=stage,
                    )
                    row["arm"] = arm.name
                    row["checkpoint"] = str(arm.checkpoint)
                    row["checkpoint_sha256"] = arm.sha256
                    rows_by_arm[arm.name].append(row)
                    print(
                        f"[{arm.name} {item_index:04d}/{len(prepared)}] "
                        f"{item.assignment.sample_key} K2={row['num_rmsd_lt2_candidates']} "
                        f"fastK2={row['num_fast_valid_rmsd_lt2_candidates']}"
                    )
            finally:
                del model
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        paired_gate = _paired_identity_gate(rows_by_arm, arms)
        replay_gate = _replay_integrity_gate(rows_by_arm, stage=stage)
        if not replay_gate["passed"]:
            raise RuntimeError(f"{stage} s50 replay integrity gate failed: {replay_gate}")
        assigned_ids = [assignment.sample_key for assignment in plan.assigned]
        for arm in arms:
            if [str(row["id"]) for row in rows_by_arm[arm.name]] != assigned_ids:
                raise RuntimeError(f"{arm.name}: output inventory/order mismatch")
        status = "complete"
    except Exception as exc:
        failures.append({**serialize_evaluation_failure("__shard__", exc), "stage": "run_shard"})
        paired_gate["passed"] = False
        status = "failed"

    elapsed = time.monotonic() - started_clock
    assigned_ids = [assignment.sample_key for assignment in plan.assigned]
    summary: dict[str, Any] = {
        "schema_version": SHARD_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "status": status,
        "run_id": args.run_id,
        "mode": stage,
        "settings": fixed_settings(stage),
        "eligibility_manifest": {
            "path": str(eligibility_path),
            "sha256": args.eligibility_manifest_sha256,
            "eligible_count": len(plan.eligible_keys),
            "eligible_ids_sha256": inventory["eligible_ids_sha256"],
            "eligible_ids_newline_sha256": inventory[
                "eligible_ids_newline_sha256"
            ],
            "eligible_system_count": inventory["eligible_system_count"],
            "ineligible_count": inventory["excluded_count"],
        },
        "inventory": {
            "full_count": len(plan.full_keys),
            "eligible_count": len(plan.eligible_keys),
            "selected_count": len(plan.selected_keys),
            "selected_ids": list(plan.selected_keys),
            "selected_ids_sha256": sorted_id_sha256(list(plan.selected_keys)),
            "num_shards": args.num_shards,
            "shard_index": args.shard_index,
            "assigned_count": len(assigned_ids),
            "assigned_ids": assigned_ids,
            "assigned_ids_sha256": sorted_id_sha256(assigned_ids),
            "arm_success_counts": {
                arm.name: len(rows_by_arm[arm.name]) for arm in arms
            },
        },
        "operational_inventory": {
            "requested_count": len(plan.full_keys),
            "evaluable_count": len(plan.eligible_keys),
            "common_preprocessing_failure_count": inventory["excluded_count"],
            "common_preprocessing_failure_ids": inventory["excluded_ids"],
            "common_preprocessing_failure_ids_sha256": inventory[
                "excluded_ids_sha256"
            ],
            "per_arm_preprocessing_failure_count": {
                arm.name: inventory["excluded_count"] for arm in arms
            },
            "operational_sensitivity_assignment": "common preprocessing failures have K2=0",
        },
        "arms": [
            {
                "name": arm.name,
                "checkpoint": str(arm.checkpoint),
                "checkpoint_sha256": arm.sha256,
                "step": arm.step,
                "source_checkpoint_step": arm.source_checkpoint_step,
            }
            for arm in arms
        ],
        "fixed_identities": manifest["inputs"]["fixed_identities"],
        "seed_contract": manifest["seed_contract"],
        "ligand_input_contract": manifest["ligand_input_contract"],
        "paired_identity_gate": paired_gate,
        "replay_integrity_gate": replay_gate,
        "failures": failures,
        "runtime": (
            _runtime_snapshot(device, started_at, elapsed)
            if device is not None
            else {
                "started_at_utc": started_at,
                "finished_at_utc": datetime.now(UTC).isoformat(),
                "elapsed_seconds": elapsed,
            }
        ),
    }
    visible_root = attempt.final_dir if status == "complete" else attempt.attempt_dir
    _write_attempt_artifacts(
        attempt,
        rows_by_arm=rows_by_arm,
        summary=summary,
        visible_root=visible_root,
    )
    output_dir = publish_complete_attempt(attempt) if status == "complete" else attempt.attempt_dir
    print(
        json.dumps(
            {
                "status": status,
                "mode": stage,
                "shard": f"{args.shard_index}/{args.num_shards}",
                "assigned": len(assigned_ids),
                "arm_success_counts": summary["inventory"]["arm_success_counts"],
                "output": str(output_dir),
            },
            sort_keys=True,
        )
    )
    if status != "complete":
        raise RuntimeError(f"paired PLINDER shard failed; diagnostics retained at {output_dir}")
    return summary


def verify_stage_summary(
    path: Path,
    *,
    stage: str,
    eligibility_manifest_sha256: str,
) -> dict[str, Any]:
    if stage not in {"smoke", "pilot"}:
        raise ValueError("only smoke and pilot are pre-full gate stages")
    payload = json.loads(path.read_text())
    if (
        payload.get("schema_version") != SHARD_SCHEMA
        or payload.get("protocol_id") != PROTOCOL_ID
        or payload.get("status") != "complete"
        or payload.get("mode") != stage
        or payload.get("settings") != fixed_settings(stage)
    ):
        raise ValueError(f"{stage} summary is not a complete frozen-protocol stage")
    if payload.get("eligibility_manifest", {}).get("sha256") != eligibility_manifest_sha256:
        raise ValueError(f"{stage} used a different eligibility manifest")
    if payload.get("paired_identity_gate", {}).get("passed") is not True:
        raise ValueError(f"{stage} paired identity gate did not pass")
    expected_arms = execution_arms(stage)
    if [entry.get("name") for entry in payload.get("arms", [])] != [
        arm.name for arm in expected_arms
    ]:
        raise ValueError(f"{stage} did not execute the frozen arms and s50 replay in order")
    counts = payload.get("inventory", {}).get("arm_success_counts", {})
    assigned = int(payload.get("inventory", {}).get("assigned_count", -1))
    expected_count = int(STAGE_SETTINGS[stage]["selected_count"])
    if assigned != expected_count or any(
        int(counts.get(arm.name, -1)) != assigned for arm in expected_arms
    ):
        raise ValueError(f"{stage} arm inventory is incomplete")
    if payload.get("replay_integrity_gate", {}).get("passed") is not True:
        raise ValueError(f"{stage} replay integrity gate did not pass")
    return payload


def verify_smoke_summary(
    path: Path,
    *,
    eligibility_manifest_sha256: str,
) -> dict[str, Any]:
    """Compatibility helper retained for focused callers."""
    return verify_stage_summary(
        path,
        stage="smoke",
        eligibility_manifest_sha256=eligibility_manifest_sha256,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight = subparsers.add_parser("preflight", help="Freeze the CPU-only eligible cohort.")
    preflight.add_argument("--raw-root", type=Path, required=True)
    preflight.add_argument("--processed-root", type=Path, default=PROCESSED_ROOT)
    preflight.add_argument("--output", type=Path, required=True)
    preflight.add_argument("--workers", type=_positive_int, default=8)
    preflight.add_argument(
        "--min-eligible-fraction",
        type=float,
        default=MIN_ELIGIBLE_FRACTION,
    )

    verify = subparsers.add_parser("verify", help="Verify a frozen eligibility manifest.")
    verify.add_argument("--raw-root", type=Path, required=True)
    verify.add_argument("--processed-root", type=Path, default=PROCESSED_ROOT)
    verify.add_argument("--eligibility-manifest", type=Path, required=True)
    verify.add_argument("--eligibility-manifest-sha256", required=True)

    run = subparsers.add_parser("run-shard", help="Run all three checkpoint arms on one GPU.")
    run.add_argument("--raw-root", type=Path, required=True)
    run.add_argument("--processed-root", type=Path, default=PROCESSED_ROOT)
    run.add_argument("--eligibility-manifest", type=Path, required=True)
    run.add_argument("--eligibility-manifest-sha256", required=True)
    run.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    run.add_argument("--run-id", required=True)
    run.add_argument("--num-shards", type=_positive_int, default=1)
    run.add_argument("--shard-index", type=int, default=0)
    run.add_argument("--stage", choices=tuple(STAGE_SETTINGS), required=True)

    stage_gate = subparsers.add_parser(
        "verify-stage", help="Gate a later run on a smoke or pilot summary."
    )
    stage_gate.add_argument("--stage", choices=("smoke", "pilot"), required=True)
    stage_gate.add_argument("--summary", type=Path, required=True)
    stage_gate.add_argument("--eligibility-manifest-sha256", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.command == "preflight":
        if not 0.0 <= args.min_eligible_fraction <= 1.0:
            raise ValueError("--min-eligible-fraction must lie in [0, 1]")
        build_eligibility_manifest(args)
    elif args.command == "verify":
        payload = validate_eligibility_manifest(
            args.eligibility_manifest.resolve(),
            args.eligibility_manifest_sha256,
            raw_root=args.raw_root.resolve(),
            processed_root=args.processed_root.resolve(),
        )
        print(
            json.dumps(
                {
                    "status": "passed",
                    "eligible_count": payload["inventory"]["eligible_count"],
                    "eligible_ids_sha256": payload["inventory"]["eligible_ids_sha256"],
                },
                sort_keys=True,
            )
        )
    elif args.command == "run-shard":
        execute_shard(args)
    elif args.command == "verify-stage":
        verify_stage_summary(
            args.summary.resolve(),
            stage=args.stage,
            eligibility_manifest_sha256=args.eligibility_manifest_sha256,
        )
        print(
            json.dumps(
                {
                    "status": "passed",
                    "stage": args.stage,
                    "summary": str(args.summary.resolve()),
                }
            )
        )
    else:  # pragma: no cover - argparse enforces the command set
        raise AssertionError(args.command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
