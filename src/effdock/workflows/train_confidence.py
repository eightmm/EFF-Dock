#!/usr/bin/env python3
"""Train the docking-graph pose confidence model."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
from collections.abc import Iterator
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from effdock.checkpoint import atomic_torch_save, load_portable_model_state
from effdock.confidence.dataset import (
    DEFAULT_CONFIDENCE_POSE_TAG,
    LigandPoseConfidenceDataset,
    PairedLigandPoseConfidenceDataset,
    collate_complexes,
    to_device,
)
from effdock.confidence.losses import pose_confidence_loss
from effdock.confidence.model import DockingGraphPoseConfidence
from effdock.confidence.selectors import select_confidence_poses
from effdock.training.trainer import configure_optimizers, get_warmup_stable_cosine_scheduler

torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

BANK_MANIFEST_SCHEMA = "effdock.s50_confidence_bank.manifest.v1"
BANK_PROTOCOL_ID = "EFFDOCK-S50-CONFIDENCE-TRAINING-BANK-V1"
SYMMETRY_TARGET_MANIFEST_SCHEMA = "EFFDOCK_S50_SYMMETRY_RMSD_MANIFEST_V1"
SYMMETRY_TARGET_KEY = "pose_rmsd_symmetry_no_align"
SYMMETRY_TARGET_METHOD = "rdkit_calc_rms_symmetry_no_align"
BANK_SETTINGS = {
    "num_samples": 100,
    "num_steps": 10,
    "sample_sigma": 2.0,
    "time_schedule": "late",
    "schedule_power": 3.0,
    "pocket_cutoff_angstrom": 10.0,
    "prior_pool_size": 100,
    "ligand_conformer_seed": 0,
    "sampling_dynamics": "deterministic_ode",
    "stochastic_gamma": 0.0,
    "translation_sde_base_sigma": 0.0,
    "guidance": False,
    "refine": "none",
    "fk_resampling": False,
    "particle_resampling": False,
    "eligibility_boundary": "input_only_no_sampled_pose_outcomes",
}
REFINED_BANK_SETTINGS = {
    **BANK_SETTINGS,
    "refine": "guidance_unified_step100",
    "refinement_steps": 100,
    "refinement_receptor_policy": "geometry_only",
}
ALLOWED_BANK_SETTINGS = (BANK_SETTINGS, REFINED_BANK_SETTINGS)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ordered_ids_sha256(ids: list[str]) -> str:
    digest = hashlib.sha256()
    for sample_id in ids:
        digest.update(sample_id.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _require_sha256(value: Any, *, label: str) -> str:
    normalized = str(value).lower()
    if len(normalized) != 64 or any(ch not in "0123456789abcdef" for ch in normalized):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return normalized


def _resolve_declared_path(value: Any, *, manifest_path: Path) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    beside_manifest = manifest_path.parent / path
    return beside_manifest if beside_manifest.exists() else path


def validate_bank_manifest(
    manifest_path: Path,
    *,
    expected_sha256: str | None,
    split_file: Path,
    pose_tag: str,
    allow_smoke: bool = False,
) -> dict[str, Any]:
    """Validate the sealed S50 bank identity and its filtered split inventory."""
    if not manifest_path.is_file():
        raise FileNotFoundError(f"bank manifest does not exist: {manifest_path}")
    actual_sha256 = _sha256_file(manifest_path)
    if expected_sha256 is not None:
        expected = _require_sha256(expected_sha256, label="bank manifest expected SHA-256")
        if actual_sha256 != expected:
            raise ValueError(
                f"bank manifest SHA-256 mismatch: expected {expected}, got {actual_sha256}"
            )

    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict):
        raise ValueError("bank manifest must be a JSON object")
    if manifest.get("schema_version") != BANK_MANIFEST_SCHEMA:
        raise ValueError(
            f"bank manifest schema_version must be {BANK_MANIFEST_SCHEMA!r}, "
            f"got {manifest.get('schema_version')!r}"
        )
    expected_status = "smoke_complete" if allow_smoke else "complete"
    if manifest.get("status") != expected_status:
        raise ValueError(
            f"bank manifest status must be {expected_status!r}, "
            f"got {manifest.get('status')!r}"
        )
    expected_claim_eligible = not allow_smoke
    if manifest.get("claim_eligible") is not expected_claim_eligible:
        raise ValueError(
            f"bank manifest claim_eligible must be {expected_claim_eligible} "
            f"for {'smoke' if allow_smoke else 'full'} training"
        )
    protocol_id = manifest.get("protocol_id")
    if protocol_id != BANK_PROTOCOL_ID:
        raise ValueError(
            f"bank manifest protocol_id must be {BANK_PROTOCOL_ID!r}, got {protocol_id!r}"
        )
    if manifest.get("settings") not in ALLOWED_BANK_SETTINGS:
        raise ValueError(
            "bank manifest settings differ from the frozen raw/refined S50 contracts"
        )
    if manifest.get("pose_tag") != pose_tag:
        raise ValueError(
            f"bank manifest pose_tag {manifest.get('pose_tag')!r} != requested {pose_tag!r}"
        )

    if not split_file.is_file():
        raise FileNotFoundError(f"filtered split file does not exist: {split_file}")
    filtered_sha = _require_sha256(
        manifest.get("filtered_split_sha256"), label="filtered_split_sha256"
    )
    actual_filtered_sha = _sha256_file(split_file)
    if actual_filtered_sha != filtered_sha:
        raise ValueError(
            f"filtered split SHA-256 mismatch: manifest {filtered_sha}, "
            f"--split_file {actual_filtered_sha}"
        )
    declared_split = _resolve_declared_path(
        manifest.get("filtered_split_path"), manifest_path=manifest_path
    )
    if not declared_split.is_file():
        raise FileNotFoundError(f"manifest filtered_split_path does not exist: {declared_split}")
    if _sha256_file(declared_split) != filtered_sha:
        raise ValueError("manifest filtered_split_path content does not match its SHA-256")

    with split_file.open(encoding="utf-8") as handle:
        split_map = json.load(handle)
    if not isinstance(split_map, dict):
        raise ValueError("filtered split file must be a JSON object")
    inventory = manifest.get("inventory")
    records = manifest.get("records")
    if not isinstance(inventory, dict) or not isinstance(records, list):
        raise ValueError("bank manifest requires inventory object and records list")

    all_filtered_ids: list[str] = []
    shard_paths: dict[str, dict[str, str]] = {"train": {}, "val": {}}
    system_ids: dict[str, dict[str, str]] = {"train": {}, "val": {}}
    record_maps: dict[str, dict[str, dict[str, Any]]] = {"train": {}, "val": {}}
    for split in ("train", "val"):
        ids = split_map.get(split)
        inv = inventory.get(split)
        if not isinstance(ids, list) or not all(isinstance(pid, str) and pid for pid in ids):
            raise ValueError(f"filtered split {split!r} must be a list of non-empty IDs")
        if len(ids) != len(set(ids)):
            raise ValueError(f"filtered split {split!r} contains duplicate IDs")
        if not isinstance(inv, dict):
            raise ValueError(f"bank inventory {split!r} must be an object")
        eligible_count = int(inv.get("eligible_count", -1))
        record_count = int(inv.get("record_count", -1))
        full_count = int(inv.get("full_count", -1))
        excluded_count = int(inv.get("excluded_count", -1))
        if record_count != len(ids) or (not allow_smoke and eligible_count != len(ids)):
            raise ValueError(
                f"bank inventory {split!r} count mismatch: split={len(ids)} "
                f"eligible={eligible_count} records={record_count}"
            )
        if allow_smoke and eligible_count < len(ids):
            raise ValueError(f"smoke bank inventory {split!r} exceeds eligible count")
        if full_count < eligible_count or excluded_count != full_count - eligible_count:
            raise ValueError(f"bank inventory {split!r} full/excluded counts are inconsistent")
        ids_sha = _require_sha256(
            inv.get("eligible_ids_sha256"), label=f"inventory.{split}.eligible_ids_sha256"
        )
        _require_sha256(inv.get("full_ids_sha256"), label=f"inventory.{split}.full_ids_sha256")
        _require_sha256(
            inv.get("excluded_ids_sha256"),
            label=f"inventory.{split}.excluded_ids_sha256",
        )
        record_ids_sha = _require_sha256(
            inv.get("record_ids_sha256"),
            label=f"inventory.{split}.record_ids_sha256",
        )
        if record_ids_sha != _ordered_ids_sha256(ids):
            raise ValueError(f"bank inventory {split!r} record ID ledger mismatch")
        if not allow_smoke and ids_sha != record_ids_sha:
            raise ValueError(f"bank inventory {split!r} eligible ID ledger mismatch")

        split_records = [row for row in records if isinstance(row, dict) and row.get("split") == split]
        if len(split_records) != len(ids):
            raise ValueError(
                f"bank record inventory {split!r} count {len(split_records)} != {len(ids)}"
            )
        try:
            split_indices = [int(row["split_index"]) for row in split_records]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"bank records for {split!r} require integer split_index") from exc
        if split_indices != sorted(set(split_indices)):
            raise ValueError(
                f"bank record {split!r} split_index inventory must be unique and increasing"
            )
        record_ids = [row.get("sample_key") for row in split_records]
        if record_ids != ids:
            raise ValueError(f"bank record {split!r} IDs do not match the filtered split order")
        for pid, row in zip(ids, split_records):
            _require_sha256(row.get("pt_sha256"), label=f"record {pid} pt_sha256")
            system_id = row.get("system_id")
            if not isinstance(system_id, str) or not system_id:
                raise ValueError(f"bank record {pid} requires a non-empty system_id")
            pose_count = int(row.get("pose_count", -1))
            expected_pose_count = int(BANK_SETTINGS["num_samples"])
            if pose_count != expected_pose_count:
                raise ValueError(
                    f"bank record {pid} pose_count {pose_count} != frozen sampler "
                    f"count {expected_pose_count}"
                )
            if int(row.get("size_bytes", -1)) < 1:
                raise ValueError(f"bank record {pid} has invalid size_bytes")
            record_pt = _resolve_declared_path(
                row.get("pt_path"), manifest_path=manifest_path
            ).resolve()
            if record_pt.name != f"confposes_{pose_tag}.pt":
                raise ValueError(
                    f"bank record {pid} pt_path basename does not match pose_tag: {record_pt}"
                )
            if not record_pt.is_file():
                raise FileNotFoundError(f"bank record {pid} shard does not exist: {record_pt}")
            if record_pt.stat().st_size != int(row["size_bytes"]):
                raise ValueError(f"bank record {pid} shard size does not match the manifest")
            shard_paths[split][pid] = str(record_pt)
            system_ids[split][pid] = system_id
            record_maps[split][pid] = dict(row)
        all_filtered_ids.extend(ids)

    if len(all_filtered_ids) != len(set(all_filtered_ids)):
        raise ValueError("filtered train and val splits overlap")
    if len(records) != len(all_filtered_ids):
        raise ValueError("bank manifest contains records outside the train/val inventory")

    return {
        "path": str(manifest_path.resolve()),
        "sha256": actual_sha256,
        "schema_version": BANK_MANIFEST_SCHEMA,
        "protocol_id": protocol_id,
        "status": expected_status,
        "claim_eligible": expected_claim_eligible,
        "pose_tag": pose_tag,
        "settings": manifest["settings"],
        "filtered_split_path": str(split_file.resolve()),
        "filtered_split_sha256": filtered_sha,
        "inventory": inventory,
        # Runtime-only lookup.  The caller removes this large map before
        # embedding compact provenance in checkpoints.
        "_shard_paths": shard_paths,
        "_system_ids": system_ids,
        "_records": record_maps,
    }


def validate_pose_target_manifest(
    manifest_path: Path,
    *,
    expected_sha256: str,
    bank_manifest_sha256: str,
    bank_records: dict[str, dict[str, Any]],
    ordered_train_ids: list[str],
) -> dict[str, Any]:
    """Validate and load the external symmetry target inventory for training."""
    expected = _require_sha256(expected_sha256, label="pose target manifest SHA-256")
    actual = _sha256_file(manifest_path)
    if actual != expected:
        raise ValueError(f"pose target manifest SHA-256 mismatch: expected {expected}, got {actual}")
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != SYMMETRY_TARGET_MANIFEST_SCHEMA
        or manifest.get("status") != "complete"
        or manifest.get("split") != "train"
        or manifest.get("method") != SYMMETRY_TARGET_METHOD
        or manifest.get("label_key") != SYMMETRY_TARGET_KEY
    ):
        raise ValueError("pose target manifest does not match the frozen symmetry contract")
    if int(manifest.get("num_samples_per_complex", -1)) != 100:
        raise ValueError("pose target manifest must contain exactly 100 poses per complex")
    if manifest.get("bank_manifest", {}).get("sha256") != bank_manifest_sha256:
        raise ValueError("pose target manifest bank identity does not match the training bank")
    _require_sha256(
        manifest.get("input_manifest", {}).get("sha256"),
        label="pose target input manifest SHA-256",
    )
    records = manifest.get("records")
    if not isinstance(records, list) or len(records) != len(ordered_train_ids):
        raise ValueError("pose target manifest record count does not match filtered train")
    if int(manifest.get("record_count", -1)) != len(records):
        raise ValueError("pose target manifest declared record_count is inconsistent")
    if int(manifest.get("pose_count", -1)) != 100 * len(records):
        raise ValueError("pose target manifest declared pose_count is inconsistent")

    target_records: dict[str, dict[str, Any]] = {}
    sidecar_digests: dict[Path, str] = {}
    for expected_id, record in zip(ordered_train_ids, records):
        if not isinstance(record, dict) or record.get("sample_key") != expected_id:
            raise ValueError("pose target manifest IDs do not match filtered train order")
        bank = bank_records.get(expected_id)
        if bank is None:
            raise ValueError(f"pose target manifest contains unknown bank ID: {expected_id}")
        for key, bank_key in (
            ("system_id", "system_id"),
            ("split_index", "split_index"),
            ("pose_count", "pose_count"),
            ("source_pt_sha256", "pt_sha256"),
            ("pose_ensemble_sha256", "pose_ensemble_sha256"),
        ):
            if record.get(key) != bank.get(bank_key):
                raise ValueError(f"{expected_id}: pose target {key} does not match bank")
        sidecar_path = Path(str(record.get("sidecar_path"))).resolve()
        sidecar_sha = _require_sha256(
            record.get("sidecar_sha256"), label=f"{expected_id} sidecar_sha256"
        )
        _require_sha256(record.get("label_sha256"), label=f"{expected_id} label_sha256")
        if sidecar_path in sidecar_digests and sidecar_digests[sidecar_path] != sidecar_sha:
            raise ValueError(f"{expected_id}: inconsistent sidecar SHA declaration")
        sidecar_digests[sidecar_path] = sidecar_sha
        target_records[expected_id] = {
            **record,
            "sidecar_path": str(sidecar_path),
            "bank_manifest_sha256": bank_manifest_sha256,
            "input_manifest_sha256": manifest.get("input_manifest", {}).get("sha256"),
        }
    for sidecar_path, expected_sidecar_sha in sidecar_digests.items():
        if not sidecar_path.is_file() or sidecar_path.is_symlink():
            raise FileNotFoundError(f"pose target sidecar is not a regular file: {sidecar_path}")
        if _sha256_file(sidecar_path) != expected_sidecar_sha:
            raise ValueError(f"pose target sidecar SHA-256 mismatch: {sidecar_path}")
    return {
        "path": str(manifest_path.resolve()),
        "sha256": actual,
        "schema_version": SYMMETRY_TARGET_MANIFEST_SCHEMA,
        "method": SYMMETRY_TARGET_METHOD,
        "label_key": SYMMETRY_TARGET_KEY,
        "record_count": len(records),
        "pose_count": 100 * len(records),
        "bank_manifest_sha256": bank_manifest_sha256,
        "_records": target_records,
    }


def _load_checkpoint(path: Path, device: torch.device | str) -> dict[str, Any]:
    """Safely load retained confidence checkpoints containing Path metadata."""
    with torch.serialization.safe_globals([type(Path())]):
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"confidence checkpoint must be a mapping, got {type(checkpoint).__name__}")
    return checkpoint


def load_config_sections(
    config_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    import yaml

    with config_path.open() as handle:
        cfg = yaml.safe_load(handle)
    model = cfg["model"]
    model_cfg = {
        "flow_hidden_dim": int(model["hidden_dim"]),
        "flow_hidden_vec_dim": int(model["hidden_vec_dim"]),
        "flow_l2_dim": int(model["l2_dim"]),
        "flow_l2o_dim": int(model["l2o_dim"]),
        "use_saved_ligand_hidden": bool(model.get("use_saved_ligand_hidden", True)),
    }
    return (
        model_cfg,
        dict(cfg.get("training", {})),
        dict(cfg.get("confidence", {})),
        dict(cfg.get("loss", {})),
    )


def _arg_or_cfg(value: Any, cfg: dict[str, Any], key: str, default: Any) -> Any:
    return cfg.get(key, default) if value is None else value


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def setup_ddp() -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("DDP confidence training requires CUDA.")
        torch.cuda.set_device(0 if torch.cuda.device_count() == 1 else local_rank)
        timeout_minutes = int(os.environ.get("CONFIDENCE_DDP_TIMEOUT_MIN", "180"))
        if timeout_minutes <= 0:
            raise ValueError("CONFIDENCE_DDP_TIMEOUT_MIN must be positive")
        dist.init_process_group(backend="nccl", timeout=timedelta(minutes=timeout_minutes))
    return rank, local_rank, world_size


def cleanup_ddp(world_size: int) -> None:
    if world_size > 1:
        dist.destroy_process_group()


class _ResumableDistributedSampler(DistributedSampler):
    """DistributedSampler with a one-shot in-epoch data-order resume offset."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._start_index = 0

    def set_start_index(self, start_index: int) -> None:
        if not 0 <= start_index <= super().__len__():
            raise ValueError(f"sampler start index out of range: {start_index}")
        self._start_index = int(start_index)

    def __iter__(self) -> Iterator[int]:
        iterator = iter(super().__iter__())
        start_index = self._start_index
        self._start_index = 0
        for _ in range(start_index):
            next(iterator)
        return iterator


def _stable_pose_order(values: torch.Tensor) -> list[int]:
    """Order finite scores ascending, resolving exact ties by saved pose index."""
    flat = values.detach().cpu().to(torch.float64).reshape(-1)
    if not bool(torch.isfinite(flat).all()):
        raise ValueError("confidence evaluation predictions must be finite")
    return sorted(range(int(flat.numel())), key=lambda index: (float(flat[index]), index))


def _restore_training_states(
    checkpoint: dict[str, Any],
    *,
    optimizers: list[Any],
    schedulers: list[Any],
) -> None:
    """Restore every optimizer and scheduler or reject a partial resume."""
    optimizer_states = checkpoint.get("optimizer_state_dicts")
    scheduler_states = checkpoint.get("scheduler_state_dicts")
    if not isinstance(optimizer_states, list) or len(optimizer_states) != len(optimizers):
        actual = len(optimizer_states) if isinstance(optimizer_states, list) else "missing"
        raise ValueError(
            "resume optimizer state count mismatch: "
            f"checkpoint={actual}, configured={len(optimizers)}"
        )
    if not isinstance(scheduler_states, list) or len(scheduler_states) != len(schedulers):
        actual = len(scheduler_states) if isinstance(scheduler_states, list) else "missing"
        raise ValueError(
            "resume scheduler state count mismatch: "
            f"checkpoint={actual}, configured={len(schedulers)}"
        )
    for optimizer, state in zip(optimizers, optimizer_states, strict=True):
        optimizer.load_state_dict(state)
    for scheduler, state in zip(schedulers, scheduler_states, strict=True):
        scheduler.load_state_dict(state)


def _summarize_eval_target(
    selected: list[float],
    top5_selected: list[float],
    oracle: list[float],
    oracle_k2: list[int],
) -> dict[str, float]:
    if not selected:
        raise ValueError("confidence evaluation target inventory is empty")
    sel = torch.tensor(selected, dtype=torch.float32)
    top5 = torch.tensor(top5_selected, dtype=torch.float32)
    ora = torch.tensor(oracle, dtype=torch.float32)
    metrics = {
        "eval_top1_mean": float(sel.mean().item()),
        "eval_top1_median": float(sel.median().item()),
        "eval_top1_lt1": float((sel < 1).float().mean().item() * 100),
        "eval_top1_lt2": float((sel < 2).float().mean().item() * 100),
        "eval_top1_lt5": float((sel < 5).float().mean().item() * 100),
        "eval_top5_mean": float(top5.mean().item()),
        "eval_top5_median": float(top5.median().item()),
        "eval_top5_lt1": float((top5 < 1).float().mean().item() * 100),
        "eval_top5_lt2": float((top5 < 2).float().mean().item() * 100),
        "eval_top5_lt5": float((top5 < 5).float().mean().item() * 100),
        "eval_oracle_mean": float(ora.mean().item()),
        "eval_oracle_median": float(ora.median().item()),
        "eval_oracle_lt2": float((ora < 2).float().mean().item() * 100),
    }
    slice_masks = {
        "0": [count == 0 for count in oracle_k2],
        "1_4": [1 <= count <= 4 for count in oracle_k2],
        "5_9": [5 <= count <= 9 for count in oracle_k2],
        "ge10": [count >= 10 for count in oracle_k2],
    }
    for name, raw_mask in slice_masks.items():
        mask = torch.tensor(raw_mask, dtype=torch.bool)
        count = int(mask.sum().item())
        prefix = f"eval_oracle_k2_{name}"
        metrics[f"{prefix}_n"] = float(count)
        metrics[f"{prefix}_top1_lt2"] = (
            float((sel[mask] < 2).float().mean().item() * 100) if count else 0.0
        )
        metrics[f"{prefix}_top5_lt2"] = (
            float((top5[mask] < 2).float().mean().item() * 100) if count else 0.0
        )
    return metrics


def _oracle_k2_slice(count: int) -> str:
    if count == 0:
        return "0"
    if count <= 4:
        return "1_4"
    if count <= 9:
        return "5_9"
    return "ge10"


def _release_eval_cuda_cache(device: torch.device) -> None:
    """Release rank-local evaluation allocations before DDP training resumes."""
    if device.type != "cuda":
        return
    gc.collect()
    torch.cuda.empty_cache()


def evaluate(
    model: DockingGraphPoseConfidence,
    loader: DataLoader,
    device: torch.device,
    *,
    max_complexes: int | None,
    eval_target_key: str = "pose_rmsd",
    eval_records: list[dict[str, Any]] | None = None,
) -> dict[str, float]:
    model.eval()
    selected = []
    selected_success = []
    selected_frozen = []
    selected_rank_vote = []
    oracle = []
    eval_selected = []
    eval_top5_selected = []
    eval_oracle = []
    eval_oracle_k2 = []
    pose_loss = []
    n_seen = 0
    with torch.no_grad():
        for batch in loader:
            for raw in batch:
                item = to_device(raw, device)
                out = model.forward_complex(item)
                true = item["pose_rmsd"]
                eval_true = item[eval_target_key]
                if eval_true.shape != true.shape:
                    raise ValueError(
                        f"{item.get('pid', '<unknown>')}: {eval_target_key} shape "
                        f"{tuple(eval_true.shape)} != pose_rmsd shape {tuple(true.shape)}"
                    )
                if not bool(torch.isfinite(eval_true).all()):
                    raise ValueError(
                        f"{item.get('pid', '<unknown>')}: {eval_target_key} must be finite"
                    )
                stable_order = _stable_pose_order(out["pose_rmsd"])
                pred_idx = stable_order[0]
                succ_idx = int(out["pose_success_logit"].argmax().item())
                selected.append(float(true[pred_idx].item()))
                selected_success.append(float(true[succ_idx].item()))
                pred_rmsd = out["pose_rmsd"].detach().cpu().to(torch.float32)
                pred_success = torch.sigmoid(
                    out["pose_success_logit"].detach().cpu().to(torch.float32)
                )
                atom_disp = torch.expm1(
                    out["atom_disp_log1p"].detach().cpu().to(torch.float32).clamp(-2.0, 5.0)
                ).clamp_min(0.0)
                atom_ok = torch.sigmoid(out["atom_ok_logit"].detach().cpu().to(torch.float32))
                atom_rmsd = torch.sqrt(atom_disp.square().mean(dim=1).clamp_min(0.0))
                atom_q90 = torch.quantile(atom_disp, 0.9, dim=1)
                atom_ok_mean = atom_ok.mean(dim=1)
                scores = [
                    {
                        "confidence_rmsd": float(pred_rmsd[i]),
                        "confidence_success": float(pred_success[i]),
                        "confidence_atom_rmsd": float(atom_rmsd[i]),
                        "confidence_atom_q90": float(atom_q90[i]),
                        "confidence_atom_ok": float(atom_ok_mean[i]),
                    }
                    for i in range(int(true.numel()))
                ]
                # Dataset graph and pose coordinates are already pocket-centered,
                # so the frozen selector receives a zero center here.
                selector_indices = select_confidence_poses(
                    [pose for pose in item["pose_atom_coords"]],
                    scores,
                    item["graph"],
                    torch.zeros(3, device=device),
                )
                selected_frozen.append(
                    float(
                        true[
                            selector_indices["pair_gate_density_rank_vote_plclash_ambig"]
                        ].item()
                    )
                )
                selected_rank_vote.append(
                    float(true[selector_indices["rank_vote"]].item())
                )
                oracle.append(float(true.min().item()))
                top5_indices = torch.tensor(
                    stable_order[: min(5, len(stable_order))],
                    dtype=torch.long,
                    device=eval_true.device,
                )
                eval_selected.append(float(eval_true[pred_idx].item()))
                eval_top5_selected.append(
                    float(eval_true.index_select(0, top5_indices).min().item())
                )
                eval_oracle.append(float(eval_true.min().item()))
                oracle_k2 = int((eval_true < 2.0).sum().item())
                eval_oracle_k2.append(oracle_k2)
                if eval_records is not None:
                    top1_value = eval_selected[-1]
                    top5_value = eval_top5_selected[-1]
                    oracle_value = eval_oracle[-1]
                    eval_records.append(
                        {
                            "pid": str(item.get("pid", "")),
                            "system_id": item.get("system_id"),
                            "pose_count": int(true.numel()),
                            "oracle_k2": oracle_k2,
                            "oracle_k2_slice": _oracle_k2_slice(oracle_k2),
                            "top1_index": pred_idx,
                            "top5_indices": [int(index) for index in top5_indices.cpu().tolist()],
                            "top1_rmsd": top1_value,
                            "top5_best_rmsd": top5_value,
                            "oracle_rmsd": oracle_value,
                            "top1_lt2": bool(top1_value < 2.0),
                            "top5_lt2": bool(top5_value < 2.0),
                            "oracle_lt2": bool(oracle_value < 2.0),
                        }
                    )
                pose_loss.append(
                    float(
                        torch.nn.functional.huber_loss(
                            out["pose_rmsd_log1p"],
                            torch.log1p(true),
                        ).item()
                    )
                )
                n_seen += 1
                if max_complexes is not None and n_seen >= max_complexes:
                    break
            if max_complexes is not None and n_seen >= max_complexes:
                break
    sel = torch.tensor(selected)
    sel_s = torch.tensor(selected_success)
    sel_frozen = torch.tensor(selected_frozen)
    sel_rank_vote = torch.tensor(selected_rank_vote)
    ora = torch.tensor(oracle)
    metrics = {
        "n": float(n_seen),
        "selected_mean": float(sel.mean().item()),
        "selected_median": float(sel.median().item()),
        "selected_lt1": float((sel < 1).float().mean().item() * 100),
        "selected_lt2": float((sel < 2).float().mean().item() * 100),
        "selected_lt5": float((sel < 5).float().mean().item() * 100),
        "success_selected_median": float(sel_s.median().item()),
        "success_selected_lt2": float((sel_s < 2).float().mean().item() * 100),
        "frozen_selected_median": float(sel_frozen.median().item()),
        "frozen_selected_lt2": float((sel_frozen < 2).float().mean().item() * 100),
        "rank_vote_selected_lt2": float((sel_rank_vote < 2).float().mean().item() * 100),
        "oracle_mean": float(ora.mean().item()),
        "oracle_median": float(ora.median().item()),
        "oracle_lt2": float((ora < 2).float().mean().item() * 100),
        "pose_loss": float(sum(pose_loss) / max(1, len(pose_loss))),
    }
    metrics.update(
        _summarize_eval_target(
            eval_selected,
            eval_top5_selected,
            eval_oracle,
            eval_oracle_k2,
        )
    )
    return metrics


def fmt_metrics(prefix: str, metrics: dict[str, float]) -> str:
    return (
        f"{prefix} n={metrics['n']:.0f} "
        f"sel_mean={metrics['selected_mean']:.3f} sel_med={metrics['selected_median']:.3f} "
        f"sel<1={metrics['selected_lt1']:.1f}% sel<2={metrics['selected_lt2']:.1f}% "
        f"sel<5={metrics['selected_lt5']:.1f}% "
        f"succ_med={metrics['success_selected_median']:.3f} succ<2={metrics['success_selected_lt2']:.1f}% "
        f"frozen_med={metrics['frozen_selected_median']:.3f} "
        f"frozen<2={metrics['frozen_selected_lt2']:.1f}% "
        f"oracle_med={metrics['oracle_median']:.3f} oracle<2={metrics['oracle_lt2']:.1f}% "
        f"eval_top1<2={metrics['eval_top1_lt2']:.1f}% "
        f"eval_top5<2={metrics['eval_top5_lt2']:.1f}%"
    )


def _wandb_safe_config(args: argparse.Namespace) -> dict[str, Any]:
    cfg: dict[str, Any] = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            cfg[key] = str(value)
        else:
            cfg[key] = value
    return cfg


def _write_eval_ledger(
    path: Path,
    *,
    step: int,
    eval_target_key: str,
    records: list[dict[str, Any]],
    bank_provenance: dict[str, Any] | None,
) -> dict[str, str]:
    payload = {
        "schema_version": "effdock.confidence_eval_ledger.v1",
        "status": "complete",
        "step": int(step),
        "update": int(step),
        "eval_target": eval_target_key,
        "bank_manifest_sha256": (
            bank_provenance.get("sha256") if bank_provenance is not None else None
        ),
        "record_count": len(records),
        "records": records,
    }
    serialized = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    expected_sha = hashlib.sha256(serialized).hexdigest()
    if path.exists():
        actual_sha = _sha256_file(path)
        if actual_sha != expected_sha:
            raise FileExistsError(
                f"refusing to overwrite non-identical evaluation ledger: {path}"
            )
        return {"path": str(path.resolve()), "sha256": actual_sha}

    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("xb") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            actual_sha = _sha256_file(path)
            if actual_sha != expected_sha:
                raise FileExistsError(
                    f"refusing to overwrite non-identical evaluation ledger: {path}"
                )
    finally:
        temporary.unlink(missing_ok=True)
    return {"path": str(path.resolve()), "sha256": expected_sha}


def _resume_eval_state(
    checkpoint: dict[str, Any],
    *,
    out_dir: Path,
    step: int,
    eval_every: int,
    eval_target_key: str,
    bank_provenance: dict[str, Any] | None,
) -> tuple[str, dict[str, str] | None, dict[str, float] | None]:
    """Classify a scheduled-boundary resume without silently losing its evaluation."""
    checkpoint_args = checkpoint.get("args")
    u0_was_scheduled = isinstance(checkpoint_args, dict) and bool(
        checkpoint_args.get("eval_on_start", False)
    )
    if step < 0 or eval_every < 1:
        raise ValueError("resume step must be non-negative and eval_every must be positive")
    if (step == 0 and not u0_was_scheduled) or (step > 0 and step % eval_every != 0):
        return "not_scheduled", None, None

    ledger_path = out_dir / f"eval_u{step:06d}.json"
    marker = checkpoint.get("evaluation_ledger")
    if not ledger_path.is_file():
        if marker is not None:
            raise FileNotFoundError(
                f"resume checkpoint commits a missing evaluation ledger: {ledger_path}"
            )
        return "evaluate_missing", None, None

    with ledger_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    expected_bank_sha = (
        bank_provenance.get("sha256") if bank_provenance is not None else None
    )
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != "effdock.confidence_eval_ledger.v1"
        or payload.get("status") != "complete"
        or int(payload.get("step", -1)) != step
        or int(payload.get("update", -1)) != step
        or payload.get("eval_target") != eval_target_key
        or payload.get("bank_manifest_sha256") != expected_bank_sha
    ):
        raise ValueError(f"invalid scheduled evaluation ledger: {ledger_path}")
    records = payload.get("records")
    if (
        not isinstance(records, list)
        or not records
        or int(payload.get("record_count", -1)) != len(records)
    ):
        raise ValueError(f"invalid record inventory in evaluation ledger: {ledger_path}")
    try:
        metrics = {
            "n": float(len(records)),
            **_summarize_eval_target(
                [float(record["top1_rmsd"]) for record in records],
                [float(record["top5_best_rmsd"]) for record in records],
                [float(record["oracle_rmsd"]) for record in records],
                [int(record["oracle_k2"]) for record in records],
            ),
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid metric records in evaluation ledger: {ledger_path}") from exc
    ledger = {"path": str(ledger_path.resolve()), "sha256": _sha256_file(ledger_path)}
    if marker is None:
        # The ledger was atomically written, but latest.pt was not yet rewritten
        # as the transaction commit marker. Recover best selection from the
        # sealed outcomes without evaluating the model a second time.
        return "commit_existing", ledger, metrics
    if not isinstance(marker, dict) or marker != ledger:
        raise ValueError("resume checkpoint evaluation-ledger commitment mismatch")
    return "committed", ledger, metrics


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/train_confidence.yaml"))
    parser.add_argument("--split_file", type=Path, default=None)
    parser.add_argument("--processed_dir", type=Path, default=None)
    parser.add_argument("--pose_tag", type=str, default=None)
    parser.add_argument("--tag", type=str, default=None, help="Deprecated alias for --pose_tag.")
    parser.add_argument("--out_dir", type=Path, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--use_wandb", action=argparse.BooleanOptionalAction, default=None)
    checkpoint_group = parser.add_mutually_exclusive_group()
    checkpoint_group.add_argument("--resume", type=Path, default=None)
    checkpoint_group.add_argument(
        "--init_checkpoint",
        type=Path,
        default=None,
        help="Load model weights only and start a fresh U0 optimizer/scheduler run.",
    )
    parser.add_argument("--init_checkpoint_sha256", type=str, default=None)
    parser.add_argument("--reset-optimizer", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--finetune-steps", type=int, default=None)
    parser.add_argument("--eval-on-start", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--total_steps", type=int, default=None)
    parser.add_argument("--batch_complexes", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--muon_lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--use_muon", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--scheduler_type", choices=["warmup_stable_cosine", "none"], default=None)
    parser.add_argument("--warmup_ratio", type=float, default=None)
    parser.add_argument("--cooldown_ratio", type=float, default=None)
    parser.add_argument("--min_lr_ratio", type=float, default=None)
    parser.add_argument("--max_grad_norm", type=float, default=None)
    parser.add_argument("--hidden", type=int, default=None)
    parser.add_argument("--num_layers", type=int, default=None)
    parser.add_argument("--n_rbf", type=int, default=None)
    parser.add_argument("--sh_lmax", type=int, default=None)
    parser.add_argument("--cond_dim", type=int, default=None)
    parser.add_argument("--contact_cutoff", type=float, default=None)
    parser.add_argument(
        "--pose_readout",
        type=str,
        default=None,
        choices=["global_pool", "contact_attention", "global_contact_attention"],
    )
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument(
        "--protein_crop_mode",
        type=str,
        default=None,
        choices=["pose_residue", "ligand_residue", "center"],
    )
    parser.add_argument("--protein_contact_cutoff", type=float, default=None)
    parser.add_argument("--protein_crop_cutoff", type=float, default=None)
    parser.add_argument("--protein_crop_cutoff_min", type=float, default=None)
    parser.add_argument("--protein_crop_cutoff_max", type=float, default=None)
    parser.add_argument("--protein_crop_jitter_sigma", type=float, default=None)
    parser.add_argument("--protein_crop_jitter_max", type=float, default=None)
    parser.add_argument("--stochastic_crop", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--disable_stochastic_crop", action="store_false", dest="stochastic_crop")
    parser.add_argument("--max_protein_atoms", type=int, default=None)
    parser.add_argument("--max_train_pose_node_product", type=int, default=None)
    parser.add_argument("--large_train_graph_node_threshold", type=int, default=None)
    parser.add_argument("--large_train_graph_max_poses", type=int, default=None)
    parser.add_argument("--max_train_poses_per_complex", type=int, default=None)
    parser.add_argument("--max_val_poses_per_complex", type=int, default=None)
    parser.add_argument(
        "--pose_sample_strategy", type=str, default=None, choices=["best_random", "stratified"]
    )
    parser.add_argument("--train_limit", type=int, default=None)
    parser.add_argument("--val_limit", type=int, default=None)
    parser.add_argument("--val_start", type=int, default=None)
    parser.add_argument("--eval_every", type=int, default=None)
    parser.add_argument("--eval_complexes", type=int, default=None)
    parser.add_argument(
        "--training_target",
        type=str,
        default=None,
        help="Per-pose target used by pose-level training losses.",
    )
    parser.add_argument(
        "--eval_target",
        type=str,
        default=None,
        help="Per-pose shard field used only for validation selection metrics.",
    )
    parser.add_argument("--bank_manifest", type=Path, default=None)
    parser.add_argument("--bank_manifest_sha256", type=str, default=None)
    parser.add_argument("--train_pose_target_manifest", type=Path, default=None)
    parser.add_argument("--train_pose_target_manifest_sha256", type=str, default=None)
    parser.add_argument("--train_aux_bank_manifest", type=Path, default=None)
    parser.add_argument("--train_aux_bank_manifest_sha256", type=str, default=None)
    parser.add_argument("--train_aux_pose_tag", type=str, default=None)
    parser.add_argument("--train_aux_pose_target_manifest", type=Path, default=None)
    parser.add_argument("--train_aux_pose_target_manifest_sha256", type=str, default=None)
    parser.add_argument("--smoke", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--save_every", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--listwise-weight", type=float, default=None)
    parser.add_argument("--success-listwise-weight", type=float, default=None)
    parser.add_argument("--setwise-success-weight", type=float, default=None)
    parser.add_argument("--setwise-success-temperature", type=float, default=None)
    parser.add_argument("--pairwise-success-weight", type=float, default=None)
    parser.add_argument("--pairwise-success-temperature", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args(argv)

    model_cfg, train_cfg, conf_cfg, loss_cfg = load_config_sections(args.config)
    loss_kwargs = {
        "atom_weight": float(loss_cfg.get("atom_weight", 0.2)),
        "atom_bce_weight": float(loss_cfg.get("atom_bce_weight", 0.2)),
        "pose_weight": float(loss_cfg.get("pose_weight", 1.0)),
        "pose_bce_weight": float(loss_cfg.get("pose_bce_weight", 0.4)),
        "rank_weight": float(loss_cfg.get("rank_weight", 0.5)),
        "listwise_weight": float(
            _arg_or_cfg(args.listwise_weight, loss_cfg, "listwise_weight", 0.0)
        ),
        "listwise_tau": float(loss_cfg.get("listwise_tau", 1.0)),
        "listwise_pred_temp": float(loss_cfg.get("listwise_pred_temp", 1.0)),
        "success_listwise_weight": float(
            _arg_or_cfg(
                args.success_listwise_weight,
                loss_cfg,
                "success_listwise_weight",
                0.0,
            )
        ),
        "success_listwise_margin": float(loss_cfg.get("success_listwise_margin", 0.5)),
        "setwise_success_weight": float(
            _arg_or_cfg(args.setwise_success_weight, loss_cfg, "setwise_success_weight", 0.0)
        ),
        "setwise_success_temperature": float(
            _arg_or_cfg(
                args.setwise_success_temperature,
                loss_cfg,
                "setwise_success_temperature",
                1.0,
            )
        ),
        "pairwise_success_weight": float(
            _arg_or_cfg(args.pairwise_success_weight, loss_cfg, "pairwise_success_weight", 0.0)
        ),
        "pairwise_success_temperature": float(
            _arg_or_cfg(
                args.pairwise_success_temperature,
                loss_cfg,
                "pairwise_success_temperature",
                1.0,
            )
        ),
        "hard_pair_weight": float(loss_cfg.get("hard_pair_weight", 0.0)),
        "hard_pair_margin": float(loss_cfg.get("hard_pair_margin", 0.2)),
        "hard_pair_positive_threshold": float(loss_cfg.get("hard_pair_positive_threshold", 2.0)),
        "hard_pair_negative_threshold": float(loss_cfg.get("hard_pair_negative_threshold", 2.0)),
        "hard_pair_min_gap": float(loss_cfg.get("hard_pair_min_gap", 0.5)),
        "success_threshold": float(loss_cfg.get("success_threshold", 2.0)),
        "rank_margin": float(loss_cfg.get("rank_margin", 0.05)),
        "min_rank_gap": float(loss_cfg.get("min_rank_gap", 0.3)),
    }
    args.loss = loss_kwargs
    args.split_file = Path(
        _arg_or_cfg(args.split_file, conf_cfg, "split_file", "data/splits/plinder.json")
    )
    args.processed_dir = Path(
        _arg_or_cfg(args.processed_dir, conf_cfg, "processed_dir", "data/plinder_processed")
    )
    cfg_pose_tag = conf_cfg.get("pose_tag", conf_cfg.get("tag", DEFAULT_CONFIDENCE_POSE_TAG))
    args.pose_tag = str(args.pose_tag or args.tag or cfg_pose_tag)
    args.out_dir = Path(
        _arg_or_cfg(args.out_dir, conf_cfg, "out_dir", "outputs/pose_confidence_full_graph")
    )
    args.run_name = str(
        _arg_or_cfg(args.run_name, conf_cfg, "run_name", "pose_confidence_full_graph")
    )
    cfg_resume = conf_cfg.get("resume", None)
    args.resume = args.resume or (Path(cfg_resume) if cfg_resume else None)
    cfg_init_checkpoint = conf_cfg.get("init_checkpoint", None)
    args.init_checkpoint = args.init_checkpoint or (
        Path(cfg_init_checkpoint) if cfg_init_checkpoint else None
    )
    if args.resume is not None and args.init_checkpoint is not None:
        raise ValueError("--resume and --init_checkpoint are mutually exclusive")
    args.init_checkpoint_sha256 = _arg_or_cfg(
        args.init_checkpoint_sha256,
        conf_cfg,
        "init_checkpoint_sha256",
        None,
    )
    args.reset_optimizer = bool(
        _arg_or_cfg(args.reset_optimizer, train_cfg, "reset_optimizer_on_resume", False)
    )
    args.eval_on_start = bool(
        _arg_or_cfg(args.eval_on_start, conf_cfg, "eval_on_start", False)
    )
    args.batch_complexes = int(_arg_or_cfg(args.batch_complexes, conf_cfg, "batch_complexes", 1))
    args.gradient_accumulation_steps = int(
        _arg_or_cfg(
            args.gradient_accumulation_steps,
            train_cfg,
            "gradient_accumulation_steps",
            1,
        )
    )
    if args.batch_complexes < 1 or args.gradient_accumulation_steps < 1:
        raise ValueError("batch_complexes and gradient_accumulation_steps must be positive")
    args.num_workers = int(_arg_or_cfg(args.num_workers, conf_cfg, "num_workers", 2))
    args.hidden = int(_arg_or_cfg(args.hidden, conf_cfg, "hidden", 512))
    args.num_layers = int(_arg_or_cfg(args.num_layers, conf_cfg, "num_layers", 4))
    args.n_rbf = int(_arg_or_cfg(args.n_rbf, conf_cfg, "n_rbf", 32))
    args.sh_lmax = int(_arg_or_cfg(args.sh_lmax, conf_cfg, "sh_lmax", 2))
    args.cond_dim = int(_arg_or_cfg(args.cond_dim, conf_cfg, "cond_dim", 128))
    args.contact_cutoff = float(_arg_or_cfg(args.contact_cutoff, conf_cfg, "contact_cutoff", 5.0))
    args.pose_readout = str(_arg_or_cfg(args.pose_readout, conf_cfg, "pose_readout", "global_pool"))
    args.dropout = float(_arg_or_cfg(args.dropout, conf_cfg, "dropout", 0.1))
    args.protein_crop_mode = str(
        _arg_or_cfg(args.protein_crop_mode, conf_cfg, "protein_crop_mode", "center")
    )
    args.protein_contact_cutoff = float(
        _arg_or_cfg(args.protein_contact_cutoff, conf_cfg, "protein_contact_cutoff", 5.0)
    )
    args.protein_crop_cutoff = float(
        _arg_or_cfg(args.protein_crop_cutoff, conf_cfg, "protein_crop_cutoff", 8.0)
    )
    args.protein_crop_cutoff_min = float(
        _arg_or_cfg(args.protein_crop_cutoff_min, conf_cfg, "protein_crop_cutoff_min", 6.0)
    )
    args.protein_crop_cutoff_max = float(
        _arg_or_cfg(args.protein_crop_cutoff_max, conf_cfg, "protein_crop_cutoff_max", 12.0)
    )
    args.protein_crop_jitter_sigma = float(
        _arg_or_cfg(args.protein_crop_jitter_sigma, conf_cfg, "protein_crop_jitter_sigma", 2.0)
    )
    args.protein_crop_jitter_max = float(
        _arg_or_cfg(args.protein_crop_jitter_max, conf_cfg, "protein_crop_jitter_max", 4.0)
    )
    args.stochastic_crop = bool(
        _arg_or_cfg(args.stochastic_crop, conf_cfg, "stochastic_crop", True)
    )
    args.max_protein_atoms = int(
        _arg_or_cfg(args.max_protein_atoms, conf_cfg, "max_protein_atoms", 2048)
    )
    args.max_train_pose_node_product = _optional_int(
        _arg_or_cfg(
            args.max_train_pose_node_product,
            conf_cfg,
            "max_train_pose_node_product",
            None,
        )
    )
    if args.max_train_pose_node_product is not None and args.max_train_pose_node_product < 1:
        raise ValueError("max_train_pose_node_product must be positive")
    args.large_train_graph_node_threshold = _optional_int(
        _arg_or_cfg(
            args.large_train_graph_node_threshold,
            conf_cfg,
            "large_train_graph_node_threshold",
            None,
        )
    )
    args.large_train_graph_max_poses = _optional_int(
        _arg_or_cfg(
            args.large_train_graph_max_poses,
            conf_cfg,
            "large_train_graph_max_poses",
            None,
        )
    )
    if (args.large_train_graph_node_threshold is None) != (
        args.large_train_graph_max_poses is None
    ):
        raise ValueError(
            "large_train_graph_node_threshold and large_train_graph_max_poses "
            "must be set together"
        )
    if (
        args.large_train_graph_node_threshold is not None
        and args.large_train_graph_node_threshold < 1
    ):
        raise ValueError("large_train_graph_node_threshold must be positive")
    if args.large_train_graph_max_poses is not None and args.large_train_graph_max_poses < 1:
        raise ValueError("large_train_graph_max_poses must be positive")
    args.max_train_poses_per_complex = _optional_int(
        _arg_or_cfg(
            args.max_train_poses_per_complex,
            conf_cfg,
            "max_train_poses_per_complex",
            None,
        )
    )
    args.max_val_poses_per_complex = _optional_int(
        _arg_or_cfg(
            args.max_val_poses_per_complex,
            conf_cfg,
            "max_val_poses_per_complex",
            None,
        )
    )
    args.pose_sample_strategy = str(
        _arg_or_cfg(
            args.pose_sample_strategy,
            conf_cfg,
            "pose_sample_strategy",
            "best_random",
        )
    )
    args.train_limit = _optional_int(_arg_or_cfg(args.train_limit, conf_cfg, "train_limit", None))
    args.val_limit = _optional_int(_arg_or_cfg(args.val_limit, conf_cfg, "val_limit", None))
    args.val_start = int(_arg_or_cfg(args.val_start, conf_cfg, "val_start", 0))
    args.eval_every = int(_arg_or_cfg(args.eval_every, conf_cfg, "eval_every", 1000))
    raw_eval_complexes = _arg_or_cfg(args.eval_complexes, conf_cfg, "eval_complexes", 256)
    args.eval_complexes = (
        None
        if raw_eval_complexes is None or int(raw_eval_complexes) <= 0
        else int(raw_eval_complexes)
    )
    args.eval_target = str(
        _arg_or_cfg(args.eval_target, conf_cfg, "eval_target", "pose_rmsd")
    )
    args.training_target = str(
        _arg_or_cfg(args.training_target, conf_cfg, "training_target", "pose_rmsd")
    )
    if not args.eval_target:
        raise ValueError("eval_target must be a non-empty shard field name")
    if args.training_target not in {"pose_rmsd", SYMMETRY_TARGET_KEY}:
        raise ValueError(
            f"training_target must be 'pose_rmsd' or {SYMMETRY_TARGET_KEY!r}"
        )
    cfg_bank_manifest = conf_cfg.get("bank_manifest", None)
    args.bank_manifest = args.bank_manifest or (
        Path(cfg_bank_manifest) if cfg_bank_manifest else None
    )
    args.bank_manifest_sha256 = _arg_or_cfg(
        args.bank_manifest_sha256,
        conf_cfg,
        "bank_manifest_sha256",
        None,
    )
    if args.bank_manifest_sha256 is not None and args.bank_manifest is None:
        raise ValueError("bank_manifest_sha256 requires bank_manifest")
    cfg_target_manifest = conf_cfg.get("train_pose_target_manifest", None)
    args.train_pose_target_manifest = args.train_pose_target_manifest or (
        Path(cfg_target_manifest) if cfg_target_manifest else None
    )
    args.train_pose_target_manifest_sha256 = _arg_or_cfg(
        args.train_pose_target_manifest_sha256,
        conf_cfg,
        "train_pose_target_manifest_sha256",
        None,
    )
    cfg_aux_bank_manifest = conf_cfg.get("train_aux_bank_manifest", None)
    args.train_aux_bank_manifest = args.train_aux_bank_manifest or (
        Path(cfg_aux_bank_manifest) if cfg_aux_bank_manifest else None
    )
    args.train_aux_bank_manifest_sha256 = _arg_or_cfg(
        args.train_aux_bank_manifest_sha256,
        conf_cfg,
        "train_aux_bank_manifest_sha256",
        None,
    )
    args.train_aux_pose_tag = _arg_or_cfg(
        args.train_aux_pose_tag,
        conf_cfg,
        "train_aux_pose_tag",
        None,
    )
    cfg_aux_target_manifest = conf_cfg.get("train_aux_pose_target_manifest", None)
    args.train_aux_pose_target_manifest = args.train_aux_pose_target_manifest or (
        Path(cfg_aux_target_manifest) if cfg_aux_target_manifest else None
    )
    args.train_aux_pose_target_manifest_sha256 = _arg_or_cfg(
        args.train_aux_pose_target_manifest_sha256,
        conf_cfg,
        "train_aux_pose_target_manifest_sha256",
        None,
    )
    args.smoke = bool(_arg_or_cfg(args.smoke, conf_cfg, "smoke", False))
    best_metric_name = str(conf_cfg.get("best_metric", "success_selected_lt2"))
    best_metric_mode = str(conf_cfg.get("best_metric_mode", "max")).lower()
    if best_metric_mode not in {"min", "max"}:
        raise ValueError(f"best_metric_mode must be 'min' or 'max', got {best_metric_mode!r}")
    args.save_every = int(_arg_or_cfg(args.save_every, conf_cfg, "save_every", 5000))
    if args.eval_every < 1 or args.save_every < 1:
        raise ValueError("eval_every and save_every must be positive")
    args.seed = int(_arg_or_cfg(args.seed, conf_cfg, "seed", 42))
    use_wandb = bool(_arg_or_cfg(args.use_wandb, conf_cfg, "use_wandb", True))
    wandb_project = str(conf_cfg.get("wandb_project", "eff-dock"))
    wandb_run_name = str(conf_cfg.get("wandb_run_name", args.run_name))

    args.total_steps = int(_arg_or_cfg(args.total_steps, train_cfg, "max_steps", 50000))
    args.lr = float(_arg_or_cfg(args.lr, train_cfg, "lr", 2e-4))
    args.muon_lr = float(_arg_or_cfg(args.muon_lr, train_cfg, "muon_lr", 0.02))
    args.weight_decay = float(_arg_or_cfg(args.weight_decay, train_cfg, "weight_decay", 0.01))
    args.use_muon = bool(_arg_or_cfg(args.use_muon, train_cfg, "use_muon", True))
    args.scheduler_type = str(
        _arg_or_cfg(args.scheduler_type, train_cfg, "scheduler_type", "warmup_stable_cosine")
    )
    args.warmup_ratio = float(_arg_or_cfg(args.warmup_ratio, train_cfg, "warmup_ratio", 0.02))
    args.cooldown_ratio = float(_arg_or_cfg(args.cooldown_ratio, train_cfg, "cooldown_ratio", 0.5))
    args.min_lr_ratio = float(_arg_or_cfg(args.min_lr_ratio, train_cfg, "min_lr_ratio", 0.05))
    args.max_grad_norm = float(_arg_or_cfg(args.max_grad_norm, train_cfg, "max_grad_norm", 1.0))
    if args.protein_crop_mode not in {"pose_residue", "ligand_residue", "center"}:
        raise ValueError(f"unknown protein_crop_mode: {args.protein_crop_mode}")

    bank_provenance: dict[str, Any] | None = None
    bank_shard_paths: dict[str, dict[str, str]] | None = None
    bank_system_ids: dict[str, dict[str, str]] | None = None
    bank_records: dict[str, dict[str, dict[str, Any]]] | None = None
    if args.bank_manifest is not None:
        bank_provenance = validate_bank_manifest(
            args.bank_manifest,
            expected_sha256=args.bank_manifest_sha256,
            split_file=args.split_file,
            pose_tag=args.pose_tag,
            allow_smoke=args.smoke,
        )
        bank_shard_paths = bank_provenance.pop("_shard_paths")
        bank_system_ids = bank_provenance.pop("_system_ids")
        bank_records = bank_provenance.pop("_records")
        # Persist the observed digest even when the optional expected digest was
        # omitted on the command line.
        args.bank_manifest_sha256 = bank_provenance["sha256"]

    target_provenance: dict[str, Any] | None = None
    train_pose_targets: dict[str, dict[str, Any]] | None = None
    if args.training_target == SYMMETRY_TARGET_KEY:
        if (
            args.train_pose_target_manifest is None
            or args.train_pose_target_manifest_sha256 is None
            or bank_provenance is None
            or bank_records is None
        ):
            raise ValueError(
                "symmetry training_target requires exact bank and train pose-target manifests"
            )
        with args.split_file.open(encoding="utf-8") as handle:
            ordered_train_ids = list(json.load(handle)["train"])
        target_provenance = validate_pose_target_manifest(
            args.train_pose_target_manifest,
            expected_sha256=args.train_pose_target_manifest_sha256,
            bank_manifest_sha256=bank_provenance["sha256"],
            bank_records=bank_records["train"],
            ordered_train_ids=ordered_train_ids,
        )
        train_pose_targets = target_provenance.pop("_records")
        args.train_pose_target_manifest_sha256 = target_provenance["sha256"]
    elif args.train_pose_target_manifest is not None:
        raise ValueError("train_pose_target_manifest is only valid for symmetry training_target")

    aux_fields = (
        args.train_aux_bank_manifest,
        args.train_aux_bank_manifest_sha256,
        args.train_aux_pose_tag,
        args.train_aux_pose_target_manifest,
        args.train_aux_pose_target_manifest_sha256,
    )
    aux_requested = any(value is not None for value in aux_fields)
    if aux_requested and not all(value is not None for value in aux_fields):
        raise ValueError(
            "paired training requires the auxiliary bank path/SHA, pose tag, and "
            "symmetry-target path/SHA"
        )
    aux_bank_provenance: dict[str, Any] | None = None
    aux_bank_shard_paths: dict[str, dict[str, str]] | None = None
    aux_bank_system_ids: dict[str, dict[str, str]] | None = None
    aux_bank_records: dict[str, dict[str, dict[str, Any]]] | None = None
    aux_target_provenance: dict[str, Any] | None = None
    aux_train_pose_targets: dict[str, dict[str, Any]] | None = None
    if aux_requested:
        if bank_provenance is None or bank_system_ids is None:
            raise ValueError("paired training requires a sealed primary bank")
        if args.training_target != SYMMETRY_TARGET_KEY:
            raise ValueError("paired training currently requires symmetry-aware RMSD targets")
        if args.max_train_poses_per_complex is None or args.max_train_poses_per_complex < 2:
            raise ValueError("paired training requires at least two train poses per complex")
        if args.max_train_poses_per_complex % 2:
            raise ValueError("paired training requires an even max_train_poses_per_complex")
        aux_bank_provenance = validate_bank_manifest(
            args.train_aux_bank_manifest,
            expected_sha256=args.train_aux_bank_manifest_sha256,
            split_file=args.split_file,
            pose_tag=str(args.train_aux_pose_tag),
            allow_smoke=args.smoke,
        )
        aux_bank_shard_paths = aux_bank_provenance.pop("_shard_paths")
        aux_bank_system_ids = aux_bank_provenance.pop("_system_ids")
        aux_bank_records = aux_bank_provenance.pop("_records")
        if aux_bank_system_ids != bank_system_ids:
            raise ValueError("paired pose banks have different system-ID inventories")
        refinements = {
            str(bank_provenance["settings"]["refine"]),
            str(aux_bank_provenance["settings"]["refine"]),
        }
        if refinements != {"none", "guidance_unified_step100"}:
            raise ValueError("paired training requires exactly one raw and one refined S50 bank")
        with args.split_file.open(encoding="utf-8") as handle:
            ordered_train_ids = list(json.load(handle)["train"])
        aux_target_provenance = validate_pose_target_manifest(
            args.train_aux_pose_target_manifest,
            expected_sha256=str(args.train_aux_pose_target_manifest_sha256),
            bank_manifest_sha256=aux_bank_provenance["sha256"],
            bank_records=aux_bank_records["train"],
            ordered_train_ids=ordered_train_ids,
        )
        aux_train_pose_targets = aux_target_provenance.pop("_records")
        args.train_aux_bank_manifest_sha256 = aux_bank_provenance["sha256"]
        args.train_aux_pose_target_manifest_sha256 = aux_target_provenance["sha256"]

    init_provenance: dict[str, Any] | None = None
    if args.init_checkpoint is not None:
        if not args.init_checkpoint.is_file():
            raise FileNotFoundError(f"initialization checkpoint does not exist: {args.init_checkpoint}")
        init_sha = _sha256_file(args.init_checkpoint)
        if args.init_checkpoint_sha256 is not None:
            expected_init_sha = _require_sha256(
                args.init_checkpoint_sha256,
                label="initialization checkpoint expected SHA-256",
            )
            if init_sha != expected_init_sha:
                raise ValueError(
                    f"initialization checkpoint SHA-256 mismatch: expected "
                    f"{expected_init_sha}, got {init_sha}"
                )
        args.init_checkpoint_sha256 = init_sha
        init_provenance = {
            "path": str(args.init_checkpoint.resolve()),
            "sha256": init_sha,
            "source_step": None,
            "mode": "weights_only_fresh_u0",
        }

    rank, local_rank, world_size = setup_ddp()
    is_main = rank == 0

    random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    paired_training = aux_bank_provenance is not None
    train_component_max_poses = args.max_train_poses_per_complex
    train_component_node_product = args.max_train_pose_node_product
    train_component_large_graph_max = args.large_train_graph_max_poses
    if paired_training:
        train_component_max_poses = args.max_train_poses_per_complex // 2
        if train_component_node_product is not None:
            train_component_node_product = max(1, train_component_node_product // 2)
        if train_component_large_graph_max is not None:
            train_component_large_graph_max = max(1, train_component_large_graph_max // 2)

    primary_train_ds = LigandPoseConfidenceDataset(
        split_file=args.split_file,
        split="train",
        processed_dir=args.processed_dir,
        pose_tag=args.pose_tag,
        protein_crop_mode=args.protein_crop_mode,
        protein_contact_cutoff=args.protein_contact_cutoff,
        protein_crop_cutoff=args.protein_crop_cutoff,
        protein_crop_cutoff_min=args.protein_crop_cutoff_min,
        protein_crop_cutoff_max=args.protein_crop_cutoff_max,
        protein_crop_jitter_sigma=args.protein_crop_jitter_sigma,
        protein_crop_jitter_max=args.protein_crop_jitter_max,
        stochastic_crop=args.stochastic_crop,
        max_protein_atoms=args.max_protein_atoms,
        max_poses_per_complex=train_component_max_poses,
        max_pose_node_product=train_component_node_product,
        large_graph_node_threshold=args.large_train_graph_node_threshold,
        large_graph_max_poses=train_component_large_graph_max,
        pose_sample_strategy=args.pose_sample_strategy,
        pose_target_key=args.training_target,
        external_pose_targets=train_pose_targets,
        shard_paths=(bank_shard_paths or {}).get("train") if bank_shard_paths else None,
        system_ids=(bank_system_ids or {}).get("train") if bank_system_ids else None,
        limit=args.train_limit,
    )
    train_ds: LigandPoseConfidenceDataset | PairedLigandPoseConfidenceDataset
    if paired_training:
        auxiliary_train_ds = LigandPoseConfidenceDataset(
            split_file=args.split_file,
            split="train",
            processed_dir=args.processed_dir,
            pose_tag=str(args.train_aux_pose_tag),
            protein_crop_mode=args.protein_crop_mode,
            protein_contact_cutoff=args.protein_contact_cutoff,
            protein_crop_cutoff=args.protein_crop_cutoff,
            protein_crop_cutoff_min=args.protein_crop_cutoff_min,
            protein_crop_cutoff_max=args.protein_crop_cutoff_max,
            protein_crop_jitter_sigma=args.protein_crop_jitter_sigma,
            protein_crop_jitter_max=args.protein_crop_jitter_max,
            stochastic_crop=args.stochastic_crop,
            max_protein_atoms=args.max_protein_atoms,
            max_poses_per_complex=train_component_max_poses,
            max_pose_node_product=train_component_node_product,
            large_graph_node_threshold=args.large_train_graph_node_threshold,
            large_graph_max_poses=train_component_large_graph_max,
            pose_sample_strategy=args.pose_sample_strategy,
            pose_target_key=args.training_target,
            external_pose_targets=aux_train_pose_targets,
            shard_paths=(aux_bank_shard_paths or {}).get("train"),
            system_ids=(aux_bank_system_ids or {}).get("train"),
            limit=args.train_limit,
        )
        train_ds = PairedLigandPoseConfidenceDataset(
            primary_train_ds,
            auxiliary_train_ds,
        )
    else:
        train_ds = primary_train_ds
    val_ds = LigandPoseConfidenceDataset(
        split_file=args.split_file,
        split="val",
        processed_dir=args.processed_dir,
        pose_tag=args.pose_tag,
        protein_crop_mode=args.protein_crop_mode,
        protein_contact_cutoff=args.protein_contact_cutoff,
        protein_crop_cutoff=args.protein_crop_cutoff,
        protein_crop_cutoff_min=args.protein_crop_cutoff_min,
        protein_crop_cutoff_max=args.protein_crop_cutoff_max,
        protein_crop_jitter_sigma=args.protein_crop_jitter_sigma,
        protein_crop_jitter_max=args.protein_crop_jitter_max,
        stochastic_crop=False,
        max_protein_atoms=args.max_protein_atoms,
        max_poses_per_complex=args.max_val_poses_per_complex,
        pose_sample_strategy="best_random",
        pose_target_key=args.training_target,
        eval_target_key=args.eval_target,
        shard_paths=(bank_shard_paths or {}).get("val") if bank_shard_paths else None,
        system_ids=(bank_system_ids or {}).get("val") if bank_system_ids else None,
        limit=args.val_limit,
        start=args.val_start,
    )
    train_sampler = (
        _ResumableDistributedSampler(
            train_ds,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=args.seed,
            drop_last=False,
        )
        if world_size > 1 or bank_provenance is not None
        else None
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_complexes,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.num_workers,
        collate_fn=collate_complexes,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_complexes,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_complexes,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    raw_model = DockingGraphPoseConfidence(
        **model_cfg,
        n_rbf=args.n_rbf,
        contact_cutoff=args.contact_cutoff,
        hidden=args.hidden,
        num_layers=args.num_layers,
        sh_lmax=args.sh_lmax,
        cond_dim=args.cond_dim,
        dropout=args.dropout,
        pose_readout=args.pose_readout,
    ).to(device)
    resume_step = 0
    if args.resume is not None:
        ckpt = _load_checkpoint(args.resume, device)
        resumed_bank = ckpt.get("bank_provenance")
        if resumed_bank is not None:
            if bank_provenance is None:
                raise ValueError(
                    "resuming a sealed-bank checkpoint requires --bank_manifest revalidation"
                )
            if resumed_bank.get("sha256") != bank_provenance.get("sha256"):
                raise ValueError("resume checkpoint bank manifest SHA-256 does not match")
        resumed_target = ckpt.get("pose_target_provenance")
        if resumed_target != target_provenance:
            raise ValueError("resume checkpoint pose-target provenance does not match")
        if ckpt.get("train_aux_bank_provenance") != aux_bank_provenance:
            raise ValueError("resume checkpoint auxiliary-bank provenance does not match")
        if ckpt.get("train_aux_pose_target_provenance") != aux_target_provenance:
            raise ValueError(
                "resume checkpoint auxiliary pose-target provenance does not match"
            )
        resumed_global_batch = ckpt.get("effective_global_batch_complexes")
        current_global_batch = (
            args.batch_complexes * args.gradient_accumulation_steps * world_size
        )
        if resumed_global_batch is not None and int(resumed_global_batch) != current_global_batch:
            raise ValueError(
                "resume effective global batch mismatch: "
                f"checkpoint={resumed_global_batch}, current={current_global_batch}"
            )
        resumed_init = ckpt.get("initialization_provenance")
        if resumed_init is not None:
            init_provenance = dict(resumed_init)
        load_portable_model_state(raw_model, ckpt["state_dict"])
        resume_step = int(ckpt.get("step", 0))
    elif args.init_checkpoint is not None:
        init_ckpt = _load_checkpoint(args.init_checkpoint, device)
        load_portable_model_state(raw_model, init_ckpt["state_dict"])
        if init_provenance is None:
            raise RuntimeError("missing initialization checkpoint provenance")
        init_provenance["source_step"] = int(init_ckpt.get("step", -1))
    if args.finetune_steps is not None:
        if args.resume is None:
            raise ValueError("--finetune-steps requires --resume")
        if args.finetune_steps < 1:
            raise ValueError("--finetune-steps must be at least 1")
        args.total_steps = resume_step + int(args.finetune_steps)
    if world_size > 1:
        ddp_device = torch.cuda.current_device()
        model = DDP(raw_model, device_ids=[ddp_device], output_device=ddp_device)
    else:
        model = raw_model
    optimizers = configure_optimizers(
        raw_model,
        lr=args.lr,
        muon_lr=args.muon_lr,
        weight_decay=args.weight_decay,
        use_muon=args.use_muon,
    )
    if not optimizers:
        raise RuntimeError("No trainable parameters found for optimizer configuration.")
    scheduler_total_steps = (
        int(args.finetune_steps)
        if args.resume is not None and args.reset_optimizer and args.finetune_steps is not None
        else args.total_steps
    )
    if args.scheduler_type == "warmup_stable_cosine":
        schedulers = [
            get_warmup_stable_cosine_scheduler(
                opt,
                scheduler_total_steps,
                warmup_ratio=args.warmup_ratio,
                cooldown_ratio=args.cooldown_ratio,
                min_lr_ratio=args.min_lr_ratio,
            )
            for opt in optimizers
        ]
    elif args.scheduler_type == "none":
        schedulers = []
    else:
        raise ValueError(f"unknown scheduler_type: {args.scheduler_type}")
    if args.resume is not None and not args.reset_optimizer:
        _restore_training_states(
            ckpt,
            optimizers=optimizers,
            schedulers=schedulers,
        )

    wandb_run = None
    if is_main and use_wandb:
        try:
            import wandb

            wandb_init = {
                "project": wandb_project,
                "name": wandb_run_name,
                "config": _wandb_safe_config(args),
                "resume": "allow",
            }
            if args.resume is not None and "ckpt" in locals() and ckpt.get("wandb_run_id"):
                wandb_init["id"] = ckpt["wandb_run_id"]
            wandb_run = wandb.init(**wandb_init)
            wandb.define_metric("global_step", hidden=True)
            wandb.define_metric("step/*", step_metric="global_step")
            wandb.define_metric("val/*", step_metric="global_step")
            wandb.define_metric("meta/*", step_metric="global_step")
        except Exception as exc:
            print(f"WARNING: wandb init failed: {exc!r}", flush=True)
            use_wandb = False

    if is_main:
        print(
            f"device={device} rank={rank} local_rank={local_rank} world_size={world_size} "
            f"train={len(train_ds)} val={len(val_ds)}"
        )
        print(
            f"model_params={sum(p.numel() for p in raw_model.parameters() if p.requires_grad):,}",
            flush=True,
        )
        print(
            f"optimizers={','.join(type(opt).__name__ for opt in optimizers)} "
            f"lr={args.lr:g} muon_lr={args.muon_lr:g} weight_decay={args.weight_decay:g} "
            f"use_muon={args.use_muon}",
            flush=True,
        )
        print(
            f"scheduler={args.scheduler_type} total_steps={args.total_steps} "
            f"scheduler_steps={scheduler_total_steps} "
            f"warmup_ratio={args.warmup_ratio:g} cooldown_ratio={args.cooldown_ratio:g} "
            f"min_lr_ratio={args.min_lr_ratio:g}",
            flush=True,
        )
        print(
            f"batch_complexes_per_rank={args.batch_complexes} "
            f"gradient_accumulation_steps={args.gradient_accumulation_steps} "
            f"effective_global_batch_complexes="
            f"{args.batch_complexes * args.gradient_accumulation_steps * world_size}",
            flush=True,
        )
        print(
            "loss=" + " ".join(f"{key}={value:g}" for key, value in loss_kwargs.items()),
            flush=True,
        )
        print(
            f"eval_complexes={'all' if args.eval_complexes is None else args.eval_complexes} "
            f"max_protein_atoms={args.max_protein_atoms} "
            f"max_train_poses={args.max_train_poses_per_complex} "
            f"max_train_pose_node_product={args.max_train_pose_node_product or 'off'} "
            f"large_train_graph_node_threshold="
            f"{args.large_train_graph_node_threshold or 'off'} "
            f"large_train_graph_max_poses={args.large_train_graph_max_poses or 'off'} "
            f"max_val_poses={args.max_val_poses_per_complex} "
            f"pose_readout={args.pose_readout}",
            flush=True,
        )
        print(
            f"training_target={args.training_target} eval_target={args.eval_target}",
            flush=True,
        )
        if paired_training:
            print(
                "train_pose_mix=paired_equal "
                f"poses_per_bank={train_component_max_poses} "
                "crystal_anchor_per_complex=1 total_max_poses="
                f"{2 * train_component_max_poses + 1}",
                flush=True,
            )
        if bank_provenance is not None:
            print(
                f"bank_manifest={bank_provenance['path']} "
                f"sha256={bank_provenance['sha256']}",
                flush=True,
            )
        if target_provenance is not None:
            print(
                f"train_pose_target_manifest={target_provenance['path']} "
                f"sha256={target_provenance['sha256']}",
                flush=True,
            )
        if aux_bank_provenance is not None:
            print(
                f"train_aux_bank_manifest={aux_bank_provenance['path']} "
                f"sha256={aux_bank_provenance['sha256']}",
                flush=True,
            )
        if aux_target_provenance is not None:
            print(
                f"train_aux_pose_target_manifest={aux_target_provenance['path']} "
                f"sha256={aux_target_provenance['sha256']}",
                flush=True,
            )
        if args.resume is not None:
            print(
                f"resumed from {args.resume} at step={resume_step} "
                f"reset_optimizer={args.reset_optimizer}",
                flush=True,
            )
        if init_provenance is not None:
            print(
                f"initialized U0 weights from {init_provenance['path']} "
                f"source_step={init_provenance['source_step']} "
                f"sha256={init_provenance['sha256']}",
                flush=True,
            )

    try:
        best_score = -float("inf") if best_metric_mode == "max" else float("inf")
        best_path = args.out_dir / "best.pt"
        if best_path.exists():
            try:
                best_ckpt = _load_checkpoint(best_path, "cpu")
                best_metrics = best_ckpt.get("metrics") or {}
                if best_metric_name in best_metrics:
                    best_score = float(best_metrics[best_metric_name])
            except Exception as exc:
                if is_main:
                    print(f"warning: failed to read existing best.pt metrics: {exc!r}", flush=True)
        step = resume_step
        last_metrics: dict[str, float] | None = None
        evaluation_ledgers: list[dict[str, Any]] = []

        def checkpoint_payload(
            metrics: dict[str, float] | None = None,
            eval_ledger: dict[str, str] | None = None,
        ) -> dict[str, Any]:
            payload: dict[str, Any] = {
                "step": step,
                "update": step,
                "state_dict": raw_model.state_dict(),
                "model_cfg": model_cfg,
                "args": vars(args),
                "model_type": "docking_graph_pose_confidence",
                "wandb_run_id": getattr(wandb_run, "id", None)
                if wandb_run is not None
                else None,
                "optimizer_state_dicts": [opt.state_dict() for opt in optimizers],
                "scheduler_state_dicts": [sched.state_dict() for sched in schedulers],
                "bank_provenance": bank_provenance,
                "pose_target_provenance": target_provenance,
                "train_aux_bank_provenance": aux_bank_provenance,
                "train_aux_pose_target_provenance": aux_target_provenance,
                "initialization_provenance": init_provenance,
                "effective_global_batch_complexes": (
                    args.batch_complexes
                    * args.gradient_accumulation_steps
                    * world_size
                ),
            }
            if metrics is not None:
                payload.update(
                    {
                        "metrics": metrics,
                        "best_metric": best_metric_name,
                        "best_metric_mode": best_metric_mode,
                        "evaluation_ledger": eval_ledger,
                    }
                )
            return payload

        def evaluate_and_write_ledger() -> tuple[dict[str, float], dict[str, str]]:
            records: list[dict[str, Any]] = []
            metrics = evaluate(
                raw_model,
                val_loader,
                device,
                max_complexes=args.eval_complexes,
                eval_target_key=args.eval_target,
                eval_records=records,
            )
            ledger = _write_eval_ledger(
                args.out_dir / f"eval_u{step:06d}.json",
                step=step,
                eval_target_key=args.eval_target,
                records=records,
                bank_provenance=bank_provenance,
            )
            # Full validation runs only on rank 0. Without an explicit cache
            # release, its 100-pose evaluation allocations can occupy nearly
            # the entire GPU while the other ranks enter the next DDP step.
            _release_eval_cuda_cache(device)
            ledger_with_step: dict[str, Any] = {"step": step, **ledger}
            if ledger_with_step not in evaluation_ledgers:
                evaluation_ledgers.append(ledger_with_step)
            return metrics, ledger

        def save_best_if_improved(
            metrics: dict[str, float],
            ledger: dict[str, str],
            *,
            label: str,
        ) -> None:
            nonlocal best_score
            metric_value = float(metrics[best_metric_name])
            improved = (
                metric_value > best_score
                if best_metric_mode == "max"
                else metric_value < best_score
            )
            if improved:
                best_score = metric_value
                atomic_torch_save(
                    checkpoint_payload(metrics, ledger),
                    best_path,
                )
                print(
                    f"  saved {label} best.pt {best_metric_name}={best_score:.3f} "
                    f"mode={best_metric_mode}",
                    flush=True,
                )

        resume_state = "not_scheduled"
        resume_ledger: dict[str, str] | None = None
        resume_metrics: dict[str, float] | None = None
        if args.resume is not None:
            resume_state, resume_ledger, resume_metrics = _resume_eval_state(
                ckpt,
                out_dir=args.out_dir,
                step=step,
                eval_every=args.eval_every,
                eval_target_key=args.eval_target,
                bank_provenance=bank_provenance,
            )
        resume_boundary_handled = resume_state != "not_scheduled"
        if args.resume is not None and world_size > 1:
            # All ranks classify the same pre-recovery filesystem snapshot.
            dist.barrier()
        if args.resume is not None and is_main:
            if resume_state == "evaluate_missing":
                last_metrics, resume_ledger = evaluate_and_write_ledger()
                print(fmt_metrics(f"[Val U{step} resume recovery]", last_metrics), flush=True)
                save_best_if_improved(
                    last_metrics,
                    resume_ledger,
                    label="resume-recovered",
                )
                atomic_torch_save(
                    checkpoint_payload(last_metrics, resume_ledger),
                    args.out_dir / "latest.pt",
                )
                print(f"  committed recovered scheduled evaluation U{step}", flush=True)
            elif resume_state in {"commit_existing", "committed"}:
                if resume_ledger is None or resume_metrics is None:
                    raise RuntimeError("resume evaluation state lost its sealed ledger metrics")
                last_metrics = resume_metrics
                evaluation_ledgers.append({"step": step, **resume_ledger})
                if resume_state == "commit_existing":
                    save_best_if_improved(
                        last_metrics,
                        resume_ledger,
                        label="ledger-recovered",
                    )
                    atomic_torch_save(
                        checkpoint_payload(last_metrics, resume_ledger),
                        args.out_dir / "latest.pt",
                    )
                    print(
                        f"  committed existing scheduled evaluation U{step} without re-evaluation",
                        flush=True,
                    )
        if args.resume is not None and world_size > 1:
            dist.barrier()

        if args.eval_on_start and not resume_boundary_handled:
            if is_main:
                last_metrics, initial_ledger = evaluate_and_write_ledger()
                print(fmt_metrics(f"[Val U{step} initial]", last_metrics), flush=True)
                save_best_if_improved(
                    last_metrics,
                    initial_ledger,
                    label="initial",
                )
            if world_size > 1:
                dist.barrier()
        consumed_train_batches = step * args.gradient_accumulation_steps
        epoch, train_batch_offset = divmod(
            consumed_train_batches,
            max(1, len(train_loader)),
        )
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
            train_sampler.set_start_index(train_batch_offset * args.batch_complexes)
        train_iter = iter(train_loader)
        if is_main and train_batch_offset:
            print(
                f"resumed data stream epoch={epoch} batch_offset={train_batch_offset}",
                flush=True,
            )
        while step < args.total_steps:
            model.train()
            for opt in optimizers:
                opt.zero_grad(set_to_none=True)
            logs: dict[str, float] = {}
            for accumulation_index in range(args.gradient_accumulation_steps):
                try:
                    batch = next(train_iter)
                except StopIteration:
                    epoch += 1
                    if train_sampler is not None:
                        train_sampler.set_epoch(epoch)
                    train_iter = iter(train_loader)
                    batch = next(train_iter)

                denominator = len(batch) * args.gradient_accumulation_steps
                for raw_index, raw in enumerate(batch):
                    should_sync = (
                        accumulation_index == args.gradient_accumulation_steps - 1
                        and raw_index == len(batch) - 1
                    )
                    sync_context = (
                        nullcontext()
                        if world_size == 1 or should_sync
                        else model.no_sync()
                    )
                    with sync_context:
                        item = to_device(raw, device)
                        out = model(item)
                        losses = pose_confidence_loss(out, item, **loss_kwargs)
                        loss = losses["loss"] / denominator
                        loss.backward()
                    for key, value in losses.items():
                        logs[key] = logs.get(key, 0.0) + float(value.detach().item()) / denominator
            grad_norm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), args.max_grad_norm)
            for opt in optimizers:
                opt.step()
            for sched in schedulers:
                sched.step()
            step += 1

            if is_main and (step == 1 or step % 20 == 0):
                lr_vals = [opt.param_groups[0]["lr"] for opt in optimizers]
                lr_text = " ".join(f"lr{i}={lr:.3e}" for i, lr in enumerate(lr_vals))
                print(
                    f"[U{step}] loss={logs['loss']:.4f} pose={logs['loss_pose']:.4f} "
                    f"rank={logs['loss_rank']:.4f} listwise={logs['loss_listwise']:.4f} "
                    f"succ_listwise={logs['loss_success_listwise']:.4f} "
                    f"setwise={logs['loss_setwise_success']:.4f} "
                    f"pairwise={logs['loss_pairwise_success']:.4f} "
                    f"hard_pair={logs['loss_hard_pair']:.4f} "
                    f"atom={logs['loss_atom']:.4f} "
                    f"grad={float(grad_norm):.3f} {lr_text}",
                    flush=True,
                )
                if use_wandb:
                    import wandb

                    log_dict = {
                        "global_step": step,
                        "step/loss": logs["loss"],
                        "step/loss_pose": logs["loss_pose"],
                        "step/loss_pose_bce": logs["loss_pose_bce"],
                        "step/loss_rank": logs["loss_rank"],
                        "step/loss_listwise": logs["loss_listwise"],
                        "step/loss_success_listwise": logs["loss_success_listwise"],
                        "step/loss_setwise_success": logs["loss_setwise_success"],
                        "step/loss_pairwise_success": logs["loss_pairwise_success"],
                        "step/loss_hard_pair": logs["loss_hard_pair"],
                        "step/loss_atom": logs["loss_atom"],
                        "step/loss_atom_bce": logs["loss_atom_bce"],
                        "step/grad_norm": float(grad_norm),
                    }
                    for i, lr in enumerate(lr_vals):
                        log_dict[f"meta/lr_{i}"] = lr
                    wandb.log(log_dict, step=step)

            if is_main and (step % args.save_every == 0 or step == args.total_steps):
                atomic_torch_save(checkpoint_payload(), args.out_dir / "latest.pt")
            if world_size > 1 and (step % args.save_every == 0 or step == args.total_steps):
                dist.barrier()

            if step % args.eval_every == 0:
                if is_main:
                    metrics, eval_ledger = evaluate_and_write_ledger()
                    last_metrics = metrics
                    print(fmt_metrics(f"[Val U{step}]", metrics), flush=True)
                    if use_wandb:
                        import wandb

                        wandb.log(
                            {
                                "global_step": step,
                                **{f"val/{key}": value for key, value in metrics.items()},
                            },
                            step=step,
                        )
                    save_best_if_improved(
                        metrics,
                        eval_ledger,
                        label="scheduled",
                    )
                    if step % args.save_every == 0 or step == args.total_steps:
                        # latest.pt is first written above as the recovery anchor.
                        # This second atomic write commits the ledger and proves
                        # best-checkpoint selection ran for this boundary.
                        atomic_torch_save(
                            checkpoint_payload(metrics, eval_ledger),
                            args.out_dir / "latest.pt",
                        )
                if world_size > 1:
                    dist.barrier()

        if is_main:
            best_metrics: dict[str, float] = {}
            if best_path.exists():
                best_metrics = dict(_load_checkpoint(best_path, "cpu").get("metrics") or {})
            metrics_summary = {
                "run_name": args.run_name,
                "resume": str(args.resume) if args.resume is not None else None,
                "resume_step": resume_step,
                "final_step": step,
                "final_update": step,
                "best_metric": best_metric_name,
                "best_score": best_score,
                "best_checkpoint": str(best_path),
                "best_metrics": best_metrics,
                "last_metrics": last_metrics or {},
                "bank_provenance": bank_provenance,
                "pose_target_provenance": target_provenance,
                "train_aux_bank_provenance": aux_bank_provenance,
                "train_aux_pose_target_provenance": aux_target_provenance,
                "train_pose_mix": (
                    {
                        "mode": "paired_equal",
                        "poses_per_bank": train_component_max_poses,
                        "crystal_anchor_per_complex": 1,
                    }
                    if paired_training
                    else None
                ),
                "training_target": args.training_target,
                "initialization_provenance": init_provenance,
                "eval_target": args.eval_target,
                "effective_global_batch_complexes": (
                    args.batch_complexes
                    * args.gradient_accumulation_steps
                    * world_size
                ),
                "evaluation_ledgers": evaluation_ledgers,
            }
            (args.out_dir / "metrics.json").write_text(
                json.dumps(metrics_summary, indent=2) + "\n"
            )
            print("done")
    finally:
        if is_main and use_wandb:
            try:
                import wandb

                wandb.finish()
            except Exception:
                pass
        cleanup_ddp(world_size)


if __name__ == "__main__":
    main()
