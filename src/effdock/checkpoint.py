"""Safe, backend-portable checkpoint helpers for EFF-Dock."""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from typing import Any

import torch
from torch import nn


def atomic_torch_save(value: Any, path: str | Path) -> Path:
    """Durably replace a Torch artifact without exposing a partial checkpoint."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(value, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def load_checkpoint_file(path: str | Path) -> dict[str, Any]:
    """Load a tensor/basic-container checkpoint on CPU with the safe unpickler."""
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"checkpoint must contain a mapping, got {type(checkpoint).__name__}")
    return checkpoint


def is_runtime_graph_buffer(key: str) -> bool:
    """Return whether a state key is a derived cuEquivariance graph constant."""
    return ".graph.c" in f".{key}"


def load_portable_model_state(model: nn.Module, state: dict[str, Any]) -> None:
    """Load learned state while tolerating only backend-derived graph buffers."""
    incompatible = model.load_state_dict(state, strict=False)
    bad_missing = [key for key in incompatible.missing_keys if not is_runtime_graph_buffer(key)]
    bad_unexpected = [
        key for key in incompatible.unexpected_keys if not is_runtime_graph_buffer(key)
    ]
    if bad_missing or bad_unexpected:
        raise RuntimeError(
            "checkpoint is incompatible with the configured EFF-Dock model: "
            f"missing={bad_missing}, unexpected={bad_unexpected}"
        )


def extract_ema_model_state(checkpoint: dict[str, Any]) -> dict[str, Any]:
    """Return an ``EFFDock.state_dict`` promoted from an AveragedModel EMA state."""
    ema_state = checkpoint.get("ema_state_dict")
    if not isinstance(ema_state, dict):
        raise RuntimeError("checkpoint does not contain an EMA state mapping")

    unexpected = [
        key
        for key in ema_state
        if key != "n_averaged" and (not isinstance(key, str) or not key.startswith("module."))
    ]
    if unexpected:
        raise RuntimeError(f"EMA state has unexpected non-module keys: {unexpected}")

    promoted = {
        key.removeprefix("module."): value
        for key, value in ema_state.items()
        if key != "n_averaged"
    }
    if not promoted:
        raise RuntimeError("EMA state contains no model tensors")

    raw_state = checkpoint.get("model_state_dict")
    if isinstance(raw_state, dict) and set(promoted) != set(raw_state):
        missing = sorted(set(raw_state) - set(promoted))
        unexpected_promoted = sorted(set(promoted) - set(raw_state))
        raise RuntimeError(
            "EMA state is incompatible with the raw model state: "
            f"missing={missing}, unexpected={unexpected_promoted}"
        )
    return promoted


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ema_n_averaged(checkpoint: dict[str, Any]) -> int:
    ema_state = checkpoint.get("ema_state_dict")
    if not isinstance(ema_state, dict) or "n_averaged" not in ema_state:
        raise RuntimeError("EMA state is missing n_averaged")
    value = ema_state["n_averaged"]
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise RuntimeError("EMA n_averaged must be scalar")
        return int(value.item())
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    raise RuntimeError("EMA n_averaged must be an integer scalar")


def export_ema_inference_checkpoint(
    source_path: str | Path,
    output_path: str | Path,
) -> Path:
    """Atomically export an inference-only checkpoint whose canonical weights are EMA.

    The source is never modified. The output deliberately omits optimizer,
    scheduler, and RNG state so it cannot silently act as a resume checkpoint.
    """
    source = Path(source_path)
    output = Path(output_path)
    if not source.is_file():
        raise FileNotFoundError(f"source checkpoint not found: {source}")
    if source.resolve() == output.resolve():
        raise ValueError("EMA export output must differ from the source checkpoint")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite EMA export: {output}")

    source_sha256 = _file_sha256(source)
    checkpoint = load_checkpoint_file(source)
    if _file_sha256(source) != source_sha256:
        raise RuntimeError("source checkpoint changed while it was being read")

    promoted = extract_ema_model_state(checkpoint)
    n_averaged = _ema_n_averaged(checkpoint)
    exported: dict[str, Any] = {
        "format_version": checkpoint.get("format_version", 1),
        "artifact_type": "effdock_ema_inference_checkpoint",
        "inference_only": True,
        "weight_source": "ema",
        "source_checkpoint_sha256": source_sha256,
        "source_checkpoint_step": checkpoint.get("step"),
        "ema_n_averaged": n_averaged,
        "step": checkpoint.get("step"),
        "model_state_dict": promoted,
        "ema_state_dict": checkpoint["ema_state_dict"],
    }
    for key in ("epoch", "data_pass_epoch", "metrics", "config"):
        if key in checkpoint:
            exported[key] = checkpoint[key]
    if "best_rmsd" in checkpoint:
        exported["source_best_rmsd"] = checkpoint["best_rmsd"]

    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(exported, handle)
            handle.flush()
            os.fsync(handle.fileno())
        # Hard-link publication is atomic and refuses to replace an existing output.
        os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output
