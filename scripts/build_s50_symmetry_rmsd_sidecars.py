#!/usr/bin/env python3
"""Materialize symmetry-aware RMSD labels for a sealed S50 pose bank.

The pose bank is immutable.  This tool reads the saved coordinates, recomputes
RDKit ``CalcRMS`` without alignment against the frozen crystal reference, and
writes small sharded label sidecars.  It never regenerates poses or hidden
features and never mutates the source bank.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import torch

from scripts.prepare_s50_confidence_training_bank import (
    NUM_SAMPLES,
    _recompute_validation_symmetry_rmsd,
    _tensor_sha256,
)

SCHEMA_VERSION = "EFFDOCK_S50_SYMMETRY_RMSD_SIDECAR_V1"
MANIFEST_SCHEMA_VERSION = "EFFDOCK_S50_SYMMETRY_RMSD_MANIFEST_V1"
METHOD = "rdkit_calc_rms_symmetry_no_align"
DEFAULT_POSE_TAG = "s50_n100_s10_sig2_latep3_pc10_rdkitseed0"


class SidecarContractError(RuntimeError):
    pass


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_verified_bytes(path: Path, expected_sha256: str, *, label: str) -> bytes:
    if not path.is_file() or path.is_symlink():
        raise SidecarContractError(f"{label} must be a regular non-symlink file: {path}")
    data = path.read_bytes()
    actual = _sha256_bytes(data)
    if actual != expected_sha256:
        raise SidecarContractError(
            f"{label} SHA-256 mismatch: expected={expected_sha256} actual={actual}"
        )
    return data


def _load_json_verified(path: Path, expected_sha256: str, *, label: str) -> dict[str, Any]:
    data = _read_verified_bytes(path, expected_sha256, label=label)
    payload = json.loads(data)
    if not isinstance(payload, dict):
        raise SidecarContractError(f"{label} must contain a JSON object")
    return payload


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise SidecarContractError(f"refusing to overwrite output: {path}")
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_torch_save(path: Path, payload: dict[str, Any]) -> str:
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    data = buffer.getvalue()
    _atomic_write_bytes(path, data)
    return _sha256_bytes(data)


def _canonical_json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _load_contract(
    input_manifest_path: Path,
    input_manifest_sha256: str,
    bank_manifest_path: Path,
    bank_manifest_sha256: str,
    *,
    split: str,
    expected_pose_tag: str = DEFAULT_POSE_TAG,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    frozen = _load_json_verified(
        input_manifest_path, input_manifest_sha256, label="frozen input manifest"
    )
    bank = _load_json_verified(bank_manifest_path, bank_manifest_sha256, label="bank manifest")
    if frozen.get("status") != "complete" or bank.get("status") != "complete":
        raise SidecarContractError("source manifests must both be complete")
    if bank.get("pose_tag") != expected_pose_tag:
        raise SidecarContractError("unexpected source pose tag")
    frozen_records = {
        str(record.get("sample_key")): record
        for record in frozen.get("records", [])
        if record.get("split") == split and record.get("status") == "eligible"
    }
    bank_records = [
        record
        for record in bank.get("records", [])
        if record.get("split") == split and record.get("status") == "complete"
    ]
    bank_records.sort(key=lambda record: int(record["split_index"]))
    if not bank_records or len(bank_records) != len(frozen_records):
        raise SidecarContractError(
            f"source inventory mismatch for {split}: bank={len(bank_records)} "
            f"frozen={len(frozen_records)}"
        )
    seen: set[str] = set()
    for record in bank_records:
        sample_key = str(record.get("sample_key"))
        frozen_record = frozen_records.get(sample_key)
        if sample_key in seen or frozen_record is None:
            raise SidecarContractError(f"duplicate or unexpected source record: {sample_key}")
        seen.add(sample_key)
        for key in ("split_index", "global_index", "system_id", "sampling_seed"):
            if record.get(key) != frozen_record.get(key):
                raise SidecarContractError(f"{sample_key}: frozen {key} mismatch")
        if int(record.get("pose_count", -1)) != NUM_SAMPLES:
            raise SidecarContractError(f"{sample_key}: pose count is not {NUM_SAMPLES}")
    return bank_records, frozen_records, bank


def _tensor_digest(sample_key: str, labels: torch.Tensor) -> str:
    values = labels.detach().cpu().contiguous().to(torch.float32)
    return hashlib.sha256(
        b"EFFDOCK_SYMMETRY_RMSD_LABEL_V1\0"
        + sample_key.encode()
        + b"\0"
        + values.numpy().tobytes(order="C")
    ).hexdigest()


def generate_shard(args: argparse.Namespace) -> None:
    started = time.monotonic()
    bank_records, frozen_records, _ = _load_contract(
        args.input_manifest,
        args.expected_input_manifest_sha256,
        args.bank_manifest,
        args.expected_bank_manifest_sha256,
        split=args.split,
        expected_pose_tag=args.expected_pose_tag,
    )
    if not 0 <= args.shard_index < args.num_shards:
        raise SidecarContractError("shard index is outside the declared shard count")
    selected = [
        record
        for position, record in enumerate(bank_records)
        if position % args.num_shards == args.shard_index
    ]
    if args.limit is not None:
        if args.limit <= 0:
            raise SidecarContractError("limit must be positive")
        selected = selected[: args.limit]
    if not selected:
        raise SidecarContractError("selected shard is empty")

    sample_keys: list[str] = []
    system_ids: list[str] = []
    split_indices: list[int] = []
    source_pt_sha256: list[str] = []
    pose_ensemble_sha256: list[str] = []
    label_sha256: list[str] = []
    labels: list[torch.Tensor] = []

    for ordinal, bank_record in enumerate(selected, start=1):
        sample_key = str(bank_record["sample_key"])
        frozen_record = frozen_records[sample_key]
        source_path = Path(str(bank_record["pt_path"]))
        source_bytes = _read_verified_bytes(
            source_path, str(bank_record["pt_sha256"]), label=f"{sample_key} source bank payload"
        )
        payload = torch.load(io.BytesIO(source_bytes), map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            raise SidecarContractError(f"{sample_key}: bank payload is not a mapping")
        poses = torch.as_tensor(payload.get("pose_atom_coords"), dtype=torch.float32)
        if poses.ndim != 3 or poses.shape[0] != NUM_SAMPLES or poses.shape[-1] != 3:
            raise SidecarContractError(f"{sample_key}: invalid saved pose tensor shape")
        if not bool(torch.isfinite(poses).all()):
            raise SidecarContractError(f"{sample_key}: saved pose tensor is non-finite")
        if payload.get("pid") != sample_key:
            raise SidecarContractError(f"{sample_key}: payload identity mismatch")
        ensemble_sha = _tensor_sha256(poses)
        if ensemble_sha != str(bank_record.get("pose_ensemble_sha256", "")):
            raise SidecarContractError(f"{sample_key}: pose ensemble identity mismatch")

        ligand_asset = frozen_record.get("processed_ligand_reference", {})
        ligand_path = Path(str(ligand_asset.get("path", "")))
        ligand_bytes = _read_verified_bytes(
            ligand_path,
            str(ligand_asset.get("sha256", "")),
            label=f"{sample_key} processed crystal ligand",
        )
        reference_ligand = torch.load(
            io.BytesIO(ligand_bytes), map_location="cpu", weights_only=False
        )
        computed = _recompute_validation_symmetry_rmsd(
            payload, frozen_record, reference_ligand, path=source_path
        ).to(torch.float32)
        if computed.shape != (NUM_SAMPLES,) or not bool(torch.isfinite(computed).all()):
            raise SidecarContractError(f"{sample_key}: invalid symmetry RMSD output")

        sample_keys.append(sample_key)
        system_ids.append(str(bank_record["system_id"]))
        split_indices.append(int(bank_record["split_index"]))
        source_pt_sha256.append(str(bank_record["pt_sha256"]))
        pose_ensemble_sha256.append(ensemble_sha)
        label_sha256.append(_tensor_digest(sample_key, computed))
        labels.append(computed)
        if ordinal % 25 == 0 or ordinal == len(selected):
            print(
                f"shard={args.shard_index:03d}/{args.num_shards:03d} "
                f"complete={ordinal}/{len(selected)}",
                flush=True,
            )

    output = args.output_root / "shards" / args.split / (
        f"shard-{args.shard_index:03d}-of-{args.num_shards:03d}.pt"
    )
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "method": METHOD,
        "split": args.split,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "num_samples_per_complex": NUM_SAMPLES,
        "input_manifest_sha256": args.expected_input_manifest_sha256,
        "bank_manifest_sha256": args.expected_bank_manifest_sha256,
        "sample_keys": sample_keys,
        "system_ids": system_ids,
        "split_indices": split_indices,
        "source_pt_sha256": source_pt_sha256,
        "pose_ensemble_sha256": pose_ensemble_sha256,
        "label_sha256": label_sha256,
        "pose_rmsd_symmetry_no_align": torch.stack(labels),
        "elapsed_seconds": time.monotonic() - started,
    }
    output_sha = _atomic_torch_save(output, artifact)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "method": METHOD,
        "split": args.split,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "record_count": len(sample_keys),
        "pose_count": len(sample_keys) * NUM_SAMPLES,
        "output_path": str(output.resolve()),
        "output_sha256": output_sha,
        "elapsed_seconds": time.monotonic() - started,
    }
    _atomic_write_bytes(output.with_suffix(".summary.json"), _canonical_json_bytes(summary))
    print(json.dumps(summary, sort_keys=True), flush=True)


def aggregate(args: argparse.Namespace) -> None:
    bank_records, _, bank = _load_contract(
        args.input_manifest,
        args.expected_input_manifest_sha256,
        args.bank_manifest,
        args.expected_bank_manifest_sha256,
        split=args.split,
        expected_pose_tag=args.expected_pose_tag,
    )
    expected = {str(record["sample_key"]): record for record in bank_records}
    manifest_records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for shard_index in range(args.num_shards):
        path = args.output_root / "shards" / args.split / (
            f"shard-{shard_index:03d}-of-{args.num_shards:03d}.pt"
        )
        summary_path = path.with_suffix(".summary.json")
        if not summary_path.is_file() or summary_path.is_symlink():
            raise SidecarContractError(f"missing shard summary: {summary_path}")
        summary = json.loads(summary_path.read_text())
        artifact_bytes = _read_verified_bytes(
            path, str(summary.get("output_sha256", "")), label=f"sidecar shard {shard_index}"
        )
        artifact = torch.load(io.BytesIO(artifact_bytes), map_location="cpu", weights_only=False)
        labels = torch.as_tensor(artifact.get("pose_rmsd_symmetry_no_align"))
        sample_keys = artifact.get("sample_keys")
        n = len(sample_keys) if isinstance(sample_keys, list) else -1
        list_fields = (
            "system_ids",
            "split_indices",
            "source_pt_sha256",
            "pose_ensemble_sha256",
            "label_sha256",
        )
        if (
            artifact.get("schema_version") != SCHEMA_VERSION
            or artifact.get("status") != "complete"
            or artifact.get("method") != METHOD
            or artifact.get("split") != args.split
            or artifact.get("shard_index") != shard_index
            or artifact.get("num_shards") != args.num_shards
            or labels.shape != (n, NUM_SAMPLES)
            or not bool(torch.isfinite(labels).all())
            or any(not isinstance(artifact.get(key), list) or len(artifact[key]) != n for key in list_fields)
        ):
            raise SidecarContractError(f"invalid sidecar shard contract: {path}")
        for row_index, sample_key_value in enumerate(sample_keys):
            sample_key = str(sample_key_value)
            source = expected.get(sample_key)
            if source is None or sample_key in seen:
                raise SidecarContractError(f"duplicate or unexpected sidecar record: {sample_key}")
            seen.add(sample_key)
            if (
                artifact["system_ids"][row_index] != source["system_id"]
                or int(artifact["split_indices"][row_index]) != int(source["split_index"])
                or artifact["source_pt_sha256"][row_index] != source["pt_sha256"]
                or artifact["pose_ensemble_sha256"][row_index]
                != source["pose_ensemble_sha256"]
                or artifact["label_sha256"][row_index]
                != _tensor_digest(sample_key, labels[row_index])
            ):
                raise SidecarContractError(f"{sample_key}: sidecar provenance mismatch")
            manifest_records.append(
                {
                    "sample_key": sample_key,
                    "system_id": source["system_id"],
                    "split": args.split,
                    "split_index": int(source["split_index"]),
                    "pose_count": NUM_SAMPLES,
                    "pose_ensemble_sha256": source["pose_ensemble_sha256"],
                    "source_pt_sha256": source["pt_sha256"],
                    "sidecar_path": str(path.resolve()),
                    "sidecar_sha256": str(summary["output_sha256"]),
                    "row_index": row_index,
                    "label_sha256": artifact["label_sha256"][row_index],
                }
            )
    if seen != set(expected):
        missing = sorted(set(expected) - seen)
        raise SidecarContractError(f"sidecar inventory incomplete; missing={missing[:5]}")
    manifest_records.sort(key=lambda record: int(record["split_index"]))
    output = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "status": "complete",
        "method": METHOD,
        "label_key": "pose_rmsd_symmetry_no_align",
        "split": args.split,
        "record_count": len(manifest_records),
        "pose_count": len(manifest_records) * NUM_SAMPLES,
        "num_shards": args.num_shards,
        "num_samples_per_complex": NUM_SAMPLES,
        "input_manifest": {
            "path": str(args.input_manifest.resolve()),
            "sha256": args.expected_input_manifest_sha256,
        },
        "bank_manifest": {
            "path": str(args.bank_manifest.resolve()),
            "sha256": args.expected_bank_manifest_sha256,
            "pose_tag": bank["pose_tag"],
        },
        "records": manifest_records,
    }
    _atomic_write_bytes(args.output_manifest, _canonical_json_bytes(output))
    print(
        json.dumps(
            {
                "status": "complete",
                "output_manifest": str(args.output_manifest.resolve()),
                "record_count": len(manifest_records),
                "pose_count": len(manifest_records) * NUM_SAMPLES,
            },
            sort_keys=True,
        )
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--input-manifest", type=Path, required=True)
    common.add_argument("--expected-input-manifest-sha256", required=True)
    common.add_argument("--bank-manifest", type=Path, required=True)
    common.add_argument("--expected-bank-manifest-sha256", required=True)
    common.add_argument("--output-root", type=Path, required=True)
    common.add_argument("--split", choices=("train", "val"), default="train")
    common.add_argument("--num-shards", type=int, required=True)
    common.add_argument("--expected-pose-tag", default=DEFAULT_POSE_TAG)

    generate = subparsers.add_parser("generate-shard", parents=[common])
    generate.add_argument("--shard-index", type=int, required=True)
    generate.add_argument("--limit", type=int)
    generate.set_defaults(func=generate_shard)

    merge = subparsers.add_parser("aggregate", parents=[common])
    merge.add_argument("--output-manifest", type=Path, required=True)
    merge.set_defaults(func=aggregate)
    return parser


def main() -> None:
    args = _parser().parse_args()
    for field in ("expected_input_manifest_sha256", "expected_bank_manifest_sha256"):
        value = getattr(args, field)
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise SidecarContractError(f"{field} is not a lowercase SHA-256 digest")
    if args.num_shards <= 0:
        raise SidecarContractError("num_shards must be positive")
    args.func(args)


if __name__ == "__main__":
    main()
