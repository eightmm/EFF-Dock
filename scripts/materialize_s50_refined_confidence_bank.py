#!/usr/bin/env python3
"""Materialize confidence features/labels for the sealed refined S50 poses."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from effdock.inference.docking import load_model
from effdock.workflows.evaluate import file_sha256
from scripts.prepare_s50_confidence_training_bank import (
    _center_graph,
    _extract_hidden_chunked,
    _fixed_labels,
    _prepare_runtime_input,
    _symmetry_no_align_rmsd,
    _tensor_sha256,
)

PROTOCOL_ID = "EFFDOCK-S50-CONFIDENCE-TRAINING-BANK-V1"
STUDY_PROTOCOL_ID = "EFFDOCK-S50-RAW-REFINED-CONFIDENCE-FINETUNE-V2"
MANIFEST_SCHEMA = "effdock.s50_confidence_bank.manifest.v1"
SHARD_SCHEMA = "effdock.s50_refined_confidence_bank.shard.v1"
REFINED_PROTOCOL_ID = "EFFDOCK-S50-REFINED-POSE-BANK-V2"
REFINED_MANIFEST_SCHEMA = "effdock.s50_refined_pose_bank.manifest.v2"
POSE_TAG = "s50_n100_s10_sig2_latep3_pc10_rdkitseed0_refine100"
NUM_SAMPLES = 100
HIDDEN_CHUNK_SIZE = 20
SETTINGS = {
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
    "refine": "guidance_unified_step100",
    "fk_resampling": False,
    "particle_resampling": False,
    "eligibility_boundary": "input_only_no_sampled_pose_outcomes",
    "refinement_steps": 100,
    "refinement_receptor_policy": "geometry_only",
}


class MaterializeError(RuntimeError):
    pass


def _canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, separators=(",", ":"), sort_keys=True) + "\n").encode()


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    fd, raw = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    attempt = Path(raw)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(attempt, path)
    finally:
        attempt.unlink(missing_ok=True)


def _atomic_torch_save(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    fd, raw = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(fd)
    attempt = Path(raw)
    try:
        torch.save(payload, attempt)
        os.link(attempt, path)
    finally:
        attempt.unlink(missing_ok=True)


def _ordered_ids_sha256(ids: list[str]) -> str:
    digest = hashlib.sha256()
    for sample_id in ids:
        digest.update(sample_id.encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise MaterializeError(f"expected JSON object: {path}")
    return value


def _load_contracts(
    raw_manifest_path: Path,
    refined_manifest_path: Path,
    input_manifest_path: Path,
) -> tuple[
    dict[str, Any],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    raw = _load_json(raw_manifest_path)
    refined = _load_json(refined_manifest_path)
    frozen = _load_json(input_manifest_path)
    if raw.get("protocol_id") != PROTOCOL_ID or raw.get("status") != "complete":
        raise MaterializeError("raw confidence bank is incomplete")
    if (
        refined.get("schema_version") != REFINED_MANIFEST_SCHEMA
        or refined.get("protocol_id") != REFINED_PROTOCOL_ID
        or refined.get("status") != "complete"
    ):
        raise MaterializeError("refined coordinate bank is incomplete")
    raw_by_id = {str(row["sample_key"]): row for row in raw["records"]}
    refined_by_id = {str(row["sample_key"]): row for row in refined["records"]}
    frozen_by_id = {
        str(row["sample_key"]): row
        for row in frozen["records"]
        if row.get("status") == "eligible"
    }
    if set(raw_by_id) != set(refined_by_id) or set(raw_by_id) != set(frozen_by_id):
        raise MaterializeError("raw/refined/frozen eligible inventories differ")
    return raw, refined_by_id, frozen_by_id


def _selected_records(
    raw: dict[str, Any], *, split: str, shard_index: int, num_shards: int
) -> list[dict[str, Any]]:
    return [
        row
        for row in sorted(
            (item for item in raw["records"] if item["split"] == split),
            key=lambda item: int(item["split_index"]),
        )
        if (int(row["split_index"]) - 1) % num_shards == shard_index
    ]


def _materialize_one(
    raw_record: dict[str, Any],
    refined_record: dict[str, Any],
    frozen_record: dict[str, Any],
    *,
    model: torch.nn.Module,
    device: torch.device,
    processed_root: Path,
    checkpoint: Path,
    checkpoint_sha256: str,
    config: Path,
    config_sha256: str,
) -> dict[str, Any]:
    sample_id = str(raw_record["sample_key"])
    raw_path = Path(str(raw_record["pt_path"])).resolve(strict=True)
    refined_path = Path(str(refined_record["pt_path"])).resolve(strict=True)
    if file_sha256(raw_path) != raw_record["pt_sha256"]:
        raise MaterializeError(f"{sample_id}: raw payload changed")
    if file_sha256(refined_path) != refined_record["pt_sha256"]:
        raise MaterializeError(f"{sample_id}: refined payload changed")
    raw_payload = torch.load(raw_path, map_location="cpu", weights_only=False)
    refined_payload = torch.load(refined_path, map_location="cpu", weights_only=False)
    poses = torch.as_tensor(
        refined_payload["pose_atom_coords_refined"], dtype=torch.float32
    )
    if (
        poses.shape != raw_payload["pose_atom_coords"].shape
        or poses.shape[0] != NUM_SAMPLES
        or not bool(torch.isfinite(poses).all())
        or refined_payload.get("source_pt_sha256") != raw_record["pt_sha256"]
    ):
        raise MaterializeError(f"{sample_id}: refined coordinate identity mismatch")
    (
        molecule_input,
        molecule_reference,
        graph,
        ligand_data,
        inference_meta,
        reference_ligand,
    ) = _prepare_runtime_input(frozen_record, processed_root)
    hidden = _extract_hidden_chunked(
        model,
        graph,
        ligand_data,
        inference_meta,
        poses,
        device=device,
        hidden_chunk_size=HIDDEN_CHUNK_SIZE,
    )
    pocket_center = inference_meta["pocket_center"].detach().cpu().to(torch.float32)
    reference_aligned, atom_disp, pose_rmsd = _fixed_labels(
        poses,
        reference_ligand,
        frozen_record["input_to_reference"],
        pocket_center,
    )
    crystal_pose = reference_aligned.unsqueeze(0).contiguous()
    crystal_hidden = _extract_hidden_chunked(
        model,
        graph,
        ligand_data,
        inference_meta,
        crystal_pose,
        device=device,
        hidden_chunk_size=1,
    )
    crystal_atom_disp = torch.zeros(
        (1, reference_aligned.shape[0]), dtype=torch.float32
    )
    crystal_rmsd = torch.zeros(1, dtype=torch.float32)
    payload = dict(raw_payload)
    payload.update(
        {
            "study_protocol_id": STUDY_PROTOCOL_ID,
            "pose_tag": POSE_TAG,
            "pose_source": "s50_guidance_unified_refinement_step100",
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": checkpoint_sha256,
            "config": str(config.resolve()),
            "config_sha256": config_sha256,
            "pose_atom_coords": poses,
            "h_lig_node": hidden["h_lig_node"],
            "lig_node_type": hidden["lig_node_type"],
            "lig_atom_coords_crystal_centered": reference_aligned,
            "atom_disp": atom_disp,
            "pose_rmsd": pose_rmsd,
            "graph_centered": _center_graph(graph, pocket_center),
            "source_raw_pt_path": str(raw_path),
            "source_raw_pt_sha256": raw_record["pt_sha256"],
            "source_refined_pt_path": str(refined_path),
            "source_refined_pt_sha256": refined_record["pt_sha256"],
            "source_refined_pose_ensemble_sha256": refined_record[
                "refined_pose_ensemble_sha256"
            ],
            "refinement_protocol_id": REFINED_PROTOCOL_ID,
            "refinement_steps": 100,
            "refinement_receptor_policy": "geometry_only",
            "crystal_anchor_pose_atom_coords": crystal_pose,
            "crystal_anchor_h_lig_node": crystal_hidden["h_lig_node"],
            "crystal_anchor_atom_disp": crystal_atom_disp,
            "crystal_anchor_pose_rmsd": crystal_rmsd,
            "crystal_anchor_rmsd_method": "exact_mapped_reference_zero",
        }
    )
    payload["graph"] = payload["graph_centered"]
    if raw_record["split"] == "val":
        payload["pose_rmsd_symmetry_no_align"] = _symmetry_no_align_rmsd(
            poses, molecule_input, molecule_reference, pocket_center
        )
        payload["symmetry_rmsd_method"] = "rdkit_calc_rms_symmetry_no_align"
    return payload


def generate_shard(args: argparse.Namespace) -> None:
    raw, refined_by_id, frozen_by_id = _load_contracts(
        args.raw_bank_manifest, args.refined_manifest, args.input_manifest
    )
    records = _selected_records(
        raw,
        split=args.split,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
    )
    if args.limit is not None:
        records = records[: args.limit]
    if not records:
        raise MaterializeError("selected materialization shard is empty")
    if file_sha256(args.checkpoint) != args.checkpoint_sha256:
        raise MaterializeError("S50 checkpoint hash mismatch")
    if file_sha256(args.config) != args.config_sha256:
        raise MaterializeError("S50 config hash mismatch")
    device = torch.device(args.device)
    model, _, _ = load_model(args.config, args.checkpoint, device)
    model.eval()
    shard_root = (
        args.output_root.resolve()
        / "shards"
        / args.split
        / f"shard-{args.shard_index:03d}-of-{args.num_shards:03d}"
    )
    summary_path = shard_root / "summary.json"
    if summary_path.exists():
        raise FileExistsError(f"refusing to reuse completed shard: {summary_path}")
    output_records: list[dict[str, Any]] = []
    for index, raw_record in enumerate(records, start=1):
        sample_id = str(raw_record["sample_key"])
        output_path = (
            shard_root
            / sample_id
            / "confidence_poses"
            / f"confposes_{POSE_TAG}.pt"
        )
        if output_path.exists():
            payload = torch.load(output_path, map_location="cpu", weights_only=False)
            if payload.get("pid") != sample_id or payload.get("pose_tag") != POSE_TAG:
                raise MaterializeError(f"{sample_id}: existing output identity mismatch")
        else:
            payload = _materialize_one(
                raw_record,
                refined_by_id[sample_id],
                frozen_by_id[sample_id],
                model=model,
                device=device,
                processed_root=args.processed_root,
                checkpoint=args.checkpoint,
                checkpoint_sha256=args.checkpoint_sha256,
                config=args.config,
                config_sha256=args.config_sha256,
            )
            _atomic_torch_save(output_path, payload)
        output_records.append(
            {
                "global_index": int(raw_record["global_index"]),
                "pose_count": NUM_SAMPLES,
                "pose_ensemble_sha256": _tensor_sha256(payload["pose_atom_coords"]),
                "pose_source": payload["pose_source"],
                "pt_path": str(output_path.resolve()),
                "pt_sha256": file_sha256(output_path),
                "sample_key": sample_id,
                "sampling_seed": int(raw_record["sampling_seed"]),
                "size_bytes": output_path.stat().st_size,
                "split": args.split,
                "split_index": int(raw_record["split_index"]),
                "status": "complete",
                "system_id": raw_record["system_id"],
            }
        )
        print(f"[{args.split}] {index}/{len(records)} {sample_id}", flush=True)
    summary = {
        "schema_version": SHARD_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "study_protocol_id": STUDY_PROTOCOL_ID,
        "status": "complete",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "split": args.split,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "pose_tag": POSE_TAG,
        "record_count": len(output_records),
        "record_ids_sha256": _ordered_ids_sha256(
            [row["sample_key"] for row in output_records]
        ),
        "records": output_records,
        "inputs": {
            "raw_bank_manifest_sha256": file_sha256(args.raw_bank_manifest),
            "refined_manifest_sha256": file_sha256(args.refined_manifest),
            "input_manifest_sha256": file_sha256(args.input_manifest),
            "checkpoint_sha256": args.checkpoint_sha256,
            "config_sha256": args.config_sha256,
        },
        "runtime": {"slurm_job_id": os.environ.get("SLURM_JOB_ID")},
    }
    _atomic_write(summary_path, _canonical_bytes(summary))


def aggregate(args: argparse.Namespace) -> None:
    raw, _, _ = _load_contracts(
        args.raw_bank_manifest, args.refined_manifest, args.input_manifest
    )
    output_root = args.output_root.resolve()
    manifest_path = output_root / "bank_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite {manifest_path}")
    records: list[dict[str, Any]] = []
    summaries: list[dict[str, str]] = []
    for split, num_shards in (("train", args.train_shards), ("val", args.val_shards)):
        for shard_index in range(num_shards):
            path = (
                output_root
                / "shards"
                / split
                / f"shard-{shard_index:03d}-of-{num_shards:03d}"
                / "summary.json"
            )
            summary = _load_json(path)
            if (
                summary.get("schema_version") != SHARD_SCHEMA
                or summary.get("status") != "complete"
                or summary.get("split") != split
                or int(summary.get("shard_index", -1)) != shard_index
            ):
                raise MaterializeError(f"invalid shard summary: {path}")
            for record in summary["records"]:
                pt_path = Path(str(record["pt_path"])).resolve(strict=True)
                if file_sha256(pt_path) != record["pt_sha256"]:
                    raise MaterializeError(f"changed materialized payload: {pt_path}")
                payload = torch.load(pt_path, map_location="cpu", weights_only=False)
                if (
                    payload.get("pose_tag") != POSE_TAG
                    or payload.get("pid") != record["sample_key"]
                    or payload["pose_atom_coords"].shape[0] != NUM_SAMPLES
                    or payload["h_lig_node"].shape[0] != NUM_SAMPLES
                    or not bool(torch.isfinite(payload["h_lig_node"]).all())
                    or payload.get("crystal_anchor_pose_atom_coords", torch.empty(0)).shape
                    != (1, payload["pose_atom_coords"].shape[1], 3)
                    or payload.get("crystal_anchor_h_lig_node", torch.empty(0)).shape[:1]
                    != (1,)
                    or payload.get("crystal_anchor_atom_disp", torch.empty(0)).shape
                    != (1, payload["pose_atom_coords"].shape[1])
                    or payload.get("crystal_anchor_pose_rmsd", torch.empty(0)).shape != (1,)
                    or float(payload["crystal_anchor_pose_rmsd"][0]) != 0.0
                    or payload.get("crystal_anchor_rmsd_method")
                    != "exact_mapped_reference_zero"
                    or not torch.equal(
                        payload["crystal_anchor_pose_atom_coords"][0],
                        payload["lig_atom_coords_crystal_centered"],
                    )
                    or bool(payload["crystal_anchor_atom_disp"].abs().max() != 0.0)
                    or any(
                        not bool(torch.isfinite(payload[key]).all())
                        for key in (
                            "crystal_anchor_pose_atom_coords",
                            "crystal_anchor_h_lig_node",
                            "crystal_anchor_atom_disp",
                            "crystal_anchor_pose_rmsd",
                        )
                    )
                ):
                    raise MaterializeError(f"invalid materialized payload: {pt_path}")
                records.append(record)
            summaries.append({"path": str(path.resolve()), "sha256": file_sha256(path)})
    source_records = sorted(
        raw["records"], key=lambda row: (row["split"], int(row["split_index"]))
    )
    records = sorted(records, key=lambda row: (row["split"], int(row["split_index"])))
    if [row["sample_key"] for row in records] != [
        row["sample_key"] for row in source_records
    ]:
        raise MaterializeError("materialized bank does not cover the exact source bank")
    inventory = raw["inventory"]
    filtered_split_path = Path(str(raw["filtered_split_path"])).resolve(strict=True)
    filtered_split_sha256 = file_sha256(filtered_split_path)
    if filtered_split_sha256 != raw["filtered_split_sha256"]:
        raise MaterializeError("source filtered split changed")
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "study_protocol_id": STUDY_PROTOCOL_ID,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "complete",
        "claim_eligible": True,
        "pose_tag": POSE_TAG,
        "settings": SETTINGS,
        "filtered_split_path": str(filtered_split_path),
        "filtered_split_sha256": filtered_split_sha256,
        "inventory": inventory,
        "records": records,
        "shard_summaries": summaries,
        "inputs": {
            "raw_bank_manifest": str(args.raw_bank_manifest.resolve()),
            "raw_bank_manifest_sha256": file_sha256(args.raw_bank_manifest),
            "refined_manifest": str(args.refined_manifest.resolve()),
            "refined_manifest_sha256": file_sha256(args.refined_manifest),
            "input_manifest": str(args.input_manifest.resolve()),
            "input_manifest_sha256": file_sha256(args.input_manifest),
        },
    }
    _atomic_write(manifest_path, _canonical_bytes(manifest))
    _atomic_write(
        manifest_path.with_suffix(".json.sha256"),
        f"{file_sha256(manifest_path)}  {manifest_path}\n".encode(),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    generate = sub.add_parser("generate-shard")
    for target in (generate,):
        target.add_argument("--raw-bank-manifest", type=Path, required=True)
        target.add_argument("--refined-manifest", type=Path, required=True)
        target.add_argument("--input-manifest", type=Path, required=True)
        target.add_argument("--output-root", type=Path, required=True)
    generate.add_argument("--processed-root", type=Path, required=True)
    generate.add_argument("--checkpoint", type=Path, required=True)
    generate.add_argument("--checkpoint-sha256", required=True)
    generate.add_argument("--config", type=Path, required=True)
    generate.add_argument("--config-sha256", required=True)
    generate.add_argument("--split", choices=("train", "val"), required=True)
    generate.add_argument("--shard-index", type=int, required=True)
    generate.add_argument("--num-shards", type=int, required=True)
    generate.add_argument("--limit", type=int)
    generate.add_argument("--device", default="cuda")
    aggregate_parser = sub.add_parser("aggregate")
    aggregate_parser.add_argument("--raw-bank-manifest", type=Path, required=True)
    aggregate_parser.add_argument("--refined-manifest", type=Path, required=True)
    aggregate_parser.add_argument("--input-manifest", type=Path, required=True)
    aggregate_parser.add_argument("--output-root", type=Path, required=True)
    aggregate_parser.add_argument("--train-shards", type=int, default=128)
    aggregate_parser.add_argument("--val-shards", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.command == "generate-shard":
        generate_shard(args)
    else:
        aggregate(args)


if __name__ == "__main__":
    main()
