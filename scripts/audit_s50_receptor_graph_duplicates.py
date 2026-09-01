#!/usr/bin/env python3
"""Audit exact receptor-coordinate duplication in the frozen S50 bank cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch

SCHEMA_VERSION = "effdock.s50_receptor_graph_duplicate_audit.v1"
OPERATIONAL_NODE_THRESHOLD = 1200
MATERIAL_DUPLICATE_FRACTION = 0.10
FALLBACK_RADIUS_INCREMENT = 5.0
FALLBACK_NEAREST_RESIDUES = 32


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_resolve(path: Path, *, label: str) -> Path:
    resolved = path.resolve(strict=True)
    if resolved.is_symlink() or not resolved.is_file():
        raise ValueError(f"{label} must be a regular file: {resolved}")
    return resolved


def _coordinate_bits(coords: torch.Tensor) -> np.ndarray:
    values = coords.detach().cpu().contiguous().to(torch.float32).numpy()
    return values.view(np.int32).reshape(values.shape[0], 3).astype(np.int64, copy=False)


def _atom_signatures(protein: dict[str, torch.Tensor], mask: torch.Tensor) -> np.ndarray:
    columns = [
        _coordinate_bits(protein["patom_coords"][mask]),
        protein["patom_token"][mask].to(torch.int64).cpu().numpy()[:, None],
    ]
    for key in (
        "patom_is_backbone",
        "patom_is_metal",
        "patom_is_donor",
        "patom_is_acceptor",
        "patom_is_positive",
        "patom_is_negative",
        "patom_is_hydrophobic",
    ):
        columns.append(protein[key][mask].to(torch.int64).cpu().numpy()[:, None])
    return np.concatenate(columns, axis=1)


def _residue_signatures(
    protein: dict[str, torch.Tensor], active_residue_ids: torch.Tensor
) -> np.ndarray:
    columns = [
        _coordinate_bits(protein["pres_coords"][active_residue_ids]),
        protein["pres_residue_type"][active_residue_ids]
        .to(torch.int64)
        .cpu()
        .numpy()[:, None],
        protein["pres_is_pseudo"][active_residue_ids]
        .to(torch.int64)
        .cpu()
        .numpy()[:, None],
    ]
    return np.concatenate(columns, axis=1)


def _unique_stats(signatures: np.ndarray) -> tuple[int, int]:
    if signatures.shape[0] == 0:
        return 0, 0
    _, counts = np.unique(signatures, axis=0, return_counts=True)
    return int(counts.shape[0]), int(counts.max(initial=0))


def _select_crop_atom_mask(
    coords: torch.Tensor,
    residue_ids: torch.Tensor,
    center: torch.Tensor,
    pocket_cutoff: float,
) -> tuple[torch.Tensor, str]:
    atom_distance = torch.linalg.vector_norm(coords - center, dim=1)
    for cutoff, route in (
        (pocket_cutoff, "radius_primary"),
        (pocket_cutoff + FALLBACK_RADIUS_INCREMENT, "radius_plus_5"),
    ):
        near_atom_mask = atom_distance <= cutoff
        if bool(near_atom_mask.any()):
            active_residue_ids = torch.unique(residue_ids[near_atom_mask], sorted=True)
            return torch.isin(residue_ids, active_residue_ids), route

    n_res = int(residue_ids.max().item()) + 1
    residue_distance = torch.full((n_res,), float("inf"), dtype=atom_distance.dtype)
    residue_distance.scatter_reduce_(
        0, residue_ids, atom_distance, reduce="amin", include_self=True
    )
    k = min(FALLBACK_NEAREST_RESIDUES, n_res)
    if k <= 0:
        raise ValueError("empty receptor residue inventory")
    active_residue_ids = torch.topk(residue_distance, k=k, largest=False).indices
    pocket_atom_mask = torch.isin(residue_ids, active_residue_ids)
    if not bool(pocket_atom_mask.any()):
        raise ValueError("nearest-residue fallback produced an empty crop")
    return pocket_atom_mask, "nearest_32_residues"


def _audit_one(task: dict[str, Any]) -> dict[str, Any]:
    sample_key = str(task["sample_key"])
    processed_root = Path(task["processed_root"])
    sample_root = processed_root / sample_key
    protein_path = _strict_resolve(sample_root / "protein.pt", label=f"{sample_key}.protein")
    meta_path = _strict_resolve(sample_root / "meta.pt", label=f"{sample_key}.meta")
    protein = torch.load(protein_path, map_location="cpu", weights_only=True)
    meta = torch.load(meta_path, map_location="cpu", weights_only=True)
    if not isinstance(protein, dict) or not isinstance(meta, dict):
        raise ValueError(f"{sample_key}: processed tensors must be mappings")
    if meta.get("plinder_system_id") != task["system_id"]:
        raise ValueError(f"{sample_key}: processed system_id mismatch")

    coords = protein["patom_coords"].to(torch.float32)
    residue_ids = protein["patom_residue_id"].to(torch.long)
    center = meta["pocket_center"].to(torch.float32)
    if coords.ndim != 2 or coords.shape[1] != 3 or center.shape != (3,):
        raise ValueError(f"{sample_key}: invalid receptor coordinates or pocket center")
    if not bool(torch.isfinite(coords).all()) or not bool(torch.isfinite(center).all()):
        raise ValueError(f"{sample_key}: non-finite receptor coordinates or pocket center")

    pocket_atom_mask, crop_route = _select_crop_atom_mask(
        coords, residue_ids, center, float(task["pocket_cutoff"])
    )
    active_residue_ids = torch.unique(residue_ids[pocket_atom_mask], sorted=True)

    pres_atom_index = protein["pres_atom_index"].to(torch.long)
    if pres_atom_index.ndim != 1 or bool(
        ((pres_atom_index < 0) | (pres_atom_index >= coords.shape[0])).any()
    ):
        raise ValueError(f"{sample_key}: invalid residue virtual-node anchor index")
    pocket_pres_mask = pocket_atom_mask[pres_atom_index]
    pocket_pres_indices = pocket_pres_mask.nonzero(as_tuple=True)[0]
    pres_residue_ids = residue_ids[pres_atom_index[pocket_pres_mask]]
    virtualized_residue_group_count = int(torch.unique(pres_residue_ids).numel())
    missing_virtual_residue_groups = (
        int(active_residue_ids.numel()) - virtualized_residue_group_count
    )
    if missing_virtual_residue_groups < 0:
        raise ValueError(f"{sample_key}: more virtual residue groups than atom residue groups")

    atom_count = int(pocket_atom_mask.sum())
    residue_count = int(pocket_pres_indices.numel())
    unique_atom_count, max_atom_multiplicity = _unique_stats(
        _atom_signatures(protein, pocket_atom_mask)
    )
    unique_residue_count, max_residue_multiplicity = _unique_stats(
        _residue_signatures(protein, pocket_pres_indices)
    )
    ligand_nodes = int(meta["num_atom"]) + int(meta["num_frag"])
    graph_nodes = ligand_nodes + atom_count + residue_count
    deduplicated_graph_nodes = ligand_nodes + unique_atom_count + unique_residue_count
    atom_removed = atom_count - unique_atom_count
    residue_removed = residue_count - unique_residue_count
    atom_duplicate_fraction = atom_removed / atom_count
    residue_duplicate_fraction = residue_removed / residue_count
    material_duplication = (
        atom_duplicate_fraction >= MATERIAL_DUPLICATE_FRACTION
        and residue_duplicate_fraction >= MATERIAL_DUPLICATE_FRACTION
    )

    return {
        "sample_key": sample_key,
        "split": task["split"],
        "system_id": task["system_id"],
        "system_id_no_biounit": task.get("system_id_no_biounit"),
        "receptor_instance_count": int(task.get("receptor_instance_count", 0)),
        "unique_receptor_chain_count": int(task.get("unique_receptor_chain_count", 0)),
        "pocket_cutoff": float(task["pocket_cutoff"]),
        "crop_route": crop_route,
        "ligand_nodes": ligand_nodes,
        "protein_atom_nodes": atom_count,
        "protein_residue_nodes": residue_count,
        "protein_atom_residue_group_count": int(active_residue_ids.numel()),
        "atom_residue_groups_without_virtual_node": missing_virtual_residue_groups,
        "graph_nodes": graph_nodes,
        "unique_protein_atom_signatures": unique_atom_count,
        "unique_protein_residue_signatures": unique_residue_count,
        "deduplicated_graph_nodes": deduplicated_graph_nodes,
        "exact_duplicate_atom_nodes": atom_removed,
        "exact_duplicate_residue_nodes": residue_removed,
        "atom_duplicate_fraction": atom_duplicate_fraction,
        "residue_duplicate_fraction": residue_duplicate_fraction,
        "max_atom_multiplicity": max_atom_multiplicity,
        "max_residue_multiplicity": max_residue_multiplicity,
        "has_any_exact_duplicate": atom_removed > 0 or residue_removed > 0,
        "material_exact_duplication": material_duplication,
        "over_operational_node_threshold": graph_nodes > OPERATIONAL_NODE_THRESHOLD,
        "deduplicated_over_operational_node_threshold": (
            deduplicated_graph_nodes > OPERATIONAL_NODE_THRESHOLD
        ),
        "duplicate_inflated_over_threshold": (
            graph_nodes > OPERATIONAL_NODE_THRESHOLD
            and deduplicated_graph_nodes <= OPERATIONAL_NODE_THRESHOLD
        ),
        "protein_path": str(protein_path),
        "protein_sha256": _sha256(protein_path),
        "meta_path": str(meta_path),
        "meta_sha256": _sha256(meta_path),
    }


def _parse_receptor_instances(system_id_no_biounit: str | None) -> tuple[int, int]:
    if not system_id_no_biounit:
        return 0, 0
    fields = str(system_id_no_biounit).split("__")
    if len(fields) < 2:
        return 0, 0
    instances = [value for value in fields[1].split("_") if value]
    return len(instances), len(set(instances))


def _top(records: list[dict[str, Any]], key: str, *, limit: int = 100) -> list[dict[str, Any]]:
    return sorted(records, key=lambda record: (record[key], record["sample_key"]), reverse=True)[
        :limit
    ]


def _aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    any_duplicate = [record for record in records if record["has_any_exact_duplicate"]]
    material = [record for record in records if record["material_exact_duplication"]]
    oversized = [record for record in records if record["over_operational_node_threshold"]]
    deduplicated_oversized = [
        record for record in records if record["deduplicated_over_operational_node_threshold"]
    ]
    duplicate_inflated = [
        record for record in records if record["duplicate_inflated_over_threshold"]
    ]
    missing_virtual = [
        record
        for record in records
        if record["atom_residue_groups_without_virtual_node"] > 0
    ]
    return {
        "record_count": len(records),
        "split_counts": {
            split: sum(record["split"] == split for record in records)
            for split in ("train", "val")
        },
        "crop_route_counts": {
            route: sum(record["crop_route"] == route for record in records)
            for route in ("radius_primary", "radius_plus_5", "nearest_32_residues")
        },
        "any_exact_duplicate_count": len(any_duplicate),
        "material_exact_duplication_count": len(material),
        "over_operational_node_threshold_count": len(oversized),
        "deduplicated_over_operational_node_threshold_count": len(deduplicated_oversized),
        "duplicate_inflated_over_threshold_count": len(duplicate_inflated),
        "atom_residue_groups_without_virtual_node_record_count": len(missing_virtual),
        "material_exact_duplication_ids": sorted(record["sample_key"] for record in material),
        "duplicate_inflated_over_threshold_ids": sorted(
            record["sample_key"] for record in duplicate_inflated
        ),
        "deduplicated_over_operational_node_threshold_ids": sorted(
            record["sample_key"] for record in deduplicated_oversized
        ),
        "top_graph_nodes": _top(records, "graph_nodes"),
        "top_deduplicated_graph_nodes": _top(records, "deduplicated_graph_nodes"),
        "top_exact_duplicate_atom_nodes": _top(records, "exact_duplicate_atom_nodes"),
        "top_atom_duplicate_fraction": _top(records, "atom_duplicate_fraction"),
        "top_atom_residue_groups_without_virtual_node": _top(
            records, "atom_residue_groups_without_virtual_node"
        ),
    }


def _load_tasks(
    bank_manifest: Path,
    pool_parquet: Path,
    processed_root: Path,
    pocket_cutoff: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    bank = json.loads(bank_manifest.read_text())
    if bank.get("status") != "complete" or not bank.get("claim_eligible"):
        raise ValueError("bank manifest must be complete and claim-eligible")
    records = bank.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("bank manifest has no records")

    table = pq.read_table(pool_parquet, columns=["system_id", "system_id_no_biounit"])
    pool_by_system: dict[str, str] = {}
    for row in table.to_pylist():
        pool_by_system.setdefault(str(row["system_id"]), str(row["system_id_no_biounit"]))

    tasks: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in records:
        sample_key = str(record.get("sample_key", ""))
        if not sample_key or sample_key in seen:
            raise ValueError(f"invalid or duplicate bank sample_key: {sample_key!r}")
        seen.add(sample_key)
        split = str(record.get("split", ""))
        if split not in {"train", "val"}:
            raise ValueError(f"{sample_key}: invalid split {split!r}")
        system_id = str(record.get("system_id", ""))
        if system_id not in pool_by_system:
            raise ValueError(f"{sample_key}: system_id missing from pool parquet")
        no_biounit = pool_by_system[system_id]
        instance_count, unique_chain_count = _parse_receptor_instances(no_biounit)
        tasks.append(
            {
                "sample_key": sample_key,
                "split": split,
                "system_id": system_id,
                "system_id_no_biounit": no_biounit,
                "receptor_instance_count": instance_count,
                "unique_receptor_chain_count": unique_chain_count,
                "processed_root": str(processed_root),
                "pocket_cutoff": pocket_cutoff,
            }
        )
    return tasks, bank


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank-manifest", type=Path, required=True)
    parser.add_argument("--pool-parquet", type=Path, default=Path("data/plinder_pool.parquet"))
    parser.add_argument("--processed-root", type=Path, default=Path("data/plinder_processed"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--pocket-cutoff", type=float, default=10.0)
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    if args.pocket_cutoff <= 0:
        raise ValueError("pocket cutoff must be positive")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output}")

    started = time.monotonic()
    bank_manifest = _strict_resolve(args.bank_manifest, label="bank manifest")
    pool_parquet = _strict_resolve(args.pool_parquet, label="pool parquet")
    processed_root = args.processed_root.resolve(strict=True)
    if not processed_root.is_dir():
        raise ValueError(f"processed root is not a directory: {processed_root}")
    tasks, bank = _load_tasks(
        bank_manifest, pool_parquet, processed_root, float(args.pocket_cutoff)
    )

    records: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_audit_one, task): task["sample_key"] for task in tasks}
        for completed, future in enumerate(as_completed(futures), start=1):
            sample_key = futures[future]
            try:
                records.append(future.result())
            except Exception as exc:
                raise RuntimeError(f"audit failed for {sample_key}") from exc
            if completed % 500 == 0 or completed == len(futures):
                print(f"audited={completed}/{len(futures)}", flush=True)
    records.sort(key=lambda record: record["sample_key"])

    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "settings": {
            "pocket_cutoff": float(args.pocket_cutoff),
            "operational_node_threshold": OPERATIONAL_NODE_THRESHOLD,
            "material_duplicate_fraction": MATERIAL_DUPLICATE_FRACTION,
            "fallback_radius_increment": FALLBACK_RADIUS_INCREMENT,
            "fallback_nearest_residues": FALLBACK_NEAREST_RESIDUES,
            "duplicate_signature": (
                "exact float32 coordinate bits plus protein atom/residue chemical features"
            ),
        },
        "inputs": {
            "bank_manifest": str(bank_manifest),
            "bank_manifest_sha256": _sha256(bank_manifest),
            "bank_record_count": len(bank["records"]),
            "pool_parquet": str(pool_parquet),
            "pool_parquet_sha256": _sha256(pool_parquet),
            "processed_root": str(processed_root),
        },
        "summary": _aggregate(records),
        "records": records,
        "elapsed_seconds": time.monotonic() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=args.output.parent, prefix=f".{args.output.name}.", delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    try:
        os.link(temporary, args.output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"status=complete output={args.output}", flush=True)


if __name__ == "__main__":
    main()
