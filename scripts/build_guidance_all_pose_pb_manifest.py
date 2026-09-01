#!/usr/bin/env python3
"""Freeze all saved eta-sweep poses into one content-addressed PB manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

PROTOCOL_ID = "EFFDOCK-GUIDANCE-ALL-POSE-PB-ETA-V1"
ETAS = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
REFERENCE_ETAS = ETAS[:5]
DATASETS = {"astex": 85, "posebusters": 308}
SHARDS = 8
POSES_PER_CELL = 100


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def eta_tag(eta: float) -> str:
    return f"eta{int(round(eta * 1000)):04d}"


def source_csv(reference_root: Path, extension_root: Path, dataset: str, eta: float, shard: int) -> Path:
    tag = eta_tag(eta)
    if eta in REFERENCE_ETAS:
        prefix = f"effdock-guidance-steric-high-eta-v1-{dataset}-n100-s10-{tag}"
        root = reference_root
    else:
        prefix = f"effdock-guidance-eta-cap-extension-v1-{dataset}-n100-s10-{tag}"
        root = extension_root
    return root / "raw" / f"{prefix}.shard-{shard:03d}-of-{SHARDS:03d}.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--extension-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records: list[dict[str, object]] = []
    source_files: list[dict[str, object]] = []
    for dataset, expected_count in DATASETS.items():
        for eta in ETAS:
            ids: set[str] = set()
            for shard in range(SHARDS):
                path = source_csv(args.reference_root, args.extension_root, dataset, eta, shard)
                if not path.is_file():
                    raise FileNotFoundError(path)
                source_files.append({"path": str(path), "sha256": file_sha256(path)})
                with path.open(newline="", encoding="utf-8") as handle:
                    rows = list(csv.DictReader(handle))
                for row in rows:
                    complex_id = row["id"]
                    if complex_id in ids:
                        raise ValueError(f"duplicate {dataset}/{eta}/{complex_id}")
                    ids.add(complex_id)
                    if int(row["all_poses_count"]) != POSES_PER_CELL:
                        raise ValueError(f"{dataset}/{eta}/{complex_id}: expected 100 poses")
                    pose_path = Path(row["all_poses_sdf"])
                    protein = Path(row["protein"])
                    ligand_ref = Path(row["ligand_ref"])
                    expected_hashes = {
                        pose_path: row["all_poses_sdf_sha256"],
                        protein: row["protein_sha256"],
                        ligand_ref: row["ligand_reference_sha256"],
                    }
                    for asset, expected_hash in expected_hashes.items():
                        if not asset.is_file() or file_sha256(asset) != expected_hash:
                            raise ValueError(f"changed or missing frozen asset: {asset}")
                    records.append(
                        {
                            "dataset": dataset,
                            "eta": eta,
                            "eta_tag": eta_tag(eta),
                            "id": complex_id,
                            "pose_path": str(pose_path),
                            "pose_sha256": row["all_poses_sdf_sha256"],
                            "pose_count": POSES_PER_CELL,
                            "protein": str(protein),
                            "protein_sha256": row["protein_sha256"],
                            "ligand_ref": str(ligand_ref),
                            "ligand_ref_sha256": row["ligand_reference_sha256"],
                            "sampling_seed": int(row["sampling_seed"]),
                            "prior_pool_sha256": row["prior_pool_sha256"],
                        }
                    )
            if len(ids) != expected_count:
                raise ValueError(f"{dataset}/{eta}: expected {expected_count} IDs, got {len(ids)}")
    records.sort(key=lambda row: (str(row["dataset"]), float(row["eta"]), str(row["id"])))
    expected_cells = sum(DATASETS.values()) * len(ETAS)
    if len(records) != expected_cells:
        raise ValueError(f"expected {expected_cells} cells, got {len(records)}")
    payload = {
        "schema_version": "effdock.guidance_all_pose_pb_manifest.v1",
        "protocol_id": PROTOCOL_ID,
        "posebusters_version": "0.6.5",
        "posebusters_config": "redock",
        "validity_definition": "all 27 non-RMSD redock checks",
        "etas": list(ETAS),
        "datasets": DATASETS,
        "poses_per_cell": POSES_PER_CELL,
        "expected_cells": expected_cells,
        "expected_poses": expected_cells * POSES_PER_CELL,
        "reference_root": str(args.reference_root),
        "extension_root": str(args.extension_root),
        "source_files": source_files,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "cells": len(records), "poses": payload["expected_poses"]}, sort_keys=True))


if __name__ == "__main__":
    main()
