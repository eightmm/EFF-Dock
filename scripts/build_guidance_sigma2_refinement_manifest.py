#!/usr/bin/env python3
"""Freeze the complete sigma-2/eta-2 saved-pose ensemble for refinement."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from effdock.workflows.evaluate import file_sha256

PROTOCOL_ID = "EFFDOCK-GUIDANCE-SIGMA2-ETA2-REFINEMENT-INPUT-V1"
SOURCE_PROTOCOL_ID = "EFFDOCK-UNIFIED-GUIDANCE-SIGMA-SWEEP-ETA2-V1"
DATASETS = {"astex": 85, "posebusters": 308}
SHARDS = 8
SIGMA = 2.0
ETA = 2.0
POSES_PER_COMPLEX = 100


def _source_stem(dataset: str) -> str:
    return f"effdock-guidance-sigma-sweep-eta2-v1-{dataset}-n100-s10-sigma2000"


def _require_asset(path: Path, expected_sha256: str) -> None:
    if not path.is_file() or file_sha256(path) != expected_sha256:
        raise ValueError(f"missing or changed frozen asset: {path}")


def _require_source_summary(path: Path, dataset: str) -> dict[str, Any]:
    summary = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "protocol_id": SOURCE_PROTOCOL_ID,
        "dataset": dataset,
        "sigma": SIGMA,
        "num_samples": POSES_PER_COMPLEX,
        "num_steps": 10,
        "unified_guidance_scale": ETA,
        "unified_guidance_mode": "normalized_drift",
    }
    changed = {
        key: (summary.get(key), value)
        for key, value in expected.items()
        if summary.get(key) != value
    }
    if changed:
        raise ValueError(f"unexpected source summary contract {path}: {changed}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    records: list[dict[str, object]] = []
    source_files: list[dict[str, str]] = []
    source_summaries: list[dict[str, str]] = []
    guidance_parameter_hashes: set[str] = set()
    for dataset, expected_count in DATASETS.items():
        ids: set[str] = set()
        stem = _source_stem(dataset)
        for shard in range(SHARDS):
            suffix = f"shard-{shard:03d}-of-{SHARDS:03d}"
            csv_path = args.source_root / "raw" / f"{stem}.{suffix}.csv"
            summary_path = args.source_root / "raw" / f"{stem}.{suffix}.summary.json"
            if not csv_path.is_file() or not summary_path.is_file():
                raise FileNotFoundError(f"missing source shard: {csv_path} / {summary_path}")
            source_files.append({"path": str(csv_path), "sha256": file_sha256(csv_path)})
            source_summaries.append(
                {"path": str(summary_path), "sha256": file_sha256(summary_path)}
            )
            summary = _require_source_summary(summary_path, dataset)
            guidance_parameter_hashes.add(
                str(summary["guidance_parameter_set"]["sha256"])
            )
            with csv_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            for row in rows:
                complex_id = row["id"].lower()
                if complex_id in ids:
                    raise ValueError(f"duplicate source complex: {dataset}/{complex_id}")
                ids.add(complex_id)
                if int(row["all_poses_count"]) != POSES_PER_COMPLEX:
                    raise ValueError(f"{dataset}/{complex_id}: expected 100 saved poses")
                if row["guidance_mode"] != "unified_normalized_drift":
                    raise ValueError(f"{dataset}/{complex_id}: unexpected guidance mode")
                pose_path = Path(row["all_poses_sdf"])
                protein = Path(row["protein"])
                ligand_ref = Path(row["ligand_ref"])
                _require_asset(pose_path, row["all_poses_sdf_sha256"])
                _require_asset(protein, row["protein_sha256"])
                _require_asset(ligand_ref, row["ligand_reference_sha256"])
                records.append(
                    {
                        "dataset": dataset,
                        "id": complex_id,
                        "eta": ETA,
                        "sigma": SIGMA,
                        "pose_path": str(pose_path),
                        "pose_sha256": row["all_poses_sdf_sha256"],
                        "pose_count": POSES_PER_COMPLEX,
                        "protein": str(protein),
                        "protein_sha256": row["protein_sha256"],
                        "ligand_ref": str(ligand_ref),
                        "ligand_ref_sha256": row["ligand_reference_sha256"],
                        "sampling_seed": int(row["sampling_seed"]),
                        "prior_pool_sha256": row["prior_pool_sha256"],
                        "guidance_mode": row["guidance_mode"],
                        "guidance_parameter_sha256": row[
                            "guidance_parameter_sha256"
                        ],
                    }
                )
        if len(ids) != expected_count:
            raise ValueError(
                f"{dataset}: expected {expected_count} source complexes, got {len(ids)}"
            )
    if len(guidance_parameter_hashes) != 1:
        raise ValueError(
            f"source guidance parameter identity changed: {guidance_parameter_hashes}"
        )
    records.sort(key=lambda row: (str(row["dataset"]), str(row["id"])))
    expected_complexes = sum(DATASETS.values())
    if len(records) != expected_complexes:
        raise ValueError(f"expected {expected_complexes} records, got {len(records)}")
    payload = {
        "schema_version": "effdock.guidance_sigma2_refinement_input.v1",
        "protocol_id": PROTOCOL_ID,
        "source_protocol_id": SOURCE_PROTOCOL_ID,
        "source_root": str(args.source_root),
        "sigma": SIGMA,
        "eta": ETA,
        "num_steps": 10,
        "poses_per_complex": POSES_PER_COMPLEX,
        "datasets": DATASETS,
        "expected_complexes": expected_complexes,
        "expected_poses": expected_complexes * POSES_PER_COMPLEX,
        "guidance_parameter_sha256": next(iter(guidance_parameter_hashes)),
        "source_files": source_files,
        "source_summaries": source_summaries,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "complexes": len(records),
                "poses": payload["expected_poses"],
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
