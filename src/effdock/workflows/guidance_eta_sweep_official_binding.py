#!/usr/bin/env python3
"""Bind one official PoseBusters shard to its exact eta-sweep sampling inputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from effdock.workflows.evaluate import sorted_id_sha256
from effdock.workflows.posebusters_report import file_sha256, load_rows


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _row_ledger(rows: list[dict[str, str]]) -> str:
    digest = hashlib.sha256()
    digest.update(b"EFFDOCK_ETA_SWEEP_OFFICIAL_INPUT_LEDGER_V1\0")
    for row in rows:
        try:
            pose_hashes = json.loads(row["saved_pose_sha256_json"])
        except (KeyError, json.JSONDecodeError) as exc:
            raise ValueError(f"{row.get('id')}: invalid saved-pose hash ledger") from exc
        oracle_hash = pose_hashes.get("oracle") if isinstance(pose_hashes, dict) else None
        record = {
            "id": row["id"],
            "protein_sha256": row["protein_sha256"],
            "ligand_reference_sha256": row["ligand_reference_sha256"],
            "oracle_pose_sha256": oracle_hash,
            "sampling_row_sha256": hashlib.sha256(
                _canonical_json(row).encode()
            ).hexdigest(),
        }
        for key in (
            "protein_sha256",
            "ligand_reference_sha256",
            "oracle_pose_sha256",
            "sampling_row_sha256",
        ):
            value = record[key]
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError(f"{row['id']}: invalid {key}")
            int(value, 16)
        digest.update(_canonical_json(record).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def build_binding(
    *,
    sampling_dir: Path,
    official_dir: Path,
    run_name: str,
    protocol_id: str,
    dataset: str,
    eta: float,
    shard_index: int,
    num_shards: int,
) -> dict[str, Any]:
    if not 0 <= shard_index < num_shards:
        raise ValueError("shard_index must satisfy 0 <= shard_index < num_shards")
    all_rows = load_rows(sampling_dir, run_name)
    rows = all_rows[shard_index::num_shards]
    if not rows:
        raise ValueError("official binding has no assigned sampling rows")
    tag = f"shard-{shard_index:03d}-of-{num_shards:03d}"
    sampling_summary = sampling_dir / f"{run_name}.{tag}.summary.json"
    official_summary = official_dir / f"{tag}.summary.json"
    official_csv = official_dir / f"{tag}.csv"
    for path in (sampling_summary, official_summary, official_csv):
        if not path.is_file():
            raise FileNotFoundError(f"missing binding input: {path}")

    sampling_meta = json.loads(sampling_summary.read_text())
    expected_sampling = {
        "protocol_id": protocol_id,
        "run_name": run_name,
        "dataset": dataset,
        "shard_index": shard_index,
        "num_shards": num_shards,
    }
    for key, expected in expected_sampling.items():
        if sampling_meta.get(key) != expected:
            raise ValueError(f"{sampling_summary}: {key} mismatch")
    if not math.isclose(
        float(sampling_meta.get("unified_guidance_scale", math.nan)),
        eta,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(f"{sampling_summary}: eta mismatch")

    official_meta = json.loads(official_summary.read_text())
    if official_meta.get("input_hashes_verified") is not True:
        raise ValueError(f"{official_summary}: input hashes were not verified")
    if int(official_meta.get("num_input_hashes_verified", -1)) != len(rows):
        raise ValueError(f"{official_summary}: verified input count mismatch")
    with official_csv.open(newline="") as handle:
        official_ids = [row["id"] for row in csv.DictReader(handle)]
    expected_ids = [row["id"] for row in rows]
    if official_ids != expected_ids:
        raise ValueError(f"{official_csv}: official IDs/order differ from sampling shard")

    return {
        "binding_contract": "EFFDOCK_ETA_SWEEP_OFFICIAL_BINDING_V1",
        "protocol_id": protocol_id,
        "run_name": run_name,
        "dataset": dataset,
        "eta": eta,
        "shard_index": shard_index,
        "num_shards": num_shards,
        "count": len(rows),
        "ids_sha256": sorted_id_sha256(expected_ids),
        "sampling_input_pose_ledger_sha256": _row_ledger(rows),
        "sampling_summary_sha256": file_sha256(sampling_summary),
        "official_summary_sha256": file_sha256(official_summary),
        "official_csv_sha256": file_sha256(official_csv),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sampling-dir", type=Path, required=True)
    parser.add_argument("--official-dir", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--protocol-id", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--eta", type=float, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    args = parser.parse_args(argv)
    binding = build_binding(
        sampling_dir=args.sampling_dir,
        official_dir=args.official_dir,
        run_name=args.run_name,
        protocol_id=args.protocol_id,
        dataset=args.dataset,
        eta=args.eta,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
    )
    tag = f"shard-{args.shard_index:03d}-of-{args.num_shards:03d}"
    output = args.official_dir / f"{tag}.binding.json"
    output.write_text(json.dumps(binding, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(binding, sort_keys=True))


if __name__ == "__main__":
    main()
