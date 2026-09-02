#!/usr/bin/env python3
"""Aggregate the completed U70k PhiBench Top-5 PoseBusters shards."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from effdock.workflows.evaluate import file_sha256
from scripts.evaluate_phibench_u70k_top5_posebusters_shard import (
    CONFIDENCE_SHA256,
    EXPECTED_COMPLEXES,
    PROTOCOL_ID,
    TOP_K,
    read_json,
)

EXPECTED_REPRODUCTION = {
    "raw_top1_rmsd_lt2": 128,
    "refined_top1_rmsd_lt2": 131,
    "refined_top1_posebusters_valid": 184,
    "refined_top1_joint_posebusters_valid_rmsd_lt2": 120,
}


def count(rows: list[dict[str, Any]], predicate) -> int:
    return sum(bool(predicate(row)) for row in rows)


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {
        "raw_top1_rmsd_lt2": count(rows, lambda row: row["raw_top1_rmsd"] < 2.0),
        "raw_top5_rmsd_lt2": count(rows, lambda row: row["raw_top5_best_rmsd"] < 2.0),
        "raw_oracle_rmsd_lt2": count(rows, lambda row: row["raw_oracle_rmsd"] < 2.0),
        "refined_top1_rmsd_lt2": count(
            rows, lambda row: row["refined_top1_rmsd"] < 2.0
        ),
        "refined_top5_rmsd_lt2": count(
            rows, lambda row: row["refined_top5_best_rmsd"] < 2.0
        ),
        "refined_oracle_rmsd_lt2": count(
            rows, lambda row: row["refined_oracle_rmsd"] < 2.0
        ),
        "refined_top1_posebusters_valid": count(
            rows, lambda row: row["refined_top1_posebusters_valid"]
        ),
        "refined_top5_posebusters_valid": count(
            rows, lambda row: row["refined_top5_posebusters_valid"]
        ),
        "refined_top1_joint_posebusters_valid_rmsd_lt2": count(
            rows, lambda row: row["refined_top1_joint"]
        ),
        "refined_top5_joint_posebusters_valid_rmsd_lt2": count(
            rows, lambda row: row["refined_top5_joint"]
        ),
    }
    if any(counts[key] != value for key, value in EXPECTED_REPRODUCTION.items()):
        raise ValueError(f"released Top-1 reproduction gate failed: {counts}")
    if not (
        counts["raw_top1_rmsd_lt2"]
        <= counts["raw_top5_rmsd_lt2"]
        <= counts["raw_oracle_rmsd_lt2"]
    ):
        raise ValueError("raw Top-1/Top-5/oracle ordering gate failed")
    if not (
        counts["refined_top1_rmsd_lt2"]
        <= counts["refined_top5_rmsd_lt2"]
        <= counts["refined_oracle_rmsd_lt2"]
    ):
        raise ValueError("refined Top-1/Top-5/oracle ordering gate failed")
    if not (
        counts["refined_top1_posebusters_valid"]
        <= counts["refined_top5_posebusters_valid"]
        <= EXPECTED_COMPLEXES
    ):
        raise ValueError("Top-1/Top-5 PB-valid ordering gate failed")
    if not (
        counts["refined_top1_joint_posebusters_valid_rmsd_lt2"]
        <= counts["refined_top5_joint_posebusters_valid_rmsd_lt2"]
        <= counts["refined_top5_rmsd_lt2"]
    ):
        raise ValueError("Top-1/Top-5 joint ordering gate failed")
    return {
        "n": EXPECTED_COMPLEXES,
        "counts": counts,
        "percent": {
            key: 100.0 * value / EXPECTED_COMPLEXES for key, value in counts.items()
        },
        "top5_rescue": {
            "raw_rmsd_count": counts["raw_top5_rmsd_lt2"]
            - counts["raw_top1_rmsd_lt2"],
            "refined_rmsd_count": counts["refined_top5_rmsd_lt2"]
            - counts["refined_top1_rmsd_lt2"],
            "refined_joint_count": counts[
                "refined_top5_joint_posebusters_valid_rmsd_lt2"
            ]
            - counts["refined_top1_joint_posebusters_valid_rmsd_lt2"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    rows: list[dict[str, Any]] = []
    shard_inventory: list[dict[str, Any]] = []
    for shard_index in range(13):
        name = f"phibench.shard-{shard_index:03d}-of-013"
        path = args.input_root / "full" / "shards" / name / "summary.json"
        shard = read_json(path)
        expected_count = len(range(shard_index, EXPECTED_COMPLEXES, 13))
        if (
            shard.get("protocol_id") != PROTOCOL_ID
            or shard.get("status") != "complete_descriptive"
            or shard.get("stage") != "full"
            or shard.get("dataset") != "phibench"
            or int(shard.get("num_shards", -1)) != 13
            or int(shard.get("shard_index", -1)) != shard_index
            or int(shard.get("num_complexes", -1)) != expected_count
            or int(shard.get("num_posebusters_evaluations", -1))
            != expected_count * TOP_K
            or shard.get("confidence_checkpoint_sha256") != CONFIDENCE_SHA256
        ):
            raise ValueError(f"invalid Top-5 shard: {path}")
        rows.extend(shard["records"])
        shard_inventory.append({"path": str(path), "sha256": file_sha256(path)})
    ids = [str(row["id"]) for row in rows]
    if len(rows) != EXPECTED_COMPLEXES or len(set(ids)) != EXPECTED_COMPLEXES:
        raise ValueError("PhiBench Top-5 complex inventory mismatch")
    if sum(len(row["top5"]) for row in rows) != EXPECTED_COMPLEXES * TOP_K:
        raise ValueError("PhiBench Top-5 pose inventory mismatch")

    result = aggregate(rows)
    report = {
        "schema_version": "effdock.phibench_u70k_top5_report.v1",
        "protocol_id": PROTOCOL_ID,
        "status": "complete_descriptive",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "checkpoint": "U70k",
        "confidence_checkpoint_sha256": CONFIDENCE_SHA256,
        "ranking": "ascending predicted RMSD with stable pose-index tie-break",
        "top_k": TOP_K,
        "aggregate": result,
        "shards": shard_inventory,
        "claim_boundary": (
            "Closest available endpoint-aligned PhiBench context, not a direct "
            "comparison: EFF-Dock uses a derived 203-system cohort while the "
            "PhysDock source-native panel contains 206 systems."
        ),
    }
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    attempt = Path(
        tempfile.mkdtemp(prefix=f".{args.output_dir.name}.", dir=args.output_dir.parent)
    )
    ordered = sorted(rows, key=lambda row: str(row["id"]))
    with (attempt / "per_complex.jsonl").open("w", encoding="utf-8") as handle:
        for row in ordered:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    (attempt / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    values = result["counts"]
    percents = result["percent"]
    lines = [
        "# PhiBench U70k Top-5 descriptive reanalysis",
        "",
        "| Endpoint | Top-1 | Top-5 | Oracle-100 |",
        "|---|---:|---:|---:|",
        (
            "| Raw RMSD `<2 A` | "
            f"{values['raw_top1_rmsd_lt2']}/203 ({percents['raw_top1_rmsd_lt2']:.2f}%) | "
            f"{values['raw_top5_rmsd_lt2']}/203 ({percents['raw_top5_rmsd_lt2']:.2f}%) | "
            f"{values['raw_oracle_rmsd_lt2']}/203 ({percents['raw_oracle_rmsd_lt2']:.2f}%) |"
        ),
        (
            "| Refined RMSD `<2 A` | "
            f"{values['refined_top1_rmsd_lt2']}/203 ({percents['refined_top1_rmsd_lt2']:.2f}%) | "
            f"{values['refined_top5_rmsd_lt2']}/203 ({percents['refined_top5_rmsd_lt2']:.2f}%) | "
            f"{values['refined_oracle_rmsd_lt2']}/203 ({percents['refined_oracle_rmsd_lt2']:.2f}%) |"
        ),
        (
            "| Refined PB-valid | "
            f"{values['refined_top1_posebusters_valid']}/203 ({percents['refined_top1_posebusters_valid']:.2f}%) | "
            f"{values['refined_top5_posebusters_valid']}/203 ({percents['refined_top5_posebusters_valid']:.2f}%) | -- |"
        ),
        (
            "| Refined joint PB-valid + RMSD `<2 A` | "
            f"{values['refined_top1_joint_posebusters_valid_rmsd_lt2']}/203 "
            f"({percents['refined_top1_joint_posebusters_valid_rmsd_lt2']:.2f}%) | "
            f"{values['refined_top5_joint_posebusters_valid_rmsd_lt2']}/203 "
            f"({percents['refined_top5_joint_posebusters_valid_rmsd_lt2']:.2f}%) | -- |"
        ),
        "",
        report["claim_boundary"],
        "",
    ]
    (attempt / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    os.rename(attempt, args.output_dir)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
