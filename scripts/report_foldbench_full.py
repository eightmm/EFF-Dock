#!/usr/bin/env python3
"""Aggregate the complete FoldBench pocket-redocking adaptation."""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from effdock.workflows.evaluate import file_sha256
from scripts.report_external_temporal_benchmark import (
    BOOL_FIELDS,
    FLOAT_FIELDS,
    parse_bool,
    summarize,
)

PROTOCOL_ID = "EFFDOCK-FOLDBENCH-POCKET-558-V1"
EXPECTED_COUNT = 558
NUM_SHARDS = 44
TEMPORAL_CUTOFF = "2024-06-30"


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--selector-checkpoint-sha256", required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    manifest = read_json(args.manifest)
    if (
        manifest.get("dataset") != "foldbench"
        or manifest.get("cohort") != "full"
        or int(manifest.get("count", -1)) != EXPECTED_COUNT
        or manifest.get("failures") != []
    ):
        raise ValueError("invalid complete FoldBench manifest")
    manifest_ids = {str(record["id"]) for record in manifest["records"]}
    if len(manifest_ids) != EXPECTED_COUNT:
        raise ValueError("FoldBench manifest IDs are not unique")

    all_rows: list[dict[str, Any]] = []
    schemas: set[tuple[str, ...]] = set()
    posebusters_cpu_seconds = 0.0
    for shard_index in range(NUM_SHARDS):
        shard_name = f"foldbench.shard-{shard_index:03d}-of-{NUM_SHARDS:03d}"
        shard_dir = args.input_root / "full" / "posebusters" / shard_name
        summary = read_json(shard_dir / "summary.json")
        if (
            summary.get("protocol_id") != PROTOCOL_ID
            or summary.get("status") != "complete"
            or summary.get("dataset") != "foldbench"
            or int(summary.get("num_shards", -1)) != NUM_SHARDS
            or int(summary.get("shard_index", -1)) != shard_index
            or summary.get("confidence_checkpoint_sha256")
            != args.selector_checkpoint_sha256
        ):
            raise ValueError(f"invalid FoldBench result shard: {shard_dir}")
        posebusters_cpu_seconds += float(summary["runtime"]["elapsed_seconds"])
        with (shard_dir / "results.csv").open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            schemas.add(tuple(reader.fieldnames or ()))
            rows = list(reader)
        if len(rows) != int(summary["num_results"]):
            raise ValueError(f"row count mismatch: {shard_dir}")
        for row in rows:
            for field in BOOL_FIELDS:
                row[field] = parse_bool(str(row[field]))
            for field in FLOAT_FIELDS:
                row[field] = float(row[field])
        all_rows.extend(rows)

    if len(schemas) != 1 or len(all_rows) != EXPECTED_COUNT:
        raise ValueError("incomplete or schema-inconsistent FoldBench result inventory")
    result_ids = {str(row["id"]) for row in all_rows}
    if result_ids != manifest_ids:
        raise ValueError("FoldBench result IDs do not match the frozen manifest")

    postcut_ids = {
        str(record["id"])
        for record in manifest["records"]
        if str(record["initial_release_date"])[:10] > TEMPORAL_CUTOFF
    }
    if len(postcut_ids) != 66:
        raise ValueError(f"expected 66 post-cutoff interfaces, found {len(postcut_ids)}")
    imputed_ids = set(manifest.get("reference_heavy_atom_imputations", {}))
    if len(imputed_ids) != 2 or not imputed_ids <= manifest_ids:
        raise ValueError("unexpected reference heavy-atom imputation inventory")

    slices = {
        "all_558": all_rows,
        "postcut_66": [row for row in all_rows if str(row["id"]) in postcut_ids],
        "pre_or_on_cutoff_492": [
            row for row in all_rows if str(row["id"]) not in postcut_ids
        ],
        "fully_observed_reference_556": [
            row for row in all_rows if str(row["id"]) not in imputed_ids
        ],
    }
    summaries = {name: summarize(name, rows) for name, rows in slices.items()}
    report = {
        "schema_version": "effdock.foldbench_full_report.v1",
        "protocol_id": PROTOCOL_ID,
        "status": "complete_descriptive",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "manifest": {"path": str(args.manifest), "sha256": file_sha256(args.manifest)},
        "sampling": {
            "num_samples": 100,
            "num_steps": 10,
            "sigma": 2.0,
            "time_schedule": "late_power_3",
            "pocket_cutoff_angstrom": 10.0,
            "guidance_mode": "none",
        },
        "refinement": {"maximum_steps": 100, "adaptive_energy_plateau": True},
        "selector": "U70k stable argmin predicted symmetry-aware RMSD",
        "selector_checkpoint_sha256": args.selector_checkpoint_sha256,
        "reference_heavy_atom_imputation_ids": sorted(imputed_ids),
        "slices": summaries,
        "aggregate_posebusters_cpu_seconds": posebusters_cpu_seconds,
        "claim_boundary": (
            "descriptive holo-pocket redocking adaptation, not the native FoldBench "
            "cofolding leaderboard; the older 492-interface slice may overlap training data"
        ),
    }

    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    attempt = Path(tempfile.mkdtemp(prefix=f".{args.output_dir.name}.", dir=args.output_dir.parent))
    with (attempt / "per_complex.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(next(iter(schemas)))
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(sorted(all_rows, key=lambda row: str(row["id"])))
    (attempt / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# FoldBench-Pocket-558 results",
        "",
        "| Slice | N | Raw Top-1 <2A | Refined Top-1 <2A | Refined oracle <2A | PB-valid | Joint PB-valid + <2A |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, result in summaries.items():
        values = result["percent"]
        lines.append(
            f"| {name} | {result['n']} | {values['raw_top1_rmsd_lt2']:.2f}% | "
            f"{values['refined_top1_rmsd_lt2']:.2f}% | "
            f"{values['refined_oracle_rmsd_lt2']:.2f}% | "
            f"{values['refined_top1_posebusters_valid']:.2f}% | "
            f"{values['refined_top1_joint_posebusters_valid_rmsd_lt2']:.2f}% |"
        )
    lines.extend(("", report["claim_boundary"] + ".", ""))
    (attempt / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    os.rename(attempt, args.output_dir)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
