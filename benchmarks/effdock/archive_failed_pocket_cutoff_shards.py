#!/usr/bin/env python3
"""Archive incomplete pocket-cutoff generation shards before a safe retry."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

SUMMARY_RE = re.compile(
    r"effdock-pocket-cutoff-v1-(astex|posebusters)-c(\d{2})-r([0-2])-n100-s10"
    r"\.shard-(\d{3})-of-008\.summary\.json$"
)
CUTOFFS = (6, 8, 10, 12)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def task_id(*, cutoff: int, repeat: int, dataset: str, shard: int) -> int:
    cutoff_index = CUTOFFS.index(cutoff)
    dataset_index = {"astex": 0, "posebusters": 1}[dataset]
    return cutoff_index * 48 + repeat * 16 + dataset_index * 8 + shard


def read_rows(path: Path | None) -> list[dict[str, str]]:
    if path is None or not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-job-id", type=int, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    output_root = args.output_root.resolve()
    archive_root = output_root / "recovery_archive" / f"generation_job_{args.source_job_id}"
    if archive_root.exists():
        raise FileExistsError(f"recovery archive already exists: {archive_root}")

    plans: list[dict[str, Any]] = []
    for summary_path in sorted(output_root.glob("cutoff_*/repeat_*/raw/*.summary.json")):
        match = SUMMARY_RE.fullmatch(summary_path.name)
        if match is None:
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if int(summary.get("num_failed", 0)) == 0:
            continue
        error_types = sorted({str(row.get("error_type")) for row in summary["failures"]})
        if error_types != ["OutOfMemoryError"]:
            raise ValueError(f"unexpected failure types in {summary_path}: {error_types}")

        dataset, cutoff_text, repeat_text, shard_text = match.groups()
        cutoff = int(cutoff_text)
        repeat = int(repeat_text)
        shard = int(shard_text)
        original_task_id = task_id(
            cutoff=cutoff,
            repeat=repeat,
            dataset=dataset,
            shard=shard,
        )
        csv_value = summary.get("csv")
        csv_path = Path(csv_value).resolve() if csv_value else summary_path.with_suffix("")
        if not csv_value:
            csv_path = None
        rows = read_rows(csv_path)
        if len(rows) != int(summary["num_success"]):
            raise ValueError(f"success-row mismatch in {summary_path}")
        complex_ids = {str(row["id"]).lower() for row in rows}
        complex_ids.update(str(row["id"]).lower() for row in summary["failures"])
        if len(complex_ids) != int(summary["num_assigned"]):
            raise ValueError(f"assigned-ID mismatch in {summary_path}")

        sources = [summary_path]
        if csv_path is not None:
            sources.append(csv_path)
        pose_root = summary_path.parent / "poses" / str(summary["run_name"]) / dataset
        for category in ("all_poses", "selected"):
            for complex_id in sorted(complex_ids):
                pose_path = pose_root / category / f"{complex_id}.sdf"
                if pose_path.is_file():
                    sources.append(pose_path)
        files = []
        for source in sorted(set(sources)):
            relative = source.resolve().relative_to(output_root)
            destination = archive_root / f"task_{original_task_id:03d}" / relative
            files.append(
                {
                    "original": str(source.resolve()),
                    "archive": str(destination),
                    "size_bytes": source.stat().st_size,
                    "sha256": file_sha256(source),
                }
            )
        plans.append(
            {
                "task_id": original_task_id,
                "cutoff": cutoff,
                "repeat": repeat,
                "dataset": dataset,
                "shard": shard,
                "num_assigned": int(summary["num_assigned"]),
                "num_success": int(summary["num_success"]),
                "num_failed": int(summary["num_failed"]),
                "complex_ids": sorted(complex_ids),
                "files": files,
            }
        )

    plans.sort(key=lambda row: int(row["task_id"]))
    task_ids = [int(row["task_id"]) for row in plans]
    if len(plans) != 70 or len(set(task_ids)) != 70:
        raise ValueError(f"expected 70 unique failed shards, found {len(plans)}")
    payload = {
        "schema_version": "effdock.pocket_cutoff_generation_recovery.v1",
        "status": "planned" if not args.apply else "archived",
        "source_job_id": args.source_job_id,
        "output_root": str(output_root),
        "failed_task_count": len(plans),
        "failed_task_ids": task_ids,
        "shards": plans,
    }
    print(json.dumps({k: payload[k] for k in payload if k != "shards"}, indent=2))
    if not args.apply:
        return

    archive_root.mkdir(parents=True)
    planned_path = archive_root / "planned.json"
    planned_path.write_text(json.dumps({**payload, "status": "moving"}, indent=2) + "\n")
    for shard in plans:
        for entry in shard["files"]:
            source = Path(entry["original"])
            destination = Path(entry["archive"])
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(source, destination)
    manifest_path = archive_root / "manifest.json"
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n")
    (archive_root / "failed_task_ids.txt").write_text(
        ",".join(str(value) for value in task_ids) + "\n"
    )
    planned_path.unlink()
    print(f"archived {len(plans)} failed shards to {archive_root}")


if __name__ == "__main__":
    main()
