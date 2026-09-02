#!/usr/bin/env python3
"""Archive failed FoldBench sampling shards and their partial pose files."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

PREFIX = "effdock-foldbench-pocket-558-v1-foldbench-n100-s10-sigma2-unguided"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--archive-dir", type=Path, required=True)
    parser.add_argument("--shards", type=int, nargs="+", required=True)
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    archive_dir = args.archive_dir.resolve()
    if archive_dir.exists():
        raise FileExistsError(archive_dir)

    sampling_root = output_root / "full" / "sampling"
    sources: list[Path] = []
    shard_records: list[dict[str, object]] = []
    for shard in args.shards:
        stem = f"{PREFIX}.shard-{shard:03d}-of-044"
        csv_path = sampling_root / f"{stem}.csv"
        summary_path = sampling_root / f"{stem}.summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if int(summary.get("num_failed", -1)) != 1 or int(summary.get("num_success", -1)) != 12:
            raise ValueError(f"shard {shard} is not the expected 12/13 partial artifact")
        with csv_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != 12:
            raise ValueError(f"shard {shard}: expected 12 successful rows")
        sources.extend((csv_path, summary_path))
        pose_files: list[str] = []
        for row in rows:
            complex_id = str(row["id"]).lower()
            pose_path = (
                sampling_root
                / "poses"
                / PREFIX
                / "foldbench"
                / "all_poses"
                / f"{complex_id}.sdf"
            )
            if not pose_path.is_file():
                raise FileNotFoundError(pose_path)
            sources.append(pose_path)
            pose_files.append(str(pose_path.relative_to(output_root)))
        shard_records.append(
            {
                "shard": shard,
                "successful_ids": sorted(str(row["id"]).lower() for row in rows),
                "failed_ids": sorted(str(row["id"]).lower() for row in summary["failures"]),
                "pose_files": sorted(pose_files),
            }
        )

    entries = []
    for source in sorted(set(sources)):
        relative = source.relative_to(output_root)
        destination = (
            archive_dir / source.name
            if source.parent == sampling_root
            else archive_dir / relative
        )
        entries.append(
            {
                "original": str(source),
                "archive": str(destination),
                "size_bytes": source.stat().st_size,
                "sha256": file_sha256(source),
            }
        )
    archive_dir.mkdir(parents=True)
    for entry in entries:
        source = Path(entry["original"])
        destination = Path(entry["archive"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(source, destination)
    payload = {
        "schema_version": "effdock.foldbench_failed_sampling_archive.v1",
        "status": "archived",
        "output_root": str(output_root),
        "shards": shard_records,
        "files": entries,
    }
    (archive_dir / "manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"status": "archived", "file_count": len(entries)}, sort_keys=True))


if __name__ == "__main__":
    main()
