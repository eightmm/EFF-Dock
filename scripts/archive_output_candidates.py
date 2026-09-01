#!/usr/bin/env python3
"""Move verified cleanup candidates into a recoverable output archive."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--archive-root", type=Path, required=True)
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    outputs = (repo_root / "outputs").resolve()
    inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
    if inventory.get("destructive_actions_performed") is not False:
        raise ValueError("inventory must be a non-destructive snapshot")
    candidates = [
        row
        for row in inventory["levels"]["outputs"]
        if row["classification"] == "cleanup_candidate"
    ]
    if not candidates:
        raise ValueError("inventory contains no cleanup candidates")
    archive_root = (
        args.archive_root
        if args.archive_root.is_absolute()
        else repo_root / args.archive_root
    ).resolve()
    if archive_root.exists():
        raise FileExistsError(f"refusing to reuse archive root: {archive_root}")
    if outputs not in archive_root.parents:
        raise ValueError("archive root must remain inside outputs")

    resolved: list[tuple[Path, Path, dict]] = []
    for row in candidates:
        source = (repo_root / row["path"]).resolve()
        if source.parent != outputs or not source.is_dir():
            raise ValueError(f"unsafe or missing cleanup candidate: {source}")
        if row["repository_reference_count"] != 0:
            raise ValueError(f"referenced path cannot be archived: {source}")
        target = archive_root / source.name
        resolved.append((source, target, row))

    archive_root.mkdir(parents=True)
    moved: list[tuple[Path, Path, dict]] = []
    try:
        for source, target, row in resolved:
            source.rename(target)
            moved.append((source, target, row))
    except Exception:
        for source, target, _ in reversed(moved):
            if target.exists() and not source.exists():
                target.rename(source)
        raise

    record = {
        "schema_version": "effdock.output_archive_manifest.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "operation": "recoverable_move",
        "source_inventory": str(args.inventory.resolve()),
        "source_inventory_created_utc": inventory["created_utc"],
        "archive_root": str(archive_root),
        "count": len(moved),
        "bytes": sum(int(row["bytes"]) for _, _, row in moved),
        "entries": [
            {
                "original_path": str(source.relative_to(repo_root)),
                "archived_path": str(target.relative_to(repo_root)),
                "bytes": int(row["bytes"]),
                "files": int(row["files"]),
                "reason": row["reason"],
            }
            for source, target, row in moved
        ],
    }
    (archive_root / "ARCHIVE_MANIFEST.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"count": record["count"], "bytes": record["bytes"]}))


if __name__ == "__main__":
    main()
