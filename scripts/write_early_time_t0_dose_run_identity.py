"""Atomically bind a paired t=0-dose branch to its Slurm producer and artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from effdock.checkpoint import load_checkpoint_file


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_identity(path: Path, args: argparse.Namespace) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        identity = json.load(handle)
    expected = {
        "producer_array_job_id": args.array_job_id,
        "producer_job_id": args.job_id,
        "producer_task_id": args.task_id,
    }
    mismatches = {
        key: (identity.get(key), value)
        for key, value in expected.items()
        if identity.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"run identity producer mismatch: {mismatches}")
    return identity


def _add_producer_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--array-job-id", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--task-id", type=int, required=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="action", required=True)

    start = subparsers.add_parser("start")
    _add_producer_args(start)
    start.add_argument("--branch", required=True)
    start.add_argument("--mode", choices=("fresh", "resume"), required=True)
    start.add_argument("--config", type=Path, required=True)
    start.add_argument("--config-sha256", required=True)
    start.add_argument("--split", type=Path, required=True)
    start.add_argument("--split-sha256", required=True)
    start.add_argument("--common-init", type=Path, required=True)
    start.add_argument("--common-init-sha256", required=True)
    start.add_argument("--common-init-step", type=int, required=True)
    start.add_argument("--code-sha256", required=True)
    start.add_argument("--output-dir", type=Path, required=True)
    start.add_argument("--expected-steps", type=int, required=True)

    complete = subparsers.add_parser("complete")
    _add_producer_args(complete)
    complete.add_argument("--latest", type=Path, required=True)
    complete.add_argument("--endpoint", type=Path, required=True)
    complete.add_argument("--baseline", type=Path, required=True)
    complete.add_argument("--expected-steps", type=int, required=True)

    failed = subparsers.add_parser("fail")
    _add_producer_args(failed)
    failed.add_argument("--exit-code", type=int, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.action == "start":
        if args.manifest.exists():
            with args.manifest.open(encoding="utf-8") as handle:
                previous = json.load(handle)
            if previous.get("status") == "completed":
                raise RuntimeError("refusing to replace a completed run identity")
        identity = {
            "schema_version": 1,
            "status": "started",
            "started_at": _utc_now(),
            "producer_array_job_id": args.array_job_id,
            "producer_job_id": args.job_id,
            "producer_task_id": args.task_id,
            "branch": args.branch,
            "mode": args.mode,
            "config": str(args.config),
            "config_sha256": args.config_sha256,
            "split": str(args.split),
            "split_sha256": args.split_sha256,
            "common_init": str(args.common_init),
            "common_init_sha256": args.common_init_sha256,
            "common_init_step": args.common_init_step,
            "code_sha256": args.code_sha256,
            "output_dir": str(args.output_dir),
            "expected_steps": args.expected_steps,
        }
        _atomic_write_json(identity, args.manifest)
        return 0

    identity = _load_identity(args.manifest, args)
    if args.action == "fail":
        if identity.get("status") != "completed":
            identity.update(
                {
                    "status": "failed",
                    "failed_at": _utc_now(),
                    "exit_code": args.exit_code,
                }
            )
            _atomic_write_json(identity, args.manifest)
        return 0

    expected_steps = int(args.expected_steps)
    artifacts = {
        "baseline": (args.baseline, 0),
        "endpoint": (args.endpoint, expected_steps),
        "latest": (args.latest, expected_steps),
    }
    artifact_records: dict[str, dict[str, Any]] = {}
    for name, (path, expected_step) in artifacts.items():
        checkpoint = load_checkpoint_file(path)
        actual_step = int(checkpoint.get("step", -1))
        if actual_step != expected_step:
            raise RuntimeError(
                f"{name} checkpoint step mismatch: {actual_step} != {expected_step}"
            )
        artifact_records[name] = {
            "path": str(path),
            "sha256": _sha256(path),
            "step": actual_step,
        }
    identity.update(
        {
            "status": "completed",
            "completed_at": _utc_now(),
            "artifacts": artifact_records,
        }
    )
    _atomic_write_json(identity, args.manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
