#!/usr/bin/env python3
"""Run official PoseBusters 0.6.5 on confidence-selected PLINDER poses."""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import io
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from plinder_guidance_common import (
    ETA_TAGS,
    ETA_VALUES,
    EXPECTED_COUNT,
    EXPECTED_POSEBUSTERS_VERSION,
    POSEBUSTERS_SCHEMA,
    PRIMARY_SELECTOR,
    PROTOCOL_ID,
    expected_ids,
    expected_num_shards,
    file_sha256,
    ids_sha256,
    load_csv,
    load_split_ids,
    posebusters_shard_dir,
    sampling_shard_dir,
    validate_passed_audit,
    write_bytes_noreplace,
    write_json_noreplace,
)
from posebusters import PoseBusters

from effdock.workflows.guidance_budget_posebusters_report import VALIDITY_CHECKS
from effdock.workflows.posebusters_report import (
    require_posebusters_runtime_version,
    verify_posebusters_input_hashes,
)


@dataclass(frozen=True)
class PoseBustersAttempt:
    attempt_dir: Path
    final_dir: Path
    publish_lock: Path


def reserve_posebusters_attempt(
    output_root: Path,
    *,
    mode: str,
    eta: float,
    shard_index: int,
    num_shards: int,
) -> PoseBustersAttempt:
    final_dir = posebusters_shard_dir(
        output_root.resolve(), mode, eta, shard_index, num_shards
    )
    if final_dir.exists():
        raise FileExistsError(f"refusing to rerun completed PoseBusters shard: {final_dir}")
    arm_root = final_dir.parent
    arm_root.mkdir(parents=True, exist_ok=True)
    incomplete_root = arm_root / ".incomplete"
    incomplete_root.mkdir(exist_ok=True)
    shard_name = final_dir.name
    attempt_dir = Path(
        tempfile.mkdtemp(prefix=f"{shard_name}.attempt-", dir=incomplete_root)
    )
    return PoseBustersAttempt(
        attempt_dir=attempt_dir,
        final_dir=final_dir,
        publish_lock=incomplete_root / f".{shard_name}.publish.lock",
    )


def publish_posebusters_attempt(attempt: PoseBustersAttempt) -> Path:
    """Atomically expose one complete official shard while preserving failed attempts."""
    with attempt.publish_lock.open("a+b") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        if attempt.final_dir.exists():
            raise FileExistsError(
                f"refusing duplicate PoseBusters publish: {attempt.final_dir}"
            )
        try:
            os.rename(attempt.attempt_dir, attempt.final_dir)
        except OSError as exc:
            raise RuntimeError(
                f"atomic PoseBusters publish failed; retained {attempt.attempt_dir}"
            ) from exc
    return attempt.final_dir


def _audit_shard_record(
    audit: dict[str, Any], *, eta: float, shard_index: int
) -> dict[str, Any]:
    cells = audit.get("cells")
    if not isinstance(cells, list):
        raise ValueError("sampling audit lacks cell inventory")
    matching_cells = [cell for cell in cells if isinstance(cell, dict) and cell.get("eta") == eta]
    if len(matching_cells) != 1:
        raise ValueError(f"sampling audit does not bind exactly one eta={eta} cell")
    shards = matching_cells[0].get("shards")
    if not isinstance(shards, list):
        raise ValueError(f"sampling audit eta={eta} lacks shard inventory")
    matching_shards = [
        shard
        for shard in shards
        if isinstance(shard, dict) and shard.get("shard_index") == shard_index
    ]
    if len(matching_shards) != 1:
        raise ValueError(f"sampling audit does not bind eta={eta} shard={shard_index}")
    return matching_shards[0]


def validate_official_checks(raw: dict[str, Any], *, sample_id: str) -> tuple[str, dict[str, bool]]:
    rmsd_checks = [key for key in raw if key.startswith("rmsd_")]
    if len(rmsd_checks) != 1:
        raise ValueError(f"{sample_id}: PoseBusters must emit exactly one separate RMSD check")
    expected = {*VALIDITY_CHECKS, rmsd_checks[0]}
    if set(raw) != expected:
        missing = sorted(expected - set(raw))
        extra = sorted(set(raw) - expected)
        raise ValueError(
            f"{sample_id}: PoseBusters 0.6.5 redock schema mismatch; "
            f"missing={missing}, extra={extra}"
        )
    checks = {
        key: False if pd.isna(value) else bool(value)
        for key, value in raw.items()
    }
    return rmsd_checks[0], checks


def _csv_bytes(rows: list[dict[str, Any]], *, rmsd_check: str) -> bytes:
    fields = ["id", "posebusters_valid", rmsd_check, *VALIDITY_CHECKS]
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="raise")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def run(
    *,
    sampling_root: Path,
    audit_path: Path,
    split_file: Path,
    output_root: Path,
    mode: str,
    eta: float,
    num_shards: int,
    shard_index: int,
) -> dict[str, Any]:
    if eta not in ETA_VALUES:
        raise ValueError(f"eta must be one of {ETA_VALUES}")
    expected_shards = expected_num_shards(mode)
    if num_shards != expected_shards or not 0 <= shard_index < num_shards:
        raise ValueError(f"{mode} PoseBusters requires exact shard contract {expected_shards}")
    sampling_root = sampling_root.resolve()
    audit = validate_passed_audit(audit_path, mode=mode, sampling_root=sampling_root)
    split_ids = load_split_ids(split_file)
    selected_ids = expected_ids(mode, split_ids)
    if (
        audit.get("selected_count") != len(selected_ids)
        or audit.get("selected_ids_sha256") != ids_sha256(selected_ids)
        or audit.get("num_shards") != num_shards
    ):
        raise ValueError("sampling audit selected-cohort contract mismatch")
    assigned_ids = selected_ids[shard_index::num_shards]
    shard_dir = sampling_shard_dir(
        sampling_root, mode, eta, shard_index, num_shards
    )
    sampling_csv = shard_dir / "results.csv"
    sampling_summary = shard_dir / "summary.json"
    binding = _audit_shard_record(audit, eta=eta, shard_index=shard_index)
    expected_bindings = {
        "assigned_count": len(assigned_ids),
        "assigned_ids_sha256": ids_sha256(assigned_ids),
        "csv": str(sampling_csv.resolve()),
        "csv_sha256": file_sha256(sampling_csv),
        "summary": str(sampling_summary.resolve()),
        "summary_sha256": file_sha256(sampling_summary),
    }
    for key, expected in expected_bindings.items():
        if binding.get(key) != expected:
            raise ValueError(f"sampling audit shard binding mismatch for {key}")
    _, sampling_rows = load_csv(sampling_csv)
    if [row.get("id") for row in sampling_rows] != assigned_ids:
        raise ValueError("sampling rows differ from exact assigned PoseBusters IDs")

    attempt = reserve_posebusters_attempt(
        output_root,
        mode=mode,
        eta=eta,
        shard_index=shard_index,
        num_shards=num_shards,
    )

    observed_version = require_posebusters_runtime_version()
    if observed_version != EXPECTED_POSEBUSTERS_VERSION:
        raise RuntimeError("unexpected PoseBusters version")
    buster = PoseBusters(config="redock", max_workers=0)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    rmsd_check: str | None = None
    for index, (sample_id, sampling_row) in enumerate(
        zip(assigned_ids, sampling_rows, strict=True), start=1
    ):
        pose_path = shard_dir / "poses" / PRIMARY_SELECTOR / f"{sample_id}.sdf"
        try:
            verify_posebusters_input_hashes(sampling_row, pose_path, PRIMARY_SELECTOR)
            frame = buster.bust(
                pose_path,
                Path(sampling_row["ligand_ref"]),
                Path(sampling_row["protein"]),
                full_report=False,
            )
            if len(frame.index) != 1:
                raise ValueError("PoseBusters returned more or fewer than one result row")
            current_rmsd_check, checks = validate_official_checks(
                frame.iloc[0].to_dict(), sample_id=sample_id
            )
            if rmsd_check is None:
                rmsd_check = current_rmsd_check
            elif current_rmsd_check != rmsd_check:
                raise ValueError("PoseBusters RMSD check column changed within a shard")
            validity = all(checks[key] for key in VALIDITY_CHECKS)
            rows.append(
                {
                    "id": sample_id,
                    "posebusters_valid": validity,
                    current_rmsd_check: checks[current_rmsd_check],
                    **{key: checks[key] for key in VALIDITY_CHECKS},
                }
            )
            print(f"[{index:04d}/{len(assigned_ids)}] {sample_id} valid={validity}")
        except Exception as exc:
            failures.append(
                {"id": sample_id, "error_type": type(exc).__name__, "message": str(exc)}
            )
            print(f"[{index:04d}/{len(assigned_ids)}] {sample_id} FAIL {exc!r}")

    if rmsd_check is None:
        # Preserve a parseable failure artifact while returning nonzero below.
        rmsd_check = "rmsd_≤_2å"
    csv_payload = _csv_bytes(rows, rmsd_check=rmsd_check)
    stored_csv_path = attempt.attempt_dir / "results.csv"
    final_csv_path = attempt.final_dir / "results.csv"
    write_bytes_noreplace(stored_csv_path, csv_payload)
    complete = len(rows) == len(assigned_ids) and not failures
    summary: dict[str, Any] = {
        "schema_version": POSEBUSTERS_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "status": "complete" if complete else "failed",
        "mode": mode,
        "eta": eta,
        "eta_tag": ETA_TAGS[eta],
        "primary_selector": PRIMARY_SELECTOR,
        "posebusters_version": observed_version,
        "config": "redock",
        "pass_all_definition": "all 27 non-RMSD redock checks",
        "validity_checks": list(VALIDITY_CHECKS),
        "rmsd_check": rmsd_check,
        "expected_denominator": EXPECTED_COUNT,
        "selected_cohort_count": len(selected_ids),
        "num_shards": num_shards,
        "shard_index": shard_index,
        "assigned_count": len(assigned_ids),
        "assigned_ids": assigned_ids,
        "assigned_ids_sha256": ids_sha256(assigned_ids),
        "success_count": len(rows),
        "failure_count": len(failures),
        "failures": failures,
        "sampling_audit": str(audit_path.resolve()),
        "sampling_audit_sha256": file_sha256(audit_path),
        "sampling_csv": str(sampling_csv.resolve()),
        "sampling_csv_sha256": file_sha256(sampling_csv),
        "sampling_summary": str(sampling_summary.resolve()),
        "sampling_summary_sha256": file_sha256(sampling_summary),
        "artifacts": {
            "csv": str(final_csv_path.resolve()),
            "csv_sha256": hashlib.sha256(csv_payload).hexdigest(),
            "summary": str((attempt.final_dir / "summary.json").resolve()),
        },
    }
    summary_path = attempt.attempt_dir / "summary.json"
    write_json_noreplace(summary_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not complete:
        raise RuntimeError(
            "official PoseBusters shard incomplete: "
            f"assigned={len(assigned_ids)} success={len(rows)} failures={len(failures)}; "
            f"attempt retained at {attempt.attempt_dir}"
        )
    publish_posebusters_attempt(attempt)
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sampling-root", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, default=Path("data/splits/plinder.json"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument("--eta", type=float, choices=ETA_VALUES, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    args = parser.parse_args(argv)
    run(
        sampling_root=args.sampling_root,
        audit_path=args.audit,
        split_file=args.split_file,
        output_root=args.output_root,
        mode=args.mode,
        eta=args.eta,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
    )


if __name__ == "__main__":
    main()
