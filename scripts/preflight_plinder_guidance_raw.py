#!/usr/bin/env python3
"""Freeze and verify every raw PLINDER archive and validation structure asset."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import google_crc32c
from plinder_guidance_common import (
    EXPECTED_COUNT,
    EXPECTED_RAW_ARCHIVE_BYTES,
    EXPECTED_RAW_ARCHIVES,
    EXPECTED_SPLIT_SHA256,
    EXPECTED_UNIQUE_SYSTEMS,
    PROTOCOL_ID,
    RAW_GATE_SCHEMA,
    canonical_json_sha256,
    file_sha256,
    ids_sha256,
    load_split_ids,
    verify_raw_manifest,
    write_bytes_noreplace,
    write_json_noreplace,
)


def _base64_md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return base64.b64encode(digest.digest()).decode("ascii")


def _base64_crc32c(path: Path) -> str:
    checksum = google_crc32c.Checksum()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            checksum.update(chunk)
    return base64.b64encode(checksum.digest()).decode("ascii")


def _verify_archive(record: dict[str, Any], *, raw_root: Path) -> dict[str, Any]:
    code = record.get("two_char_code")
    relative = record.get("relative_path")
    gcs = record.get("gcs")
    if not isinstance(code, str) or not isinstance(relative, str) or not isinstance(gcs, dict):
        raise ValueError("raw manifest archive record is incomplete")
    path = raw_root / relative
    if path.resolve().parent != (raw_root / "systems").resolve():
        raise ValueError(f"unsafe archive path in raw manifest: {relative}")
    if path.name != f"{code}.zip" or not path.is_file():
        raise FileNotFoundError(f"missing frozen PLINDER archive: {path}")
    expected_size = gcs.get("size_bytes")
    expected_md5 = gcs.get("md5_hash_base64")
    expected_crc32c = gcs.get("crc32c_base64")
    generation = gcs.get("generation")
    if (
        not isinstance(expected_size, int)
        or not isinstance(expected_md5, str)
        or not expected_md5
        or not isinstance(expected_crc32c, str)
        or not expected_crc32c
        or not isinstance(generation, str)
        or not generation
    ):
        raise ValueError(f"{code}: complete GCS size/generation/MD5/CRC32C is required")
    if path.stat().st_size != expected_size or record.get("local_size_bytes") != expected_size:
        raise ValueError(f"{code}: local archive size differs from frozen GCS metadata")
    observed_md5 = _base64_md5(path)
    observed_crc32c = _base64_crc32c(path)
    if observed_md5 != expected_md5:
        raise ValueError(f"{code}: local archive MD5 differs from GCS metadata")
    if observed_crc32c != expected_crc32c:
        raise ValueError(f"{code}: local archive CRC32C differs from GCS metadata")
    with zipfile.ZipFile(path) as archive:
        bad_member = archive.testzip()
        member_count = len(archive.infolist())
    if bad_member is not None:
        raise ValueError(f"{code}: ZIP member CRC failed: {bad_member}")
    return {
        "two_char_code": code,
        "path": str(path.resolve()),
        "size_bytes": expected_size,
        "generation": generation,
        "md5_hash_base64": observed_md5,
        "crc32c_base64": observed_crc32c,
        "zip_member_count": member_count,
        "zip_crc_passed": True,
    }


def _sample_assets(sample_id: str, *, raw_root: Path) -> dict[str, Any]:
    system_id, ligand_chain = sample_id.rsplit("__", 1)
    receptor = raw_root / "systems" / system_id / "receptor.pdb"
    ligand = raw_root / "systems" / system_id / "ligand_files" / f"{ligand_chain}.sdf"
    for label, path in (("receptor", receptor), ("ligand", ligand)):
        if not path.is_file() or path.stat().st_size < 1:
            raise FileNotFoundError(f"{sample_id}: missing or empty {label}: {path}")
    return {
        "sample_id": sample_id,
        "system_id": system_id,
        "ligand_chain": ligand_chain,
        "receptor": {
            "path": str(receptor.resolve()),
            "size_bytes": receptor.stat().st_size,
            "sha256": file_sha256(receptor),
        },
        "ligand": {
            "path": str(ligand.resolve()),
            "size_bytes": ligand.stat().st_size,
            "sha256": file_sha256(ligand),
        },
    }


def run(
    *,
    split_file: Path,
    raw_manifest: Path,
    raw_root: Path,
    output: Path,
    sidecar: Path,
    published_output: Path | None,
    workers: int,
) -> dict[str, Any]:
    if not 1 <= workers <= 16:
        raise ValueError("workers must be between 1 and 16")
    split_ids = load_split_ids(split_file)
    manifest = verify_raw_manifest(raw_manifest, split_ids=split_ids)
    gcs_inventory = manifest["archive_inventory"]["gcs_metadata"]
    if gcs_inventory.get("status") != "complete" or gcs_inventory.get("missing_objects") != []:
        raise ValueError("complete GCS object metadata is required before sampling")
    archive_records = manifest["archive_inventory"]["archives"]
    if not isinstance(archive_records, list) or len(archive_records) != EXPECTED_RAW_ARCHIVES:
        raise ValueError("raw manifest archive ledger is incomplete")
    with ThreadPoolExecutor(max_workers=workers) as executor:
        archives = list(
            executor.map(lambda record: _verify_archive(record, raw_root=raw_root), archive_records)
        )
        assets = list(
            executor.map(lambda sample_id: _sample_assets(sample_id, raw_root=raw_root), split_ids)
        )
    archives.sort(key=lambda item: item["two_char_code"])
    assets.sort(key=lambda item: item["sample_id"])
    total_size = sum(int(item["size_bytes"]) for item in archives)
    unique_systems = {item["system_id"] for item in assets}
    if total_size != EXPECTED_RAW_ARCHIVE_BYTES or len(unique_systems) != EXPECTED_UNIQUE_SYSTEMS:
        raise ValueError("raw archive size or unique-system coverage mismatch")
    result: dict[str, Any] = {
        "schema_version": RAW_GATE_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "status": "passed",
        "split_sha256": EXPECTED_SPLIT_SHA256,
        "sample_count": EXPECTED_COUNT,
        "sample_ids_sha256": ids_sha256(split_ids),
        "unique_system_count": len(unique_systems),
        "archive_count": len(archives),
        "archive_total_size_bytes": total_size,
        "raw_manifest": str(raw_manifest.resolve()),
        "raw_manifest_sha256": file_sha256(raw_manifest),
        "raw_root": str(raw_root.resolve()),
        "gcs_metadata_status": "complete",
        "gcs_md5_verified_count": len(archives),
        "gcs_crc32c_verified_count": len(archives),
        "zip_crc_verified_count": sum(item["zip_crc_passed"] for item in archives),
        "verified_receptor_sample_count": len(assets),
        "verified_ligand_count": len(assets),
        "mismatches": [],
        "archive_ledger_sha256": canonical_json_sha256(archives),
        "asset_ledger_sha256": canonical_json_sha256(assets),
        "archives": archives,
        "assets": assets,
    }
    write_json_noreplace(output, result)
    digest = file_sha256(output)
    bound_output = output if published_output is None else published_output
    write_bytes_noreplace(sidecar, f"{digest}  {bound_output.resolve()}\n".encode("utf-8"))
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-file", type=Path, default=Path("data/splits/plinder.json"))
    parser.add_argument("--raw-manifest", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument(
        "--published-output",
        type=Path,
        default=None,
        help="final path to bind in the sidecar when publishing an attempt directory",
    )
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)
    result = run(
        split_file=args.split_file,
        raw_manifest=args.raw_manifest,
        raw_root=args.raw_root,
        output=args.output,
        sidecar=args.sidecar,
        published_output=args.published_output,
        workers=args.workers,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "archives": result["archive_count"],
                "samples": result["sample_count"],
                "output": str(args.output),
                "sidecar": str(args.sidecar),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
