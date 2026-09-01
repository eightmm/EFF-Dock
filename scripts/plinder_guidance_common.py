#!/usr/bin/env python3
"""Shared fail-closed contracts for the PLINDER guidance-development pipeline."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Iterable

PROTOCOL_ID = "EFFDOCK-PLINDER-GUIDANCE-DEV-V1"
SAMPLING_SCHEMA = "effdock.plinder_guidance_validation_shard.v1"
AUDIT_SCHEMA = "effdock.plinder_guidance_audit.v1"
POSEBUSTERS_SCHEMA = "effdock.plinder_guidance_posebusters_shard.v1"
REPORT_SCHEMA = "effdock.plinder_guidance_report.v1"
RAW_GATE_SCHEMA = "effdock.plinder_guidance_raw_gate.v1"

EXPECTED_COUNT = 1_076
EXPECTED_SPLIT_SHA256 = (
    "3ac570bf08bced053f1ce040b57efca27c3be616f29a82cd66ef887c08860e6b"
)
EXPECTED_DOCKING_SHA256 = (
    "6932fb3ba6ebac770f714453529656a44b8f33cf15119d23c9e675d2d60b36db"
)
EXPECTED_CONFIDENCE_SHA256 = (
    "e31fde6f351284205c78f7a1510002779c43312e94d9f82003d47a14d72bc78f"
)
EXPECTED_CONFIG_SHA256 = (
    "39aa62e4a48ed6f3aa4ff59345fb43a81220e2baba22edfd5beb0c4981b307ec"
)
EXPECTED_GUIDANCE_PARAMETER_SHA256 = (
    "6621d17c41aeb6c9685075209155850018c5eb9882489ae209c7c30b8070e89f"
)
EXPECTED_GUIDANCE_IMPLEMENTATION_SHA256 = (
    "04271077bfc9fe255e370cb5b985efe4df7242ba700abc6f81c50ec12aff6b25"
)
EXPECTED_POSEBUSTERS_VERSION = "0.6.5"
EXPECTED_RAW_ARCHIVES = 475
EXPECTED_RAW_ARCHIVE_BYTES = 71_372_079_105
EXPECTED_UNIQUE_SYSTEMS = 1_058

BASE_SEED = 42
ETA_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0)
ETA_TAGS = {
    0.0: "eta0000",
    0.5: "eta0500",
    1.0: "eta1000",
    1.5: "eta1500",
    2.0: "eta2000",
}
TAG_TO_ETA = {tag: eta for eta, tag in ETA_TAGS.items()}
FULL_SHARDS = 32
SMOKE_SHARDS = 1
SMOKE_ID = "7z9g__1__1.A_1.B_1.C_1.D__1.J__1.J"
PRIMARY_SELECTOR = "confidence"
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def require_file_hash(path: Path, expected: str, *, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    observed = file_sha256(path)
    if observed != expected:
        raise ValueError(f"{label} SHA-256 mismatch: expected={expected} observed={observed}")


def parse_bool(value: object, *, label: str) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"{label} must be true or false, got {value!r}")


def finite_float(value: object, *, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def exact_int(value: object, *, label: str) -> int:
    try:
        result = int(str(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer") from exc
    if str(result) != str(value):
        raise ValueError(f"{label} must be a canonical integer")
    return result


def load_split_ids(path: Path) -> list[str]:
    require_file_hash(path, EXPECTED_SPLIT_SHA256, label="frozen PLINDER split")
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw = payload.get("val") if isinstance(payload, dict) else None
    if not isinstance(raw, list) or not all(isinstance(value, str) and value for value in raw):
        raise ValueError("frozen PLINDER split must contain a string list at key 'val'")
    ordered = sorted(raw)
    if len(ordered) != EXPECTED_COUNT or len(set(ordered)) != EXPECTED_COUNT:
        raise ValueError(
            f"frozen PLINDER validation must contain {EXPECTED_COUNT} unique sample keys"
        )
    if SMOKE_ID not in set(ordered):
        raise ValueError(f"frozen size-based smoke ID is absent: {SMOKE_ID}")
    return ordered


def ids_sha256(ids: Iterable[str]) -> str:
    ordered = sorted(ids)
    if len(ordered) != len(set(ordered)):
        raise ValueError("ID inventory contains duplicates")
    digest = hashlib.sha256()
    digest.update(b"EFFDOCK_SORTED_COMPLEX_IDS_V1\0")
    for complex_id in ordered:
        digest.update(complex_id.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def expected_ids(mode: str, split_ids: list[str]) -> list[str]:
    if mode == "full":
        return list(split_ids)
    if mode == "smoke":
        return [SMOKE_ID]
    raise ValueError(f"mode must be 'smoke' or 'full', got {mode!r}")


def expected_num_shards(mode: str) -> int:
    if mode == "full":
        return FULL_SHARDS
    if mode == "smoke":
        return SMOKE_SHARDS
    raise ValueError(f"mode must be 'smoke' or 'full', got {mode!r}")


def sampling_shard_dir(
    sampling_root: Path, mode: str, eta: float, shard_index: int, num_shards: int
) -> Path:
    return (
        sampling_root
        / mode
        / ETA_TAGS[eta]
        / f"shard-{shard_index:03d}-of-{num_shards:03d}"
    )


def posebusters_shard_dir(
    posebusters_root: Path, mode: str, eta: float, shard_index: int, num_shards: int
) -> Path:
    return (
        posebusters_root
        / mode
        / ETA_TAGS[eta]
        / f"shard-{shard_index:03d}-of-{num_shards:03d}"
    )


def load_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: missing CSV header")
        fields = list(reader.fieldnames)
        if len(fields) != len(set(fields)):
            raise ValueError(f"{path}: duplicate CSV columns")
        return fields, list(reader)


def write_bytes_noreplace(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite artifact: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json_noreplace(path: Path, payload: dict[str, Any]) -> None:
    data = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    write_bytes_noreplace(path, data)


def verify_raw_manifest(path: Path, *, split_ids: list[str]) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"raw-download manifest is missing: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise ValueError("raw-download manifest schema mismatch")
    if raw.get("status") != "complete":
        raise ValueError("raw-download manifest is not complete")
    source = raw.get("source")
    if not isinstance(source, dict):
        raise ValueError("raw-download manifest lacks source identity")
    expected_source = {
        "name": "PLINDER",
        "dataset_id": "plinder_2024_06_v2",
        "release": "2024-06",
        "iteration": "v2",
        "plinder_package_version": "0.2.26",
        "bucket": "plinder",
        "systems_gcs_prefix": "gs://plinder/2024-06/v2/systems/",
    }
    for key, expected in expected_source.items():
        if source.get(key) != expected:
            raise ValueError(f"raw-download manifest source.{key} mismatch")
    split = raw.get("split")
    if not isinstance(split, dict):
        raise ValueError("raw-download manifest lacks split identity")
    expected_split = {
        "key": "val",
        "sha256": EXPECTED_SPLIT_SHA256,
        "sample_count": EXPECTED_COUNT,
        "unique_sample_key_count": EXPECTED_COUNT,
        "unique_system_id_count": EXPECTED_UNIQUE_SYSTEMS,
        "required_archive_count": EXPECTED_RAW_ARCHIVES,
    }
    for key, expected in expected_split.items():
        if split.get(key) != expected:
            raise ValueError(f"raw-download manifest split.{key} mismatch")
    request = raw.get("request")
    if not isinstance(request, dict) or request.get("sample_keys") != split_ids:
        raise ValueError("raw-download manifest sample-key inventory mismatch")
    archive = raw.get("archive_inventory")
    if not isinstance(archive, dict):
        raise ValueError("raw-download manifest lacks archive inventory")
    expected_archive = {
        "expected_archive_count": EXPECTED_RAW_ARCHIVES,
        "observed_archive_count": EXPECTED_RAW_ARCHIVES,
        "expected_total_size_bytes": EXPECTED_RAW_ARCHIVE_BYTES,
        "observed_total_size_bytes": EXPECTED_RAW_ARCHIVE_BYTES,
    }
    for key, expected in expected_archive.items():
        if archive.get(key) != expected:
            raise ValueError(f"raw-download manifest archive_inventory.{key} mismatch")
    verification = raw.get("verification")
    if not isinstance(verification, dict):
        raise ValueError("raw-download manifest lacks verification")
    expected_verification = {
        "expected_sample_count": EXPECTED_COUNT,
        "expected_unique_system_count": EXPECTED_UNIQUE_SYSTEMS,
        "verified_sample_count": EXPECTED_COUNT,
        "verified_receptor_sample_count": EXPECTED_COUNT,
        "verified_unique_receptor_count": EXPECTED_UNIQUE_SYSTEMS,
        "verified_ligand_count": EXPECTED_COUNT,
        "missing_sample_count": 0,
        "missing_assets": [],
        "mismatches": [],
    }
    for key, expected in expected_verification.items():
        if verification.get(key) != expected:
            raise ValueError(f"raw-download manifest verification.{key} mismatch")
    return raw


def validate_raw_gate(
    path: Path,
    sidecar: Path,
    *,
    raw_manifest: Path,
    raw_root: Path,
    split_ids: list[str],
) -> dict[str, Any]:
    if not path.is_file() or not sidecar.is_file():
        raise FileNotFoundError("verified raw gate or SHA-256 sidecar is missing")
    sidecar_fields = sidecar.read_text(encoding="utf-8").strip().split()
    if len(sidecar_fields) != 2:
        raise ValueError("raw-gate SHA-256 sidecar must contain one digest/path record")
    expected_digest = require_sha256(sidecar_fields[0], label="raw-gate sidecar")
    sidecar_path = Path(sidecar_fields[1])
    if sidecar_path.resolve() != path.resolve() or file_sha256(path) != expected_digest:
        raise ValueError("raw-gate SHA-256 sidecar binding failed")
    raw = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": RAW_GATE_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "status": "passed",
        "split_sha256": EXPECTED_SPLIT_SHA256,
        "sample_count": EXPECTED_COUNT,
        "sample_ids_sha256": ids_sha256(split_ids),
        "unique_system_count": EXPECTED_UNIQUE_SYSTEMS,
        "archive_count": EXPECTED_RAW_ARCHIVES,
        "archive_total_size_bytes": EXPECTED_RAW_ARCHIVE_BYTES,
        "raw_manifest": str(raw_manifest.resolve()),
        "raw_manifest_sha256": file_sha256(raw_manifest),
        "raw_root": str(raw_root.resolve()),
        "gcs_metadata_status": "complete",
        "gcs_md5_verified_count": EXPECTED_RAW_ARCHIVES,
        "gcs_crc32c_verified_count": EXPECTED_RAW_ARCHIVES,
        "zip_crc_verified_count": EXPECTED_RAW_ARCHIVES,
        "verified_receptor_sample_count": EXPECTED_COUNT,
        "verified_ligand_count": EXPECTED_COUNT,
        "mismatches": [],
    }
    for key, expected_value in expected.items():
        if raw.get(key) != expected_value:
            raise ValueError(f"raw-gate {key} mismatch")
    require_sha256(raw.get("archive_ledger_sha256"), label="raw archive ledger")
    archive_records = raw.get("archives")
    if not isinstance(archive_records, list) or len(archive_records) != EXPECTED_RAW_ARCHIVES:
        raise ValueError("raw-gate archive records are incomplete")
    if canonical_json_sha256(archive_records) != raw.get("archive_ledger_sha256"):
        raise ValueError("raw-gate archive ledger hash mismatch")
    require_sha256(raw.get("asset_ledger_sha256"), label="raw asset ledger")
    assets = raw.get("assets")
    if not isinstance(assets, list) or len(assets) != EXPECTED_COUNT:
        raise ValueError("raw-gate asset records are incomplete")
    if [asset.get("sample_id") if isinstance(asset, dict) else None for asset in assets] != split_ids:
        raise ValueError("raw-gate asset inventory/order mismatch")
    if canonical_json_sha256(assets) != raw.get("asset_ledger_sha256"):
        raise ValueError("raw-gate asset ledger hash mismatch")
    for asset in assets:
        for label in ("receptor", "ligand"):
            identity = asset.get(label)
            if not isinstance(identity, dict):
                raise ValueError(f"raw-gate asset lacks {label} identity")
            require_sha256(identity.get("sha256"), label=f"raw-gate {label}")
            if not isinstance(identity.get("path"), str) or not identity["path"]:
                raise ValueError(f"raw-gate {label} path is missing")
            if not isinstance(identity.get("size_bytes"), int) or identity["size_bytes"] < 1:
                raise ValueError(f"raw-gate {label} size is invalid")
    return raw


def validate_passed_audit(
    path: Path,
    *,
    mode: str,
    sampling_root: Path | None = None,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing PLINDER guidance audit: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": AUDIT_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "status": "passed",
        "mode": mode,
        "primary_selector": PRIMARY_SELECTOR,
        "expected_denominator": EXPECTED_COUNT,
        "eta_values": list(ETA_VALUES),
    }
    for key, value in expected.items():
        if raw.get(key) != value:
            raise ValueError(f"{path}: audit {key} mismatch")
    if sampling_root is not None:
        observed = raw.get("sampling_root")
        if not isinstance(observed, str) or Path(observed).resolve() != sampling_root.resolve():
            raise ValueError(f"{path}: audit sampling root mismatch")
    require_sha256(raw.get("global_sampling_ledger_sha256"), label="audit ledger")
    return raw
