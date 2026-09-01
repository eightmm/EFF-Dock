#!/usr/bin/env python3
"""Download and verify the frozen PLINDER guidance-validation raw structures."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from plinder.core.utils.config import get_config
from plinder.core.utils.unpack import get_zips_to_unpack

PLINDER_VERSION = "0.2.26"
PLINDER_RELEASE = "2024-06"
PLINDER_ITERATION = "v2"
EXPECTED_SPLIT_SHA256 = (
    "3ac570bf08bced053f1ce040b57efca27c3be616f29a82cd66ef887c08860e6b"
)
EXPECTED_SAMPLE_COUNT = 1_076
EXPECTED_SYSTEM_COUNT = 1_058
EXPECTED_ARCHIVE_COUNT = 475
EXPECTED_ARCHIVE_BYTES = 71_372_079_105
_TWO_CHAR_CODE = re.compile(r"^[a-z0-9]{2}$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_identifier(value: str, *, label: str) -> str:
    if not value or value in {".", ".."} or Path(value).name != value:
        raise ValueError(f"unsafe {label}: {value!r}")
    if "/" in value or "\\" in value:
        raise ValueError(f"unsafe {label}: {value!r}")
    return value


def _load_frozen_samples(split_file: Path) -> tuple[str, list[dict[str, str]]]:
    if not split_file.is_file():
        raise FileNotFoundError(f"missing split file: {split_file}")
    split_sha256 = _sha256(split_file)
    if split_sha256 != EXPECTED_SPLIT_SHA256:
        raise ValueError(
            "PLINDER split SHA-256 mismatch: "
            f"expected {EXPECTED_SPLIT_SHA256}, observed {split_sha256}"
        )

    payload = json.loads(split_file.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("val"), list):
        raise ValueError("split must be a JSON object containing a list at key 'val'")
    sample_keys = payload["val"]
    if len(sample_keys) != EXPECTED_SAMPLE_COUNT:
        raise ValueError(
            f"validation sample count mismatch: expected {EXPECTED_SAMPLE_COUNT}, "
            f"observed {len(sample_keys)}"
        )
    if not all(isinstance(sample_key, str) for sample_key in sample_keys):
        raise ValueError("every validation sample key must be a string")
    if len(set(sample_keys)) != EXPECTED_SAMPLE_COUNT:
        raise ValueError("validation sample keys are not unique")

    samples: list[dict[str, str]] = []
    for sample_key in sorted(sample_keys):
        try:
            system_id, ligand_chain = sample_key.rsplit("__", 1)
        except ValueError as exc:
            raise ValueError(f"invalid sample key: {sample_key!r}") from exc
        _safe_identifier(system_id, label="PLINDER system ID")
        _safe_identifier(ligand_chain, label="ligand instance chain")
        if len(system_id) < 3:
            raise ValueError(f"invalid PLINDER system ID: {system_id!r}")
        two_char_code = system_id[1:3]
        if _TWO_CHAR_CODE.fullmatch(two_char_code) is None:
            raise ValueError(
                f"invalid two-character archive code {two_char_code!r} "
                f"from system {system_id!r}"
            )
        samples.append(
            {
                "sample_key": sample_key,
                "system_id": system_id,
                "ligand_instance_chain": ligand_chain,
                "two_char_code": two_char_code,
            }
        )

    system_ids = {sample["system_id"] for sample in samples}
    if len(system_ids) != EXPECTED_SYSTEM_COUNT:
        raise ValueError(
            f"unique system count mismatch: expected {EXPECTED_SYSTEM_COUNT}, "
            f"observed {len(system_ids)}"
        )
    codes = {sample["two_char_code"] for sample in samples}
    if len(codes) != EXPECTED_ARCHIVE_COUNT:
        raise ValueError(
            f"required archive count mismatch: expected {EXPECTED_ARCHIVE_COUNT}, "
            f"observed {len(codes)}"
        )
    return split_sha256, samples


def _remote_blob_metadata(
    *, bucket_name: str, object_prefix: str, required_objects: set[str]
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Return public GCS object metadata, or a bounded unavailability record."""
    try:
        from google.cloud import storage

        client = storage.Client.create_anonymous_client()
        found: dict[str, dict[str, Any]] = {}
        for blob in client.list_blobs(bucket_name, prefix=object_prefix):
            if blob.name not in required_objects:
                continue
            found[blob.name] = {
                "size_bytes": int(blob.size) if blob.size is not None else None,
                "generation": str(blob.generation)
                if blob.generation is not None
                else None,
                "md5_hash_base64": blob.md5_hash,
                "crc32c_base64": blob.crc32c,
                "etag": blob.etag,
                "storage_class": blob.storage_class,
                "updated_utc": blob.updated.isoformat()
                if blob.updated is not None
                else None,
            }
        return found, {
            "status": "complete" if set(found) == required_objects else "incomplete",
            "requested_object_count": len(required_objects),
            "returned_object_count": len(found),
            "missing_objects": sorted(required_objects - set(found)),
        }
    except Exception as exc:  # metadata is supplementary to verified local bytes
        return {}, {
            "status": "unavailable",
            "requested_object_count": len(required_objects),
            "returned_object_count": 0,
            "missing_objects": [],
            "error_type": type(exc).__name__,
        }


def _relative_display(path: Path, *, base: Path) -> str:
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _expected_ids_by_code(samples: Iterable[dict[str, str]]) -> dict[str, list[str]]:
    grouped: defaultdict[str, set[str]] = defaultdict(set)
    for sample in samples:
        grouped[sample["two_char_code"]].add(sample["system_id"])
    return {code: sorted(system_ids) for code, system_ids in sorted(grouped.items())}


def _bounded_parallelism(value: str | int) -> int:
    try:
        parallelism = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("parallelism must be an integer") from exc
    if not 1 <= parallelism <= 32:
        raise argparse.ArgumentTypeError("parallelism must be between 1 and 32")
    return parallelism


def run(
    *,
    split_file: Path,
    output: Path,
    plinder_mount: str | None,
    parallelism: int = 8,
) -> None:
    parallelism = _bounded_parallelism(parallelism)
    package_version = importlib.metadata.version("plinder")
    if package_version != PLINDER_VERSION:
        raise RuntimeError(
            f"PLINDER package mismatch: expected {PLINDER_VERSION}, "
            f"observed {package_version}"
        )

    split_sha256, samples = _load_frozen_samples(split_file)
    system_ids = sorted({sample["system_id"] for sample in samples})
    codes = sorted({sample["two_char_code"] for sample in samples})
    expected_ids_by_code = _expected_ids_by_code(samples)

    data_config: dict[str, Any] = {
        "plinder_release": PLINDER_RELEASE,
        "plinder_iteration": PLINDER_ITERATION,
    }
    if plinder_mount is not None:
        data_config["plinder_mount"] = plinder_mount
    cfg = get_config(config={"data": data_config}, cached=True)
    if cfg.data.plinder_release != PLINDER_RELEASE:
        raise RuntimeError("PLINDER resolved an unexpected release")
    if cfg.data.plinder_iteration != PLINDER_ITERATION:
        raise RuntimeError("PLINDER resolved an unexpected iteration")

    bucket_name = str(cfg.data.plinder_bucket)
    object_prefix = f"{PLINDER_RELEASE}/{PLINDER_ITERATION}/systems/"
    required_objects = {f"{object_prefix}{code}.zip" for code in codes}
    remote_metadata, remote_inventory = _remote_blob_metadata(
        bucket_name=bucket_name,
        object_prefix=object_prefix,
        required_objects=required_objects,
    )
    remote_total_size: int | None = None
    if remote_inventory["status"] == "incomplete":
        raise RuntimeError(
            "public GCS archive inventory is incomplete: "
            f"{len(remote_inventory['missing_objects'])} object(s) missing"
        )
    if remote_inventory["status"] == "complete":
        remote_sizes = [metadata["size_bytes"] for metadata in remote_metadata.values()]
        if any(size is None for size in remote_sizes):
            raise RuntimeError("public GCS archive inventory has missing object sizes")
        remote_total_size = sum(int(size) for size in remote_sizes)
        if remote_total_size != EXPECTED_ARCHIVE_BYTES:
            raise RuntimeError(
                "public GCS archive size mismatch: "
                f"expected {EXPECTED_ARCHIVE_BYTES}, observed {remote_total_size}"
            )

    code_batches = [
        codes[offset : offset + parallelism]
        for offset in range(0, len(codes), parallelism)
    ]
    zip_mapping: dict[Path, list[str]] = {}
    download_batches: list[dict[str, Any]] = []
    merged_codes: set[str] = set()
    for batch_index, batch_codes in enumerate(code_batches):
        batch_system_ids = [
            system_id
            for code in batch_codes
            for system_id in expected_ids_by_code[code]
        ]
        batch_mapping = get_zips_to_unpack(
            kind="systems", system_ids=batch_system_ids, cfg=cfg
        )
        returned_codes = {Path(path).stem for path in batch_mapping}
        if returned_codes != set(batch_codes):
            raise RuntimeError(
                f"PLINDER batch {batch_index} archive mismatch: "
                f"expected {batch_codes}, observed {sorted(returned_codes)}"
            )
        duplicate_codes = merged_codes.intersection(returned_codes)
        duplicate_paths = set(zip_mapping).intersection(batch_mapping)
        if duplicate_codes or duplicate_paths:
            raise RuntimeError(
                f"PLINDER batch {batch_index} returned duplicate archives: "
                f"{sorted(duplicate_codes)}"
            )
        zip_mapping.update(batch_mapping)
        merged_codes.update(returned_codes)
        download_batches.append(
            {
                "batch_index": batch_index,
                "two_char_codes": batch_codes,
                "archive_count": len(batch_mapping),
                "system_id_count": len(batch_system_ids),
            }
        )
    if len(zip_mapping) != EXPECTED_ARCHIVE_COUNT:
        raise RuntimeError(
            f"PLINDER returned {len(zip_mapping)} archives; "
            f"expected {EXPECTED_ARCHIVE_COUNT}"
        )

    zip_by_code: dict[str, Path] = {}
    mapping_mismatches: list[dict[str, Any]] = []
    for raw_path, returned_ids in zip_mapping.items():
        zip_path = Path(raw_path)
        code = zip_path.stem
        if code in zip_by_code:
            mapping_mismatches.append(
                {"kind": "duplicate_archive_code", "two_char_code": code}
            )
        zip_by_code[code] = zip_path
        expected_ids = expected_ids_by_code.get(code)
        if expected_ids is None or sorted(set(returned_ids)) != expected_ids:
            mapping_mismatches.append(
                {
                    "kind": "archive_system_id_mapping",
                    "two_char_code": code,
                    "expected_system_ids": expected_ids or [],
                    "observed_system_ids": sorted(set(returned_ids)),
                }
            )
    if set(zip_by_code) != set(codes):
        mapping_mismatches.append(
            {
                "kind": "archive_code_inventory",
                "missing_codes": sorted(set(codes) - set(zip_by_code)),
                "unexpected_codes": sorted(set(zip_by_code) - set(codes)),
            }
        )

    archives: list[dict[str, Any]] = []
    local_archive_bytes = 0
    systems_root: Path | None = None
    for code in codes:
        zip_path = zip_by_code.get(code)
        local_size: int | None = None
        if zip_path is None:
            mapping_mismatches.append(
                {"kind": "missing_archive_mapping", "two_char_code": code}
            )
        else:
            systems_root = zip_path.parent if systems_root is None else systems_root
            if zip_path.parent != systems_root:
                mapping_mismatches.append(
                    {"kind": "multiple_system_roots", "two_char_code": code}
                )
            if not zip_path.is_file():
                mapping_mismatches.append(
                    {
                        "kind": "missing_local_archive",
                        "two_char_code": code,
                        "relative_path": f"systems/{code}.zip",
                    }
                )
            else:
                local_size = zip_path.stat().st_size
                local_archive_bytes += local_size

        object_name = f"{object_prefix}{code}.zip"
        gcs_metadata = remote_metadata.get(object_name)
        if (
            local_size is not None
            and gcs_metadata is not None
            and gcs_metadata["size_bytes"] is not None
            and local_size != gcs_metadata["size_bytes"]
        ):
            mapping_mismatches.append(
                {
                    "kind": "archive_size_vs_gcs",
                    "two_char_code": code,
                    "local_size_bytes": local_size,
                    "gcs_size_bytes": gcs_metadata["size_bytes"],
                }
            )
        archives.append(
            {
                "two_char_code": code,
                "relative_path": f"systems/{code}.zip",
                "gcs_uri": f"gs://{bucket_name}/{object_name}",
                "local_size_bytes": local_size,
                "gcs": gcs_metadata,
                "requested_system_count": len(expected_ids_by_code[code]),
            }
        )

    if local_archive_bytes != EXPECTED_ARCHIVE_BYTES:
        mapping_mismatches.append(
            {
                "kind": "archive_total_size",
                "expected_bytes": EXPECTED_ARCHIVE_BYTES,
                "observed_bytes": local_archive_bytes,
            }
        )
    if systems_root is None:
        raise RuntimeError("no local PLINDER systems archive root was resolved")

    missing_assets: list[dict[str, Any]] = []
    verified_sample_count = 0
    verified_receptor_sample_count = 0
    verified_ligand_count = 0
    verified_receptor_paths: set[Path] = set()
    for sample in samples:
        system_dir = systems_root / sample["system_id"]
        receptor = system_dir / "receptor.pdb"
        ligand = (
            system_dir
            / "ligand_files"
            / f"{sample['ligand_instance_chain']}.sdf"
        )
        missing: list[str] = []
        if receptor.is_file() and receptor.stat().st_size > 0:
            verified_receptor_sample_count += 1
            verified_receptor_paths.add(receptor)
        else:
            missing.append("receptor.pdb")
        if ligand.is_file() and ligand.stat().st_size > 0:
            verified_ligand_count += 1
        else:
            missing.append("ligand_sdf")
        if not missing:
            verified_sample_count += 1
        else:
            missing_assets.append(
                {
                    "sample_key": sample["sample_key"],
                    "system_id": sample["system_id"],
                    "ligand_instance_chain": sample["ligand_instance_chain"],
                    "missing_or_empty": missing,
                    "expected_receptor": (
                        Path("systems") / sample["system_id"] / "receptor.pdb"
                    ).as_posix(),
                    "expected_ligand": (
                        Path("systems")
                        / sample["system_id"]
                        / "ligand_files"
                        / f"{sample['ligand_instance_chain']}.sdf"
                    ).as_posix(),
                }
            )

    verification_mismatches = list(mapping_mismatches)
    if verified_sample_count != EXPECTED_SAMPLE_COUNT or missing_assets:
        verification_mismatches.append(
            {
                "kind": "raw_asset_coverage",
                "expected_samples": EXPECTED_SAMPLE_COUNT,
                "verified_samples": verified_sample_count,
                "missing_sample_count": len(missing_assets),
            }
        )
    if len(verified_receptor_paths) != EXPECTED_SYSTEM_COUNT:
        verification_mismatches.append(
            {
                "kind": "unique_receptor_coverage",
                "expected_systems": EXPECTED_SYSTEM_COUNT,
                "verified_systems": len(verified_receptor_paths),
            }
        )

    cwd = Path.cwd()
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "complete" if not verification_mismatches else "failed",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "name": "PLINDER",
            "dataset_id": "plinder_2024_06_v2",
            "release": PLINDER_RELEASE,
            "iteration": PLINDER_ITERATION,
            "plinder_package_version": package_version,
            "bucket": bucket_name,
            "systems_gcs_prefix": f"gs://{bucket_name}/{object_prefix}",
        },
        "split": {
            "path": _relative_display(split_file, base=cwd),
            "key": "val",
            "sha256": split_sha256,
            "sample_count": len(samples),
            "unique_sample_key_count": len({s["sample_key"] for s in samples}),
            "unique_system_id_count": len(system_ids),
            "required_archive_count": len(codes),
        },
        "request": {
            "sample_keys": [sample["sample_key"] for sample in samples],
            "samples": samples,
            "system_ids": system_ids,
            "required_two_char_codes": codes,
            "expected_ids_by_two_char_code": expected_ids_by_code,
        },
        "archive_inventory": {
            "parallelism": parallelism,
            "batch_count": len(code_batches),
            "download_batches": download_batches,
            "expected_archive_count": EXPECTED_ARCHIVE_COUNT,
            "observed_archive_count": len(zip_mapping),
            "expected_total_size_bytes": EXPECTED_ARCHIVE_BYTES,
            "observed_total_size_bytes": local_archive_bytes,
            "observed_gcs_total_size_bytes": remote_total_size,
            "gcs_metadata": remote_inventory,
            "archives": archives,
        },
        "verification": {
            "expected_sample_count": EXPECTED_SAMPLE_COUNT,
            "expected_unique_system_count": EXPECTED_SYSTEM_COUNT,
            "verified_sample_count": verified_sample_count,
            "verified_receptor_sample_count": verified_receptor_sample_count,
            "verified_unique_receptor_count": len(verified_receptor_paths),
            "verified_ligand_count": verified_ligand_count,
            "missing_sample_count": len(missing_assets),
            "missing_assets": missing_assets,
            "mismatches": verification_mismatches,
        },
    }
    _write_json_atomic(output, manifest)
    if verification_mismatches:
        raise RuntimeError(
            f"PLINDER raw validation failed with {len(verification_mismatches)} "
            f"mismatch(es); see {output}"
        )

    print(
        f"verified {verified_sample_count}/{EXPECTED_SAMPLE_COUNT} samples, "
        f"{len(system_ids)} systems, {len(codes)} archives; manifest={output}"
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split-file",
        type=Path,
        default=Path("data/splits/plinder.json"),
        help="frozen EFF-Dock PLINDER split JSON",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="atomic raw-download manifest path",
    )
    parser.add_argument(
        "--plinder-mount",
        default=None,
        help="optional PLINDER cache mount; defaults to PLINDER's configured mount",
    )
    parser.add_argument(
        "--parallelism",
        type=_bounded_parallelism,
        default=8,
        help="maximum archives downloaded/extracted concurrently (1..32; default: 8)",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    run(
        split_file=args.split_file,
        output=args.output,
        plinder_mount=args.plinder_mount,
        parallelism=args.parallelism,
    )


if __name__ == "__main__":
    main()
