#!/usr/bin/env python3
"""Bind one confidence-selector PoseBusters shard to exact sampling inputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from effdock.workflows.evaluate import sorted_id_sha256
from effdock.workflows.posebusters_report import file_sha256, load_rows

PROTOCOL_ID = "EFFDOCK-UNIFIED-GUIDANCE-ETA-SWEEP-CONFIDENCE-PB-V1"
CONFIDENCE_CHECKPOINT_SHA256 = "e31fde6f351284205c78f7a1510002779c43312e94d9f82003d47a14d72bc78f"
SELECTORS = ("confidence", "confidence_filter")
BINDING_CONTRACT = "EFFDOCK_ETA_SWEEP_CONFIDENCE_OFFICIAL_BINDING_V1"


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{label} must be a SHA-256 hex digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{label} must be a SHA-256 hex digest") from exc
    return value


def _ordered_ids_sha256(ids: list[str]) -> str:
    digest = hashlib.sha256()
    digest.update(b"EFFDOCK_ORDERED_COMPLEX_IDS_V1\0")
    for complex_id in ids:
        digest.update(complex_id.encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _resolve_declared_file(
    declaration: object,
    *,
    summary_path: Path,
    expected_path: Path,
    label: str,
) -> Path:
    if not isinstance(declaration, str) or not declaration:
        raise ValueError(f"{summary_path}: {label} must be a non-empty path")
    raw = Path(declaration)
    candidates = [raw] if raw.is_absolute() else [Path.cwd() / raw, summary_path.parent / raw]
    hits: list[Path] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_file() and resolved not in hits:
            hits.append(resolved)
    if not hits:
        raise FileNotFoundError(f"{summary_path}: declared {label} does not exist: {declaration}")
    if expected_path.resolve() not in hits:
        raise ValueError(f"{summary_path}: {label} must resolve to {expected_path.resolve()}")
    if len(hits) != 1:
        raise ValueError(f"{summary_path}: ambiguous relative {label}: {declaration}")
    return hits[0]


def _read_csv_ids(path: Path) -> list[str]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "id" not in reader.fieldnames:
            raise ValueError(f"{path}: CSV requires an id column")
        ids = [str(row["id"]) for row in reader]
    if any(not complex_id or complex_id.strip() != complex_id for complex_id in ids):
        raise ValueError(f"{path}: invalid complex ID")
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path}: duplicate complex IDs")
    return ids


def _file_ledger(
    rows: list[dict[str, str]],
    *,
    selector: str,
    pose_dir: Path,
) -> str:
    """Verify and hash sampling-time protein/reference/selected-pose identities."""
    if selector not in SELECTORS:
        raise ValueError(f"selector must be one of {SELECTORS}")
    digest = hashlib.sha256()
    digest.update(b"EFFDOCK_ETA_SWEEP_CONFIDENCE_INPUT_FILE_LEDGER_V1\0")
    for row in rows:
        complex_id = row.get("id", "")
        if not complex_id:
            raise ValueError("sampling row is missing id")
        try:
            pose_hashes = json.loads(row["saved_pose_sha256_json"])
        except (KeyError, json.JSONDecodeError) as exc:
            raise ValueError(f"{complex_id}: invalid saved-pose hash ledger") from exc
        if not isinstance(pose_hashes, dict):
            raise ValueError(f"{complex_id}: saved-pose hash ledger must be an object")

        paths_and_hashes = (
            (Path(row["protein"]), row.get("protein_sha256"), "protein_sha256"),
            (
                Path(row["ligand_ref"]),
                row.get("ligand_reference_sha256"),
                "ligand_reference_sha256",
            ),
            (
                pose_dir / f"{complex_id}.sdf",
                pose_hashes.get(selector),
                "selected_pose_sha256",
            ),
        )
        record: dict[str, str] = {"id": complex_id, "selector": selector}
        for path, expected_raw, hash_label in paths_and_hashes:
            expected = _require_sha256(expected_raw, label=f"{complex_id}.{hash_label}")
            if not path.is_file():
                raise FileNotFoundError(f"{complex_id}: missing bound file {path}")
            observed = file_sha256(path)
            if observed != expected:
                raise ValueError(f"{complex_id}: {hash_label} differs from sampling-time hash")
            record[hash_label] = expected
        record["sampling_row_sha256"] = hashlib.sha256(_canonical_json(row).encode()).hexdigest()
        digest.update(_canonical_json(record).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def build_binding(
    *,
    sampling_dir: Path,
    official_dir: Path,
    run_name: str,
    protocol_id: str,
    dataset: str,
    eta: float,
    selector: str,
    shard_index: int,
    num_shards: int,
    pose_dir: Path | None = None,
) -> dict[str, Any]:
    """Build a fail-closed binding for one official PoseBusters shard."""
    if protocol_id != PROTOCOL_ID:
        raise ValueError(f"protocol_id must be {PROTOCOL_ID!r}")
    if selector not in SELECTORS:
        raise ValueError(f"selector must be one of {SELECTORS}")
    if not 0 <= shard_index < num_shards:
        raise ValueError("shard_index must satisfy 0 <= shard_index < num_shards")

    all_rows = load_rows(sampling_dir, run_name)
    rows = all_rows[shard_index::num_shards]
    if not rows:
        raise ValueError("official binding has no assigned sampling rows")
    expected_ids = [row["id"] for row in rows]
    tag = f"shard-{shard_index:03d}-of-{num_shards:03d}"
    sampling_summary = sampling_dir / f"{run_name}.{tag}.summary.json"
    sampling_csv = sampling_dir / f"{run_name}.{tag}.csv"
    official_summary = official_dir / f"{tag}.summary.json"
    official_csv = official_dir / f"{tag}.csv"
    for path in (sampling_summary, sampling_csv, official_summary, official_csv):
        if not path.is_file():
            raise FileNotFoundError(f"missing binding input: {path}")

    sampling_meta = json.loads(sampling_summary.read_text())
    if not isinstance(sampling_meta, dict):
        raise ValueError(f"{sampling_summary}: summary must be a JSON object")
    expected_sampling = {
        "protocol_id": PROTOCOL_ID,
        "run_name": run_name,
        "dataset": dataset,
        "shard_index": shard_index,
        "num_shards": num_shards,
        "confidence_checkpoint_sha256": CONFIDENCE_CHECKPOINT_SHA256,
        "expected_discovered_count": len(all_rows),
        "num_discovered_total": len(all_rows),
        "num_assigned": len(rows),
        "num_success": len(rows),
        "num_failed": 0,
        "num_samples": 100,
        "num_steps": 10,
        "model_pose_step_budget": 1000,
        "require_complete_success": True,
    }
    for key, expected in expected_sampling.items():
        if sampling_meta.get(key) != expected:
            raise ValueError(
                f"{sampling_summary}: {key} mismatch; "
                f"expected {expected!r}, got {sampling_meta.get(key)!r}"
            )
    if sampling_meta.get("failures") != []:
        raise ValueError(f"{sampling_summary}: complete sampling shard requires no failures")
    if not math.isclose(
        float(sampling_meta.get("unified_guidance_scale", math.nan)),
        eta,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(f"{sampling_summary}: eta mismatch")
    resolved_sampling_csv = _resolve_declared_file(
        sampling_meta.get("csv"),
        summary_path=sampling_summary,
        expected_path=sampling_csv,
        label="sampling csv",
    )
    if _read_csv_ids(resolved_sampling_csv) != expected_ids:
        raise ValueError(f"{sampling_csv}: IDs/order differ from deterministic sampling shard")

    official_meta = json.loads(official_summary.read_text())
    if not isinstance(official_meta, dict):
        raise ValueError(f"{official_summary}: summary must be a JSON object")
    expected_official = {
        "posebusters_version": "0.6.5",
        "config": "redock",
        "selector": selector,
        "input_hashes_verified": True,
        "num_input_hashes_verified": len(rows),
        "expected_discovered_count": len(all_rows),
        "num_discovered_total": len(all_rows),
        "num_assigned": len(rows),
        "num_success": len(rows),
        "num_failed": 0,
        "require_complete_success": True,
    }
    for key, expected in expected_official.items():
        if official_meta.get(key) != expected:
            raise ValueError(
                f"{official_summary}: {key} mismatch; "
                f"expected {expected!r}, got {official_meta.get(key)!r}"
            )
    if official_meta.get("failures") != []:
        raise ValueError(f"{official_summary}: strict binding rejects failures")
    resolved_official_csv = _resolve_declared_file(
        official_meta.get("csv"),
        summary_path=official_summary,
        expected_path=official_csv,
        label="official csv",
    )
    if _read_csv_ids(resolved_official_csv) != expected_ids:
        raise ValueError(f"{official_csv}: official IDs/order differ from sampling shard")

    selected_pose_dir = (
        pose_dir if pose_dir is not None else sampling_dir / "poses" / run_name / dataset / selector
    )
    return {
        "binding_contract": BINDING_CONTRACT,
        "protocol_id": PROTOCOL_ID,
        "run_name": run_name,
        "dataset": dataset,
        "eta": eta,
        "selector": selector,
        "selector_role": "primary" if selector == "confidence" else "diagnostic",
        "confidence_checkpoint_sha256": CONFIDENCE_CHECKPOINT_SHA256,
        "shard_index": shard_index,
        "num_shards": num_shards,
        "count": len(rows),
        "ordered_ids_sha256": _ordered_ids_sha256(expected_ids),
        "sorted_ids_sha256": sorted_id_sha256(expected_ids),
        "sampling_input_file_ledger_sha256": _file_ledger(
            rows,
            selector=selector,
            pose_dir=selected_pose_dir,
        ),
        "sampling_summary_sha256": file_sha256(sampling_summary),
        "sampling_csv_sha256": file_sha256(sampling_csv),
        "official_summary_sha256": file_sha256(official_summary),
        "official_csv_sha256": file_sha256(official_csv),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sampling-dir", type=Path, required=True)
    parser.add_argument("--official-dir", type=Path, required=True)
    parser.add_argument("--pose-dir", type=Path)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--protocol-id", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--eta", type=float, required=True)
    parser.add_argument("--selector", choices=SELECTORS, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    args = parser.parse_args(argv)
    binding = build_binding(
        sampling_dir=args.sampling_dir,
        official_dir=args.official_dir,
        pose_dir=args.pose_dir,
        run_name=args.run_name,
        protocol_id=args.protocol_id,
        dataset=args.dataset,
        eta=args.eta,
        selector=args.selector,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
    )
    tag = f"shard-{args.shard_index:03d}-of-{args.num_shards:03d}"
    output = args.official_dir / f"{tag}.binding.json"
    output.write_text(json.dumps(binding, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(binding, sort_keys=True))


if __name__ == "__main__":
    main()
