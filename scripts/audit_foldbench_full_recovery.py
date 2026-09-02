#!/usr/bin/env python3
"""Verify that recovery added only the previously failed FoldBench rows.

GPU replay is numerically, but not byte-for-byte, deterministic.  Preserve the
strict checks for IDs, decisions, labels, and provenance while allowing bounded
floating-point drift in replayed coordinates and confidence scores.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path

from rdkit import Chem

COORDINATE_TOLERANCE_ANGSTROM = 1e-3

HASH_FIELDS = {
    "all_poses_sdf_sha256",
    "candidate_ensemble_sha256",
    "saved_pose_sha256_json",
}
JSON_NUMERIC_TOLERANCES = {
    "candidate_rmsds_json": 2e-4,
    "confidence_candidate_scores_json": 5e-2,
}
SCALAR_NUMERIC_TOLERANCES = {
    "first_rmsd": 2e-4,
    "oracle_rmsd": 2e-4,
    "pairwise_heavy_atom_rmsd_mean": 2e-4,
    "pairwise_heavy_atom_rmsd_median": 2e-4,
    "pairwise_heavy_atom_rmsd_ge2_fraction": 2e-4,
    "nearest_neighbor_heavy_atom_rmsd_median": 2e-4,
    "fast_valid_oracle_rmsd": 2e-4,
    "mean_sample_rmsd": 2e-4,
    "confidence_rmsd": 2e-4,
    "confidence_pred_rmsd": 5e-3,
    "confidence_pred_success": 5e-2,
    "confidence_filter_rmsd": 2e-4,
    "confidence_filter_pred_rmsd": 5e-3,
    "confidence_filter_pred_success": 5e-2,
}


def max_numeric_delta(previous: object, current: object) -> float:
    """Return the largest numeric leaf delta, requiring identical structure."""
    if isinstance(previous, bool) or isinstance(current, bool):
        if previous != current:
            raise ValueError("boolean JSON leaf changed")
        return 0.0
    if isinstance(previous, (int, float)) and isinstance(current, (int, float)):
        return abs(float(previous) - float(current))
    if isinstance(previous, list) and isinstance(current, list):
        if len(previous) != len(current):
            raise ValueError("JSON list length changed")
        return max(
            (max_numeric_delta(left, right) for left, right in zip(previous, current)),
            default=0.0,
        )
    if isinstance(previous, dict) and isinstance(current, dict):
        if set(previous) != set(current):
            raise ValueError("JSON object keys changed")
        return max(
            (max_numeric_delta(previous[key], current[key]) for key in previous),
            default=0.0,
        )
    if previous != current:
        raise ValueError("non-numeric JSON leaf changed")
    return 0.0


def audit_replayed_row(
    previous: dict[str, str], current: dict[str, str]
) -> tuple[dict[str, float], list[str]]:
    """Compare one replayed row under the bounded numerical replay contract."""
    if set(previous) != set(current):
        raise ValueError("CSV columns changed")
    deltas: dict[str, float] = {}
    changed_hashes: list[str] = []
    for field, previous_value in previous.items():
        current_value = current[field]
        if previous_value == current_value:
            continue
        if field in HASH_FIELDS:
            changed_hashes.append(field)
            continue
        if field in JSON_NUMERIC_TOLERANCES:
            delta = max_numeric_delta(
                json.loads(previous_value), json.loads(current_value)
            )
            tolerance = JSON_NUMERIC_TOLERANCES[field]
        elif field in SCALAR_NUMERIC_TOLERANCES:
            delta = abs(float(previous_value) - float(current_value))
            tolerance = SCALAR_NUMERIC_TOLERANCES[field]
        else:
            raise ValueError(f"exact replay field changed: {field}")
        if delta > tolerance:
            raise ValueError(
                f"replay field {field} delta {delta:.9g} exceeds {tolerance:.9g}"
            )
        deltas[field] = delta
    return deltas, changed_hashes


def read_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    mapping = {str(row["id"]): row for row in rows}
    if len(mapping) != len(rows):
        raise ValueError(f"duplicate IDs in {path}")
    return mapping


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_replayed_coordinates(
    previous_sdf: Path, current_sdf: Path
) -> tuple[float, float]:
    """Require atom-identical ensembles with only sub-milliangstrom replay drift."""
    previous_molecules = [
        molecule
        for molecule in Chem.SDMolSupplier(
            str(previous_sdf), removeHs=False, sanitize=False
        )
        if molecule is not None
    ]
    current_molecules = [
        molecule
        for molecule in Chem.SDMolSupplier(str(current_sdf), removeHs=False, sanitize=False)
        if molecule is not None
    ]
    if len(previous_molecules) != len(current_molecules):
        raise ValueError("pose count changed")

    max_atom_displacement = 0.0
    max_pose_coordinate_rmsd = 0.0
    for pose_index, (previous_molecule, current_molecule) in enumerate(
        zip(previous_molecules, current_molecules, strict=True)
    ):
        previous_elements = [atom.GetAtomicNum() for atom in previous_molecule.GetAtoms()]
        current_elements = [atom.GetAtomicNum() for atom in current_molecule.GetAtoms()]
        if previous_elements != current_elements:
            raise ValueError(f"atom identity changed at pose {pose_index}")
        previous_positions = previous_molecule.GetConformer().GetPositions()
        current_positions = current_molecule.GetConformer().GetPositions()
        squared_displacements = [
            sum(float(delta) ** 2 for delta in previous_position - current_position)
            for previous_position, current_position in zip(
                previous_positions, current_positions, strict=True
            )
        ]
        max_atom_displacement = max(
            max_atom_displacement,
            math.sqrt(max(squared_displacements, default=0.0)),
        )
        max_pose_coordinate_rmsd = max(
            max_pose_coordinate_rmsd,
            math.sqrt(sum(squared_displacements) / max(len(squared_displacements), 1)),
        )
    if max_atom_displacement > COORDINATE_TOLERANCE_ANGSTROM:
        raise ValueError(
            f"max atom displacement {max_atom_displacement:.9g} A exceeds "
            f"{COORDINATE_TOLERANCE_ANGSTROM:.9g} A"
        )
    return max_atom_displacement, max_pose_coordinate_rmsd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--archive-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.run_root = args.run_root.resolve()
    args.archive_dir = args.archive_dir.resolve()
    args.output = args.output.resolve()
    if args.output.exists():
        raise FileExistsError(args.output)

    prefix = "effdock-foldbench-pocket-558-v1-foldbench-n100-s10-sigma2-unguided"
    recovered_ids: list[str] = []
    unchanged_ids: list[str] = []
    replay_max_numeric_delta: dict[str, float] = {}
    replay_changed_hash_counts: dict[str, int] = {}
    replay_max_atom_displacement = 0.0
    replay_max_pose_coordinate_rmsd = 0.0
    for shard_index in (21, 22):
        suffix = f"shard-{shard_index:03d}-of-044.csv"
        previous = read_rows(args.archive_dir / f"{prefix}.{suffix}")
        current = read_rows(args.run_root / "full" / "sampling" / f"{prefix}.{suffix}")
        if len(previous) != 12 or len(current) != 13:
            raise ValueError(f"shard {shard_index}: expected 12 previous and 13 current rows")
        if not set(previous) < set(current):
            raise ValueError(f"shard {shard_index}: recovery did not add exactly one ID")
        for complex_id, previous_row in previous.items():
            try:
                deltas, changed_hashes = audit_replayed_row(
                    previous_row, current[complex_id]
                )
            except ValueError as error:
                raise ValueError(
                    f"shard {shard_index}: changed prior row {complex_id}: {error}"
                ) from error
            for field, delta in deltas.items():
                replay_max_numeric_delta[field] = max(
                    replay_max_numeric_delta.get(field, 0.0), delta
                )
            for field in changed_hashes:
                replay_changed_hash_counts[field] = (
                    replay_changed_hash_counts.get(field, 0) + 1
                )
            current_sdf = Path(current[complex_id]["all_poses_sdf"])
            relative_sdf = current_sdf.relative_to(
                args.run_root / "full" / "sampling" / "poses"
            )
            previous_sdf = (
                args.archive_dir / "full" / "sampling" / "poses" / relative_sdf
            )
            if sha256(previous_sdf) != previous_row["all_poses_sdf_sha256"]:
                raise ValueError(f"shard {shard_index}: archived SDF hash mismatch")
            if sha256(current_sdf) != current[complex_id]["all_poses_sdf_sha256"]:
                raise ValueError(f"shard {shard_index}: current SDF hash mismatch")
            try:
                atom_displacement, coordinate_rmsd = audit_replayed_coordinates(
                    previous_sdf, current_sdf
                )
            except ValueError as error:
                raise ValueError(
                    f"shard {shard_index}: changed prior coordinates {complex_id}: {error}"
                ) from error
            replay_max_atom_displacement = max(
                replay_max_atom_displacement, atom_displacement
            )
            replay_max_pose_coordinate_rmsd = max(
                replay_max_pose_coordinate_rmsd, coordinate_rmsd
            )
            unchanged_ids.append(complex_id)
        recovered_ids.extend(sorted(set(current) - set(previous)))
        shard_summary = (
            args.run_root
            / "full"
            / "shards"
            / f"foldbench.shard-{shard_index:03d}-of-044.json"
        )
        summary = json.loads(shard_summary.read_text(encoding="utf-8"))
        if summary.get("status") != "complete" or int(summary.get("num_completed", -1)) != 13:
            raise ValueError(f"shard {shard_index}: incomplete recovery summary")

    all_shards = list((args.run_root / "full" / "shards").glob("foldbench.shard-*.json"))
    if len(all_shards) != 44:
        raise ValueError(f"expected 44 complete shard summaries, found {len(all_shards)}")
    expected_recovered = {
        "8ous-assembly1__protein-a__ligand-b__ccd-w3i",
        "8out-assembly1__protein-a__ligand-b__ccd-w3c",
    }
    if set(recovered_ids) != expected_recovered:
        raise ValueError(f"unexpected recovered IDs: {recovered_ids}")

    payload = {
        "schema_version": "effdock.foldbench_full_recovery_audit.v2",
        "status": "complete",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "recovered_ids": sorted(recovered_ids),
        "unchanged_replayed_ids": sorted(unchanged_ids),
        "num_unchanged_replayed": len(unchanged_ids),
        "replay_contract": {
            "exact_for_all_unlisted_fields": True,
            "json_numeric_tolerances": JSON_NUMERIC_TOLERANCES,
            "scalar_numeric_tolerances": SCALAR_NUMERIC_TOLERANCES,
            "hash_fields_allowed_to_change": sorted(HASH_FIELDS),
            "coordinate_tolerance_angstrom": COORDINATE_TOLERANCE_ANGSTROM,
        },
        "replay_max_numeric_delta": dict(sorted(replay_max_numeric_delta.items())),
        "replay_changed_hash_counts": dict(sorted(replay_changed_hash_counts.items())),
        "replay_max_atom_displacement_angstrom": replay_max_atom_displacement,
        "replay_max_pose_coordinate_rmsd_angstrom": replay_max_pose_coordinate_rmsd,
        "num_complete_shards": len(all_shards),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
