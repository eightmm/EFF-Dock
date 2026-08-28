"""Run official PoseBusters redock checks on selected PoseBusters poses."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from importlib.metadata import version as distribution_version
from pathlib import Path

import pandas as pd
from posebusters import PoseBusters

EXPECTED_POSEBUSTERS_VERSION = "0.6.5"

# PoseBusters 0.6.5 ``redock`` binary checks used for pass-all validity.
# The separately reported RMSD check is intentionally not part of this tuple.
VALIDITY_CHECKS = (
    "mol_pred_loaded",
    "mol_true_loaded",
    "mol_cond_loaded",
    "sanitization",
    "inchi_convertible",
    "all_atoms_connected",
    "no_radicals",
    "molecular_formula",
    "molecular_bonds",
    "double_bond_stereochemistry",
    "tetrahedral_chirality",
    "bond_lengths",
    "bond_angles",
    "internal_steric_clash",
    "aromatic_ring_flatness",
    "non-aromatic_ring_non-flatness",
    "double_bond_flatness",
    "internal_energy",
    "protein-ligand_maximum_distance",
    "minimum_distance_to_protein",
    "minimum_distance_to_organic_cofactors",
    "minimum_distance_to_inorganic_cofactors",
    "minimum_distance_to_waters",
    "volume_overlap_with_protein",
    "volume_overlap_with_organic_cofactors",
    "volume_overlap_with_inorganic_cofactors",
    "volume_overlap_with_waters",
)


def require_posebusters_runtime_version() -> str:
    """Fail before evaluation when the installed official checker has drifted."""
    observed = distribution_version("posebusters")
    if observed != EXPECTED_POSEBUSTERS_VERSION:
        raise RuntimeError(
            "PoseBusters runtime version mismatch: "
            f"expected {EXPECTED_POSEBUSTERS_VERSION}, got {observed}"
        )
    return observed


def load_rows(input_dir: Path, run_name: str) -> list[dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    for path in sorted(input_dir.glob(f"{run_name}*.csv")):
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                if row["id"] in rows:
                    raise ValueError(f"duplicate PoseBusters row: {row['id']}")
                rows[row["id"]] = row
    return [rows[key] for key in sorted(rows)]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_posebusters_input_hashes(row: dict[str, str], pose_path: Path, selector: str) -> None:
    """Bind official checks to the files written and hashed during sampling."""
    for path_key, hash_key in (
        ("protein", "protein_sha256"),
        ("ligand_ref", "ligand_reference_sha256"),
    ):
        expected = row.get(hash_key, "")
        if len(expected) != 64:
            raise ValueError(f"{row.get('id')}: missing sampling-time {hash_key}")
        path = Path(row[path_key])
        if not path.is_file() or file_sha256(path) != expected:
            raise ValueError(f"{row.get('id')}: {path_key} changed after sampling")

    try:
        pose_hashes = json.loads(row.get("saved_pose_sha256_json", ""))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{row.get('id')}: invalid saved-pose hash record") from exc
    expected_pose = pose_hashes.get(selector) if isinstance(pose_hashes, dict) else None
    if not isinstance(expected_pose, str) or len(expected_pose) != 64:
        raise ValueError(f"{row.get('id')}: missing sampling-time {selector} pose hash")
    if not pose_path.is_file() or file_sha256(pose_path) != expected_pose:
        raise ValueError(f"{row.get('id')}: selected pose changed after sampling")


def require_complete_posebusters_run(
    *, num_assigned: int, num_success: int, failures: list[dict]
) -> None:
    """Turn recorded per-complex PoseBusters failures into a nonzero shard exit."""
    if num_success != num_assigned or failures:
        failed_ids = sorted(str(failure.get("id", "<unknown>")) for failure in failures)
        preview = ", ".join(failed_ids[:8])
        suffix = " ..." if len(failed_ids) > 8 else ""
        raise RuntimeError(
            "PoseBusters shard did not complete successfully: "
            f"assigned={num_assigned} success={num_success} failures={len(failures)}"
            + (f" failed_ids={preview}{suffix}" if failed_ids else "")
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("outputs/benchmarks/raw"))
    parser.add_argument("--run-name", default="effdock-redock-ema-n40-s25-v1-posebusters")
    parser.add_argument(
        "--pose-dir",
        type=Path,
        default=Path("outputs/benchmarks/raw/poses/posebusters/vina"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/benchmarks/posebusters_official")
    )
    parser.add_argument("--selector", default="effdock_torch_vina_plus_dg")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--expected-discovered-count", type=int, default=None)
    parser.add_argument(
        "--only-id",
        default=None,
        help=(
            "Evaluate one complex after validating the complete input inventory. "
            "This is intended for smoke checks only."
        ),
    )
    parser.add_argument(
        "--require-input-hashes",
        action="store_true",
        help="Verify sampling-time protein, ligand, and selected-pose SHA-256 values.",
    )
    parser.add_argument(
        "--require-complete-success",
        action="store_true",
        help="Write the shard summary, then exit nonzero if any assigned pose failed.",
    )
    args = parser.parse_args(argv)

    posebusters_version = require_posebusters_runtime_version()
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard_index must satisfy 0 <= shard_index < num_shards")
    if args.expected_discovered_count is not None and args.expected_discovered_count < 1:
        parser.error("--expected-discovered-count must be positive")
    all_rows = load_rows(args.input_dir, args.run_name)
    num_discovered_total = len(all_rows)
    if (
        args.expected_discovered_count is not None
        and len(all_rows) != args.expected_discovered_count
    ):
        raise RuntimeError(
            "PoseBusters input coverage mismatch: "
            f"expected {args.expected_discovered_count} complexes, found {len(all_rows)}"
        )
    if args.only_id is not None:
        matching = [row for row in all_rows if row["id"] == args.only_id]
        if len(matching) != 1:
            raise ValueError(
                f"--only-id {args.only_id!r} must identify exactly one discovered complex"
            )
        all_rows = matching
    rows = all_rows[args.shard_index :: args.num_shards]
    if not rows:
        raise ValueError("no PoseBusters rows assigned")

    buster = PoseBusters(config="redock", max_workers=0)
    results: list[dict] = []
    failures: list[dict] = []
    num_input_hashes_verified = 0
    for index, row in enumerate(rows, start=1):
        complex_id = row["id"]
        try:
            pose_path = args.pose_dir / f"{complex_id}.sdf"
            if args.require_input_hashes:
                verify_posebusters_input_hashes(row, pose_path, args.selector)
                num_input_hashes_verified += 1
            frame = buster.bust(
                pose_path,
                Path(row["ligand_ref"]),
                Path(row["protein"]),
                full_report=False,
            )
            checks = {
                key: False if pd.isna(value) else bool(value)
                for key, value in frame.iloc[0].to_dict().items()
            }
            validity_checks = {
                key: value for key, value in checks.items() if not key.startswith("rmsd_")
            }
            results.append(
                {
                    "id": complex_id,
                    "posebusters_valid": all(validity_checks.values()),
                    **checks,
                }
            )
            print(
                f"[{index:04d}/{len(rows)}] {complex_id} valid={results[-1]['posebusters_valid']}"
            )
        except Exception as exc:
            failures.append({"id": complex_id, "error": repr(exc)})
            print(f"[{index:04d}/{len(rows)}] {complex_id} FAIL {exc!r}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tag = f"shard-{args.shard_index:03d}-of-{args.num_shards:03d}"
    csv_path = args.output_dir / f"{tag}.csv"
    if results:
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(results[0]))
            writer.writeheader()
            writer.writerows(results)
    summary = {
        "posebusters_version": posebusters_version,
        "config": "redock",
        "selector": args.selector,
        "only_id": args.only_id,
        "pass_all_definition": "all 27 non-RMSD PoseBusters 0.6.5 redock checks",
        "rmsd_check_excluded_from_validity": True,
        "input_hashes_verified": args.require_input_hashes,
        "num_input_hashes_verified": num_input_hashes_verified,
        "expected_discovered_count": args.expected_discovered_count,
        "require_complete_success": args.require_complete_success,
        "num_discovered_total": num_discovered_total,
        "num_assigned": len(rows),
        "num_success": len(results),
        "num_failed": len(failures),
        "posebusters_valid_pct": (
            sum(result["posebusters_valid"] for result in results) / len(results) * 100
            if results
            else None
        ),
        "failures": failures,
        "csv": str(csv_path) if results else None,
    }
    summary_path = args.output_dir / f"{tag}.summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.require_complete_success:
        require_complete_posebusters_run(
            num_assigned=len(rows),
            num_success=len(results),
            failures=failures,
        )


if __name__ == "__main__":
    main()
