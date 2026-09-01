#!/usr/bin/env python
"""Audit crystal-local versus SMILES/RDKit fragment geometry on PLINDER."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, rdBase

from effdock.evaluation.fragment_geometry import (
    analyze_fragment_geometry_pair,
    partitions_equivalent,
)
from effdock.inference.preprocess import generate_smiles_conformer
from effdock.preprocess.fragments import decompose_fragments
from effdock.preprocess.ligand import load_molecule
from effdock.workflows.prepare import sample_key

SCHEMA_VERSION = "effdock.rdkit_fragment_geometry_audit.v1"
DEFAULT_THRESHOLDS = (0.25, 0.50, 1.00, 2.00)
PROTOCOL_PATH = Path("docs/RDKIT_FRAGMENT_GEOMETRY_AUDIT_PROTOCOL.md")
IMPLEMENTATION_FILES = (
    Path("scripts/analyze_rdkit_fragment_geometry.py"),
    Path("src/effdock/evaluation/fragment_geometry.py"),
    Path("src/effdock/inference/preprocess.py"),
    Path("src/effdock/preprocess/fragments.py"),
    Path("src/effdock/preprocess/ligand.py"),
    Path("src/effdock/evaluation/benchmark.py"),
    Path("src/effdock/workflows/benchmark_inputs.py"),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _quantiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    array = np.asarray(values, dtype=np.float64)
    return {
        label: float(np.quantile(array, quantile))
        for label, quantile in (
            ("min", 0.00),
            ("p25", 0.25),
            ("p50", 0.50),
            ("p75", 0.75),
            ("p90", 0.90),
            ("p95", 0.95),
            ("p99", 0.99),
            ("max", 1.00),
        )
    }


def _threshold_summary(values: list[float]) -> dict[str, dict[str, float | int]]:
    return {
        f"ge_{threshold:g}A": {
            "count": sum(value >= threshold for value in values),
            "fraction": (
                sum(value >= threshold for value in values) / len(values) if values else 0.0
            ),
        }
        for threshold in DEFAULT_THRESHOLDS
    }


def _input_cohort_sha256(records: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256(b"EFFDOCK_RDKIT_FRAGMENT_GEOMETRY_INPUT_COHORT_V1\0")
    for record in sorted(records, key=lambda item: item["sample_key"]):
        digest.update(str(record["sample_key"]).encode("utf-8"))
        digest.update(b"\0")
        hashes = record.get("input_sha256", {})
        for name in ("processed_ligand_pt", "processed_meta_pt", "raw_ligand_sdf"):
            digest.update(name.encode("utf-8"))
            digest.update(b"=")
            digest.update(str(hashes.get(name, "missing")).encode("utf-8"))
            digest.update(b"\0")
    return digest.hexdigest()


def _apply_hydrogen_policy(mol: Chem.Mol, policy: str) -> Chem.Mol:
    """Apply an explicit, label-blind ligand hydrogen policy."""
    if policy == "remove_hs":
        return mol
    if policy != "remove_all_hs":
        raise ValueError(f"unknown hydrogen policy: {policy!r}")
    normalized = Chem.RemoveAllHs(mol)
    hydrogen_count = sum(atom.GetAtomicNum() == 1 for atom in normalized.GetAtoms())
    if hydrogen_count:
        raise ValueError(
            "heavy-only normalization left "
            f"{hydrogen_count} hydrogen atom(s) after RemoveAllHs"
        )
    if normalized.GetNumConformers() != 1 or not normalized.GetConformer().Is3D():
        raise ValueError("heavy-only normalization did not preserve one 3D conformer")
    return normalized


def _analyze_one(task: dict[str, Any]) -> dict[str, Any]:
    torch.set_num_threads(1)
    key = str(task["sample_key"])
    stage = "resolve_inputs"
    record: dict[str, Any] = {
        "sample_key": key,
        "system_id": str(task["system_id"]),
        "ligand_instance_chain": str(task["ligand_instance_chain"]),
        "status": "failed",
    }
    try:
        processed_dir = Path(task["processed_root"]) / key
        ligand_path = processed_dir / "ligand.pt"
        meta_path = processed_dir / "meta.pt"
        if not ligand_path.is_file() or not meta_path.is_file():
            raise FileNotFoundError("processed ligand.pt/meta.pt missing")
        stage = "load_processed"
        ligand = torch.load(ligand_path, map_location="cpu", weights_only=True)
        meta = torch.load(meta_path, map_location="cpu", weights_only=True)

        raw_sdf = (
            Path(task["plinder_root"])
            / "systems"
            / str(task["system_id"])
            / "ligand_files"
            / f"{task['ligand_instance_chain']}.sdf"
        )
        if not raw_sdf.is_file():
            raise FileNotFoundError("raw crystal ligand SDF missing")
        stage = "hash_inputs"
        record["input_sha256"] = {
            "processed_ligand_pt": _sha256(ligand_path),
            "processed_meta_pt": _sha256(meta_path),
            "raw_ligand_sdf": _sha256(raw_sdf),
        }
        stage = "load_crystal"
        crystal_mol, _, crystal_sanitize_ok = load_molecule(raw_sdf)
        if crystal_mol is None:
            raise ValueError("crystal SDF load failed")
        crystal_coords_raw = torch.as_tensor(
            crystal_mol.GetConformer().GetPositions(),
            dtype=torch.float64,
        )
        crystal_coords = ligand["atom_coords"].to(torch.float64)
        if crystal_coords_raw.shape != crystal_coords.shape:
            raise ValueError("raw and processed crystal atom counts differ")
        crystal_coordinate_max_abs_error = float(
            (crystal_coords_raw - crystal_coords).abs().max().item()
        )
        if crystal_coordinate_max_abs_error > float(task["processed_coordinate_tolerance"]):
            raise ValueError(
                "raw and processed crystal coordinates differ by "
                f"{crystal_coordinate_max_abs_error:.6g} A"
            )
        crystal_fragment_check = decompose_fragments(crystal_mol, crystal_coords_raw)
        if crystal_fragment_check is None or not partitions_equivalent(
            crystal_fragment_check["fragment_id"],
            ligand["fragment_id"],
        ):
            raise ValueError("raw and processed crystal fragment partitions differ")

        smiles = str(task["smiles"])
        stage = "generate_smiles_conformer"
        inference_mol, conformer_metadata = generate_smiles_conformer(
            smiles,
            random_seed=int(task["rdkit_seed"]),
        )
        inference_mol = _apply_hydrogen_policy(
            inference_mol,
            str(task["hydrogen_policy"]),
        )
        conformer_metadata["hydrogen_policy"] = str(task["hydrogen_policy"])
        stage = "map_and_fit_fragments"
        result = analyze_fragment_geometry_pair(
            crystal_mol,
            inference_mol,
            crystal_coords,
            ligand["fragment_id"],
            max_matches=int(task["max_matches"]),
        )

        for fragment in result["fragments"]:
            ring_atom_count = sum(
                crystal_mol.GetAtomWithIdx(int(atom_index)).IsInRing()
                for atom_index in fragment["crystal_atom_indices"]
            )
            fragment["ring_atom_count"] = int(ring_atom_count)
            fragment["contains_ring"] = ring_atom_count > 0

        expected_atoms = int(meta["num_atom"])
        if result["atom_count"] != expected_atoms:
            raise ValueError("analyzed atom count differs from processed metadata")

        stage = "complete"
        record.update(
            {
                "status": "ok",
                "smiles_sha256": hashlib.sha256(smiles.encode("utf-8")).hexdigest(),
                "crystal_sanitize_ok": bool(crystal_sanitize_ok),
                "processed_coordinate_max_abs_error": crystal_coordinate_max_abs_error,
                "stored_fragment_count_from_meta": int(meta["num_frag"]),
                "conformer": conformer_metadata,
                **result,
            }
        )
    except Exception as error:
        record["failure_stage"] = stage
        record["failure_code"] = f"{stage}:{type(error).__name__}"
        record["error_type"] = type(error).__name__
        record["error"] = str(error)
    return record


def _aggregate(
    records: list[dict[str, Any]],
    *,
    min_coverage: float,
    min_symmetry_coverage: float,
) -> dict[str, Any]:
    valid = [record for record in records if record["status"] == "ok"]
    floors = [float(record["rigid_fragment_floor_rmsd"]) for record in valid]
    pair_errors = [float(record["pair_distance_rmse"]) for record in valid]
    max_fragment_errors = [float(record["max_fragment_rmsd"]) for record in valid]
    fragments = [fragment for record in valid for fragment in record["fragments"]]
    fragment_errors = [float(fragment["rigid_fit_rmsd"]) for fragment in fragments]
    ring_fragment_errors = [
        float(fragment["rigid_fit_rmsd"])
        for fragment in fragments
        if bool(fragment["contains_ring"])
    ]
    nonring_fragment_errors = [
        float(fragment["rigid_fit_rmsd"])
        for fragment in fragments
        if not bool(fragment["contains_ring"])
    ]
    mmff_status_slices = {
        str(status): [
            float(record["rigid_fragment_floor_rmsd"])
            for record in valid
            if int(record["conformer"]["mmff_status"]) == status
        ]
        for status in sorted(
            {int(record["conformer"]["mmff_status"]) for record in valid}
        )
    }
    attempted = len(records)
    coverage = len(valid) / attempted if attempted else 0.0
    symmetry_complete_count = sum(bool(record["symmetry_complete"]) for record in valid)
    symmetry_coverage = symmetry_complete_count / len(valid) if valid else 0.0
    total_squared_error = sum(float(record["squared_error"]) for record in valid)
    total_atoms = sum(int(record["atom_count"]) for record in valid)
    p90 = _quantiles(floors).get("p90", math.nan)
    fraction_ge_half = (
        sum(value >= 0.5 for value in floors) / len(floors) if floors else math.nan
    )
    if coverage < min_coverage:
        decision = "invalid_coverage"
    elif symmetry_coverage < min_symmetry_coverage:
        decision = "invalid_symmetry_coverage"
    elif p90 >= 0.5 or fraction_ge_half >= 0.10:
        decision = "material_mismatch_run_paired_rdkit_local_ablation"
    elif p90 < 0.35 and fraction_ge_half < 0.05:
        decision = "small_mismatch_do_not_prioritize_rdkit_local_finetune"
    else:
        decision = "intermediate_mismatch_require_candidate_level_paired_probe"

    return {
        "attempted_complexes": attempted,
        "valid_complexes": len(valid),
        "coverage": coverage,
        "input_cohort_sha256": _input_cohort_sha256(records),
        "failure_counts": dict(
            sorted(
                Counter(
                    record.get("failure_code", "unknown")
                    for record in records
                    if record["status"] != "ok"
                ).items()
            )
        ),
        "mapping_method_counts": dict(
            sorted(Counter(str(record["mapping_method"]) for record in valid).items())
        ),
        "mapping_truncated_complexes": sum(bool(record["mapping_truncated"]) for record in valid),
        "symmetry_coverage": {
            "complete_count": symmetry_complete_count,
            "incomplete_count": len(valid) - symmetry_complete_count,
            "fraction": symmetry_coverage,
            "required_fraction": min_symmetry_coverage,
        },
        "mmff_status_counts": dict(
            sorted(Counter(str(record["conformer"]["mmff_status"]) for record in valid).items())
        ),
        "mmff_status_floor_slices": {
            status: {
                "count": len(status_floors),
                "quantiles": _quantiles(status_floors),
                "thresholds": _threshold_summary(status_floors),
            }
            for status, status_floors in mmff_status_slices.items()
        },
        "stored_partition_mismatch_complexes": {
            "count": sum(not bool(record["stored_partition_equal"]) for record in valid),
            "fraction": (
                sum(not bool(record["stored_partition_equal"]) for record in valid) / len(valid)
                if valid
                else 0.0
            ),
        },
        "complex_rigid_fragment_floor_rmsd": {
            "quantiles": _quantiles(floors),
            "thresholds": _threshold_summary(floors),
            "atom_pooled_rmsd": (
                math.sqrt(total_squared_error / total_atoms) if total_atoms else None
            ),
        },
        "complex_pair_distance_rmse": {"quantiles": _quantiles(pair_errors)},
        "complex_max_fragment_rmsd": {"quantiles": _quantiles(max_fragment_errors)},
        "fragment_rigid_fit_rmsd": {
            "count": len(fragment_errors),
            "quantiles": _quantiles(fragment_errors),
            "thresholds": _threshold_summary(fragment_errors),
        },
        "ring_fragment_rigid_fit_rmsd": {
            "count": len(ring_fragment_errors),
            "quantiles": _quantiles(ring_fragment_errors),
        },
        "nonring_fragment_rigid_fit_rmsd": {
            "count": len(nonring_fragment_errors),
            "quantiles": _quantiles(nonring_fragment_errors),
        },
        "decision": decision,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool-parquet", type=Path, default=Path("data/plinder_pool.parquet"))
    parser.add_argument("--split-file", type=Path, default=Path("data/splits/plinder.json"))
    parser.add_argument("--split-key", choices=("train", "val"), default="val")
    parser.add_argument("--processed-root", type=Path, default=Path("data/plinder_processed"))
    parser.add_argument(
        "--plinder-root",
        type=Path,
        default=Path.home() / ".local/share/plinder/2024-06/v2",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rdkit-seed", type=int, default=0)
    parser.add_argument("--max-matches", type=int, default=1024)
    parser.add_argument("--processed-coordinate-tolerance", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--min-coverage", type=float, default=0.95)
    parser.add_argument("--min-symmetry-coverage", type=float, default=0.95)
    parser.add_argument(
        "--hydrogen-policy",
        choices=("remove_hs", "remove_all_hs"),
        default="remove_hs",
        help=(
            "remove_hs preserves the current public loader output; remove_all_hs applies "
            "the explicit heavy-atom normalization used by frozen external benchmarks"
        ),
    )
    parser.add_argument("--protocol-path", type=Path, default=PROTOCOL_PATH)
    parser.add_argument(
        "--protocol-id",
        default="EFFDOCK-RDKIT-FRAGMENT-GEOMETRY-AUDIT-V1",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.workers <= 0:
        raise ValueError("workers must be positive")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("limit must be positive")
    if not 0.0 <= args.min_coverage <= 1.0:
        raise ValueError("min-coverage must lie in [0, 1]")
    if not 0.0 <= args.min_symmetry_coverage <= 1.0:
        raise ValueError("min-symmetry-coverage must lie in [0, 1]")
    if not args.protocol_path.is_file():
        raise FileNotFoundError(f"protocol file is missing: {args.protocol_path}")

    with args.split_file.open() as handle:
        split = json.load(handle)
    sample_ids = [str(value) for value in split[args.split_key]]
    if args.limit is not None:
        sample_ids = sample_ids[: args.limit]

    pool = pd.read_parquet(args.pool_parquet)
    pool = pool.assign(
        sample_key=[
            sample_key(str(system_id), str(chain))
            for system_id, chain in zip(
                pool["system_id"],
                pool["ligand_instance_chain"],
                strict=True,
            )
        ]
    )
    if bool(pool["sample_key"].duplicated().any()):
        raise ValueError("pool parquet contains duplicate sample keys")
    rows = pool.set_index("sample_key", drop=False).to_dict("index")
    missing_rows = [key for key in sample_ids if key not in rows]
    if missing_rows:
        raise ValueError(f"{len(missing_rows)} split IDs are absent from the pool parquet")

    common = {
        "processed_root": str(args.processed_root),
        "plinder_root": str(args.plinder_root),
        "rdkit_seed": args.rdkit_seed,
        "max_matches": args.max_matches,
        "processed_coordinate_tolerance": args.processed_coordinate_tolerance,
        "hydrogen_policy": args.hydrogen_policy,
    }
    tasks = [
        {
            **common,
            "sample_key": key,
            "system_id": rows[key]["system_id"],
            "ligand_instance_chain": rows[key]["ligand_instance_chain"],
            "smiles": rows[key]["ligand_rdkit_canonical_smiles"],
        }
        for key in sample_ids
    ]

    if args.workers == 1:
        records = [_analyze_one(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            records = list(executor.map(_analyze_one, tasks, chunksize=4))
    records.sort(key=lambda record: record["sample_key"])
    summary = _aggregate(
        records,
        min_coverage=args.min_coverage,
        min_symmetry_coverage=args.min_symmetry_coverage,
    )

    payload = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": args.protocol_id,
        "question": (
            "How much crystal-vs-SMILES/RDKit intrafragment error remains after the "
            "best independent proper rigid fit of every fragment?"
        ),
        "non_goal": "RDKit whole-conformer placement is never used as a pose prior.",
        "config": {
            "split_key": args.split_key,
            "requested_complexes": len(sample_ids),
            "rdkit_seed": args.rdkit_seed,
            "conformer_recipe": (
                "MolFromSmiles/AddHs/ETKDGv3/MMFFOptimizeMolecule(200)/"
                + ("RemoveHs" if args.hydrogen_policy == "remove_hs" else "RemoveHs/RemoveAllHs")
            ),
            "hydrogen_policy": args.hydrogen_policy,
            "max_symmetry_matches": args.max_matches,
            "processed_coordinate_tolerance_A": args.processed_coordinate_tolerance,
            "min_coverage": args.min_coverage,
            "min_symmetry_coverage": args.min_symmetry_coverage,
            "decision_rule": {
                "validity": "mapping coverage and complete-symmetry coverage are each >=95%",
                "material": "p90 floor >=0.5 A or >=10% complexes have floor >=0.5 A",
                "small": "p90 floor <0.35 A and <5% complexes have floor >=0.5 A",
                "otherwise": "candidate-level paired probe",
            },
        },
        "inputs": {
            "pool_parquet": str(args.pool_parquet),
            "pool_parquet_sha256": _sha256(args.pool_parquet),
            "split_file": str(args.split_file),
            "split_file_sha256": _sha256(args.split_file),
            "processed_root": str(args.processed_root),
            "plinder_release": "2024-06/v2",
            "protocol": str(args.protocol_path),
            "protocol_sha256": _sha256(args.protocol_path),
            "implementation_sha256": {
                str(path): _sha256(path) for path in IMPLEMENTATION_FILES
            },
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "rdkit": rdBase.rdkitVersion,
        },
        "summary": summary,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(args.output)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if summary["coverage"] < args.min_coverage:
        print("coverage gate failed", file=sys.stderr)
        return 2
    if summary["symmetry_coverage"]["fraction"] < args.min_symmetry_coverage:
        print("symmetry-coverage gate failed", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
