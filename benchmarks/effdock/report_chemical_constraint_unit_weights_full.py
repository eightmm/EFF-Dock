#!/usr/bin/env python3
"""Strict aggregate for the chemical-constraint unit-weights full run."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from rdkit import Chem

from effdock.workflows.benchmark_inputs import load_benchmark_inputs, load_benchmark_ligand
from effdock.workflows.stereo_filter import explicit_stereo_constraints, stereo_compatibility

PROTOCOL_ID = "EFFDOCK-CHEMICAL-CONSTRAINT-UNIT-WEIGHTS-FULL-V1"
DATASET_COUNTS = {"astex": 85, "posebusters": 308}
ARMS = ("physical_interaction_1", "all_three_1")
CHEMICAL_STRENGTHS = {"physical_interaction_1": 0.0, "all_three_1": 1.0}
DOCKING_SHA256 = "65be44d7dc8f0867eb9fc5d22214b80f93971ea4702679a527c665046e91e6b6"
CONFIDENCE_SHA256 = "ce59be42f0ca613871ca079127c3296f5ca9a4ec72e44a9e5cf61878351c2638"


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    raise ValueError(f"expected boolean field, got {value!r}")


def _run_name(dataset: str, arm: str) -> str:
    return f"effdock-chemical-unit1-full-{dataset}-{arm}"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _sampling_records(
    output_root: Path,
    dataset: str,
    arm: str,
    expected_ids: set[str],
) -> tuple[dict[str, dict[str, str]], list[dict[str, Any]]]:
    run_name = _run_name(dataset, arm)
    paths = sorted((output_root / "raw").glob(f"{run_name}.shard-*-of-008.csv"))
    if len(paths) != 8:
        raise ValueError(f"{run_name}: expected 8 sampling CSVs, got {len(paths)}")
    rows: dict[str, dict[str, str]] = {}
    summaries: list[dict[str, Any]] = []
    for path in paths:
        summary_path = path.with_suffix(".summary.json")
        summary = json.loads(summary_path.read_text())
        summaries.append(summary)
        exact = {
            "protocol_id": PROTOCOL_ID,
            "dataset": dataset,
            "run_name": run_name,
            "selector_profile": "confidence_cluster_free",
            "num_samples": 100,
            "num_steps": 10,
            "model_pose_step_budget": 1000,
            "sigma": 2.0,
            "time_schedule": "late",
            "schedule_power": 3.0,
            "pocket_cutoff": 10.0,
            "center_jitter_sigma": 0.0,
            "ligand_conformer_seed": 0,
            "refine": "none",
            "unified_guidance_scale": 1.0,
            "unified_guidance_mode": "normalized_drift",
            "unified_guidance_start_t": 0.5,
            "unified_guidance_chemical_constraint_strength": CHEMICAL_STRENGTHS[arm],
            "checkpoint_sha256": DOCKING_SHA256,
            "confidence_checkpoint_sha256": CONFIDENCE_SHA256,
            "num_discovered_total": DATASET_COUNTS[dataset],
            "num_shards": 8,
        }
        for key, expected in exact.items():
            if summary.get(key) != expected:
                raise ValueError(
                    f"{summary_path}: {key} must be {expected!r}, got {summary.get(key)!r}"
                )
        if summary.get("num_failed") != 0 or summary.get("failures") != []:
            raise ValueError(f"{summary_path}: sampling failures are not allowed")
        if summary.get("num_success") != summary.get("num_assigned"):
            raise ValueError(f"{summary_path}: incomplete sampling shard")
        for row in _read_csv(path):
            complex_id = row["id"]
            if complex_id in rows:
                raise ValueError(f"{run_name}: duplicate complex {complex_id}")
            rows[complex_id] = row
    if set(rows) != expected_ids:
        raise ValueError(
            f"{run_name}: ID mismatch; missing={sorted(expected_ids - set(rows))[:8]} "
            f"extra={sorted(set(rows) - expected_ids)[:8]}"
        )
    return rows, summaries


def _official_records(
    output_root: Path,
    dataset: str,
    arm: str,
    expected_ids: set[str],
) -> dict[str, dict[str, str]]:
    run_name = _run_name(dataset, arm)
    root = output_root / "posebusters_official" / run_name
    paths = sorted(root.glob("shard-*-of-008.csv"))
    if len(paths) != 8:
        raise ValueError(f"{run_name}: expected 8 official CSVs, got {len(paths)}")
    rows: dict[str, dict[str, str]] = {}
    for path in paths:
        summary = json.loads(path.with_suffix(".summary.json").read_text())
        if summary.get("num_failed") != 0 or summary.get("failures") != []:
            raise ValueError(f"{path}: official PoseBusters failures are not allowed")
        if summary.get("num_success") != summary.get("num_assigned"):
            raise ValueError(f"{path}: incomplete official PoseBusters shard")
        if summary.get("selector") != "confidence":
            raise ValueError(f"{path}: wrong selector")
        if summary.get("input_hashes_verified") is not True:
            raise ValueError(f"{path}: sampling input hashes were not verified")
        for row in _read_csv(path):
            complex_id = row["id"]
            if complex_id in rows:
                raise ValueError(f"{run_name}: duplicate official row {complex_id}")
            rows[complex_id] = row
    if set(rows) != expected_ids:
        raise ValueError(f"{run_name}: official PoseBusters ID mismatch")
    return rows


def _candidate_stereo(
    dataset: str,
    rows: dict[str, dict[str, str]],
    ligand_mapping: dict[str, Any],
) -> tuple[dict[str, int], dict[str, dict[str, int | bool]]]:
    totals = {
        "candidate_count": 0,
        "candidate_stereo_valid_count": 0,
        "candidate_stereo_and_rmsd_lt2_count": 0,
        "complex_any_stereo_valid_count": 0,
        "complex_any_stereo_and_rmsd_lt2_count": 0,
        "confidence_stereo_valid_count": 0,
    }
    per_id: dict[str, dict[str, int | bool]] = {}
    for complex_id in sorted(rows):
        row = rows[complex_id]
        mol_input, _ = load_benchmark_ligand(ligand_mapping[complex_id], random_seed=0)
        expected = explicit_stereo_constraints(mol_input)
        pose_path = Path(row["all_poses_sdf"])
        poses = [mol for mol in Chem.SDMolSupplier(str(pose_path), removeHs=False) if mol]
        rmsds = [float(value) for value in json.loads(row["candidate_rmsds_json"])]
        if len(poses) != 100 or len(rmsds) != 100:
            raise ValueError(f"{dataset}/{complex_id}: expected 100 ordered candidates")
        valid = [stereo_compatibility(expected, mol).valid for mol in poses]
        confidence_index = int(row["confidence_index"])
        stereo_count = sum(valid)
        stereo_rmsd2_count = sum(ok and rmsd < 2.0 for ok, rmsd in zip(valid, rmsds, strict=True))
        totals["candidate_count"] += len(valid)
        totals["candidate_stereo_valid_count"] += stereo_count
        totals["candidate_stereo_and_rmsd_lt2_count"] += stereo_rmsd2_count
        totals["complex_any_stereo_valid_count"] += stereo_count > 0
        totals["complex_any_stereo_and_rmsd_lt2_count"] += stereo_rmsd2_count > 0
        totals["confidence_stereo_valid_count"] += valid[confidence_index]
        per_id[complex_id] = {
            "candidate_stereo_valid_count": stereo_count,
            "candidate_stereo_and_rmsd_lt2_count": stereo_rmsd2_count,
            "confidence_stereo_valid": valid[confidence_index],
        }
    return totals, per_id


def _aggregate_arm(
    dataset: str,
    arm: str,
    sampling: dict[str, dict[str, str]],
    summaries: list[dict[str, Any]],
    official: dict[str, dict[str, str]],
    stereo_totals: dict[str, int],
) -> dict[str, Any]:
    ids = sorted(sampling)
    confidence_success = sum(float(sampling[key]["confidence_rmsd"]) < 2.0 for key in ids)
    oracle_success = sum(float(sampling[key]["oracle_rmsd"]) < 2.0 for key in ids)
    confidence_fast = sum(_bool(sampling[key]["confidence_fast_valid"]) for key in ids)
    official_valid = sum(_bool(official[key]["posebusters_valid"]) for key in ids)
    joint = sum(
        _bool(official[key]["posebusters_valid"])
        and float(sampling[key]["confidence_rmsd"]) < 2.0
        for key in ids
    )
    selected_stereo_checks = sum(
        _bool(official[key]["tetrahedral_chirality"])
        and _bool(official[key]["double_bond_stereochemistry"])
        for key in ids
    )
    runtime_stats = [summary.get("guidance_runtime_stats", {}) for summary in summaries]

    def sum_stat(name: str) -> int:
        return sum(int(stats.get(name, 0) or 0) for stats in runtime_stats)

    return {
        "complex_count": len(ids),
        "confidence_rmsd_lt2_count": confidence_success,
        "oracle_rmsd_lt2_count": oracle_success,
        "confidence_fast_valid_count": confidence_fast,
        "official_posebusters_valid_count": official_valid,
        "official_posebusters_valid_and_rmsd_lt2_count": joint,
        "official_selected_stereo_checks_pass_count": selected_stereo_checks,
        **stereo_totals,
        "chemical_constraint_pose_applications": sum_stat(
            "direct_chemical_constraint_pose_applied"
        ),
        "chemical_constraint_cap_triggers": sum_stat(
            "direct_chemical_constraint_cap_trigger_count"
        ),
        "chemical_constraint_nonfinite_poses": sum_stat(
            "direct_chemical_constraint_nonfinite_poses"
        ),
        "direct_nonfinite_poses": sum_stat("direct_nonfinite_poses"),
        "cuda_max_memory_allocated_bytes": max(
            int(summary["runtime"]["cuda_max_memory_allocated_bytes"]) for summary in summaries
        ),
    }


def main() -> None:
    args = _args()
    audit = json.loads((args.output_root / "audit" / "combined.json").read_text())
    result: dict[str, Any] = {
        "schema_version": "effdock.chemical_constraint_unit_weights_full.v1",
        "protocol_id": PROTOCOL_ID,
        "status": "complete",
        "claim_boundary": "paired repeated-external descriptive characterization",
        "datasets": {},
    }
    for dataset, expected_count in DATASET_COUNTS.items():
        audit_entry = audit.get("datasets", {}).get(dataset, {})
        expected_ids = set(audit_entry.get("success_ids", audit_entry.get("ids", [])))
        if len(expected_ids) != expected_count:
            raise ValueError(f"{dataset}: fresh audit does not contain {expected_count} IDs")
        ligand_mapping, mapping_identity = load_benchmark_inputs(
            dataset,
            Path("data/external_test"),
            Path("docs/GUIDANCE_BUDGET1000_FULL_INPUTS.json"),
        )
        dataset_result: dict[str, Any] = {
            "count": expected_count,
            "mapping_identity_sha256": mapping_identity["sha256"],
            "arms": {},
            "paired_prior_verified": True,
        }
        prior_by_arm: dict[str, dict[str, str]] = {}
        for arm in ARMS:
            sampling, summaries = _sampling_records(
                args.output_root, dataset, arm, expected_ids
            )
            official = _official_records(args.output_root, dataset, arm, expected_ids)
            stereo_totals, _ = _candidate_stereo(dataset, sampling, ligand_mapping)
            prior_by_arm[arm] = {
                complex_id: row["prior_pool_sha256"] for complex_id, row in sampling.items()
            }
            dataset_result["arms"][arm] = _aggregate_arm(
                dataset,
                arm,
                sampling,
                summaries,
                official,
                stereo_totals,
            )
        if prior_by_arm[ARMS[0]] != prior_by_arm[ARMS[1]]:
            dataset_result["paired_prior_verified"] = False
            raise ValueError(f"{dataset}: paired prior hashes differ")
        dataset_result["all_three_minus_physical_interaction"] = {
            key: dataset_result["arms"][ARMS[1]][key]
            - dataset_result["arms"][ARMS[0]][key]
            for key in (
                "confidence_rmsd_lt2_count",
                "oracle_rmsd_lt2_count",
                "confidence_fast_valid_count",
                "official_posebusters_valid_count",
                "official_posebusters_valid_and_rmsd_lt2_count",
                "official_selected_stereo_checks_pass_count",
                "candidate_stereo_valid_count",
                "candidate_stereo_and_rmsd_lt2_count",
                "confidence_stereo_valid_count",
            )
        }
        result["datasets"][dataset] = dataset_result
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
