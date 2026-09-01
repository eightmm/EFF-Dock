#!/usr/bin/env python3
"""Evaluate frozen U50 Top-1 poses with PoseBusters 0.6.5."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import statistics
import tempfile
import time
from collections import Counter
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any, Iterable

from posebusters import PoseBusters
from rdkit import Chem

PROTOCOL_ID = "EFFDOCK-FIXED-NFE-STEP-POSE-PB-V1"
SOURCE_PROTOCOL_ID = "EFFDOCK-FIXED-NFE-STEP-POSE-V1"
SOURCE_SCHEMA = "effdock.fixed_nfe_step_pose_report.v1"
U50_SHA256 = "fd49fa86f67187bf26d6c1bcf2daf925ba3e3b19dfeae733e57535d183280469"
EXPECTED_COUNTS = {"astex": 85, "posebusters": 308}
EXPECTED_POSES = {"s10_n100": 100, "s25_n40": 40}
SCORE_PROTOCOLS = {
    "s10_n100": "EFFDOCK-GUIDANCE-SDF-POST-REFINEMENT-CONFIDENCE-V2",
    "s25_n40": "EFFDOCK-FIXED-NFE-STEP-POSE-U50-CONFIDENCE-V1",
}
REFINEMENT_PROTOCOLS = {
    "s10_n100": "EFFDOCK-GUIDANCE-SDF-POST-REFINEMENT-V1",
    "s25_n40": "EFFDOCK-FIXED-NFE-STEP-POSE-REFINEMENT-V1",
}
ARMS = ("s10_n100", "s25_n40")
STAGES = {"raw": "step_000", "refined": "step_100"}
SMOKE_KEYS = (("astex", "1jje"), ("posebusters", "7b2c_tp7"))
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
COFACTOR_AND_WATER_CHECKS = (
    "minimum_distance_to_organic_cofactors",
    "minimum_distance_to_inorganic_cofactors",
    "minimum_distance_to_waters",
    "volume_overlap_with_organic_cofactors",
    "volume_overlap_with_inorganic_cofactors",
    "volume_overlap_with_waters",
)
PL_VALIDITY_CHECKS = tuple(
    name for name in VALIDITY_CHECKS if name not in COFACTOR_AND_WATER_CHECKS
)
BOOLEAN_FIELDS = (
    "rmsd_lt2",
    "pl_valid",
    "posebusters_valid",
    "joint_pl_valid_rmsd_lt2",
    "joint_posebusters_valid_rmsd_lt2",
    "separate_posebusters_rmsd_check",
    *VALIDITY_CHECKS,
)

if len(VALIDITY_CHECKS) != 27 or len(PL_VALIDITY_CHECKS) != 21:
    raise RuntimeError("unexpected frozen validity schema")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _as_bool(value: Any) -> bool:
    if value is None:
        return False
    try:
        if math.isnan(float(value)):
            return False
    except (TypeError, ValueError):
        pass
    return bool(value)


def _truth(value: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"invalid boolean {value!r}")


def _validated_checks(raw: dict[str, Any], label: str) -> tuple[str, dict[str, bool]]:
    rmsd = [key for key in raw if str(key).startswith("rmsd_")]
    if len(rmsd) != 1 or set(raw) != {*VALIDITY_CHECKS, rmsd[0]}:
        raise ValueError(f"{label}: unexpected PoseBusters redock schema")
    return rmsd[0], {name: _as_bool(value) for name, value in raw.items()}


def _load_source_report(
    path: Path, expected_sha256: str
) -> tuple[dict[str, Any], dict[tuple[str, str], dict[str, str]]]:
    if file_sha256(path) != expected_sha256:
        raise ValueError("fixed-budget source report SHA-256 mismatch")
    report = json.loads(path.read_text(encoding="utf-8"))
    if (
        report.get("schema_version") != SOURCE_SCHEMA
        or report.get("protocol_id") != SOURCE_PROTOCOL_ID
        or report.get("status") != "complete_descriptive"
        or tuple(report.get("arms", {})) != ARMS
    ):
        raise ValueError("invalid fixed-budget source report")
    spec = report.get("artifacts", {}).get("complex_metrics.csv", {})
    ledger = Path(str(spec.get("path", "")))
    if not ledger.is_file() or file_sha256(ledger) != spec.get("sha256"):
        raise ValueError("missing or changed fixed-budget selected-pose ledger")
    with ledger.open(newline="", encoding="utf-8") as handle:
        raw_rows = list(csv.DictReader(handle))
    rows: dict[tuple[str, str], dict[str, str]] = {}
    for row in raw_rows:
        key = (row["dataset"], row["id"].lower())
        if key in rows:
            raise ValueError(f"duplicate source row {key}")
        rows[key] = row
    counts = Counter(dataset for dataset, _ in rows)
    if counts != Counter(EXPECTED_COUNTS):
        raise ValueError(f"unexpected source cohort {dict(counts)}")
    return report, rows


def _require_file(path: Path, expected_sha256: str, label: str) -> None:
    if not path.is_file() or file_sha256(path) != expected_sha256:
        raise ValueError(f"missing or changed {label}: {path}")


def _selected_molecule(path: Path, pose_index: int, expected_count: int) -> Chem.Mol:
    selected: Chem.Mol | None = None
    count = 0
    with path.open("rb") as handle:
        supplier = Chem.ForwardSDMolSupplier(handle, removeHs=False, sanitize=False)
        for current_index, molecule in enumerate(supplier):
            if molecule is None:
                raise ValueError(f"failed to parse pose {current_index} in {path}")
            if molecule.HasProp("sample_index") and int(molecule.GetProp("sample_index")) != current_index:
                raise ValueError(f"sample_index mismatch at pose {current_index} in {path}")
            if current_index == pose_index:
                selected = Chem.Mol(molecule)
            count += 1
    if count != expected_count or selected is None or not 0 <= pose_index < expected_count:
        raise ValueError(f"invalid pose inventory/index in {path}: {count}/{expected_count}")
    return selected


def _selected_conditions(
    *,
    dataset: str,
    complex_id: str,
    source: dict[str, str],
    score_roots: dict[str, Path],
) -> tuple[list[dict[str, Any]], Path, Path]:
    conditions: list[dict[str, Any]] = []
    identity: tuple[str, str] | None = None
    protein_path: Path | None = None
    ligand_reference_path: Path | None = None
    for arm in ARMS:
        score_summary_path = score_roots[arm] / dataset / complex_id / "summary.json"
        score_summary = json.loads(score_summary_path.read_text(encoding="utf-8"))
        inputs = score_summary.get("inputs", {})
        if (
            score_summary.get("status") != "complete_descriptive"
            or score_summary.get("protocol_id") != SCORE_PROTOCOLS[arm]
            or score_summary.get("dataset") != dataset
            or score_summary.get("complex_id") != complex_id
            or int(score_summary.get("pose_count", -1)) != EXPECTED_POSES[arm]
            or float(score_summary.get("sigma", -1)) != 2.0
            or inputs.get("confidence_checkpoint_sha256") != U50_SHA256
        ):
            raise ValueError(f"{dataset}/{complex_id}/{arm}: invalid U50 score summary")
        refinement_summary_path = Path(str(inputs.get("refinement_summary", "")))
        _require_file(
            refinement_summary_path,
            str(inputs.get("refinement_summary_sha256", "")),
            "refinement summary",
        )
        refinement = json.loads(refinement_summary_path.read_text(encoding="utf-8"))
        if (
            refinement.get("protocol_id") != REFINEMENT_PROTOCOLS[arm]
            or refinement.get("status") != "complete_descriptive"
            or int(refinement.get("counts", {}).get("poses", -1)) != EXPECTED_POSES[arm]
            or int(refinement.get("counts", {}).get("failed", -1)) != 0
        ):
            raise ValueError(f"{dataset}/{complex_id}/{arm}: invalid refinement summary")
        refinement_inputs = refinement["inputs"]
        current_identity = (
            str(refinement_inputs["protein_sha256"]),
            str(refinement_inputs["ligand_reference_sha256"]),
        )
        if identity is None:
            identity = current_identity
            protein_path = Path(refinement_inputs["protein"])
            ligand_reference_path = Path(refinement_inputs["ligand_reference"])
            _require_file(protein_path, current_identity[0], "protein")
            _require_file(ligand_reference_path, current_identity[1], "reference ligand")
        elif current_identity != identity:
            raise ValueError(f"{dataset}/{complex_id}: arm input identity mismatch")
        for stage, step_name in STAGES.items():
            selected = score_summary.get("selected", {}).get(step_name, {})
            selected_index = int(source[f"{arm}_{stage}_selected_index"])
            selected_rmsd = float(source[f"{arm}_{stage}_selected_rmsd"])
            if (
                int(selected.get("pose_index", -1)) != selected_index
                or not math.isclose(
                    float(selected.get("symmetry_rmsd_angstrom", math.nan)),
                    selected_rmsd,
                    abs_tol=1e-12,
                )
            ):
                raise ValueError(
                    f"{dataset}/{complex_id}/{arm}/{stage}: selected-pose mismatch"
                )
            artifact = refinement["artifacts"][f"{step_name}_sdf"]
            pose_path = Path(artifact["path"])
            _require_file(pose_path, artifact["sha256"], f"{arm}/{stage} pose SDF")
            molecule = _selected_molecule(pose_path, selected_index, EXPECTED_POSES[arm])
            molecule.SetProp("_Name", f"{dataset}__{complex_id}__{arm}__{stage}")
            molecule.SetProp("effdock_arm", arm)
            molecule.SetProp("effdock_stage", stage)
            molecule.SetIntProp("effdock_selected_index", selected_index)
            coordinate_sha256 = hashlib.sha256(
                Chem.MolToMolBlock(molecule).encode("utf-8")
            ).hexdigest()
            conditions.append(
                {
                    "arm": arm,
                    "stage": stage,
                    "selected_index": selected_index,
                    "selected_rmsd": selected_rmsd,
                    "pose_sdf": str(pose_path.resolve()),
                    "pose_sdf_sha256": artifact["sha256"],
                    "selected_coordinate_sha256": coordinate_sha256,
                    "molecule": molecule,
                }
            )
    assert protein_path is not None and ligand_reference_path is not None
    return conditions, ligand_reference_path, protein_path


def _write_conditions(conditions: Iterable[dict[str, Any]], path: Path) -> None:
    writer = Chem.SDWriter(str(path))
    count = 0
    try:
        for condition in conditions:
            writer.write(condition["molecule"])
            count += 1
    finally:
        writer.close()
    if count != 4 or not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError("failed to materialize four selected poses")


def _cohort(
    rows: dict[tuple[str, str], dict[str, str]], mode: str
) -> list[tuple[str, str]]:
    if mode == "smoke":
        missing = [key for key in SMOKE_KEYS if key not in rows]
        if missing:
            raise ValueError(f"missing smoke keys {missing}")
        return list(SMOKE_KEYS)
    return sorted(rows)


def run_shard(args: argparse.Namespace) -> None:
    expected_shards = 2 if args.mode == "smoke" else 32
    if args.num_shards != expected_shards or not 0 <= args.shard_index < args.num_shards:
        raise ValueError(f"{args.mode} requires exactly {expected_shards} shards")
    _, source_rows = _load_source_report(args.source_report, args.expected_source_sha256)
    assigned = _cohort(source_rows, args.mode)[args.shard_index :: args.num_shards]
    if not assigned:
        raise ValueError("empty selected-pose shard")
    posebusters_version = version("posebusters")
    if posebusters_version != "0.6.5":
        raise RuntimeError(f"expected PoseBusters 0.6.5, got {posebusters_version}")
    workers = max(1, args.workers)
    buster = PoseBusters(config="redock", max_workers=workers, chunk_size=1)
    final_dir = args.output_root / f"shard-{args.shard_index:03d}-of-{args.num_shards:03d}"
    if final_dir.exists():
        raise FileExistsError(f"refusing to overwrite {final_dir}")
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    incomplete = args.output_root / ".incomplete"
    incomplete.mkdir(parents=True, exist_ok=True)
    attempt = Path(tempfile.mkdtemp(prefix=f"{final_dir.name}.attempt-", dir=incomplete))
    started = time.monotonic()
    result_rows: list[dict[str, Any]] = []
    rmsd_check_name: str | None = None
    score_roots = {"s10_n100": args.n100_scores_root, "s25_n40": args.n40_scores_root}
    for progress, (dataset, complex_id) in enumerate(assigned, start=1):
        conditions, ligand_reference, protein = _selected_conditions(
            dataset=dataset,
            complex_id=complex_id,
            source=source_rows[(dataset, complex_id)],
            score_roots=score_roots,
        )
        with tempfile.TemporaryDirectory(prefix="effdock-fixed-nfe-pb-") as temporary:
            selected_sdf = Path(temporary) / "selected.sdf"
            _write_conditions(conditions, selected_sdf)
            frame = buster.bust(selected_sdf, ligand_reference, protein, full_report=False)
        if len(frame.index) != len(conditions):
            raise ValueError(f"{dataset}/{complex_id}: expected four PoseBusters rows")
        for condition, (_, raw_row) in zip(conditions, frame.iterrows(), strict=True):
            current_rmsd_name, checks = _validated_checks(
                raw_row.to_dict(),
                f"{dataset}/{complex_id}/{condition['arm']}/{condition['stage']}",
            )
            if rmsd_check_name is None:
                rmsd_check_name = current_rmsd_name
            elif rmsd_check_name != current_rmsd_name:
                raise ValueError("PoseBusters RMSD column changed within shard")
            rmsd_lt2 = float(condition["selected_rmsd"]) < 2.0
            pl_valid = all(checks[name] for name in PL_VALIDITY_CHECKS)
            posebusters_valid = all(checks[name] for name in VALIDITY_CHECKS)
            result_rows.append(
                {
                    "dataset": dataset,
                    "id": complex_id,
                    "arm": condition["arm"],
                    "stage": condition["stage"],
                    "selected_index": condition["selected_index"],
                    "selected_rmsd_angstrom": condition["selected_rmsd"],
                    "rmsd_lt2": rmsd_lt2,
                    "pl_valid": pl_valid,
                    "posebusters_valid": posebusters_valid,
                    "joint_pl_valid_rmsd_lt2": pl_valid and rmsd_lt2,
                    "joint_posebusters_valid_rmsd_lt2": posebusters_valid and rmsd_lt2,
                    "separate_posebusters_rmsd_check": checks[current_rmsd_name],
                    "pose_sdf": condition["pose_sdf"],
                    "pose_sdf_sha256": condition["pose_sdf_sha256"],
                    "selected_coordinate_sha256": condition["selected_coordinate_sha256"],
                    **{name: checks[name] for name in VALIDITY_CHECKS},
                }
            )
        print(f"[{progress}/{len(assigned)}] {dataset}/{complex_id}", flush=True)
    assert rmsd_check_name is not None
    fields = list(result_rows[0])
    with gzip.open(attempt / "selected.csv.gz", "wt", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(result_rows)
    summary = {
        "schema_version": "effdock.fixed_nfe_step_pose_pb_shard.v1",
        "protocol_id": PROTOCOL_ID,
        "status": "complete_descriptive",
        "mode": args.mode,
        "posebusters_version": posebusters_version,
        "posebusters_config": "redock",
        "validity_checks": list(VALIDITY_CHECKS),
        "pl_validity_checks": list(PL_VALIDITY_CHECKS),
        "rmsd_check": rmsd_check_name,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "assigned_complexes": len(assigned),
        "result_poses": len(result_rows),
        "source_report": {
            "path": str(args.source_report.resolve()),
            "sha256": args.expected_source_sha256,
        },
        "runtime": {
            "elapsed_seconds": time.monotonic() - started,
            "workers": workers,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "finished_at_utc": datetime.now(UTC).isoformat(),
        },
    }
    (attempt / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.rename(attempt, final_dir)


def _typed_rows(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["selected_index"] = int(row["selected_index"])
        row["selected_rmsd_angstrom"] = float(row["selected_rmsd_angstrom"])
        for field in BOOLEAN_FIELDS:
            row[field] = _truth(row[field])
    return rows


def _metric(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot aggregate an empty cell")
    rmsds = [float(row["selected_rmsd_angstrom"]) for row in rows]
    result: dict[str, Any] = {
        "complexes": len(rows),
        "median_selected_rmsd_angstrom": statistics.median(rmsds),
        "mean_selected_rmsd_angstrom": statistics.fmean(rmsds),
    }
    for field in (
        "rmsd_lt2",
        "pl_valid",
        "joint_pl_valid_rmsd_lt2",
        "posebusters_valid",
        "joint_posebusters_valid_rmsd_lt2",
    ):
        count = sum(bool(row[field]) for row in rows)
        result[f"{field}_count"] = count
        result[f"{field}_pct"] = 100.0 * count / len(rows)
    result["check_pass_pct"] = {
        name: 100.0 * sum(bool(row[name]) for row in rows) / len(rows)
        for name in VALIDITY_CHECKS
    }
    return result


def _transition(
    rows_by_key: dict[tuple[str, str, str, str], dict[str, Any]],
    keys: list[tuple[str, str]],
    left: tuple[str, str],
    right: tuple[str, str],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for field in (
        "rmsd_lt2",
        "pl_valid",
        "joint_pl_valid_rmsd_lt2",
        "posebusters_valid",
        "joint_posebusters_valid_rmsd_lt2",
    ):
        before = [bool(rows_by_key[(*key, *left)][field]) for key in keys]
        after = [bool(rows_by_key[(*key, *right)][field]) for key in keys]
        gain = sum(not a and b for a, b in zip(before, after, strict=True))
        loss = sum(a and not b for a, b in zip(before, after, strict=True))
        result[field] = {
            "before_count": sum(before),
            "after_count": sum(after),
            "delta_count": sum(after) - sum(before),
            "delta_percentage_points": 100.0 * (sum(after) - sum(before)) / len(keys),
            "false_to_true": gain,
            "true_to_false": loss,
        }
    return result


def report(args: argparse.Namespace) -> None:
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_root}")
    _, source_rows = _load_source_report(args.source_report, args.expected_source_sha256)
    result_rows: list[dict[str, Any]] = []
    for shard_index in range(args.num_shards):
        directory = args.shards_root / f"shard-{shard_index:03d}-of-{args.num_shards:03d}"
        summary_path = directory / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if (
            summary.get("protocol_id") != PROTOCOL_ID
            or summary.get("status") != "complete_descriptive"
            or summary.get("mode") != "full"
            or summary.get("posebusters_version") != "0.6.5"
            or summary.get("validity_checks") != list(VALIDITY_CHECKS)
            or summary.get("pl_validity_checks") != list(PL_VALIDITY_CHECKS)
            or int(summary.get("num_shards", -1)) != args.num_shards
            or int(summary.get("shard_index", -1)) != shard_index
            or summary.get("source_report", {}).get("sha256")
            != args.expected_source_sha256
        ):
            raise ValueError(f"invalid shard summary {summary_path}")
        rows = _typed_rows(directory / "selected.csv.gz")
        if len(rows) != int(summary.get("result_poses", -1)):
            raise ValueError(f"row-count mismatch in shard {shard_index}")
        result_rows.extend(rows)
    expected_result_count = sum(EXPECTED_COUNTS.values()) * len(ARMS) * len(STAGES)
    if len(result_rows) != expected_result_count:
        raise ValueError(f"expected {expected_result_count} selected poses, got {len(result_rows)}")
    rows_by_key: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in result_rows:
        key = (row["dataset"], row["id"], row["arm"], row["stage"])
        if key in rows_by_key:
            raise ValueError(f"duplicate result row {key}")
        source = source_rows[(row["dataset"], row["id"])]
        if (
            row["selected_index"] != int(source[f"{row['arm']}_{row['stage']}_selected_index"])
            or not math.isclose(
                row["selected_rmsd_angstrom"],
                float(source[f"{row['arm']}_{row['stage']}_selected_rmsd"]),
                abs_tol=1e-12,
            )
        ):
            raise ValueError(f"selected-pose ledger mismatch for {key}")
        rows_by_key[key] = row
    expected_keys = {
        (dataset, complex_id, arm, stage)
        for dataset, complex_id in source_rows
        for arm in ARMS
        for stage in STAGES
    }
    if set(rows_by_key) != expected_keys:
        raise ValueError("selected-pose condition inventory mismatch")
    aggregate: list[dict[str, Any]] = []
    transitions: list[dict[str, Any]] = []
    for dataset, expected_count in EXPECTED_COUNTS.items():
        dataset_ids = sorted(key for key in source_rows if key[0] == dataset)
        if len(dataset_ids) != expected_count:
            raise ValueError(f"{dataset}: unexpected ID inventory")
        for arm in ARMS:
            for stage in STAGES:
                cell = [rows_by_key[(*key, arm, stage)] for key in dataset_ids]
                aggregate.append(
                    {"dataset": dataset, "arm": arm, "stage": stage, **_metric(cell)}
                )
            transitions.append(
                {
                    "dataset": dataset,
                    "comparison": "raw_to_refined",
                    "arm": arm,
                    "metrics": _transition(
                        rows_by_key, dataset_ids, (arm, "raw"), (arm, "refined")
                    ),
                }
            )
        for stage in STAGES:
            transitions.append(
                {
                    "dataset": dataset,
                    "comparison": "s10_n100_to_s25_n40",
                    "stage": stage,
                    "metrics": _transition(
                        rows_by_key,
                        dataset_ids,
                        ("s10_n100", stage),
                        ("s25_n40", stage),
                    ),
                }
            )
    destination = args.output_root.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    attempt = Path(tempfile.mkdtemp(prefix=f".{destination.name}.attempt-", dir=destination.parent))
    row_fields = list(result_rows[0])
    with (attempt / "selected_pose_validity.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=row_fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(sorted(result_rows, key=lambda row: (row["dataset"], row["id"], row["arm"], row["stage"])))
    payload = {
        "schema_version": "effdock.fixed_nfe_step_pose_pb_report.v1",
        "protocol_id": PROTOCOL_ID,
        "status": "complete_descriptive",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "claim_boundary": "Repeated-use paired Astex/PoseBusters descriptive validity characterization; no model or setting selection.",
        "posebusters_version": "0.6.5",
        "posebusters_config": "redock",
        "primary_validity": "21-check protein-ligand-only PL-valid",
        "secondary_validity": "all 27 non-RMSD PoseBusters redock checks",
        "u50_checkpoint_sha256": U50_SHA256,
        "source_report_sha256": args.expected_source_sha256,
        "aggregate": aggregate,
        "transitions": transitions,
        "artifacts": {
            "selected_pose_validity.csv": {
                "path": str(destination / "selected_pose_validity.csv"),
                "sha256": file_sha256(attempt / "selected_pose_validity.csv"),
            }
        },
    }
    (attempt / "report.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# Fixed-NFE U50 Top-1 PoseBusters results",
        "",
        "> Repeated-use paired descriptive result. U50 indices were frozen before validity evaluation.",
        "",
        "| Dataset | Arm | Stage | RMSD <2A | PL-valid (21) | Joint PL | PB-valid (27) | Joint PB | Median RMSD |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    labels = {"s10_n100": "10 steps x 100 poses", "s25_n40": "25 steps x 40 poses"}
    for row in aggregate:
        lines.append(
            f"| {row['dataset']} | {labels[row['arm']]} | {row['stage']} | "
            f"{row['rmsd_lt2_pct']:.2f}% | {row['pl_valid_pct']:.2f}% | "
            f"{row['joint_pl_valid_rmsd_lt2_pct']:.2f}% | "
            f"{row['posebusters_valid_pct']:.2f}% | "
            f"{row['joint_posebusters_valid_rmsd_lt2_pct']:.2f}% | "
            f"{row['median_selected_rmsd_angstrom']:.3f} A |"
        )
    lines.extend(
        [
            "",
            "PL-valid excludes the six cofactor/water distance and overlap checks. PB-valid requires all 27 non-RMSD redock checks.",
            "",
        ]
    )
    (attempt / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    os.rename(attempt, destination)
    print(json.dumps({"status": "complete", "output_root": str(destination)}, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    shard = subparsers.add_parser("shard")
    shard.add_argument("--source-report", type=Path, required=True)
    shard.add_argument("--expected-source-sha256", required=True)
    shard.add_argument("--n100-scores-root", type=Path, required=True)
    shard.add_argument("--n40-scores-root", type=Path, required=True)
    shard.add_argument("--output-root", type=Path, required=True)
    shard.add_argument("--mode", choices=("smoke", "full"), required=True)
    shard.add_argument("--num-shards", type=int, required=True)
    shard.add_argument("--shard-index", type=int, required=True)
    shard.add_argument("--workers", type=int, default=1)
    aggregate = subparsers.add_parser("report")
    aggregate.add_argument("--source-report", type=Path, required=True)
    aggregate.add_argument("--expected-source-sha256", required=True)
    aggregate.add_argument("--shards-root", type=Path, required=True)
    aggregate.add_argument("--num-shards", type=int, default=32)
    aggregate.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "shard":
        run_shard(args)
    else:
        report(args)


if __name__ == "__main__":
    main()
