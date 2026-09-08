#!/usr/bin/env python3
"""Reselect refined poses with input-SMILES stereo constraints, then run PB checks."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from posebusters import PoseBusters
from rdkit import Chem

from effdock.workflows.benchmark_inputs import load_benchmark_inputs
from effdock.workflows.evaluate import file_sha256
from effdock.workflows.guidance_budget_posebusters_report import VALIDITY_CHECKS
from effdock.workflows.posebusters_report import require_posebusters_runtime_version
from effdock.workflows.stereo_filter import (
    explicit_stereo_constraints,
    select_stereo_filtered_index,
    stereo_compatibility,
)

DOCKING_SHA256 = "65be44d7dc8f0867eb9fc5d22214b80f93971ea4702679a527c665046e91e6b6"
CONFIDENCE_SHA256 = "ce59be42f0ca613871ca079127c3296f5ca9a4ec72e44a9e5cf61878351c2638"
PROTOCOL_ID = "EFFDOCK-INPUT-STEREO-FILTER-SELECTION-V4"
EXPECTED = {"astex": 85, "posebusters": 308}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition-root", type=Path, required=True)
    parser.add_argument("--cutoff", type=int, choices=(10,), required=True)
    parser.add_argument("--repeat-index", type=int, choices=range(3), required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument(
        "--benchmark-input-manifest",
        type=Path,
        default=Path("docs/GUIDANCE_BUDGET1000_FULL_INPUTS.json"),
    )
    parser.add_argument("--external-dir", type=Path, default=Path("data/external_test"))
    return parser.parse_args()


def _bool(value: object) -> bool:
    return str(value).strip().lower() == "true"


def _load_original_pb_rows(condition_root: Path) -> dict[tuple[str, str], dict[str, str]]:
    rows: dict[tuple[str, str], dict[str, str]] = {}
    for path in sorted((condition_root / "full" / "selected_posebusters").glob("shard-*/results.csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                key = (row["dataset"], row["id"])
                if key in rows:
                    raise ValueError(f"duplicate original PoseBusters row: {key}")
                if row["error"]:
                    raise ValueError(f"original PoseBusters row has an error: {key}")
                rows[key] = row
    if len(rows) != sum(EXPECTED.values()):
        raise ValueError(f"incomplete original PoseBusters inventory: {len(rows)}")
    return rows


def _load_scores(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    indices = [int(row["pose_index"]) for row in rows]
    if len(rows) != 100 or indices != list(range(100)):
        raise ValueError(f"expected ordered pose indices 0..99 in {path}")
    return rows


def _load_poses(path: Path) -> list[Chem.Mol]:
    with path.open("rb") as handle:
        poses = list(Chem.ForwardSDMolSupplier(handle, removeHs=False, sanitize=False))
    if len(poses) != 100 or any(pose is None for pose in poses):
        raise ValueError(f"expected 100 readable poses in {path}")
    return [pose for pose in poses if pose is not None]


def _write_pose(molecule: Chem.Mol, index: int, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    selected = Chem.Mol(molecule)
    selected.SetIntProp("effdock_selected_pose_index", index)
    selected.SetProp("effdock_selector", "input_stereo_filter_then_predicted_rmsd")
    writer = Chem.SDWriter(str(output))
    writer.SetForceV3000(True)
    writer.write(selected)
    writer.close()
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError(f"failed to write {output}")


def _validated_checks(raw: dict[str, Any], label: str) -> tuple[str, dict[str, bool]]:
    rmsd = [key for key in raw if str(key).startswith("rmsd_")]
    if len(rmsd) != 1 or set(raw) != {*VALIDITY_CHECKS, rmsd[0]}:
        raise ValueError(f"{label}: unexpected PoseBusters redock schema")
    return rmsd[0], {
        key: False if pd.isna(raw[key]) else bool(raw[key])
        for key in VALIDITY_CHECKS
    }


def main() -> None:
    args = parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid shard specification")
    manifest = json.loads((args.condition_root / "manifest.json").read_text(encoding="utf-8"))
    condition = manifest.get("condition", {})
    if (
        manifest.get("protocol_id") != "EFFDOCK-POCKET-CUTOFF-ROBUSTNESS-MANIFEST-V1"
        or int(condition.get("pocket_cutoff_angstrom", -1)) != args.cutoff
        or int(condition.get("repeat_index", -1)) != args.repeat_index
    ):
        raise ValueError("condition manifest mismatch")
    records = sorted(manifest["records"], key=lambda row: (row["dataset"], row["id"]))
    counts = {dataset: sum(row["dataset"] == dataset for row in records) for dataset in EXPECTED}
    if counts != EXPECTED or len(records) != sum(EXPECTED.values()):
        raise ValueError(f"manifest coverage mismatch: {counts}")
    assigned = records[args.shard_index :: args.num_shards]
    if not assigned:
        raise ValueError("empty assigned shard")

    smiles_by_dataset = {
        dataset: load_benchmark_inputs(
            dataset,
            args.external_dir,
            args.benchmark_input_manifest,
        )[0]
        for dataset in EXPECTED
    }
    original_pb = _load_original_pb_rows(args.condition_root)
    version = require_posebusters_runtime_version()
    buster = PoseBusters(config="redock", max_workers=0)

    output_root = args.condition_root / "full" / "stereo_filtered_selection_v4"
    shard_name = f"shard-{args.shard_index:03d}-of-{args.num_shards:03d}"
    final_dir = output_root / shard_name
    if final_dir.exists():
        raise FileExistsError(final_dir)
    incomplete = output_root / ".incomplete"
    incomplete.mkdir(parents=True, exist_ok=True)
    attempt = Path(tempfile.mkdtemp(prefix=f"{shard_name}.", dir=incomplete))
    selected_dir = attempt / "selected"

    started = time.monotonic()
    results: list[dict[str, Any]] = []
    rmsd_check: str | None = None
    for position, record in enumerate(assigned, start=1):
        dataset, complex_id = str(record["dataset"]), str(record["id"])
        confidence_path = (
            args.condition_root / "full" / "confidence_chunk20_fresh"
            / dataset / complex_id / "summary.json"
        )
        refinement_path = (
            args.condition_root / "full" / "refinement" / dataset / complex_id / "summary.json"
        )
        result: dict[str, Any] = {
            "dataset": dataset,
            "id": complex_id,
            "cutoff_angstrom": args.cutoff,
            "repeat_index": args.repeat_index,
            "original_selected_pose_index": -1,
            "selected_pose_index": -1,
            "stereo_valid_candidate_count": 0,
            "stereo_filter_fallback": False,
            "selection_changed": False,
            "selected_input_tetrahedral_valid": False,
            "selected_input_double_bond_valid": False,
            "original_selected_rmsd": math.inf,
            "selected_rmsd": math.inf,
            "selected_rmsd_lt2": False,
            "posebusters_valid": False,
            "joint_rmsd_lt2_pb_valid": False,
            "oracle_rmsd": math.inf,
            "oracle_rmsd_lt2": False,
            "official_pb_reused": False,
            "error": "",
        }
        try:
            confidence = json.loads(confidence_path.read_text(encoding="utf-8"))
            refinement = json.loads(refinement_path.read_text(encoding="utf-8"))
            if confidence.get("status") != "complete_descriptive":
                raise ValueError("confidence summary is incomplete")
            if refinement.get("status") != "complete_descriptive":
                raise ValueError("refinement summary is incomplete")
            inputs = confidence["inputs"]
            if inputs["docking_checkpoint_sha256"] != DOCKING_SHA256:
                raise ValueError("docking checkpoint mismatch")
            if inputs["confidence_checkpoint_sha256"] != CONFIDENCE_SHA256:
                raise ValueError("confidence checkpoint mismatch")
            if float(confidence["pocket_cutoff_angstrom"]) != 10.0:
                raise ValueError("confidence crop is not fixed at 10 A")
            if float(refinement["inputs"]["pocket_cutoff_angstrom"]) != 10.0:
                raise ValueError("refinement crop is not fixed at 10 A")

            score_path = Path(confidence["artifacts"]["scores_csv"]["path"])
            if file_sha256(score_path) != confidence["artifacts"]["scores_csv"]["sha256"]:
                raise ValueError("confidence score artifact changed")
            scores = _load_scores(score_path)
            pose_spec = refinement["artifacts"]["step_100_sdf"]
            pose_path = Path(pose_spec["path"])
            if file_sha256(pose_path) != pose_spec["sha256"]:
                raise ValueError("refined pose artifact changed")
            poses = _load_poses(pose_path)

            input_mol = Chem.MolFromSmiles(smiles_by_dataset[dataset][complex_id])
            if input_mol is None:
                raise ValueError("frozen input SMILES cannot be parsed")
            expected_constraints = explicit_stereo_constraints(input_mol)
            compatibility = [
                stereo_compatibility(expected_constraints, pose)
                for pose in poses
            ]
            stereo_valid = [item.valid for item in compatibility]
            predicted_rmsd = [float(row["after_confidence_rmsd"]) for row in scores]
            original = confidence["selected"]["step_100"]
            original_index = int(original["pose_index"])
            selected_index, fallback = select_stereo_filtered_index(
                predicted_rmsd,
                stereo_valid,
                fallback_index=original_index,
            )
            selected_path = selected_dir / dataset / f"{complex_id}.sdf"
            _write_pose(poses[selected_index], selected_index, selected_path)

            score = scores[selected_index]
            selected_rmsd = float(score["final_symmetry_rmsd_angstrom"])
            oracle_rmsd = min(float(row["final_symmetry_rmsd_angstrom"]) for row in scores)
            original_row = original_pb[(dataset, complex_id)]
            if int(original_row["selected_pose_index"]) != original_index:
                raise ValueError("original confidence and official PB indices disagree")

            if selected_index == original_index:
                checks = {key: _bool(original_row[key]) for key in VALIDITY_CHECKS}
                current_rmsd = None
                official_reused = True
            else:
                frame = buster.bust(
                    selected_path,
                    Path(record["ligand_ref"]),
                    Path(record["protein"]),
                    full_report=False,
                )
                if len(frame.index) != 1:
                    raise ValueError("expected one PoseBusters result")
                current_rmsd, checks = _validated_checks(frame.iloc[0].to_dict(), complex_id)
                official_reused = False
            if current_rmsd is not None:
                if rmsd_check is None:
                    rmsd_check = current_rmsd
                elif rmsd_check != current_rmsd:
                    raise ValueError("PoseBusters RMSD column changed within shard")

            valid = all(checks.values())
            result.update(
                {
                    "original_selected_pose_index": original_index,
                    "selected_pose_index": selected_index,
                    "stereo_valid_candidate_count": sum(stereo_valid),
                    "stereo_filter_fallback": fallback,
                    "selection_changed": selected_index != original_index,
                    "selected_input_tetrahedral_valid": compatibility[selected_index].tetrahedral,
                    "selected_input_double_bond_valid": compatibility[selected_index].double_bond,
                    "original_selected_rmsd": float(original["symmetry_rmsd_angstrom"]),
                    "selected_rmsd": selected_rmsd,
                    "selected_rmsd_lt2": selected_rmsd < 2.0,
                    "posebusters_valid": valid,
                    "joint_rmsd_lt2_pb_valid": valid and selected_rmsd < 2.0,
                    "oracle_rmsd": oracle_rmsd,
                    "oracle_rmsd_lt2": oracle_rmsd < 2.0,
                    "official_pb_reused": official_reused,
                    **checks,
                }
            )
        except Exception as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
            result.update({key: False for key in VALIDITY_CHECKS})
        results.append(result)
        print(
            f"[{position}/{len(assigned)}] r{args.repeat_index}/{dataset}/{complex_id} "
            f"valid_candidates={result['stereo_valid_candidate_count']} "
            f"changed={result['selection_changed']} fallback={result['stereo_filter_fallback']} "
            f"RMSD={result['selected_rmsd']:.3f} PB={result['posebusters_valid']} "
            f"error={bool(result['error'])}",
            flush=True,
        )

    with (attempt / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    summary = {
        "schema_version": 1,
        "status": "complete" if not any(row["error"] for row in results) else "failed",
        "protocol_id": PROTOCOL_ID,
        "selection_information_boundary": "frozen input SMILES and U70k predicted RMSD only",
        "external_outcome_policy": "descriptive_only_not_selector_admission",
        "cutoff_angstrom": args.cutoff,
        "repeat_index": args.repeat_index,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "num_results": len(results),
        "num_errors": sum(bool(row["error"]) for row in results),
        "num_selection_changed": sum(bool(row["selection_changed"]) for row in results),
        "num_stereo_filter_fallback": sum(bool(row["stereo_filter_fallback"]) for row in results),
        "num_official_pb_reused": sum(bool(row["official_pb_reused"]) for row in results),
        "posebusters_version": version,
        "posebusters_config": "redock",
        "validity_checks": list(VALIDITY_CHECKS),
        "rmsd_check": rmsd_check,
        "runtime": {
            "elapsed_seconds": time.monotonic() - started,
            "finished_at_utc": datetime.now(UTC).isoformat(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        },
    }
    (attempt / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    os.rename(attempt, final_dir)


if __name__ == "__main__":
    main()
