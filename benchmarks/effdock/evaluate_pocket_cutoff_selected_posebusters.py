#!/usr/bin/env python3
"""Evaluate official PB-validity for refined U70k Top-1 poses in one shard."""

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

from effdock.workflows.evaluate import file_sha256
from effdock.workflows.guidance_budget_posebusters_report import VALIDITY_CHECKS
from effdock.workflows.posebusters_report import require_posebusters_runtime_version

DOCKING_SHA256 = "65be44d7dc8f0867eb9fc5d22214b80f93971ea4702679a527c665046e91e6b6"
CONFIDENCE_SHA256 = "ce59be42f0ca613871ca079127c3296f5ca9a4ec72e44a9e5cf61878351c2638"
EXPECTED = {"astex": 85, "posebusters": 308}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition-root", type=Path, required=True)
    parser.add_argument("--cutoff", type=int, choices=(6, 8, 10, 12), required=True)
    parser.add_argument("--repeat-index", type=int, choices=range(3), required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def validated_checks(raw: dict[str, Any], label: str) -> tuple[str, dict[str, bool]]:
    rmsd = [key for key in raw if str(key).startswith("rmsd_")]
    if len(rmsd) != 1 or set(raw) != {*VALIDITY_CHECKS, rmsd[0]}:
        raise ValueError(f"{label}: unexpected PoseBusters redock schema")
    return rmsd[0], {
        key: False if pd.isna(raw[key]) else bool(raw[key])
        for key in VALIDITY_CHECKS
    }


def extract_pose(source: Path, index: int, output: Path) -> None:
    with source.open("rb") as handle:
        molecules = list(Chem.ForwardSDMolSupplier(handle, removeHs=False, sanitize=False))
    if len(molecules) != 100 or any(mol is None for mol in molecules):
        raise ValueError(f"expected 100 readable poses in {source}")
    molecule = molecules[index]
    if molecule is None:
        raise AssertionError("unreachable unreadable selected pose")
    molecule.SetIntProp("effdock_selected_pose_index", index)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = Chem.SDWriter(str(output))
    writer.SetForceV3000(True)
    writer.write(molecule)
    writer.close()
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError(f"failed to write {output}")


def main() -> None:
    args = parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid shard specification")
    manifest_path = args.condition_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    condition = manifest.get("condition", {})
    if (
        manifest.get("protocol_id") != "EFFDOCK-POCKET-CUTOFF-ROBUSTNESS-MANIFEST-V1"
        or int(condition.get("pocket_cutoff_angstrom", -1)) != args.cutoff
        or int(condition.get("repeat_index", -1)) != args.repeat_index
    ):
        raise ValueError("manifest condition mismatch")
    records = sorted(manifest["records"], key=lambda row: (row["dataset"], row["id"]))
    expected = {"astex": 1} if args.smoke else EXPECTED
    counts = {dataset: sum(row["dataset"] == dataset for row in records) for dataset in expected}
    if counts != expected or len(records) != sum(expected.values()):
        raise ValueError(f"manifest coverage mismatch: {counts}")
    assigned = records[args.shard_index :: args.num_shards]
    if not assigned:
        raise ValueError("empty assigned shard")

    output_root = args.condition_root / "full" / "selected_posebusters"
    shard_name = f"shard-{args.shard_index:03d}-of-{args.num_shards:03d}"
    final_dir = output_root / shard_name
    if final_dir.exists():
        raise FileExistsError(final_dir)
    incomplete = output_root / ".incomplete"
    incomplete.mkdir(parents=True, exist_ok=True)
    attempt = Path(tempfile.mkdtemp(prefix=f"{shard_name}.", dir=incomplete))
    selected_dir = attempt / "selected"

    version = require_posebusters_runtime_version()
    buster = PoseBusters(config="redock", max_workers=0)
    started = time.monotonic()
    results: list[dict[str, Any]] = []
    rmsd_check: str | None = None
    for position, record in enumerate(assigned, start=1):
        dataset, complex_id = str(record["dataset"]), str(record["id"])
        refinement_path = (
            args.condition_root / "full" / "refinement" / dataset / complex_id / "summary.json"
        )
        confidence_path = (
            args.condition_root
            / "full"
            / "confidence_chunk20_fresh"
            / dataset
            / complex_id
            / "summary.json"
        )
        result: dict[str, Any] = {
            "dataset": dataset,
            "id": complex_id,
            "cutoff_angstrom": args.cutoff,
            "repeat_index": args.repeat_index,
            "selected_pose_index": -1,
            "selected_rmsd": math.inf,
            "selected_rmsd_lt2": False,
            "posebusters_valid": False,
            "joint_rmsd_lt2_pb_valid": False,
            "oracle_rmsd": math.inf,
            "oracle_rmsd_lt2": False,
            "error": "",
        }
        try:
            refinement = json.loads(refinement_path.read_text(encoding="utf-8"))
            confidence = json.loads(confidence_path.read_text(encoding="utf-8"))
            if refinement.get("status") != "complete_descriptive":
                raise ValueError("refinement is incomplete")
            if float(refinement["inputs"].get("pocket_cutoff_angstrom", -1)) != 10.0:
                raise ValueError("refinement crop is not fixed at 10 A")
            if confidence.get("status") != "complete_descriptive":
                raise ValueError("confidence is incomplete")
            if float(confidence.get("pocket_cutoff_angstrom", -1)) != 10.0:
                raise ValueError("confidence crop is not fixed at 10 A")
            conf_inputs = confidence["inputs"]
            if conf_inputs["docking_checkpoint_sha256"] != DOCKING_SHA256:
                raise ValueError("docking checkpoint mismatch")
            if conf_inputs["confidence_checkpoint_sha256"] != CONFIDENCE_SHA256:
                raise ValueError("confidence checkpoint mismatch")
            selected = confidence["selected"]["step_100"]
            selected_index = int(selected["pose_index"])
            selected_rmsd = float(selected["symmetry_rmsd_angstrom"])
            pose_spec = refinement["artifacts"]["step_100_sdf"]
            pose_path = Path(pose_spec["path"])
            if file_sha256(pose_path) != pose_spec["sha256"]:
                raise ValueError("refined pose artifact changed")
            selected_path = selected_dir / dataset / f"{complex_id}.sdf"
            extract_pose(pose_path, selected_index, selected_path)
            frame = buster.bust(
                selected_path,
                Path(record["ligand_ref"]),
                Path(record["protein"]),
                full_report=False,
            )
            if len(frame.index) != 1:
                raise ValueError("expected one PoseBusters result")
            current_rmsd, checks = validated_checks(frame.iloc[0].to_dict(), complex_id)
            if rmsd_check is None:
                rmsd_check = current_rmsd
            elif rmsd_check != current_rmsd:
                raise ValueError("PoseBusters RMSD column changed within shard")
            valid = all(checks.values())
            oracle_rmsd = min(float(row["final_symmetry_rmsd_angstrom"]) for row in refinement["poses"])
            result.update(
                {
                    "selected_pose_index": selected_index,
                    "selected_rmsd": selected_rmsd,
                    "selected_rmsd_lt2": selected_rmsd < 2.0,
                    "posebusters_valid": valid,
                    "joint_rmsd_lt2_pb_valid": valid and selected_rmsd < 2.0,
                    "oracle_rmsd": oracle_rmsd,
                    "oracle_rmsd_lt2": oracle_rmsd < 2.0,
                    **checks,
                }
            )
        except Exception as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
            result.update({key: False for key in VALIDITY_CHECKS})
        results.append(result)
        print(
            f"[{position}/{len(assigned)}] c{args.cutoff}/r{args.repeat_index}/"
            f"{dataset}/{complex_id} RMSD={result['selected_rmsd']:.3f} "
            f"PB={result['posebusters_valid']} error={bool(result['error'])}",
            flush=True,
        )

    with (attempt / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    summary = {
        "schema_version": 1,
        "status": "complete",
        "protocol_id": "EFFDOCK-POCKET-CUTOFF-ROBUSTNESS-V1",
        "cutoff_angstrom": args.cutoff,
        "repeat_index": args.repeat_index,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "num_results": len(results),
        "num_errors": sum(bool(row["error"]) for row in results),
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
