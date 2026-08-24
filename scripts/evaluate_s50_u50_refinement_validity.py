#!/usr/bin/env python3
"""Measure official PoseBusters validity around U50 refinement selection."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import statistics
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

PROTOCOL_ID = "EFFDOCK-S50-U50-REFINEMENT-PB-VALIDITY-V1"
SOURCE_PROTOCOL_ID = "EFFDOCK-S50-SYMMETRY-CONFIDENCE-REFINED-EXTERNAL-V1"
REFINEMENT_PROTOCOL_ID = "EFFDOCK-GUIDANCE-SDF-POST-REFINEMENT-V1"
U50_SHA256 = "fd49fa86f67187bf26d6c1bcf2daf925ba3e3b19dfeae733e57535d183280469"
EXPECTED_COUNTS = {"astex": 85, "posebusters": 308}
EXPECTED_POSES = 100


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
    return rmsd[0], {
        key: False if pd.isna(value) else bool(value) for key, value in raw.items()
    }


def _source_rows(path: Path, dataset: str) -> dict[str, dict[str, Any]]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if (
        report.get("protocol_id") != SOURCE_PROTOCOL_ID
        or report.get("status") != "complete_descriptive"
        or report.get("arms", {}).get("u050000") != U50_SHA256
    ):
        raise ValueError("invalid frozen source report")
    rows = {
        str(row["id"]): row
        for row in report.get("complex_rows", [])
        if row.get("dataset") == dataset
    }
    expected = EXPECTED_COUNTS[dataset]
    if len(rows) != expected:
        raise ValueError(f"expected {expected} {dataset} rows, got {len(rows)}")
    return rows


def _selected_molecule(path: Path, pose_index: int) -> Chem.Mol:
    molecules: list[Chem.Mol] = []
    with path.open("rb") as handle:
        for molecule in Chem.ForwardSDMolSupplier(handle, removeHs=False, sanitize=False):
            if molecule is None:
                raise ValueError(f"failed to parse a pose in {path}")
            molecules.append(molecule)
    if len(molecules) != EXPECTED_POSES or not 0 <= pose_index < EXPECTED_POSES:
        raise ValueError(f"invalid pose inventory/index for {path}")
    return molecules[pose_index]


def _write_selected_sdf(molecule: Chem.Mol, path: Path) -> None:
    writer = Chem.SDWriter(str(path))
    try:
        writer.write(molecule)
    finally:
        writer.close()
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"failed to write selected pose {path}")


def run_shard(args: argparse.Namespace) -> None:
    if args.num_shards != 32 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("the frozen contract is exactly 32 shards")
    source_rows = _source_rows(args.source_report, args.dataset)
    selected_ids = sorted(source_rows)[args.shard_index :: args.num_shards]
    if not selected_ids:
        raise ValueError("empty shard")
    final_dir = args.output_root / f"shard-{args.shard_index:03d}-of-032"
    if final_dir.exists():
        raise FileExistsError(f"refusing to overwrite {final_dir}")
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    incomplete = args.output_root / ".incomplete"
    incomplete.mkdir(parents=True, exist_ok=True)
    attempt = Path(tempfile.mkdtemp(prefix=f"{final_dir.name}.attempt-", dir=incomplete))

    version = require_posebusters_runtime_version()
    if version != "0.6.5":
        raise RuntimeError(f"expected PoseBusters 0.6.5, got {version}")
    buster = PoseBusters(config="redock", max_workers=1, chunk_size=1)
    started = time.monotonic()
    result_rows: list[dict[str, Any]] = []
    rmsd_name: str | None = None
    for progress, complex_id in enumerate(selected_ids, start=1):
        source = source_rows[complex_id]
        selected_index = int(source["u050000_step_000_selected_index"])
        score_summary_path = args.scores_root / "u050000" / args.dataset / complex_id / "summary.json"
        score_summary = json.loads(score_summary_path.read_text(encoding="utf-8"))
        if (
            score_summary.get("status") != "complete_descriptive"
            or score_summary.get("dataset") != args.dataset
            or score_summary.get("complex_id") != complex_id
            or score_summary.get("inputs", {}).get("confidence_checkpoint_sha256") != U50_SHA256
            or int(score_summary.get("selected", {}).get("step_000", {}).get("pose_index", -1))
            != selected_index
        ):
            raise ValueError(f"{complex_id}: invalid U50 score summary")
        refinement_path = args.refinement_root / "refinement" / args.dataset / complex_id / "summary.json"
        refinement = json.loads(refinement_path.read_text(encoding="utf-8"))
        if (
            refinement.get("protocol_id") != REFINEMENT_PROTOCOL_ID
            or refinement.get("status") != "complete_descriptive"
            or int(refinement.get("counts", {}).get("poses", -1)) != EXPECTED_POSES
            or int(refinement.get("counts", {}).get("failed", -1)) != 0
            or score_summary.get("inputs", {}).get("refinement_summary_sha256")
            != file_sha256(refinement_path)
        ):
            raise ValueError(f"{complex_id}: invalid refinement summary")
        inputs = refinement["inputs"]
        step_spec = refinement["artifacts"]["step_000_sdf"]
        step_path = Path(step_spec["path"])
        ligand_ref = Path(inputs["ligand_reference"])
        protein = Path(inputs["protein"])
        for path, expected in (
            (step_path, step_spec["sha256"]),
            (ligand_ref, inputs["ligand_reference_sha256"]),
            (protein, inputs["protein_sha256"]),
        ):
            if not path.is_file() or file_sha256(path) != expected:
                raise ValueError(f"{complex_id}: missing or changed input {path}")
        with tempfile.TemporaryDirectory(prefix="effdock-u50-pb-") as temporary:
            selected_path = Path(temporary) / "selected.sdf"
            _write_selected_sdf(_selected_molecule(step_path, selected_index), selected_path)
            frame = buster.bust(selected_path, ligand_ref, protein, full_report=False)
        if len(frame.index) != 1:
            raise ValueError(f"{complex_id}: expected one PoseBusters row")
        current_rmsd, checks = _validated_checks(frame.iloc[0].to_dict(), complex_id)
        if rmsd_name is None:
            rmsd_name = current_rmsd
        elif rmsd_name != current_rmsd:
            raise ValueError("PoseBusters RMSD column changed within shard")
        result_rows.append(
            {
                "id": complex_id,
                "pose_index": selected_index,
                "posebusters_valid": all(checks[name] for name in VALIDITY_CHECKS),
                "separate_rmsd_check": checks[current_rmsd],
                **{name: checks[name] for name in VALIDITY_CHECKS},
            }
        )
        print(f"[{progress}/{len(selected_ids)}] {args.dataset}/{complex_id}", flush=True)

    assert rmsd_name is not None
    with gzip.open(attempt / "selected_step000.csv.gz", "wt", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result_rows[0]), extrasaction="raise")
        writer.writeheader()
        writer.writerows(result_rows)
    summary = {
        "schema_version": "effdock.s50_u50_refinement_pb_validity_shard.v1",
        "protocol_id": PROTOCOL_ID,
        "status": "complete_descriptive",
        "posebusters_version": version,
        "posebusters_config": "redock",
        "dataset": args.dataset,
        "validity_checks": list(VALIDITY_CHECKS),
        "rmsd_check": rmsd_name,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "assigned": len(result_rows),
        "valid": sum(bool(row["posebusters_valid"]) for row in result_rows),
        "source_report": {"path": str(args.source_report.resolve()), "sha256": file_sha256(args.source_report)},
        "runtime": {
            "elapsed_seconds": time.monotonic() - started,
            "finished_at_utc": datetime.now(UTC).isoformat(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        },
    }
    (attempt / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.rename(attempt, final_dir)


def _load_step100(root: Path, dataset: str) -> dict[tuple[str, int], dict[str, bool]]:
    rows: dict[tuple[str, int], dict[str, bool]] = {}
    for shard in range(32):
        directory = root / f"shard-{shard:03d}-of-032"
        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        if summary.get("status") != "complete":
            raise ValueError(f"incomplete step100 shard {shard}")
        with gzip.open(directory / "poses.csv.gz", "rt", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row["dataset"] != dataset:
                    continue
                key = (row["id"], int(row["pose_index"]))
                if key in rows:
                    raise ValueError(f"duplicate step100 row {key}")
                rows[key] = {name: _truth(row[name]) for name in VALIDITY_CHECKS}
    if len(rows) != EXPECTED_COUNTS[dataset] * EXPECTED_POSES:
        raise ValueError(f"unexpected step100 inventory: {len(rows)}")
    return rows


def _metric(rows: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    valid = [bool(row[f"{prefix}_valid"]) for row in rows]
    lt2 = [bool(row[f"{prefix}_lt2"]) for row in rows]
    rmsd = [float(row[f"{prefix}_rmsd"]) for row in rows]
    joint = [a and b for a, b in zip(valid, lt2, strict=True)]
    return {
        "complexes": len(rows),
        "official_valid_count": sum(valid),
        "official_valid_pct": 100.0 * sum(valid) / len(rows),
        "rmsd_lt2_count": sum(lt2),
        "rmsd_lt2_pct": 100.0 * sum(lt2) / len(rows),
        "joint_count": sum(joint),
        "joint_pct": 100.0 * sum(joint) / len(rows),
        "mean_rmsd": statistics.fmean(rmsd),
        "median_rmsd": statistics.median(rmsd),
    }


def _transition(rows: list[dict[str, Any]], left: str, right: str) -> dict[str, Any]:
    before = [bool(row[f"{left}_valid"]) for row in rows]
    after = [bool(row[f"{right}_valid"]) for row in rows]
    return {
        "delta_percentage_points": 100.0 * (sum(after) - sum(before)) / len(rows),
        "invalid_to_valid": sum(not a and b for a, b in zip(before, after, strict=True)),
        "valid_to_invalid": sum(a and not b for a, b in zip(before, after, strict=True)),
    }


def report(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    source_rows = _source_rows(args.source_report, args.dataset)
    raw_rows: dict[str, dict[str, Any]] = {}
    for shard in range(32):
        directory = args.shards_root / f"shard-{shard:03d}-of-032"
        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        if (
            summary.get("protocol_id") != PROTOCOL_ID
            or summary.get("status") != "complete_descriptive"
            or int(summary.get("num_shards", -1)) != 32
            or int(summary.get("shard_index", -1)) != shard
            or summary.get("dataset") != args.dataset
            or summary.get("source_report", {}).get("sha256") != file_sha256(args.source_report)
        ):
            raise ValueError(f"invalid raw-validity shard {shard}")
        with gzip.open(directory / "selected_step000.csv.gz", "rt", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row["id"] in raw_rows:
                    raise ValueError(f"duplicate raw-validity row {row['id']}")
                raw_rows[row["id"]] = row
    if set(raw_rows) != set(source_rows):
        raise ValueError("raw-validity ID inventory mismatch")
    step100 = _load_step100(args.step100_official_root, args.dataset)
    rows: list[dict[str, Any]] = []
    for complex_id in sorted(source_rows):
        source = source_rows[complex_id]
        raw = raw_rows[complex_id]
        index0 = int(source["u050000_step_000_selected_index"])
        index100 = int(source["u050000_step_100_selected_index"])
        if int(raw["pose_index"]) != index0:
            raise ValueError(f"{complex_id}: raw selected-index mismatch")
        score_summary = json.loads(
            (args.scores_root / "u050000" / args.dataset / complex_id / "summary.json").read_text(encoding="utf-8")
        )
        score_csv = Path(score_summary["artifacts"]["scores_csv"]["path"])
        if file_sha256(score_csv) != score_summary["artifacts"]["scores_csv"]["sha256"]:
            raise ValueError(f"{complex_id}: score CSV hash mismatch")
        with score_csv.open(newline="", encoding="utf-8") as handle:
            scores = list(csv.DictReader(handle))
        if len(scores) != EXPECTED_POSES:
            raise ValueError(f"{complex_id}: score inventory mismatch")
        raw_valid = _truth(raw["posebusters_valid"])
        same_checks = step100[(complex_id, index0)]
        selected_checks = step100[(complex_id, index100)]
        rows.append(
            {
                "id": complex_id,
                "step000_selected_index": index0,
                "step100_selected_index": index100,
                "step000_valid": raw_valid,
                "step000_lt2": bool(source["u050000_step_000_selected_lt2"]),
                "step000_rmsd": float(source["u050000_step_000_selected_rmsd"]),
                "step100_same_index_valid": all(same_checks.values()),
                "step100_same_index_lt2": float(scores[index0]["final_symmetry_rmsd_angstrom"]) < 2.0,
                "step100_same_index_rmsd": float(scores[index0]["final_symmetry_rmsd_angstrom"]),
                "step100_reselected_valid": all(selected_checks.values()),
                "step100_reselected_lt2": bool(source["u050000_step_100_selected_lt2"]),
                "step100_reselected_rmsd": float(source["u050000_step_100_selected_rmsd"]),
            }
        )
    result = {
        "schema_version": "effdock.s50_u50_refinement_pb_validity_report.v1",
        "protocol_id": PROTOCOL_ID,
        "status": "complete_descriptive",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "claim_boundary": "Repeated-use PoseBusters descriptive refinement decomposition; not a checkpoint-selection result.",
        "u50_checkpoint_sha256": U50_SHA256,
        "dataset": args.dataset,
        "stages": {
            "step000_selected": _metric(rows, "step000"),
            "step100_same_index": _metric(rows, "step100_same_index"),
            "step100_reselected": _metric(rows, "step100_reselected"),
        },
        "validity_transitions": {
            "refinement_same_index": _transition(rows, "step000", "step100_same_index"),
            "post_refinement_reselection": _transition(rows, "step100_same_index", "step100_reselected"),
            "end_to_end": _transition(rows, "step000", "step100_reselected"),
        },
        "complex_rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.link(temporary, args.output)
    temporary.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    shard = subparsers.add_parser("shard")
    shard.add_argument("--source-report", type=Path, required=True)
    shard.add_argument("--dataset", choices=tuple(EXPECTED_COUNTS), required=True)
    shard.add_argument("--scores-root", type=Path, required=True)
    shard.add_argument("--refinement-root", type=Path, required=True)
    shard.add_argument("--output-root", type=Path, required=True)
    shard.add_argument("--num-shards", type=int, default=32)
    shard.add_argument("--shard-index", type=int, required=True)
    aggregate = subparsers.add_parser("report")
    aggregate.add_argument("--source-report", type=Path, required=True)
    aggregate.add_argument("--dataset", choices=tuple(EXPECTED_COUNTS), required=True)
    aggregate.add_argument("--scores-root", type=Path, required=True)
    aggregate.add_argument("--step100-official-root", type=Path, required=True)
    aggregate.add_argument("--shards-root", type=Path, required=True)
    aggregate.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "shard":
        run_shard(args)
    else:
        report(args)


if __name__ == "__main__":
    main()
