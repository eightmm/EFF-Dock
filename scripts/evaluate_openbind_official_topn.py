#!/usr/bin/env python3
"""Evaluate saved EFF-Dock poses with the OpenBind Top-N aggregation contract."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROTOCOL_ID = "EFFDOCK-OPENBIND-OFFICIAL-TOP25-V1"
EXPECTED_SOURCE_COMPLEXES = 925
EXPECTED_OFFICIAL_COMPLEXES = 802
EXPECTED_POSES = 100
DEFAULT_TOP_N = 25
TOP_NS = (1, 5, 25)
POSEBUSTERS_VERSION = "0.6.5"
OPENSTRUCTURE_VERSION = "2.11.1"

# PoseBusters 0.6.5 redock outputs used for pass-all validity. The separate
# PoseBusters RMSD output is deliberately excluded: the OpenBind endpoint uses
# OpenStructure BiSyRMSD below.
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


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ids_sha256(ids: list[str]) -> str:
    payload = "\n".join(sorted(ids)) + "\n"
    return hashlib.sha256(payload.encode()).hexdigest()


def parse_bool(value: Any, *, label: str) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"{label}: invalid boolean {value!r}")


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def load_official_cohort(
    metadata_path: Path,
    *,
    expected_source_count: int | None = EXPECTED_SOURCE_COMPLEXES,
    expected_cohort_count: int | None = EXPECTED_OFFICIAL_COMPLEXES,
) -> list[str]:
    """Reproduce OpenBind's filtered scaffold-only denominator from source metadata."""
    with metadata_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "complex_name",
        "covalent",
        "fragment_screen",
        "pb_valid_prepared",
        "suspected_artefact",
    }
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"{metadata_path}: missing OpenBind annotation columns")
    if expected_source_count is not None and len(rows) != expected_source_count:
        raise ValueError(
            f"{metadata_path}: expected {expected_source_count} rows, found {len(rows)}"
        )
    selected: list[str] = []
    seen: set[str] = set()
    for row_number, row in enumerate(rows, start=2):
        complex_id = str(row["complex_name"]).strip().lower()
        if not complex_id or complex_id in seen:
            raise ValueError(f"{metadata_path}:{row_number}: duplicate/empty complex ID")
        seen.add(complex_id)
        admitted = (
            parse_bool(row["pb_valid_prepared"], label=f"{complex_id}.pb_valid_prepared")
            and not parse_bool(row["suspected_artefact"], label=f"{complex_id}.suspected_artefact")
            and not parse_bool(row["fragment_screen"], label=f"{complex_id}.fragment_screen")
        )
        if admitted:
            if parse_bool(row["covalent"], label=f"{complex_id}.covalent"):
                raise ValueError(f"{complex_id}: official scaffold cohort contains covalent ligand")
            selected.append(complex_id)
    selected.sort()
    if expected_cohort_count is not None and len(selected) != expected_cohort_count:
        raise ValueError(
            f"{metadata_path}: expected {expected_cohort_count} official complexes, "
            f"found {len(selected)}"
        )
    return selected


def load_ranked_scores(scores_path: Path, *, top_n: int = DEFAULT_TOP_N) -> list[dict[str, Any]]:
    """Return zero-based confidence ranks, stable by original pose index."""
    with scores_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "pose_index",
        "after_confidence_rmsd",
        "final_symmetry_rmsd_angstrom",
    }
    if len(rows) != EXPECTED_POSES or not rows or not required.issubset(rows[0]):
        raise ValueError(f"{scores_path}: expected 100 complete confidence rows")
    parsed: list[dict[str, Any]] = []
    for row in rows:
        pose_index = int(row["pose_index"])
        confidence_rmsd = float(row["after_confidence_rmsd"])
        internal_rmsd = float(row["final_symmetry_rmsd_angstrom"])
        if not math.isfinite(confidence_rmsd) or not math.isfinite(internal_rmsd):
            raise ValueError(f"{scores_path}: non-finite score")
        parsed.append(
            {
                "pose_index": pose_index,
                "confidence_predicted_rmsd": confidence_rmsd,
                "internal_symmetry_rmsd": internal_rmsd,
            }
        )
    if {row["pose_index"] for row in parsed} != set(range(EXPECTED_POSES)):
        raise ValueError(f"{scores_path}: pose indices must be exactly 0..99")
    parsed.sort(key=lambda row: (row["confidence_predicted_rmsd"], row["pose_index"]))
    for rank, row in enumerate(parsed):
        row["rank"] = rank
    return parsed[:top_n]


def source_inputs(
    source_run: Path,
    complex_id: str,
    *,
    expected_confidence_sha256: str | None = None,
) -> dict[str, Path] | None:
    confidence_dir = source_run / "full" / "confidence" / "openbind" / complex_id
    refinement_dir = source_run / "full" / "refinement" / "openbind" / complex_id
    confidence_path = confidence_dir / "summary.json"
    refinement_path = refinement_dir / "summary.json"
    if not confidence_path.is_file() or not refinement_path.is_file():
        return None
    confidence = read_json(confidence_path)
    refinement = read_json(refinement_path)
    if confidence.get("status") != "complete_descriptive":
        raise ValueError(f"{complex_id}: incomplete confidence summary")
    if refinement.get("status") != "complete_descriptive":
        raise ValueError(f"{complex_id}: incomplete refinement summary")
    if confidence.get("complex_id") != complex_id:
        raise ValueError(f"{complex_id}: confidence identity mismatch")
    if (
        expected_confidence_sha256 is not None
        and confidence.get("inputs", {}).get("confidence_checkpoint_sha256")
        != expected_confidence_sha256
    ):
        raise ValueError(f"{complex_id}: confidence checkpoint SHA mismatch")

    scores_path = confidence_dir / "scores.csv"
    pose_path = refinement_dir / "step_100.sdf"
    protein_path = Path(str(refinement["inputs"]["protein"]))
    reference_path = Path(str(refinement["inputs"]["ligand_reference"]))
    checks = (
        (scores_path, confidence["artifacts"]["scores_csv"]["sha256"]),
        (pose_path, refinement["artifacts"]["step_100_sdf"]["sha256"]),
        (protein_path, refinement["inputs"]["protein_sha256"]),
        (reference_path, refinement["inputs"]["ligand_reference_sha256"]),
    )
    for path, expected in checks:
        if not path.is_file() or file_sha256(path) != expected:
            raise ValueError(f"{complex_id}: missing or changed frozen input {path}")
    return {
        "scores": scores_path,
        "poses": pose_path,
        "protein": protein_path,
        "reference": reference_path,
        "confidence_summary": confidence_path,
        "refinement_summary": refinement_path,
    }


def assigned_ids(
    cohort: list[str], *, num_shards: int, shard_index: int, max_complexes: int | None
) -> list[str]:
    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError("invalid shard specification")
    assigned = cohort[shard_index::num_shards]
    if max_complexes is not None:
        if max_complexes < 1:
            raise ValueError("max_complexes must be positive")
        assigned = assigned[:max_complexes]
    return assigned


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def attempt_dir(final_dir: Path) -> Path:
    if final_dir.exists():
        raise FileExistsError(final_dir)
    incomplete = final_dir.parent / ".incomplete"
    incomplete.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=f"{final_dir.name}.", dir=incomplete))


def validated_posebusters_row(raw: dict[str, Any], *, label: str) -> dict[str, bool]:
    import pandas as pd

    rmsd_columns = [key for key in raw if str(key).startswith("rmsd_")]
    expected = set(VALIDITY_CHECKS)
    allowed = expected | set(rmsd_columns)
    if not set(raw).issubset(allowed) or not expected.issubset(raw) or len(rmsd_columns) > 1:
        raise ValueError(f"{label}: unexpected PoseBusters redock schema")
    return {key: False if pd.isna(raw[key]) else bool(raw[key]) for key in VALIDITY_CHECKS}


def posebusters_shard(args: argparse.Namespace) -> None:
    import posebusters
    from posebusters import PoseBusters
    from rdkit import Chem

    if posebusters.__version__ != POSEBUSTERS_VERSION:
        raise RuntimeError(
            f"PoseBusters must be {POSEBUSTERS_VERSION}, found {posebusters.__version__}"
        )
    cohort = load_official_cohort(args.metadata)
    ids = assigned_ids(
        cohort,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
        max_complexes=args.max_complexes,
    )
    shard_name = f"shard-{args.shard_index:03d}-of-{args.num_shards:03d}"
    final_dir = args.output_root / args.stage / "posebusters" / shard_name
    attempt = attempt_dir(final_dir)
    started = time.monotonic()
    redock = PoseBusters(config="redock", max_workers=0)
    validity_config = copy.deepcopy(redock.config)
    validity_config["modules"] = [
        module for module in validity_config["modules"] if module.get("function") != "rmsd"
    ]
    buster = PoseBusters(config=validity_config, max_workers=0)
    coverage_rows: list[dict[str, Any]] = []
    pose_rows: list[dict[str, Any]] = []
    try:
        for position, complex_id in enumerate(ids, start=1):
            inputs = source_inputs(
                args.source_run,
                complex_id,
                expected_confidence_sha256=args.confidence_checkpoint_sha256,
            )
            if inputs is None:
                coverage_rows.append(
                    {
                        "id": complex_id,
                        "prediction_available": False,
                        "num_ranked_poses": 0,
                        "reason": "excluded_from_source_run_by_pb_valid_ref_filter",
                    }
                )
                print(
                    f"[{position}/{len(ids)}] {complex_id}: missing source prediction", flush=True
                )
                continue
            ranked = load_ranked_scores(inputs["scores"], top_n=args.top_n)
            with inputs["poses"].open("rb") as handle:
                molecules = list(Chem.ForwardSDMolSupplier(handle, removeHs=False, sanitize=False))
            if len(molecules) != EXPECTED_POSES or any(mol is None for mol in molecules):
                raise ValueError(f"{complex_id}: expected 100 readable refined poses")
            selected = [molecules[row["pose_index"]] for row in ranked]
            frame = buster.bust(
                selected,
                inputs["reference"],
                inputs["protein"],
                full_report=False,
            )
            if len(frame.index) != args.top_n:
                raise ValueError(f"{complex_id}: expected {args.top_n} PoseBusters rows")
            for offset, ranked_row in enumerate(ranked):
                raw = frame.iloc[offset].to_dict()
                checks = validated_posebusters_row(
                    raw, label=f"{complex_id}.rank{ranked_row['rank']}"
                )
                pose_rows.append(
                    {
                        "id": complex_id,
                        **ranked_row,
                        "posebusters_valid": all(checks.values()),
                        **checks,
                    }
                )
            coverage_rows.append(
                {
                    "id": complex_id,
                    "prediction_available": True,
                    "num_ranked_poses": args.top_n,
                    "reason": "",
                }
            )
            valid_count = sum(bool(row["posebusters_valid"]) for row in pose_rows[-args.top_n :])
            print(
                f"[{position}/{len(ids)}] {complex_id}: PB-valid {valid_count}/{args.top_n}",
                flush=True,
            )

        write_csv(
            attempt / "coverage.csv",
            coverage_rows,
            ["id", "prediction_available", "num_ranked_poses", "reason"],
        )
        pose_fields = [
            "id",
            "rank",
            "pose_index",
            "confidence_predicted_rmsd",
            "internal_symmetry_rmsd",
            "posebusters_valid",
            *VALIDITY_CHECKS,
        ]
        write_csv(attempt / "poses.csv", pose_rows, pose_fields)
        summary = {
            "schema_version": "effdock.openbind_official_topn.posebusters_shard.v1",
            "protocol_id": PROTOCOL_ID,
            "status": "complete",
            "stage": args.stage,
            "num_shards": args.num_shards,
            "shard_index": args.shard_index,
            "top_n": args.top_n,
            "official_denominator": len(cohort),
            "official_ids_sha256": ids_sha256(cohort),
            "num_assigned": len(ids),
            "num_with_predictions": sum(
                parse_bool(row["prediction_available"], label=row["id"]) for row in coverage_rows
            ),
            "num_pose_rows": len(pose_rows),
            "posebusters_version": posebusters.__version__,
            "posebusters_config": "redock_without_separate_rmsd_module",
            "validity_checks": list(VALIDITY_CHECKS),
            "inputs": {
                "source_run": str(args.source_run.resolve()),
                "metadata": str(args.metadata.resolve()),
                "metadata_sha256": file_sha256(args.metadata),
                "confidence_checkpoint_sha256": args.confidence_checkpoint_sha256,
            },
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
        final_dir.parent.mkdir(parents=True, exist_ok=True)
        os.rename(attempt, final_dir)
    except Exception:
        shutil.rmtree(attempt, ignore_errors=True)
        raise


def split_sdf_records(path: Path) -> list[bytes]:
    payload = path.read_bytes()
    records = []
    for part in payload.split(b"$$$$"):
        if part.strip():
            records.append(part.strip(b"\r\n") + b"\n$$$$\n")
    if len(records) != EXPECTED_POSES:
        raise ValueError(f"{path}: expected 100 SDF records, found {len(records)}")
    return records


def parse_ost_csv(path: Path, *, label: str) -> dict[str, float]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError(f"{label}: expected one OpenStructure CSV row, found {len(rows)}")
    row = rows[0]
    required = {"rmsd", "rmsd_coverage", "lddt_pli", "lddt_pli_coverage"}
    if not required.issubset(row):
        raise ValueError(f"{label}: missing OpenStructure output columns")
    parsed = {key: float(row[key]) for key in required}
    if not all(math.isfinite(value) for value in parsed.values()):
        raise ValueError(f"{label}: non-finite OpenStructure score")
    return parsed


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def openstructure_shard(args: argparse.Namespace) -> None:
    version = subprocess.run(
        [str(args.ost_bin), "--version"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if version != f"OpenStructure {OPENSTRUCTURE_VERSION}":
        raise RuntimeError(f"unexpected OpenStructure version: {version!r}")
    cohort = load_official_cohort(args.metadata)
    ids = assigned_ids(
        cohort,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
        max_complexes=args.max_complexes,
    )
    shard_name = f"shard-{args.shard_index:03d}-of-{args.num_shards:03d}"
    pb_dir = args.output_root / args.stage / "posebusters" / shard_name
    pb_summary = read_json(pb_dir / "summary.json")
    if pb_summary.get("protocol_id") != PROTOCOL_ID or pb_summary.get("status") != "complete":
        raise ValueError(f"{pb_dir}: incomplete PoseBusters prerequisite")
    if (
        args.confidence_checkpoint_sha256 is not None
        and pb_summary.get("inputs", {}).get("confidence_checkpoint_sha256")
        != args.confidence_checkpoint_sha256
    ):
        raise ValueError(f"{pb_dir}: confidence checkpoint SHA mismatch")
    coverage = {row["id"]: row for row in read_csv_rows(pb_dir / "coverage.csv")}
    by_id: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in read_csv_rows(pb_dir / "poses.csv"):
        by_id[row["id"]].append(row)
    if set(coverage) != set(ids):
        raise ValueError(f"{pb_dir}: coverage identity mismatch")

    final_dir = args.output_root / args.stage / "openstructure" / shard_name
    attempt = attempt_dir(final_dir)
    started = time.monotonic()
    score_rows: list[dict[str, Any]] = []
    outcome_rows: list[dict[str, Any]] = []
    try:
        for position, complex_id in enumerate(ids, start=1):
            available = parse_bool(
                coverage[complex_id]["prediction_available"],
                label=f"{complex_id}.prediction_available",
            )
            if not available:
                outcome_rows.append(
                    {
                        "id": complex_id,
                        "prediction_available": False,
                        "evaluated_pose_count": 0,
                        "first_rmsd_valid_rank": "",
                        "first_success_valid_rank": "",
                    }
                )
                print(
                    f"[{position}/{len(ids)}] {complex_id}: missing source prediction", flush=True
                )
                continue
            inputs = source_inputs(
                args.source_run,
                complex_id,
                expected_confidence_sha256=args.confidence_checkpoint_sha256,
            )
            if inputs is None:
                raise ValueError(f"{complex_id}: source vanished after PoseBusters stage")
            ranked_rows = sorted(by_id[complex_id], key=lambda row: int(row["rank"]))
            if len(ranked_rows) != args.top_n or [int(row["rank"]) for row in ranked_rows] != list(
                range(args.top_n)
            ):
                raise ValueError(f"{complex_id}: incomplete ranked PoseBusters rows")
            records = split_sdf_records(inputs["poses"])
            first_rmsd_rank: int | None = None
            first_success_rank: int | None = None
            evaluated = 0
            with tempfile.TemporaryDirectory(prefix=f"effdock-ost-{complex_id}-") as tmp_raw:
                tmp = Path(tmp_raw)
                pose_path = tmp / "pose.sdf"
                result_path = tmp / "result.csv"
                for row in ranked_rows:
                    rank = int(row["rank"])
                    if not parse_bool(
                        row["posebusters_valid"], label=f"{complex_id}.rank{rank}.pb"
                    ):
                        continue
                    pose_index = int(row["pose_index"])
                    pose_path.write_bytes(records[pose_index])
                    result_path.unlink(missing_ok=True)
                    command = [
                        str(args.ost_bin),
                        "compare-ligand-structures",
                        "-m",
                        str(inputs["protein"]),
                        "-ml",
                        str(pose_path),
                        "-r",
                        str(inputs["protein"]),
                        "-rl",
                        str(inputs["reference"]),
                        "--lddt-pli",
                        "--rmsd",
                        "-of",
                        "csv",
                        "-csvm",
                        "-o",
                        str(result_path),
                    ]
                    completed = subprocess.run(command, capture_output=True, text=True)
                    if completed.returncode != 0 or not result_path.is_file():
                        raise RuntimeError(
                            f"{complex_id}.rank{rank}: OpenStructure failed: "
                            f"{completed.stderr[-2000:]}"
                        )
                    scores = parse_ost_csv(result_path, label=f"{complex_id}.rank{rank}")
                    evaluated += 1
                    rmsd_valid = scores["rmsd"] <= 2.0
                    lddt_valid = scores["lddt_pli"] >= 0.8
                    success_valid = rmsd_valid and lddt_valid
                    if rmsd_valid and first_rmsd_rank is None:
                        first_rmsd_rank = rank
                    if success_valid and first_success_rank is None:
                        first_success_rank = rank
                    score_rows.append(
                        {
                            "id": complex_id,
                            "rank": rank,
                            "pose_index": pose_index,
                            "posebusters_valid": True,
                            "rmsd": scores["rmsd"],
                            "rmsd_coverage": scores["rmsd_coverage"],
                            "lddt_pli": scores["lddt_pli"],
                            "lddt_pli_coverage": scores["lddt_pli_coverage"],
                            "rmsd_valid": rmsd_valid,
                            "lddt_pli_valid": lddt_valid,
                            "success_valid": success_valid,
                        }
                    )
                    if success_valid:
                        break
            outcome_rows.append(
                {
                    "id": complex_id,
                    "prediction_available": True,
                    "evaluated_pose_count": evaluated,
                    "first_rmsd_valid_rank": "" if first_rmsd_rank is None else first_rmsd_rank,
                    "first_success_valid_rank": (
                        "" if first_success_rank is None else first_success_rank
                    ),
                }
            )
            print(
                f"[{position}/{len(ids)}] {complex_id}: OST={evaluated}, "
                f"first_rmsd={first_rmsd_rank}, first_success={first_success_rank}",
                flush=True,
            )

        write_csv(
            attempt / "scores.csv",
            score_rows,
            [
                "id",
                "rank",
                "pose_index",
                "posebusters_valid",
                "rmsd",
                "rmsd_coverage",
                "lddt_pli",
                "lddt_pli_coverage",
                "rmsd_valid",
                "lddt_pli_valid",
                "success_valid",
            ],
        )
        write_csv(
            attempt / "outcomes.csv",
            outcome_rows,
            [
                "id",
                "prediction_available",
                "evaluated_pose_count",
                "first_rmsd_valid_rank",
                "first_success_valid_rank",
            ],
        )
        summary = {
            "schema_version": "effdock.openbind_official_topn.openstructure_shard.v1",
            "protocol_id": PROTOCOL_ID,
            "status": "complete",
            "stage": args.stage,
            "num_shards": args.num_shards,
            "shard_index": args.shard_index,
            "top_n": args.top_n,
            "official_denominator": len(cohort),
            "official_ids_sha256": ids_sha256(cohort),
            "num_assigned": len(ids),
            "num_score_rows": len(score_rows),
            "openstructure_version": version,
            "command_contract": (
                "compare-ligand-structures with explicit identical prepared model/reference "
                "protein PDBs, explicit predicted/reference ligand SDFs, --lddt-pli, --rmsd"
            ),
            "early_stop": (
                "per complex after first PB-valid pose satisfying both RMSD<=2A and "
                "LDDT-PLI>=0.8; otherwise all PB-valid Top-25 poses"
            ),
            "inputs": {
                "source_run": str(args.source_run.resolve()),
                "confidence_checkpoint_sha256": args.confidence_checkpoint_sha256,
            },
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
        final_dir.parent.mkdir(parents=True, exist_ok=True)
        os.rename(attempt, final_dir)
    except Exception:
        shutil.rmtree(attempt, ignore_errors=True)
        raise


def optional_rank(value: str) -> int | None:
    value = str(value).strip()
    return None if not value else int(value)


def report(args: argparse.Namespace) -> None:
    cohort = load_official_cohort(args.metadata)
    cohort_set = set(cohort)
    coverage_rows: list[dict[str, str]] = []
    pb_rows: list[dict[str, str]] = []
    ost_rows: list[dict[str, str]] = []
    outcome_rows: list[dict[str, str]] = []
    expected_hash = ids_sha256(cohort)
    for shard_index in range(args.num_shards):
        shard_name = f"shard-{shard_index:03d}-of-{args.num_shards:03d}"
        pb_dir = args.output_root / "full" / "posebusters" / shard_name
        ost_dir = args.output_root / "full" / "openstructure" / shard_name
        for directory in (pb_dir, ost_dir):
            summary = read_json(directory / "summary.json")
            if (
                summary.get("protocol_id") != PROTOCOL_ID
                or summary.get("status") != "complete"
                or int(summary.get("shard_index", -1)) != shard_index
                or int(summary.get("num_shards", -1)) != args.num_shards
                or summary.get("official_ids_sha256") != expected_hash
            ):
                raise ValueError(f"{directory}: incomplete or mismatched shard")
            if (
                args.confidence_checkpoint_sha256 is not None
                and summary.get("inputs", {}).get("confidence_checkpoint_sha256")
                != args.confidence_checkpoint_sha256
            ):
                raise ValueError(f"{directory}: confidence checkpoint SHA mismatch")
        coverage_rows.extend(read_csv_rows(pb_dir / "coverage.csv"))
        pb_rows.extend(read_csv_rows(pb_dir / "poses.csv"))
        ost_rows.extend(read_csv_rows(ost_dir / "scores.csv"))
        outcome_rows.extend(read_csv_rows(ost_dir / "outcomes.csv"))

    if len(coverage_rows) != len(cohort) or {row["id"] for row in coverage_rows} != cohort_set:
        raise ValueError("full coverage does not match official OpenBind denominator")
    if len(outcome_rows) != len(cohort) or {row["id"] for row in outcome_rows} != cohort_set:
        raise ValueError("OpenStructure outcomes do not match official denominator")
    coverage = {row["id"]: row for row in coverage_rows}
    outcomes = {row["id"]: row for row in outcome_rows}
    pb_by_id: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in pb_rows:
        pb_by_id[row["id"]].append(row)
    n_with_predictions = sum(
        parse_bool(row["prediction_available"], label=row["id"]) for row in coverage_rows
    )
    if len(pb_rows) != n_with_predictions * args.top_n:
        raise ValueError("PoseBusters pose-row coverage mismatch")

    complex_rows: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    for top_n in TOP_NS:
        if top_n > args.top_n:
            continue
        current: list[dict[str, Any]] = []
        for complex_id in cohort:
            available = parse_bool(
                coverage[complex_id]["prediction_available"],
                label=f"{complex_id}.prediction_available",
            )
            selected_pb = [row for row in pb_by_id.get(complex_id, []) if int(row["rank"]) < top_n]
            pb_valid_any = any(
                parse_bool(row["posebusters_valid"], label=f"{complex_id}.pb")
                for row in selected_pb
            )
            internal_rmsd_pb_valid = any(
                parse_bool(row["posebusters_valid"], label=f"{complex_id}.pb")
                and float(row["internal_symmetry_rmsd"]) <= 2.0
                for row in selected_pb
            )
            first_rmsd = optional_rank(outcomes[complex_id]["first_rmsd_valid_rank"])
            first_success = optional_rank(outcomes[complex_id]["first_success_valid_rank"])
            rmsd_valid = first_rmsd is not None and first_rmsd < top_n
            success_valid = first_success is not None and first_success < top_n
            row = {
                "id": complex_id,
                "top_n": top_n,
                "prediction_available": available,
                "n_poses": len(selected_pb),
                "pb_valid_any": pb_valid_any,
                "internal_symmetry_rmsd_pb_valid": internal_rmsd_pb_valid,
                "rmsd_valid": rmsd_valid,
                "success_valid": success_valid,
                "first_rmsd_valid_rank": "" if first_rmsd is None else first_rmsd,
                "first_success_valid_rank": "" if first_success is None else first_success,
            }
            current.append(row)
            complex_rows.append(row)
        n_rmsd_valid = sum(bool(row["rmsd_valid"]) for row in current)
        n_success_valid = sum(bool(row["success_valid"]) for row in current)
        summaries[str(top_n)] = {
            "top_n": top_n,
            "n_total": len(cohort),
            "n_with_predictions": n_with_predictions,
            "mean_n_poses": sum(int(row["n_poses"]) for row in current) / len(cohort),
            "n_pb_valid_any": sum(bool(row["pb_valid_any"]) for row in current),
            "n_internal_symmetry_rmsd_pb_valid": sum(
                bool(row["internal_symmetry_rmsd_pb_valid"]) for row in current
            ),
            "n_rmsd_valid": n_rmsd_valid,
            "n_success_valid": n_success_valid,
            "pb_valid_any_pct": 100
            * sum(bool(row["pb_valid_any"]) for row in current)
            / len(cohort),
            "internal_symmetry_rmsd_pb_valid_pct": 100
            * sum(bool(row["internal_symmetry_rmsd_pb_valid"]) for row in current)
            / len(cohort),
            "rmsd_valid_pct": 100 * n_rmsd_valid / len(cohort),
            "success_valid_pct": 100 * n_success_valid / len(cohort),
        }

    report_dir = args.output_root / "report"
    if report_dir.exists():
        raise FileExistsError(report_dir)
    attempt = attempt_dir(report_dir)
    write_csv(
        attempt / "complex_results.csv",
        complex_rows,
        [
            "id",
            "top_n",
            "prediction_available",
            "n_poses",
            "pb_valid_any",
            "internal_symmetry_rmsd_pb_valid",
            "rmsd_valid",
            "success_valid",
            "first_rmsd_valid_rank",
            "first_success_valid_rank",
        ],
    )
    write_csv(
        attempt / "posebusters_poses.csv",
        pb_rows,
        list(pb_rows[0]) if pb_rows else [],
    )
    write_csv(
        attempt / "openstructure_scores_evaluated.csv",
        ost_rows,
        list(ost_rows[0]) if ost_rows else [],
    )
    summary = {
        "schema_version": "effdock.openbind_official_topn.report.v1",
        "protocol_id": PROTOCOL_ID,
        "status": "complete",
        "official_contract": {
            "denominator": len(cohort),
            "filtered": True,
            "scaffold_only": True,
            "rank_rule": "zero-based stable confidence rank < top_n",
            "rmsd_valid": "any selected pose with PoseBusters-valid and OST BiSyRMSD <= 2A",
            "success_valid": (
                "any selected pose with PoseBusters-valid, OST BiSyRMSD <= 2A, and LDDT-PLI >= 0.8"
            ),
            "missing_predictions_count_as_failures": True,
        },
        "coverage": {
            "n_total": len(cohort),
            "n_with_predictions": n_with_predictions,
            "n_missing_predictions": len(cohort) - n_with_predictions,
            "official_ids_sha256": expected_hash,
        },
        "metrics": summaries,
        "versions": {
            "posebusters": POSEBUSTERS_VERSION,
            "openstructure": OPENSTRUCTURE_VERSION,
        },
        "inputs": {
            "source_run": str(args.source_run.resolve()),
            "metadata": str(args.metadata.resolve()),
            "metadata_sha256": file_sha256(args.metadata),
            "confidence_checkpoint_sha256": args.confidence_checkpoint_sha256,
        },
        "finished_at_utc": datetime.now(UTC).isoformat(),
    }
    (attempt / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# EFF-Dock OpenBind official-style Top-N aggregation",
        "",
        f"Protocol: `{PROTOCOL_ID}`",
        "",
        (
            f"Official filtered scaffold-only denominator: **{len(cohort)}**; "
            f"predictions: **{n_with_predictions}**; missing predictions counted as failures: "
            f"**{len(cohort) - n_with_predictions}**."
        ),
        "",
        "| Rank budget | PB-valid + BiSyRMSD <= 2 A | + LDDT-PLI >= 0.8 | Any PB-valid |",
        "| --- | ---: | ---: | ---: |",
    ]
    for top_n in TOP_NS:
        if str(top_n) not in summaries:
            continue
        metric = summaries[str(top_n)]
        lines.append(
            f"| Top-{top_n} | {metric['n_rmsd_valid']}/{len(cohort)} "
            f"({metric['rmsd_valid_pct']:.2f}%) | {metric['n_success_valid']}/{len(cohort)} "
            f"({metric['success_valid_pct']:.2f}%) | {metric['n_pb_valid_any']}/{len(cohort)} "
            f"({metric['pb_valid_any_pct']:.2f}%) |"
        )
    lines.extend(
        [
            "",
            "The internal symmetry-RMSD column is retained only as a diagnostic. The two primary "
            "columns use OpenStructure BiSyRMSD/LDDT-PLI and PoseBusters pass-all validity.",
        ]
    )
    (attempt / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    report_dir.parent.mkdir(parents=True, exist_ok=True)
    os.rename(attempt, report_dir)
    print(json.dumps(summary["metrics"], indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--source-run", type=Path, required=True)
    common.add_argument("--output-root", type=Path, required=True)
    common.add_argument(
        "--metadata",
        type=Path,
        default=Path("data/external_benchmarks/data/OpenBind_EV-A71_2A/EV-A71_2A_metadata.csv"),
    )
    common.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    common.add_argument("--num-shards", type=int, required=True)
    common.add_argument("--confidence-checkpoint-sha256")

    for name in ("posebusters-shard", "openstructure-shard"):
        current = subparsers.add_parser(name, parents=[common])
        current.add_argument("--stage", choices=("smoke", "full"), required=True)
        current.add_argument("--shard-index", type=int, required=True)
        current.add_argument("--max-complexes", type=int, default=None)
    ost = subparsers.choices["openstructure-shard"]
    ost.add_argument("--ost-bin", type=Path, default=Path(".venvs/openstructure/bin/ost"))
    subparsers.add_parser("report", parents=[common])
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.top_n != DEFAULT_TOP_N:
        raise ValueError(f"frozen protocol requires top_n={DEFAULT_TOP_N}")
    if args.command == "posebusters-shard":
        posebusters_shard(args)
    elif args.command == "openstructure-shard":
        openstructure_shard(args)
    else:
        report(args)


if __name__ == "__main__":
    main()
