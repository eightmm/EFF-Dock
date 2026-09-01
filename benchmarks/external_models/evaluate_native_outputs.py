#!/usr/bin/env python3
"""Evaluate native external-model poses with one strict no-align RMSD metric.

Missing complexes and topology failures remain in the frozen dataset denominator.
The script intentionally does not run PoseBusters; it writes the selected-pose
inventory needed by the separate official 27-check stage.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolAlign

from effdock.evaluation.benchmark import _strip_charges, load_sdf_robust

ROOT = Path(__file__).resolve().parents[2]
DATASETS = {
    "astex_diverse": (
        85,
        ROOT
        / "outputs/external_models/inputs/posebench_native/astex_diverse/"
        "vina_astex_diverse_inputs.csv",
    ),
    "posebusters_benchmark": (
        308,
        ROOT
        / "outputs/external_models/inputs/posebench_native/posebusters_benchmark/"
        "vina_posebusters_benchmark_inputs.csv",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        required=True,
        choices=(
            "diffdock_pocket",
            "rldiff",
            "rldiff_rlpp",
            "surfdock",
            "diffbindfr",
            "interformer",
            "posebench_diffdock",
            "posebench_fabind",
            "posebench_dynamicbind",
            "posebench_vina",
            "sigmadock",
        ),
    )
    parser.add_argument("--dataset", required=True, choices=tuple(DATASETS))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--official-seed",
        type=int,
        help=(
            "Evaluate one repeat from the frozen 2026-08-31 supplied-pocket "
            "campaign instead of the historical merged output roots."
        ),
    )
    return parser.parse_args()


def read_manifest(dataset: str, limit: int | None) -> list[dict[str, str]]:
    expected, path = DATASETS[dataset]
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != expected:
        raise ValueError(f"Frozen denominator mismatch: {path}: {len(rows)} != {expected}")
    if len({row["complex_name"] for row in rows}) != expected:
        raise ValueError(f"Duplicate complex_name in {path}")
    return rows[:limit] if limit is not None else rows


def coverage_entries(paths: list[Path]) -> dict[str, dict[str, object]]:
    merged: dict[str, dict[str, object]] = {}
    for path in sorted(paths):
        payload = json.loads(path.read_text())
        for target, entry in payload.get("coverage", {}).items():
            old = merged.get(target)
            if old is None or int(entry.get("pose_count", 0)) >= int(old.get("pose_count", 0)):
                merged[target] = entry
    return merged


def ranked_files(directory: Path, pattern: str) -> list[Path]:
    if not directory.is_dir():
        return []
    regex = re.compile(pattern)
    ranks: dict[int, Path] = {}
    for path in directory.glob("*.sdf"):
        match = regex.search(path.name)
        if match:
            ranks[int(match.group(1))] = path
    return [ranks[rank] for rank in sorted(ranks)]


def sdf_record_count(path: Path) -> int:
    with path.open(errors="replace") as handle:
        count = sum(line.strip() == "$$$$" for line in handle)
    return count or int(path.stat().st_size > 0)


def load_pose_sdf_preserve_components(path: Path) -> Chem.Mol:
    """Load one pose record without discarding non-largest components."""
    supplier = Chem.SDMolSupplier(str(path), sanitize=True, removeHs=False)
    molecule = next(supplier)
    if molecule is not None:
        return molecule

    supplier = Chem.SDMolSupplier(str(path), sanitize=False, removeHs=False)
    molecule = next(supplier)
    if molecule is None:
        raise ValueError(f"RDKit cannot parse {path}")
    molecule.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(molecule)
    relaxed = (
        Chem.SanitizeFlags.SANITIZE_FINDRADICALS
        | Chem.SanitizeFlags.SANITIZE_SETAROMATICITY
        | Chem.SanitizeFlags.SANITIZE_SETCONJUGATION
        | Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION
        | Chem.SanitizeFlags.SANITIZE_SYMMRINGS
    )
    Chem.SanitizeMol(molecule, sanitizeOps=relaxed)
    return molecule


def indexed_rank_directories(root: Path, pattern: str) -> dict[str, list[Path]]:
    regex = re.compile(pattern)
    found: dict[str, list[Path]] = defaultdict(list)
    if not root.is_dir():
        return found
    for path in root.rglob("*.sdf"):
        match = regex.search(path.name)
        if not match:
            continue
        target_match = re.search(
            r"(?:^|___)([0-9A-Za-z]{4}_[0-9A-Za-z]+?)(?:_|$)",
            path.parent.name,
        )
        if target_match is None:
            target_match = re.fullmatch(r"([0-9A-Za-z]{4})", path.parent.name)
        target = target_match.group(1) if target_match else ""
        if target:
            found[target].append(path)
    output: dict[str, list[Path]] = {}
    for target, paths in found.items():
        by_rank: dict[int, Path] = {}
        for path in paths:
            match = regex.search(path.name)
            assert match is not None
            by_rank[int(match.group(1))] = path
        output[target] = [by_rank[rank] for rank in sorted(by_rank)]
    return output


def collect_official_poses(
    model: str, dataset: str, seed: int
) -> dict[str, list[Path]]:
    """Collect one immutable repeat from the official supplied-pocket campaign."""
    runs = ROOT / "outputs/external_models/runs"
    if model == "diffdock_pocket":
        return indexed_rank_directories(
            runs
            / "diffdock_pocket/official_s30_n40_r2_20260831"
            / dataset
            / f"seed_{seed}",
            r"^rank(\d+)_confidence.*\.sdf$",
        )
    if model == "rldiff_rlpp":
        output = indexed_rank_directories(
            runs
            / "rldiff/official_rlpp_r2_20260831"
            / dataset
            / "rlpp"
            / f"seed_{seed}",
            r"^rank(\d+)_gnina.*\.sdf$",
        )
        recovery = indexed_rank_directories(
            runs
            / "rldiff/rlpp_recovery_20260901"
            / dataset
            / f"seed_{seed}",
            r"^rank(\d+)_gnina.*\.sdf$",
        )
        for target, paths in recovery.items():
            if len(paths) >= 40:
                output[target] = paths
        return output
    if model == "diffbindfr":
        base = (
            runs
            / "diffbindfr/official_holo_ec_r1_20260830"
            / dataset
            / f"seed_{seed}"
        )
        output: dict[str, list[tuple[float, Path]]] = defaultdict(list)
        for path in base.rglob("results_ec.csv"):
            with path.open(newline="") as handle:
                for row in csv.DictReader(handle):
                    try:
                        score = float(row["mdn_score"])
                    except (KeyError, TypeError, ValueError):
                        continue
                    pose = Path(row["docked_lig"])
                    if pose.is_file() and math.isfinite(score):
                        output[row["complex_name"]].append((score, pose))
        return {
            target: [pose for _, pose in sorted(values, key=lambda item: item[0], reverse=True)]
            for target, values in output.items()
        }
    if model == "posebench_vina":
        base = (
            runs
            / "posebench_vina/official_ex32_n40_r3_20260831"
            / dataset
            / f"seed_{seed}"
        )
        entries = coverage_entries(list(base.rglob("coverage.json")))
        return {
            target: [Path(str(entry["prediction"]))]
            for target, entry in entries.items()
            if entry.get("prediction") and Path(str(entry["prediction"])).is_file()
        }
    raise ValueError(f"No official supplied-pocket collector for model={model!r}")


def collect_poses(
    model: str, dataset: str, official_seed: int | None = None
) -> dict[str, list[Path]]:
    if official_seed is not None:
        return collect_official_poses(model, dataset, official_seed)
    runs = ROOT / "outputs/external_models/runs"
    if model == "diffdock_pocket":
        base = runs / "diffdock_pocket/full" / dataset / "native_s30_n40_predicted_pocket_crop"
        output = indexed_rank_directories(base, r"^rank(\d+)_confidence.*\.sdf$")
        recovery = indexed_rank_directories(
            runs / "diffdock_pocket/recovery_pose_integrity_20260830_v1" / dataset,
            r"^rank(\d+)_confidence.*\.sdf$",
        )
        for target, paths in recovery.items():
            if len(paths) >= 40:
                output[target] = paths
        return output
    if model in {"rldiff", "rldiff_rlpp"}:
        arm = (
            "native_s20_n40_pocket_crop_clean"
            if model == "rldiff"
            else "rlpp_raw_s20_n40_pocket_crop"
        )
        return indexed_rank_directories(
            runs / "rldiff/full" / dataset / arm,
            r"^rank(\d+)\.sdf$",
        )
    if model == "posebench_diffdock":
        primary_only: dict[str, list[Path]] = {}
        with DATASETS[dataset][1].open(newline="") as handle:
            pdb_aliases = {
                row["complex_name"][:4].upper(): row["complex_name"]
                for row in csv.DictReader(handle)
            }
        for root in (
            runs / "posebench_diffdock/primary_only_20260830_v1/full" / dataset,
            runs / "posebench_diffdock/primary_only_20260830_v1/recovery" / dataset,
        ):
            for target, paths in indexed_rank_directories(
                root, r"^rank(\d+)_confidence.*\.sdf$"
            ).items():
                canonical_target = pdb_aliases.get(target.upper(), target)
                if len(paths) >= len(primary_only.get(canonical_target, [])):
                    primary_only[canonical_target] = paths
        # Never mix ligand-description protocols.  A completion marker makes
        # the primary-reference-SMILES campaign authoritative even when the
        # pinned upstream model explicitly rejects a frozen-denominator target.
        # Those targets must remain failures rather than silently falling back
        # to poses generated with the old multi-component ligand description.
        campaign_marker = (
            runs
            / "posebench_diffdock/primary_only_20260830_v1"
            / f"{dataset}_campaign_complete.json"
        )
        if campaign_marker.is_file():
            return primary_only
        output: dict[str, list[Path]] = {}
        bases = [
            runs / "posebench_diffdock/full" / dataset / "native_s20_n5_clean",
            runs / "posebench_diffdock/recovery_20260828_v1" / dataset,
        ]
        for base in bases:
            for target, paths in indexed_rank_directories(
                base, r"^rank(\d+)_confidence.*\.sdf$"
            ).items():
                if len(paths) > len(output.get(target, [])):
                    output[target] = paths
        return output
    if model == "surfdock":
        short = "astex" if dataset == "astex_diverse" else "posebusters"
        roots = [
            runs / "surfdock" / f"{short}_native_s20_n40_seed0",
            runs / "surfdock/recovery_20260828_v1" / dataset,
            runs / "surfdock/recovery_pose_integrity_20260830_v1" / dataset,
        ]
        entries = coverage_entries(
            [path for root in roots if root.is_dir() for path in root.rglob("coverage.json")]
        )
        return {
            target: ranked_files(Path(str(entry["result_dir"])), r"_rank_(\d+)_rmsd_")
            for target, entry in entries.items()
            if entry.get("result_dir")
        }
    if model == "posebench_dynamicbind":
        clean_recovery_root = (
            runs / "posebench_dynamicbind/recovery_pose_integrity_20260830_v2" / dataset
        )
        roots = [
            runs / "posebench_dynamicbind/recovery_20260828_v1/full" / dataset,
            runs / "posebench_dynamicbind/recovery_20260829_8f4j_batch1",
            runs / "posebench_dynamicbind/recovery_pose_integrity_20260830_v1" / dataset,
            clean_recovery_root,
        ]
        entries = coverage_entries(
            [path for root in roots if root.is_dir() for path in root.rglob("coverage.json")]
        )
        # The first integrity rerun reused upstream result tags, so old and new
        # files accumulated in one directory (53--78 files instead of 40).
        # A clean-tag rerun is authoritative whenever it emitted poses. The
        # pinned DynamicBind names poses by rounded scores, so equal-score file
        # collisions make 30--40 outputs per target a native variable-N result.
        clean_entries = coverage_entries(
            list(clean_recovery_root.rglob("coverage.json"))
            if clean_recovery_root.is_dir()
            else []
        )
        for target, entry in clean_entries.items():
            if int(entry.get("pose_count", 0)) > 0:
                entries[target] = entry
        return {
            target: ranked_files(Path(str(entry["result_dir"])), r"^rank(\d+)_ligand_")
            for target, entry in entries.items()
            if entry.get("result_dir")
        }
    if model == "posebench_fabind":
        base = runs / "posebench_fabind/full" / dataset / "native_postoptim_seed0"
        output: dict[str, list[Path]] = defaultdict(list)
        for path in base.rglob("*.sdf"):
            match = re.match(r"^([0-9A-Za-z]{4}_[0-9A-Za-z]+)_\d+\.sdf$", path.name)
            if match:
                output[match.group(1)].append(path)
        return dict(output)
    if model == "posebench_vina":
        roots = [
            runs / "posebench_vina/recovery_20260828_v3/full" / dataset,
            runs / "posebench_vina/official_20260830_v1/full" / dataset,
            runs / "posebench_vina/official_20260830_v2/full" / dataset,
            runs / "posebench_vina/official_20260830_v2/recovery" / dataset,
            runs / "posebench_vina/official_20260830_v2/recovery_wave2" / dataset,
            runs / "posebench_vina/official_20260830_v2/recovery_wave3" / dataset,
        ]
        output: dict[str, list[Path]] = {}
        for root in roots:
            if not root.is_dir():
                continue
            for path in root.rglob("*.sdf"):
                if path.parent.name != path.stem:
                    continue
                target = path.parent.name
                if target not in output or sdf_record_count(path) > sdf_record_count(
                    output[target][0]
                ):
                    output[target] = [path]
        return output
    if model == "interformer":
        short = "astex" if dataset == "astex_diverse" else "posebusters"
        base = runs / "interformer" / f"{short}_native_n20_seed0"
        entries = coverage_entries(list(base.rglob("coverage.json")))
        score_base = (
            runs
            / "interformer_pose_score/official_v02_20260830_v1/full"
            / dataset
        )
        score_csv_by_target: dict[str, Path] = {}
        for score_csv in score_base.rglob("query.round0_ensemble.csv"):
            with score_csv.open(newline="") as handle:
                for row in csv.DictReader(handle):
                    target = str(row.get("Target", ""))
                    if target:
                        score_csv_by_target[target.lower()] = score_csv
        return {
            target: [Path(str(entry["output_sdf"])), score_csv_by_target[target[:4].lower()]]
            for target, entry in entries.items()
            if entry.get("output_sdf") and target[:4].lower() in score_csv_by_target
        }
    if model == "diffbindfr":
        short = "astex" if dataset == "astex_diverse" else "posebusters"
        base = runs / "diffbindfr" / f"{short}_native_s20_n40_seed0"
        output: dict[str, list[tuple[float, Path]]] = defaultdict(list)
        for path in base.rglob("results.csv"):
            with path.open(newline="") as handle:
                for row in csv.DictReader(handle):
                    try:
                        score = float(row["mdn_score"])
                    except (KeyError, TypeError, ValueError):
                        continue
                    pose = Path(row["docked_lig"])
                    if pose.is_file() and math.isfinite(score):
                        output[row["complex_name"]].append((score, pose))
        return {
            target: [pose for _, pose in sorted(values, key=lambda item: item[0], reverse=True)]
            for target, values in output.items()
        }
    raise AssertionError(model)


def load_pose_records(
    paths: list[Path], model: str
) -> tuple[list[tuple[int, Path | str, Chem.Mol]], list[str], int]:
    records: list[tuple[int, Path | str, Chem.Mol]] = []
    errors: list[str] = []
    available = 0
    if model == "interformer":
        sdf_paths = [path for path in paths if path.suffix.lower() == ".sdf"]
        score_paths = [path for path in paths if path.suffix.lower() == ".csv"]
        if len(sdf_paths) != 1 or len(score_paths) != 1:
            raise ValueError("Interformer requires one pose SDF and one PoseScore CSV")
        sdf_path, score_path = sdf_paths[0], score_paths[0]
        target = sdf_path.stem.split("_docked", 1)[0].lower()
        with score_path.open(newline="") as handle:
            score_rows = [
                row
                for row in csv.DictReader(handle)
                if str(row.get("Target", "")).lower() == target
                and int(float(row["pose_rank"])) < 20
            ]
        ranks = [int(float(row["pose_rank"])) for row in score_rows]
        if len(ranks) != len(set(ranks)):
            raise ValueError(f"duplicate Interformer PoseScore ranks for {target}")
        ranked = sorted(score_rows, key=lambda row: float(row["pred_pose"]), reverse=True)
        molecules = list(Chem.SDMolSupplier(str(sdf_path), sanitize=True, removeHs=True))
        available = len(ranked)
        for selected_index, row in enumerate(ranked):
            native_rank = int(float(row["pose_rank"]))
            if native_rank >= len(molecules) or molecules[native_rank] is None:
                errors.append(
                    f"candidate_{selected_index}:native_rank_{native_rank}:unparseable:{sdf_path}"
                )
                continue
            source = f"{sdf_path}#record={native_rank};pred_pose={float(row['pred_pose']):.8g}"
            records.append((selected_index, source, molecules[native_rank]))
        return records, errors, available
    for path in paths:
        if model == "posebench_vina":
            supplier = Chem.SDMolSupplier(str(path), sanitize=True, removeHs=True)
            limit = 40
            for record_index, mol in enumerate(supplier):
                available += 1
                if mol is not None:
                    records.append((record_index, path, mol))
                else:
                    errors.append(f"candidate_{record_index}:unparseable:{path}")
                if available == limit:
                    return records, errors, available
        else:
            candidate_index = available
            if model not in {"diffbindfr", "posebench_fabind"}:
                rank_match = re.search(r"(?:^|_)rank_?(\d+)(?:_|\.)", path.name)
                if rank_match:
                    candidate_index = int(rank_match.group(1)) - 1
            available += 1
            try:
                records.append(
                    (candidate_index, path, load_pose_sdf_preserve_components(path))
                )
            except Exception as exc:
                errors.append(
                    f"candidate_{candidate_index}:{type(exc).__name__}:{exc}:{path}"
                )
    return records, errors, available


def _single_component_rmsd(pose: Chem.Mol, reference: Chem.Mol) -> tuple[float, str]:
    if pose.GetNumAtoms() != reference.GetNumAtoms():
        raise ValueError(
            f"heavy_atom_count_mismatch:{pose.GetNumAtoms()}!={reference.GetNumAtoms()}"
        )
    try:
        return float(rdMolAlign.CalcRMS(pose, reference)), "rdkit_calc_rms"
    except Exception:
        pose_nocharge = _strip_charges(pose)
        reference_nocharge = _strip_charges(reference)
        try:
            return (
                float(rdMolAlign.CalcRMS(pose_nocharge, reference_nocharge)),
                "rdkit_calc_rms_charge_agnostic",
            )
        except Exception as exc:
            raise ValueError(f"full_topology_match_failed:{type(exc).__name__}") from exc


def no_align_rmsd(pose: Chem.Mol, reference: Chem.Mol) -> tuple[float, str]:
    # RDKit's normal RemoveHs intentionally retains stereochemistry-defining
    # explicit hydrogens (e.g. Interformer's imine `[H]/N=` records).  They are
    # not heavy atoms and must not enter the heavy-atom topology/RMSD contract.
    pose = Chem.RemoveAllHs(pose)
    reference = Chem.RemoveAllHs(reference)
    fragments = Chem.GetMolFrags(pose, asMols=True, sanitizeFrags=False)
    if len(fragments) == 1:
        return _single_component_rmsd(fragments[0], reference)

    # PoseBench predictions can contain the requested primary ligand together
    # with docked cofactors/ions.  Frozen references contain exactly one
    # primary component, so retain the first disconnected component with a
    # full topology match rather than comparing the combined entity.
    failures: list[str] = []
    for fragment_index, fragment in enumerate(fragments):
        if fragment.GetNumAtoms() != reference.GetNumAtoms():
            continue
        try:
            value, method = _single_component_rmsd(fragment, reference)
            return value, f"{method}_primary_fragment_{fragment_index}"
        except ValueError as exc:
            failures.append(str(exc))
    detail = failures[0] if failures else "no_fragment_with_reference_atom_count"
    raise ValueError(f"primary_ligand_fragment_match_failed:{detail}")


def evaluate(
    model: str,
    dataset: str,
    limit: int | None,
    official_seed: int | None = None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    if model == "sigmadock":
        if official_seed is not None:
            raise ValueError("SigmaDock official repeats use evaluate_sigmadock_official.py")
        return evaluate_sigmadock(dataset, limit)
    manifest = read_manifest(dataset, limit)
    poses_by_target = collect_poses(model, dataset, official_seed)
    rows: list[dict[str, object]] = []
    for item in manifest:
        target = item["complex_name"]
        paths = poses_by_target.get(target, [])
        row: dict[str, object] = {
            "model": model,
            "dataset": dataset,
            "complex_name": target,
            "reference_ligand": item["reference_ligand"],
            "available_pose_count": len(paths),
            "evaluated_pose_count": 0,
            "top1_rmsd": math.inf,
            "oracle_rmsd": math.inf,
            "top1_pose": "",
            "oracle_pose": "",
            "rmsd_method": "",
            "error": "",
            "pose_error_count": 0,
            "pose_errors": "",
        }
        if not paths:
            row["error"] = "missing_prediction"
            rows.append(row)
            continue
        try:
            reference = load_sdf_robust(Path(item["reference_ligand"]))
            records, pose_errors, available = load_pose_records(paths, model)
            values: list[tuple[int, float, Path, str]] = []
            for candidate_index, path, pose in records:
                try:
                    value, method = no_align_rmsd(pose, reference)
                    values.append((candidate_index, value, path, method))
                except Exception as exc:
                    pose_errors.append(
                        f"candidate_{candidate_index}:{type(exc).__name__}:{exc}:{path}"
                    )
            row.update(
                {
                    "available_pose_count": available,
                    "evaluated_pose_count": len(values),
                    "pose_error_count": len(pose_errors),
                    "pose_errors": " | ".join(pose_errors),
                }
            )
            if not values:
                raise ValueError("no_parseable_pose")
            top1 = next((value for value in values if value[0] == 0), None)
            oracle = min(values, key=lambda value: value[1])
            row.update(
                {
                    "available_pose_count": available,
                    "evaluated_pose_count": len(values),
                    "top1_rmsd": top1[1] if top1 is not None else math.inf,
                    "oracle_rmsd": oracle[1],
                    "top1_pose": str(top1[2]) if top1 is not None else "",
                    "oracle_pose": str(oracle[2]),
                    "rmsd_method": top1[3] if top1 is not None else "",
                }
            )
            if top1 is None:
                row["error"] = "top1_unparseable_or_unmappable"
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}:{exc}"
        rows.append(row)

    denominator = len(rows)
    top1_count = sum(float(row["top1_rmsd"]) < 2.0 for row in rows)
    oracle_available_count = sum(float(row["oracle_rmsd"]) < 2.0 for row in rows)
    oracle40_count = sum(
        int(row["evaluated_pose_count"]) >= 40 and float(row["oracle_rmsd"]) < 2.0
        for row in rows
    )
    complete = sum(int(row["evaluated_pose_count"]) >= 40 for row in rows)
    any_pose = sum(int(row["evaluated_pose_count"]) > 0 for row in rows)
    pose_error_count = sum(int(row["pose_error_count"]) for row in rows)
    pose_error_target_count = sum(int(row["pose_error_count"]) > 0 for row in rows)
    primary_fragment_top1_count = sum(
        "_primary_fragment_" in str(row["rmsd_method"]) for row in rows
    )
    summary = {
        "schema_version": 1,
        "model": model,
        "dataset": dataset,
        "denominator": denominator,
        "targets_with_any_evaluated_pose": any_pose,
        "targets_with_at_least_40_evaluated_poses": complete,
        "top1_rmsd_lt2_count": top1_count,
        "top1_rmsd_lt2_pct": 100.0 * top1_count / denominator,
        "oracle_available_rmsd_lt2_count": oracle_available_count,
        "oracle_available_rmsd_lt2_pct": 100.0 * oracle_available_count / denominator,
        "oracle40_rmsd_lt2_count": oracle40_count,
        "oracle40_rmsd_lt2_pct": 100.0 * oracle40_count / denominator,
        "failed_target_count": denominator - any_pose,
        "pose_error_count": pose_error_count,
        "pose_error_target_count": pose_error_target_count,
        "primary_fragment_top1_count": primary_fragment_top1_count,
        "rmsd_definition": (
            "RDKit symmetry-aware heavy-atom CalcRMS without alignment; all "
            "explicit hydrogens removed; first full-topology-matching primary "
            "component selected when cofactors are present; charge-agnostic "
            "full-topology retry only"
        ),
        "oracle40_policy": "targets with fewer than 40 evaluated poses count as failures",
    }
    if official_seed is not None:
        summary.update(
            {
                "repeat_seed": official_seed,
                "campaign": "official supplied-pocket 2026-08-31",
            }
        )
    return rows, summary


def evaluate_sigmadock(
    dataset: str, limit: int | None
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Evaluate independent SigmaDock seeds, ranked by native Vinardo affinity."""

    import gc

    import torch
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
    manifest = read_manifest(dataset, limit)
    wanted = {row["complex_name"] for row in manifest}
    references = {
        row["complex_name"]: load_sdf_robust(Path(row["reference_ligand"]))
        for row in manifest
    }
    candidates: dict[str, list[tuple[float, float, str, str]]] = defaultdict(list)
    base = (
        ROOT
        / "outputs/external_models/runs/sigmadock/"
        "s25_n40_predicted_receptor_20260828/full_compat_v2/results"
        / dataset
        / "sigmadock_v0p1_beta_s25"
    )
    for seed_dir in sorted(base.glob("seed_*"), key=lambda path: int(path.name.split("_")[1])):
        predictions_path = seed_dir / "predictions.pt"
        rescoring_path = seed_dir / "rescoring.pt"
        coverage_path = seed_dir / "coverage.json"
        if not (predictions_path.is_file() and rescoring_path.is_file() and coverage_path.is_file()):
            continue
        coverage = json.loads(coverage_path.read_text())
        if int(coverage.get("complete_targets", 0)) != DATASETS[dataset][0]:
            continue
        predictions = torch.load(predictions_path, map_location="cpu", weights_only=False)
        rescoring = torch.load(rescoring_path, map_location="cpu", weights_only=False)
        for key, records in predictions["results"].items():
            target = key.split("::", 1)[0]
            if target not in wanted or not records:
                continue
            score_rows = rescoring.get("scores", {}).get(key, [])
            if not score_rows:
                continue
            try:
                score = float(score_rows[0]["Affinity"])
                record = records[0]
                pose = Chem.Mol(record["lig_ref"])
                coordinates = record["x0_hat"].detach().cpu().numpy()
                if coordinates.shape != (pose.GetNumAtoms(), 3):
                    raise ValueError(
                        f"coordinate_shape_mismatch:{coordinates.shape}!={(pose.GetNumAtoms(), 3)}"
                    )
                conformer = pose.GetConformer()
                for atom_index, xyz in enumerate(coordinates):
                    conformer.SetAtomPosition(atom_index, [float(value) for value in xyz])
                rmsd, method = no_align_rmsd(pose, references[target])
                candidates[target].append(
                    (score, rmsd, f"{predictions_path}#{key}", method)
                )
            except Exception:
                continue
        del predictions, rescoring
        gc.collect()

    rows: list[dict[str, object]] = []
    for item in manifest:
        target = item["complex_name"]
        values = candidates.get(target, [])
        row: dict[str, object] = {
            "model": "sigmadock",
            "dataset": dataset,
            "complex_name": target,
            "reference_ligand": item["reference_ligand"],
            "available_pose_count": len(values),
            "evaluated_pose_count": len(values),
            "top1_rmsd": math.inf,
            "oracle_rmsd": math.inf,
            "top1_pose": "",
            "oracle_pose": "",
            "rmsd_method": "",
            "error": "",
        }
        if values:
            ranked = sorted(values, key=lambda value: value[0])
            oracle = min(values, key=lambda value: value[1])
            row.update(
                {
                    "top1_rmsd": ranked[0][1],
                    "oracle_rmsd": oracle[1],
                    "top1_pose": ranked[0][2],
                    "oracle_pose": oracle[2],
                    "rmsd_method": ranked[0][3],
                }
            )
        else:
            row["error"] = "missing_or_unmappable_prediction"
        rows.append(row)

    denominator = len(rows)
    top1_count = sum(float(row["top1_rmsd"]) < 2.0 for row in rows)
    oracle_available_count = sum(float(row["oracle_rmsd"]) < 2.0 for row in rows)
    oracle40_count = sum(
        int(row["evaluated_pose_count"]) >= 40 and float(row["oracle_rmsd"]) < 2.0
        for row in rows
    )
    any_pose = sum(int(row["evaluated_pose_count"]) > 0 for row in rows)
    complete = sum(int(row["evaluated_pose_count"]) >= 40 for row in rows)
    summary = {
        "schema_version": 1,
        "model": "sigmadock",
        "dataset": dataset,
        "denominator": denominator,
        "completed_seed_count": max(
            (int(row["evaluated_pose_count"]) for row in rows), default=0
        ),
        "targets_with_any_evaluated_pose": any_pose,
        "targets_with_at_least_40_evaluated_poses": complete,
        "top1_rmsd_lt2_count": top1_count,
        "top1_rmsd_lt2_pct": 100.0 * top1_count / denominator,
        "oracle_available_rmsd_lt2_count": oracle_available_count,
        "oracle_available_rmsd_lt2_pct": 100.0 * oracle_available_count / denominator,
        "oracle40_rmsd_lt2_count": oracle40_count,
        "oracle40_rmsd_lt2_pct": 100.0 * oracle40_count / denominator,
        "failed_target_count": denominator - any_pose,
        "rmsd_definition": "RDKit symmetry-aware heavy-atom CalcRMS without alignment; charge-agnostic full-topology retry only",
        "top1_selector": "minimum native Vinardo Affinity across completed independent seeds",
        "oracle40_policy": "targets with fewer than 40 evaluated poses count as failures",
    }
    return rows, summary


def main() -> None:
    args = parse_args()
    # Parsing failures are retained per pose in the CSV; repeated RDKit stderr
    # diagnostics otherwise dominate large N40 evaluation logs.
    RDLogger.DisableLog("rdApp.*")
    rows, summary = evaluate(
        args.model,
        args.dataset,
        args.limit,
        args.official_seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / f"{args.model}__{args.dataset}.csv"
    json_path = args.output_dir / f"{args.model}__{args.dataset}.json"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
