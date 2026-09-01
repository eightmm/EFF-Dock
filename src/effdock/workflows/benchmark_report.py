"""Aggregate completed EFF-Dock benchmark shards into one machine ledger."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import torch
from rdkit.Chem import rdMolDescriptors

from effdock.evaluation.benchmark import load_ligand
from effdock.preprocess.fragments import decompose_fragments
from effdock.preprocess.ligand import featurize_ligand
from effdock.preprocess.protein import (
    AA3_TO_IDX,
    METAL_ELEMENTS,
    NUCLEIC_ACID_RESIDUES,
    PTM_MAPPING,
    WATER_RESIDUES,
)
from effdock.workflows.evaluate import summarize_rows

EXPECTED_COUNTS = {"astex": 85, "posebusters": 308}
CONSISTENCY_KEYS = (
    "checkpoint_sha256",
    "confidence_checkpoint_sha256",
    "config_sha256",
    "pocket_centers_sha256",
    "eligibility_manifest_sha256",
    "num_samples",
    "num_steps",
    "sigma",
    "time_schedule",
    "schedule_power",
    "pocket_cutoff",
    "center_jitter_sigma",
    "prior_pool_size",
    "unified_guidance_scale",
    "unified_guidance_start_t",
    "unified_guidance_ramp_power",
    "unified_guidance_max_force",
    "unified_guidance_max_velocity",
    "unified_guidance_max_angular_velocity",
    "unified_guidance_max_atom_displacement",
    "unified_guidance_max_backtracks",
    "unified_guidance_protein_shell",
    "refine",
    "seed",
)


def code_tree_sha256(root: Path = Path.cwd()) -> str:
    paths = [root / "pyproject.toml", root / "uv.lock"]
    for directory in (root / "src", root / "configs", root / "scripts" / "slurm"):
        paths.extend(
            path
            for path in directory.rglob("*")
            if path.is_file()
            and "__pycache__" not in path.parts
            and path.suffix in {".py", ".yaml", ".yml", ".sbatch", ".sh"}
        )
    digest = hashlib.sha256()
    for path in sorted(paths):
        relative = path.relative_to(root).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _cofactor_class(protein_path: Path) -> str:
    has_metal = False
    has_organic = False
    with protein_path.open() as handle:
        for line in handle:
            if not line.startswith("HETATM"):
                continue
            residue = line[17:20].strip()
            element = line[76:78].strip().upper() if len(line) >= 78 else ""
            if residue in WATER_RESIDUES or residue in NUCLEIC_ACID_RESIDUES:
                continue
            if residue in METAL_ELEMENTS or element in METAL_ELEMENTS:
                has_metal = True
                continue
            if PTM_MAPPING.get(residue, residue) not in AA3_TO_IDX:
                has_organic = True
    if has_metal and has_organic:
        return "organic_and_metal"
    if has_organic:
        return "organic"
    if has_metal:
        return "metal"
    return "none"


def enrich_rows(rows: list[dict]) -> None:
    for row in rows:
        ligand_path = Path(row["ligand_ref"])
        ligand = load_ligand(ligand_path, ligand_path.suffix.lower().lstrip("."))
        ligand_data = featurize_ligand(ligand)
        if ligand_data is None:
            raise ValueError(f"failed to featurize benchmark ligand: {ligand_path}")
        coords = torch.tensor(ligand.GetConformer().GetPositions(), dtype=torch.float32)
        fragment_data = decompose_fragments(ligand, coords)
        if fragment_data is None:
            raise ValueError(f"failed to fragment benchmark ligand: {ligand_path}")
        row["heavy_atoms"] = ligand.GetNumHeavyAtoms()
        row["num_fragments"] = int(fragment_data["n_frags"])
        row["rotatable_bonds"] = int(rdMolDescriptors.CalcNumRotatableBonds(ligand))
        row["cofactor_class"] = _cofactor_class(Path(row["protein"]))


def _bin(value: int, thresholds: tuple[int, int], labels: tuple[str, str, str]) -> str:
    if value <= thresholds[0]:
        return labels[0]
    if value <= thresholds[1]:
        return labels[1]
    return labels[2]


def _selectors(rows: list[dict]) -> list[str]:
    return [
        selector
        for selector in (
            "first",
            "vina",
            "confidence",
            "confidence_filter",
            "confidence_final",
            "oracle",
        )
        if rows and f"{selector}_rmsd" in rows[0]
    ]


def compute_slices(rows: list[dict]) -> dict:
    definitions = {
        "heavy_atoms": lambda row: _bin(
            int(row["heavy_atoms"]), (20, 40), ("le20", "21to40", "gt40")
        ),
        "num_fragments": lambda row: _bin(int(row["num_fragments"]), (1, 3), ("1", "2to3", "ge4")),
        "rotatable_bonds": lambda row: _bin(
            int(row["rotatable_bonds"]), (3, 7), ("le3", "4to7", "ge8")
        ),
        "cofactor_class": lambda row: str(row["cofactor_class"]),
    }
    result: dict[str, dict] = {}
    for dimension, group_fn in definitions.items():
        groups: dict[str, list[dict]] = {}
        for row in rows:
            groups.setdefault(group_fn(row), []).append(row)
        result[dimension] = {
            group: {
                "n": len(group_rows),
                **{
                    f"{selector}_lt2_pct": sum(
                        float(row[f"{selector}_rmsd"]) < 2.0 for row in group_rows
                    )
                    / len(group_rows)
                    * 100
                    for selector in _selectors(rows)
                },
            }
            for group, group_rows in sorted(groups.items())
        }
    return result


def aggregate_dataset(
    input_dir: Path, run_name: str, expected_count: int
) -> tuple[list[dict], dict]:
    summaries = sorted(input_dir.glob(f"{run_name}*.summary.json"))
    if not summaries:
        raise FileNotFoundError(f"no summaries for {run_name} in {input_dir}")
    metadata = [json.loads(path.read_text()) for path in summaries]
    reference = metadata[0]
    for current in metadata[1:]:
        mismatches = [key for key in CONSISTENCY_KEYS if current.get(key) != reference.get(key)]
        if mismatches:
            raise ValueError(f"inconsistent {run_name} shards: {mismatches}")

    rows_by_id: dict[str, dict] = {}
    for shard in metadata:
        csv_path = Path(shard["csv"])
        if not csv_path.is_absolute():
            csv_path = Path.cwd() / csv_path
        with csv_path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                complex_id = row["id"]
                if complex_id in rows_by_id:
                    raise ValueError(f"duplicate {run_name} row: {complex_id}")
                converted = dict(row)
                for selector in _selectors([converted]):
                    converted[f"{selector}_rmsd"] = float(converted[f"{selector}_rmsd"])
                    converted[f"{selector}_fast_valid"] = (
                        converted[f"{selector}_fast_valid"].lower() == "true"
                    )
                rows_by_id[complex_id] = converted

    rows = [rows_by_id[key] for key in sorted(rows_by_id)]
    enrich_rows(rows)
    all_failures = {
        failure["id"]: failure for shard in metadata for failure in shard.get("failures", [])
    }
    rescued_failures = [all_failures[key] for key in sorted(all_failures) if key in rows_by_id]
    failures_by_id = {
        key: failure for key, failure in all_failures.items() if key not in rows_by_id
    }
    failures = [failures_by_id[key] for key in sorted(failures_by_id)]
    if len(rows) + len(failures) != expected_count:
        raise ValueError(
            f"{run_name}: expected {expected_count} IDs, found {len(rows)} successes "
            f"and {len(failures)} failures"
        )
    aggregate = {
        "expected": expected_count,
        "success": len(rows),
        "failed": len(failures),
        "failure_pct": len(failures) / expected_count * 100,
        "stats": summarize_rows(rows),
        "slices": compute_slices(rows),
        "failures": failures,
        "rescued_failures": rescued_failures,
        "shard_summaries": [str(path) for path in summaries],
        "runtime": [shard.get("runtime", {}) for shard in metadata],
        "checkpoint_sha256": reference["checkpoint_sha256"],
        "confidence_checkpoint_sha256": reference.get("confidence_checkpoint_sha256"),
        "config_sha256": reference["config_sha256"],
        "pocket_centers_sha256": reference["pocket_centers_sha256"],
        "num_samples": reference["num_samples"],
        "num_steps": reference["num_steps"],
        "sigma": reference["sigma"],
        "pocket_cutoff": reference["pocket_cutoff"],
        "seed": reference["seed"],
    }
    return rows, aggregate


def aggregate_official_posebusters(input_dir: Path, expected_count: int) -> dict:
    summaries = sorted(input_dir.glob("shard-*.summary.json"))
    if not summaries:
        raise FileNotFoundError(f"no official PoseBusters summaries in {input_dir}")
    rows: dict[str, dict[str, bool]] = {}
    failures: list[dict] = []
    for summary_path in summaries:
        summary = json.loads(summary_path.read_text())
        failures.extend(summary.get("failures", []))
        if summary.get("csv"):
            with Path(summary["csv"]).open(newline="") as handle:
                for row in csv.DictReader(handle):
                    complex_id = row.pop("id")
                    if complex_id in rows:
                        raise ValueError(f"duplicate official PoseBusters row: {complex_id}")
                    rows[complex_id] = {key: value.lower() == "true" for key, value in row.items()}
    if len(rows) + len(failures) != expected_count:
        raise ValueError(
            f"official PoseBusters: expected {expected_count}, found {len(rows)} successes "
            f"and {len(failures)} failures"
        )
    check_names = sorted(next(iter(rows.values()))) if rows else []
    check_pass_pct = {
        check: sum(result[check] for result in rows.values()) / len(rows) * 100
        for check in check_names
    }
    return {
        "version": "0.6.5",
        "config": "redock",
        "success": len(rows),
        "failed": len(failures),
        "posebusters_valid_pct": check_pass_pct.get("posebusters_valid"),
        "check_pass_pct": check_pass_pct,
        "failures": failures,
        "shard_summaries": [str(path) for path in summaries],
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("outputs/benchmarks/raw"))
    parser.add_argument("--output", type=Path, default=Path("outputs/benchmarks/summary.json"))
    parser.add_argument("--run-prefix", default="effdock-redock-ema-n40-s25-v1")
    parser.add_argument("--protocol-id", default="EFFDOCK-REDOCK-EMA-N40-S25-V1")
    parser.add_argument(
        "--posebusters-official-dir",
        type=Path,
        default=Path("outputs/benchmarks/posebusters_official"),
    )
    args = parser.parse_args(argv)

    result = {
        "protocol_id": args.protocol_id,
        "scope": "reference-defined oracle-pocket redocking diagnostic",
        "created_at": datetime.now(UTC).isoformat(),
        "code_tree_sha256": code_tree_sha256(),
        "datasets": {},
    }
    combined_dir = args.output.parent / "combined"
    combined_dir.mkdir(parents=True, exist_ok=True)
    for dataset, expected in EXPECTED_COUNTS.items():
        run_name = f"{args.run_prefix}-{dataset}"
        rows, aggregate = aggregate_dataset(args.input_dir, run_name, expected)
        combined_path = combined_dir / f"{dataset}.csv"
        with combined_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        aggregate["rows_csv"] = str(combined_path)
        result["datasets"][dataset] = aggregate

    if args.posebusters_official_dir.exists():
        result["datasets"]["posebusters"]["official_posebusters"] = aggregate_official_posebusters(
            args.posebusters_official_dir, EXPECTED_COUNTS["posebusters"]
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    posebusters_num_samples = result["datasets"]["posebusters"]["num_samples"]
    ledger_metrics = {
        "posebusters_vina_top1_lt2_pct": result["datasets"]["posebusters"]["stats"]["vina"][
            "pct_lt_2A"
        ],
        "posebusters_first_lt2_pct": result["datasets"]["posebusters"]["stats"]["first"][
            "pct_lt_2A"
        ],
        f"posebusters_oracle_top{posebusters_num_samples}_lt2_pct": result["datasets"][
            "posebusters"
        ]["stats"]["oracle"]["pct_lt_2A"],
        "posebusters_official_valid_pct": result["datasets"]["posebusters"]
        .get("official_posebusters", {})
        .get("posebusters_valid_pct"),
        "astex_vina_top1_lt2_pct": result["datasets"]["astex"]["stats"]["vina"]["pct_lt_2A"],
    }
    pb_stats = result["datasets"]["posebusters"]["stats"]
    if "confidence_final" in pb_stats:
        ledger_metrics.update(
            {
                "posebusters_confidence_top1_lt2_pct": pb_stats["confidence"]["pct_lt_2A"],
                "posebusters_confidence_final_top1_lt2_pct": pb_stats["confidence_final"][
                    "pct_lt_2A"
                ],
                "astex_confidence_final_top1_lt2_pct": result["datasets"]["astex"]["stats"][
                    "confidence_final"
                ]["pct_lt_2A"],
            }
        )
    ledger_path = args.output.parent / "ledger_metrics.json"
    ledger_path.write_text(json.dumps(ledger_metrics, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
