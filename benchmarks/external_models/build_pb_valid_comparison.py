#!/usr/bin/env python3
"""Build the clean paper-classical/local-AI PB-valid comparison ledger."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PB_ROOT = (
    ROOT / "outputs/external_models/evaluation/official_selected_posebusters_20260901"
)
DEFAULT_CLASSICAL = (
    ROOT / "docs/results/external_models/posebusters_classical_paper_values.json"
)
DEFAULT_EXECUTED = (
    ROOT / "docs/results/external_models/pocket_only_executed_reruns.json"
)
DEFAULT_EFFDOCK = (
    ROOT / "docs/results/external_models/effdock_u70k_benchmark.json"
)
DEFAULT_OUTPUT = (
    ROOT / "docs/results/external_models/pocket_only_pb_valid_comparison.json"
)
DATASETS = {
    "astex": ("astex_diverse", 85),
    "posebusters": ("posebusters_benchmark", 308),
}
MODELS = {
    "diffdock_pocket": ("DiffDock-Pocket", "learning"),
    "rldiff_rlpp": ("RLDiff RL++", "hybrid"),
    "diffbindfr": ("DiffBindFR + MDN/EC", "hybrid"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--posebusters-root", type=Path, default=DEFAULT_PB_ROOT)
    parser.add_argument("--classical", type=Path, default=DEFAULT_CLASSICAL)
    parser.add_argument("--executed", type=Path, default=DEFAULT_EXECUTED)
    parser.add_argument("--effdock", type=Path, default=DEFAULT_EFFDOCK)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def aggregate(values: list[float]) -> dict[str, object]:
    if len(values) != 3:
        raise ValueError(f"expected three repeats, found {len(values)}")
    return {
        "values": values,
        "mean": statistics.mean(values),
        "std": statistics.stdev(values),
        "std_definition": "sample standard deviation (ddof=1)",
    }


def load_pb_repeats(
    root: Path, model: str, dataset: str, denominator: int
) -> tuple[list[float], list[float], list[int], list[str]]:
    rmsd_values: list[float] = []
    joint_values: list[float] = []
    errors: list[int] = []
    source_dirs: list[str] = []
    for repeat in range(3):
        repeat_dir = root / model / dataset / f"repeat_{repeat}"
        shard_dirs = sorted(repeat_dir.glob("shard_*_of_*"))
        if not shard_dirs:
            raise FileNotFoundError(f"no completed shards in {repeat_dir}")
        summaries = [read_json(path / "summary.json") for path in shard_dirs]
        expected_shards = {int(summary["num_shards"]) for summary in summaries}
        if len(expected_shards) != 1 or len(shard_dirs) != next(iter(expected_shards)):
            raise ValueError(f"incomplete shard inventory: {repeat_dir}")
        rows: list[dict[str, str]] = []
        for shard_dir in shard_dirs:
            with (shard_dir / "results.csv").open(newline="", encoding="utf-8") as handle:
                rows.extend(csv.DictReader(handle))
        ids = [row["complex_name"] for row in rows]
        if len(rows) != denominator or len(set(ids)) != denominator:
            raise ValueError(f"coverage mismatch in {repeat_dir}: {len(rows)}")
        rmsd_count = sum(row["top1_rmsd_lt2"] == "True" for row in rows)
        joint_count = sum(
            row["joint_rmsd_lt2_pb_valid"] == "True" for row in rows
        )
        if joint_count > rmsd_count:
            raise ValueError(f"joint exceeds RMSD success in {repeat_dir}")
        rmsd_values.append(100.0 * rmsd_count / denominator)
        joint_values.append(100.0 * joint_count / denominator)
        errors.append(sum(bool(row["error"]) for row in rows))
        source_dirs.append(str(repeat_dir.resolve().relative_to(ROOT.resolve())))
    return rmsd_values, joint_values, errors, source_dirs


def main() -> None:
    args = parse_args()
    classical = read_json(args.classical)
    executed = read_json(args.executed)
    effdock = read_json(args.effdock)
    output_datasets: dict[str, object] = {}
    for dataset_key, (dataset_name, denominator) in DATASETS.items():
        methods: list[dict[str, object]] = []
        eff = effdock["datasets"][dataset_key]
        methods.append(
            {
                "method": "EFF-Dock",
                "family": "ours",
                "source_type": "our_run",
                "repeat_count": 1,
                "top1_rmsd_lt2": {"mean": float(eff["refined_top1_lt2"]["pct"]), "std": 0.0},
                "top1_joint_rmsd_lt2_pb_valid": {
                    "mean": float(eff["refined_joint_lt2_pb_valid"]["pct"]),
                    "std": 0.0,
                },
                "source": effdock["source_report"],
            }
        )

        executed_rows = {
            row["method"]: row for row in executed["datasets"][dataset_key]["methods"]
        }
        sigma = executed_rows["SigmaDock"]
        methods.append(
            {
                "method": "SigmaDock",
                "family": "hybrid",
                "source_type": "our_run",
                "repeat_count": 3,
                "top1_rmsd_lt2": sigma["top1"],
                "top1_joint_rmsd_lt2_pb_valid": sigma["joint"],
                "source": sigma["source_summaries"],
            }
        )
        for model, (display, family) in MODELS.items():
            rmsd, joint, errors, sources = load_pb_repeats(
                args.posebusters_root, model, dataset_name, denominator
            )
            common = executed_rows[display]["top1"]
            if any(abs(a - b) > 1e-9 for a, b in zip(rmsd, common["values"], strict=True)):
                raise ValueError(f"PoseBusters inventory changed RMSD results for {display}")
            methods.append(
                {
                    "method": display,
                    "family": family,
                    "source_type": "our_run",
                    "repeat_count": 3,
                    "top1_rmsd_lt2": aggregate(rmsd),
                    "top1_joint_rmsd_lt2_pb_valid": aggregate(joint),
                    "posebusters_error_count": errors,
                    "source": sources,
                }
            )

        for row in classical["datasets"][dataset_key]["methods"]:
            methods.append(
                {
                    "method": row["method"],
                    "family": "classical",
                    "source_type": "paper",
                    "repeat_count": 1,
                    "top1_rmsd_lt2": {
                        "mean": float(row["top1_rmsd_lt2"]["pct"]),
                        "std": 0.0,
                    },
                    "top1_joint_rmsd_lt2_pb_valid": {
                        "mean": float(row["top1_joint_rmsd_lt2_pb_valid"]["pct"]),
                        "std": 0.0,
                    },
                    "source": classical["provenance"],
                }
            )
        output_datasets[dataset_key] = {
            "name": classical["datasets"][dataset_key]["name"],
            "n": denominator,
            "methods": methods,
        }

    payload = {
        "schema_version": 1,
        "comparison_scope": "supplied_pocket_only",
        "metric_contract": (
            "One Top-1 pose per target. Hatched total is symmetry-aware heavy-atom "
            "RMSD <2 A; solid subset additionally passes all official PoseBusters "
            "redock validity checks. Missing/error targets remain denominator failures."
        ),
        "source_policy": (
            "Classical GOLD/Vina use deposited PoseBusters paper values without "
            "minimization; AI/hybrid models use our three-repeat executions; "
            "EFF-Dock uses the promoted U70k refined result."
        ),
        "datasets": output_datasets,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
