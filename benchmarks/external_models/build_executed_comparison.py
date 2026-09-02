#!/usr/bin/env python3
"""Freeze completed three-repeat supplied-pocket RMSD results for plotting."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVALUATION_ROOT = (
    ROOT / "outputs/external_models/evaluation/official_repeat_rmsd_20260901"
)
DEFAULT_OUTPUT = ROOT / "benchmarks/results/external_models/pocket_only_executed_reruns.json"

DATASETS = {
    "astex": ("astex_diverse", 85),
    "posebusters": ("posebusters_benchmark", 308),
}
MODELS = {
    "diffdock_pocket": ("DiffDock-Pocket", "learned"),
    "rldiff_rlpp": ("RLDiff RL++", "hybrid"),
    "diffbindfr": ("DiffBindFR + MDN/EC", "hybrid"),
    "posebench_vina": ("Vina", "classical"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-root", type=Path, default=DEFAULT_EVALUATION_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT.resolve()))


def aggregate(values: list[float]) -> dict[str, object]:
    if len(values) != 3:
        raise ValueError(f"exactly three repeats are required, found {len(values)}")
    return {
        "values": values,
        "mean": statistics.mean(values),
        "std": statistics.stdev(values),
        "std_definition": "sample standard deviation (ddof=1)",
    }


def load_common_repeats(
    root: Path,
    model: str,
    dataset: str,
    denominator: int,
) -> tuple[list[dict], list[str]]:
    summaries: list[dict] = []
    paths: list[str] = []
    for repeat in range(3):
        path = root / model / dataset / f"repeat_{repeat}" / f"{model}__{dataset}.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        summary = json.loads(path.read_text())
        if int(summary["denominator"]) != denominator:
            raise ValueError(f"denominator mismatch in {path}")
        if summary.get("model") != model or summary.get("dataset") != dataset:
            raise ValueError(f"identity mismatch in {path}")
        summaries.append(summary)
        paths.append(relative(path))
    return summaries, paths


def load_sigmadock_repeats(
    dataset_key: str,
    dataset: str,
    denominator: int,
) -> tuple[list[dict], list[str]]:
    tag = "sigmadock_official_r4_20260830" if dataset_key == "astex" else "sigmadock_official_r6_20260831"
    base = ROOT / "outputs/external_models/evaluation" / tag / dataset
    summaries: list[dict] = []
    paths: list[str] = []
    for repeat in range(3):
        path = base / f"repeat_{repeat}" / "summary.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        summary = json.loads(path.read_text())
        if int(summary["denominator"]) != denominator:
            raise ValueError(f"denominator mismatch in {path}")
        summaries.append(summary)
        paths.append(relative(path))
    return summaries, paths


def metric(summaries: list[dict], key: str) -> dict[str, object]:
    return aggregate([float(summary[key]) for summary in summaries])


def main() -> None:
    args = parse_args()
    payload: dict[str, object] = {
        "schema_version": 1,
        "comparison_scope": "supplied_pocket_only",
        "provenance": "locally executed official/repository-default three-repeat reruns",
        "metric": "symmetry-aware heavy-atom RMSD without alignment; full frozen denominator",
        "datasets": {},
        "pending_not_plotted": [
            "SurfDock official three-repeat campaign",
            "Interformer official three-repeat campaign",
        ],
    }
    datasets: dict[str, object] = {}
    for dataset_key, (dataset, denominator) in DATASETS.items():
        methods: list[dict[str, object]] = []
        sigma_summaries, sigma_paths = load_sigmadock_repeats(
            dataset_key, dataset, denominator
        )
        methods.append(
            {
                "method": "SigmaDock",
                "run_label": "SigmaDock (our rerun)",
                "family": "hybrid",
                "repeat_count": 3,
                "top1": metric(sigma_summaries, "top1_rmsd_lt2_pct"),
                "oracle": metric(sigma_summaries, "oracle_rmsd_lt2_pct"),
                "joint": metric(sigma_summaries, "top1_joint_pct"),
                "coverage": {
                    "pool_size": [int(summary["pool_size"]) for summary in sigma_summaries]
                },
                "source_summaries": sigma_paths,
            }
        )
        for model, (method, family) in MODELS.items():
            summaries, paths = load_common_repeats(
                args.evaluation_root, model, dataset, denominator
            )
            methods.append(
                {
                    "method": method,
                    "run_label": f"{method} (our rerun)",
                    "family": family,
                    "repeat_count": 3,
                    "top1": metric(summaries, "top1_rmsd_lt2_pct"),
                    "oracle": metric(summaries, "oracle_available_rmsd_lt2_pct"),
                    "coverage": {
                        "targets_with_any_evaluated_pose": [
                            int(summary["targets_with_any_evaluated_pose"])
                            for summary in summaries
                        ],
                        "targets_with_at_least_40_evaluated_poses": [
                            int(summary["targets_with_at_least_40_evaluated_poses"])
                            for summary in summaries
                        ],
                        "pose_error_count": [
                            int(summary["pose_error_count"]) for summary in summaries
                        ],
                    },
                    "source_summaries": paths,
                }
            )
        datasets[dataset_key] = {
            "name": "Astex Diverse" if dataset_key == "astex" else "PoseBusters v2",
            "n": denominator,
            "methods": methods,
        }
    payload["datasets"] = datasets
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
