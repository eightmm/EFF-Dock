#!/usr/bin/env python3
"""Render the current EFF-Dock benchmark and pocket-cutoff overview."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CUTOFF = (
    ROOT
    / "outputs/benchmarks/effdock_pocket_cutoff_robustness_runs"
    / "cutoff-r3-production-20260901-r2/report.json"
)
DEFAULT_CORE = ROOT / "benchmarks/results/external_models/effdock_u70k_benchmark.json"
DEFAULT_TEMPORAL = ROOT / "benchmarks/results/external_models/temporal_literature.json"
DEFAULT_EXTERNAL = (
    ROOT / "benchmarks/results/external_models/pocket_only_pb_valid_comparison.json"
)
DEFAULT_OPENBIND = (
    ROOT
    / "outputs/benchmarks/s50_raw_refined_confidence_temporal_external_runs"
    / "d97d5eb907acc485dfde4b7fcf88d87b4d5fd8576014d2cfb89dd0518b9c9bb4"
    / "report/summary.json"
)
DEFAULT_OUTPUT_DIR = ROOT / "docs/results/current_overview"

INK = "#29384D"
MUTED = "#66758A"
EDGE = "#718096"
GRID = "#E5EAF0"
COLORS = {
    "raw": "#C8D7EA",
    "top1": "#7F9FCB",
    "joint": "#6FAFA8",
    "valid": "#A8D5BA",
    "oracle": "#E7A77E",
    "dl": "#8FA9D0",
    "classical": "#AED8C7",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cutoff-report", type=Path, default=DEFAULT_CUTOFF)
    parser.add_argument("--core-report", type=Path, default=DEFAULT_CORE)
    parser.add_argument("--temporal-report", type=Path, default=DEFAULT_TEMPORAL)
    parser.add_argument("--openbind-report", type=Path, default=DEFAULT_OPENBIND)
    parser.add_argument("--external-report", type=Path, default=DEFAULT_EXTERNAL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def save_figure(fig: plt.Figure, base: Path, dpi: int) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".png"), dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def clean_axis(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.9, zorder=0)
    ax.tick_params(colors=MUTED, labelsize=9)


def plot_cutoff(payload: dict, output_dir: Path, dpi: int) -> None:
    if payload.get("status") != "complete":
        raise ValueError("pocket-cutoff report is not complete")
    if payload.get("protocol_id") != "EFFDOCK-POCKET-CUTOFF-ROBUSTNESS-V1":
        raise ValueError("unexpected pocket-cutoff protocol")

    specs = (
        ("selected_rmsd_lt2", "Top-1 RMSD < 2 Å", COLORS["top1"], "o", "-"),
        (
            "joint_rmsd_lt2_pb_valid",
            "Top-1 RMSD < 2 Å & PB-valid",
            COLORS["joint"],
            "s",
            "-",
        ),
        ("posebusters_valid", "PB-valid", COLORS["valid"], "D", "--"),
        ("oracle_rmsd_lt2", "Oracle-100 RMSD < 2 Å", COLORS["oracle"], "^", "--"),
    )
    rows: list[dict] = []
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.9), sharey=True)

    for ax, dataset_key, letter in zip(
        axes, ("astex", "posebusters"), ("A", "B"), strict=True
    ):
        dataset = payload["datasets"][dataset_key]
        cutoffs = np.asarray(
            [item["pocket_cutoff_angstrom"] for item in dataset["cutoffs"]], dtype=float
        )
        if list(cutoffs) != [6.0, 8.0, 10.0, 12.0]:
            raise ValueError(f"unexpected cutoffs for {dataset_key}: {cutoffs}")
        ax.axvspan(9.72, 10.28, color="#EEF2F7", zorder=0)
        for metric, label, color, marker, linestyle in specs:
            mean = np.asarray(
                [item["aggregate"][metric]["mean"] for item in dataset["cutoffs"]]
            )
            std = np.asarray(
                [item["aggregate"][metric]["std"] for item in dataset["cutoffs"]]
            )
            if np.any((mean < 0) | (mean > 100)):
                raise ValueError(f"out-of-range percentage in {dataset_key}/{metric}")
            ax.errorbar(
                cutoffs,
                mean,
                yerr=std,
                label=label,
                color=color,
                marker=marker,
                markersize=5.8,
                markeredgecolor="white",
                markeredgewidth=0.8,
                linewidth=2.0,
                linestyle=linestyle,
                capsize=3.0,
                capthick=1.0,
                elinewidth=1.0,
                zorder=4,
            )
            for cutoff, value, error in zip(cutoffs, mean, std, strict=True):
                rows.append(
                    {
                        "dataset": dataset_key,
                        "n": dataset["n"],
                        "pocket_cutoff_A": int(cutoff),
                        "metric": metric,
                        "mean_pct": f"{value:.8f}",
                        "sample_std_pct": f"{error:.8f}",
                        "repeats": 3,
                    }
                )

        top1 = [
            item["aggregate"]["selected_rmsd_lt2"]["mean"]
            for item in dataset["cutoffs"]
        ]
        joint = [
            item["aggregate"]["joint_rmsd_lt2_pb_valid"]["mean"]
            for item in dataset["cutoffs"]
        ]
        for x, y in zip(cutoffs, top1, strict=True):
            ax.text(x, y + 1.65, f"{y:.1f}", ha="center", fontsize=8.2, color=INK)
        for x, y in zip(cutoffs, joint, strict=True):
            ax.text(x, y - 2.1, f"{y:.1f}", ha="center", va="top", fontsize=8.2, color=INK)

        ax.text(10, 99.1, "default", ha="center", va="top", fontsize=8, color=MUTED)
        ax.set_title(
            f"{letter}  {dataset['name']}  (N={dataset['n']})",
            loc="left",
            fontsize=13,
            fontweight="bold",
            color=INK,
            pad=10,
        )
        ax.set_xticks(cutoffs)
        ax.set_xlabel("Docking pocket cutoff (Å)", fontsize=10, color=INK)
        ax.set_ylim(68, 101)
        ax.set_yticks(np.arange(70, 101, 5))
        clean_axis(ax)

    axes[0].set_ylabel("Success rate (%)", fontsize=10, color=INK)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.015),
        ncol=4,
        frameon=False,
        fontsize=8.8,
        columnspacing=1.6,
        handlelength=2.5,
    )
    fig.subplots_adjust(left=0.075, right=0.99, top=0.84, bottom=0.14, wspace=0.12)
    save_figure(fig, output_dir / "figures/pocket_cutoff_robustness", dpi)
    write_csv(
        output_dir / "pocket_cutoff_robustness.csv",
        [
            "dataset",
            "n",
            "pocket_cutoff_A",
            "metric",
            "mean_pct",
            "sample_std_pct",
            "repeats",
        ],
        rows,
    )


def benchmark_rows(core: dict, temporal: dict, openbind: dict) -> list[dict]:
    rows: list[dict] = []
    for key in ("astex", "posebusters"):
        item = core["datasets"][key]
        rows.append(
            {
                "key": key,
                "label": item["name"],
                "n": item["n"],
                "raw": item["raw_top1_lt2"]["pct"],
                "refined": item["refined_top1_lt2"]["pct"],
                "joint": item["refined_joint_lt2_pb_valid"]["pct"],
                "pb_valid": item["refined_pb_valid"]["pct"],
                "scope": "core",
            }
        )

    cohorts = temporal["effdock"]["cohorts"]
    for key, label, source_key in (
        ("phibench", "PhiBench", "phibench_derived"),
        ("foldbench", "FoldBench pocket", "foldbench_pocket_558_all"),
    ):
        item = cohorts[source_key]
        rows.append(
            {
                "key": key,
                "label": label,
                "n": item["n"],
                "raw": item["raw_top1_rmsd_lt_2_pct"],
                "refined": item["refined_top1_rmsd_lt_2_pct"],
                "joint": item["joint_refined_rmsd_pb_valid_pct"],
                "pb_valid": item["pb_valid_pct"],
                "scope": "temporal",
            }
        )

    item = openbind["aggregates"]["u070000"]["openbind"]
    pct = item["percent"]
    rows.append(
        {
            "key": "openbind",
            "label": "OpenBind (aux.)",
            "n": item["n"],
            "raw": pct["raw_top1_rmsd_lt2"],
            "refined": pct["refined_top1_rmsd_lt2"],
            "joint": pct["refined_top1_joint_posebusters_valid_rmsd_lt2"],
            "pb_valid": pct["refined_top1_posebusters_valid"],
            "scope": "auxiliary",
        }
    )
    for row in rows:
        if not (0 <= row["joint"] <= row["refined"] <= 100):
            raise ValueError(f"invalid Top-1/joint relationship for {row['key']}")
    return rows


def plot_benchmark(rows: list[dict], output_dir: Path, dpi: int) -> None:
    labels = [f"{row['label']}\nN={row['n']}" for row in rows]
    x = np.arange(len(rows), dtype=float)
    width = 0.34
    raw = np.asarray([row["raw"] for row in rows])
    refined = np.asarray([row["refined"] for row in rows])
    joint = np.asarray([row["joint"] for row in rows])
    valid = np.asarray([row["pb_valid"] for row in rows])

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.9), sharey=True)
    ax = axes[0]
    bars_raw = ax.bar(
        x - width / 2,
        raw,
        width,
        color=COLORS["raw"],
        edgecolor=EDGE,
        linewidth=0.8,
        label="ODE Top-1",
        zorder=3,
    )
    bars_refined = ax.bar(
        x + width / 2,
        refined,
        width,
        color=COLORS["top1"],
        edgecolor=EDGE,
        linewidth=0.8,
        label="ODE + refinement Top-1",
        zorder=3,
    )
    for bars, values in ((bars_raw, raw), (bars_refined, refined)):
        for bar, value in zip(bars, values, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 1.0,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=7.8,
                color=INK,
            )
    ax.set_title("A  Refinement effect", loc="left", fontsize=13, fontweight="bold", color=INK)
    ax.legend(frameon=False, fontsize=8.8, loc="upper right")

    ax = axes[1]
    bars_joint = ax.bar(
        x,
        joint,
        0.62,
        color=COLORS["joint"],
        edgecolor=EDGE,
        linewidth=0.9,
        label="RMSD < 2 Å & PB-valid",
        zorder=3,
    )
    ax.bar(
        x,
        refined - joint,
        0.62,
        bottom=joint,
        color=COLORS["top1"],
        alpha=0.52,
        edgecolor=EDGE,
        linewidth=0.9,
        hatch="///",
        label="RMSD < 2 Å but PB-invalid",
        zorder=3,
    )
    ax.scatter(
        x,
        valid,
        marker="D",
        s=27,
        color=COLORS["valid"],
        edgecolor=EDGE,
        linewidth=0.7,
        label="PB-valid",
        zorder=5,
    )
    for bar, top, joint_value in zip(bars_joint, refined, joint, strict=True):
        center = bar.get_x() + bar.get_width() / 2
        ax.text(center, top + 1.0, f"{top:.1f}", ha="center", fontsize=7.8, color=INK)
        ax.text(
            center,
            max(joint_value - 1.7, 2),
            f"{joint_value:.1f}",
            ha="center",
            va="top",
            fontsize=7.8,
            fontweight="bold",
            color="white",
        )
    ax.set_title("B  Physical validity of refined Top-1", loc="left", fontsize=13, fontweight="bold", color=INK)
    ax.legend(frameon=False, fontsize=8.4, loc="upper right")

    for ax in axes:
        ax.set_xticks(x, labels)
        ax.set_ylim(0, 103)
        ax.set_yticks(np.arange(0, 101, 20))
        ax.tick_params(axis="x", length=0, pad=7, labelsize=8.2)
        clean_axis(ax)
        ax.axvline(3.5, color="#D5DCE5", linewidth=1.0, linestyle=(0, (3, 3)), zorder=1)
    axes[0].set_ylabel("Success rate (%)", fontsize=10, color=INK)
    fig.subplots_adjust(left=0.065, right=0.99, top=0.91, bottom=0.18, wspace=0.12)
    save_figure(fig, output_dir / "figures/effdock_benchmark_summary", dpi)

    write_csv(
        output_dir / "effdock_benchmark_summary.csv",
        ["dataset", "n", "scope", "raw_top1_pct", "refined_top1_pct", "joint_pct", "pb_valid_pct"],
        [
            {
                "dataset": row["key"],
                "n": row["n"],
                "scope": row["scope"],
                "raw_top1_pct": f"{row['raw']:.8f}",
                "refined_top1_pct": f"{row['refined']:.8f}",
                "joint_pct": f"{row['joint']:.8f}",
                "pb_valid_pct": f"{row['pb_valid']:.8f}",
            }
            for row in rows
        ],
    )


def plot_external(payload: dict, output_dir: Path, dpi: int) -> None:
    if payload.get("comparison_scope") != "supplied_pocket_only":
        raise ValueError("external comparison must be supplied-pocket-only")
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 6.2), sharex=True)
    legend_handles = [
        Patch(facecolor=COLORS["dl"], edgecolor=EDGE, label="DL"),
        Patch(facecolor=COLORS["classical"], edgecolor=EDGE, label="Classical"),
        Patch(facecolor="#A7BBD2", edgecolor=EDGE, label="RMSD < 2 Å & PB-valid"),
        Patch(
            facecolor="#A7BBD2",
            edgecolor=EDGE,
            alpha=0.5,
            hatch="///",
            label="RMSD < 2 Å",
        ),
    ]
    csv_rows: list[dict] = []
    for ax, key, letter in zip(axes, ("astex", "posebusters"), ("A", "B"), strict=True):
        dataset = payload["datasets"][key]
        # Keep the production EFF-Dock condition first, rank learned docking
        # methods by Top-1, then retain classical docking as a bottom block.
        methods = [method for method in dataset["methods"] if method["method"] != "SurfDock"]
        effdock = [method for method in methods if method["method"] == "EFF-Dock"]
        if len(effdock) != 1:
            raise ValueError(f"expected exactly one EFF-Dock row for {key}")
        rank_key = lambda method: (
            -float(method["top1_rmsd_lt2"]["mean"]),
            0 if method["method"] == "SigmaDock" else 1,
            method["method"],
        )
        learned = sorted(
            (
                method
                for method in methods
                if method["method"] != "EFF-Dock" and method["family"] != "classical"
            ),
            key=rank_key,
        )
        classical = sorted(
            (method for method in methods if method["family"] == "classical"),
            key=rank_key,
        )
        methods = effdock + learned + classical
        if any(method["source_type"] == "our_run" and method["repeat_count"] != 3 for method in methods):
            raise ValueError(f"every locally executed method must have three repeats for {key}")
        y = np.arange(len(methods))[::-1]
        total = np.asarray([m["top1_rmsd_lt2"]["mean"] for m in methods])
        joint = np.asarray([m["top1_joint_rmsd_lt2_pb_valid"]["mean"] for m in methods])
        total_std = np.asarray([m["top1_rmsd_lt2"].get("std", 0.0) for m in methods])
        colors = [
            COLORS["classical"] if m["family"] == "classical" else COLORS["dl"]
            for m in methods
        ]
        if np.any(joint > total + 1e-10):
            raise ValueError(f"joint exceeds Top-1 for {key}")

        ax.barh(y, joint, height=0.65, color=colors, edgecolor=EDGE, linewidth=0.8, zorder=3)
        ax.barh(
            y,
            total - joint,
            left=joint,
            height=0.65,
            color=colors,
            alpha=0.5,
            edgecolor=EDGE,
            linewidth=0.8,
            hatch="///",
            zorder=3,
        )
        repeated = np.asarray([m["repeat_count"] == 3 for m in methods])
        ax.errorbar(
            total[repeated],
            y[repeated],
            xerr=total_std[repeated],
            fmt="none",
            ecolor="#526276",
            capsize=2.4,
            elinewidth=0.9,
            zorder=6,
        )
        for index, (yy, top, valid_joint) in enumerate(
            zip(y, total, joint, strict=True)
        ):
            label_x = top + (total_std[index] if repeated[index] else 0.0) + 1.0
            ax.text(label_x, yy, f"{top:.1f}", va="center", fontsize=8, color=INK)
            if valid_joint > 19:
                ax.text(
                    valid_joint - 1.0,
                    yy,
                    f"{valid_joint:.1f}",
                    ha="right",
                    va="center",
                    fontsize=7.7,
                    fontweight="bold",
                    color="white",
                )
        labels = [
            "SurfDock" if method["method"] == "SurfDock + force optimization" else method["method"]
            for method in methods
        ]
        ax.set_yticks(y, labels)
        ax.set_xlim(0, 101)
        ax.set_xticks(np.arange(0, 101, 20))
        ax.set_xlabel("Top-1 success rate (%)", fontsize=10, color=INK)
        ax.set_title(
            f"{letter}  {dataset['name']}  (N={dataset['n']})",
            loc="left",
            fontsize=13,
            fontweight="bold",
            color=INK,
            pad=10,
        )
        ax.grid(axis="x", color=GRID, linewidth=0.9, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=8.8)

        for method in methods:
            csv_rows.append(
                {
                    "dataset": key,
                    "n": dataset["n"],
                    "method": method["method"],
                    "family": method["family"],
                    "source_type": method["source_type"],
                    "repeats": method["repeat_count"],
                    "top1_mean_pct": f"{method['top1_rmsd_lt2']['mean']:.8f}",
                    "top1_sample_std_pct": f"{method['top1_rmsd_lt2'].get('std', 0.0):.8f}",
                    "joint_mean_pct": f"{method['top1_joint_rmsd_lt2_pb_valid']['mean']:.8f}",
                    "joint_sample_std_pct": f"{method['top1_joint_rmsd_lt2_pb_valid'].get('std', 0.0):.8f}",
                }
            )

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=4,
        frameon=False,
        fontsize=9,
    )
    fig.subplots_adjust(left=0.13, right=0.985, top=0.88, bottom=0.12, wspace=0.30)
    save_figure(fig, output_dir / "figures/pocket_model_comparison", dpi)
    write_csv(
        output_dir / "pocket_model_comparison.csv",
        [
            "dataset",
            "n",
            "method",
            "family",
            "source_type",
            "repeats",
            "top1_mean_pct",
            "top1_sample_std_pct",
            "joint_mean_pct",
            "joint_sample_std_pct",
        ],
        csv_rows,
    )


def main() -> None:
    args = parse_args()
    cutoff = read_json(args.cutoff_report)
    core = read_json(args.core_report)
    temporal = read_json(args.temporal_report)
    openbind = read_json(args.openbind_report)
    external = read_json(args.external_report)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    plot_cutoff(cutoff, args.output_dir, args.dpi)
    plot_benchmark(benchmark_rows(core, temporal, openbind), args.output_dir, args.dpi)
    plot_external(external, args.output_dir, args.dpi)
    print(args.output_dir)


if __name__ == "__main__":
    main()
