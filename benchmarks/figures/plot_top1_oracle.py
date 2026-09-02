#!/usr/bin/env python3
"""Create a pocket-only comparison using the promoted EFF-Dock U70k result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BENCHMARK = ROOT / "benchmarks/results/external_models/effdock_u70k_benchmark.json"
DEFAULT_LITERATURE = ROOT / "docs/LITERATURE_RMSD_COMPARISON.json"
DEFAULT_EXECUTED = (
    ROOT / "benchmarks/results/external_models/pocket_only_executed_reruns.json"
)
DEFAULT_OUTPUT = ROOT / "outputs/figures/effdock_u70k_joint_oracle_comparison.png"

LEARNED = "#B8CCE4"
CLASSICAL = "#B9DDCF"
EFFDOCK = "#7E9BCB"
EXECUTED_LEARNED = "#84A9D8"
EXECUTED_HYBRID = "#A995CC"
EXECUTED_CLASSICAL = "#80BEA8"
ORACLE = "#F5C3A6"
BAR_EDGE = "#7A8797"
ORACLE_EDGE = "#C98565"
GRID = "#E4E9EF"
INK = "#263447"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--literature", type=Path, default=DEFAULT_LITERATURE)
    parser.add_argument("--executed", type=Path, default=DEFAULT_EXECUTED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dpi", type=int, default=240)
    return parser.parse_args()


def load_effdock(benchmark_path: Path) -> dict[str, dict[str, float | int]]:
    with benchmark_path.open() as handle:
        summary = json.load(handle)

    if summary.get("comparison_scope") != "supplied_pocket_only":
        raise ValueError("EFF-Dock benchmark must be supplied-pocket-only")

    result: dict[str, dict[str, float | int]] = {}
    for key, expected_n in (("astex", 85), ("posebusters", 308)):
        dataset = summary["datasets"][key]
        if dataset["n"] != expected_n:
            raise ValueError(f"Unexpected {key} denominator: {dataset['n']}")
        result[key] = {
            "n": expected_n,
            "top1": float(dataset["refined_joint_lt2_pb_valid"]["pct"]),
            "oracle": float(dataset["refined_oracle_lt2"]["pct"]),
        }
    return result


def load_literature(path: Path) -> dict:
    with path.open() as handle:
        literature = json.load(handle)
    if literature.get("comparison_scope") != "supplied_pocket_only":
        raise ValueError("Literature comparison must be supplied-pocket-only")
    return literature


def load_executed(path: Path) -> dict[str, dict]:
    with path.open() as handle:
        report = json.load(handle)
    if report.get("comparison_scope") != "supplied_pocket_only":
        raise ValueError("Executed comparison must be supplied-pocket-only")
    result = report["datasets"]
    for key, expected in (("astex", 85), ("posebusters", 308)):
        if int(result[key]["n"]) != expected:
            raise ValueError(f"Unexpected executed {key} denominator")
        if any(int(row["repeat_count"]) != 3 for row in result[key]["methods"]):
            raise ValueError(f"Executed {key} contains a non-three-repeat row")
    return result


def plot_panel(
    ax: plt.Axes,
    dataset: dict,
    effdock: dict[str, float | int],
    executed: dict,
    panel: str,
) -> None:
    rows = [{**row, "source": "literature"} for row in dataset["values"]]
    rows.extend(
        {
            "method": row["run_label"],
            "value": float(row["top1"]["mean"]),
            "std": float(row["top1"]["std"]),
            "family": row["family"],
            "source": "executed",
        }
        for row in executed["methods"]
    )
    rows.append(
        {
            "method": "EFF-Dock U70k\n(refined Joint)",
            "value": effdock["top1"],
            "family": "ours",
            "source": "effdock",
        }
    )
    rows.sort(key=lambda row: float(row["value"]), reverse=True)

    methods = [row["method"] for row in rows]
    values = np.asarray([float(row["value"]) for row in rows])
    colors = []
    for row in rows:
        if row["family"] == "ours":
            colors.append(EFFDOCK)
        elif row["source"] == "executed" and row["family"] == "classical":
            colors.append(EXECUTED_CLASSICAL)
        elif row["source"] == "executed" and row["family"] == "hybrid":
            colors.append(EXECUTED_HYBRID)
        elif row["source"] == "executed":
            colors.append(EXECUTED_LEARNED)
        elif row["family"] == "classical":
            colors.append(CLASSICAL)
        else:
            colors.append(LEARNED)
    x = np.arange(len(rows))
    bars = ax.bar(
        x,
        values,
        width=0.68,
        color=colors,
        edgecolor=BAR_EDGE,
        linewidth=0.8,
        zorder=3,
    )
    executed_indices = [
        index for index, row in enumerate(rows) if row["source"] == "executed"
    ]
    if executed_indices:
        ax.errorbar(
            executed_indices,
            [values[index] for index in executed_indices],
            yerr=[float(rows[index]["std"]) for index in executed_indices],
            fmt="none",
            ecolor="#536377",
            elinewidth=0.9,
            capsize=2.5,
            capthick=0.9,
            zorder=6,
        )

    eff_index = methods.index("EFF-Dock U70k\n(refined Joint)")
    top1 = float(effdock["top1"])
    oracle = float(effdock["oracle"])
    ax.bar(
        eff_index,
        oracle - top1,
        bottom=top1,
        width=0.68,
        color=ORACLE,
        edgecolor=ORACLE_EDGE,
        linewidth=0.8,
        hatch="///",
        zorder=4,
    )

    for index, (bar, value) in enumerate(zip(bars, values, strict=True)):
        if index == eff_index:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value - 2.0,
                f"{value:.1f}",
                va="top",
                ha="center",
                fontsize=8.2,
                fontweight="bold",
                color="white",
                zorder=6,
            )
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1.2,
            f"{value:.1f}",
            va="bottom",
            ha="center",
            fontsize=7.8,
            fontweight="semibold",
            color=INK,
        )
    ax.text(
        eff_index,
        oracle + 1.2,
        f"{oracle:.1f}",
        va="bottom",
        ha="center",
        fontsize=8.2,
        fontweight="bold",
        color="#A35F43",
    )

    ax.set_xticks(x, methods, rotation=61, ha="right", rotation_mode="anchor")
    for tick, row in zip(ax.get_xticklabels(), rows, strict=True):
        if row["family"] == "ours":
            tick.set_fontweight("bold")
            tick.set_color(EFFDOCK)
    ax.set_ylim(0, 104)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", length=0, pad=7, labelsize=6.7)
    ax.tick_params(axis="y", colors="#52606D", labelsize=8.5)
    ax.set_title(
        f"{panel}  {dataset['name']}  (N={dataset['n']})",
        loc="left",
        fontsize=12.5,
        fontweight="bold",
        color=INK,
        pad=9,
    )


def main() -> None:
    args = parse_args()
    effdock = load_effdock(args.benchmark)
    literature = load_literature(args.literature)
    executed = load_executed(args.executed)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.labelsize": 11,
        }
    )
    fig, axes = plt.subplots(
        1, 2, figsize=(18.2, 6.2), sharey=True, constrained_layout=False
    )
    plot_panel(axes[0], literature["astex"], effdock["astex"], executed["astex"], "A")
    plot_panel(
        axes[1],
        literature["posebusters"],
        effdock["posebusters"],
        executed["posebusters"],
        "B",
    )
    axes[0].set_ylabel("Success rate (%)", fontsize=10)
    legend = (
        Patch(facecolor=LEARNED, edgecolor=BAR_EDGE, linewidth=0.8, label="Pocket-based learning"),
        Patch(
            facecolor=CLASSICAL,
            edgecolor=BAR_EDGE,
            linewidth=0.8,
            label="Pocket-based classical / hybrid",
        ),
        Patch(
            facecolor=EXECUTED_LEARNED,
            edgecolor=BAR_EDGE,
            linewidth=0.8,
            label="Our rerun: learning-based (mean ± SD)",
        ),
        Patch(
            facecolor=EXECUTED_HYBRID,
            edgecolor=BAR_EDGE,
            linewidth=0.8,
            label="Our rerun: hybrid (mean ± SD)",
        ),
        Patch(
            facecolor=EXECUTED_CLASSICAL,
            edgecolor=BAR_EDGE,
            linewidth=0.8,
            label="Our rerun: classical (mean ± SD)",
        ),
        Patch(
            facecolor=EFFDOCK,
            edgecolor=BAR_EDGE,
            linewidth=0.8,
            label="EFF-Dock U70k: refined Joint",
        ),
        Patch(
            facecolor=ORACLE,
            edgecolor=ORACLE_EDGE,
            linewidth=0.8,
            hatch="///",
            label="EFF-Dock refined Oracle-100",
        ),
    )
    axes[1].legend(
        handles=legend,
        loc="upper right",
        ncols=2,
        frameon=True,
        framealpha=0.96,
        edgecolor=GRID,
        fontsize=7.0,
    )
    fig.subplots_adjust(left=0.052, right=0.995, top=0.90, bottom=0.33, wspace=0.10)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    print(args.output)


if __name__ == "__main__":
    main()
