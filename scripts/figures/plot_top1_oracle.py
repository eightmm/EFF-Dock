#!/usr/bin/env python3
"""Create a publication-style EFF-Dock literature benchmark comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SUMMARY = ROOT / "outputs/benchmarks/confidence/summary.json"
DEFAULT_LITERATURE = ROOT / "docs/LITERATURE_RMSD_COMPARISON.json"
DEFAULT_OUTPUT = ROOT / "outputs/figures/effdock_literature_comparison.png"

LEARNED = "#B8CCE4"
CLASSICAL = "#B9DDCF"
EFFDOCK = "#7E9BCB"
ORACLE = "#F5C3A6"
BAR_EDGE = "#7A8797"
ORACLE_EDGE = "#C98565"
GRID = "#E4E9EF"
INK = "#263447"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--literature", type=Path, default=DEFAULT_LITERATURE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dpi", type=int, default=240)
    return parser.parse_args()


def load_effdock(summary_path: Path) -> dict[str, dict[str, float | int]]:
    with summary_path.open() as handle:
        summary = json.load(handle)

    result: dict[str, dict[str, float | int]] = {}
    for key, expected_n in (("astex", 85), ("posebusters", 308)):
        dataset = summary["datasets"][key]
        if dataset["expected"] != expected_n or dataset["failed"] != 0:
            raise ValueError(
                f"Unexpected {key} evaluation boundary: "
                f"expected={dataset['expected']}, failed={dataset['failed']}"
            )
        result[key] = {
            "n": expected_n,
            "top1": float(dataset["stats"]["confidence"]["pct_lt_2A"]),
            "oracle": float(dataset["stats"]["oracle"]["pct_lt_2A"]),
        }
    return result


def load_literature(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def plot_panel(
    ax: plt.Axes,
    dataset: dict,
    effdock: dict[str, float | int],
    panel: str,
) -> None:
    rows = [*dataset["values"]]
    rows.append({"method": "EFF-Dock (ours)", "value": effdock["top1"], "family": "ours"})
    rows.sort(key=lambda row: float(row["value"]), reverse=True)

    methods = [row["method"] for row in rows]
    values = np.asarray([float(row["value"]) for row in rows])
    colors = [
        EFFDOCK if row["family"] == "ours" else CLASSICAL if row["family"] == "classical" else LEARNED
        for row in rows
    ]
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

    eff_index = methods.index("EFF-Dock (ours)")
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

    ax.set_xticks(x, methods, rotation=58, ha="right", rotation_mode="anchor")
    for tick, row in zip(ax.get_xticklabels(), rows, strict=True):
        if row["family"] == "ours":
            tick.set_fontweight("bold")
            tick.set_color(EFFDOCK)
    ax.set_ylim(0, 104)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", length=0, pad=7, labelsize=7.7)
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
    effdock = load_effdock(args.summary)
    literature = load_literature(args.literature)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.labelsize": 11,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.8), sharey=True, constrained_layout=False)
    plot_panel(axes[0], literature["astex"], effdock["astex"], "A")
    plot_panel(axes[1], literature["posebusters"], effdock["posebusters"], "B")
    axes[0].set_ylabel("Top-1 success: ligand RMSD < 2 A (%)", fontsize=10)
    legend = (
        Patch(facecolor=LEARNED, edgecolor=BAR_EDGE, linewidth=0.8, label="Learning-based"),
        Patch(facecolor=CLASSICAL, edgecolor=BAR_EDGE, linewidth=0.8, label="Classical / hybrid"),
        Patch(facecolor=EFFDOCK, edgecolor=BAR_EDGE, linewidth=0.8, label="EFF-Dock Top-1"),
        Patch(
            facecolor=ORACLE,
            edgecolor=ORACLE_EDGE,
            linewidth=0.8,
            hatch="///",
            label="EFF-Dock Oracle-80",
        ),
    )
    axes[1].legend(
        handles=legend,
        loc="upper right",
        ncols=2,
        frameon=True,
        framealpha=0.96,
        edgecolor=GRID,
        fontsize=7.8,
    )
    fig.subplots_adjust(left=0.06, right=0.995, top=0.93, bottom=0.31, wspace=0.12)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    print(args.output)


if __name__ == "__main__":
    main()
