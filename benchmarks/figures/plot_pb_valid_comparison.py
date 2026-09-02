#!/usr/bin/env python3
"""Plot clean pocket-only RMSD/PB-valid Top-1 bars."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = (
    ROOT / "benchmarks/results/external_models/pocket_only_pb_valid_comparison.json"
)
DEFAULT_OUTPUT = (
    ROOT / "benchmarks/results/external_models/pocket_only_pb_valid_comparison.png"
)

COLORS = {
    "ours": "#6FAFA8",
    "learning": "#8FB3D9",
    "hybrid": "#B3A4D6",
    "classical": "#ABD6C5",
}
EDGE = "#68788B"
INK = "#263447"
GRID = "#E5EAF0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dpi", type=int, default=260)
    return parser.parse_args()


def panel(ax: plt.Axes, dataset: dict, letter: str) -> None:
    rows = dataset["methods"]
    labels = [row["method"] for row in rows]
    total = np.asarray([row["top1_rmsd_lt2"]["mean"] for row in rows], dtype=float)
    joint = np.asarray(
        [row["top1_joint_rmsd_lt2_pb_valid"]["mean"] for row in rows], dtype=float
    )
    total_sd = np.asarray([row["top1_rmsd_lt2"].get("std", 0.0) for row in rows])
    joint_sd = np.asarray(
        [row["top1_joint_rmsd_lt2_pb_valid"].get("std", 0.0) for row in rows]
    )
    if np.any(joint > total + 1e-10):
        raise ValueError("joint success cannot exceed RMSD success")
    x = np.arange(len(rows), dtype=float)
    x[5:] += 0.32
    colors = [COLORS[row["family"]] for row in rows]

    bars = ax.bar(
        x,
        joint,
        width=0.68,
        color=colors,
        edgecolor=EDGE,
        linewidth=0.85,
        zorder=3,
    )
    ax.bar(
        x,
        total - joint,
        bottom=joint,
        width=0.68,
        color=colors,
        alpha=0.62,
        edgecolor=EDGE,
        linewidth=0.85,
        hatch="///",
        zorder=4,
    )
    rerun = np.asarray([row["repeat_count"] == 3 for row in rows])
    if rerun.any():
        ax.errorbar(
            x[rerun],
            total[rerun],
            yerr=total_sd[rerun],
            fmt="none",
            ecolor="#4D5C70",
            elinewidth=0.9,
            capsize=2.4,
            capthick=0.9,
            zorder=7,
        )
        ax.errorbar(
            x[rerun],
            joint[rerun],
            yerr=joint_sd[rerun],
            fmt="none",
            ecolor="#FFFFFF",
            elinewidth=0.9,
            capsize=2.1,
            capthick=0.9,
            zorder=7,
        )

    for bar, top, valid in zip(bars, total, joint, strict=True):
        center = bar.get_x() + bar.get_width() / 2
        ax.text(
            center,
            top + 1.4,
            f"{top:.1f}",
            ha="center",
            va="bottom",
            fontsize=8.4,
            fontweight="bold",
            color=INK,
        )
        ax.text(
            center,
            max(valid - 2.0, 2.0),
            f"{valid:.1f}",
            ha="center",
            va="top",
            fontsize=8.1,
            fontweight="bold",
            color="white",
        )

    divider = (x[4] + x[5]) / 2
    ax.axvline(divider, color="#CBD3DC", linewidth=1.0, linestyle=(0, (3, 3)), zorder=1)
    ax.text(
        np.mean(x[:5]),
        -0.19,
        "AI / hybrid — our runs",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=8.5,
        color="#647386",
    )
    ax.text(
        np.mean(x[5:]),
        -0.19,
        "Classical — paper",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=8.5,
        color="#647386",
    )
    ax.set_xticks(x, labels, rotation=48, ha="right", rotation_mode="anchor")
    for tick, row in zip(ax.get_xticklabels(), rows, strict=True):
        if row["family"] == "ours":
            tick.set_color(COLORS["ours"])
            tick.set_fontweight("bold")
    ax.set_ylim(0, 104)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.grid(axis="y", color=GRID, linewidth=0.85, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", length=0, pad=7, labelsize=7.8)
    ax.tick_params(axis="y", colors="#52606D", labelsize=8.8)
    ax.set_title(
        f"{letter}  {dataset['name']}  (N={dataset['n']})",
        loc="left",
        fontsize=13,
        fontweight="bold",
        color=INK,
        pad=9,
    )


def main() -> None:
    args = parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    if payload.get("comparison_scope") != "supplied_pocket_only":
        raise ValueError("comparison must be supplied-pocket-only")
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(1, 2, figsize=(15.7, 6.0), sharey=True)
    panel(axes[0], payload["datasets"]["astex"], "A")
    panel(axes[1], payload["datasets"]["posebusters"], "B")
    axes[0].set_ylabel("Top-1 success rate (%)", fontsize=10.5, color=INK)
    legend = (
        Patch(facecolor="#9FBBCF", edgecolor=EDGE, label="RMSD < 2 Å and PB-valid"),
        Patch(
            facecolor="#9FBBCF",
            edgecolor=EDGE,
            alpha=0.62,
            hatch="///",
            label="RMSD < 2 Å but PB-invalid",
        ),
        Patch(facecolor=COLORS["ours"], edgecolor=EDGE, label="EFF-Dock"),
        Patch(facecolor=COLORS["learning"], edgecolor=EDGE, label="Learning-based"),
        Patch(facecolor=COLORS["hybrid"], edgecolor=EDGE, label="Hybrid"),
        Patch(facecolor=COLORS["classical"], edgecolor=EDGE, label="Classical"),
    )
    fig.legend(
        handles=legend,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=3,
        frameon=False,
        fontsize=8.5,
        columnspacing=1.6,
        handlelength=1.9,
    )
    fig.subplots_adjust(left=0.055, right=0.99, top=0.84, bottom=0.29, wspace=0.12)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
