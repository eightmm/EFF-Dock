#!/usr/bin/env python3
"""Render all completed production cutoff-by-jitter metrics as heatmaps."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / (
    "outputs/benchmarks/effdock_pocket_cutoff_jitter_robustness_runs/"
    "production-jitter-r3-20260904-v3/report.json"
)
OUTPUT = ROOT / "docs/results/current_overview/figures/pocket_cutoff_jitter_complete.png"
CSV_OUTPUT = ROOT / "docs/results/current_overview/pocket_cutoff_jitter_complete.csv"
METRICS = (
    ("selected_rmsd_lt2", "Top-1 RMSD < 2 Å"),
    ("posebusters_valid", "PB-valid"),
    ("joint_rmsd_lt2_pb_valid", "Joint: < 2 Å & PB-valid"),
    ("oracle_rmsd_lt2", "Refined Oracle-100 < 2 Å"),
)
CUTOFFS = (6, 8, 10, 12)
JITTERS = (0, 1, 2)
INK = "#29384D"
MUTED = "#68788E"


def load_rows() -> list[dict[str, float | int | str]]:
    payload = json.loads(REPORT.read_text(encoding="utf-8"))
    if (
        payload.get("status") != "complete"
        or payload.get("protocol_id") != "EFFDOCK-POCKET-CUTOFF-JITTER-ROBUSTNESS-V1"
        or payload.get("total_selected_posebusters_errors") != 0
    ):
        raise ValueError("expected a completed, zero-error production cutoff-by-jitter report")
    rows: list[dict[str, float | int | str]] = []
    for dataset, data in payload["datasets"].items():
        for cutoff_row in data["cutoffs"]:
            cutoff = int(cutoff_row["pocket_cutoff_angstrom"])
            for jitter_row in cutoff_row["jitters"]:
                jitter = int(jitter_row["center_jitter_sigma_angstrom_per_axis"])
                aggregate = jitter_row["aggregate"]
                for metric, _ in METRICS:
                    rows.append(
                        {
                            "dataset": dataset,
                            "cutoff_A": cutoff,
                            "jitter_sigma_A_per_axis": jitter,
                            "metric": metric,
                            "mean_pct": float(aggregate[metric]["mean"]),
                            "std_pct": float(aggregate[metric]["std"]),
                        }
                    )
    expected = 2 * len(CUTOFFS) * len(JITTERS) * len(METRICS)
    if len(rows) != expected:
        raise ValueError(f"incomplete report matrix: {len(rows)} != {expected}")
    return rows


def matrix(rows: list[dict[str, float | int | str]], dataset: str, metric: str, key: str) -> np.ndarray:
    values = {
        (int(row["cutoff_A"]), int(row["jitter_sigma_A_per_axis"])): float(row[key])
        for row in rows
        if row["dataset"] == dataset and row["metric"] == metric
    }
    return np.asarray([[values[(cutoff, jitter)] for jitter in JITTERS] for cutoff in CUTOFFS])


def draw_panel(
    ax: plt.Axes,
    means: np.ndarray,
    stds: np.ndarray,
    title: str,
    cmap: LinearSegmentedColormap,
    norm: Normalize,
) -> plt.AxesImage:
    image = ax.imshow(means, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")
    ax.set_xticks(np.arange(len(JITTERS)), [str(value) for value in JITTERS])
    ax.set_yticks(np.arange(len(CUTOFFS)), [str(value) for value in CUTOFFS])
    ax.set_title(title, loc="left", color=INK, fontsize=10.5, fontweight="bold", pad=7)
    ax.tick_params(length=0, colors=MUTED, labelsize=9)
    ax.set_xticks(np.arange(-0.5, len(JITTERS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(CUTOFFS), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2.0)
    ax.tick_params(which="minor", bottom=False, left=False)
    for row in range(means.shape[0]):
        for column in range(means.shape[1]):
            value = means[row, column]
            color = "white" if value >= 84 else INK
            ax.text(
                column,
                row,
                f"{value:.1f}\n±{stds[row, column]:.1f}",
                ha="center",
                va="center",
                color=color,
                fontsize=8.4,
                fontweight="bold",
            )
    for spine in ax.spines.values():
        spine.set_color("#CBD5E1")
        spine.set_linewidth(0.8)
    return image


def main() -> None:
    rows = load_rows()
    cmap = LinearSegmentedColormap.from_list(
        "effdock_complete", ("#F4C3B7", "#F7E4B5", "#B8DCC8", "#789ED0")
    )
    norm = Normalize(vmin=60.0, vmax=100.0)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(2, 4, figsize=(15.2, 7.2), sharex=True, sharey=True)
    for column, (metric, title) in enumerate(METRICS):
        image = draw_panel(
            axes[0, column],
            matrix(rows, "astex", metric, "mean_pct"),
            matrix(rows, "astex", metric, "std_pct"),
            title,
            cmap,
            norm,
        )
        draw_panel(
            axes[1, column],
            matrix(rows, "posebusters", metric, "mean_pct"),
            matrix(rows, "posebusters", metric, "std_pct"),
            title,
            cmap,
            norm,
        )
        axes[1, column].set_xlabel("Center jitter σ (Å per axis)", color=INK, fontsize=9)
    axes[0, 0].set_ylabel("Astex cutoff (Å)", color=INK, fontsize=10)
    axes[1, 0].set_ylabel("PoseBusters cutoff (Å)", color=INK, fontsize=10)
    fig.suptitle(
        "EFF-Dock production pocket-cutoff × center-jitter robustness",
        x=0.065,
        y=0.985,
        ha="left",
        fontsize=16,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.065,
        0.937,
        "U70k confidence after adaptive physical refinement · mean ± SD across 3 paired seeds · N100/S10, prior σ=2",
        ha="left",
        fontsize=9,
        color=MUTED,
    )
    fig.subplots_adjust(left=0.07, right=0.90, bottom=0.105, top=0.86, wspace=0.19, hspace=0.25)
    color_axis = fig.add_axes((0.925, 0.18, 0.013, 0.62))
    colorbar = fig.colorbar(image, cax=color_axis)
    colorbar.set_label("Success rate (%)", color=INK)
    colorbar.ax.tick_params(colors=MUTED, labelsize=9)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(OUTPUT.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    with CSV_OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(OUTPUT)


if __name__ == "__main__":
    main()
