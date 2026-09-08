#!/usr/bin/env python3
"""Render the completed prior-sigma-by-center-jitter joint-success matrix."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / (
    "outputs/benchmarks/effdock_pocket_prior_robustness_extension_runs/"
    "production-cutoff14-sigma-r3-20260906-v1/report.json"
)
OUTPUT_DIR = ROOT / "docs/results/current_overview/figures"
CSV_OUTPUT = ROOT / "docs/results/current_overview/prior_sigma_jitter_top1_joint_metrics.csv"
SIGMAS = (1, 2, 4)
JITTERS = (0, 1, 2)
METRIC = "joint_rmsd_lt2_pb_valid"
INK = "#29384D"
MUTED = "#66758A"


def load_rows() -> list[dict[str, float | int | str]]:
    payload = json.loads(REPORT.read_text(encoding="utf-8"))
    if (
        payload.get("status") != "complete"
        or payload.get("protocol_id") != "EFFDOCK-POCKET-PRIOR-ROBUSTNESS-EXTENSION-V1"
        or payload.get("total_selected_posebusters_errors") != 0
    ):
        raise ValueError("expected a completed prior-sigma report with zero PB errors")

    rows: list[dict[str, float | int | str]] = []
    for dataset, dataset_rows in payload["datasets"].items():
        for sigma_row in dataset_rows["prior_sigma_sweep_fixed_cutoff_10"]:
            sigma = int(sigma_row["prior_sigma_angstrom"])
            for jitter_row in sigma_row["jitters"]:
                aggregate = jitter_row["aggregate"][METRIC]
                rows.append(
                    {
                        "dataset": dataset,
                        "prior_sigma_A": sigma,
                        "jitter_sigma_A_per_axis": int(
                            jitter_row["center_jitter_sigma_angstrom_per_axis"]
                        ),
                        "metric": METRIC,
                        "mean_pct": float(aggregate["mean"]),
                        "std_pct": float(aggregate["std"]),
                    }
                )
    expected = 2 * len(SIGMAS) * len(JITTERS)
    if len(rows) != expected:
        raise ValueError(f"incomplete sigma-by-jitter matrix: {len(rows)} != {expected}")
    return rows


def matrix(rows: list[dict[str, float | int | str]], dataset: str, key: str) -> np.ndarray:
    lookup = {
        (int(row["prior_sigma_A"]), int(row["jitter_sigma_A_per_axis"])): float(row[key])
        for row in rows
        if row["dataset"] == dataset
    }
    return np.asarray([[lookup[(sigma, jitter)] for jitter in JITTERS] for sigma in SIGMAS])


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
    ax.set_yticks(np.arange(len(SIGMAS)), [str(value) for value in SIGMAS])
    ax.set_xlabel("Center jitter σ (Å per axis)", color=INK)
    ax.set_ylabel("Prior σ (Å)", color=INK)
    ax.set_title(title, loc="left", color=INK, fontweight="bold", pad=10)
    ax.tick_params(length=0, colors=MUTED)
    ax.set_xticks(np.arange(-0.5, len(JITTERS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(SIGMAS), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2.4)
    ax.tick_params(which="minor", bottom=False, left=False)
    for row in range(means.shape[0]):
        for column in range(means.shape[1]):
            value = means[row, column]
            color = "white" if norm(value) >= 0.56 else INK
            ax.text(
                column,
                row,
                f"{value:.1f}\n± {stds[row, column]:.1f}",
                ha="center",
                va="center",
                color=color,
                fontsize=11,
                fontweight="bold",
            )
    for spine in ax.spines.values():
        spine.set_color("#CBD5E1")
        spine.set_linewidth(0.8)
    return image


def main() -> None:
    rows = load_rows()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cmap = LinearSegmentedColormap.from_list(
        "effdock_prior_sigma", ("#F4C7C3", "#F7E2B6", "#B8DBC9", "#83A9D8")
    )
    norm = Normalize(vmin=60.0, vmax=100.0)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.3))
    image = draw_panel(
        axes[0], matrix(rows, "astex", "mean_pct"), matrix(rows, "astex", "std_pct"),
        "A  Astex Diverse (N=85)", cmap, norm,
    )
    draw_panel(
        axes[1], matrix(rows, "posebusters", "mean_pct"), matrix(rows, "posebusters", "std_pct"),
        "B  PoseBusters v2 (N=308)", cmap, norm,
    )
    fig.suptitle(
        "Top-1 RMSD < 2 Å & PB-valid", x=0.5, y=0.985, ha="center", fontsize=15,
        fontweight="bold", color=INK,
    )
    fig.subplots_adjust(left=0.08, right=0.89, top=0.80, bottom=0.18, wspace=0.30)
    color_axis = fig.add_axes((0.92, 0.23, 0.018, 0.52))
    colorbar = fig.colorbar(image, cax=color_axis)
    colorbar.set_label("Success rate (%)", color=INK)
    colorbar.ax.tick_params(colors=MUTED)
    figure_path = OUTPUT_DIR / "prior_sigma_jitter_top1_joint_heatmap.png"
    fig.savefig(figure_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(figure_path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    with CSV_OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
