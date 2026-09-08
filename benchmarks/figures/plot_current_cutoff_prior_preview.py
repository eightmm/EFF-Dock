#!/usr/bin/env python3
"""Render the complete raw-Oracle portion of the active robustness extension."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BASE_JITTER0 = ROOT / (
    "outputs/benchmarks/effdock_pocket_cutoff_robustness_runs/"
    "cutoff-r3-production-20260901-r2"
)
BASE_JITTER = ROOT / (
    "outputs/benchmarks/effdock_pocket_cutoff_jitter_robustness_runs/"
    "production-jitter-r3-20260904-v3"
)
EXTENSION = ROOT / (
    "outputs/benchmarks/effdock_pocket_prior_robustness_extension_runs/"
    "production-cutoff14-sigma-r3-20260906-v1"
)
OUTPUT = ROOT / "docs/results/current_overview/figures/cutoff_prior_raw_oracle_preview.png"
CSV_OUTPUT = ROOT / "docs/results/current_overview/cutoff_prior_raw_oracle_preview.csv"
DATASETS = {"astex": 85, "posebusters": 308}
JITTERS = (0, 1, 2)
REPEATS = (0, 1, 2)
CUTOFFS = (6, 8, 10, 12, 14)
SIGMAS = (1, 2, 4)
INK = "#2B3950"
MUTED = "#6D7D92"
JITTER_STYLE = {
    0: ("#7FA6D8", "o", "jitter σ=0 Å"),
    1: ("#75BFA8", "s", "jitter σ=1 Å"),
    2: ("#E89B79", "D", "jitter σ=2 Å"),
}
LABEL_OFFSET_POINTS = {0: 8, 1: -15, 2: 8}


def old_raw_dir(cutoff: int, jitter: int, repeat: int) -> Path:
    if jitter == 0:
        return BASE_JITTER0 / f"cutoff_{cutoff:02d}" / f"repeat_{repeat}" / "raw"
    return (
        BASE_JITTER
        / f"jitter_{jitter:02d}"
        / f"cutoff_{cutoff:02d}"
        / f"repeat_{repeat}"
        / "raw"
    )


def extension_raw_dir(cutoff: int, sigma: int, jitter: int, repeat: int) -> Path:
    return (
        EXTENSION
        / f"cutoff_{cutoff:02d}"
        / f"sigma_{sigma:02d}"
        / f"jitter_{jitter:02d}"
        / f"repeat_{repeat}"
        / "raw"
    )


def raw_oracle_pct(raw_dir: Path, dataset: str, expected: int) -> float:
    frames = [pd.read_csv(path, usecols=["id", "oracle_rmsd"]) for path in sorted(raw_dir.glob(f"*{dataset}*.csv"))]
    if not frames:
        raise FileNotFoundError(f"missing raw CSVs for {dataset}: {raw_dir}")
    frame = pd.concat(frames, ignore_index=True)
    if len(frame) != expected or frame["id"].nunique() != expected:
        raise ValueError(f"incomplete {dataset} raw cell at {raw_dir}: {len(frame)}/{expected}")
    return 100.0 * float((frame["oracle_rmsd"] < 2.0).mean())


def summarise() -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for dataset, expected in DATASETS.items():
        for jitter in JITTERS:
            for cutoff in CUTOFFS:
                values = [
                    raw_oracle_pct(
                        extension_raw_dir(cutoff, 2, jitter, repeat)
                        if cutoff == 14
                        else old_raw_dir(cutoff, jitter, repeat),
                        dataset,
                        expected,
                    )
                    for repeat in REPEATS
                ]
                rows.append(
                    {
                        "view": "cutoff",
                        "dataset": dataset,
                        "jitter_sigma_A_per_axis": jitter,
                        "cutoff_A": cutoff,
                        "prior_sigma_A": 2,
                        "mean_pct": float(np.mean(values)),
                        "std_pct": float(np.std(values, ddof=1)),
                    }
                )
            for sigma in SIGMAS:
                values = [
                    raw_oracle_pct(
                        old_raw_dir(10, jitter, repeat)
                        if sigma == 2
                        else extension_raw_dir(10, sigma, jitter, repeat),
                        dataset,
                        expected,
                    )
                    for repeat in REPEATS
                ]
                rows.append(
                    {
                        "view": "prior_sigma",
                        "dataset": dataset,
                        "jitter_sigma_A_per_axis": jitter,
                        "cutoff_A": 10,
                        "prior_sigma_A": sigma,
                        "mean_pct": float(np.mean(values)),
                        "std_pct": float(np.std(values, ddof=1)),
                    }
                )
    return rows


def draw(ax: plt.Axes, rows: list[dict[str, float | int | str]], *, key: str, title: str) -> None:
    for jitter in JITTERS:
        points = sorted(
            (row for row in rows if int(row["jitter_sigma_A_per_axis"]) == jitter),
            key=lambda row: float(row[key]),
        )
        x = np.asarray([float(row[key]) for row in points])
        y = np.asarray([float(row["mean_pct"]) for row in points])
        err = np.asarray([float(row["std_pct"]) for row in points])
        color, marker, label = JITTER_STYLE[jitter]
        ax.errorbar(
            x,
            y,
            yerr=err,
            color=color,
            marker=marker,
            markersize=6.5,
            markeredgecolor="white",
            markeredgewidth=0.9,
            linewidth=2.2,
            capsize=3,
            label=label,
            zorder=3,
        )
        for x_value, y_value in zip(x, y, strict=True):
            ax.annotate(
                f"{y_value:.1f}",
                (x_value, y_value),
                xytext=(0, LABEL_OFFSET_POINTS[jitter]),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                color=INK,
                fontweight="bold",
            )
    ax.set_title(title, loc="left", fontsize=11.5, fontweight="bold", color=INK, pad=8)
    ax.set_ylim(55, 101)
    ax.set_yticks((60, 70, 80, 90, 100))
    ax.grid(axis="y", color="#DCE4EE", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#AAB7C7")


def main() -> None:
    rows = summarise()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 7.7), sharey=True)
    for row, dataset in enumerate(("astex", "posebusters")):
        label = "Astex Diverse (N=85)" if dataset == "astex" else "PoseBusters v2 (N=308)"
        cutoff_rows = [item for item in rows if item["dataset"] == dataset and item["view"] == "cutoff"]
        sigma_rows = [item for item in rows if item["dataset"] == dataset and item["view"] == "prior_sigma"]
        draw(axes[row, 0], cutoff_rows, key="cutoff_A", title=f"{label} · pocket cutoff")
        draw(axes[row, 1], sigma_rows, key="prior_sigma_A", title=f"{label} · translation prior σ")
        axes[row, 0].set_xticks(CUTOFFS)
        axes[row, 1].set_xticks(SIGMAS)
        axes[row, 0].set_xlabel("Docking pocket cutoff (Å)", color=INK)
        axes[row, 1].set_xlabel("Translation prior σ (Å), cutoff=10 Å", color=INK)
        axes[row, 0].set_ylabel("Oracle-100 success, RMSD < 2 Å (%)", color=INK)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.925), fontsize=9)
    fig.suptitle("Current EFF-Dock robustness preview", x=0.07, y=0.985, ha="left", fontsize=16, fontweight="bold", color=INK)
    fig.text(
        0.07,
        0.948,
        "Raw ODE candidate-generation Oracle-100 · mean ± SD across 3 paired seeds · refinement / confidence / PB pending",
        ha="left",
        fontsize=9,
        color=MUTED,
    )
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.09, top=0.86, hspace=0.38, wspace=0.20)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    with CSV_OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(OUTPUT)


if __name__ == "__main__":
    main()
