#!/usr/bin/env python3
"""Plot the current EFF-Dock raw Oracle cutoff-by-jitter matrix."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize

ROOT = Path(__file__).resolve().parents[2]
JITTER0_ROOT = ROOT / (
    "outputs/benchmarks/effdock_pocket_cutoff_robustness_runs/"
    "cutoff-r3-production-20260901-r2"
)
JITTER_ROOT = ROOT / (
    "outputs/benchmarks/effdock_pocket_cutoff_jitter_robustness_runs/"
    "production-jitter-r3-20260904-v3"
)
OUTPUT = ROOT / "docs/results/current_overview/figures/pocket_cutoff_jitter_raw_oracle.png"
CSV_OUTPUT = ROOT / "docs/results/current_overview/pocket_cutoff_jitter_raw_oracle.csv"
CUTOFFS = (6, 8, 10, 12)
JITTERS = (0, 1, 2)
REPEATS = (0, 1, 2)
EXPECTED = {"astex": 85, "posebusters": 308}
INK = "#29384D"
MUTED = "#66758A"


def load_rows() -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for dataset, expected in EXPECTED.items():
        for cutoff in CUTOFFS:
            for jitter in JITTERS:
                repeat_values = []
                for repeat in REPEATS:
                    if jitter == 0:
                        raw = JITTER0_ROOT / f"cutoff_{cutoff:02d}/repeat_{repeat}/raw"
                    else:
                        raw = JITTER_ROOT / (
                            f"jitter_{jitter:02d}/cutoff_{cutoff:02d}/repeat_{repeat}/raw"
                        )
                    frames = [
                        pd.read_csv(path, usecols=["id", "oracle_rmsd"])
                        for path in sorted(raw.glob(f"*{dataset}*.csv"))
                    ]
                    if not frames:
                        raise ValueError(f"missing raw CSVs: {dataset=} {cutoff=} {jitter=} {repeat=}")
                    frame = pd.concat(frames, ignore_index=True)
                    if len(frame) != expected or frame["id"].nunique() != expected:
                        raise ValueError(
                            f"incomplete cell: {dataset=} {cutoff=} {jitter=} {repeat=} "
                            f"rows={len(frame)} unique={frame['id'].nunique()} expected={expected}"
                        )
                    repeat_values.append(100.0 * float((frame["oracle_rmsd"] < 2.0).mean()))
                rows.append(
                    {
                        "dataset": dataset,
                        "cutoff_A": cutoff,
                        "jitter_sigma_A_per_axis": jitter,
                        "mean_pct": float(np.mean(repeat_values)),
                        "std_pct": float(np.std(repeat_values, ddof=1)),
                    }
                )
    return rows


def matrix(rows: list[dict[str, float | int | str]], dataset: str, key: str) -> np.ndarray:
    lookup = {
        (int(row["cutoff_A"]), int(row["jitter_sigma_A_per_axis"])): float(row[key])
        for row in rows
        if row["dataset"] == dataset
    }
    return np.asarray([[lookup[(cutoff, jitter)] for jitter in JITTERS] for cutoff in CUTOFFS])


def draw_panel(
    ax: plt.Axes,
    means: np.ndarray,
    stds: np.ndarray,
    title: str,
    cmap: LinearSegmentedColormap,
    norm: Normalize,
) -> None:
    image = ax.imshow(means, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")
    ax.set_xticks(np.arange(len(JITTERS)), [str(value) for value in JITTERS])
    ax.set_yticks(np.arange(len(CUTOFFS)), [str(value) for value in CUTOFFS])
    ax.set_xlabel("Center jitter σ (Å per axis)", color=INK)
    ax.set_ylabel("Pocket cutoff (Å)", color=INK)
    ax.set_title(title, loc="left", color=INK, fontweight="bold", pad=10)
    ax.tick_params(length=0, colors=MUTED)
    ax.set_xticks(np.arange(-0.5, len(JITTERS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(CUTOFFS), 1), minor=True)
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
                fontsize=10,
                fontweight="bold",
            )
    for spine in ax.spines.values():
        spine.set_color("#CBD5E1")
        spine.set_linewidth(0.8)
    return image


def main() -> None:
    rows = load_rows()
    cmap = LinearSegmentedColormap.from_list(
        "effdock_current", ("#F4C7C3", "#F7E2B6", "#B8DBC9", "#83A9D8")
    )
    norm = Normalize(vmin=60.0, vmax=100.0)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.9))
    image = draw_panel(
        axes[0], matrix(rows, "astex", "mean_pct"), matrix(rows, "astex", "std_pct"),
        "Astex Diverse (N=85)", cmap, norm,
    )
    draw_panel(
        axes[1], matrix(rows, "posebusters", "mean_pct"),
        matrix(rows, "posebusters", "std_pct"), "PoseBusters v2 (N=308)", cmap, norm,
    )
    fig.suptitle(
        "Pocket cutoff × center jitter robustness",
        x=0.075,
        y=0.985,
        ha="left",
        fontsize=15,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.075,
        0.925,
        "Raw Oracle RMSD < 2 Å (%) · mean ± SD across 3 paired seeds",
        ha="left",
        fontsize=9,
        color=MUTED,
    )
    fig.subplots_adjust(left=0.08, right=0.89, top=0.82, bottom=0.16, wspace=0.30)
    cax = fig.add_axes((0.92, 0.20, 0.018, 0.58))
    colorbar = fig.colorbar(image, cax=cax)
    colorbar.set_label("Oracle success rate (%)", color=INK)
    colorbar.ax.tick_params(colors=MUTED)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    CSV_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with CSV_OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(OUTPUT)


if __name__ == "__main__":
    main()
