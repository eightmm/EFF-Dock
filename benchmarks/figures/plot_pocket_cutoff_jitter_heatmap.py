#!/usr/bin/env python3
"""Plot the completed legacy cutoff-by-center-jitter sensitivity matrix."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "outputs/benchmarks/pocket_sensitivity_n80_s25_v2/raw"
DEFAULT_OUTPUT = (
    ROOT / "docs/results/current_overview/figures/pocket_cutoff_jitter_legacy_heatmap.png"
)
INK = "#29384D"
MUTED = "#66758A"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def load_cells(input_dir: Path) -> tuple[list[dict], dict]:
    cells: list[dict] = []
    identity: dict | None = None
    for path in sorted(input_dir.glob("*.summary.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("protocol_id") != "EFFDOCK-POCKET-SENSITIVITY-N80-S25-V2":
            continue
        current = {
            "num_samples": payload["num_samples"],
            "num_steps": payload["num_steps"],
            "sigma": payload["sigma"],
            "checkpoint": payload["checkpoint"],
            "confidence_checkpoint": payload["confidence_checkpoint"],
        }
        if identity is None:
            identity = current
        elif current != identity:
            raise ValueError(f"mixed protocol identity in {path}")
        if payload["num_failed"] != 0:
            raise ValueError(f"failed complexes in {path}")
        cells.append(
            {
                "dataset": payload["dataset"],
                "cutoff": int(payload["pocket_cutoff"]),
                "jitter": int(payload["center_jitter_sigma"]),
                "n": int(payload["num_success"]),
                "top1": float(payload["stats"]["confidence_final"]["pct_lt_2A"]),
                "oracle": float(payload["stats"]["oracle"]["pct_lt_2A"]),
                "median_rmsd": float(
                    payload["stats"]["confidence_final"]["median_rmsd"]
                ),
                "source": str(path.relative_to(ROOT)),
            }
        )
    if identity is None:
        raise ValueError(f"no admitted summaries in {input_dir}")
    expected = {
        (dataset, cutoff, jitter)
        for dataset in ("astex", "posebusters")
        for cutoff in (6, 8, 10, 12)
        for jitter in (0, 1, 2)
    }
    observed = {(row["dataset"], row["cutoff"], row["jitter"]) for row in cells}
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(f"incomplete matrix: missing={missing}, extra={extra}")
    return cells, identity


def matrix(cells: list[dict], dataset: str, metric: str) -> np.ndarray:
    lookup = {(row["cutoff"], row["jitter"]): row[metric] for row in cells if row["dataset"] == dataset}
    return np.asarray(
        [[lookup[(cutoff, jitter)] for cutoff in (6, 8, 10, 12)] for jitter in (0, 1, 2)],
        dtype=float,
    )


def add_heatmap(
    ax: plt.Axes,
    values: np.ndarray,
    *,
    title: str,
    letter: str,
    cmap: LinearSegmentedColormap,
    norm: Normalize,
) -> None:
    image = ax.imshow(values, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")
    ax.set_xticks(np.arange(4), ("6", "8", "10", "12"))
    ax.set_yticks(np.arange(3), ("0", "1", "2"))
    ax.set_xlabel("Pocket cutoff (Å)", color=INK)
    ax.set_ylabel("Center jitter σ (Å per axis)", color=INK)
    ax.set_title(f"{letter}  {title}", loc="left", color=INK, fontweight="bold", pad=9)
    ax.tick_params(length=0, colors=MUTED)
    ax.set_xticks(np.arange(-0.5, 4, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 3, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2.0)
    ax.tick_params(which="minor", bottom=False, left=False)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            color = "white" if norm(value) > 0.57 else INK
            ax.text(
                column,
                row,
                f"{value:.1f}",
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
    args = parse_args()
    cells, identity = load_cells(args.input_dir)
    cmap = LinearSegmentedColormap.from_list(
        "effdock_pastel", ("#F5DFC2", "#C9DDBD", "#8EC4C1", "#7295C5")
    )
    norm = Normalize(vmin=20.0, vmax=100.0)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.6), constrained_layout=False)
    panels = (
        ("astex", "top1", "Astex Top-1 RMSD < 2 Å", "A"),
        ("astex", "oracle", "Astex Oracle-80 RMSD < 2 Å", "B"),
        ("posebusters", "top1", "PoseBusters v2 Top-1 RMSD < 2 Å", "C"),
        ("posebusters", "oracle", "PoseBusters v2 Oracle-80 RMSD < 2 Å", "D"),
    )
    image = None
    for ax, (dataset, metric, title, letter) in zip(axes.flat, panels, strict=True):
        image = add_heatmap(
            ax,
            matrix(cells, dataset, metric),
            title=title,
            letter=letter,
            cmap=cmap,
            norm=norm,
        )
    fig.suptitle(
        "Pocket cutoff × center jitter robustness",
        x=0.08,
        y=0.988,
        ha="left",
        fontsize=15,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.08,
        0.947,
        "Legacy matched ablation · N80/S25 · prior σ=0.5 · extmatch confidence 42.5k · no guidance/refinement",
        ha="left",
        fontsize=9,
        color=MUTED,
    )
    fig.subplots_adjust(
        left=0.085,
        right=0.875,
        top=0.88,
        bottom=0.09,
        wspace=0.25,
        hspace=0.36,
    )
    colorbar_axis = fig.add_axes((0.905, 0.20, 0.018, 0.59))
    colorbar = fig.colorbar(image, cax=colorbar_axis)
    colorbar.set_label("Success rate (%)", color=INK)
    colorbar.ax.tick_params(colors=MUTED)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)

    csv_path = args.output.parents[1] / "pocket_cutoff_jitter_legacy.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "dataset",
                "n",
                "pocket_cutoff_A",
                "center_jitter_sigma_A_per_axis",
                "top1_rmsd_lt2_pct",
                "oracle80_rmsd_lt2_pct",
                "top1_median_rmsd_A",
                "source",
            ),
        )
        writer.writeheader()
        for row in sorted(cells, key=lambda value: (value["dataset"], value["jitter"], value["cutoff"])):
            writer.writerow(
                {
                    "dataset": row["dataset"],
                    "n": row["n"],
                    "pocket_cutoff_A": row["cutoff"],
                    "center_jitter_sigma_A_per_axis": row["jitter"],
                    "top1_rmsd_lt2_pct": f"{row['top1']:.8f}",
                    "oracle80_rmsd_lt2_pct": f"{row['oracle']:.8f}",
                    "top1_median_rmsd_A": f"{row['median_rmsd']:.8f}",
                    "source": row["source"],
                }
            )
    print(args.output)


if __name__ == "__main__":
    main()
