"""Operator-change uncertainty and effective-selector bottleneck panels."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from benchmarks.figures.labels import interval_label, segment_labels
from benchmarks.figures.paper import NAMES, save

DATA = Path(__file__).resolve().parents[2] / "benchmarks/results/paper/evidence"


def render(out):
    changes = json.loads((DATA / "rw_operator_change.json").read_text())["comparisons"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7), sharey=True, layout="constrained")
    for ax, metric, title in zip(
        axes, ("rmsd_lt2", "joint"), ("A  RMSD success", "B  PB-valid success"), strict=True
    ):
        for i, ds in enumerate(NAMES):
            r = next(r for r in changes if r["dataset"] == ds and r["metric"] == metric)
            e = r["estimate"]
            lo, hi = e["ci95_pp"]
            v = e["delta_pp"]
            fresh = r["reference"] == "paired_rerun"
            ax.errorbar(
                (lo + hi) / 2,
                i,
                xerr=(hi - lo) / 2,
                fmt="none",
                ecolor="#526273",
                capsize=3,
            )
            ax.plot(
                v,
                i,
                "o" if fresh else "D",
                color="#8FB9D8" if fresh else "#E8B395",
                ms=6,
            )
            interval_label(ax, i, v)
        ax.axvline(0, color="#8C98A5", ls="--", lw=0.8)
        ax.set_title(title, loc="left", weight="bold")
        ax.set_xlabel("Rw − Rᵀw (percentage points)")
        ax.grid(axis="x", color="#E1E5EA", lw=0.7)
    axes[0].set_yticks(range(5), NAMES.values())
    axes[0].invert_yaxis()
    fig.legend(
        handles=[
            Line2D([], [], marker="o", color="#8FB9D8", ls="none", label="Paired rerun"),
            Line2D([], [], marker="D", color="#E8B395", ls="none", label="Historical bank"),
        ],
        loc="outside lower center",
        ncol=2,
        frameon=False,
    )
    save(fig, out, "S11_rw_operator_change")
    rows = json.loads((DATA / "selector_bottleneck.json").read_text())["rows"]
    colors = ["#B9C3CE", "#D7B5C5", "#E8B395", "#B8A6CE", "#9AC9B4"]
    labels = [
        "Generation failure",
        "Chirality exclusion",
        "Ranking failure",
        "PB failure",
        "Success",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), layout="constrained")
    selected = [
        next(r for r in rows if r["dataset"] == ds and r["stage"] == "refined") for ds in NAMES
    ]
    bottom = np.zeros(5)
    for j, c in enumerate(colors):
        h = np.array([r["mean"][j] for r in selected])
        axes[0].bar(
            range(5),
            h,
            bottom=bottom,
            color=c,
            width=0.67,
            hatch="////" if j == 3 else None,
            edgecolor="white" if j == 3 else c,
            linewidth=0,
        )
        segment_labels(axes[0], range(5), h, bottom)
        bottom += h
    axes[0].set(ylim=(0, 100), ylabel="Complexes (%)")
    axes[0].set_xticks(
        range(5),
        [f"{name}\n(n={r['n']})" for name, r in zip(NAMES.values(), selected, strict=True)],
        rotation=22,
        ha="right",
    )
    axes[0].set_title("A  Failure attribution", loc="left", weight="bold")
    for ds, c in zip(NAMES, ["#8FB9D8", "#E8B395", "#9AC9B4", "#B8A6CE", "#C8B77D"], strict=True):
        r = next(r for r in selected if r["dataset"] == ds)
        cdf = r["cdf"]
        axes[1].plot(cdf["rmsd_regret_angstrom"], cdf["mean"], color=c, lw=1.9, label=NAMES[ds])
    axes[1].set(
        xlim=(0, 5),
        ylim=(0, 100),
        xlabel="Selected − eligible oracle RMSD (Å)",
        ylabel="Cumulative complexes (%)",
    )
    axes[1].set_title("B  Selection regret", loc="left", weight="bold")
    axes[1].legend(frameon=False, fontsize=8, loc="lower right")
    for ax in axes:
        ax.grid(axis="y", color="#E1E5EA", lw=0.7)
        ax.set_axisbelow(True)
    fig.legend(
        handles=[
            Patch(
                facecolor=c,
                label=l,
                hatch="////" if j == 3 else None,
                edgecolor="white" if j == 3 else c,
            )
            for j, (c, l) in enumerate(zip(colors, labels, strict=True))
        ],
        loc="outside lower center",
        ncol=5,
        frameon=False,
        fontsize=9,
    )
    save(fig, out, "S12_selector_bottleneck")
