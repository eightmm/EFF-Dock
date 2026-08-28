#!/usr/bin/env python3
"""Plot fixed-NFE Top-1 and full-budget oracle RMSD distributions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
from pathlib import Path

SOURCE_SHA256 = {
    "86ddf0da1f179d2afd702dbabdbb0de18d8ec68b76ea74a0f284c53aac508aaa",
    "b4f0efd67458652f039a689754cf1b37ed333089c11eaa76d41647149a76ed8e",
}
DATASETS = {
    "astex": {"title": "Astex Diverse", "complexes": 85},
    "posebusters": {"title": "PoseBusters v2", "complexes": 308},
}
ARMS = {
    "s10_n100": {
        "label": "S10 / N100",
        "raw_color": "#3F6FAE",
        "refined_color": "#8EB5E3",
        "marker": "o",
    },
    "s25_n40": {
        "label": "S25 / N40",
        "raw_color": "#B95F36",
        "refined_color": "#F0A477",
        "marker": "s",
    },
}
STAGES = {
    "raw": {"label": "Raw", "linestyle": (0, (5, 2.5)), "linewidth": 2.35},
    "refined": {"label": "Adaptive", "linestyle": "solid", "linewidth": 3.25},
}
PANELS = {
    "selected": "A  U50 Top-1 selection",
    "oracle": "B  Full-budget oracle (100 vs 40 poses)",
}
PLOT_STAGE_ORDER = ("refined", "raw")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_rmsds(
    path: Path,
    dataset: str,
) -> dict[tuple[str, str, str], list[float]]:
    if file_sha256(path) not in SOURCE_SHA256:
        raise ValueError("frozen complex ledger SHA-256 mismatch")
    expected = int(DATASETS[dataset]["complexes"])
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["dataset"] == dataset]
    if len(rows) != expected or len({row["id"] for row in rows}) != expected:
        raise ValueError(f"unexpected {dataset} complex inventory")

    result: dict[tuple[str, str, str], list[float]] = {}
    for metric in PANELS:
        for stage in STAGES:
            for arm in ARMS:
                values = sorted(float(row[f"{arm}_{stage}_{metric}_rmsd"]) for row in rows)
                if len(values) != expected or any(
                    not math.isfinite(value) or value < 0 for value in values
                ):
                    raise ValueError(f"invalid RMSD distribution for {metric}/{stage}/{arm}")
                result[(metric, stage, arm)] = values
    return result


def success_rate(values: list[float], cutoff: float = 2.0) -> float:
    return 100.0 * sum(value < cutoff for value in values) / len(values)


def render(
    rmsds: dict[tuple[str, str, str], list[float]],
    dataset: str,
    output: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    metadata = DATASETS[dataset]
    observed_max = max(value for values in rmsds.values() for value in values)
    x_max = max(6.0, math.ceil(observed_max * 2.0) / 2.0)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titlesize": 13.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 11.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(13.0, 5.2), sharex=True, sharey=True)

    for axis, (metric, title) in zip(axes, PANELS.items(), strict=True):
        cutoff_values: dict[tuple[str, str], float] = {}
        for arm, arm_spec in ARMS.items():
            for stage in PLOT_STAGE_ORDER:
                stage_spec = STAGES[stage]
                values = rmsds[(metric, stage, arm)]
                cumulative = [100.0 * index / len(values) for index in range(1, len(values) + 1)]
                axis.step(
                    [0.0, *values],
                    [0.0, *cumulative],
                    where="post",
                    color=arm_spec[f"{stage}_color"],
                    linestyle=stage_spec["linestyle"],
                    linewidth=stage_spec["linewidth"],
                    marker=arm_spec["marker"] if stage == "refined" else None,
                    markersize=3.4 if stage == "refined" else 0,
                    markerfacecolor=arm_spec["refined_color"],
                    markeredgecolor="white",
                    markeredgewidth=0.55,
                    markevery=max(8, len(values) // 10),
                    alpha=0.98,
                    zorder=4 if stage == "raw" else 3,
                )
                cutoff_values[(stage, arm)] = success_rate(values)

        axis.axvline(2.0, color="#BAC5D1", linewidth=1.2, linestyle=(0, (3, 4)), zorder=0)
        for arm, arm_spec in ARMS.items():
            axis.scatter(
                2.0,
                cutoff_values[("refined", arm)],
                s=48,
                marker=arm_spec["marker"],
                color=arm_spec["refined_color"],
                edgecolor="white",
                linewidth=0.9,
                zorder=5,
            )

        summary_lines = ["At RMSD < 2 Å   raw → adaptive"]
        for arm, arm_spec in ARMS.items():
            raw_value = cutoff_values[("raw", arm)]
            adaptive_value = cutoff_values[("refined", arm)]
            summary_lines.append(
                f"{arm_spec['label']}:  {raw_value:4.1f} → {adaptive_value:4.1f}%"
            )
        axis.text(
            0.975,
            0.055,
            "\n".join(summary_lines),
            transform=axis.transAxes,
            ha="right",
            va="bottom",
            fontsize=9.2,
            color="#35475B",
            linespacing=1.35,
            bbox={
                "boxstyle": "round,pad=0.48",
                "facecolor": "#F7F9FB",
                "edgecolor": "#D7E0E9",
                "linewidth": 0.8,
                "alpha": 0.96,
            },
            zorder=6,
        )
        axis.set_title(title, loc="left", color="#293A50")
        axis.set_xlim(0, x_max)
        axis.set_ylim(0, 100)
        axis.xaxis.set_major_locator(MaxNLocator(nbins=8, steps=[1, 2, 2.5, 5, 10]))
        axis.set_yticks([0, 20, 40, 60, 80, 100])
        axis.grid(axis="y", color="#DEE6EE", linewidth=0.9)
        axis.set_axisbelow(True)
        axis.tick_params(colors="#566779")
        axis.set_xlabel("RMSD threshold (Å)")

    axes[0].set_ylabel("Cumulative complexes below threshold (%)")
    curve_handles = []
    for arm_spec in ARMS.values():
        for stage in ("raw", "refined"):
            stage_spec = STAGES[stage]
            curve_handles.append(
                Line2D(
                    [0],
                    [0],
                    color=arm_spec[f"{stage}_color"],
                    linewidth=stage_spec["linewidth"],
                    linestyle=stage_spec["linestyle"],
                    marker=arm_spec["marker"] if stage == "refined" else None,
                    markersize=4.5 if stage == "refined" else 0,
                    markeredgecolor="white",
                    markeredgewidth=0.6,
                    label=f"{arm_spec['label']} — {stage_spec['label']}",
                )
            )
    figure.legend(
        handles=curve_handles,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.905),
        columnspacing=1.8,
        handlelength=2.8,
    )
    figure.suptitle(
        f"{metadata['title']} RMSD distributions (N={metadata['complexes']})",
        fontsize=17,
        fontweight="bold",
        color="#293A50",
        y=0.985,
    )
    figure.subplots_adjust(top=0.76, bottom=0.15, left=0.075, right=0.99, wspace=0.08)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=260, bbox_inches="tight", facecolor="white")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render(load_rmsds(args.input, args.dataset), args.dataset, args.output)


if __name__ == "__main__":
    main()
