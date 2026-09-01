#!/usr/bin/env python3
"""Render PoseBusters v2 Top-1 and full-budget oracle RMSD CDFs."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import math
from pathlib import Path

SOURCE_SHA256 = "86ddf0da1f179d2afd702dbabdbb0de18d8ec68b76ea74a0f284c53aac508aaa"
EXPECTED_COMPLEXES = 308
ARMS = {
    "s10_n100": {
        "label": "S10 / N100",
        "short": "S10 / N100",
        "raw_color": "#3F6FAE",
        "refined_color": "#8EB5E3",
        "marker": "o",
    },
    "s25_n40": {
        "label": "S25 / N40",
        "short": "S25 / N40",
        "raw_color": "#B95F36",
        "refined_color": "#F0A477",
        "marker": "s",
    },
}
STAGES = {
    "raw": {"label": "Raw", "linestyle": (0, (5, 2.5)), "linewidth": 2.35},
    "refined": {"label": "Adaptive", "linestyle": "solid", "linewidth": 3.25},
}
PLOT_STAGE_ORDER = ("refined", "raw")
PANELS = {
    "selected": "A  U50 Top-1 selection",
    "oracle": "B  Full-budget oracle (100 vs 40 poses)",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_rmsds(path: Path) -> dict[tuple[str, str, str], list[float]]:
    if file_sha256(path) != SOURCE_SHA256:
        raise ValueError("frozen complex ledger SHA-256 mismatch")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["dataset"] == "posebusters"]
    if len(rows) != EXPECTED_COMPLEXES or len({row["id"] for row in rows}) != EXPECTED_COMPLEXES:
        raise ValueError("unexpected PoseBusters v2 complex inventory")

    result: dict[tuple[str, str, str], list[float]] = {}
    for metric in PANELS:
        for stage in STAGES:
            for arm in ARMS:
                column = f"{arm}_{stage}_{metric}_rmsd"
                values = sorted(float(row[column]) for row in rows)
                if len(values) != EXPECTED_COMPLEXES or any(
                    not math.isfinite(value) or value < 0 for value in values
                ):
                    raise ValueError(f"invalid RMSD distribution for {metric}/{stage}/{arm}")
                result[(metric, stage, arm)] = values
    return result


def success_rate(values: list[float], cutoff: float = 2.0) -> float:
    return 100.0 * sum(value < cutoff for value in values) / len(values)


def render(
    rmsds: dict[tuple[str, str, str], list[float]],
    output: Path,
    *,
    stacked: bool = False,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

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
    figure, axes = plt.subplots(
        2 if stacked else 1,
        1 if stacked else 2,
        figsize=(5.9, 9.2) if stacked else (13.0, 5.2),
        sharex=True,
        sharey=True,
    )

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
                    markevery=30,
                    alpha=0.98,
                    zorder=4 if stage == "raw" else 3,
                )
                cutoff_values[(stage, arm)] = success_rate(values)

        axis.axvline(2.0, color="#BAC5D1", linewidth=1.2, linestyle=(0, (3, 4)), zorder=0)
        for arm, arm_spec in ARMS.items():
            adaptive_value = cutoff_values[("refined", arm)]
            axis.scatter(
                2.0,
                adaptive_value,
                s=42,
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
                f"{arm_spec['short']}:  {raw_value:4.1f} → {adaptive_value:4.1f}%"
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
            zorder=5,
        )
        axis.set_title(title, loc="left", color="#293A50")
        axis.set_xlim(0, 10.5)
        axis.set_ylim(0, 100)
        axis.set_xticks([0, 1, 2, 3, 4, 6, 8, 10])
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
        ncol=4 if not stacked else 2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.905 if not stacked else 0.935),
        columnspacing=1.8,
        handlelength=2.8,
    )
    figure.suptitle(
        "PoseBusters v2 RMSD distributions (N=308)",
        fontsize=17,
        fontweight="bold",
        color="#293A50",
        y=0.985,
    )
    if stacked:
        figure.subplots_adjust(top=0.78, bottom=0.075, left=0.145, right=0.985, hspace=0.34)
    else:
        figure.subplots_adjust(top=0.76, bottom=0.15, left=0.075, right=0.99, wspace=0.08)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=260, bbox_inches="tight", facecolor="white")
    if not stacked:
        figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(figure)


def write_inline_html(wide_image: Path, stacked_image: Path, destination: Path) -> None:
    wide_encoded = base64.b64encode(wide_image.read_bytes()).decode("ascii")
    stacked_encoded = base64.b64encode(stacked_image.read_bytes()).decode("ascii")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        "\n".join(
            (
                '<div id="effdock-posebusters-top1-oracle-cdf">',
                '  <img class="wide-figure" alt="PoseBusters v2 RMSD cumulative distributions for U50 Top-1 selection and full-budget oracle, comparing raw guidance with adaptive refinement."',
                f'       src="data:image/png;base64,{wide_encoded}">',
                '  <img class="stacked-figure" alt="PoseBusters v2 RMSD cumulative distributions for U50 Top-1 selection and full-budget oracle, comparing raw guidance with adaptive refinement."',
                f'       src="data:image/png;base64,{stacked_encoded}">',
                "</div>",
                "<style>",
                "#effdock-posebusters-top1-oracle-cdf { width: 100%; background: transparent; }",
                "#effdock-posebusters-top1-oracle-cdf img { display: block; width: 100%; height: auto; }",
                "#effdock-posebusters-top1-oracle-cdf .stacked-figure { display: none; }",
                "@media (max-width: 600px) {",
                "  #effdock-posebusters-top1-oracle-cdf .wide-figure { display: none; }",
                "  #effdock-posebusters-top1-oracle-cdf .stacked-figure { display: block; }",
                "}",
                "</style>",
                "",
            )
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--html", type=Path)
    args = parser.parse_args()
    rmsds = load_rmsds(args.input)
    render(rmsds, args.output)
    stacked = args.output.with_name(f"{args.output.stem}-stacked{args.output.suffix}")
    render(rmsds, stacked, stacked=True)
    if args.html is not None:
        write_inline_html(args.output, stacked, args.html)


if __name__ == "__main__":
    main()
