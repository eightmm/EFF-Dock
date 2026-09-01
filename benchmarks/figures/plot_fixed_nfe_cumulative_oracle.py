#!/usr/bin/env python3
"""Plot fixed-NFE cumulative oracle success against generated pose count."""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
from pathlib import Path

SOURCE_SHA256 = {
    "1dba36b65339f6a1897b572ef13d82aa2e867894564eac8e9d685150ca4253b4",
    "d482832ad53a869b8187f714e29e73c39e3c170838217cb1023c14de9efca160",
}
DATASETS = {
    "astex": {"title": "Astex Diverse", "complexes": 85},
    "posebusters": {"title": "PoseBusters v2", "complexes": 308},
}
ARMS = {
    "s10_n100": {
        "poses": 100,
        "label": "S10 / N100",
        "color": "#84A7D7",
        "marker": "o",
    },
    "s25_n40": {
        "poses": 40,
        "label": "S25 / N40",
        "color": "#EEA178",
        "marker": "s",
    },
}
STAGES = {"raw": "A  Raw ODE + guidance", "refined": "B  Adaptive refinement"}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_curves(
    path: Path,
    dataset: str,
) -> dict[tuple[str, str], list[tuple[int, float]]]:
    if file_sha256(path) not in SOURCE_SHA256:
        raise ValueError("frozen cumulative-oracle CSV SHA-256 mismatch")
    expected = int(DATASETS[dataset]["complexes"])
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    curves: dict[tuple[str, str], list[tuple[int, float]]] = {}
    for stage in STAGES:
        for arm, spec in ARMS.items():
            selected = [
                row
                for row in rows
                if row["dataset"] == dataset
                and row["stage"] == stage
                and row["arm"] == arm
            ]
            points = [(int(row["k"]), float(row["sr_pct"])) for row in selected]
            if (
                len(points) != spec["poses"]
                or [point[0] for point in points] != list(range(1, spec["poses"] + 1))
                or any(int(row["complexes"]) != expected for row in selected)
                or any(not math.isfinite(point[1]) for point in points)
            ):
                raise ValueError(f"invalid curve inventory for {dataset}/{stage}/{arm}")
            curves[(stage, arm)] = points
    return curves


def render(
    curves: dict[tuple[str, str], list[tuple[int, float]]],
    dataset: str,
    output: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metadata = DATASETS[dataset]
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "axes.labelsize": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(12.8, 4.9), sharex=True, sharey=True)
    for axis, (stage, title) in zip(axes, STAGES.items(), strict=True):
        for arm, spec in ARMS.items():
            points = curves[(stage, arm)]
            x = [point[0] for point in points]
            y = [point[1] for point in points]
            axis.step(
                x,
                y,
                where="post",
                linewidth=3,
                color=spec["color"],
                label=spec["label"],
                zorder=2,
            )
            marked = [index for index, k in enumerate(x) if k in {1, 5, 10, 20, 40, 100}]
            axis.scatter(
                [x[index] for index in marked],
                [y[index] for index in marked],
                s=42,
                marker=spec["marker"],
                color=spec["color"],
                edgecolor="white",
                linewidth=0.9,
                zorder=3,
            )

        n100 = dict(curves[(stage, "s10_n100")])
        n40 = dict(curves[(stage, "s25_n40")])
        axis.axvline(40, color="#BAC5D1", linewidth=1.1, linestyle=(0, (3, 4)), zorder=0)
        axis.annotate(
            f"{n100[40]:.1f}",
            (40, n100[40]),
            xytext=(-7, -16),
            textcoords="offset points",
            ha="right",
            color=ARMS["s10_n100"]["color"],
            fontweight="bold",
        )
        axis.annotate(
            f"{n40[40]:.1f}",
            (40, n40[40]),
            xytext=(7, 8),
            textcoords="offset points",
            ha="left",
            color=ARMS["s25_n40"]["color"],
            fontweight="bold",
        )
        axis.annotate(
            f"{n100[100]:.1f}",
            (100, n100[100]),
            xytext=(-5, 8),
            textcoords="offset points",
            ha="right",
            color=ARMS["s10_n100"]["color"],
            fontweight="bold",
        )
        axis.set_title(title, loc="left", color="#293A50")
        axis.set_xlim(1, 100)
        axis.set_ylim(0, 100)
        axis.set_xticks([1, 10, 20, 40, 60, 80, 100])
        axis.set_yticks([0, 20, 40, 60, 80, 100])
        axis.grid(axis="y", color="#DEE6EE", linewidth=0.9)
        axis.set_axisbelow(True)
        axis.tick_params(colors="#566779")
        axis.set_xlabel("Number of generated poses (k)")
    axes[0].set_ylabel("Cumulative oracle SR: min RMSD < 2 Å (%)")
    axes[0].legend(frameon=False, loc="lower right")
    figure.suptitle(
        f"{metadata['title']} cumulative success (N={metadata['complexes']})",
        fontsize=17,
        fontweight="bold",
        color="#293A50",
        y=0.985,
    )
    figure.subplots_adjust(top=0.82, bottom=0.15, left=0.075, right=0.99, wspace=0.08)
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
    render(load_curves(args.input, args.dataset), args.dataset, args.output)


if __name__ == "__main__":
    main()
