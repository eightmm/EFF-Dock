#!/usr/bin/env python3
"""Render PoseBusters v2 U50 Top-1 RMSD empirical CDFs."""

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
    "s10_n100": {"label": "10 steps × 100 poses", "color": "#84A7D7"},
    "s25_n40": {"label": "25 steps × 40 poses", "color": "#EEA178"},
}
STAGES = {"raw": "A  Raw ODE + guidance", "refined": "B  + adaptive refinement"}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_rmsds(path: Path) -> dict[tuple[str, str], list[float]]:
    if file_sha256(path) != SOURCE_SHA256:
        raise ValueError("frozen selected-pose ledger SHA-256 mismatch")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["dataset"] == "posebusters"]
    if len(rows) != EXPECTED_COMPLEXES or len({row["id"] for row in rows}) != EXPECTED_COMPLEXES:
        raise ValueError("unexpected PoseBusters v2 complex inventory")
    result: dict[tuple[str, str], list[float]] = {}
    for stage in STAGES:
        for arm in ARMS:
            values = sorted(float(row[f"{arm}_{stage}_selected_rmsd"]) for row in rows)
            if len(values) != EXPECTED_COMPLEXES or any(
                not math.isfinite(value) or value < 0 for value in values
            ):
                raise ValueError(f"invalid RMSD distribution for {stage}/{arm}")
            result[(stage, arm)] = values
    return result


def render(
    rmsds: dict[tuple[str, str], list[float]],
    output: Path,
    *,
    stacked: bool = False,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

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
    figure, axes = plt.subplots(
        2 if stacked else 1,
        1 if stacked else 2,
        figsize=(5.8, 8.4) if stacked else (12.8, 4.9),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    for axis, (stage, title) in zip(axes, STAGES.items(), strict=True):
        cutoff_values: dict[str, float] = {}
        for arm, spec in ARMS.items():
            values = rmsds[(stage, arm)]
            cumulative = [100.0 * index / len(values) for index in range(1, len(values) + 1)]
            axis.step(
                [0.0, *values],
                [0.0, *cumulative],
                where="post",
                linewidth=3,
                color=spec["color"],
                label=spec["label"],
                zorder=2,
            )
            cutoff_values[arm] = 100.0 * sum(value < 2.0 for value in values) / len(values)
        axis.axvline(2.0, color="#BAC5D1", linewidth=1.2, linestyle=(0, (3, 4)), zorder=0)
        first = cutoff_values["s10_n100"]
        second = cutoff_values["s25_n40"]
        blue_offset = 9 if first >= second else -17
        orange_offset = -17 if first >= second else 9
        axis.scatter(
            [2.0, 2.0],
            [first, second],
            s=44,
            color=[ARMS["s10_n100"]["color"], ARMS["s25_n40"]["color"]],
            edgecolor="white",
            linewidth=0.9,
            zorder=4,
        )
        axis.annotate(
            f"{first:.1f}",
            (2.0, first),
            xytext=(-7, blue_offset),
            textcoords="offset points",
            ha="right",
            color=ARMS["s10_n100"]["color"],
            fontweight="bold",
        )
        axis.annotate(
            f"{second:.1f}",
            (2.0, second),
            xytext=(7, orange_offset),
            textcoords="offset points",
            ha="left",
            color=ARMS["s25_n40"]["color"],
            fontweight="bold",
        )
        axis.set_title(title, loc="left", color="#293A50")
        axis.set_xlim(0, 10.5)
        axis.set_ylim(0, 100)
        axis.set_xticks([0, 1, 2, 3, 4, 6, 8, 10])
        axis.set_yticks([0, 20, 40, 60, 80, 100])
        axis.grid(axis="y", color="#DEE6EE", linewidth=0.9)
        axis.set_axisbelow(True)
        axis.tick_params(colors="#566779")
        axis.set_xlabel("U50 Top-1 RMSD threshold (Å)")
    axes[0].set_ylabel("Cumulative complexes below threshold (%)")
    axes[0].legend(frameon=False, loc="lower right")
    figure.suptitle(
        "PoseBusters v2 U50 Top-1 RMSD distribution (N=308)",
        fontsize=17,
        fontweight="bold",
        color="#293A50",
    )
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
                '<div id="effdock-posebusters-rmsd-cdf">',
                '  <img class="wide-figure" alt="Empirical cumulative distribution of U50 Top-1 RMSD on PoseBusters v2 for 10-step 100-pose and 25-step 40-pose EFF-Dock sampling, before and after refinement."',
                f'       src="data:image/png;base64,{wide_encoded}">',
                '  <img class="stacked-figure" alt="Empirical cumulative distribution of U50 Top-1 RMSD on PoseBusters v2 for 10-step 100-pose and 25-step 40-pose EFF-Dock sampling, before and after refinement."',
                f'       src="data:image/png;base64,{stacked_encoded}">',
                "</div>",
                "<style>",
                "#effdock-posebusters-rmsd-cdf { width: 100%; background: transparent; }",
                "#effdock-posebusters-rmsd-cdf img { display: block; width: 100%; height: auto; }",
                "#effdock-posebusters-rmsd-cdf .stacked-figure { display: none; }",
                "@media (max-width: 600px) {",
                "  #effdock-posebusters-rmsd-cdf .wide-figure { display: none; }",
                "  #effdock-posebusters-rmsd-cdf .stacked-figure { display: block; }",
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
