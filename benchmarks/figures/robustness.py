"""Render corrected-Rw sensitivity results from verified aggregate tables."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.patches import Patch

from benchmarks.figures.labels import success_labels

NAMES = {"astex": "Astex Diverse Set", "posebusters": "PoseBusters v2"}
COUNTS = {"astex": 85, "posebusters": 308}
COLORS = ["#8FB9D8", "#E8B395"]
ROOT = Path(__file__).resolve().parents[2]


def save(fig, output, name):
    for suffix in ["pdf", "png"]:
        metadata = {"CreationDate": None, "ModDate": None} if suffix == "pdf" else None
        fig.savefig(output / f"{name}.{suffix}", dpi=240, bbox_inches="tight", metadata=metadata)
    plt.close(fig)


def cell(data, dataset, *, cutoff=10, sigma=2, jitter=0, n=100, guided=False, stage="refined"):
    matches = [
        r
        for r in data["rows"]
        if r["dataset"] == dataset
        and r["stage"] == stage
        and r["policy"] == "filtered"
        and all(
            r["condition"][k] == v
            for k, v in {
                "cutoff": cutoff,
                "sigma": sigma,
                "jitter": jitter,
                "n": n,
                "guided": guided,
            }.items()
        )
    ]
    if len(matches) != 1:
        raise ValueError("Missing or duplicated sensitivity cell")
    return matches[0]


def pocket_prior(data, output):
    # Keep a linear 50--100 scale; concentrate hue changes in the observed 70--80 band.
    cmap = LinearSegmentedColormap.from_list(
        "pocket_focus",
        [(0.0, "#717FA8"), (0.4, "#96BDD8"), (0.5, "#F7F0D9"), (0.6, "#E9AA92"), (1.0, "#AC6467")],
    )
    cmap.set_under("#59698F")
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.3), layout="constrained")
    image = None
    for i, (values, axis_label) in enumerate(
        [
            ([6, 8, 10, 12, 14], "Pocket radius (Å)"),
            ([1, 2, 4], "Translation prior σ (Å)"),
        ]
    ):
        for j, dataset in enumerate(NAMES):
            ax = axes[i, j]
            matrix = np.array(
                [
                    [
                        cell(
                            data, dataset, **({"cutoff": x} if i == 0 else {"sigma": x}), jitter=y
                        )["joint"]["mean"]
                        for x in values
                    ]
                    for y in [0, 1, 2]
                ]
            )
            image = ax.imshow(matrix, cmap=cmap, vmin=50, vmax=100, aspect="auto")
            for y in range(3):
                for x in range(len(values)):
                    ax.text(
                        x,
                        y,
                        f"{matrix[y, x]:.1f}",
                        ha="center",
                        va="center",
                        fontsize=10,
                        color="#20252C" if 60 <= matrix[y, x] <= 90 else "white",
                    )
            ax.set_xticks(range(len(values)), values)
            ax.set_yticks([0, 1, 2], ["0", "1", "2"])
            ax.set_xlabel(axis_label)
            if j == 0:
                ax.set_ylabel("Center jitter σ (Å per axis)")
            ax.set_title(
                f"{chr(65 + 2 * i + j)}  {NAMES[dataset]}\n(n={COUNTS[dataset]})",
                loc="left",
                fontweight="bold",
            )
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color("black")
                spine.set_linewidth(0.9)
    colorbar = fig.colorbar(
        image,
        ax=axes,
        shrink=0.84,
        fraction=0.035,
        ticks=[50, 60, 70, 75, 80, 90, 100],
        extend="min",
    )
    colorbar.set_label("RMSD <2 Å & PB-valid (%)")
    save(fig, output, "08_pocket_prior")


def guidance_budget(data, output):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True, layout="constrained")
    specs = [(100, False), (100, True), (40, False), (40, True)]
    for j, (ax, dataset) in enumerate(zip(axes, NAMES, strict=True)):
        for i, stage in enumerate(["raw", "refined"]):
            rows = [cell(data, dataset, n=n, guided=g, stage=stage) for n, g in specs]
            x = np.arange(4) + (i - 0.5) * 0.34
            joint = np.array([r["joint"]["mean"] for r in rows])
            rmsd = np.array([r["rmsd_lt2"]["mean"] for r in rows])
            color = COLORS[i]
            ax.bar(x, joint, width=0.30, color=color, linewidth=0)
            ax.bar(
                x,
                rmsd - joint,
                bottom=joint,
                width=0.30,
                facecolor=tuple(0.18 * c + 0.82 for c in to_rgb(color)),
                edgecolor=color,
                linewidth=0,
                hatch="////",
            )
            for metric, heights in [("joint", joint), ("rmsd_lt2", rmsd)]:
                ax.errorbar(
                    x,
                    heights,
                    yerr=[r[metric]["sd"] for r in rows],
                    fmt="none",
                    ecolor="#333333",
                    elinewidth=0.8,
                    capsize=2,
                    capthick=0.8,
                )
            for position, row in zip(x, rows, strict=True):
                success_labels(
                    ax,
                    position,
                    row["rmsd_lt2"]["mean"],
                    row["joint"]["mean"],
                    row["rmsd_lt2"]["sd"],
                    row["joint"]["sd"],
                )
        ax.set_xticks(
            np.arange(4),
            ["N100/S10\nUnguided", "N100/S10\nGuided", "N40/S25\nUnguided", "N40/S25\nGuided"],
        )
        ax.set_ylim(0, 112)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_title(
            f"{chr(65 + j)}  {NAMES[dataset]}\n(n={COUNTS[dataset]})", loc="left", fontweight="bold"
        )
        ax.grid(axis="y", color="#D9DFE5", linewidth=0.7)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Success rate (%)")
    fig.legend(
        handles=[
            Patch(facecolor=COLORS[0], label="Raw"),
            Patch(facecolor=COLORS[1], label="Refined"),
            Patch(facecolor="#B9C3CE", label="RMSD <2 Å & PB-valid"),
            Patch(
                facecolor="#F3F4F6",
                edgecolor="#B9C3CE",
                hatch="////",
                label="RMSD <2 Å & PB-invalid",
            ),
        ],
        ncol=2,
        frameon=False,
        loc="outside lower center",
    )
    save(fig, output, "06_guidance_budget")


def render(data, output):
    output.mkdir(exist_ok=True, parents=True)
    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    ):
        pocket_prior(data, output)
        guidance_budget(data, output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, default=ROOT / "benchmarks/results/paper/rw_robustness.json"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "docs/paper/figures")
    args = parser.parse_args()
    from benchmarks.analysis.rw_robustness import verify

    table = json.loads(args.data.read_text())
    verify(table)
    render(table, args.output)
