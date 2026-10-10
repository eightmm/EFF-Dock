"""Render main Fig. 5: confidence ranking, eligibility-aware outcomes and joint headroom.

Inputs are the published numerical source data (papers/data/source_data.json) and
the all-candidate PoseBusters summary (papers/data/all_candidate_pb.json).
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from benchmarks.figures.labels import mean_label

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "papers/data"
NAMES = {
    "astex": "Astex Diverse Set",
    "posebusters": "PoseBusters v2",
    "phibench": "PhiBench-derived",
    "foldbench": "FoldBench",
    "openbind": "OpenBind",
}
COLORS = ["#8FB9D8", "#E8B395", "#9AC9B4", "#B8A6CE", "#C7BD8C"]
DARK = "#41434A"
STATES = [
    ("generation_failure", "No RMSD-successful candidate", "#7F8792"),
    ("chirality_exclusion", "Chirality exclusion", "#C0504D"),
    ("ranking_failure", "Ranking failure", "#E8B395"),
    ("pb_failure", "PoseBusters failure", "#7E57C2"),
    ("success", "RMSD–PoseBusters success", "#9AC9B4"),
]


def panel(ax, letter, title):
    ax.set_title(f"{letter}  {title}", loc="left", fontweight="bold", pad=12)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#E4E6EB", linewidth=0.7)


def xticks(ax, keys):
    short = {"astex": "Astex", "posebusters": "PoseBusters\nv2",
             "phibench": "PhiBench-\nderived", "foldbench": "FoldBench", "openbind": "OpenBind"}
    ax.set_xticks(np.arange(len(keys)), [short[k] for k in keys], fontsize=7.5)


def bars(ax, groups, series, ylabel, digits):
    x = np.arange(len(groups))
    width = 0.8 / len(series)
    for j, (label, rows) in enumerate(series):
        positions = x + (j - (len(series) - 1) / 2) * width
        ax.bar(positions, [r["mean"] for r in rows], width * 0.94, color=COLORS[j], label=label,
               yerr=[r["sd"] or 0 for r in rows], error_kw={"ecolor": DARK, "capsize": 3, "elinewidth": 1.2})
        for position, r in zip(positions, rows, strict=True):
            mean_label(ax, position, r["mean"], r["sd"], digits=digits, fontsize=7)
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False, fontsize=9, ncol=len(series), loc="upper right")


def render(out):
    values = json.loads((DATA / "source_data.json").read_text())["values"]
    joint = json.loads((DATA / "all_candidate_pb.json").read_text())["datasets"]
    conf = {(r["dataset"], r["stage"]): r for r in values["evidence"]["confidence"]}
    keys = list(NAMES)
    fig, axes = plt.subplots(2, 2, figsize=(9.4, 7.0))

    ax = axes[0, 0]
    panel(ax, "A", "Within-bank ranking")
    bars(ax, keys, [(s.capitalize(), [conf[k, s]["spearman"] for k in keys]) for s in ("raw", "refined")],
         "Spearman ρ", 2)
    ax.set_ylim(0, 1.2), ax.set_yticks([0, 0.25, 0.5, 0.75, 1]), xticks(ax, keys)

    ax = axes[0, 1]
    panel(ax, "B", "Near-native discrimination (refined)")
    bars(ax, keys, [(name, [conf[k, "refined"][m] for k in keys])
                    for name, m in (("AUROC", "auroc"), ("Average precision", "average_precision"))],
         "Mean per-bank score", 2)
    ax.set_ylim(0, 1.2), ax.set_yticks([0, 0.25, 0.5, 0.75, 1]), xticks(ax, keys)

    ax = axes[1, 0]
    panel(ax, "C", "Eligibility-aware outcome partition")
    rows = {(r["dataset"], r["stage"]): r for r in values["selector_bottleneck"]["rows"]}
    x = np.arange(len(keys))
    bottom = np.zeros(len(keys))
    for i, (state, label, color) in enumerate(STATES):
        heights = np.array([rows[k, "refined"]["mean"][i] for k in keys])
        ax.bar(x, heights, 0.62, bottom=bottom, color=color, label=label, edgecolor="white", linewidth=0.6)
        for xi, b, h in zip(x, bottom, heights, strict=True):
            if h >= 6:
                ax.text(xi, b + h / 2, f"{h:.1f}", ha="center", va="center", fontsize=8,
                        color="white" if state in ("generation_failure", "pb_failure") else DARK)
        bottom += heights
    ax.set_ylabel("Complexes (%)"), ax.set_ylim(0, 100), xticks(ax, keys)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.17))

    ax = axes[1, 1]
    panel(ax, "D", "Selected pose and bank oracles (refined)")
    two = ["astex", "posebusters"]
    series = [("Selected top-1", "top1_joint"), ("Joint oracle", "joint_oracle"), ("RMSD oracle", "rmsd_oracle")]
    x = np.arange(len(two))
    width = 0.26
    for j, (label, key) in enumerate(series):
        rs = [joint[k]["refined"][key] for k in two]
        positions = x + (j - 1) * width
        hatch = "//" if key == "rmsd_oracle" else None
        ax.bar(positions, [r["mean"] for r in rs], width * 0.94,
               color=[COLORS[2], COLORS[0], "white"][j], edgecolor=[COLORS[2], COLORS[0], DARK][j],
               hatch=hatch, label=label, yerr=[r["sd"] for r in rs],
               error_kw={"ecolor": DARK, "capsize": 3, "elinewidth": 1.2})
        for p, r in zip(positions, rs, strict=True):
            mean_label(ax, p, r["mean"], r["sd"], digits=1, fontsize=8)
    ax.set_xticks(x, [NAMES[k].replace(" Diverse Set", "\nDiverse Set") for k in two], fontsize=9)
    ax.set_ylabel("Complexes (%)"), ax.set_ylim(0, 115), ax.set_yticks([0, 25, 50, 75, 100])
    ax.legend(handles=[Patch(facecolor=COLORS[2], label="Selected top-1 (RMSD & PB)"),
                       Patch(facecolor=COLORS[0], label="Joint oracle (RMSD & PB)"),
                       Patch(facecolor="white", edgecolor=DARK, hatch="//", label="RMSD-only oracle")],
              frameon=False, fontsize=8, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.17))
    fig.subplots_adjust(wspace=0.26, hspace=0.42)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(out.with_suffix(".png"), dpi=170, bbox_inches="tight")
    plt.close(fig)


def render_regret(out):
    """Supplementary figure: regret and conditional selection (refined, primary selector)."""
    values = json.loads((DATA / "source_data.json").read_text())["values"]
    conf = {(r["dataset"], r["stage"]): r for r in values["evidence"]["confidence"]}
    rows = {(r["dataset"], r["stage"]): r for r in values["selector_bottleneck"]["rows"]}
    keys = list(NAMES)
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.6))
    ax = axes[0]
    panel(ax, "A", "Median regret to full-bank oracle")
    bars(ax, keys, [(s.capitalize(), [conf[k, s]["filtered_regret"] for k in keys]) for s in ("raw", "refined")],
         "Median RMSD regret (Å)", 2)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.2), xticks(ax, keys)
    ax = axes[1]
    panel(ax, "B", "Top-1 given an RMSD-successful candidate")
    bars(ax, keys, [(s.capitalize(), [conf[k, s]["filtered_conditional_success"] for k in keys])
                    for s in ("raw", "refined")], "Top-1 success (%)", 1)
    ax.set_ylim(0, 125), ax.set_yticks([0, 25, 50, 75, 100]), xticks(ax, keys)
    ax = axes[2]
    panel(ax, "C", "Regret within the effective eligible set")
    for k, color in zip(keys, COLORS, strict=True):
        cdf = rows[k, "refined"]["cdf"]
        ax.plot(cdf["rmsd_regret_angstrom"], cdf["mean"], color=color, lw=1.8, label=NAMES[k])
    ax.set_xlabel("Selected − eligible-oracle RMSD (Å)"), ax.set_ylabel("Cumulative complexes (%)")
    ax.set_xlim(0, 5), ax.set_ylim(0, 102), ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.subplots_adjust(wspace=0.3)
    fig.savefig(out, bbox_inches="tight", metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(out.with_suffix(".png"), dpi=170, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "papers/assets/confidence_diagnostics.pdf")
    parser.add_argument("--regret-output", type=Path, default=ROOT / "papers/assets/selection_regret.pdf")
    args = parser.parse_args()
    with plt.rc_context({"font.size": 10, "axes.titlesize": 11, "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False}):
        render(args.output)
        render_regret(args.regret_output)


if __name__ == "__main__":
    main()
