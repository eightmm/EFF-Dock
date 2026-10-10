"""Render the readout-ablation Supplementary figure from papers/data/readout_ablation.json."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from benchmarks.figures.labels import mean_label

ROOT = Path(__file__).resolve().parents[2]
ARMS = {"newton_euler": ("Newton–Euler readout", "#8FB9D8"), "fragment_direct": ("Direct fragment readout", "#E8B395")}
DARK = "#41434A"


def panel(ax, letter, title):
    ax.set_title(f"{letter}  {title}", loc="left", fontweight="bold", pad=10)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#E4E6EB", linewidth=0.7)


def render(data, out):
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4))
    ax = axes[0]
    panel(ax, "A", "Internal-validation rollout")
    for arm, (label, color) in ARMS.items():
        curve = data["curves"][arm]
        ax.plot([c["step"] / 1000 for c in curve], [c["success_2A"] for c in curve], "o-", color=color,
                label=label, lw=1.8, ms=4)
    ax.set_xlabel("Training updates (thousands)"), ax.set_ylabel("RMSD < 2 Å (%)")
    ax.set_ylim(0, None), ax.legend(frameon=False, fontsize=9)
    for k, (letter, ds, title) in enumerate((("B", "astex", "Astex Diverse Set"), ("C", "posebusters", "PoseBusters v2"))):
        ax = axes[k + 1]
        panel(ax, letter, f"{title} (unrefined)")
        keys = [("oracle10", "Oracle@10"), ("oracle100", "Oracle@100"), ("candidate_lt2", "Near-native\ncandidates")]
        x = np.arange(len(keys))
        for j, (arm, (label, color)) in enumerate(ARMS.items()):
            vals = [data["report"]["arms"][arm][ds][key] for key, _ in keys]
            pos = x + (j - 0.5) * 0.36
            ax.bar(pos, vals, 0.34, color=color, label=label)
            for p, v in zip(pos, vals, strict=True):
                mean_label(ax, p, v, 0, digits=1, fontsize=8)
        ax.set_xticks(x, [name for _, name in keys]), ax.set_ylabel("Percent"), ax.set_ylim(0, 110)
    fig.subplots_adjust(wspace=0.28)
    fig.savefig(out, bbox_inches="tight", metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(out.with_suffix(".png"), dpi=170, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "papers/data/readout_ablation.json")
    parser.add_argument("--output", type=Path, default=ROOT / "papers/assets/readout_ablation.pdf")
    args = parser.parse_args()
    with plt.rc_context({"font.size": 10, "axes.titlesize": 11, "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False}):
        render(json.loads(args.data.read_text()), args.output)


if __name__ == "__main__":
    main()
