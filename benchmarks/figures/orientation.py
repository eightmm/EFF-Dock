"""Render the separate, full-cohort orientation diagnostic from aggregate data."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[2]


def render(P):
    summary = json.loads((ROOT / "benchmarks/results/paper/orientation/summary.json").read_text())
    assert summary["status"] == "complete_with_supplement" and summary["completed_pb_n"] == 308
    P.mkdir(parents=True, exist_ok=True)
    cohorts = [
        (key, summary["cohorts"][key]["title"], None, None) for key in ("astex", "posebusters")
    ]
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    colors = ["#B6A4D5", "#80C6B5"]
    edge = "#59636F"
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.3), layout="constrained", sharey=True)
    for ax, (key, title, _, _) in zip(axs, cohorts):
        c = summary["cohorts"][key]
        subtitle = f"n = {c['n']}"
        ax.set_title(
            ("A" if key == "astex" else "B") + "  " + title + "\n" + subtitle,
            loc="left",
            weight="bold",
        )
        for stage, x in [("raw", 0), ("refined", 1)]:
            rm = next(
                r
                for r in c["metrics"]
                if (r["stage"], r["selector"], r["metric"]) == (stage, "filtered", "rmsd_success")
            )
            pv = next(
                r
                for r in c["metrics"]
                if (r["stage"], r["selector"], r["metric"])
                == (stage, "filtered", "pb_valid_success")
            )
            for j, arm_prefix in enumerate(["old", "new"]):
                xx = x + (j - 0.5) * 0.32
                total = rm[arm_prefix + "_pct"]
                valid = pv[arm_prefix + "_pct"]
                ax.bar(xx, valid, width=0.29, color=colors[j], edgecolor=edge, linewidth=0.5)
                ax.bar(
                    xx,
                    total - valid,
                    bottom=valid,
                    width=0.29,
                    color=colors[j],
                    edgecolor=edge,
                    hatch="////",
                    linewidth=0.5,
                )
                ax.text(xx, total + 1.5, f"{total:.1f}", ha="center", va="bottom", fontsize=9)
                if total - valid > 2:
                    ax.text(
                        xx,
                        valid - 2,
                        f"{valid:.1f}",
                        ha="center",
                        va="top",
                        fontsize=8,
                        color="#35424A",
                    )
        ax.set(xticks=[0, 1], xticklabels=["Raw", "Refined"], ylim=(0, 105))
        ax.grid(axis="y", alpha=0.18)
        ax.set_axisbelow(True)
    axs[0].set_ylabel("Success rate (%)")
    fig.legend(
        handles=[
            Patch(facecolor=colors[0], label=r"Original $R^{\mathsf{T}}w$"),
            Patch(facecolor=colors[1], label=r"Corrected $Rw$"),
            Patch(facecolor="white", edgecolor=edge, hatch="////", label="PB-invalid"),
        ],
        loc="outside lower center",
        ncol=3,
        frameon=False,
    )
    fig.savefig(P / "Rw_comparison.pdf", metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(P / "Rw_comparison.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/orientation_figure")
    render(parser.parse_args().output)
