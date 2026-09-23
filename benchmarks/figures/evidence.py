"""Render saved-bank empirical evidence without private pose banks or new inference."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "benchmarks/results/paper/evidence"
NAMES = {
    "astex": "Astex Diverse Set",
    "posebusters": "PoseBusters v2",
    "phibench": "PhiBench",
    "foldbench": "FoldBench",
    "openbind": "OpenBind",
}
COLORS = ["#8FB9D8", "#E8B395", "#9AC9B4", "#B8A6CE", "#C7BD8C"]
DARK = "#41434A"


def save(fig, out, name, **pdf_options):
    fig.savefig(
        out / f"{name}.pdf",
        bbox_inches="tight",
        metadata={"CreationDate": None, "ModDate": None},
        **pdf_options,
    )
    fig.savefig(out / f"{name}.png", dpi=190, bbox_inches="tight")
    plt.close(fig)


def panel(ax, letter, title):
    ax.set_title(f"{letter}  {title}", loc="left", fontweight="bold", pad=12)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#E4E6EB", linewidth=0.7)


def ranking(data, out):
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 7.6))
    x = np.arange(5)
    lookup = {(r["dataset"], r["stage"]): r for r in data["confidence"]}
    specs = [
        ("A", "Within-bank ranking", "spearman", "Spearman ρ"),
        ("B", "Near-native discrimination", None, "Macro score"),
        ("C", "Selection regret", "filtered_regret", "Median RMSD regret (Å)"),
        (
            "D",
            "Selection given a successful candidate",
            "filtered_conditional_success",
            "Top-1 success (%)",
        ),
    ]
    for ax, (letter, title, key, ylabel) in zip(axes.flat, specs, strict=True):
        panel(ax, letter, title)
        for j in range(2):
            stage = "refined" if key is None else ("raw", "refined")[j]
            metric = ("auroc", "average_precision")[j] if key is None else key
            rows = [lookup[ds, stage][metric] for ds in NAMES]
            ax.bar(
                x + (j - 0.5) * 0.34,
                [r["mean"] for r in rows],
                0.32,
                color=COLORS[j],
                label=("AUROC", "Average precision")[j] if key is None else stage.capitalize(),
                yerr=[r["sd"] or 0 for r in rows],
                error_kw={"ecolor": DARK, "capsize": 3, "elinewidth": 1.3},
            )
        ax.set_xticks(
            x, [n.replace(" Diverse Set", "\nDiverse Set") for n in NAMES.values()], fontsize=9
        )
        ax.set_ylabel(ylabel)
        ax.legend(frameon=False, fontsize=9, ncol=2)
        if key in ("spearman", None):
            ax.set_ylim(0, 1.07)
        elif key == "filtered_conditional_success":
            ax.set_ylim(0, 112)
        else:
            ax.set_ylim(bottom=0)
    fig.subplots_adjust(wspace=0.23, hspace=0.4)
    save(fig, out, "S4_confidence_diagnostics")


def reliability(data, out):
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.3))
    panel(axes[0], "A", "Success-probability reliability")
    panel(axes[1], "B", "RMSD prediction reliability")
    panel(axes[2], "C", "Near-native candidate density")
    max_xy = 0
    for ds, color in zip(NAMES, COLORS, strict=True):
        for i, metric in enumerate(("success_probability", "predicted_rmsd")):
            rows = [
                r
                for r in data["reliability"]
                if r["dataset"] == ds and r["stage"] == "refined" and r["metric"] == metric
            ]
            axes[i].plot(
                [r["prediction"] for r in rows],
                [r["observed"] for r in rows],
                "o-",
                color=color,
                ms=4,
                lw=1.8,
            )
            if i == 1:
                max_xy = max(max_xy, max(max(r["prediction"], r["observed"]) for r in rows))
        rows = [r for r in data["density"] if r["dataset"] == ds and r["stage"] == "refined"]
        axes[2].errorbar(
            np.arange(5),
            [r["success"]["mean"] if r["success"]["mean"] is not None else np.nan for r in rows],
            yerr=[r["success"]["sd"] or 0 for r in rows],
            color=color,
            ecolor=DARK,
            lw=1.8,
            marker="o",
            ms=4,
            capsize=2,
            elinewidth=0.8,
        )
    for ax, lim in zip(axes[:2], (1, max_xy * 1.04), strict=True):
        ax.plot([0, lim], [0, lim], "--", color="#70727A", lw=1, zorder=0)
        ax.set(xlim=(0, lim), ylim=(0, lim))
    axes[0].set(xlabel="Predicted success probability", ylabel="Observed near-native fraction")
    axes[1].set(xlabel="Predicted RMSD (Å)", ylabel="Observed RMSD (Å)")
    axes[2].set(
        xlabel="Near-native poses per 100 candidates", ylabel="Top-1 success (%)", ylim=(-3, 103)
    )
    axes[2].set_xticks(range(5), ["0", "1–5", "6–20", "21–50", "51–100"])
    fig.legend(
        [Line2D([], [], color=c, marker="o", lw=1.8) for c in COLORS],
        list(NAMES.values()),
        loc="lower center",
        ncol=5,
        frameon=False,
        fontsize=9,
    )
    fig.subplots_adjust(wspace=0.32, bottom=0.23, top=0.86)
    save(fig, out, "S5_confidence_reliability")


def subset(data, out):
    fig, axes = plt.subplots(1, 5, figsize=(14, 4.4), sharey=True)
    for i, (ax, ds) in enumerate(zip(axes, NAMES, strict=True)):
        panel(ax, chr(65 + i), NAMES[ds])
        rows = [r for r in data["subsets"] if r["dataset"] == ds]
        for x, row in enumerate(rows):
            if not row["n"]:
                ax.text(x, 7, "Empty", ha="center", color=DARK, fontsize=10)
                continue
            total, joint = row["top1"]["mean"], row["joint"]["mean"]
            color = COLORS[0 if x == 0 else 2]
            ax.bar(x, joint, width=0.56, color=color)
            ax.bar(
                x,
                total - joint,
                bottom=joint,
                width=0.56,
                color=color,
                hatch="////",
                edgecolor="white",
                linewidth=0,
            )
            for key in ("top1", "joint"):
                ax.errorbar(
                    x,
                    row[key]["mean"],
                    yerr=row[key]["sd"],
                    color=DARK,
                    capsize=3,
                    elinewidth=1.1,
                    fmt="none",
                )
            ax.errorbar(
                x,
                row["oracle"]["mean"],
                yerr=row["oracle"]["sd"],
                color=DARK,
                capsize=3,
                fmt="D",
                ms=4,
                elinewidth=1.1,
            )
        ax.set_xticks([0, 1], [f"{r['subset']}\n(n={r['n']})" for r in rows])
        ax.set(xlim=(-0.65, 1.65), ylim=(0, 110))
    axes[0].set_ylabel("Success rate (%)")
    fig.legend(
        [
            Patch(facecolor=COLORS[0]),
            Patch(facecolor=COLORS[0], hatch="////", edgecolor="white"),
            Line2D([], [], color=DARK, marker="D", lw=0),
        ],
        ["RMSD <2 Å, PB-valid", "RMSD <2 Å, PB-invalid", "RMSD oracle"],
        ncol=3,
        loc="lower center",
        frameon=False,
    )
    fig.subplots_adjust(wspace=0.12, bottom=0.22, top=0.85)
    save(fig, out, "S6_stringent_subset")


def physical(data, out):
    metrics = [
        "Bond geometry",
        "Internal clash",
        "Receptor clash",
        "Stereochemistry",
        "Ring planarity",
        "Internal energy",
        "All PB checks",
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12.7, 8.7), layout="constrained")
    cmap = LinearSegmentedColormap.from_list(
        "failure", ["#F4F6FA", "#C2CFE5", "#8B92BE", "#615582"]
    )
    titles = [
        ("A", "Raw selected-pose failure", "primary_selection", "raw_fail"),
        ("B", "Refined selected-pose failure", "primary_selection", "refined_fail"),
        ("C", "Matched candidate: fail → pass", "matched_evaluated_candidate", "rescued"),
        ("D", "Matched candidate: pass → fail", "matched_evaluated_candidate", "harmed"),
    ]
    for ax, (letter, title, mode, metric) in zip(axes.flat, titles, strict=True):
        lookup = {(r["dataset"], r["metric"]): r for r in data["physical"] if r["mode"] == mode}
        mat = np.array([[lookup[ds, key][metric]["mean"] for ds in NAMES] for key in metrics])
        im = ax.imshow(mat, vmin=0, vmax=50, cmap=cmap, aspect="auto")
        ax.set_title(f"{letter}  {title}", loc="left", fontweight="bold", pad=10)
        ax.set_yticks(range(len(metrics)), metrics, fontsize=9)
        ax.set_xticks(
            range(5),
            [n.replace(" Diverse Set", "\nDiverse Set") for n in NAMES.values()],
            fontsize=9,
        )
        for y in range(len(metrics)):
            for x in range(5):
                ax.text(
                    x,
                    y,
                    f"{mat[y, x]:.1f}",
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="white" if mat[y, x] > 30 else DARK,
                )
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(DARK)
            spine.set_linewidth(0.7)
    fig.colorbar(im, ax=axes, shrink=0.7, label="Fraction of evaluated pairs (%)", pad=0.025)
    save(fig, out, "S7_physical_validity")


def structures(rows, out):
    from benchmarks.figures.structure_views import ELEMENT_COLORS, POSE_COLORS

    views = DATA / "structure_views"
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.6))
    for i, row in enumerate(rows):
        ax = axes[i]
        ax.imshow(plt.imread(views / f"{row['id']}_pocket.png"))
        ax.set_axis_off()
        ax.set_title(
            f"{chr(65 + i)}  {row['title']}",
            loc="left",
            fontweight="bold",
            pad=7,
        )
        label_box = dict(facecolor="white", edgecolor="none", alpha=0.86, pad=3)
        ax.text(
            0.035,
            0.965,
            row["id"].upper(),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=11,
            fontweight="bold",
            color=DARK,
            bbox=label_box,
        )
        detail = f"Selected RMSD: {row['selected_rmsd']:.2f} Å"
        if row["comparator_rmsd"] is not None:
            label = "Raw (same pose)" if i == 1 else "Oracle RMSD"
            detail += f"\n{label}: {row['comparator_rmsd']:.2f} Å"
        ax.text(
            0.035,
            0.035,
            detail,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=10,
            color=DARK,
            linespacing=1.35,
            bbox=label_box,
        )
    handles = [Line2D([], [], color=c, lw=4) for c in POSE_COLORS.values()]
    labels = ["Crystal carbon", "Selected carbon", "Raw / oracle carbon"]
    present = set(
        e for row in rows for key in POSE_COLORS if row[key] for e in row[key]["elements"]
    )
    for element, color in ELEMENT_COLORS.items():
        if element in present:
            handles.append(Line2D([], [], marker="o", color=color, linestyle="none", ms=6))
            labels.append(element)
    fig.legend(handles, labels, ncol=len(labels), frameon=False, loc="lower center", fontsize=10)
    fig.subplots_adjust(left=0.02, right=0.985, top=0.88, bottom=0.10, wspace=0.06)
    save(fig, out, "S8_structure_examples", dpi=400)


def baselines(data, out):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
    methods = list(dict.fromkeys(r["method"] for r in data["comparisons"]))
    for ax, ds, letter in zip(axes, ("astex", "posebusters"), "AB", strict=True):
        panel(ax, letter, NAMES[ds])
        for j, metric in enumerate(("rmsd_lt2", "joint")):
            rows = [
                next(
                    r
                    for r in data["comparisons"]
                    if r["dataset"] == ds and r["method"] == method and r["metric"] == metric
                )["pdb_bootstrap"]
                for method in methods
            ]
            point = np.array([r["delta_pp"] for r in rows])
            bounds = np.array([r["ci95_pp"] for r in rows]).T
            ax.errorbar(
                point,
                np.arange(len(methods)) + (j - 0.5) * 0.23,
                xerr=np.array([point - bounds[0], bounds[1] - point]),
                fmt=("o", "s")[j],
                color=COLORS[j],
                ecolor=DARK,
                elinewidth=1.1,
                capsize=3,
                ms=7,
                label=("RMSD success", "PB-valid success")[j],
            )
        ax.set_yticks(
            range(len(methods)),
            [m.replace("+", "+\n", 1) if len(m) > 20 else m for m in methods],
            fontsize=10,
        )
        ax.invert_yaxis()
        ax.axvline(0, color="#6A6A73", ls="--", lw=1)
        ax.set_xlabel("EFF-Dock − baseline (percentage points)")
        ax.set_xlim(-35, 75)
        ax.grid(axis="x", color="#E4E6EB", lw=0.7)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.2, top=0.87, wspace=0.7)
    save(fig, out, "S9_baseline_uncertainty")


def render(out):
    out.mkdir(parents=True, exist_ok=True)
    data = json.loads((DATA / "results.json").read_text())
    with plt.rc_context(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    ):
        for plot in (ranking, reliability, subset, physical):
            plot(data, out)
        structures(json.loads((DATA / "structures.json").read_text()), out)
        baselines(json.loads((DATA / "baseline_uncertainty.json").read_text()), out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/paper_figures")
    args = parser.parse_args()
    render(args.output)
    print(f"Rendered six empirical evidence figures in {args.output}")


if __name__ == "__main__":
    main()
