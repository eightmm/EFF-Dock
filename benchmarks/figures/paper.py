"""Render the 14 manuscript figures using only versioned numerical tables.

Run ``uv run python -m benchmarks.figures.paper --output outputs/paper_figures``.
This does not run docking, refinement, sequence search, or bootstrap sampling.
"""

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.patches import ConnectionPatch, Patch, Rectangle

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "benchmarks/results/paper"
NAMES = dict(
    astex="Astex Diverse Set",
    posebusters="PoseBusters v2",
    phibench="PhiBench",
    foldbench="FoldBench",
    openbind="OpenBind",
)
ALL_NAMES = dict(NAMES, validation="Validation")
COUNTS = dict(astex=85, posebusters=308, phibench=206, foldbench=558, openbind=925, validation=1076)
DESCRIPTORS = dict(
    heavy_atoms="Heavy atoms", rotatable_bonds="Rotatable bonds", fragments="Fragments"
)
FIELDS = DESCRIPTORS
COLORS = dict(top1="#8FB9D8", joint="#E8B395", oracle="#9AC9B4")
BIN_COLORS = ["#E8B395", "#B9C3CE", "#8FB9D8", "#B8A6CE"]
BINS = ["<30", "30–<70", "70–<90", "90–100"]
STATES = dict(
    coverage_failure=("Near-native pose generation failure", "#B9C3CE"),
    selection_failure=("Selection failure", "#E8B395"),
    physical_failure=("PB failure", "#B8A6CE"),
    success=("Success", "#9AC9B4"),
)


def category(value):
    return int(np.searchsorted([30, 70, 90], value, side="right"))


def save(fig, out, name, dpi=200):
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def style(ax):
    ax.set_ylim(-2, 103)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.grid(axis="y", color="#D9DFE5", linewidth=0.7)
    ax.set_axisbelow(True)


def bin_label(row):
    return f"{row['lower']}–{row['upper'] - 1}" if row["upper"] else f"≥{row['lower']}"


def complexity_panel(ax, rows, title):
    x = np.arange(len(rows))
    for key, label, linestyle in [
        ("oracle", "Oracle", ":"),
        ("top1", "Top-1", "-"),
        ("joint", "Top-1 + PB-valid", "--"),
    ]:
        means = [r[key]["mean"] if r[key] else np.nan for r in rows]
        sd = [r[key]["sd"] if r[key] else np.nan for r in rows]
        ax.errorbar(
            x,
            means,
            yerr=sd,
            color=COLORS[key],
            ecolor=tuple(0.75 * channel for channel in to_rgb(COLORS[key])),
            label=label,
            linestyle=linestyle,
            marker={"oracle": "^", "top1": "o", "joint": "s"}[key],
            markerfacecolor="white" if key == "top1" else COLORS[key],
            markersize=5 if key == "top1" else 3,
            capsize=3 if key == "top1" else 1.5,
            linewidth=1.9 if key == "top1" else 1.4,
        )
    ax.set_xticks(
        x,
        [
            f"{bin_label(r)}\nn={r['complexes']}{'*' if 0 < r['complexes'] < 5 else ''}"
            for r in rows
        ],
    )
    ax.set_title(title, fontweight="bold", loc="left")
    style(ax)


def shared_legend(fig, ax, columns=3):
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="outside lower center", ncol=columns, frameon=False, handlelength=3.5
    )


def cohort_panels():
    return plt.subplots(1, 5, figsize=(12.5, 3.2), sharey=True, layout="constrained")


def curve_panel(ax, row, specs, *, show_markers=True):
    x = np.array(row["n"])
    for key, label, color, ls in specs:
        mean = np.array(row[key]["mean"])
        sd = np.array(row[key]["sd"])
        baseline = key == "prefix_top1" or label == "Refined"
        oracle = key == "prefix_oracle"
        ax.plot(
            x,
            mean,
            label=label,
            color=color,
            ls=ls,
            lw=1.9 if baseline else 1.5,
            marker=None if oracle or not show_markers else "o" if baseline else "s",
            markevery=None if oracle else [9, 29, 49, 69, 89] if baseline else [19, 39, 59, 79, 99],
            markersize=4,
            markerfacecolor="white" if baseline else color,
            markeredgewidth=1,
            zorder=3 if baseline else 4,
        )
        ax.fill_between(
            x,
            np.maximum(mean - sd, 0),
            np.minimum(mean + sd, 100),
            color=color,
            alpha=0.18,
            linewidth=0,
            zorder=1,
        )
    style(ax)
    ax.set_xlim(1, 100)
    ax.set_xticks([1, 20, 40, 60, 80, 100])


def comparison(data, out):
    rows = data["rows"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.2), sharex=True, layout="constrained")
    for i, (ax, dataset, name) in enumerate(
        zip(axes, ("astex", "posebusters"), ("Astex Diverse Set", "PoseBusters v2"), strict=True)
    ):
        group = [r for r in rows if r["dataset"] == dataset]
        y = np.arange(len(group), dtype=float)
        local = np.array([r["source_type"] == "our_run" for r in group])
        y[~local] += 0.65
        joint = np.array([r["joint"]["mean"] for r in group])
        rmsd = np.array([r["rmsd_lt2"]["mean"] for r in group])
        colors = [
            "#8FB9D8"
            if r["method"] == "EFF-Dock"
            else "#E8B395"
            if r["source_type"] == "paper"
            else "#B9C3CE"
            for r in group
        ]
        ax.barh(y, joint, color=colors, height=0.62, edgecolor="white", linewidth=0)
        ax.barh(
            y,
            rmsd - joint,
            left=joint,
            height=0.62,
            facecolor=[
                tuple(0.18 * c + 0.82 for c in matplotlib.colors.to_rgb(color)) for color in colors
            ],
            edgecolor=colors,
            hatch="////",
            linewidth=0,
        )
        for metric, values in (("joint", joint), ("rmsd_lt2", rmsd)):
            ax.errorbar(
                values[local],
                y[local],
                xerr=[r[metric]["sd"] for r in group if r["source_type"] == "our_run"],
                fmt="none",
                ecolor="#333333",
                elinewidth=0.8,
                capsize=2,
                capthick=0.8,
                zorder=3,
            )
        ax.axhline(5.75, color="#B9C3CE", linewidth=0.7, linestyle="--")
        ax.text(100, 6.25, "Literature", ha="right", va="center", fontsize=9, color="#68737D")
        ax.set_yticks(y, [r["method"] for r in group])
        ax.invert_yaxis()
        ax.set_xlim(0, 100)
        ax.set_xlabel("Success rate (%)")
        ax.set_title(f"{chr(65 + i)}  {name}", loc="left", fontweight="bold")
        ax.grid(axis="x", color="#E1E5EA", lw=0.6)
        ax.set_axisbelow(True)
    fig.legend(
        handles=[
            Patch(facecolor="#B9C3CE", label="RMSD < 2 Å & PB-valid"),
            Patch(
                facecolor="#EEF1F3",
                edgecolor="#B9C3CE",
                hatch="////",
                label="RMSD < 2 Å & PB-invalid",
            ),
        ],
        ncol=2,
        loc="outside lower center",
        frameon=False,
    )
    save(fig, out, "01_model_comparison")


def stages_and_pocket(result, out):
    colors = ["#8FB9D8", "#E8B395", "#9AC9B4", "#B8A6CE"]

    fig, ax = plt.subplots(figsize=(10, 5.2))
    fig.subplots_adjust(left=0.075, right=0.99, top=0.98, bottom=0.25)
    x = np.arange(5)
    for j, (stage, policy, label) in enumerate(
        [
            ("raw", "baseline", "Raw"),
            ("raw", "filtered", "Raw + chirality"),
            ("refined", "baseline", "Refined"),
            ("refined", "filtered", "Refined + chirality"),
        ]
    ):
        rs = [
            next(
                r
                for r in result["rows"]
                if (r["dataset"], r["n"], r["guidance"], r["stage"], r["policy"])
                == (d, 100, "off", stage, policy)
            )
            for d in NAMES
        ]
        positions = x + (j - 1.5) * 0.2
        joint = np.array([r["joint"]["mean"] for r in rs])
        rmsd = np.array([r["rmsd_lt2"]["mean"] for r in rs])
        if np.any(joint > rmsd):
            raise ValueError("PB-valid RMSD success cannot exceed RMSD success")
        ax.bar(
            positions,
            joint,
            width=0.19,
            color=colors[j],
            edgecolor="white",
            linewidth=0,
            label=label,
        )
        ax.bar(
            positions,
            rmsd - joint,
            bottom=joint,
            width=0.19,
            facecolor=tuple(0.18 * c + 0.82 for c in to_rgb(colors[j])),
            edgecolor=colors[j],
            hatch="////",
            linewidth=0,
        )
        for metric, means in (("joint", joint), ("rmsd_lt2", rmsd)):
            ax.errorbar(
                positions,
                means,
                yerr=[r[metric]["sd"] for r in rs],
                fmt="none",
                ecolor="#333333",
                elinewidth=0.8,
                capsize=2,
                capthick=0.8,
                zorder=3,
            )
    ax.set_xticks(x, [f"{NAMES[d]}\n(n={COUNTS[d]})" for d in NAMES])
    ax.set_ylim(0, 100)
    ax.set_ylabel("Success rate (%)")
    fig.legend(
        *ax.get_legend_handles_labels(),
        ncol=4,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.085),
        frameon=False,
    )
    fig.legend(
        handles=[
            Patch(facecolor="#777777", label="RMSD < 2 Å & PB-valid"),
            Patch(
                facecolor="#E7E7E7",
                edgecolor="#777777",
                hatch="////",
                label="RMSD < 2 Å & PB-invalid",
            ),
        ],
        ncol=2,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        frameon=False,
    )
    save(fig, out, "02_refinement_chirality")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True, layout="constrained")
    for ax, dataset in zip(axes, ("astex", "posebusters"), strict=True):
        for j, (n, g, label) in enumerate(
            [
                (100, "off", "N100/S10 off"),
                (100, "eta2", "N100/S10 eta2"),
                (40, "off", "N40/S25 off"),
                (40, "eta2", "N40/S25 eta2"),
            ]
        ):
            rs = [
                next(
                    r
                    for r in result["rows"]
                    if (r["dataset"], r["n"], r["guidance"], r["stage"], r["policy"])
                    == (dataset, n, g, s, p)
                )
                for s, p in [
                    ("raw", "baseline"),
                    ("raw", "filtered"),
                    ("refined", "baseline"),
                    ("refined", "filtered"),
                ]
            ]
            ax.errorbar(
                range(4),
                [r["joint"]["mean"] for r in rs],
                yerr=[r["joint"]["sd"] for r in rs],
                label=label,
                marker="o" if n == 100 else "s",
                linestyle="--" if g == "eta2" else "-",
                color=colors[j],
                ecolor=tuple(0.75 * channel for channel in to_rgb(colors[j])),
                capsize=2,
            )
        ax.set_xticks(range(4), ["Raw", "Raw +\nchirality", "Refined", "Refined +\nchirality"])
        ax.set_title(NAMES[dataset])
        ax.set_ylim(40, 90)
    axes[0].set_ylabel("RMSD <2 Å and PB-valid (%)")
    fig.legend(
        *axes[0].get_legend_handles_labels(), ncol=4, loc="outside lower center", frameon=False
    )
    save(fig, out, "06_guidance_budget")

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout="constrained")
    for i, dataset in enumerate(("astex", "posebusters")):
        for j, (key, field, label) in enumerate(
            [
                (
                    "cutoff_sweep_fixed_sigma_2",
                    "pocket_cutoff_angstrom",
                    "Pocket cutoff (Å)",
                ),
                (
                    "prior_sigma_sweep_fixed_cutoff_10",
                    "prior_sigma_angstrom",
                    "Prior σ (Å)",
                ),
            ]
        ):
            entries = result["robustness"]["datasets"][dataset][key]
            values = np.array(
                [
                    [c["aggregate"]["joint_rmsd_lt2_pb_valid"]["mean"] for c in e["jitters"]]
                    for e in entries
                ]
            ).T
            ax = axes[i, j]
            im = ax.imshow(
                values,
                vmin=50,
                vmax=100,
                cmap=LinearSegmentedColormap.from_list(
                    "pocket_focus_70_80",
                    [
                        (0.0, "#364D7A"),
                        (0.3, "#547FA4"),
                        (0.4, "#75B5BB"),
                        (0.5, "#F4E9C8"),
                        (0.6, "#E49B75"),
                        (0.7, "#C26863"),
                        (1.0, "#803E65"),
                    ],
                ),
                aspect="auto",
            )
            ax.set_xticks(range(len(entries)), [e[field] for e in entries])
            ax.set_yticks(range(3), [0, 1, 2])
            ax.set_xlabel(label)
            ax.set_ylabel("Center jitter σ (Å)")
            if i == 0:
                ax.set_title(("A  Pocket size", "B  Prior scale")[j], loc="left", fontweight="bold")
            if j == 0:
                ax.set_ylabel(f"{NAMES[dataset]}\nCenter jitter σ (Å)")
            for y in range(values.shape[0]):
                for x in range(values.shape[1]):
                    rgb = np.array(im.cmap(im.norm(values[y, x]))[:3])
                    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
                    luminance = linear @ np.array([0.2126, 0.7152, 0.0722])
                    ax.text(
                        x,
                        y,
                        f"{values[y, x]:.1f}",
                        ha="center",
                        va="center",
                        color="black" if luminance > 0.179 else "white",
                    )
    fig.colorbar(
        im,
        ax=axes,
        label="RMSD < 2 Å & PB-valid (%)",
        ticks=[50, 60, 70, 75, 80, 90, 100],
        shrink=0.8,
    )
    for ax in axes.flat:
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("black")
            spine.set_linewidth(0.8)
    save(fig, out, "08_pocket_prior")


def complexity(data, out):
    fig = plt.figure(figsize=(12.5, 8.8), layout="constrained")
    row_figures = fig.subfigures(3, 1)
    row_titles = [
        "Ligand size dependence",
        "Rotatable bond dependence",
        "Fragment count dependence",
    ]
    for i, (descriptor, label) in enumerate(DESCRIPTORS.items()):
        row_figures[i].suptitle(
            f"{chr(65 + i)}  {row_titles[i]}", x=0.01, ha="left", fontsize=12, fontweight="bold"
        )
        axes = row_figures[i].subplots(1, 5, sharey=True)
        for j, (dataset, name) in enumerate(NAMES.items()):
            rows = [
                r
                for r in data["complexity"]
                if (r["dataset"], r["descriptor"]) == (dataset, descriptor)
            ]
            complexity_panel(axes[j], rows, name if i == 0 else "")
            axes[j].set_xlabel(label)
            if j == 0:
                axes[j].set_ylabel("Success (%)")
    shared_legend(fig, axes[0])
    save(fig, out, "03_ligand_complexity")


def cumulative(data, out):
    fig = plt.figure(figsize=(16, 7.5), layout="constrained")
    for subfig, (metric, title, xlabel) in zip(
        fig.subfigures(2, 1),
        [
            ("ranked_topk", "A  Confidence-ranked poses", "Ranked poses (k)"),
            ("prefix_oracle", "B  Generation-order poses", "Generated poses (N)"),
        ],
        strict=True,
    ):
        subfig.suptitle(title, x=0.01, ha="left", fontweight="bold", fontsize=13)
        axes = subfig.subplots(1, 5, sharey=True)
        for ax, (ds, name) in zip(axes, list(NAMES.items())[:5], strict=True):
            for stage, color in [("raw", "#A9B5C1"), ("refined", "#8FB9D8")]:
                row = next(r for r in data["curves"] if (r["dataset"], r["stage"]) == (ds, stage))
                curve_panel(
                    ax,
                    row,
                    [(metric, stage.title(), color, "--" if stage == "raw" else "-")],
                    show_markers=False,
                )
                other = next(r for r in data["curves"] if (r["dataset"], r["stage"]) == (ds, stage))
                assert (
                    abs(other["ranked_topk"]["mean"][-1] - other["prefix_oracle"]["mean"][-1])
                    < 1e-8
                )
            inset = ax.inset_axes([0.52, 0.12, 0.44, 0.39])
            for stage, color in [("raw", "#A9B5C1"), ("refined", "#8FB9D8")]:
                row = next(r for r in data["curves"] if (r["dataset"], r["stage"]) == (ds, stage))
                x = np.asarray(row["n"])
                mask = x <= 20
                mean = np.asarray(row[metric]["mean"])[mask]
                sd = np.asarray(row[metric]["sd"])[mask]
                inset.plot(x[mask], mean, color=color, ls="--" if stage == "raw" else "-", lw=1.4)
                inset.fill_between(
                    x[mask],
                    np.maximum(0, mean - sd),
                    np.minimum(100, mean + sd),
                    color=color,
                    alpha=0.18,
                    linewidth=0,
                )
            inset.set(xlim=(1, 20), ylim=(0, 103), xticks=[1, 10, 20], yticks=[0, 25, 50, 75, 100])
            inset.tick_params(labelsize=7, length=2, pad=1)
            inset.set_title("1–20 poses", fontsize=8, pad=3)
            for spine in inset.spines.values():
                spine.set_visible(True)
                spine.set_color("#87939E")
                spine.set_linewidth(0.6)
            inset.set_facecolor("white")
            inset.set_axisbelow(True)
            inset.grid(axis="y", color="#D3DCE4", linewidth=0.6)
            ax.add_patch(
                Rectangle(
                    (1, 0),
                    19,
                    103,
                    facecolor="#DCEAF3",
                    edgecolor="#91A6B6",
                    linewidth=0.7,
                    linestyle="--",
                    alpha=0.35,
                    zorder=0.5,
                )
            )
            for source_y, target_y in [(0, 0), (103, 1)]:
                ax.add_artist(
                    ConnectionPatch(
                        xyA=(20, source_y),
                        coordsA=ax.transData,
                        xyB=(0, target_y),
                        coordsB=inset.transAxes,
                        color="#91A6B6",
                        linewidth=0.65,
                        linestyle="--",
                        alpha=0.65,
                        zorder=0.8,
                    )
                )
            ax.set_title(name, fontweight="bold", fontsize=11)
            ax.set_xlabel(xlabel)
        axes[0].set_ylabel("Cumulative success (%)")
    fig.legend(
        *axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=2, frameon=False
    )
    save(fig, out, "04_cumulative_success", dpi=220)


def budget(data, out):
    specs = [
        ("prefix_oracle", "Oracle", "#9AC9B4", ":"),
        ("prefix_top1", "Top-1", "#8FB9D8", "-"),
        ("prefix_filtered_top1", "Top-1 + chirality", "#E8B395", "--"),
    ]
    for stage in ("refined",):
        fig, axes = cohort_panels()
        for panel, (ax, (dataset, name)) in enumerate(zip(axes, NAMES.items(), strict=True)):
            row = next(r for r in data["curves"] if (r["dataset"], r["stage"]) == (dataset, stage))
            curve_panel(ax, row, specs)
            ax.set_title(name, loc="left", fontweight="bold")
            ax.set_xlabel("Generated poses (N)")
        axes[0].set_ylabel("Success (%)")
        shared_legend(fig, axes[0])
        save(fig, out, "05_pose_budget")


def runtime(rows, out):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), layout="constrained")
    for j, (arm, label, color) in enumerate(
        [("unguided_n100_s10", "N100/S10", "#8FB9D8"), ("unguided_n40_s25", "N40/S25", "#E8B395")]
    ):
        rs = [
            next(r for r in rows if r["arm"] == arm and r["dataset"] == d)
            for d in ("astex", "posebusters")
        ]
        x = np.arange(2) + (j - 0.5) * 0.32
        bars = axes[0].bar(
            x,
            [r["pipeline_mean_s"] for r in rs],
            width=0.30,
            yerr=[r["pipeline_repeat_sd_s"] for r in rs],
            capsize=3,
            label=label,
            color=color,
        )
        axes[0].bar_label(bars, labels=[f"{r['pipeline_mean_s']:.1f}" for r in rs], padding=5)
        bars = axes[1].bar(
            x, [r["sampling_allocator_peak_gib"] for r in rs], width=0.30, label=label, color=color
        )
        axes[1].bar_label(
            bars, labels=[f"{r['sampling_allocator_peak_gib']:.1f}" for r in rs], padding=5
        )
    axes[0].set_ylabel("Wall time / complex (s)")
    axes[0].set_title("A  Runtime", loc="left", fontweight="bold")
    axes[1].set_ylabel("CUDA allocated peak (GiB)")
    axes[1].set_title("B  Memory", loc="left", fontweight="bold")
    for ax in axes:
        ax.set_xticks([0, 1], ["Astex Diverse Set", "PoseBusters v2"])
        ax.margins(y=0.2)
    fig.legend(
        *axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=2, frameon=False
    )
    save(fig, out, "07_runtime_memory")


def relatedness(result, out):
    labels = [
        "Same-sample overlap",
        "Separate matches",
        "Ligand only",
        "Sequence only",
        "Neither observed",
    ]
    colors = ["#B8A6CE", "#E8B395", "#9AC9B4", "#8FB9D8", "#B9C3CE"]
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(16, 5.5),
        sharey=True,
        layout="constrained",
        gridspec_kw={"width_ratios": [1.2, 1.2, 1]},
    )
    for y, (ds, name) in enumerate(ALL_NAMES.items()):
        counts = np.bincount(
            [category(r["max_sequence_identity"]) for r in result["records"] if r["dataset"] == ds],
            minlength=4,
        )
        overlaps = result["summary"][ds]["counts"]
        for ax, ns, cs in [
            (axes[0], counts, BIN_COLORS),
            (axes[1], [overlaps.get(k, 0) for k in labels], colors),
        ]:
            left = 0
            assert sum(ns) == COUNTS[ds]
            for n, color in zip(ns, cs, strict=True):
                width = 100 * n / COUNTS[ds]
                ax.barh(y, width, left=left, color=color, height=0.62)
                if width >= 12:
                    ax.text(left + width / 2, y, str(n), ha="center", va="center", fontsize=9)
                left += width
        n = overlaps.get("Same-sample overlap", 0)
        pct = 100 * n / COUNTS[ds]
        axes[2].barh(y, pct, color=colors[0], height=0.62)
        axes[2].text(pct + 1, y, f"{n}/{COUNTS[ds]} ({pct:.1f}%)", va="center", fontsize=9)
    axes[0].set_yticks(range(6), [f"{name}\n(n={COUNTS[ds]})" for ds, name in ALL_NAMES.items()])
    axes[0].invert_yaxis()
    for ax, title in zip(
        axes,
        ["A  Sequence identity", "B  Ligand–protein overlap", "C  Same training sample"],
        strict=True,
    ):
        ax.set_title(title, loc="left", fontweight="bold", fontsize=12)
        ax.set_xlim(0, 100)
        ax.set_xlabel("Complexes (%)")
    axes[0].legend(
        handles=[Patch(color=c, label=b + "%") for c, b in zip(BIN_COLORS, BINS, strict=True)],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=2,
        frameon=False,
        title="Sequence identity (%)",
    )
    axes[1].legend(
        handles=[Patch(color=c, label=l) for c, l in zip(colors, labels, strict=True)],
        loc="upper left",
        bbox_to_anchor=(0, -0.15),
        ncol=3,
        frameon=False,
    )
    save(fig, out, "09_training_relatedness", dpi=220)


def ligand_similarity(data, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    plt.rcParams["pdf.fonttype"] = 42
    x = np.arange(5)
    strata = [
        "exact_identity",
        "no_observed_exact_>=0.8",
        "no_observed_exact_0.5_to_<0.8",
        "no_observed_exact_<0.5",
    ]
    labels = [
        "Exact ligand identity",
        "T ≥0.8",
        "0.5 ≤T <0.8",
        "T <0.5",
    ]
    colors = ["#B8A6CE", "#E8B395", "#B1CDDF", "#9AC9B4"]
    bottom = np.zeros(5)
    counts = {ds: Counter(data["composition"][ds]) for ds in NAMES}
    for s, label, color in zip(strata, labels, colors, strict=True):
        values = np.array([100 * counts[d][s] / sum(counts[d].values()) for d in NAMES])
        axes[0].bar(x, values, bottom=bottom, color=color, label=label, width=0.65)
        bottom += values
    axes[0].set_ylabel("Benchmark complexes (%)")
    axes[0].set_title("A  Ligand similarity", loc="left", fontweight="bold")
    axes[0].legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.34), ncol=2, fontsize=8, frameon=False
    )
    for j, (stratum, label, color) in enumerate(
        [
            ("observed_overlap", "Exact match observed", "#B8A6CE"),
            ("no_observed_match", "No observed exact match", "#8FB9D8"),
        ]
    ):
        rows = [
            next(
                r
                for r in data["rows"]
                if (r["dataset"], r["stage"], r["policy"], r["axis"], r["stratum"])
                == (d, "refined", "filtered", "exact_train_ligand", stratum)
            )
            for d in NAMES
        ]
        bars = axes[1].bar(
            x + (j - 0.5) * 0.34,
            [r["joint"]["mean"] for r in rows],
            width=0.32,
            yerr=[r["joint"]["sd"] for r in rows],
            capsize=2,
            label=label,
            color=color,
        )
        axes[1].bar_label(
            bars, labels=[f"n={r['count_per_repeat']}" for r in rows], fontsize=7, padding=6
        )
    axes[1].set_ylabel("RMSD <2 Å and PB-valid (%)")
    axes[1].set_title("B  RMSD < 2 Å & PB-valid", loc="left", fontweight="bold")
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.34), fontsize=9, frameon=False)
    for ax in axes:
        ax.set_xticks(
            x,
            [f"{name}\n(n={sum(counts[ds].values())})" for ds, name in NAMES.items()],
            rotation=25,
            ha="right",
        )
        ax.set_ylim(0, 100)
        ax.spines[["top", "right"]].set_visible(False)
    save(fig, out, "10_ligand_similarity")


def sequence_similarity(result, performance, out):
    rows = result["records"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), sharey=True, layout="constrained")
    perf = []
    for ax, (ds, name), letter in zip(axes.flat, list(NAMES.items())[:5], "ABCDE", strict=False):
        counts = []
        for b, color in enumerate(BIN_COLORS):
            ids = {
                r["id"]
                for r in rows
                if r["dataset"] == ds and category(r["max_sequence_identity"]) == b
            }
            counts.append(len(ids))
            if not ids:
                continue
            entry = next(r for r in performance if r["dataset"] == ds and r["bin"] == BINS[b])
            if entry["n"] != len(ids):
                raise ValueError("Sequence performance denominator mismatch")
            mean, sd = np.array(entry["mean"]), np.array(entry["sd"])
            ax.bar(
                b,
                mean[0] - mean[1],
                bottom=mean[1],
                width=0.66,
                facecolor="#EFF2F5",
                edgecolor=color,
                hatch="////",
                linewidth=0,
            )
            ax.bar(b, mean[1], width=0.66, color=color, edgecolor="white", linewidth=0)
            for m, s in zip(mean, sd, strict=True):
                ax.errorbar(b, m, yerr=s, fmt="none", ecolor="#303030", capsize=3, elinewidth=1.25)
            perf.append(
                dict(dataset=ds, bin=BINS[b], n=len(ids), mean=mean.tolist(), sd=sd.tolist())
            )
        ax.set_title(f"{letter}  {name}", loc="left", fontweight="bold")
        ax.set_xticks(
            range(4), [f"{b}\nn={n}" for b, n in zip(BINS, counts, strict=True)], fontsize=10
        )
        ax.set_ylim(0, 105)
        ax.set_xlabel("Sequence identity (%)")
    axes[0, 0].set_ylabel("Success rate (%)")
    axes[1, 0].set_ylabel("Success rate (%)")
    axes[1, 2].axis("off")
    fig.legend(
        handles=[
            Patch(facecolor="#71869A", label="RMSD < 2 Å & PB-valid"),
            Patch(
                facecolor="#EFF2F5",
                edgecolor="#71869A",
                hatch="////",
                label="RMSD < 2 Å & PB-invalid",
            ),
        ],
        loc="outside lower center",
        ncol=2,
        frameon=False,
    )
    save(fig, out, "11_sequence_similarity", dpi=220)


def uncertainty(diag, out):
    contrasts = {
        "refinement_baseline": "Refined − raw, no chirality filter",
        "refinement_filtered": "Refined − raw, with chirality filter",
        "chirality_raw": "Chirality filter off → on, raw",
        "chirality_refined": "Chirality filter off → on, refined",
    }
    plt.rcParams.update(
        {"font.size": 10, "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False}
    )
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    for i, (contrast, title) in enumerate(contrasts.items()):
        ax = axes.flat[i]
        rows = [
            next(
                r
                for r in diag["contrasts"]
                if (r["dataset"], r["contrast"], r["metric"]) == (ds, contrast, "joint")
            )
            for ds in NAMES
        ]
        for j, (key, color, label, offset) in enumerate(
            [
                ("complex_bootstrap", "#8FB9D8", "Complex bootstrap", -0.12),
                ("exact_pdb_bootstrap", "#E8B395", "PDB-cluster bootstrap", 0.12),
            ]
        ):
            for y, r in enumerate(rows):
                est = r[key]
                if est is None:
                    continue
                low, high = est["ci95_pp"]
                center = est["delta_pp"]
                ax.errorbar(
                    center,
                    y + offset,
                    xerr=np.array([[center - low], [high - center]]),
                    fmt="o" if j == 0 else "s",
                    color=color,
                    markersize=4,
                    capsize=3,
                    label=label if y == 0 else None,
                )
        ax.axvline(0, color="#B9C3CE", linestyle="--", linewidth=0.8)
        ax.set_yticks(range(5), NAMES.values())
        ax.invert_yaxis()
        ax.set_xlabel("Success-rate improvement (percentage points)")
        ax.set_title(f"{chr(65 + i)}  {title}", loc="left", fontweight="bold")
    fig.legend(
        *axes.flat[0].get_legend_handles_labels(), loc="outside lower center", ncol=2, frameon=False
    )
    save(fig, out, "S1_paired_uncertainty")


def candidates(data, out):
    rows = data["rows"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    x = np.arange(5)
    mainrows = [
        next(
            r
            for r in rows
            if r["dataset"] == d
            and r["stage"] == "refined"
            and r["arm"] in ("unguided_n100_s10", "temporal_n100_s10")
        )
        for d in NAMES
    ]
    for j, (metric, label, color) in enumerate(
        [
            ("top1_lt2", "Top-1", "#8FB9D8"),
            ("top5_lt2", "Top-5", "#E8B395"),
            ("oracle_lt2", "Oracle-100", "#9AC9B4"),
        ]
    ):
        bars = axes[0].bar(
            x + (j - 1) * 0.24,
            [r[metric]["mean"] for r in mainrows],
            width=0.23,
            yerr=[r[metric]["sd"] for r in mainrows],
            capsize=2,
            label=label,
            color=color,
        )
        for bar, row in zip(bars, mainrows, strict=True):
            value = row[metric]["mean"]
            axes[0].text(
                bar.get_x() + bar.get_width() / 2,
                value + row[metric]["sd"] + 1.0,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=6.5,
                color="#333333",
            )
    axes[0].set_ylabel("Success (%)")
    axes[0].set_title("A  Selection", loc="left", fontweight="bold")
    fig.legend(
        *axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=3, frameon=False
    )
    vals = [r["candidate_lt2"]["mean"] for r in mainrows]
    bars = axes[1].bar(
        x,
        vals,
        yerr=[r["candidate_lt2"]["sd"] for r in mainrows],
        capsize=2,
        color="#8FB9D8",
        width=0.65,
    )
    axes[1].bar_label(bars, labels=[f"{v:.1f}%" for v in vals], padding=8)
    axes[1].set_ylabel("Near-native poses (%)")
    axes[1].set_title("B  Near-native density", loc="left", fontweight="bold")
    for ax in axes:
        ax.set_xticks(x, NAMES.values(), rotation=25, ha="right")
        ax.set_ylim(0, 108 if ax == axes[0] else 45)
    save(fig, out, "S2_candidate_ranking")


def failures(data, out):
    rows = data["rows"]
    fig = plt.figure(figsize=(12.5, 8.8), layout="constrained")
    row_figures = fig.subfigures(3, 1)
    row_titles = [
        "Ligand size dependence",
        "Rotatable bond dependence",
        "Fragment count dependence",
    ]
    for i, (field, title) in enumerate(FIELDS.items()):
        row_figures[i].suptitle(
            f"{chr(65 + i)}  {row_titles[i]}", x=0.01, ha="left", fontsize=12, fontweight="bold"
        )
        axes = row_figures[i].subplots(1, 5, sharey=True)
        for j, (dataset, name) in enumerate(NAMES.items()):
            ax = axes[j]
            group = [r for r in rows if (r["dataset"], r["descriptor"]) == (dataset, field)]
            x = np.arange(len(group))
            bottom = np.zeros(len(group))
            for state, (legend, color) in STATES.items():
                heights = np.array([r[state] if r[state] is not None else 0 for r in group])
                ax.bar(x, heights, bottom=bottom, color=color, width=0.72, label=legend)
                bottom += heights
            ax.set_xticks(x, [f"{bin_label(r)}\nn={r['complexes']}" for r in group])
            ax.set_ylim(0, 100)
            ax.set_yticks([0, 25, 50, 75, 100])
            ax.set_xlabel(title)
            ax.set_title(
                name if i == 0 else "",
                loc="left",
                fontweight="bold",
            )
            if j == 0:
                ax.set_ylabel("Fraction (%)")
            for k, r in enumerate(group):
                if not r["complexes"]:
                    ax.text(k, 45, "—", rotation=0, ha="center", va="center", color="#68737D")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False)
    save(fig, out, "S3_complexity_failures")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_repeats(value):
    if isinstance(value, dict):
        if (
            all(k in value for k in ("mean", "sd", "per_repeat"))
            and value["per_repeat"] is not None
        ):
            repeats = np.asarray(value["per_repeat"], dtype=float)
            require(repeats.shape[0] == 3 and np.isfinite(repeats).all(), "Invalid repeat table")
            require(
                np.allclose(repeats.mean(axis=0), value["mean"], atol=1e-9, rtol=0),
                "Repeat mean mismatch",
            )
            require(
                np.allclose(repeats.std(axis=0, ddof=1), value["sd"], atol=1e-9, rtol=0),
                "Sample SD mismatch",
            )
        for item in value.values():
            validate_repeats(item)
    elif isinstance(value, list):
        for item in value:
            validate_repeats(item)


def validate(data, bundle):
    """Cross-check shared endpoints, identities, denominators and frozen hashes."""
    manifest = json.loads((ROOT / "docs/paper/manifest.json").read_text())
    for row in manifest["numerical_inputs"]:
        path = ROOT / row["path"]
        require(
            hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"], f"Changed input: {path}"
        )
    validate_repeats(data)
    validate_repeats(bundle)
    main = bundle["benchmark_results"]["rows"]
    for row in main:
        require(
            row["complexes_per_repeat"] == COUNTS[row["dataset"]], "Benchmark denominator mismatch"
        )
        require(
            0 <= row["joint"]["mean"] <= row["rmsd_lt2"]["mean"] <= 100, "Invalid success endpoints"
        )
    for row in data["comparison"]["rows"]:
        require(row["count"] == COUNTS[row["dataset"]], "Comparison denominator mismatch")
        if row["method"] == "EFF-Dock":
            own = next(
                r
                for r in main
                if (r["dataset"], r["n"], r["guidance"], r["stage"], r["policy"])
                == (row["dataset"], 100, "off", "refined", "filtered")
            )
            for key in ("joint", "rmsd_lt2"):
                require(
                    np.allclose(row[key]["per_repeat"], own[key]["per_repeat"], atol=1e-9, rtol=0),
                    "Comparison/main condition mismatch",
                )
    complexity_rows = data["complexity_budget"]["complexity"]
    for ds in NAMES:
        for descriptor in DESCRIPTORS:
            require(
                sum(
                    r["complexes"]
                    for r in complexity_rows
                    if r["dataset"] == ds and r["descriptor"] == descriptor
                )
                == COUNTS[ds],
                "Complexity bins do not cover cohort",
            )
    for row in data["complexity_budget"]["curves"]:
        require(row["n"] == list(range(1, 101)), "Expected all prefixes 1–100")
        require(row["complexes"] == COUNTS[row["dataset"]], "Curve denominator mismatch")
        for key in ("ranked_topk", "prefix_oracle"):
            values = np.asarray(row[key]["per_repeat"])
            require(
                values.shape == (3, 100) and np.all(np.diff(values, axis=1) >= -1e-9),
                "Nonmonotone cumulative success",
            )
        require(
            np.allclose(
                np.asarray(row["ranked_topk"]["per_repeat"])[:, -1],
                np.asarray(row["prefix_oracle"]["per_repeat"])[:, -1],
                atol=1e-9,
                rtol=0,
            ),
            "Cumulative endpoints disagree",
        )
    for row, original in zip(data["complexity_failures"]["rows"], complexity_rows, strict=True):
        require(
            all(
                row[k] == original[k]
                for k in ("dataset", "descriptor", "lower", "upper", "complexes")
            ),
            "Failure bin mismatch",
        )
        if row["complexes"]:
            oracle, top1, joint = [original[k]["mean"] for k in ("oracle", "top1", "joint")]
            require(
                np.allclose(
                    [row[k] for k in STATES],
                    [100 - oracle, oracle - top1, top1 - joint, joint],
                    atol=1e-9,
                    rtol=0,
                ),
                "Failure partition mismatch",
            )
    membership = ROOT / "benchmarks/inputs/training_membership"
    with (membership / "samples.csv").open() as handle:
        samples = list(csv.DictReader(handle))
    require(
        len(samples) == len({r["sample_id"] for r in samples}) == 48386,
        "Training sample IDs duplicated/missing",
    )
    sets = {
        key: {r["sample_id"] for r in samples if r[key] == "1"}
        for key in (
            "docking_train_member",
            "confidence_train_member",
            "confidence_validation_member",
        )
    }
    metadata = json.loads((membership / "manifest.json").read_text())
    for key, members in sets.items():
        digest = hashlib.sha256(("\n".join(sorted(members)) + "\n").encode()).hexdigest()
        require(
            len(members) == metadata["sets"][key]["count"]
            and digest == metadata["sets"][key]["sorted_ids_sha256"],
            f"Training set mismatch: {key}",
        )
    require(
        len(sets["docking_train_member"] & sets["confidence_train_member"]) == 43067,
        "Training intersection mismatch",
    )
    rows = data["sequence"]["records"]
    require(
        len(rows) == len({(r["dataset"], r["id"]) for r in rows}) == 3158,
        "Sequence IDs duplicated/missing",
    )
    require(Counter(r["dataset"] for r in rows) == COUNTS, "Sequence cohort mismatch")
    for row in rows:
        require(
            np.isfinite(row["max_sequence_identity"]) and 0 <= row["max_sequence_identity"] <= 100,
            "Invalid sequence score",
        )
        for key in ("sequence_witness", "same_sample_witness"):
            require(
                row[key] is None or row[key] in sets["docking_train_member"],
                "Sequence witness outside docking train",
            )
    for ds in ALL_NAMES:
        require(
            Counter(r["status"] for r in rows if r["dataset"] == ds)
            == data["sequence"]["summary"][ds]["counts"],
            "Overlap category mismatch",
        )
    with (DATA / "selected_outcomes.csv").open() as handle:
        outcomes = list(csv.DictReader(handle))
    require(
        len(outcomes)
        == len(
            {tuple(r[k] for k in ("dataset", "id", "repeat", "stage", "policy")) for r in outcomes}
        )
        == 24984,
        "Outcome IDs duplicated/missing",
    )
    for row in main:
        if row["n"] != 100 or row["guidance"] != "off":
            continue
        for repeat in range(3):
            group = [
                r
                for r in outcomes
                if (r["dataset"], r["stage"], r["policy"], int(r["repeat"]))
                == (row["dataset"], row["stage"], row["policy"], repeat)
            ]
            require(len(group) == COUNTS[row["dataset"]], "Outcome denominator mismatch")
            for key in ("rmsd_lt2", "pb_valid", "joint"):
                value = 100 * sum(r[key] == "True" for r in group) / len(group)
                require(
                    abs(value - row[key]["per_repeat"][repeat]) < 1e-9,
                    "Outcomes/main table mismatch",
                )
    for entry in data["sequence_performance"]:
        ids = {
            r["id"]
            for r in rows
            if r["dataset"] == entry["dataset"]
            and BINS[category(r["max_sequence_identity"])] == entry["bin"]
        }
        require(len(ids) == entry["n"], "Sequence stratum mismatch")
        vals = []
        for repeat in range(3):
            group = [
                r
                for r in outcomes
                if r["dataset"] == entry["dataset"]
                and r["id"] in ids
                and int(r["repeat"]) == repeat
                and r["stage"] == "refined"
                and r["policy"] == "filtered"
            ]
            require(len(group) == len(ids), "Sequence outcome join mismatch")
            vals.append(
                [100 * sum(r[k] == "True" for r in group) / len(ids) for k in ("rmsd_lt2", "joint")]
            )
        require(
            np.allclose(np.mean(vals, axis=0), entry["mean"], atol=1e-9, rtol=0)
            and np.allclose(np.std(vals, axis=0, ddof=1), entry["sd"], atol=1e-9, rtol=0),
            "Sequence performance mismatch",
        )


def export_csv(value, path):
    """Export scalar inputs in long form with JSON Pointer paths."""
    with path.open("w") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["json_pointer", "value_json"])

        def visit(item, pointer):
            if isinstance(item, dict):
                for key, child in item.items():
                    visit(child, pointer + "/" + key.replace("~", "~0").replace("/", "~1"))
            elif isinstance(item, list):
                for index, child in enumerate(item):
                    visit(child, pointer + "/" + str(index))
            else:
                writer.writerow([pointer, json.dumps(item, ensure_ascii=False, allow_nan=False)])

        visit(value, "")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/paper_figures")
    parser.add_argument(
        "--check", action="store_true", help="Validate numerical inputs without rendering"
    )
    args = parser.parse_args()
    data = json.loads((DATA / "figure_data.json").read_text())
    bundle = {
        name: json.loads((DATA / f"{name}.json").read_text())
        for name in (
            "benchmark_results",
            "candidate_metrics",
            "overlap_summary",
            "runtime_comparison",
        )
    }
    validate(data, bundle)
    from benchmarks.analysis.verify_evidence import verify

    verify()
    from benchmarks.figures.trajectory import verify as verify_trajectory

    verify_trajectory()
    if args.check:
        print(
            "Verified figure inputs, repeat statistics, 24,984 outcomes, 3,158 relatedness records and training memberships"
        )
        return
    args.output.mkdir(parents=True, exist_ok=True)
    tasks = [
        (comparison, (data["comparison"],), 10),
        (stages_and_pocket, (bundle["benchmark_results"],), 10),
        (complexity, (data["complexity_budget"],), 10),
        (cumulative, (data["complexity_budget"],), 10),
        (budget, (data["complexity_budget"],), 10),
        (runtime, (bundle["runtime_comparison"],), 10),
        (relatedness, (data["sequence"],), 10),
        (ligand_similarity, (bundle["overlap_summary"],), 10),
        (sequence_similarity, (data["sequence"], data["sequence_performance"]), 11),
        (uncertainty, (data["uncertainty"],), 10),
        (candidates, (bundle["candidate_metrics"],), 10),
        (failures, (data["complexity_failures"],), 10),
    ]
    for render, arguments, font_size in tasks:
        with plt.rc_context(
            {
                "font.size": font_size,
                "pdf.fonttype": 42,
                "axes.spines.top": False,
                "axes.spines.right": False,
            }
        ):
            render(*arguments, args.output)
    from benchmarks.figures.evidence import render as render_evidence

    render_evidence(args.output)
    from benchmarks.figures.trajectory import render as render_trajectory

    render_trajectory(args.output)
    export_csv(dict(bundle, **data), args.output / "source_data.csv")
    print(f"Rendered 21 figures (PDF/PNG) and source_data.csv in {args.output}")


if __name__ == "__main__":
    main()
