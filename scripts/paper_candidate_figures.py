"""Candidate bottleneck tables and plots from the validated collector output."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmarks/results/paper"
NAMES = {
    "astex": "Astex",
    "posebusters": "PoseBusters",
    "phibench": "PhiBench",
    "foldbench": "FoldBench",
    "openbind": "OpenBind",
}


def main():
    data = json.loads((OUT / "candidate_metrics.json").read_text())
    rows = data["rows"]
    plt.rcParams.update(
        {"font.size": 10, "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False}
    )
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
            ("top1_lt2", "Top-1", "#0072B2"),
            ("top5_lt2", "Top-5", "#E69F00"),
            ("oracle_lt2", "Oracle-100", "#009E73"),
        ]
    ):
        axes[0].bar(
            x + (j - 1) * 0.24,
            [r[metric]["mean"] for r in mainrows],
            width=0.23,
            yerr=[r[metric]["sd"] for r in mainrows],
            capsize=2,
            label=label,
            color=color,
        )
    axes[0].set_ylabel("Complexes with RMSD <2 Å (%)")
    axes[0].set_title("Confidence selection versus candidate coverage", pad=32)
    axes[0].legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False)
    vals = [r["candidate_lt2"]["mean"] for r in mainrows]
    bars = axes[1].bar(
        x,
        vals,
        yerr=[r["candidate_lt2"]["sd"] for r in mainrows],
        capsize=2,
        color="#0072B2",
        width=0.65,
    )
    axes[1].bar_label(bars, labels=[f"{v:.1f}%" for v in vals], padding=8)
    axes[1].set_ylabel("Mean fraction of 100 poses with RMSD <2 Å (%)")
    axes[1].set_title("Near-native candidate density")
    for ax in axes:
        ax.set_xticks(x, NAMES.values(), rotation=25, ha="right")
        ax.set_ylim(0, 100 if ax == axes[0] else 45)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"04_candidate_bottleneck.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    lines = [
        "# Candidate generation and confidence selection",
        "",
        "RMSD-only endpoints: strict symmetry-aware no-alignment RMSD <2 Å. Three-repeat mean ± sample SD (%). "
        "Top-1/Top-5 rank all candidates by predicted RMSD, with stable candidate-index ties; no chirality filter. "
        "Oracle uses reference RMSD only as a post-hoc upper bound. Candidate fraction averages equally over complexes.",
        "",
        "| Dataset | Arm | Stage | Top-1 | Top-5 | Oracle | Candidate fraction |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        vals = " | ".join(
            f"{r[k]['mean']:.2f} ± {r[k]['sd']:.2f}"
            for k in ("top1_lt2", "top5_lt2", "oracle_lt2", "candidate_lt2")
        )
        lines.append(f"| {NAMES[r['dataset']]} | {r['arm']} | {r['stage']} | {vals} |")
    lines += [
        "",
        "## Manuscript interpretation",
        "",
        "For refined unguided N100/S10, the oracle-to-Top-1 gaps are:",
        "",
    ]
    for r in mainrows:
        lines.append(
            f"- {NAMES[r['dataset']]}: {r['oracle_lt2']['mean'] - r['top1_lt2']['mean']:.2f} percentage points; "
            f"mean {r['candidate_lt2']['mean']:.2f} near-native candidates per 100."
        )
    lines += [
        "",
        "These gaps identify selection headroom in the saved banks, not guaranteed gains from another confidence model. "
        "Repeated seeds are sampling repeats on the same targets, not independent target cohorts. "
        "The candidate fraction does not measure geometric diversity.",
        "",
        "**Figure 4.** Refined unguided N100/S10, five supplied-pocket cohorts. "
        "Left: ordinary confidence Top-1/Top-5 versus oracle coverage. Right: candidate near-native density. "
        "Whiskers are sample SD across three seeds. All endpoints are RMSD-only, not joint PB-valid.",
        "",
        "![Candidate bottleneck](04_candidate_bottleneck.png)",
        "",
        "## Unavailable endpoints",
        "",
        "Saved selected-pose PB records cover baseline and chirality-filtered Top-1 only; temporal records include the disclosed compatibility repairs. Full-bank PB-valid fraction, "
        "joint-PB Top-5, and joint-PB oracle are **not available** and were not replaced by fast geometry or chirality labels. "
        "Producing these requires additional official PB evaluations, not another inference run.",
        "",
        "## Reproducibility",
        "",
        "`scripts/collect_paper_candidate_metrics.py` produces `candidate_metrics.json` and `candidate_cases.csv`; "
        "`scripts/paper_candidate_figures.py` produces this table and PNG/PDF. "
        "The collector verifies every confidence summary/score CSV hash, candidate order and both cached vectors. "
        "Cohort count, stage/repeat IDs, finite values, strict threshold, stable ranking and fallback are checked.",
        "",
    ]
    (OUT / "CANDIDATES.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
