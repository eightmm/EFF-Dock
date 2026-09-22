"""Build manuscript tables and static figures from frozen completed reports."""

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
B = ROOT / "outputs/benchmarks"
FACT = B / "astex_pb_unguided_r3_hostmatched_v2/factorial_report.json"
TEMP = B / "external_chirality_u70k_temporal_full_r3_v1/pb_inchi_compat_v1/three_seed_summary.json"
ROBUST = (
    B
    / "effdock_pocket_prior_robustness_extension_runs/production-cutoff14-sigma-r3-20260906-v1/report.json"
)
NAMES = {
    "astex": "Astex",
    "posebusters": "PoseBusters",
    "phibench": "PhiBench",
    "foldbench": "FoldBench",
    "openbind": "OpenBind",
}
COUNTS = dict(astex=85, posebusters=308, phibench=206, foldbench=558, openbind=925)


def read(path):
    return json.loads(path.read_text())


def source(path):
    return dict(
        path=str(path.relative_to(ROOT)), sha256=hashlib.sha256(path.read_bytes()).hexdigest()
    )


def portable_paths(value):
    """Publish repository-relative provenance rather than workstation paths."""
    if isinstance(value, dict):
        return {k: portable_paths(v) for k, v in value.items()}
    if isinstance(value, list):
        return [portable_paths(v) for v in value]
    if isinstance(value, str) and value.startswith(str(ROOT) + "/"):
        return str(Path(value).relative_to(ROOT))
    return value


def collect():
    fact, temporal, robust = read(FACT), read(TEMP), read(ROBUST)
    assert robust["status"] == "complete" and robust["total_selected_posebusters_errors"] == 0
    rows = list(fact["rows"])
    for dataset, stages in temporal.items():
        for stage, policies in stages.items():
            for policy, metrics in policies.items():
                row = dict(
                    dataset=dataset, n=100, steps=10, guidance="off", stage=stage, policy=policy
                )
                for name, m in metrics.items():
                    assert len(m["per_repeat_percent"]) == 3
                    row[name] = dict(
                        mean=m["mean_percent"],
                        sd=m["sample_sd_percent"],
                        per_repeat=m["per_repeat_percent"],
                    )
                rows.append(row)
    assert len(rows) == 44
    for r in rows:
        r["complexes_per_repeat"] = COUNTS[r["dataset"]]
        for k in ("rmsd_lt2", "pb_valid", "joint"):
            assert len(r[k]["per_repeat"]) == 3 and 0 <= r[k]["mean"] <= 100
        assert r["joint"]["mean"] <= min(r["rmsd_lt2"]["mean"], r["pb_valid"]["mean"]) + 1e-9
    return dict(rows=rows, robustness=robust, sources=[source(p) for p in (FACT, TEMP, ROBUST)])


def plot(result, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "savefig.dpi": 200,
        }
    )
    colors = ["#0072B2", "#E69F00", "#009E73", "#CC79A7"]

    def save(fig, name):
        for ext in ("png", "pdf"):
            fig.savefig(out / f"{name}.{ext}", bbox_inches="tight")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.8), layout="constrained")
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
        ax.bar(
            x + (j - 1.5) * 0.2,
            [r["joint"]["mean"] for r in rs],
            width=0.19,
            yerr=[r["joint"]["sd"] for r in rs],
            capsize=2,
            color=colors[j],
            label=label,
        )
    ax.set_xticks(x, [f"{NAMES[d]}\n(n={COUNTS[d]})" for d in NAMES])
    ax.set_ylim(0, 100)
    ax.set_ylabel("RMSD <2 Å and PB-valid (%)")
    ax.set_title("Unguided N100/S10: effect of refinement and chirality selection")
    ax.legend(ncol=2, loc="upper right", frameon=False)
    save(fig, "01_stage_ablation")

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
                capsize=2,
            )
        ax.set_xticks(range(4), ["Raw", "Raw +\nchirality", "Refined", "Refined +\nchirality"])
        ax.set_title(NAMES[dataset])
        ax.set_ylim(40, 90)
    axes[0].set_ylabel("RMSD <2 Å and PB-valid (%)")
    axes[1].legend(frameon=False, fontsize=8, loc="lower right")
    save(fig, "02_guidance_budget")

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout="constrained")
    for i, dataset in enumerate(("astex", "posebusters")):
        for j, (key, field, label) in enumerate(
            [
                (
                    "cutoff_sweep_fixed_sigma_2",
                    "pocket_cutoff_angstrom",
                    "Pocket cutoff (Å), prior σ=2 Å",
                ),
                (
                    "prior_sigma_sweep_fixed_cutoff_10",
                    "prior_sigma_angstrom",
                    "Prior σ (Å), pocket cutoff=10 Å",
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
            im = ax.imshow(values, vmin=0, vmax=100, cmap="cividis", aspect="auto")
            ax.set_xticks(range(len(entries)), [e[field] for e in entries])
            ax.set_yticks(range(3), [0, 1, 2])
            ax.set_xlabel(label)
            ax.set_ylabel("Center jitter σ per axis (Å)")
            ax.set_title(NAMES[dataset])
            for y in range(values.shape[0]):
                for x in range(values.shape[1]):
                    ax.text(
                        x,
                        y,
                        f"{values[y, x]:.1f}",
                        ha="center",
                        va="center",
                        color="black" if values[y, x] > 55 else "white",
                    )
    fig.colorbar(im, ax=axes, label="Joint success (%)", shrink=0.8)
    save(fig, "03_pocket_prior_robustness")


def write_report(result, out):
    lines = [
        "# Frozen external benchmark results",
        "",
        "Three-repeat means ± sample SD (%). U70k confidence; early-time S50 sampler; supplied pockets.",
        "These repeated-use external analyses do not select checkpoints or admit inference settings.",
        "",
        "## Uniform unguided N100/S10 main table",
        "",
        "Refinement plus input-SMILES chirality-only selection; fallback to original confidence Top-1 if all candidates fail.",
        "",
        "| Dataset | n/seed | RMSD <2 Å | PB-valid | Joint |",
        "|---|---:|---:|---:|---:|",
    ]
    for d in NAMES:
        r = next(
            r
            for r in result["rows"]
            if (r["dataset"], r["n"], r["guidance"], r["stage"], r["policy"])
            == (d, 100, "off", "refined", "filtered")
        )
        vals = " | ".join(
            f"{r[k]['mean']:.2f} ± {r[k]['sd']:.2f}" for k in ("rmsd_lt2", "pb_valid", "joint")
        )
        lines.append(f"| {NAMES[d]} | {COUNTS[d]} | {vals} |")
    lines += [
        "",
        "## All generation/postprocessing conditions",
        "",
        "| Dataset | N/S | Guidance | Stage | Selector | RMSD <2 Å | PB-valid | Joint |",
        "|---|---|---|---|---|---:|---:|---:|",
    ]
    for r in result["rows"]:
        vals = " | ".join(
            f"{r[k]['mean']:.2f} ± {r[k]['sd']:.2f}" for k in ("rmsd_lt2", "pb_valid", "joint")
        )
        lines.append(
            f"| {NAMES[r['dataset']]} | {r['n']}/{r['steps']} | {r['guidance']} | {r['stage']} | {r['policy']} | {vals} |"
        )
    lines += [
        "",
        "## Figure captions",
        "",
        "**Figure 1 — Stage ablation.** Official selected-pose joint success for unguided N100/S10. "
        "Bars are three-seed means; whiskers are sample SD, not confidence intervals. "
        "All original cases remain in denominators. PhiBench includes three reconstructed cases; "
        "OpenBind is an auxiliary single-protease cohort with quality/covalent flags. "
        "Three FoldBench PB shards use the disclosed energy-reference InChI compatibility repair.",
        "",
        "![Stage ablation](01_stage_ablation.png)",
        "",
        "**Figure 2 — Guidance and pose/step budget.** Same-budget guided/unguided initial priors were "
        "verified by exact hash for all 2,358 new complex-repeat records. N100/S10 and N40/S25 "
        "have equal learned pose-step count, not equal runtime; priors are not asserted nested across budgets. "
        "Refinement reduces the final difference between guided and unguided conditions.",
        "",
        "![Guidance and budget](02_guidance_budget.png)",
        "",
        "**Figure 3 — Existing pocket/prior robustness study.** Joint success means from the frozen "
        "guided-eta2 robustness campaign, refined/confidence-selected without the new chirality filter. "
        "Only the sampling crop changes; refinement/scoring retain 10 Å crops. Jitter σ is per Cartesian axis. "
        "These cells are NOT unguided results and cannot be substituted into Figure 1.",
        "",
        "![Pocket and prior](03_pocket_prior_robustness.png)",
        "",
        "## Source identities",
        "",
        "```json",
        json.dumps(result["sources"], indent=2),
        "```",
        "",
    ]
    (out / "RESULTS.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "benchmarks/results/paper")
    parser.add_argument(
        "--from-aggregate",
        action="store_true",
        help="Regenerate from published benchmark_results.json without local banks",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    result = portable_paths(
        read(args.output / "benchmark_results.json") if args.from_aggregate else collect()
    )
    (args.output / "benchmark_results.json").write_text(json.dumps(result, indent=2))
    plot(result, args.output)
    write_report(result, args.output)


if __name__ == "__main__":
    main()
