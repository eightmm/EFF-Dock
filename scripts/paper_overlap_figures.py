"""Exposure-stratified report with explicit incomplete-train annotations."""

import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/paper/20260919"
NAMES = {
    "astex": "Astex",
    "posebusters": "PoseBusters",
    "phibench": "PhiBench",
    "foldbench": "FoldBench",
    "openbind": "OpenBind",
}


def main():
    local = OUT / "overlap_metrics.json"
    if local.exists():
        original = json.loads(local.read_text())
        metadata = dict(original["metadata"])
        metadata["train_invalid_smiles"] = [
            dict(reason=r["reason"]) for r in metadata["train_invalid_smiles"]
        ]
        metadata["validation_invalid_smiles"] = [
            dict(reason=r["reason"]) for r in metadata["validation_invalid_smiles"]
        ]
        data = {k: original[k] for k in ("rows", "per_repeat", "sources", "pb_sources")}
        data.update(metadata=metadata, composition={}, exposure={})
        for ds in NAMES:
            rs = [r for r in original["annotations"] if r["dataset"] == ds]
            data["composition"][ds] = dict(Counter(r["similarity_stratum"] for r in rs))
            data["exposure"][ds] = dict(
                count=len(rs),
                exact=sum(r["observed_train_ligand_identity"] for r in rs),
                pdb=None
                if all(r["exact_train_pdb_status"] == "missing" for r in rs)
                else sum(r["exact_train_pdb_status"] == "exact_match" for r in rs),
                validation_only_observed=sum(
                    r["observed_validation_ligand_identity"]
                    and not r["observed_train_ligand_identity"]
                    for r in rs
                ),
            )
        (OUT / "overlap_summary.json").write_text(json.dumps(data, indent=2))
    else:
        data = json.loads((OUT / "overlap_summary.json").read_text())
    metadata = data["metadata"]
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
        "No observed exact, T ≥0.8",
        "No observed exact, 0.5 ≤T <0.8",
        "No observed exact, T <0.5",
    ]
    colors = ["#CC79A7", "#E69F00", "#56B4E9", "#009E73"]
    bottom = np.zeros(5)
    counts = {ds: Counter(data["composition"][ds]) for ds in NAMES}
    for s, label, color in zip(strata, labels, colors, strict=True):
        values = np.array([100 * counts[d][s] / sum(counts[d].values()) for d in NAMES])
        axes[0].bar(x, values, bottom=bottom, color=color, label=label, width=0.65)
        bottom += values
    axes[0].set_ylabel("Benchmark complexes (%)")
    axes[0].set_title("Exposure to parseable training ligands")
    axes[0].legend(
        loc="upper center", bbox_to_anchor=(0.5, 1.36), ncol=2, fontsize=8, frameon=False
    )
    for j, (stratum, label, color) in enumerate(
        [
            ("observed_overlap", "Exact match observed", "#CC79A7"),
            ("no_observed_match", "No observed exact match", "#0072B2"),
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
    axes[1].set_title("Refined + chirality selection, unguided N100/S10")
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, 1.25), fontsize=9, frameon=False)
    for ax in axes:
        ax.set_xticks(x, NAMES.values(), rotation=25, ha="right")
        ax.set_ylim(0, 100)
        ax.spines[["top", "right"]].set_visible(False)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"06_training_exposure.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    lines = [
        "# Training exposure and stratified external performance",
        "",
        "**This is not a proof of an overlap-free benchmark.** It audits the preserved checkpoint split, not the later strict split builder.",
        "",
        f"Executed fine-tuning membership: {metadata['train_members']:,} train; preserved validation: {metadata['validation_members']:,} IDs. "
        f"The metadata contains {metadata['parseable_train_members']:,} parseable train entries and {len(metadata['train_invalid_smiles'])} unresolved train SMILES. "
        f"There are {metadata['unique_train_ligands']:,} unique parseable heavy-isomeric training ligands. "
        "The 47,277-system loader index was verified against the ordered preserved 47,310-system split and loader filters by cache-key reconstruction and a content hash. "
        "This is the S50 fine-tuning membership, not a union of all predecessor pretraining exposure.",
        "",
        "Exact identity uses RDKit canonical heavy-atom isomeric SMILES with explicit-H normalization, without tautomer/protonation standardization. "
        "Morgan radius-2 2,048-bit Tanimoto excludes chirality; nearest neighbors are over unique parseable training ligands. "
        "Exact identity takes priority over similarity bins. Bins (<0.5, 0.5–<0.8, ≥0.8) are descriptive, not tuned thresholds.",
        "",
        "Because one training metadata ligand is unresolved, **no observed match** means no match in the parseable index, "
        "not certified absence from all training inputs. Nearest Tanimoto is likewise relative to the parseable index. "
        "No benchmark cases were removed and no malformed ligand was repaired silently.",
        "",
        "| Dataset | Total | Observed exact train ligand | Exact train PDB accession | Observed val match but no observed train match |",
        "|---|---:|---:|---:|---:|",
    ]
    for ds, name in NAMES.items():
        exposure = data["exposure"][ds]
        count, exact = exposure["count"], exposure["exact"]
        pdb = "unavailable" if exposure["pdb"] is None else str(exposure["pdb"])
        val = exposure["validation_only_observed"]
        lines.append(f"| {name} | {count} | {exact} ({100 * exact / count:.2f}%) | {pdb} | {val} |")
    lines += [
        "",
        "## Exact-ligand strata: complete original cohorts",
        "",
        "Refined + chirality-filtered unguided N100/S10; percent mean ± sample SD over three seeds.",
        "",
        "| Dataset | Stratum | n/seed | RMSD <2 Å | PB-valid | Joint |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for r in data["rows"]:
        if (r["stage"], r["policy"], r["axis"]) != ("refined", "filtered", "exact_train_ligand"):
            continue
        vals = " | ".join(
            f"{r[k]['mean']:.2f} ± {r[k]['sd']:.2f}" for k in ("rmsd_lt2", "pb_valid", "joint")
        )
        lines.append(
            f"| {NAMES[r['dataset']]} | {r['stratum']} | {r['count_per_repeat']} | {vals} |"
        )
    lines += [
        "",
        "## Similarity strata",
        "",
        "| Dataset | Stratum | n/seed | RMSD <2 Å | PB-valid | Joint |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for r in data["rows"]:
        if (r["stage"], r["policy"], r["axis"]) != ("refined", "filtered", "ligand_similarity"):
            continue
        vals = " | ".join(
            f"{r[k]['mean']:.2f} ± {r[k]['sd']:.2f}" for k in ("rmsd_lt2", "pb_valid", "joint")
        )
        lines.append(
            f"| {NAMES[r['dataset']]} | {r['stratum']} | {r['count_per_repeat']} | {vals} |"
        )
    lines += [
        "",
        "## Protein/pocket boundary",
        "",
        "Exact receptor PDB-accession overlap is an exposure indicator, not sequence homology or pocket similarity. "
        "OpenBind has no benchmark per-case PDB accession and is explicitly missing. No validated external-to-training "
        "sequence-identity or pocket-similarity matrix was used; PLINDER-internal pocket-community labels cannot be "
        "assigned to external pockets without a separate mapping. In particular, ligand nonmatch does not establish pocket novelty.",
        "",
        "FoldBench has substantial exact ligand and PDB-accession exposure under this preserved split; it must not be "
        "described as uniformly unseen-target validation. All overlap strata are reported, including strata where overlap "
        "performs worse. Stratum differences are confounded by target/ligand composition and do not estimate a causal memorization effect.",
        "",
        "**Figure 6.** Left: benchmark composition by observed train-ligand identity and nearest parseable-train Morgan similarity. "
        "Right: selected-pose joint success by observed exact-ligand match, retaining full denominators. Error bars are "
        "three-seed sample SD. Unresolved training entries and three FoldBench PB compatibility-repair shards are disclosed above.",
        "",
        "![Training exposure](06_training_exposure.png)",
        "",
        "## Unresolved training metadata",
        "",
        "```json",
        json.dumps(metadata["train_invalid_smiles"], indent=2),
        "```",
        "",
        "Reproduce with `scripts/collect_paper_overlap.py` then `scripts/paper_overlap_figures.py`. "
        "The local-only `overlap_metrics.json` contains all 2,082 per-case annotations. "
        "Published [overlap_summary.json](overlap_summary.json) contains aggregate compositions, all four selection conditions by stratum, "
        "repeat values and source hashes; the figure script can use that summary without per-case data. Train and validation identity are never pooled.",
        "",
    ]
    (OUT / "OVERLAP.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
