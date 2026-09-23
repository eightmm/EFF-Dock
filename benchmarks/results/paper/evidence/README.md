# Saved-bank empirical evidence

Numerical sources for supplementary Figures S4–S9. No new docking inference or
training was run. The primary protocol and fixed analyses are in
[PROTOCOL.md](PROTOCOL.md); manuscript interpretation is in
[EVIDENCE.md](../../../../docs/paper/EVIDENCE.md).

| File | Content |
|---|---|
| `results.json` | Repeat summaries, density counts, pooled reliability sufficient statistics, stringent subsets, all 27 PB checks and grouped transitions; source-path SHA-256 inventory |
| `confidence_cases.csv` | 12,492 complex/repeat/stage banks: within-bank ranking, regret, selection outcomes, density, Brier score |
| `relatedness.csv` | 2,082 complex identities, maximum ligand similarity, exact-match flag and fixed AND subset membership |
| `pb_checks.csv` | 24,984 selected-pose evaluations, 27 Boolean PB checks each (0/1), retained candidate indices |
| `structures.json` | Three illustrative receptor-frame heavy-atom structures, bonds, complete ligand-contacting receptor chains, deterministic selection metadata and input hashes |
| `baseline_cases.csv` | 5,895 local baseline outcomes matched to EFF-Dock by complex and repeat |
| `baseline_uncertainty.json` | Twenty RMSD/PB-valid contrasts, paired complex and PDB-accession bootstrap intervals and source hashes |
| `structure_views/` | Six software WebGL captures and their input/renderer hashes; used by the lightweight PDF renderer |
| `model_cost.json` | Parameter-object counts from CPU-loaded release models, existing pipeline throughput and explicit training-time availability |

Verify and render the six added figures from these public tables:

```bash
uv run python -m benchmarks.analysis.verify_evidence
uv run python -m benchmarks.figures.evidence --output outputs/paper_figures
```

The unified `python -m benchmarks.figures.paper` command verifies and renders
all 20 figures. Package checks additionally verify the manifest, page order,
English captions and Prism ZIP. Counts, selected outcomes, threshold membership,
repeat aggregation and calibration denominators are checked against case-level
records and the original published tables. AUROC uses average ranks for ties;
average precision uses score-group step area, not trapezoidal PR integration.
Missing/undefined metrics are empty CSV fields or JSON null, never zero-filled.

Recollection requires the separately retained private result banks and the
historical ligand-relatedness table; these are not needed to render or verify
the publication tables. Source paths identify those records and do not imply
that raw pose banks are distributed in this repository:

```bash
uv run python -m benchmarks.analysis.evidence --root /path/to/full/results/checkout --output benchmarks/results/paper/evidence
uv run python -m benchmarks.analysis.baseline_uncertainty --root /path/to/full/results/checkout --output benchmarks/results/paper/evidence
uv run python -m benchmarks.analysis.model_cost
```

The first collector checks frozen candidate-ledger/score hashes and selected
outcomes before aggregation, aborting on drift or missing cohorts. The baseline
collector checks every native repeat result against the frozen comparison.
Public checksum identities are in the manuscript manifest. Parameter counting
loads the released models on CPU; throughput reuses earlier measurements.

## Rebuilding molecular views

Figure S8 embeds molecular captures at 400 dpi with vector labels. It uses [3Dmol.js cartoons](https://3dmol.org/doc/CartoonStyleSpec.html)
and ball-and-stick ligands. Only carbon colors differ between poses; other
elements retain the palette in `structure_views/manifest.json`. Display chains
are selected by any protein heavy atom within 5 Å of the crystal ligand, then
retained in full. This is a visualization selection, not a new pocket or metric.

The ordinary 20-figure renderer uses the three checked pocket PNG captures, so it needs
no browser or additional molecular-view dependencies. To regenerate the captures,
use the optional `py3Dmol==2.5.5` and `playwright==1.62.0` packages with Playwright
Chromium and its OS libraries. Download the pinned JavaScript file named below;
the capture command verifies its SHA-256 before loading it. Capture uses local
inputs and software SwiftShader WebGL, with no inference or GPU allocation.

```bash
curl -fL https://cdn.jsdelivr.net/npm/3dmol@2.5.5/build/3Dmol-min.js -o /tmp/effdock-3Dmol-min.js
uv run --with py3Dmol==2.5.5 --with playwright==1.62.0 python -m benchmarks.figures.structure_views --javascript /tmp/effdock-3Dmol-min.js
```

Camera and style settings are in `benchmarks/figures/structure_views.py`. The
manifest records browser/library versions and per-view checksums. Rebuilding on
a different software-rendering stack may change pixels, even with identical
coordinates; update capture and paper-manifest hashes deliberately after review.
Only pocket close-ups are shown. A and C retain their original Astex repeat-0
median selections. B is the largest eligible same-candidate RMSD improvement
across five cohorts and three repeats, an explicitly post-hoc extreme example
(see the protocol amendment). `rescue_candidates.json` records all 138 eligible
complex-repeat selections; `rescue_search_sources.json` records the 15 input
ledger checksums. Re-running `benchmarks.analysis.evidence` exports these audits
along with the structures. No new docking inference or PB evaluation is needed.
