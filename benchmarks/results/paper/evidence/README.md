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
| `structures.json` | Three illustrative receptor-frame heavy-atom structures, bonds, nearby Cα traces, deterministic selection metadata and input hashes |
| `baseline_cases.csv` | 5,895 local baseline outcomes matched to EFF-Dock by complex and repeat |
| `baseline_uncertainty.json` | Twenty RMSD/PB-valid contrasts, paired complex and PDB-accession bootstrap intervals and source hashes |
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
