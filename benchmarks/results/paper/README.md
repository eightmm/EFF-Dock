# Numerical source data for the manuscript figures

These files reproduce the current 21 figures in
[docs/paper](../../../docs/paper/README.md) using a public checkout, without
raw structures, checkpoints, private pose banks or cluster directories.

```bash
uv run python -m benchmarks.figures.paper --check
uv run python -m benchmarks.figures.paper --output outputs/paper_figures
```

For a figure-only CPU environment, use
`uv run --locked --only-group dev python -m benchmarks.figures.paper`.
The development group pins Matplotlib; this form omits the model's CUDA stack.

The renderer produces 21 PDF/PNG pairs with the published names and a
`source_data.csv` containing the input values in long form. Its columns are
`json_pointer` and `value_json`: pointers index the named input tables/arrays,
and JSON scalars retain numbers, strings, booleans and explicit nulls. The
hierarchical JSON inputs remain the canonical numerical source.

| Figures | Inputs |
|---|---|
| 1 | `figure_data.json` → `comparison` |
| 2, 6, 8 | `benchmark_results.json` |
| 3, 4, 5 | `figure_data.json` → `complexity_budget` |
| 7 | `runtime_comparison.json` |
| 9 | `figure_data.json` → `sequence` (3,158 query records and training witnesses) |
| 10 | `overlap_summary.json` |
| 11 | `figure_data.json` → `sequence_performance`, joined to `sequence` |
| S1 | `figure_data.json` → `uncertainty` |
| S2 | `candidate_metrics.json` |
| S3 | `figure_data.json` → `complexity_failures` |
| S4–S9 | `evidence/` JSON and case-level CSV files |
| S10 | `trajectory/trace.json` and checked molecular captures |

`selected_outcomes.csv` contains 24,984 rows: 2,082 external complexes × three
seeds × raw/refined × ordinary/chirality-filtered selection. Boolean RMSD/PB
labels refer to the same selected pose. `repeat` is the paired repeat index
0, 1 or 2, not the generator's seed integer. `state` is the audited failure
partition. The CSV preserves full denominators and exact-PDB group IDs;
RMSD-only candidate-bank oracle labels are not substituted for PB labels.

`runtime_metrics.json` and `runtime_comparison.csv` preserve the existing cost
aggregates. Public [training membership](../../inputs/training_membership/README.md)
contains the 47,277-sample docking reference and the separate confidence set.
The [figure manifest](../../../docs/paper/manifest.json) binds numerical inputs,
figure pages and captions to hashes. `figure_data.json` retains original
aggregate hashes and local source identifiers; those paths are provenance,
not required external files for this renderer.

## Verification and limits

The renderer verifies input hashes, repeat means/sample SD, cohort/stratum
counts, cumulative endpoints, failure partitions, training ID hashes and
witness eligibility. It reconstructs the headline metrics and sequence-bin
performance from selected-outcome rows. At the earlier 14-figure release, all regenerated PNGs matched the
published images pixel-for-pixel in the pinned environment during release
verification; PDF bytes can differ through creation metadata.

This is figure reproduction from frozen numerical records. It does not rerun
model inference, refinement, official PoseBusters, nearest-neighbor sequence
search, or bootstrap sampling. Source values remain unchanged. The existing
collector scripts require separately obtained data/pose banks and are not an
end-to-end raw-data reproduction promise. Fonts/library versions can change
rendered pixels outside the pinned environment. Use the captions for which
conditions are main, guided, auxiliary or literature-reported.

The added Figures S4–S9 use [saved-bank evidence](evidence/README.md), with
[results and interpretation](../../../docs/paper/EVIDENCE.md). Original 14
figure files and their numerical sources remain unchanged. New source tables
are supplied alongside `source_data.csv`; that original CSV covers Figures
1–11 and S1–S3.

The 20-figure extension was also regenerated in the figure-only CPU environment
on 2026-09-23; all 20 PNGs matched the packaged previews pixel-for-pixel.
