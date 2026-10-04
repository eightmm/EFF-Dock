# Numerical source data for the manuscript figures

All submitted EFF-Dock measurements and illustrations use corrected Rw.
Primary results come from the complete frozen-weight N100/S10 three-seed study;
`rw_report.json` is its campaign summary. Released docking/confidence weights
retain their documented historical training provenance. Pocket/prior and guidance/N40 figures now use complete Rw sensitivity runs;
the operator-comparison figure remains outside this submission.

The current 20 figures in [docs/paper](../../../docs/paper/README.md) can be
rendered from this checkout without checkpoints or private benchmark banks:

```bash
uv run python -m benchmarks.figures.paper --check
uv run python -m benchmarks.figures.paper --output outputs/paper_figures
```

For a figure-only CPU environment, use
`uv run --locked --only-group dev python -m benchmarks.figures.paper`.
The development group pins Matplotlib and omits the model's CUDA stack.
The renderer produces 20 PDF/PNG pairs plus `source_data.csv`, whose
`json_pointer` and `value_json` columns index the canonical JSON inputs.
Source filenames and LaTeX labels remain stable; PDF page order and working
figure IDs are defined by the [manifest](../../../docs/paper/manifest.json).

| Working figure | Inputs |
|---|---|
| 1 | `figure_data.json` → `comparison` |
| 2 | `benchmark_results.json` → current Rw rows |
| 3, 4, 5 | `figure_data.json` → `complexity_budget` |
| 6 | `rw_runtime.json` |
| 7 | `figure_data.json` → `sequence` |
| 8 | `overlap_summary.json` |
| 9 | `figure_data.json` → `sequence_performance`, joined to `sequence` |
| S1 | `figure_data.json` → `uncertainty` |
| S2 | `candidate_metrics.json` |
| S3 | `figure_data.json` → `complexity_failures` |
| S4–S9 | `evidence/` JSON and case-level CSV files |
| S10 | `trajectory/trace.json` and checked molecular captures |
| S11 | `evidence/selector_bottleneck.json` |

`selected_outcomes.csv` contains 24,984 rows: 2,082 external complexes × three
seeds × raw/refined × ordinary/chirality-filtered selection. RMSD/PB labels
refer to the same selected pose. Repeat indices are 0, 1 and 2, not seed
integers. Full denominators, exact-PDB groups and audited failure partitions
are retained; RMSD-only oracle labels are never substituted for PB labels.

`rw_runtime.json` aggregates all 6,246 Rw case-seed executions from 240 complete
pipeline shards. Runtime sums shard wall durations per complex within each
seed, then reports mean/sample SD over three seeds. Memory is the maximum initial generation/evaluation-process CUDA allocated
peak, not reserved or whole-pipeline memory. Astex/PoseBusters export candidates
without initial confidence scoring; PhiBench/FoldBench/OpenBind include raw
confidence in that process. Separate refinement/scoring processes are excluded
from the memory peak. Process profiles are retained and are not controlled
to isolate sampling-only cost.
Mixed GPU device counts and 720 source-log hashes are retained. This is not
a controlled hardware comparison or saturated throughput measurement.
`benchmarks.analysis.rw_runtime` re-collects these records when the immutable
private source logs are available. Older cost/ablation records remain historical
provenance and are not plotted in the current package.

Public [training membership](../../inputs/training_membership/README.md)
contains the executed 47,277-sample docking reference and separate confidence
set. The sequence analysis retains 3,158 query records and eligible training
witnesses. Primary selected outcomes, descriptors and similarity inputs are
unchanged by the submission consolidation.

The renderer verifies hashes, repeat statistics, cohort/stratum counts,
cumulative endpoints, failure partitions, training membership and witness
eligibility. It reconstructs headline and sequence-bin performance from
case-level outcomes. Molecular captures verify their exact source hashes,
frame times, camera and atom identity. Figure rendering does not run inference,
refinement, PoseBusters, sequence searches or new bootstrap sampling. The
separate fixed-complex Rw N1 illustration is documented in
[trajectory/](trajectory/README.md); it is not a new benchmark evaluation.

See [evidence and limits](../../../docs/paper/EVIDENCE.md) and
[figure captions](../../../docs/paper/FIGURE_CAPTIONS.md) for selection rules,
retrospective diagnostics, applicability domains and literature-only values.
Fonts/library versions can change pixels outside the pinned environment.

`rw_robustness.json` contains verified 24-condition Rw sensitivity aggregates, all three repeat values, fixed denominators and selection-ledger hashes. Recollect with `benchmarks.analysis.rw_robustness`; render with `benchmarks.figures.robustness`.
