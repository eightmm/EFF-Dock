# External-model benchmark results

This directory is the single public location for compact supplied-pocket
benchmark results and figures. Raw structures, complete pose ensembles,
official PoseBusters per-pose tables, scheduler logs, third-party weights, and
installed environments are intentionally excluded from Git.

## Files

- `RESULTS.md`: concise human-readable comparison and completion status.
- `effdock_u70k_benchmark.json`: promoted EFF-Dock U70k result ledger.
- `pocket_only_executed_reruns.json`: three-repeat locally executed external-AI
  RMSD results.
- `posebusters_classical_paper_values.json`: deposited classical GOLD/Vina
  values and provenance.
- `TEMPORAL_LITERATURE.md` and `temporal_literature.json`: source-native
  PhiBench/FoldBench literature context and contract-aware side-by-side tables,
  with explicit non-comparability boundaries for EFF-Dock-derived cohorts.
- `foldbench_pocket_558.json`: public ledger for the completed 558-interface
  holo-pocket redocking campaign and its temporal slices.
- `phibench_u70k_top5.json`: endpoint-aligned Top-5 reanalysis of the U70k
  PhiBench-derived 203-system pose bank, including all five PoseBusters checks.
- `pocket_only_pb_valid_comparison.json`: generated Top-1 RMSD/PB-valid Joint
  comparison across EFF-Dock, local external-AI runs, and reported classical
  methods.
- `*.png`: rendered comparison plots generated from the JSON ledgers.

Every locally executed three-repeat row reports its individual values, mean,
and sample standard deviation (`ddof=1`). Missing or failed targets remain
denominator failures. Method-native generation, scoring/reranking, and
minimization contracts are recorded under `benchmarks/external_models/`.

## Local-only artifacts

- Runtime repositories, weights, caches, and environments:
  `benchmarks/external_models/runtime/`.
- Raw inference and official evaluation outputs:
  `outputs/benchmarks/external_models/`.
- Legacy top-level paths may remain as relative compatibility links because
  some third-party environments contain absolute installation prefixes.
