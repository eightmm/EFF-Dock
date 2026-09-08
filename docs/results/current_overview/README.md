# Current benchmark overview

These figures are generated from completed, admitted EFF-Dock benchmark
reports. The in-progress full chemical-constraint campaign is intentionally
excluded.

## Figures

- `figures/pocket_cutoff_robustness.{png,pdf}`: docking ODE receptor cutoff
  6/8/10/12 Å, three repeats per condition, with the confidence and refinement
  crops fixed at 10 Å.
- `figures/effdock_benchmark_summary.{png,pdf}`: current U70k-selected EFF-Dock
  results across Astex Diverse, PoseBusters v2, PhiBench, FoldBench pocket, and
  auxiliary OpenBind cohorts.
- `figures/pocket_model_comparison.{png,pdf}`: supplied-pocket-only Top-1
  comparison. Solid fill is joint RMSD `<2 Å` and PB-valid success; hatching is
  the PB-invalid remainder of RMSD `<2 Å` success.
- `figures/pocket_cutoff_jitter_legacy_heatmap.{png,pdf}`: the complete legacy
  N80/S25 cutoff-by-center-jitter matrix. This is explicitly separated from
  the promoted U70k/N100/S10 production stack.
- `figures/pocket_cutoff_jitter_top1_heatmap.{png,pdf}`: promoted U70k Top-1
  RMSD `<2 Å` robustness across cutoff `6–14 Å`.
- `figures/pocket_cutoff_jitter_top1_joint_heatmap.{png,pdf}`: promoted U70k
  Top-1 joint RMSD `<2 Å` and official PB-validity robustness across cutoff
  `6–14 Å`.
- `figures/pocket_cutoff_jitter_top5_heatmap.{png,pdf}`: promoted U70k
  confidence-ranked Top-5 RMSD `<2 Å` selection headroom across cutoff
  `6–14 Å`.
- `figures/prior_sigma_jitter_top1_joint_heatmap.{png,pdf}`: fixed 10 Å-pocket
  prior-σ (`1/2/4 Å`) and center-jitter robustness for Top-1 joint RMSD `<2 Å`
  and official PB-validity.
- `figures/pocket_cutoff_jitter_complete.{png,pdf}`: retained compact overview.
  Its oracle panel is RMSD-only, not an official PB-valid oracle.

The matching CSV files contain the exact plotted values. Error bars are sample
standard deviations (`ddof=1`) for completed three-repeat cells. One-repeat and
paper-reported cells do not claim repeat variance.

## Source ledgers

- `outputs/benchmarks/effdock_pocket_cutoff_robustness_runs/cutoff-r3-production-20260901-r2/report.json`
- `outputs/benchmarks/effdock_pocket_cutoff_jitter_robustness_runs/production-jitter-r3-20260904-v3/report.json`
- `benchmarks/results/external_models/effdock_u70k_benchmark.json`
- `benchmarks/results/external_models/temporal_literature.json`
- `outputs/benchmarks/s50_raw_refined_confidence_temporal_external_runs/d97d5eb907acc485dfde4b7fcf88d87b4d5fd8576014d2cfb89dd0518b9c9bb4/report/summary.json`
- `benchmarks/results/external_models/pocket_only_pb_valid_comparison.json`

Regenerate with:

```bash
uv run python benchmarks/figures/plot_current_results_overview.py
uv run python benchmarks/figures/plot_complete_pocket_cutoff_jitter.py
uv run python benchmarks/figures/plot_pocket_cutoff_jitter_selected_metrics.py
uv run python benchmarks/figures/plot_prior_sigma_jitter_heatmap.py
```
