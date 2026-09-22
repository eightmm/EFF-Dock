# Workflow helpers

Start with the `eff-dock` CLI for data preparation, docking/confidence training,
inference and benchmark evaluation. See [reproducibility](../docs/REPRODUCIBILITY.md)
for commands and released checkpoint identities.

| Purpose | Entry points |
|---|---|
| Development checks | `check.sh fast`, `check.sh ml-smoke` |
| Released weights | `verify_release.py`, `export_ema_inference_checkpoint.py` |
| Confidence banks | `prepare_s50_confidence_training_bank.py`, `refine_s50_confidence_pose_bank.py`, `materialize_s50_refined_confidence_bank.py` |
| Temporal evaluation | `run_external_temporal_benchmark_shard.py`, `report_external_temporal_benchmark.py` |
| Manuscript analysis | `paper_results.py`, `collect_paper_*`, `paper_*_figures.py` |
| Figure packaging | `package_paper_figures.py` |
| Cluster wrappers | [slurm/](slurm/README.md) |

Other retained helpers are dependencies of these workflows or their regression
tests. Slurm defaults and archived-input paths are site-specific. Full benchmark
inputs and saved pose banks must be supplied separately.

External model code, environments and job scripts are under
[benchmarks/external_models](../benchmarks/external_models/README.md). The old
`scripts/external_models`, `scripts/figures` and external Slurm aliases have
been retired from the public checkout; invoke their canonical paths instead.
Historical sweep launchers and their report-specific tests remain local.
