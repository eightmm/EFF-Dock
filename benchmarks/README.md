# Benchmarks

The public benchmark code contains model adapters, runners, evaluators and
compact result artifacts. Reusable metrics and EFF-Dock workflows live in
`src/effdock/evaluation` and `src/effdock/workflows`.

- EFF-Dock evaluation: `uv run eff-dock evaluate --help`
- EFF-Dock aggregation: `uv run eff-dock benchmark --help`
- [External model setup and execution](external_models/README.md)
- [Comparison plotting](figures/README.md)
- [Result artifacts](results/README.md)
- [Manuscript figures and captions](../docs/paper/README.md)

External model manifests, environment locks and runtime wrappers are retained.
Upstream repositories, weights, installed environments, pose ensembles and raw
logs are generated or supplied separately and are not distributed here.

Superseded EFF-Dock sweep campaigns, one-off recovery launchers and intermediate
plotting scripts are retained locally. The public checkout is not a complete
archive of every experiment performed during development. Retained historical
analysis helpers provide context and regression coverage, not production defaults.

Use `benchmarks/external_models` and `benchmarks/figures` directly. External
Slurm launchers are under `benchmarks/external_models/slurm`; duplicate aliases
under `scripts/` are no longer published. Resource defaults remain site-specific.

## Inputs and reference data

`inputs/` contains frozen cohort, eligibility and complex-input manifests used
by retained benchmark workflows. `reference/` contains literature comparison
values used by the plotting tools. These files are inputs, not current results.
Their bytes and checksums are unchanged by the directory reorganization;
historical provenance strings inside the JSON may retain their original paths.
