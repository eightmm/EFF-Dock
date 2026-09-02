# EFF-Dock benchmarks

This directory is the public home for benchmark-specific code. Reusable RMSD,
validity, pose-scoring, and aggregation primitives remain in
`src/effdock/evaluation` and `src/effdock/workflows`; dataset/model adapters,
launchers, and result-figure scripts live here.

```text
benchmarks/
├── effdock/               EFF-Dock robustness/evaluation campaigns
├── external_models/       model manifests, adapters, runners, evaluators
│   ├── docs/              protocols, audits, run records
│   ├── environments/      tracked uv locks; ignored local environments
│   ├── runtime/           ignored sources, weights, caches, logs
│   ├── tools/             environment and execution wrappers
│   └── slurm/             external-model install/inference/evaluation jobs
├── figures/               benchmark and presentation figure scripts
└── results/               compact public result artifacts
```

## Stable entry points

- EFF-Dock evaluation: `uv run eff-dock evaluate --help`
- EFF-Dock benchmark workflow: `uv run eff-dock benchmark --help`
- External baselines: [`external_models/README.md`](external_models/README.md)
- Result figures: [`figures/README.md`](figures/README.md)

Raw datasets and generated poses remain under ignored `data/` and
`outputs/benchmarks/` paths. External upstream repositories, weights, caches,
logs, and installed environments remain under the ignored
`external_models/runtime/` and `external_models/environments/` subtrees.

## GitHub release contract

The public repository includes enough information to reproduce and audit every
reported benchmark result:

- exact dataset manifest identity and coverage denominator;
- model/source revision and pinned environment manifests or lock files;
- inference, candidate-generation, selection, refinement, and evaluation
  settings, including seeds;
- compact per-seed and aggregate metrics, missing-target accounting, runtime
  summaries, and publication figures;
- commands or Slurm entry points that regenerate each admitted result.

Installed virtual/Conda environments, upstream repository checkouts, caches,
raw or private datasets, raw scheduler logs, large pose ensembles, machine
paths, secrets, and weights without verified redistribution rights are not
published. They are reproducible runtime material, not benchmark evidence.

## Compatibility paths

Historical commands and archived Slurm jobs use
`scripts/external_models`, `scripts/slurm/external_*`, `scripts/figures`, and
`configs/external_models.json`; local environments may also contain absolute
`external_models/` or `others/` prefixes. These are symlink aliases to the
canonical files in this directory. Do not create a second implementation or
runtime copy.
