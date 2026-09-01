# EFF-Dock benchmarks

This directory is the public home for benchmark-specific code. Reusable RMSD,
validity, pose-scoring, and aggregation primitives remain in
`src/effdock/evaluation` and `src/effdock/workflows`; dataset/model adapters,
launchers, and paper-figure scripts live here.

```text
benchmarks/
├── external_models/       model manifests, adapters, runners, evaluators
│   └── slurm/             external-model install/inference/evaluation jobs
├── figures/               benchmark and presentation figure scripts
└── results/               compact, claim-bearing public result artifacts
```

## Stable entry points

- EFF-Dock evaluation: `uv run eff-dock evaluate --help`
- EFF-Dock benchmark workflow: `uv run eff-dock benchmark --help`
- External baselines: [`external_models/README.md`](external_models/README.md)
- Result figures: [`figures/README.md`](figures/README.md)

Raw datasets, generated poses, logs, checkpoints, and model-local upstream
repositories do not belong here. They remain under ignored `data/`, `outputs/`,
`external_models/`, and `others/` paths.

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

Historical commands and already-submitted Slurm jobs use
`scripts/external_models`, `scripts/slurm/external_*`, `scripts/figures`, and
`configs/external_models.json`. These are symlink aliases to the canonical
files in this directory. Keep them until all archived protocols and external
automation have migrated; do not create a second copy of an implementation.
