# External-model benchmarks

This directory contains the reproducible adapter layer used to compare
EFF-Dock with external docking methods on Astex Diverse and PoseBusters v2.

The claim-bearing comparison is supplied-pocket-only. Adapters and outputs for
blind/full-receptor, pocket-prediction, or co-folding models are historical
non-comparison archives and are never included in result tables or figures.

## Layout

- `models.json`: pinned sources and pose/scoring/refinement taxonomy.
- `prepare_*.py`: audited input and receptor preparation.
- `run_*.py`: thin model-native inference adapters.
- `postprocess_*.py`: declared compatibility/minimization/reranking stages.
- `evaluate_*.py`, `aggregate_*.py`, `summarize_*.py`: coverage and metric
  admission.
- `slurm/`: installation, inference, native selection, and evaluation jobs.
- `docs/`: external-model protocols, run records, audits, and result notes.
- `tools/`: shared environment synchronization and execution wrappers.
- `environments/<model>/`: tracked uv project/lock files plus ignored local
  environments.
- `runtime/`: ignored upstream repositories, weights, binaries, caches, and
  logs.
- `../results/external_models/`: compact public JSON, tables, and figures.

The upstream repositories, installed environments, downloaded weights, caches,
and raw outputs are intentionally excluded from Git. Their reproducible
manifests and lock files are published; the populated runtime trees are not.
The populated installations now live under ignored `runtime/` and
`environments/` trees. Legacy `external_models/` and `others/` paths are
relative compatibility links because several pinned environments contain
absolute prefixes.

Every reported campaign must retain its dataset manifest, receptor policy,
model-native inference setting, candidate count, seed, selection stage,
coverage gate, and runtime accounting. The frozen protocol and run records are
under `docs/`; the root run ledger retains submission provenance.

Use paths in this directory for new work. Corresponding top-level paths under
`scripts/`, `configs/`, `others/`, and `external_models/` are compatibility
aliases only.

Final GitHub-facing summaries belong under `benchmarks/results/`. Full pose
banks and raw PoseBusters tables remain ignored; the tracked summary must retain
the denominator, missing-target failures, per-seed values, mean/sample standard
deviation, selector definition, timing stages, and provenance hashes.
