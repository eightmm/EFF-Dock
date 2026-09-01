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

The upstream repositories, installed environments, downloaded weights, caches,
and raw outputs are intentionally excluded from Git. Their reproducible
manifests and lock files are published; the populated runtime trees are not.
The current installations remain reachable through `others/<model>/` and the
ignored root `external_models/` workspace while active jobs complete. A later
physical relocation under an ignored `benchmarks/external_models/runtime/`
tree must retain those paths as compatibility aliases because several pinned
environments contain absolute prefixes.

Every reported campaign must retain its dataset manifest, receptor policy,
model-native inference setting, candidate count, seed, selection stage,
coverage gate, and runtime accounting. The current frozen protocol is
documented in `docs/EXTERNAL_MODEL_OFFICIAL_INFERENCE_PROTOCOL.md`, and concrete
job IDs are recorded in `docs/EXTERNAL_MODEL_SUBMISSION_20260831.md` and
`docs/EXPERIMENTS.jsonl`.

Use paths in this directory for new work. The corresponding paths under
`scripts/` and `configs/` are compatibility aliases only.

Final GitHub-facing summaries belong under `benchmarks/results/`. Full pose
banks and raw PoseBusters tables remain ignored; the tracked summary must retain
the denominator, missing-target failures, per-seed values, mean/sample standard
deviation, selector definition, timing stages, and provenance hashes.
