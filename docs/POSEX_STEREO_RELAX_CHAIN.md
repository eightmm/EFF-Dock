# Chirality + E/Z → PoseX relaxation chain

Explicitly authorized after raw ablation evaluation, 2026-09-15. This is a
separate condition: input-declared tetrahedral AND E/Z consistency, original
U70K ranking/fallback, then the repaired PoseX relaxation protocol. No EFF-Dock
refinement and no post-relaxation reranking are introduced.

All relaxation arrays depend on successful completion of raw evaluation array
74924. Each array runs at most one GPU shard (nine SD shards / sixteen CD
shards); six arrays permit at most six concurrent GPUs. Evaluators and final
summary use cpu_only. Every dependency uses afterok and kill-on-invalid-dep.

| Dataset / seed | Relaxation | Relaxed evaluation |
| --- | --- | --- |
| SD101 | 75004 | 75005 |
| SD202 | 75006 | 75007 |
| SD303 | 75008 | 75009 |
| CD101 | 75010 | 75011 |
| CD202 | 75012 | 75013 |
| CD303 | 75014 | 75015 |

Final three-seed summary job 75016 depends on all six evaluations.

Inputs come only from each run's `chirality_ez_selection_v1/official_raw` and
must have a completed coverage-checked `evaluation.json`. Relaxation uses the
distributed PoseX CIF, `posex_chain_mapping_repair.py` and the retained
residue-ID repair, with unchanged upstream minimization/force-field settings.
Case exceptions cannot count as success: each shard verifies all final pairs,
assembly verifies full coverage, and evaluation verifies the exact result ID
set before summary. Original raw/chirality-only/confidence-only results remain
untouched.

Per-run output: `chirality_ez_selection_v1/posex_relaxed_v1` (shards, assembled
outputs, separate benchmark CSV, official results and `summary.json`).
Final JSON: `outputs/benchmarks/posex_official_protocol/chirality_ez_posex_relaxed_summary.json`.

Validation: changed-file lint, Python and shell syntax passed. Initial pytest
collection lacked the repository import path; rerun explicitly with
`PYTHONPATH=.:src`. Jobs are submitted with dependency gates, not claimed done.
