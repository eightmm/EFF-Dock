# PoseX — separate protocol and aggregation

These results must not be pooled with the supplied-pocket five-cohort main
table. Each condition has seeds 101, 202 and 303, with all 718 SD and 1,312 CD
cases per seed. CD is aggregated over 109 groups; SD uses 718 individual groups.
Reported percentages are means ± sample SD across the three seeds.

| Dataset | Selection before PoseX relaxation | RMSD ≤2 Å | RMSD ≤2 Å + PB-valid |
|---|---|---:|---:|
| SD | Confidence only | 74.56 ± 1.12 | 59.52 ± 1.26 |
| SD | Confidence + chirality/E/Z mask | 72.89 ± 1.08 | 69.64 ± 0.64 |
| CD | Confidence only | 63.91 ± 2.80 | 54.43 ± 1.40 |
| CD | Confidence + chirality/E/Z mask | 59.94 ± 3.71 | 59.33 ± 3.47 |

Selection occurs **before** relaxation. There is no post-relaxation reranking
and no EFF-Dock own refinement in these conditions. PoseX uses RMSD ≤2 Å,
not the strict <2 Å used in the main five-cohort analysis. The native
PDB_GROUP → GROUP averaging and joint mean-RMSD ≤2 Å / mean-validity ≥0.5
rule are retained; the CD percentages are not simple case-level rates.

The recovered run uses distributed CIF structures and upstream force-field,
restraint and minimization settings, followed by upstream alignment and
evaluation. Receptor residue/chain identity compatibility repairs are disclosed;
this is **not** a claim of byte-identical unmodified upstream preprocessing.
Earlier fixed-receptor adapter results are not substituted into this table.
Other models' published numbers are separate literature results, not reruns.

The mask increases the mean joint endpoint here but decreases mean RMSD-only
success. It therefore must not be described as improving every endpoint.
These post-hoc comparisons do not establish statistical significance.

## Evidence and reproducibility

- Baseline: `outputs/benchmarks/posex_official_protocol/upstream_recovered_evaluation_v1_summary.json`.
- Masked: `outputs/benchmarks/posex_official_protocol/chirality_ez_posex_relaxed_summary.json`.
- Both reports retain all three seed summaries and original result CSV paths.
- [Recovered evaluation procedure](../../POSEX_RECOVERED_EVALUATION.md).
- [Relaxation recovery contract](../../POSEX_RELAX_RECOVERY_20260914.md).
- [Mask → relaxation chain](../../POSEX_STEREO_RELAX_CHAIN.md).

This package only consolidates completed PoseX results; it launches no new
PoseX docking or relaxation.
