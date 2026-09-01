# U70k/U100k temporal external confidence results

Protocol: `EFFDOCK-S50-RAW-REFINED-CONFIDENCE-TEMPORAL-EXTERNAL-V1`

The complete run contains 2,258 confidence summaries, 144 confidence shards,
144 selected-pose PoseBusters shards, and 2,258 selected SDF files. All Slurm
tasks completed with exit code 0.

| Dataset | Arm | N | Raw Top-1 `<2A` | Refined Top-1 `<2A` | PB-valid | Joint PB-valid + `<2A` |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| PhiBench | U70k | 203 | 128 (63.05%) | 131 (64.53%) | 184 (90.64%) | 120 (59.11%) |
| PhiBench | U100k | 203 | 129 (63.55%) | 130 (64.04%) | 185 (91.13%) | 119 (58.62%) |
| FoldBench | U70k | 66 | 42 (63.64%) | 45 (68.18%) | 60 (90.91%) | 44 (66.67%) |
| FoldBench | U100k | 66 | 44 (66.67%) | 44 (66.67%) | 60 (90.91%) | 43 (65.15%) |
| OpenBind | U70k | 860 | 422 (49.07%) | 477 (55.47%) | 848 (98.60%) | 470 (54.65%) |
| OpenBind | U100k | 860 | 427 (49.65%) | 472 (54.88%) | 847 (98.49%) | 465 (54.07%) |

For refined Top-1 `<2A`, U100k minus U70k has paired gain/loss counts of 5/6
on PhiBench, 2/3 on FoldBench, and 13/18 on OpenBind. The corresponding joint
PB-valid + `<2A` counts are 4/5, 2/3, and 13/18. U100k improves raw Top-1 by
one, two, and five complexes, respectively, but gives those gains back after
refinement and validity-aware evaluation.

Relative to the historical U50k selector, U70k refined Top-1 changes by -1
complex on PhiBench, +4 on FoldBench, and +32 on OpenBind. The result supports
retaining the internally selected U70k checkpoint: it is not uniformly better
than U50k across every cohort, but it is the stronger U70k/U100k refined and
joint endpoint on all three cohorts.

These are repeated-use descriptive pocket-redocking adaptations. PhiBench and
FoldBench are the core temporal checks; OpenBind remains an auxiliary dense
single-protease series. External outcomes do not select or promote a
checkpoint.

- Full report:
  `outputs/benchmarks/s50_raw_refined_confidence_temporal_external_runs/d97d5eb907acc485dfde4b7fcf88d87b4d5fd8576014d2cfb89dd0518b9c9bb4/report/summary.json`
- Report SHA-256:
  `3365e59753b13464f4911d28f0983e121f3f7be3ea00521f6942a17e167071bf`
