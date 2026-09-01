# Sigma-2 / Eta-2 adaptive post-refinement results

- Status: complete post-hoc descriptive characterization.
- Cohort: Astex Diverse `85`; PoseBusters v2 `308`; `100` poses per complex.
- Source: `sigma=2.0`, normalized-drift `eta=2.0`, `N=100`, `S=10`.
- Refinement: unified in-repository GuidanceEnergy, at most `100` rigid-fragment
  steps, adaptive energy-plateau stopping from step `25`.
- Confidence: frozen step-42500 checkpoint, freshly scored at `sigma=2.0` in
  chunks of `20` before and after refinement.
- Pose validity: PoseBusters `0.6.5`; primary validity is the 21-check
  protein-ligand-only PL-valid definition.
- Artifact root:
  `outputs/benchmarks/guidance_sdf_post_refinement_runs/sigma2-eta2-adaptive-20260819T062833Z/full`
- Completion jobs: recovery refinement `56080`, confidence `56081`,
  PoseBusters `56113`; all array tasks completed with exit `0`.

## Final sigma-2 results

`After fixed` follows the fresh step-0 confidence index through refinement.
`After reselected` applies confidence to all refined poses and selects a new
index. Joint is symmetry-aware RMSD `<2 Angstrom` and PL-valid for the same
pose.

| Dataset | State | Top-1 <2A | PL-valid | Joint | Median RMSD |
|---|---|---:|---:|---:|---:|
| Astex | Step-0 confidence | 72.94% | not recomputed | not recomputed | 1.171 A |
| Astex | After fixed | 72.94% | 90.59% | 68.24% | not reported |
| Astex | After reselected | 71.76% | 92.94% | 68.24% | 1.095 A |
| PoseBusters | Step-0 confidence | 74.35% | not recomputed | not recomputed | 1.248 A |
| PoseBusters | After fixed | 74.03% | 94.16% | 71.43% | not reported |
| PoseBusters | After reselected | 75.00% | 94.81% | 72.73% | 1.217 A |

| Dataset | All-pose PL-valid | RMSD oracle | PL-valid joint oracle |
|---|---:|---:|---:|
| Astex | 87.01% | 96.47% | 94.12% |
| PoseBusters | 87.49% | 95.78% | 94.81% |

The source sigma-2 selected-pose PoseBusters run used the historical selector;
fresh step-0 scoring changed one Astex and two PoseBusters indices. Therefore
source PL-valid and joint values are not copied into the fresh step-0 rows.

## Existing small-sigma comparison

The existing comparison arm is the completed saved-pose refinement at
`sigma=0.5`, `eta=0`, fixed `100` steps, recorded in
`docs/GUIDANCE_SDF_POST_REFINEMENT_RESULTS.md`. The table below is a pipeline
comparison, not a controlled sigma-only effect: prior sigma, ODE guidance, and
the refinement stopping rule all differ.

| Dataset | Metric | sigma=0.5 / eta=0 | sigma=2 / eta=2 | Delta |
|---|---|---:|---:|---:|
| Astex | Fixed-index joint | 70.59% | 68.24% | -2.35 pp |
| Astex | Reselected joint | 78.82% | 68.24% | -10.59 pp |
| Astex | Joint oracle | 90.59% | 94.12% | +3.53 pp |
| Astex | All-pose PL-valid | 87.81% | 87.01% | -0.80 pp |
| PoseBusters | Fixed-index joint | 68.51% | 71.43% | +2.92 pp |
| PoseBusters | Reselected joint | 66.23% | 72.73% | +6.49 pp |
| PoseBusters | Joint oracle | 92.53% | 94.81% | +2.27 pp |
| PoseBusters | All-pose PL-valid | 87.35% | 87.49% | +0.14 pp |

The larger-prior guided ensemble has more oracle headroom on both datasets,
but the frozen confidence selector converts that headroom into a Top-1 gain
only on PoseBusters v2. Astex Top-1 selection degrades, while its joint oracle
improves. This is consistent with a selection/calibration bottleneck under
distribution shift, but the external benchmark alone does not identify prior
sigma as the cause.

## Adaptive stopping diagnostics

| Dataset | Poses | Energy-plateau stops | Mean terminal step | Numerical failures |
|---|---:|---:|---:|---:|
| Astex | 8,500 | 69.38% | 66.96 | 0 |
| PoseBusters | 30,800 | 64.53% | 70.53 | 0 |

One Astex pose retained a finite `line_search_failed` terminal coordinate; it
is tracked separately and is not counted as a numerical failure. A controlled
sigma-only comparison requires applying the same `eta=2` adaptive refinement,
sigma-conditioned confidence scoring, and PoseBusters protocol to the existing
`sigma=0.5`, `eta=2`, `N100/S10` ensemble.
