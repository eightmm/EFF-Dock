# Saved-pose GuidanceEnergy post-refinement results

- Protocol: `EFFDOCK-GUIDANCE-SDF-POST-REFINEMENT-V1`, confidence extension V2
- Status: complete post-hoc descriptive characterization; not production-admitted
- Cohort: 85 Astex Diverse and 308 PoseBusters v2 complexes, 100 poses each
- Runtime jobs: refinement `52616`, confidence `53159`, PoseBusters `53162`,
  effect-decomposition report `53630`
- Artifact root:
  `outputs/benchmarks/guidance_sdf_post_refinement_runs/20260812T044100Z/full`

## Effect decomposition

The fresh step-0 confidence Top-1 is the reference index. `After fixed` follows
that exact pose index through 100 post-ODE refinement steps. `After reselected`
then applies the same chunk-20 confidence model to all refined poses and chooses
a new Top-1.

| Dataset | State | RMSD <2A | PL-valid | Joint | Median RMSD |
|---|---|---:|---:|---:|---:|
| Astex | Before | 75.29% | 28.24% | 27.06% | 1.148 A |
| Astex | After fixed | 75.29% | 92.94% | 70.59% | 1.102 A |
| Astex | After reselected | 82.35% | 94.12% | 78.82% | 0.959 A |
| PoseBusters | Before | 71.75% | 30.19% | 24.68% | 1.275 A |
| PoseBusters | After fixed | 72.40% | 92.53% | 68.51% | 1.263 A |
| PoseBusters | After reselected | 70.45% | 91.56% | 66.23% | 1.188 A |

Same-index refinement accounts for nearly all validity repair:

- Astex: PL-valid `+64.71 pp`, joint `+43.53 pp`, RMSD success `+0.00 pp`.
- PoseBusters: PL-valid `+62.34 pp`, joint `+43.83 pp`, RMSD success `+0.65 pp`.
- All-pose PL-valid changes from `25.19%` to `87.81%` on Astex and from
  `24.02%` to `87.35%` on PoseBusters. This result is independent of Top-1
  selection.

Reselection is not consistent across datasets:

- Astex: RMSD success `+7.06 pp`, PL-valid `+1.18 pp`, joint `+8.24 pp`.
- PoseBusters: RMSD success `-1.95 pp`, PL-valid `-0.97 pp`, joint `-2.27 pp`.
- For joint success, reselection changes 9 false-to-true and 2 true-to-false
  Astex complexes, but 23 false-to-true and 30 true-to-false PoseBusters
  complexes.

The physical correction is therefore a real same-pose effect rather than an
artifact of confidence reselection. The external evidence does not support a
dataset-independent benefit from rescoring after refinement.

## Diagnostics and decision

- Fresh step-0 confidence differs from the historical selector for 0/85 Astex
  and 3/308 PoseBusters complexes. This remains diagnostic only.
- The result supports carrying a fixed selected pose through bounded
  post-refinement as the conservative arm in the next internal validation.
- It does not admit post-refinement or choose a deployment selector. A
  pre-registered held-out PLINDER comparison must decide between fixed-index
  refinement and post-refinement reselection, and must evaluate adaptive
  stopping/trust-region guards without using these external outcomes for
  tuning.

This is a 100-step rigid-fragment post-ODE correction using reference-pocket
centers. It is not evidence for direct ODE guidance, molecular dynamics,
binding free energy, or affinity prediction. PL-valid excludes the declared
organic-cofactor and water checks; official 27-check values remain in the raw
aggregate.
