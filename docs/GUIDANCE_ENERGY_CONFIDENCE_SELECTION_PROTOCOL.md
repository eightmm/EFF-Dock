# GuidanceEnergy and confidence selection characterization

- Protocol: `EFFDOCK-GUIDANCE-ENERGY-CONFIDENCE-SELECTION-V1`
- Status: post-hoc descriptive characterization only
- Inputs: completed saved-pose post-refinement cohort, 85 Astex Diverse and
  308 PoseBusters v2 complexes with 100 poses per complex

## Question

Measure whether pose selection using both frozen confidence and in-repository
`GuidanceEnergy` changes RMSD success, PL-validity, or their conjunction.
Evaluate ODE output (`step 0`) and refined output (`step 100`) separately.

## Information boundary

Each selector receives only the 100 within-complex confidence predicted-RMSD
values and corresponding total GuidanceEnergy values. RMSD, PoseBusters checks,
dataset identity, crystal coordinates, and historical selected indices are
outcomes only and never enter a selector.

Raw confidence and energy values are not added or multiplied because their
units and ligand-dependent scales differ. Convert both to stable ordinal
qualities within each complex:

`q(x_i) = (N - rank(x_i)) / N`, where lower raw values are better and ties are
resolved by pose index. With `N=100`, quality lies in `[0.01, 1.00]`.

Freeze these selector families before calculating their outcomes:

- `confidence`: minimum predicted RMSD.
- `energy`: minimum total GuidanceEnergy.
- Additive rank fusion:
  `score_add = (1-alpha) q_conf + alpha q_energy`, maximize, with
  `alpha in {0.05, 0.10, 0.25, 0.50, 0.75}`.
- Multiplicative/geometric rank fusion:
  `score_geo = q_conf^(1-alpha) q_energy^alpha`, maximize, with the same
  alpha values.
- Energy filter: retain the best `{25%, 50%, 75%}` by energy, then choose
  minimum predicted RMSD among retained poses.

All selectors use stable pose-index tie breaking. `initial_total_energy` pairs
with step-0 confidence and coordinates; `final_total_energy` pairs with
step-100 confidence and coordinates.

## Outcomes and interpretation

For every dataset, stage, and selector report selected RMSD `<2 Angstrom`,
21-check PL-validity, their conjunction, official 27-check PoseBusters
validity, median selected RMSD, and index change relative to pure confidence.

Astex and PoseBusters outcomes were already opened before this analysis. No
coefficient, formula, filter fraction, or stage may be selected or admitted to
production from these values. The complete fixed grid is reported, including
negative results. Any selector decision requires a separately frozen internal
PLINDER calibration/confirmation study.
