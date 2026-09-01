# Raw GuidanceEnergy and confidence selection characterization

- Protocol: `EFFDOCK-GUIDANCE-ENERGY-CONFIDENCE-RAW-SELECTION-V1`
- Status: post-hoc descriptive characterization only
- Inputs: completed saved-pose post-refinement cohort, 85 Astex Diverse and
  308 PoseBusters v2 complexes with 100 poses per complex

## Question

Measure whether a candidate-count-invariant raw GuidanceEnergy correction to
confidence changes selected RMSD success, PL-validity, or their conjunction.
Evaluate ODE output (`step 0`) and refined output (`step 100`) separately.

## Information and scale boundary

For pose `i`, let `C_i` be confidence-predicted RMSD in Angstrom, `E_i` be
total GuidanceEnergy in kcal/mol, and `N_H` be ligand heavy-atom count. Every
score is computed from that pose and fixed complex metadata only. Candidate
rank, percentile, mean, variance, minimum, maximum, and candidate count are
forbidden score inputs.

The current selector is `argmin_i C_i`. Freeze two raw-score families:

- Total-energy correction: `S_i = C_i + lambda_total E_i`, with
  `lambda_total in {0.001, 0.0025, 0.005, 0.01, 0.025, 0.05}` Angstrom per
  kcal/mol.
- Size-normalized correction: `S_i = C_i + lambda_atom E_i/N_H`, with
  `lambda_atom in {0.05, 0.10, 0.25, 0.50, 1.00, 2.00}` Angstrom per
  kcal/mol/heavy-atom.

All scores are minimized with stable pose-index tie breaking. No clipping or
pose-set normalization is used. The total-energy family deliberately measures
the ligand-size-sensitive direct formula; the per-heavy-atom family supplies
the candidate-count- and size-normalized alternative. `initial_total_energy`
pairs with step-0 confidence and coordinates; `final_total_energy` pairs with
step-100 confidence and coordinates.

The coefficient grids were frozen from outcome-blind input scale inspection.
Across complexes, median within-complex IQR is about `0.5 Angstrom` for
confidence-predicted RMSD and `0.24--0.30 kcal/mol/heavy-atom` for final
GuidanceEnergy. The grid spans a negligible correction through an
energy-dominant correction without using RMSD or PoseBusters outcomes.

## Outcomes and interpretation

For every dataset, stage, and selector report selected RMSD `<2 Angstrom`,
21-check PL-validity, their conjunction, official 27-check PoseBusters
validity, median selected RMSD, and index change relative to pure confidence.

Astex and PoseBusters outcomes were already opened. The full fixed grid is
reported, including negative results; it cannot select a production formula or
coefficient. Any selector decision requires separately frozen internal
PLINDER calibration and external confirmation.
