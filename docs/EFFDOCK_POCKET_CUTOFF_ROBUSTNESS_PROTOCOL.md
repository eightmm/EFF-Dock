# EFF-Dock pocket-cutoff robustness

Protocol ID: `EFFDOCK-POCKET-CUTOFF-ROBUSTNESS-V1`

Status: frozen before any production-checkpoint cutoff result is opened.

## Question

Measure the sensitivity of the complete supplied-pocket EFF-Dock inference
pipeline to the receptor pocket crop. This is a descriptive robustness study,
not a cutoff-selection study. Astex Diverse and PoseBusters v2 have already
been used repeatedly during development, so their outcomes cannot select a new
default.

## Frozen factorial design

- Datasets: all 85 Astex Diverse and all 308 PoseBusters v2 complexes.
- Pocket cutoffs: `6, 8, 10, 12 Angstrom`.
- Independent repeats: three prior-seed domains per cutoff. The ligand
  conformer seed remains fixed at zero so the repeat variance measures the
  sampled docking prior rather than a mixture of conformer and docking noise.
- Docking checkpoint: `weights/effdock_docking_early_time_t0p10_50k.pt`,
  SHA-256 `65be44d7dc8f0867eb9fc5d22214b80f93971ea4702679a527c665046e91e6b6`.
- Confidence checkpoint: `weights/effdock_confidence_s50_raw_refined_u70k.pt`,
  SHA-256 `ce59be42f0ca613871ca079127c3296f5ca9a4ec72e44a9e5cf61878351c2638`.
- Sampling: `N100/S10`, translation sigma `2.0`, late-power-3 schedule,
  supplied crystal pocket center, normalized-drift GuidanceEnergy with
  `eta=2.0`, and the frozen geometry-only receptor policy and caps.
- Refinement: all 100 poses, at most 100 in-repository Torch-autograd rigid
  fragment steps; energy plateau absolute `0.02 kcal/mol`, relative `0.001`,
  patience `5`, minimum `25` steps.
- Selection: stable minimum U70k predicted symmetry-aware RMSD after
  refinement, scored in fixed chunks of 20.
- Validity: official PoseBusters `0.6.5`, `redock`, all 27 non-RMSD validity
  checks on the selected refined pose.

The tested pocket cutoff is changed **only for docking ODE preprocessing**.
Refinement preprocessing and U70k confidence preprocessing remain fixed at the
training/default `10 Angstrom` crop, and the physical interaction shell remains
fixed at `18 Angstrom`. This isolates docking-crop robustness instead of
silently evaluating four different confidence-model input distributions.

## Primary reporting

For every dataset and cutoff, report across the three repeats:

- Top-1 symmetry-aware heavy-atom RMSD `<2 Angstrom`;
- selected-pose PB-validity;
- Joint: RMSD `<2 Angstrom` and PB-valid;
- Oracle-100 RMSD `<2 Angstrom` after refinement;
- mean and sample standard deviation (`ddof=1`).

Missing targets, generation/refinement/confidence failures, and PoseBusters
errors remain failures in the frozen dataset denominator. No result from this
study changes the public 10 Angstrom default without an internal PLINDER
validation decision.
