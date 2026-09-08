# EFF-Dock pocket-cutoff x center-jitter robustness

Protocol ID: `EFFDOCK-POCKET-CUTOFF-JITTER-ROBUSTNESS-V1`

Status: frozen before the production-matched jitter `1 A` and `2 A` outcomes
are opened.

## Question

Measure the supplied-pocket EFF-Dock pipeline's sensitivity to both receptor
crop radius and Gaussian perturbation of the supplied crystal pocket center.
This is a descriptive robustness study. Astex Diverse and PoseBusters v2 have
already been used during development, so these outcomes cannot select a new
production default.

## Frozen factorial design

- Datasets: all 85 Astex Diverse and all 308 PoseBusters v2 complexes.
- Docking pocket cutoffs: `6, 8, 10, 12 Angstrom`.
- Center jitter sigma per Cartesian axis: `0, 1, 2 Angstrom`.
- Independent repeats: three docking-prior/jitter seed domains per cell. The
  ligand conformer seed remains fixed at zero.
- The completed zero-jitter cells are reused without regeneration from
  `outputs/benchmarks/effdock_pocket_cutoff_robustness_runs/cutoff-r3-production-20260901-r2`.
  Their report must be complete, have zero selected-pose evaluation errors,
  and retain the hashes and inference contract below.
- Docking checkpoint:
  `weights/effdock_docking_early_time_t0p10_50k.pt`, SHA-256
  `65be44d7dc8f0867eb9fc5d22214b80f93971ea4702679a527c665046e91e6b6`.
- Confidence checkpoint:
  `weights/effdock_confidence_s50_raw_refined_u70k.pt`, SHA-256
  `ce59be42f0ca613871ca079127c3296f5ca9a4ec72e44a9e5cf61878351c2638`.
- Sampling: `N100/S10`, translation prior sigma `2.0`, late-power-3 schedule,
  normalized-drift GuidanceEnergy with `eta=2.0`, and frozen geometry-only
  receptor policy and caps.
- Refinement: all 100 poses, at most 100 in-repository Torch-autograd rigid
  fragment steps; absolute energy plateau `0.02 kcal/mol`, relative plateau
  `0.001`, patience `5`, and minimum `25` steps.
- Selection: stable minimum U70k predicted symmetry-aware RMSD after
  refinement, scored in fixed chunks of 20.
- Validity: official PoseBusters `0.6.5`, `redock`, all 27 non-RMSD validity
  checks on the selected refined pose.

The cutoff and jitter change only docking ODE preprocessing. Refinement and
U70k confidence preprocessing remain fixed at the production `10 Angstrom`
crop, and the interaction shell remains fixed at `18 Angstrom`. For a repeat,
the base seed is held across cutoff/jitter cells so comparisons are paired.

## Primary reporting

For each dataset, cutoff, and jitter cell, report the three-repeat mean and
sample standard deviation (`ddof=1`) for:

- Top-1 symmetry-aware heavy-atom RMSD `<2 Angstrom`;
- selected-pose PB-validity;
- Joint: RMSD `<2 Angstrom` and PB-valid;
- Oracle-100 RMSD `<2 Angstrom` after refinement.

Missing targets, generation/refinement/confidence failures, and PoseBusters
errors remain failures in the frozen denominator. The heatmap uses repeat
means and must label center jitter as sigma per Cartesian axis.
