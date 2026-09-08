# EFF-Dock pocket-cutoff and prior-sigma robustness extension

Protocol ID: `EFFDOCK-POCKET-PRIOR-ROBUSTNESS-EXTENSION-V1`

Status: frozen before any new cutoff-14 or production-stack sigma-1/sigma-4
outcome is opened.

## Question and interpretation boundary

Characterize two independent supplied-pocket inference sensitivities under the
complete promoted EFF-Dock stack:

1. whether receptor-crop performance has plateaued by `12 Angstrom` or changes
   again at `14 Angstrom`; and
2. how the fragment-translation prior scale changes sampling, refined oracle,
   U70k selection, and selected-pose physical validity.

Astex Diverse and PoseBusters v2 have already been used during development.
All outcomes are repeated-use descriptive characterizations. They cannot
select a new production cutoff, prior sigma, guidance coefficient, refinement
setting, or confidence checkpoint. A deployable change requires a separately
registered PLINDER validation decision followed by fixed external reporting.

## Factor-isolated design

- Datasets: all `85` Astex Diverse and all `308` PoseBusters v2 complexes.
- New cutoff arm: docking cutoff `14 Angstrom`, translation prior sigma `2`,
  and center-jitter sigma `{0,1,2} Angstrom` per Cartesian axis.
- Prior-sigma arms: docking cutoff `10 Angstrom`, translation prior sigma
  `{1,2,4}`, and center-jitter sigma `{0,1,2} Angstrom` per Cartesian axis.
- Independent repeats: three paired seed domains per cell. The base seeds are
  `42`, `100042`, and `200042`; the ligand conformer seed remains zero.
- A repeated seed uses the same standard-normal translation draws and rotation
  draws across prior-sigma arms. Sigma changes only the scale of the coupled
  translation draw. The same base seed is held across cutoff and jitter cells.
- The completed sigma-2 cutoff `{6,8,10,12}` by jitter `{0,1,2}` report is an
  immutable prerequisite. Its cutoff-10 cells supply the sigma-2 prior baseline
  and its full grid supplies the cutoff trend through `12 Angstrom`.
- Only the `9` cutoff-14 repeat cells and `18` cutoff-10 sigma-1/sigma-4 repeat
  cells are newly generated. There is no crossed `5 cutoffs x 3 prior sigmas`
  experiment, so cutoff and sigma effects remain identifiable.

## Frozen inference stack

- Docking checkpoint:
  `weights/effdock_docking_early_time_t0p10_50k.pt`, SHA-256
  `65be44d7dc8f0867eb9fc5d22214b80f93971ea4702679a527c665046e91e6b6`.
- Confidence checkpoint:
  `weights/effdock_confidence_s50_raw_refined_u70k.pt`, SHA-256
  `ce59be42f0ca613871ca079127c3296f5ca9a4ec72e44a9e5cf61878351c2638`.
- Candidate generation: `N100/S10`, late-power-3 time schedule, prior pool
  `100`, normalized-drift GuidanceEnergy with `eta=2`, start `t=0.5`, ramp
  power `1`, force cap `20`, translation/angular cap `5`, atom-displacement
  cap `0.25 Angstrom`, at most `8` backtracks, protein shell `18 Angstrom`,
  and receptor policy `geometry_only`.
- Refinement: all `100` saved poses, at most `100` in-repository Torch-autograd
  rigid-fragment steps; absolute energy plateau `0.02 kcal/mol`, relative
  plateau `0.001`, patience `5`, and minimum `25` steps.
- Refinement and confidence receptor preprocessing stay fixed at the production
  `10 Angstrom` crop. Only docking-ODE preprocessing uses the tested cutoff.
- Confidence is conditioned on the actual source prior sigma (`1`, `2`, or
  `4`) and scored in fixed chunks of `20`. The primary selector is stable
  minimum predicted symmetry-aware RMSD after refinement.
- Validity: official PoseBusters `0.6.5`, `redock`, all `27` non-RMSD validity
  checks on the selected refined pose.

The U70k checkpoint was trained on sigma-2 raw/refined banks. Sigma-1 and
sigma-4 selection are therefore explicit confidence-domain-shift measurements,
not presumed calibration improvements.

## Hypotheses and disconfirming outcomes

- Cutoff trend hypothesis: cutoff `14 Angstrom` will be within one repeat
  standard deviation of cutoff `12 Angstrom` at each jitter, indicating a
  plateau. A consistent loss or gain larger than that is evidence against a
  plateau and must be reported without changing the production default.
- Prior trend hypothesis: sigma `4` may improve Oracle-100 through diversity
  but can reduce selected Top-1 through harder denoising and confidence domain
  shift; sigma `1` may show the reverse trade-off. Failure to change oracle or
  selected success beyond repeat variability disconfirms a material sigma
  effect under this budget.

## Fail-closed execution and reporting

The chain is end-to-end GPU smoke, full GPU generation, CPU manifest audit,
GPU refinement, GPU confidence scoring, CPU selected-pose official
PoseBusters, and CPU aggregation. CPU-only stages run exclusively on
`cpu_only`. GPU stages require one supported GPU with at least `48,000 MiB`
reported memory. Every full cell must contain exactly `85 + 308` complexes,
exactly `100` readable poses per complex, finite required values, complete
hash-checked inputs, and zero selected-pose evaluation errors.

For every dataset and cell, report three-repeat mean and sample standard
deviation (`ddof=1`) for:

- refined U70k Top-1 symmetry-aware RMSD `<2 Angstrom`;
- selected-pose PB-validity;
- joint RMSD `<2 Angstrom` and PB-validity; and
- refined Oracle-100 RMSD `<2 Angstrom`.

The final artifact contains two views only: cutoff `{6,8,10,12,14}` at fixed
sigma `2`, and prior sigma `{1,2,4}` at fixed cutoff `10`. Missing complexes or
failed stages remain failures in the frozen denominator; no successful-only
subset is reported as the primary result.
