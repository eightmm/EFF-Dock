# Sigma-2 / Eta-2 Saved-Pose Refinement Protocol

## Scope

This is a descriptive post-hoc characterization of the already-completed
EFF-Dock `sigma=2.0`, normalized-drift `eta=2.0`, `N=100`, `S=10` ensembles.
It does not tune or admit guidance, refinement, or selection as a production
default. Astex Diverse and PoseBusters v2 outcomes have already been opened.

## Frozen inputs

- Input manifest:
  `outputs/benchmarks/guidance_sigma2_eta2_refinement_runs/20260819T060555Z/manifest.json`
- Manifest SHA256:
  `9e8be4d47dba8e346a6900b6bf02f5b853a93141f571f5c65b4c719de632d695`
- Inventory: Astex `85`, PoseBusters v2 `308`, `100` poses per complex.
- Source sampler: `sigma=2.0`, normalized direct-drift guidance `eta=2.0`,
  `N=100`, `S=10`, frozen geometry checkpoint step `100000`.
- Frozen confidence checkpoint:
  `weights/effdock_confidence_extmatch_n80_s25_step42500.pt`, SHA256
  `e31fde6f351284205c78f7a1510002779c43312e94d9f82003d47a14d72bc78f`.

## Refinement

- Optimize every saved pose independently in rigid-fragment SE(3) coordinates.
- Objective: the in-repository unified GuidanceEnergy; no Vina or external
  minimization/force-field engine.
- Maximum: `100` accepted-step attempts; materialized frames remain
  `0, 25, 50, 75, 100`, carrying a finite terminal pose forward after an early
  stop.
- Pose-wise monotone backtracking line search, maximum atom displacement
  `0.10 Angstrom`, maximum `12` backtracks.
- Energy-plateau stopping is evaluated only after an accepted update and only
  from step `25` onward. Define

  `delta_E_t = E_t - E_(t+1)`

  `epsilon_t = 0.02 kcal/mol + 1e-3 * max(1 kcal/mol, abs(E_t))`.

  A pose stops with `converged_energy_plateau` after
  `delta_E_t <= epsilon_t` for `5` consecutive accepted updates. This is a
  scale-aware numerical plateau criterion, not a cross-molecule target-energy
  threshold. The existing displacement convergence remains enabled.

## Pre-full-run numerical gate

One frozen complex may be run with and without energy stopping solely to
verify solver safety and runtime. No RMSD, confidence, or PoseBusters outcome
may alter the thresholds above. The full run proceeds only if both arms keep
all `100` poses finite, preserve atom-displacement/shell gates, and the adaptive
arm introduces no new numerical failure status.

The first calibration candidate (`1e-4 + 1e-6 * max(1, abs(E_t))`) kept all
poses finite but stopped none of the `100` frozen PoseBusters poses before step
`100`. Its fixed-step energy trace showed a median decrease of
`0.0318 kcal/mol/step` over steps `75--100`. Therefore the candidate above was
revised from that energy trace alone; no RMSD, confidence, or PoseBusters
outcome was inspected or used for this revision.

The revised numerical gate (Slurm `55771`, RTX 2080 Ti) completed all `100`
poses without a line-search or nonfinite failure. Energy stopping triggered for
`52/100` poses; the mean terminal step was `83.77` (range `39--100`). Relative
to the fixed `100`-step arm, runtime changed from `174.22 s` to `139.41 s`,
while terminal GuidanceEnergy was higher by `0.375 kcal/mol` on average
(`0.230` median; `1.390` maximum). The shell-envelope-valid count remained
`67/100` and the final chiral-improper inversion count remained zero. This
passed the numerical gate; these are solver diagnostics, not benchmark claims.

## Evaluation

After all refinement artifacts complete:

1. Freshly score step `0` and step `100` with the frozen confidence model,
   conditioned at the source pose `sigma=2.0`. Step-0 scoring must reproduce
   the source sigma-2 confidence ledger up to the recorded CUDA tolerance.
2. Run official PoseBusters `0.6.5` redock checks on every step-100 pose using
   protein-ligand-only PL-validity and the separately recorded official
   27-check validity.
3. Report fixed-index physical correction and post-refinement confidence
   reselection separately for RMSD `<2 Angstrom`, PL-validity, and their joint.
