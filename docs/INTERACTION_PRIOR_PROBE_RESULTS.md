# EFF-Dock interaction prior probe results

Protocol: `EFFDOCK-INTERACTION-PRIOR-PROBE-V2`

Run date: 2026-07-31

Status: completed diagnostic; no new interaction term admitted to the default
profile or production ODE sampler.

## Execution

- Slurm array: `44659`
- Tasks: 12/12 completed with exit code 0; every stderr file was empty.
- Ensemble: 2 priors × 6 arms × 8 paired seeds = 96 trajectories.
- Solver: 500 rigid-fragment SE(3) descent steps, pose-wise line search.
- Receptor shell: 22 A around the frozen pocket center for relaxation.
- Numerical check: no accepted-step monotonicity violations.
- Tests at completion: 228 passed, 3 skipped; the two warnings were the
  expected `cuequivariance_ops_torch` fallback.

V1 job `44637` also completed, but its 18 A pocket-centered receptor shell gave
only an 8 A valid ligand envelope while the crystal/local ligand radius reached
9.62/10.99 A. V1 is therefore an invalid numerical preflight and is excluded
from every conclusion. V2 changed only the shell to 22 A; energy formulas,
parameters, cutoffs, typing, priors, seeds, solver steps, and gates were
unchanged.

## Crystal energies

All values are kcal/mol. A zero means the typed motif is absent or inactive in
that structure, not that the implementation was skipped.

| Case | Hydrophobic | H-bond | Charge | Pi stack | Cation-pi | Halogen | Metal | Total interaction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Astex `1LPZ_CMB` | -1.4386 | -1.4551 | 0.0000 | -0.0176 | 0.0000 | -0.0139 | 0.0000 | -2.9251 |
| PoseBusters `6Z14_Q4Z` | -0.2500 | -3.0580 | -1.2461 | 0.0000 | -0.2063 | 0.0000 | 0.0000 | -4.7604 |
| Astex `1R1H_BIR` | -2.4123 | -0.9148 | 0.0000 | -0.3999 | -0.4028 | 0.0000 | +0.2063 | -3.9235 |

The typed contacts are structure-specific as intended: `1LPZ_CMB` exercises
pi/halogen, `6Z14_Q4Z` exercises formal charge/cation-pi, and `1R1H_BIR`
exercises Zn coordination. The positive metal value and high crystal metal
force remain an admission warning; the metal term was not relaxed or tuned
from this external case.

For the pre-registered `1LPZ_CMB` crystal gate:

- halogen passed: crystal energy -0.01386, contact weight 0.02773, below 7/8
  local-prior initial energies;
- pi stacking failed: crystal energy -0.01763, contact weight 0.07054, below
  only 4/8 local-prior initial energies.

## Relaxation ensemble

| Prior | Arm | Joint pass | Median initial RMSD | Median final RMSD | Median final interaction |
|---|---|---:|---:|---:|---:|
| local | guard only | 0/8 | 0.899 A | 1.140 A | 0.000 |
| local | default interactions | 3/8 | 0.899 A | 1.006 A | -1.822 |
| local | default + pi | 4/8 | 0.899 A | 1.001 A | -1.829 |
| local | default + halogen | 3/8 | 0.899 A | 1.009 A | -1.824 |
| local | all terms | 3/8 | 0.899 A | 1.010 A | -1.847 |
| local | raw all interactions | 0/8 | 0.899 A | 1.152 A | -5.443 |
| model | guard only | 0/8 | 6.330 A | 6.680 A | 0.000 |
| model | default interactions | 0/8 | 6.330 A | 6.705 A | -1.680 |
| model | default + pi | 0/8 | 6.330 A | 6.705 A | -1.680 |
| model | default + halogen | 0/8 | 6.330 A | 6.705 A | -1.723 |
| model | all terms | 0/8 | 6.330 A | 6.705 A | -1.723 |
| model | raw all interactions | 0/8 | 6.330 A | 6.396 A | -4.373 |

The paired single-new-term effects versus `guard_default` were:

| Prior | Term | Median delta final RMSD | Seeds improved | New protocol failures | Admission |
|---|---|---:|---:|---:|---:|
| local | halogen | +0.000156 A | 3/8 | 0 | no |
| local | pi | -0.000326 A | 4/8 | 0 | no |
| model | halogen | -0.000002 A | 4/8 | 0 | no |
| model | pi | 0.000000 A | 0/8 | 0 | no |

Neither new term approached the required median improvement of -0.25 A and
5/8 improved seeds. Halogen passed its crystal gate but did not improve pose
recovery; pi failed both the crystal gate and pose-recovery effect gate.

The default interaction set helped relative to guard-only in the local basin
(median final RMSD 1.006 versus 1.140 A), but still moved the median pose away
from the 0.899 A initialization. It should be treated as a late-stage
filter/correction candidate, not a standalone optimizer.

The raw interaction-only control lowered its local interaction energy from a
median -2.298 to -5.443 while all 8 poses violated the cut-bond geometry gate,
5/8 violated the clash gate, and median RMSD worsened. This is direct evidence
that attraction energy alone is not a valid docking objective.

The exact model prior failed 0/8 for every arm. Guarded objectives reduced very
large overlap/connectivity energies monotonically, but after 500 steps the
median cut-bond error remained about 4.0 A and median RMSD about 6.7 A. The
current energy is therefore not a replacement for the learned ODE; it is at
most a late-stage guidance/filter layer.

## Recorded artifacts

- Compact table:
  `outputs/guidance/interaction_prior_probe_v2/1LPZ_CMB/aggregate/RESULTS.md`
  (`sha256 e7d6a02aeb861ac3ccd883862ce9be3219ee96b3ac35eb6339908274bdb3fd96`)
- Per-run CSV:
  `outputs/guidance/interaction_prior_probe_v2/1LPZ_CMB/aggregate/runs.csv`
  (`sha256 373c9ce8d5fe6816236c967b74a0381afdfd33ed43dbf0fa75214974b73c81d6`)
- Combined trajectory:
  `outputs/guidance/interaction_prior_probe_v2/1LPZ_CMB/aggregate/trajectory.pt`
  (`sha256 4ed34a0c41a7516419ec547098dcb42e2191f50f1e963ae17b2ddaefab403773`)
- Full summary:
  `outputs/guidance/interaction_prior_probe_v2/1LPZ_CMB/aggregate/summary.json`
- Crystal traces:
  `outputs/guidance/interaction_prior_probe_v2/crystal/`

The large outputs remain local and Git-ignored. This document is the tracked
result record.
