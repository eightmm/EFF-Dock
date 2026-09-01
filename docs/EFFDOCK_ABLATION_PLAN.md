# EFF-Dock ablation plan

Status: deferred plan; not submitted.

This document records the follow-up ablations to run only after the active
pocket-cutoff robustness study and external-model benchmark are complete. It
does not change the production inference defaults.

## Scientific boundary

- Hyperparameter or term admission must be decided on the frozen PLINDER
  validation cohort. Astex Diverse and PoseBusters v2 have already been opened
  repeatedly and can provide descriptive comparisons only.
- The production docking and confidence checkpoints remain frozen.
- Use paired candidate identities whenever a comparison permits it.
- Primary endpoints are Top-1 symmetry-aware RMSD `< 2 A`, official
  PoseBusters validity, their conjunction (Joint SR), Oracle SR, runtime, and
  failure coverage. Report three independent prior-seed repeats as mean plus
  sample standard deviation.

## Priority 1: inference-stage decomposition

Use the same `N100/S10`, sigma-2 prior contract and U70k selector.

| Arm | ODE guidance | Adaptive refinement | Selection |
|---|---:|---:|---|
| Base | off | off | U70k confidence |
| ODE only | on | off | U70k confidence |
| Refinement only | off | on | U70k confidence |
| Full | on | on | U70k confidence |
| Full, no learned selection | on | on | fixed candidate index |
| Full oracle | on | on | RMSD oracle, report-only |

This is the primary paper ablation because it isolates candidate generation,
post-ODE correction, and learned selection.

## Priority 2: cumulative guidance-term ablation

Add terms cumulatively so each arm retains a physically meaningful base:

1. no guidance;
2. ligand intra-geometry;
3. plus protein-ligand steric barrier;
4. plus hydrogen-bond and hydrophobic interactions;
5. plus screened electrostatics;
6. plus pi stacking, cation-pi, halogen bonding, and profile-dispatched metal
   coordination;
7. full guidance.

The `polar_unsatisfied_proxy` remains trace metadata and is excluded from
energy, force, ranking, and selection in every arm.

## Priority 3: refinement budget

Compare no refinement, fixed 25 steps, fixed 100 steps, and adaptive refinement
with maximum 100 steps. In addition to accuracy and validity, report the actual
mean/median refinement steps, early-stop fraction, and seconds per pose.

## Priority 4: prior diversity and guidance recovery

Run the minimal factorial comparison `sigma in {1, 2}` by ODE guidance
`{off, on}`. Add sigma 3 only if the PLINDER validation result shows that
guidance recovers the extra diversity without materially reducing Top-1 or
Joint SR.

## Priority 5: equal-compute ODE allocation

Hold the learned-model pose-step budget at 1,000 and compare:

- `S10 x N100`;
- `S20 x N50`;
- `S25 x N40`.

Report Top-1, Oracle, cumulative Oracle@N, wall time, pose-generation time, and
selection time separately.

## Priority 6: selector interaction

Compare U70k confidence on raw poses, U70k confidence on refined poses,
confidence after a pre-registered hard physical-invalidity filter, and a
pre-registered interaction-energy tie-break. Do not normalize or rank energy
using the number of sampled poses. RMSD and PB-validity never enter a selector.

## Execution order

1. Finish and freeze the active pocket-cutoff and external-model benchmarks.
2. Materialize one paired PLINDER validation candidate bank.
3. Run Priority 1 and the cumulative term ablation.
4. Admit no setting until the validation report is frozen.
5. Run one confirmation evaluation on Astex/PoseBusters without further
   tuning.
