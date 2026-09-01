# EFF-Dock interaction prior probe

Protocol ID: `EFFDOCK-INTERACTION-PRIOR-PROBE-V2`

Status: frozen before the V2 crystal traces and relaxation ensemble. V1 was
invalidated before scientific interpretation because its 18 A receptor shell
did not contain the full 10 A interaction neighborhood of the crystal/local
ligand envelope. V2 changes only that numerical domain to 22 A; no interaction
formula, coefficient, cutoff, type, seed, prior, objective, or decision gate
changed. V1 outputs are retained as invalid preflight evidence and cannot
support a claim or parameter change.

## Question and claim boundary

This experiment asks whether the self-contained typed interaction terms define
a useful late-stage rigid-fragment basin, and whether they can recover a pose
from the exact active EFF-Dock `t=0` prior without the learned vector field.
It is not molecular dynamics, affinity prediction, blind pocket discovery, or
production guided sampling.

Crystal coordinates define the diagnostic reference and fragment-local
geometry. They may initialize the explicitly named local-prior arm, but never
enter an energy, force, line search, or model-prior initialization. RMSD is
evaluation-only. Frozen Astex/PoseBusters pocket centers are
reference-ligand-derived and that limitation is carried in every output.

The external benchmark outcomes are report-only. Their results may not
change a formula, coefficient, cutoff, atom type, schedule, solver setting, or
term activation. A fail-closed numerical-domain check may invalidate a run and
force a new protocol ID, as happened between V1 and V2.

## Frozen systems

Cases were chosen by static motif coverage, not by favorable energy:

- Astex `1LPZ_CMB`: non-metal relaxation probe with typed aromatic and halogen
  motifs.
- PoseBusters `6Z14_Q4Z`: cation-pi crystal characterization.
- Astex `1R1H_BIR`: strict supported Zn(II) crystal characterization.

All three receive an all-callable-term crystal trace. The seeded relaxation
ensemble is restricted to `1LPZ_CMB`; cation-pi and Zn relaxation require their
own target-specific follow-up protocols. Crystal traces select receptor atoms
within 18 A of the ligand coordinates, which fully covers the largest 10 A
interaction cutoff. Relaxation instead uses one frozen pocket center and
therefore requires the larger shell specified below.

## Frozen priors

Seeds are `20260731` through `20260738`, paired exactly across every ablation.
Each prior family is represented as one tensor batch of eight independent
poses.

1. `local`: each crystal fragment receives an independent Gaussian
   translation with `sigma=0.5 A` and a proper axis-angle rotation whose angle
   has `sigma=15 degrees`. This is a crystal-informed late-stage basin probe.
2. `model`: exact active-sampler draw order, with fragment centers sampled from
   `N(pocket_center, 0.5^2 I)` followed by `Uniform(SO(3))` for multi-atom
   fragments. This is the inference-valid initialization, although the learned
   ODE is absent.

Both use CPU float32 random draws and retain crystal fragment-local geometry.

## Frozen objectives

The guard is:

```text
E_guard =
  ligand intra bond + angle + proper + improper
  + ligand intra Lennard-Jones repulsive + attractive
  + protein-ligand Lennard-Jones repulsive
```

Protein-ligand Lennard-Jones attraction is excluded so pocket attraction in a
guarded arm comes from typed interaction terms.

Paired arms:

1. `guard_only`
2. `guard_default`: guard plus hydrophobic, hydrogen bond, and screened formal
   charge
3. `guard_pi`: default plus pi stacking
4. `guard_halogen`: default plus halogen bond
5. `guard_all`: guard plus all seven callable interaction terms
6. `interaction_all_raw`: all seven interaction terms without the guard

`guard_pi` and `guard_halogen` are the only single-new-term causal comparisons
on this case. `guard_all` and `interaction_all_raw` are exploratory
visualizations, not admission evidence. The dimensionless
`polar_unsatisfied_proxy` is trace-only in every arm.

## Solver and numerical contract

- variables: rigid-fragment translations `[B,F,3]` and scalar-first
  quaternions `[B,F,4]`;
- force: `F = -dE/dx`;
- batch gradient: gradient of the sum of independent per-pose energies;
- independent per-pose monotone line search; a batch sum/mean can never accept
  a pose;
- 500 maximum steps, save every 5;
- maximum fragment translation `0.10 A` per accepted step;
- maximum rotation `5 degrees` per accepted step;
- maximum atom displacement `0.10 A` per accepted step;
- backtrack factor `0.5`, at most 12 backtracks;
- fixed 22 A receptor shell around the frozen pocket center, giving a 12 A
  valid ligand-center envelope for the largest 10 A active interaction cutoff;
- float64 energy, force, projection, and integration.

Every accepted pose objective must be non-increasing within
`1e-10 * max(1, abs(E))`. Batch and independent-loop smoke results must agree
within `rtol=1e-8` and `atol=1e-8`.

## Pre-registered predictions and gates

Predictions:

- The crystal target-term energy is below at least six of eight local-prior
  initial energies when the corresponding typed contact is nonzero.
- Single-term guarded arms lower their target interaction energy, but exact
  model-prior global capture remains unsupported because finite-cutoff
  interaction fields vanish outside their local basins.
- Raw interaction-only descent lowers energy but produces more cut-bond,
  clash, or fragment-connectivity failures than guarded descent.

Local single-term admission signal:

- paired final raw RMSD median improves by at least `0.25 A` versus
  `guard_default`;
- at least five of eight seeds improve RMSD;
- no additional geometry, clash, shell, chirality, or numerical failures.

Exact-prior standalone capture is supported only if at least five of eight
poses pass all of:

- finite trajectory and force;
- fixed-shell validity;
- final raw atom-index RMSD `<2 A`;
- maximum cut-bond error `<=0.2 A`;
- minimum protein-ligand distance divided by UFF-x `>=0.65`;
- no improper/chirality inversion.

Local pose recovery uses raw RMSD `<1 A`. Energy decrease alone is never a
success criterion. A zero target term means the structure does not exercise
that motif; it is not silently counted as evidence.

## Outputs

Every run records input and protocol hashes, code/diff identity, seeds, RNG
draws, active terms, per-pose line-search decisions, per-frame energies,
contacts, geometry metrics, shell validity, and trajectory coordinates.
Results are shown through the inline Run/Play/frame/3D viewer; GIF is not
generated unless explicitly requested.
