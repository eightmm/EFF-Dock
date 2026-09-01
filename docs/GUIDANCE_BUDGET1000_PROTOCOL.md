# Frozen unified-guidance fixed-budget protocol

Protocol ID: `EFFDOCK-UNIFIED-GUIDANCE-BUDGET1000-V1`

Frozen: 2026-07-31, before guided Astex/PoseBusters sampling outcomes are
opened.

## Question and non-claim

At a fixed learned-model budget of 1,000 pose-steps per complex, does wider
sampling or finer ODE integration produce better oracle coverage, and does the
experimental self-contained `GuidanceEnergy` corrector improve the joint
accurate-and-valid pose set?

This is a reference-pocket redocking diagnostic. Astex Diverse and
PoseBusters v2 use the already frozen reference-derived pocket centers only to
define the benchmark site, and crystal coordinates are used only after
sampling for RMSD. External outcomes cannot select or change a guidance term,
coefficient, scale, schedule, force cap, trust region, or production default.
The experiment does not admit guidance to the production sampler.

## Falsifiable hypotheses

- Budget allocation: oracle RMSD `<2 Å` is highest for `100 poses × 10
  steps`, followed by `50 × 20` and `40 × 25`, because candidate diversity
  dominates integration accuracy at this 1,000 pose-step budget.
- Guidance effect: on the pre-outcome chemistry-eligible cohort, unified
  guidance improves `any(fast-valid and RMSD <2 Å)` by at least 2 percentage
  points versus the paired unguided arm.
- Guard: guided oracle RMSD `<2 Å` must not decrease by more than 2 percentage
  points, and no accepted correction may violate finite, descent, or maximum
  atom-displacement checks.

A null or reversed ordering, a joint-success gain below 2 points, an oracle
loss beyond the guard, or systematic rejection/non-finite behavior
disconfirms the corresponding claim.

## Frozen model and input boundary

- Docking checkpoint:
  `weights/effdock_geometry_ft_100k_best.pt`
  (`sha256:6932fb3ba6ebac770f714453529656a44b8f33cf15119d23c9e675d2d60b36db`).
- Confidence is disabled for the primary run. It is distribution-matched to
  N80/S25 and is irrelevant to the sampling-oracle question.
- Ligand prior sigma: `0.5 Å`.
- Pocket crop: `10 Å`; center jitter: `0 Å`.
- Time schedule: late, power `3`.
- Base seed: `42`, offset by global sorted complex ID exactly as in the frozen
  benchmark evaluator.
- A deterministic 100-pose translation/SO(3) prior pool is generated once per
  complex. N100 uses all poses, N50 the first 50, and N40 the first 40.
  Guided and unguided arms therefore begin from identical states, and budget
  cells use nested prior prefixes.
- Datasets: Astex Diverse 85 and PoseBusters v2 308 frozen IDs.
- Guidance eligibility is determined before outcomes using only ligand
  chemistry, receptor chemistry, the frozen pocket center, and the declared
  `18 Å` guidance shell. Exact included IDs, complement exclusion counts,
  structured failure-code counts, and hashes are frozen in
  `GUIDANCE_BUDGET1000_ELIGIBILITY.json`.

Unsupported active cofactors, unresolved/clustered metal sites, identity
mismatches, or other unsupported chemistry fail closed. Headline results must
show full-dataset eligibility coverage and supported-cohort results
separately; supported-only percentages must never use 85/308 as an implicit
denominator.

## Arms and fixed budget

| Cell | ODE steps | Poses | Learned model pose-steps |
|---|---:|---:|---:|
| N100/S10 | 10 | 100 | 1,000 |
| N50/S20 | 20 | 50 | 1,000 |
| N40/S25 | 25 | 40 | 1,000 |

Every cell has a paired `unguided` and `guided` arm. The 1,000 budget counts
learned model evaluations, not equal FLOPs: the guided arm additionally
evaluates energy, gradients, and backtracking trials, which are recorded.

## Frozen experimental guidance corrector

- Energy: `GuidanceEnergy = PhysicalEnergy + InteractionEnergy`.
- Interaction profile: all seven current default terms.
- Coupling: learned ODE proposal first, followed by an operator-split guidance
  correction.
- Scale: `0.1`, fixed before external outcomes.
- Start time: `t=0.5`; linear ramp.
- Physical soft-core annealing: `1.5 Å → 0.75 Å`.
- Atom-force cap: `20`; fragment translation/angular velocity caps: `5/5`.
- Maximum accepted atom displacement per ODE step: `0.25 Å`.
- Backtracking: factor `0.5`, at most `8` reductions.
- Protein shell around the explicit pocket center: `18 Å`.
- Acceptance: every pose independently requires finite coordinates/energy,
  non-increasing energy within `atol=rtol=1e-6`, and the displacement trust
  region. Otherwise the correction is shrunk or exactly rejected.

Scale `0.1` is a conservative pre-outcome numerical choice: after caps the
largest un-ramped added fragment velocity is `0.5` in each translation/angular
channel before multiplication by the ODE interval; the stricter atom
displacement gate remains authoritative. It is not copied from Vina and will
not be tuned on Astex or PoseBusters.

## Metrics and interpretation

Primary:

- per-complex minimum symmetry-aware heavy-atom RMSD;
- oracle success at `<2 Å` on the frozen eligible cohort.

Secondary:

- oracle `<1/<3/<5 Å`, mean and median RMSD;
- fast-validity of the RMSD oracle;
- number/fraction of fast-valid candidates;
- `any(fast-valid and RMSD <2 Å)`;
- minimum RMSD among fast-valid candidates;
- official PoseBusters redock validity of the saved RMSD-oracle pose;
- preprocessing/chemistry failure and eligibility coverage;
- accepted/rejected corrections, backtracks, non-finite trials, maximum
  accepted atom displacement;
- wall time and CUDA peak allocated/reserved memory.

Guided-minus-unguided effects use identical IDs and nested initial priors.
Report paired bootstrap 95% confidence intervals over complexes with a fixed
bootstrap seed. Any runtime failure is reported as diagnostics and invalidates
the strict paired effect report; no survivor-only or post-failure complete-case
effect is emitted. One external run is exploratory evidence and does not
measure sampling-seed uncertainty.

## Execution gates

Before the full Slurm array:

1. focused CPU tests for operator order, exact no-op behavior, energy descent,
   trust-region rejection, batched fused-ring aggregation, and shared priors;
2. one N100/S10 GPU smoke on a chemistry-eligible large ligand;
3. no CUDA OOM/non-finite result; peak allocated/reserved memory recorded;
4. smoke guided and unguided arms start from the same stored prior pool;
5. full aggregation rejects missing/duplicate IDs, inconsistent hashes or
   settings, non-1,000 budgets, and silent shard failures.

The frozen launcher is `scripts/slurm/guidance_budget1000_array.sbatch`; the
strict aggregator is `python -m effdock.workflows.guidance_budget_report`.
