# Corrected-Rw sensitivity experiments

Status: complete; all 24 conditions verified. Frozen before new outcomes are opened.

Evaluate supplied-pocket sensitivity and the historical guidance/budget controls
with the released docking and confidence weights, corrected `rw` inference,
and seeds 42, 100042, and 200042. Retain all 85 Astex Diverse Set and 308
PoseBusters v2 complexes, their globally sorted per-complex seeds, full heavy-atom
mapping, and conformer seed 0 for generation. References enter retrospective
evaluation only. No retraining, selector tuning, or outcome-dependent retries.

## Conditions

1. Generation pocket radius {6, 8, 10, 12, 14} Å at translation sigma 2,
   crossed with supplied-center jitter sigma {0, 1, 2} Å per Cartesian axis.
2. Translation sigma {1, 2, 4} at generation radius 10 Å, crossed with the same
   center jitter. Sigma-2 cells are shared with the first grid, not regenerated.
3. Four equal learned pose-step budget controls: N100/S10 and N40/S25, each
   unguided or normalized-drift guidance eta 2, at radius 10 Å, sigma 2,
   and zero center jitter. Equal pose-step counts do not imply equal runtime.

The two sensitivity grids use unguided N100/S10 generation and the current
chirality-filtered minimum-predicted-RMSD selector. This intentionally replaces
the historical guided, unfiltered robustness protocol; direct before/after
claims about the orientation operator cannot be made from that comparison.
The already completed Rw N100/S10, sigma-2, radius-10, zero-jitter baseline is
reused by source hashes. There are 24 distinct conditions including that
baseline, 23 newly executed conditions, and 27,117 new complex-seed executions.
There is no full radius-by-prior factorial; each axis changes one factor.

## Fixed pipeline and paired randomness

Use the docking early-time/t0-replay 50k weights (SHA-256
`65be44d7dc8f0867eb9fc5d22214b80f93971ea4702679a527c665046e91e6b6`)
and raw+refined U70k confidence weights (SHA-256
`ce59be42f0ca613871ca079127c3296f5ca9a4ec72e44a9e5cf61878351c2638`).
Both retain their historical training operator. Fresh inference and confidence
features use `rw`; cross-operator confidence provenance remains explicit.

Keep late-power-3 sampling and generation chunks of 10. The center-jitter RNG
is independent of the docking prior RNG. Matching seeds use the same normal
translation and rotation draws across sigma arms; sigma scales translations
only. Bitwise prior identity was the registered within-budget check; the
post-execution CPU audit below documents its hardware-related deviation. No nested
N40/N100 prior claim is made.

Only generation preprocessing receives the tested crop and center perturbation.
Refinement and confidence use the original supplied center and a fixed 10 Å
crop. All generated candidates receive the unchanged rigid energy refinement:
100-step cap, 0.1 Å translation/atom displacement caps, 5° rotation cap,
12 backtracks, absolute/relative energy plateaus 0.02 kcal/mol and 0.001,
patience 5 and minimum 25 steps, physical cutoff 8 Å and receptor shell 18 Å.
Confidence uses the actual source sigma and chunks of 20. Chirality selection
uses input tetrahedral constraints only, with unfiltered minimum-pRMSD fallback
if no candidate is eligible. Also retain unfiltered/raw outcomes for controls.
Guided budget arms retain the historical eta-2 normalized drift, start 0.5,
ramp power 1, force cap 20, translation/angular caps 5, atom cap 0.25 Å,
8 backtracks, shell 18 Å, and conservative geometry-only receptor policy.

## Hypotheses and reporting

Larger center perturbations may reduce success; a crop increase may plateau;
larger prior sigma may trade oracle coverage against selection accuracy.
Directions are hypotheses, not required admission outcomes. Report gains,
losses and null effects without changing production settings. External sets
have been repeatedly examined; these are descriptive sensitivity results,
not independent model selection or proof of unseen-target generalization.

Primary metric: selected refined RMSD <2 Å and all 27 non-RMSD official
PoseBusters 0.6.5 redock checks, higher is better. Report RMSD-only success,
PB validity, Oracle-N RMSD success, raw/refined and filtered/unfiltered results,
and three-seed mean ± sample SD (ddof=1), with fixed cohort denominators.
Small-sigma and large-sigma confidence are domain-shift measurements because
the confidence training banks used sigma 2.

## Execution and publication gates

Freeze source, cohort, checkpoint and baseline-report hashes. Before full
execution, check first and largest ligands, extreme radius/prior settings,
guided N40, complete finite candidate banks, fresh Rw features, paired prior
identity, and native selected-pose PoseBusters. Resource probes do not enter
reported measurements. Only validated GPU backends may receive full tasks;
CPU-only work stays on cpu_only. Use scheduler-minimum CPU allocations and
no agent-defined GPU concurrency cap; scheduler limits still apply.

Missing/duplicate IDs, nonfinite values, changed hashes, incomplete atom maps,
or evaluation errors stop strict aggregation. Preserve failed attempts and
repair identified execution faults without selecting a favorable realization.
No success-only denominator or missing-cell imputation. Publish new numerical
tables, restored figures, captions, combined PDF and Prism ZIP only after the
complete cohort and packaging checks pass. Mixed GPU execution is recorded and
does not establish hardware-invariant predictions.

Execution amendment before the full cohort: existing numerical stages passed
68 CPU checks, and all 20 Ada generation/refinement/confidence smoke cases
completed. N40 selected-pose PB extraction initially rejected the valid
40-pose bank because the older helper hard-coded 100. Retain those failed
attempts and use a named adapter that checks the declared 40/100 bank size,
readability and selected pose identity without changing coordinates, selection
or PB checks. Reevaluate only unfinished PB labels from the same saved banks.
The largest-PoseBusters, radius-14 resource probe exceeded A5000's 24 GB memory
before producing valid outputs. Start full execution on the passed 48 GB Ada
backend; pending 96 GB resource probes may enable pending-task reassignment
after integrity and memory checks. No generation chunk reduction or failed-case
exclusion is permitted. Resource probes remain outside reported measurements.

## Post-execution prior identity audit

Strict aggregation initially stopped at unequal initial-pose hashes. Reconstructing
all 28,296 observed pools (4,716 unique seed/fragment/budget/sigma combinations)
on the original two CPU backends exactly reproduced every recorded hash. Within
877 matched combinations, executed conditions used different backend hashes.
Translation arrays were identical; quaternion components differed by at most
1.7881393432617188e-7, corresponding to a maximum rotation difference of
2.8106261854680336e-7 radians (0.0000161 degrees). Thus random seeds and fragment
inventories match, while initial rotations are not bitwise identical across CPUs.
This is a documented deviation from the original exact-hash pairing requirement.
It does not establish hardware-invariant model predictions.

The amended aggregation checks every original hash against its reproduced pool,
unchanged sampling CSV hashes, complete IDs/seeds and the bounded tensor comparison.
No candidate generation was rerun, no outcomes were used to select a realization,
and all numerical model/refinement/confidence settings remain frozen. The public
numerical export retains the audit summary and its source-manifest hash.
