# EFF-Dock cluster-free confidence filter study

Protocol ID: `EFFDOCK-CONFIDENCE-FILTER-V1`

Pre-registered: 2026-07-20, before fitting filter thresholds.

## Question and hypothesis

Can a conservative, cluster-free filter combine the retained confidence
model's pose and atom heads with an absolute protein-ligand clash signal to
select top-1 poses better than the pure predicted-RMSD head?

The hypothesis is that switching away from the pure-confidence pose only when
another pose is within a small predicted-RMSD margin and is supported by the
success/atom heads or removes a severe clash improves PLINDER validation
top-1 RMSD <2A by at least 1.0 percentage point. The selector must not use
candidate ranks, cluster sizes, pairwise pose distances, crystal coordinates,
or true RMSD at inference.

## Fixed information and data boundary

- Prediction unit: one PLINDER protein/ligand/pocket complex with 80 generated
  poses from `conf_ligonly_extmatch_n80_s25_sig0p5_pc10`.
- Checkpoint:
  `weights/effdock_confidence_extmatch_n80_s25_step42500.pt`.
- Inputs to the selector: predicted pose RMSD, pose success probability,
  mean atom-success probability, and protein-ligand contacts at <=1.6A per
  ligand atom. True RMSD is evaluation-only.
- Threshold fitting: first 1,024 matched PLINDER train complexes only.
- Confirmation: all 1,076 matched PLINDER validation complexes, opened once
  after choosing one configuration on train.
- External benchmark results are forbidden for threshold fitting or
  model selection.

## Selector family

The pure-confidence pose is the base. A head-consensus switch may select the
lowest predicted-RMSD alternative that is within a fixed RMSD margin, does not
increase the clash rate, and improves both the success and atom-success heads
by fixed minimum gains. If the base exceeds an absolute clash limit, a physical
fallback may instead select a within-margin pose below that clash limit while
allowing only fixed, bounded degradation of the two learned success heads.

Every comparison is against fixed values or the base pose. The rule does not
normalize by the number or composition of sampled poses.

## Fit grid and decision rule

- predicted-RMSD margin: `{0.03, 0.05, 0.10, 0.20}` A;
- head gains: `{0.00, 0.02, 0.05, 0.10}` independently for pose and atom heads;
- physical fallback clash limit: `{0.0, 0.05, 0.10, disabled}` contacts per
  ligand atom;
- fallback head tolerance: `{0.00, 0.02, 0.05}`.

Choose the train configuration by lexicographic objective: highest top-1
RMSD <2A, then lowest median RMSD, then fewest switches. The configuration is
admitted only if train improves by at least 1.0 percentage point over pure
confidence and does not worsen median RMSD. It is deployed only if the frozen
validation confirmation also improves by at least 1.0 point, does not worsen
median RMSD, has finite outputs for all 1,076 complexes, and changes at most
35% of selections. Otherwise retain pure confidence for deployment and keep
the historical composite only for compatibility reproduction.

## Artifacts

- Workflow: `python -m effdock.workflows.tune_confidence_filter`.
- Output root: `outputs/eff-dock/confidence-filter-v1/`.
- The result JSON records the checkpoint hash, config, split ranges, commands,
  baseline, fitted configuration, validation result, and decision.

## Pre-validation amendment: asymmetric head guard

Added 2026-07-20 after reading only the train-fit cache and before the
validation cache or metrics existed. The strict rule requiring simultaneous
positive gains from both success heads improved train top-1 <2A only from
57.03% to 57.23% (+0.20 point), below the pre-registered +1.0-point admission
bar. It is rejected.

One additional filter family is permitted before validation is read. An
alternative pose must remain within the same fixed predicted-RMSD margin and
must not increase the absolute clash rate. It may pass by one of three fixed
consensus modes:

1. pose-success improves by a minimum gain while atom-success degrades by no
   more than a fixed tolerance;
2. atom-success improves by a minimum gain while pose-success degrades by no
   more than that tolerance; or
3. either head improves by the gain while the other stays within tolerance.

The train-only grid uses RMSD margins `{0.03, 0.05, 0.10, 0.20}` A, head gains
`{0.02, 0.05, 0.10}`, and other-head tolerances `{0.00, 0.02, 0.05}`. The same
absolute clash fallback grid remains available. Selection and admission use
the original lexicographic objective and +1.0-point/non-worse-median gate. At
most one train-selected asymmetric configuration will be applied to the still
unread full validation cache; validation will not choose or alter it.

## Pre-validation amendment: atom-displacement guard

Added 2026-07-20 while the validation forward pass was still running and no
validation cache or metric existed. The asymmetric probability-head family
improved train <2A only to 57.32% (+0.29 point) and is rejected. A final
already-produced model output, predicted atom RMS displacement, is admitted as
a filter guard; this does not change the model or inference information.

The train-only grid kept the predicted-RMSD base and tested RMSD margins
`{0.03, 0.05, 0.10, 0.20}` A, required atom-RMSD improvements
`{0.00, 0.01, 0.02, 0.05, 0.10}` A, pose-success tolerances
`{0.00, 0.02, 0.05, 0.10}`, and atom-success tolerances
`{0.00, 0.02, 0.05, 0.10}`. Candidates must also have no higher absolute clash
rate than the base. Configurations switching more than 35% on train are
excluded to keep the intervention filter-like.

Before validation was read, the frozen train-selected configuration was:

- predicted-RMSD margin `0.20` A;
- atom-RMSD gain `0.00` A (no worse than base);
- pose-success tolerance `0.05`;
- atom-success tolerance `0.02`;
- clash no worse than base.

It changed 34.18% of train selections and improved train <2A from 57.03% to
58.30% (+1.27 points), with median RMSD improving from 1.729A to 1.715A. This
single configuration, and no other, will be evaluated on full validation. The
original validation deployment gate remains unchanged.

## Outcome

Completed 2026-07-20. The retained checkpoint was evaluated on 1,024 PLINDER
train complexes for fitting and all 1,076 PLINDER validation complexes for the
single frozen confirmation.

| Selector | Train <2A | Train median | Train switch | Val <2A | Val median | Val switch |
|---|---:|---:|---:|---:|---:|---:|
| Pure confidence | 57.03% | 1.729A | 0.00% | 53.90% | 1.839A | 0.00% |
| Strict consensus/clash filter | 57.23% | 1.726A | 4.69% | 54.18% | 1.834A | 3.72% |
| Frozen atom-RMSD guard | 58.30% | 1.715A | 34.18% | 53.72% | 1.835A | 32.90% |

The strict filter improved validation by only +0.28 point, below the +1.0-point
deployment gate. The atom-RMSD guard met the train gate (+1.27 points) but
reversed on validation (-0.19 point). Both hypotheses are rejected for
deployment. External benchmark labels were not opened. Pure confidence becomes
the cluster-free `auto` default; both filters remain explicit diagnostics, and
the historical composite remains available only by its full selector name.

## External characterization

Authorized by the user on 2026-07-20 after the PLINDER deployment decision was
already frozen. Although neither filter passed the validation deployment gate,
both frozen configurations are evaluated once on the existing matched
N80/S25/sigma0.5/pocket10 Astex and PoseBusters candidate artifacts. This is
external characterization only: the results cannot alter thresholds, selector
logic, or the pure-confidence deployment decision.

Primary external metric is selected-pose symmetry-aware RMSD <2A. Median RMSD
and the stored fast-validity flag are secondary. Official PoseBusters pass-all
requires complete selected SDFs and is reported separately; it must not be
conflated with the stored fast-validity proxy.

The frozen external characterization produced:

| Dataset | Pure <2A | Strict <2A | Atom guard <2A | Strict switches | Atom switches |
|---|---:|---:|---:|---:|---:|
| Astex Diverse (85) | 78.82% (67) | 78.82% (67) | 78.82% (67) | 3 | 29 |
| PoseBusters v2 (308) | 73.38% (226) | 73.38% (226) | 74.03% (228) | 10 | 113 |

The pre-registered external non-inferiority prediction for the strict filter
was met exactly: neither dataset lost an RMSD<2A success. It also produced no
net gain. The rejected atom guard preserved Astex and added two net
PoseBusters successes, but this does not overturn its PLINDER validation
failure or the pure-confidence deployment decision. The stored fast-validity
proxy was 100% for all selected poses; official PoseBusters pass-all was not
run because complete selected SDFs for this historical N80 candidate set were
not retained.
