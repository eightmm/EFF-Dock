# S50 Confidence Score-only PLINDER Results

Protocol: `EFFDOCK-S50-CONFIDENCE-SCORE-ONLY-PLINDER-V1`

Evaluation date: 2026-08-16

Protocol SHA256: `bab703f0de5c10baff5506afbe88354f3109b2f483dff8425fbbad512fd9eb1b`

## Decision

Retain the S50 sampler and prioritize confidence retraining on sampler-matched
S50/N100/sigma-2 candidate banks. The existing confidence model is a useful
ranking signal, but it is the dominant remaining bottleneck on this bank: it
recovers only `496/805 = 61.61%` of samples for which S50 produced a pose below
2 Angstrom. This is in the preregistered severe-bottleneck band.

The existing selector still improves Top-1 success strongly over both the
first pose and random selection. The result therefore supports retraining the
confidence model rather than discarding it or continuing the retained sampler
recipe. A new confidence model should be trained on allowed training-split
S50/sigma-2 banks and evaluated under a separately frozen confirmation
protocol; this repeated-use cohort must not become an independent-confirmation
claim.

## Frozen cohort and scoring contract

The primary estimand contains 1,035 evaluable samples from 1,020 PLINDER
systems, with exactly 100 saved S50 poses per sample. The confidence checkpoint
and saved candidates were fixed; scoring did not resample, refine, or reorder
poses. The primary `s50_backbone` arm used the retained S50 docking backbone,
and the `matched_backbone` arm used the docking backbone paired with the
original confidence training as a diagnostic only. Both arms used
`sigma=2.0`, chunks of 20, and stable minimum predicted-RMSD selection.

The full PLINDER split has 1,076 samples. The same 41 preregistered
preprocessing failures were excluded from the primary paired estimand and
assigned failure in both arms for the operational full-split sensitivity.

## Primary S50 results

All intervals are 20,000-resample, PLINDER-system-cluster bootstrap 95%
intervals.

| Endpoint | S50 result | 95% CI |
|---|---:|---:|
| Top-1 RMSD <2 A | `496/1,035 = 47.92%` | `[44.84, 51.01]%` |
| First-pose RMSD <2 A | `189/1,035 = 18.26%` | -- |
| Top-1 minus first | `+29.66 pp` | `[+26.28, +33.04] pp` |
| Random expected RMSD <2 A | `17.94%` | -- |
| Top-1 minus random | `+29.98 pp` | `[+27.30, +32.62] pp` |
| Oracle RMSD <2 A | `805/1,035 = 77.78%` | -- |
| Oracle recovery | `496/805 = 61.61%` | `[58.25, 64.99]%` |
| Oracle gap | `309/1,035 = 29.86 pp` | `[27.10, 32.62] pp` |
| Top-5 RMSD <2 A | `691/1,035 = 66.76%` | -- |
| Top-5 rescue over Top-1 | `195/1,035 = +18.84 pp` | `[+16.47, +21.25] pp` |

The S50 bank has 309 selection misses among the 805 oracle-solvable samples,
compared with 230 sampler-unreachable samples. Their registered difference is
`+79/1,035 = +7.63 pp` (95% CI `[+3.28, +11.97] pp`), placing the present error
budget on the confidence-selection side. The actionable `+18.84 pp` Top-5
rescue likewise shows that many near-native poses are already ranked into a
shortlist but not consistently promoted to rank one.

For the joint RMSD-below-2-Angstrom and saved fast-valid endpoint, S50 selected
`425/1,035 = 41.06%` (95% CI `[38.03, 44.13]%`). The fast-valid oracle was
`737/1,035 = 71.21%`, of which the selector recovered `57.67%`.

Under the 1,076-sample operational sensitivity, S50 Top-1 is
`496/1,076 = 46.10%` and the oracle is `805/1,076 = 74.81%`. The diagnostic
matched arm is `512/1,076 = 47.58%`; all three denominators include the same
41 fixed preprocessing failures as failures.

## Matched-backbone diagnostic

The matched arm reached Top-1 `512/1,035 = 49.47%` (95% CI
`[46.36, 52.52]%`) and recovered `63.60%` of oracle-solvable samples (95% CI
`[60.25, 66.87]%`). Its Top-5 rescue was `+17.49 pp` (95% CI
`[+15.18, +19.85] pp`).

The paired S50-minus-matched Top-1 contrast was `-1.55 pp` with 95% CI
`[-3.31, +0.19] pp`, and the selected pose index agreed in `578/1,035 = 55.85%`
of samples. This interval meets neither registered direction gate, so the
backbone effect is inconclusive. It does not justify changing the retained S50
sampler or choosing a confidence backbone from this diagnostic.

## Integrity, execution, and audit

Outcome-blind smoke array `54563` completed all eight shards. It produced the
exact 16 one-sample arm artifacts, verified finite scores and same-runtime
replay within `1e-5`, and emitted no efficacy result. Its array wall span was
about 1 minute 25 seconds. Full array `54571` completed all eight shards and
both arms: 16 sealed score artifacts, 1,035 samples and 103,500 scores per arm.
With four concurrent RTX 6000 Ada tasks, its two-wave wall span was about
25 minutes 34 seconds; individual shard tasks took 12 minutes 11 seconds to
12 minutes 59 seconds.

The full reporter rehashed all 1,035 source SDFs, parsed all 103,500 saved poses,
verified the exact two-arm inventory, finite six-head score arrays, unchanged
candidate identity/order, and stable selector recomputation before joining
labels. Runtime code identity was
`47bddf89a08dcfd95a095a690f41f21a03c2b22a5ccd617e4c7e535f5062fae3`.

The independent auditor source SHA256 was
`25cf974aa983d14e6e6d8f7da7581e2c3cf2c57173939fb5c912bea921d30761`.
It independently rejoined the source inventory, recomputed stable selector
indices, metrics, paired contrasts, and the 20,000-resample bootstrap. All 65
checked numeric report fields agreed exactly (`maximum_absolute_delta=0.0`,
tolerance `1e-12`).

Initial smoke array `54557` stopped on a false topology rejection caused by an
attached hydrogen moving between RDKit's explicit- and implicit-H
representations during V2000 round-tripping. No efficacy labels were opened.
The topology signature was corrected to compare total attached hydrogen count
while retaining exact atom order, element, formal charge, isotope, radical,
aromatic state, bond endpoints, and bond order. The corrected protocol and V2
label-free manifest were then sealed before the successful smoke and full run;
the failed artifacts remain non-claim-bearing provenance.

## Claim boundary

This is a distribution-shift diagnostic on a repeated-use PLINDER validation
cohort whose sampler outcomes had already been inspected. It demonstrates that
the retained confidence signal is operationally useful on the frozen S50 bank
and that confidence selection is the larger current bottleneck. It is not an
independent generalization estimate and cannot select a backbone, selector,
checkpoint, loss, or hyperparameter. External benchmarks remain closed for
confidence retraining and model selection.

## Artifacts

- [Frozen protocol](EARLY_TIME_SAMPLER_S50_CONFIDENCE_PROTOCOL.md), SHA256
  `bab703f0de5c10baff5506afbe88354f3109b2f483dff8425fbbad512fd9eb1b`
- [V2 label-free bank manifest](../outputs/benchmarks/early_time_sampler_s50_confidence_runs/frozen_inputs/label_free_bank.v2.json),
  SHA256 `928b7219ed1ef8375c1ee52470f6ef606b8fca4d5bf4ea5c51355e8332e29a4b`
- [Smoke integrity report](../outputs/benchmarks/early_time_sampler_s50_confidence_runs/1a7491f547e466a1ee0a76fcfee4b90b508249126f98b625c9b65a882b908619/reports/smoke_integrity.json),
  job `54563`, SHA256
  `fabc254c12734a94eb8278fb80d0c63fedd08fbf115b3eb123ac96799e21d317`
- [Full strict report](../outputs/benchmarks/early_time_sampler_s50_confidence_runs/1a7491f547e466a1ee0a76fcfee4b90b508249126f98b625c9b65a882b908619/reports/full_report.json),
  job `54571`, SHA256
  `2141c36146a08c370a5ba8c330ccd752a2eafb83c3f4d5ce8d4ac7387ae6527b`
- [Independent audit](../outputs/benchmarks/early_time_sampler_s50_confidence_runs/1a7491f547e466a1ee0a76fcfee4b90b508249126f98b625c9b65a882b908619/reports/full_independent_audit.json),
  SHA256 `162f55910138b1392bbd3ebcb9e319604b42aa6224fe8c605f0b9e820cd7aa18`
- [Independent auditor source](../scripts/audit_early_time_sampler_plinder_confidence_full.py),
  SHA256 `25cf974aa983d14e6e6d8f7da7581e2c3cf2c57173939fb5c912bea921d30761`
