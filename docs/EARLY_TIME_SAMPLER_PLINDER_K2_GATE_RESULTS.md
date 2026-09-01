# Early-Time Sampler PLINDER K2 Gate Results

Protocol: `EFFDOCK-EARLY-TIME-SAMPLER-PLINDER-K2-GATE-V1`
Evaluation date: 2026-08-16
Protocol SHA256: `0250853ae0793db288be2a6a8dc775db391d25aae32835b65b061782f34ab518`

## Decision

Keep the `t=0` 10% 50k EMA checkpoint and stop the same time-sampling
continuation. Do not promote the existing 50k-plus-10k checkpoint.

The extra 10k updates increased the mean number of candidates below 2
Angstrom by only `+0.1865` per 100-pose set, well below the registered `+1.0`
gate. More importantly, coverage decreased from 805 to 793 of 1,035 evaluable
samples. The additional density was concentrated almost entirely in complexes
that were already easy for the 50k checkpoint.

The next model-development step should be confidence retraining against samples
from the retained 50k EMA checkpoint, not another continuation of this sampler
recipe.

## Frozen comparison

The full PLINDER validation split contains 1,076 samples. A preregistered,
outcome-independent mapping audit admitted 1,035 samples from 1,020 systems and
fixed 41 common preprocessing failures. The primary estimand uses the 1,035
evaluable paired samples; a full-1,076 sensitivity assigns zero candidates to
the same 41 failures in every arm.

All three checkpoints used the deployment-aligned candidate-only inference
contract: `sigma=2.0`, 100 poses, 10 steps, late schedule with power 3, pocket
cutoff 10 Angstrom, prior pool 100, conformer seed 0, and no confidence,
guidance, FK/SDE, or refinement. Per-sample priors, seeds, proteins, ligand
inputs, and references were paired across arms.

The 25k EMA arm was diagnostic only. The registered selection contrast was
50k EMA to the existing 50k-plus-10k `t0p10` continuation.

## Results

| Arm | K2 total | Mean K2 / 100 | K2>=1 | K2>=5 | K2>=10 | Fast-valid K2 total | Fast-valid K2>=1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 25k EMA, diagnostic | 18,262 | 17.6444 | 800 | 649 | 534 | 11,659 | 734 |
| 50k EMA, baseline | 18,569 | 17.9411 | 805 | 638 | 539 | 11,824 | 737 |
| 50k + 10k, treatment | 18,762 | 18.1275 | 793 | 649 | 544 | 11,990 | 725 |

For the primary 50k-to-treatment contrast:

- K2 increased by 193 candidates, or `+0.18647` per complex; the system-cluster
  bootstrap 95% CI was `+0.09565` to `+0.27864`.
- K2>=1 coverage fell by 12 samples, from `77.78%` to `76.62%`; the change was
  `-1.159 pp` with 95% CI `-1.929` to `-0.483 pp`.
- Only one previously unsolved sample gained coverage, while 13 lost it.
- Of 167 fragile baseline samples with K2=1--4, 154 retained coverage
  (`92.22%`), below the 95% guard.
- Fast-valid K2 increased by 166 (`+0.16039` per complex), but fast-valid
  coverage also fell by 12, from 737 to 725 samples.

The gain was strongly tail-concentrated. Of the 193 added K2 candidates, 189
(`97.9%`) came from samples whose baseline K2 was already at least five.
Across all 230 baseline-K2-zero samples, the aggregate change was only `+1`.

First-pose mean RMSD changed from `4.54183` to `4.52269` Angstrom. Oracle mean
RMSD changed from `1.49076` to `1.49666` Angstrom, a slight worsening consistent
with the coverage loss.

## Diversity and selection gates

There was no mode-collapse signal:

- nearest-neighbor RMSD ratio: `0.99198` (95% CI `0.98997` to `0.99404`);
- C2 connected-component ratio: `0.98203` (95% CI `0.97701` to `0.98683`);
- coordinate-unique fraction: `100%` in both arms.

The registered decision nevertheless failed six gates: mean K2 efficacy,
coverage count, coverage CI lower bound, fragile retention, fast-valid coverage
count, and fast-valid coverage CI lower bound. Positive K2 confidence and all
diversity guards passed, but they do not override the coverage failures.

The 41-common-zero full-split sensitivity gave the same direction:
`+0.17937` mean K2 and `-1.115 pp` K2>=1 coverage over all 1,076 samples.

## Integrity and independent coordinate audit

Slurm array `54518` completed all eight shards with exit code `0:0`. The final
inventory contains 24 CSVs, 24 arm summaries, eight paired summaries, 3,105
sample-arm rows, and 310,500 retained candidate poses. Independent CSV
recomputation found no ID, seed, prior-hash, K2, first-RMSD, oracle-RMSD, or
fast-valid ledger mismatch.

The independent coordinate auditor did not import the evaluator or gate-report
aggregator. It sequentially parsed all 310,500 poses plus the selected-pose
artifacts, recomputed RMSD, K2, first/oracle RMSD, unique counts, nearest-neighbor
RMSD, C2 components, the 20,000-resample cluster bootstrap, and the selection
decision. Job `54552` completed with exit code `0:0` in 17 minutes 35 seconds.

The saved-coordinate and CSV-bound decisions both rejected promotion. Among 42
candidates within the frozen 0.0005-Angstrom band around the 2-Angstrom
threshold, none changed K2 classification. Likewise, 2,556 diversity edges
were near the 2-Angstrom boundary but changed no C2 row or gate. The decision
was stable to SDF coordinate quantization. A secondary recheck of 9,600
fast-valid labels found two rounded-coordinate boundary differences; this was
predeclared non-gating and cannot rescue the failed primary and coverage gates.
No mapped-index RMSD fallback was used.

An initial audit attempt (`54523`) correctly stopped on an overly strict audit
contract that treated pose-derived tetrahedral inversion as an atom-topology
change. Controlled reproduction showed identical atom order and constitutional
graph, with only generated 3D chirality changing. The auditor was corrected to
keep bond order, charge, isotope, hydrogen, radical, atom-order, and connectivity
checks while treating generated stereo as a pose outcome. A second wrapper
attempt (`54551`) stopped before the auditor because compute nodes cannot query
the Slurm accounting database. Both failed artifacts were preserved; neither
changed or regenerated the producer outputs.

## Artifacts

- Frozen protocol: `docs/EARLY_TIME_SAMPLER_PLINDER_K2_GATE_PROTOCOL.md`
- Strict report:
  `outputs/benchmarks/early_time_sampler_plinder_k2_paired_runs/t0p10-continuation-plinder-k2-v1-pathfix1-20260815/reports/full_strict_report.json`
  (SHA256 `d4814796a9d274f836888dd614e5b6a4a5fba6b86001da83bea6720fabf02316`)
- Independent coordinate audit:
  `outputs/benchmarks/early_time_sampler_plinder_k2_paired_runs/t0p10-continuation-plinder-k2-v1-pathfix1-20260815/reports/full_coordinate_audit.v3.json`
  (SHA256 `3b6daa4a3d4c74ae384e7c3d2199d3d26f9360fe4b64a33e1c6ab16f4b83eabc`)
- Retained sampler checkpoint:
  `outputs/eff-dock/early-time-t0p10-50k-v1-20260813/checkpoints/step50000_ema_common_init.pt`
  (SHA256 `65be44d7dc8f0867eb9fc5d22214b80f93971ea4702679a527c665046e91e6b6`)
