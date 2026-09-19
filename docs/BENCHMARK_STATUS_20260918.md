# Benchmark inventory — 2026-09-18 snapshot

Historical snapshot. For completed repair/unguided results and the audited
manuscript evidence package, see [2026-09-19 results](paper/20260919/README.md).
Pending labels below describe the snapshot date, not the current analysis state.

This is a current-status snapshot, not a replacement of historical single-bank
results or a new production/checkpoint selection. Percentages are three-seed
means ± sample standard deviations unless noted. No new evaluations were
submitted for this inventory.

## Current EFF-Dock external campaigns

Early-time S50 sampler + U70k confidence, N100/S10/sigma2. The five rows below
use raw/refined banks, pure predicted-RMSD ranking and an input-SMILES
**chirality-only** reselection ablation (not E/Z). RMSD is symmetry-aware,
no-alignment, strictly <2 A. PB-valid excludes RMSD; joint requires both.
The filter falls back to baseline Top-1 when every candidate fails.

Provenance correction after inspecting baseline manifests: Astex/PB uses
normalized-drift guidance eta2 in the frozen cutoff campaign; temporal
PhiBench/FoldBench/OpenBind here is unguided. These are not one homogeneous
unguided production experiment. N40/S25 follow-up preserves Astex/PB eta2.

| Dataset | N per seed | Status |
| --- | ---: | --- |
| Astex | 85 | All three repeats and paired PB evaluation complete |
| PoseBusters v2 | 308 | All three repeats and paired PB evaluation complete |
| PhiBench reconstructed full | 206 | All three repeats and paired PB evaluation complete |
| FoldBench holo-pocket | 558 | All generation/refinement/selection complete; 3 PB shards recovering |
| OpenBind full | 925 | All three repeats and paired PB evaluation complete |

Temporal generation accounts for 5,067 complexes. The 69/72 original temporal
PB shards are complete; replacement array 77487 is running, after successful
three-seed YKX smoke 77432. Aggregate 77490 waits on it. No FoldBench PB/joint
percentage is inferred from the incomplete cohort. The recovery uses an
explicitly disclosed PB energy-reference InChI representation adapter and
separate output root; see `PB_YKX_DIAGNOSIS.md`.

### Four-condition results (means, %)

Each cell is RMSD <2 A / PB-valid / joint. Full standard deviations for the
refined+filtered arm are given below. Pending is not zero.

| Dataset | Raw | Raw + chirality filter | Refined | Refined + chirality filter |
| --- | --- | --- | --- | --- |
| Astex | 81.18 / 70.59 / 61.96 | 82.35 / 81.57 / 70.59 | 82.35 / 93.33 / 78.04 | 82.35 / 94.51 / 78.43 |
| PoseBusters | 80.63 / 61.69 / 55.30 | 79.87 / 69.70 / 61.26 | 82.68 / 93.07 / 78.90 | 82.47 / 94.81 / 79.55 |
| PhiBench206 | 60.36 / 51.46 / 37.54 | 58.41 / 56.31 / 39.64 | 63.27 / 91.42 / 58.90 | 62.62 / 94.01 / 60.19 |
| FoldBench558 | 72.46 / pending / pending | 72.04 / pending / pending | 74.43 / pending / pending | 74.49 / pending / pending |
| OpenBind925 | 49.55 / 49.15 / 26.56 | 44.90 / 63.06 / 32.83 | 52.79 / 98.74 / 52.29 | 52.58 / 99.64 / 52.58 |

| Refined + chirality filter | RMSD <2 A | PB-valid | Joint |
| --- | ---: | ---: | ---: |
| Astex | 82.35 ± 1.18 | 94.51 ± 0.68 | 78.43 ± 1.36 |
| PoseBusters | 82.47 ± 0.65 | 94.81 ± 0.86 | 79.55 ± 1.49 |
| PhiBench206 | 62.62 ± 3.79 | 94.01 ± 0.28 | 60.19 ± 3.36 |
| FoldBench558 | 74.49 ± 0.37 | pending | pending |
| OpenBind925 | 52.58 ± 0.61 | 99.64 ± 0.17 | 52.58 ± 0.61 |

Interpretation: refinement produces a large validity/joint improvement on all
four complete cohorts. Adding chirality selection after refinement has a
smaller positive mean joint change: +0.39/+0.65/+1.29/+0.29 percentage points
for Astex/PB/PhiBench/OpenBind. It does not uniformly improve RMSD-only success.
These are descriptive means, not significance claims or selector admission.

## PoseX: separate official-style relaxation protocol

All SD 718 and CD 1,312 cases completed for seeds 101/202/303, in both the
baseline and chirality+E/Z selection arms. The latter selects first, then
runs PoseX relaxation; it is not our own deterministic refinement and not
post-relaxation reselection. PoseX uses RMSD <=2 A and its native grouping.
CD is aggregated over 109 groups, not a simple percentage of 1,312 cases.
Its grouped joint test requires mean group validity >=0.5. Do not combine
these endpoints with the five per-complex redocking cohorts above.

| Dataset / arm | RMSD success | Joint |
| --- | ---: | ---: |
| PoseX-SD baseline + PoseX relax | 74.56 ± 1.12 | 59.52 ± 1.26 |
| PoseX-SD chirality+E/Z selection + PoseX relax | 72.89 ± 1.08 | 69.64 ± 0.64 |
| PoseX-CD baseline + PoseX relax | 63.91 ± 2.80 | 54.43 ± 1.40 |
| PoseX-CD chirality+E/Z selection + PoseX relax | 59.94 ± 3.71 | 59.33 ± 3.47 |

Mean joint rises +10.12 pp (SD) and +4.89 pp (CD), while RMSD-only success
drops -1.67 and -3.98 pp. Improved validity is not universal docking accuracy
improvement. All 12 source CSVs were re-aggregated using the official-style
code, checked against saved counts/metrics, and baseline source hashes checked.
Only pandas performance warnings occurred; validation passed.

## Historical and external-model artifacts

- Historical U50/U70/U100 comparisons and single-bank Astex85/PB308,
  PhiBench203, FoldBench558/fixed66 and OpenBind860 remain in
  `BENCHMARK_RESULTS.md`. Those numbers are not current full-cohort 3-seed means.
- Earlier PhiBench203 Top-5: RMSD 76.85%, joint 73.89%, single bank. There is
  no verified new PhiBench206 three-seed Top-5 result in this snapshot.
- Saved local three-repeat RMSD comparison reports cover SigmaDock,
  SurfDock+force optimization, RLDiff, DiffBindFR, DiffDock-Pocket and Vina
  on Astex/PB. SigmaDock and SurfDock also have joint endpoints in the report.
  Interformer is marked pending in that report; its live jobs were not audited
  here. No new claims about currently running baseline jobs are made.
- FoldBench paper cofolding numbers are not directly comparable with our
  experimental-holo-receptor pocket-redocking task. PhiBench paper Top-5 is
  not directly comparable with our current Top-1 table.
- Full PhiBench206 includes three reconstructed/imputed systems, not a
  demonstrated exact match to the authors' manifest. OpenBind925 includes
  quality-flagged and two covalent cases treated as noncovalent approximations.
  This snapshot applies no new overlap or quality exclusions. Old guided banks
  and new unguided temporal banks are not interchangeable repeats.

## Sources and verification

### N40/S25 follow-up submitted

Current numbers are now the reporting baseline in README and BENCHMARK_RESULTS.
Matched guided-eta2 Astex85/PB308 N40/S25 protocol is in
`ASTEX_PB_N40_S25_PROTOCOL.md`; all old banks and weights remain unchanged.
Preparation 77561 completed. All six generation/refinement/confidence smoke
tasks 77564 completed; CPU selection/PB smoke 77578 verified all 24 paired
rows and completed in 21 seconds. Full generation 77567 was observed running
four GPU tasks. Selection 77570, PB 77572 and report 77574 form an afterok chain.
Full output: `outputs/benchmarks/astex_pb_n40_s25_r3_v1/comparison.md` when complete.
Thirteen focused tests and changed-file Ruff/Python/shell checks passed.

### Unguided matched controls submitted

Recovery update: V1 smoke 77780 failed all twelve exact-prior checks after
generation/refinement/scoring. CPU reproductions 77850/77851 isolated
host-dependent quaternion rounding: gpu1 matched all failed unguided hashes,
gpu3 matched all guided hashes; translations were bitwise equal and maximum
quaternion difference was 1.1920928955078125e-7. No pairing tolerance changed.
See `ASTEX_PB_UNGUIDED_HOST_RECOVERY.md` for the host-matched successor.

Replacement preparation 77852 completed. GPU smoke 77853 is pinned to gpu3
and was pending for resources at submission verification; actual end-to-end
smoke success is not yet claimed. CPU post-smoke 77854 -> full generation
77855 (also after guided 77567) -> selection 77856 -> PB 77857 -> report
77858 (also after guided report 77574). Superseded pending jobs 77781–77785
were cancelled after replacement submission. V1 data and all guided data
remain intact. New output root:
`outputs/benchmarks/astex_pb_unguided_r3_hostmatched_v2`.
Sixteen focused tests passed with `PYTHONPATH=.:src`; the initial invocation
without that path failed collection and was corrected. Changed-file Ruff,
Python compilation and shell syntax passed. Full fast checks were not rerun
because of the previously recorded unrelated lint failures.

The following paragraph records the original V1 submission, not current jobs.

Protocol: `ASTEX_PB_UNGUIDED_R3_PROTOCOL.md`. Astex85/PB308, three seeds,
both N100/S10 and N40/S25 now have queued unguided controls. Existing guided
eta2 banks are reused. Each budget will report guidance on/off × refinement
on/off × chirality filtering on/off, not an individual energy-term ablation.
The generation gate requires identical initial-prior hashes and sampling
seeds against the corresponding guided bank, as well as matching input hashes
and pose counts; mismatches stop downstream work.

Preparation 77779 completed. At submission verification, GPU smoke 77780
(12 tasks, concurrency 2) was pending for resources, not yet running.
CPU post-smoke 77781 follows it. Full generation 77782 (96 tasks, concurrency
4; 2,358 complex-repeat runs) waits for both 77781 and existing guided
generation 77567. Selection 77783 and PB evaluation 77784 follow successively.
Final report 77785 waits for both 77784 and guided report 77574.
This is a Slurm afterok chain, not a separate notification monitor.

Output: `outputs/benchmarks/astex_pb_unguided_r3_v1/factorial_report.md`
and `.json` after successful completion. The JSON includes paired
guided-minus-unguided gains/losses and refinement/filter deltas. No new
unguided performance numbers or successful runtime pairing checks are claimed
yet. Twelve focused tests, changed-file Ruff, Python compilation and shell
syntax checks passed. Existing weights and banks remain unchanged.

Paths below are relative to the repository; benchmark artifacts remain ignored.

- `outputs/benchmarks/external_chirality_u70k_three_seed_v1/pb_full/three_seed_summary.json`
- `outputs/benchmarks/external_chirality_u70k_temporal_full_r3_v1/full/*/records.json`
- `outputs/benchmarks/external_chirality_u70k_temporal_full_r3_v1/pb_full/*/shard-*/results.json`
- Pending recovery: same parent, `pb_inchi_compat_v1/`; eventual source hashes
  and evaluation profiles in `provenance.json`.
- `outputs/benchmarks/posex_official_protocol/upstream_recovered_evaluation_v1_summary.json`
- `outputs/benchmarks/posex_official_protocol/chirality_ez_posex_relaxed_summary.json`
- `benchmarks/results/external_models/pocket_only_executed_reruns.json`

Temporal means/SD were recomputed from all three selection ledgers and the
complete eight-shard PB cohorts, with exact per-arm count assertions. Astex/PB
values were read from their completed three-seed aggregate. No training,
selection, PB configuration, report values or job dependencies changed during
this inventory. No git publication was performed.
