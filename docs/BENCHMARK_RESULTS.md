# EFF-Dock benchmark results

This document separates the current guided/refined research characterization
from older compatibility baselines. Astex Diverse and PoseBusters v2 have been
used repeatedly during development, so their current values are descriptive
and cannot select or promote a checkpoint. Raw benchmark structures and
generated poses remain outside Git.

## Current N100/S10 guided and refined characterization

Protocol: `EFFDOCK-S50-SYMMETRY-CONFIDENCE-REFINED-EXTERNAL-V1`

The frozen candidate bank contains 100 poses for every Astex Diverse (`N=85`)
and PoseBusters v2 (`N=308`) complex. Candidates use 10 ODE steps, translation
prior `sigma=2`, normalized direct-drift GuidanceEnergy with `eta=2`, and the
adaptive in-repository physical refinement. Current reporting uses the
terminal U50k confidence checkpoint.

| Dataset | N | Raw Top-1 <2A | Refined Top-1 <2A | Refined oracle <2A | Refined PB-valid | Refined joint valid+<2A |
|---|---:|---:|---:|---:|---:|---:|
| Astex Diverse | 85 | 70/85 (82.35%) | 73/85 (85.88%) | 82/85 (96.47%) | 80/85 (94.12%) | 69/85 (81.18%) |
| PoseBusters v2 | 308 | 241/308 (78.25%) | 259/308 (84.09%) | 295/308 (95.78%) | 289/308 (93.83%) | 250/308 (81.17%) |

Top-1 is the stable argmin of confidence-predicted RMSD. RMSD is
symmetry-aware, no-alignment heavy-atom RMSD `<2 A`. `joint` requires the same
selected pose to satisfy RMSD and all 27 non-RMSD PoseBusters 0.6.5 `redock`
checks. The registered internal PLINDER rule selected U25k; U50 is a post-hoc
reporting convention and does not rewrite that selection. The exact checkpoint
hashes, candidate information boundary, adaptive stopping equation, selector,
denominators, and validity definition are in the
[`S50 refined external protocol`](S50_SYMMETRY_CONFIDENCE_REFINED_EXTERNAL_PROTOCOL.md);
complete training and external results are in the
[`S50 symmetry-confidence results`](S50_SYMMETRY_CONFIDENCE_RESULTS.md).

## Reported literature context

The table below is a compact snapshot of recent primary-source reports. It is
not a controlled leaderboard: pocket information, receptor input, candidate
count, ranking, minimization, RMSD code, and PoseBusters version differ.
`Joint` means that the selected pose satisfies both the source's 2 A RMSD
criterion and its PB-valid definition.

| Dataset | Method | Top-1 RMSD (%) | Top-1 joint (%) | Important setting difference |
|---|---|---:|---:|---|
| Astex `N=85` | SurfDock | 95.29 | 91.76 | minimized |
| Astex `N=85` | SigmaDock | 90.60 | 90.60 | N40, energy/PB ranker |
| Astex `N=85` | Matcha | 85.90 | 82.40 | blind; GNINA minimization/filter/ranking |
| Astex `N=85` | **EFF-Dock U50k** | **85.88** | **81.18** | N100/S10, guidance + adaptive refinement |
| PBv2 `N=308` | AlphaFold 3 | 93.20 | 84.40 | co-folding, pocket specified |
| PBv2 `N=308` | PocketXMol self-ranking | NR | 84.70 | source reports PBv2 joint only |
| PBv2 `N=308` | **EFF-Dock U50k** | **84.09** | **81.17** | N100/S10, guidance + adaptive refinement |
| PBv2 `N=308` | SigmaDock | 80.50 | 79.90 | N40, energy/PB ranker |
| PBv2 `N=308` | DiffDock-Pocket RL++ | 80.20 | 78.20 | Smina/Vina minimization + GNINA reranking |
| PBv2 `N=308` | Matcha | 65.90 | 63.00 | blind; GNINA minimization/filter/ranking |

The complete Astex, PBv2, legacy PB `N=428`, denominator-mismatched
`N=306`, and OpenBind cross-method tables are in the
[`literature benchmark comparison`](LITERATURE_BENCHMARK_COMPARISON.md).
The machine-readable plotting subset is
[`LITERATURE_RMSD_COMPARISON.json`](LITERATURE_RMSD_COMPARISON.json).

## Recent and target-family cohorts

Protocol: `EFFDOCK-EXTERNAL-TEMPORAL-GUIDED-REFINED-V1`

The identical frozen N100/S10, sigma-2, eta-2 guided and adaptively refined
stack was then evaluated without retuning on three additional pocket-redocking
cohorts. PhiBench is an EFF-Dock-derived, high-identity-deduplicated cohort;
FoldBench is a pocket-redocking adaptation rather than its native leaderboard
task; the 860-complex OpenBind cohort is a clean non-covalent target-family
characterization and is distinct from the 802-complex official-style cohort
below.

| Dataset | N | Raw Top-1 <2A | Refined Top-1 <2A | Refined oracle <2A | Refined PB-valid | Refined joint valid+<2A |
|---|---:|---:|---:|---:|---:|---:|
| PhiBench derived | 203 | 125/203 (61.58%) | 132/203 (65.02%) | 179/203 (88.18%) | 186/203 (91.63%) | 122/203 (60.10%) |
| FoldBench P-L adaptation | 66 | 43/66 (65.15%) | 41/66 (62.12%) | 58/66 (87.88%) | 61/66 (92.42%) | 40/66 (60.61%) |
| OpenBind clean non-covalent | 860 | 414/860 (48.14%) | 445/860 (51.74%) | 773/860 (89.88%) | 847/860 (98.49%) | 438/860 (50.93%) |

Top-1 is the stable U50k confidence selection; oracle is the best of the same
100 candidates and is not deployable. RMSD is symmetry-aware,
no-alignment heavy-atom RMSD `<2 A`. PB-valid requires all 27 non-RMSD
PoseBusters 0.6.5 `redock` checks, and joint requires validity and RMSD success
for the same confidence-selected pose. Raw structures and pose-level outputs
remain outside Git. Dataset provenance and cohort construction are in the
[`external benchmark registry`](EXTERNAL_TEMPORAL_BENCHMARKS.md), the frozen
measurement contract is in the
[`guided/refined protocol`](EXTERNAL_TEMPORAL_GUIDED_REFINED_PROTOCOL.md), and
the U50 score-only reporting override is in the
[`U50 reporting protocol`](EXTERNAL_TEMPORAL_U50_REPORT_PROTOCOL.md). The
full counts and artifact hashes are in the
[`guided/refined results`](EXTERNAL_TEMPORAL_GUIDED_REFINED_RESULTS.md).

## OpenBind EV-A71 2A official-style Top-N aggregation

Protocol: `EFFDOCK-OPENBIND-OFFICIAL-TOP25-V1`

The same N100/S10, sigma-2, eta-2 guided and adaptively refined inference stack
was re-aggregated under the public OpenBind `filtered=True,
scaffold_only=True` contract. The denominator is 802 complexes; EFF-Dock has
predictions for 786, and all 16 missing predictions remain in the denominator
as failures. Candidates are ranked by the U50k confidence checkpoint.

| Rank budget | Any PB-valid | PB-valid + BiSyRMSD <=2 A | + LDDT-PLI >=0.8 |
|---|---:|---:|---:|
| Top-1 | 773/802 (96.38%) | 417/802 (52.00%) | 359/802 (44.76%) |
| Top-5 | 786/802 (98.00%) | 603/802 (75.19%) | 511/802 (63.72%) |
| Top-25 | 786/802 (98.00%) | **695/802 (86.66%)** | **581/802 (72.44%)** |

OpenBind's public cross-method figure is an any-pose Top-25 endpoint rather
than a deployable Top-1 selector comparison. EFF-Dock Top-1 is reported above
to expose the remaining selection gap; Top-25 is the aligned public comparison.
The complete method comparison, source-table provenance, hashes, and claim
boundary are in the
[`OpenBind official-style results`](OPENBIND_OFFICIAL_TOP25_RESULTS.md); the
frozen metric contract is in the
[`OpenBind official-style protocol`](OPENBIND_OFFICIAL_TOP25_PROTOCOL.md), with
the selector-only U50 override in the
[`U50 reporting addendum`](OPENBIND_OFFICIAL_TOP25_U50_PROTOCOL.md).

## Physical-selector compatibility baseline

Protocol: `EFFDOCK-REDOCK-EMA-N40-S25-V1`

Completed: 2026-07-19 (Asia/Seoul)

Scope: reference-defined oracle-pocket redocking diagnostic. These are not
target-independent pocket-finding or prospective inference results. The exact
frozen protocol and source hashes are in the
[`compatibility benchmark protocol`](BENCHMARK_PROTOCOL.md).

All rates are percentages of the full frozen dataset. RMSD success uses
symmetry-aware heavy-atom RMSD <2A. `Vina+DG` is the selected top-1 pose;
`oracle-40` is the best RMSD among the same 40 samples and is not a deployable
selector.

| Dataset | N | First pose <2A | Vina+DG top-1 <2A | Improvement | Oracle-40 <2A | Vina median RMSD | Vina fast-valid | Final failures |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Astex Diverse | 85 | 35.29 | **64.71** | +29.41 pp | 94.12 | 1.432A | 90.59 | 0 |
| PoseBusters v2 | 308 | 31.82 | **64.29** | +32.47 pp | 91.56 | 1.446A | 74.68 | 0 |

Absolute Vina+DG <2A successes are 55/85 and 198/308. Oracle-40 successes are
80/85 and 282/308. All 393 complexes have final rows.

| Dataset | Vina <1A | Vina <3A | Vina mean RMSD |
|---|---:|---:|---:|
| Astex Diverse | 41.18 | 77.65 | 2.106A |
| PoseBusters v2 | 31.49 | 80.52 | 2.062A |

## Official PoseBusters validity

PoseBusters 0.6.5 `redock` was run on all 308 Vina+DG-selected PoseBusters
poses. The pass-all structural/chemical validity rate, excluding the RMSD
criterion, is **60.71% (187/308)**. The official RMSD criterion independently
reproduces 64.29%.

Important component pass rates are protein minimum-distance 76.30%,
tetrahedral chirality 73.05%, internal steric clash 97.73%, internal energy
98.70%, bond lengths 99.03%, and bond angles 99.03%. Protein volume-overlap
passes 100%. The self-contained fast-valid subset is reported separately and
must not be called the official PoseBusters pass rate.

## Pre-registered decision

The hypothesis passed all frozen thresholds:

- PoseBusters oracle-40 91.56% is above the pre-registered 85% threshold.
- Astex oracle-40 94.12% is above the pre-registered 90% threshold.
- PoseBusters Vina+DG improves over first-pose order by 32.47 percentage
  points, above the pre-registered 5-point threshold.
- Every dataset has 0% unresolved failures, below the 2% invalidation limit.

The retained EMA model therefore has strong sampling coverage even when the
trained confidence stack is ablated. This is an ablation/compatibility baseline,
not the final EFF-Dock selector result. The remaining Vina-to-oracle gap is the
pose-selection opportunity addressed by the retained confidence model below.

## Retained trained confidence model

The selected confidence asset is
`weights/effdock_confidence_extmatch_n80_s25_step42500.pt`, paired with
`weights/effdock_geometry_ft_100k_best.pt`. It was trained on generated PLINDER
pose sets matched to N80/S25/sigma0.5/pocket10 inference. A 500-step hard-pair
fine-tune did not improve the frozen validation metric, so step 42500 remains
the selected checkpoint.

Historical full single-run evidence, recovered from the original per-pose JSONL
and frozen selector artifacts, is:

| Dataset | N | Pure confidence <2A | Frozen confidence selector <2A | Oracle-80 <2A |
|---|---:|---:|---:|---:|
| Astex Diverse | 85 | 78.82 | **81.18** | 92.94 |
| PoseBusters v2 | 308 | 73.38 | **77.60** | 94.16 |

These historical values used the same model pair and N80 preset but predated
the active EFF-Dock frozen pocket manifests. They are context, not substituted
for the active two-dataset rerun defined in
`CONFIDENCE_BENCHMARK_PROTOCOL.md`.

The later cluster-free filter study applied its already-frozen rules to these
same historical single-run candidate artifacts:

| Dataset | N | Pure confidence <2A | Strict filter <2A | Atom-RMSD guard <2A | Historical composite <2A |
|---|---:|---:|---:|---:|---:|
| Astex Diverse | 85 | 78.82 (67) | 78.82 (67) | 78.82 (67) | **81.18 (69)** |
| PoseBusters v2 | 308 | 73.38 (226) | 73.38 (226) | 74.03 (228) | **77.60 (239)** |

The strict filter changed 3/85 Astex and 10/308 PoseBusters selections without
changing either <2A count. The atom guard changed 29/85 and 113/308, preserving
Astex and adding two net PoseBusters successes, but it had already failed the
PLINDER validation gate and is not promoted. These are RMSD-selection results,
not official PoseBusters pass-all validity; the stored fast-validity proxy was
100% for all selectors and is not substituted for an official SDF-based run.
Machine artifacts are under
`outputs/eff-dock/confidence-filter-v1-external/`.

### Active frozen-manifest N80 rerun

All selectors below consume the same 80 sampled poses per complex. The
composite is the pre-registered
`pair_gate_density_rank_vote_plclash_ambig` policy; oracle-80 is diagnostic.

| Dataset | N | First <2A | Vina+DG <2A | Pure confidence <2A | Frozen composite <2A | Composite vs Vina | Oracle-80 <2A |
|---|---:|---:|---:|---:|---:|---:|---:|
| Astex Diverse | 85 | 32.94 | 77.65 | 76.47 | **78.82** | +1.18 pp | 95.29 |
| PoseBusters v2 | 308 | 35.71 | 71.10 | **73.05** | 72.73 | +1.62 pp | 94.81 |

Absolute composite successes are 67/85 and 224/308. Pure confidence has 290
<2A successes across all 393 complexes, versus 291 for the composite and 285
for Vina+DG. Composite median RMSD is 1.074A and 1.293A;
failure rate is 0% after two recorded Astex numerical rescues.

Official PoseBusters 0.6.5 `redock` on all 308 composite-selected poses gives
**54.87% (169/308)** pass-all validity excluding RMSD. Component pass rates
include protein minimum-distance 69.16%, tetrahedral chirality 78.25%, internal
steric clash 95.78%, internal energy 98.70%, bond lengths 100%, and bond angles
97.73%. The official RMSD criterion reproduces 72.73%.

The pre-registered primary prediction passed: PoseBusters composite improved
over same-candidate Vina by 1.62 percentage points. The historical-reproduction
prediction failed: Astex was 2.35 points below 81.18%, and PoseBusters was 4.87
points below 77.60%, outside the ±2-point criterion. Therefore the trained
checkpoint remains active, while any selector recalibration must be a
new validation-only study rather than post-hoc tuning on these benchmarks.

Machine artifacts are `outputs/benchmarks/confidence/summary.json`, combined
rows under `outputs/benchmarks/confidence/combined/`, and ledger metrics under
`outputs/benchmarks/confidence/ledger_metrics.json`. All inference, recovery,
and official-validity stages completed successfully. The recorded active
code-tree SHA256 is
`5ffe2c0d3abec7748e59c134efd61997893727632304409d5559577eaf886cf9`.

## Slices and applicability

The machine summary records per-dataset heavy-atom, active fragment,
rotatable-bond, and receptor cofactor slices. On PoseBusters, Vina+DG <2A is
70.00% for ligands with at most 20 heavy atoms but 36.00% above 40 atoms; it is
68.14% for at most 3 rotatable bonds and 48.48% for 8 or more. This identifies
large/flexible ligands as the clearest compatibility weakness.

Train-neighbor ligand similarity and pocket similarity are deliberately not
reported yet: the preserved compatibility split predates the strict
two-benchmark exclusion contract. Those applicability-domain slices require
the new strict split and will belong to a publishable target-independent
evaluation, not this compatibility diagnostic.

## Execution and artifacts

- EMA checkpoint SHA256:
  `3ee604ec2338532532fa23a2ae91d0d540322defc32f5e453c8e7e12e389d36a`
- Config SHA256:
  `39aa62e4a48ed6f3aa4ff59345fb43a81220e2baba22edfd5beb0c4981b307ec`
- Final active code-tree SHA256:
  `9ec1ff3edfc61abc2b8127a3800293aaab4c2bba2005428a24b6846440846101`
- Runtime: PyTorch 2.10.0+cu130 / CUDA 13.0.
- Machine summary: `outputs/benchmarks/summary.json`.
- Per-complex combined rows: `outputs/benchmarks/combined/{astex,posebusters}.csv`.
- Selected pose SDFs, shard summaries, official PoseBusters rows, and Slurm
  logs remain under ignored `outputs/benchmarks/`.
