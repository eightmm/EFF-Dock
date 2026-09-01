# EFF-Dock OpenBind official-style Top-N results

Protocol: `EFFDOCK-OPENBIND-OFFICIAL-TOP25-V1`

Completed: 2026-08-25

## Evaluation contract

- Cohort: OpenBind `filtered=True, scaffold_only=True`, `N=802`.
- Predictions: 786 complexes; the 16 missing predictions remain in the
  denominator and count as failures.
- Candidate generation: `N100/S10`, fragment-prior `sigma=2`, normalized-drift
  guidance `eta=2`, followed by the frozen adaptive physical refinement with a
  maximum of 100 steps.
- Ranking: ascending post-refinement confidence-predicted RMSD, with original
  pose index as the stable tie-breaker, using the internally selected U25k
  symmetry-confidence checkpoint.
- Valid pose: all 27 non-RMSD checks from PoseBusters 0.6.5 `redock` pass.
- RMSD endpoint: at least one valid pose has OpenStructure 2.11.1 BiSyRMSD
  `<=2 A`.
- Strict endpoint: the same pose also has LDDT-PLI `>=0.8`.
- The full frozen contract is in `OPENBIND_OFFICIAL_TOP25_PROTOCOL.md`.

## EFF-Dock results

| Rank budget | Any PB-valid | PB-valid + BiSyRMSD <=2 A | + LDDT-PLI >=0.8 |
| --- | ---: | ---: | ---: |
| Top-1 | 779/802 (97.13%) | 406/802 (50.62%) | 338/802 (42.14%) |
| Top-5 | 786/802 (98.00%) | 592/802 (73.82%) | 500/802 (62.34%) |
| Top-25 | 786/802 (98.00%) | **694/802 (86.53%)** | **581/802 (72.44%)** |

The repository's RDKit symmetry-aware RMSD diagnostic and OpenStructure
BiSyRMSD produced identical `<=2 A` complex classifications after the
PoseBusters filter at Top-1, Top-5, and Top-25 (zero mismatched complexes).

## Public OpenBind Top-25 context

The comparison values below are copied from the public OpenBind figure table
`plotting/tables/allmethods_plot_data_scaffolds_filtered_top25.csv` at commit
`8849566aeb6b22c39589918d8ac00c24c0983aba`.

| Method | Setting | Predictions | Mean poses / denominator | PB-valid + BiSyRMSD <=2 A | + LDDT-PLI >=0.8 |
| --- | --- | ---: | ---: | ---: | ---: |
| GNINA multi | redocking | 802/802 | 25.00 | 739/802 (92.14%) | 684/802 (85.29%) |
| **EFF-Dock** | pocket redocking | 786/802 | 24.50 | **694/802 (86.53%)** | **581/802 (72.44%)** |
| Best co-folding | union of reported co-folding methods | 802/802 | 149.50 | 687/802 (85.66%) | 579/802 (72.19%) |
| Protenix | co-folding | 802/802 | 25.00 | 548/802 (68.33%) | 437/802 (54.49%) |
| OpenFold3-p2 | co-folding | 802/802 | 25.00 | 410/802 (51.12%) | 285/802 (35.54%) |
| AlphaFold3 | co-folding | 802/802 | 25.00 | 358/802 (44.64%) | 207/802 (25.81%) |
| GNINA multi | fragment cross-docking | 802/802 | 25.00 | 303/802 (37.78%) | 109/802 (13.59%) |
| RosettaFold 3 | co-folding | 798/802 | 24.88 | 143/802 (17.83%) | 27/802 (3.37%) |
| Boltz-1 | co-folding | 796/802 | 24.81 | 114/802 (14.21%) | 60/802 (7.48%) |
| Boltz-2 | co-folding | 796/802 | 24.81 | 81/802 (10.10%) | 64/802 (7.98%) |

`Best co-folding` is a per-complex union over multiple methods and is not one
deployable model. The public comparison is an any-pose Top-25 endpoint, not a
Top-1 selector benchmark. EFF-Dock Top-1 above is therefore useful as its own
deployable confidence-selection result, while EFF-Dock Top-25 is the directly
aligned row for this table.

Redocking, fragment cross-docking, and co-folding have different input
contracts. GNINA redocking is the closest public task setting to EFF-Dock;
co-folding rows are context rather than task-identical baselines. These values
are an official-style local diagnostic, not an OpenBind leaderboard submission.

Public source:
<https://github.com/OpenBind-Consortium/EV-A71_2A_benchmark>

## Artifacts and verification

- Source pose run:
  `outputs/benchmarks/external_temporal_guided_refined_runs/20260825T000806Z`
- Complete aggregation:
  `outputs/benchmarks/openbind_official_top25_runs/20260825T075023Z/report`
- Machine summary: `report/summary.json`
- Per-complex Top-1/5/25 outcomes: `report/complex_results.csv`
- All 19,650 confidence-ranked PoseBusters rows:
  `report/posebusters_poses.csv`
- All 6,569 OpenStructure-evaluated pose rows:
  `report/openstructure_scores_evaluated.csv`
- Official cohort ID SHA-256:
  `a5ba75493d58fe5744a8c96552e7aa5cd339d7fd867b8189b687533a530418b2`
- Metadata SHA-256:
  `389a7edca3ac8034d6533da5a3f3235619e7206aef7284441fd52d350bb1c652`
- Slurm: smoke PoseBusters `59006`, smoke OpenStructure `59007`, full
  PoseBusters array `59008`, full OpenStructure array `59009`, report `59010`.
  All completed with exit code `0:0`.

Independent post-report checks verified 2,406 unique complex/rank-budget rows,
19,650 unique complex/pose-rank PoseBusters rows, all 64 PoseBusters shards,
all 64 OpenStructure shards, and exact agreement between the raw-row counts
and `summary.json`.
