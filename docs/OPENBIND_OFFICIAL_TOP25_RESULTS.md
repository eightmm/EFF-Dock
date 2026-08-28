# EFF-Dock OpenBind official-style Top-N results (U50 reporting)

Base protocol: `EFFDOCK-OPENBIND-OFFICIAL-TOP25-V1`

Reporting addendum: `EFFDOCK-OPENBIND-OFFICIAL-TOP25-U50-REPORT-V1`

Completed: 2026-08-26

## Evaluation contract

- Cohort: OpenBind `filtered=True, scaffold_only=True`, `N=802`.
- Predictions: 786 complexes; all 16 missing predictions remain in the
  denominator and count as failures.
- Inference: `N100/S10`, translation prior `sigma=2`, normalized-drift guidance
  `eta=2`, followed by adaptive physical refinement with at most 100 steps.
- Selection: stable ascending U50k confidence-predicted RMSD after refinement,
  checkpoint SHA-256
  `fd49fa86f67187bf26d6c1bcf2daf925ba3e3b19dfeae733e57535d183280469`.
- Valid pose: all 27 non-RMSD checks from PoseBusters 0.6.5 `redock` pass.
- RMSD endpoint: at least one valid pose has OpenStructure 2.11.1 BiSyRMSD
  `<=2 A`.
- Strict endpoint: the same pose also has LDDT-PLI `>=0.8`.

The complete frozen metric contract and score-only reporting change are in
[`OPENBIND_OFFICIAL_TOP25_PROTOCOL.md`](OPENBIND_OFFICIAL_TOP25_PROTOCOL.md)
and
[`OPENBIND_OFFICIAL_TOP25_U50_PROTOCOL.md`](OPENBIND_OFFICIAL_TOP25_U50_PROTOCOL.md).

## EFF-Dock results

| Rank budget | Any PB-valid | PB-valid + BiSyRMSD <=2 A | + LDDT-PLI >=0.8 |
|---|---:|---:|---:|
| Top-1 | 773/802 (96.38%) | 417/802 (52.00%) | 359/802 (44.76%) |
| Top-5 | 786/802 (98.00%) | 603/802 (75.19%) | 511/802 (63.72%) |
| Top-25 | 786/802 (98.00%) | **695/802 (86.66%)** | **581/802 (72.44%)** |

The RDKit symmetry-RMSD diagnostic and OpenStructure BiSyRMSD produced
identical `<=2 A` complex classifications after the PoseBusters filter at
Top-1, Top-5, and Top-25: zero complex-level mismatches.

The gap from Top-1 to Top-25 is 34.66 percentage points for PB-valid RMSD and
27.68 points for the strict endpoint. The frozen sampler therefore has much
more native-pose coverage than the current confidence selector recovers.

## Public OpenBind Top-25 context

The comparison values below are copied from
`plotting/tables/allmethods_plot_data_scaffolds_filtered_top25.csv` in the
public OpenBind repository at commit
`8849566aeb6b22c39589918d8ac00c24c0983aba`.

| Method | Setting | Predictions | Mean poses / denominator | PB-valid + BiSyRMSD <=2 A | + LDDT-PLI >=0.8 |
|---|---|---:|---:|---:|---:|
| GNINA multi | redocking | 802/802 | 25.00 | 739/802 (92.14%) | 684/802 (85.29%) |
| **EFF-Dock** | pocket redocking | 786/802 | 24.50 | **695/802 (86.66%)** | **581/802 (72.44%)** |
| Best co-folding | union of reported co-folding methods | 802/802 | 149.50 | 687/802 (85.66%) | 579/802 (72.19%) |
| Protenix | co-folding | 802/802 | 25.00 | 548/802 (68.33%) | 437/802 (54.49%) |
| OpenFold3-p2 | co-folding | 802/802 | 25.00 | 410/802 (51.12%) | 285/802 (35.54%) |
| AlphaFold3 | co-folding | 802/802 | 25.00 | 358/802 (44.64%) | 207/802 (25.81%) |
| GNINA multi | fragment cross-docking | 802/802 | 25.00 | 303/802 (37.78%) | 109/802 (13.59%) |
| RosettaFold 3 | co-folding | 798/802 | 24.88 | 143/802 (17.83%) | 27/802 (3.37%) |
| Boltz-1 | co-folding | 796/802 | 24.81 | 114/802 (14.21%) | 60/802 (7.48%) |
| Boltz-2 | co-folding | 796/802 | 24.81 | 81/802 (10.10%) | 64/802 (7.98%) |

`Best co-folding` is a per-complex union over several methods and is not one
deployable model. GNINA redocking is the closest public task setting to
EFF-Dock. Co-folding and fragment cross-docking rows provide context but are
not task-identical baselines.

Relative to GNINA redocking, EFF-Dock is lower by 5.48 points on PB-valid RMSD
and 12.85 points on the strict endpoint. The larger strict-endpoint gap points
to protein-ligand interaction fidelity, in addition to confidence selection,
as a remaining limitation.

Public source:
[OpenBind Top-25 table](https://github.com/OpenBind-Consortium/EV-A71_2A_benchmark/blob/8849566aeb6b22c39589918d8ac00c24c0983aba/plotting/tables/allmethods_plot_data_scaffolds_filtered_top25.csv).

## Artifact verification

Generated poses and raw evaluator rows remain ignored by Git. The completed
local report was independently checked for:

- 802 unique complexes at each of Top-1, Top-5, and Top-25;
- 19,650 unique confidence-ranked PoseBusters rows (`786 * 25`);
- 6,311 OpenStructure-evaluated pose rows;
- 64/64 complete PoseBusters shards and 64/64 complete OpenStructure shards;
- exact agreement between raw-row recounts and the machine summary;
- successful completion of every smoke, full-evaluation, and report stage.

Content identities:

| Artifact | SHA-256 |
|---|---|
| Official cohort IDs | `a5ba75493d58fe5744a8c96552e7aa5cd339d7fd867b8189b687533a530418b2` |
| OpenBind metadata | `389a7edca3ac8034d6533da5a3f3235619e7206aef7284441fd52d350bb1c652` |
| U50 confidence checkpoint | `fd49fa86f67187bf26d6c1bcf2daf925ba3e3b19dfeae733e57535d183280469` |
| Source inference summary | `3a69d9642268d828088305ac8bf334f6df91ed7f6deb7b38a6c8edb243cfba39` |
| Official-style summary | `6a56a5da6d9fc771482af19273f559917b8b2b03da94f1a70f9d3f255ce2b05d` |
| Per-complex Top-N rows | `3879d63ebcdfe3f6a3e09906f2317936f65fcd40524df548f1aa8ccd58a94aac` |
| PoseBusters pose rows | `32db949d2772cc5b8ad7fe09d5583275ec55d266782318be28f65f73fde08c9e` |
| OpenStructure score rows | `a9380cdffbbbf70c69adb8b08ed7c56b6fb75c373e41b158e738a21f3a5966f8` |

These values are an official-style local diagnostic, not an OpenBind
leaderboard submission.
