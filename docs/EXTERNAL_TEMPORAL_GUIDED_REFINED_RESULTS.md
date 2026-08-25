# Recent external guided/refined benchmark results (U50 reporting)

Base protocol: `EFFDOCK-EXTERNAL-TEMPORAL-GUIDED-REFINED-V1`

Reporting addendum: `EFFDOCK-EXTERNAL-TEMPORAL-U50-REPORT-V1`

Completed: 2026-08-26

## Evaluation contract

- Candidates: `N100/S10`, translation prior `sigma=2`, normalized direct-drift
  guidance `eta=2`, then adaptive physical refinement up to 100 steps.
- Selector: stable U50k confidence argmin of predicted symmetry-aware RMSD,
  checkpoint SHA-256
  `fd49fa86f67187bf26d6c1bcf2daf925ba3e3b19dfeae733e57535d183280469`.
- RMSD success: symmetry-aware, no-alignment heavy-atom RMSD `<2 A`.
- PB-valid: all 27 non-RMSD PoseBusters 0.6.5 `redock` checks pass.
- Joint: PB-valid and RMSD success for the same confidence-selected pose.
- Every rate uses the complete frozen cohort denominator.

The exact cohort construction and frozen method are documented in
[`EXTERNAL_TEMPORAL_BENCHMARKS.md`](EXTERNAL_TEMPORAL_BENCHMARKS.md) and
[`EXTERNAL_TEMPORAL_GUIDED_REFINED_PROTOCOL.md`](EXTERNAL_TEMPORAL_GUIDED_REFINED_PROTOCOL.md).
The score-only selector override and its post-hoc claim boundary are in
[`EXTERNAL_TEMPORAL_U50_REPORT_PROTOCOL.md`](EXTERNAL_TEMPORAL_U50_REPORT_PROTOCOL.md).

## Results

| Dataset | N | Raw Top-1 <2A | Refined Top-1 <2A | Raw oracle <2A | Refined oracle <2A | Refined PB-valid | Refined joint valid+<2A |
|---|---:|---:|---:|---:|---:|---:|---:|
| PhiBench derived | 203 | 125 (61.58%) | 132 (65.02%) | 177 (87.19%) | 179 (88.18%) | 186 (91.63%) | 122 (60.10%) |
| FoldBench P-L adaptation | 66 | 43 (65.15%) | 41 (62.12%) | 57 (86.36%) | 58 (87.88%) | 61 (92.42%) | 40 (60.61%) |
| OpenBind clean non-covalent | 860 | 414 (48.14%) | 445 (51.74%) | 749 (87.09%) | 773 (89.88%) | 847 (98.49%) | 438 (50.93%) |

Mean selected RMSD changed from 2.250 to 2.212 A on PhiBench, 2.646 to
2.743 A on FoldBench, and 2.126 to 2.044 A on the clean OpenBind cohort. The
median refined selected RMSD values were 1.542, 1.803, and 1.939 A,
respectively.

Within the U50 ledger, refinement changed Top-1 by +3.45 percentage points on
PhiBench, -3.03 points on FoldBench, and +3.60 points on OpenBind. Relative to
the historical U25 report, U50 changed refined Top-1 by +4.93, -3.03, and
+1.51 points, respectively. The mixed FoldBench direction shows why physical
refinement and confidence reselection are reported separately. The oracle gaps
remain large on all three sets and quantify candidate coverage, not deployable
selection performance.

The paired U25-to-U50 transitions expose both gains and regressions:

| Dataset | Refined RMSD gain/loss | PB-valid gain/loss | Joint gain/loss |
|---|---:|---:|---:|
| PhiBench derived | 17/7 | 6/5 | 17/8 |
| FoldBench P-L adaptation | 3/5 | 1/1 | 4/5 |
| OpenBind clean non-covalent | 44/31 | 1/7 | 42/32 |

`gain/loss` counts complexes that changed from fail to pass or pass to fail,
respectively; all comparisons use the identical saved candidate coordinates.

## Claim boundaries

- PhiBench is the explicit 203-system EFF-Dock-derived cohort, not the hidden
  206-system PhysDock paper curation.
- FoldBench uses crystal-pocket redocking and is not a native FoldBench
  co-folding/structure-prediction leaderboard result.
- The clean OpenBind `N=860` characterization is not the filtered
  scaffold-only `N=802` official-style Top-25 comparison. That separate result
  is reported in
  [`OPENBIND_OFFICIAL_TOP25_RESULTS.md`](OPENBIND_OFFICIAL_TOP25_RESULTS.md).
- These datasets are descriptive repeated external evaluations and cannot be
  used to select or tune a checkpoint or guidance coefficient.

## Artifact verification

Generated poses and pose-level evaluator output remain ignored by Git. The
completed report contains exactly 1,129 rows with dataset multiplicity and was
checked for full cohort coverage, unique IDs, complete shard coverage, exact
checkpoint/input identities, finite selected poses, and successful completion
of every smoke, sampling, refinement, confidence, validity, and aggregation
stage.

| Artifact | SHA-256 |
|---|---|
| U50 confidence checkpoint | `fd49fa86f67187bf26d6c1bcf2daf925ba3e3b19dfeae733e57535d183280469` |
| Aggregate summary | `3a69d9642268d828088305ac8bf334f6df91ed7f6deb7b38a6c8edb243cfba39` |
| Per-complex rows | `28eeaad3bea8c37eaaae3a1aae43ce7f2c8caeb61d84306899a4d76485db7c5a` |
| PhiBench result IDs | `c8a70e5cfecf4fa1464b18c0a48ebf13c309b18cbd948aaea364c856e988e2b9` |
| FoldBench result IDs | `117ad1395d261b0956d17f569a024237a90221f733d02a3a63b8608ae87525c8` |
| OpenBind result IDs | `acadd32c6267e199981d13e2b436982570fc25eca77f2c033ed19323f0387957` |

These values are local descriptive diagnostics, not official leaderboard
submissions.
