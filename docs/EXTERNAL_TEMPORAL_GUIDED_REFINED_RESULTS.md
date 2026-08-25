# Recent external guided/refined benchmark results

Protocol: `EFFDOCK-EXTERNAL-TEMPORAL-GUIDED-REFINED-V1`

Completed: 2026-08-25

## Evaluation contract

- Candidates: `N100/S10`, translation prior `sigma=2`, normalized direct-drift
  guidance `eta=2`, then adaptive physical refinement up to 100 steps.
- Selector: stable U25k confidence argmin of predicted symmetry-aware RMSD.
- RMSD success: symmetry-aware, no-alignment heavy-atom RMSD `<2 A`.
- PB-valid: all 27 non-RMSD PoseBusters 0.6.5 `redock` checks pass.
- Joint: PB-valid and RMSD success for the same confidence-selected pose.
- Every rate uses the complete frozen cohort denominator.

The exact cohort construction and frozen method are documented in
[`EXTERNAL_TEMPORAL_BENCHMARKS.md`](EXTERNAL_TEMPORAL_BENCHMARKS.md) and
[`EXTERNAL_TEMPORAL_GUIDED_REFINED_PROTOCOL.md`](EXTERNAL_TEMPORAL_GUIDED_REFINED_PROTOCOL.md).

## Results

| Dataset | N | Raw Top-1 <2A | Refined Top-1 <2A | Raw oracle <2A | Refined oracle <2A | Refined PB-valid | Refined joint valid+<2A |
|---|---:|---:|---:|---:|---:|---:|---:|
| PhiBench derived | 203 | 123 (60.59%) | 122 (60.10%) | 177 (87.19%) | 179 (88.18%) | 185 (91.13%) | 113 (55.67%) |
| FoldBench P-L adaptation | 66 | 39 (59.09%) | 43 (65.15%) | 57 (86.36%) | 58 (87.88%) | 61 (92.42%) | 41 (62.12%) |
| OpenBind clean non-covalent | 860 | 384 (44.65%) | 432 (50.23%) | 749 (87.09%) | 773 (89.88%) | 853 (99.19%) | 428 (49.77%) |

Mean selected RMSD changed from 2.356 to 2.395 A on PhiBench, 2.764 to
2.706 A on FoldBench, and 2.200 to 2.093 A on the clean OpenBind cohort. The
median refined selected RMSD values were 1.595, 1.662, and 1.992 A,
respectively.

Refinement improved Top-1 by -0.49 percentage points on PhiBench, +6.06 points
on FoldBench, and +5.58 points on OpenBind. The PhiBench result shows that the
physical corrector is not uniformly pose-accuracy improving; the frozen
confidence selector and refinement should therefore be reported separately.
The oracle gaps remain large on all three sets and quantify candidate coverage,
not deployable selection performance.

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
| Aggregate summary | `2206a4ab86e834001a8a9ec93661db042174183f6755aca88b818294ca7dfbd5` |
| Per-complex rows | `ae457058165cc4f96fd672cde49756b543602df1ac326ed07951161cfee2bc2a` |
| PhiBench result IDs | `c8a70e5cfecf4fa1464b18c0a48ebf13c309b18cbd948aaea364c856e988e2b9` |
| FoldBench result IDs | `117ad1395d261b0956d17f569a024237a90221f733d02a3a63b8608ae87525c8` |
| OpenBind result IDs | `acadd32c6267e199981d13e2b436982570fc25eca77f2c033ed19323f0387957` |

These values are local descriptive diagnostics, not official leaderboard
submissions.
