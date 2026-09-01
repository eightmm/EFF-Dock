# U70k/U100k temporal external confidence protocol

Protocol ID: `EFFDOCK-S50-RAW-REFINED-CONFIDENCE-TEMPORAL-EXTERNAL-V1`

Status: frozen before U70k/U100k temporal-external scores are generated.

## Purpose

Check whether the raw+refined confidence training result transfers beyond the
previously inspected Astex and PoseBusters cohorts. The internally selected
U70k checkpoint and terminal U100k checkpoint are compared on identical saved
candidate and refinement banks. No docking pose or refinement trajectory is
regenerated.

## Frozen inputs

- Source run:
  `outputs/benchmarks/external_temporal_guided_refined_runs/20260825T000806Z`
- Cohorts: PhiBench `N=203`, FoldBench `N=66`, OpenBind clean pocket-redocking
  cohort `N=860`.
- Candidate bank: `N=100`, `S=10`, `sigma=2`, normalized-drift physical
  guidance with `eta=2`.
- Refinement: saved adaptive trajectory, at most 100 steps.
- Docking checkpoint SHA-256:
  `65be44d7dc8f0867eb9fc5d22214b80f93971ea4702679a527c665046e91e6b6`
- U70k checkpoint SHA-256:
  `ce59be42f0ca613871ca079127c3296f5ca9a4ec72e44a9e5cf61878351c2638`
- U100k checkpoint SHA-256:
  `2ea1aca4f1c326cd0841e76c3597e3749231854a523d1ba8bd923c6fb5a9bff8`

Every source shard, refinement summary, structure input, and checkpoint is
hash-checked before use.

## Selection and endpoints

Each confidence checkpoint independently selects the stable minimum predicted
RMSD among 100 poses at pre-refinement `step_000` and post-refinement
`step_100`. Reference RMSD and validity do not enter selection.

For each cohort and arm, report symmetry-aware Top-1 RMSD `<2 A`, 100-pose
oracle `<2 A`, PL-validity, official PoseBusters validity, and their conjunction
with refined Top-1 `<2 A`. Report paired U100k-minus-U70k gains and losses on
the identical complexes.

## Interpretation boundary

PhiBench and FoldBench are the core temporal checks. These are EFF-Dock
pocket-redocking adaptations; FoldBench is not evaluated under its native
leaderboard contract. OpenBind is an auxiliary stress check because it is a
dense enterovirus 2A-protease series rather than a diverse target benchmark.
All three cohorts have already been used in prior project analysis, so these
results are descriptive and cannot replace the internal U70k checkpoint
selection.
