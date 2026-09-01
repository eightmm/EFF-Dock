# S50 raw+refined confidence on frozen Astex/PoseBusters

Protocol ID: `EFFDOCK-S50-RAW-REFINED-CONFIDENCE-EXTERNAL-V1`

Status: frozen before U70k/U100k external scores are generated.

This is a repeated-use, descriptive external comparison. Astex Diverse and
PoseBusters v2 outcomes have already been opened, so these results cannot
select, tune, or promote either checkpoint. The 100k run's internal validation
rule already selected U70k as `best.pt`; U100k is the terminal `latest.pt`.

## Frozen inputs

- Cohort: exactly 85 Astex Diverse and 308 PoseBusters v2 complexes in the
  completed sigma=2, eta=2 post-refinement bank
  `guidance_sdf_post_refinement_runs/sigma2-eta2-adaptive-20260819T062833Z`.
- Candidate inventory: 100 paired poses per complex at refinement step 0
  (`raw`) and step 100 (`refined`); no docking or refinement is rerun.
- Confidence feature backbone: S50 EMA checkpoint SHA-256
  `65be44d7dc8f0867eb9fc5d22214b80f93971ea4702679a527c665046e91e6b6`.
- Confidence runtime: sigma=2, fixed chunks of 20 poses, pure stable argmin of
  predicted RMSD.
- U70k internal-best checkpoint SHA-256:
  `ce59be42f0ca613871ca079127c3296f5ca9a4ec72e44a9e5cf61878351c2638`.
- U100k terminal checkpoint SHA-256:
  `2ea1aca4f1c326cd0841e76c3597e3749231854a523d1ba8bd923c6fb5a9bff8`.

## Execution and metrics

Score the same fixed Astex and PoseBusters smoke complexes for both arms. Only
after all smoke tasks succeed, score 32 deterministic shards per arm. Outputs
are content-addressed, atomic, and non-overwriting.

Report raw and refined selected symmetry-aware RMSD below 2 Angstrom, Top-5
recovery, oracle coverage, mean K2, and selected mean/median RMSD. At refined
step 100, join the already-completed official PoseBusters validity ledger and
report validity plus the joint validity-and-RMSD endpoint. Report paired
U70k-to-U100k gain/loss counts on the identical candidates.

Any incomplete inventory, non-finite score, checkpoint/hash mismatch, pose
count mismatch, selector mismatch, or official-result mismatch invalidates the
run. External outcomes do not replace U70k's internal checkpoint selection.
