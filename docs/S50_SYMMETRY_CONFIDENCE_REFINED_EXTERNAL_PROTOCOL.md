# S50 symmetry-confidence on refined Astex/PoseBusters

Protocol ID: `EFFDOCK-S50-SYMMETRY-CONFIDENCE-REFINED-EXTERNAL-V1`

Status: frozen before new confidence scores are generated.

This is a repeated-use, descriptive external comparison. Astex and PoseBusters
outcomes have already been opened; this run cannot select or promote a
checkpoint. The internally selected U25k checkpoint remains the selected model
regardless of this result.

## Frozen inputs

- Cohort: exact 85 Astex Diverse and 308 PoseBusters v2 complexes in the
  completed sigma=2, eta=2 post-refinement bank
  `guidance_sdf_post_refinement_runs/sigma2-eta2-adaptive-20260819T062833Z`.
- Candidate inventory: 100 paired poses per complex at refinement step 0 and
  step 100; no pose regeneration.
- Confidence feature backbone: S50 EMA checkpoint SHA-256
  `65be44d7dc8f0867eb9fc5d22214b80f93971ea4702679a527c665046e91e6b6`.
- Confidence runtime: sigma=2, chunks of 20 poses, pure stable argmin predicted
  RMSD selector.
- Arms:
  - U1.5k recovery anchor: `2af26bf66bec53676b8344e811911bbf47ee85aa6550610f35c3812b7a7f9d15`;
  - U25k internal best: `1c59034172fb925cc8a70777dcba236be349f1a1de1775d49cc17d492b17c030`;
  - U50k latest: `fd49fa86f67187bf26d6c1bcf2daf925ba3e3b19dfeae733e57535d183280469`.

## Execution and metrics

Score one fixed Astex and one fixed PoseBusters complex for all three arms as a
smoke gate. After all smoke arms succeed, score 32 deterministic shards per
arm. Outputs are content-addressed, atomic, and non-overwriting.

Report step-0 and step-100 selected symmetry-aware RMSD below 2 Angstrom,
Top-5 recovery, oracle coverage, mean K2, and selected mean/median RMSD. Report
official PoseBusters validity and the validity-and-RMSD joint endpoint only at
step 100, where the official post-refinement validity ledger applies. Report
paired gain/loss counts between checkpoints at step 100. Candidate RMSD and
official validity are read only after all three score inventories are complete.

Any incomplete inventory, non-finite score, checkpoint/hash mismatch, pose
count mismatch, selector mismatch, or official-result mismatch invalidates the
run. No confidence checkpoint is retrained or selected from these results.
