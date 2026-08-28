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
- Candidate generation: `N=100`, `S=10`, translation prior `sigma=2.0`, and
  normalized direct-drift GuidanceEnergy coupling with `eta=2.0`. The docking
  geometry checkpoint SHA-256 is
  `65be44d7dc8f0867eb9fc5d22214b80f93971ea4702679a527c665046e91e6b6`.
- Information boundary: the frozen pocket center is computed from the crystal
  complex as the centroid of receptor residue virtual nodes within 8 A of the
  reference ligand, with the ligand centroid as fallback. That derived
  three-vector, the prepared receptor, and ligand chemistry enter inference.
  Reference ligand atom coordinates do not otherwise enter the model or
  GuidanceEnergy and are used directly only after inference to calculate RMSD.
- Refinement: optimize each saved pose independently in rigid-fragment SE(3)
  coordinates with the in-repository unified GuidanceEnergy. No Vina or
  external force-field/minimization engine is called. The solver allows at most
  100 update iterations, caps one-step atom displacement at `0.10 A`, and
  uses at most 12 monotone line-search backtracks.
- Adaptive stopping: starting at step 25, define
  `delta_E_t = E_t - E_(t+1)` and
  `epsilon_t = 0.02 kcal/mol + 1e-3 * max(1 kcal/mol, abs(E_t))`. Five
  consecutive accepted updates with `delta_E_t <= epsilon_t` stop the pose;
  its finite terminal coordinates are carried forward to the materialized
  step-100 bank.
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

Report step-0 and step-100 selected symmetry-aware, no-alignment heavy-atom
RMSD below 2 Angstrom, Top-5 recovery, oracle coverage, mean K2, and selected
mean/median RMSD. The selected pose is the stable argmin of confidence-predicted
RMSD; ties are resolved by the original pose index. Top-5 and oracle are
candidate-coverage diagnostics, not deployable selectors.

Official validity means that all 27 non-RMSD binary checks from PoseBusters
0.6.5 `redock` pass; a missing or non-finite check is a failure. The joint
endpoint requires official validity and RMSD `<2 A` for the same selected pose.
The primary joint comparison uses the complete step-100 official-validity
ledger. A separately frozen U50 decomposition applies the identical 27-check
definition to raw, same-index refined, and refined-reselected poses; no raw
joint value is inferred for another arm. Every rate uses the full 85- or
308-complex denominator without dropping failed complexes.

Report paired gain/loss counts between checkpoints at step 100. Candidate RMSD
and official validity are read only after all three score inventories are
complete.

Any incomplete inventory, non-finite score, checkpoint/hash mismatch, pose
count mismatch, selector mismatch, or official-result mismatch invalidates the
run. No confidence checkpoint is retrained or selected from these results.
