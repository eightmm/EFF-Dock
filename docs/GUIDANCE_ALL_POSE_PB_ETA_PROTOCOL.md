# All-pose official PoseBusters eta characterization

- Protocol: `EFFDOCK-GUIDANCE-ALL-POSE-PB-ETA-V1`
- Status: paired descriptive characterization.
- Purpose: measure official PoseBusters validity over every generated pose,
  without confidence selection, as a function of direct-guidance strength.
- Claim boundary: Astex Diverse and PoseBusters v2 are opened external sets;
  this study describes a response curve and cannot select or admit eta.

## Frozen inputs

- Complete Astex Diverse (`85`) and PoseBusters v2 (`308`) cohorts.
- `100` saved poses per complex and eta.
- `eta = {0, 0.5, 1.0, 1.5, 2.0}` from
  `guidance_steric_high_eta_confidence_runs/20260807T045916Z`.
- `eta = {2.5, 3.0}` from
  `guidance_eta_cap_extension_runs/20260810T034923Z`.
- Sampling is not repeated. Every all-pose SDF and its original sampling-row
  SHA-256 are verified before official evaluation.
- PoseBusters runtime version: `0.6.5`, `redock` configuration.
- PoseBusters validity is the conjunction of all 27 non-RMSD redock checks;
  the separate RMSD check is recorded but excluded from validity.

The inventory is exactly `393 × 7 = 2,751` complex/eta cells and
`275,100` poses. Pose index is paired across eta through the frozen shared
prior and sampling seed contract.

## Reported quantities

- Pooled all-pose PB-valid percentage for each dataset and eta.
- Per-complex macro mean validity. With exactly 100 poses per complex this
  must equal the pooled percentage, but both are audited explicitly.
- Per-check pass percentage for all 27 validity checks.
- Paired transition counts versus eta 0: invalid→valid, valid→invalid,
  valid→valid, and invalid→invalid for the same complex and pose index.
- No confidence, confidence filter, joint RMSD/validity, or automatic eta
  selection is used.

## Execution

The chain is manifest -> smoke official PB -> smoke audit -> full official PB
-> full audit -> report. Full work is sharded over CPU tasks and every stage is
connected with `afterok`. Failed attempts are retained and completed shards
are never overwritten.
