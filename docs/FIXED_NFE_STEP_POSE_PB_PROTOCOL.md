# Fixed-NFE U50 Top-1 PoseBusters protocol

Protocol ID: `EFFDOCK-FIXED-NFE-STEP-POSE-PB-V1`

Status: frozen before any new selected-pose PoseBusters result is opened.

## Scope and claim boundary

This is a post-hoc, paired descriptive extension of
`EFFDOCK-FIXED-NFE-STEP-POSE-V1`. Astex Diverse and PoseBusters v2 outcomes
have already been opened. These results cannot select an ODE allocation,
guidance setting, refinement rule, or confidence checkpoint.

The comparison holds the completed U50k selector fixed and measures whether
its selected raw and refined poses satisfy PoseBusters structural checks. No
validity value, crystal RMSD, energy, or benchmark label may change a selected
pose index.

## Frozen inputs

- Cohorts: all 85 Astex Diverse and 308 PoseBusters v2 complexes.
- Arms:
  - `s10_n100`: 10 ODE steps and 100 poses;
  - `s25_n40`: 25 ODE steps and 40 poses.
- Stages: raw `step_000` and adaptive-refined `step_100`.
- Refinement artifact protocols are
  `EFFDOCK-GUIDANCE-SDF-POST-REFINEMENT-V1` for `s10_n100` and
  `EFFDOCK-FIXED-NFE-STEP-POSE-REFINEMENT-V1` for `s25_n40`; both implement
  the same frozen 100-step adaptive refinement endpoint used by the source
  comparison.
- Selector: stable argmin U50k-predicted RMSD independently at each stage.
- U50k checkpoint SHA-256:
  `fd49fa86f67187bf26d6c1bcf2daf925ba3e3b19dfeae733e57535d183280469`.
- Frozen fixed-budget report:
  `outputs/benchmarks/fixed_nfe_step_pose_runs/20260827T022010Z/report/report.json`,
  SHA-256
  `b66fb23436c4d5b89f3089232c8a40fcf9147010ae9ca67c5f29b6007f2d146f`.
- Its selected-pose ledger `complex_metrics.csv` has SHA-256
  `86ddf0da1f179d2afd702dbabdbb0de18d8ec68b76ea74a0f284c53aac508aaa`.
- PoseBusters runtime: exactly version `0.6.5`, configuration `redock`.

Every selected index, RMSD, score summary, refinement summary, protein,
reference ligand, and source pose SDF is hash-checked before evaluation. The
two arms must resolve to the same protein and ligand-reference identities for
each complex.

## Validity definitions

PoseBusters emits 27 non-RMSD redock checks plus one separate RMSD check.

- Primary validity: **PL-valid**, the 21 protein/ligand checks obtained by
  excluding the six organic-cofactor, inorganic-cofactor, and water minimum-
  distance/volume-overlap checks.
- Secondary validity: **official PB-valid**, pass-all over all 27 non-RMSD
  redock checks.
- Primary joint endpoint:
  `U50 Top-1 symmetry RMSD < 2 A AND PL-valid`.
- Secondary joint endpoint:
  `U50 Top-1 symmetry RMSD < 2 A AND official PB-valid`.

The symmetry-aware RMSD recorded by the frozen U50 scoring ledger is the RMSD
endpoint. The PoseBusters RMSD column is retained only as a separate schema and
runtime diagnostic.

## Execution and completion gates

1. A two-complex smoke evaluates Astex `1jje` and PoseBusters `7b2c_tp7`, four
   selected poses per complex.
2. The full run uses 32 deterministic CPU-only shards and evaluates exactly
   `393 complexes x 2 arms x 2 stages = 1,572` selected poses.
3. Slurm jobs use only the `cpu_only` partition and do not request an explicit
   CPU count. PoseBusters workers follow `SLURM_CPUS_PER_TASK`, defaulting to
   one.
4. Any missing/duplicate complex, selected-index mismatch, hash mismatch,
   pose-count mismatch, parse failure, non-finite RMSD, changed PoseBusters
   schema/version, or incomplete shard prevents aggregation.
5. Report RMSD success, PL-validity, official PB-validity, both joint metrics,
   per-check pass rates, and paired gain/loss counts. Report the result in
   every direction; there is no efficacy stopping threshold.
