# EFF-Dock PLINDER guidance-development protocol

Protocol ID: `EFFDOCK-PLINDER-GUIDANCE-DEV-V1`

Status: frozen before raw-structure download and ODE sampling. This is an
internal guidance-development study. It may compare and later freeze a guidance
scale, but it is not an untouched confirmation result and must not be reported
as a new external benchmark.

## Cohort and claim boundary

- Source release: PLINDER `2024-06/v2`.
- Cohort: all 1,076 sample keys in the preserved EFF-Dock compatibility
  validation split, `data/splits/plinder.json` key `val`.
- Frozen split SHA-256:
  `3ac570bf08bced053f1ce040b57efca27c3be616f29a82cd66ef887c08860e6b`.
- Local processed coverage must be exactly `1,076/1,076`. The cohort contains
  1,058 distinct PLINDER systems.
- The split is sample-key disjoint from the 47,310-row training split, and the
  registered OMS ID-manifest check and leakage check must pass before launch.
- This historical validation set was already used to select the retained
  confidence checkpoint and to study confidence filters. It is therefore a
  guidance-development set, not an untouched confirmation set. The docking and
  confidence checkpoints remain frozen throughout this study.
- A later confirmation claim requires a new, untouched temporal cohort that is
  disjoint from model training and confidence selection on sample ID,
  canonical ligand chemistry, pocket group, and PDB entry.

## Raw inputs

The processed protein/ligand tensors are already present. Full unified-guidance
typing and official PoseBusters redock checks additionally require the original
PLINDER `receptor.pdb` and `ligand_files/<instance>.<chain>.sdf` assets.

- Download only the 475 `systems/<two_char_code>.zip` archives needed by the
  frozen validation IDs; do not download unrelated PLINDER collections.
- The public-object inventory measured before download is 71,372,079,105 bytes
  (66.4704 GiB compressed). Record object size, generation, MD5/CRC metadata,
  PLINDER package version, and the exact requested IDs in an ignored local
  manifest.
- PLINDER groups system structures into zipped `two_char_code` shards and each
  extracted system contains receptor and ligand structure files. Source
  reference: <https://plinder-org.github.io/plinder/dataset.html>.
- Downloaded raw data and generated results remain ignored and must not be
  committed.

## Prediction setting

This is reference-pocket redocking, not blind docking. The frozen pocket center
stored in each processed `meta.pt` was derived during PLINDER preprocessing from
the bound complex and is supplied explicitly to inference. Crystal ligand
coordinates may be used only as the declared input conformer/internal-geometry
reference and post-hoc RMSD target; they must never be read by the ODE drift,
guidance schedule, confidence selector, or any adaptive decision during a run.

For every complex and arm:

- Docking checkpoint:
  `weights/effdock_geometry_ft_100k_best.pt`, SHA-256
  `6932fb3ba6ebac770f714453529656a44b8f33cf15119d23c9e675d2d60b36db`.
- Confidence checkpoint:
  `weights/effdock_confidence_extmatch_n80_s25_step42500.pt`, SHA-256
  `e31fde6f351284205c78f7a1510002779c43312e94d9f82003d47a14d72bc78f`.
- Candidate budget: `N100/S10`; scalar prior `sigma=0.5`; prior pool 100.
- Base seed 42 with sorted sample-key offset. All eta arms for one complex must
  have the same `sampling_seed` and `prior_pool_sha256`.
- Pocket cutoff 10 Angstrom; center jitter 0; late schedule, power 3.
- Unified-guidance coupling: `normalized_drift`, start `t=0.5`, ramp power 1.
- Eta arms: `{0.0, 0.5, 1.0, 1.5, 2.0}`. Eta zero is regenerated in the same
  run and is the only paired baseline.
- Receptor policy `geometry_only`; protein shell 18 Angstrom; no refinement;
  Vina guidance disabled.
- Fixed caps: atom force 20, translation velocity 5, angular velocity 5,
  estimated atom displacement 0.25 Angstrom, maximum backtracks 8.
- Primary Top-1 selector: minimum predicted RMSD from the retained confidence
  model. No interaction energy, cluster size, density, Vina score, or
  PoseBusters outcome enters selection.

## Evaluation

PoseBusters `0.6.5`, configuration `redock`, is post-hoc evaluation only. For
each eta report over the exact 1,076-complex denominator:

- confidence-selected symmetry-aware RMSD `<2 Angstrom`;
- confidence-selected median RMSD;
- oracle-of-100 RMSD `<2 Angstrom`;
- confidence-selected pass-all over the 27 non-RMSD redock checks;
- joint confidence-selected RMSD `<2 Angstrom` and pass-all;
- per-check PoseBusters failure rates;
- paired deltas and paired complex-ID bootstrap intervals versus eta zero;
- non-finite counters, cap rates, applied/model ratios, CUDA peaks, and all
  missing/failure reasons.

Validity alone is not a success metric because moving a ligand away from the
correct pocket geometry can reduce clashes. The primary outcome is the joint
RMSD-and-validity rate. The pre-declared development target is at least `+2.0`
percentage points in joint success versus eta zero while confidence-selected
RMSD `<2 Angstrom` decreases by no more than `2.0` percentage points. Report
all arms without automatically selecting a winner; the user makes the final
development decision.

## Fail-closed execution order

1. Verify split/data hashes, exact 1,076 processed coverage, and train/val ID
   leakage checks.
2. Download and verify only the required raw PLINDER system shards.
3. Run a fixed-ID all-arm CUDA smoke and official PoseBusters smoke.
4. Run the full 1,076 x 5 paired ODE/guidance sampling array.
5. Audit exact inventory, prior pairing, numerical gates, parameter identity,
   source/runtime manifest, and saved selected-pose hashes.
6. Run official PoseBusters on the confidence-selected pose only.
7. Produce one strict aggregate. Missing or failed complexes never shrink the
   denominator; any incomplete stage blocks its dependents.

Scheduler-availability correction (2026-08-05): CUDA smoke and full sampling
may be scheduled on the ordered Slurm partition list `6000ada,heavy`, always
with exactly one visible GPU and at least 48,000 MiB visible memory. A task
placed on `6000ada` still fails closed unless the device is an RTX 6000 Ada.
The `heavy` partition is restricted by cluster inventory to H100 and RTX PRO
6000-class devices; the sampling stage records the actual partition, GPU name,
and total visible memory, and the audit retains the observed GPU-name and
memory sets. This changes scheduler eligibility only. Cohort, eta arms,
`N100/S10`, seeds, priors, checkpoints, guidance equations and caps, selector,
and outcome gates are unchanged.

Every replacement run executes from a per-run, read-only code capsule under
`.effdock_execution_capsules/`. The execution manifest hashes capsule files,
not the mutable live worktree, and the capsule source path is placed ahead of
the editable environment on `PYTHONPATH`. Large data, weights, outputs, and the
Python environment remain linked to their declared repository locations;
every selected frozen input retains its exact hash gate. Thus later source
edits cannot alter a queued run, while any mutation of a declared linked input
still fails closed.
