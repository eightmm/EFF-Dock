# S50 Raw + Refined Confidence 100k Protocol

## Status

User-authorized one-stage training protocol, registered 2026-08-28 before the
100k run starts. This is a new content identity and does not rewrite the
previously registered 10k continuation.

## Objective and claim boundary

The hypothesis is that the 10k continuation is exposure-limited: with 43,092
training complexes and an effective global batch of two complexes, 10k updates
cover less than half of one complex-level epoch. A single 100k schedule gives
about 4.64 complex-level epochs and permits repeated stratified draws from the
sealed raw/refined pose banks.

This is repeated-use PLINDER adaptation. The 1,035-complex validation split may
select `best.pt`; it is not independent external confirmation. External Astex
and PoseBusters results cannot tune this training run.

## Frozen inputs

- Initialization: immutable terminal U50,000 symmetry-confidence `latest.pt`.
- Training inventory: the existing filtered 43,092-complex train split.
- Validation inventory: the existing 1,035-complex validation split.
- Per training complex: 32 raw sigma-2 poses, 32 deterministic step-100 refined
  poses, and one exact mapped-crystal anchor.
- Validation: all 100 refined poses; no crystal anchor.
- Pose labels and the primary selection metric use symmetry-aware no-alignment
  RDKit `CalcRMS` RMSD in Angstrom.
- Seed: 45. Effective global batch: two complexes on two GPUs.
- The returned graph is always the refined primary graph. Cross-bank graph
  validation requires identical topology and node coordinates. The auxiliary
  `edge_ref_dist` is an independently regenerated, unused derived value, so
  only its shape, dtype, and finiteness are checked; its numeric values do not
  enter the model or the cross-bank equality contract.

## One-stage optimization

- Total updates: 100,000.
- Optimizers: Muon at `2e-4` and AdamW at `3e-6`; weight decay `0.01`.
- Scheduler: one warmup-stable-cosine schedule over all 100k updates.
- Warmup: first 2% (2,000 updates).
- Cosine cooldown: final 50% (U50,000 through U100,000).
- Minimum LR ratio: 0.05.
- Validation every 5,000 updates; `latest.pt` every 500 updates.
- `best.pt` maximizes validation Top-1 symmetry-aware RMSD `<2 Angstrom`.

Slurm execution may resume an atomically saved checkpoint after an operational
interruption, but optimizer, scheduler, sampler position, and global update are
restored exactly. Such a restart is not a second training stage.

## Stop and integrity rules

The run is not stopped early for a flat validation point. It stops only at
U100,000 or on a numerical/data/provenance failure. All input hashes, config
hashes, job IDs, logs, and checkpoint identities are recorded. `latest.pt` and
`best.pt` are kept separately and the prior 10k outputs are never overwritten.
