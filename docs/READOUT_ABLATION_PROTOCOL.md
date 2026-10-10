# Readout ablation protocol (frozen before any result)

Frozen 2026-10-09, before either arm was trained.

## Question

Under a matched, reduced training budget, does the least-squares Newton–Euler
readout (atom vectors aggregated into fragment translation and angular
velocity) generate better candidates than a direct fragment readout? This tests
candidate generation only. It does not assess the ranked pipeline, production-
scale training, or variability across training seeds.

## Arms

Identical data, split (`data/splits/plinder.json`, 47,277 train / 1,076 val),
configuration (`configs/ablation_readout_<arm>.yaml`), optimizer, schedule,
seed 42, hardware type (one RTX PRO 6000 Max-Q per arm) and update count.
Only `model.readout` differs.

- `newton_euler`: production head; per-atom 1o vectors, unit-mass least-squares
  rigid-motion fit with truncated inertia pseudoinverse.
- `fragment_direct`: the same head structure applied to fragment-node features,
  emitting 1x1o translation and 1x1e angular velocity; angular velocity is
  projected onto the same observable subspace used by the loss.

Training: from scratch, 100,000 updates, global batch 16, AdamW 3e-4,
warmup 2%, cosine cooldown over the final 50% to 5%, EMA 0.999. Trainable
parameter counts are reported for both arms.

## Checkpoint rule

The terminal EMA checkpoint at update 100,000 is evaluated for each arm. No
checkpoint is chosen by any outcome.

## Endpoints

Primary: internal-validation rollout success, fraction of the 1,076 PLINDER
validation complexes with single-sample heavy-atom RMSD < 2 Å after 20 late-
biased steps, at the terminal checkpoint (trainer rollout).

Secondary:
- internal-validation median rollout RMSD; learning curves every 10,000 updates;
- Astex (85) and PoseBusters v2 (308) unguided N100/S10 raw banks with
  sigma 2, 10 Å crop, seed 42: oracle@100 and oracle@10 (RMSD < 2 Å), median
  minimum RMSD. No refinement, confidence ranking or PoseBusters, because the
  released confidence model consumes production docking features.

Paired differences use the same complexes and seeds; complex-level bootstrap
(10,000 resamples) intervals describe evaluation uncertainty only.

## Reporting

All outcomes are reported whatever their direction. Null: "no clear advantage
under this protocol; equivalence is not established." Negative: "the direct
readout performed better under this protocol; the results do not support an
advantage of the least-squares readout." A negative result is not attributed
to undertraining without evidence.
