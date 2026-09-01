# Early-Time Fine-Tune Results

## Decision

The registered `t=0` 10% early-time replay screen did **not** pass its primary
gate. At the fixed step-2,000 endpoint, uniform-S20 single-pose success below
2 Angstrom was `39/256 = 15.234375%`, versus the same-job step-0 baseline of
`40/256 = 15.625%`. The change was `-0.390625` percentage points (one fewer
successful system); the registered `+1` percentage-point gate required at
least `43/256` successes.

This result does not support promoting the `0.80 SimpleFold + 0.10 U(0,0.3) +
0.10 delta(t=0)` sampler to the production training default or using
Astex/PoseBusters to select among these checkpoints without a new registered
decision.

## Registered protocol

- Initialization: `weights/effdock_geometry_ft_100k_best.pt`
- Training: 2,000 fresh AdamW updates, four GPUs, global batch size 64,
  `lr=2e-5`, EMA decay `0.999`, seed 42
- Time distribution: `0.80 SimpleFold + 0.10 U(0,0.3) + 0.10 exact t=0`
- Evaluation: first 256 PLINDER validation systems, sigma `0.5`, uniform S20,
  one pose per system
- Primary endpoint: step-2,000 `rollout/success_2A` relative to the identical
  step-0 evaluation
- Slurm job: `53692`, `COMPLETED (0:0)` in `00:40:21`
- Infrastructure-only exception: `NCCL_P2P_DISABLE=1` was used after a
  controlled large-collective probe reproduced a node-local NCCL P2P timeout.
  It does not change model math or the registered sampling distribution.

## Results

| Update | Validation loss | Median RMSD (A) | Success below 2 A | Delta vs step 0 | Success below 5 A |
|---:|---:|---:|---:|---:|---:|
| 0 | 5.9437 | 4.43* | 40/256 (15.625%) | 0.000 pp | 145/256 (56.641%) |
| 500 | 6.2970 | 4.366800 | 41/256 (16.016%) | +0.391 pp | 150/256 (58.594%) |
| 1,000 | 6.1006 | 4.321666 | 41/256 (16.016%) | +0.391 pp | 148/256 (57.812%) |
| 1,500 | 6.0910 | 4.291815 | 41/256 (16.016%) | +0.391 pp | 148/256 (57.812%) |
| 2,000 | 5.7157 | 4.343503 | 39/256 (15.234%) | -0.391 pp | 150/256 (58.594%) |

`*` The step-0 median is available only at the two-decimal precision printed in
the job log. All success fractions are exact counts over 256 systems.

The secondary median RMSD improved modestly through step 1,500, then regressed
slightly at the registered endpoint. Step 1,500 had the lowest observed median
RMSD but tied steps 500 and 1,000 on the primary metric and still missed the
gate. Validation loss is diagnostic because validation times are resampled;
it is not a paired deterministic comparison.

No non-finite loss/gradient, CUDA OOM, NCCL error, or rollout rank skew occurred
in job 53692. All four ranks completed every 64-sample rollout shard, and all
five evaluations covered exactly 256 systems.

## Artifacts and provenance

- Config: `configs/train_early_time_ft.yaml`
  (`7a0563bc07216eadbcd4f7f74911d200eec62195c3c60db02bbc00daab52626a`)
- Initialization checkpoint:
  `weights/effdock_geometry_ft_100k_best.pt`
  (`6932fb3ba6ebac770f714453529656a44b8f33cf15119d23c9e675d2d60b36db`)
- Step-2,000 resumable checkpoint:
  `outputs/eff-dock/early-time-ft-t0p10-v1-retry1-20260813/checkpoints/rollout_step0002000.pt`
  (`0b29d2ca9b664bc0cac897d60828af36b0dbb5b1c603474c905d5785a4d88fcd`)
- Step-2,000 EMA inference checkpoint:
  `outputs/eff-dock/early-time-ft-t0p10-v1-retry1-20260813/checkpoints/early_time_t0p10_step2000_ema_inference.pt`
  (`e0e85ae44c416ee4bdebb6e72ff2c26f015d0de4ccc8cfc9c43e6152cd280d4a`)
- Stdout: `outputs/slurm/effdock-early-time-ft-53692.out`
  (`833447770d693f2e8ec48daf5e81575db042ee3e77d8e5368a51b145c3cfa676`)
- Stderr: `outputs/slurm/effdock-early-time-ft-53692.err`
  (`e45e2be2e39bf5837f740544b15f2305609cde867bf7a663f7fd4583cd789816`)

The rollout metrics were computed with EMA weights. The inference artifact
therefore promotes the stored EMA state to canonical `model_state_dict`, keeps
the original checkpoint untouched, records its source SHA256, and deliberately
omits optimizer/scheduler/RNG state. The repository's standard inference loader
successfully loaded this artifact as an `EFFDock` model on CPU.

Post-run integrity review found that the run-time save order left the
`best_rmsd` resume threshold one evaluation stale in `best.pt` and the named
step-500/1,000/1,500 rollout checkpoints. Their weight and metric payloads were
already correct. The current files have been atomically corrected to the actual
cumulative minima (`4.366800`, `4.321666`, and `4.291815`); byte-preserving
pre-fix copies use the suffix `.pre_best_rmsd_fix.pt`. A recursive comparison
verified that `best_rmsd` is the only changed field. The trainer save ordering
and regression test are also fixed for future runs. The recorded step-2,000 and
EMA inference hashes above were unaffected.

The run ledger was reconciled after completion: job `53692` is recorded as
`COMPLETED`, exit `0:0`, elapsed `00:40:21`.
