# Paired Exact-Zero Dose Continuation Protocol

Protocol ID: `EFFDOCK-EARLY-TIME-T0-DOSE-10K-V1`
Registered: 2026-08-14, before either 10,000-update branch

## Question and interpretation boundary

After the opened 50,000-update `80/10/10` run plateaued, does reallocating
five percentage points of continuous early-time sampling to exact `t=0`
improve the retained checkpoint beyond a matched continuation?

This is an adaptive internal-validation dose study. It is not an independent
confirmation, and external benchmarks remain closed for training,
checkpoint selection, and dose selection. A result cannot establish that the
parent 50k gain was caused by exact-zero sampling because the original run had
no SimpleFold-only continuation control.

This single-seed comparison holds every non-time distribution fixed but is not
a common-random-number causal estimate. With `deterministic: false`, choosing a
continuous-early versus exact-zero component consumes a different number of
worker RNG draws, so the two arms subsequently realize different (but
identically distributed) prior/sigma/augmentation samples. The endpoint is
therefore an adaptive dose screen; any positive result still needs independent
seed or closed-benchmark confirmation.

## Opened parent result and common initialization

- Parent job `53726` completed 50,000 updates in 10:32:02; audit job `53737`
  passed. Full-val uniform-S20 success below 2 Angstrom increased from
  `192/1,076` (`17.8439%`) to `219/1,076` (`20.3532%`).
- Most improvement occurred by step 25,000 (`218/1,076`). Steps 25,000 to
  50,000 added one success, so an unbounded same-mixture continuation is not
  justified.
- Parent `best.pt` is step 50,000, SHA256
  `ad6c794851698294b38246dd173035e0d336b9e12bc5a1c91289b241c22b3756`.
- Its EMA is promoted to the canonical raw model and retained as the new EMA
  in the common weight-only initialization
  `step50000_ema_common_init.pt`, SHA256
  `65be44d7dc8f0867eb9fc5d22214b80f93971ea4702679a527c665046e91e6b6`.
  Both branches start from these identical 490 tensors; optimizer and
  scheduler state are reset identically.

## Frozen data and implementation

- Existing split only: `data/splits/plinder.json`, SHA256
  `3ac570bf08bced053f1ce040b57efca27c3be616f29a82cd66ef887c08860e6b`.
  It is not regenerated or reshuffled. Runtime must resolve exactly 47,277
  filtered train and 1,076 validation systems.
- Complete `src/effdock` Python tree plus `pyproject.toml` and `uv.lock`
  aggregate SHA256:
  `6b41a2f744b9f9678f870f4121160479ed559546306b134ca808fae40664e3cc`.
- Control config SHA256:
  `13d2f9d8b7d64eb4b0286c6cdf84cba0110364991f9d7f16eba94e5f1be1cbde`.
- Treatment config SHA256:
  `e24d3068d8c2d768c1643fec581cca41098acb2edb1bd431b29c16ddde0972f4`.
- The configs differ only in the registered mixture and branch-specific
  output/logging names.

## Arms and training budget

- Control: `0.80 SimpleFold + 0.10 U(0,0.3) + 0.10 delta(t=0)`.
- Treatment: `0.80 SimpleFold + 0.05 U(0,0.3) + 0.15 delta(t=0)`.
- The treatment holds SimpleFold replay at 80% and total explicit early
  intervention at 20%; it only moves 5% from the early neighborhood to the
  exact first-call point.
- Each arm: 10,000 updates, four RTX 6000 Ada GPUs, batch 16 per rank, global
  batch 64, seed 42, full filtered train split, and 19 epoch-rotating padding
  examples per 739-update data pass.
- AdamW peak LR `1e-5`, weight decay `0.01`, clipping `1.0`; 500-update
  warmup, 7,500 stable updates, and 2,000-update cosine cooldown to `1e-6`.
- EMA decay `0.999`; EMA weights are used for every validation and rollout.
  Architecture, losses, augmentation, sigma mixture, receptor policy, and
  every non-time configuration value are identical.

## Evaluation and decision rule

- Both branches save and verify an identical step-0 full-val baseline before
  any update.
- Full 1,076-system PLINDER uniform-S20 single-pose rollout at steps
  `0, 2k, 4k, 6k, 8k, 10k`, sigma `0.5`, seed `42 + validation index`.
- Registered primary endpoint: treatment minus control step-10,000
  `rollout/success_2A`. Intermediate best steps are trajectory diagnostics and
  cannot replace the endpoint.
- Prediction: treatment exceeds control by at least `+1.0` percentage point,
  which requires at least 11 additional successes out of 1,076.
- Treatment must also retain the shared step-0 success, remain within
  `-0.5` percentage point of control for success below 5 Angstrom, and remain
  within `+0.10` Angstrom of control median RMSD. All four gates must pass
  before considering an exact-zero 20% follow-up.
- A null/negative endpoint keeps the current 10% exact-zero checkpoint and
  ends this dose escalation. No adaptive extension beyond 10,000 updates.

## Reliability and outputs

- Stop on non-finite loss/gradient/output, incomplete coverage, source/split/
  config/code hash mismatch, or irrecoverable checkpoint inconsistency.
- Atomic `latest.pt` every 1,000 updates contains model, EMA, optimizer,
  scheduler, distinct RNG state for all four ranks, data-pass position,
  current best threshold, exact config, and metrics.
- Synchronized branch locks prevent duplicate writers. A rerun resumes only
  from exact-config `latest.pt`; it never silently restarts from the common
  initialization when other checkpoints exist.
- Each branch atomically records a `run_identity.json` tied to the producing
  Slurm array/job/task IDs, common-init/split/config/code hashes, and hashes of
  its completed S0, S10k, and `latest.pt` artifacts. A failed or stale producer
  cannot satisfy the dependent audit.
- Control output:
  `outputs/eff-dock/early-time-t0-dose-control-t0p10-10k-v1-20260814`.
- Treatment output:
  `outputs/eff-dock/early-time-t0-dose-treatment-t0p15-10k-v1-20260814`.
- A run-scoped dependent CPU audit validates both complete rollout inventories,
  common-init-identical S0 raw/EMA tensors and metrics, every named config/RNG/
  metric schema, endpoint/latest and named-best tensor identity, integer-count
  gate arithmetic, producer manifests, and every registered gate. Audit code
  and checkpoint-loader source hashes are frozen before launch.
