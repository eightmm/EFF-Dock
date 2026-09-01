# Long-Horizon Early-Time Fine-Tune Protocol

Protocol ID: `EFFDOCK-EARLY-TIME-T0P10-50K-V1`
Registered: 2026-08-13, before the 50,000-update run

## Question and interpretation boundary

Does extending the replay-preserving `t=0` 10% intervention to 50,000 fresh
optimizer updates improve the retained geometry checkpoint under uniform-time
inference?

The earlier 2,000-update screen was already opened and its registered endpoint
was negative (`40/256` to `39/256`). This 50,000-update experiment is therefore
a user-authorized long-horizon follow-up, not an independent confirmation of
the original hypothesis. External benchmarks are not used for
checkpoint selection or hyperparameter tuning.

## Frozen inputs

- Initialization: original
  `weights/effdock_geometry_ft_100k_best.pt`, step 100,000,
  SHA256 `6932fb3ba6ebac770f714453529656a44b8f33cf15119d23c9e675d2d60b36db`.
  The 2,000-update screen checkpoint is not used.
- Existing split: `data/splits/plinder.json`, SHA256
  `3ac570bf08bced053f1ce040b57efca27c3be616f29a82cd66ef887c08860e6b`.
  It declares 47,310 train and 1,076 validation IDs. No split is generated,
  reshuffled between train/val, or modified for this run.
- Dataset filters retain 47,277 train systems. Size-aware 4-GPU batching uses
  all 47,277 once per data pass and pads the last global batch with 19
  epoch-rotating duplicates. No fixed tail is discarded. Runtime construction
  must resolve exactly 47,277 train and 1,076 validation systems or fail before
  the first update.
- Frozen configuration SHA256:
  `439d95d7b56e49de6239e113ae4b7e4db94be9e10afe5b8c5f0ece48f076a369`.
  The complete `src/effdock` Python tree plus `pyproject.toml` and `uv.lock`
  have aggregate SHA256
  `6b41a2f744b9f9678f870f4121160479ed559546306b134ca808fae40664e3cc`.
- Seed: 42. Architecture, graph construction, losses, augmentation, prior
  sigma mixture, and receptor policy are held to the registered 2,000-update
  screen configuration.

## Training intervention

- 50,000 fresh AdamW updates from the original checkpoint; optimizer and LR
  scheduler are intentionally reset.
- Time sampling:
  `0.80 SimpleFold + 0.10 U(0,0.3) + 0.10 exact delta(t=0)`.
- Four GPUs, batch 16 per rank, global batch 64, no gradient accumulation.
- AdamW peak LR `2e-5`, weight decay `0.01`, gradient clipping `1.0`.
- LR schedule: 1,000-step warmup, constant through step 40,000, then
  10,000-step cosine decay to `2e-6`.
- EMA decay `0.999`; EMA weights are used for validation and rollout.
- Expected exposure: 50,000 global batches, approximately 67.7 complete data
  passes and 3.2 million sample presentations.

## Evaluation and checkpoint rule

- Same-job step 0 is the baseline and is saved as the named step-0 rollout
  checkpoint before any optimizer update. It is eligible for `best.pt`, so a
  degraded fine-tune cannot replace the retained baseline as best.
- Validation loss: all existing validation systems every 1,000 updates under
  the retained SimpleFold time distribution. It is diagnostic because time is
  resampled.
- Docking rollout: all 1,076 existing validation systems every 5,000 updates,
  one pose per system, sigma 0.5, uniform S20, seed `42 + validation index`.
- Primary metric: step-50,000 `rollout/success_2A` minus same-job step-0
  `rollout/success_2A`.
- Prediction/gate: at least +1.0 percentage point at step 50,000. A smaller,
  null, or negative endpoint disconfirms the long-horizon benefit under this
  protocol and the run is not extended adaptively.
- `best.pt` maximizes the registered validation `rollout/success_2A`; named
  5,000-step checkpoints remain available for the complete trajectory. The
  step-50,000 named/final checkpoint, rather than a post-hoc best step, is the
  registered endpoint.
- Secondary metrics: median/mean RMSD, success below 5 Angstrom, centroid
  distance, fragment RMSD, and validation loss.

## Reliability and stop rules

- Stop immediately on non-finite loss/gradient/output, incomplete validation
  coverage, split/config/code hash mismatch, or irrecoverable checkpoint
  failure.
- `latest.pt` is atomically replaced every 1,000 updates and contains model,
  EMA, optimizer, scheduler, separate RNG state for every DDP rank,
  sampler/data-pass state, config, current best threshold, and metrics.
- A rerun of the same Slurm script resumes only from `latest.pt`, requires the
  exact embedded config and frozen code/config/split hashes, and never silently
  falls back to the source checkpoint when other checkpoint files exist.
- GPU transport uses `NCCL_P2P_DISABLE=1`, the controlled workaround that
  passed the prior 4-GPU large-collective and training smoke tests. This is an
  infrastructure-only change.

## Outcome (opened 2026-08-14)

- Job `53726` completed 50,000/50,000 updates in 10:32:02 with exit `0:0`;
  dependent audit job `53737` passed.
- Registered success below 2 Angstrom improved from `192/1,076` (`17.8439%`)
  at step 0 to `219/1,076` (`20.3532%`) at step 50,000, a `+2.5093`
  percentage-point change. The `+1.0`-point gate passed.
- Median RMSD improved from `4.2186` to `3.9174` Angstrom and success below
  5 Angstrom from `58.7361%` to `62.6394%`.
- The trajectory plateaued: step 25,000 already had `218/1,076` successes and
  the best median RMSD (`3.8114` Angstrom). The adaptive paired dose follow-up
  is frozen separately in `docs/EARLY_TIME_T0_DOSE_10K_PROTOCOL.md`.
