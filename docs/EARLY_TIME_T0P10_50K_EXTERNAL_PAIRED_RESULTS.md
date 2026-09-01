# Early-Time t=0 10% 50k External Paired Results

Protocol: `EFFDOCK-EARLY-TIME-T0P10-50K-EXTERNAL-PAIRED-V1`
Evaluation date: 2026-08-15
Protocol SHA256: `976a292470fa03bef85ac6e8a711a74e1ad32f4b1ca870fa69dcaa3c89c1d203`

## Conclusion

The registered `t=0` 10% 50,000-update fine-tune increased the number of
near-native candidates on both external datasets. Relative to the parent EMA,
it added 1,000 candidates below 2 Angstrom across 393 complexes, or
`+2.545` candidates per 100-pose set (paired 95% CI `+2.130` to `+2.977`).

This is a candidate-density improvement, not a coverage improvement. The
number of complexes with at least one sub-2-Angstrom pose decreased from
378/393 to 373/393. The result is therefore useful for downstream confidence
selection on already-solvable complexes, but it does not justify claiming that
the sampler solves more complexes.

The benchmarks and endpoint had already been opened, so this is a descriptive
compatibility result rather than an independent model-selection or production
admission result.

## Frozen comparison

The primary training-effect contrast is `parent_ema -> t0p10_50k_ema`. A
separate `current_raw -> t0p10_50k_ema` contrast measures the practical effect
of replacing the currently deployed raw checkpoint.

All arms used the same deterministic ODE inference: `sigma=2.0`, 100 poses,
10 steps, late schedule power 3, pocket cutoff 10 Angstrom, prior pool 100,
seed 42, and no confidence, refinement, guidance, FK, or translation SDE.

Strict validation passed for all 85 Astex and 308 PoseBusters complexes. All
three arms had identical per-complex seeds, prior-pool hashes, proteins,
ligand inputs, and references. Every candidate RMSD vector contained 100 finite
values, stored K2 values were exactly recomputed, and all 48 shard summaries
were complete with zero failures.

### Metric implementation caveat

The evaluator uses RDKit `CalcRMS` symmetry matching when it succeeds and a
frozen full-heavy-atom index mapping when it does not. PoseBusters
`5sak_zry` and `6zk5_imh` used the index-mapped fallback in all three arms
because the input and reference bond-order representations prevented
`CalcRMS` matching. All other complexes used the intended symmetry-aware path.

A diagnostic bond-order-normalized symmetry enumeration changed the combined
primary K2 delta only from `+1000` to `+1002`; K>=1 coverage and the conclusion
were unchanged. The reported primary numbers retain the exact frozen evaluator
outputs rather than substituting this post-hoc sensitivity calculation.

## Primary result: parent EMA to t0 EMA

| Dataset | Complexes | K2 total | K2 mean | Delta K2 mean | Paired 95% CI | K>=1 |
|---|---:|---:|---:|---:|---:|---:|
| Astex | 85 | 2464 -> 2698 | 28.988 -> 31.741 | +2.753 | +1.918 to +3.612 | 82 -> 81 |
| PoseBusters | 308 | 8484 -> 9250 | 27.545 -> 30.032 | +2.487 | +2.006 to +2.974 | 296 -> 292 |
| Combined | 393 | 10948 -> 11948 | 27.858 -> 30.402 | +2.545 | +2.130 to +2.977 | 378 -> 373 |

Across the combined cohort, 251 complexes gained K2 candidates, 82 lost them,
and 60 tied. The total number of candidates that were both fast-valid and
below 2 Angstrom increased by 486 (`+1.237` per complex).

The higher-density thresholds improved even though the any-hit threshold did
not:

- `K>=5`: 336/393 to 344/393 (`+8` net).
- `K>=10`: 296/393 to 307/393 (`+11` net).
- First candidate below 2 Angstrom: 99/393 to 107/393 (`+8` net).
- Mean first-candidate RMSD: `-0.166` Angstrom.
- Mean oracle RMSD: `-0.012` Angstrom; the small change is consistent with the
  loss of five any-hit complexes despite the large density gain elsewhere.

PoseBusters K>=1 gained `7xfa_d9j` and lost `7m6k_yrj`, `7mwn_wi5`,
`7omx_cna`, `7rh3_59o`, and `7rni_60i`. Astex lost `1meh` and gained no new
K>=1 complex.

## Checkpoint-effect decomposition

| Contrast | Combined delta K2 total | Delta K2 mean | Paired 95% CI | Net K>=1 |
|---|---:|---:|---:|---:|
| current raw -> parent EMA | +397 | +1.010 | +0.761 to +1.267 | +1 |
| parent EMA -> t0 EMA | +1000 | +2.545 | +2.130 to +2.977 | -5 |
| current raw -> t0 EMA | +1397 | +3.555 | +3.097 to +4.036 | -4 |

Approximately 72% of the current-raw-to-t0 K2 gain came from the registered
fine-tune rather than the raw-to-EMA checkpoint change.

## Execution and artifacts

- Astex Slurm array: `54432`, all eight tasks completed with exit code 0.
- PoseBusters Slurm array: `54442`, all eight tasks completed with exit code 0.
  Each shard ran the three arms in 25:00 to 28:23; the resource-staggered full
  array spanned 65 minutes 35 seconds.
- Strict combined report:
  `outputs/benchmarks/early_time_t0p10_50k_external_paired_runs/t0p10-50k-v1-976a292-20260815/reports/combined_paired_report_final_v2.json`.
- Frozen protocol: `docs/EARLY_TIME_T0P10_50K_EXTERNAL_PAIRED_PROTOCOL.md`.

## Decision implication

The t0 fine-tune is supported for the intended goal of giving a later
confidence model more near-native candidates to rank. It should not yet be
treated as an unconditional sampler replacement because the any-hit tail lost
five complexes. Confidence retraining should use t0-model samples, while the
coverage trade-off should be checked on the frozen internal validation set or
handled by a predeclared mixed/checkpoint-ensemble sampling policy rather than
by further adapting to these opened external benchmarks.
