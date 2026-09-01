# Early-Time t=0 10% 50k External Paired Evaluation Protocol

Protocol ID: `EFFDOCK-EARLY-TIME-T0P10-50K-EXTERNAL-PAIRED-V1`
Initially frozen: 2026-08-14
Amended: 2026-08-15, before any candidate sampling under this protocol

The amendment followed two preflight-only Slurm failures that occurred before
model loading or candidate generation. It replaces a guidance-specific cohort
validator whose implementation identity had changed, even though all guidance
is disabled here, with an exact static membership/input/reference validator.
It also removes an internally inconsistent outcome-dependent PoseBusters gate:
the declared combined 393-complex estimand requires both datasets regardless of
the already-opened Astex result.

## Question and interpretation boundary

Does the registered `t=0` 10% 50,000-update fine-tune increase the number of
near-native candidates under the intended sigma-2 inference regime on Astex
and PoseBusters?

These external benchmarks and the internal 50k endpoint have already been
opened. This is therefore a paired descriptive compatibility check, not an
independent confirmation, a checkpoint-selection set, or a production
admission test. No result from this evaluation may trigger another adaptive
sampling-model fine-tune.

## Frozen checkpoint arms

All arms use `configs/train.yaml`, SHA256
`39aa62e4a48ed6f3aa4ff59345fb43a81220e2baba22edfd5beb0c4981b307ec`.

1. `current_raw`: current deployed checkpoint
   `weights/effdock_geometry_ft_100k_best.pt`, step 100,000, canonical raw
   weights, SHA256
   `6932fb3ba6ebac770f714453529656a44b8f33cf15119d23c9e675d2d60b36db`.
2. `parent_ema`: EMA from the same parent checkpoint, exported without
   changing tensors, EMA count 300,000, SHA256
   `166d92a7f74015b0011451ad70c71601d72769da00ce1206c8a6a27832e40d97`.
3. `t0p10_50k_ema`: the named step-50,000 endpoint of the registered
   fine-tune, promoted EMA count 350,000, SHA256
   `65be44d7dc8f0867eb9fc5d22214b80f93971ea4702679a527c665046e91e6b6`.

The primary training-effect contrast is `parent_ema -> t0p10_50k_ema`.
`current_raw -> t0p10_50k_ema` is the practical replacement contrast, and
`current_raw -> parent_ema` isolates the parent raw/EMA choice. The file named
`best.pt` in the fine-tune run is not eligible because its canonical state is
raw rather than the validated endpoint EMA.

## Frozen cohort and inputs

- Astex: all 85 systems.
- PoseBusters Benchmark: all 308 systems.
- Input manifest:
  `docs/GUIDANCE_BUDGET1000_FULL_INPUTS.json`, SHA256
  `99f15f557644cc51c3dd1f559b0dd97dd4259c1de3e1403fb761b7c7e079f668`.
- Full-cohort audit:
  `docs/GUIDANCE_BUDGET1000_FULL_COHORT.json`, SHA256
  `d7321f847c8d6d08950e02d5f41ff42b62fd29ccea78072f27078aa039791c45`.
- Astex pocket centers SHA256:
  `1ac4d8629a7ee2adb785173db56fb69ec4140d68e3057631ae10df6ef88d0d85`.
- PoseBusters pocket centers SHA256:
  `2d3db55c8cc75650cff85d8e3c12445fb8f45fbe2673d8bbc32045ee8c0f6ad0`.
- Ligands are constructed from the frozen SMILES inputs; the crystal ligand is
  used only for symmetry-aware evaluation. Full heavy-atom mapping is required.

Each dataset is split into eight deterministic shards. Every shard task runs
the checkpoint arms sequentially on the same GPU. Coverage must be exactly
85/85 and 308/308 with zero recorded failures.

## Frozen inference settings

- The complete `src/effdock` Python tree plus `pyproject.toml` and `uv.lock`
  have aggregate SHA256
  `a3c7b5da8898f4057fcf0a71771738495ddec0ca37213c43c04a51e3aa980a35`.
  The evaluator file that records the complete candidate RMSD vector has
  SHA256
  `91db3ffd2953a04a46b2af2e8854a1fdf78328cb83d88c83aa9f3199e706734c`.
- Deterministic ODE rollout; translation SDE is disabled.
- No confidence model, refinement, Vina guidance, unified guidance, or
  Feynman--Kac resampling/constraint term.
- `sigma=2.0`, `N=100`, `S=10`, prior pool size 100.
- Late time schedule with power 3, pocket cutoff 10 Angstrom, center jitter 0.
- Base seed 42. The per-complex seed is 42 plus the one-based position in the
  globally sorted dataset ID list.
- All 100 candidate poses are retained for audit.

For every complex, `sampling_seed` and `prior_pool_sha256` must be identical
across all three arms. Candidate ordering is paired by the shared prior index.
Any seed, prior, input, reference, or setting mismatch invalidates the paired
comparison.

## Metrics and estimands

The primary per-complex metric is

`K2 = number of the 100 candidates with symmetry-aware heavy-atom RMSD < 2 A`.

The strict inequality is frozen. The primary aggregate is the complex-weighted
mean and total paired change in K2 for `parent_ema -> t0p10_50k_ema`, reported
for Astex, PoseBusters, and the combined 393-complex cohort. Uncertainty is a
deterministic paired bootstrap over complex IDs, never over candidate poses.

Key secondary metrics are:

- number and fraction of complexes with `K2 >= 1`, including gained and lost
  transitions;
- fractions with `K2 >= 5` and `K2 >= 10`;
- oracle minimum RMSD and first-candidate RMSD;
- number of candidates that are both fast-valid and below 2 Angstrom;
- positive, negative, and tied per-complex K2 changes;
- complete candidate RMSD vectors, retained so K2 can be independently
  recomputed.

The practical replacement and raw/EMA contrasts receive the same secondary
summary but are not substituted for the primary training-effect contrast.

## Smoke, reliability floor, and execution order

Before full sampling, run Astex `1jje` and PoseBusters `7b2c_tp7` with
`N=2`, `S=2`, and prior pool size 2. The smoke also repeats `current_raw` as
`current_raw_replay` with identical settings. All arms must finish, produce
finite two-candidate RMSD vectors, and share the exact seed and prior hash.
The repeated raw arms must agree in K2 and in the `K2 >= 1` classification;
coordinate hashes need not be identical. A disagreement is a numerical
reliability failure and full sampling is not launched.

Astex is run and strictly audited before PoseBusters as an engineering staging
step. Once the smoke and Astex integrity checks pass, PoseBusters is run
unconditionally; Astex efficacy metrics do not control that decision. This
preserves the declared combined 393-complex estimand and avoids making an
opened benchmark outcome part of the sampling path.

The static cohort gate retains the frozen cohort-manifest SHA and requires the
current frozen-SMILES mapping, current discovery, cohort ID/audited/success
sets and hashes, canonical benchmark input identity, and every current protein
and ligand-reference hash to match exactly. It requires complete coverage and
zero failures. Guidance implementation, parameter, and receptor-policy hashes
are intentionally not compared because Vina guidance, unified guidance,
Feynman--Kac resampling, and translation SDE are all disabled.

## Failure and reporting rules

Stop on non-finite metrics, incomplete coverage, recorded evaluation failure,
checkpoint/input/config hash drift, mismatched paired seeds or priors, malformed
candidate RMSD vectors, or disagreement between stored and recomputed K2.
Engineering failures may be repaired without changing the frozen scientific
settings. Reports must retain all failures and explicitly label this experiment
descriptive.
