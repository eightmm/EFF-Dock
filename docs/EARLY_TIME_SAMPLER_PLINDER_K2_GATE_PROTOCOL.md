# Early-time sampler PLINDER K2 gate protocol

Protocol ID: `EFFDOCK-EARLY-TIME-SAMPLER-PLINDER-K2-GATE-V1`

Status: frozen before deployment-aligned PLINDER candidate sampling.

## Question

Does another 10,000 updates with the existing `80% SimpleFold / 10% U(0,0.3) /
10% t=0` time mixture improve the 50k sampler under the intended inference
regime without losing rare-complex coverage or pose diversity?

This is an internal development/selection experiment. It is not an external
generalization claim.

## Frozen arms

The primary selection contrast is `s50_ema -> parent50k_plus10k_t0p10_ema`.
All arms use promoted EMA weights, never the raw `model_state_dict` contained in
a training checkpoint.

| Arm | Role | Artifact | SHA-256 |
|---|---|---|---|
| `s25_ema` | report-only plateau diagnostic; not eligible for selection | `outputs/eff-dock/early-time-t0p10-50k-v1-20260813/checkpoints/step25000_ema_inference.pt` | `c343ebc34cea3395762cd82e1c54b8c7b847dc04c4fa9e80b9813a864cafa0e1` |
| `s50_ema` | selection baseline | `outputs/eff-dock/early-time-t0p10-50k-v1-20260813/checkpoints/step50000_ema_common_init.pt` | `65be44d7dc8f0867eb9fc5d22214b80f93971ea4702679a527c665046e91e6b6` |
| `parent50k_plus10k_t0p10_ema` | selection treatment | `outputs/eff-dock/early-time-t0-dose-control-t0p10-10k-v1-20260814/checkpoints/parent50k_plus10k_t0p10_ema_inference.pt` | `0a48577379e286c584abd8c652d079b09dd6fff3c06a1a2f433d617ab0cd6074` |

The 15% exact-zero arm is excluded because its earlier frozen dose experiment
already triggered the stop-escalation rule. The 25k arm was chosen after viewing
the older single-pose validation curve, so it can diagnose a plateau but cannot
be promoted by this experiment.

## Frozen inputs and cohort

- PLINDER release: `2024-06/v2`.
- Validation split: `data/splits/plinder.json`, SHA-256
  `3ac570bf08bced053f1ce040b57efca27c3be616f29a82cd66ef887c08860e6b`,
  exactly 1,076 unique sample keys.
- Canonical ligand inputs: `data/plinder_pool.parquet`, SHA-256
  `0ff455da77ce5540b839918cccb96f45414e91efff6272d7da3a65337ab1fe91`.
- Model/inference config: `configs/train.yaml`, SHA-256
  `39aa62e4a48ed6f3aa4ff59345fb43a81220e2baba22edfd5beb0c4981b307ec`.
- Raw-asset gate: `outputs/benchmarks/plinder_guidance_validation_runs/20260804T042517Z/raw_gate/verified.json`,
  SHA-256 `1ac146cfbec49ebfd1eb4452219320f134b0261bc8dc1bc196bcdab91b60f546`.
- Outcome-independent conformer/mapping audit:
  `outputs/analysis/rdkit_fragment_geometry_v2/val1076_seed0_heavy_only.json`,
  SHA-256 `d30f7380186d914b60964e120280dd84470b0f67b5a8aa9548e499af0aa942bf`.

All 1,076 split entries remain in the attempted and operational denominators.
The paired K2 estimand is frozen to the 1,035 entries (1,020 unique PLINDER
systems) whose pre-existing audit record has all of:

- `status == "ok"`;
- `mapping_method == "strict_stereo"`;
- `symmetry_complete == true`;
- production conformer seed `0` and heavy-only normalization.

The eligible sample keys, lexicographically sorted and encoded as UTF-8 with
one key per line plus a terminal newline, have SHA-256
`005577bbf2b0c1c1e98bac3092b8e5350a6aa06597442b4c86d05f24e763593f`.

The 41 ineligible entries are recorded as preprocessing failures for every arm
and are never replaced with a crystal-derived input conformer. A full-1,076
operational sensitivity assigns those common preprocessing failures `K2=0`;
the primary paired treatment effect remains the evaluable-cohort result.

Input ligand coordinates are generated only from the frozen canonical SMILES
using production ETKDGv3/MMFF and conformer seed 0. The crystal ligand is used
only as the RMSD reference. Pocket centers are the frozen crystal-defined
processed PLINDER centers, so the scope is known-pocket redocking, not blind
pocket prediction.

## Frozen inference contract

Each eligible sample uses the same conditions in all arms:

- `sigma=2.0`, 100 candidates, 100 shared priors;
- 10 deterministic ODE steps with `late`, power 3;
- pocket cutoff 10 A and center jitter 0;
- sampling seed `42 + one-based index in the complete sorted 1,076-key split`;
- ligand conformer seed 0, independent of the sampling seed;
- confidence, Vina selection/guidance, unified guidance, FK resampling, SDE,
  post-refinement, and stochastic sampling all disabled;
- candidate-only evaluation, full heavy-atom mapping required;
- all 100 RMSDs, fast-valid labels, prior hash, candidate hash, and candidate
  diversity fields retained.

Within a shard the three arms run sequentially on the same GPU. An ID is paired
only when its sampling seed and exact CPU prior-pool SHA-256 match across all
arms. Any checkpoint, config, input, inventory, seed, prior, finite-value,
candidate-count, or output-completeness mismatch invalidates the comparison.

## Outcomes

For sample `i`, `K2_i` is the number of its 100 runtime pose RMSDs strictly
below 2 A. The evaluator prefers RDKit symmetry-aware no-alignment `CalcRMS`
and may use its existing full-heavy-atom mapped-index fallback when RDKit cannot
compare equivalent molecular representations; any observed fallback is listed.

Primary efficacy outcome:

- sample-weighted mean paired change in K2 for
  `s50_ema -> parent50k_plus10k_t0p10_ema` over the fixed 1,035 samples.

Uncertainty is a 20,000-resample percentile bootstrap using NumPy `PCG64` with
seed 20260815,
resampling the 1,020 `system_id` clusters and retaining every ligand sample in a
drawn system. A system-balanced mean is reported as a sensitivity.

Hard coverage guards:

1. `K2>=1` sample coverage must not decrease.
2. The paired-bootstrap 95% interval lower bound for the `K2>=1` coverage
   percentage-point change must be at least `-1.0 pp`.
3. Among baseline-fragile samples with `s50_ema K2` in `[1,4]`, at least 95%
   must retain `K2>=1` in the treatment.

Hard fast-valid guards:

1. Mean fast-valid K2 must not decrease.
2. Fast-valid `K2>=1` sample coverage must not decrease, and its paired
   cluster-bootstrap interval lower bound must be at least `-1.0 pp`.
3. The overall fraction of fast-valid candidates may decrease by at most
   `1.0 pp`.

Hard diversity guard:

- Define `NN_j` as the within-sample median nearest-neighbor receptor-frame
  same-index heavy-atom RMSD. Define `C2_j` as the number of connected
  components in the 100-pose graph with an edge whenever pair RMSD is strictly
  below 2 A.
- The sample-weighted treatment/baseline aggregate ratio must be at least 0.95
  for both NN and C2, and each system-cluster-bootstrap 95% interval lower bound
  must be at least 0.90.
- After rounding coordinates to 0.001 A, the treatment unique-pose fraction
  must be at least 0.99 and may be at most 0.005 below baseline.

Secondary outcomes are total/median K2, `K2>=5`, `K2>=10`, oracle RMSD, first
candidate RMSD, fast-valid K2, gained/lost ID lists, and the corresponding 25k
diagnostic comparisons.

## Decision rule

The existing +10k continuation passes only if every hard coverage/diversity
guard passes and both efficacy conditions hold:

- mean K2 improves by at least `+1.0` candidate per 100 poses; and
- the cluster-bootstrap 95% confidence interval lower bound is greater than 0.

If it passes, promote the existing +10k EMA sampler and end sampler training;
confidence-model retraining is the next stage. If efficacy is near-null,
negative, inconclusive, or any coverage/fast-valid/diversity guard fails, keep
the 50k EMA and stop additional t0 continuation. This experiment does not
automatically authorize another adaptive 5--50k continuation; that would need a
new temporal holdout and matched-control protocol.

## Execution stages

1. CPU preflight verifies the split, audit, exact 1,076-to-1,035 accounting,
   eligible digest, input assets, code, config, and checkpoint hashes.
2. Engineering smoke uses the first 8 eligible keys, `N=4/S=2`, the three arms,
   and an `s50_ema` replay. It inspects only finite/count/prior/replay integrity,
   not efficacy.
3. Numerical pilot uses the first 32 eligible keys at full `N=100/S=10` with an
   `s50_ema` replay. It requires zero `K2>=1` and fast-valid `K2>=1` replay
   classification mismatches, replay mean absolute K2 difference at most 0.25,
   and replay diversity aggregate ratios in `[0.98,1.02]`.
4. Only after those checks pass, run all 1,035 eligible entries in 8 fixed
   shards, with at most 4 one-GPU tasks concurrently, followed by the strict CPU
   audit/report.

No threshold or cohort may be changed after candidate outcomes are opened.
