# Fixed-NFE step/pose allocation protocol

Protocol ID: `EFFDOCK-FIXED-NFE-STEP-POSE-V1`

Status: frozen before the new `N40/S25` Astex or PoseBusters outputs are
generated.

## Question and claim boundary

Under the current EFF-Dock inference stack, does spending the same learned
model budget on more ODE integration steps per pose (`N40/S25`) or more poses
(`N100/S10`) provide better RMSD success coverage?

Astex Diverse and PoseBusters v2 have already been inspected repeatedly. This
is therefore a paired descriptive allocation study, not independent external
validation. Its outcomes cannot select or production-admit an ODE schedule,
guidance coefficient, refinement rule, docking checkpoint, or confidence
checkpoint.

## Frozen arms

| Arm | ODE steps per pose | Poses per complex | Learned pose-steps |
|---|---:|---:|---:|
| `s10_n100` | 10 | 100 | 1,000 |
| `s25_n40` | 25 | 40 | 1,000 |

The completed `s10_n100` arm is reused without regeneration. Only `s25_n40`
is newly sampled. Both arms use:

- all 85 Astex Diverse and 308 PoseBusters v2 complexes in the frozen full
  benchmark-input manifest;
- docking checkpoint `weights/effdock_geometry_ft_100k_best.pt`, SHA-256
  `6932fb3ba6ebac770f714453529656a44b8f33cf15119d23c9e675d2d60b36db`;
- `sigma=2.0`, late schedule with power `3`, pocket cutoff `10 A`, and zero
  center jitter;
- normalized direct-drift unified guidance with `eta=2.0`, start `t=0.5`,
  ramp power `1`, force cap `20`, translation/angular caps `5/5`, atom-step
  cap `0.25 A`, and the `geometry_only` receptor policy;
- seed `42` plus the frozen global complex offset; and
- one deterministic 100-pose translation/SO(3) prior pool per complex.

The new `s25_n40` arm must have the same `sampling_seed` and exact
`prior_pool_sha256` as the reused `s10_n100` arm for every complex. Its 40
poses are therefore the first 40 members of the same prior pool. Any mismatch
invalidates aggregation.

The stored shared-prior hash includes the byte-exact normalized quaternion
pool. A pre-full smoke on 2026-08-26 established that the cluster's `gpu1` and
`gpu3` CPU families can differ in the final floating-point bits of quaternion
normalization despite identical Torch seeds and random variates. Controlled
Slurm probes on 2026-08-27 reproduced the reused arm's exact hash on both
`gpu1` and the `test` partition's `gpu2`, while `gpu3` reproduced the failed
cross-node smoke hash. Because all `heavy` GPUs were occupied, new `s25_n40`
sampling is pinned to the preflighted `test`/RTX A5000 node family. Runtime
identity and the exact per-complex prior hash remain mandatory gates. The
failed cross-node smoke is retained as non-claim-bearing provenance and is not
aggregated. No hash tolerance or post-hoc equivalence rule is used. Since the
two arms use different GPU families, their wall-clock runtimes are descriptive
only and are not compared as a speed benchmark.

Sampling runs with the immutable source capsule used for the completed
sigma-2/eta-2 arm:

- capsule manifest SHA-256
  `62c698577a3b4ad407b9926ec922dae201fbce45e189e6aae2b83c4d4fe0cb35`;
- reused `s10_n100` source manifest SHA-256
  `9e8be4d47dba8e346a6900b6bf02f5b853a93141f571f5c65b4c719de632d695`.

## Frozen post-processing

Every newly generated `s25_n40` pose is passed through the same adaptive
rigid-fragment post-refinement used for the reused arm:

- unified in-repository GuidanceEnergy only; no Vina or external minimizer;
- at most 100 accepted-step attempts with materialized steps
  `0,25,50,75,100`;
- maximum atom displacement `0.10 A` and at most 12 backtracks;
- energy-plateau stopping from step 25 with
  `0.02 kcal/mol + 1e-3 * max(1 kcal/mol, |E|)`, patience 5.

Refinement uses the frozen recovery capsule whose manifest SHA-256 is
`4891267ff04d52915be1f3c39a9a78ffa82a19e87589c28971f3cf7d63becb75`.

Final Top-1 selection uses pure stable argmin predicted RMSD from the U50k
symmetry-confidence checkpoint, conditioned at sigma 2:

- feature-backbone checkpoint SHA-256
  `65be44d7dc8f0867eb9fc5d22214b80f93971ea4702679a527c665046e91e6b6`;
- U50k confidence checkpoint SHA-256
  `fd49fa86f67187bf26d6c1bcf2daf925ba3e3b19dfeae733e57535d183280469`;
- pose chunks of 20 on the `6000ada` partition.

## Metrics

The primary curve is cumulative refined RMSD oracle success in original
sampling order:

`OracleSR(k) = mean_i[ min_{0 <= j < k} RMSD(i,j) < 2 A ]`.

The curve is reported for `k=1..100` for `s10_n100` and `k=1..40` for
`s25_n40`. A raw post-ODE/pre-refinement version is reported separately.
Endpoints at the equal 1,000 learned pose-step budget are `k=100` and `k=40`,
respectively. Secondary metrics are U50-selected Top-1 `<2 A`, Oracle@40 on
the common pose-count support, final oracle `<2 A`, median selected RMSD, and
wall-clock/runtime diagnostics. This protocol does not run or claim official
PoseBusters validity; RMSD SR and PB-validity remain distinct endpoints.

## Gates and stopping rules

1. A two-complex smoke (one Astex, one PoseBusters) must pass sampling, prior
   pairing, refinement, and U50 scoring before the full sampling array starts.
2. Full aggregation requires exact `85/85 + 308/308` coverage, 40 readable
   poses per new complex, zero failed/non-finite terminal poses, and exact
   seed/prior-pool pairing to all 393 reused records.
3. Any changed source capsule, checkpoint, input manifest, reference structure,
   guidance parameter identity, or duplicate/missing complex stops the report.
4. The single frozen paired seed is reported as such; no variance or
   multi-seed generalization claim is made.
