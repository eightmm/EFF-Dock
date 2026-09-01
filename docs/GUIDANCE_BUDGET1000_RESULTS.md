# Unified-guidance 1,000-step budget results

Protocol: `EFFDOCK-UNIFIED-GUIDANCE-BUDGET1000-V1`
Completed: 2026-07-31
Docking checkpoint:
`weights/effdock_geometry_ft_100k_best.pt`
(`sha256:6932fb3ba6ebac770f714453529656a44b8f33cf15119d23c9e675d2d60b36db`)

## Decision

Use **100 poses × 10 ODE steps (N100/S10)** as the default fixed-1,000-step
allocation.

- It maximizes or ties guided oracle `<2 Å` on both supported cohorts.
- It has the best guided joint accurate-and-fast-valid candidate coverage on
  Astex and ties the best result on PoseBusters.
- N40/S25 has somewhat better official PoseBusters pass-all validity, but loses
  Astex oracle/joint coverage. The paired uncertainty does not support replacing
  N100/S10 as the general default.

The frozen guidance hypothesis is **not supported**. The preregistered target
was at least `+2 percentage points` in
`any(fast-valid candidate and RMSD <2 Å)`. Observed guided-minus-unguided
effects were `0 pp` in five cells and `+1.06 pp` in PoseBusters N40/S25.

The safety guard passed: oracle `<2 Å` never decreased, all 96 sampling shards
and 96 official-validity shards completed without a failed complex, non-finite
guidance evaluations were zero, and the maximum accepted atom displacement was
`0.02237 Å` under the frozen `0.25 Å` limit.

## Scope

Guidance chemistry fails closed. Results below are for the pre-outcome
supported cohorts, not the full benchmark:

| Dataset | Supported | Full set | Eligibility |
|---|---:|---:|---:|
| Astex Diverse | 36 | 85 | 42.35% |
| PoseBusters v2 | 94 | 308 | 30.52% |

The exact eligible IDs and exclusion reasons are in
`docs/GUIDANCE_BUDGET1000_ELIGIBILITY.json`. Percentages must not be presented
as full-set results.

## Astex Diverse supported cohort

`PB valid` means PoseBusters 0.6.5 `redock` pass-all checks applied to the saved
Astex RMSD-oracle pose; it is not a PoseBusters-v2 benchmark result.

| Budget | Oracle <2 Å U→G | Oracle median U→G (Å) | Joint valid & <2 Å U→G | PB valid U→G |
|---|---:|---:|---:|---:|
| N100/S10 | 91.67→91.67 | 0.7621→0.7617 | 91.67→91.67 | 61.11→61.11 |
| N50/S20 | 91.67→91.67 | 0.7844→0.7833 | 86.11→86.11 | 66.67→66.67 |
| N40/S25 | 88.89→88.89 | 0.8014→0.8010 | 80.56→80.56 | 66.67→66.67 |

Guided budget contrasts against N100/S10:

- N50/S20: oracle `<2 Å` `0.00 pp` (95% CI `[0.00, 0.00]`);
  joint `-5.56 pp` (`[-13.89, 0.00]`).
- N40/S25: oracle `<2 Å` `-2.78 pp` (`[-8.33, 0.00]`);
  joint `-11.11 pp` (`[-22.22, -2.78]`).
- Official pass-all validity is `+5.56 pp` for both N50/S20 and N40/S25
  versus N100/S10, but both CIs are `[-11.11, 22.22]`.

## PoseBusters v2 supported cohort

| Budget | Oracle <2 Å U→G | Oracle median U→G (Å) | Joint valid & <2 Å U→G | PB valid U→G |
|---|---:|---:|---:|---:|
| N100/S10 | 95.74→95.74 | 0.8715→0.8667 | 84.04→84.04 | 50.00→51.06 |
| N50/S20 | 94.68→94.68 | 0.8386→0.8370 | 81.91→81.91 | 50.00→50.00 |
| N40/S25 | 95.74→95.74 | 0.8412→0.8393 | 82.98→84.04 | 53.19→54.26 |

Guidance effects:

- N100/S10 official pass-all: `+1.06 pp` (one invalid→valid transition;
  95% CI `[0.00, 3.19]`); oracle/joint thresholds are unchanged.
- N50/S20: no oracle, joint, or official pass-all change.
- N40/S25 joint and official pass-all: each `+1.06 pp`
  (`[0.00, 3.19]`); oracle `<2 Å` is unchanged.

Guided budget contrasts against N100/S10:

- N50/S20: oracle `<2 Å` `-1.06 pp` (`[-4.26, 2.13]`);
  joint `-2.13 pp` (`[-6.38, 2.13]`).
- N40/S25: oracle `<2 Å` `0.00 pp` (`[-3.19, 3.19]`);
  joint `0.00 pp` (`[-6.38, 6.38]`);
  official pass-all `+3.19 pp` (`[-5.32, 11.70]`).

## Runtime and numerical diagnostics

- Sampling Slurm array: job `44981`; every task completed with exit code `0`.
- Sampling summaries: `96/96`; eligible-complex failures: `0`.
- Guidance pose corrections: `311,411/312,000` accepted (`99.81%`);
  `589` safely rejected; `7,052` backtracks.
- Non-finite base poses/trials: `0/0`.
- Maximum accepted atom displacement: `0.022372 Å`.
- Peak CUDA allocated/reserved: `13.05/46.85 GiB`; no CUDA OOM.
- Official PoseBusters: `96/96` shards, `780/780` saved oracle poses,
  failures `0`; every Slurm task exited `0`.
- Bootstrap: paired by exact complex ID, 10,000 resamples, seed `20260731`.
- All six cells per dataset use the same per-complex sampling seed and exact
  100-pose prior-pool hash. N50 and N40 use nested prefixes.

Energy-descent acceptance is evaluated with the frozen absolute/relative
tolerance, not as exact floating-point monotonicity.

## Artifacts

- Strict sampling report:
  `outputs/benchmarks/guidance_budget1000_v1/guidance_budget1000_report.json`
- Strict official-validity report:
  `outputs/benchmarks/guidance_budget1000_v1/guidance_budget1000_posebusters_report.json`
- Raw sampling CSV/JSON and saved poses:
  `outputs/benchmarks/guidance_budget1000_v1/raw/`
- Official PoseBusters shard outputs:
  `outputs/benchmarks/guidance_budget1000_v1/posebusters_official/`

These reports cover complex-level sampling uncertainty only. They use one
frozen sampling seed and do not estimate seed-to-seed variance. The benchmark
outcomes must not be used to tune guidance coefficients; any tuning should use
a disjoint development cohort and be re-evaluated on a new holdout.
