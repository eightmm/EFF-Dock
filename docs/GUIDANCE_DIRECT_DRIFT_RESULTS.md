# Normalized direct GuidanceEnergy drift results

Protocol: `EFFDOCK-UNIFIED-GUIDANCE-DIRECT-DRIFT-BUDGET1000-V1`

Completed: 2026-08-01

Status: complete, strict full-cohort paired descriptive result. This document
records values only. It does not choose a budget cell or sampler mode.

## Frozen comparison

The only arm-level change is direct normalized ODE drift:

```text
unguided: v_total = v_model
direct:   v_total = v_model + 0.1 * interval_ramp * normalized(-grad E)
```

The shared `-grad E` fragment direction is normalized pose-wise by one scalar
in induced atom-velocity RMS space. The normal SE(3) solver applies `dt` once.
The full equation and information boundary are frozen in
[`GUIDANCE_DIRECT_DRIFT_PROTOCOL.md`](GUIDANCE_DIRECT_DRIFT_PROTOCOL.md).

| Item | Frozen value |
|---|---|
| Checkpoint SHA-256 | `6932fb3ba6ebac770f714453529656a44b8f33cf15119d23c9e675d2d60b36db` |
| Config SHA-256 | `39aa62e4a48ed6f3aa4ff59345fb43a81220e2baba22edfd5beb0c4981b307ec` |
| Input manifest SHA-256 | `99f15f557644cc51c3dd1f559b0dd97dd4259c1de3e1403fb761b7c7e079f668` |
| Fresh combined audit SHA-256 | `ea895dee81364a034201902831ca60a8c6055ca195528d7a230b1b948b3adff4` |
| Guidance implementation SHA-256 | `d82e9cb7dab0951ab356af556b16a7aeac0168c026088f505ce4dd27a228d216` |
| Guidance parameter-set SHA-256 | `7851dfe3cb2f290d3fce6e3ae2e2fe1d785cd5bc2c730e6d13bbcfb67e2b6012` |
| Prior / pocket / jitter | sigma `0.5 A` / cutoff `10 A` / jitter `0 A` |
| Arms | scale `0` / direct scale `0.1` |
| Cells | N100/S10, N50/S20, N40/S25; 1,000 learned pose-steps each |
| Pairing | same global sampling seed and exact 100-pose prior-pool hash |
| Selection / refinement | confidence off / refinement none / RMSD oracle saved |

## Completion and numerical evidence

| Gate | Result |
|---|---:|
| Fresh crystal audit | Astex `85/85`; PoseBusters v2 `308/308`; failure `0` |
| Sampling coverage per arm and cell | Astex `85/85`; PoseBusters v2 `308/308` |
| Total sampled complex-arm rows | `2,358/2,358` |
| Official PoseBusters oracle poses | `2,358/2,358` |
| Direct steps attempted | `17,292` |
| Direct pose-energy/gradient evaluations | `943,200` |
| Direct pose applications | `943,200` |
| Non-finite direct poses / sampling failures | `0 / 0` |
| Prior seed/hash pairing | verified for all `393` complexes in all cells |
| Maximum CUDA allocated / reserved | `16.743 / 46.668 GiB` |

The configured guide-only caps were translation `5`, angular `5`, and
conservative per-step atom displacement `0.25 A`. Observed values at the cap
can exceed the decimal literal by less than `5e-7` from float32 arithmetic;
the strict tolerance was `1e-6`.

## Sampling values

The joint diagnostic is `any(fast-valid candidate with symmetry-aware RMSD <
2 A)`. Every arrow is `unguided -> direct`; `Delta` is direct minus unguided.
Intervals are paired complex-ID bootstrap 95% intervals with 10,000 resamples
and seed `20260731`.

### Astex Diverse, 85 complexes

| Cell | Oracle <2 A | Oracle median RMSD (A) | Joint valid and <2 A | Joint Delta, 95% CI (pp) |
|---|---:|---:|---:|---:|
| N100/S10 | 92.94 -> 92.94 | 0.8007 -> 0.7871 | 89.41 -> 91.76 | +2.35 [0.00, 5.88] |
| N50/S20 | 92.94 -> 92.94 | 0.7923 -> 0.7775 | 88.24 -> 87.06 | -1.18 [-3.53, 0.00] |
| N40/S25 | 91.76 -> 91.76 | 0.8323 -> 0.8270 | 84.71 -> 85.88 | +1.18 [0.00, 3.53] |

Oracle `<2 A` delta was `0.00 pp` with interval `[0.00, 0.00]` in all three
cells. Oracle-median RMSD deltas and intervals were:

| Cell | Delta (A), 95% CI |
|---|---:|
| N100/S10 | -0.0136 [-0.0458, 0.0246] |
| N50/S20 | -0.0148 [-0.0262, 0.0130] |
| N40/S25 | -0.0053 [-0.0274, 0.0131] |

### PoseBusters v2, 308 complexes

| Cell | Oracle <2 A | Oracle median RMSD (A) | Joint valid and <2 A | Joint Delta, 95% CI (pp) |
|---|---:|---:|---:|---:|
| N100/S10 | 92.86 -> 92.86 | 0.8715 -> 0.8637 | 82.47 -> 83.12 | +0.65 [-0.97, 2.27] |
| N50/S20 | 92.21 -> 92.21 | 0.8661 -> 0.8649 | 77.92 -> 81.17 | +3.25 [1.30, 5.19] |
| N40/S25 | 92.86 -> 92.86 | 0.8782 -> 0.8690 | 78.90 -> 80.19 | +1.30 [-0.97, 3.57] |

Oracle `<2 A` delta was `0.00 pp` with interval `[0.00, 0.00]` in all three
cells. Oracle-median RMSD deltas and intervals were:

| Cell | Delta (A), 95% CI |
|---|---:|
| N100/S10 | -0.0078 [-0.0178, 0.0121] |
| N50/S20 | -0.0012 [-0.0208, 0.0064] |
| N40/S25 | -0.0092 [-0.0260, 0.0050] |

## Official PoseBusters 0.6.5 pass-all

Pass-all requires all 27 non-RMSD `redock` checks for the saved RMSD-oracle
pose. RMSD is excluded from the validity label. Every arrow is
`unguided -> direct`.

| Dataset | Cell | Valid count | Pass-all % | Delta, 95% CI (pp) | Invalid -> valid / valid -> invalid |
|---|---|---:|---:|---:|---:|
| Astex | N100/S10 | 51 -> 52 / 85 | 60.00 -> 61.18 | +1.18 [-3.53, 5.88] | 3 / 2 |
| Astex | N50/S20 | 52 -> 51 / 85 | 61.18 -> 60.00 | -1.18 [-4.71, 2.35] | 1 / 2 |
| Astex | N40/S25 | 51 -> 49 / 85 | 60.00 -> 57.65 | -2.35 [-7.06, 2.35] | 1 / 3 |
| PoseBusters v2 | N100/S10 | 158 -> 162 / 308 | 51.30 -> 52.60 | +1.30 [-0.97, 3.90] | 9 / 5 |
| PoseBusters v2 | N50/S20 | 156 -> 168 / 308 | 50.65 -> 54.55 | +3.90 [1.95, 6.17] | 12 / 0 |
| PoseBusters v2 | N40/S25 | 161 -> 170 / 308 | 52.27 -> 55.19 | +2.92 [0.97, 5.19] | 10 / 1 |

## Direct-runtime measurements

| Dataset | Cell | Direct steps | Pose evaluations / applied | Max translation | Max angular | Max displacement (A) | Max allocated (GiB) |
|---|---|---:|---:|---:|---:|---:|---:|
| Astex | N100/S10 | 680 | 68,000 / 68,000 | 3.0351 | 5.0000 | 0.25000003 | 11.494 |
| Astex | N50/S20 | 1,360 | 68,000 / 68,000 | 2.3899 | 4.9711 | 0.17351557 | 5.663 |
| Astex | N40/S25 | 1,700 | 68,000 / 68,000 | 2.4173 | 5.0000 | 0.13785629 | 4.512 |
| PoseBusters v2 | N100/S10 | 2,464 | 246,400 / 246,400 | 2.8490 | 5.00000048 | 0.25000003 | 16.743 |
| PoseBusters v2 | N50/S20 | 4,928 | 246,400 / 246,400 | 2.8432 | 5.00000048 | 0.24999999 | 8.231 |
| PoseBusters v2 | N40/S25 | 6,160 | 246,400 / 246,400 | 2.7626 | 5.00000048 | 0.21107133 | 6.562 |

## Execution and report identity

- fresh audit: Slurm `46141`, both tasks exit `0`; merge `46145`, exit `0`;
- representative GPU smoke: `46146` (`1jje`) and `46147` (`7b2c_tp7`),
  both exit `0`, each with `800/800` direct pose applications;
- full GPU sampling: Slurm `46155`, all `48/48` tasks exit `0`;
- official validity: Slurm `46275`, all `96/96` tasks exit `0`;
- strict report: Slurm `46389`, exit `0`, empty stdout and stderr;
- strict sampling report SHA-256:
  `0ac7191796d597674ed2f7bd94ceeae97815ac4362c7630e5b71f2d3e508c2b0`;
- strict official report SHA-256:
  `8e86af62213126b21455c595d2a9b1ce0b3512e0109a33b9ea2f914835494b9f`.

Before submission, the full test gate passed `306` tests with `3` expected
skips. Focused Ruff, Python compilation, shell syntax, and `git diff --check`
passed. The repository-wide fast wrapper remained nonzero only because of a
pre-existing unrelated import-order issue in
`scripts/figures/plot_top1_oracle.py`; the skip reason is preserved in the
experiment ledger.

Generated strict reports and raw outputs remain under the ignored directory:

```text
outputs/benchmarks/guidance_direct_drift_v1/
├── aggregate.json
├── posebusters_aggregate.json
├── audit/combined.json
├── raw/
└── posebusters_official/
```

No automatic model, budget, or guidance-mode selection is made here.

## Post-hoc paired coupling comparison

For interpretation only, the frozen direct guided arm was paired by complex ID
against the earlier full-cohort operator-split guided arm. Every comparison
has the same exact `85` Astex or `308` PoseBusters IDs, sampling seed,
100-pose prior hash, protein/reference/input identity, guidance parameter hash,
and receptor-policy hash. Intervals use 10,000 paired complex-ID bootstrap
resamples with seed `20260731`. This comparison does not change either frozen
protocol or select a production default.

| Dataset | Cell | Joint valid and <2 A, direct - operator (pp) | Official pass-all, direct - operator (pp) |
|---|---|---:|---:|
| Astex | N100/S10 | +2.35 [0.00, 5.88] | +2.35 [0.00, 5.88] |
| Astex | N50/S20 | -1.18 [-3.53, 0.00] | 0.00 [-3.53, 3.53] |
| Astex | N40/S25 | 0.00 [0.00, 0.00] | -1.18 [-4.74, 2.35] |
| PoseBusters v2 | N100/S10 | +1.30 [0.32, 2.60] | +1.30 [-0.65, 3.25] |
| PoseBusters v2 | N50/S20 | +2.92 [0.97, 5.19] | +3.25 [1.30, 5.19] |
| PoseBusters v2 | N40/S25 | +0.65 [-1.30, 2.60] | +2.60 [0.65, 4.55] |

Oracle `<2 A` is identical between couplings in all six cells. Direct median
oracle RMSD is lower by `0.0011-0.0192 A` in all cells, but every paired median
interval includes zero. Thus PB validity favors direct drift across budgets,
whereas Astex remains mixed and oracle success coverage is unchanged.
