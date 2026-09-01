# Normalized direct-guidance eta sweep protocol

Protocol: `EFFDOCK-UNIFIED-GUIDANCE-ETA-SWEEP-V2`

Status: pre-registered descriptive execution protocol. No scale is selected by
this external benchmark sweep.

## Question and claim boundary

This study measures how the magnitude, direction, cap activation, numerical
stability, sampling quality, and official structural validity of normalized
direct `GuidanceEnergy` drift change as its single dimensionless strength
coefficient `eta` is increased from `0` to `0.5`.

Astex Diverse and PoseBusters v2 outcomes were opened before this study and the
retained compatibility checkpoint is not strictly external-exclusive.
Consequently, every result in this protocol is a paired descriptive ablation.
No Astex or PoseBusters value may select `eta`, tune an energy-term weight,
alter the schedule or caps, admit guidance to the production sampler, or
support an independent external-generalization claim.

The scope of this run is exactly the two complete external redocking cohorts:
Astex Diverse and PoseBusters v2. It is an all-scale descriptive comparison;
the report must show every arm and must not nominate an automatically selected
coefficient.

## Frozen intervention

The learned and guidance fields use the direct normalized coupling frozen in
`GUIDANCE_DIRECT_DRIFT_PROTOCOL.md`:

```text
A_i(v, omega) = v_f + omega_f cross (x_i - T_f)
m = RMS_i(A_i(v_model, omega_model))
g = RMS_i(A_i(v_raw, omega_raw))
u_guide = gamma * eta * ramp_bar * (m / (g + eps)) * u_raw
u_total = u_model + u_guide
```

The standard SE(3) solver applies `dt` exactly once to `u_total`. One
pose-wise `gamma <= 1` jointly enforces the frozen translation, angular, and
per-step atom-displacement caps. All energy terms, term-relative weights,
typing, force clipping, normalization, schedule, caps, checkpoint, prior,
pocket, and selection rules are held fixed. Only `eta` changes.

## Pre-registered arms and prediction

The complete grid is:

```text
0, 0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5
```

Before cap activation, the active-call mean and last-step guide/model atom-RMS
ratios are expected to be:

| eta | Active-call mean | Last active step |
|---:|---:|---:|
| 0.025 | 1.86% | 2.5% |
| 0.05 | 3.72% | 5% |
| 0.10 | 7.43% | 10% |
| 0.20 | 14.87% | 20% |
| 0.30 | 22.30% | 30% |
| 0.40 | 29.73% | 40% |
| 0.50 | 37.17% | 50% |

Prediction: the measured applied/model ratio will be monotone in `eta` until
the shared caps truncate it; cap activation will increase at large `eta`.
The performance response is deliberately not assigned a winning threshold.
All RMSD and validity values are reported, including negative and null effects.

A technical result is disconfirmed if any required complex is omitted, a
paired prior differs across arms, a guidance value becomes non-finite, a
declared cap is violated, or a scale/run identity is incomplete or duplicated.

## Frozen data, model, and sampling budget

- datasets: all audited `85` Astex Diverse and `308` PoseBusters v2 complexes;
- input identity: a fresh content-addressed audit generated after the telemetry
  implementation is frozen;
- checkpoint: `weights/effdock_geometry_ft_100k_best.pt`;
- config: `configs/train.yaml`;
- candidate budget: `N100/S10`, exactly `1,000` learned model pose-steps;
- prior: exact shared 100-pose pool, translation sigma `0.5 Angstrom`, seed
  `42 + sorted-global-ID offset`;
- time grid: late, power `3`;
- guidance schedule: start `0.5`, linear interval-average ramp;
- pocket cutoff `10 Angstrom`, center jitter `0`;
- receptor policy `geometry_only`, guidance shell `18 Angstrom`;
- force cap `20`, translation cap `5`, angular cap `5`, conservative
  guide-only atom-displacement cap `0.25 Angstrom`;
- confidence disabled, refinement `none`, RMSD oracle pose saved only for
  measurement and official PoseBusters checks.

Every scale for one complex must use the same sampling seed and exact prior
pool hash. `eta=0` is the common paired baseline for all seven positive arms.

## Required telemetry and outcomes

Telemetry is measured in the same induced atom-velocity space used for
normalization:

- model, applied-guide, and total atom-speed RMS;
- applied-guide/model RMS ratio;
- model-guide cosine and parallel contribution `(guide/model) * cosine`;
- `dt`-weighted RMS path-length proxies, explicitly not molecular trajectory
  arc length or physical-time displacement;
- raw normalization and cap scalar;
- translation, angular, displacement, any-cap, and multiple-cap trigger counts;
- active interval, ramp, scale, counts, and p05/p50/p95/p99 trace summaries;
- zero-direction, zero-reference, non-finite, CUDA memory, and failure counts.

Sampling outcomes report oracle `<2 Angstrom`, median oracle RMSD,
fast-valid-and-`RMSD<2 Angstrom`, and paired complex-ID bootstrap intervals.
Official PoseBusters `0.6.5` reports pass-all over the same 27 non-RMSD
`redock` checks for every saved RMSD-oracle pose. No survivor-only denominator
is permitted.

## Execution gates

Execution uses a fresh ignored output root and the dependency chain:

```text
fresh Astex/PoseBusters audit
  -> strict audit merge
  -> fixed-ID all-eta GPU smoke
  -> full eta x dataset x shard GPU sampling
  -> official PoseBusters checks
  -> strict aggregate reports
```

The smoke IDs are the previously exercised `1jje` (Astex) and `7b2c_tp7`
(PoseBusters v2), and all eight scales must complete for both before the full
array is released. Long or GPU work runs only through Slurm on `6000ada`.
Every downstream stage uses `afterok`; any missing input, failed complex,
non-finite value, cap violation, hash mismatch, missing shard, or duplicate
cell makes the relevant stage nonzero and prevents the report.

Raw structures, generated outputs, logs, and benchmark poses remain ignored by
Git. The protocol, code, immutable hashes, commands, Slurm IDs, and final
aggregate values are retained in versioned documentation.
