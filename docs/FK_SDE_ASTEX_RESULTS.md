# FK Translation-SDE Astex Results

- Protocol: `EFFDOCK-FK-TRANSLATION-SDE-ASTEX-V1`
- Completed: 2026-08-12
- Status: complete paired descriptive external evaluation; hypothesis not
  supported; no production admission
- Claim boundary: Astex had already been opened, so these results cannot tune
  a setting or admit a method
- Machine-readable report:
  `outputs/benchmarks/fk_sde_astex_runs/20260812T071037Z/report.json`

## Integrity

All four arms completed all 85 Astex complexes. Every arm used exactly
`N40/S25 = 1,000` model pose-steps per complex. The audit verified identical
per-complex sampling seeds and prior-pool hashes across arms, exact all-pose
SDF hashes and 40-record counts, and 255 complete finite FK events per FK arm.
The technical smoke additionally verified that no-jitter FK-ODE outputs were
a numerical subset of ODE trajectories (maximum atom error `0.0001 A`) and
that every duplicated FK-SDE parent group produced distinct descendants.
The completed capsule's FK-ODE summaries inherited an imprecise legacy
`heuristic_not_marginal_preserving_sde` classification label even though both
jitter fields were exactly zero; the zero fields and subset audit establish
that no mutation ran. The label was corrected after the run without changing
sampling code, outputs, or metrics.

## Results

| Arm | Selected `<2 A` | Oracle `<2 A` | Selected median | Selected fast-valid | Fast-valid and `<2 A` | Mean valid candidates | Terminal unique fraction |
|---|---:|---:|---:|---:|---:|---:|---:|
| ODE | 68/85 (80.00%) | 78/85 (91.76%) | 1.203 A | 71.76% | 61.18% | 15.13 | 1.000 |
| SDE | 67/85 (78.82%) | 79/85 (92.94%) | 1.171 A | 78.82% | 62.35% | 15.15 | 1.000 |
| FK-ODE | 66/85 (77.65%) | 76/85 (89.41%) | 1.207 A | 75.29% | 63.53% | 18.08 | 0.531 |
| FK-SDE | 62/85 (72.94%) | 76/85 (89.41%) | 1.200 A | 80.00% | 63.53% | 18.56 | 1.000 |

The predeclared primary contrast, FK-SDE minus FK-ODE, was:

- confidence-selected `<2 A`: `-4/85`, or `-4.71` percentage points;
- paired selected-RMSD median difference: `+0.011 A`;
- oracle `<2 A`: `0/85`, or `0.00` percentage points;
- selected fast-valid: `+4/85`, or `+4.71` percentage points;
- selected fast-valid and `<2 A`: `0/85`, or `0.00` percentage points;
- terminal unique-coordinate fraction: `+0.469` (`0.531 -> 1.000`).

There were three complexes that gained selected `<2 A` success and seven that
lost it. FK weighting itself remained nearly unchanged: median ESS fraction
was `0.814` for FK-ODE and `0.815` for FK-SDE, while mean final unique initial
ancestor fraction was `0.523` and `0.528`, respectively. Brownian translation
dynamics therefore diversified cloned descendants without materially changing
the resampling genealogy.

The contextual SDE-minus-ODE contrast was `-1.18` percentage points for
selected `<2 A`, `+1.18` percentage points for oracle `<2 A`, and `+7.06`
percentage points for selected fast-valid poses.

## Decision

The predeclared hypothesis is not supported. FK-SDE solved the clone-collapse
symptom at the coordinate level and improved fast validity, but that extra
diversity did not enrich near-native candidates or improve the frozen
confidence selector. The current FK constraint potential also reduced oracle
and selected success relative to the matched non-FK controls, so terminal
diversity alone is not the limiting factor.

The translation score-corrected SDE remains available only through its explicit
experimental evaluation flag. It is not a default sampler, a full SE(3) SDE,
or production-admitted: rotations remain on the deterministic SO(3) flow.
There will be no Astex-driven sweep of `g_0`, FK beta/times, or selectors.
Any follow-up calibration must be designed and selected on an internal held-out
PLINDER split before another frozen external evaluation.
