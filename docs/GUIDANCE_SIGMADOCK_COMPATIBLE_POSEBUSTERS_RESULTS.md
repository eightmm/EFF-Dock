# SigmaDock-compatible official PoseBusters results

## Evaluation identity

- Sampling: eta=2.0, sigma={0.5,1.0,2.0,3.0,4.0}, N100/S10.
- Selector: frozen primary `confidence` Top-1 only.
- Cohorts: Astex 85 and PoseBusters v2 308, exact frozen denominators.
- Checker: PoseBusters 0.6.5, `redock`, all 27 non-RMSD checks.
- Joint: the same selected pose must have symmetry-aware RMSD <2 Angstrom and
  pass all PB validity checks.
- SigmaDock-list compatibility: 26 listed non-RMSD checks. It equals the
  27-check result in every cell because `no_radicals` passed 100% throughout.
- Full official inventory: 80/80 primary shard tasks completed with exit 0.
  The pending diagnostic-selector half was cancelled before execution because
  auxiliary selector comparisons are outside this result scope.

## Astex

| sigma | RMSD <2A | PB-valid | Joint | median RMSD |
|---:|---:|---:|---:|---:|
| 0.5 | 76.5% | 65.9% | 54.1% | 1.23 |
| 1.0 | 75.3% | 65.9% | 56.5% | 1.15 |
| 2.0 | 72.9% | 68.2% | 52.9% | 1.17 |
| 3.0 | 68.2% | 65.9% | 49.4% | 1.11 |
| 4.0 | 65.9% | 64.7% | 48.2% | 1.22 |

The largest observed Astex joint value in this grid is 56.5% at sigma=1.0.
The main low-pass checks across the grid are tetrahedral chirality and minimum
distance to protein.

## PoseBusters v2

| sigma | RMSD <2A | PB-valid | Joint | median RMSD |
|---:|---:|---:|---:|---:|
| 0.5 | 72.4% | 57.5% | 47.1% | 1.21 |
| 1.0 | 76.0% | 59.1% | 48.7% | 1.20 |
| 2.0 | 75.0% | 63.0% | 53.2% | 1.25 |
| 3.0 | 70.8% | 61.0% | 48.7% | 1.35 |
| 4.0 | 66.2% | 56.8% | 42.9% | 1.41 |

The largest observed PoseBusters joint value in this grid is 53.2% at
sigma=2.0. At that cell the two lowest-pass checks are minimum distance to
protein (77.3%) and tetrahedral chirality (80.2%); internal steric clash passes
96.4%.

These are descriptive external-benchmark outcomes. They do not by themselves
admit a production sigma; the follow-up plan requires held-out validation
before a default is chosen.

## Artifacts

- Full aggregate JSON:
  `outputs/benchmarks/guidance_sigma_sweep_eta2_runs/20260809T031535Z/sigmadock_posebusters/aggregate.json`
- Human-readable full report:
  `outputs/benchmarks/guidance_sigma_sweep_eta2_runs/20260809T031535Z/sigmadock_posebusters/RESULTS.md`
- Official per-pose shard outputs:
  `outputs/benchmarks/guidance_sigma_sweep_eta2_runs/20260809T031535Z/sigmadock_posebusters/official/`
