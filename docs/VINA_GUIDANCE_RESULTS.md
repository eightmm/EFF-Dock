# Vina-guided Sampling Results

Protocol: `EFFDOCK-VINA-GUIDANCE-V1`

Date: 2026-07-19

## Outcome

The frozen Vina+DG guidance run improved official PoseBusters pass-all from
169/308 (54.87%) to 173/308 (56.17%), a paired change of +1.30 percentage
points. Six baseline-invalid selected poses became valid and two
baseline-valid poses became invalid. The exact McNemar p-value is 0.289 and the
paired bootstrap 95% interval for the change is [-0.32, +3.25] points.

The pre-registered +3-point validity target was not met. The accuracy guardrail
was met: frozen-composite selected RMSD<2A remained 72.73%, and oracle-80
remained 94.81%. This is weak positive evidence, not a confirmed improvement.

| Metric | Unguided | Vina+DG guided | Change |
|---|---:|---:|---:|
| Official PoseBusters pass-all | 54.87% (169/308) | 56.17% (173/308) | +1.30 pp |
| Frozen-composite RMSD<2A | 72.73% | 72.73% | 0.00 pp |
| Frozen-composite fast-valid | 66.88% | 67.53% | +0.65 pp |
| Oracle-80 RMSD<2A | 94.81% | 94.81% | 0.00 pp |

The official checks with the largest positive changes were minimum distance to
protein (+0.65 pp), internal energy (+0.65 pp), and internal steric clash
(+0.32 pp). Bond-angle validity decreased by 0.32 pp; other reported checks
were unchanged or smaller.

## Frozen Configuration

- Geometry/confidence checkpoints: geometry-FT step 100000 and confidence step
  42500
- Sampler: N80, S25, sigma 0.5, late schedule power 3, pocket cutoff 10A,
  per-ID seed inherited from the frozen baseline
- Guidance: scale 0.05, start_t 0.5, linear ramp, atom force cap 10,
  translation/angular caps 5, DG weight 1, receptor shell 18A
- Final selector: frozen `pair_gate_density_rank_vote_plclash_ambig`
- Official evaluator: PoseBusters 0.6.5 `redock`

## Completeness and Validation

- Sampling: 308/308 final rows. One `linalg.eigh` failure (`7zhp_iqy`) was
  preserved in the original shard and reproduced successfully with the same
  seed/config on H100.
- Official PoseBusters: 308/308, zero evaluator failures.
- CPU suite: 102 passed and 3 GPU-only skipped before the benchmark; focused
  Vina/guidance/sampler suite: 47 passed after the final scorer changes.
- Exact explicit-XS Torch kernel follows AutoDock Vina v1.2.7 coefficients,
  radii/flags, center-distance cutoff, and torsion normalization.
- End-to-end automatic PDBQT type reconstruction is not called exact: on the
  official 1iep example it gave -12.217 versus Vina 1.2.7 -12.513 (2.37%
  difference), because Vina's internal flexible-bond mobility is not fully
  present in the reconstructed typing path. Active PDB/RDKit guidance is
  therefore documented as best-effort typing over the exact score kernel.

## Artifacts

- Sampling jobs: `38791`, `38797`, `38799`; rescue `38801`
- Official PoseBusters job: `38802`
- Raw generated summaries: ignored
  `outputs/benchmarks/raw/effdock-vina-guidance-n80-s25-v1-posebusters*`
- Official reports: ignored
  `outputs/benchmarks/vina_guidance_posebusters_official/`
- Machine-readable tracked summary: `docs/VINA_GUIDANCE_RESULTS.json`

## Decision

Do not make scale 0.05 guidance the default yet. Keep it opt-in and retain the
implementation. A next experiment should choose a small scale/start-time grid
on PLINDER validation only, then reopen PoseBusters once with a newly frozen
configuration. The current PoseBusters result must not be used to choose that
grid.
