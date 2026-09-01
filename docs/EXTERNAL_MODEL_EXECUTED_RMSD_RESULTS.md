# Executed external-model RMSD results

> Superseded snapshot. Use
> [`EXTERNAL_MODEL_AUDIT_20260830.md`](EXTERNAL_MODEL_AUDIT_20260830.md) and the
> machine-readable `common_rmsd_v8` outputs. This older table predates explicit
> hydrogen handling, primary-ligand/cofactor separation, completed recovery
> jobs, and the corrected official Vina protocol. Its Vina rows are invalid as
> benchmark baselines. It also contains historical blind/full-receptor rows;
> those rows are archived diagnostics and are excluded from all current
> EFF-Dock comparisons and plots.

Snapshot: 2026-08-29 08:30 KST

These tables contain locally executed inference results, not values copied from
papers.  Every row uses the frozen Astex Diverse `N=85` or PoseBusters Benchmark
v2 `N=308` denominator.  RMSD is the same RDKit symmetry-aware heavy-atom
`CalcRMS` without alignment, with a charge-agnostic full-topology retry only.
Missing predictions and non-bijective topology failures remain failures in the
full denominator.

The tables are RMSD-only.  Official PoseBusters 0.6.5 27-check validity and
Joint (`RMSD < 2 A` and PB-valid) have not yet been run for these external-model
Top-1 selections and must not be inferred from these values.

## Astex Diverse

| Executed arm | Native candidates | RMSD-evaluable coverage | Top-1 RMSD <2 A | Oracle at available N <2 A | Strict Oracle@40 <2 A |
|---|---:|---:|---:|---:|---:|
| RLDiff native + confidence | 40 | 85/85 | 65/85 (76.47%) | 77/85 (90.59%) | 77/85 (90.59%) |
| SurfDock + MDN | 40 | 84/85 | 64/85 (75.29%) | 78/85 (91.76%) | 78/85 (91.76%) |
| DiffBindFR + MDN | 40 | 85/85 | 60/85 (70.59%) | 82/85 (96.47%) | 82/85 (96.47%) |
| SigmaDock + Vinardo | 40 independent seeds | 85/85 | 53/85 (62.35%) | 76/85 (89.41%) | 76/85 (89.41%) |
| DiffDock-Pocket + confidence | 40 | 85/85 | 44/85 (51.76%) | 57/85 (67.06%) | 57/85 (67.06%) |
| RLDiff RL++ raw + native confidence | 40 | 85/85 | 40/85 (47.06%) | 77/85 (90.59%) | 77/85 (90.59%) |
| DynamicBind + predicted LDDT rank | 30--40 | 85/85 | 40/85 (47.06%) | 61/85 (71.76%) | 36/85 (42.35%) |
| Interformer + learned energy | 20 | 80/85 | 44/85 (51.76%) | 55/85 (64.71%) | -- |
| PoseBench DiffDock + confidence | 5 | 63/85 | 37/85 (43.53%) | 53/85 (62.35%) | -- |
| FABind post-optimized | 1 | 85/85 | 31/85 (36.47%) | 31/85 (36.47%) | -- |
| PoseBench Vina, exhaustiveness 32 | 1 emitted | 85/85 | 11/85 (12.94%) | 11/85 (12.94%) | -- |

## PoseBusters Benchmark v2

| Executed arm | Native candidates | RMSD-evaluable coverage | Top-1 RMSD <2 A | Oracle at available N <2 A | Strict Oracle@40 <2 A |
|---|---:|---:|---:|---:|---:|
| SurfDock + MDN | 40 | 308/308 | 183/308 (59.42%) | 256/308 (83.12%) | 256/308 (83.12%) |
| RLDiff native + confidence | 40 | 308/308 | 166/308 (53.90%) | 241/308 (78.25%) | 241/308 (78.25%) |
| DiffBindFR + MDN, pre-recovery snapshot | 40 | 282/308 | 140/308 (45.45%) | 238/308 (77.27%) | 238/308 (77.27%) |
| DiffDock-Pocket + confidence | 40 | 306/308 | 99/308 (32.14%) | 163/308 (52.92%) | 163/308 (52.92%) |
| RLDiff RL++ raw + native confidence | 40 | 308/308 | 93/308 (30.19%) | 240/308 (77.92%) | 240/308 (77.92%) |
| DynamicBind + predicted LDDT rank | 34--40, one missing | 307/308 | 60/308 (19.48%) | 116/308 (37.66%) | 74/308 (24.03%) |
| Interformer + learned energy | 20 | 306/308 | 130/308 (42.21%) | 183/308 (59.42%) | -- |
| PoseBench DiffDock + confidence | 5 | 236/308 | 73/308 (23.70%) | 105/308 (34.09%) | -- |
| FABind post-optimized | 1 | 308/308 | 39/308 (12.66%) | 39/308 (12.66%) | -- |
| PoseBench Vina, partial through shard 9 | 1 emitted | 251/308 | 26/308 (8.44%) | 26/308 (8.44%) | -- |

SigmaDock PoseBusters is not in the result table because only 26/40 independent
seeds were complete at this snapshot.  PoseBench Vina is also explicitly
partial; its remaining CPU shards are still queued.

## Coverage interpretation

- `RMSD-evaluable coverage` requires a full heavy-atom topology comparison, not
  merely an emitted file.  For example, PoseBench DiffDock emitted more targets
  than the strict evaluator could map, and those mapping failures remain in the
  denominator.
- `Oracle at available N` is descriptive for variable-N, N20, N5, and N1 arms.
  It is not relabeled Oracle@40.
- `Strict Oracle@40` counts a target as successful only when at least 40 poses
  were evaluated and one is below 2 A.  This is why DynamicBind's strict column
  is lower than its available-pose oracle.

## Active recovery jobs

| Job | Purpose |
|---:|---|
| `60605` | DiffBindFR PB shard 5 after Python 3.9 compatibility fix |
| `60626` | one-target PoseBench DiffDock reference-SDF input gate |
| `60627` / `60628` | dependent PB/Astex DiffDock recovery if the gate passes |
| `60629` | DynamicBind `8F4J_PHO` retry with batch size 1 |
| `60422` | remaining PoseBench Vina CPU-only shards |
| `60239` / `60242` | remaining SigmaDock PB seeds and aggregate |

Machine-readable per-target rows and aggregate JSON files are under ignored
runtime root `outputs/external_models/evaluation/common_rmsd_v1/`.
