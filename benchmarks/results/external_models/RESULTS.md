# Supplied-pocket external-model comparison

Status: external-model three-repeat evaluation complete on 2026-09-01.

The locally executed AI/hybrid rows use three independent inference seeds and
report mean plus sample standard deviation. GOLD and AutoDock Vina are
deposited PoseBusters benchmark values and therefore have no local repeat
variance. EFF-Dock is the currently promoted single U70k aggregate and will be
replaced by the active production-checkpoint three-repeat result after that run
finishes.

`RMSD` is Top-1 symmetry-aware ligand heavy-atom RMSD `< 2 A`. `Joint` requires
the same selected pose to pass RMSD `< 2 A` and all official PoseBusters 0.6.5
redocking checks. Missing or evaluator-error targets remain denominator
failures.

## Astex Diverse (N=85)

| Method | Repeats | RMSD (%) | Joint (%) |
|---|---:|---:|---:|
| EFF-Dock U70k | 1 | 85.88 | 81.18 |
| SigmaDock | 3 | 90.59 ± 1.18 | 89.02 ± 1.36 |
| DiffDock-Pocket | 3 | 53.73 ± 0.68 | 31.37 ± 3.40 |
| RLDiff RL++ | 3 | 84.31 ± 2.45 | 82.35 ± 4.08 |
| DiffBindFR + MDN/EC | 3 | 80.78 ± 2.96 | 68.63 ± 3.59 |
| GOLD (reported) | 1 | 67.06 | 63.53 |
| AutoDock Vina (reported) | 1 | 57.65 | 56.47 |

## PoseBusters v2 (N=308)

| Method | Repeats | RMSD (%) | Joint (%) |
|---|---:|---:|---:|
| EFF-Dock U70k | 1 | 84.09 | 81.17 |
| SigmaDock | 3 | 78.90 ± 0.32 | 76.41 ± 0.19 |
| DiffDock-Pocket | 3 | 31.82 ± 0.65 | 14.83 ± 0.68 |
| RLDiff RL++ | 3 | 74.46 ± 1.35 | 72.94 ± 0.68 |
| DiffBindFR + MDN/EC | 3 | 54.22 ± 2.27 | 36.04 ± 2.45 |
| GOLD (reported) | 1 | 58.12 | 54.55 |
| AutoDock Vina (reported) | 1 | 59.74 | 58.12 |

All `216/216` selected-pose evaluation shards completed. DiffDock-Pocket had
one PoseBusters evaluator error in repeat 2; the corresponding target remains a
failure in the fixed denominator. The other newly evaluated local rows had no
PoseBusters evaluator errors.

The exact per-repeat values, provenance paths, and error counts are in
[`pocket_only_pb_valid_comparison.json`](pocket_only_pb_valid_comparison.json).
Comparison figures are intentionally deferred until the production EFF-Dock
three-repeat result is complete.
