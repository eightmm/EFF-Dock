# Pocket cutoff and center-jitter sensitivity

Study ID: `EFFDOCK-POCKET-SENSITIVITY-LEGACY-V1`

Status: recovered historical matched ablation; no new external-test tuning.

## Scope and interpretation boundary

This study measures sensitivity to the receptor crop radius (`pocket_cutoff`)
and perturbation of the supplied residue-defined pocket center
(`center_jitter_sigma`). It is a redocking robustness diagnostic, not blind
pocket discovery. The benchmark center is reference-defined and the jitter is
an artificial perturbation around that center.

All rows use one fixed legacy FlowFrag stack: geometry step 100000, confidence
step 20000, N40/S25/sigma1.0, center crop, and the same selector. This makes the
rows comparable to one another, but they must not be mixed with the retained
EFF-Dock step-42500/N80/sigma0.5 headline benchmark.

Primary metric: confidence-selected top-1 symmetry-aware heavy-atom RMSD <2A.
Oracle-40 is diagnostic and measures candidate coverage, not deployable
selection.

## Result A: cutoff sweep at zero jitter

| Pocket cutoff | Astex selected <2A | PoseBusters selected <2A | Astex oracle-40 | PoseBusters oracle-40 |
|---:|---:|---:|---:|---:|
| 6A | 47.06% | 47.08% | 85.88% | 78.25% |
| 8A | 70.59% | **72.08%** | 89.41% | **91.23%** |
| 10A | **75.29%** | **72.08%** | **91.76%** | **91.23%** |
| 12A | 71.76% | 66.56% | **91.76%** | 89.94% |

The response is an inverted U rather than a monotonic gain. A 6A crop removes
useful receptor context. Expanding beyond 10A adds context but does not add
candidate coverage and can hurt ranking. The shared robust region is 8--10A;
10A remains the defensible fixed default because it is best on Astex and tied
on PoseBusters, not because the external sets were used to fit a new value.

## Result B: Astex cutoff x jitter matrix

Selected top-1 RMSD <2A (%):

| Cutoff / jitter sigma | 0A | 1A | 2A |
|---:|---:|---:|---:|
| 6A | 47.06 | 42.35 | 27.06 |
| 8A | 70.59 | 64.71 | 47.06 |
| 10A | **75.29** | **70.59** | 44.71 |
| 12A | 71.76 | 64.71 | **50.59** |

Oracle-40 RMSD <2A (%):

| Cutoff / jitter sigma | 0A | 1A | 2A |
|---:|---:|---:|---:|
| 6A | 85.88 | 81.18 | 62.35 |
| 8A | 89.41 | 87.06 | 72.94 |
| 10A | 91.76 | **92.94** | 68.24 |
| 12A | 91.76 | 88.24 | 64.71 |

One-Angstrom jitter causes a moderate selected-score drop while 2A jitter
causes a large drop at every cutoff. Oracle coverage also collapses at 2A, so
the failure is not only confidence misranking: the generator is no longer
placing enough good candidates. The 12A crop partially buffers selected top-1
at 2A jitter, but its no-jitter score is worse; this is a robustness tradeoff,
not evidence to change the default.

The matched PoseBusters jitter matrix does not exist yet. It should remain an
explicit missing experiment rather than being inferred from Astex.

## Slide placement

Use this as `Result 1 -- Pocket robustness` in a seven-slide deck:

1. Introduction: task, why fragment-level SE(3) docking, one-sentence claim.
2. Dataset and protocol: PLINDER train/validation; frozen Astex and
   PoseBusters tests; explicit oracle-pocket limitation.
3. Architecture -- generator: fragment translation/rotation flow and
   protein-ligand graph.
4. Architecture -- confidence: multi-head outputs and cluster-free top-1
   ranking/filtering.
5. Result 1 -- pocket robustness: cutoff line chart on the left and Astex
   cutoff-by-jitter heatmap on the right. Put `8--10A robust region` and
   `2A center error becomes generator-limited` as the only two callouts.
6. Result 2 -- final benchmark: retained N80 confidence results for Astex and
   PoseBusters, with pure confidence and frozen selector clearly separated.
7. Conclusion: sampling coverage, selection gap, pocket-center limitation,
   and next experiment.

Do not put all numeric tables on the slide. Keep them in backup/notes and show
only the line chart, heatmap, and two conclusions. Label slide 5 `legacy matched
ablation (N40)` so it cannot be mistaken for the final N80 benchmark.

## Next pre-registered experiment

To complete the robustness claim, run the same 4 x 3 matrix on PoseBusters and
rerun both datasets with the retained step-42500/N80 stack. Freeze cutoff values
{6, 8, 10, 12}A, jitter sigma {0, 1, 2}A, candidate count, steps, seeds, and
selector before execution. Report selected <2A, median RMSD, oracle <2A,
official PoseBusters pass-all, failures, and runtime. Do not select a cutoff on
Astex or PoseBusters; use PLINDER validation for any default change.

Machine-readable recovered values are in
`docs/POCKET_SENSITIVITY_RESULTS.json`.
