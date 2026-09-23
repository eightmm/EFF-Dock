# Saved-data evidence for the manuscript

Updated 2026-09-23. No training or new GPU inference was performed. The original
14 figure files are preserved; six supplementary figures are appended as
pages 15–20 of the [combined PDF](paper_figures.pdf). Captions and the Prism
package use the same order. [Protocol and source tables](../../benchmarks/results/paper/evidence/README.md).

## What the evidence supports

The saved confidence scores discriminate near-native candidates, but a sizable
selection loss remains even when a near-native candidate exists. Refinement
substantially reduces selected-pose PB failures, particularly receptor clashes.
The fixed simultaneous low-sequence/low-ligand-similarity subset is too small to
support a broad generalization claim. Local-baseline comparisons include both
negative and inconclusive differences; they do not establish universal or
equal-compute superiority.

These are retrospective descriptive analyses of already evaluated cohorts.
They neither isolate architectural contributions nor close the conformer and
primary-pocket-robustness gaps identified by the review.

## Confidence diagnosis — Figures S4–S5

Refined banks; 100 candidates per complex; means over three repeats. Spearman,
AUROC and average precision are macro within-complex statistics. AUROC/AP omit
single-class banks; eligible counts are shown explicitly. Regret is the mean of
repeat-specific medians for the primary chirality-filtered selector.

| Dataset | Spearman | AUROC | Average precision | Eligible n, repeats 0/1/2 | Regret (Å) | Top-1 given oracle success (%) | Brier |
|---|---:|---:|---:|---|---:|---:|---:|
| Astex Diverse Set | 0.698 | 0.937 | 0.825 | 83 / 82 / 82 | 0.201 | 85.0 | 0.103 |
| PoseBusters v2 | 0.628 | 0.915 | 0.805 | 291 / 289 / 290 | 0.183 | 86.7 | 0.108 |
| PhiBench | 0.529 | 0.878 | 0.670 | 183 / 181 / 185 | 0.368 | 70.5 | 0.093 |
| FoldBench | 0.598 | 0.902 | 0.744 | 513 / 509 / 511 | 0.254 | 81.2 | 0.103 |
| OpenBind | 0.508 | 0.926 | 0.603 | 818 / 812 / 808 | 0.330 | 59.8 | 0.053 |

An oracle-positive bank still fails primary selection in approximately 13–40%
of cases, depending on the cohort. This supports remaining selection loss;
it does not attribute every failure to the ranking network because the
chirality mask is part of the selector. The success-probability head is an
auxiliary diagnostic; production ranking uses predicted RMSD. Reliability
curves show departures from identity, including overconfident OpenBind
probabilities. A lower Brier score across datasets is not by itself better
calibration: prevalence differs. No calibrator or new selector was fitted.

Candidate-density strata also differ in size. The following counts are
complex banks in repeats 0/1/2, not independent complexes pooled across repeats.

| Dataset | 0 near-native | 1–5 | 6–20 | 21–50 | 51–100 |
|---|---|---|---|---|---|
| Astex Diverse Set | 2/3/3 | 10/9/7 | 14/17/17 | 35/34/36 | 24/22/22 |
| PoseBusters v2 | 17/19/18 | 28/16/24 | 77/83/80 | 116/125/122 | 70/65/64 |
| PhiBench | 23/25/21 | 43/40/40 | 51/54/58 | 60/60/56 | 29/27/31 |
| FoldBench | 45/47/47 | 68/64/69 | 173/177/164 | 185/181/190 | 87/89/88 |
| OpenBind | 107/113/117 | 434/400/399 | 343/378/382 | 40/34/27 | 1/0/0 |

## Stringent simultaneous restriction — Figure S6

Maximum observed binding-chain sequence identity <30% **AND** maximum training
ligand Morgan Tanimoto <0.5 **AND** no observed exact training-ligand match.
This is the existing query-normalized sequence metric, not pocket structural
similarity. Existing reference coverage and predecessor-exposure limitations
remain. Thresholds were fixed before the new aggregation.

| Dataset | Retained / full | Top-1 RMSD success (%) | PB-valid success (%) | RMSD oracle (%) |
|---|---:|---:|---:|---:|
| Astex Diverse Set | 2 / 85 | 66.7 | 66.7 | 100.0 |
| PoseBusters v2 | 4 / 308 | 75.0 | 75.0 | 100.0 |
| PhiBench | 7 / 206 | 57.1 | 57.1 | 81.0 |
| FoldBench | 9 / 558 | 51.9 | 48.1 | 77.8 |
| OpenBind | 0 / 925 | — | — | — |

Only 22 complexes survive in total. OpenBind is empty; its performance is not
zero. Do not present these numbers as strong evidence of unseen-target
performance or loosen the thresholds to obtain a more attractive result.
Repeat SD measures generation variability on these few complexes, not sampling
uncertainty over novel proteins.

## Refinement and physical validity — Figure S7

All 27 official non-RMSD PB checks are retained. Percentages below concern PB
failure regardless of RMSD; they differ from RMSD-plus-PB success. A selected
pose may change between raw and refined stages.

| Dataset | Raw selected failure (%) | Refined selected failure (%) | Matched candidate n, repeats 0/1/2 | Matched fail→pass (%) | Matched pass→fail (%) |
|---|---:|---:|---|---:|---:|
| Astex Diverse Set | 26.27 | 4.31 | 40 / 26 / 29 | 36.23 | 0.00 |
| PoseBusters v2 | 35.93 | 4.44 | 112 / 93 / 111 | 35.81 | 0.60 |
| PhiBench | 43.69 | 5.99 | 78 / 75 / 70 | 36.28 | 0.90 |
| FoldBench | 31.54 | 3.64 | 183 / 183 / 188 | 31.57 | 0.00 |
| OpenBind | 36.94 | 0.36 | 483 / 446 / 432 | 43.06 | 0.00 |

Matching uses the same candidate index and only already evaluated candidates.
These subsets include ordinary and chirality-filtered selections, deduplicated
by candidate identity. They are selection-biased and must not be extrapolated
to all generated poses. Rescue/harm denominators include all matched pairs,
not only initially failing/passing candidates. The plotted check groups overlap.
Refinement improves clashes and geometry, while residual stereochemistry
failures remain; it is not a universal stereochemical repair claim.

## Structure examples — Figure S8

A and C retain their original lower-median selections in Astex repeat 0.
At the user's request, B uses an expanded, explicitly post-hoc extreme-example
rule: maximize same-candidate RMSD improvement among primary refined selections
in five cohorts and three repeats, requiring raw RMSD ≥2 Å, refined RMSD <2 Å
and refined PB validity. Dataset, repeat and ID break ties. The rule was fixed
before the expanded search; this is not a typical effect or prevalence estimate.

| Mechanism | Dataset / repeat | Complex | Selected refined RMSD | Comparator | Eligible examples |
|---|---|---|---:|---|---:|
| Successful pose | Astex / 0 | 1OWE | 0.80 Å | — | 66 |
| Refinement rescue | PoseBusters v2 / 0 | 7FB7–8NF | 0.30 Å | 2.08 Å (raw, same index 76) | 138 |
| Selection failure | Astex / 0 | 1TZ8 | 2.27 Å | 0.65 Å (refined oracle) | 14 |

The rescue improves by 1.7784 Å; all 27 saved non-RMSD PB checks pass after
refinement. The 138 eligible records are complex-repeat selections, not 138
independent complexes. The audit lists all eligible candidates in
`benchmarks/results/paper/evidence/rescue_candidates.json` and the 15 ledger
checksums in `rescue_search_sources.json`. No new pose generation or PB
measurement was run. Numerical aggregate results elsewhere are unchanged.

Only binding-site close-ups are shown; the full-protein overview row was
removed. Carbon colors identify poses; heteroatom colors identify elements.
Stored coordinates, displayed receptor chains, connectivity and three software
rendered captures are distributed for re-rendering. No independent ligand
superposition or contact/interaction assignment is used.

## Local-baseline paired uncertainty — Figure S9

Difference is EFF-Dock minus the baseline in **percentage points**. Table shows
PB-valid success; RMSD-only intervals are in the same source data and figure.
Three-repeat means are paired by complex, with 2,000 bootstrap resamples.

| Dataset | Baseline | Difference | 95% interval |
|---|---|---:|---:|
| Astex Diverse Set | SigmaDock | -9.80 | [-17.25, -3.53] |
| Astex Diverse Set | DiffDock-Pocket | +47.84 | [+36.86, +58.43] |
| Astex Diverse Set | RLDiff RL++ | -3.14 | [-10.59, +4.31] |
| Astex Diverse Set | DiffBindFR + MDN/EC | +10.59 | [+0.39, +20.78] |
| Astex Diverse Set | SurfDock | -6.67 | [-15.69, +2.35] |
| PoseBusters v2 | SigmaDock | +2.81 | [-2.06, +7.36] |
| PoseBusters v2 | DiffDock-Pocket | +64.39 | [+58.98, +69.70] |
| PoseBusters v2 | RLDiff RL++ | +6.28 | [+1.19, +11.47] |
| PoseBusters v2 | DiffBindFR + MDN/EC | +43.18 | [+37.55, +49.03] |
| PoseBusters v2 | SurfDock | +1.41 | [-3.68, +6.71] |

There is one exact PDB accession per complex here, so complex and PDB-group
intervals coincide. Neither resampling scheme accounts for protein-family
relationships. The intervals are descriptive, unadjusted for multiplicity,
and conditional on the three stored repeats. Baselines retain their native
budgets and selectors. EFF-Dock's Astex PB-valid result is lower than SigmaDock;
PoseBusters intervals versus SigmaDock and SurfDock include zero. Retain these
findings rather than describing a uniform advantage.

## Model size and existing throughput

Counts are actual parameter objects in CPU-instantiated release models,
excluding buffers; no inference was run for this count.

| Model | Parameters |
|---|---:|
| Docking | 5,719,970 |
| Confidence | 9,403,274 |

Existing N100/S10 timing, NVIDIA RTX 6000 Ada Generation:

| Dataset | Mean pipeline seconds / complex | Repeat SD (s) | Amortized poses / second |
|---|---:|---:|---:|
| Astex Diverse Set | 103.91 | 0.37 | 0.962 |
| PoseBusters v2 | 103.67 | 1.02 | 0.965 |
| PhiBench | 115.80 | 11.61 | 0.864 |
| FoldBench | 103.92 | 0.35 | 0.962 |
| OpenBind | 107.25 | 1.77 | 0.932 |

Throughput is 100 divided by the existing mean pipeline time, including sampling,
refinement and confidence evaluation of both banks. It is not a newly measured
saturated service rate and excludes separate official PB assessment. A complete,
interruption-aware training wall-time ledger was not audited, so total training
wall time remains unreported rather than being inferred from step counts.

## Remaining work

- **Primary unguided pocket robustness:** deferred by explicit user choice;
  the guided pocket/prior diagnostic cannot substitute for it. No new GPU job
  was submitted for this analysis.
- **Component ablation and prospective conformer sensitivity:** still require
  controlled new training/inference; these data do not resolve those claims.
- **Generality:** low-overlap support is sparse, and all predecessor-checkpoint
  exposure is not certified. This analysis does not establish leakage absence.
- **Full-bank PB transitions:** unavailable; only previously evaluated selected
  candidates are analyzed.

The reproducible checks reconcile selected labels with the unchanged original
results, recompute repeat aggregates from case records, verify strict-subset
membership and calibration denominators, and bind inputs with SHA-256 hashes.
The model and evaluation pipeline were not changed.

Verification: six focused metric tests and the fast repository check passed.
All 20 figures regenerated pixel-for-pixel from public tables in the figure-only
CPU environment. The PDF/manifest/caption/ZIP checks passed. Prism's reference
LaTeX was compiled with Tectonic; the Å unit markup was corrected for math mode
without changing any equations or scientific content.
