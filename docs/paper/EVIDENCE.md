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

All five external cohorts now have three different complex IDs, selected from
all three saved repeats. This replaces the earlier mixed-dataset triplet.
Rescue is chosen first by maximum same-candidate RMSD improvement, requiring raw
RMSD ≥2 Å, refined RMSD <2 Å and refined PB validity. Success uses the lower
median selected RMSD among PB-valid successes after excluding that complex ID.
Selection failure uses the lower median regret among selected RMSD failures with
an oracle below 2 Å, excluding both earlier complex IDs. Ties: repeat, then ID.
These are post-hoc illustrations. Median cases describe their selected pools;
rescue is explicitly extreme, not an estimate of typical improvement or frequency.

| Dataset | Example | Complete sample ID | Repeat (zero-based) | Selected RMSD (Å) | Comparator RMSD (Å) | Eligible records |
|---|---|---|---:|---:|---:|---:|
| Astex Diverse Set | Successful pose | `1n1m` | 2 | 0.754 | — | 201 |
| Astex Diverse Set | Refinement rescue | `1hvy` | 0 | 1.980 | 2.317 (raw, same candidate) | 4 |
| Astex Diverse Set | Selection failure | `1tz8` | 0 | 2.272 | 0.648 (refined oracle) | 36 |
| PoseBusters v2 | Successful pose | `7dua_hj0` | 2 | 0.875 | — | 729 |
| PoseBusters v2 | Refinement rescue | `7fb7_8nf` | 0 | 0.302 | 2.081 (raw, same candidate) | 10 |
| PoseBusters v2 | Selection failure | `8ay3_oe3` | 1 | 3.043 | 1.167 (refined oracle) | 116 |
| PhiBench | Successful pose | `9dhh_a1a4o_a_1` | 2 | 1.046 | — | 371 |
| PhiBench | Refinement rescue | `9fyb_nmn_a_1_b` | 0 | 1.509 | 2.227 (raw, same candidate) | 11 |
| PhiBench | Selection failure | `7hhm_uwd_c_1` | 0 | 2.205 | 0.530 (refined oracle) | 161 |
| FoldBench | Successful pose | `9bbh-assembly1__protein-a__ligand-e__ccd-vvp` | 2 | 0.905 | — | 1217 |
| FoldBench | Refinement rescue | `8ug3-assembly1__protein-a__ligand-c__ccd-wre` | 2 | 1.225 | 2.524 (raw, same candidate) | 34 |
| FoldBench | Selection failure | `8g7f-assembly1__protein-b__ligand-h__ccd-1jj` | 0 | 2.696 | 1.026 (refined oracle) | 287 |
| OpenBind | Successful pose | `a71ev2a-x3177a` | 1 | 1.121 | — | 1457 |
| OpenBind | Refinement rescue | `a71ev2a-x7379a` | 1 | 1.600 | 2.881 (raw, same candidate) | 79 |
| OpenBind | Selection failure | `a71ev2a-x3858b` | 2 | 2.645 | 1.207 (refined oracle) | 978 |

Eligible records are complex-repeat observations, not independent complex counts.
The same complex may qualify in several repeats, but no displayed triplet repeats
a complex ID. All successful and rescued selections pass all 27 saved non-RMSD
PB checks; an oracle is the minimum RMSD candidate and need not be PB-valid.
Astex's largest eligible rescue is only 2.317 →1.980 Å; the panel retains this
small improvement rather than substituting a stronger case from another dataset.

`rescue_candidates.json` lists the 138 eligible rescues across all five datasets;
`rescue_search_sources.json` records the 15 ledger checksums. `structures.json`
contains the full selected/comparator geometry and exact candidate indices.
No new docking or PB evaluation was run. Aggregate benchmark outcomes are unchanged.

Rows A–E identify datasets; the three columns identify mechanisms. PDB/sample
labels and RMSDs are placed inside each pocket image. FoldBench labels shorten
the full identifier to PDB, CCD ligand and ligand chain; OpenBind labels are its
sample IDs, not PDB accessions. Carbon colors identify poses, heteroatom colors
identify elements. All 15 software-rendered captures and receptor coordinates
are included, with no independent ligand alignment or assigned interactions.

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

## Saved fragment trajectory — Figure S10

Astex Diverse Set 1T46–STI, shown in a fixed receptor coordinate frame and camera. The five panels show actual saved states at t=0.000, 0.271, 0.488, 0.784 and 1.000; no coordinates are interpolated. These are the available states nearest to five equally spaced target times. Carbon colors track the same six rigid fragments throughout; nitrogen is blue and oxygen red. The pale protein cartoon provides binding-site context. Interfragment bonds are omitted in the first four panels for visibility and shown in grey at the last panel using the known ligand connectivity. This display convention does not represent chemical bond formation. Time t is the dimensionless generative-flow coordinate, not physical time or energy-refinement progress. The stored illustrative run used one sample, ten ODE steps, a late schedule with power 3, positional prior sigma 2 Å, pocket cutoff 10 Å, seed 42 and no guidance or confidence selection. The recorded checkpoint is effdock_docking_early_time_t0p10_50k.pt. This is a separate N1 illustration, not the 7FB7 refinement rescue of Figure S8 or an N100 benchmark-selected pose; no success, PB-validity or representative-trajectory claim is made. All 11 saved frames and the original fragment assignments are retained in the source data.

This additional illustration reuses an existing trace and requires no new model
inference. See `benchmarks/results/paper/trajectory/README.md` for the frame,
fragment, camera and source-integrity checks. The ODE illustration is separate from the 15 S8 structure examples; existing
benchmark outcomes are unchanged.

The trajectory artwork is intended as manuscript Figure 1; it has time labels
without panel letters. Page 21 / S10 remains its working-bundle identifier until
the manuscript figure order is finalized.
