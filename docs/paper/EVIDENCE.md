# Historical Rw figure evidence

This document describes the earlier saved-bank figure release. Its tables and
case selections have not been refreshed for the corrected generation-reference
and confidence-frame evaluation. Use the [canonical article](../../papers/main.pdf),
[Supplementary Information](../../papers/SI.pdf) and [Source Data guide](../../papers/data/README.md)
for the current manuscript results. Retaining this document supports interpretation
and reproduction of the historical tables; its values are not current submission results.

Completed corrected Rw, frozen released docking/confidence weights, unguided N100/S10, 10 Å supplied pocket, physical/interaction energy refinement, input-chirality-filtered minimum-pRMSD selection. Seeds 42, 100042 and 200042. Every original complex is retained. Benchmark statistics are derived from the completed study without new cohort inference, training, reference-based selection or tuned thresholds. One fixed-complex Rw N1 run supplies the separate method illustration.

## Primary selected-pose performance

| Dataset | N per seed | RMSD <2 Å (%) | PB-valid (%) | RMSD <2 Å & PB-valid (%) |
|---|---:|---:|---:|---:|
| Astex Diverse Set | 85 | 83.14 ± 1.80 | 95.69 ± 0.68 | 80.00 ± 2.04 |
| PoseBusters v2 | 308 | 81.82 ± 0.56 | 95.02 ± 0.50 | 79.00 ± 0.82 |
| PhiBench | 206 | 62.94 ± 3.23 | 93.69 ± 0.00 | 60.36 ± 3.08 |
| FoldBench | 558 | 75.27 ± 0.54 | 96.54 ± 0.10 | 73.78 ± 0.63 |
| OpenBind | 925 | 53.19 ± 0.56 | 99.68 ± 0.19 | 53.19 ± 0.56 |

Values are three-seed means and sample SD on fixed cohorts. The endpoint uses the same selected pose for RMSD and all 27 non-RMSD PoseBusters 0.6.5 checks. Temporal cohorts retain their audited InChI energy-reference compatibility path; Astex/PoseBusters use native official redock.

## Frozen-weight provenance

Rw follows the local-to-world equivariance convention. Docking and confidence
weights remain frozen from legacy-operator training; this is a cross-operator
evaluation, not retraining under Rw. The current submission does not claim a
universal accuracy gain from the correction. Operator-control comparisons remain
in the versioned numerical report and correction documentation, outside the
submitted figure selection. OpenBind has one exact-PDB group, so its PDB-cluster
interval is unavailable.

## Confidence diagnostics

| Dataset | Refined Spearman | Refined AUROC | Refined average precision | AUROC eligible cases by seed | Mean of per-seed median selected-to-full-oracle regret (Å) |
|---|---:|---:|---:|---|---:|
| Astex Diverse Set | 0.699 | 0.938 | 0.820 | [83, 82, 82] | 0.229 |
| PoseBusters v2 | 0.628 | 0.916 | 0.806 | [292, 289, 290] | 0.194 |
| PhiBench | 0.529 | 0.879 | 0.673 | [182, 181, 184] | 0.359 |
| FoldBench | 0.598 | 0.903 | 0.745 | [513, 510, 510] | 0.258 |
| OpenBind | 0.507 | 0.927 | 0.605 | [817, 814, 807] | 0.309 |

Correlation and discrimination are macro-averaged within candidate banks. AUROC/AP exclude banks containing a single outcome class; eligible denominators are retained. Reliability plots describe the frozen confidence outputs and do not fit a calibration transform.

## Effective-selector failure attribution

| Dataset | No successful pose (%) | Chirality exclusion (%) | Ranking failure (%) | PB failure (%) | Success (%) | Effective-regret median by seed (Å) | Effective-regret p90 by seed (Å) |
|---|---:|---:|---:|---:|---:|---|---|
| Astex Diverse Set | 3.14 | 0.00 | 13.73 | 3.14 | 80.00 | [0.21, 0.243, 0.207] | [1.927, 1.326, 1.297] |
| PoseBusters v2 | 5.74 | 0.22 | 12.23 | 2.81 | 79.00 | [0.16, 0.202, 0.179] | [1.568, 1.549, 1.739] |
| PhiBench | 11.49 | 0.00 | 25.57 | 2.59 | 60.36 | [0.318, 0.435, 0.3] | [2.728, 2.865, 3.165] |
| FoldBench | 8.36 | 0.18 | 16.19 | 1.49 | 73.78 | [0.221, 0.275, 0.25] | [1.899, 2.26, 2.077] |
| OpenBind | 12.14 | 0.29 | 34.38 | 0.00 | 53.19 | [0.315, 0.316, 0.282] | [2.053, 2.194, 2.145] |

Effective chirality eligibility includes the frozen unfiltered fallback; PB labels refer to selected poses only.

The partition distinguishes generation coverage, chirality eligibility and confidence ranking. Effective-oracle regret measures ranking within the set the real selector was permitted to use; full-bank regret also includes losses caused by eligibility. Neither analysis evaluates PB validity for every candidate.

## Stringent relatedness subset

The prespecified conjunction remains sequence identity <30%, Morgan Tanimoto <0.5, and no observed exact ligand match. It contains 22 complexes in total (Astex 2, PoseBusters 4, PhiBench 7, FoldBench 9, OpenBind 0); this refresh changes outcomes, not similarity definitions or membership. Empty strata remain undefined.

| Dataset | Subset n | RMSD success (%) | PB-valid success (%) | Oracle (%) |
|---|---:|---:|---:|---:|
| Astex Diverse Set | 2 | 83.33 | 83.33 | 100.00 |
| PoseBusters v2 | 4 | 75.00 | 75.00 | 100.00 |
| PhiBench | 7 | 47.62 | 47.62 | 80.95 |
| FoldBench | 9 | 59.26 | 55.56 | 77.78 |
| OpenBind | 0 | — | — | — |

## Real structure examples

Fifteen Rw examples follow the existing deterministic rule: maximum same-candidate RMSD rescue within each dataset, then lower-median PB-valid success and lower-median RMSD selection failure, excluding previously selected complex IDs. Rescue cases are extremes and are not estimates of average refinement benefit. Cameras rotate the complete scene together; no coordinates or RMSDs are changed.

| Dataset | Mechanism | ID | Seed repeat | Selected RMSD (Å) | Raw same-pose / oracle RMSD (Å) |
|---|---|---|---:|---:|---:|
| Astex Diverse Set | Successful pose | `2br1` | 2 | 0.761 | — |
| Astex Diverse Set | Refinement rescue | `1hvy` | 0 | 1.973 | 2.316 |
| Astex Diverse Set | Selection failure | `1hp0` | 2 | 2.181 | 0.512 |
| PoseBusters v2 | Successful pose | `7p5t_5yg` | 0 | 0.884 | — |
| PoseBusters v2 | Refinement rescue | `7fb7_8nf` | 0 | 0.301 | 2.082 |
| PoseBusters v2 | Selection failure | `8eye_x4i` | 2 | 2.619 | 0.656 |
| PhiBench | Successful pose | `9e8k_adp_b_1` | 2 | 1.031 | — |
| PhiBench | Refinement rescue | `9fyb_nmn_a_1_b` | 0 | 1.508 | 2.164 |
| PhiBench | Selection failure | `9e8n_adp_d_1_e` | 2 | 2.271 | 0.640 |
| FoldBench | Successful pose | `7vf6-assembly1__protein-a__ligand-e__ccd-gdp` | 1 | 0.926 | — |
| FoldBench | Refinement rescue | `8bmv-assembly1__protein-a__ligand-c__ccd-urc` | 0 | 0.271 | 2.100 |
| FoldBench | Selection failure | `8ihg-assembly1__protein-c__ligand-g__ccd-2af` | 1 | 2.242 | 0.604 |
| OpenBind | Successful pose | `a71ev2a-x5608b` | 1 | 1.121 | — |
| OpenBind | Refinement rescue | `a71ev2a-x7379a` | 1 | 1.601 | 2.865 |
| OpenBind | Selection failure | `a71ev2a-x3977a` | 1 | 2.289 | 0.848 |

## Figure provenance and remaining limits

Rw updates the main comparison, postprocessing, complexity, cumulative/prefix success, ligand/sequence-stratified performance, uncertainty, confidence, stringent subset, selected PB transitions, native-baseline pairing and structure examples. Training-relatedness inputs remain identical. Legacy guidance/N40 and pocket/prior figures and the operator-comparison figure are excluded from the current 20-page submission. Runtime/memory are reaggregated from complete Rw study logs. Fig1 and the saved trajectory now use one prespecified Rw N1 run of 1T46–STI, seed 42, plus the separately generated Rw N100 repeat-0 candidate bank for confidence selection. No outcome-based case/seed selection or new benchmark generation is involved. There is no new component ablation, input-conformer experiment, confidence retraining or primary-protocol pocket robustness measurement in this saved-data refresh.

No additional all-candidate PB labels, training wall time or equal-compute baseline evidence is inferred. Relatedness strata and repeatedly inspected external results are descriptive; fixed-cohort bootstrap intervals do not resolve all target or seed dependence.

Source: `rw_report.json`, `figure_data.json`, case-level confidence/PB tables and the source hashes in the manifest.

## Applicability domain

PhiBench uses the retained reconstructed 206-complex cohort, not the complete native benchmark. FoldBench is adapted to supplied-holo-pocket redocking, not its native co-folding task. OpenBind is an auxiliary single-protease cohort evaluated through the recorded noncovalent approximation. These cohorts do not supply a blind, apo, covalent or native co-folding claim. External relatedness and precursor-exposure limitations remain as described in the captions.

## Reading the evidence

Refined confidence discrimination is high, but eligible near-native poses still lose to the selected pose in 12.23–34.38% of all complexes across cohorts. Chirality exclusion accounts for 0.00–0.29%. The effective-set diagnostic therefore directly supports residual ranking loss, while avoiding attribution of mask exclusions to ranking. These are descriptive fractions, not a component ablation or a calibrated success guarantee.

Raw/refined selected-pose validity and the matched-candidate transitions answer different questions: the former includes a changed selected index; the latter controls index correspondence only in the saved, selection-biased evaluated subset. Both improvements and harm remain in Figure S7. Figure S9 pairs Rw with the five locally executed native baselines using fixed case-level outcomes; its descriptive intervals do not establish equal-compute superiority.

## Model size and corrected-Rw cost

| Model | Parameter objects |
|---|---:|
| Docking | 5,719,970 |
| Confidence | 9,403,274 |

Figure 6 (PDF page 6) and `evidence/model_cost.json` now use the complete Rw campaign logs: 6,246 case-seed executions in 240 pipeline shards. Runtime is summed shard wall time divided by the cohort size within each seed, then averaged across three seeds. Memory is the maximum initial generation/evaluation-process CUDA allocated peak across shards and seeds, not whole-pipeline or reserved memory. Astex/PoseBusters use generation-only candidate export; PhiBench/FoldBench/OpenBind include raw-confidence scoring in the initial process. Refinement and subsequent confidence evaluation run in separate processes. Their actual timing overhead is retained; these profiles are not controlled to isolate a single stage. These logs mix RTX A5000, RTX 6000 Ada and RTX PRO 6000 backends; their device counts are retained in `rw_runtime.json`. Costs are descriptive, not a controlled hardware comparison. The reported pose throughput is amortized pipeline throughput including setup, refinement, fresh confidence and I/O, excluding separate PB evaluation. A complete interruption-aware training wall-time ledger is unavailable; step counts are not converted to elapsed time.

## Corrected-Rw sensitivity completion

The [registered sensitivity study](../RW_ROBUSTNESS_PROTOCOL.md) is complete on all 393 Astex/PoseBusters complexes and three seeds: 24 distinct conditions, 23 newly executed, and 27,117 new complex-seed runs. The completed production baseline is reused by source hash. Pocket/prior sensitivity now uses unguided, chirality-filtered selection, replacing historical guided/unfiltered sensitivity; downstream refinement/confidence remain at the original center and 10 Å crop. The separate eta-2/N40 budget controls remain descriptive. No training, default fitting or reference-based selection occurred. All 27 official PoseBusters checks and cohort denominators were verified. The prior audit reproduced every recorded initial-pose hash on the original CPU backends; translations matched and initial rotations differed by at most 0.0000161 degrees. This documented deviation from exact bitwise pairing required no new candidate generation. Full repeat values, sample SD and oracle endpoints are in [the numerical table](../../benchmarks/results/paper/rw_robustness.json). These protocol changes prevent inferring the operator effect from a comparison with the old guided figure.
