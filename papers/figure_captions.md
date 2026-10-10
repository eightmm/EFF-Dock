# Figure captions

Captions below are extracted from the canonical LaTeX manuscript. Keep the LaTeX mathematics and labels when copying into Prism.

## Main figures

### Figure 1

- [Fig1.pdf](assets/Fig1.pdf)

```latex
\caption{\textbf{Overview of the \method pipeline, illustrated on Astex complex 1T46.} Input: a prepared receptor, the ligand and a supplied pocket center; the ligand is cut into rigid fragments (colors). Pose generation: fragments move along the learned \(\SE(3)\) flow from a random prior (\(t=0\)) to a predicted pose (\(t=1\)); one prespecified trajectory (seed 42) from the supplied conformer is shown, whereas the benchmark bank uses a conformer generated from SMILES. Pose refinement: bounded descent on the geometric interaction energy for the same pose. Confidence ranking: candidates from the separately generated 100-pose benchmark bank are ordered by predicted RMSD (pRMSD), and the top-ranked candidate (check mark) is selected. RMSD values and crystal overlays (orange) are retrospective and are not available to the method; the 1.36\,\angstrom{} selected-pose RMSD belongs to the ranked bank, not to the illustrated trajectory.}
\label{fig:workflow-overview}
```

### Figure 2

- [Fig2.pdf](assets/Fig2.pdf)

```latex
\caption{\textbf{Docking and confidence architectures.} (A) The docking network applies six conditioned equivariant interaction layers and converts ligand-atom features to fragment translation and angular velocities with the Newton--Euler readout. For each candidate, terminal docking features are recomputed at \(t=1\) using the translation-prior standard deviation used to generate it. The confidence network combines these with a new embedding, applies four interaction layers with zero conditioning vector \(c=0\), converts the resulting features into rotation-invariant quantities, and predicts pose-level RMSD and success scores from global and contact-aware summaries. (B) Each interaction layer contains normalization, equivariant convolution, post-convolution transformation, a residual connection, and adaptive layer normalization. (C) The equivariant convolution uses radial and gating networks, shared tensor products through \(\ell\le2\), distance decay, and gate-normalized aggregation with invariant nonscalar rescaling. (D) Scalar and nonscalar channels use type-appropriate adaptive normalization and equivariant activation. Detailed tensor definitions are provided in Supplementary Information, Sec.~S3.}
\label{fig:model-architecture}
```

### Figure 3

- [model_comparison.pdf](assets/model_comparison.pdf)

```latex
\caption{\textbf{External docking performance on supplied-pocket benchmarks.} (A) Astex Diverse Set (85 complexes) and (B) PoseBusters v2 (308 complexes). Solid bars show RMSD \(<2\,\angstrom\) and PoseBusters success; hatched segments complete RMSD-only success. Local results are means with sample SD across three repeats; EFF-Dock repeats share one starting conformer, so their SD excludes conformer variation. EFF-Dock uses 100 poses, 10 sampling steps, refinement, and chirality-filtered confidence selection. GOLD and AutoDock Vina are literature values; the remaining methods use their native sampling and ranking protocols. DiffDock-Pocket uses a holo-aligned predicted receptor; DiffBindFR samples pocket side-chain torsions on a fixed backbone and uses cognate-ligand coordinates in its upstream scoring graph; its selected ligand poses are evaluated against the common frozen holo receptor. Budgets, receptor preparation, selectors and evaluation versions differ, so these are protocol-conditioned comparisons.}
\label{fig:benchmark-comparison}
```

### Figure 4

- [refinement_chirality.pdf](assets/refinement_chirality.pdf)

```latex
\caption{\textbf{Effects of refinement and chirality-aware selection.} \method results are shown for Astex, PoseBusters v2, PhiBench-derived, FoldBench, and OpenBind using the frozen released weights and unguided 100-pose, 10-step sampling. Raw and refined banks are ranked with or without the chirality filter. Solid bars show RMSD and PoseBusters success; hatched segments complete RMSD-only success. Bars and error bars are the mean and sample SD across three repeats on fixed cohorts; for Astex and PoseBusters v2 the repeats share one starting conformer, whereas for the other cohorts the conformer varies with the repeat.}
\label{fig:refinement-selection}
```

### Figure 5

- [confidence_diagnostics.pdf](assets/confidence_diagnostics.pdf)

```latex
\caption{\textbf{Confidence ranking and selection headroom.} (A) Within-complex Spearman correlation between predicted and observed RMSD for raw and refined banks. (B) Refined-bank AUROC and average precision for RMSD \(<2\,\angstrom\) labels, averaged over banks; banks containing only one class are excluded. (C) Partition of refined primary outcomes: no RMSD-successful candidate in the bank; all such candidates excluded by the chirality screen; an eligible RMSD-successful candidate not selected (ranking failure); the selected RMSD-successful pose fails PoseBusters; or RMSD--PoseBusters success. Candidate success in C is defined by RMSD alone; the corresponding partition with all-candidate PoseBusters labels for Astex and PoseBusters v2 is given in Supplementary Table~S21. (D) For Astex and PoseBusters v2, selected-pose RMSD--PoseBusters success compared with the joint oracle (any candidate with RMSD \(<2\,\angstrom\) passing all PoseBusters checks) and the RMSD-only oracle, from official checks on every refined candidate. Bars show three-repeat means and sample SD.}
\label{fig:confidence-diagnostics}
```

### Figure 6

- [pocket_prior.pdf](assets/pocket_prior.pdf)

```latex
\caption{\textbf{Sensitivity to supplied-pocket radius, center error and translation-prior scale.} (A,B) Joint selected-pose RMSD \(<2\,\angstrom\) and PoseBusters success across generation crop radii at prior scale \(2\,\angstrom\), for Astex Diverse Set (85 complexes) and PoseBusters v2 (308 complexes). (C,D) Prior-scale dependence at a \(10\,\angstrom\) generation crop on the same cohorts. Rows vary Gaussian center error, with standard deviation reported per Cartesian axis. Cells show three-inference-seed means on a linear color scale from 50\% to 100\%, with the neutral color at 75\% and darker blue below 50\%. All conditions use unguided 100-pose, ten-step generation, energy refinement and chirality-filtered confidence ranking. Refinement and ranking retain the original center, the refinement energy shell and the \(10\,\angstrom\) scoring crop. Terminal docking features used for confidence are recomputed with the translation-prior scale used to generate each candidate. The grid isolates the crop and prior axes rather than crossing every pair. Downstream spatial context is fixed; prior-scale effects include sampling and terminal feature extraction.}
\label{fig:pocket-sensitivity}
```

## Supplementary figures

### Figure S1

- [ligand_complexity.pdf](assets/ligand_complexity.pdf)

```latex
\caption{\textbf{Docking performance across ligand-complexity strata.} \method results use refined candidates and chirality-filtered selection. The five benchmarks are grouped by (A) ligand heavy-atom count, (B) strict rotatable-bond count, and (C) saved rigid-fragment count. Top-1 denotes RMSD \(<2\,\angstrom\); Top-1 + PB-valid requires the same selected pose to pass PoseBusters; Oracle searches all 100 refined candidates using RMSD only. Points show mean \(\pm\) sample SD across three repeats. Labels give unique-complex counts, and asterisks mark bins with fewer than five complexes.}
\label{fig:si-complexity-performance}
```

### Figure S2

- [cumulative_success.pdf](assets/cumulative_success.pdf)

```latex
\caption{\textbf{Cumulative pose success under confidence ranking and generation order.} \method results are shown for (A) the first \(k\) candidates after unfiltered confidence ranking of the complete bank and (B) the first \(N\) candidates in saved generation order without confidence sorting. Solid and dashed lines denote refined and raw banks; bands show \(\pm1\) sample SD across three repeats. Insets expand candidates 1--20. Both orderings reach the full-bank RMSD-only oracle at 100 poses; PoseBusters validity is not included.}
\label{fig:si-pose-order}
```

### Figure S3

- [pose_budget.pdf](assets/pose_budget.pdf)

```latex
\caption{\textbf{Selected-pose success as a function of available candidate count.} The first \(N\) refined candidates are retained from each \method 100-pose bank. Top-1 reselects the minimum predicted-RMSD pose, Top-1 + chirality applies the input-chirality mask with the production fallback, and Oracle tests whether the prefix contains any RMSD-successful pose. Lines and bands show mean \(\pm1\) sample SD across three repeats. The curves reuse saved prefixes and are not independently executed smaller campaigns or runtime measurements.}
\label{fig:si-candidate-prefix}
```

### Figure S4

- [historical_runtime.pdf](assets/historical_runtime.pdf)

```latex
\caption{\textbf{Initial candidate-acquisition pipeline runtime.} Mean wall time per complex for the unguided 100-pose, 10-step evaluation. Runtime covers setup, generation, original refinement and confidence scoring, and I/O. It excludes official PoseBusters evaluation and subsequent refinement and rescoring passes, so it is not the end-to-end runtime of the primary results. Whiskers show sample SD across three fixed-weight inference repeats. Mixed devices and cohort-specific process profiles preclude a controlled speed comparison.}
\label{fig:si-runtime}
```

### Figure S5

- [generation_memory.pdf](assets/generation_memory.pdf)

```latex
\caption{\textbf{Initial pose-generation process memory.} Maximum CUDA allocated-memory peak across three repeats of the unguided 100-pose, 10-step evaluation. This is the original generation/evaluation process allocator peak, not reserved memory or a whole-pipeline peak. Astex and PoseBusters used candidate-only generation processes, whereas PhiBench-derived, FoldBench and OpenBind processes included initial confidence scoring. Separate post-generation refinement and rescoring processes are excluded. Mixed devices and these different process profiles preclude a controlled comparison of sampling-only memory.}
\label{fig:si-generation-memory}
```

### Figure S6

- [training_relatedness.pdf](assets/training_relatedness.pdf)

```latex
\caption{\textbf{Sequence and ligand relatedness to the docking-training set.} The five external cohorts and 1,076-complex docking-validation set are compared with 47,277 executed docking-training samples. (A) Maximum query-normalized binding-chain sequence identity. (B) Joint ligand and sequence overlap classified by whether the matches occur in the same training sample. (C) The same-sample overlap fraction with counts. Ligand-binding chains use protein--ligand heavy-atom contacts within \(6\,\angstrom\).}
\label{fig:si-relatedness-overview}
```

### Figure S7

- [ligand_similarity.pdf](assets/ligand_similarity.pdf)

```latex
\caption{\textbf{Training-ligand relatedness and stratified docking performance.} (A) Exact heavy-isomeric ligand identity and nearest training-ligand Morgan-fingerprint Tanimoto similarity relative to the 47,277 executed docking-training samples. (B) \method RMSD--PoseBusters success with and without an observed exact ligand match. Bars show mean \(\pm\) sample SD across three repeats; group differences are descriptive and confounded by cohort composition.}
\label{fig:si-relatedness-ligand}
```

### Figure S8

- [sequence_similarity.pdf](assets/sequence_similarity.pdf)

```latex
\caption{\textbf{Docking performance by training-set sequence identity.} Strata use maximum query-normalized binding-chain identity to the docking-training set. Solid bars show RMSD--PoseBusters success; hatching extends to RMSD-only success. Bars and whiskers show three-repeat means and sample SD; labels give stratum sizes, including empty and sparse groups.}
\label{fig:si-relatedness-sequence}
```

### Figure S9

- [paired_uncertainty.pdf](assets/paired_uncertainty.pdf)

```latex
\caption{\textbf{Paired changes in docking success with refinement and chirality filtering.} \method results compare refined versus raw candidates with and without the chirality filter and filter off versus on at each stage. The endpoint is RMSD \(<2\,\angstrom\) and PoseBusters success. Per-complex paired differences are averaged across seeds before 2,000-draw percentile bootstrapping. Whiskers show complex and exact-PDB-cluster 95\% intervals; OpenBind lacks a cluster interval.}
\label{fig:si-paired-effects}
```

### Figure S10

- [candidate_ranking.pdf](assets/candidate_ranking.pdf)

```latex
\caption{\textbf{Confidence selection and near-native candidate density.} (A) Top-1, top-5, and oracle-100 RMSD success for refined unguided banks. (B) Fraction of individual refined candidates with RMSD \(<2\,\angstrom\). Bars show mean \(\pm\) sample SD across three repeats. Panel A is measured per complex, whereas panel B is measured per candidate.}
\label{fig:si-selection-density}
```

### Figure S11

- [complexity_failures.pdf](assets/complexity_failures.pdf)

```latex
\caption{\textbf{Descriptive failure decomposition across ligand-complexity strata.} \method, refined, chirality-filtered outcomes are partitioned into no RMSD-successful candidate, an available successful candidate not selected, selected RMSD success with PoseBusters failure, or RMSD--PoseBusters success. The fractions are the algebraic decomposition \(100-O\), \(O-T\), \(T-J\), and \(J\) of oracle coverage \(O\), selected RMSD success \(T\), and RMSD--PoseBusters success \(J\). The \(100-O\) category denotes absence of an RMSD-successful candidate after refinement. The \(O-T\) category combines chirality exclusion and ranking loss, which main Fig.~5C separates.}
\label{fig:si-complexity-failures}
```

### Figure S12

- [confidence_reliability.pdf](assets/confidence_reliability.pdf)

```latex
\caption{\textbf{Reliability of saved confidence predictions and candidate-density dependence.} \method refined candidates are pooled across three repeats. (A) Empirical near-native fraction versus the saved success-head probability. (B) Mean observed versus predicted RMSD in fixed predicted-RMSD bins. No recalibration is fitted. (C) Primary chirality-filtered top-1 success versus the number of near-native candidates in the 100-pose bank. Candidate observations within complexes are correlated, and the production selector uses predicted RMSD rather than the auxiliary probability head.}
\label{fig:si-confidence-reliability}
```

### Figure S13

- [stringent_subset.pdf](assets/stringent_subset.pdf)

```latex
\caption{\textbf{Performance under simultaneous sequence and ligand restrictions.} The fixed subset requires maximum training-relative binding-chain identity \(<30\%\), Morgan Tanimoto \(<0.5\), and no observed exact training-ligand match. Selected RMSD and RMSD--PoseBusters success and full-bank RMSD oracle are shown for EFF-Dock. Remaining counts are 2, 4, 7, 9 and 0 for Astex, PoseBusters, PhiBench-derived, FoldBench and OpenBind, respectively; the slices are descriptive.}
\label{fig:si-stringent-slices}
```

### Figure S14

- [baseline_uncertainty.pdf](assets/baseline_uncertainty.pdf)

```latex
\caption{\textbf{Paired uncertainty relative to locally executed baselines.} \method minus baseline percentage-point differences are shown for RMSD success and same-selected-pose RMSD--PoseBusters success on (A) Astex and (B) PoseBusters. Intervals are 2,000-draw percentile 95\% paired bootstrap intervals over exact PDB-accession groups; every retained complex has a distinct accession in these cohorts. Literature-only rows are excluded. Positive values favor EFF-Dock, but differing budgets and native selectors prevent an equal-compute superiority claim.}
\label{fig:si-baseline-differences}
```

### Figure S15

- [fragment_trajectory.pdf](assets/fragment_trajectory.pdf)

```latex
\caption{\textbf{Fragment motion along a saved ODE trajectory.} Astex 1T46--STI at five saved flow times in a fixed receptor frame. Colors track six rigid fragments; omission of early interfragment bonds is a display convention. Dimensionless flow time describes generation, separate from refinement. This prespecified seed-42, unguided one-pose illustration starts from the supplied conformer and is separate from the 100-pose benchmark banks, which use conformers generated from SMILES.}
\label{fig:si-generation-trajectory}
```

### Figure S16

- [physical_validity.pdf](assets/physical_validity.pdf)

```latex
\caption{\textbf{Physical-validity failures before and after refinement.} (A,B) Heatmaps report failure percentages among the primary selected raw and refined poses; the selected candidate may differ between stages. (C,D) Fail-to-pass and pass-to-fail percentages use candidate indices with saved official PoseBusters evaluations at both stages. Rows summarize bond geometry, internal clash, receptor clash, stereochemistry, ring planarity, internal energy, and all 27 non-RMSD checks. Categories overlap, and the matched subset is selection biased.}
\label{fig:pb-failure-profiles}
```

### Figure S17

- [structure_examples.pdf](assets/structure_examples.pdf)
- [structure_examples_continued.pdf](assets/structure_examples_continued.pdf)

```latex
\caption{\textbf{Dataset-specific examples of success, refinement rescue, and selection failure.} (A--C) Astex Diverse Set, PoseBusters v2 and PhiBench-derived; (D,E) FoldBench and OpenBind appear on the continuation page. Rescue panels show the largest same-candidate RMSD decrease meeting the stated success and validity criteria in each dataset. Successful and selection-failure cases are deterministic lower-median examples from the remaining qualifying records. Rescue panels therefore show extremes rather than typical effects. Coordinates remain in the receptor frame without independent ligand alignment. A failed primary selection is compared with the minimum-RMSD pose from the full saved bank; this comparator is not required to pass the chirality screen or PoseBusters and does not by itself isolate a confidence-ranking failure.}
\label{fig:representative-cases}
```

### Figure S18

- [selection_regret.pdf](assets/selection_regret.pdf)

```latex
\caption{\textbf{Selection regret and conditional top-1 success.} \method results use the primary chirality-filtered confidence selector. (A) Median selected-minus-full-bank-oracle RMSD regret for raw and refined banks. (B) Selected top-1 RMSD success among complexes whose full bank contains an RMSD-successful candidate. Losses in A and B can include chirality-eligibility exclusions. (C) Cumulative refined selected-pose RMSD regret relative to the minimum RMSD within the same effective eligible set, including the unfiltered fallback when no candidate passes the mask. The corresponding outcome partition is shown in main Fig.~5C. Reference RMSD is used only for retrospective analysis. Bars show three-repeat means and sample SD.}
\label{fig:selection-bottlenecks}
```

### Figure S19

- [guidance_budget.pdf](assets/guidance_budget.pdf)

```latex
\caption{\textbf{Guidance and candidate allocation at a fixed learned pose-step count.} (A) Astex Diverse Set (85 complexes) and (B) PoseBusters v2 (308 complexes). Each 100-pose/ten-step or 40-pose/25-step condition uses 1,000 learned pose-steps, with unguided generation or normalized energy guidance of strength \(\eta=2\). Raw and refined banks use the input-chirality-filtered minimum-predicted-RMSD selector, with an unfiltered fallback. Solid segments show RMSD and PoseBusters success; hatched extensions show RMSD-successful but PB-invalid outcomes. Bars and whiskers show the mean and sample SD over three inference seeds. Guided and unguided initial draws are paired within each budget; the two budgets are separately executed. Equal pose-step counts are not equal-runtime comparisons.}
\label{fig:si-guidance-budget}
```

### Figure S20

- [saved_endpoint_analysis.pdf](assets/saved_endpoint_analysis.pdf)

```latex
\caption{\textbf{Benchmark-specific evaluation of saved candidates.} Supplementary endpoint analyses of saved EFF-Dock candidates. (A) Top-1 and Top-5 fixed-receptor symmetry-corrected RMSD success (\(<2\,\angstrom\)), with and without all 27 non-RMSD PoseBusters checks, in the locally selected PhiBench-derived cohort. This is not asserted to reproduce the native PhysDock PAL-RMSD, cohort or 18-check conjunction. (B) Raw and refined Top-1 BiSyRMSD (\(<2\,\angstrom\)), LDDT-PLI (\(>0.8\)), and their same-pose conjunction on all 558 FoldBench interfaces using OpenStructure 2.8.0. The supplied holo receptor makes this a redocking evaluation rather than the source cofolding task. (C) Top-1/5/25 on the official 802 OpenBind follow-on IDs, using OpenStructure 2.11.1 BiSyRMSD \(\le2\,\angstrom\), PoseBusters 0.6.5, and the additional same-pose LDDT-PLI \(\ge0.8\) criterion. A and C use refined banks and the frozen chirality-filtered confidence selector, including its no-eligible-candidate fallback. Hatching denotes RMSD-successful but PB-invalid cases. 40-prefix and 25-prefix restrict the original saved N100 banks before selection; they do not represent fresh smaller-budget generation or equal compute. Numbers in A and C give RMSD--PoseBusters success; numbers in B give the corresponding criterion. Bars and error lines show means and sample standard deviations across three fixed-weight inference repeats. Rates retain the entire cohort denominator; endpoint-specific determinate denominators and unresolved counts are provided in the source data. Native preparation differences remain.}
\label{fig:si-native-endpoints}
```

### Figure S21

- [readout_ablation.pdf](assets/readout_ablation.pdf)

```latex
\caption{\textbf{Matched readout ablation.} Two docking models trained from scratch for 100,000 updates with a global batch of 16 complexes differ only in the output head: the Newton--Euler least-squares readout of atom vectors or a direct fragment readout. (A) Internal-validation rollout success (single sample, 20 steps, 1,076 complexes) every 10,000 updates using the EMA weights. (B,C) Unrefined 100-pose, ten-step banks on Astex Diverse Set and PoseBusters v2 at the terminal checkpoint: RMSD \(<2\,\angstrom\) oracle coverage among the first 10 and all 100 generated candidates and the mean fraction of near-native candidates. One training seed per arm; no refinement or confidence ranking.}
\label{fig:readout-ablation}
```
