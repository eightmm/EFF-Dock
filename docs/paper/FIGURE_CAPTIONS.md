# Manuscript figure captions

All submitted EFF-Dock measurements and illustrations use corrected Rw with frozen released weights. Training provenance remains documented in [the orientation note](../ORIENTATION_INJECTION.md). PB-valid success is solid; RMSD-successful PB-invalid portions are hatched. Page/figure IDs below are working reference IDs; manuscript numbering can be adapted.

PDF SHA-256: `287f792c8b1638998bb2a45a5151c4d1d77424e251395723ac1f6159f5cff21b`

## PDF page 1 · Figure 1

**External docking performance on two supplied-pocket benchmarks.**

EFF-Dock results use corrected Rw with frozen released weights. (A) Astex Diverse Set (85 complexes) and (B) PoseBusters v2 (308 complexes). Solid bar segments indicate selected poses with RMSD <2 Å that also pass PoseBusters; hatched extensions indicate RMSD-successful but PB-invalid poses. Total bar length therefore gives RMSD-only success. Local-run results are means over three repeats, with sample SD shown separately at the PB-valid and RMSD-only boundaries. EFF-Dock uses unguided generation with 100 poses and 10 sampling steps, energy-based refinement, and chirality-filtered confidence selection. Other methods retain their evaluated native budgets and selection procedures; SurfDock is represented by its force-optimized variant. GOLD and AutoDock Vina rows below the separator are literature-reported values, not local reruns, and have no repeat SD. Different computational budgets and PoseBusters versions prevent interpreting this as an equal-compute comparison. Numerical labels inside solid bars give PB-valid success (%); labels to the right give RMSD-only success (%), rounded to one decimal.

- 원본: [01_model_comparison.pdf](figures/01_model_comparison.pdf)
- LaTeX label: `fig:paper-1`
- 저자 메모: 외부 모델은 native 조건 유지. GOLD/Vina 출처 인용 필요. PB v2는 데이터셋 구분이며 소프트웨어 버전이 아니다.

## PDF page 2 · Figure 2

**Effects of refinement and chirality-aware pose selection.**

EFF-Dock results use corrected Rw with frozen released weights. Selected-pose docking performance is shown for Astex Diverse Set (n=85), PoseBusters v2 (n=308), PhiBench (n=206), FoldBench (n=558), and OpenBind (n=925), using unguided 100-pose, 10-step generation. Raw and energy-refined banks are evaluated with ordinary confidence selection or chirality-filtered confidence selection. Solid segments represent RMSD <2 Å and PB-valid success; hatched segments represent RMSD <2 Å with PB invalidity. Each total bar height is RMSD-only success. Bars show means over three sampling repeats on the same complexes. Error bars at the solid-segment boundary and total height are the respective sample SDs, not confidence intervals or the SD of the hatched difference. Cohort denominators are retained across conditions. Numerical labels inside solid segments give PB-valid success (%); labels above the error bars give RMSD-only success (%), rounded to one decimal.

- 원본: [02_refinement_chirality.pdf](figures/02_refinement_chirality.pdf)
- LaTeX label: `fig:paper-2`
- 저자 메모: Raw→refined는 에너지 기반 후처리. Chirality 필터와 official PB-valid는 다르다. 복구·호환성 처리는 Methods에 명시.

## PDF page 3 · Figure 3

**Docking performance across ligand-complexity strata.**

EFF-Dock results use corrected Rw with frozen released weights. Rows stratify the five external benchmarks by (A) ligand heavy-atom count, (B) RDKit strict rotatable-bond count, and (C) saved rigid-fragment count. Results use unguided 100-pose, 10-step generation, refined candidates, and chirality-filtered confidence selection. Top-1 denotes selected-pose RMSD <2 Å success; Top-1 + PB-valid additionally requires the same pose to pass PoseBusters. Oracle denotes at least one RMSD-successful pose among all 100 refined candidates and does not require PB validity. Points and error bars show the mean and sample SD across three repeats. Labels give the number of unique complexes per bin; asterisks identify bins with fewer than five complexes. Empty bins are not imputed. Fragment counts come from saved fragment assignments rather than being inferred from rotatable-bond counts. These strata describe associations with performance, not isolated causal effects of molecular complexity.

- 원본: [03_ligand_complexity.pdf](figures/03_ligand_complexity.pdf)
- LaTeX label: `fig:paper-3`
- 저자 메모: 작은 bin의 변동을 일반 경향으로 단정하지 않는다. 이 그림의 Top-1은 chirality-filtered selection이다.

## PDF page 4 · Figure 4

**Cumulative pose success under confidence ranking and generation order.**

EFF-Dock results use corrected Rw with frozen released weights. (A) The fraction of complexes with at least one RMSD <2 Å pose among the k best confidence-ranked candidates from a complete 100-pose bank. (B) The corresponding fraction among the first N candidates in their saved generation order, without confidence sorting. Raw and refined candidates preserve pose identities. Confidence ranking in A is unfiltered, with stable pose-index tie breaking. Lines and shaded bands show means and ±1 sample SD over three repeats for each external benchmark. Solid and dashed lines denote refined and raw banks, respectively. Lower-right insets expand poses 1–20; shaded regions and connectors identify the expanded interval, with the same success-rate scale retained. Both orderings reach the full-bank RMSD oracle at 100 poses. These endpoints measure candidate coverage rather than deployable selected Top-1 success and do not incorporate PoseBusters validity.

- 원본: [04_cumulative_success.pdf](figures/04_cumulative_success.pdf)
- LaTeX label: `fig:paper-4`
- 저자 메모: k는 생성 예산 N이 아니다. A는 전체 bank ranking, B는 생성 prefix. 확대 구간은 실제 관측값 1–20이다.

## PDF page 5 · Figure 5

**Selected-pose success as a function of the available candidate count.**

EFF-Dock results use corrected Rw with frozen released weights. For each of the five external benchmarks, the first N refined poses in saved generation order are retained from the existing unguided 100-pose, 10-step banks. Top-1 reselects the best predicted-RMSD candidate within each prefix; Top-1 + chirality restricts this selection to candidates passing the chirality mask, with fallback to unfiltered Top-1 when no candidate passes. Oracle reports whether the same prefix contains any pose with RMSD <2 Å. Lines and shaded bands show the mean and ±1 sample SD across three repeats. All endpoints are RMSD-only because newly selected prefix poses were not subjected to additional official PoseBusters evaluation. Selected Top-1 need not improve monotonically as candidates are added. Reusing saved prefixes is not a separately executed N-pose sampling experiment and does not provide measured runtime at each N.

- 원본: [05_pose_budget.pdf](figures/05_pose_budget.pdf)
- LaTeX label: `fig:paper-5`
- 저자 메모: Oracle는 누적 그림 B의 refined와 동일. Top-1은 prefix 안에서 재선택하므로 full-bank Top-k와 다르다.

## PDF page 6 · Figure 6

**Guidance and candidate allocation at a fixed learned pose-step count.**

Corrected Rw inference with frozen released weights is evaluated on (A) Astex Diverse Set (n=85) and (B) PoseBusters v2 (n=308). N100/S10 and N40/S25 each use 1,000 learned pose-steps, with unguided generation or the prespecified eta-2 normalized-drift guidance. Raw and energy-refined banks use the same input-chirality-filtered minimum-predicted-RMSD selector, with unfiltered fallback when no candidate is eligible. Solid segments denote RMSD <2 Å and PB-valid success; hatched extensions denote RMSD-successful but PB-invalid poses. Bars and whiskers give means and sample SD across seeds 42, 100042 and 200042, with SD at both endpoint boundaries. The completed unguided N100/S10 baseline is reused by source hash. Guided/unguided priors share seeds and fragment inventories within each budget; a full CPU audit found identical translations and at most 0.0000161 degrees of initial-rotation difference across backends; N40 and N100 are separately executed, without a nested-prior claim. Equal pose-step counts do not imply equal elapsed time because guidance and candidate processing add work. Labels inside solid segments give PB-valid success (%); labels above whiskers give RMSD-only success (%). All conditions are descriptive and do not select production guidance or budget defaults.

- 원본: [06_guidance_budget.pdf](figures/06_guidance_budget.pdf)
- LaTeX label: `fig:paper-6`
- 저자 메모: Rw 신규 실행. Guidance는 eta2 진단 조건이며 기본값으로 채택하지 않는다. N40은 저장된 N100 prefix 분석과 다르다.

## PDF page 7 · Figure 7

**Measured corrected-Rw runtime and generation-process memory.**

(A) Pipeline wall time per complex and (B) initial pose-generation-process peak CUDA allocated memory for the corrected-Rw, unguided N100/S10 evaluation on the five external benchmarks. Every cohort retains its full complex inventory across seeds 42, 100042 and 200042. Runtime sums completed shard wall durations within each seed and divides by the cohort size; bars give the three-seed mean and whiskers the sample SD. Timings include setup, generation, energy refinement, confidence scoring and I/O, but exclude separately executed official PoseBusters evaluation. Summed shard durations are not campaign elapsed time or pure sampling latency. Memory is the maximum initial evaluation-process allocator peak across all 16 shards and three seeds, in GiB, with no SD bar. Astex and PoseBusters use generation-only candidate export; PhiBench, FoldBench and OpenBind include raw-confidence scoring in that initial process. Separate refinement and post-refinement confidence processes are not included in this memory peak; it is neither reserved memory nor a whole-pipeline peak. Executions used a recorded mixture of RTX A5000, RTX 6000 Ada and RTX PRO 6000 backends, with per-cohort device counts and process profiles retained in the source data. These descriptive costs are not a controlled hardware/profile comparison or saturated throughput benchmark. Numerical labels give bar values to one decimal; n denotes unique complexes per seed.

- 원본: [07_runtime_memory.pdf](figures/07_runtime_memory.pdf)
- LaTeX label: `fig:paper-7`
- 저자 메모: Rw 저장 로그 집계. SD는 3시드 및 GPU 혼합 변동을 포함. Memory는 초기 생성/evaluation 프로세스 peak이며 temporal 셋에서는 raw confidence도 포함; sampling-only나 whole-pipeline peak로 해석하지 않는다.

## PDF page 8 · Figure 8

**Sensitivity to supplied-pocket radius, center error and translation prior scale.**

All cells use corrected Rw, unguided N100/S10 generation, frozen released weights, energy refinement and input-chirality-filtered minimum-predicted-RMSD selection. (A,B) Generation pocket radius {6,8,10,12,14} Å at translation prior sigma 2 Å, for Astex Diverse Set (n=85) and PoseBusters v2 (n=308), respectively. (C,D) Prior sigma {1,2,4} Å at generation radius 10 Å on the same cohorts. Rows vary Gaussian supplied-center error with sigma {0,1,2} Å per Cartesian axis. Refinement and confidence retain the original unperturbed supplied center and a fixed 10 Å crop; the experiment measures generation-stage sensitivity with fixed downstream context, not an end-to-end erroneous-center deployment. Confidence uses the actual source prior sigma. Matching seeds couple prior draws; the jitter RNG is independent. A full prior audit reproduced all 28,296 recorded hashes; CPU-dependent quaternion roundoff gave at most 0.0000161 degrees of initial-rotation difference, with identical translations. Bitwise initial-pose identity across CPUs and hardware-invariant predictions are not claimed. Cells report three-seed mean selected-pose RMSD <2 Å and PB-valid success (%), annotated to one decimal. Sample SD and RMSD-only/PB/oracle endpoints are retained in the numerical tables. The linear color scale is 50–100%, with stronger hue variation at 70–80%; values below 50% use the under-range color. Shared sigma-2 cells and the completed baseline are counted once. The grids isolate the two axes rather than crossing every radius and prior scale. These descriptive external results do not tune production settings; sigma-1/4 also measure confidence domain shift.

- 원본: [08_pocket_prior.pdf](figures/08_pocket_prior.pdf)
- LaTeX label: `fig:paper-8`
- 저자 메모: 예전 guided·unfiltered 포켓 그림을 대체한다. 생성만 jitter/cutoff 변경; 후처리는 원래 center/crop10 유지. 서로 다른 프로토콜로 Rw 효과를 주장하지 않는다.

## PDF page 9 · Figure 9

**Sequence relatedness and joint ligand–protein overlap with docking-training data.**

All six cohorts are compared with the same 47,277 executed docking-training samples. Eligible ligand-binding chains follow the frozen 6 Å heavy-atom contact definition. (A) Distribution of the maximum binding-chain sequence identity. Here sequence identity is the number of identical known residues, summed over one-to-one binding-chain assignments within a training sample, divided by the total observed query binding-chain length; unaligned residues and unmatched query chains remain in the denominator. (B) Exact canonical heavy-isomeric ligand identity and a sequence score ≥70% are classified by whether they occur in the same training sample, in separate samples only, individually, or neither is observed. All exact-ligand training candidates are searched for the joint condition. (C) The same-sample fraction already contained in B is shown separately with counts and percentages. Cohorts comprise the five external benchmarks and Validation (n=1,076). Sequence bins and the 70% threshold are descriptive. These results do not establish identical pockets or proven leakage, and no observed match does not certify biological novelty.

- 원본: [09_training_relatedness.pdf](figures/09_training_relatedness.pdf)
- LaTeX label: `fig:paper-9`
- 저자 메모: Train은 47,277개. Confidence subset이나 역사적 PLINDER community 지표와 혼동하지 않는다. 후보 집합·UNK 한계는 Methods에 명시.

## PDF page 10 · Figure 10

**Training-ligand relatedness and stratified docking performance.**

EFF-Dock results use corrected Rw with frozen released weights. (A) Composition of each external benchmark by observed exact ligand identity and nearest training-ligand fingerprint similarity. Exact identity uses canonical heavy-atom isomeric SMILES with explicit-hydrogen normalization and takes precedence over similarity bins. Other ligands are grouped by maximum Tanimoto similarity using radius-2, 2,048-bit Morgan fingerprints without chirality. The reference is the parseable ligand index from the 47,277 executed docking-training samples. (B) Selected-pose RMSD <2 Å and PB-valid success is compared between complexes with and without an observed exact training-ligand match, using the refined, chirality-filtered unguided condition. Bars and whiskers show the mean and sample SD across three repeats; n denotes complexes per stratum. Ligands are not standardized across tautomers or protonation states. No observed exact match is not proof of complete training-data absence, and group differences are confounded by target and ligand composition rather than being causal estimates of memorization. In A, segments of at least 8% are annotated with their percentage; smaller segments remain visible without numerical labels. In B, labels above the whiskers give mean success (%) and labels at the base give stratum counts.

- 원본: [10_ligand_similarity.pdf](figures/10_ligand_similarity.pdf)
- LaTeX label: `fig:paper-10`
- 저자 메모: Ligand Tanimoto와 protein identity는 다르다. 예전 문서의 서열 미계산 설명은 현재 분석에 적용되지 않는다.

## PDF page 11 · Figure 11

**Docking performance across training-sequence identity strata.**

EFF-Dock results use corrected Rw with frozen released weights. The five external benchmarks are stratified by maximum binding-chain sequence identity to the 47,277 executed docking-training samples, using bins <30%, 30–<70%, 70–<90%, and 90–100%. Sequence identity is query-normalized: identical known residues under one-to-one chain assignments are divided by the total observed query binding-chain length, including unaligned residues and unmatched chains. Results use refined candidates and chirality-filtered confidence selection from unguided 100-pose, 10-step banks. Solid segments show RMSD <2 Å and PB-valid success; hatched extensions show RMSD-successful but PB-invalid poses. Total bar heights show RMSD-only success. Bars and error bars are means and sample SD across three repeats, with uncertainty shown at both endpoint boundaries. Labels give the number of complexes per bin; empty bins have no estimate. Validation is excluded because no equivalent three-repeat evaluation bank is available. Small strata and target-composition differences limit interpretation of apparent trends. Numerical labels inside solid segments give PB-valid success (%); labels above the error bars give RMSD-only success (%), rounded to one decimal.

- 원본: [11_sequence_similarity.pdf](figures/11_sequence_similarity.pdf)
- LaTeX label: `fig:paper-11`
- 저자 메모: OpenBind 925개 모두 90–100 구간이라 이 구간화로 OpenBind 내부의 identity 의존성은 평가할 수 없다.

## PDF page 12 · Figure S1

**Paired changes in docking success with refinement and chirality filtering.**

EFF-Dock results use corrected Rw with frozen released weights. Success-rate improvements are absolute differences in the percentage of complexes satisfying RMSD <2 Å and PoseBusters validity. Panels compare (A) refined versus raw poses without the chirality filter, (B) refined versus raw poses with the filter, (C) filter off versus on for raw poses, and (D) filter off versus on for refined poses. Within each complex, paired differences are averaged across three seeds before resampling. Points show the observed mean improvement in percentage points; whiskers show 95% percentile confidence intervals from 2,000 bootstrap draws. Complex bootstrap resamples individual complexes. PDB-cluster bootstrap resamples exact-PDB groups while keeping their member complexes together and computes a complex-weighted mean. Both methods share the same point estimate. Exact-PDB grouping does not capture protein-family dependence; PDB-cluster estimates are unavailable for OpenBind. Intervals condition on the three seeds and are not adjusted for multiple comparisons. Signed labels to the right give the common point estimate in percentage points, rounded to two decimals; the two bootstrap procedures differ in uncertainty rather than in the estimated mean.

- 원본: [S1_paired_uncertainty.pdf](figures/S1_paired_uncertainty.pdf)
- LaTeX label: `fig:paper-s1`
- 저자 메모: 여기는 SD가 아닌 95% CI. Percentage points는 상대 증가율 %가 아니다. OpenBind로 다른 단백질에 대한 일반화를 추정하지 않는다.

## PDF page 13 · Figure S2

**Confidence selection and near-native candidate density.**

EFF-Dock results use corrected Rw with frozen released weights. Refined unguided 100-pose, 10-step banks are evaluated on the five external benchmarks. (A) A complex is counted as successful when at least one pose among the unfiltered confidence-ranked Top-1, Top-5, or all 100 candidates has RMSD <2 Å. The Oracle-100 values represent full-bank RMSD coverage. These values summarize the k=1, 5, and 100 positions of the refined confidence-ranked cumulative curves. (B) The percentage of generated candidate poses with RMSD <2 Å quantifies near-native candidate density, not geometric diversity. Bars and whiskers show means and sample SD across three repeats. All endpoints are RMSD-only; no full-bank PB-valid fraction or PB-valid Top-5/oracle is inferred. The panels have different units of success: complexes containing a successful candidate in A, versus individual candidate poses in B.

- 원본: [S2_candidate_ranking.pdf](figures/S2_candidate_ranking.pdf)
- LaTeX label: `fig:paper-s2`
- 저자 메모: A는 ranked curve 요약. B는 포즈가 분모이므로 complex 기준 oracle와 같은 성공률로 취급하지 않는다.

## PDF page 14 · Figure S3

**Descriptive failure decomposition across ligand-complexity strata.**

EFF-Dock results use corrected Rw with frozen released weights. Rows stratify refined, chirality-filtered results by (A) heavy-atom count, (B) strict rotatable-bond count, and (C) saved rigid-fragment count for the five external benchmarks. Each complex belongs to one of four mutually exclusive categories: no refined candidate with RMSD <2 Å; an RMSD-successful candidate exists but the selected pose fails the threshold; the selected pose passes RMSD but fails PoseBusters; or the selected pose passes both. Bars show mean fractions over three repeats, using all complexes within each stratum. If O, T, and J denote RMSD oracle coverage, selected Top-1 RMSD success, and selected RMSD-plus-PB success, respectively, the four fractions are 100−O, O−T, T−J, and J. Thus this is a decomposition of the complexity-performance results, not an independent causal experiment. Generation failure refers to the refined bank and does not isolate the sampler from refinement effects; selection failure includes chirality-mask exclusions. Labels give complex counts, and dashes denote empty bins.

- 원본: [S3_complexity_failures.pdf](figures/S3_complexity_failures.pdf)
- LaTeX label: `fig:paper-s3`
- 저자 메모: 네 범주는 같은 결과의 대수적 분해이며 독립 원인 검증이 아니다. Full-bank PB oracle는 계산하지 않았다.

## PDF page 15 · Figure S4

**Confidence ranking and remaining selection loss.**

EFF-Dock results use corrected Rw with frozen released weights. (A) Within-complex Spearman correlation between predicted and symmetry-aware observed RMSD for the 100 raw or refined candidates. (B) Refined-bank AUROC and average precision for strict RMSD <2 Å labels, ranked by negative predicted RMSD. Each complex receives equal weight; single-class banks are excluded from both binary metrics. Eligible complex counts for the three repeats are 83/82/82 (Astex Diverse Set), 292/289/290 (PoseBusters v2), 182/181/184 (PhiBench), 513/510/510 (FoldBench), 817/814/807 (OpenBind). (C) Median selected-minus-oracle RMSD regret within each repeat. (D) Selected Top-1 success conditional on the bank containing at least one near-native candidate. C and D use the primary chirality-filtered selector, so their loss includes mask exclusions. Bars show repeat means and error bars the sample SD across three repeats. The full cohorts contain 85, 308, 206, 558 and 925 complexes, respectively. Binary discrimination and final pose selection are different endpoints; a high AUROC does not imply reliable Top-1 selection. Labels above the whiskers give means, with two decimals for correlation, discrimination and RMSD regret, and one decimal for percentage success.

- 원본: [S4_confidence_diagnostics.pdf](figures/S4_confidence_diagnostics.pdf)
- LaTeX label: `fig:paper-s4`
- 저자 메모: AUROC/AP는 양·음성 후보가 모두 있는 complex만 집계. 높은 AUROC와 선택 손실이 함께 존재한다.

## PDF page 16 · Figure S5

**Reliability of saved confidence predictions and candidate-density dependence.**

EFF-Dock results use corrected Rw with frozen released weights. (A) Empirical near-native fraction versus the saved success-head probability, using ten equal-width probability bins. (B) Mean observed RMSD versus mean predicted RMSD in fixed bins [0,1), [1,2), [2,3), [3,4), [4,6), [6,10), and [10,infinity). Dashed diagonals denote agreement. A and B pool all unfiltered refined candidates from three repeats; empty bins are omitted and each point represents a bin mean. No probability recalibration or threshold fitting was performed. Candidate observations within complexes are correlated, so no independent-candidate confidence intervals are shown. (C) Primary chirality-filtered Top-1 success versus the number of near-native refined candidates among 100, in fixed bins 0, 1–5, 6–20, 21–50, and 51–100. Points and dark error bars are repeat means and sample SD; empty repeat strata are omitted from that aggregate. Bin counts and eligible repeats are supplied in the source data. These are descriptive reliability and density associations, not evidence that the auxiliary probability head is the production selector; production selection uses predicted RMSD.

- 원본: [S5_confidence_reliability.pdf](figures/S5_confidence_reliability.pdf)
- LaTeX label: `fig:paper-s5`
- 저자 메모: 확률 head를 새로 fitting하지 않았다. OpenBind의 낮은 Brier는 클래스 불균형 영향도 있으므로 calibration 우수성으로 단정하지 않는다.

## PDF page 17 · Figure S6

**Performance under simultaneous sequence and ligand relatedness restrictions.**

EFF-Dock results use corrected Rw with frozen released weights. Each benchmark compares its full cohort with a fixed stringent subset: maximum observed training-relative binding-chain sequence identity <30%, maximum Morgan fingerprint Tanimoto similarity <0.5, and no observed exact training-ligand match. All conditions must hold for the same complex. Sequence identity retains the query-normalized definition and reference-eligibility limitations described for Figure 9. Surviving counts are 2/85, 4/308, 7/206, 9/558 and 0/925 for Astex Diverse Set, PoseBusters v2, PhiBench, FoldBench and OpenBind, respectively. Refined primary Top-1 success is split into PB-valid solid segments and RMSD-successful but PB-invalid hatched extensions; diamonds indicate the full-bank RMSD oracle. Bars and points show means over three repeats with sample SD. OpenBind has no eligible complex and no imputed performance. Thresholds were not relaxed after observing these counts. Small retained cohorts preclude a strong generalization claim; this is a descriptive restriction of previously evaluated benchmarks, not a newly held-out test or an audit of all precursor training exposure. Numerical labels inside solid segments give PB-valid success (%); labels above the selected-pose error bars give RMSD-only success (%), rounded to one decimal. Diamond markers retain the separately plotted RMSD oracle.

- 원본: [S6_stringent_subset.pdf](figures/S6_stringent_subset.pdf)
- LaTeX label: `fig:paper-s6`
- 저자 메모: 남는 표본이 총 22개뿐이다. 성능 증명보다 엄격한 비교의 표본 한계를 보고하는 SI로 사용.

## PDF page 18 · Figure S7

**Physical-validity failures before and after refinement.**

EFF-Dock results use corrected Rw with frozen released weights. (A,B) Failure percentages among primary chirality-filtered selected poses in raw and refined banks. Each stage may select a different candidate; these panels describe the complete pipeline, not a fixed-pose intervention. (C,D) Fail-to-pass and pass-to-fail percentages among candidate indices for which saved official PoseBusters evaluations exist at both stages, deduplicating identical ordinary and chirality-filtered selections. These matched subsets contain 41/26/29, 112/100/104, 72/76/63, 171/173/208, 473/443/435 candidate pairs across three repeats for Astex Diverse Set, PoseBusters v2, PhiBench, FoldBench and OpenBind, respectively. Values are means of repeat-specific percentages, with all evaluated pairs as the denominator, not just initially failing or passing pairs. A grouped check fails if any constituent check fails. Bond geometry combines bond length and angle checks; receptor clash combines protein minimum-distance and volume-overlap checks; stereochemistry combines tetrahedral and double-bond checks; ring planarity combines aromatic-ring and double-bond flatness checks. All PB checks combines all 27 non-RMSD checks, including those not displayed separately. The common color scale is 0–50%. Categories overlap and do not sum to 100%. The matched subset is selection-biased and cannot establish full-bank repair rates. This figure reports PB validity irrespective of RMSD success.

- 원본: [S7_physical_validity.pdf](figures/S7_physical_validity.pdf)
- LaTeX label: `fig:paper-s7`
- 저자 메모: 동일 complex의 재선택과 동일 pose의 refinement를 분리. full-bank PB 평가로 오해하지 않도록 한다.

## PDF page 19 · Figure S8

**Dataset-specific examples of success, refinement rescue and selection failure.**

EFF-Dock results use corrected Rw with frozen released weights. Rows A–E show Astex Diverse Set, PoseBusters v2, PhiBench, FoldBench and OpenBind; columns show successful selection, refinement rescue and selection failure. Examples use saved primary refined selections pooled across three repeats. Within each dataset, the rescue is selected first as the largest same-candidate RMSD decrease from ≥2 Å raw to <2 Å refined with final PB validity. Successful selection uses the lower-median selected RMSD among PB-valid successes, excluding the rescue complex. Selection failure uses the lower-median selected-minus-oracle RMSD among cases with selected RMSD ≥2 Å and an oracle <2 Å, excluding the two previously chosen complex IDs. Ties are broken by repeat and ID. Rescue panels illustrate extremes, not typical improvements; these examples do not estimate prevalence. Eligible rescue counts are 3, 7, 10, 31, 80 complex-repeat records, respectively. Green, blue and orange carbons denote crystal, selected and raw/oracle poses; heteroatoms use element colors. All structures retain the receptor frame without independent ligand alignment. Each panel uses one case-specific viewing rotation shared by the receptor and all poses, chosen for visibility; projected separation is not a quantitative measure. Protein cartoons use stored coordinates of chains contacting the crystal ligand within 5 Å for display only. Oracle denotes minimum saved-bank RMSD, without a PB-validity requirement. Full sample IDs, repeat indices and eligible counts are provided in the accompanying evidence table; FoldBench image labels abbreviate PDB/ligand/chain and OpenBind uses sample IDs.

- 원본: [S8_structure_examples.pdf](figures/S8_structure_examples.pdf)
- LaTeX label: `fig:paper-s8`
- 저자 메모: 5개 외부 셋마다 서로 다른 complex 3건. Rescue는 셋 내 최대 개선 사례이며 대표적 평균효과가 아니다. 성공·선택실패는 3회 반복을 합친 적격 기록의 lower median이며 먼저 뽑은 complex ID를 제외한다. n은 complex-repeat 수이다.

## PDF page 20 · Figure S9

**Paired uncertainty relative to locally executed baselines.**

EFF-Dock results use corrected Rw with frozen released weights. (A) Astex Diverse Set (n=85) and (B) PoseBusters v2 (n=308). Points show EFF-Dock minus baseline success-rate differences in percentage points for RMSD <2 Å and same-selected-pose RMSD-plus-PB success. Each complex contributes the difference between method-specific three-repeat means. Horizontal intervals are percentile 95% paired bootstrap intervals from 2,000 resamples (seed 20260923), resampling exact PDB-accession groups and preserving complex-weighted means. Every retained complex has a distinct PDB accession in these two cohorts, so PDB-group and complex resampling coincide; this does not account for protein-family dependence. EFF-Dock uses the primary refined chirality-filtered protocol. The five locally run baselines retain the native budgets and selectors of Figure 1; SurfDock uses its force-optimized variant. Literature-only rows are excluded. Positive differences favor EFF-Dock. Intervals are descriptive, unadjusted for multiple comparisons, and conditional on the three available repeats; they do not establish equal-compute superiority. Signed labels at the right of each interval row give the corresponding observed difference in percentage points, rounded to two decimals.

- 원본: [S9_baseline_uncertainty.pdf](figures/S9_baseline_uncertainty.pdf)
- LaTeX label: `fig:paper-s9`
- 저자 메모: 음수·0을 포함하는 CI도 모두 유지. Exact PDB가 모두 달라서 여기서는 complex/PDB bootstrap이 같다.

## PDF page 21 · Figure S10

**Fragment motion along a saved ODE generation trajectory.**

Corrected Rw with the frozen released docking checkpoint. Astex Diverse Set 1T46–STI, shown in a fixed receptor coordinate frame and camera. The five panels show actual saved states at t=0.000, 0.271, 0.488, 0.784 and 1.000; no coordinates are interpolated. These are the available states nearest to five equally spaced target times. Carbon colors track the same six rigid fragments throughout; nitrogen is blue and oxygen red. The pale protein cartoon provides binding-site context. Interfragment bonds are omitted in the first four panels for visibility and shown in grey at the last panel using the known ligand connectivity. This display convention does not represent chemical bond formation. Time t is the dimensionless generative-flow coordinate, not physical time or energy-refinement progress. The stored illustrative run used one sample, ten ODE steps, a late schedule with power 3, positional prior sigma 2 Å, pocket cutoff 10 Å, seed 42 and no guidance or confidence selection. The recorded checkpoint is effdock_docking_early_time_t0p10_50k.pt. This is a separate N1 illustration, not an N100 benchmark-selected pose; no success, PB-validity or representative-trajectory claim is made. All 11 saved frames and the original fragment assignments are retained in the source data.

- 원본: [S10_fragment_trajectory.pdf](figures/S10_fragment_trajectory.pdf)
- LaTeX label: `fig:paper-s10`
- 저자 메모: Fig1과 같은 Rw N1 생성 경로의 시간별 그림. 현재 PDF page 19 / S10. 생성 ODE 시간이며 energy refinement 시간이 아니다. Fragment 사이 결합을 숨기는 것은 표시 방식이며 화학 반응을 의미하지 않는다.

## PDF page 22 · Figure S11

**Eligibility-aware selection bottlenecks and regret.**

EFF-Dock results use corrected Rw with frozen released weights. All five cohorts use corrected Rw, refined N100/S10 banks and the frozen chirality-filtered confidence selector. (A) Mutually exclusive outcomes are no RMSD <2 Å candidate in the full bank; full-bank success but no success in the effective chirality-eligible set; eligible success but an unsuccessful selected pose; selected RMSD success with PB invalidity; and RMSD success with PB validity. If no pose passes chirality, the effective set follows the actual unfiltered fallback. The PB-invalid segment is hatched. Bars are three-seed mean fractions and retain every complex. (B) Cumulative selected-pose RMSD regret relative to the minimum RMSD in that same effective candidate set, shown through 5 Å; the source table retains the complete distributions and per-seed medians and 90th percentiles. Reference RMSD enters retrospective diagnostics only, never selection. PB validity is known for selected poses, not for every candidate. In A, segments of at least 8% are annotated with their mean percentage; smaller segments remain visible without numerical labels.

- 원본: [S12_selector_bottleneck.pdf](figures/S12_selector_bottleneck.pdf)
- LaTeX label: `fig:paper-s12`
- 저자 메모: 저장된 Rw 결과의 사후 진단. 외부 결과로 selector나 threshold를 튜닝하지 않는다.
