# Current manuscript figure captions — Prism reference

현재 **20페이지 PDF** 기준: 본문용 Figure 1–11, 보충 Figure S1–S9. 최종 논문 번호는 편집 시 변경 가능하다. 각 영문 캡션은 복사해서 사용할 수 있으며, 한국어 메모는 저자 참고용이다.

PDF SHA-256: `fd6c2b8d7839c9ce39e6be113fc76ad0d2fd11fab17495a44089e8cc409b05fc`

## 공통 Methods 메모

- RMSD는 symmetry-aware, no-alignment RMSD <2 Å 기준이다. 주어진 포켓 조건 평가이며 blind docking으로 기술하지 않는다.
- 주 분석은 unguided N100/S10 및 세 반복이다. 그림6의 예산/guidance 변형, 그림8의 guided robustness, 그림1의 외부 모델·문헌 조건은 별도다.
- 기본 오차막대는 같은 complex 집합에서 세 생성 반복의 sample SD다. Figure S1과 S9는 paired bootstrap 95% CI이며, S5 A/B는 pooled reliability, S7은 반복 평균, S8은 사례 그림이다.
- PhiBench 복구 3건, OpenBind의 단일 protease·quality/covalent flags, FoldBench PB 3개 shard의 InChI 호환성 처리는 데이터/평가 Methods에 유지한다.
- 학습 관련성은 실행된 docking train 47,277개 기준이다. 모든 선행 pretraining 노출의 합집합 감사라고 주장하지 않는다.
- Figure1 GOLD/Vina 문헌값 인용: https://doi.org/10.1039/D3SC04185A 및 https://zenodo.org/records/8278563 . Bibliography의 실제 cite key로 연결한다.
- 과거 디렉터리별 캡션에는 이전 페이지 번호와 폐기된 그림 설명이 남아 있다. 현재 제출용 매핑은 이 문서와 manifest.json을 기준으로 한다.

## PDF page 1 · Figure 1

**External docking performance on two supplied-pocket benchmarks.**

(A) Astex Diverse Set (85 complexes) and (B) PoseBusters v2 (308 complexes). Solid bar segments indicate selected poses with RMSD <2 Å that also pass PoseBusters; hatched extensions indicate RMSD-successful but PB-invalid poses. Total bar length therefore gives RMSD-only success. Local-run results are means over three repeats, with sample SD shown separately at the PB-valid and RMSD-only boundaries. EFF-Dock uses unguided generation with 100 poses and 10 sampling steps, energy-based refinement, and chirality-filtered confidence selection. Other methods retain their evaluated native budgets and selection procedures; SurfDock is represented by its force-optimized variant. GOLD and AutoDock Vina rows below the separator are literature-reported values, not local reruns, and have no repeat SD. Different computational budgets and PoseBusters versions prevent interpreting this as an equal-compute comparison.

- 원본: [01_model_comparison.pdf](figures/01_model_comparison.pdf)
- LaTeX label: `fig:paper-1`
- 저자 메모: 외부 모델은 native 조건 유지. GOLD/Vina 출처 인용 필요. PB v2는 데이터셋 구분이며 소프트웨어 버전이 아니다.

## PDF page 2 · Figure 2

**Effects of refinement and chirality-aware pose selection.**

Selected-pose docking performance is shown for Astex Diverse Set (n=85), PoseBusters v2 (n=308), PhiBench (n=206), FoldBench (n=558), and OpenBind (n=925), using unguided 100-pose, 10-step generation. Raw and energy-refined banks are evaluated with ordinary confidence selection or chirality-filtered confidence selection. Solid segments represent RMSD <2 Å and PB-valid success; hatched segments represent RMSD <2 Å with PB invalidity. Each total bar height is RMSD-only success. Bars show means over three sampling repeats on the same complexes. Error bars at the solid-segment boundary and total height are the respective sample SDs, not confidence intervals or the SD of the hatched difference. Cohort denominators are retained across conditions.

- 원본: [02_refinement_chirality.pdf](figures/02_refinement_chirality.pdf)
- LaTeX label: `fig:paper-2`
- 저자 메모: Raw→refined는 에너지 기반 후처리. Chirality 필터와 official PB-valid는 다르다. 복구·호환성 처리는 Methods에 명시.

## PDF page 3 · Figure 3

**Docking performance across ligand-complexity strata.**

Rows stratify the five external benchmarks by (A) ligand heavy-atom count, (B) RDKit strict rotatable-bond count, and (C) saved rigid-fragment count. Results use unguided 100-pose, 10-step generation, refined candidates, and chirality-filtered confidence selection. Top-1 denotes selected-pose RMSD <2 Å success; Top-1 + PB-valid additionally requires the same pose to pass PoseBusters. Oracle denotes at least one RMSD-successful pose among all 100 refined candidates and does not require PB validity. Points and error bars show the mean and sample SD across three repeats. Labels give the number of unique complexes per bin; asterisks identify bins with fewer than five complexes. Empty bins are not imputed. Fragment counts come from saved fragment assignments rather than being inferred from rotatable-bond counts. These strata describe associations with performance, not isolated causal effects of molecular complexity.

- 원본: [03_ligand_complexity.pdf](figures/03_ligand_complexity.pdf)
- LaTeX label: `fig:paper-3`
- 저자 메모: 작은 bin의 변동을 일반 경향으로 단정하지 않는다. 이 그림의 Top-1은 chirality-filtered selection이다.

## PDF page 4 · Figure 4

**Cumulative pose success under confidence ranking and generation order.**

(A) The fraction of complexes with at least one RMSD <2 Å pose among the k best confidence-ranked candidates from a complete 100-pose bank. (B) The corresponding fraction among the first N candidates in their saved generation order, without confidence sorting. Raw and refined candidates preserve pose identities. Confidence ranking in A is unfiltered, with stable pose-index tie breaking. Lines and shaded bands show means and ±1 sample SD over three repeats for each external benchmark. Solid and dashed lines denote refined and raw banks, respectively. Lower-right insets expand poses 1–20; shaded regions and connectors identify the expanded interval, with the same success-rate scale retained. Both orderings reach the full-bank RMSD oracle at 100 poses. These endpoints measure candidate coverage rather than deployable selected Top-1 success and do not incorporate PoseBusters validity.

- 원본: [04_cumulative_success.pdf](figures/04_cumulative_success.pdf)
- LaTeX label: `fig:paper-4`
- 저자 메모: k는 생성 예산 N이 아니다. A는 전체 bank ranking, B는 생성 prefix. 확대 구간은 실제 관측값 1–20이다.

## PDF page 5 · Figure 5

**Selected-pose success as a function of the available candidate count.**

For each of the five external benchmarks, the first N refined poses in saved generation order are retained from the existing unguided 100-pose, 10-step banks. Top-1 reselects the best predicted-RMSD candidate within each prefix; Top-1 + chirality restricts this selection to candidates passing the chirality mask, with fallback to unfiltered Top-1 when no candidate passes. Oracle reports whether the same prefix contains any pose with RMSD <2 Å. Lines and shaded bands show the mean and ±1 sample SD across three repeats. All endpoints are RMSD-only because newly selected prefix poses were not subjected to additional official PoseBusters evaluation. Selected Top-1 need not improve monotonically as candidates are added. Reusing saved prefixes is not a separately executed N-pose sampling experiment and does not provide measured runtime at each N.

- 원본: [05_pose_budget.pdf](figures/05_pose_budget.pdf)
- LaTeX label: `fig:paper-5`
- 저자 메모: Oracle는 누적 그림 B의 refined와 동일. Top-1은 prefix 안에서 재선택하므로 full-bank Top-k와 다르다.

## PDF page 6 · Figure 6

**Guidance and pose–step allocation across post-processing conditions.**

Performance on Astex Diverse Set and PoseBusters v2 is compared for 100 poses with 10 sampling steps (N100/S10) and 40 poses with 25 steps (N40/S25), with guidance disabled or enabled at η=2. The four selection conditions are raw, raw with chirality filtering, refined, and refined with chirality filtering. The endpoint is selected-pose RMSD <2 Å together with PoseBusters validity. Points and error bars indicate means and sample SD across three repeats. Guided and unguided runs within a matched budget share the verified initial priors; priors are not assumed nested between the two budgets. Both allocations have the same product N×S, but this does not imply equal wall time because candidate-dependent refinement and confidence evaluation costs differ.

- 원본: [06_guidance_budget.pdf](figures/06_guidance_budget.pdf)
- LaTeX label: `fig:paper-6`
- 저자 메모: 동일 N×S는 동일 runtime이 아니다. 비용 그림과 별도로 사용한다.

## PDF page 7 · Figure 7

**Measured runtime and sampling memory for two unguided budgets.**

(A) Aggregate pipeline wall time per complex and (B) peak CUDA allocated memory during sampling for unguided N100/S10 and N40/S25 on Astex Diverse Set and PoseBusters v2. Runtime is the sum of completed shard wall durations divided by the number of complexes within each repeat, averaged across three repeats; whiskers indicate sample SD. It includes recorded pipeline setup, refinement, confidence scoring, and I/O overhead, but excludes separate official PoseBusters evaluation. Because parallel shard durations are summed, this is not user-perceived campaign elapsed time or pure sampling latency. Memory is the maximum recorded sampling allocator peak across shards and repeats, expressed in GiB, and has no SD bar. Measurements use the recorded RTX 6000 Ada runs. Memory is neither whole-pipeline peak usage nor a per-pose quantity.

- 원본: [07_runtime_memory.pdf](figures/07_runtime_memory.pdf)
- LaTeX label: `fig:paper-7`
- 저자 메모: A는 shard wall의 합을 나눈 비용, B는 sampling allocated peak. Guided나 외부 모델 비용으로 확대 해석하지 않는다.

## PDF page 8 · Figure 8

**Sensitivity to pocket crop, prior scale, and pocket-center perturbation.**

(A) Sampling-pocket cutoff and (B) initial-prior scale are varied for Astex Diverse Set and PoseBusters v2 in the frozen guided (η=2) robustness campaign. Rows within each heatmap correspond to Gaussian center-jitter standard deviations of 0, 1, or 2 Å per Cartesian axis. Cells show the mean percentage of selected poses satisfying RMSD <2 Å and PoseBusters validity across three repeats, after refinement and ordinary confidence selection without the additional chirality filter. The pocket-cutoff sweep fixes the prior scale at 2 Å, and the prior-scale sweep fixes the sampling cutoff at 10 Å. Only the sampling crop changes in the pocket-cutoff sweep; refinement and scoring retain 10 Å crops. All heatmaps use a shared linear 50–100% numerical scale, with color contrast emphasized around 70–80%; the 75% color anchor is not a statistical threshold. These guided robustness results are distinct from the main unguided evaluation condition.

- 원본: [08_pocket_prior.pdf](figures/08_pocket_prior.pdf)
- LaTeX label: `fig:paper-8`
- 저자 메모: Pocket sweep은 prior σ=2 Å, prior sweep은 cutoff=10 Å로 고정. Center jitter와 prior σ는 서로 다르다.

## PDF page 9 · Figure 9

**Sequence relatedness and joint ligand–protein overlap with docking-training data.**

All six cohorts are compared with the same 47,277 executed docking-training samples. Eligible ligand-binding chains follow the frozen 6 Å heavy-atom contact definition. (A) Distribution of the maximum binding-chain sequence identity. Here sequence identity is the number of identical known residues, summed over one-to-one binding-chain assignments within a training sample, divided by the total observed query binding-chain length; unaligned residues and unmatched query chains remain in the denominator. (B) Exact canonical heavy-isomeric ligand identity and a sequence score ≥70% are classified by whether they occur in the same training sample, in separate samples only, individually, or neither is observed. All exact-ligand training candidates are searched for the joint condition. (C) The same-sample fraction already contained in B is shown separately with counts and percentages. Cohorts comprise the five external benchmarks and Validation (n=1,076). Sequence bins and the 70% threshold are descriptive. These results do not establish identical pockets or proven leakage, and no observed match does not certify biological novelty.

- 원본: [09_training_relatedness.pdf](figures/09_training_relatedness.pdf)
- LaTeX label: `fig:paper-9`
- 저자 메모: Train은 47,277개. Confidence subset이나 역사적 PLINDER community 지표와 혼동하지 않는다. 후보 집합·UNK 한계는 Methods에 명시.

## PDF page 10 · Figure 10

**Training-ligand relatedness and stratified docking performance.**

(A) Composition of each external benchmark by observed exact ligand identity and nearest training-ligand fingerprint similarity. Exact identity uses canonical heavy-atom isomeric SMILES with explicit-hydrogen normalization and takes precedence over similarity bins. Other ligands are grouped by maximum Tanimoto similarity using radius-2, 2,048-bit Morgan fingerprints without chirality. The reference is the parseable ligand index from the 47,277 executed docking-training samples. (B) Selected-pose RMSD <2 Å and PB-valid success is compared between complexes with and without an observed exact training-ligand match, using the refined, chirality-filtered unguided condition. Bars and whiskers show the mean and sample SD across three repeats; n denotes complexes per stratum. Ligands are not standardized across tautomers or protonation states. No observed exact match is not proof of complete training-data absence, and group differences are confounded by target and ligand composition rather than being causal estimates of memorization.

- 원본: [10_ligand_similarity.pdf](figures/10_ligand_similarity.pdf)
- LaTeX label: `fig:paper-10`
- 저자 메모: Ligand Tanimoto와 protein identity는 다르다. 예전 문서의 서열 미계산 설명은 현재 분석에 적용되지 않는다.

## PDF page 11 · Figure 11

**Docking performance across training-sequence identity strata.**

The five external benchmarks are stratified by maximum binding-chain sequence identity to the 47,277 executed docking-training samples, using bins <30%, 30–<70%, 70–<90%, and 90–100%. Sequence identity is query-normalized: identical known residues under one-to-one chain assignments are divided by the total observed query binding-chain length, including unaligned residues and unmatched chains. Results use refined candidates and chirality-filtered confidence selection from unguided 100-pose, 10-step banks. Solid segments show RMSD <2 Å and PB-valid success; hatched extensions show RMSD-successful but PB-invalid poses. Total bar heights show RMSD-only success. Bars and error bars are means and sample SD across three repeats, with uncertainty shown at both endpoint boundaries. Labels give the number of complexes per bin; empty bins have no estimate. Validation is excluded because no equivalent three-repeat evaluation bank is available. Small strata and target-composition differences limit interpretation of apparent trends.

- 원본: [11_sequence_similarity.pdf](figures/11_sequence_similarity.pdf)
- LaTeX label: `fig:paper-11`
- 저자 메모: OpenBind 925개 모두 90–100 구간이라 이 구간화로 OpenBind 내부의 identity 의존성은 평가할 수 없다.

## PDF page 12 · Figure S1

**Paired changes in docking success with refinement and chirality filtering.**

Success-rate improvements are absolute differences in the percentage of complexes satisfying RMSD <2 Å and PoseBusters validity. Panels compare (A) refined versus raw poses without the chirality filter, (B) refined versus raw poses with the filter, (C) filter off versus on for raw poses, and (D) filter off versus on for refined poses. Within each complex, paired differences are averaged across three seeds before resampling. Points show the observed mean improvement in percentage points; whiskers show 95% percentile confidence intervals from 2,000 bootstrap draws. Complex bootstrap resamples individual complexes. PDB-cluster bootstrap resamples exact-PDB groups while keeping their member complexes together and computes a complex-weighted mean. Both methods share the same point estimate. Exact-PDB grouping does not capture protein-family dependence; PDB-cluster estimates are unavailable for OpenBind. Intervals condition on the three seeds and are not adjusted for multiple comparisons.

- 원본: [S1_paired_uncertainty.pdf](figures/S1_paired_uncertainty.pdf)
- LaTeX label: `fig:paper-s1`
- 저자 메모: 여기는 SD가 아닌 95% CI. Percentage points는 상대 증가율 %가 아니다. OpenBind로 다른 단백질에 대한 일반화를 추정하지 않는다.

## PDF page 13 · Figure S2

**Confidence selection and near-native candidate density.**

Refined unguided 100-pose, 10-step banks are evaluated on the five external benchmarks. (A) A complex is counted as successful when at least one pose among the unfiltered confidence-ranked Top-1, Top-5, or all 100 candidates has RMSD <2 Å. The Oracle-100 values represent full-bank RMSD coverage. These values summarize the k=1, 5, and 100 positions of the refined confidence-ranked cumulative curves. (B) The percentage of generated candidate poses with RMSD <2 Å quantifies near-native candidate density, not geometric diversity. Bars and whiskers show means and sample SD across three repeats. All endpoints are RMSD-only; no full-bank PB-valid fraction or PB-valid Top-5/oracle is inferred. The panels have different units of success: complexes containing a successful candidate in A, versus individual candidate poses in B.

- 원본: [S2_candidate_ranking.pdf](figures/S2_candidate_ranking.pdf)
- LaTeX label: `fig:paper-s2`
- 저자 메모: A는 ranked curve 요약. B는 포즈가 분모이므로 complex 기준 oracle와 같은 성공률로 취급하지 않는다.

## PDF page 14 · Figure S3

**Descriptive failure decomposition across ligand-complexity strata.**

Rows stratify refined, chirality-filtered results by (A) heavy-atom count, (B) strict rotatable-bond count, and (C) saved rigid-fragment count for the five external benchmarks. Each complex belongs to one of four mutually exclusive categories: no refined candidate with RMSD <2 Å; an RMSD-successful candidate exists but the selected pose fails the threshold; the selected pose passes RMSD but fails PoseBusters; or the selected pose passes both. Bars show mean fractions over three repeats, using all complexes within each stratum. If O, T, and J denote RMSD oracle coverage, selected Top-1 RMSD success, and selected RMSD-plus-PB success, respectively, the four fractions are 100−O, O−T, T−J, and J. Thus this is a decomposition of the complexity-performance results, not an independent causal experiment. Generation failure refers to the refined bank and does not isolate the sampler from refinement effects; selection failure includes chirality-mask exclusions. Labels give complex counts, and dashes denote empty bins.

- 원본: [S3_complexity_failures.pdf](figures/S3_complexity_failures.pdf)
- LaTeX label: `fig:paper-s3`
- 저자 메모: 네 범주는 같은 결과의 대수적 분해이며 독립 원인 검증이 아니다. Full-bank PB oracle는 계산하지 않았다.

## PDF page 15 · Figure S4

**Confidence ranking and remaining selection loss.**

(A) Within-complex Spearman correlation between predicted and symmetry-aware observed RMSD for the 100 raw or refined candidates. (B) Refined-bank AUROC and average precision for strict RMSD <2 Å labels, ranked by negative predicted RMSD. Each complex receives equal weight; single-class banks are excluded from both binary metrics. Eligible complex counts for the three repeats are 83/82/82 (Astex Diverse Set), 291/289/290 (PoseBusters v2), 183/181/185 (PhiBench), 513/509/511 (FoldBench), and 818/812/808 (OpenBind). (C) Median selected-minus-oracle RMSD regret within each repeat. (D) Selected Top-1 success conditional on the bank containing at least one near-native candidate. C and D use the primary chirality-filtered selector, so their loss includes mask exclusions. Bars show repeat means and error bars the sample SD across three repeats. The full cohorts contain 85, 308, 206, 558 and 925 complexes, respectively. Binary discrimination and final pose selection are different endpoints; a high AUROC does not imply reliable Top-1 selection.

- 원본: [S4_confidence_diagnostics.pdf](figures/S4_confidence_diagnostics.pdf)
- LaTeX label: `fig:paper-s4`
- 저자 메모: AUROC/AP는 양·음성 후보가 모두 있는 complex만 집계. 높은 AUROC와 선택 손실이 함께 존재한다.

## PDF page 16 · Figure S5

**Reliability of saved confidence predictions and candidate-density dependence.**

(A) Empirical near-native fraction versus the saved success-head probability, using ten equal-width probability bins. (B) Mean observed RMSD versus mean predicted RMSD in fixed bins [0,1), [1,2), [2,3), [3,4), [4,6), [6,10), and [10,infinity). Dashed diagonals denote agreement. A and B pool all unfiltered refined candidates from three repeats; empty bins are omitted and each point represents a bin mean. No probability recalibration or threshold fitting was performed. Candidate observations within complexes are correlated, so no independent-candidate confidence intervals are shown. (C) Primary chirality-filtered Top-1 success versus the number of near-native refined candidates among 100, in fixed bins 0, 1–5, 6–20, 21–50, and 51–100. Points and dark error bars are repeat means and sample SD; empty repeat strata are omitted from that aggregate. Bin counts and eligible repeats are supplied in the source data. These are descriptive reliability and density associations, not evidence that the auxiliary probability head is the production selector; production selection uses predicted RMSD.

- 원본: [S5_confidence_reliability.pdf](figures/S5_confidence_reliability.pdf)
- LaTeX label: `fig:paper-s5`
- 저자 메모: 확률 head를 새로 fitting하지 않았다. OpenBind의 낮은 Brier는 클래스 불균형 영향도 있으므로 calibration 우수성으로 단정하지 않는다.

## PDF page 17 · Figure S6

**Performance under simultaneous sequence and ligand relatedness restrictions.**

Each benchmark compares its full cohort with a fixed stringent subset: maximum observed training-relative binding-chain sequence identity <30%, maximum Morgan fingerprint Tanimoto similarity <0.5, and no observed exact training-ligand match. All conditions must hold for the same complex. Sequence identity retains the query-normalized definition and reference-eligibility limitations described for Figure 9. Surviving counts are 2/85, 4/308, 7/206, 9/558 and 0/925 for Astex Diverse Set, PoseBusters v2, PhiBench, FoldBench and OpenBind, respectively. Refined primary Top-1 success is split into PB-valid solid segments and RMSD-successful but PB-invalid hatched extensions; diamonds indicate the full-bank RMSD oracle. Bars and points show means over three repeats with sample SD. OpenBind has no eligible complex and no imputed performance. Thresholds were not relaxed after observing these counts. Small retained cohorts preclude a strong generalization claim; this is a descriptive restriction of previously evaluated benchmarks, not a newly held-out test or an audit of all precursor training exposure.

- 원본: [S6_stringent_subset.pdf](figures/S6_stringent_subset.pdf)
- LaTeX label: `fig:paper-s6`
- 저자 메모: 남는 표본이 총 22개뿐이다. 성능 증명보다 엄격한 비교의 표본 한계를 보고하는 SI로 사용.

## PDF page 18 · Figure S7

**Physical-validity failures before and after refinement.**

(A,B) Failure percentages among primary chirality-filtered selected poses in raw and refined banks. Each stage may select a different candidate; these panels describe the complete pipeline, not a fixed-pose intervention. (C,D) Fail-to-pass and pass-to-fail percentages among candidate indices for which saved official PoseBusters evaluations exist at both stages, deduplicating identical ordinary and chirality-filtered selections. These matched subsets contain 40/26/29, 112/93/111, 78/75/70, 183/183/188 and 483/446/432 candidate pairs across three repeats for Astex Diverse Set, PoseBusters v2, PhiBench, FoldBench and OpenBind, respectively. Values are means of repeat-specific percentages, with all evaluated pairs as the denominator, not just initially failing or passing pairs. A grouped check fails if any constituent check fails. Bond geometry combines bond length and angle checks; receptor clash combines protein minimum-distance and volume-overlap checks; stereochemistry combines tetrahedral and double-bond checks; ring planarity combines aromatic-ring and double-bond flatness checks. All PB checks combines all 27 non-RMSD checks, including those not displayed separately. The common color scale is 0–50%. Categories overlap and do not sum to 100%. The matched subset is selection-biased and cannot establish full-bank repair rates. This figure reports PB validity irrespective of RMSD success.

- 원본: [S7_physical_validity.pdf](figures/S7_physical_validity.pdf)
- LaTeX label: `fig:paper-s7`
- 저자 메모: 동일 complex의 재선택과 동일 pose의 refinement를 분리. full-bank PB 평가로 오해하지 않도록 한다.

## PDF page 19 · Figure S8

**Receptor-frame examples of success, refinement rescue and selection failure.**

Illustrative Astex Diverse Set cases from repeat 0. Upper views show complete ligand-contacting protein chains as cartoons; lower views enlarge the binding site. Displayed chains have at least one heavy atom within 5 Å of a crystal-ligand heavy atom; this display rule does not change evaluation inputs. Ligands are ball-and-stick, with green crystal, blue selected and orange comparator carbon atoms; heteroatoms use element colors (nitrogen blue, oxygen red, sulfur yellow). Sticks show connectivity. The cartoons are rendered from stored receptor coordinates using 3Dmol.js secondary-structure assignment. All poses retain the original receptor frame, with no independent ligand alignment. (A) 1OWE, the lower-median selected RMSD among 66 PB-valid successes (0.80 Å). (B) 1HVY, the sole eligible same-candidate RMSD rescue among this repeat's refined primary selections: 2.32 Å before refinement and 1.98 Å after refinement, with the final selected pose PB-valid. This small threshold crossing is illustrative. (C) 1TZ8, the lower-median regret among 14 selection failures with a near-native candidate available: selected RMSD 2.27 Å versus oracle RMSD 0.65 Å. Median rules and ID tie breaking were specified before inspecting the structures. The views do not assign contacts or interaction types.

- 원본: [S8_structure_examples.pdf](figures/S8_structure_examples.pdf)
- LaTeX label: `fig:paper-s8`
- 저자 메모: 시각적으로 가장 좋은 사례를 고르지 않았다. Rescue는 해당 정의에서 1건뿐이며 2 Å 경계 부근의 작은 변화이다.

## PDF page 20 · Figure S9

**Paired uncertainty relative to locally executed baselines.**

(A) Astex Diverse Set (n=85) and (B) PoseBusters v2 (n=308). Points show EFF-Dock minus baseline success-rate differences in percentage points for RMSD <2 Å and same-selected-pose RMSD-plus-PB success. Each complex contributes the difference between method-specific three-repeat means. Horizontal intervals are percentile 95% paired bootstrap intervals from 2,000 resamples (seed 20260923), resampling exact PDB-accession groups and preserving complex-weighted means. Every retained complex has a distinct PDB accession in these two cohorts, so PDB-group and complex resampling coincide; this does not account for protein-family dependence. EFF-Dock uses the primary refined chirality-filtered protocol. The five locally run baselines retain the native budgets and selectors of Figure 1; SurfDock uses its force-optimized variant. Literature-only rows are excluded. Positive differences favor EFF-Dock. Intervals are descriptive, unadjusted for multiple comparisons, and conditional on the three available repeats; they do not establish equal-compute superiority.

- 원본: [S9_baseline_uncertainty.pdf](figures/S9_baseline_uncertainty.pdf)
- LaTeX label: `fig:paper-s9`
- 저자 메모: 음수·0을 포함하는 CI도 모두 유지. Exact PDB가 모두 달라서 여기서는 complex/PDB bootstrap이 같다.
