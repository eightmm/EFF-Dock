# Manuscript figures

Primary benchmark figures now use the complete corrected **Rw**, frozen weights, N100/S10 and three seeds. Historical guidance/budget/runtime/pocket/prior panels and the saved N1 trajectory remain explicitly **Rᵀw**. Training-relatedness inputs are unchanged.

- [Combined result PDF](paper_figures.pdf): 23 pages.
- [Prism ZIP](prism/prism_figure_reference.zip): all figure PDFs, PNG previews, Fig1/2 SVGs, captions and LaTeX.
- [Captions](FIGURE_CAPTIONS.md)
- [Rw evidence and numerical comparisons](EVIDENCE.md)
- [Numerical source data](../../benchmarks/results/paper/README.md)
- [Representative workflow](REPRESENTATIVE_FIGURE.md)
- [Architecture](ARCHITECTURE_FIGURE.md)

| Page | Figure | PDF | Preview |
|---:|---|---|---|
| 1 | 외부 모델 비교 | [PDF](figures/01_model_comparison.pdf) | [PNG](figures/01_model_comparison.png) |
| 2 | Raw/refined·chirality 효과 | [PDF](figures/02_refinement_chirality.pdf) | [PNG](figures/02_refinement_chirality.png) |
| 3 | 리간드 복잡도별 성능 | [PDF](figures/03_ligand_complexity.pdf) | [PNG](figures/03_ligand_complexity.png) |
| 4 | 누적 성능: confidence 순위·생성 순서 | [PDF](figures/04_cumulative_success.pdf) | [PNG](figures/04_cumulative_success.png) |
| 5 | 생성 후보 수별 선택 성능 | [PDF](figures/05_pose_budget.pdf) | [PNG](figures/05_pose_budget.png) |
| 6 | Guidance·계산 예산별 성능 | [PDF](figures/06_guidance_budget.pdf) | [PNG](figures/06_guidance_budget.png) |
| 7 | Runtime·memory | [PDF](figures/07_runtime_memory.pdf) | [PNG](figures/07_runtime_memory.png) |
| 8 | Pocket cutoff·prior 민감도 | [PDF](figures/08_pocket_prior.pdf) | [PNG](figures/08_pocket_prior.png) |
| 9 | 학습 관련성: 서열·리간드–단백질 중복 | [PDF](figures/09_training_relatedness.pdf) | [PNG](figures/09_training_relatedness.png) |
| 10 | 리간드 유사도·성능 | [PDF](figures/10_ligand_similarity.pdf) | [PNG](figures/10_ligand_similarity.png) |
| 11 | 서열 유사도별 성능 | [PDF](figures/11_sequence_similarity.pdf) | [PNG](figures/11_sequence_similarity.png) |
| 12 | 보충 S1: 개선량과 95% 신뢰구간 | [PDF](figures/S1_paired_uncertainty.pdf) | [PNG](figures/S1_paired_uncertainty.png) |
| 13 | 보충 S2: Top-k 요약·near-native 밀도 | [PDF](figures/S2_candidate_ranking.pdf) | [PNG](figures/S2_candidate_ranking.png) |
| 14 | 보충 S3: 복잡도별 실패 분해 | [PDF](figures/S3_complexity_failures.pdf) | [PNG](figures/S3_complexity_failures.png) |
| 15 | 보충 S4: Confidence ranking·selection 진단 | [PDF](figures/S4_confidence_diagnostics.pdf) | [PNG](figures/S4_confidence_diagnostics.png) |
| 16 | 보충 S5: Confidence reliability·후보 밀도 | [PDF](figures/S5_confidence_reliability.pdf) | [PNG](figures/S5_confidence_reliability.png) |
| 17 | 보충 S6: 서열·리간드 동시 저유사도 subset | [PDF](figures/S6_stringent_subset.pdf) | [PNG](figures/S6_stringent_subset.png) |
| 18 | 보충 S7: Refinement PB 실패·전이 | [PDF](figures/S7_physical_validity.pdf) | [PNG](figures/S7_physical_validity.png) |
| 19 | 보충 S8: 데이터셋별 성공·rescue·선택 실패 구조 | [PDF](figures/S8_structure_examples.pdf) | [PNG](figures/S8_structure_examples.png) |
| 20 | 보충 S9: 로컬 baseline과 paired 신뢰구간 | [PDF](figures/S9_baseline_uncertainty.pdf) | [PNG](figures/S9_baseline_uncertainty.png) |
| 21 | 보충 S10: Fragment 생성 경로 | [PDF](figures/S10_fragment_trajectory.pdf) | [PNG](figures/S10_fragment_trajectory.png) |
| 22 | Rw 변경량과 신뢰구간 | [PDF](figures/S11_rw_operator_change.pdf) | [PNG](figures/S11_rw_operator_change.png) |
| 23 | Chirality 제외·ranking 병목 | [PDF](figures/S12_selector_bottleneck.pdf) | [PNG](figures/S12_selector_bottleneck.png) |

Solid segments are RMSD <2 Å and PB-valid. Hatched extensions are RMSD <2 Å and PB-invalid.

```bash
uv run python -m benchmarks.figures.paper --check
uv run python -m benchmarks.figures.paper --output outputs/paper_figures
uv run python scripts/package_paper_figures.py --check
```

The renderer uses versioned numerical records and reproduces no docking, PB evaluation or training. Saved-bank recollection requires the immutable study outputs and previous input tables; see `benchmarks.analysis.refresh_rw` and `benchmarks.analysis.rw_diagnostics`.
