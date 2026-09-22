# Manuscript figures

Working figures and captions for writing the EFF-Dock manuscript. Figure
numbering and placement may change during writing.

- [Combined PDF](paper_figures.pdf): 14 pages.
- [Figure captions](FIGURE_CAPTIONS.md): English captions and author notes.
- [Sequence-relatedness definitions](RELATEDNESS.md)
- [Prism package](prism/prism_figure_reference.zip): PDFs, equations and LaTeX reference blocks.
- [Editable methods](prism/methods.tex)
- [Numerical source data and reproduction](../../benchmarks/results/paper/README.md)
- [Source and caption manifest](manifest.json)

## Figures

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

Pages 1–11 are currently main figures; pages 12–14 are supplementary.
PB-valid success is solid; only RMSD-passing but PB-invalid portions are hatched.

All 14 selected figures can be rendered using the versioned numerical inputs:

```bash
uv run python -m benchmarks.figures.paper --check
uv run python -m benchmarks.figures.paper --output outputs/paper_figures
```

The renderer produces PDF/PNG pairs and a long-form source-data CSV. It verifies
hashes, counts, repeat statistics, outcome joins and training-witness membership.
This rebuild starts from frozen numerical records; it does not rerun docking,
PB evaluation, sequence alignment or bootstrap sampling. Historical collection
helpers still require separately supplied pose banks. Older drafts and internal
verification logs remain local.

Replace the selected figure files with new analysis outputs before running
`uv run python scripts/package_paper_figures.py` to rebuild the combined PDF,
manifest and Prism ZIP. Use `--check` to verify the existing package. Packaging
requires Poppler's `pdfinfo`, `pdftotext` and `pdfunite` on PATH and does not
rerun scientific analyses. LaTeX blocks need manuscript-specific
numbering and bibliography keys. See [Prism notes](prism/README.md) for compilation.
