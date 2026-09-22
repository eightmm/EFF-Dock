# 논문 그림 통합본

**현재 사용할 파일: [paper_figures.pdf](paper_figures.pdf)**

핵심 그림 11개(1–11페이지)와 보충 진단 3개(12–14페이지)를 함께 묶은 단일 PDF.

| 페이지 | 내용 |
|---:|---|
| 1 | 외부 모델 비교 |
| 2 | Raw/refined·chirality 효과 |
| 3 | 리간드 복잡도별 성능 |
| 4 | 누적 성능: confidence 순위·생성 순서 |
| 5 | 생성 후보 수별 선택 성능 |
| 6 | Guidance·계산 예산별 성능 |
| 7 | Runtime·memory |
| 8 | Pocket cutoff·prior 민감도 |
| 9 | 학습 관련성: 서열·리간드–단백질 중복 |
| 10 | 리간드 유사도·성능 |
| 11 | 서열 유사도별 성능 |
| 12 | 보충 S1: 개선량과 95% 신뢰구간 |
| 13 | 보충 S2: Top-k 요약·near-native 밀도 |
| 14 | 보충 S3: 복잡도별 실패 분해 |

기존 pocket/community 비교와 서열 구성비에 중복되는 서열 누적 분포는 제외했다. 리간드 유사도 분석은 별도 정보를 제공하므로 유지했다.
현재 통합본은 위 파일 하나다. 원본 분석·개별 그림은 재현용으로 보존했다. PB-valid는 단색, RMSD 통과·PB-invalid 구간만 빗금이다.

서열 점수/동일 학습 샘플 교집합 정의: [캡션](../sequence_overlap/CAPTIONS.md). 확정 leakage 또는 동일 pocket을 뜻하지 않는다.

현재 페이지별 영문 논문 캡션과 Prism용 자료: [캡션 문서](../prism/FIGURE_CAPTIONS.md) · [Prism 참고 패키지](../prism/prism_figure_reference.zip).
