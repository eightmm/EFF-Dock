"""Package the current manuscript selection without rasterizing PDF plots."""

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "docs/paper"
OUT = BASE / "figure_gallery"
SELECTED = [
    ("recent_benchmarks", "07_recent_model_comparison"),
    ("20260919", "01_stage_ablation"),
    ("20260920", "complexity_overview"),
    ("combined_panels", "pose_order_comparison"),
    ("20260920", "pose_budget_refined"),
    ("20260919", "02_guidance_budget"),
    ("20260919", "05_runtime_memory"),
    ("20260919", "03_pocket_prior_robustness"),
    ("combined_panels", "training_relatedness"),
    ("20260919", "06_training_exposure"),
    ("sequence_overlap", "sequence_performance"),
    ("manuscript_support", "paired_uncertainty"),
    ("20260919", "04_candidate_bottleneck"),
    ("20260920", "complexity_failure_explanation"),
]
MAIN_COUNT = 11


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


TITLES = [
    "외부 모델 비교",
    "Raw/refined·chirality 효과",
    "리간드 복잡도별 성능",
    "누적 성능: confidence 순위·생성 순서",
    "생성 후보 수별 선택 성능",
    "Guidance·계산 예산별 성능",
    "Runtime·memory",
    "Pocket cutoff·prior 민감도",
    "학습 관련성: 서열·리간드–단백질 중복",
    "리간드 유사도·성능",
    "서열 유사도별 성능",
    "보충 S1: 개선량과 95% 신뢰구간",
    "보충 S2: Top-k 요약·near-native 밀도",
    "보충 S3: 복잡도별 실패 분해",
]


def main():
    files = [BASE / directory / (name + ".pdf") for directory, name in SELECTED]
    assert len(files) == len(set(files)) == len(TITLES)
    entries = []
    source_text = []
    for page, (pdf, title) in enumerate(zip(files, TITLES, strict=True), 1):
        info = subprocess.check_output(["pdfinfo", str(pdf)], text=True)
        count = next(
            line.split(":", 1)[1].strip() for line in info.splitlines() if line.startswith("Pages:")
        )
        assert count == "1", pdf
        source_text.append(
            subprocess.check_output(["pdftotext", "-layout", str(pdf), "-"], text=True)
        )
        entries.append(
            dict(
                page=page,
                title=title,
                section="main" if page <= MAIN_COUNT else "supplement",
                source=str(pdf.relative_to(ROOT)),
                sha256=digest(pdf),
            )
        )
    final = OUT / "paper_figures.pdf"
    temporary = OUT / "paper_figures.build.pdf"
    subprocess.run(["pdfunite", *map(str, files), str(temporary)], check=True)
    combined = subprocess.check_output(["pdftotext", "-layout", str(temporary), "-"], text=True)
    assert combined == "".join(source_text), "Merged PDF text differs from source pages"
    temporary.replace(final)
    manifest = dict(
        pdf=str(final.relative_to(ROOT)),
        sha256=digest(final),
        pages=entries,
        excluded="Historical PLINDER community/pocket comparisons; sequence ECDF redundant with composition. Superseded combined PDFs were removed.",
        verification=f"{len(files)} unique single-page vector PDFs; merged text equals ordered source text",
    )
    (OUT / "paper_figures_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    lines = [
        "# 논문 그림 통합본",
        "",
        "**현재 사용할 파일: [paper_figures.pdf](paper_figures.pdf)**",
        "",
        f"핵심 그림 {MAIN_COUNT}개(1–{MAIN_COUNT}페이지)와 보충 진단 {len(files) - MAIN_COUNT}개({MAIN_COUNT + 1}–{len(files)}페이지)를 함께 묶은 단일 PDF.",
        "",
        "| 페이지 | 내용 |",
        "|---:|---|",
    ]
    lines += [f"| {r['page']} | {r['title']} |" for r in entries]
    lines += [
        "",
        "기존 pocket/community 비교와 서열 구성비에 중복되는 서열 누적 분포는 제외했다. 리간드 유사도 분석은 별도 정보를 제공하므로 유지했다.",
        "현재 통합본은 위 파일 하나다. 원본 분석·개별 그림은 재현용으로 보존했다. PB-valid는 단색, RMSD 통과·PB-invalid 구간만 빗금이다.",
        "",
        "서열 점수/동일 학습 샘플 교집합 정의: [캡션](../sequence_overlap/CAPTIONS.md). 확정 leakage 또는 동일 pocket을 뜻하지 않는다.",
        "",
        "페이지별 영문 캡션 및 Prism 자료: [캡션 문서](../prism/FIGURE_CAPTIONS.md) · [Prism 패키지](../prism/prism_figure_reference.zip).",
    ]
    (OUT / "PDF_EXPORTS.md").write_text("\n".join(lines) + "\n")
    print(f"Created {final}: {len(files)} verified pages")


if __name__ == "__main__":
    main()
