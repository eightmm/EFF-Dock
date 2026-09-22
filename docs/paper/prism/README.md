# Prism용 그림·캡션 참고 자료

- `FIGURE_CAPTIONS.md`: 현재 PDF 14페이지별 영문 캡션 + 한국어 저자 메모.
- `figure_captions.tex`: 본문11개/보충3개 LaTeX figure 블록.
- `captions.json`: 페이지, 원본 파일, 해시, 제목, 캡션, label의 정확한 연결.
- `prism_figure_reference.zip`: 위 문서, main.tex, figure_captions.tex와 현재14개 개별 PDF를 묶은 독립 참고 패키지.

ZIP 내부 main.tex는 XeLaTeX/LuaLaTeX 기준의 참고용 문서다. 논문 본문에는
필요한 figure 블록을 옮기고 최종 figure 번호/배치를 조정한다. S1–S3 전환은
참고용 문서에 적용되어 있으므로 기존 보충자료 파일에 넣을 때 중복 적용하지 않는다.
로컬에 XeLaTeX/LuaLaTeX가 없어 실제 TeX 컴파일은 실행하지 않았다. 구조·파일 경로·ZIP 무결성은 검증했다.
이 작업은 로컬 참고 자료 작성이며 Prism에 업로드하거나 논문을 게시하지 않았다.

캡션의 설명과 다른 데이터를 임의로 추가하지 않는다. 특히 PB-invalid 빗금,
RMSD-only oracle, confidence ranking과generation prefix 차이, query-normalized
sequence identity, SD 대95% CI의 구분을 유지한다. 문헌 cite key는 실제
프로젝트 bibliography에 맞게 추가해야 한다. 최신 PDF 해시는 captions.json에 있다.
