# Prism manuscript reference package

[Download the ZIP](prism_figure_reference.zip) for the combined 21-page figure
PDF, individual PDFs, English captions, source mapping and editable LaTeX.
`methods.tex` contains the current model, loss, refinement, selection and
relatedness equations. `main.tex` includes the methods followed by
`figure_captions.tex`; the separate `paper_figures.pdf` remains figures only.
ZIP paths are self-contained for compilation.

Use XeLaTeX or LuaLaTeX with `amsmath`, `amssymb`, `graphicx`, `fontspec` and
the TeX-distributed Latin Modern font files. For an existing Prism manuscript,
copy `methods.tex` and use `\input{methods.tex}` with `amsmath` and `amssymb`
rather than replacing the main
manuscript. Review wording, numbering, placements and bibliography keys.
Detailed typed-energy gates and parameter tables remain in the repository
[methods](../../methods/README.md).

Current sources: [captions](../FIGURE_CAPTIONS.md), [manifest](../manifest.json),
[page index](../README.md), and [numerical reproduction](../../../benchmarks/results/paper/README.md).
The manifest's numerical inputs refer to the repository; they are not required
to compile the figure/methods ZIP. These files are manuscript working materials.

The assembled ZIP compiled successfully with Tectonic 0.17.0 (XeTeX) on
2026-09-23, including all methods and figure blocks, with no TeX warnings.
The font files are resolved from the TeX distribution rather than a
system-installed font family. LuaLaTeX and the Prism upload interface were
not separately exercised.
