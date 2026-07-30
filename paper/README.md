# EFF-Dock paper

This directory is the canonical local source for the EFF-Dock manuscript.
It is intentionally based on the standard `article` class so that the content
can later be moved into a conference or journal template with minimal changes.

## Structure

- `main.tex`: manuscript entry point and shared macros.
- `sections/method.tex`: model and learning method.
- `sections/experimental_setup.tex`: data, training, selection, and evaluation contracts.
- `sections/results.tex`: frozen diagnostic results currently supported by the repository.
- `sections/declarations.tex`: venue-neutral availability, contribution, funding, and conflict sections.
- `sections/evidence_checklist.tex`: internal claim/evidence gaps; intentionally excluded from `main.tex`.
- `references.bib`: initial primary references.
- `figures/`: manuscript figures only.
- `tables/`: generated or hand-written table fragments.

## Local compilation

The repository-local Tectonic compiler is installed under `.venvs/latex`.
Compile from this directory with:

```bash
cd paper
../.venvs/latex/bin/tectonic main.tex --keep-logs --keep-intermediates
```

The manuscript was successfully compiled with Tectonic 0.17.0. The current
output is `main.pdf`; Tectonic runs BibTeX and the required reruns
automatically.

## Moving to Overleaf

Zip the contents of this directory, not the parent repository, and upload the
archive as a new Overleaf project. Keep all paths relative to `paper/`.

## Scientific boundaries to resolve before submission

1. Select the target venue and apply its official template.
2. Freeze a publishable PLINDER split that strictly excludes all external
   benchmark ligand SMILES and rerun the leakage audit.
3. Trace every headline result to the exact checkpoint, config, seed, and code
   state used to produce it.
4. Add matched literature baselines under the same pocket and evaluation
   protocol; do not compare oracle-pocket redocking numbers with blind docking.
5. Add uncertainty across independent seeds or clearly label the current
   frozen run as a single-run diagnostic.
6. Verify the final model parameter count and compute/runtime statistics.
7. Complete the acknowledgement and funding statements.
