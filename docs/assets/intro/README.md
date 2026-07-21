# EFF-Dock introduction visuals

These slide-ready figures are generated from the project's pinned RDKit
environment and frozen local Astex benchmark snapshot.

```bash
.venv/bin/python scripts/figures/make_intro_figures.py
```

- `fragmentation_examples.png`: one simple 2D ligand-to-fragments panel for
  1T46–STI/imatinib, decomposed with
  `effdock.preprocess.fragments.decompose_fragments` (not BRICS/RECAP).
- `protein_pocket_context.png`: the 1T46–STI/imatinib receptor, explicit
  reference pocket center, 10 Å residue-aware crop, and fragment-colored ligand.
- `intro_visuals_manifest.json`: source IDs, versions, counts, coordinates, and
  the pocket-information boundary.

The pocket panel is explicitly a reference-defined holo visualization. It
illustrates pocket-conditioned docking and must not be presented as blind or
target-independent pocket detection.
