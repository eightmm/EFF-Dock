# Architecture schematic specification

`spec.json` records the audited released-model dimensions, active loss weights
and SHA-256 hashes of the implementation/configuration files. This is an
architecture specification, not experimental data. The renderer verifies source
identity and dimension arithmetic before exporting one four-panel vector figure.

- [Figure and manuscript caption](../../../../docs/paper/ARCHITECTURE_FIGURE.md)
- Renderer: `python -m benchmarks.figures.architecture --output outputs/paper_figures`

The graph and fragment-motion geometry are schematic; no model was run to create
this illustration. Detailed branches omitted from the drawing are documented in
the caption companion.
