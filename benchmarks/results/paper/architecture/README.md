# Architecture schematic specification

`spec.json` records released-model dimensions, RMSNorm/AdaLN operations and
SHA-256 hashes of the implementation/configuration files. This is an architecture
specification, not experimental data. The renderer verifies source identity and
dimension arithmetic before exporting one four-panel vector figure.

- [Figure and manuscript caption](../../../../docs/paper/ARCHITECTURE_FIGURE.md)
- Renderer: `python -m benchmarks.figures.architecture --output outputs/paper_figures`

Panels show the model overview, interaction layer, RMSNorm/AdaLN operations
and activation. The overview shows the atom-head sum, gated ligand-state input
and concatenated global/contact pooling, including both contact-descriptor
injections. The layer expands input/output radial scaling and shows its single additive
identity skip before AdaLN.
No model was run. Detailed branches omitted from the drawing are documented in
the caption companion. Graph construction and training objectives are outside
this figure's scope.
