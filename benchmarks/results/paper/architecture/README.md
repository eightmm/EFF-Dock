# Architecture schematic specification

`spec.json` records released-model dimensions, RMSNorm/AdaLN operations and
SHA-256 hashes of the implementation/configuration files. This is an architecture
specification, not experimental data. The renderer verifies source identity and
dimension arithmetic before exporting one four-panel vector figure.

- [Figure and manuscript caption](../../../../docs/paper/ARCHITECTURE_FIGURE.md)
- Renderer: `python -m benchmarks.figures.architecture --output outputs/paper_figures`

Panels show the model overview, horizontal interaction layer, convolution
detail and normalization/activation. The overview shows the atom-head sum, gated ligand-state input
and concatenated global/contact pooling, including both contact-descriptor
injections. The Newton–Euler readout expands mean and torque/inertia branches;
feature degrees ℓ = 0, 1, 2 and both AdaLN outputs are explicit. The layer shows its single additive identity skip before AdaLN;
the separate convolution panel expands input/output radial scaling and the
gate MLP, sigmoid and distance-decay product. Post-message linear, activation
and dropout are separate boxes in the interaction layer.
No model was run. Detailed branches omitted from the drawing are documented in
the caption companion. Graph construction and training objectives are outside
this figure's scope.
