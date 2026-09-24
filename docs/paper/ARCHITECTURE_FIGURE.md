# EFF-Dock architecture figure

- [PDF](figures/Fig2_architecture.pdf)
- [Editable vector SVG](figures/Fig2_architecture.svg)
- [PNG preview](figures/Fig2_architecture.png)

Figure numbering is provisional. This four-panel figure complements the
[representative workflow](REPRESENTATIVE_FIGURE.md); it describes model operations,
not a measured trajectory, benchmark result or component-ablation result.

## Manuscript caption

**EFF-Dock architecture and pose-confidence readout.**
**A**, Ligand atoms, rigid fragments, receptor atoms and virtual residue nodes form
one heterogeneous graph. Static chemical and membership edges are augmented with
protein–ligand contacts within 5 Å at each forward pass. Node features initialize
scalar, vector and rank-2 channels; time and prior scale condition the docking
network. **B**, Each of six docking interaction layers applies equivariant
pre-normalization, a shared tensor product with spherical harmonics through
ℓ = 2, edge-conditioned radial scaling and gated aggregation. An equivariant
linear/activation/dropout block precedes residual addition and conditioned
normalization. **C**, Linear and self-tensor-product heads produce a learned atom
vector field. Newton–Euler aggregation maps this field to fragment translation
and angular velocities, preserving intrafragment geometry. The objective combines
translation and observable-rotation flow matching with atom-velocity and
interfragment distance-geometry losses. **D**, A separately parameterized
four-layer confidence network processes each candidate graph independently.
Invariant features feed global node-type pooling and contact-aware atom pooling;
their concatenation predicts pose RMSD and a success logit. Atom displacement and
success heads provide additional supervision. Selection minimizes predicted RMSD
among eligible candidates. Confidence uses a fixed zero conditioning vector and
no fragment-frame input to its interaction layers; saved ligand states can enter
through a learned gate when available. Graphs and motion arrows are schematic.

## Panel definitions and implementation audit

| Panel | Depicted operation | Implementation |
|---|---|---|
| A | Four node types; scalar embedding; geometric initialization; dynamic contacts | [EFFDockNodeEmbedding / EFFDock](../../src/effdock/models/effdock.py), [graph schema](../methods/01_graph_features.md) |
| B | Pre-norm → tensor-product messages → gate-normalized aggregation → post block → residual addition → AdaLN | [EFFDockInteractionLayer](../../src/effdock/models/effdock.py), [GatedEquivariantConv / EquivariantAdaLN](../../src/effdock/models/equivariant.py) |
| C | 736→544 head, linear + self TP to 1o, Newton–Euler readout, active loss weights | [Docking head](../../src/effdock/models/effdock.py), [loss implementation](../../src/effdock/training/losses.py), [released docking config](../../configs/train_early_time_ft_50k.yaml) |
| D | Independent four-layer scorer, invariant/contact features, global/contact pooling and heads | [Confidence model](../../src/effdock/confidence/model.py), [released confidence config](../../configs/train_confidence_s50_raw_refined_100k.yaml) |

### A. Graph and representation

The seven ligand atoms, two fragments and small receptor graph are schematic,
not a real complex or a reconstruction of the molecular example in Figure 1.
Circles denote atoms, diamonds fragment nodes and rounded squares virtual residue
nodes. Solid lines illustrate chemical/coarse relations; dashed membership and
cross-interface contact edges are distinguished by context. The picture samples
relations rather than enumerating all ten directed edge types.

The scalar input embedding is 208→384→384, with SiLU between linear layers.
The state is 384×0e + 32×1o + 32×1e + 16×2e + 16×2o: 736 components and
480 invariant channels (scalars plus non-scalar channel norms). The initial 1o
features combine gated displacements with a learned fragment-orientation mix;
1e/2e/2o channels initially vanish. Fragment size and conditioning additionally
enter fragment scalar initialization. The diagram compresses these initialization
branches into the state block. Time and log-prior-scale embeddings each map to
128 dimensions and are added, not concatenated to 256.

### B. Interaction layer

The 1,020-dimensional edge input contains 32 RBF values, 16 edge-type channels,
20 bond-attribute channels, 16 evolving-distance channels, 8 fragment-hop channels,
16 source-frame channels, 16 chemical-pair channels, two 384-wide endpoint scalar
states and the 128-wide condition. Edge features feed a 128-wide radial trunk
and separate gate network. Input/output radial scalings surround a shared-weight
tensor product with real spherical harmonics of degree 0/1/2. Aggregation uses
normalized scalar and per-channel gates, edge-type distance decay and non-scalar
norm rescaling; it is not a ten-network mixture or ordinary softmax attention.
The edge-MLP and spherical-harmonic arrows indicate inputs to this message block;
the detailed gate branch is compressed into the aggregation module.

The exact outer order is `AdaLN(h + post_block(conv(pre_norm(h))))`.
The residual bypasses the pre-norm/message/post block, not AdaLN. AdaLN applies
RMS normalization followed by conditional scalar affine modulation and non-scalar
gating. The ×6 annotation denotes six separately parameterized layers, not six
applications of one tied-weight module. O(3) irrep notation does not establish
reflection equivariance of the full stereochemical input pipeline; the docking
contract is joint SE(3) transformation of all geometric inputs.

### C. Readout and training

The 736-dimensional ligand-atom state is projected to 544 channels (192 scalars
and unchanged non-scalar widths), activated, and read through a linear 1o path
plus a self-tensor-product 1o path. The self-TP weights are zero initialized.
The output is a **learned atom vector field**, not a physical force obtained as
an energy gradient; Newton–Euler is the aggregation geometry.

For each fragment, translation is the mean atom field, torque is the sum of
lever-arm cross products, and angular velocity is the inertia pseudo-inverse
applied to torque. The inertia uses unit atom weights and an eigenvalue threshold
of 1% of the fragment maximum; unobservable rotations are removed. The same
observable-subspace projector is applied to angular flow targets. All vectors
are in the receptor coordinate frame and flow time is dimensionless.

The released objective is L_v + 8 L_omega + 0.3 L_atom + 3 L_DG. The atom term
supervises instantaneous rigid-body velocities, not coordinates. DG supervises
interfragment distances of the one-step predicted endpoint with a t²-weighted
per-complex reduction. These are training terms; inference updates fragment
transforms without reference coordinates. Energy-based post-refinement is a
separate downstream operation already shown in the representative workflow and
is not depicted as a neural layer here.

### D. Confidence readout

Each candidate repeats the receptor graph and rebuilds contacts at 5 Å. There is
no message passing or attention across different candidate poses. Its four
interaction layers use independent parameters from docking. They receive a zero
128-dimensional condition and no fragment rotation matrices, unlike the docking
layers in panel B. Saved ligand/fragment states, when present, are added through
a learned scalar gate initialized to 0.25. The scorer also operates without them.

The 480 invariant channels are concatenated with 44 atom-contact features and
projected to 512 dimensions. Atom heads predict log1p displacement and a success
logit. Contact pooling concatenates attention-weighted contact features, maximum
contact features, mean atom features and maximum atom features: 4×512 = 2,048.
A separate projection of invariant node states supplies mean/max pools for each
of four node types: 8×512 = 4,096. Concatenating both routes gives 6,144 dimensions
for LayerNorm and a depth-3 pose MLP with hidden width 512 and two outputs.
Internal LayerNorm/activation/dropout operations are compressed in the pooling
and MLP boxes to keep the panel readable.

The pose outputs are log1p RMSD and a success logit. Displayed predicted RMSD is
`max(0, expm1(clamp(log1p_RMSD, -2, 5)))`. The success output is an auxiliary
prediction, not the deployed selector score or a claim of calibrated probability.
Reference RMSD/success labels exist during confidence training, not at inference.

## Reproduction and use

```bash
python -m benchmarks.figures.architecture --output outputs/paper_figures
```

The [source specification](../../benchmarks/results/paper/architecture/spec.json)
records dimensions and source/config SHA-256 hashes. Rendering fails if those
files change, requiring another implementation audit. The renderer checks shape
arithmetic and text bounds/overlap. PDF/SVG are fully vector, with editable SVG
text; the preview is 300 dpi. Exports are deterministic. The source width is
304.8 mm; at 180 mm the 11 pt body text becomes 6.50 pt. Place at two-column
width and inspect the final journal proof. Existing figures, data and the combined
benchmark PDF are unchanged; no model execution, training or evaluation is needed.


## Visual design reference

The visual organization was checked against Figure 1 of
[BA-Pred and RMSD-Pred](https://doi.org/10.1021/acs.jcim.5c02591)
([open-access figure](https://pmc.ncbi.nlm.nih.gov/articles/PMC13080981/figure/fig1/)).
Only layout principles inform this original drawing: compact functional blocks,
thin connectors, and a clear separation between graph inputs, internal layers,
and readout. No panels, molecular images, or model operations were reused.
EFF-Dock operations are independently audited against the source files above.

Plain panel letters and open whitespace replace enclosing rounded cards.
Pale colors distinguish geometric inputs, equivariant operations, motion readout,
and conditioning/pooling; they do not encode quantitative values. Channel widths
and implementation details remain in this document rather than being repeated
inside every module. Solid arrows indicate feature or vector flow; the dashed
atom-head branch denotes auxiliary supervision. Training losses are separated
from the inference readout in panel C.
