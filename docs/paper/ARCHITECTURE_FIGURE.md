# EFF-Dock architecture figure

- [PDF](figures/Fig2_architecture.pdf)
- [Editable vector SVG](figures/Fig2_architecture.svg)
- [PNG preview](figures/Fig2_architecture.png)

Figure numbering is provisional. This model-only figure complements the
[representative workflow](REPRESENTATIVE_FIGURE.md). Graph construction,
post-refinement and training objectives are outside its scope.

## Manuscript caption

**EFF-Dock architecture and equivariant building blocks.**
**A**, The docking network embeds input features and applies six interaction
layers. A learned atom-vector head followed by Newton–Euler aggregation produces
fragment translation and angular velocities. A separately parameterized
four-layer confidence network maps candidate-pose features to invariants and
combines global and contact-aware pooling to predict RMSD and a pose-success
logit. Docking uses additive time and log-prior-scale embeddings; confidence uses
a fixed zero conditioning vector. **B**, Each interaction layer applies
blockwise equivariant RMS normalization, an edge-conditioned tensor product with
spherical harmonics through degree two, message activation, gated aggregation,
and an equivariant linear/activation/dropout block. The residual is added before
AdaLN. Scalar activation is elementwise SiLU; non-scalar channels are multiplied
by sigmoid gates computed from their invariant norms. **C**, Equivariant RMSNorm
normalizes each irrep block using the channel-mean squared irrep norm and applies
learned channel gains. The drawing illustrates a vector block; scalar and
rank-two blocks are normalized separately by the same rule. **D**, AdaLN first
applies RMSNorm. A linear projection of the condition produces scalar scale and
shift parameters and bounded multiplicative scales for non-scalar channels.
The latter act identically on every component of a channel, preserving its
direction. Bars and arrows are schematic illustrations, not measured activations.

## Panel definitions and implementation audit

| Panel | Depicted operation | Source |
|---|---|---|
| A | Independent docking/confidence models, embedding, readouts | [Docking model](../../src/effdock/models/effdock.py), [confidence model](../../src/effdock/confidence/model.py) |
| B | Pre-norm → tensor product → message activation → gated aggregation → linear/activation/dropout → residual → AdaLN | [EFFDockInteractionLayer](../../src/effdock/models/effdock.py), [GatedEquivariantConv / EquivariantBlock](../../src/effdock/models/equivariant.py) |
| C | RMS reduction per irrep block, learned channel gains | [EquivariantRMSNorm](../../src/effdock/models/equivariant.py) |
| D | RMSNorm, scalar affine modulation, bounded non-scalar modulation | [EquivariantAdaLN](../../src/effdock/models/equivariant.py) |

Released settings are taken from the [docking config](../../configs/train_early_time_ft_50k.yaml)
and [confidence config](../../configs/train_confidence_s50_raw_refined_100k.yaml).

### A. Model overview

Input features include chemistry, coordinates and supplied graph relations.
The diagram begins after graph construction. Scalar embedding is 208→384→384.
The equivariant state is 384×0e + 32×1o + 32×1e + 16×2e + 16×2o, giving
736 components and 480 invariant channels. Initial vector channels combine gated
coordinate displacements and, for docking fragments, a learned orientation mix.
Other non-scalar blocks initially vanish. Fragment size and conditioning enter
fragment scalar initialization. Time and log-prior-scale sinusoidal embeddings
each pass through a 128-wide MLP and are added, not concatenated.

Six independently parameterized interaction layers feed the docking head.
Ligand-atom states pass through a 736→544 linear/activation head, then linear 1o
and self-tensor-product 1o paths are summed. The resulting atom field is learned;
it is not the gradient of a physical energy. Fragment translation is its atom
mean. Torque is the sum of lever-arm cross products, and angular velocity is
obtained with the fragment inertia pseudo-inverse. Unit atom weights and a 1%
relative eigenvalue threshold define observable rotations. These velocities feed
the rigid SE(3) integrator shown in the separate workflow figure.

The confidence network uses four layers with independent parameters. Each
candidate is processed independently; there is no cross-candidate attention.
It uses a zero 128-wide condition and no fragment-frame input to the interaction
layers. Saved ligand/fragment hidden states can enter via a learned gate when
available; this optional branch is omitted from the overview.

Invariant node features contain scalars and non-scalar channel norms. Global
mean/max pools over four node types produce 4,096 features. A contact-aware atom
MLP combines 480 invariant features with 44 contact descriptors, then attention,
contact-maximum, atom-mean and atom-maximum pools produce 2,048 features. The
6,144-dimensional concatenation feeds a depth-three pose MLP with hidden width
512 and two outputs: log1p RMSD and a success logit. Displayed pRMSD is
`max(0, expm1(clamp(log1p_RMSD, -2, 5)))`; selection minimizes pRMSD among eligible
poses. The success logit is not the selector or a calibrated probability claim.
Auxiliary atom-level heads are omitted from this model overview.

### B. Interaction layer and activation

The exact outer order is `AdaLN(h + post_block(conv(pre_norm(h))))`.
The residual bypasses the message/post block, not AdaLN. The six docking layers
and four confidence layers are separately parameterized; repetition counts do
not denote tied weights.

A 1,020-wide edge vector contains 32 RBF values, edge/bond descriptors, distance
evolution, fragment-hop and local-frame features, contact chemistry, normalized
endpoint scalar features, and conditioning. Edge MLPs supply input/output radial
scales and aggregation gates. Tensor-product weights are shared across edge
types within a layer. Real spherical harmonics include degrees 0, 1 and 2.
Message activation occurs after output radial scaling, before aggregation.

Aggregation uses gate-normalized sums, edge-type-dependent distance decay and
additional non-scalar norm rescaling grouped by degree. It is not ordinary
softmax attention. These internal reductions are compressed into the gated
aggregation module. Edge scalar construction is also compressed; coordinates
and conditioning feed it in addition to the two inputs named in the drawing.

Equivariant linear maps mix multiplicity channels within compatible irreps.
For activation, even scalars receive SiLU. All non-scalar channel norms are
concatenated and passed through an MLP and sigmoid to produce per-channel gates:
`u_c' = g_c u_c`. Every component of a vector or tensor receives the same gate.
The message and post-linear activation modules use this pattern with separate
parameters. Dropout uses one mask per irrep channel, broadcast over its spatial
components; it is inactive at inference. Both released configs use dropout 0.1.

### C. Equivariant RMSNorm

Let b identify one (degree, parity) block, C_b its multiplicity, and h_{b,c}
its channel vector (one component for a scalar, 2ℓ+1 otherwise). For each node:

- `r_b = sqrt(mean_c ||h_{b,c}||² + epsilon)`, with `epsilon = 1e-6`.
- `hhat_{b,c} = a_{b,c} h_{b,c} / r_b`, with learned channel gains initialized to one.

The component norm sums over the irrep dimension; the outer mean is over channels
within the block. This is not one RMS over the entire 736-component state, a
batch statistic, or separate normalization of x/y/z. No mean is subtracted.
All components of a channel receive the same scalar multiplier. The pictured
arrows illustrate positive gains and a common rescaling; lengths are schematic.
Learned gains themselves are unconstrained and may change sign.

### D. Equivariant AdaLN

AdaLN has its own RMSNorm, distinct from the pre-message RMSNorm in panel B.
The condition passes through one linear projection, producing gamma_s, beta_s
and gamma_u. Feature and conditioning paths occupy separate sides of the stacked
modulation rows; dots mark branch points. The purple path carries conditioning
parameters, while the gray path carries normalized features. Normalized features
follow two branches:

- Scalars: `s' = (1 + gamma_s) * shat + beta_s`.
- Non-scalars: `u' = (1 + 0.1 * tanh(gamma_u)) * uhat`.

Non-scalar scales lie between 0.9 and 1.1; no vector/tensor offset is added.
The scale is shared across the 2ℓ+1 components of each channel. The scalar-bar
example illustrates a shift; the vector example illustrates a positive scale.
Zero initialization of the conditioning projection initially leaves the
RMS-normalized features unchanged, not the raw input unchanged. A zero confidence
condition need not imply zero modulation after training because the projection
has a learned bias. Equivariance here is with respect to joint SE(3)
transformations of the model inputs; the figure does not establish full-pipeline
reflection equivariance.

## Reproduction and use

```bash
python -m benchmarks.figures.architecture --output outputs/paper_figures
```

The [source specification](../../benchmarks/results/paper/architecture/spec.json)
records dimensions, normalization operations and source/config SHA-256 hashes.
Rendering fails if those sources change. It also checks dimension arithmetic,
text overlap, module padding and connector routing (unmarked intersections,
collinear overlaps, text/module intrusion and arrowheads without adequate shafts).
PDF/SVG are vector, SVG text is editable and
the PNG preview is 300 dpi. Exports are deterministic. At 180 mm width the
11 pt source body text becomes 6.50 pt; inspect the final manuscript scale.
No model execution or measured activation data are needed for this schematic.

The restrained functional-block styling was informed by Figure 1 of
[BA-Pred and RMSD-Pred](https://doi.org/10.1021/acs.jcim.5c02591).
This is an original drawing audited against EFF-Dock source code; no reference
panels, molecular images or model operations are reused.
