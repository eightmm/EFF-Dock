# EFF-Dock architecture figure

- [PDF](figures/Fig2_architecture.pdf)
- [Editable vector SVG](figures/Fig2_architecture.svg)
- [PNG preview](figures/Fig2_architecture.png)

Figure numbering is provisional. This model-only figure complements the
[representative workflow](REPRESENTATIVE_FIGURE.md). Graph construction,
post-refinement and training objectives are outside its scope.

## Manuscript caption

**EFF-Dock architecture and equivariant building blocks.**
**A**, Six docking interaction layers feed a ligand-atom head whose linear and
self-tensor-product outputs are added before Newton–Euler aggregation yields
fragment translational and angular velocities. The separately parameterized
four-layer confidence network uses scalar features and irrep-channel norms.
Global pooled features and a contact-aware readout are concatenated before the
pose MLP predicts RMSD and a pose-success logit. Pose–protein contact descriptors
enter both the atom MLP and the contact readout. A fresh docking-network forward
pass at t = 1 on each scored candidate supplies ligand hidden states, weighted
by a learned scalar coefficient α and added to the confidence embedding.
Docking conditions on additive time and log-prior-scale embeddings; confidence
uses c = 0. **B**, Each layer applies pre-message RMSNorm, input radial scaling,
a shared tensor product with spherical harmonics through degree two, output
radial scaling, activation and gate-normalized aggregation. Normalized endpoint
scalars, condition and edge descriptors feed both the radial MLP and the gate
MLP. Sigmoid gates include an edge-type-dependent distance decay. The ordered
post-message transform is equivariant linear → activation → channel dropout.
The original input follows one identity skip, added before AdaLN. **C**,
RMSNorm acts separately within each (degree, parity) block. AdaLN applies its own
RMSNorm and condition-dependent scalar affine modulation or bounded non-scalar
scaling. Exact reductions and modulation equations are given below. **D**, Even
scalars receive SiLU; invariant non-scalar norms pass through an MLP and sigmoid,
and the resulting channel gates g multiply the original vector/tensor input.
Dots mark branches, + denotes addition, × denotes multiplication and “concat”
denotes concatenation. α is a learned scalar parameter; g is feature-dependent.

## Panel definitions and implementation audit

| Panel | Depicted operation | Source |
|---|---|---|
| A | Separately parameterized docking/confidence models, embedding, readouts | [Docking model](../../src/effdock/models/effdock.py), [confidence model](../../src/effdock/confidence/model.py) |
| B | Pre-norm → input scale → tensor product → output scale → activation → aggregation → post transform → residual → AdaLN | [EFFDockInteractionLayer](../../src/effdock/models/effdock.py), [GatedEquivariantConv / EquivariantBlock](../../src/effdock/models/equivariant.py) |
| C | RMSNorm per irrep block with learned gains; AdaLN (own RMSNorm, scalar affine, bounded non-scalar scale) | [EquivariantRMSNorm, EquivariantAdaLN](../../src/effdock/models/equivariant.py) |
| D | Scalar SiLU; norm-derived sigmoid gating of non-scalar input | [EquivariantActivation](../../src/effdock/models/equivariant.py) |

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
layers. The released dataset and inference runtime supply ligand-atom and fragment
hidden states from a docking-model forward pass at t = 1 for each candidate.
These states are multiplied by a learned scalar coefficient α (initialized to 0.25) and
added only at ligand-atom and fragment slots of the confidence embedding. The figure
draws it as a separate t = 1 docking-network block feeding the × α and + nodes;
it is not a wire from an intermediate generation step. The docking network is evaluated on the
scored pose, including a refined pose when applicable. The model class permits
omitting this branch when saved states are absent or explicitly disabled, but
the released scoring path supplies them. See the
[feature extractor](../../src/effdock/confidence/features.py),
[inference runtime](../../src/effdock/confidence/runtime.py) and
[dataset](../../src/effdock/confidence/dataset.py).

Invariant node features contain scalars and non-scalar channel norms. Global
mean/max pools over four node types produce 4,096 features. A contact-aware atom
MLP combines 480 invariant features with 44 contact descriptors, then attention,
contact-maximum, atom-mean and atom-maximum pools produce 2,048 features. The
6,144-dimensional concatenation (“concat” in panel A) feeds a depth-three pose MLP
with hidden width 512 and two outputs: log1p RMSD and a success logit. Displayed pRMSD is
`max(0, expm1(clamp(log1p_RMSD, -2, 5)))`; selection minimizes pRMSD among eligible
poses. The success logit is not the selector or a calibrated probability claim.
“Scalars and irrep norms” denotes a concatenated invariant feature vector, not
an elementwise sum. The “Contact readout” box includes contact projection,
attention/max pooling and atom mean/max pooling; the arrow into each of the two
contact-branch boxes marks a separate descriptor concatenation. “Logit” in the
pose head is the auxiliary pose-success logit. Auxiliary atom-level heads are
omitted from this model overview.

### B. Interaction layer

The exact outer order is `AdaLN(h + post_block(conv(pre_norm(h))))`.
The identity skip carries the original h around pre-normalization and the
message/post block, ending at the + node before AdaLN. There is exactly one
additive residual per interaction layer; neither the post block nor AdaLN has
an additional residual. The output arrow follows AdaLN. The six docking layers
and four confidence layers are separately parameterized; repetition counts do
not denote tied weights.

A 1,020-wide edge vector contains 32 RBF values, edge/bond descriptors, distance
evolution, fragment-hop and local-frame features, contact chemistry, normalized
endpoint scalar features, and conditioning. One radial trunk produces both input and output scales (each 1 + a learned
delta); a separate gate MLP produces the aggregation gates. Tensor-product weights are shared across edge
types within a layer. Real spherical harmonics include degrees 0, 1 and 2.
Message activation occurs after output radial scaling, before aggregation.

Aggregation uses gate-normalized sums, edge-type-dependent distance decay and
additional non-scalar norm rescaling grouped by degree. It is not ordinary
softmax attention. These internal reductions are compressed into the gated
aggregation module. The hats on ŝ_i and ŝ_j mark normalized endpoint scalars;
they are concatenated with the condition c and edge descriptors before the two
MLP branches. The shared input branch supplies those same descriptors to both
MLPs; it does not feed the gate MLP with the radial MLP's output. The spherical
harmonic argument appears inside the tensor-product block to avoid a crossing
external wire. “Shared” refers to tensor-product weights shared across edge
types within a layer, not weights tied across successive interaction layers.

Equivariant linear maps mix multiplicity channels within compatible irreps.
The message activation and the activation inside the combined post-message
box use the operation in panel D with independent parameters. Channel dropout uses one mask per irrep channel, broadcast over its
spatial components; it is inactive at inference. Both released configs use 0.1.

### C. Equivariant RMSNorm and AdaLN

#### RMSNorm

Let b identify one (degree, parity) block, C_b its multiplicity, and h_{b,c}
its channel vector (one component for a scalar, 2ℓ+1 otherwise). The schematic
uses RMS_b(h) = r_b. For each node:

- `r_b = sqrt(mean_c ||h_{b,c}||² + epsilon)`, with `epsilon = 1e-6`.
- `hhat_{b,c} = a_{b,c} h_{b,c} / r_b`, with learned channel gains initialized to one.

The component norm sums over the irrep dimension; the outer mean is over channels
within the block. This is not one RMS over the entire 736-component state, a
batch statistic, or separate normalization of x/y/z. No mean is subtracted.
All components of a channel receive the same scalar multiplier. Learned gains are unconstrained and may change sign.

#### AdaLN

AdaLN has its own RMSNorm, distinct from the pre-message RMSNorm in panel B.
The condition passes through one linear projection, producing gamma_s, beta_s
and gamma_u. Feature and conditioning paths occupy separate sides of the stacked
modulation rows; dots mark branch points. The purple path carries conditioning
parameters, while the gray path carries normalized features. Normalized features
follow two branches:

- Scalars: `s' = (1 + gamma_s) * shat + beta_s`.
- Non-scalars: `u' = (1 + 0.1 * tanh(gamma_u)) * uhat`.

Non-scalar scales lie between 0.9 and 1.1; no vector/tensor offset is added.
The scale is shared across the 2ℓ+1 components of each channel. Zero initialization
of the conditioning projection initially leaves the
RMS-normalized features unchanged, not the raw input unchanged. A zero confidence
condition need not imply zero modulation after training because the projection
has a learned bias. Equivariance here is with respect to joint SE(3)
transformations of the model inputs; the figure does not establish full-pipeline
reflection equivariance.

### D. Equivariant activation

Even scalars receive elementwise SiLU. All non-scalar channel norms are
concatenated and passed through an MLP and sigmoid to produce per-channel gates.
The bypass in panel D carries the original vector/tensor input to the product:
`u_c' = g_c u_c`; s and u denote even-scalar and vector/tensor channels. The ×
node denotes channelwise multiplication, broadcasting
one gate over all 2ℓ+1 components of each channel. Norms are used to calculate
the gate, not substituted for the original input. The message activation and
post-linear activation use separate parameters; their panel-D references denote
the same operation type, not tied weights.

## Reproduction and use

```bash
python -m benchmarks.figures.architecture --output outputs/paper_figures
```

The [source specification](../../benchmarks/results/paper/architecture/spec.json)
records dimensions, normalization operations, the ligand-state input and
source/config SHA-256 hashes, including feature extraction and scoring code.
Rendering fails if those sources change. It also checks dimension arithmetic,
text overlap, module padding and connector routing (unmarked intersections,
collinear overlaps, text/module intrusion and arrowheads without adequate shafts).
PDF/SVG are vector, SVG text is editable and
the PNG preview is 300 dpi. Exports are deterministic. At 180 mm width the
12.5 pt source body text becomes 7.38 pt. A 180 mm raster proof was inspected;
math subscripts follow standard typesetting and are smaller than body text.
No model execution or measured activation data are needed for this schematic.

The restrained functional-block styling was informed by Figure 1 of
[BA-Pred and RMSD-Pred](https://doi.org/10.1021/acs.jcim.5c02591).
This is an original drawing audited against EFF-Dock source code; no reference
panels, molecular images or model operations are reused.
