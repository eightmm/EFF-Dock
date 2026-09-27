# EFF-Dock architecture figure

- [PDF](figures/Fig2_architecture.pdf)
- [Editable vector SVG](figures/Fig2_architecture.svg)
- [PNG preview](figures/Fig2_architecture.png)

Figure numbering is provisional. This model-only figure complements the
[representative workflow](REPRESENTATIVE_FIGURE.md). Graph construction,
post-refinement and training objectives are outside its scope.

Source-review status (2026-09-27): the layer/readout topology and panel-D
equations match the implementation. A rotation-indexing discrepancy in docking
initialization was tested using the released checkpoint; see the measured
rotation diagnostic below. This distinguishes exact equivariance from empirical
docking performance and does not invalidate the existing benchmark measurements.

## Manuscript caption

**EFF-Dock architecture and equivariant building blocks.**
**A**, Six docking interaction layers feed a ligand-atom equivariant linear stage followed
by activation. Subsequent equivariant linear and self-tensor-product outputs are added. Newton–Euler readout branches into a
per-fragment mean for translational velocity and a torque followed by the
inertia pseudo-inverse for angular velocity. The interaction states contain
irreducible features of degrees ℓ = 0, 1 and 2 (scalar, vector and rank-2
features, respectively). The separately parameterized
four-layer confidence network combines ℓ = 0 features with norms of ℓ > 0
channels.
Global pooled features and a contact-aware readout are concatenated before the
pose MLP predicts RMSD and a pose-success logit. Pose–protein contact descriptors
enter both the atom MLP and the contact readout. A fresh docking-network forward
pass at t = 1 on each scored candidate supplies ligand hidden states, weighted
by a learned scalar coefficient α and added to the confidence embedding.
Docking conditions on additive time and log-prior-scale embeddings; confidence
uses c = 0. **B**, The interaction layer applies pre-message RMSNorm,
equivariant convolution, followed by separate equivariant linear, activation
and channel-dropout stages. The bracket labels this post-convolution transform by its role; convolution
itself is also equivariant. The original input follows one identity skip,
added before AdaLN.
**C**, The convolution maps normalized node features h_in to aggregated node
features h_conv, using input radial scaling, a shared tensor product
with spherical harmonics through degree two, output radial scaling, activation
and gate-normalized aggregation. Normalized endpoint scalars, condition and
edge descriptors feed both the radial MLP and the separate gate MLP. The
shared edge gate passes through sigmoid and is multiplied by a separately
computed, edge-type-dependent distance decay exp(−dᵢⱼ/σ_type) before aggregation.
The radial-scale boxes show the residual scale factors 1 + δ_in and 1 + δ_out. **D**, AdaLN applies its
own RMSNorm and condition-dependent scalar affine modulation or bounded
non-scalar scaling. The two degree-specific equations occupy separately labeled
ℓ = 0 and ℓ > 0 rows within one modulation box. Feature and conditioning
streams enter from above, and the complete feature
output leaves below, labeled beside its arrow. The activation schematic underneath
is a separate reusable operation, not a stage appended to AdaLN.
The RMSNorm box defines h → hhat internally, leaving the connector to modulation
clear. The scalar linear projection explicitly outputs γ₀, β₀ and γ₍>₀₎.
The symbol ⊙ denotes channelwise multiplication, with each non-scalar scale
broadcast over its irrep components.
RMSNorm acts separately within each (degree, parity) block;
exact reductions and modulation equations are given below. Activation applies
SiLU to even scalars; invariant non-scalar norms pass through an MLP and sigmoid,
and the resulting channel gates g multiply the original vector/tensor input.
T-junctions mark branches, + denotes addition, × denotes multiplication and “concat”
denotes concatenation. α is a learned scalar parameter; g is feature-dependent. Bold h denotes
node features, hats denote normalized features, and subscripts 0 and >0 select
degree classes. Bold c denotes conditioning. B and C use the same h_in/h_conv
labels at the convolution interfaces. Equivariant linear maps mix channels
within matching degree/parity blocks, with the same weights applied to each
spatial component.

## Notation

Bold h consistently denotes a feature tensor. Superscripts (k) and (k+1)
index full interaction layers; subscripts in and conv identify the normalized
input and aggregated output of the convolution. Subscripts 0 and >0 select
ℓ = 0 and ℓ > 0 channels. A hat marks normalization; a prime marks the local
output of AdaLN or activation in D, rather than a full next-layer state.
Bold c is the conditioning vector. γ and β are its channelwise modulation
parameters; g denotes feature-dependent activation gates and α the learned
coefficient for docking-state input to confidence. Diagram labels use this h
notation for all degree classes, without separate s/u feature variables.

## Linear and equivariance labels

| Figure location | Label | Implementation and representation |
|---|---|---|
| A, ligand-atom preprocessing | Equivariant linear | `atom_head_pre`: `cuet.Linear`; mixes compatible degree/parity channels, retains ℓ = 0, 1, 2 |
| A, branch beside self tensor product | Equivariant linear | `f_atom_linear`: `cuet.Linear` to one 1o vector; direct linear paths use compatible 1o inputs |
| B, after convolution | Equivariant linear | `post_block.linear`: `cuet.Linear` over the full mixed-irrep state |
| D, condition projection | Scalar linear | `cond`: `nn.Linear` on invariant conditioning; outputs γ₀, β₀ and γ_{>0} |
| A/C/D, MLPs | MLP | Ordinary `nn.Linear` on invariant scalar descriptors, pooled features or channel norms |

“Equivariant linear” here denotes a mixed-irrep linear map with weights shared
across the spatial components of each irrep. Different degrees/parities do not
mix through this linear map. The scalar projections also respect the symmetry
of their invariant inputs; “scalar” does not mean that they break equivariance.
The self tensor product supplies degree-coupling paths absent from the linear
vector readout. Channel dropout shares its mask across irrep components and is
disabled for inference. The panel-D activation reference in A/B/C denotes the
equivariant scalar-SiLU/non-scalar-gating operation, not scalar-only SiLU on all
feature components. These labels were checked against the source paths below.

## Panel definitions and implementation audit

| Panel | Depicted operation | Source |
|---|---|---|
| A | Separately parameterized docking/confidence models, embedding, readouts | [Docking model](../../src/effdock/models/effdock.py), [confidence model](../../src/effdock/confidence/model.py) |
| B | Pre-norm → convolution → post transform → residual → AdaLN | [EFFDockInteractionLayer](../../src/effdock/models/effdock.py), [EquivariantBlock](../../src/effdock/models/equivariant.py) |
| C | Input scale → tensor product → output scale → activation → gated aggregation | [GatedEquivariantConv](../../src/effdock/models/equivariant.py) |
| D | AdaLN with its own RMSNorm; scalar SiLU and norm-derived non-scalar gates | [EquivariantRMSNorm, EquivariantAdaLN, EquivariantActivation](../../src/effdock/models/equivariant.py) |

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
obtained with the fragment inertia pseudo-inverse. The two readout branches
in panel A show `v_f = mean_{i∈f} f_i` and
`omega_f = I_f^+ sum_{i∈f} (x_i - T_f) × f_i`. Here `f_i` is the learned
atom-wise vector, `T_f` is the fragment center, and the `I_f^+` box applies
the thresholded inverse on observable rotation axes. The diagram omits the
coordinate inputs to this compact readout; torque and inertia both depend
on the atom-to-fragment-center lever arms. Unit atom weights and a 1%
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
“ℓ = 0 features / ℓ > 0 norms” denotes a concatenated invariant feature
vector, not an elementwise sum. The “Contact readout” box includes contact projection,
attention/max pooling and atom mean/max pooling; the arrow into each of the two
contact-branch boxes marks a separate descriptor concatenation. “Logit” in the
pose head is the auxiliary pose-success logit. Auxiliary atom-level heads are
omitted from this model overview.

### B. Interaction layer

The exact outer order is `AdaLN(h + post_block(conv(pre_norm(h))))`.
The labels at the convolution interfaces define `h_in = pre_norm(h)` and
`h_conv = conv(h_in)`. Both are node-indexed feature tensors with the same
736-component irrep layout. `h_conv` is the aggregated convolution output,
not the next layer state; the post-message stages, residual addition and AdaLN
still follow. Panel C repeats the same interface names.
The identity skip carries the original h around pre-normalization and the
message/post block, ending at the + node before AdaLN. There is exactly one
additive residual per interaction layer; neither the post block nor AdaLN has
an additional residual. The output arrow follows AdaLN; its condition enters
from above. The six docking layers and four confidence layers are separately
parameterized; repetition counts do
not denote tied weights.

Equivariant linear maps mix multiplicity channels within compatible irreps.
The three post-message boxes show equivariant linear, activation and channel
dropout in their execution order. Message activation in C and post-linear
activation in B use the operation in panel D with independent parameters. Channel dropout
uses one mask per irrep channel, broadcast over its spatial components; it is inactive at inference. Both released configs use 0.1.

### C. Equivariant convolution

A 1,020-wide edge vector contains 32 RBF values, edge/bond descriptors, distance
evolution, fragment-hop and local-frame features, contact chemistry, normalized
endpoint scalar features, and conditioning. One radial trunk produces both input and output scales (each 1 + a learned
delta); a separate gate MLP produces the aggregation gates. Tensor-product weights are shared across edge
types within a layer. Real spherical harmonics include degrees 0, 1 and 2.
Message activation occurs after output radial scaling, before aggregation.

The expanded gate branch shows the shared weight for each edge i→j:
`z_ij = gate_mlp(e_ij)` and
`g_ij = sigmoid(z_ij[0]) * exp(-d_ij / sigma_edge_type)`.
Distance and edge type supply the decay directly. Additional sigmoid channel
gates from the same gate MLP multiply this shared weight for non-scalar
aggregation. Those channel gates and the degree-grouped norm rescaling remain
inside the aggregation abstraction.

Aggregation uses gate-normalized sums, edge-type-dependent distance decay and
additional non-scalar norm rescaling grouped by degree. It is not ordinary
softmax attention. These internal reductions are compressed into the gated
aggregation module. The symbols hhat_{i,0} and hhat_{j,0} mark normalized
ℓ = 0 endpoint features;
they are concatenated with the condition c and edge descriptors before the two
MLP branches. The shared input branch supplies those same descriptors to both
MLPs; it does not feed the gate MLP with the radial MLP's output. The spherical
harmonic argument appears inside the tensor-product block to avoid a crossing
external wire. “Shared” refers to tensor-product weights shared across edge
types within a layer, not weights tied across successive interaction layers.

### D. Normalization and activation

#### RMSNorm

Let b identify one (degree, parity) block, C_b its multiplicity, and h_{b,c}
its channel vector (one component for a scalar, 2ℓ+1 otherwise).
Write RMS_b(h) = r_b. For each node:

- `r_b = sqrt(mean_c ||h_{b,c}||² + epsilon)`, with `epsilon = 1e-6`.
- `hhat_{b,c} = a_{b,c} h_{b,c} / r_b`, with learned channel gains initialized to one.

The component norm sums over the irrep dimension; the outer mean is over channels
within the block. This is not one RMS over the entire 736-component state, a
batch statistic, or separate normalization of x/y/z. No mean is subtracted.
All components of a channel receive the same scalar multiplier. Learned gains are unconstrained and may change sign.

#### AdaLN

AdaLN has its own RMSNorm, distinct from the pre-message RMSNorm in panel B.
The condition passes through one linear projection, producing gamma_0, beta_0
and gamma_{>0}. Panel D flows from top to bottom: h passes through RMSNorm,
c through the conditioning projection, and both enter a common modulation box.
The two labeled rows show the scale-and-shift rule for ℓ = 0 and the bounded
1 + 0.1 tanh(γ) scale for ℓ > 0. They apply to separate degree classes,
not sequentially to the same channels. The output h′ contains both degree
classes; the projection lists the three modulation-parameter groups explicitly.
Normalized features follow two branches:

- ℓ = 0: `h'_0 = (1 + gamma_0) * hhat_0 + beta_0`.
- ℓ > 0: `h'_{>0} = (1 + 0.1 * tanh(gamma_{>0})) * hhat_{>0}`.

Non-scalar scales lie between 0.9 and 1.1; no vector/tensor offset is added.
The scale is shared across the 2ℓ+1 components of each channel. Zero initialization
of the conditioning projection initially leaves the
RMS-normalized features unchanged, not the raw input unchanged. A zero confidence
condition need not imply zero modulation after training because the projection
has a learned bias. These normalization/modulation operations respect rotations
of their irrep inputs. This does not establish full-pipeline equivariance; see
the initialization caveat below.

#### Equivariant activation

Even scalars receive elementwise SiLU. All non-scalar channel norms are
concatenated and passed through an MLP and sigmoid to produce per-channel gates.
The bypass in panel D carries the original vector/tensor input to the product:
`h'_{>0,c} = g_c h_{>0,c}`. Subscripts 0 and >0 denote degree classes,
not layer indices; >0 includes degrees 1 and 2 in the released model. The
ℓ = 0 path contains even scalars (0e). The × node denotes channelwise multiplication, broadcasting
one gate over all 2ℓ+1 components of each channel. Norms are used to calculate
the gate, not substituted for the original input. The message activation and
post-linear activation use separate parameters; their panel-D references denote
the same operation type, not tied weights.

### Equivariant linear implementation

For one node and fixed degree/parity (ℓ,p), write the feature block as
`h_{ℓ,p} ∈ R^(C_in × (2ℓ+1))`. Its linear map is
`h'_{ℓ,p} = W_{ℓ,p} h_{ℓ,p}`, where
`W_{ℓ,p} ∈ R^(C_out × C_in)`. The same W acts on every spatial component;
different degree/parity blocks have independent weights. The effective W
includes library path normalization. The deployed `cuet.Linear` calls use
internal shared weights and no additive bias. Weights are not tied across
separate layers or heads.

The atom preprocessing and B post-convolution maps retain the mixed-irrep
structure. The atom-vector projection requests only 1o output and directly
uses compatible 1o inputs; its parallel self tensor product supplies additional
degree-coupling paths. These details explain the Equivariant linear labels in
A/B without a separate figure panel.

### Full diagram audit (2026-09-26)

The released source/config hashes were rechecked alongside these paths:

- A: six/four independent interaction layers; atom-head linear/activation,
  parallel linear/self tensor product, mean and torque/inertia readout;
  candidate-specific t = 1 confidence features and both contact injections.
- B: pre-RMSNorm, convolution, post transform, one original-input residual,
  then AdaLN. The post-transform bracket denotes execution order, not the
  extent of equivariance.
- C: 1 + δ radial factors, shared tensor product, activation, separate
  sigmoid gate and distance decay. Additional channel gates and degree-grouped
  norm rescaling remain explicitly documented inside aggregation. The current
  forward returns the aggregate directly; an old class-docstring reference to
  a self-linear residual is stale and is not depicted as an extra model stage.
- D: blockwise RMSNorm, scalar condition projection, exact degree-wise
  modulation, scalar SiLU and norm-derived non-scalar activation gates.
- Linear maps: mixed-irrep channel maps at all three depicted call sites. The installed
  `cuequivariance_torch.operations.linear.Linear` constructor and its linear
  descriptor were inspected for component sharing and absence of bias.

This is a source and schematic audit, not a new numerical equivariance test or
model evaluation. No trained model or benchmark result was changed.

### MLP implementation details

MLP inputs are invariant scalar quantities (ℓ = 0): edge features,
non-scalar channel norms, or pooled/projected invariant confidence features.
Their outputs parameterize equivariant scaling or produce invariant readouts.
These MLPs use ordinary scalar linear layers. B's Equivariant linear instead mixes
multiplicity channels within compatible irreps, followed by the equivariant
activation in D and channel dropout. The bracket in B is labeled “Post-convolution transform” and corresponds to
the deployed `EquivariantBlock` class. It describes where those three operations
occur, not a boundary between equivariant and non-equivariant processing.
Convolution, the post transform, RMSNorm and AdaLN are all equivariant; there
is no additional internal residual.

The radial MLP uses a shared `Linear(1020,128) → SiLU` trunk and two independent
`Linear(128,480)` heads. Its outputs are δ_in and δ_out; C's radial scales
apply 1 + δ to each irrep channel. The gate MLP uses
`Linear(1020,64) → SiLU → Linear(64,97)`; its sigmoid is shown separately in C.
The norm MLP in D uses `Linear(n_v,n_v) → SiLU → Linear(n_v,n_v)`, followed by
the separate sigmoid and multiplication with the original ℓ > 0 features.
Here n_v counts non-scalar multiplicity channels (96 in both the interaction
layers and the atom-head activation). Gate and norm MLPs have the same topology,
not shared weights.

The confidence `_mlp` constructor repeats Linear–SiLU–Dropout `depth−1` times
before a terminal Linear. The atom trunk uses depth 2; the hybrid pose head
uses depth 3. The atom trunk has one hidden stage and the pose head has two,
with independent weights at each repetition. The terminal linear has no appended activation or
dropout. Other scalar confidence projections use the same constructor; their
pooling and normalization remain in the contact/global readout abstraction.
Dropout is inactive when evaluating the model.

The reusable `EquivariantMLP` class is not instantiated by the released docking
or confidence models; the deployed equivariant block and scalar MLPs above
are the operations represented here. The compact figure retains MLP module
names; their internal layers and dimensions are specified in this section.

## Reproduction and use

```bash
python -m benchmarks.figures.architecture --output outputs/paper_figures
```

The [source specification](../../benchmarks/results/paper/architecture/spec.json)
records dimensions, normalization operations, the ligand-state input and
source/config SHA-256 hashes, including feature extraction and scoring code.
Rendering fails if those sources change. It also checks dimension arithmetic,
text overlap, module padding and connector routing (undeclared intersections,
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


## Source audit and unresolved initialization defect (2026-09-27)

The review compared the current A–D graphic with executable forward paths,
the released configurations, and all eight source/config hashes in spec.json.
Hash agreement identifies the inspected implementation; it is not a functional
equivariance test. No model or checkpoint was changed.

| Panel | Verified against implementation | Abstraction or remaining issue |
|---|---|---|
| A, docking | Six separate layers; atom preprocessing; sum of linear and self-TP outputs; mean/torque/inertia readout | Initialization has the indexing defect below. Conditioning also enters fragment initialization, although its diagram arrow points only to the layer stack. Readout lever-arm inputs and the thresholded inverse are documented rather than drawn. |
| A, confidence | Four separate layers; fresh per-pose t=1 ligand states added with learned α; invariant features; global/contact concatenation | α acts only on ligand atom/fragment slots. Global pooling includes LayerNorm and a node MLP. Contact readout includes projection and attention/max/mean reductions. Auxiliary atom heads and the final log1p-to-RMSD conversion are compressed/omitted. |
| B | AdaLN(h + post_block(conv(pre_norm(h)))); one additive skip; linear → activation → dropout | c also enters convolution edge features (shown in C), not only AdaLN. Dropout is disabled during evaluation. |
| C | Radial input scale → shared TP → radial output scale → activation → aggregation; shared edge gate times distance decay | The 96 additional non-scalar gates and per-degree Linear–SiLU norm rescaling are inside the aggregation box. This is not a complete scalar/non-scalar reduction diagram. |
| D | Blockwise RMSNorm; scalar affine rule; bounded non-scalar scaling; scalar SiLU and norm-derived gates | AdaLN and activation are separate operations. ℓ=0 denotes 0e in this model. The two modulation rows act on different feature subsets in parallel. |

Panel D was aligned so both activation input/output labels share the same
horizontal anchors. The AdaLN output label now sits beside its terminal arrow,
with separation from the activation heading. The hhat definition was moved
inside the RMSNorm box, and the parallel condition-projection box has matching
height. Its operations and formulas did not change.

### Rotation injection: a verified local counterexample

In [the docking forward](../../src/effdock/models/effdock.py), line 881 uses:

```python
h_R = torch.einsum("nki,ck->nci", R_t, self.R_frag_mix.weight)
```

The adjacent comment describes mixing columns of R, but the expression mixes
rows: for one channel weight w, it computes Rᵀw instead of Rw.
[The geometry convention](../../src/effdock/geometry/se3.py) and
[sampler coordinate reconstruction](../../src/effdock/inference/sampler.py)
use x_global = R @ x_local. A joint world rotation Q therefore changes a
fragment frame to QR. A vector-valued initialization should satisfy
F(QR, w) = Q F(R, w).

A CPU float64 counterexample evaluated the einsum string extracted from the
actual source, with R=I, w=(1,0,0), and Q a 90° rotation around z. The current
expression produced approximately (0,−1,0), whereas the rotated original
feature was (0,1,0): maximum absolute component error 2.0. The intended column
expression nik,ck->nci had zero error on the same example.

This proves that the injection is not an equivariant map for general nonzero
weights under the stated joint-rotation action. The toy error of 2.0 is **not**
a measured error of the trained model. The checkpoint-specific follow-up below
measures the docking output deviation. Existing tests/test_equivariance.py
exercises individual building blocks, not this EFFDock initialization path.

Changing these indices would change model computation under existing weights.
The audit therefore leaves the model and checkpoint intact. The earlier
source-hash/layout checks must not be presented as proof that the whole model
is strictly equivariant. The input embedding
contains a measured departure from strict equivariance; the equivariant
backbone operations and existing docking results are separate claims.


### Released checkpoint and saved-trajectory recheck

The released 50,000-step EMA checksum matches the model manifest. Its
R_frag_mix has 96/96 nonzero entries (Frobenius norm 0.749953, maximum absolute
entry 0.202768). The row-mixing expression is also present in the initial public
source commit 3e26f41 (2026-07-21); the figure edits did not introduce it.

The test used the saved 1T46_STI trajectory with 37 ligand atoms, six fragments,
375 graph nodes and 2,256 static edges, prior sigma 2, and three stored states.
Fragment-local coordinates and graph topology were fixed. Protein/ligand
world coordinates and fragment centers were jointly rotated, while fragment
frames transformed by left multiplication QR. Rotations were z90°, x90°, and
60° about (1,2,3). This is a test of the prepared model state, not a rerun of
raw-structure preprocessing or complete ODE sampling.

The checkpoint's learned parameters were loaded into the library's CPU naive
backend, using the same tensor-product descriptors in float32. No new docking
poses, training or GPU inference were run. Identical-input repeat errors were
zero. The reported quantity is 100 × relative Frobenius-norm deviation from
the expected rotated **instantaneous velocity output**, not RMSD, a docking
success-rate difference or accumulated trajectory error.

| Saved state | Translation output deviation (%) | Angular output deviation (%) |
|---|---:|---:|
| t=0.000 | 0.161–0.196 | 0.613–0.887 |
| t=0.488 | 0.086–0.143 | 0.166–0.422 |
| t=1.000 | 0.056–0.087 | 0.147–0.187 |

Ranges are over the three prescribed rotations, not confidence intervals.
Zeroing only the injection weights or changing only its einsum indices in
memory reduced relative deviations below 0.00061% across these tests. These
controls isolate the source of the excess deviation; they are not proposed
inference settings or evidence that changed-index weights improve docking.
The released source, weights and existing results remain unchanged.

This establishes a small but nonzero departure from exact equivariance **in
this case**. It does not establish benchmark-wide robustness, the effect on
confidence ranking, or the need for retraining. Already measured docking
scores remain measurements of the existing trained implementation. Changing
indices after training would define a different computation and cannot be
silently substituted into those results. Panel A uses the neutral label
“Node embedding”; the backbone operations retain their explicit equivariant
labels. The training local-frame augmentation also does not mathematically
repair the row-mixing operation.

[Full diagnostic values](../../benchmarks/results/paper/architecture/rotation_diagnostic.json)
include absolute errors, repeat controls, all rotations and artifact hashes.
