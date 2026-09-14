# Equivariant graph backbone

Implementation sources: `src/effdock/models/effdock.py` and
`src/effdock/models/equivariant.py`. Production architecture values come from
`configs/train.yaml` and the released confidence configuration.

## 1. Node state and irreps

Every node type in the heterogeneous graph uses the same O(3) state

\[
384\!\times\!0e + 32\!\times\!1o + 32\!\times\!1e +
16\!\times\!2e + 16\!\times\!2o.
\]

The flattened feature width is `736`:

| block | channels | Cartesian width |
|---|---:|---:|
| scalar `0e` | 384 | 384 |
| odd vector `1o` | 32 | 96 |
| even vector `1e` | 32 | 96 |
| even rank-2 `2e` | 16 | 80 |
| odd rank-2 `2o` | 16 | 80 |
| **total** | 480 irrep channels | **736** |

For invariant readout, the 384 scalars are concatenated with norms of the
`32+32+16+16=96` non-scalar channels, giving `480` invariant channels per
node. Initial scalars are the 384-dimensional graph embedding from
`01_graph_features.md`. Initial `1o` channels are gated displacements from the
per-complex graph centre; `1e`, `2e`, and `2o` start at zero. Fragment
orientation is injected through a zero-initialized rotation-dependent mix.

## 2. Conditioning and edge representation

Time uses `sinusoidal(32) -> MLP(32,128,128)` with SiLU. The optional prior
sigma condition has the same 128-wide projection and is added to the time
condition. For directed edge `i -> j`, the interaction layer concatenates:

| edge field | width |
|---|---:|
| 32 radial basis values | 32 |
| edge-type embedding (10 types) | 16 |
| bond type/conjugation/ring/stereo embeddings | 20 |
| evolving/reference-distance encoding | 16 |
| fragment-hop embedding | 8 |
| source-frame relative coordinate `R_i^T(x_j-x_i)` | 16 |
| six ligand--protein chemical pair flags | 16 |
| source scalar state | 384 |
| destination scalar state | 384 |
| time/sigma condition | 128 |
| **edge MLP input** | **1,020** |

The six pair flags are hydrogen bond, salt bridge, like-charge clash,
hydrophobe, ligand-halogen/protein-acceptor, and ligand-acceptor-or-negative/
protein-metal. They are features for learned message passing, not a physical
energy term. The radial encoder has 32 basis functions, the local spherical
harmonics have maximum degree `l=2`, and dynamic atom--atom contact edges are
rebuilt at 5 A for each forward pass.

## 3. Interaction layer

There are six docking layers and four confidence layers. Each layer applies
pre-normalization, an equivariant tensor-product convolution using spherical
harmonics, gated residual update, and time-conditioned equivariant adaptive
layer normalization. Schematically, with `h_i^(l)` an irrep state and
`Y_{<=2}(r_ij)` the real spherical harmonics,

\[
m_{ij}^{(l)} = g_{ij}^{(l)}\,\mathcal{T}^{(l)}
\left(h_i^{(l)},Y_{<=2}(x_j-x_i);\phi_e(e_{ij})\right),
\]
\[
h_j^{(l+1)}=h_j^{(l)}+
\operatorname{AdaLN}_t\!\left(\sum_{i\to j}
\alpha_{ij}^{(l)}m_{ij}^{(l)}\right).
\]

`phi_e` is the radial/edge MLP; `g` and `alpha` are learned gates. Attention
is normalized over incoming edges and also has a learned edge-type distance
scale. This is one shared equivariant tensor-product mechanism conditioned on
edge type and attributes, not ten separate full GNNs. Dropout is 0.1.

## 4. Equivariance boundary

Only relative coordinates, source local-frame coordinates, distances, and
spherical harmonics enter geometric messages. Consequently a joint rigid
O(3) transform of ligand and receptor rotates vector/tensor outputs and leaves
scalar predictions invariant. Categorical chemistry and graph topology are
coordinate independent. The supplied pocket centre chooses the conditional
translation frame but is not learned from a target crystal ligand.
