# Graph schema and input feature dimensions

Implementation sources: `src/effdock/preprocess/{ligand,protein,graph,graph_types}.py` and `src/effdock/models/effdock.py`.

## Node sets and coordinate tensors

The unified graph is `V = V_LA ∪ V_F ∪ V_PA ∪ V_PR` with node type IDs
`{ligand_atom: 0, fragment: 1, protein_atom: 2, protein_residue: 3}`.

| Set | Count | Coordinate | Main discrete fields |
|---|---:|---|---|
| ligand atoms | `A` | `[A, 3]` | element, charge, aromaticity, hybridization, rings, degree, valence, chirality, six flags |
| fragments | `F` | centroid `[F, 3]` | fragment ID and size |
| protein atoms | `P` | `[P, 3]` | `(residue, atom)` token, backbone/metal and five flags |
| residue virtuals | `R` | `[R, 3]` | 22-way residue type, pseudo-CB indicator |

Protein residue virtuals use CB when present. For glycine or incomplete
residues the construction uses a pseudo-CB or CA-based fallback and records the
`pseudo` flag. Non-applicable fields are filled with sentinel values; this lets
one embedding module operate on all node types without treating a missing field
as a chemical value.

## Raw node embedding

The node embedding concatenates a fixed 208-dimensional scalar vector and maps
it through `Linear(208, 384) → SiLU → Linear(384, 384)`.

| Feature group | Input vocabulary / values | Embedded width |
|---|---|---:|
| ligand element | 13 categories (`C` through `Se` plus other) | 32 |
| ligand formal charge | scalar | 8 |
| aromaticity | 2 categories | 8 |
| hybridization | 7 categories | 16 |
| ring count | clipped 0–4 | 8 |
| atom degree | clipped 0–6 | 8 |
| implicit valence | clipped 0–7 | 6 |
| explicit valence | clipped 0–7 | 6 |
| chirality | 5 categories | 8 |
| ligand flags | donor, acceptor, positive, negative, hydrophobe, halogen | 16 |
| protein atom token | `(residue, atom)` vocabulary with unknown/metal fallbacks | 32 |
| protein structural flags | backbone, metal | 8 |
| node type | 4 categories | 16 |
| residue virtual type | 20 canonical amino acids, unknown, metal | 16 |
| residue virtual flag | pseudo-CB | 4 |
| protein flags | donor, acceptor, positive, negative, hydrophobic | 16 |
| **total** |  | **208** |

The protein-pharmacophore projection is zero initialized for schema
compatibility; it is a learned input pathway, not a hard-coded interaction
energy.

## Edge vocabulary

All edges are directed. The static graph has the following ten type IDs.

| ID | Edge type | Construction |
|---:|---|---|
| 0 | ligand bond | covalent ligand bonds, with bond attributes |
| 1 | ligand triangulation | local cross-fragment geometry near a cut bond |
| 2 | ligand cut | explicit rotatable cut bond |
| 3 | ligand atom–fragment | bidirectional membership edges |
| 4 | ligand fragment–fragment | fragment adjacency edges |
| 5 | protein bond | inferred intra-residue and peptide/disulfide bonds |
| 6 | protein atom–residue | bidirectional atom-to-virtual-residue edges |
| 7 | protein residue–residue | virtual-residue pairs within 10 A |
| 8 | protein residue–fragment | initial residue–fragment proximity relation |
| 9 | dynamic contact | protein-atom/ligand-atom pairs within 5 A during a forward pass |

Dynamic contact edges are appended separately for each complex in a batch, so
there are no cross-complex contacts. Fragment–fragment and residue–fragment
edges use an evolving-distance sentinel rather than a fixed reference distance.

## Tensor contract

The static graph returns `node_coords [N,3]`, `node_type [N]`,
`node_fragment_id [N]`, `edge_index [2,E]`, `edge_type [E]`, bond attributes
`[E]`, reference distance `edge_ref_dist [E]`, and fragment hop count
`edge_frag_hop [E]`. Collation adds `batch [N]`, `frag_batch [F]`, the flow
time `t [B,1]`, fragment state `T_frag [F,3]`, `q_frag [F,4]`, and optional
prior sigma `prior_sigma [B,1]`.
