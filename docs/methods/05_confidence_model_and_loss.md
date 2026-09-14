# Ensemble confidence model and ranking objective

Implementation sources: `src/effdock/confidence/model.py`,
`src/effdock/confidence/losses.py`, and
`configs/train_confidence_s50_raw_refined_100k.yaml`.

## 1. Prediction unit and inputs

The confidence model ranks the `K` poses generated for **one fixed
protein--ligand complex**. It is not an affinity predictor and its values are
not calibrated across targets. Its input is `pose_atom_coords [K,A,3]`, the
static heterogeneous graph, fragment assignment, optional saved docking
ligand representations `h_lig_node [K,N_lig,736]`, and labels available only
in training. A batched graph is constructed by repeating the receptor graph
once per candidate; dynamic protein--ligand contacts are then rebuilt from
each candidate's coordinates at 5 A.

The backbone uses the same `736`-wide O(3) representation as docking:
`384x0e + 32x1o + 32x1e + 16x2e + 16x2o`, with four interaction layers.
Saved ligand hidden states are added through a learned scalar gate initialized
to 0.25; if absent, the model remains a complete graph scorer.

## 2. Atom contact representation and heads

Per ligand atom, invariant backbone channels (`480`) are concatenated with a
`44`-wide receptor-contact vector:

| contact component | width |
|---|---:|
| minimum protein-atom distance / 10 | 1 |
| RBF(minimum distance; range 0--10 A) | 32 |
| `log(1 + count)` at <2 A, <3.5 A, <5 A, and soft-contact sum | 4 |
| soft-distance-weighted protein flags | 7 |
| **total** | **44** |

Thus atom heads consume `524` channels and apply
`LayerNorm(524) -> MLP(524,512,512; depth 2) -> MLP(512,512,2; depth 2)`.
They output `atom_disp_log1p [K,A]` and `atom_ok_logit [K,A]`.

For the released `global_contact_attention` readout, four 512-wide contact
summaries are concatenated: learned attention pool, contact maximum, atom
mean, atom maximum (`2048`). A separate global pool makes mean and max vectors
for each of ligand atom, fragment, protein atom, and protein residue nodes:
eight 512-wide vectors (`4096`). Their concatenation is `6144 = 12 x 512`,
then `LayerNorm(6144) -> MLP(6144,512,2; depth 3)`, producing
`pose_rmsd_log1p [K]` and `pose_success_logit [K]`.

The reported predicted RMSD is
`max(0, expm1(clamp(pose_rmsd_log1p, -2, 5)))`; ranking uses the unambiguous
minimum predicted RMSD convention of the inference API.

## 3. Labels

`atom_disp` is the atomwise displacement to the reference pose. `pose_rmsd`
is symmetry-aware, no-alignment, heavy-atom RMSD using RDKit `CalcRMS`; its
unit is Angstrom. The binary atom and pose labels use the strict boundary
`<2.0 A`. A mapped crystal anchor has RMSD exactly zero. All labels are
computed after pose generation; no crystal coordinates enter the scorer at
inference.

## 4. Loss

For `y_a=log(1+atom_disp_a)` and `y_k=log(1+RMSD_k)`, the regression terms are
Huber losses and the binary terms are BCE-with-logits:

\[
L_{atom}=Huber(\hat y_a,y_a),\quad L_{pose}=Huber(\hat y_k,y_k),
\]
\[
L_{atom-ok}=BCE(z_a,[d_a<2]),\quad
L_{pose-ok}=BCE(z_k,[r_k<2]).
\]

Pairwise RMSD ranking is active only for pairs whose target log-RMSD gap
exceeds 0.3:

\[
L_{rank}=\operatorname{mean}_{y_j-y_i>0.3}
\max(0,0.05-(\hat y_j-\hat y_i)).
\]

The active ensemble loss is a cross-entropy from the smoothed success target
to the pose-success logits. Let `q_k=sigmoid((2-r_k)/0.2)` and
`p_k=q_k/sum_j q_j`; then

\[
L_{success-list}=-\sum_k p_k\log\operatorname{softmax}(z)_k.
\]

The released configuration is

\[
0.2L_{atom}+0.2L_{atom-ok}+0.3L_{pose}+0.4L_{pose-ok}
+0.1L_{rank}+1.0L_{success-list}.
\]

Ordinary listwise, setwise, pairwise-success, and hard-negative alternatives
exist as explicitly configured zero-weight options; they are not active in the
released U70k objective.
