# EFF-Dock method specification

This directory is the equation- and dimension-level companion to
[`../PAPER_METHODS_CURRENT.md`](../PAPER_METHODS_CURRENT.md). Every document
names the implementation source and the production configuration from which its
values are taken. Dimensions below describe the released U70k production stack,
not a claim that every historical checkpoint has the same schema.

| Document | Question answered |
|---|---|
| [`01_graph_features.md`](01_graph_features.md) | What nodes, edges, categorical features, and embedding dimensions enter the model? |
| [`02_fragment_se3_flow.md`](02_fragment_se3_flow.md) | How are ligands fragmented, parameterized on SE(3), and supervised by flow matching? |
| [`03_equivariant_backbone.md`](03_equivariant_backbone.md) | What is the irreps layout, edge feature vector, and message-passing equation? |
| [`04_docking_head_and_objective.md`](04_docking_head_and_objective.md) | How are atom fields mapped to fragment velocities and losses? |
| [`05_confidence_model_and_loss.md`](05_confidence_model_and_loss.md) | What ranks candidate poses and what objective trains it? |
| [`06_training_and_checkpoint_selection.md`](06_training_and_checkpoint_selection.md) | Which data, optimization, and internal selection rules produced the released pair? |
| [`07_inference_and_evaluation.md`](07_inference_and_evaluation.md) | What exact inference contract and endpoints define paper results? |

Notation: `N` is total nodes in a batched graph, `E` edges, `A` ligand atoms,
`F` ligand fragments, and `B` complexes. Coordinates are Angstroms. A tensor
shape such as `[N, d]` is after collation unless stated otherwise.
