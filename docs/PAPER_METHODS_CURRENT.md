# Current manuscript methods

The canonical manuscript [Methods](../papers/main.tex) and
[Supplementary Information](../papers/SI.tex) define the current scientific
protocols and equations. [Figure captions](../papers/figure_captions.md)
specify each reported condition. This overview and the
[implementation notes](methods/README.md) describe the released pair and its
source code; they are not separate versions of the manuscript.

The manuscript evaluation uses explicit `rw` angular features,
`contextual_v1` confidence frames and generation-consistent ligand references,
with the released weights frozen. Training and compatibility conventions are
documented in [the coordinate contract](ORIENTATION_INJECTION.md).

## Task and checkpoints

EFF-Dock performs supplied-pocket protein–ligand redocking. Inputs are a
receptor structure, ligand chemistry and an explicit pocket centre. The
released pair consists of the early-time/t=0-replay 50k docking EMA and the
S50 raw+refined U70k confidence checkpoint. Exact SHA-256 identities are in
[weights/MANIFEST.md](../weights/MANIFEST.md).

Main sampling is unguided N100/S10, translation sigma 2 Å, 10 Å receptor crop
and a late-power-3 time grid. Ordinary ranking minimizes predicted pose RMSD.
The public sampler returns raw poses; the main benchmark condition additionally
applies energy refinement and tetrahedral-chirality-filtered confidence
selection. These are explicitly separate stages.

## Data and training membership

PLINDER 2024-06/v2 sample identity is `(system_id, ligand_instance_chain)`.
The preserved compatibility split has 47,310 training and 1,076 validation IDs.
The released docking loader contains 47,277 samples (45,441 unique systems;
43,392 PDB IDs). Confidence uses 43,092 training and 1,035 validation samples.
The two training sets share 43,067 IDs, with 4,210 docking-only and 25
confidence-only entries. Public [ID lists, exclusions and hashes](../benchmarks/inputs/training_membership/README.md)
make these distinctions auditable. They describe loader/split membership,
not an exhaustive audit of all predecessor checkpoint exposure.

The historical split used relaxed external exclusion: exact ligand match AND
membership in an external pocket-community set constructed from benchmark PDB
hits in the processed pool. This is not equivalent to the newer strict split
builder, which removes all matching external canonical SMILES and enforces
sample/SMILES/pocket70 disjointness. The newer builder must not be cited as the
procedure that produced the released checkpoint. Broader relatedness matches
remain and are reported, not retrospectively removed.

The [membership manifest and historical audit](../benchmarks/inputs/training_membership/README.md)
retain the original 904 exclusions, the validation community-overlap counts,
and the 34 broader external-overlap witnesses. Community membership is not a
direct pairwise-identity bound; the released cohorts must not be described as
fully disjoint under the broader mapping.

Ligands are sanitized heavy-atom graphs split at eligible rotatable bonds;
planar-conjugated bonds remain rigid and singleton fragments are merged.
Training augments pocket crops from 6–12 Å and translation-prior sigma with
{0.5,1,2,3,4} Å and probabilities {0.10,0.25,0.30,0.25,0.10}.
Detailed [graph features](methods/01_graph_features.md) and
[fragment representation](methods/02_fragment_se3_flow.md) specify the schema.

## Model, objective and internal selection

The graph contains ligand atoms, fragments, protein atoms and residue virtual
nodes. Six equivariant docking layers use degree-2 spherical harmonics,
32 radial basis functions, dynamic 5 Å protein–ligand contacts and
`384x0e+32x1o+32x1e+16x2e+16x2o` hidden irreps. The learned atom field is
aggregated into fragment translation and observable angular velocity by
unit-weight Newton–Euler equations.

The active docking objective is translation MSE + 8 × angular MSE + 0.3 ×
**atom-velocity MSE** + 3 × distance-geometry loss. The latter compares
inter-fragment pair distances after one-step endpoint reconstruction, using
a normalized `t²`-weighted mean across complexes. It is distinct from the
atom-velocity auxiliary term. [Exact equations and reductions](methods/04_docking_head_and_objective.md)
also specify observability and distributed normalization.

The 50k continuation starts from the preceding geometry model and samples
`0.80 SimpleFold-style + 0.10 Uniform(0,0.3) + 0.10 exact t=0`. It uses fresh
AdamW state, global batch 64 across four GPUs, peak LR 2e-5 and EMA 0.999.
The complete retained time law and schedules are in the
[training specification](methods/06_training_and_checkpoint_selection.md).

The four-layer confidence scorer uses paired docking hidden features and
contact/global pooling to predict pose RMSD/success and atom displacement/
success. Each training bank supplies 32 raw poses, 32 refined poses and a
mapped crystal anchor; the loader draws a bounded subset. Its Huber, BCE,
pairwise-ranking and smoothed-success listwise terms are specified in
[the confidence objective](methods/05_confidence_model_and_loss.md).
U70k was selected only on the fixed 1,035-complex internal validation bank:
622/1,035 (60.10%) Top-1 RMSD <2 Å, compared with 617/1,035 at U100k.
Confidence is a within-ensemble ranking signal, not calibrated affinity.

## Refinement and evaluation

Energy refinement minimizes the in-repository physical plus seven-term
interaction energy with a fixed receptor, using mass/inertia-preconditioned
rigid-fragment descent and independent per-pose backtracking. The executed
external protocol allows at most 100 iterations, caps translation/rotation/
atom displacement at 0.10 Å / 5 degrees / 0.10 Å, and enables an energy-plateau
stop from iteration 25. Chiral improper restraints are active; the separate
chemical-constraint channel is zero-weight. [Inference and evaluation methods](methods/07_inference_and_evaluation.md)
define all terms, constants, stopping rules, failure handling and the
all-candidates-fail chirality fallback.

RMSD is symmetry-aware, no-alignment heavy-atom RDKit CalcRMS, with strict
threshold <2 Å. PB-valid success requires that same selected pose to pass the
official PoseBusters evaluator. All original denominators remain. Benchmarks
are Astex Diverse Set (85), PoseBusters v2 (308), PhiBench-derived (206 locally
selected cases; three reconstructed cases), FoldBench (558) and OpenBind
(925; auxiliary single-protease set including two flagged noncovalent
approximations of covalent systems). Astex/PoseBusters use native redock;
the three temporal cohorts use the disclosed InChI energy-reference
compatibility adapter. Dataset v2 and PoseBusters software version are distinct
identifiers.

Frozen benchmark pocket centres can be reference-ligand-derived, and receptors
can be holo structures. Primary benchmark ligand templates are prepared from
ligand chemistry using the recorded generation preparation. Crystal-ligand
coordinates provide the supplied pocket definition, atom-mapping references
and evaluation labels; they are not candidate-ranking inputs or the molecular
templates used for primary refinement.
These retrospective redocking results do not establish blind docking,
prospective screening, affinity prediction or cofolding performance.

## Figure data and statistical interpretation

The canonical [Source Data](../papers/data) provide the numerical inputs and
provenance of current manuscript figures. The older
[renderer inputs](../benchmarks/results/paper/README.md) describe historical
analyses and must not be substituted for the current results. Figures report
sample SD across three fixed-weight inference repeats; paired differences use
2,000-resample 95% percentile intervals, comparing complex and exact-PDB-group
resampling where specified in the captions.

Relatedness uses the 47,277 docking samples. Binding-chain identity is
query-normalized over one-to-one chain assignments within a training sample;
exact-ligand AND sequence≥70% overlap requires the same training witness.
It does not certify pocket identity or proven leakage. Definitions and known
sequence-reference limitations are in [RELATEDNESS.md](paper/RELATEDNESS.md).
Validation is included in relatedness composition but has no matched
three-repeat external performance bank.

Opened external cohorts are descriptive and do not select checkpoints.
Unguided generation-input sensitivity, supplementary guided controls and
literature baselines retain their separately captioned protocols. Released
weights remain frozen; these evaluations do not tune checkpoints or thresholds
using external outcomes.
