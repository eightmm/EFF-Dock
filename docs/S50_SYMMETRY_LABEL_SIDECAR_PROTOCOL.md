# S50 Symmetry-Aware Confidence Label Sidecar Protocol

Status: frozen before train-label materialization on 2026-08-20.

## Objective

Correct the S50 confidence-training pose target so that pose-level regression,
binary success, ranking, and success-listwise supervision use the same
symmetry-aware, no-alignment heavy-atom RMSD used by validation and deployment
evaluation. Atomwise displacement remains a fixed-map auxiliary target because
an atomwise target is not uniquely defined across equivalent automorphisms.

This is a label-only intervention. It does not regenerate, refine, filter,
reorder, or select poses and it does not recompute S50 hidden features.

## Frozen raw bank

- Input manifest SHA-256:
  `6a991e964715f9ddb51ce48ccbea5a948eb3b2d3c51de1f26929389365aec5f3`
- Complete bank manifest SHA-256:
  `d45e36f3f2d75fb8ba9553715e1ec45031e4ad31881631d8632b6c19999f2d2b`
- Train inventory: exactly 43,092 eligible complexes and 4,309,200 ordered
  poses. The existing 1,035-complex validation bank already contains the same
  symmetry-aware target and is not regenerated.
- Pose contract: S50 EMA, deterministic ODE, N100/S10, sigma 2.0, late-power3,
  pocket 10 A, canonical-SMILES conformer seed 0, no guidance/FK/SDE/refinement.

## Label computation

For every ordered saved train pose, reconstruct the canonical-SMILES input and
the sealed processed crystal reference, require the frozen complete strict-
stereo graph mapping, place the saved receptor-frame coordinates on the input
molecule, and evaluate RDKit `rdMolAlign.CalcRMS` without alignment. The output
key and method are:

- `pose_rmsd_symmetry_no_align`
- `rdkit_calc_rms_symmetry_no_align`

Every source payload, processed crystal ligand, pose-ensemble identity, and
sample/split/seed identity is checked against the two frozen manifests. Any
missing file, hash drift, graph/mapping drift, CalcRMS failure, non-finite
value, duplicate, or incomplete inventory fails the shard. There is no
fallback RMSD and no outcome-dependent exclusion or replacement.

The source `.pt` bank remains immutable. Results are written as 128 small CPU-
generated label shards plus one complete manifest. Candidate order is
preserved and each label tensor has shape `[100]`, float32. Atomic no-overwrite
publication is required.

## Training transition

The currently running fixed-map-label job 56312 is retained only as a
compatibility/control branch while labels are computed. It is not a valid
symmetry-label continuation point. After the sidecar is complete and loader
smoke tests pass, the symmetry-label branch must restart from the retained
confidence step-42500 weights with fresh optimizer and scheduler state.

Pose RMSD regression, pose success BCE, ranking, and success-listwise losses
must all consume the symmetry-aware target. Atom RMSD regression and atom BCE
continue to consume the fixed-map `atom_disp`. Validation and best-checkpoint
selection remain symmetry-aware. Raw and future refined banks are separate
training stages; the refined bank receives its own sidecar after its complete
pose inventory is sealed.

## Execution and audit

Raw train labels run on the `cpu_only` partition as 128 shards with bounded
concurrency. The aggregate requires exactly all 128 successful artifacts and
4,309,200 finite labels. A small real-payload smoke must pass before the full
array. No GPU is requested for label calculation.
