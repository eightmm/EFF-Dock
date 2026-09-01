# RDKit Fragment-Local Geometry Audit Protocol

Protocol ID: `EFFDOCK-RDKIT-FRAGMENT-GEOMETRY-AUDIT-V1`

Status: pre-registered for the full 1,076-system PLINDER validation audit.

## Question

The retained geometry model was trained with fragment-local coordinates from
the deposited PLINDER ligand, while SMILES inference constructs one
label-blind conformer with RDKit. How much intrafragment error remains when
every inference fragment is placed optimally by an independent proper rigid
transformation?

This protocol does **not** preserve or use the RDKit whole-conformer placement.
Fragment translations and rotations remain independent, exactly as in the
current training and inference prior.

## Hypothesis and decision rule

The hypothesis is that RDKit-versus-crystal internal geometry is material for
a nontrivial validation subset, particularly ring-containing fragments.

The primary estimand is the per-complex atom-weighted optimistic RMSD floor
after a separate proper Kabsch fit for every fragment defined by the inference
SMILES conformer:

```text
floor = sqrt(sum_f min_{R in SO(3), t} sum_{i in f} ||R x_i + t - y_map(i)||^2 / N)
```

The full-cohort interpretation was fixed before opening the 1,076-system
result:

- material mismatch: the complex-level floor p90 is at least `0.5 A`, or at
  least 10% of valid complexes have a floor of at least `0.5 A`;
- small mismatch: p90 is below `0.35 A` and fewer than 5% have a floor of at
  least `0.5 A`;
- otherwise: intermediate, requiring a candidate-level paired probe before
  geometry fine-tuning.

At least 95% of requested complexes must have a complete accepted result. A
coverage failure invalidates the decision rule rather than silently changing
the cohort. At least 95% of valid complexes must additionally have complete,
untruncated stereo-preserving symmetry enumeration; otherwise the result is
`invalid_symmetry_coverage` and no materiality decision is made.

## Frozen inputs and conformer contract

- Cohort: `val` from the existing `data/splits/plinder.json`, in its stored
  membership; all 1,076 IDs are requested.
- Ligand identity: `ligand_rdkit_canonical_smiles` joined from
  `data/plinder_pool.parquet` by the immutable processed sample key.
- Crystal geometry: `data/plinder_processed/<sample_key>/ligand.pt`, checked
  against the original PLINDER 2024-06 v2 ligand SDF before analysis.
- RDKit recipe: public SMILES loader with seed `0`:
  `MolFromSmiles -> AddHs -> ETKDGv3 -> MMFFOptimizeMolecule(maxIters=200) -> RemoveHs`.
- Failure disposition: an embedding failure is a failed complex. MMFF return
  statuses `0`, `1`, and `-1` are all retained in the primary audit because
  the current public loader silently consumes the resulting conformer; status
  counts and slices are reported rather than filtered post hoc.
- RDKit version: the environment-pinned version is recorded in the output.
- Symmetry: enumerate exact stereo-preserving full graph matches, capped at
  1,024 and explicitly flag truncation. Representation fallbacks accept only
  one complete element/connectivity-preserving mapping and must also pass the
  stereochemistry check; their symmetry enumeration is reported incomplete.
- Fragmentation: run the production greedy fragmentation independently on the
  RDKit conformer. A difference from the stored crystal-derived partition is a
  reported diagnostic, not an exclusion.

The public docking path currently defaults SMILES conformer generation to seed
`0`; the external benchmark evaluator uses a different per-complex seed
contract. This audit addresses the public default only. No seed is selected
using the crystal residual.

## Outputs

The complete JSON contains pool/split/code/protocol hashes and, for every
requested complex, SHA-256 identities of the processed `ligand.pt`, processed
`meta.pt`, and raw ligand SDF. It also contains their deterministic cohort
digest, environment identity, every success and failure record, the selected
full atom mapping, fragmentwise residuals, ring slices, partition parity, and
aggregate distributions.

Registered full output:

```text
outputs/analysis/rdkit_fragment_geometry_v1/val1076_seed0.json
```

An engineering smoke on the first 20 stored validation IDs was opened before
the full run. It established only code and schema behavior; it is not an
independent confirmation result. The thresholds above were encoded before
that smoke was executed and are not changed by it.

## Downstream boundary

A material result authorizes preparation of a paired, internal PLINDER
fine-tuning ablation; it does not itself admit a new model. The correct
treatment uses the inference RDKit fragment-local template, recomputes all
geometry-derived graph references, and defines translation/rotation labels by
proper Kabsch alignment to the crystal pose. The independent fragment prior is
unchanged. Both treatment and control must later be evaluated from actual
SMILES inputs at the frozen deployment `sigma`, pocket, candidate count, solver
steps, and time grid, using crystal coordinates as evaluation truth.
