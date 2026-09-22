# Released training membership

`samples.csv` publishes only immutable PLINDER sample IDs, original split
assignment and input-eligibility decisions. It contains no molecular structures,
sequences, coordinates, machine paths or optimizer state. Obtain PLINDER
2024-06/v2 separately under its own terms.

| Set | Samples |
|---|---:|
| Preserved training split | 47,310 |
| Docking fine-tune loader | 47,277 |
| Confidence training split | 43,092 |
| Common docking/confidence training samples | 43,067 |
| Docking only | 4,210 |
| Confidence only | 25 |
| Union of these two training sets | 47,302 |
| Preserved validation split / relatedness query cohort | 1,076 |
| Confidence validation selection bank | 1,035 |

The unit is `(system_id, ligand_instance_chain)`, encoded as
`<system_id>__<ligand_instance_chain>`. Docking membership contains 45,441 unique
system IDs and 43,392 PDB IDs; these counts are not alternative training sample
counts. The reference for the published relatedness figures is the **47,277
sample docking set**, not the 43,092 confidence set or their union.

Columns `docking_train_member`, `confidence_train_member` and
`confidence_validation_member` are binary. `split=val` includes all 1,076
preserved validation IDs, including the 41 excluded from confidence input
preparation. All train-membership flags on validation rows are zero. Original
split order is retained within train then val; set hashes use the encoding
specified in `manifest.json`.

## Why the two training sets differ

Starting independently from the preserved 47,310 training IDs:

- Docking filters remove 32 samples with fewer than 50 protein residues and
  one with more than 30 fragments.
- Confidence input preparation removes 3,466 processed-reference atom-mapping
  incompatibilities and 752 canonical-ligand preparation failures.
- Of the 33 docking exclusions, 25 short-protein inputs pass confidence
  preparation. Confidence is therefore not a subset of docking training.
- Confidence validation removes 22 mapping failures and 19 preparation
  failures from 1,076, leaving 1,035.

Exclusion columns record the original eligibility categories, without
reinterpreting every mapping failure as stereochemistry or every preparation
failure as conformer embedding. These are input filters, separate from the
historical external-benchmark exclusion predicate. No recovered input was
silently added to the released training sets.

## Evidence and scope

`manifest.json` binds the CSV and sorted-ID sets to SHA-256 hashes and records
the released checkpoint hashes and frozen local provenance. The docking set
was checked against the actual filtered loader index; the confidence set was
checked against the filtered split named in the U70k checkpoint and every
eligibility record in the frozen input manifest.

The local provenance paths are identifiers for historical audit inputs, not
files shipped in this checkout. Counts/IDs/reasons and their checksums are
public and can be verified without those inputs:

```bash
uv run python -m benchmarks.figures.paper --check
```

This is loader/split membership provenance, not a log of every optimizer read,
not an exact replay of training, and not a union of all earlier pretraining or
warm-start exposure. The newer strict split builder must not be substituted
when explaining the released checkpoints. The historical relaxed exclusion
used a ligand-match **AND** pocket-community-match predicate, with the pocket
set derived from benchmark PDB hits in the processed pool. It did not certify
absence of every broader same-ligand/similar-protein match.

The historical audit in `manifest.json` reproduces 904 removals using 217
external pocket communities and zero retained matches to that original
predicate. Full-index mapping with the current evaluation ligands nevertheless
identifies same-training-sample protein-community/pocket-community/exact-ligand
overlap for 7 Astex and 27 PoseBusters complexes. Six Astex and all 27
PoseBusters PDBs were absent from the processed pool; Astex 1meh had a different
ligand in the original exclusion input. The manifest retains the 34 witnesses.

Validation shares no `pocket_fident__70__community` with executed training,
but 43/1,076 queries share a `protein_fident_qcov_weighted_sum__70__community`
and 460/1,076 share a `pocket_lddt_qcov__70__community`. These historical
community checks neither impose a direct pairwise-identity bound nor certify
novel protein/pocket structure. They do not replace the current sequence-based
figure definitions or imply that the external cohorts are fully disjoint.
