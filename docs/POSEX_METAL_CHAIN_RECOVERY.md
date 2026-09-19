# PoseX metal-chain recovery (2026-09-14)

## Reproduced cause

The remaining SD KeyErrors are not node interruptions. A CPU reproduction of
the unmodified sequence mapper found distinct topology chain objects with the
same PDB chain ID:

| Case | Protein chain(s) | Additional chain(s) causing the error |
| --- | --- | --- |
| 7HKD_8BT | A, 581 residues | A, two Zn residues |
| 7VQ9_ISY | A/B, 206/203 residues | A/B, one Mg residue each |
| 8WWQ_XQ8 | A/B, 346/317 residues | A/B, one Mg residue each |

Upstream `findMissingResidues` populates `chainWithGaps` only for the first
chain with a given ID, then indexes it while iterating all matching chain
objects. The metal-only objects lack an entry, producing the KeyError.

## Minimal compatibility fix

`scripts/posex_chain_mapping_repair.py` adds a guarded skip before that lookup:
an unmapped chain is excluded from **protein sequence matching only** if all
its residue names belong to the upstream metal list. Empty, mixed, protein,
water, and unknown unmapped chains still raise an error. No atoms, ions,
coordinates, or topology bonds are deleted. Already mapped chains follow
the original method unchanged. The existing residue-ID repair is retained.

The runtime patch requires an exact single upstream source anchor and fails
closed if it changes. The vendored upstream source and currently running v1
jobs are not edited. This remains a disclosed preprocessing compatibility
change, not a claim of byte-identical upstream processing.

## Execution and validation

`scripts/retry_posex_metal_chain_cases.py` retries only the three named cases
from their immutable raw EFF-Dock inputs and distributed PoseX CIF. Input and
code SHA256 hashes are written before execution. New outputs are isolated in
each SD run's `upstream_metal_chain_recovery_v2/processed/`; they are not yet
merged with the complete cohort or evaluated as a final benchmark.

Completion requires both final files, an RDKit-readable single ligand, and
preserved Zn/Mg atom counts relative to the raw receptor. A per-case coverage
report is written; case exceptions cannot masquerade as full success.

- Seed101 three-case validation: job 73953, observed RUNNING.
- Seed202/303 retries: jobs 73956/73957, depend on 73953 success with
  `kill-on-invalid-dep=yes`.
- Unit tests: 3 passed (metal preservation, mapped protein, fail-closed cases).
- Changed-file Ruff checks and shell syntax: passed.
- Project fast check: compilation passed; repository-wide Ruff failed on
  eight unrelated existing findings in benchmark figure/fixed-receptor files.
  The new import-order finding was fixed, and changed-file lint passed.
  The final EFF-Dock import check was run separately and passed.
- GPU completion and the final whole-cohort evaluation remain pending at
  submission time. Other models continue to use published paper numbers.

## Remaining-case recovery — 2026-09-15

Jobs 73953/73956/73957 subsequently completed successfully, including preserved
metal counts in all three seeds. Full v1 coverage plus these outputs is 716/718
SD cases per seed; CD is 1,312/1,312 per seed. Two later SD failures reproduce
the same unmapped metal-chain lookup:

- `7N8E_0UD`: protein A/B plus separate A/B chains containing three Mg each.
- `8EGF_WIK`: protein A plus a separate A chain containing one K.

The existing compatibility fix is unchanged. A `--remaining` option isolates
these two retries under `upstream_metal_chain_remaining_v2`. The output audit
now counts the complete upstream METALS list, including K, instead of Zn/Mg
only. Five unit tests, changed-file lint, Python syntax and shell syntax pass.
No existing successful structures or original run manifests were overwritten.

Recovery jobs: 74720 (101), 74721 (202), 74722 (303); 74720 was observed RUNNING
and the other two resource/priority pending. These jobs perform relaxation and
coverage checks, not the final official RMSD/PoseBusters evaluation.
