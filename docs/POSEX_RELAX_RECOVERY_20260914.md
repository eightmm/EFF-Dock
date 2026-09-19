# PoseX relaxation recovery — 2026-09-14

Scope: recover EFF-Dock relaxation outputs. Other models use their published
paper numbers; no other model inference is submitted.

## Evidence and limits

- Historical jobs include FAILED and CANCELLED states. Examined records do not
  establish node failure as the sole cause. Case-level OpenMM errors also exist.
- Upstream `relax_model_outputs.py` catches case exceptions, logs them, then
  returns success. Process exit status alone does not establish output coverage.
- On the distributed PoseX CIF, `7YYB_PNS` reproduces three unmatched residues
  after the exact RDKit-to-PDB conversion: GLY 465, GLY 469, GLN 470 (zero-based
  topology indices). GLY C154 atom identities collide following reconstruction.
  Unexpected cross-residue bonds appear; these are not duplicate bond objects.
- Removing CONECT records diagnostically removes the template mismatch, but
  this is NOT used as a repair because it can remove legitimate connectivity.
- Renumbering only chains with repeated residue IDs before upstream RDKit
  conversion preserves the original topology bonds. The 7YYB test preserved
  all 7,149 resulting RDKit atoms across PDB conversion, found zero unmatched
  force-field residues, and verified that applying the repair again is a no-op.
- This establishes one failure mechanism, not the explanation of every case.
  The separate `findMissingResidues` KeyError remains unresolved.

## Recovery contract

Use distributed CIF files, original EFF-Dock raw selected poses, unchanged
upstream force field, restraint and minimization settings. Do not substitute
fixed-receptor adapter outputs. Identity repair is an explicit compatibility
change, NOT an assertion of byte-identical upstream processing. Print the
old/new residue mapping for provenance; retain atom order, coordinates, bonds.

`recover_posex_upstream_relax.py` records hashes and fails if any expected final
protein/ligand pair is missing or empty. File existence is not a substitute for
the downstream structural/PoseBusters evaluation. Do not report an incomplete
cohort as a completed benchmark.

## Submitted work

- Unmodified distributed-CIF retries: 73830 (SD303 shard 0, 80 cases), 73831
  (CD101 shard 0, 82 cases). These were observed RUNNING.
- Identity-repair GPU smoke: 73842 (7YYB), 73843 (8X35), observed RUNNING.
- Full identity-repair recovery requires BOTH smoke jobs to succeed, with
  `afterok:73842:73843` and `kill-on-invalid-dep=yes`:

| Seed | SD array (718 cases) | CD array (1,312 cases) |
| --- | --- | --- |
| 101 | 73844 | 73845 |
| 202 | 73846 | 73847 |
| 303 | 73848 | 73849 |

Each array permits one running shard, at most six concurrent GPUs for full
recovery. Corrected outputs are isolated under each run's
`upstream_residue_identity_v1/attempts/`. Prior results remain unchanged.
Full corrected runs do not reuse old CIF/identity-condition outputs.

Validation performed: Python AST and shell syntax checks; real 7YYB CPU
topology/atom/template assertions. GPU smoke, full coverage, unresolved error
repair, and final official evaluation remain pending at submission time.

## Follow-up during the same recovery

Both 73842 and 73843 completed actual GPU relaxation and wrote final outputs.
The newly added shell coverage loop then incorrectly treated export JSON/CSV
metadata files as case directories, causing job failure and automatic
cancellation of 73844–73849. The loop now skips non-directories.

Replacement smoke jobs: 73850 (7YYB), 73851 (8X35). Replacement full arrays,
dependent on BOTH replacement smoke successes:

| Seed | SD | CD |
| --- | --- | --- |
| 101 | 73852 | 73853 |
| 202 | 73854 | 73855 |
| 303 | 73856 | 73857 |

Actual relaxation success for the two formerly failing cases is verified in
73842/73843 stderr, including final protein and ligand writes. The full cohort
is not yet complete. The repaired protocol must still disclose this explicit
preprocessing compatibility change when compared to published numbers.
