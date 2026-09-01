# RDKit Fragment-Local Geometry Heavy-Atom Audit Protocol

Protocol ID: `EFFDOCK-RDKIT-FRAGMENT-GEOMETRY-AUDIT-V2`

Status: frozen after diagnosing the V1 coverage failure and before opening the
18 newly recoverable complex-level geometry floors. This is a corrected
follow-up, not an independent replication of V1.

## Reason for V2

V1 exactly reproduced the public SMILES loader and returned 1,017 valid results
from 1,076 requested validation systems (`94.52%`), below its frozen `95%`
coverage gate. Failure diagnosis found that 18 of the 28 atom-count mismatches
were caused solely by RDKit `RemoveHs` retaining one explicit hydrogen used to
encode terminal C=N stereochemistry. The model and frozen external benchmark
contract are heavy-atom-only; the external benchmark loader already applies
`Chem.RemoveAllHs`.

V2 changes only this label-blind normalization. It does not change the
materiality thresholds, select a conformer using the crystal structure, or
repair any chemically ambiguous record.

## Frozen conformer and failure contract

- Cohort: all 1,076 stored `val` IDs from `data/splits/plinder.json`.
- Ligand identity and crystal inputs: identical to V1.
- RDKit seed: `0`.
- Conformer recipe:
  `MolFromSmiles -> AddHs -> ETKDGv3 -> MMFF(200) -> RemoveHs -> RemoveAllHs`.
- `RemoveAllHs` is applied uniformly to every generated conformer, before atom
  mapping or fragmentation, without consulting crystal coordinates.
- No alternative embedding seeds, `useRandomCoords`, largest-component
  selection, partial-atom mapping, or stereo wildcard is allowed.
- ETKDG failures, unresolved crystal heavy atoms, multi-component target-policy
  mismatches, and input-versus-crystal stereo conflicts remain failed records.
- Whole-conformer RDKit placement is discarded. Every inference-defined
  fragment is still fitted independently by a proper Kabsch transform.

The expected coverage improvement from the already opened failure taxonomy is
reported as provenance, not as a result: the 18 normalization failures are
deterministically recoverable. Their geometry floors and the resulting V2
aggregate decision were not opened when this protocol was frozen.

## Unchanged decision rule

- Valid only if complete accepted mapping coverage is at least `95%` and
  complete, untruncated stereo-preserving symmetry coverage among valid records
  is at least `95%`.
- Material mismatch: complex-level floor p90 is at least `0.5 A`, or at least
  `10%` of valid complexes have floor at least `0.5 A`.
- Small mismatch: p90 is below `0.35 A` and fewer than `5%` have floor at least
  `0.5 A`.
- Otherwise the result is intermediate and requires a candidate-level paired
  probe before any RDKit-local geometry fine-tuning.

## Registered output

```text
outputs/analysis/rdkit_fragment_geometry_v2/val1076_seed0_heavy_only.json
```

The JSON must retain the full V1 provenance, per-record input hashes, exact
mapping/symmetry metadata, fragment residuals, ring and MMFF slices, and record
the `remove_all_hs` policy and this protocol hash.
