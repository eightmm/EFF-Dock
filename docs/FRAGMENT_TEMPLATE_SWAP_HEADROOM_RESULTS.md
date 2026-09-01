# Fragment-Template Swap Headroom Results

## Decision

A long RDKit-local geometry fine-tune is **not prioritized**. On the completed
sigma-2, N=100 Astex and PoseBusters ensembles, even an optimistic oracle that
also relaxes 14 real input-versus-crystal stereochemistry conflicts increased
the full-cohort mean by only `+78/393 = +0.19847 K2` candidates per 100. This
is far below the frozen strong-headroom gate of `+1.0 K2`.

The intended intervention is smaller still. The stereo-preserving evaluable
cohort gained `+55` candidates over 379 complexes (`+0.14512 K2`), and a
clearly labelled zero-effect imputation for the 14 non-evaluable stereo
mismatches gives `+55/393 = +0.13995 K2`. Seventeen of the additional 23
candidates in the optimistic bound come from Astex `1gpk`, where the oracle
replaces two opposite stereocenters inside one 18-atom rigid fragment. A
RDKit-template fine-tune cannot reproduce that stereochemical correction.

The exact frozen V1 deprioritization rule cannot formally pass because its
whole-molecule stereo-preserving contract left 14 complexes non-evaluable.
The full optimistic sensitivity also has two, rather than fewer than two, new
`K>=1` complexes and is therefore formally intermediate. Scientifically, the
strong-headroom hypothesis is rejected robustly and there is no support for a
long run. If revisited after higher-priority model changes, the maximum
justified investment is one short internally paired ablation; confidence must
remain frozen until geometry and deployment inference are frozen.

## Full-cohort optimistic sensitivity

| Dataset | Complexes | K2 before | K2 after | Mean delta / 100 | New `K>=1` |
|---|---:|---:|---:|---:|---:|
| Astex | 85 | 2,382 | 2,400 | +0.21176 | 1 |
| PoseBusters | 308 | 8,299 | 8,359 | +0.19481 | 1 |
| Combined | 393 | 10,681 | 10,759 | **+0.19847** | **2** |

Across all 39,300 candidates, 100 poses entered `<2 A` and 22 exited it, for
the net gain of 78. The complex-level crossings were `+2/-0` for `K>=1`,
`+4/-0` for `K>=5`, and `+2/-0` for `K>=10`. Astex and PoseBusters both had a
non-negative macro delta, but the combined effect reached only 19.8% of the
registered `+1.0` strong-headroom threshold.

This full result is an upper bound, not the V1 primary estimate. Fourteen rows
have `mapping.sensitivity_only=true`: exact atom identity and connectivity are
preserved, but all exact mappings conflict in R/S stereochemistry. Their total
was `389 -> 412`, or `+23`; `1gpk` alone contributed `+17`.

## Stereo-preserving result and failure handling

The original V1 job `54218` reported 378 successful stereo-preserving rows:

- Astex: 82/85, `+1 K2`, mean `+0.01220`;
- PoseBusters: 296/308, `+54 K2`, mean `+0.18243`;
- combined: `+55/378 = +0.14550 K2`;
- new `K>=1`: one complex.

Its remaining failures were 14 genuine R/S mismatches and one apparent SDF
parse failure. The latter, PoseBusters `7rou_66i` candidate 95, was not corrupt:
its registered SHA256 and serialized record are intact. RDKit 2025.09.5's
indexed supplier mis-seeked the record at a 4096-byte buffer boundary. A
sequential `ForwardSDMolSupplier` recovered all 100 records without changing
the file; `7rou_66i` has a valid stereo-preserving map and `15 -> 15 K2`.
Thus the corrected stereo-preserving evaluable cohort is 379/393 with the same
net `+55`.

There is no mapping-only repair for the 14 remaining systems. Every one has an
exact full constitutional graph map, but none has a stereo-compatible map,
including after enumerating all non-chiral symmetries and reassigning stereo
from 3D. They remain non-evaluable under frozen V1 and are not silently treated
as observed zero-effect rows.

## Scope

The probe preserves each saved fragment centroid and orientation and replaces
only fragment-internal geometry via an independent proper Kabsch frame. It
never copies whole-ligand crystal placement. It is still a post-hoc endpoint
counterfactual: graph features, guidance forces, and the sampling trajectory
were generated from the original RDKit template. It therefore cannot prove
that a fine-tuned model would attain even this small gain.

The source candidates used sigma `2.0`, N=100, S=10 and normalized-drift
guidance `eta=2`; this result is not an unguided deployment-solver admission
test. Saved confidence and validity labels were not reused after geometry
replacement.

## Provenance

- Primary protocol: `docs/FRAGMENT_TEMPLATE_SWAP_HEADROOM_PROTOCOL.md`
- Sensitivity protocol:
  `docs/FRAGMENT_TEMPLATE_SWAP_HEADROOM_SENSITIVITY_PROTOCOL.md`
- Primary job: `54218` (expected completeness failure after writing partial
  results)
- Full sensitivity job: `54233`, both tasks `COMPLETED (0:0)`
- Primary Astex JSON SHA256:
  `ea0a1217b8cefada77e16a3c9ec4ce3063bbe11a96237b97271b487fb554586e`
- Primary PoseBusters JSON SHA256:
  `e1bf84b74685fece96033822079a790d28f016a89371444a7d5ee2d99b8c9201`
- Sensitivity Astex JSON SHA256:
  `8e9827509449fcbd721fe40f341ecf6d9d9fc1453e45caf77fbe52cfdf9b65a3`
- Sensitivity PoseBusters JSON SHA256:
  `32eba7636c9e6e29e595ae4c72264de0b3c578b838fc1d0a5c0755264cb9ca36`
