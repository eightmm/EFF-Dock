# Chemical-constraint unit-weights full-cohort protocol

Status: frozen before full-cohort execution; repeated external descriptive
characterization, not a production-setting selection.

## Question

Across all 85 Astex Diverse and 308 PoseBusters v2 reference-pocket redocking
complexes, what changes when the separately normalized chemical-constraint
drift is enabled at outer strength 1 while the existing combined
physical-plus-interaction drift remains at outer strength 1?

## Paired arms

- `physical_interaction_1`: physical-plus-interaction outer strength 1;
  chemical strength 0.
- `all_three_1`: physical-plus-interaction outer strength 1; chemical strength
  1.

Both arms use identical per-complex prior pools. “Strength 1” applies only to
the normalized drift channels. Versioned term-level physical and interaction
constants retain their declared units and values.

## Frozen inference

- docking checkpoint: `weights/effdock_docking_early_time_t0p10_50k.pt`,
  SHA-256 `65be44d7dc8f0867eb9fc5d22214b80f93971ea4702679a527c665046e91e6b6`;
- confidence checkpoint: `weights/effdock_confidence_s50_raw_refined_u70k.pt`,
  SHA-256 `ce59be42f0ca613871ca079127c3296f5ca9a4ec72e44a9e5cf61878351c2638`;
- sampling: N100/S10, translation sigma 2, late-power-3 time grid, pocket
  cutoff 10 A, no center jitter, seed 42, ligand conformer seed 0;
- guidance: normalized direct drift, shared start t=0.5, linear ramps,
  physical-plus-interaction atom-displacement cap 0.25 A, chemical-channel
  cap 0.10 A, and `geometry_only` receptor policy;
- ranking: pure U70k predicted-RMSD argmin (`confidence` selector from the
  cluster-free profile);
- refinement: none.

## Coverage and metrics

A fresh current-implementation audit must admit exactly 85/85 Astex and
308/308 PoseBusters inputs before GPU work. Every sampling and official
PoseBusters shard fails on a missing or failed assigned complex.

Report for each dataset and arm:

- confidence Top-1 and oracle symmetry-aware RMSD-under-2-A success;
- all-candidate declared-stereo validity and stereo-valid RMSD-under-2-A;
- fast-valid and joint candidate/Top-1 diagnostics;
- official PoseBusters 0.6.5 `redock` pass-all over non-RMSD checks for the
  confidence-selected pose, plus its conjunction with RMSD under 2 A;
- tetrahedral and double-bond stereo checks on the selected pose;
- chemical-channel applications, cap triggers, nonfinite events, and runtime.

## Interpretation boundary

The external benchmark outcomes and the three-case smoke were already opened.
All full-cohort values are paired descriptive measurements. They cannot tune
the strength, select a production setting, or replace internal PLINDER
validation.
