# S50 Refined Symmetry-Aware Confidence Label Sidecar Protocol

Status: frozen before refined train-label materialization on 2026-08-20.

## Objective

After the complete refined N100 confidence bank is sealed, recompute the
training pose target from each saved refined coordinate using symmetry-aware,
no-alignment heavy-atom RMSD. Pose regression, success BCE, ranking, and
success-listwise supervision in the later refined-confidence continuation use
this target. Fixed-map `atom_disp` remains the atomwise auxiliary target.

This is a label-only transform. It never regenerates or refines coordinates,
changes candidate order, filters complexes, or recomputes hidden features.

## Frozen inputs and computation

- Frozen input manifest SHA-256:
  `6a991e964715f9ddb51ce48ccbea5a948eb3b2d3c51de1f26929389365aec5f3`.
- Source pose tag:
  `s50_n100_s10_sig2_latep3_pc10_rdkitseed0_refine100`.
- Expected inventory: 43,092 train complexes, exactly 100 ordered poses each.
- The refined confidence bank manifest SHA-256 is read only from its sealed
  no-overwrite sidecar after job 56622 succeeds and is copied into every label
  shard and the aggregate manifest.

For each saved pose, reconstruct the canonical-SMILES molecule and sealed
processed crystal reference, require the frozen strict-stereo full mapping,
and run RDKit `rdMolAlign.CalcRMS` without alignment. The field and method are
`pose_rmsd_symmetry_no_align` and
`rdkit_calc_rms_symmetry_no_align`. There is no RMSD fallback.

Any source/hash/mapping/order/count drift, CalcRMS failure, non-finite value,
duplicate, or incomplete inventory fails closed. Outputs are 128 CPU-generated
train label shards plus one atomic no-overwrite aggregate manifest.

## Execution boundary

A two-complex real-payload smoke gates the 128-shard full run. The aggregate
must contain all 4,309,200 labels. Refined-confidence training may start only
after this manifest, the refined feature bank, and the current 50k
symmetry-label confidence checkpoint are all sealed.
