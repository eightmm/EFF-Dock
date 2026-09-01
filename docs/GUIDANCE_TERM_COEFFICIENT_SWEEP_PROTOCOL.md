# Guidance term-coefficient sweep protocol

- Protocol: `EFFDOCK-UNIFIED-GUIDANCE-TERM-COEFFICIENT-SWEEP-V1`
- Status: pre-registered paired descriptive ablation.
- Purpose: isolate whether the protein-ligand steric contact radius and the
  ligand stereochemical improper restraint improve generated-pose geometry.
- Claim boundary: Astex Diverse and PoseBusters v2 have already been opened.
  These results are descriptive and may not select a production coefficient.
  A later internal PLINDER study is required for selection or admission.

## Frozen sampling contract

- Docking checkpoint: `weights/effdock_geometry_ft_100k_best.pt`
- Confidence checkpoint:
  `weights/effdock_confidence_extmatch_n80_s25_step42500.pt`
- Primary selector: pure minimum predicted RMSD (`confidence_cluster_free`).
- Datasets: complete Astex Diverse (`85`) and PoseBusters v2 (`308`) cohorts.
- Budget: `N=100`, `S=10`, scalar prior `sigma=0.5`, paired global seed `42`.
- Solver: `normalized_drift`, `eta=2.0`, late schedule, guidance start `t=0.5`.
- All other energy terms, caps, pocket settings, and receptor policy remain
  fixed at the current diagnostic defaults.

## Arms

| arm | PL steric radius scale | chiral improper scale |
|---|---:|---:|
| `base_r080_c1` | 0.80 | 1.0 |
| `steric_r090_c1` | 0.90 | 1.0 |
| `chiral_r080_c2` | 0.80 | 2.0 |
| `combined_r090_c2` | 0.90 | 2.0 |

The steric value changes the activation surface
`r_safe = scale * (r_vdw,ligand + r_vdw,protein)`. The chiral value multiplies
only non-planar improper energy; planar impropers and all other ligand-internal
terms are unchanged.

## Execution and reporting

The chain is smoke sampling -> smoke audit -> full sampling -> full audit ->
report, with `afterok` dependencies inside the chain. A new immutable execution
capsule records the exact dirty worktree and implementation. No arm is selected
automatically. The first report contains RMSD, internal fast-validity, and
guidance/cap telemetry; official PoseBusters evaluation is a separate follow-up
stage and must retain the same confidence-selected pose hashes.
