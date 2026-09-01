# EFF-Dock early-time/t=0 docking checkpoint

- File: `effdock_docking_early_time_t0p10_50k.pt`
- SHA-256: `65be44d7dc8f0867eb9fc5d22214b80f93971ea4702679a527c665046e91e6b6`
- Model type: fragment-level SE(3)-equivariant flow-matching docking model
- Training endpoint: 50,000-update early-time/t=0 replay fine-tune EMA
- Paired confidence checkpoint: `effdock_confidence_s50_raw_refined_u70k.pt`

## Intended use

Generate candidate ligand poses for a receptor, ligand chemistry, and explicit
binding-pocket center. The promoted inference preset uses 100 poses, 10 ODE
steps, translation sigma 2.0, a 10-Angstrom pocket crop, and a late-power-3
time grid. The paired U70k confidence model ranks the generated poses.

## Training intervention

The 50,000-update run initialized from the previous geometry checkpoint and
used all 47,277 filtered training systems from the preserved PLINDER split.
Its time distribution was `0.80 SimpleFold + 0.10 U(0,0.3) + 0.10 exact t=0`.
The run used fresh AdamW state, EMA decay 0.999, and a registered internal
PLINDER-validation endpoint.

Internal single-pose rollout success below 2 Angstrom improved from
192/1,076 at step 0 to 219/1,076 at step 50,000. On the subsequently inspected
Astex and PoseBusters N100 banks, the model increased the number of
sub-2-Angstrom candidates but reduced any-hit coverage by five complexes. The
supported interpretation is improved candidate density on already-solvable
complexes, not universal coverage improvement.

## Limitations

- Requires an explicit pocket; it does not discover pockets.
- Output depends on receptor preparation, ligand protonation/stereochemistry,
  pose count, sigma, ODE budget, crop, and seed.
- The model does not predict binding affinity or binder status.
- External redocking cohorts were opened during development and are
  descriptive rather than independent model-selection sets.
- The pinned public environment targets Linux and NVIDIA CUDA 13.

The registered training and external characterization are documented in
`docs/EARLY_TIME_FINE_TUNE_50K_PROTOCOL.md` and
`docs/EARLY_TIME_T0P10_50K_EXTERNAL_PAIRED_RESULTS.md`.
