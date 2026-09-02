# FoldBench-Pocket-558 protocol

Protocol ID: `EFFDOCK-FOLDBENCH-POCKET-558-V1`

Status: frozen before any FoldBench-Pocket-558 model output is generated.

## Purpose

Measure EFF-Dock coverage and pose quality on all 558 protein-ligand interface
rows in the official FoldBench interface table. This is a descriptive
crystal-pocket redocking adaptation, not the native FoldBench cofolding
leaderboard contract.

## Inputs

- Official FoldBench protein-ligand interface CSV: 558 rows and 554 assemblies.
- Prediction unit: one protein-chain/ligand-chain/CCD interface. Stable IDs
  include all three fields so the four multi-ligand assemblies remain distinct.
- Holo protein chain and reference-defined 10 A pocket center.
- Ligand input: canonical isomeric SMILES; reference coordinates are used only
  for the frozen pocket center and post-hoc evaluation.
- Two bridged ligands (`W3I`, `W3C`) fail the primary seed-specific ETKDGv3
  embedding. They use the same-seed ETKDGv3 fallback with chirality enforcement
  relaxed during distance-geometry embedding only; all SMILES-declared chiral
  centers must match the generated 3D structure or preprocessing still fails.
- Two official reference chains omit one heavy atom each: `8ok4/GLC O1` and
  `8jik/PLP O4A`. Their missing coordinate is reconstructed by local rigid
  alignment of the ideal CCD conformer while every observed coordinate is
  preserved. Results are reported both for all 558 interfaces and the 556
  fully observed references.

## Frozen model and inference

- Docking checkpoint SHA-256:
  `65be44d7dc8f0867eb9fc5d22214b80f93971ea4702679a527c665046e91e6b6`.
- U70k confidence checkpoint SHA-256:
  `ce59be42f0ca613871ca079127c3296f5ca9a4ec72e44a9e5cf61878351c2638`.
- Candidate generation: `N=100`, `S=10`, translation sigma `2`, late-power-3
  time schedule, 10 A pocket crop, seed `42`, no inference-time gradient
  guidance, and no center jitter.
- Refinement: all 100 candidates, at most 100 steps, with the existing adaptive
  energy-plateau stop (`0.02 kcal/mol`, relative `0.001`, patience `5`, minimum
  `25` steps).
- Selection: stable minimum U70k predicted RMSD, independently at raw step 0
  and refined step 100. Reference RMSD and validity never enter selection.

## Endpoints

Report symmetry-aware Top-1 RMSD `<2 A`, 100-pose oracle RMSD `<2 A`, PL
validity, official PoseBusters redock validity, and the validity-plus-RMSD
joint endpoint. Report four fixed slices: all 558, strict post-2024-06-30 66,
the remaining 492, and the 556 fully observed references.

## Interpretation boundary

The post-cutoff 66 slice preserves the prior temporal definition. The older
492 interfaces may overlap PLINDER training data and are therefore a coverage
and stress characterization, not an independent generalization estimate.
Neither slice is directly comparable to native FoldBench leaderboard values,
which use a cofolding input contract and joint LRMSD/LDDT-PLI success.
