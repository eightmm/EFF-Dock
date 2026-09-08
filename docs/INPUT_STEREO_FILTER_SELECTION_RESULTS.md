# Input-SMILES Stereo-filter Selection Results

## Status and claim boundary

This post-hoc descriptive ablation is complete. It does not promote or admit a
new production selector because Astex and PoseBusters outcomes were already
open. The selector uses only the frozen inference-time ligand SMILES and U70k
predicted RMSD; crystal coordinates, reference-ligand stereochemistry, RMSD,
and PoseBusters outcomes are unavailable until after Top-1 selection.

## Frozen comparison

- Candidates: completed cutoff-10, sigma-2, eta-2 normalized-drift,
  `N=100/S=10` Early-time-50k ensembles after adaptive 100-step refinement.
- Repeats: three existing sampling repeats.
- Baseline: minimum U70k predicted RMSD over all 100 refined poses.
- Intervention: derive tetrahedral and double-bond stereo from each candidate's
  3D coordinates, discard candidates that conflict with stereochemistry
  explicitly declared by the frozen input SMILES, then minimize the unchanged
  U70k predicted RMSD.
- Empty-filter policy: retain the original Top-1 and record an explicit
  fallback; never silently remove a benchmark complex.
- Evaluation: symmetry-aware RMSD below 2 Angstrom and official PoseBusters
  0.6.5 `redock` 27-check validity after selection.

## Three-repeat mean

| Dataset | Selector | RMSD <2A (%) | PB-valid (%) | Joint (%) |
|---|---|---:|---:|---:|
| Astex | U70k | 82.35 | 93.33 | 78.04 |
| Astex | stereo filter + U70k | 82.35 | 94.51 | 78.43 |
| Astex | delta (pp) | +0.00 | +1.18 | +0.39 |
| PoseBusters v2 | U70k | 82.68 | 93.07 | 78.90 |
| PoseBusters v2 | stereo filter + U70k | 82.36 | 95.35 | 79.98 |
| PoseBusters v2 | delta (pp) | -0.32 | +2.27 | +1.08 |

PoseBusters per-repeat joint changes were `+1.30`, `+0.97`, and `+0.97`
percentage points. Astex joint changes were `+0.00`, `+1.18`, and `+0.00`
percentage points.

Across the three PoseBusters repeats, the filter changed 60 selections and had
no zero-candidate fallback. The minimum number of compatible candidates in a
100-pose complex was six. The corresponding Astex counts were eight changed
selections, no fallback, and a minimum of 31 compatible candidates.

The final V4 filter preserves the raw-SMILES heavy-atom order and constrains
only model-observable stereo. In particular, it does not constrain imine E/Z
notation whose defining substituent is an explicit hydrogen removed by the
frozen heavy-atom input policy. V1 compared whole InChI stereo layers and was
superseded because it incorrectly constrained unassigned (`?`) centers. V2
fixed that issue but still constrained explicit-H-only E/Z. V3 was cancelled
after an atom-order mismatch was found in canonical-SMILES reparsing. None of
these superseded runs is claim-bearing.

## Verification and provenance

- Selection/PoseBusters array: Slurm `65188`, 48/48 tasks completed with exit
  code 0.
- Aggregate report: Slurm `65199`, completed with exit code 0.
- Result inventory: 48 shard summaries, 1,179 selected rows, zero recorded
  errors.
- Focused verification: Ruff, Python compilation, and eight stereochemistry
  filter tests passed.
- Ignored machine-local aggregate JSON SHA-256:
  `d3a40735d440ccfb944061a31540ff37842ac05e9c0974f8bac328510b3e4de7`.
- Ignored machine-local aggregate Markdown SHA-256:
  `664f22c0b9bb0f2709062f6e7873a90c8e7555ffa278b36b0b5aac05779ce5b7`.
