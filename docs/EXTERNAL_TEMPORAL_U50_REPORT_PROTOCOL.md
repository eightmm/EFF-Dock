# External temporal U50 reporting protocol

Protocol ID: `EFFDOCK-EXTERNAL-TEMPORAL-U50-REPORT-V1`

## Purpose

Report the saved PhiBench, FoldBench, and OpenBind pose banks with the terminal
50,000-update symmetry-confidence checkpoint.  This changes only pose ranking;
it does not regenerate docking poses or repeat physical refinement.

## Frozen inputs

- Source run:
  `outputs/benchmarks/external_temporal_guided_refined_runs/20260825T000806Z`
- Docking checkpoint SHA-256:
  `65be44d7dc8f0867eb9fc5d22214b80f93971ea4702679a527c665046e91e6b6`
- U50 confidence checkpoint SHA-256:
  `fd49fa86f67187bf26d6c1bcf2daf925ba3e3b19dfeae733e57535d183280469`
- Candidate bank per complex: `N=100`, `S=10`, `sigma=2`, normalized-drift
  physical guidance with `eta=2`.
- Refinement: the already-saved adaptive physical refinement trajectory, at
  most 100 steps.
- Cohorts: PhiBench `N=203`, FoldBench `N=66`, OpenBind clean pocket-redocking
  cohort `N=860`.

Every source refinement summary, trajectory, protein, reference ligand,
docking checkpoint, and confidence checkpoint is hash-checked before scoring.

## Selection and metrics

The selector is the stable minimum U50-predicted RMSD over the 100 poses.  It
is applied independently to the saved pre-refinement (`step_000`) and
post-refinement (`step_100`) ensembles.  RMSD and validity are outcomes and do
not enter selection.

For the selected post-refinement pose, report:

- symmetry-aware ligand RMSD `< 2 A`;
- PL-validity over the protein-ligand PoseBusters subset;
- official PoseBusters 0.6.5 pass-all validity;
- the conjunction of RMSD `< 2 A` and validity;
- the unchanged 100-pose oracle.

OpenBind's filtered scaffold-only `N=802` Top-1/5/25 table is reranked from the
same U50 confidence ledger and reevaluated with PoseBusters 0.6.5 and
OpenStructure 2.11.1.

## Claim boundary

The U25 external results were already opened before the request to use U50 as
the project reporting convention.  U50 values are therefore descriptive and
must not be presented as a blind external checkpoint-selection result.  The
historical U25 protocol and artifacts remain unchanged for audit provenance.
