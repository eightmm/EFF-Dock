# OpenBind official-style Top-25 aggregation

Protocol ID: `EFFDOCK-OPENBIND-OFFICIAL-TOP25-V1`

Status: frozen before the Top-25 PoseBusters or OpenStructure outcomes are inspected.

## Scope

This evaluation re-aggregates the completed EFF-Dock OpenBind redocking run at
`outputs/benchmarks/external_temporal_guided_refined_runs/20260825T000806Z` under
the public OpenBind figure contract. No pose is regenerated, rescored, or selected
using an outcome metric.

## Denominator and ranking

- Reproduce the OpenBind `filtered=True, scaffold_only=True` denominator from
  `EV-A71_2A_metadata.csv`: prepared ground truth must be PoseBusters-valid,
  suspected artefacts are excluded, and fragment-screen structures are excluded.
- The frozen denominator is 802 complexes. The source EFF-Dock run contains 786
  of them. Its additional `pb_valid_ref=True` preparation filter excluded 16
  official-denominator complexes; these remain in the denominator as missing
  predictions and therefore failures.
- Rank each complex's 100 post-refinement poses by ascending frozen
  `after_confidence_rmsd`, breaking ties by original pose index. Ranks are
  zero-based. Top-N means `rank < N` for N in 1, 5, and 25.

## Pose-level metrics

- PoseBusters: version 0.6.5, `redock`, pass all 27 non-RMSD binary checks. The
  separate PoseBusters RMSD module is omitted because its output is not part of
  validity and OpenStructure supplies the endpoint RMSD.
- Ligand RMSD: OpenStructure 2.11.1 `compare-ligand-structures --rmsd`
  BiSyRMSD, using the same prepared protein as model and reference and explicit
  predicted/reference ligand SDFs.
- LDDT-PLI: the same OpenStructure call with `--lddt-pli`.
- `rmsd_valid`: PoseBusters-valid and BiSyRMSD <= 2 Angstrom.
- `success_valid`: `rmsd_valid` and LDDT-PLI >= 0.8.

The existing RDKit symmetry-aware RMSD is retained only as a diagnostic and is
not used for the two primary OpenBind-compatible endpoints.

## Complex aggregation

For each N, a complex passes an endpoint if any of its confidence-ranked Top-N
poses passes. Missing predictions count as failures. OpenStructure evaluation
may stop after the first `success_valid` pose for a complex because that pose
also establishes every later Top-N endpoint containing its rank; otherwise all
PB-valid Top-25 poses are evaluated.

## Execution

PoseBusters and OpenStructure run only on Slurm `cpu_only`, without an explicit
CPU-count request. A one-complex PB-to-OpenStructure smoke chain gates the full
64-shard arrays. The report runs only after every full shard exits successfully.
