# SigmaDock-compatible official PoseBusters protocol

## Purpose

This protocol evaluates frozen EFF-Dock Top-1 poses with the same core
PoseBusters redocking convention used by SigmaDock. It is a post-selection
evaluation and does not use PoseBusters outcomes to choose the reported pose.

## Frozen evaluation

- Source sampling study: eta=2.0, sigma={0.5,1.0,2.0,3.0,4.0}, N100/S10.
- Cohorts: all 85 Astex Diverse complexes and all 308 PoseBusters v2
  complexes frozen in `GUIDANCE_BUDGET1000_FULL_INPUTS.json`.
- Selector: only the frozen primary `confidence` Top-1 is reported. Diagnostic
  selectors are excluded from this protocol.
- Checker: PoseBusters 0.6.5, `PoseBusters(config="redock", max_workers=0)`,
  `full_report=False`.
- Inputs: selected SDF, cognate reference ligand, and the full protein path
  recorded during sampling. All three are verified against sampling-time
  SHA-256 values before evaluation.
- PB validity excludes the separate RMSD column.
- Primary PB-valid is the conjunction of all 27 non-RMSD checks emitted by
  PoseBusters 0.6.5.
- A comparison-only SigmaDock-legacy view is also recorded using the 26 checks
  explicitly listed in SigmaDock's current statistics code. The only omitted
  0.6.5 check is `no_radicals`; this compatibility view never replaces the
  stricter primary result.
- The joint headline metric is selected symmetry-aware RMSD <2 Angstrom and
  PB-valid on the same Top-1 pose and the same exact cohort denominator.

## Claim boundary

SigmaDock's public package does not pin a PoseBusters version. Therefore an
unqualified comparison of its listed 26-check implementation to EFF-Dock's
27-check pinned runtime would be ambiguous. Both views are stored, with the
version and complete check lists in the aggregate JSON. EFF-Dock selection is
confidence-based and is not claimed to reproduce SigmaDock's GNINA/Vinardo
ranking.

Official references:

- https://github.com/alvaroprat97/sigmadock
- https://github.com/alvaroprat97/sigmadock/blob/main/src/sigmadock/chem/statistics.py
- https://github.com/maabuu/posebusters

## Execution chain

The fail-closed Slurm chain is: 10-cell one-complex smoke, smoke audit,
80-task full evaluation (1 selector x 2 datasets x 5 sigmas x 8 shards),
then exact-inventory aggregation. Any missing pose, changed input hash,
PoseBusters exception, schema drift, or incomplete shard fails the chain.
