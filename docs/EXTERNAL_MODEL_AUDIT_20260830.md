# External-model execution audit

Snapshot: 2026-08-30 KST

This is the authoritative local-inference audit for the external docking models.
An emitted coordinate file is not automatically a benchmark-valid prediction.
All rates use the frozen Astex Diverse `N=85` or PoseBusters Benchmark v2
`N=308` denominator; missing or topology-unmappable predictions are failures.

The comparison tables below are now supplied-pocket-only. Blind/full-receptor,
pocket-prediction, and co-folding models are excluded from reporting; their
previously generated files remain local archival diagnostics.

## Evaluation contract

- Metric: RDKit symmetry-aware heavy-atom `CalcRMS`, without alignment.
- Hydrogen policy: remove all explicit hydrogens before topology matching.
- Cofactor policy: preserve components during loading, then select the first
  component with a full topology match to the frozen primary reference ligand.
- Top-1: each method's released/native selector. A broken rank 1 is a failure;
  rank 2 is never silently promoted.
- Oracle at available N: the best RMSD among every evaluable pose emitted by
  that method. It is not relabeled Oracle@40 for N1, N5, N20, or variable-N
  methods.
- The tables below are **RMSD-only**. Official PoseBusters 0.6.5 27-check
  validity and joint success (`RMSD < 2 A` and PB-valid) have not been applied
  to these external-model Top-1 selections.

## Final executed RMSD results

### Astex Diverse

| Executed arm | Native selection / candidates | Coverage | Top-1 <2 A | Oracle available <2 A |
|---|---|---:|---:|---:|
| RLDiff | confidence / 40 | 85/85 | 65/85 (76.47%) | 77/85 (90.59%) |
| SurfDock | MDN / 40 | 85/85 | 64/85 (75.29%) | 78/85 (91.76%) |
| DiffBindFR | MDN / 40 | 85/85 | 60/85 (70.59%) | 82/85 (96.47%) |
| SigmaDock | Vinardo / 40 independent seeds | 85/85 | 53/85 (62.35%) | 76/85 (89.41%) |
| DiffDock-Pocket | confidence / 40 | 85/85 | 44/85 (51.76%) | 57/85 (67.06%) |
| RLDiff RL++ | native confidence / 40 | 85/85 | 40/85 (47.06%) | 77/85 (90.59%) |
| Interformer | official PoseScore ensemble / 20 | 85/85 | 39/85 (45.88%) | 58/85 (68.24%) |
| PoseBench Vina | Vina affinity / up to 40 | 85/85 | 13/85 (15.29%) | 29/85 (34.12%) |

### PoseBusters Benchmark v2

| Executed arm | Native selection / candidates | Coverage | Top-1 <2 A | Oracle available <2 A |
|---|---|---:|---:|---:|
| SurfDock | MDN / 40 | 308/308 | 183/308 (59.42%) | 256/308 (83.12%) |
| RLDiff | confidence / 40 | 308/308 | 166/308 (53.90%) | 241/308 (78.25%) |
| SigmaDock | Vinardo / 40 independent seeds | 308/308 | 157/308 (50.97%) | 244/308 (79.22%) |
| DiffBindFR | MDN / 40 | 308/308 | 150/308 (48.70%) | 252/308 (81.82%) |
| Interformer | official PoseScore ensemble / 20 | 308/308 | 100/308 (32.47%) | 185/308 (60.06%) |
| DiffDock-Pocket | confidence / 40 | 308/308 | 99/308 (32.14%) | 163/308 (52.92%) |
| RLDiff RL++ | native confidence / 40 | 308/308 | 93/308 (30.19%) | 240/308 (77.92%) |
| PoseBench Vina | Vina affinity / up to 40 | 308/308 | 27/308 (8.77%) | 81/308 (26.30%) |

Machine-readable summaries and per-target rows are under
`outputs/external_models/evaluation/common_rmsd_v8/`.

## Model-specific execution notes

### PoseBench Vina

The corrected protocol uses the PoseBench-pinned Vina engine, exhaustiveness
32, a 25 A box with 1 A spacing, explicit seed 1, and at most 40 output modes.
The receptor pipeline is PoseBench Reduce -> OpenBabel pH 7.4 -> clean ADFR
`prepare_receptor4.py`. Each target runs with the scheduler's default one CPU.
All ranked PDBQT models are retained in one multi-record SDF together with
`vina_mode`, affinity, and Vina RMSD-bound properties. This replaces the
rejected historical run that used a compatibility receptor writer,
nondeterministic `seed=0`, and retained only one SDF record.

Vina may emit fewer than 40 modes because its default output energy window is
3 kcal/mol. Therefore Oracle-available is the honest primary oracle statistic;
strict Oracle@40 additionally treats every target with fewer than 40 evaluable
poses as a failure.

### Interformer

Top-1 is selected by the released v0.2 four-checkpoint PoseScore ensemble using
maximum `pred_pose`; the appended input conformer is excluded. Native
energy-sampling order was retained only as a diagnostic and is not reported in
the final tables. For reference, that diagnostic Top-1 was 47/85 on Astex and
132/308 on PB, versus official PoseScore's 39/85 and 100/308.

### PoseBench DiffDock

Inputs were rebuilt from the frozen primary reference ligand's isomeric SMILES,
then sampled with the released confidence selector, 20 inference steps, five
poses, and seed 0. Astex reached 84/85 evaluable targets; `1T9B_1CS` repeatedly
failed upstream with batch sizes 1, 2, and 5 due to a tensor-shape error. PB
reached 304/308: `7FRX_O88`, `7M31_TDR`, and `8F4J_PHO` exceed the upstream
3000-residue hard limit, while `7XJN_NSD` repeatedly failed with the same
tensor-shape class. These are native limitations and remain denominator
failures rather than being replaced with another protocol.

### DynamicBind

Missing-rank recovery was rerun under clean, unique run tags so upstream
timestamp tags could not merge stale files. All targets now have an evaluable
rank 1. DynamicBind still emits variable N (approximately 35--40 here), partly
because score-based filenames can collide. The evaluator preserves native
rank indices rather than compacting missing ranks.

### Lower-rank integrity

- DiffDock-Pocket PB retains two malformed lower-ranked poses across two
  targets; Top-1 coverage remains 308/308.
- SurfDock Astex retains five coordinate-explosion lower-ranked poses in one
  target after recovery; Top-1 coverage remains 85/85.
- PoseBench DiffDock retains six malformed lower ranks across four Astex
  targets and 22 across 12 PB targets. Its reported coverage and Oracle use
  every evaluable emitted pose and record each bad candidate separately.

## Corrections made during the audit

1. Interformer's stereochemistry-defining explicit `[H]` atoms are excluded
   consistently from the heavy-atom metric with `RemoveAllHs`.
2. Multi-component predictions no longer choose the largest component, which
   could select ATP or another cofactor instead of the named primary ligand.
3. A malformed lower rank no longer invalidates a valid Top-1.
4. DynamicBind filename ranks are preserved; later ranks cannot be promoted.
5. PoseBench DiffDock is now primary-ligand-only and carries explicit terminal
   failure markers, preventing fallback to the older multicomponent run.
6. Vina preserves every emitted mode and uses deterministic, official receptor
   preparation rather than the earlier compatibility shortcut.

## Environment and verification evidence

- All pinned source revisions match `benchmarks/external_models/models.json`.
- Required checkpoint sentinels exist and checksum-gated artifacts pass.
- Model-local imports pass for SigmaDock 2.13.0+cu126, SurfDock 2.2.2+cu121,
  DiffBindFR 1.13.1+cu117, and Interformer 2.4.0+cu118.
- PoseBench core/fork environments and the corrected ADFR/MolKit runtime pass
  `benchmarks/external_models/verify_installations.sh`.
- Expensive runs were submitted through the run ledger. CPU-only inference and
  evaluation were submitted only to `cpu_only` without an explicit CPU count.
- Final focused evaluator/recovery tests: `22 passed`; `scripts/check.sh fast`
  and `git diff --check` pass.
