# EFF-Dock S50 raw+refined pose-confidence checkpoint

**Feature provenance:** this unchanged checkpoint was trained on docking features from `legacy_rt_w`. Scoring corrected `rw` features is a cross-operator frozen-weight evaluation, reported separately; it is not a retrained confidence model. See [orientation compatibility](../docs/ORIENTATION_INJECTION.md).

- File: `effdock_confidence_s50_raw_refined_u70k.pt`
- SHA-256: `ce59be42f0ca613871ca079127c3296f5ca9a4ec72e44a9e5cf61878351c2638`
- Model type: `docking_graph_pose_confidence`
- Training update: 70,000 of the registered 100,000-update run
- Selection metric: validation Top-1 symmetry-aware RMSD `<2A`
- Paired docking checkpoint: `effdock_docking_early_time_t0p10_50k.pt`
- Default sampler: N100/S10/sigma-2/late-power-3/pocket-10A
- Default selector: stable minimum predicted pose RMSD

## Intended use

Rank multiple poses generated for the same receptor, ligand, and explicit
pocket. The promoted contract uses 100 poses, 10 ODE steps, translation sigma
2.0, and a 10A protein crop. The checkpoint predicts pose RMSD and success plus
per-atom displacement and success heads. These are within-complex ranking
signals; they are not calibrated across targets and do not predict binding
affinity.

The confidence model consumes t=1 ligand hidden representations from the
paired docking model. The default files, sampling preset, and pure-confidence
selector form one versioned stack. Using a different generator, sigma, pose
count, ODE budget, or protein crop is a distribution shift and must be reported
explicitly.

## Training and internal selection

The run used 43,092 training samples and initialized from the terminal U50k
symmetry-confidence state, with
one balanced per-complex mixture of 32 raw sigma-2 poses, 32 deterministically
refined poses, and one mapped crystal anchor. Pose-level training and selection
labels use RDKit `CalcRMS` symmetry-aware no-alignment heavy-atom RMSD.

The checkpoint was selected only on the fixed 1,035-complex PLINDER validation
bank. U70k reached 622/1,035 (60.10%) Top-1 `<2A`, the best registered value in
the 100k run. U100k reached 617/1,035 (59.61%) and remains the terminal training
state, not the deployment checkpoint.

## External characterization

The manuscript evaluation uses three fixed-weight inference repeats, explicit
`rw` angular features, `contextual_v1` confidence frames, and ligand reference
conformers reconstructed from the generation preparation. Sampling is unguided
N100/S10/sigma-2, followed by explicit energy refinement and input-chirality-filtered
predicted-RMSD selection, including the all-ineligible fallback. These settings
must be requested explicitly; compatibility defaults do not reproduce this evaluation.

Values are percent mean ± sample SD. RMSD is symmetry-aware heavy-atom RMSD
without receptor alignment. RMSD–PoseBusters success requires the same selected
pose to have RMSD <2 Å and pass all 27 non-RMSD PoseBusters checks. An unavailable
required check is conservatively treated as nonpassing.

| Dataset | N per repeat | RMSD <2 Å | PB-valid poses | RMSD–PoseBusters success |
|---|---:|---:|---:|---:|
| Astex Diverse Set | 85 | 83.14 ± 0.68 | 95.69 ± 0.68 | 80.00 ± 0.00 |
| PoseBusters v2 | 308 | 81.49 ± 0.56 | 95.24 ± 0.19 | 79.11 ± 0.19 |
| PhiBench-derived | 206 | 63.11 ± 3.36 | 93.69 ± 0.00 | 60.52 ± 3.23 |
| FoldBench-Pocket | 558 | 75.27 ± 0.54 | 96.54 ± 0.10 | 73.78 ± 0.63 |
| OpenBind | 925 | 53.19 ± 0.56 | 99.68 ± 0.19 | 53.19 ± 0.56 |

These are supplied-pocket redocking cohorts. PhiBench-derived is not asserted
to match the published PhiBench cohort or PAL-RMSD endpoint. The FoldBench row
uses fixed-receptor RMSD rather than the native cofolding endpoint. OpenBind
contains all 925 local cases; the separate benchmark-specific analysis uses the
official 802-case follow-on subset. Those endpoint analyses and their protocol
limitations are in the [Supplementary Information](../papers/SI.tex).

Raw/refined and chirality-selection results are reported separately in
Supplementary Figure S9. U70k was selected on internal validation; external
results are descriptive. Refinement and chirality filtering are explicit
postprocessing stages, rather than implicit operations in the public `dock()` API.

## Limitations

- Requires an explicit pocket and compatible EFF-Dock hidden features.
- Performance can shift with receptor preparation, ligand protonation or
  stereochemistry, pose count, sigma, ODE budget, crop, or generator weights.
- Reference structures are used only for evaluation labels, not ranking.
- The external studies are pocket-redocking evaluations, not blind pocket
  discovery or prospective screening.
- The model does not predict affinity or binder/non-binder status.

Exact [training membership](../benchmarks/inputs/training_membership/README.md),
[method equations](../docs/methods/05_confidence_model_and_loss.md),
[training settings](../docs/methods/06_training_and_checkpoint_selection.md), and
[current manuscript results](../papers/main.tex) are public. The
43,092 confidence training IDs and 47,277 docking IDs are independently filtered
sets sharing 43,067 IDs, not a nested pair.

This checkpoint is released together with the paired docking checkpoint under
Apache-2.0. See `DOCKING_MODEL_CARD.md` and `MANIFEST.md` for the complete
deployment identity.
