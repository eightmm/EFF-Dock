# EFF-Dock documentation

This directory contains the public method contract, result evidence, and
reproducibility records. Start with the short documents below; detailed
protocols are retained as supporting evidence rather than as the primary user
interface.

## Start here

- [`MODEL.md`](MODEL.md): architecture, objectives, sampler, and confidence model.
- [`DATA.md`](DATA.md): PLINDER processing, split, and leakage controls.
- [`EVALUATION.md`](EVALUATION.md): RMSD, validity, ranking, and refinement definitions.
- [`BENCHMARK_RESULTS.md`](BENCHMARK_RESULTS.md): released U70k results and external comparisons.
- [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md): environment and run identities.
- [`STRUCTURE.md`](STRUCTURE.md): repository ownership boundaries.

## Released-model evidence

The released model path is documented by:

1. [`EARLY_TIME_FINE_TUNE_50K_PROTOCOL.md`](EARLY_TIME_FINE_TUNE_50K_PROTOCOL.md)
   for the docking-model intervention and internal endpoint;
2. [`EARLY_TIME_T0P10_50K_EXTERNAL_PAIRED_RESULTS.md`](EARLY_TIME_T0P10_50K_EXTERNAL_PAIRED_RESULTS.md)
   for candidate-density characterization;
3. [`S50_RAW_REFINED_CONFIDENCE_100K_PROTOCOL.md`](S50_RAW_REFINED_CONFIDENCE_100K_PROTOCOL.md)
   for raw/refined/crystal-anchor confidence training;
4. [`S50_RAW_REFINED_CONFIDENCE_EXTERNAL_RESULTS.md`](S50_RAW_REFINED_CONFIDENCE_EXTERNAL_RESULTS.md)
   for Astex and PoseBusters results;
5. [`S50_RAW_REFINED_CONFIDENCE_TEMPORAL_EXTERNAL_RESULTS.md`](S50_RAW_REFINED_CONFIDENCE_TEMPORAL_EXTERNAL_RESULTS.md)
   for PhiBench, FoldBench, and auxiliary OpenBind results.

## Ablations and supplementary evidence

- Deferred post-benchmark ablation sequence:
  [`EFFDOCK_ABLATION_PLAN.md`](EFFDOCK_ABLATION_PLAN.md).
- Time sampling and exact-zero dose:
  `EARLY_TIME_FINE_TUNE_RESULTS.md`, `EARLY_TIME_T0_DOSE_10K_RESULTS.md`.
- Fixed inference budget:
  `FIXED_NFE_STEP_POSE_PROTOCOL.md` and `results/fixed_nfe_u50/`.
- Fragment initialization:
  `RDKIT_FRAGMENT_GEOMETRY_AUDIT_V2_PROTOCOL.md` and
  `FRAGMENT_TEMPLATE_SWAP_HEADROOM_RESULTS.md`.
- Confidence objectives and selectors:
  `CONFIDENCE_SELECTION_STUDY.md`, `CONFIDENCE_FILTER_STUDY.md`, and the
  S50 confidence protocols.
- Refinement and validity:
  `GUIDANCE_SDF_POST_REFINEMENT_RESULTS.md` and
  `S50_REFINEMENT_BUDGET_CALIBRATION_PROTOCOL.md`.
- External methods:
  `../benchmarks/external_models/docs/EXTERNAL_MODEL_OFFICIAL_INFERENCE_PROTOCOL.md`
  and `../benchmarks/external_models/docs/EXTERNAL_MODEL_EXECUTED_RMSD_RESULTS.md`.

## Diagnostic-only studies

The FK-SDE and differentiable-guidance studies are retained for negative or
diagnostic evidence. They are not part of the released inference stack and
should not be presented as production components:

- `FK_SDE_*`;
- `GUIDANCE_*`;
- `INTERACTION_PRIOR_PROBE_*`;
- `PLINDER_GUIDANCE_VALIDATION_PROTOCOL.md`.

## Record policy

Protocol and result documents are immutable scientific records once their
outcomes are opened. Corrections should be explicit and append-only. Private
machine paths, raw scheduler logs, pose banks, source datasets, and the
machine-generated run ledger stay outside Git. The curated narrative index is
[`EXPERIMENTS.md`](EXPERIMENTS.md).

Some frozen historical protocols reference superseded checkpoints that are no
longer distributed. Those references are retained as provenance, not as
supported public runtime dependencies. The only released and supported model
pair is listed in `weights/MANIFEST.md`.
