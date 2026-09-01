# Paper evidence map

This document maps the released EFF-Dock implementation to claims that can be
supported in the paper. It is an authoring guide, not a substitute for the
frozen protocol and result documents linked below.

## Released system

The paper's primary system is the paired checkpoint stack:

- docking: `effdock_docking_early_time_t0p10_50k.pt`;
- confidence: `effdock_confidence_s50_raw_refined_u70k.pt`;
- inference: N100/S10, sigma 2.0, 10-Angstrom pocket crop, late-power-3 grid;
- selection: minimum predicted pose RMSD;
- optional reported post-processing: deterministic refinement, always labeled
  separately from raw sampling.

No FK-SDE or differentiable interaction guidance is part of this released
system.

## Claim-to-evidence map

| Paper topic | Supported statement | Primary evidence | Required caveat |
|---|---|---|---|
| Method | Fragment-level SE(3)-equivariant flow matching predicts ligand poses conditioned on an explicit pocket. | [`MODEL.md`](MODEL.md), source under `src/effdock/` | Not blind pocket discovery or affinity prediction. |
| Time sampling | The registered 50k early-time/t=0 replay run improved internal PLINDER validation success from 192/1,076 to 219/1,076. | [`EARLY_TIME_FINE_TUNE_50K_PROTOCOL.md`](EARLY_TIME_FINE_TUNE_50K_PROTOCOL.md) | The earlier 2k screen was negative; the 50k run was a registered follow-up. |
| Candidate density | On opened Astex+PoseBusters candidate banks, the t0p10 model added 1,000 sub-2-Angstrom poses across 393 complexes. | [`EARLY_TIME_T0P10_50K_EXTERNAL_PAIRED_RESULTS.md`](EARLY_TIME_T0P10_50K_EXTERNAL_PAIRED_RESULTS.md) | Any-hit coverage decreased by five complexes; claim density, not universal coverage. |
| Confidence training | U70k was trained on raw sigma-2 poses, deterministic refined poses, and mapped crystal anchors with symmetry-aware RMSD labels. | [`S50_RAW_REFINED_CONFIDENCE_100K_PROTOCOL.md`](S50_RAW_REFINED_CONFIDENCE_100K_PROTOCOL.md), [`weights/CONFIDENCE_MODEL_CARD.md`](../weights/CONFIDENCE_MODEL_CARD.md) | Predictions are within-complex ranking signals, not calibrated RMSD or affinity. |
| Model selection | U70k was selected on the fixed PLINDER validation bank at 622/1,035 Top-1 below 2 Angstrom. | [`weights/CONFIDENCE_MODEL_CARD.md`](../weights/CONFIDENCE_MODEL_CARD.md) | External benchmarks did not select the checkpoint. |
| Main redocking result | Released raw/refined/joint results are the five rows in `BENCHMARK_RESULTS.md`. | [`BENCHMARK_RESULTS.md`](BENCHMARK_RESULTS.md) | Supplied-pocket redocking; opened external cohorts are descriptive. |
| Refinement | Deterministic refinement improves RMSD on the reported aggregate rows while retaining high PoseBusters validity. | [`S50_RAW_REFINED_CONFIDENCE_EXTERNAL_RESULTS.md`](S50_RAW_REFINED_CONFIDENCE_EXTERNAL_RESULTS.md), [`S50_RAW_REFINED_CONFIDENCE_TEMPORAL_EXTERNAL_RESULTS.md`](S50_RAW_REFINED_CONFIDENCE_TEMPORAL_EXTERNAL_RESULTS.md) | Refinement is a separate evaluated stage and is not silently run by `dock()`. |
| External methods | Comparisons are restricted to methods run or reported under an explicit supplied-pocket boundary. | [`EXTERNAL_MODEL_OFFICIAL_INFERENCE_PROTOCOL.md`](EXTERNAL_MODEL_OFFICIAL_INFERENCE_PROTOCOL.md), [`EXTERNAL_MODEL_EXECUTED_RMSD_RESULTS.md`](EXTERNAL_MODEL_EXECUTED_RMSD_RESULTS.md) | Keep paper-reported and locally rerun values visually and textually distinct. |

## Recommended paper organization

1. **Method:** fragment construction, equivariant network, flow objective,
   sampling, and confidence ranking.
2. **Training:** PLINDER split, time distribution, raw/refined confidence bank,
   symmetry-aware labels, and checkpoint selection.
3. **Primary evaluation:** Astex and PoseBusters supplied-pocket redocking.
4. **Temporal evaluation:** PhiBench and FoldBench; report OpenBind separately
   as an auxiliary dense single-protease cohort.
5. **Ablations:** time sampling, inference-step budget, refinement, and
   confidence training composition.
6. **Limitations:** explicit pockets, repeated-use benchmarks, dependence on
   receptor/ligand preparation, CUDA environment, and lack of affinity claims.

FK-SDE and experimental guidance belong in supplementary negative/diagnostic
analysis if used at all. They should not be mixed into the main-method diagram
or released default description.

## Numbers to keep synchronized

The canonical public table is [`BENCHMARK_RESULTS.md`](BENCHMARK_RESULTS.md).
The README and confidence model card must reproduce those values exactly; the
docking model card separately tracks the sampler-training evidence. Machine
readable U70k aggregates are under `docs/results/external_models/`. Any later
correction should update all claim-bearing surfaces in the same commit and
retain exact counts, not percentages alone.

## Claim guardrails

- Say **supplied-pocket redocking**, not blind docking.
- Say **symmetry-aware heavy-atom RMSD without alignment**.
- Distinguish `Raw`, `Refined`, `PB-valid`, and `Joint`.
- Do not state that U70k wins every benchmark; it slightly trails U50k on
  PhiBench while improving FoldBench and auxiliary OpenBind.
- Do not use external outcomes to justify checkpoint selection.
- Do not aggregate OpenBind with target-diverse cohorts without showing its
  disproportionate size and single-protease composition.
- Do not describe confidence output as binding affinity or calibrated RMSD.
