# EFF-Dock eta-2 prior-sigma sweep protocol

## Status and scope

This is a paired descriptive inference ablation. It fixes the current
repository-native unified guidance at `eta=2.0` and changes only the Gaussian
fragment-translation prior scale. External benchmark outcomes cannot admit a
production setting or tune the guidance implementation.

## Frozen conditions

- Datasets: all 85 Astex Diverse and all 308 PoseBusters v2 complexes from
  `docs/GUIDANCE_BUDGET1000_FULL_INPUTS.json`.
- New sigma arms: `sigma={1.0,2.0,3.0,4.0}`. The completed `sigma=0.5`,
  `eta=2.0` arm at
  `outputs/benchmarks/guidance_steric_high_eta_confidence_runs/20260807T045916Z`
  is the frozen reference arm.
- Sampling budget: `N100/S10`, prior pool 100, seed 42 plus frozen complex
  position, late schedule with power 3.
- Guidance: `normalized_drift`, `eta=2.0`, starts at `t=0.5`, ramp power 1,
  force cap 20, translation/angular caps 5, atom-displacement cap 0.25
  Angstrom, maximum 8 backtracks, 18 Angstrom protein shell.
- Pocket: reference center, cutoff 10 Angstrom, center jitter 0.
- Receptor policy: `geometry_only`; refinement disabled; Vina disabled.
- Docking and confidence checkpoints are those frozen by
  `docs/GUIDANCE_STERIC_HIGH_ETA_CONFIDENCE_PB_PROTOCOL.md`.

The docking model was trained with sigma values `{0.5,1.0,2.0,3.0,4.0}`.
The confidence checkpoint was trained on sigma-0.5-generated pose sets, so
selection performance at larger sigma is explicitly treated as a measured
domain-shift outcome rather than assumed calibration.

## Execution and audit

The fail-closed chain is smoke sampling, smoke audit, full sampling, full
audit, and aggregate report. Full sampling uses 64 GPU tasks:
`2 datasets x 4 new sigma arms x 8 shards`. Every task must have one supported
GPU, complete its assigned inventory without failures, emit 100 candidates per
complex, and preserve finite confidence and RMSD values. CUDA allocated memory
must remain below 48 GiB. CUDA reserved cache is recorded but is not treated as
allocated tensor memory and is not an OOM gate.

The aggregate report includes confidence Top-1, confidence-filter Top-1, raw
RMSD oracle, internal fast-valid oracle, joint `<2 Angstrom` and internal
fast-valid rate, valid-candidate density, cap telemetry, and comparison to the
frozen sigma-0.5 reference. Internal fast-valid checks are not official
PoseBusters validity.
