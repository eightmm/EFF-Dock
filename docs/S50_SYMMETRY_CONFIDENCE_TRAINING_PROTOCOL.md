# S50 symmetry-target confidence training protocol

Protocol ID: `EFFDOCK-S50-SYMMETRY-CONFIDENCE-TRAINING-V1`

## Objective

Train the confidence model on the frozen S50 N100/S10/sigma2 pose bank while
using RDKit `CalcRMS` symmetry-aware, no-alignment RMSD for every pose-level
loss. Atom-level regression/BCE continues to use the original fixed-map
`atom_disp`; this run does not reinterpret atom labels.

## Frozen inputs

- Bank manifest SHA-256: `d45e36f3f2d75fb8ba9553715e1ec45031e4ad31881631d8632b6c19999f2d2b`
- Filtered split SHA-256: `1f23b50ef3a5eff73fd8cad683c1f3adfe4e1ab235199274d0dbc220c1d22507`
- Train symmetry manifest SHA-256: `89ea40b02b121387228e3d47461a68a30bd749143b6281fe4d5ea9ab89056981`
- Warm-start confidence checkpoint SHA-256: `e31fde6f351284205c78f7a1510002779c43312e94d9f82003d47a14d72bc78f`
- Train inventory: 43,092 complexes, 4,309,200 labels.
- Validation inventory: 1,035 complexes, 103,500 embedded symmetry labels.

The train target manifest and all 128 sidecars are verified before model
construction. Each loaded row is checked against sample key, system ID,
split index, source payload SHA, pose-ensemble SHA, label SHA, bank SHA and
frozen-input SHA. Any mismatch fails the run.

## Optimization and topology

- Initialize weights from retained confidence step 42,500; optimizer,
  scheduler and update counter start fresh at U0.
- AdamW LR `3e-5`; Muon LR `2e-3`; weight decay `0.01`.
- WSD schedule: 2% warmup, 50% cooldown, minimum LR ratio 0.05.
- 50,000 updates, four DDP ranks on one `heavy` allocation. The allocated
  device types are recorded; the exact same four-GPU topology is exercised by
  the smoke before the full run is admitted.
- One complex per rank, 64 of 100 train poses selected using symmetry labels;
  validation uses all 100 poses.
- Full validation every 5,000 updates. Publish `latest.pt` every 500 updates
  and `best.pt` by symmetry-aware Top-1 `<2A` validation success.
- Confidence step 42,500 is a weights-only warm start, not an optimizer resume.

## Launch gate

First run the exact four-GPU topology with 8 train complexes, 2 validation
complexes and one update. The smoke must publish finite metrics, `latest.pt`,
`best.pt`, and exact target provenance. Only then may the 50k job start.

The older fixed-map training job is a control. Once the symmetry smoke passes,
cancel it but retain its logs and checkpoints as provenance; do not delete its
artifacts.
