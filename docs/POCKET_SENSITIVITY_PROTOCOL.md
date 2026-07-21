# Frozen current-stack pocket sensitivity protocol

Protocol ID: `EFFDOCK-POCKET-SENSITIVITY-N80-S25-V2`

Frozen: 2026-07-21, after the stable-inertia implementation and before the V2 sweep.

V2 supersedes the incomplete V1 compute. V1 exposed numerical failures in the
float32 batched CUDA eigendecomposition and was stopped before completion; its
raw outputs remain preserved but are excluded from all final tables.

## Question and information boundary

How do receptor pocket crop radius and error in the supplied pocket center
affect the retained EFF-Dock sampler and pure-confidence top-1 selector?

This is a reference-defined oracle-pocket redocking robustness diagnostic on
Astex Diverse and PoseBusters v2. The frozen center is target-derived and may
be perturbed, so this experiment does not measure blind pocket discovery or
prospective docking. Reference ligand coordinates are used only to define the
already-frozen center and calculate RMSD.

## Frozen hypothesis and prediction

- Hypothesis: 8--10A supplies sufficient receptor context at accurate or mildly
  perturbed centers, while 6A removes useful context and 2A center error lowers
  generator coverage rather than only changing confidence ranking.
- Prediction: at zero jitter, cutoff 6A is at least 10 percentage points below
  cutoff 10A in pure-confidence RMSD <2A; cutoff 8A or 10A is best. At cutoff
  10A, 1A jitter loses no more than 7 points, while 2A jitter loses at least 10
  points and lowers oracle-80 by at least 5 points.
- Disconfirming observations: cutoff 6A is competitive with 10A, 12A improves
  consistently, or 2A jitter leaves oracle coverage unchanged.
- This is characterization only. Astex and PoseBusters are not used to select
  a new default; any change from 10A requires PLINDER validation.

## Frozen execution matrix

- Datasets: frozen Astex Diverse (85) and PoseBusters v2 (308) manifests.
- Pocket cutoff: {6, 8, 10, 12}A.
- Center jitter sigma: {0, 1, 2}A.
- Docking checkpoint: `weights/effdock_geometry_ft_100k_best.pt`, step 100000.
- Confidence checkpoint:
  `weights/effdock_confidence_extmatch_n80_s25_step42500.pt`, step 42500.
- Sampling: N80, 25 ODE steps, translation sigma 0.5, late schedule power 3,
  no refinement, base seed 42 plus frozen global complex index.
- Jitter pairing: a separate per-complex CPU generator uses the complex seed;
  1A and 2A perturbations therefore share direction, and jitter does not shift
  the pose-sampling RNG stream.
- Inertia solve: symmetric fragment inertia tensors are constructed and
  decomposed in float64; single-atom fragments bypass eigendecomposition;
  non-finite tensors fail explicitly; CUDA convergence failures retry the same
  matrix on CPU float64. This changes numerical execution only, not the
  Newton--Euler definition, checkpoint, candidates, or selector.
- Primary selector: pure minimum predicted RMSD (`confidence`).
- Diagnostic selectors: first, Vina+DG, frozen historical composite, oracle-80.
- Primary metric: pure-confidence selected top-1 symmetry-aware heavy-atom
  RMSD <2A on each dataset.
- Secondary metrics: selected median RMSD, oracle-80 <2A, first/Vina/composite
  <2A, failure rate, and official PoseBusters pass-all on saved selected poses.
- Failure gate: no condition supports a conclusion if unresolved failures
  exceed 2%, hashes differ across its outputs, or fewer than all frozen IDs are
  represented after documented numerical rescue.

## Compute and artifacts

- Partition: Slurm `6000ada`, one RTX 6000 Ada per task.
- Array: 24 conditions, at most 8 simultaneous tasks; one dataset-condition per
  task to avoid an unnecessarily large 192-task shard array.
- Raw outputs: `outputs/benchmarks/pocket_sensitivity_n80_s25_v2/raw/`.
- Logs: `outputs/benchmarks/pocket_sensitivity_n80_s25_v2/logs/`.
- Final table and plot are produced only after completeness and hash checks.
