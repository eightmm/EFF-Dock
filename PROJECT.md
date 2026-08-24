# PROJECT.md

## Status

- State: confirmed
- Name: EFF-Dock
- Type: ml
- Last updated: 2026-08-24
- Gate: confirmed for non-destructive migration and EFF-Dock bootstrap.

## Project

- Goal: provide one clean, reproducible protein-ligand docking project that
  supports data preparation, training, evaluation, checkpoint resume, and
  end-to-end inference.
- Users / workflow: single-researcher local and multi-GPU workflows; `uv` for
  environments; PyTorch DDP for training; a unified `eff-dock` CLI for active
  user paths.
- Initial model baseline: preserve the existing fragment-based SE(3)-equivariant
  flow-matching architecture closely enough to train from scratch and load the
  retained FlowFrag docking weights. Architecture changes happen only after
  the compatibility baseline is verified.
- Scope:
  - PLINDER curation, preprocessing, split generation, and manifest checks.
  - Fragment-level SE(3) flow-matching training and exact checkpoint resume.
  - External docking evaluation on PoseBusters v2 and Astex Diverse.
  - Pocket-conditioned inference from a protein structure, ligand, and an
    explicit pocket definition.
  - Portable release weights and a minimal legacy-weight compatibility path.
- Non-goals for the initial baseline:
  - New adaptive selector sweeps, local-resample experiments, or additional
    confidence fine-tuning driven by the completed external benchmark.
  - Blind binding-site discovery, affinity prediction, or virtual-screening
    classification.
  - Covalent docking or induced-fit receptor modeling.
  - Reproducing every historical FlowFrag experiment from the active tree.

The confidence-selection study below is an explicitly approved post-baseline
experiment and does not replace the retained step-42500 compatibility stack
unless its validation and external gates pass.

### Guidance Contract

- Status: diagnostic implementation active. The pure-Torch energy kernel,
  force/fragment projection, crystal perturbation trace, and saved-trajectory
  trace are implemented. They are not yet admitted to the production sampler;
  operator-split correction and an internal held-out validation ablation
  (currently PLINDER validation) remain gates.
- Goal: build one inference-time
  `GuidanceEnergy = PhysicalEnergy + InteractionEnergy` corrector for the
  fragment SE(3) ODE while keeping docking and confidence checkpoints frozen.
- Runtime boundary: energy, force, fragment projection, scheduling, and
  correction must be evaluated by code and versioned parameter tables inside
  this repository. The active path may use PyTorch tensor/autograd operations
  and existing molecular parsing, but it may not call or import an external
  force-field, docking, minimization, or molecular-simulation engine.
- External references: published equations, constants, atom/residue parameters,
  and offline reference outputs may be copied into versioned in-repository
  tables or golden fixtures when their provenance and redistribution terms are
  recorded. External software may be used offline only to produce independent
  validation values; it is never a runtime or required-test dependency.
- Scientific boundary: `PhysicalEnergy` contains declared generic geometry and
  force-field-like terms; `InteractionEnergy` contains typed directional
  motifs. Both are components of one diagnostic `GuidanceEnergy`. Hydrophobic
  contact, idealized missing-valence-cone heavy-atom hydrogen bond, and
  screened formal-charge-group terms are active diagnostics; metal
  coordination remains contract-only. Vina is explicitly excluded from this
  guidance objective. Its legacy code and historical results are retained but
  are not imported, weighted, or combined here.
- Claim boundary: this is force-field energy-gradient guidance on a
  dimensionless generative ODE, not molecular dynamics, binding free-energy
  calculation, or affinity prediction.
- Missing parameterization fails explicitly or uses a separately named and
  reported `geometry-only` mode; no atom type, charge, or energy term is
  silently zero-filled or delegated to an external engine.
- Receptor admission follows normalized residue chemistry, not PDB
  `ATOM`/`HETATM` formatting. Active-shell cofactors, ions, and other
  unsupported nonprotein residues fail with structured provenance.
- Full term, solver, provenance, naming, and evaluation contract:
  `docs/GUIDANCE_CONTRACT.md`.
- Frozen pre-formal-charge V3 diagnostic results and exact input/parameter
  hashes: `docs/GUIDANCE_DIAGNOSTIC_RESULTS.json`. The screened-charge V4
  external fixed-coordinate characterization is recorded in
  `docs/GUIDANCE_FORMAL_CHARGE_BENCHMARK_CHARACTERIZATION_V4.json`.
- Interaction terms are admitted one at a time. The screened formal-charge
  study freezes its typing, constants, numerical gates, and validation order in
  `docs/INTERACTION_GUIDANCE_STUDY.md`; physically valid terms may remain as
  traced diagnostics even when they do not pass the separate sampler-activation
  gate.
- `docs/GUIDANCE_FORMAL_CHARGE_COVERAGE_V4.json` preserves the earlier
  PLINDER net-charge proxy inventory as historical provenance. Net charge is
  not the eligibility rule because it misses zwitterions. Stage 1B now uses
  any nonzero ligand formal-charge site in the frozen Astex/PoseBusters raw
  structures and is strictly report-only; external outcomes cannot select a
  formula, coefficient, schedule, term, or sampler setting.

### Archived Vina-guided Sampling Experiment (inactive)

This protocol is retained only as historical evidence. It is not part of the
current guidance target, is not a baseline component, and must not supply
equations, coefficients, typing, gradients, or selection values to
`GuidanceEnergy`.

- Protocol ID: `EFFDOCK-VINA-GUIDANCE-V1`.
- Intervention: inference-only, late-time negative gradient of the official
  AutoDock Vina 1.2 scoring function plus the existing ligand DG strain term,
  aggregated into fragment translation and rotation velocities. Training data,
  docking weights, confidence weights, candidate count, ODE steps, seeds, and
  final selector remain frozen.
- Information boundary: the guidance may use only receptor coordinates/types,
  ligand chemistry/starting geometry, the declared pocket center, and current
  sampled coordinates. It may not use the crystal pose, RMSD, benchmark ID, or
  PoseBusters outcome.
- Tuning policy: numerical stability and the guidance scale are checked on
  smoke/PLINDER validation inputs only. PoseBusters is evaluation-only and is
  not used to tune the scale, start time, force cap, or strain weight.
- Primary hypothesis: selected-pose PoseBusters pass-all improves by at least
  3 percentage points versus the unguided frozen baseline while selected
  RMSD<2A decreases by no more than 2 percentage points.
- Full protocol and provenance fields: `docs/VINA_GUIDANCE_PROTOCOL.md`.

### Confidence Selection Study

- Protocol ID: `EFFDOCK-CONFIDENCE-SELECTION-V1`.
- Goal: improve top-1 pose selection by training only on matched PLINDER
  N80/S25/sigma0.5/pocket10 pose sets. The docking generator and frozen
  external benchmark manifests remain unchanged.
- Model selection boundary: loss variants and checkpoints are selected using
  PLINDER validation only. PoseBusters and Astex are not used for loss,
  hyperparameter, checkpoint, or selector tuning and are opened only after a
  candidate is frozen.
- Primary screen metric: validation success-head selected RMSD <2A. The frozen
  composite selector and selected median RMSD are secondary guard metrics.
- Full hypotheses, variants, thresholds, and provenance are in
  `docs/CONFIDENCE_SELECTION_STUDY.md`.

### Cluster-free Confidence Filter Study

- Protocol ID: `EFFDOCK-CONFIDENCE-FILTER-V1`.
- The deployable default with a confidence checkpoint is pure predicted-RMSD
  ranking. It is independent of cluster size, candidate ranks, and pairwise
  pose distances.
- A conservative strict filter and an atom-displacement guard were fitted on
  PLINDER train and confirmed once on full PLINDER validation. Neither met the
  pre-registered +1.0-point validation gate, so neither replaces pure
  confidence. They remain explicit experimental/diagnostic selectors.
- The historical `pair_gate_density_rank_vote_plclash_ambig` selector remains
  explicit for exact benchmark reproduction only.
- Protocol and outcomes: `docs/CONFIDENCE_FILTER_STUDY.md` and
  `docs/CONFIDENCE_FILTER_RESULTS.json`.

## Repository Layout

- Active package: `src/effdock/`
- Canonical configs: `configs/`
- Canonical commands: `eff-dock data`, `eff-dock train`,
  `eff-dock confidence`, `eff-dock evaluate`, and `eff-dock dock`.
- Active tests: `tests/`
- Retained release artifacts: `weights/`
- Historical code/config/docs: `archive/flowfrag_legacy/`
- Preserved local data: `data/` (never moved or deleted during migration).
- Preserved historical runs: `outputs/` (not part of the active interface).

### Migration Policy

- Phase 1 is non-destructive: move non-canonical source, configs, scripts, and
  docs into `archive/flowfrag_legacy/`; do not delete them.
- Keep `data/`, `outputs/`, `archive/`, checkpoints, caches, and generated
  benchmark artifacts ignored by Git.
- Do not rewrite or reprocess existing data during repository cleanup.
- Preserve the following docking artifacts as named, hashed legacy inputs:
  - released 200k EMA weight;
  - source 200k resume checkpoint;
  - small-sigma fine-tuned best checkpoint.
- Preserve the selected extmatch confidence step-42500 checkpoint and its paired
  geometry-FT docking checkpoint as named, hashed active artifacts. Other
  confidence experiments remain preserved under ignored `outputs/`.
- No file is deleted until the EFF-Dock compatibility load and checksum checks
  pass and deletion is separately approved.

## Interface and Data

### Public CLI

```text
eff-dock data prepare [options]
eff-dock data split [options]
eff-dock train --config configs/train.yaml [--resume CHECKPOINT]
eff-dock confidence prepare --checkpoint DOCKING_CHECKPOINT [options]
eff-dock confidence train --config configs/train_confidence.yaml [--resume CHECKPOINT]
eff-dock evaluate --dataset DATASET --data-dir DIR \
  --pocket-centers CENTERS.json --checkpoint CHECKPOINT
eff-dock dock --protein P.pdb --ligand L.sdf --pocket-center X,Y,Z \
  --config configs/train.yaml --checkpoint WEIGHT --output-dir outputs/docked
```

- `--pocket-center` or another explicit pocket definition is required for the
  public docking boundary. Crystal target coordinates must not be used to infer
  the pocket implicitly.
- The initial CLI has one canonical path per operation. Historical shell
  launchers and selector-specific entry points remain in the archive only.

### Prediction Contract

- Prediction unit: one protein-structure / ligand-chemical-entity / pocket
  context complex.
- Immutable sample ID: PLINDER `system_id` plus `ligand_instance_chain`, stored
  as `sample_key = <system_id>__<ligand_instance_chain>`.
- Inputs available at inference:
  - receptor coordinates and atom/residue identity;
  - ligand graph, stereochemistry, protonation state, and generated or supplied
    starting conformer;
  - an explicit pocket center or pocket residue definition.
- Target: experimental bound ligand fragment poses in Angstrom and proper
  rotations in SO(3).
- Outputs: sampled ligand poses plus declared selection scores and provenance.
- Forbidden inference features: target/crystal ligand coordinates, labels,
  test-set membership, or features derived from downstream benchmark results.

### Data and Provenance

- Primary source: PLINDER 2024-06 v2.
- Existing curated pool: `data/plinder_pool.parquet`.
- Existing processed samples: `data/plinder_processed/`.
- Existing split: `data/splits/plinder.json` with 47,310 train and 1,076 val
  samples; compatibility-only until the EFF-Dock split contract is finalized.
- External benchmarks: PoseBusters v2 and Astex Diverse; benchmark
  snapshots and molecule mappings must be versioned in manifests even though
  raw structures remain ignored by Git.
- Coordinate unit: Angstrom throughout preprocessing, training, inference, and
  RMSD evaluation.
- Invalid or missing structures are quarantined with a reason; scientifically
  meaningful features are never silently zero-filled.

### ML Scientific Contract

- Inference-time information boundary: receptor coordinates and identities,
  ligand chemistry/starting conformer, and an explicit pocket definition only;
  no crystal target pose, label-derived feature, or benchmark outcome.
- Entity IDs/standardization: immutable PLINDER `sample_key`; canonical RDKit
  SMILES for ligand grouping; PLINDER pocket70 community for pocket grouping.
- Source snapshot/provenance: PLINDER 2024-06 v2 plus frozen PoseBusters v2 and
  Astex Diverse snapshots; preprocessing version and hashes are
  carried by manifests/checkpoints.
- Label/target definition: experimental bound per-fragment translation and
  proper rotation, with flow velocity targets derived from the declared prior
  and interpolation.
- Label units/direction/censoring/replicates: Angstrom for coordinates and
  translation; radians/world frame for angular velocity; no censoring;
  alternate structures/poses of one entity stay in one split group.
- Split policy: canonical-SMILES-disjoint and pocket70-disjoint train/val;
  strict external-benchmark canonical-SMILES exclusion from train.
- Split/group keys: `sample_key`, canonical SMILES, and
  `pocket_fident__70__community`.
- Leakage risks: repeated ligand chemistry, homologous pockets, alternate
  structures, bound-pose-derived pocket hints, and adaptive external-test use.
- Train-only fitted transforms: any learned normalization, threshold,
  calibration, selector, or retrieval/index state; the baseline featurizer is
  deterministic and label-blind.
- Data manifest: `eff-dock-plinder`, to be created and checked before training.
- Calibration/applicability-domain plan: the retained confidence score is used
  for within-pose-set ranking, not claimed as calibrated RMSD or success
  probability; report ligand- and pocket-neighbor similarity and coverage.

### Split and Leakage Contract

- Split on canonical ligand SMILES and protein pocket cluster rather than rows.
- Train and validation must be disjoint on sample ID, canonical SMILES, and
  pocket70 community.
- Alternate structures, poses, and derived views of one complex remain on the
  same side.
- External benchmark complexes are frozen before training. Canonical ligand
  SMILES from PoseBusters v2 and Astex Diverse are strictly excluded
  from train. Pocket similarity is retained as an evaluation slice rather than
  used as an additional destructive exclusion rule.
- Every registered split gets an OMS data manifest; `check` and `leakage` must
  pass before training.
- Learned preprocessing, thresholds, calibration, and model selection use
  training/validation data only. External benchmarks are opened only for a
  frozen model and selector.

## Model and Training Contract

- Baseline state: per-fragment translation in R^3 and rotation in SO(3).
- Baseline model: one SE(3)-equivariant protein-ligand graph network with
  Newton-Euler aggregation from ligand-atom forces to fragment velocity.
- Loss units are explicit:
  - translation and observable angular losses reduce over valid fragments;
  - atom auxiliary loss reduces over valid ligand atoms;
  - distance-geometry loss reduces over declared valid intra-ligand pairs.
- Variable-size graphs use visible masks/counts; DDP aggregation must match a
  single-process global masked mean for unequal valid counts.
- Default optimizer baseline: AdamW with semantically defined parameter groups.
- Muon is opt-in and requires a matched AdamW ablation; embeddings,
  normalization, bias, and scalar parameters stay in AdamW.
- Batch size is declared per rank and effective global batch size is logged.
- AMP/autocast remains disabled until cuEquivariance kernel compatibility is
  demonstrated. TF32 may be enabled and must be recorded.
- DDP uses conservative defaults, `DistributedSampler.set_epoch`, rank-0
  logging/checkpoint writes, and explicit process-group cleanup.

### Checkpoint Contract

- Release checkpoint: portable tensor-only unwrapped model weights, model/data/
  featurizer/split/config IDs, code commit, and inference defaults.
- Resume checkpoint: model, every optimizer/scheduler, scaler when applicable,
  global step/epoch, best metric, RNG state, and sampler state required for the
  declared resume guarantee.
- Loading starts on CPU with `weights_only=True`; shape or key mismatches fail
  explicitly unless a named migration is applied.
- Checkpoint structure must not depend on whether CUDA cuEquivariance kernels or
  CPU fallback kernels were available when the model was instantiated.
- The selected confidence checkpoint and paired docking checkpoint are a frozen
  compatibility stack. Its historical training-matched preset is
  N80/S25/sigma0.5 with a 10A pocket crop. The public deployment default is
  N100/S10/sigma2 with the same crop; this candidate-distribution shift must
  remain explicit in user-facing documentation and result provenance.

## Evaluation

- Primary metric: PoseBusters v2 selected top-1 fraction with symmetry-aware
  ligand RMSD < 2 Angstrom, using the frozen trained-confidence selector.
- Secondary metrics:
  - Astex Diverse selected top-1 < 2 Angstrom;
  - PoseBusters chemical/structural validity;
  - oracle top-k success to separate sampling from selection quality.
- Baselines:
  - retained FlowFrag 200k EMA checkpoint;
  - retained small-sigma fine-tuned checkpoint;
  - Vina/Vina+strain selection where available;
  - prior/no-learned-velocity rollout sanity baseline.
  - first-pose and Vina+DG selection on the identical confidence candidate set.
- Required slices: ligand heavy atoms, fragment count, rotatable bonds,
  cofactor class, ligand similarity to train, and pocket similarity to train.
- Applicability domain: report train-neighbor ligand and pocket similarity with
  coverage; do not claim the retained confidence outputs are calibrated.

## Commands

- Setup: `uv sync --group dev`
- Unit tests: `uv run pytest -q`
- Lint: `uv run ruff check src tests`
- CPU smoke: `uv run pytest -q tests/smoke/test_cpu_step.py`
- GPU smoke: `uv run pytest -q -m gpu`
- DDP smoke: `uv run torchrun --standalone --nproc-per-node=2 -m effdock.train.smoke`
- Train: `uv run eff-dock train --config configs/train.yaml`
- Confidence train: `uv run eff-dock confidence train --config configs/train_confidence.yaml`
- Dock: `uv run eff-dock dock --help`

## Slurm Execution Contract

- Partition/account: supplied by the target cluster environment; bootstrap and
  CPU checks do not submit jobs.
- CPU/GPU/memory/time: resources are declared per submitted run from the local
  generated cluster contract; no GPU model is assumed in shared project files.
- Logs/checkpoints: scheduler logs under ignored `outputs/slurm/`; portable and
  resume checkpoints under the run-specific ignored `outputs/` directory.

## Verification

- Success criteria: the active EFF-Dock paths satisfy all checks below while
  retained data and historical artifacts remain intact and ignored by Git.

### Success Criteria

- Active repository contains EFF-Dock docking and the selected confidence
  baseline; other historical FlowFrag/confidence experiments remain isolated
  under the ignored archive.
- All five retained docking/confidence artifacts have recorded SHA-256 hashes.
- Retained weights load through an explicit compatibility migration and produce
  a deterministic inference smoke result on the intended CUDA runtime.
- One CPU batch completes forward, loss, backward, and optimizer step.
- One production-device batch passes with finite outputs and gradients.
- Save/load and exact-resume regression checks pass.
- Two-rank DDP smoke matches the single-process masked-loss contract.
- Raw-network and public-inference SE(3) transformation tests pass.
- Data manifest drift and leakage checks pass before any long training run.
- Active tests and lint are green.

### Required Checks

- `uv lock --check`
- `uv pip check`
- `uv run ruff check src tests`
- `uv run pytest -q`
- one release-weight load/inference smoke;
- one checkpoint round-trip/next-step resume test;
- one two-rank DDP smoke on CUDA;
- `oms data-manifest check --name eff-dock-plinder`;
- `oms data-manifest leakage --name eff-dock-plinder`.

## Experiment Pre-Registration

- The physical N40 baseline is frozen in `docs/BENCHMARK_PROTOCOL.md`.
- The trained-confidence N80 study is frozen in
  `docs/CONFIDENCE_BENCHMARK_PROTOCOL.md`.
- Future experiments must declare hypothesis, baseline, metric, seed/update
  budget, and disconfirming result before compute.

### S50 Symmetry-Confidence Training (2026-08-22)

- Status: completed. The four-GPU recovery run reached U50,000 on the frozen
  43,092-complex S50/sigma-2 train bank and 1,035-complex validation bank.
  Every pose-level confidence target and selection metric used RDKit `CalcRMS`
  symmetry-aware no-alignment RMSD; fixed-map atom displacement remained only
  for the atom-level auxiliary heads.
- Internal selection: U25k achieved `605/1,035 = 58.45%` Top-1 `<2A`, versus
  `497/1,035 = 48.02%` at U0 and `588/1,035 = 56.81%` at U50k. Therefore the
  registered validation rule selects U25k `best.pt`; U50k `latest.pt` is the
  terminal state and continuation source.
- Experimental checkpoint SHA-256 values are
  `1c59034172fb925cc8a70777dcba236be349f1a1de1775d49cc17d492b17c030`
  for U25k best and
  `fd49fa86f67187bf26d6c1bcf2daf925ba3e3b19dfeae733e57535d183280469`
  for U50k latest. They remain under ignored `outputs/` and have not replaced
  the packaged step-42,500 compatibility checkpoint in `weights/`.
- Repeated-use Astex/PoseBusters characterization favored U50k descriptively,
  especially after refinement, but cannot override the internally selected
  U25k checkpoint or promote a production default.
- Frozen contract and complete result:
  `docs/S50_SYMMETRY_CONFIDENCE_TRAINING_PROTOCOL.md` and
  `docs/S50_SYMMETRY_CONFIDENCE_RESULTS.md`.

### S50 Raw + Refined Confidence Continuation (2026-08-23)

- The separately registered branch mixes 32 raw sigma-2 poses, 32
  deterministically refined poses, and one mapped-crystal anchor per complex.
  Validation uses all 100 refined poses and excludes crystal anchors.
- It initializes weights-only from immutable U50k `latest.pt`, uses a fresh
  optimizer/scheduler for 10,000 updates, and writes new `latest.pt` and
  `best.pt` artifacts under a distinct output identity. This continuation does
  not retroactively promote U50k over the internally selected U25k checkpoint.
- The current frozen budget is 10,000 updates. Any longer continuation requires
  a new registered content identity.
- Frozen contract: `docs/S50_RAW_REFINED_CONFIDENCE_FINETUNE_PROTOCOL.md`.

## Decisions

### Confirmed

- Rename and reorganize the project as EFF-Dock.
- Support the full data -> training -> evaluation -> inference lifecycle.
- Preserve existing datasets locally and keep them ignored by Git.
- Archive non-canonical legacy material instead of deleting it.
- Promote the selected extmatch confidence baseline and frozen selector into the
  active package; archive other confidence/reranking experiments.
- Preserve usable docking weights, not merely checkpoint bytes.
- Use `uv`; do not introduce pip/conda workflows.
- Strictly exclude external-benchmark canonical ligand SMILES from train while
  reporting pocket similarity as an evaluation slice.
- Preserve legacy load/inference compatibility; exact historical training
  reproduction is not a requirement for the first EFF-Dock baseline.

### Open

- None for the initial migration and bootstrap.
