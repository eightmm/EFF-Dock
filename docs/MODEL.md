# Model contract

EFF-Dock is a fragment-level SE(3)-equivariant flow-matching docking model. A
single heterogeneous graph contains ligand atoms, ligand fragments, protein
atoms, and residue virtual nodes. Edge-typed equivariant message passing
predicts atom forces, which Newton-Euler aggregation maps to per-fragment
translation velocity in R3 and observable angular velocity in SO(3).

The compatibility architecture is configured in `configs/train.yaml` and
implemented under `src/effdock/models/`. Inputs are variable-size tensor
mappings produced by `effdock_collate`; outputs include `v_pred`, `omega_pred`,
and the observable-rotation projection.

Invariants:

- coordinates and translations use Angstroms; angular velocity uses radians;
- proper rotations use the active quaternion convention in
  `effdock.geometry.se3`;
- single-atom and rank-deficient fragments do not receive supervision on
  unobservable rotation axes;
- masked means are scaled across DDP ranks to match a global unequal-count
  mean;
- AdamW is the default. Optional Muon owns only non-degenerate
  `nn.Linear.weight` matrices; all other parameters remain in AdamW;
- AMP is off by default because the current cuEquivariance fused path expects
  FP32. TF32 is enabled by the training entry point.

Checkpoint loading is CPU-first with `weights_only=True`. Learned key/shape
mismatches fail. Only deterministic cuEquivariance runtime graph buffers may
differ between CPU fallback and CUDA construction.

```bash
uv run pytest -q tests/test_equivariance.py tests/test_losses.py
uv run pytest -q tests/test_checkpoint.py tests/test_optimizer.py
```

Architecture, representation, preprocessing, or output-head changes require a
new model compatibility version and cannot silently reuse current weights.

## Pose confidence

`src/effdock/confidence/` contains the active docking-graph pose-confidence
model, PLINDER pose-shard dataset, multitask ranking losses, safe checkpoint
runtime, and the frozen historical composite selector. The scorer reuses the docking
model's t=1 ligand hidden irreps and adds protein-ligand contact message passing,
pose RMSD/success heads, and per-atom displacement/success heads.

The selected checkpoint is U70k of the S50 raw+refined confidence run. It is
paired with the early-time/t=0-replay docking checkpoint and the
N100/S10/sigma-2/pocket-10A deployment distribution. Pose-shard preparation
and training are available through:

```bash
uv run eff-dock confidence prepare \
  --checkpoint weights/effdock_docking_early_time_t0p10_50k.pt --split train
uv run eff-dock confidence train \
  --config configs/train_confidence_s50_raw_refined_100k.yaml
```

The run warm-started the terminal symmetry-confidence state and trained on a
balanced per-complex mixture of 32 raw sigma-2 poses, 32 deterministically
refined poses, and one crystal anchor. Pose-level targets use symmetry-aware
no-alignment heavy-atom RMSD. The fixed 1,035-complex PLINDER validation bank
selected U70k at 622/1,035 (60.10%) Top-1 `<2A`; U100k reached 617/1,035
(59.61%) and remains the terminal state rather than the default.

Existing pose shards are skipped unless `--overwrite` is explicitly given;
each success, skip, and failure is appended to a JSONL manifest. The previous
N80/S25/sigma-0.5 extmatch confidence checkpoint remains a named compatibility
artifact but is not the public default.

The trained outputs are used for within-candidate-set ranking. They are not
declared calibrated estimates across datasets or sampling distributions.
Deployment `auto` uses the pure predicted-RMSD head. The cluster-free filter
study did not meet its validation gate; its filters remain explicit diagnostics,
while the cluster-based composite remains compatibility-only.

The checkpoint identity, training composition, external characterization, and
limitations are recorded in `weights/CONFIDENCE_MODEL_CARD.md` and
`docs/BENCHMARK_RESULTS.md`.
