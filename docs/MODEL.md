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

The selected checkpoint is step 42500 of the extmatch N80/S25/sigma0.5/pocket10
run. Hard-pair fine-tunes were retained only in historical outputs because they
did not improve the frozen validation metric. Pose-shard preparation and
training are available through:

```bash
uv run eff-dock confidence prepare \
  --checkpoint weights/effdock_geometry_ft_100k_best.pt --split train
uv run eff-dock confidence train --config configs/train_confidence.yaml
```

Preparation defaults to the matched N80/S25/sigma0.5/pocket10 sampling
distribution. Existing shards are skipped unless `--overwrite` is explicitly
given; each success, skip, and failure is appended to a JSONL manifest.

On the completed active three-dataset diagnostic, pure predicted-RMSD ranking
was slightly stronger overall than the frozen composite (488 versus 486 <2A
successes among 678 complexes). The composite is retained for exact historical
reproducibility and remains an explicit selector, but is not evidence of a
universally better ranking policy.

The trained outputs are used for within-candidate-set ranking. They are not
declared calibrated estimates across datasets or sampling distributions.
Deployment `auto` uses the pure predicted-RMSD head. The cluster-free filter
study did not meet its validation gate; its filters remain explicit diagnostics,
while the cluster-based composite remains compatibility-only.
