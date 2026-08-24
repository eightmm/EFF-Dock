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

A separate S50 N100/S10/sigma-2 experiment warm-started these weights and
trained pose-level heads against symmetry-aware no-alignment RMSD. Its internal
validation rule selected U25k `best.pt` (58.45% Top-1 `<2A`); U50k
`latest.pt` (56.81%) is the terminal continuation state. Both remain
experimental files under ignored `outputs/` and require an explicit checkpoint
override. See `docs/S50_SYMMETRY_CONFIDENCE_RESULTS.md` for exact hashes and
the repeated-use evaluation boundary.

Preparation defaults to the matched N80/S25/sigma0.5/pocket10 sampling
distribution. Existing shards are skipped unless `--overwrite` is explicitly
given; each success, skip, and failure is appended to a JSONL manifest.

That confidence-preparation default preserves the retained checkpoint's
training provenance. It is distinct from public inference: `eff-dock dock` and
`eff-dock evaluate` default to N100/S10/sigma2 with a 10A pocket crop.

On the completed active Astex/PoseBusters diagnostic, pure predicted-RMSD
ranking produced 290 <2A successes versus 291 for the frozen composite among
393 complexes. The composite is retained for exact historical
reproducibility and remains an explicit selector, but is not evidence of a
universally better ranking policy.

The trained outputs are used for within-candidate-set ranking. They are not
declared calibrated estimates across datasets or sampling distributions.
Deployment `auto` uses the pure predicted-RMSD head. The cluster-free filter
study did not meet its validation gate; its filters remain explicit diagnostics,
while the cluster-based composite remains compatibility-only.
