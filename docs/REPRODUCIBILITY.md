# Reproducibility

The environment is locked by `pyproject.toml` and `uv.lock`, including PyTorch
2.10.0 CUDA 13.0 wheels and matching PyG extension wheels.

```bash
uv sync --group dev
uv run pytest -q
uv run ruff check src tests
uv run eff-dock train --config configs/train.yaml
uv run eff-dock train --config configs/train.yaml --resume outputs/RUN/checkpoints/latest.pt
uv run eff-dock confidence prepare \
  --checkpoint weights/effdock_geometry_ft_100k_best.pt --split train
uv run eff-dock confidence train --config configs/train_confidence.yaml
```

The training entry point seeds Python, NumPy, and PyTorch. Resume checkpoints
contain model, EMA, every optimizer and scheduler, config, global step/epoch,
best metric, RNG states, metrics, and run ID. `--resume` requires the same
optimizer/scheduler layout; `--init-from` performs an explicit weights-only
migration with new optimizer state.

Record the Git commit, `uv.lock`, config, split/data manifest, checkpoint hash,
hardware, world size, effective batch size, seed, and command for every run.
CUDA scatter/atomic kernels and multi-GPU reduction order can remain
nondeterministic; do not claim bitwise reproducibility unless separately
verified.

GPU work runs through the project Slurm scripts. `confidence_prepare.sbatch`
generates immutable-by-default labeled pose shards and
`confidence_train.sbatch` consumes them. The active confidence inference path
has N2 and N80 CUDA smokes (jobs `38750`/`38751`), the confidence trainer has a
one-step CUDA smoke (`38762`), and pose preparation has an N2 CUDA smoke
(`38773`). Benchmark job IDs, hashes, and logs are retained under ignored
`outputs/` and recorded in the experiment ledger.
