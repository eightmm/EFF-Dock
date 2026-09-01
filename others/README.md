# External docking models

Each comparison model owns an isolated uv project:

```text
others/<model>/
├── pyproject.toml   # tracked dependency contract
├── uv.lock          # tracked resolved Python lock
├── .python-version  # tracked interpreter series
├── .venv/           # ignored, model-local environment
├── .cache/uv/       # ignored, model-local download/build cache
├── upstream         # ignored pinned upstream checkout
├── weights          # ignored model-local checkpoint view
└── bin/             # ignored non-Python native tools, when required
```

The active environment is never shared between models and micromamba is not
part of this runtime path.  Existing source and weight downloads in the legacy
`external_models/` archive may be linked into a model workspace without moving
or deleting them.  The source revision is still checked against
`configs/external_models.json` before synchronization.

Synchronize and verify one model with:

```bash
bash scripts/others/sync_model.sh sigmadock
bash scripts/others/sync_model.sh surfdock
bash scripts/others/sync_model.sh diffbindfr
bash scripts/others/sync_model.sh interformer
```

Run a command in a synchronized model without touching another environment:

```bash
bash scripts/others/run_model.sh interformer python -c \
  'import torch, pyvina_core; print(torch.__version__)'
```

Long installations run through `scripts/slurm/others_uv_sync.sbatch` on the
`cpu_only` partition.  GPU inference scripts consume these same model-local
environments. Interformer's Boost 1.84, Reduce 4.14, and `obrms` runtime are
also model-local. If no archived copy exists, synchronization downloads the
exact checksum-pinned native packages and materializes only the required files
below `others/interformer/bin`.

SigmaDock uses the official `v0.1.0-beta` checkpoint and GNINA 1.3.2 from the
ignored legacy artifact cache.  Both artifacts are checksum-verified before
the model-local environment is accepted.  Its native comparison arm uses 40
independent seeds, 25 diffusion steps, and Vinardo ranking.
