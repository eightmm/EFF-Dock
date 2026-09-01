# External model installation status

> Historical installation snapshot. Current source/environment/checkpoint
> verification is summarized in
> [`EXTERNAL_MODEL_AUDIT_20260830.md`](EXTERNAL_MODEL_AUDIT_20260830.md).

Snapshot: 2026-08-28 12:32 KST
EFF-Dock revision at submission: `862409d19159936e556c1bc5e0e2d16dfa1b5cfb`
External runtime implementation: `00a47ae55eb144eb9a8702b0d38bf4d57f5503cf`

Machine-readable source revisions, pipeline taxonomy, and compatibility
overrides are in `benchmarks/external_models/models.json`. Artifact URLs, sizes, and
digests are in `configs/external_model_artifacts.json`.

## Active runtime layout

The active comparison-model runtime is now model-local:

```text
others/<model>/{pyproject.toml,uv.lock,.python-version,.venv,upstream,weights,bin}
```

- `pyproject.toml`, `uv.lock`, and `.python-version` are reproducible tracked
  inputs.
- `.venv`, uv cache, upstream checkout, checkpoints, and native tools are
  ignored runtime artifacts owned by exactly one model.
- `uv sync --project others/<model>` creates the environment; inference uses
  `uv run --project others/<model> --no-sync` through
  `scripts/others/run_model.sh`.
- The old ignored `external_models/` tree is retained as an archive and initial
  source/weight cache. It is no longer the active environment path for the
  models below.

## Completed model environments and inference gates

All installation work is on Slurm `cpu_only`; no CPU-count override is
requested.

| Model | Python | Torch/CUDA contract | Final sync | End-to-end smoke | Coverage |
|---|---:|---|---:|---:|---:|
| SurfDock | 3.10.20 | Torch 2.2.2 / cu121 | `60105` | `60112` | 1/1 target, 1/1 pose |
| DiffBindFR | 3.9.25 | Torch 1.13.1 / cu117 | `60106` | `60107` | 2/2 targets, 2/2 poses, MDN scores present |
| Interformer | 3.12.3 | Torch 2.4.0 / cu118 | `60128` | `60129` | 1/1 target, 1/1 sampled pose |

Each smoke executed the released checkpoint and native inference entrypoint;
an import-only check is not counted here. Coverage JSON is stored below the
corresponding ignored `outputs/external_models/runs/<model>/smoke/` directory.

The superseded shared-micromamba retries `60058`, `60061`, and `60070` were
cancelled after the layout decision. Downloaded sources, weights, and partial
environments were retained. Failed uv attempts were used only to identify
missing build/runtime contracts; the table records the final successful jobs.

## Submitted full benchmark runs

All full jobs use the frozen Astex `N=85` and PoseBusters v2 `N=308`
denominators, `seed=0`, fail-on-incomplete coverage checks, and at most two
concurrent array tasks per model/dataset.

| Model | Astex job/configuration | PoseBusters v2 job/configuration | State at snapshot |
|---|---|---|---|
| DiffBindFR | `60116`, native S20/N40 | `60115`, native S20/N40 | PENDING (Priority) |
| SurfDock | `60119`, native S20/N40 | `60118`, native S20/N40 | PENDING (Priority) |
| Interformer | `60130`, native N20 | `60131`, native N20 | PENDING (Priority) |

## Existing inference preserved during migration

The already validated DiffDock-Pocket and RLDiff RL++ environments remain in
place until their submitted full runs finish. This avoids invalidating active
processes while the remaining models move to `others/`.

- DiffDock-Pocket Astex `60062`: 4/4 tasks completed. PoseBusters `60064`:
  4/12 completed and two running at this snapshot.
- RLDiff RL++ Astex `60065`: 4/4 tasks completed. PoseBusters `60066`:
  two tasks running at this snapshot.
- The cropped long-chain smoke tests completed with full `1/1` target and
  `5/5` pose coverage before those full submissions.

## Compatibility corrections

- DiffDock-Pocket: inject the missing `filtering_dir` alias and isolate only the
  unused optional OpenFF import when relaxation is disabled.
- DiffBindFR: use `scikit-learn==1.4.1.post1` and compatible
  `joblib==1.3.2`; keep Torch 1.13.1 while installing matching official PyG
  wheels. Pin `setuptools==80.9.0` because Torch 1.13's extension loader still
  imports `pkg_resources`. The no-error-correction arm uses a fail-closed PyMOL
  import shim, because PyMOL is neither called with supplied crystal ligands
  nor available as a Python 3.9 uv wheel.
- SurfDock: pin the exact Dimorphite-DL 1.3.2 fork commit and official Meta ESM
  2.0.1 commit, `setuptools==80.9.0`, and IPython 8.37.0. The runtime wrapper
  supplies the malformed upstream `global_vars.py` constants, launches APBS
  beside its basename-referenced PQR, and stores SO(3)/torsion lookup arrays in
  the model-local cache. The no-refinement arm uses a fail-closed shim for the
  eagerly imported OpenFF force-optimization module; requesting force
  optimization raises instead of silently changing behavior.
- Interformer: build PyVina inside its uv environment against a model-local copy
  of the pinned Boost 1.84 headers/runtime. Reduce, `obrms`, and their native
  runtime libraries are materialized into `others/interformer/bin` from exact
  checksum-pinned packages when no archived copy exists. A discovery marker
  bridges the upstream `dbm.dumb` `*.db.dat` glob to Python 3.12's single-file
  `dbm.gnu` output. The `obrms` CLI and its OpenBabel library are isolated from
  the Python OpenBabel wheel to avoid ABI interposition.
- Every model source must match the pinned 40-character commit before uv sync,
  and each checkpoint directory must exist before its environment is accepted.

## Verification

```bash
bash scripts/others/sync_model.sh <model>
bash scripts/others/run_model.sh <model> python -c 'import torch; print(torch.__version__)'
```

The focused repository test verifies the isolated layout and native ABI pins:

```bash
.venv/bin/python -m pytest -q tests/test_external_model_recovery.py
```

Current focused result: `2 passed`.
