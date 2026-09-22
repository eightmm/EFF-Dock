# Repository structure

| Path | Purpose |
|---|---|
| `src/effdock/` | Docking/confidence models, graph preparation, training, inference and evaluation |
| `configs/` | Model, training and evaluation configurations |
| `weights/` | Released docking/confidence pair, checksums and model cards |
| `benchmarks/external_models/` | External model adapters, environments, runners and evaluators |
| `benchmarks/results/` | Compact comparison metrics and provenance |
| `benchmarks/figures/` | Retained comparison and visualization scripts |
| `scripts/` | Workflow helpers, release checks and supported launcher paths |
| `tests/` | Unit, scientific-contract and regression tests |
| `docs/` | Method, usage and manuscript documentation |
| `docs/paper/` | Manuscript working figures, captions and source mapping |

The main Python inference interface is `effdock.inference.DockingOptions` and
`effdock.inference.dock`. The `eff-dock` CLI wraps data preparation, training,
confidence and benchmark workflows. Run examples from the repository root so
configuration and weight paths resolve consistently.

```text
protein + ligand + explicit pocket
  -> graph preparation
  -> docking model samples poses
  -> paired confidence model scores poses
  -> frozen selection policy
  -> SDF + results.pt + provenance
```

Benchmark implementations and external Slurm launchers live under `benchmarks/`.
The retained `configs/external_models.json` alias points to the model manifest;
old benchmark aliases under `scripts/` are no longer published.
Retained analysis and guidance diagnostics are not automatically part of the
released inference preset. See [evaluation](EVALUATION.md) for metric boundaries
and [reproducibility](REPRODUCIBILITY.md) for checkpoint identities.

Raw data, pose banks, scheduler logs, installed upstream environments and
historical experiment archives remain in ignored local storage. Superseded
campaign tools and intermediate results are excluded from the public checkout.
Some retained historical report helpers expect those separately supplied inputs.
The public repository does not contain every development run or third-party
artifact.

Start with [the documentation index](README.md). The [paper guide](paper/README.md)
identifies the figure bundle prepared for manuscript drafting.
