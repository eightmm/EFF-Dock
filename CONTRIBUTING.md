# Contributing to EFF-Dock

EFF-Dock is an experimental molecular-docking codebase. Contributions should
keep scientific claims, runtime behavior, and generated artifacts clearly
separated.

## Development setup

Use Python 3.12, Git LFS, and the pinned `uv.lock` environment:

```bash
git lfs install
uv sync --frozen --group dev
./scripts/check.sh fast
./scripts/check.sh ml-smoke
```

The pinned runtime targets Linux and NVIDIA CUDA 13. Run the local checks below
before submitting a change; CUDA behavior must be verified on a compatible GPU
when the modified path requires it.

## Pull requests

- Keep changes scoped and avoid committing raw datasets, generated outputs,
  credentials, machine-specific paths, or scheduler logs.
- Add focused tests for behavioral changes and document any skipped GPU test.
- Preserve frozen benchmark manifests and report experimental results with the
  exact checkpoint, configuration, input identities, and random seeds.
- Do not present reference-defined redocking results as blind-pocket or
  prospective docking performance.
- Run `git diff --check` and the two checks above before opening a pull request.

For repository layout and reproducibility requirements, see
[`docs/STRUCTURE.md`](docs/STRUCTURE.md),
[`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md), and
[`docs/BENCHMARK_RESULTS.md`](docs/BENCHMARK_RESULTS.md).

Unless explicitly stated otherwise, contributions intentionally submitted to
EFF-Dock are accepted under the Apache License 2.0.
