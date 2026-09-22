# Tests

The public suite covers model geometry and equivariance, graph preparation,
training/checkpoint behavior, confidence labels and selection, inference,
physical diagnostics, benchmark adapters and current manuscript aggregation.
Run it from the repository root after installing the development environment:

```bash
uv run pytest -q
```

The suite uses synthetic examples, frozen fixtures and mocked external runners;
it does not train new models or reproduce full benchmark campaigns. Some tensor
tests use CUDA when available; set `CUDA_VISIBLE_DEVICES=` to run CPU checks.

Tests dedicated to retired sweep reports, historical checkpoint comparisons and
one-off recovery campaigns are archived with those experiments. Their removal
does not replace the runtime, data-integrity, metric or checkpoint tests above.
External-model imports use `benchmarks.external_models` directly.
