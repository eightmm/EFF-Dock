# Manuscript materials

The accompanying manuscript is under review and is not distributed in this
repository. Its numerical Source Data are in [`papers/data`](../../papers/data).

Recompute the benchmark-specific rates and sample SD from the released case
outcomes, including both-metric-assigned conditional denominators:

```bash
uv run python -m benchmarks.analysis.published_endpoints
```

This verifies the numerical aggregation in Supplementary Table S15 and
Figure S20's endpoint summaries. Molecular scoring and candidate selection are
separate steps; their protocols and limitations are documented in the SI.
