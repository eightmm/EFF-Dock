# Manuscript materials

The article, Supplementary Information, figures and numerical Source Data now
live in the canonical [`papers`](../../papers) directory.

- [Article PDF](../../papers/main.pdf)
- [Supplementary PDF](../../papers/SI.pdf)
- [Figure captions](../../papers/figure_captions.md)
- [Prism authoring ZIP](../../papers/Prism.zip)

Use the figure numbering and captions in those canonical manuscript files.

Recompute the benchmark-specific rates and sample SD from the released case
outcomes, including both-metric-assigned conditional denominators:

```bash
uv run python -m benchmarks.analysis.published_endpoints
```

This verifies the numerical aggregation in Supplementary Table S15 and
Figure S20's endpoint summaries. Molecular scoring and candidate selection are
separate steps; their protocols and limitations are documented in the SI.
