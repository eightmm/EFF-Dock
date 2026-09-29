# Frozen-weight orientation diagnostic

`summary.json` contains the one-seed paired comparison: Astex 85/85 and
PoseBusters v2 **307/308 evaluated**. `verification.json` records code regression
and finite full-model rotation checks. These are separate from the legacy
three-repeat manuscript benchmark data.

[Equations, discussion, compatibility and limitations](../../../../docs/ORIENTATION_INJECTION.md)
include the unresolved original-model repeatability guard for 7UJ5_DGL,
missing-outcome bounds and the Astex replay sensitivity. Uncertainty intervals
resample complexes from one execution; they exclude seed/execution variance.
No missing-case failure imputation or corrected-model accuracy claim is made.

Regenerate the separate comparison using aggregate data only:

```bash
uv run python -m benchmarks.figures.orientation --output outputs/orientation_figure
```

The source comparison used immutable released weights and the historical source
revision in `summary.json`, changing only the orientation contraction in memory.
Raw structures, pose banks, checkpoint tensors and local execution details are
not included here. The new named runtime options preserve those operator
semantics; a fresh execution need not reproduce numeric trajectories bitwise.
