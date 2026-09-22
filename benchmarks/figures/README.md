# Comparison plotting

Retained scripts render model comparisons from compact result artifacts under
`benchmarks/results/external_models`. Use [the manuscript figure index](../../docs/paper/README.md)
for the current paper-writing PDF and matching captions.

- `plot_joint_oracle.py`: refinement and oracle comparison.
- `plot_top1_oracle.py`: compact confidence-selected and oracle comparison.
- `plot_pb_valid_comparison.py`: PoseBusters-valid comparison.

The comparison plots distinguish local repeated runs from literature-reported
values and enforce the supplied-pocket comparison boundary. Their individual
input conditions must be preserved. Intermediate fixed-NFE and pocket/prior
plot scripts are no longer included in the public tree.

`scripts/figures` remains a compatibility alias. Introduction and visualization
helpers are separate from the manuscript's benchmark figure selection.
