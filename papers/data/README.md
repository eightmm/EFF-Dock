# Numerical data accompanying EFF-Dock

Use the endpoint definitions in [the article](../main.tex), [Supplementary
Information](../SI.tex) and [figure captions](../figure_captions.md) when
interpreting these files. Rates are percentages; rate differences are percentage
points. Means and sample standard deviations summarize three inference repeats
of frozen networks, rather than three independently trained models.

| File | Contents |
|---|---|
| [source_data.json](source_data.json) | Nine numerical bundles covering primary performance, candidate and ranking diagnostics, training-set relatedness, paired uncertainty, generation-input sensitivity and the historical acquisition runtime measurements. |
| [source_data.csv](source_data.csv) | The same numerical content as JSON-pointer/value rows. Each pointer identifies a scalar or empty container in the JSON; this is distinct from the complex-level endpoint CSV below. |
| [published_endpoints.json](published_endpoints.json) | Benchmark-specific aggregate endpoints, repeat counts, evaluator versions, score availability and denominator definitions for Supplementary Table S15 and Figure S20. |
| [published_endpoints.csv](published_endpoints.csv) | Complex-level endpoint outcomes used to calculate those aggregates. Each row identifies a complex, repeat, candidate-bank stage, saved prefix and Top-k selection. |
| [foldbench_strata.json](foldbench_strata.json) | Fixed docking-training accession and release-date strata for Supplementary Table S16, including membership IDs and aggregate outcomes. |
| [confidence_only_overlap.json](confidence_only_overlap.json) | Additional training-set overlap checks for the 25 confidence-training samples absent from the docking-training inventory. |
| [figure1.json](figure1.json) | Candidate ranks, predicted and retrospective RMSD, input provenance and the scope of the separate illustrative trajectory in Figure 1. |
| [literature_context.json](literature_context.json) | Source-defined published comparator values and cohort/protocol qualifications; these are not locally executed paired comparisons. |

The primary OpenBind cohort contains 925 complexes; the benchmark-specific
follow-on analysis uses the official 802-case annotation subset. FoldBench uses
558 interfaces. The locally selected PhiBench-derived cohort contains 206
complexes and is not claimed to reproduce the original published cohort or
protein-aligned RMSD evaluation.

For FoldBench, the benchmark-specific conjunction requires binding-site-superposed,
symmetry-corrected RMSD below 2 Å and LDDT-PLI above 0.8 on the same pose.
For OpenBind, RMSD–PoseBusters requires native RMSD at most 2 Å and all 27
non-RMSD PoseBusters checks; the additional conjunction also requires
LDDT-PLI at least 0.8 on that pose. EFF-Dock uses a supplied holo receptor;
matching structural endpoints does not match the original FoldBench cofolding
task or every benchmark's preparation and sampling budget.

Full-cohort rates divide confirmed successful complexes by all complexes.
Blank CSV endpoint fields are unassigned or inapplicable, not zero. A known
failed component can resolve a conjunction as false despite another unassigned
component. Endpoint-decidable conditional rates therefore differ from the
subset with both structural metrics assigned; the JSON reports them separately.
For Top-k conjunctions, one candidate must satisfy all requirements.
Saved-prefix analyses restrict the existing 100-pose banks before selection;
they do not measure fresh smaller-budget generation or equal compute.

The sensitivity data contain 24 conditions evaluated on all 85 Astex and 308
PoseBusters complexes across three repeats, including the reused primary
condition. Generation crop and center jitter vary while refinement and ranking
retain the original center and a 10 Å crop. Prior scale also enters the docking
features recomputed for ranking. These data do not measure complete-pipeline
robustness to an incorrectly supplied pocket. The runtime bundle describes
initial candidate acquisition and excludes subsequent downstream corrections.
The sensitivity field `new_executions` counts the 27,117 nonbaseline banks
processed downstream in this analysis, rather than newly generated candidates;
the analysis uses saved generation outputs and adds no candidates. The remaining
1,179 banks reuse the primary condition.
