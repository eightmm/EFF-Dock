# Paper materials

## Current manuscript

- [Single current PDF](figure_gallery/paper_figures.pdf)
- [Page index](figure_gallery/PDF_EXPORTS.md)
- [English captions and author notes](prism/FIGURE_CAPTIONS.md)
- [Prism reference package](prism/prism_figure_reference.zip)
- [Machine-readable page/caption/source mapping](prism/captions.json)

The PDF contains 11 main figures and 3 supplementary diagnostics. The mapping
and captions above are authoritative for the current layout. Figure sources in
20260919/, 20260920/, combined_panels/, recent_benchmarks/, sequence_overlap/
and manuscript_support/ remain in place for reproducibility and stable links.
The Prism ZIP contains the individual PDFs, LaTeX blocks and a reference main.tex;
it is the only maintained export ZIP. Runtime/memory and guidance/budget
performance are separate figures.

## Supporting analysis evidence

The following directories retain numerical inputs, per-case audits and historical
methods. They are not additional figures to insert into the current manuscript.
Do not substitute their earlier page numbers or captions for the current mapping.
This inventory includes local-only analysis directories; not all underlying
records are published in GitHub. The current manuscript links above are published.

| Directory | Retained purpose |
|---|---|
| `20260919/`, `20260920/` | Frozen benchmark aggregates, complexity/budget analyses and failure decomposition |
| `sequence_overlap/` | Current sequence-relatedness and exact-ligand same-sample overlap definitions |
| `six_cohort_similarity/`, `external_direct_similarity/` | Earlier direct sequence/pocket calculations and coordinate checks |
| `plinder_exposure/`, `triple_exposure/`, `ligand_pocket_community_overlap/` | Historical PLINDER-community mapping and recovery audits |
| `split_exclusion_audit/` | Reconstruction of the pretraining split exclusion mechanism |
| `pocket_similarity_pilot/` | Bounded exploratory pocket calculation |
| `confidence_input_recovery/` | Separate confidence-input exclusions and recovery diagnostics |
| `manuscript_support/` | Statistical comparisons and reporting-condition evidence |

Older pocket/community measurements were replaced in the selected figure set,
not invalidated or silently converted into sequence identities. Their numerical
records remain evidence. Root-level experiment protocols and outcomes in docs/
are likewise retained; deleting them would erase reproducibility and negative
results. Code still reads some of these paths.

## Cleanup

Old selected/all ZIP variants, obsolete contact sheets and the rejected combined
quality/cost figure were removed from the local workspace. These 29 files were
untracked and were never part of the GitHub release. Current figures, captions, datasets, aggregate
JSON/CSV and protocols were preserved. See the [cleanup record](../maintenance/20260922_cleanup.json).
