# Verification and reproducibility — 2026-09-19

## Scope and ownership

GPT-5.6-sol investigated/reviewed training exposure, GPT-5.6-terra supplied and
reviewed candidate analysis, and GPT-5.6-luna supplied and reviewed runtime
accounting. The parent integrated the code, executed it and checked outputs.
All work here is frozen-bank analysis: no new model training, docking,
production-default change. A subsequent user request authorized publication
of the analysis code, aggregate results, figures and documentation.

## Completed analysis jobs

All jobs used the CPU-only partition; no new GPU allocation was used.

| Job | Work | Outcome |
|---|---|---|
| 78751 | Main benchmark tables and Figures 1–3 | Complete |
| 78757 | Candidate collector including underlying score-CSV hash/vector checks | Complete |
| 78770 | Final runtime collector and runtime/candidate figures | Complete |
| 78778 | Exact executed-training membership audit, stratification and Figure 6 | Complete, 34 s |
| 78779 | Final candidate/exposure figure legend and label correction | Complete, 3 s |
| 78783 | Publication aggregates, per-complex/per-pose cost tables and portable figures | Complete, 7 s |
| 78784 | Staged public-only checkout: 19 tests and all six figures regenerated without raw banks | Complete, 15 s |

Earlier runtime job 78758 stopped because the collector expected eight shards
for a temporal cohort that correctly has sixteen; the explicit per-cohort
expectation was corrected and the complete collector reran successfully.
Earlier overlap job 78764 stopped on one invalid-valence training SMILES.
That system is now explicitly missing for ligand identity/similarity, never
silently repaired or treated as certified non-overlap. Its PDB accession still
participates in the independent accession audit.

## Output contracts and checks

- Main benchmark: 44 condition rows, three repeats, explicit cohort counts;
  joint success cannot exceed RMSD or PB success.
- Candidate analysis: 22 aggregate rows and 19,566 complex/stage records;
  all candidate indices, finite scores, symmetry-RMSD vectors, stable ties,
  filter fallback, summary hashes and score-file hashes checked. All 22
  ordinary and filtered Top-1 aggregates match existing selected-pose reports.
- Exposure: all 2,082 external IDs annotated. The 47,277-system fine-tuning
  loader index is a verified ordered subset of the preserved 47,310 train IDs.
  The loader cache key is independently reconstructed; content SHA256 is
  `21b194112242d0645cd61faeddec09d3188c0c5e15b66ab0ffee4c6ccf2a02b4`.
  There are 47,276 parseable entries and 33,513 unique canonical ligands.
  All 152 stratified summary rows retain per-repeat denominators.
- Selected PB sources: 48 Astex/PB shards and 72 temporal shards; temporal
  source hashes/profiles checked (69 ordinary, three disclosed FoldBench
  energy-reference compatibility shards).
- Runtime: measured fields, per-cohort shard counts, completed-case coverage,
  and source manifest hashes checked. Sampling allocator peaks are not host
  RAM requests. Summed shard-wall cost is not campaign elapsed time.
- PoseX: complete 718 SD / 1,312 CD inputs for each of three seeds in each
  reported condition; native 718 / 109 group denominators retained.

All **19 focused tests passed** in the publication follow-up, including N versus
2N normalization and preservation of process memory peaks. Tests cover thresholds, ranking, fallback, invalid/missing numeric
values, stereochemical identity, H normalization and membership-key drift.
Changed Python files pass targeted lint and compilation; the Slurm entry
point passes shell syntax validation. Exact final commands/results are in
the task handoff and logs.

The repository-wide `scripts/check.sh fast` failed on eight pre-existing,
unrelated lint findings in four `benchmarks/figures/` scripts and
`scripts/relax_posex_fixed_receptor.py`. Those files were not modified for
this analysis. Later full-gate steps were consequently not executed; a clean
full-repository gate is not claimed.

Publication checks also passed: staged diff whitespace, shell syntax, targeted
lint, JSON parsing, public-only Markdown links and exclusion of machine-local
absolute paths/raw banks/per-case ligand metadata. The public-only checkout
passed 19 tests and regenerated all six PNG/PDF figures using only released
aggregates. The full-bank collectors still require the retained local inputs.

## Reproduction and source provenance

`scripts/slurm/paper_analysis.sbatch` regenerates tables, figures and aggregates
using the retained local banks, metadata and loader index. JSON files record
input paths/hashes; the runtime manifest is retained under
`outputs/benchmarks/paper_analysis_20260919/`. Run metadata is in
`outputs/benchmarks/paper_analysis_20260919_ledger.jsonl`. The manuscript prose
and the separately consolidated PoseX table require editorial updates if
source results change. Third-party metadata is not redistributed by this task.

## Measurements not available from current artifacts

1. External-to-training protein/pocket nearest-neighbor matrices. No validated
   sequence/pocket index exists locally; PDB mismatch cannot replace homology.
2. Complete historical pretraining exposure before the audited S50 fine-tune.
3. Full-bank official PB labels, joint-PB Top-5/oracle or PB-valid density.
   Selected-pose PB validity and candidate chirality are not substitutes.
4. Isolated sampling latency, host peak RSS, refinement/confidence GPU peaks.
5. Target-level confidence intervals or a preregistered independent test.

These are explicitly unavailable, not zero or silently estimated. The current
manuscript draft limits its claims accordingly. Additional data extraction or
measurement campaigns would be needed before adding those endpoints.
