# Paper analysis: frozen-bank completion

Authorized 2026-09-19. This is descriptive analysis of already-opened external
results, not a new checkpoint, hyperparameter, selector or production-admission
study. Existing weights, splits, candidate banks and official evaluation files
are immutable. No new docking or training is included.

## Scope and units

1. Training overlap: annotate ligand identity and available receptor/pocket
   similarity against the **verified executed fine-tuning loader index from the preserved training split**, not a future
   strict split. Report known overlap, known non-overlap and missing/unmapped
   separately. Do not infer independence from a missing similarity edge.
2. Candidate/selector decomposition: Top-1 and Top-5 use ascending predicted
   RMSD with deterministic candidate-order ties; oracle is the best actual
   symmetry-aware no-alignment RMSD among all saved candidates. The candidate
   success fraction counts poses with RMSD <2 A within each complex, then
   averages equally across complexes. Oracle is a diagnostic, never a deployed
   selection method. All thresholds and metrics remain unchanged.
3. Cost: report observed sampling/refinement/scoring stage times where recorded,
   Slurm elapsed/allocation separately, and memory only where measured. Host RSS
   and requested memory are not peak GPU memory. Missing measurements remain
   missing; equal N*S does not mean equal runtime.

Main supplied-pocket cohorts: Astex85, PoseBusters308, PhiBench206, FoldBench558,
OpenBind925, three repeats. Use N100/S10 unguided as the uniform main row where
available; preserve guided eta2 and N40/S25 as separately labelled ablations.
PhiBench includes three reconstructed cases; OpenBind includes flagged systems
and noncovalent approximations of two covalent ligands. No new exclusions.
PoseX remains separate: SD718/CD1312, seeds101/202/303, native group aggregation,
RMSD <=2 A and upstream-style relaxation. Do not pool with strict <2 A metrics.

Official PB-valid candidate metrics are computed only from actual official PB
labels. Fast geometry checks are not official PB labels. If only selected
poses have PB labels, no full-bank joint oracle/Top-5 or candidate PB fraction
is claimed. Any missing-label workload is inventoried explicitly before new
expensive evaluation is considered; no pass/fail values are imputed.

## Evidence and execution

Primary paired source: `outputs/benchmarks/astex_pb_unguided_r3_hostmatched_v2`.
Temporal selected-pose source index:
`outputs/benchmarks/external_chirality_u70k_temporal_full_r3_v1/pb_inchi_compat_v1/provenance.json`.
The latter combines 69 official-redock shards with three disclosed FoldBench
energy-reference InChI-repair shards; never describe all 72 as unmodified.

Every result stores cohort denominators, repeat metrics, sample SD, source
paths/hashes and limitations. Scientific tables and static PNG/PDF figures are
generated from machine-readable aggregate files, not manually copied values.
Large bank scans and overlap computations run on cpu_only; lightweight unit
tests and metadata inspection may run locally. No public upload, commit or push
is part of the initial analysis request. Parent verifies all delegated recommendations/code.

## Authorized publication follow-up

The user subsequently authorized per-complex/per-pose runtime reporting and
GitHub publication of the analysis code, aggregate results, figures and prose.
Keep raw banks, per-case third-party ligand metadata, scheduler logs, local
machine paths and unrelated worktree changes out of this publication.
Whole-pipeline/refinement cost is amortized by N generated poses; confidence
cost by 2N raw/refined evaluations. No single-pose latency or per-pose GPU
memory is inferred. Existing frozen reports and model defaults remain unchanged.
