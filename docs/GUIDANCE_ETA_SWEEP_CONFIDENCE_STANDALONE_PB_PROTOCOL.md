# Standalone one-pass confidence-selected eta-sweep protocol

Protocol: `EFFDOCK-UNIFIED-GUIDANCE-ETA-SWEEP-CONFIDENCE-STANDALONE-PB-V1`

Status: frozen before submission. This is a fresh, parent-free, one-pass
characterization. It does not replay or compare candidate coordinates against a
previous eta-sweep run and makes no deterministic-replay claim.

Astex Diverse and PoseBusters v2 outcomes from related experiments have already
been opened. Consequently, every value produced here is paired descriptive
evidence only. The run cannot select an `eta`, tune or replace a selector,
change a model, or admit guidance to the production sampler. All arms are
reported; the user, not the reporting program, decides how to interpret them.

## Question and fixed estimands

On one newly sampled candidate set for every frozen eta arm, what top-1 outcome
is obtained from:

- `confidence` (primary): the candidate with minimum predicted RMSD from the
  frozen confidence checkpoint; and
- `confidence_filter` (diagnostic): the already-frozen, cluster-free near-tie
  and clash filter around pure confidence.

The diagnostic filter did not pass its PLINDER deployment gate and cannot
replace pure confidence. Sampling uses the explicit evaluator profile
`confidence_cluster_free`. That profile computes and persists only pure
confidence and the fixed cluster-free filter as selected-pose outputs. It does
not call legacy Torch-Vina scoring/selection or pairwise cluster/density
`confidence_final`, and the integrity audit rejects either output if present.
First-pose and oracle RMSD remain evaluation-only diagnostics; neither selects
the reported top-1. Vina guidance and Vina selection are explicitly outside
this protocol.

The evaluation unit is one top-1 pose per complex, eta, and selector. For every
dataset/eta/selector cell, the strict report records:

- symmetry-aware selected RMSD `<2 Angstrom`, count and percent;
- median selected RMSD;
- official PoseBusters `0.6.5` `redock` pass-all over the frozen 27 non-RMSD
  checks, count and percent; and
- `RMSD <2 Angstrom AND PB-valid`, count and percent.

Within each selector, every eta is paired by immutable complex ID against the
same-run eta `0` arm. At each eta,
`confidence_filter - confidence` is paired the same way. The report includes
percentile 95% paired complex-ID bootstrap intervals and transition counts. It
must not emit a winning eta or selector.

## Frozen cohorts and model stack

- cohorts: all audited `85` Astex Diverse and `308` PoseBusters v2 complexes,
  for `393` complexes per eta and `3,144` sampled complex/eta rows overall;
- eligibility input only:
  `outputs/benchmarks/guidance_eta_sweep_v2_runs/20260801T102903Z/audit/combined.json`,
  SHA-256
  `dac7903488ccd36552a9bca134e37e633e3f07166d94f0389837012081ff3048`;
- eta: `0, 0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5`;
- docking checkpoint:
  `weights/effdock_geometry_ft_100k_best.pt`, SHA-256
  `6932fb3ba6ebac770f714453529656a44b8f33cf15119d23c9e675d2d60b36db`;
- confidence checkpoint:
  `weights/effdock_confidence_extmatch_n80_s25_step42500.pt`, SHA-256
  `e31fde6f351284205c78f7a1510002779c43312e94d9f82003d47a14d72bc78f`;
- docking config: `configs/train.yaml`, SHA-256
  `39aa62e4a48ed6f3aa4ff59345fb43a81220e2baba22edfd5beb0c4981b307ec`;
- benchmark input manifest: `docs/GUIDANCE_BUDGET1000_FULL_INPUTS.json`,
  SHA-256
  `99f15f557644cc51c3dd1f559b0dd97dd4259c1de3e1403fb761b7c7e079f668`;
- Astex pocket centers SHA-256:
  `1ac4d8629a7ee2adb785173db56fb69ec4140d68e3057631ae10df6ef88d0d85`;
- PoseBusters pocket centers SHA-256:
  `2d3db55c8cc75650cff85d8e3c12445fb8f45fbe2673d8bbc32045ee8c0f6ad0`.

The inherited cohort audit supplies eligibility and exact complex IDs only. No
parent sampling CSV, pose, score, aggregate, selector outcome, or
PoseBusters result is an input to this run.

The confidence model was retained for `N80/S25/sigma0.5/pocket10`; this
`N100/S10` use is a declared distribution-shifted characterization. Predicted
RMSD is used only for within-set ranking and is not treated as a calibrated
RMSD or success probability.

## Frozen sampling intervention

Every complex uses one set of `100` candidates, `10` learned ODE steps, prior
pool size `100`, sigma `0.5`, seed `42`, late schedule power `3`, pocket cutoff
`10 Angstrom`, zero center jitter, and no refinement. The guidance coupling is
the repository's unified physical-plus-interaction energy in
`normalized_drift` mode with:

- start time `0.5`, ramp power `1.0`;
- max force `20.0`, max translational velocity `5.0`, max angular velocity
  `5.0`;
- max atom displacement `0.25 Angstrom`, max backtracks `8`;
- protein shell `18.0 Angstrom`; and
- receptor policy `geometry_only`.

The only swept input is the frozen guidance scale `eta`. All eta arms for a
complex must report the same sampling seed and prior-pool SHA-256. Eta `0` is
the unguided within-run baseline; it is not imported from a prior run.

This historical standalone V1 audit retains exact cross-eta prior-pool digest
equality. The later steric high-eta audit V2 diagnostic policy does not loosen
or reinterpret this completed V1 contract.

The sampler is evaluated on CUDA without changing TF32, scatter kernels,
deterministic-algorithm settings, or any other numerical backend relative to
the frozen direct-drift configuration. CUDA scatter/atomic and fused-kernel
execution is not guaranteed to be bitwise reproducible. This protocol therefore
characterizes exactly this one fresh execution and does not rerun it to claim
candidate identity.

## Parent-free integrity contract

Before either official full-cohort PoseBusters stage is released, the audit
`EFFDOCK_CONFIDENCE_STANDALONE_INTEGRITY_V1` must fail closed unless all of the
following hold:

- exact inventory: 16 smoke summaries/CSVs with one row each, then 128 full
  summaries/CSVs covering 16 dataset/eta cells and 8 shards per cell;
- exact full coverage: `85` Astex plus `308` PoseBusters IDs per eta, `3,144`
  unique complex/eta rows, no duplicates, and zero recorded failures;
- exact protocol, run-name, checkpoint, confidence-checkpoint, config,
  benchmark-input, pocket-center, cohort, scientific-setting, and runtime
  provenance;
- exact evaluator selector profile `confidence_cluster_free`, with no Vina or
  pairwise cluster/density selector columns or saved-pose artifacts;
- identical seed and prior-pool SHA-256 across the eight eta arms for each
  complex;
- exactly `100` finite, range-valid entries in every candidate score ledger,
  with `confidence_atom_ok`, `confidence_atom_q90`,
  `confidence_atom_rmsd`, `confidence_rmsd`, `confidence_success`,
  `confidence_success_logit`, and `pl_clash_1p6` present per candidate;
- exact recomputation of the pure confidence argmin and frozen cluster-free
  filter, valid indices in `[0, 99]`, and exact hashes for their saved SDFs;
- current protein, reference-ligand, and selected-pose bytes equal their
  sampling-time hashes;
- complete paired artifact/file ledgers and exact ID/order/shard coverage; and
- an RTX 6000 Ada runtime record with positive peak CUDA memory below the
  `48 GiB` device limit, with no OOM or non-finite failure.

The pre-confidence candidate digest is accepted only as
`digest_present_and_producer_bound`: it binds the scores to the candidate
tensor emitted by this frozen sampler process. Because all 100 candidate
coordinates are not persisted, the audit cannot independently reconstruct that
digest and must not describe it as independently verified.

The saved audit must state:

```text
mode=fresh_one_pass_characterization
parent_compared=false
deterministic_replay_claim=false
status=passed
```

The final report rebuilds the full integrity audit from current CSV, summary,
SDF, protein, and reference files and requires exact equality with the saved
audit before reading official outcomes. Each official shard then binds its
PoseBusters CSV/summary to the exact sampling row and current input hashes.
PoseBusters must be runtime version `0.6.5`, configuration `redock`, and expose
the same frozen 27 non-RMSD checks. Missing/duplicate IDs, survivor-only
denominators, absent bindings, hash drift, a PoseBusters exception, or any
partial shard fails the report.

The two-selector runtime smoke has a separate fail-closed post-run gate. It
rechecks the installed `0.6.5` runtime, `redock`, the exact 27-check non-RMSD
schema, the separately excluded `rmsd_≤_2å` check, both selector labels,
sampling-time protein/reference/selected-pose hashes against current bytes,
exactly one successful `7b2c_tp7` row, zero failures, and the current
summary-to-CSV path, ID, pass-all conjunction, and validity percentage. It does
not create a full-cohort binding: the `--only-id` sampling smoke intentionally
retains the dataset-wide discovered total (`308`) while its CSV contains one
assigned row, so applying the full binding's equal-total invariant would be
incorrect. Exact standalone bindings begin with the 256-task full official
array.

## Submission and execution gates

The launcher accepts one safe run ID and atomically reserves only:

```text
outputs/benchmarks/guidance_eta_sweep_confidence_standalone_runs/<RUN_ID>
```

An existing root is never reused. Each stage also reserves its exact task once;
failed or partial tasks are preserved and are not silently treated as complete
or overwritten. A new run ID is required for another execution.

Before submission, the launcher verifies the declared immutable hashes and
writes both a frozen-input manifest and an execution manifest. The execution
manifest covers the complete active `src/effdock` package, `pyproject.toml`,
`uv.lock`, config, benchmark-input manifest, pocket centers, eligibility audit,
this protocol, and every script in this chain. Every delayed stage verifies
both the manifest file's SHA-256 and every listed file before importing code or
reading outcomes. If launcher submission fails partway, all job IDs already
returned by that launcher are cancelled and `.submission.failed` is retained;
the run root and any produced evidence are not deleted.

```text
16-task GPU smoke (2 datasets x 8 eta; fixed ID in each cell)
  -> standalone smoke integrity audit
  -> hash-bound official PoseBusters runtime smoke (2 selectors)
  -> 128-task full GPU sampling (2 datasets x 8 eta x 8 shards)
  -> standalone full integrity audit
  -> 256-task official PB array
     (2 selectors x 2 datasets x 8 eta x 8 shards)
  -> one strict combined report
```

Every edge is Slurm `afterok`. Sampling uses the ordered partition list
`6000ada,heavy`, with no more than eight simultaneous GPU tasks. Each task has
exactly one visible GPU and at least 48,000 MiB visible memory; `6000ada`
requires RTX 6000 Ada, while `heavy` permits H100 or RTX PRO 6000-class
devices. Runtime summaries and audits retain the actual partition, GPU name,
and total memory. Official checks run on `cpu_only`, with no more than 16
simultaneous full PB tasks. The smoke IDs are `1jje` for Astex and `7b2c_tp7`
for PoseBusters. The 128 and 256 numbers above are scheduler task counts, not
complex counts. This scheduler expansion does not change eta arms, `N100/S10`,
seeds, priors, guidance, or selectors.

## Stop and interpretation rules

Any smoke, manifest, sampling, integrity, binding, version, coverage, memory,
official-check, or final rebuild failure blocks every downstream stage through
`afterok`. No tolerance is widened and no failed run is salvaged into a
claim-bearing aggregate. The unchanged failed deterministic replay remains
separate provenance and is not an input here.

Success produces data artifacts under the fresh run root: smoke artifacts, raw
full sampling and selected SDFs, smoke/full audits, official PoseBusters shards
and bindings, manifests/submission metadata, and `aggregate.json`. Slurm
stdout/stderr remain job-ID-scoped in the shared ignored
`outputs/benchmarks/logs/` directory. These artifacts remain ignored data;
protocol and job/result provenance can be recorded in the repository after
completion without changing the frozen run.
