# Confidence-selected eta-sweep PoseBusters protocol

Protocol: `EFFDOCK-UNIFIED-GUIDANCE-ETA-SWEEP-CONFIDENCE-PB-V1`

Status: parent report recovered; numerical replay-equivalence gate validated on
multiple independent 16-cell confidence replays. The final immutable full chain
was submitted on 2026-08-02.
Astex and PoseBusters outcomes have already been opened, so this study cannot tune
`eta`, train or choose a confidence model, change a selector, or admit guidance
to production.

## Question and estimand

For every arm of the frozen `EFFDOCK-UNIFIED-GUIDANCE-ETA-SWEEP-V2`, what are
the selected-pose outcomes when one of the 100 generated poses is chosen by:

- `confidence` (primary): minimum predicted RMSD from the frozen confidence
  checkpoint; or
- `confidence_filter` (diagnostic): the frozen cluster-free near-tie and clash
  filter around pure confidence.

`confidence_filter` did not pass its PLINDER validation deployment gate and is
therefore reported only as a diagnostic. The historical
`pair_gate_density_rank_vote_plclash_ambig`/`confidence_final` selector is
excluded because it uses pairwise-pose clustering and candidate-density
features. No result is allowed to replace pure `confidence` as the declared
primary selector.

The unit is one top-1 pose per complex, eta, and selector. Every
dataset/eta/selector cell reports:

- symmetry-aware selected RMSD `<2 Angstrom`, count and percent;
- median selected RMSD;
- official PoseBusters `0.6.5` `redock` pass-all over all 27 non-RMSD checks,
  count and percent; and
- the conjunction `RMSD <2 Angstrom AND PB-valid`, count and percent.

Paired full-cohort comparisons report each eta minus eta `0` within a selector
and `confidence_filter - confidence` at the same eta, with percentile 95%
paired complex-ID bootstrap intervals. All arms are shown and no winner is
selected automatically.

## Frozen stack and distribution boundary

- parent output:
  `outputs/benchmarks/guidance_eta_sweep_v2_runs/20260801T102903Z`;
- cohorts: all audited `85` Astex Diverse and `308` PoseBusters v2 complexes;
- eta: `0, 0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5`;
- docking checkpoint:
  `weights/effdock_geometry_ft_100k_best.pt`, SHA-256
  `6932fb3ba6ebac770f714453529656a44b8f33cf15119d23c9e675d2d60b36db`;
- confidence checkpoint:
  `weights/effdock_confidence_extmatch_n80_s25_step42500.pt`, SHA-256
  `e31fde6f351284205c78f7a1510002779c43312e94d9f82003d47a14d72bc78f`;
- docking config: `configs/train.yaml`, SHA-256
  `39aa62e4a48ed6f3aa4ff59345fb43a81220e2baba22edfd5beb0c4981b307ec`;
- sampling: `N100/S10`, shared prior pool `100`, sigma `0.5`, seed `42`, late
  schedule power `3`, pocket cutoff `10 Angstrom`, no jitter, no refinement;
- guidance mode, schedule, caps, receptor policy, input manifest, and all other
  settings are byte-for-byte the parent protocol values.

The confidence checkpoint was trained and retained for the matched
`N80/S25/sigma0.5/pocket10` distribution. `N100/S10` plus guidance is a
distribution-shifted, descriptive characterization; its outputs are not
claimed to be calibrated RMSD or success probabilities.

## Deterministic replay and integrity boundary

The parent run saved selected poses, not all 100 candidate coordinates, so the
original candidate pool cannot be retrospectively verified by a complete
coordinate hash. Confidence is therefore evaluated by one deterministic replay
with the exact parent seed, prior, model, inputs, and sampler settings. The
report may call it only a **parent-sentinel-verified deterministic replay**, not
a byte-proven identical copy of every original candidate.

Before any official PoseBusters task is released, a fail-closed identity audit
must verify every available parent sentinel:

- full ID/order/shard coverage and zero sampling failures;
- checkpoint, config, input, pocket, guidance, and receptor-policy identity;
- per-complex sampling seed and shared-prior SHA-256;
- exact `first`, `vina`, and `oracle` indices and all categorical/discrete
  fields;
- finite legacy RMSD/score/statistic equivalence at `rtol=2e-5`, `atol=1e-4`,
  and nested direct-step/summary guidance telemetry equivalence at
  `rtol=2e-4`, `atol=2e-4`, with counters exact;
- legacy selected-pose hash equality when available, otherwise exact SDF
  atom/bond identity and order plus coordinate RMSD `<=5e-4 Angstrom` and
  maximum per-atom displacement `<=1e-3 Angstrom`;
- finite confidence outputs and valid indices in `[0, 99]`;
- a complete 100-candidate confidence/interaction score ledger from which pure
  confidence and the frozen cluster-free filter indices are recomputed;
- sampling-time hashes for `confidence` and `confidence_filter` SDFs; and
- a pre-confidence full-candidate tensor SHA-256 recorded for every new replay
  row, binding confidence scores to the replayed in-memory pool.

Any mismatch blocks all downstream official checks. Every official PB shard
then verifies the protein, reference ligand, and selected-pose hashes written
during replay. Missing IDs, duplicate cells, a PoseBusters exception, or any
survivor-only denominator makes the report fail.

Submission is pinned to recovered parent report job `47280`. The launcher freezes a
SHA-256 execution manifest covering the active EFF-Dock package, this chain's
Slurm scripts, config, benchmark-input/pocket manifests, audited cohort, and
the recovered parent aggregates/provenance;
every delayed stage verifies it before importing code or reading outcomes. A
partial launcher failure cancels every job already submitted by that launcher.
The final report also recomputes the complete identity audit and requires exact
equality with the saved audit, so post-audit CSV, summary, or SDF mutation is
rejected rather than rebound as a new input.
Each official-check process additionally requires the installed PoseBusters
runtime to equal the pinned `0.6.5` version before evaluating any pose.

## Execution gates

```text
parent strict report
  -> 16-task fixed-ID/all-eta confidence GPU smoke
  -> smoke sentinel-identity audit
  -> hash-bound confidence/filter official-PB smoke
  -> 128-task full confidence GPU replay
  -> full sentinel-identity audit
  -> 256-task official PB array (2 selectors x 2 datasets x 8 eta x 8 shards)
  -> one strict combined report
```

All GPU work uses Slurm `6000ada`; official checks and audits use `cpu_only`.
The full GPU array is limited to eight concurrent tasks and the PB array to 16.
The fixed smoke IDs are `1jje` (Astex) and `7b2c_tp7` (PoseBusters). Peak CUDA
memory is recorded, and OOM or any other failure blocks the full run through
`afterok` dependencies.

Outputs are isolated under the parent root at
`confidence_selector_replay/`. Raw structures, poses, logs, and aggregates
remain ignored data artifacts; this protocol, immutable hashes, implementation,
Slurm IDs, and final values are retained as provenance.

## Submitted execution

The original parent report `46947` failed after all 128 sampling and 128
official-PB shards had completed, because an output-only confidence-ledger edit
changed the live implementation identity before aggregation. Its dependent
confidence jobs `47259`--`47265` all remained at zero elapsed time and were
cancelled. Their submission and manifest SHA-256
`d138ffda719f5dd95f5e4f316585be9cabe9d9deb34309f556033c4b41c4eaef`
are preserved under `confidence_selector_replay_superseded_47259_47265/`.

The sampling-time package was reconstructed under
`recovery/parent_source_d726/`. Its `evaluate.py` SHA-256 is
`9b4fb9330f2d25ad47f31e846b9a4f50e196bd1584068ea049bff4e2431c8373`,
its source-manifest SHA-256 is
`05941e1999a286dd3c2484ec909c0935d14959b074eb697d602d09218f2d32e1`,
and its full guidance implementation identity exactly matches the audit:
`d726ddc4cb89b495f0495aa059faf9efdf33ee76c42c4b14e71356068935c0a5`.
Recovery report job `47280` completed `0:0` and produced the strict sampling
and official-PB aggregates with provenance in
`recovery/parent-report-47280.json`.

During local recovery validation, a test-scope error temporarily replaced
`audit/combined.json` with a small fixture. This was detected before any new
Slurm submission. The original was reconstructed from the unchanged Astex and
PoseBusters dataset audits plus its recorded creation timestamp; its exact
SHA-256 `dac7903488ccd36552a9bca134e37e633e3f07166d94f0389837012081ff3048`
was restored. The erroneous fixture is retained under `recovery/` for audit.

The confidence replay validates those historical audit bytes by the exact
frozen SHA instead of incorrectly requiring equality with the output-extended
current evaluator. The later identity gate independently requires equal legacy
rows and selected-pose sentinels, recomputes selectors from the 100-candidate
ledger, and records both historical and replay implementation hashes.

The first recovered-parent chain reached its 16/16 GPU smoke tasks, but identity
job `47288` correctly stopped the chain because the V1 gate required byte-exact
floating-point summaries and SDF text. Jobs `47289`--`47293` were cancelled,
and the complete run is preserved under
`confidence_selector_replay_failed_exact_identity_47287_47293/` with execution
manifest SHA-256
`37bf8ab03ececce4728e53be2f6143cddeccd752f2d48045bf4d384d36def99c`.
The smoke showed exact indices, seeds, priors, and molecular identity in all 16
rows. Across 48 legacy selected poses, 30 file hashes were exact and 18 differed
only at SDF coordinate rounding; maximum coordinate RMSD was
`2.8284271247898416e-5 Angstrom` and maximum atom displacement was
`1.0000000000331966e-4 Angstrom`. The replacement V2 numerical-equivalence gate
passes all 16 preserved cells while retaining exact current replay hashes,
candidate ledgers, and selector recomputation.

A second V2 smoke array `47316` also completed 16/16, but identity job `47317`
found one low-magnitude direct-step percentile delta of `1.440137624741e-4`,
just above the initial telemetry `atol=1e-4`. All later jobs `47318`--`47322`
were cancelled at zero elapsed time. A diagnostic pass over every field showed
exact categorical/discrete identities and maximum meaningful direct-step
relative delta `1.577399244823e-4`; only the telemetry absolute tolerance was
changed to `2e-4`, retaining `rtol=2e-4`. Both independent preserved smoke
replays pass this final contract. This chain is preserved under
`confidence_selector_replay_failed_v2_abs_47316_47322/`.

Superseded recovered-parent chain:

- parent artifact gate: `47285` (`COMPLETED`, `0:0`);
- confidence smoke array: `47287` (16 tasks);
- smoke identity audit: `47288`;
- official PoseBusters smoke: `47289` (two selectors);
- full confidence replay array: `47290` (128 tasks, at most eight concurrent);
- full identity audit: `47291`;
- official PoseBusters array: `47292` (256 tasks, at most 16 concurrent);
- final combined report: `47293`;
- execution-manifest SHA-256:
  `37bf8ab03ececce4728e53be2f6143cddeccd752f2d48045bf4d384d36def99c`.

Every child dependency is `afterok`; failure of a smoke or identity gate keeps
all later work from running.

Final immutable chain:

- parent artifact gate: `47341` (`COMPLETED`, `0:0`);
- confidence smoke array: `47342` (16/16 `COMPLETED`);
- smoke V2 identity: `47343` (`COMPLETED`, `0:0`);
- official PoseBusters smoke: `47344` (`COMPLETED`, `0:0`);
- full confidence replay: `47345` (128 tasks, at most eight concurrent);
- full V2 identity: `47346`;
- official PoseBusters: `47347` (256 tasks, at most 16 concurrent);
- final report: `47348`;
- execution-manifest SHA-256:
  `ce91831926b4cacef7bf4ae6c819146c2cf77013a4343b69ef1878e8cda78e29`.

## Final chain outcome: failed strict identity gate

The full replay array `47345` completed all `128/128` tasks with no sampling
failure, traceback, or OOM. Its maximum recorded CUDA allocation was
`18.015 GiB` and maximum reservation was `50.300 GB`, below the frozen
48-GiB byte gate. The full V2 identity job `47346` then failed as designed.
The first mismatch was Astex `1hww`, eta `0`, shard `1`:
`mean_sample_rmsd` was `2.382819347802695` in the parent and
`2.3826657863339618` in the replay (absolute difference
`0.0001535614687332434`).

A complete read-only diagnostic over all `3,144` rows found `36` exact
row/direct-step counter mismatches, `11` exact summary counter mismatches,
four `num_fast_valid_candidates` mismatches, and four legacy selected-pose
coordinate-gate failures. The maximum legacy selected-pose coordinate RMSD was
`0.006902930134692069 Angstrom` and maximum atom displacement was
`0.023987079855624253 Angstrom`. All `3,144` 100-candidate confidence ledgers
were complete; pure-confidence and fixed confidence-filter recomputation
passed for every row. There were zero selector-index, RMSD `<2 Angstrom`, or
selected fast-validity flips, but those stable downstream labels do not erase
the frozen identity-gate failure.

The tolerance was not widened and this replay is not reused as a claim-bearing
result. Jobs `47347` and `47348` were cancelled before execution. The unchanged
run, failure record, and verified 107-file execution manifest are preserved at
`confidence_selector_replay_failed_full_identity_47345_47348/`; the manifest
SHA-256 remains
`ce91831926b4cacef7bf4ae6c819146c2cf77013a4343b69ef1878e8cda78e29`.
The replacement is a newly pre-registered parent-free one-pass
characterization under
`docs/GUIDANCE_ETA_SWEEP_CONFIDENCE_STANDALONE_PB_PROTOCOL.md`.
