# EFF-Dock guidance evaluation follow-up plan

## Current completed evidence

- The eta=2, sigma={0.5,1,2,3,4} N100/S10 Astex/PoseBusters sampling
  grid is complete and passed its sampling audit.
- The frozen primary confidence Top-1 from every cell has now been evaluated
  with official PoseBusters 0.6.5 `redock` checks.
- Primary validity is pass-all-27 non-RMSD checks. A SigmaDock-list-compatible
  26-check value is recorded separately; it is not used as the EFF-Dock
  headline.
- No PoseBusters outcome was used to choose a pose or tune a selector.

## Ordered follow-up

1. Finish the already-submitted eta={2.5,3.0}, sigma=0.5 sampling extension.
   Admit it only if smoke, full sampling, and exact-cohort audits all pass.
2. Build one primary-confidence official-PB eta table by combining the frozen
   eta={0,0.5,1,1.5,2.0} run with the admitted eta={2.5,3.0} extension. Reuse
   the same PoseBusters runtime, redock config, 27-check conjunction, cohorts,
   hash binding, and joint RMSD<2 definition used by the sigma study.
3. Record guidance cap telemetry beside the eta table: any-cap fraction,
   translation/angular/displacement cap fractions, applied/model drift ratio,
   and non-finite/backtrack counts. These are mechanism diagnostics, not
   selector inputs.
4. Run the chosen small set of eta/sigma candidates on the held-out PLINDER
   validation cohort before adopting a default. Astex and PoseBusters remain
   external descriptive benchmarks and do not choose production settings.
5. Re-run Astex/PoseBusters once with the validation-chosen setting and freeze
   that as the reportable test result. Do not continue tuning on these test
   outcomes.

## Output retention

- Retain current sampling SDFs, confidence-selected SDFs, reference files,
  sampling audits, official PB shard CSV/JSON, aggregate JSON/Markdown, and
  exact execution/experiment ledger entries.
- Small unreferenced dry/smoke/debug directories may be moved into a dated
  recoverable archive. No output is deleted by this policy.
- Large training or smoke-training directories remain review-only until the
  final docking and confidence checkpoint lineage is frozen. Moving them may
  hide the only optimizer/checkpoint state even when their directory names
  look temporary.
