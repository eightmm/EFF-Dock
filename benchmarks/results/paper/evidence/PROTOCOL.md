# Saved-bank empirical evidence extension

Frozen 2026-09-23 before aggregating the new endpoints. No retraining, new
docking inference, tuned confidence selector or fitted calibration is included.
The primary condition is unguided N100/S10, refined, chirality-filtered U70k.
All five external cohorts retain their original IDs and three repeats.

- Confidence: within-complex Spearman correlation of predicted and observed
  RMSD; AUROC and average precision (step-area PR) using negative predicted
  RMSD for strict RMSD <2 Å labels. Exclude single-class complexes from both
  binary ranking metrics and report eligible n; constant ranks have undefined
  Spearman. Macro-average within repeat, then report repeat mean/sample SD.
- Selection: report raw/refined primary Top-1 regret relative to the full-bank
  minimum RMSD (median within repeat), and selected success conditional on at
  least one near-native candidate. Chirality-mask effects remain included.
- Density: fixed candidate-success bins 0, 1–5, 6–20, 21–50 and 51–100 of 100.
  Report conditional primary selected success and bank counts in every bin.
- Reliability: unfiltered candidates, success-head probabilities in ten equal
  bins; predicted RMSD bins [0,1,2,3,4,6,10,infinity). Show empirical outcome
  vs mean prediction. Brier score applies only to the saved success probability.
  Calibration is descriptive, not refitted; probability bins pool candidates
  over all three repeats and do not provide independent-candidate CIs.
- Relatedness: sequence identity <30% AND maximum training Morgan Tanimoto <0.5
  AND no observed exact training-ligand match. Thresholds are fixed regardless
  of surviving n. Show original/all and stringent n, Top-1, RMSD oracle and
  same-selected-pose PB-valid success. Empty strata are explicitly empty.
  Identity is the existing query-normalized binding-chain metric; reference
  eligibility/unknown-residue limits carry over. This is not certified absence
  of all predecessor-checkpoint exposure or an independently held-out dataset.
- Refinement: all 27 non-RMSD official PB checks, plus interpretable groups
  (bond geometry, internal clash, receptor clash, stereochemistry). Report
  selected-pose failure rates and pass/fail transitions. Also match candidate
  indices where saved selected-pose evaluations exist at both stages; this
  conditional subset isolates pose correspondence but is selection-biased.
  It must not be extrapolated to the full 100-pose bank.
- Structure examples: Astex repeat 0, primary selection. Successful refined
  cases use the median selected RMSD among PB-valid successes; rescue uses
  median raw-to-refined same-index RMSD improvement among cases crossing 2 Å
  and ending PB-valid; selection failure uses median selected-minus-oracle
  regret where the refined bank contains a near-native candidate. IDs break
  ties. Crystal pose, selected pose and relevant comparator share the receptor
  frame without independent ligand alignment. Examples are illustrative.

All external results are descriptive. Expected patterns (positive rank
correlation, remaining selection loss, reduced geometry/clash failure after
refinement) are hypotheses, not inclusion gates; null/negative effects remain.
Hash-check source ledgers and score CSVs, cross-check selected outcomes against
the published main table, and abort on missing IDs, nonfinite scores or drift.
Use CPU jobs (1 CPU, 8 GiB, at most 30 minutes per stage); no new CI workflow.

New unguided pocket robustness is deferred at the user's request. The existing
guided pocket/prior panel remains labelled as a diagnostic and cannot establish
robustness of the primary unguided method.

## Secondary baseline and cost summaries

For the five locally executed baseline rows already present in Figure 1,
retain their native budgets/selectors and all 85 Astex or 308 PoseBusters IDs.
Pair methods by complex, average the three repeat outcomes within each complex,
and bootstrap these paired differences 2,000 times (seed 20260923), separately
by complex and exact PDB accession, with complex-weighted means. Report RMSD
and PB-valid success differences in percentage points with percentile 95% CIs.
These are descriptive, unadjusted intervals, not equal-compute causal effects
or protein-family-level resampling. Preserve negative and inconclusive effects.

Count parameter objects in the released models instantiated on CPU, excluding
non-parameter buffers. Derive amortized pose throughput from existing mean
pipeline timing only. Do not infer training wall time from checkpoint steps.

### Pocket-only example and rescue selection amendment (2026-09-23)

At the user's request, remove the full-protein overview row from page19/S8.
Keep the original successful and selection-failure examples. Replace the rescue
with the largest absolute same-candidate raw-to-refined RMSD decrease among
primary refined selections across all five external datasets and three repeats,
requiring raw RMSD >=2 A, refined RMSD <2 A and refined PB validity. Break ties
by dataset, repeat and complex ID. This post-hoc extreme illustration is not a
representative effect or a prevalence estimate. Freeze this rule before the
expanded search; export eligible candidates and source-ledger hashes for audit.
No new inference or PB evaluation; reuse saved coordinates and labels. Use
pocket-only cartoons, element-colored ligands and existing CPU render budget.

### Dataset-specific structure triplets (2026-09-23)

Expand S8 to all five external datasets, three examples per dataset. Use all
three repeats of the frozen primary selected poses. Within each dataset select
maximum same-index raw-to-refined RMSD improvement crossing 2 A and ending
PB-valid; then the lower-median selected RMSD among PB-valid successes excluding
that complex ID; then lower-median regret among RMSD selection failures with an
oracle below 2 A, excluding the two already chosen complex IDs. Ties: repeat, ID.
These are post-hoc illustrative examples; rescue is extreme, not representative
of typical improvement. Do not substitute another dataset if eligibility is
empty, or change selection based on visual appearance. Record eligible counts
and complete IDs in the report. One 5-row by 3-column pocket-only figure, with
ID/RMSD inside each image. No new inference/PB evaluation; CPU-only 2 CPUs/8G,
10min collection/capture. Preserve other PDF pages and all aggregate metrics.
