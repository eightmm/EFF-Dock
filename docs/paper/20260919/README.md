# EFF-Dock manuscript evidence package — 2026-09-19

This package consolidates completed frozen-bank analyses; it does not change
weights, data splits, inference defaults or candidate selection. All numerical
tables are descriptive external characterizations of the retained checkpoint.
They are not new independent model-selection evidence.

## Reading order

1. [Results and captions](RESULTS.md): uniform unguided N100/S10 main table;
   all guidance/budget/postprocessing conditions; pocket/prior robustness.
2. [Candidate generation versus selection](CANDIDATES.md): Top-1/Top-5/oracle,
   near-native candidate density, raw/refined and all registered budgets.
3. [Training exposure](OVERLAP.md): actual preserved split, exact ligand/PDB
   exposure, nearest-training Morgan similarity, stratified performance.
4. [Runtime and measured memory](RUNTIME.md): summed shard wall cost per
   complex and amortized per generated/scored pose, measured refinement/confidence
   times, sampling CUDA allocator peaks; machine-readable JSON and CSV included.
5. [Manuscript-ready text](MANUSCRIPT_DRAFT.md): Methods/Results/Limitations
   paragraphs with the corresponding evidence and condition boundaries.
6. [Separate PoseX evaluation](POSEX.md): native SD/CD aggregation and relaxation.
7. [Verification and reproducibility](VERIFICATION.md): jobs, tests, sources,
   analysis recovery and remaining unavailable measurements.

The architecture, feature dimensions, equations and training details remain in
[the Methods index](../../methods/README.md) and
[the checkpoint-specific method record](../../PAPER_METHODS_CURRENT.md).
The public benchmark entry point is [BENCHMARK_RESULTS](../../BENCHMARK_RESULTS.md).

## Figures

Each figure has matching PNG and vector PDF files; captions are in its section.

| Figure | Question | PNG | PDF |
|---|---|---|---|
| 1 | What do refinement and chirality selection add? | [PNG](01_stage_ablation.png) | [PDF](01_stage_ablation.pdf) |
| 2 | Do guidance and pose/step allocation help? | [PNG](02_guidance_budget.png) | [PDF](02_guidance_budget.pdf) |
| 3 | How sensitive is the frozen guided protocol to crop, prior and center? | [PNG](03_pocket_prior_robustness.png) | [PDF](03_pocket_prior_robustness.pdf) |
| 4 | Is the bottleneck candidate generation or confidence selection? | [PNG](04_candidate_bottleneck.png) | [PDF](04_candidate_bottleneck.pdf) |
| 5 | What observed time and allocator memory did the budgets use? | [PNG](05_runtime_memory.png) | [PDF](05_runtime_memory.pdf) |
| 6 | How much training exposure exists and how do strata perform? | [PNG](06_training_exposure.png) | [PDF](06_training_exposure.pdf) |

## Claim boundaries that must survive manuscript editing

- Main table: unguided N100/S10, own refinement, tetrahedral-chirality selection.
  Ordinary Top-k in Figure 4 does **not** use that chirality filter.
- Guided eta2, N40/S25, and pocket/prior sweeps are separately labelled.
  The historical pocket/prior study uses guided sampling and no new chirality filter.
- Confidence is a within-bank ranking signal, not a calibrated affinity/RMSD guarantee.
- Train exposure is substantial, especially FoldBench. No global unseen-target,
  overlap-free or prospective-validation claim follows from these results.
- Overlap uses the verified executed 47,277-system fine-tuning loader index,
  not the union of all predecessor pretraining data. One invalid training SMILES remains unresolved. “No observed
  match” is limited to the parseable index, not certified absence of overlap.
- Exact PDB accession is not protein homology or pocket similarity. OpenBind
  PDB accession is missing. No validated external-to-training pocket/sequence
  nearest-neighbor matrix is supplied by this package.
- PhiBench includes three reconstructed systems; OpenBind includes quality
  flags and two noncovalent approximations of covalent systems.
- Three FoldBench PB shards use the disclosed InChI energy-reference repair.
- PoseX uses a distinct relaxation protocol, RMSD ≤2 Å and native grouping;
  CD success is over 109 groups, not a simple rate over 1,312 cases.
- Full-bank official PB labels are absent. No joint-PB Top-5/oracle or full-bank
  PB-valid density is inferred from chirality or fast geometry labels.
- Sample SD across three seeds is not a target-level confidence interval.
  Stratum differences are descriptive and composition-confounded.

## Reproduction

Run `scripts/slurm/paper_analysis.sbatch` on **cpu_only** with the existing
immutable banks present. The scripts regenerate aggregate JSON/CSV, Markdown,
and PNG/PDF figures under this directory. Source hashes and measured fields
are recorded in the JSON outputs. No GPU inference, training or publication
is performed. Per-case ligand annotations and candidate-case CSV remain local;
the release includes aggregate results, not third-party ligand metadata.

Without local banks, regenerate figures from the released aggregates with
`python scripts/paper_results.py --from-aggregate`,
`python scripts/paper_candidate_figures.py`,
`python scripts/paper_runtime_figures.py` and
`python scripts/paper_overlap_figures.py` in the project environment.
