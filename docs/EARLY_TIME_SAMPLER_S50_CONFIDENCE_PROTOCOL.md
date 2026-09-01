# S50 confidence score-only PLINDER protocol

Protocol ID: `EFFDOCK-S50-CONFIDENCE-SCORE-ONLY-PLINDER-V1`

Status: frozen before any confidence score is calculated for this candidate
bank. The smoke is integrity-only and may not open efficacy outcomes.

## Question and scope

On the already frozen S50 PLINDER bank, how much of the sampler's available
sub-2 A pose coverage can the retained confidence model recover when its
docking-feature backbone is either the deployed S50 sampler or the backbone
with which confidence was trained?

This is a score-only deployment diagnostic. It does not generate, refine,
filter, remove, or reorder candidates before scoring; retrain a model; or
authorize a new checkpoint. All 100 saved poses remain in their frozen order.

The same PLINDER validation identities have been used in historical model
development, and the S50 candidate labels were already opened for the sampler
decision. Consequently this report is a descriptive repeated-validation and
deployment-shift diagnostic, not an independent confirmation or an unbiased
generalization estimate. Any later confidence retraining or selector choice
must be developed on train-derived data disjoint from these identities and
confirmed on a separately frozen temporal holdout. External benchmark outcomes
remain closed and cannot be used to tune this protocol.

## Frozen source bank and cohort

Paths below are relative to the repository root unless shown as absolute.

- Candidate root:
  `outputs/benchmarks/early_time_sampler_plinder_k2_paired_runs/t0p10-continuation-plinder-k2-v1-pathfix1-20260815/full`.
- Source sampler strict report:
  `outputs/benchmarks/early_time_sampler_plinder_k2_paired_runs/t0p10-continuation-plinder-k2-v1-pathfix1-20260815/reports/full_strict_report.json`,
  SHA-256
  `d4814796a9d274f836888dd614e5b6a4a5fba6b86001da83bea6720fabf02316`.
- Source sampler protocol:
  `docs/EARLY_TIME_SAMPLER_PLINDER_K2_GATE_PROTOCOL.md`, SHA-256
  `0250853ae0793db288be2a6a8dc775db391d25aae32835b65b061782f34ab518`.
- Source coordinate audit V3:
  `outputs/benchmarks/early_time_sampler_plinder_k2_paired_runs/t0p10-continuation-plinder-k2-v1-pathfix1-20260815/reports/full_coordinate_audit.v3.json`,
  SHA-256
  `3b6daa4a3d4c74ae384e7c3d2199d3d26f9360fe4b64a33e1c6ab16f4b83eabc`.
- Eligibility manifest:
  `outputs/benchmarks/early_time_sampler_plinder_k2_paired_runs/t0p10-continuation-plinder-k2-v1-pathfix1-20260815/eligibility_manifest.json`,
  SHA-256
  `6ebeb2d165e1def6ebf7b5bba301f82d4a9c3ff9d6c5cd43616dcf09edbd38ac`.
- Validation split: `data/splits/plinder.json`, SHA-256
  `3ac570bf08bced053f1ce040b57efca27c3be616f29a82cd66ef887c08860e6b`.
- Inference/model config: `configs/train.yaml`, SHA-256
  `39aa62e4a48ed6f3aa4ff59345fb43a81220e2baba22edfd5beb0c4981b307ec`.

The fixed primary cohort contains 1,035 ligand samples from 1,020 PLINDER
systems. Each sample has exactly 100 S50 poses, for 103,500 poses per scoring
arm. The 41 eligibility failures from the complete 1,076-key validation split
are never scored or imputed in the primary cohort. They enter only the frozen
full-1,076 operational sensitivity as common zero-coverage samples.

The frozen eligible-key identity is SHA-256
`005577bbf2b0c1c1e98bac3092b8e5350a6aa06597442b4c86d05f24e763593f`
for lexicographically sorted UTF-8 keys, one key per line with a terminal
newline. The manifest's eligible-ID JSON identity is
`4b22461f2445971eca84d1b050221dca205cfad8a787dbf2774999d2b7ca7a38`.

### Exact S50 inventory

The authoritative inventory is the ordered row set in these source CSVs. A
scoring manifest must retain only an allowlist of label-free fields, but must
verify every CSV and every referenced SDF against the following pins.

| Shard | Samples | Source CSV | SHA-256 |
|---:|---:|---|---|
| 000 | 130 | `full/shard-000-of-008/arms/s50_ema/results.csv` | `17d737c1339b8385f2cff2767b6af2b13cefb13944b7fccfdf05eca8820abf98` |
| 001 | 130 | `full/shard-001-of-008/arms/s50_ema/results.csv` | `e8246381e55e9eb1258c722fe617adc6c1c45e424ee6d7907de6e93ef43640d9` |
| 002 | 130 | `full/shard-002-of-008/arms/s50_ema/results.csv` | `ed458d9616b13ce3f08c4429b56aa86d5b69d740eef674293e09680f61fb1184` |
| 003 | 129 | `full/shard-003-of-008/arms/s50_ema/results.csv` | `064a075fca38f0447b6ea1cc685b42b49a8dac972a0df150fd3fbb8f4a571301` |
| 004 | 129 | `full/shard-004-of-008/arms/s50_ema/results.csv` | `ef388ceea5c22f6f0c79342274622ab1622957322126aa985fb0faeb12e085fa` |
| 005 | 129 | `full/shard-005-of-008/arms/s50_ema/results.csv` | `7628fc782176f19ee6cc9a2cf573b40647e2a66cdd98ca3b1cf6774ee4c89cba` |
| 006 | 129 | `full/shard-006-of-008/arms/s50_ema/results.csv` | `e7da8b24be4eac979fb7066cfc6e78aa98d2581f73e7a976731a0961564ef5ca` |
| 007 | 129 | `full/shard-007-of-008/arms/s50_ema/results.csv` | `1079e508111095341705717c3f98f2fb528db1bd00c73a76f42ddf361c7918ef` |

Here `full/` is the absolute candidate root named above. The verified union is
1,035 unique SDFs, every SDF contains exactly 100 records, and the total SDF
payload is 299,361,693 bytes. Each actual SDF SHA-256 must equal its row's
`all_poses_sdf_sha256`; each row's candidate identity is additionally bound by
`candidate_ensemble_sha256`. The read-only S50-only inventory digest is
`d4e69ca40f7582f7ae0df1cb273c017b5b8ef5c0cdbaca0379b540e565ac22d8`,
formed from the byte prefix
`EFFDOCK_FULL_COORDINATE_AUDIT_SDF_INVENTORY_V1\0` followed in shard and CSV
row order by `s50_ema\0<sample_key>\0<all_poses_sdf_sha256>\0`. This digest is
a derived cross-check, not a substitute for the per-file hashes.

The bank was generated once with `sigma=2.0`, `N=100`, `S=10`, deterministic
ODE, `late` schedule with power 3, pocket cutoff 10 A, zero center jitter,
prior pool 100, ligand conformer seed 0, and no SDE, FK resampling, guidance,
confidence, or refinement. It is never regenerated for this protocol.

## Frozen model arms

Both arms use the same retained confidence checkpoint and differ only in the
docking model used to reconstruct the confidence model's time-one ligand
hidden features.

| Arm | Role | Docking backbone | SHA-256 |
|---|---|---|---|
| `s50_backbone` | primary deployment diagnostic | `outputs/eff-dock/early-time-t0p10-50k-v1-20260813/checkpoints/step50000_ema_common_init.pt` | `65be44d7dc8f0867eb9fc5d22214b80f93971ea4702679a527c665046e91e6b6` |
| `matched_backbone` | compatibility diagnostic only | `weights/effdock_geometry_ft_100k_best.pt` | `6932fb3ba6ebac770f714453529656a44b8f33cf15119d23c9e675d2d60b36db` |

The shared confidence checkpoint is
`weights/effdock_confidence_extmatch_n80_s25_step42500.pt`, SHA-256
`e31fde6f351284205c78f7a1510002779c43312e94d9f82003d47a14d72bc78f`.
It is the repository's retained/default deployment checkpoint. It was trained
at step 42,500 on an N80/S25/sigma0.5/pocket10 pose distribution with
`matched_backbone`. Its model card is `weights/CONFIDENCE_MODEL_CARD.md`.

The confidence runtime is not a standalone SDF scorer: it requires a docking
backbone to extract time-one ligand irreps and hidden features. Therefore the
`s50_backbone` arm answers the actual deployment question but deliberately
combines a new generator/backbone and sigma-2 bank with a confidence model
trained on the older backbone and sigma-0.5 bank. The `matched_backbone` arm
separates backbone-representation compatibility from that deployment shift;
it is diagnostic-only and cannot replace the primary arm post hoc.

## Label-free scoring boundary

Before GPU work, `freeze-inputs` writes one immutable, atomic, no-overwrite
manifest with schema `effdock.early_time_sampler_s50_confidence_bank.v1`. Its
SHA-256 is supplied independently to every task. The manifest allowlist
contains only identity, topology/conformer reconstruction, pocket center,
sigma, pose order, source path, and source hashes. It must not contain or expose
crystal/reference coordinates, pose RMSD, K2, success labels, fast-valid
labels, oracle fields, selector outcomes, or any efficacy aggregate.

The GPU scorer reads only that label-free manifest and its referenced SDFs. It
must not read the source result CSVs, source reports, crystal ligand files, or
reference labels. For every pose it records both confidence heads and immutable
join keys, including `sample_key`, `system_id`, arm, `sample_index`,
`candidate_ensemble_sha256`, and `all_poses_sdf_sha256`. Scores are sealed in
atomic, no-overwrite shard artifacts before any report joins labels.

Producer SDF records contain a `fast_valid` property, but the score stage is
software-restricted to an exact allowlist of identity, coordinate, seed, and
sigma properties and never requests or exports `fast_valid`. Pose RMSD is not
stored in those SDF records. Exact regression checks reject forbidden outcome
fields in both the label-free manifest and score artifacts. Thus possession of
the saved coordinate file does not authorize using its validity property during
smoke or full GPU scoring.

All scoring uses the frozen per-pose `sigma=2.0`; falling back to the package
default sigma 0.5 is a hard error. Saved SDF positions are absolute receptor
frame coordinates, while the confidence runtime expects pocket-centered
coordinates, so the scorer subtracts the exact manifest `pocket_center` before
feature extraction. The ligand graph is rebuilt from the frozen canonical
SMILES with conformer seed 0 and the saved coordinates assigned in verified
atom order. SDF-coordinate-derived stereochemistry must not replace the frozen
topology. The SDF serialization is quantized to 0.001 A (coordinate agreement
tolerance 0.0005 A), so these are explicitly saved-coordinate scores.

The ordered heavy-atom topology check normalizes only RDKit's V2000
explicit-versus-implicit attached-hydrogen representation by comparing total
attached H per atom. Atom order, element, formal charge, isotope, radical,
aromatic state, bond endpoints, and bond order remain exact. This preserves
protonation and constitutional fail-closed checks without treating an SDF
serialization detail as a molecule mutation.

Each arm uses pose chunks of exactly 20 on one RTX 6000 Ada GPU. Candidate
order is never changed. Any missing/non-finite score, candidate-count mismatch,
topology/order mismatch, input mutation, SHA mismatch, device mismatch, or
partial shard invalidates the stage.

## Frozen selectors

- Primary selector: stable ascending `confidence_rmsd`; ties retain ascending
  `sample_index`. Top-1 is the first ranked pose.
- Secondary selector: stable descending `confidence_success`; ties retain
  ascending `sample_index`.
- No cluster, density, energy, Vina, validity, physical, consensus, filter, or
  composite term is permitted.

The primary result is always the `s50_backbone` primary selector. The
success-head and `matched_backbone` results are secondary diagnostics.

## Smoke and full execution gates

The integrity smoke is the first eight lexicographically sorted eligible keys.
Because the frozen full run uses a strided eight-shard assignment, these are
also the first key in shards 000 through 007, respectively:

| Shard | Smoke key |
|---:|---|
| 000 | `1a5s__1__1.A__1.C__1.C` |
| 001 | `1a69__1__1.A_1.C__1.D_1.E__1.E` |
| 002 | `1bjo__1__1.A_1.B__1.C_1.D__1.D` |
| 003 | `1c29__1__1.A__1.C__1.C` |
| 004 | `1c39__1__1.B__1.F_1.H__1.F` |
| 005 | `1c9d__1__1.A__1.C__1.C` |
| 006 | `1cw2__1__2.A__2.C__2.C` |
| 007 | `1cx9__1__1.A__1.C__1.C` |

For both arms, each smoke key's identical saved coordinates are scored twice
in the same loaded eval-mode runtime. This detects within-runtime replay drift;
it is not a fresh-process or fresh-model-load reproducibility test. The stage
passes only with exact selected-index replay, maximum absolute per-head score
difference at most `1e-5`, exactly 100 poses per pass, zero missing/non-finite
scores, exact hashes, and no input mutation. The smoke report is
integrity-only: it may not join or print RMSD, K2, validity, oracle, or other
efficacy fields.

Only a sealed passing smoke report authorizes the full score stage. Full uses
exactly 8 shards and must produce exactly 1,035 samples and 103,500 poses per
arm, with zero failures, missing/non-finite values, duplicate/missing indices,
hash mismatches, or source mutation. Expected artifacts are
`<content-root>/<stage>/shard-XXX-of-008/arms/<arm>/scores.json`; each has
schema `effdock.early_time_sampler_s50_confidence_scores.v1`. The strict report
is written only after every full score artifact is sealed.

## Frozen reporting endpoints

The primary endpoint is sample-weighted Top-1 success below 2 A (`Top1 SR2`)
for the primary predicted-RMSD selector in `s50_backbone` over the fixed 1,035
samples.

Secondary endpoints are:

- oracle recovery among samples with `K2>=1`, oracle gap, the confidence-miss
  indicator `M` (oracle-solvable but selected pose is not below 2 A), the
  unreachable indicator `U` (`K2=0`), and mean `M-U` in percentage points;
- Top-3/5/10 success, deltas from the first saved candidate, and the exact
  analytic random-selector expectation `K2/100` averaged over samples;
- selected-pose RMSD mean, median, fractions below 1 A and 5 A, and regret
  from the per-sample oracle RMSD;
- fast-valid selected rate, joint sub-2-A-and-fast-valid success, fast-valid
  oracle recovery, and corresponding oracle gaps;
- success-head Top-1; reciprocal and median rank of the first correct pose;
  macro per-complex AUC where both classes exist; and frozen K2 strata;
- all corresponding `matched_backbone` diagnostics and the paired
  `s50_backbone - matched_backbone` Top1-SR2 difference;
- the full-1,076 common-zero operational sensitivity, labelled separately
  from the 1,035-sample primary estimand.

All paired uncertainty uses 20,000 percentile bootstrap draws from NumPy
`PCG64(seed=20260815)`, resampling `system_id` clusters and retaining every
ligand sample in a drawn system. Point estimates remain sample-weighted. The
report joins scores to frozen labels only by the complete key
`(sample_key, candidate_ensemble_sha256, all_poses_sdf_sha256, sample_index)`;
any incomplete, one-to-many, or hash-inconsistent join is fatal.

## Pre-registered interpretation bands

- Oracle recovery below 0.75 is a severe confidence bottleneck; `[0.75,0.90)`
  is material; at least 0.90 is near-oracle.
- For mean `M-U`, a 95% interval with lower bound above zero is
  confidence-dominant, an upper bound below zero is sampler-dominant, and an
  interval crossing zero is mixed.
- Confidence selection is called useful only if the 95% interval lower bounds
  for both Top1-SR2 deltas versus the first pose and versus analytic random
  selection are above zero.
- `Top5 - Top1` at least 5 percentage points is actionable reranking headroom;
  2 to less than 5 points is modest; less than 2 points is little.
- For paired Top1-SR2 `s50_backbone - matched_backbone`, a 95% interval wholly
  inside `[-2,+2]` percentage points is backbone-equivalent; upper bound below
  -2 points means S50 representation drift hurts; lower bound above +2 points
  means S50 helps; every other interval is inconclusive.

These bands diagnose whether confidence retraining should be prioritized after
the S50 sampler is frozen. They do not permit selector tuning on this report,
change the retained checkpoint, or reopen sampler training automatically.

## Execution identity, resources, and order

The executable contract is:

- scorer: `scripts/score_early_time_sampler_plinder_confidence.py`, SHA-256
  `193f35972076d87d413ee09d50053e55a44cde53e55c5abeac28ca686098d6e6`;
- reporter: `scripts/report_early_time_sampler_plinder_confidence.py`, SHA-256
  `98cf21c15633cd9f9d46d1b99fd62b899199a9ebd6ad8448a7d70da461cd0add`;
- score-runtime aggregate identity, SHA-256
  `47bddf89a08dcfd95a095a690f41f21a03c2b22a5ccd617e4c7e535f5062fae3`;
- launcher: `scripts/slurm/early_time_sampler_s50_confidence.sbatch`.

The protocol file cannot contain its own SHA-256 without a self-reference;
its externally computed final value is pinned in the launcher and the
label-free bank manifest.
The scorer and reporter hashes above are finalized before `freeze-inputs`.
The runtime aggregate is domain-separated by
`EFFDOCK_S50_CONFIDENCE_RUNTIME_CODE_V1\0` and covers the scorer, confidence
runtime/features/model, docking/preprocess/sampler paths, checkpoint/model/SE(3)
code, ligand/protein/fragment/graph preprocessing, dataset/benchmark helpers,
and `uv.lock`. The manifest carries the complete per-file identity map.

The label-free manifest SHA-256 content-addresses a fresh output root.
`freeze-inputs` alone verifies and seals the eligibility/source-protocol/source-
report/source-audit and eight source-CSV identities. GPU tasks trust those
identities only through the exact bank-manifest SHA-256: they do not open or
rehash any source CSV, source report, source audit, or eligibility artifact.
Every task rehashes the scoring protocol, scorer, reporter, runtime code,
config, checkpoints, and manifest before model load. Atomic task reservations
and no-overwrite outputs forbid reruns into a used root.

Execution order is fixed:

1. Run label-free `freeze-inputs`, record its manifest SHA-256, and derive the
   fresh content-addressed root.
2. Submit the 8-task smoke array, at most 4 tasks concurrently, and build the
   outcome-blind integrity report.
3. Only after that report passes and its SHA-256 is supplied, submit the exact
   8-task full array, again at most 4 tasks concurrently.
4. Seal the full score inventory, then and only then run the strict reporter
   that joins frozen labels and emits the registered endpoints.

Both GPU stages use partition `6000ada`, one RTX 6000 Ada GPU, 4 CPUs, 32 GiB
host RAM, a 3-hour task limit, and score chunk size 20. Each scoring task is
recorded by `oms research-runner` in a task-local ledger to avoid concurrent
ledger writes.

The closest accepted runtime evidence is Slurm job 53159: 78,600 pose scores
on RTX 6000 Ada with chunk size 20 completed in 10,206 aggregate task-seconds
(2.835 GPU-hours), with task mean 318.9 seconds and CPU MaxRSS about 2.58 GiB.
Linear pose throughput predicts about 3.73 GPU-hours for one 103,500-pose arm;
a conservative complex-overhead bound is 7.46 GPU-hours per arm. The two arms
therefore require roughly 7.5--15 GPU-hours in total; with 8 shards and `%4`,
the expected execution wall time is roughly 1.9--3.8 hours plus queueing if the
arms run sequentially in each task. Three hours per task is the frozen request,
not a guarantee. Score-only peak GPU memory was not recorded; only the 48-GB
RTX 6000 Ada device family is proven, so no lower VRAM claim is made.

No threshold, cohort, selector, arm role, join key, or bootstrap setting may be
changed after any full efficacy outcome is opened.
