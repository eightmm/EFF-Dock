# S50-matched confidence training protocol

Protocol ID: `EFFDOCK-S50-MATCHED-CONFIDENCE-TRAIN-VAL-V1`

Status: pre-registered before pose generation or confidence updating. A run is
valid only when the final protocol, builder, trainer, config, and launcher
identities are sealed after parent review in one read-only execution capsule
and content-addressed output root. This file does not authorize changing the
retained sampler.

## Question, hypothesis, and one intended change

Question: with the retained sampler fixed, does adapting the existing
confidence model to poses produced by that sampler improve recovery of the
sampler's sub-2-A candidates?

Hypothesis: the existing confidence model is useful but distribution-shifted
because it was trained on N80/S25/sigma-0.5 candidates from the older docking
backbone. Training it on S50/N100/S10/sigma-2 candidates will improve pure
predicted-RMSD Top-1 success on the fixed PLINDER validation bank.

Prediction: the best scheduled checkpoint will improve validation Top-1
success below 2 A by at least 3 percentage points over its own pre-update U0
warm-start score, with the paired 95% confidence interval lower bound above
zero.

The intended independent variable is the confidence-training pose
distribution. The S50 sampler checkpoint, sampler settings, confidence
architecture, confidence loss, crop, and selector formula are fixed. This is
confidence-only adaptation; it does not continue docking-model training.

## Claim boundary and split

The exact existing split is used unchanged:

- `data/splits/plinder.json`, SHA-256
  `3ac570bf08bced053f1ce040b57efca27c3be616f29a82cd66ef887c08860e6b`;
- 47,310 declared train sample keys and 1,076 declared validation sample keys;
- sample key is the immutable prediction-unit ID; PLINDER `system_id` is the
  grouping key for uncertainty;
- duplicate keys or train/validation overlap are fatal.

The split was deliberately constructed as the project's maximally separated
PLINDER split and is not rewritten or externally filtered. Every declared key
is attempted and accounted for. An aggregate filtered split may omit only a
record with a sealed, outcome-independent internal preprocessing or atom-map
failure. It must retain the original attempted denominators and the exact
reason for every omission. Benchmark identity, benchmark membership, and
benchmark outcome are never eligibility features.

These validation identities have been used repeatedly in model development,
and their S50 outcomes have already been opened. Validation can select a
checkpoint for this internal adaptation but is not an independent
generalization claim. Later Astex Diverse and PoseBusters checks must use one
checkpoint frozen here and cannot tune the loss, step, bank, or selector.

## Frozen inputs and information boundary

### Models and canonical ligand input

- Retained S50 EMA sampler:
  `outputs/eff-dock/early-time-t0p10-50k-v1-20260813/checkpoints/step50000_ema_common_init.pt`,
  SHA-256
  `65be44d7dc8f0867eb9fc5d22214b80f93971ea4702679a527c665046e91e6b6`.
- Confidence warm start:
  `weights/effdock_confidence_extmatch_n80_s25_step42500.pt`, SHA-256
  `e31fde6f351284205c78f7a1510002779c43312e94d9f82003d47a14d72bc78f`.
- Docking/inference config: `configs/train.yaml`, SHA-256
  `39aa62e4a48ed6f3aa4ff59345fb43a81220e2baba22edfd5beb0c4981b307ec`.
- Canonical ligand table: `data/plinder_pool.parquet`, SHA-256
  `0ff455da77ce5540b839918cccb96f45414e91efff6272d7da3a65337ab1fe91`.
- Processed PLINDER root: `data/plinder_processed`.

Train inference starts from `ligand_rdkit_canonical_smiles` with production
ETKDGv3/MMFF conformer seed 0. It uses the processed protein graph and frozen
pocket metadata. The processed crystal ligand is never the model's initial
ligand pose; it is read only to freeze the heavy-atom correspondence and
produce reference labels. Raw PLINDER receptor/ligand files are not required
for train generation.

The existing validation coordinates are reused, not regenerated:

- label-free S50 bank:
  `outputs/benchmarks/early_time_sampler_s50_confidence_runs/frozen_inputs/label_free_bank.v2.json`,
  SHA-256
  `928b7219ed1ef8375c1ee52470f6ef606b8fca4d5bf4ea5c51355e8332e29a4b`;
- exactly 1,035 eligible samples with 100 ordered poses each;
- the same 41 records already frozen as ineligible remain declared against
  the full 1,076-key operational denominator and are never imputed.

Pocket centers are crystal-defined processed PLINDER centers, so this is
known-pocket redocking. The confidence input contains protein/ligand graph
features and candidate coordinates, but no reference coordinates, RMSD,
validity, oracle count, or benchmark outcome.

## Pose-bank contract

### Generation settings

Every eligible train record is generated once with:

- S50 EMA checkpoint above, in eval mode;
- deterministic ODE, `N=100`, `S=10`, `sigma=2.0`;
- `late` time schedule with power 3;
- pocket cutoff 10 A, center jitter 0;
- 100 initial priors and the manifest-recorded deterministic sampling seed;
- canonical-SMILES conformer seed 0;
- guidance, FK resampling, SDE, confidence selection, filtering, and
  post-refinement disabled.

The pose tag is
`s50_n100_s10_sig2_latep3_pc10_rdkitseed0`. Candidate order is immutable.
Each shard stores all 100 poses and the S50 time-one hidden features required
by confidence. Training later draws at most 80 poses per complex with the
frozen stratified strategy; the bank itself is not truncated.

Validation feature extraction reads the 1,035 exact saved SDF ensembles and
verifies their file and candidate hashes, topology, atom order, seed, and pose
count before constructing a shard. It uses all 100 poses during evaluation.

### Labels

Train supervision is a fixed-map, receptor-frame, no-alignment heavy-atom
target. The canonical-SMILES-to-crystal atom correspondence is frozen before
candidate outcomes are inspected and is identical for all 100 candidates of a
sample. Atomwise displacement labels and pose RMSD labels follow that map.

Validation additionally stores
`pose_rmsd_symmetry_no_align`: symmetry-aware, no-alignment heavy-atom RMSD in
the receptor frame. This target is evaluation-only. The model still ranks by
its predicted RMSD; the symmetry target never enters a selector score.

### Inventory and failure policy

`freeze-inputs` must first seal one input manifest that accounts for all
47,310 train and 1,076 validation keys. It pins the split, pool, processed
inputs, val bank, S50 checkpoint, inference config, builder, runtime files,
and their hashes. Missing or duplicate input rows, mutable paths, hash drift,
no stereo-compatible full map, truncated symmetry enumeration, incompatible
atom order, non-finite data, or undeclared val eligibility state fail the
preflight. Complete chemical symmetries are resolved by the frozen
minimum-fragment-floor rule: every map within an absolute `1e-12 A` window of
the minimum is tied, then the lexicographically smallest mapping tuple wins.
Atom indices and every discrete mapping field remain exact across freeze,
generation, and aggregation. Only the diagnostic
`rigid_fragment_floor_rmsd` and `pair_distance_rmse` floats may vary across CPU
implementations; both values must be finite and match with relative and
absolute tolerance `1e-12`.

Generation writes one atomic, no-overwrite `.pt` artifact per eligible sample
and one sealed summary per shard. A task reservation prevents two Slurm tasks
from claiming the same `(stage, split, shard)` tuple. A failed or partial shard
cannot be treated as complete.

The full aggregate succeeds only if every original split key has exactly one
terminal state: complete artifact or declared internal ineligibility. It
rehashes every artifact, verifies 100 poses, required fields, finite values,
map identity, split ownership, and uniqueness, then writes:

- a content-addressed bank manifest whose records contain authoritative
  artifact paths and hashes;
- a derived loader split containing only complete internal records;
- attempted, complete, and ineligible counts for both original splits.

The aggregate is the sole authority for training. Ad-hoc globbing, silent
skip, overwrite, imputation, or a manually edited split is forbidden.

## Confidence training contract

Config: `configs/train_confidence_s50_matched.yaml`.

The architecture and loss are copied exactly from
`configs/train_confidence.yaml`: atom regression/BCE weights `0.2/0.2`, pose
regression/BCE weights `0.3/0.4`, rank weight `0.1`, success-listwise weight
`1.0`, and all other listwise/setwise/pairwise additions disabled.

Training is frozen to:

- load confidence step 42,500 as model weights only;
- fresh optimizer, scheduler, RNG stream, and update counter U0;
- evaluate the complete eligible validation bank at U0 before any update;
- 50,000 optimizer updates, seed 43, effective global batch 4;
- Muon enabled, Adam-like LR `3e-5`, Muon LR `2e-3`, weight decay `0.01`;
- the original warmup-stable-cosine ratios `0.02/0.5/0.05` and gradient-norm
  cap `1.0`;
- all complete train complexes, at most 80 stratified poses per complex;
- all complete validation complexes and all 100 poses per complex;
- full validation at U0 and every 5,000 updates; checkpoints every 2,500.

Four-rank DDP uses one complex per rank and accumulation 1, for effective
global batch 4. No single-GPU fallback is admitted because it has not passed
the matching smoke and memory contract. Resume is permitted only from the exact
content root's `latest.pt` with matching config, bank-manifest, code, and world
size/accumulation provenance. It is not a second initialization run.

## Metric, checkpoint selection, and admission gate

For each validation complex, the primary selector is a stable ascending sort
of predicted pose RMSD; exact ties retain original pose index. The primary
endpoint is sample-weighted `eval_top1_lt2`, the percentage whose selected pose
has `pose_rmsd_symmetry_no_align < 2.0 A`, over the full eligible validation
bank. U0 is the baseline. The best scheduled checkpoint maximizes this metric;
no success-head, composite, cluster, density, validity, or energy score may
choose a checkpoint.

The evaluation ledger seals per-ID selected index, Top-1 outcome, Top-5
outcome, oracle RMSD, and oracle K2 at every scheduled evaluation. The paired
Top-1 difference from U0 uses 20,000 percentile-bootstrap draws from NumPy
`PCG64(seed=20260816)`, resampling PLINDER `system_id` clusters and retaining
all ligand samples in each drawn system.

The new confidence checkpoint is admitted for later frozen external checking
only if both conditions hold:

1. absolute paired `eval_top1_lt2` improvement is at least `+3.0` percentage
   points versus U0; and
2. the paired 95% interval lower bound is strictly above zero.

Otherwise the retained confidence checkpoint remains the deployment default
and the null/negative result is recorded. Secondary diagnostics include
`eval_top5_lt2`, oracle coverage, oracle recovery, K2 strata, selected RMSD,
and training losses. The final report seals selected-RMSD summaries and
Top-1 recovery within K2 `0`, `1-4`, `5-9`, and `>=10` strata at U0 and every
scheduled look. They cannot override the primary gate.

Because the same validation set both selects the step and applies this gate,
passing is an internal repeated-validation decision, not independent evidence.

## Execution order and resources

All stages use the same read-only execution capsule and content-addressed run
root. No stage may reuse a root created by another submission. The content
identity includes a deterministic runtime fingerprint for the linked `.venv`:
Python, PyTorch, PyTorch CUDA/cuDNN build, RDKit, and cuEquivariance package
versions plus the resolved environment identity. Every Slurm stage recomputes
that fingerprint and fails before work if it differs from the sealed artifact.

1. CPU preflight freezes the complete input manifest and split accounting.
   Process workers equal allocated CPUs while OMP, MKL, and OpenBLAS threads
   are fixed to one per worker.
2. Exact-budget smoke generation produces two train and two reused-val
   records with N100/S10/sigma-2 settings; smoke aggregation and a U0/one-update
   confidence smoke must pass on the same four-rank `heavy` topology as the
   full run, using 80 train poses and all 100 validation poses per complex.
3. Full train-bank generation runs as `0-127%8` on `6000ada`, one RTX 6000 Ada
   GPU per task. Full validation feature extraction runs as `0-7%8` under the
   same device contract. Each task requests 4 CPUs, 32 GiB, and a conservative
   23:59 limit within the account's `normal` QOS.
4. A CPU aggregate runs only after both full arrays finish successfully and
   seals the bank manifest and derived loader split.
5. The confidence run uses the `heavy` partition's generic four-GPU
   allocation (`1 x H100` and `3 x 6000pro_maxq` in the verified cluster
   inventory), one node, four DDP ranks, and the admitted `long` QOS with a
   2-day 23:59 limit. No alternative training topology is part of this frozen
   run. The four-GPU smoke overrides this to the `normal` QOS and four hours.
6. The paired U0 gate report runs only after training finishes and requires all
   scheduled ledgers and hashes to be complete.

Slurm dependencies are `afterok`; any preflight, smoke, shard, aggregate,
training, or report failure stops its descendants. Each command is recorded
in a task-local OMS ledger. Submission records the capsule identity, content
identity, Slurm job IDs, model/data/config hashes, seed, resources, and logs.
