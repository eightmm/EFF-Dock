# EFF-Dock confidence selection study

Protocol ID: `EFFDOCK-CONFIDENCE-SELECTION-V1`

Pre-registered: 2026-07-19, before launching the new fine-tuning runs.

## Question

Can a pose-set-level training objective improve selection from the retained
step-42500 confidence checkpoint without changing the docking generator,
candidate distribution, model architecture, or frozen selector?

## Data and information boundary

- Prediction unit: one PLINDER protein/ligand/pocket complex with 80 generated
  poses.
- Inputs: receptor and ligand graph features, generated pose coordinates, and
  the declared pocket center. Labels are symmetry-aware pose/atom displacement
  targets in Angstrom.
- Train data: 4,096 matched
  `conf_ligonly_extmatch_n80_s25_sig0p5_pc10` complexes.
- Screen validation: the first 512 complexes in the frozen PLINDER validation
  split, evaluated on all 80 poses with no stochastic crop.
- Split audit: 47,310 train and 1,076 validation sample keys; exact sample-key,
  canonical-SMILES, and pocket70-community overlaps are all zero.
- External PoseBusters, Astex, and CASF labels/results are forbidden for model,
  loss, checkpoint, or hyperparameter selection.

## Frozen baseline

- Confidence initialization:
  `weights/effdock_confidence_extmatch_n80_s25_step42500.pt`.
- Matched docking checkpoint:
  `weights/effdock_geometry_ft_100k_best.pt`.
- Historical matched-val256 success-head top-1 <2A: 49.61%.
- Historical 500-step hard-pair fine-tuning fell to 46.88%; that intervention
  is abandoned and will not be repeated.

## Stage-1 falsification screen

All runs use the same initialization, seed 43, data order, fresh optimizer,
1,500 updates, schedule, batch size, validation rows, and evaluation cadence.
Only the declared loss intervention changes.

| Variant | Independent change | Prediction |
|---|---|---|
| `control` | no loss change | estimates ordinary low-LR continuation drift |
| `setwise` | add setwise-success weight 1.0 | concentrate categorical mass on any <2A pose |
| `pairwise` | add all success/failure pairwise weight 0.5 | improve positive-vs-negative ordering |
| `rmsd_listwise` | add RMSD-listwise weight 0.5 | improve continuous pose ranking |

Primary metric is PLINDER val512 success-head selected RMSD <2A. A variant
passes the screen only if its best checkpoint:

1. improves by at least 1.5 percentage points over its step-42500 initial
   score;
2. exceeds the matched control run;
3. does not reduce frozen-composite val512 <2A by more than 0.5 points; and
4. completes with finite losses and zero missing validation complexes.

Failure to meet the bar abandons that loss at the tested weight. A passing
variant must next be repeated across seeds and confirmed on the full 1,076-case
validation set before any external benchmark is opened. External success is
not guaranteed by a validation gain.

## Execution and artifacts

- Launcher: `scripts/slurm/confidence_selection_sweep.sbatch`.
- Outputs: `outputs/eff-dock/confidence-selection-v1/<variant>/`.
- Each Slurm task records its question, change, command, checkpoint metrics,
  and exit status through `oms research-runner`.
- Existing data, checkpoints, and historical runs are preserved; no benchmark
  result is overwritten.

## Stage-1 outcome

Completed 2026-07-20. The shared step-42500 val512 baseline was
success-head `58.59%`, pure RMSD-head `58.40%`, frozen-composite `57.03%`, and
oracle-80 `77.93%`. No fine-tuned checkpoint exceeded the baseline primary
metric, so every run retained step 42500 as its best checkpoint.

| Variant | Best step | Best success <2A | Final success <2A | Final frozen <2A | Primary delta |
|---|---:|---:|---:|---:|---:|
| `control` | 42500 | 58.59 | 57.81 | 56.45 | -0.78 pp |
| `setwise` | 42500 | 58.59 | 58.40 | 57.62 | -0.20 pp |
| `pairwise` | 42500 | 58.59 | 58.20 | 57.03 | -0.39 pp |
| `rmsd_listwise` | 42500 | 58.59 | 57.62 | 57.81 | -0.98 pp |

The three first submissions on 48 GB RTX 6000 Ada GPUs completed the initial
validation but OOMed on the first backward pass. These are operational
failures, not scientific outcomes. Exact reruns on an 80 GB H100 completed
successfully without changing the 80-pose contract.

The pre-registered `+1.5 pp` success bar was not met. Multi-seed/full-val and
external PoseBusters/Astex/CASF evaluation were therefore not run. Keep
`weights/effdock_confidence_extmatch_n80_s25_step42500.pt`; abandon these three
loss additions at the tested weights. The next falsifying experiment should
change data coverage or representation rather than perform another short
loss-only fine-tune on the same 4,096 complexes.

Completed Slurm tasks were control `38824`, setwise `38831`, pairwise `38880`,
and RMSD-listwise `38830`. Machine metrics are under
`outputs/eff-dock/confidence-selection-v1/<variant>/metrics.json`.
