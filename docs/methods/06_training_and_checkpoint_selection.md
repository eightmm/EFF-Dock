# Training data, optimization, and checkpoint selection

Evidence: [released model cards](../../weights/MANIFEST.md),
[public membership manifest](../../benchmarks/inputs/training_membership/README.md),
[50k docking configuration](../../configs/train_early_time_ft_50k.yaml), and
[confidence configuration](../../configs/train_confidence_s50_raw_refined_100k.yaml).

## 1. Data and split boundary

Training structures are PLINDER 2024-06/v2. The immutable identity is
`<system_id>__<ligand_instance_chain>`. The released docking fine-tune used the
preserved compatibility split (47,310 train / 1,076 validation identities
before filtering; 47,277 filtered training samples). This is checkpoint
provenance, not the split recommended for a future claim-bearing replacement.

The current strict split builder removes external canonical ligand-SMILES
matches and assigns validation by `pocket_fident__70__community`. It verifies
disjointness by identity, canonical SMILES, and pocket70 community. Reference
pose coordinates provide supervision and evaluation labels. Frozen
pocket definitions in retrospective redocking can be reference-ligand-derived;
the reference pose is not a model/scorer feature or refinement target.

The confidence checkpoint uses 43,092 eligible training samples and 1,035
validation samples. Docking and confidence share 43,067 training samples;
4,210 are docking-only and 25 are confidence-only. Both pipelines started from
the preserved 47,310 IDs and applied different input filters. Exact IDs,
exclusion reasons and set hashes are public in the linked membership manifest.
Relatedness figures use the docking set, not this confidence set or their union.
These are loader/split inventories, not a union of every predecessor checkpoint's
training exposure.

## 2. Released docking fine-tune

The released docking checkpoint starts from the retained geometry fine-tune
and performs 50,000 **fresh** AdamW updates. It uses four GPUs, batch 16 per
rank (global 64), learning rate `2e-5`, weight decay `0.01`, gradient clipping
`1.0`, and EMA decay `0.999`. The schedule has 1,000-update warmup, a peak
plateau through update 40,000, then cosine decay to `2e-6`. The released
artifact is the terminal EMA, not an external-benchmark-selected checkpoint.

Its only architectural/data intervention is the early-time/t=0 mixture stated
in `02_fragment_se3_flow.md`. Other base configuration values are six layers,
384 scalar hidden channels, 32 RBFs, `l_max=2`, 5-A dynamic contacts, and
dropout 0.1. The raw baseline configuration's 200k-step schedule is not the
50k fine-tune schedule and must not be cited as such.

## 3. U70k confidence continuation

U70k warm-starts from the terminal S50 symmetry-confidence checkpoint. Every
training complex provides a balanced candidate mixture of 32 raw production
(`sigma=2`, N100/S10) poses, 32 deterministic-refinement poses, and one
mapped crystal anchor. A loader may draw at most 64 train poses per complex;
the maximum pose-node product is 64,000, with a protective eight-pose cap for
very large graphs. Validation uses up to 100 poses. The crystal anchor is
excluded from validation selection.

The run is configured for 100k updates, Muon at `2e-4` plus `3e-6` learning
rate for the remaining optimizer path, weight decay `0.01`, clipping `1.0`,
2% warmup, 50% cosine cooldown, final-LR ratio 0.05, batch one complex per rank
(two ranks, effective global batch two), and seed 45. It saves a latest
checkpoint every 500 updates; `best` selection is
separate from recovery checkpoints.

## 4. Selection rule and release decision

The fixed internal selection bank contains 1,035 PLINDER complexes. The
selection metric is candidate-pool Top-1 symmetry-aware/no-alignment RMSD
below 2 A, after pure minimum-predicted-RMSD ranking. U70k scored
622/1,035 (60.10%); terminal U100k scored 617/1,035 (59.61%). U70k is therefore
the released confidence model.

Astex, PoseBusters, PhiBench, FoldBench, and OpenBind are not part of this
checkpoint choice. Astex and PoseBusters have been opened during development,
so their reported values are descriptive rather than independent model
selection. Refinement and raw candidates remain separately labelled evaluation
conditions even though both occur in confidence training.
