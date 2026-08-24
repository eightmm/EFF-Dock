# S50 raw + refined confidence continuation protocol

Protocol ID: `EFFDOCK-S50-RAW-REFINED-CONFIDENCE-FINETUNE-V2`.

## Objective

Adapt the completed S50 symmetry-confidence model to rank both its original
sigma-2 candidates and the deterministic step-100 refined candidates without
forgetting either domain. The existing U50,000 `latest.pt` and `best.pt` are
immutable inputs; every new checkpoint is written to a distinct output root.

## Frozen data contract

- Complex inventory: the same frozen 43,092 train and 1,035 validation IDs.
- Raw source: sealed S50 N100/S10/sigma-2 deterministic-ODE bank.
- Refined source: the complete V2 geometry-only unified-guidance transform of
  those exact 100 poses. No RMSD or confidence outcome affects membership.
- Labels: RDKit `CalcRMS` symmetry-aware heavy-atom RMSD without alignment,
  recomputed and sealed independently for each raw and refined pose.
- Crystal anchor: one exact reference pose in frozen input atom order, with a
  separately extracted S50 t=1 hidden representation and exact zero RMSD/atom
  displacement. It is an in-complex anchor, not another complex or split row.
- Train item: one complex containing 32 stratified raw poses, 32 stratified
  refined poses, and one crystal anchor. This preserves equal complex weight.
- Validation: all 100 refined poses only. Crystal anchors are never included in
  validation selection metrics.

The loader fails closed on ID/system mismatch, graph or topology drift,
non-finite tensors, unequal raw/refined candidate counts, changed bank/label
hashes, or a non-zero crystal-anchor label.

## Optimization and checkpoints

- Initialization: weights-only from the completed S50 symmetry-confidence
  U50,000 `latest.pt`; optimizer and scheduler start fresh.
- Updates: 10,000, four GPUs, one complex per rank, N64 candidates plus one
  anchor, AdamW/Muon with `lr=3e-6` and `muon_lr=2e-4`.
- Loss contract is unchanged from the completed symmetry-confidence run.
- Validation at U0 and every 2,000 updates; `latest.pt` every 500 updates and at
  the terminal update; `best.pt` selects refined-bank symmetry Top-1 <2A.
- A one-update four-GPU smoke must verify the exact mixed item, forward,
  backward, evaluation ledger, and atomic `latest.pt`/`best.pt` before full
  training starts.

This is a continuation on repeated PLINDER identities, not an independent
generalization claim. External Astex/PoseBusters evaluation remains a separate
post-freeze decision.
