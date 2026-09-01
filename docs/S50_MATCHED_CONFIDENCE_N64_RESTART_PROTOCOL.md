# S50 matched confidence N64 restart

## Decision

The failed N80 run (Slurm 55336) is preserved. It reached U1960 but failed on
a genuine per-complex CUDA capacity OOM before its first U2500 checkpoint, so
none of those updates are resumable.

The replacement warm-starts the retained confidence step-42500 weights with a
fresh optimizer and scheduler. The sealed S50 N100 train/validation bank,
split, loss, four-rank global complex batch, learning rates, and 50,000-update
schedule remain fixed. The only training-data change is stratified sampling of
64 instead of 80 poses per complex. Validation continues to use all 100 poses.

`latest.pt` is atomically published every 500 updates and at the terminal
update. `best.pt` remains selected only at the frozen 5,000-update full
validation looks. The N64 outputs use new no-overwrite directories and never
reuse the failed N80 training directory.

## Execution gate

A four-GPU heavy-partition U0/one-update smoke using N64 and N100 validation
must complete before the 50k job may start. If N64 still exceeds the 80-GiB
H100 capacity, the full job remains dependency-blocked and a new protocol is
required; it does not silently lower the pose count.

The first mechanical submission (56288) failed before loading a training item:
it passed the full-bank manifest together with the trainer's smoke-bank schema
flag. The corrected retry reads the sealed full bank with explicit train/val
limits but does not mislabel it as a smoke bank. The failed output is preserved.
