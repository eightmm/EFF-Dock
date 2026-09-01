# S50 symmetry-confidence submission — 2026-08-20

- Replaced fixed-map pose-target control job `56312` after 1:50:52 elapsed.
  It was cancelled by the owner; its logs and checkpoints remain intact.
- Symmetry-target four-GPU smoke: job `56495`, completed in 2:00 with exit 0.
- Symmetry-target 50k training: job `56496`, submitted with
  `afterok:56495` and started on `gpu1` after the smoke passed.
- Execution identity:
  `88509b2189ab619c91feee843e4d90388b85b36b181532ca0a54ebf271f69b97`.
- Train target manifest SHA-256:
  `89ea40b02b121387228e3d47461a68a30bd749143b6281fe4d5ea9ab89056981`.
- The smoke log confirmed
  `training_target=pose_rmsd_symmetry_no_align`, 64 train poses, 100 val
  poses, four DDP ranks, finite U0/U1 metrics, and both `best.pt` and
  `latest.pt` publication.
- The full job uses all 43,092 train and 1,035 validation complexes, starts
  from retained confidence step 42,500 weights with a fresh optimizer and
  scheduler, evaluates every 5,000 updates, and publishes `latest.pt` every
  500 updates.

## U1960 recovery

- Job `56496` and bounded recovery jobs `56630`/`56632` all stopped making
  progress immediately after U1960. In each case rank 0 retained roughly
  94--97 GiB while the other ranks spun in a collective. The last complete
  atomic checkpoint was preserved as U1500 with SHA-256
  `2af26bf66bec53676b8344e811911bbf47ee85aa6550610f35c3812b7a7f9d15`.
- The next rank-0 sample was deterministically identified as
  `8qfe__1__2.A_3.A_4.A_5.A_6.A__2.B_3.B_4.B_5.B_6.B__2.B`: its saved graph
  has 1,892 nodes and 21,326 edges, versus roughly 284--345 nodes for the
  other ranks in that update.
- Exact single-GPU forward/backward probes in job `56637` measured peak CUDA
  reserved memory of 3,966, 7,678, 15,326, and 31,134 MiB for respectively
  1, 2, 4, and 8 poses of this complex.
- Recovery job `56642` therefore retains 64 poses for normal complexes but
  caps training graphs above 1,200 nodes at 8 poses. The existing
  `nodes * poses <= 64,000` guard remains active, and rank-0 evaluation now
  releases the CUDA cache before training resumes. The loss, symmetry target,
  global complex batch, optimizer, scheduler, and validation contract are
  unchanged.
- Recovery runtime identity:
  `e0484955d44159ea0ed0b5b8a53f7853ee7a09a23bf34319616e7ea21338a9eb`.
- Job `56642` crossed the former stall (`U1960 -> U1980`) and atomically
  published U2000 `latest.pt` with SHA-256
  `84bbe4366c129b2f5972a9a5efa3e2fcd7a002c48519663a136b92e75c3dbda2`.
  Immediately afterward all four GPUs were active at 100% utilization and
  used 45--54 GiB, rather than leaving rank 0 idle at 94--97 GiB.

## Final completion

- Recovery job `56642` completed all 50,000 updates and all eleven registered
  full-validation looks. The sealed final metrics report U25k as the internal
  best checkpoint and U50k as the terminal latest checkpoint.
- U25k `best.pt` SHA-256:
  `1c59034172fb925cc8a70777dcba236be349f1a1de1775d49cc17d492b17c030`.
- U50k `latest.pt` SHA-256:
  `fd49fa86f67187bf26d6c1bcf2daf925ba3e3b19dfeae733e57535d183280469`.
- Final `metrics.json` SHA-256:
  `5fab2a3dce00332b4d46858e16d1586d37f86cb6b04adf03c006bb9997c015f7`.
- The primary symmetry-aware Top-1 `<2A` metric was `48.02%` at U0,
  `58.45%` at U25k, and `56.81%` at U50k. Thus `best.pt` correctly remains
  U25k even though `latest.pt` is the complete U50k state.
- Complete metrics, hard K2 slices, external descriptive characterization,
  artifact hashes, and the promotion boundary are recorded in
  `docs/S50_SYMMETRY_CONFIDENCE_RESULTS.md`.
