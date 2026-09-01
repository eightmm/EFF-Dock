# Released EFF-Dock model artifacts

The two model artifacts under `weights/` are tracked with Git LFS. Their source
runs remain intact under ignored `outputs/`; this paired stack is the complete
public training and inference boundary.

| File | Role | SHA-256 |
|---|---|---|
| `effdock_docking_early_time_t0p10_50k.pt` | Default docking checkpoint; early-time/t=0 replay fine-tune, step 50,000 EMA | `65be44d7dc8f0867eb9fc5d22214b80f93971ea4702679a527c665046e91e6b6` |
| `effdock_confidence_s50_raw_refined_u70k.pt` | Default pose-confidence checkpoint; raw+refined sigma-2 training, internally selected U70k | `ce59be42f0ca613871ca079127c3296f5ca9a4ec72e44a9e5cf61878351c2638` |

The promoted default is the first two files with N100/S10, translation
sigma 2.0, a 10A pocket crop, and pure minimum predicted-RMSD ranking. The
confidence checkpoint was trained and evaluated against hidden features from
the paired docking checkpoint; mixing it with the older geometry checkpoint
is outside the promoted contract.

Checkpoint loading is CPU-first with `weights_only=True`. Confidence metadata
contains `pathlib.PosixPath` values and is loaded with the explicit safe-global
allowlist in `effdock.confidence.runtime`. Learned key or shape mismatches fail
explicitly.

`--resume` is exact only for checkpoints created with the same optimizer and
scheduler layout. Historical and intermediate checkpoints are intentionally
not distributed. Their hashes may remain in frozen experiment protocols,
while the local files stay in the ignored archive/output store.

See `DOCKING_MODEL_CARD.md` and `CONFIDENCE_MODEL_CARD.md` for intended use,
selection evidence, external results, and limitations. The EFF-Dock source and
these two released artifacts are provided under Apache-2.0; third-party data
and software retain their own terms.
