# Retained EFF-Dock model artifacts

These are the model artifacts promoted into active EFF-Dock. Their historical
sources remain intact under ignored `outputs/`; the copies here are the stable
training/inference boundary.

| File | Role | SHA-256 |
|---|---|---|
| `effdock_legacy_flowfrag_200k_ema.pt` | Portable 200k EMA inference weights | `3ee604ec2338532532fa23a2ae91d0d540322defc32f5e453c8e7e12e389d36a` |
| `effdock_legacy_flowfrag_200k_resume.pt` | Historical full-state checkpoint; use `--init-from` with the new AdamW baseline | `ec0a5f2f08072a3f6b52b37db83d585d409241b7bf1c13c0ee4d6f854449c734` |
| `effdock_legacy_flowfrag_small_sigma_best.pt` | Historical small-sigma full-state checkpoint; weights-only migration supported | `10d4384d988aff6dfe0ec8de8a6691b7f1255ead1089a6172bf1ccd4f157ffc2` |
| `effdock_geometry_ft_100k_best.pt` | Geometry fine-tuned docking checkpoint paired with the extmatch confidence model | `6932fb3ba6ebac770f714453529656a44b8f33cf15119d23c9e675d2d60b36db` |
| `effdock_confidence_extmatch_n80_s25_step42500.pt` | Selected docking-graph pose-confidence checkpoint; N80/S25/sigma0.5/pocket10 extmatch training distribution | `e31fde6f351284205c78f7a1510002779c43312e94d9f82003d47a14d72bc78f` |

`--resume` is exact only for checkpoints created with the same EFF-Dock
optimizer/scheduler layout. Legacy full-state files remain intact, but their
weights should enter a new baseline through `--init-from`.

The confidence checkpoint is safely loaded with `weights_only=True` plus the
explicit `pathlib.PosixPath` allowlist required by its retained argument
metadata. The hard-pair fine-tunes remain historical because they did not beat
this step-42500 checkpoint on the frozen validation subset.
