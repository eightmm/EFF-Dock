# S50 symmetry-confidence training results

Protocol ID: `EFFDOCK-S50-SYMMETRY-CONFIDENCE-TRAINING-V1`

Status: completed. The four-GPU recovery run reached U50,000 with a
complete 1,035-complex validation ledger at every registered 5,000-update
look. The result is an internal, repeated-PLINDER model-selection study; it is
not an independent external-generalization claim.

## Frozen data and target

- S50 bank manifest SHA-256:
  `d45e36f3f2d75fb8ba9553715e1ec45031e4ad31881631d8632b6c19999f2d2b`
- Filtered split SHA-256:
  `1f23b50ef3a5eff73fd8cad683c1f3adfe4e1ab235199274d0dbc220c1d22507`
- Symmetry-label manifest SHA-256:
  `89ea40b02b121387228e3d47461a68a30bd749143b6281fe4d5ea9ab89056981`
- Warm-start confidence SHA-256:
  `e31fde6f351284205c78f7a1510002779c43312e94d9f82003d47a14d72bc78f`
- Train inventory: 43,092 complexes and 4,309,200 poses.
- Validation inventory: 1,035 complexes and 103,500 poses.
- Pose-level training and evaluation target: RDKit `CalcRMS`, symmetry-aware,
  no alignment, stored as `pose_rmsd_symmetry_no_align`.
- Atom displacement regression/BCE retained the fixed-map `atom_disp` target;
  symmetry-equivalent per-atom displacement is not uniquely defined.

## Internal selection result

The table is recomputed from the sealed per-complex U0, U25k, and U50k
validation ledgers. Each row uses the same 1,035 complexes and all 100 poses.

| Checkpoint | Top-1 <2A | Top-5 <2A | Selected mean RMSD | Selected median RMSD |
|---|---:|---:|---:|---:|
| U0 warm start | 497/1,035 (48.02%) | 66.76% | 3.235 A | 2.132 A |
| U25k | 605/1,035 (58.45%) | 69.95% | 2.735 A | 1.602 A |
| U50k | 588/1,035 (56.81%) | 69.57% | 2.686 A | 1.651 A |

U25k improved the primary metric by 108 complexes, or `+10.43` percentage
points, over U0. U50k remained `+8.79` points above U0 but was `-1.64` points
below U25k. The frozen primary therefore selects U25k as `best.pt`; U50k is the
terminal `latest.pt` and a valid continuation source, not the internally
selected checkpoint.

The primary hard slices show where the improvement occurred:

| Oracle K2 slice | N | U0 Top-1 <2A | U25k Top-1 <2A | U50k Top-1 <2A |
|---|---:|---:|---:|---:|
| 0 | 230 | 0.00% | 0.00% | 0.00% |
| 1-4 | 167 | 27.54% | 37.13% | 38.32% |
| 5-9 | 99 | 49.49% | 60.61% | 55.56% |
| >=10 | 539 | 74.58% | 89.61% | 87.01% |

## Checkpoints and provenance

- Run root:
  `outputs/eff-dock/s50-confidence-symmetry-training/88509b2189ab619c91feee843e4d90388b85b36b181532ca0a54ebf271f69b97/full`
- Selected U25k `best.pt` SHA-256:
  `1c59034172fb925cc8a70777dcba236be349f1a1de1775d49cc17d492b17c030`
- Terminal U50k `latest.pt` SHA-256:
  `fd49fa86f67187bf26d6c1bcf2daf925ba3e3b19dfeae733e57535d183280469`
- Final `metrics.json` SHA-256:
  `5fab2a3dce00332b4d46858e16d1586d37f86cb6b04adf03c006bb9997c015f7`

Neither experimental checkpoint has been copied into `weights/` or made the
public `dock`/`evaluate` default. The retained step-42,500 extmatch checkpoint
remains the packaged compatibility model.

## Repeated-use external characterization

The frozen external run compared U1.5k, U25k, and U50k on the same frozen
sigma-2 N100 Astex/PoseBusters candidates before and after deterministic
step-100 refinement. This is descriptive because both benchmarks had already
been opened; it cannot select a checkpoint.

| Dataset/stage | U25k Top-1 <2A | U50k Top-1 <2A | U25k joint valid+<2A | U50k joint valid+<2A |
|---|---:|---:|---:|---:|
| Astex raw | 80.00% | 82.35% | n/a | n/a |
| Astex refined | 84.71% | 85.88% | 80.00% | 81.18% |
| PoseBusters raw | 75.97% | 78.25% | n/a | 56.82%* |
| PoseBusters refined | 81.17% | 84.09% | 77.60% | 81.17% |

`*` The U50 raw joint value comes from the separate official PoseBusters
validity decomposition. It measured raw official validity at 64.94%, refined
same-index validity at 93.51%, and refined-reselected validity at 93.83%.
Refinement supplied most of the validity gain; reselection supplied the
additional RMSD gain.

The external report SHA-256 is
`501d2010a4df65fb0d9779e66113c7f3f423cd418d4ac683d903ec9b3fe1590a`.
The U50 PoseBusters validity report SHA-256 is
`e60bd0562035aa60e593028f821985d8b632f23a0ff098af56287aa932cbb082`.
The exact external contract is in
`docs/S50_SYMMETRY_CONFIDENCE_REFINED_EXTERNAL_PROTOCOL.md`.

## Decision

- Use U25k `best.pt` when the decision must follow the registered internal
  checkpoint-selection rule.
- Preserve U50k `latest.pt` as the terminal training state and as the immutable
  initialization for the separately registered raw+refined continuation.
- Do not promote either checkpoint into `weights/` or change public defaults
  from these repeated-use results alone.
