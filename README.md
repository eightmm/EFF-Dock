# EFF-Dock

EFF-Dock is a trainable protein-ligand docking project built around
fragment-level SE(3)-equivariant flow matching. It supports PLINDER
preprocessing, PyTorch DDP training, learned pose-confidence ranking, external
benchmark evaluation, exact checkpoint resume, and single-complex docking
through one CLI.

The public inference boundary requires a receptor structure, ligand chemistry,
and an explicit pocket center. EFF-Dock does not perform blind pocket discovery
or binding-affinity prediction.

## Requirements

- Python 3.12
- [`uv`](https://docs.astral.sh/uv/)
- Linux with an NVIDIA GPU for the pinned CUDA 13 training/inference stack
- Git LFS when cloning the retained model weights

## Setup

```bash
git lfs install
uv sync --group dev
uv run eff-dock --help
```

Run the CPU verification suite with:

```bash
uv run pytest -q
uv run ruff check .
```

## Canonical workflows

```bash
uv run eff-dock data curate --help
uv run eff-dock data prepare --help
uv run eff-dock data split --help
uv run eff-dock data benchmark --help
uv run eff-dock train --config configs/train.yaml
uv run eff-dock confidence prepare \
  --checkpoint weights/effdock_geometry_ft_100k_best.pt --split train
uv run eff-dock confidence train --config configs/train_confidence.yaml
uv run eff-dock evaluate --dataset astex --data-dir DATASET \
  --pocket-centers FROZEN_CENTERS.json
uv run eff-dock benchmark --help
uv run eff-dock dock --protein receptor.pdb --ligand ligand.sdf \
  --pocket-center X,Y,Z
uv run eff-dock physical trace --protein receptor.pdb \
  --ligand crystal_ligand.sdf --output outputs/guidance/trace.json
```

`physical trace` is the backward-compatible command for the diagnostic-only,
self-contained Torch guidance trace. It records physical, hydrophobic,
idealized missing-valence-cone hydrogen-bond, screened formal-charge-group, and
combined energies/forces for a crystal pose or saved trajectory; it does not
optimize the crystal or enable production ODE guidance. Vina is not part of
this path. See
[`docs/GUIDANCE_CONTRACT.md`](docs/GUIDANCE_CONTRACT.md).

## Retained weights

Model artifacts under `weights/` are tracked with Git LFS. The public default
inference stack is:

- `effdock_geometry_ft_100k_best.pt`
- `effdock_confidence_extmatch_n80_s25_step42500.pt`
- 100 candidate poses, 10 ODE steps, translation sigma 2.0, and a 10A pocket crop

Both `eff-dock dock` and `eff-dock evaluate` therefore run with
`--num-samples 100 --num-steps 10` unless explicitly overridden. Use
`--no-confidence` to disable learned reranking.

The packaged confidence checkpoint was originally trained on N80/S25/sigma0.5
pose banks. N100/S10/sigma2 is the current deployment sampling budget, so this
is an intentional candidate-distribution shift rather than a claim that the
retained checkpoint was trained with the new defaults. The historical
N80/S25/sigma0.5 contract remains in the model card for exact reproduction.

Checksums, compatibility notes, and the confidence model card are in
[`weights/MANIFEST.md`](weights/MANIFEST.md) and
[`weights/CONFIDENCE_MODEL_CARD.md`](weights/CONFIDENCE_MODEL_CARD.md).

The later S50/sigma-2 symmetry-confidence experiment is complete but remains
an unpromoted research result. Current benchmark reporting uses the terminal
U50k checkpoint on the frozen sigma-2 N100 candidate banks:

| Dataset | N | Raw Top-1 <2A | Refined Top-1 <2A | Refined oracle <2A | Refined PB-valid | Refined joint valid+<2A |
|---|---:|---:|---:|---:|---:|---:|
| Astex Diverse | 85 | 70/85 (82.35%) | 73/85 (85.88%) | 82/85 (96.47%) | 80/85 (94.12%) | 69/85 (81.18%) |
| PoseBusters v2 | 308 | 241/308 (78.25%) | 259/308 (84.09%) | 295/308 (95.78%) | 289/308 (93.83%) | 250/308 (81.17%) |

These Astex/PoseBusters results are descriptive external evaluations. The
registered internal PLINDER rule selected U25k (`58.45%` Top-1 `<2A`; U50k
`56.81%`), and that historical selection is not rewritten. U50k is used here
as the project reporting convention after external outcomes were opened; this
does not by itself promote the checkpoint into public deployment defaults.
Exact checkpoint hashes, validity decomposition, and evaluation boundaries are
documented in
[`docs/S50_SYMMETRY_CONFIDENCE_RESULTS.md`](docs/S50_SYMMETRY_CONFIDENCE_RESULTS.md).
The score-only U50 reporting override for the additional cohorts is frozen in
[`docs/EXTERNAL_TEMPORAL_U50_REPORT_PROTOCOL.md`](docs/EXTERNAL_TEMPORAL_U50_REPORT_PROTOCOL.md).

The same frozen N100/S10 guided/refined stack was run without retuning on
recent external pocket-redocking cohorts:

| Dataset | N | Raw Top-1 <2 A | Refined Top-1 <2 A | Refined joint PB-valid + <2 A |
|---|---:|---:|---:|---:|
| PhiBench derived | 203 | 61.58% | 65.02% | 60.10% |
| FoldBench P-L adaptation | 66 | 65.15% | 62.12% | 60.61% |
| OpenBind clean non-covalent | 860 | 48.14% | 51.74% | 50.93% |

These are descriptive pocket-redocking adaptations; PhiBench and FoldBench are
not claimed as native author-leaderboard reproductions. Cohort provenance,
exact U50-selected counts, validity, and artifact hashes are in
[`docs/EXTERNAL_TEMPORAL_GUIDED_REFINED_RESULTS.md`](docs/EXTERNAL_TEMPORAL_GUIDED_REFINED_RESULTS.md).

The same guided/refined inference stack has also been ranked with U50k
confidence and aggregated under the public OpenBind filtered scaffold-only
Top-25 contract (`N=802`, with 16 missing EFF-Dock predictions counted as
failures):

| Rank budget | PB-valid + BiSyRMSD <=2 A | + LDDT-PLI >=0.8 |
|---|---:|---:|
| Top-1 | 417/802 (52.00%) | 359/802 (44.76%) |
| Top-5 | 603/802 (75.19%) | 511/802 (63.72%) |
| Top-25 | **695/802 (86.66%)** | **581/802 (72.44%)** |

OpenBind's cross-method figure is an any-pose Top-25 comparison, not a
deployable Top-1 benchmark. Exact denominator construction, PoseBusters 0.6.5
validity, OpenStructure 2.11.1 BiSyRMSD/LDDT-PLI commands, public comparison
values, and artifact hashes are documented in
[`docs/OPENBIND_OFFICIAL_TOP25_RESULTS.md`](docs/OPENBIND_OFFICIAL_TOP25_RESULTS.md).

The portable code paths used by this study are published with the repository:

```bash
uv run python scripts/prepare_s50_confidence_training_bank.py --help
uv run python scripts/build_s50_symmetry_rmsd_sidecars.py --help
uv run python scripts/calibrate_s50_refinement_budget.py --help
uv run python scripts/refine_s50_confidence_pose_bank.py --help
uv run python scripts/materialize_s50_refined_confidence_bank.py --help
uv run python scripts/report_s50_confidence_training.py --help
uv run python scripts/run_guidance_sdf_post_refinement.py --help
uv run python scripts/score_guidance_sdf_post_refinement_confidence.py --help
uv run python scripts/report_s50_symmetry_confidence_refined_external.py --help
uv run python scripts/evaluate_s50_u50_refinement_validity.py --help
```

`configs/train_confidence_s50_symmetry.yaml` reproduces the symmetry-target
confidence setup. `configs/train_confidence_s50_raw_refined_10k.yaml` defines
the paired raw/refined continuation with 32 poses from each bank plus one
mapped-crystal anchor. The scripts require explicit manifests, hashes, paths,
and output roots. The external scripts reproduce the frozen refined-pose
rescoring and official PoseBusters validity decomposition. Generated banks,
checkpoints, and cluster-specific submission wrappers are intentionally not
stored in Git.

Raw data, generated outputs, and historical migration material stay local and
are ignored by Git. The active package, configs, tests, documentation, and
retained weights form the published interface.

See [`docs/STRUCTURE.md`](docs/STRUCTURE.md) for the repository layout,
[`docs/EVALUATION.md`](docs/EVALUATION.md) for the evaluation contract, and
[`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) for reproducibility notes.
