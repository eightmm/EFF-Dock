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

Model artifacts under `weights/` are tracked with Git LFS. The recommended
matched inference stack is:

- `effdock_geometry_ft_100k_best.pt`
- `effdock_confidence_extmatch_n80_s25_step42500.pt`
- 80 poses, 25 ODE steps, translation sigma 0.5, and a 10A pocket crop

Both `eff-dock dock` and `eff-dock evaluate` use this matched stack by
default. Pass explicit checkpoint or sampling arguments to override it, or
`--no-confidence` to disable learned reranking.

Checksums, compatibility notes, and the confidence model card are in
[`weights/MANIFEST.md`](weights/MANIFEST.md) and
[`weights/CONFIDENCE_MODEL_CARD.md`](weights/CONFIDENCE_MODEL_CARD.md).

The later S50/sigma-2 symmetry-confidence experiment is complete but remains
an unpromoted research result. The registered internal PLINDER rule selected
U25k (`58.45%` Top-1 `<2A`; U50k `56.81%`), while repeated-use external
characterization consistently favored U50k on the same frozen sigma-2 N100
Astex/PoseBusters candidate banks:

| Dataset/stage | U25k Top-1 <2A | U50k Top-1 <2A | U25k joint valid+<2A | U50k joint valid+<2A |
|---|---:|---:|---:|---:|
| Astex raw | 80.00% | 82.35% | n/a | n/a |
| Astex refined | 84.71% | 85.88% | 80.00% | 81.18% |
| PoseBusters raw | 75.97% | 78.25% | n/a | 56.82% |
| PoseBusters refined | 81.17% | 84.09% | 77.60% | 81.17% |

These Astex/PoseBusters results are descriptive external evaluations, not an
absence of external testing. Because both benchmarks had already been opened,
they cannot override the preregistered internal checkpoint selection or alone
promote U25k/U50k into the public deployment defaults. U25k remains the
internally selected `best.pt`; U50k remains the terminal `latest.pt` and the
more promising deployment candidate under this repeated-use characterization.
Exact checkpoint hashes, validity decomposition, and the frozen evaluation
contract are documented in
[`docs/S50_SYMMETRY_CONFIDENCE_RESULTS.md`](docs/S50_SYMMETRY_CONFIDENCE_RESULTS.md).

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

## Reference-defined redocking diagnostics

The retained N80 confidence stack produced the following frozen-manifest
diagnostic results. These use reference-defined pocket centers and must not be
presented as blind-pocket or prospective docking performance.

| Dataset | N | Pure confidence RMSD <2A | Frozen composite RMSD <2A | Oracle-80 RMSD <2A |
|---|---:|---:|---:|---:|
| Astex Diverse | 85 | 76.47% | 78.82% | 95.29% |
| PoseBusters v2 | 308 | 73.05% | 72.73% | 94.81% |
| CASF-2016 | 285 | 69.47% | 68.42% | 91.93% |

The official PoseBusters pass-all validity of the composite-selected poses was
54.87%. Full protocol, limitations, failure/rescue provenance, and selector
comparisons are in [`docs/BENCHMARK_RESULTS.md`](docs/BENCHMARK_RESULTS.md).

Prospective docking and publishable target-independent evaluation require
target-independent pocket definitions. The retained-weight compatibility
benchmarks are separately labeled as reference-defined oracle-pocket redocking
diagnostics in [`docs/BENCHMARK_PROTOCOL.md`](docs/BENCHMARK_PROTOCOL.md) and
[`docs/CONFIDENCE_BENCHMARK_PROTOCOL.md`](docs/CONFIDENCE_BENCHMARK_PROTOCOL.md).

Raw data, generated outputs, and historical migration material stay local and
are ignored by Git. The active package, configs, tests, documentation, and
retained weights form the published interface.

See [`PROJECT.md`](PROJECT.md) for the scientific and engineering contract,
[`docs/STRUCTURE.md`](docs/STRUCTURE.md) for the repository layout, and
[`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) for reproducibility notes.
