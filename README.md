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
an unpromoted research result. Its internally selected U25k checkpoint,
terminal U50k checkpoint, exact hashes, and repeated-use Astex/PoseBusters
diagnostics are documented in
[`docs/S50_SYMMETRY_CONFIDENCE_RESULTS.md`](docs/S50_SYMMETRY_CONFIDENCE_RESULTS.md).

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
