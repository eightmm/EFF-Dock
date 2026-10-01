# EFF-Dock

EFF-Dock performs fragment-level flow matching on SE(3) for protein-ligand
docking, using an equivariant graph backbone. The repository contains the Python implementation,
training and evaluation workflows, released docking/confidence weights,
and benchmark figures prepared for manuscript drafting.

EFF-Dock predicts ligand poses inside an explicitly supplied binding pocket.
It does not perform blind pocket discovery, binding-affinity prediction, or
binder/non-binder classification.

## Released model

**Orientation correction:** [Rw versus historical Rᵀw](docs/ORIENTATION_INJECTION.md) documents the mathematical fix, checkpoint compatibility and completed three-seed evaluation. Fresh training with `configs/train_rw.yaml` uses Rw; loading released weights preserves their historical operator unless `--orientation-injection rw` is requested. Primary benchmark figures now use the completed corrected Rw three-seed results. Historical ablations retain their explicit operator provenance.

The public model is one paired deployment stack:

- docking: `weights/effdock_docking_early_time_t0p10_50k.pt`;
- confidence: `weights/effdock_confidence_s50_raw_refined_u70k.pt`;
- sampling: 100 poses, 10 ODE steps, translation sigma 2.0;
- pocket crop: 10 Angstrom;
- time grid: late schedule with power 3;
- selection: minimum predicted pose RMSD.

The confidence model consumes hidden features from the paired docking model.
Changing either checkpoint, the pose generator, sigma, ODE budget, or pocket
crop is a distribution shift and should be reported explicitly. Checksums and
model-card links are in [`weights/MANIFEST.md`](weights/MANIFEST.md).

## Requirements

- Python 3.12;
- [`uv`](https://docs.astral.sh/uv/);
- Linux with an NVIDIA GPU compatible with the pinned CUDA 13 stack;
- Git LFS for the released weights.

```bash
git lfs install
uv sync --frozen --group dev
uv run python scripts/verify_release.py
```

The repository is currently distributed as a research codebase rather than a
general-purpose PyPI package. Run examples from the repository root so the
versioned configs and weights resolve to the released files.

## Python inference

The primary inference interface is `effdock.inference.DockingOptions` plus
`effdock.inference.dock`:

```python
from pathlib import Path

import torch

from effdock.inference import DockingOptions, dock

options = DockingOptions(
    protein=Path("receptor.pdb"),
    ligand="ligand.sdf",  # an SDF path or a SMILES string
    pocket_center=torch.tensor([12.4, -3.1, 8.7]),
    checkpoint=Path("weights/effdock_docking_early_time_t0p10_50k.pt"),
    confidence_checkpoint=Path(
        "weights/effdock_confidence_s50_raw_refined_u70k.pt"
    ),
    config=Path("configs/train.yaml"),
    num_samples=100,
    num_steps=10,
    sigma=2.0,
    pocket_cutoff=10.0,
    time_schedule="late",
    schedule_power=3.0,
    rank_by="confidence",
    out_dir=Path("outputs/docked"),
    device="cuda",
    seed=42,
)

dock(options)
```

The call writes the complete pose ensemble to `docked_poses.sdf`, raw tensors
and provenance to `results.pt`, and convenience selected-pose artifacts under
the requested output directory. A receptor, ligand chemistry, and explicit
pocket center are required; target/crystal ligand coordinates must not be used
to define the pocket in a prospective setting.

The `eff-dock` command remains as a thin wrapper around the same Python
workflows for reproducibility and Slurm jobs. It is not the primary public API.

## Training and evaluation

Reusable components are importable from the package:

```python
from effdock.confidence import DockingGraphPoseConfidence
from effdock.training import Trainer, flow_matching_loss
```

Experiment entry points live in `effdock.workflows`; the corresponding
configuration files are under `configs/`. Dataset construction, split rules,
checkpoint selection, and exact evaluation definitions are documented in:

- [`docs/DATA.md`](docs/DATA.md);
- [`docs/MODEL.md`](docs/MODEL.md);
- [`docs/EVALUATION.md`](docs/EVALUATION.md);
- [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

## Main results

The current reporting baseline uses **corrected Rw, three seeds, unguided N100/S10/sigma2**, using the
frozen released docking/U70k pair, explicit physical refinement and input-chirality
selection. These postprocessing steps are not silently applied by `dock()`.
RMSD is symmetry-aware heavy-atom RMSD without alignment. PB-valid success
requires the same pose to have RMSD <2 Angstrom and pass PoseBusters. Values
are percent mean ± sample SD. Released weights retain legacy-trained features;
these Rw results are a cross-operator frozen-weight evaluation, using an explicit
operator override documented in the [protocol](docs/RW_THREE_SEED_PROTOCOL.md).

| Dataset | N per seed | RMSD <2 Å | PB-valid poses | PB-valid success |
|---|---:|---:|---:|---:|
| Astex Diverse Set | 85 | 83.14 ± 1.80 | 95.69 ± 0.68 | 80.00 ± 2.04 |
| PoseBusters v2 | 308 | 81.82 ± 0.56 | 95.02 ± 0.50 | 79.00 ± 0.82 |
| PhiBench reconstructed full | 206 | 62.94 ± 3.23 | 93.69 ± 0.00 | 60.36 ± 3.08 |
| FoldBench-Pocket full | 558 | 75.27 ± 0.54 | 96.54 ± 0.10 | 73.78 ± 0.63 |
| OpenBind full | 925 | 53.19 ± 0.56 | 99.68 ± 0.19 | 53.19 ± 0.56 |

Postprocessing and metric definitions are in the
[figure captions](docs/paper/FIGURE_CAPTIONS.md).

These are supplied-pocket redocking results, not blind docking or prospective
screening. Astex, PoseBusters, and the temporal cohorts were inspected during
development, so their results are descriptive. U70k was selected only on the
fixed 1,035-complex PLINDER validation bank. PhiBench and FoldBench are the
core temporal checks; OpenBind is reported separately as a dense
single-protease auxiliary cohort.

FoldBench-Pocket uses holo-receptor, crystal-pocket redocking targets, rather
than the native FoldBench cofolding task. Benchmark conditions and comparisons
are documented in [the benchmark report](docs/BENCHMARK_RESULTS.md).

## Manuscript figures

[Paper materials](docs/paper/README.md) contain a 23-page result PDF,
individual PDF/PNG files, captions and a Prism reference package. They are
working materials for writing the manuscript; figure numbering and placement
can change. Their presence here does not indicate a published paper.

[Detailed equations](docs/methods/README.md), [editable LaTeX methods](docs/paper/prism/methods.tex),
[training membership](benchmarks/inputs/training_membership/README.md), and
[figure source data](benchmarks/results/paper/README.md) connect the manuscript
to the released implementation. Recreate all 14 figures from public numerical
records with `uv run python -m benchmarks.figures.paper`.

## Repository map

- `src/effdock/`: reusable Python package;
- `configs/`: training and evaluation configurations;
- `weights/`: the two released model artifacts and model cards;
- `benchmarks/`: external-model adapters and compact result artifacts;
- `scripts/`: retained workflow helpers, checks and Slurm entry points;
- `tests/`: unit and scientific-contract tests;
- `docs/`: model, data, evaluation and manuscript documentation.

Start with [`docs/README.md`](docs/README.md) for the documentation index and
[`docs/STRUCTURE.md`](docs/STRUCTURE.md) for ownership boundaries. Raw data,
pose banks, scheduler logs, complete run ledgers, and historical checkpoints
remain local and are not part of the public repository. Superseded campaign
launchers and intermediate experiment records are also kept outside the public tree.

## License

The EFF-Dock source code and released EFF-Dock model artifacts are provided
under the [Apache License 2.0](LICENSE). Third-party datasets, software, and
model artifacts remain subject to their respective terms.
