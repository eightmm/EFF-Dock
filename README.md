# EFF-Dock

EFF-Dock performs fragment-level flow matching on SE(3) for protein-ligand
docking, using an equivariant graph backbone. The repository contains the Python implementation,
training and evaluation workflows, released docking/confidence weights,
and benchmark figures prepared for manuscript drafting.

EFF-Dock predicts ligand poses inside an explicitly supplied binding pocket.
It does not perform blind pocket discovery, binding-affinity prediction, or
binder/non-binder classification.

## Released model

The manuscript evaluates the released weights with explicit `rw` local-to-world feature injection and `contextual_v1` coordinate-only confidence frame recovery. Checkpoint-compatible defaults remain available; the example below opts into the evaluated conventions. [Coordinate conventions and checkpoint compatibility](docs/ORIENTATION_INJECTION.md) and the [model contract](docs/MODEL.md) describe these fixed-weight changes. Fresh training can use `configs/train_rw.yaml`.

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
    orientation_injection="rw",
    confidence_frame_policy="contextual_v1",
    out_dir=Path("outputs/docked"),
    device="cuda",
    seed=42,
)

dock(options)
```

This example generates and scores a raw ensemble. The manuscript's primary evaluation additionally uses energy refinement and input-stereochemistry-aware selection, as documented in the [reproducibility guide](docs/REPRODUCIBILITY.md). The call writes the complete pose ensemble to `docked_poses.sdf`, raw tensors
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

The current reporting baseline uses **Rw, contextual confidence frames, three inference repeats and unguided N100/S10/sigma2**, using the
frozen released docking/U70k pair, explicit physical refinement and input-chirality
selection. These postprocessing steps are not silently applied by `dock()`.
RMSD is symmetry-aware heavy-atom RMSD without alignment. PB-valid success
requires the same pose to have RMSD <2 Angstrom and pass PoseBusters. Values
are percent mean ± sample SD. Released weights retain legacy-trained features;
these Rw results are a cross-operator frozen-weight evaluation, using explicit
coordinate conventions documented in the [model contract](docs/MODEL.md).

| Dataset | N per seed | RMSD <2 Å | PB-valid poses | PB-valid success |
|---|---:|---:|---:|---:|
| Astex Diverse Set | 85 | 83.14 ± 0.68 | 95.69 ± 0.68 | 80.00 ± 0.00 |
| PoseBusters v2 | 308 | 81.49 ± 0.56 | 95.24 ± 0.19 | 79.11 ± 0.19 |
| PhiBench-derived | 206 | 63.11 ± 3.36 | 93.69 ± 0.00 | 60.52 ± 3.23 |
| FoldBench-Pocket | 558 | 75.27 ± 0.54 | 96.54 ± 0.10 | 73.78 ± 0.63 |
| OpenBind | 925 | 53.19 ± 0.56 | 99.68 ± 0.19 | 53.19 ± 0.56 |

Postprocessing and metric definitions are given in the Supplementary Information of the accompanying manuscript.

These are supplied-pocket redocking results, not blind docking or prospective
screening. Astex, PoseBusters, and the temporal cohorts were inspected during
development, so their results are descriptive. U70k was selected only on the
fixed 1,035-complex PLINDER validation bank. PhiBench-derived and FoldBench are the
core temporal checks; OpenBind is reported separately as a dense
single-protease auxiliary cohort.

FoldBench-Pocket uses holo-receptor, crystal-pocket redocking targets, rather
than the native FoldBench cofolding task. Benchmark conditions, preparation differences and comparison endpoints
are documented in the Supplementary Information of the accompanying manuscript.

## Manuscript materials

The accompanying manuscript is under review and is not distributed in this
repository. The [numerical Source Data](papers/data) for its figures and tables
are released here.

The figure captions and numerical data specify the evaluated protocols and
which analyses use stored candidate restrictions. [Training membership and
eligibility](benchmarks/inputs/training_membership/README.md) and
[implementation details](docs/methods/README.md) accompany the released code.


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
