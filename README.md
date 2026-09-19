# EFF-Dock

EFF-Dock is a fragment-level SE(3)-equivariant flow-matching model for
protein-ligand docking. The repository contains the Python implementation,
training and evaluation workflows, the promoted docking/confidence weights,
and the evidence used for the accompanying paper.

EFF-Dock predicts ligand poses inside an explicitly supplied binding pocket.
It does not perform blind pocket discovery, binding-affinity prediction, or
binder/non-binder classification.

## Released model

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

The current reporting baseline is **three seeds, unguided N100/S10/sigma2**, using the
released docking/U70k pair, explicit physical refinement and input-chirality
selection. These postprocessing steps are not silently applied by `dock()`.
RMSD is symmetry-aware heavy-atom RMSD without alignment; Joint requires both
RMSD <2 Angstrom and PB-validity. Values are percent mean ± sample SD.

| Dataset | N per seed | Refined + filtered `<2A` | PB-valid | Joint |
|---|---:|---:|---:|---:|
| Astex Diverse | 85 | 82.35 ± 2.04 | 95.69 ± 0.68 | 79.22 ± 2.72 |
| PoseBusters v2 | 308 | 81.60 ± 0.94 | 95.56 ± 0.50 | 79.22 ± 1.42 |
| PhiBench reconstructed full | 206 | 62.62 ± 3.79 | 94.01 ± 0.28 | 60.19 ± 3.36 |
| FoldBench-Pocket full | 558 | 74.49 ± 0.37 | 96.36 ± 0.37 | 72.82 ± 0.63 |
| OpenBind full | 925 | 52.58 ± 0.61 | 99.64 ± 0.17 | 52.58 ± 0.61 |

The matched Astex/PB guided/unguided N100/S10 and N40/S25 comparisons are
complete; the earlier eta2 rows remain separate ablations. FoldBench includes
three PB shards with a disclosed energy-reference InChI compatibility repair.
Historical single-bank numbers remain in the [benchmark report](docs/BENCHMARK_RESULTS.md).
Current tables, reproducible figures and condition-specific captions are in
the [paper results](docs/paper/20260919/RESULTS.md).

PoseX is reported separately: input chirality+E/Z selection followed by
PoseX relaxation, not our own refinement. Across three seeds, SD718 gives
72.89 ± 1.08% RMSD success and 69.64 ± 0.64% Joint; CD1312 gives
59.94 ± 3.71% and 59.33 ± 3.47%. CD uses the official-style 109-group
aggregation, not per-case success; these endpoints are not interchangeable
with the table above. See the inventory for baseline comparisons and provenance.

These are supplied-pocket redocking results, not blind docking or prospective
screening. Astex, PoseBusters, and the temporal cohorts were inspected during
development, so their results are descriptive. U70k was selected only on the
fixed 1,035-complex PLINDER validation bank. PhiBench and FoldBench are the
core temporal checks; OpenBind is reported separately as a dense
single-protease auxiliary cohort.

For historical endpoint-aligned PhiBench context, U70k refined confidence Top-5 reaches
`156/203` (`76.85%`) RMSD success and `150/203` (`73.89%`) same-pose
PB-valid/RMSD joint success. The source-native paper cohort has 206 systems,
so this remains descriptive rather than a direct head-to-head claim.

FoldBench-Pocket uses the 558 released protein-ligand interfaces as
holo-receptor, crystal-pocket redocking targets. It is not the native
FoldBench cofolding leaderboard. Contract-aware side-by-side literature tables
are provided in
[`benchmarks/results/external_models/TEMPORAL_LITERATURE.md`](benchmarks/results/external_models/TEMPORAL_LITERATURE.md).

Released tables are in [`docs/BENCHMARK_RESULTS.md`](docs/BENCHMARK_RESULTS.md),
with machine-readable comparison artifacts under
[`benchmarks/results/`](benchmarks/results/).

## Repository map

The [manuscript evidence package](docs/paper/20260919/README.md) includes
six figures, source-hashed aggregates and manuscript text. Its
[runtime report](docs/paper/20260919/RUNTIME.md) distinguishes seconds per
complex, amortized seconds per generated/scored pose, and sampling-process
GPU peaks; these are saved-run costs, not isolated single-pose latency.

- `src/effdock/`: reusable Python package;
- `configs/`: training and evaluation configurations;
- `weights/`: the two released model artifacts and model cards;
- `benchmarks/`: external-model adapters and compact result artifacts;
- `scripts/`: experiment and Slurm launchers;
- `tests/`: unit and scientific-contract tests;
- `docs/`: method, data, evaluation, protocol, and result-evidence documents.

Start with [`docs/README.md`](docs/README.md) for the documentation index and
[`docs/STRUCTURE.md`](docs/STRUCTURE.md) for ownership boundaries. Raw data,
pose banks, scheduler logs, complete run ledgers, and historical checkpoints
remain local and are not part of the public repository.

## License

The EFF-Dock source code and released EFF-Dock model artifacts are provided
under the [Apache License 2.0](LICENSE). Third-party datasets, software, and
model artifacts remain subject to their respective terms.
