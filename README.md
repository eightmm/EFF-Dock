# EFF-Dock

**Fragment-based protein–ligand docking with SE(3) flow matching.**
Generate poses in a supplied protein pocket and rank them with a paired confidence model.

[Install](#install) · [Usage](#usage) · [Results](#benchmark-results) · [Demos](#docking-in-motion)

[![EFF-Dock workflow from a supplied pocket and ligand fragments to pose generation, refinement, and confidence ranking.](docs/assets/workflow.png)](docs/assets/workflow.png)

## Install

Install [uv](https://docs.astral.sh/uv/) and [Git LFS](https://git-lfs.com/).
Inference uses Python 3.12 and selects CUDA on compatible Linux/NVIDIA systems
(CUDA 13.0), MPS on Apple Silicon, or CPU. Use `--device cpu` to select CPU explicitly.

```bash
git lfs install
git clone https://github.com/eightmm/EFF-Dock.git
cd EFF-Dock
git lfs pull
uv sync --frozen --python 3.12
uv run python scripts/verify_release.py --smoke
```

The smoke test verifies both released checkpoints and docks one synthetic ligand.

## Usage

<details>
<summary>CLI and Python API examples</summary>

From the repository root, supply a prepared receptor PDB, ligand SDF or SMILES,
and pocket center in the receptor's coordinate frame.

### CLI

#### Dock a ligand

```bash
uv run eff-dock dock \
  --protein receptor.pdb --ligand ligand.sdf \
  --pocket-center '12.4,-3.1,8.7' \
  --rank-by confidence --no-score --seed 42 --output-dir outputs/docked
```

The default budget is 100 poses, 10 sampling steps, a 2 Å translation prior,
and a 10 Å receptor crop. Each run writes three files.

| File | Contents |
| --- | --- |
| `docked_poses.sdf` | Raw poses in the selected rank order |
| `docked_poses_minimized.sdf` | GuidanceEnergy-minimized poses with per-pose energies and status |
| `results.pt` | Raw coordinates, scores, and provenance |

The two SDFs share record order and `sample_index`. Each stores confidence
computed on its own coordinates, including `confidence_rmsd` and the one-based
`confidence_rank`, even with `--no-score`. For one pose, the SDF names are
`docked.sdf` and `docked_minimized.sdf`. With MPS, docking and confidence use
MPS and minimization runs on CPU.
[Coordinate conventions and checkpoint compatibility](docs/ORIENTATION_INJECTION.md).

#### Score or minimize saved poses

Both commands accept one or more SDF records for the same ligand in the same
heavy-atom order. `minimize` uses fixed-receptor GuidanceEnergy and defaults to
at most 100 steps. It writes a new SDF with per-pose energies and status.
`confidence score` accepts raw or minimized poses and writes predicted RMSD and
rank to JSON in input order.

```bash
uv run eff-dock minimize \
  --protein receptor.pdb --poses existing_poses.sdf \
  --pocket-center '12.4,-3.1,8.7' --ligand-template ligand.sdf \
  --output outputs/minimized.sdf

uv run eff-dock confidence score \
  --protein receptor.pdb --poses outputs/minimized.sdf \
  --pocket-center '12.4,-3.1,8.7' --ligand-template ligand.sdf \
  --output outputs/confidence.json
```

Minimization runs without model checkpoints. To reproduce a docking score,
use the original ligand template, crop, prior sigma, and feature conventions.

### Python API

```python
from pathlib import Path

import torch

from effdock.inference import (
    ConfidenceOptions,
    DockingOptions,
    MinimizationOptions,
    dock,
    minimize,
    score_saved_poses,
)

site = {
    "protein": Path("receptor.pdb"),
    "pocket_center": torch.tensor([12.4, -3.1, 8.7]),
}
ligand = "ligand.sdf"
docked = Path("outputs/docked")

dock(DockingOptions(
    **site, ligand=ligand, rank_by="confidence",
    score=False, seed=42, out_dir=docked,
))

# Process poses from a separate run or another program.
minimized = Path("outputs/minimized.sdf")
minimize(MinimizationOptions(
    **site, poses=Path("existing_poses.sdf"), ligand_template=ligand, output=minimized,
))
score_saved_poses(ConfidenceOptions(
    **site, poses=minimized, ligand_template=ligand,
    output=Path("outputs/confidence.json"),
))
```

</details>

## Benchmark results

Reference-pocket redocking with the released checkpoints, 100 poses, 10 unguided
sampling steps, energy refinement, and chirality-filtered confidence selection.
Values are percentages, reported as mean ± sample SD over three inference repeats.

<!-- benchmark-summary:start -->

| Benchmark (n) | Top-1 RMSD | Top-1 RMSD + PB | Oracle@100 |
| :--- | ---: | ---: | ---: |
| Astex Diverse Set (85) | 83.1 ± 0.7 | 80.0 ± 0.0 | 96.5 ± 1.2 |
| PoseBusters v2 (308) | 82.1 ± 0.6 | 79.8 ± 0.2 | 94.9 ± 0.5 |
| PhiBench-derived (206) | 63.8 ± 3.1 | 60.5 ± 3.2 | 89.5 ± 0.7 |
| FoldBench-Pocket (558) | 75.3 ± 0.5 | 73.8 ± 0.6 | 91.6 ± 0.3 |
| OpenBind (925) | 53.2 ± 0.6 | 53.2 ± 0.6 | 87.9 ± 0.6 |

<!-- benchmark-summary:end -->

Success uses symmetry-aware heavy-atom RMSD <2 Å in the fixed receptor frame.
**+ PB** requires the same pose to pass all 27 non-RMSD PoseBusters checks.
**Oracle@100** counts targets with any refined pose below 2 Å.

These are descriptive supplied-holo results. Astex and PoseBusters were used
during development, and complete disjointness from training is unverified.
[Benchmark protocol and source values](docs/EVALUATION.md#reported-benchmark-results).

## Docking in motion

<p align="center">
  <a href="docs/assets/pocket-context.gif"><img src="docs/assets/pocket-context.gif" width="32%" alt="Camera zoom into the supplied 1T46 pocket."></a>
  <a href="docs/assets/fragment-flow.gif"><img src="docs/assets/fragment-flow.gif" width="32%" alt="Eleven saved ODE states of six rigid ligand fragments."></a>
  <a href="docs/assets/confidence-ranking.gif"><img src="docs/assets/confidence-ranking.gif" width="32%" alt="Four refined candidates and their predicted RMSD ranks."></a>
</p>

Astex 1T46. The trajectory shows one saved rollout. Ranking shows four candidates from a
separate 100-pose refined bank. Amber marks the crystal reference used for RMSD.
[Animation details and provenance](docs/assets/README.md#animations).

## Documentation

- [Architecture diagram](docs/assets/architecture.png) · [Model and methods](docs/methods/README.md)
- [Checkpoints and model cards](weights/MANIFEST.md)
- [Data, training, and evaluation](docs/README.md) · [Reproducibility](docs/REPRODUCIBILITY.md)
- [Workflow helpers](scripts/README.md) · [External-model adapters](benchmarks/README.md)

## License

Source code and released weights use the [Apache License 2.0](LICENSE).
Third-party software and datasets retain their own terms.
