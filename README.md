# EFF-Dock

**Fragment-based protein–ligand docking with SE(3) flow matching.**

Generate poses in a supplied protein pocket and rank them with a paired
confidence model. Includes runnable code and two pretrained checkpoints.

[Results](#benchmark-results) · [Animations](#docking-in-motion) · [Install](#install) · [Usage](#dock-a-ligand) · [Architecture](#architecture) · [Documentation](#documentation)

[![EFF-Dock workflow: prepare a supplied pocket and ligand fragments, generate poses, refine them, rank candidates by predicted RMSD and select a pose.](docs/assets/workflow.png)](docs/assets/workflow.png)

From ligand fragments and a supplied pocket to pose generation, energy-based
refinement and confidence selection. Crystal overlays and RMSDs are
retrospective annotations, not ranking inputs.

## Benchmark results

Reference-pocket redocking with 100 poses, 10 unguided sampling steps,
energy refinement and chirality-filtered confidence selection. Values are
percent mean ± sample SD over three inference repeats of the released weights.

<!-- benchmark-summary:start -->

| Benchmark (n) | Top-1 RMSD | Top-1 RMSD + PB | Oracle@100 |
| :--- | ---: | ---: | ---: |
| Astex Diverse Set (85) | 83.1 ± 0.7 | 80.0 ± 0.0 | 96.5 ± 1.2 |
| PoseBusters v2 (308) | 82.1 ± 0.6 | 79.8 ± 0.2 | 94.9 ± 0.5 |
| PhiBench-derived (206) | 63.8 ± 3.1 | 60.5 ± 3.2 | 89.5 ± 0.7 |
| FoldBench-Pocket (558) | 75.3 ± 0.5 | 73.8 ± 0.6 | 91.6 ± 0.3 |
| OpenBind (925) | 53.2 ± 0.6 | 53.2 ± 0.6 | 87.9 ± 0.6 |

<!-- benchmark-summary:end -->

**Top-1 RMSD:** selected-pose symmetry-aware heavy-atom RMSD <2 Å.
**Top-1 RMSD + PB:** the same pose also passes all 27 non-RMSD PoseBusters checks.
**Oracle@100:** any of the 100 refined poses has RMSD <2 Å, irrespective of validity.

These are descriptive results under a supplied-holo protocol. Astex and
PoseBusters were used during development; the historical training exclusions
do not establish complete benchmark disjointness.
[Protocol, cohort definitions and source values](docs/EVALUATION.md#reported-benchmark-results).

## Docking in motion

<p align="center">
  <a href="docs/assets/pocket-context.gif"><img src="docs/assets/pocket-context.gif" width="32%" alt="Astex 1T46 receptor: camera zoom from the full protein into the supplied pocket."></a>
  <a href="docs/assets/fragment-flow.gif"><img src="docs/assets/fragment-flow.gif" width="32%" alt="Eleven actual saved ODE states of six rigid ligand fragments in the fixed 1T46 pocket."></a>
  <a href="docs/assets/confidence-ranking.gif"><img src="docs/assets/confidence-ranking.gif" width="32%" alt="Four refined 1T46 candidates with predicted RMSD ranks and retrospective crystal overlays; rank1 is selected."></a>
</p>

3Dmol.js views of **Astex 1T46**. The fragment flow is one saved illustrative
trajectory; confidence ranking shows four poses from a **separate 100-pose
refined bank**. Amber denotes the crystal reference, used only for retrospective
RMSD. Click a view to enlarge it. [Animation details and provenance](docs/assets/README.md#animations).

## Install

The accelerated environment is Linux, Python 3.12 and an NVIDIA GPU/driver
compatible with CUDA 13.0. Apple Silicon Macs support inference, single-device
training and evaluation with native PyTorch. MPS is selected automatically
when available, with CPU as a fallback; multi-process training remains
CUDA-only. Install [uv](https://docs.astral.sh/uv/) and
[Git LFS](https://git-lfs.com/) first. The Python version and dependencies are
pinned in `.python-version` and `uv.lock`.

```bash
git lfs install
git clone https://github.com/eightmm/EFF-Dock.git
cd EFF-Dock
git lfs pull
uv sync --frozen --python 3.12
uv run python scripts/verify_release.py
```

On Apple Silicon, use the same `uv sync --frozen --python 3.12` command. The
lockfile selects macOS PyTorch and skips Linux-only CUDA and PyG extension
wheels. Use `--device mps` to request Apple GPU inference explicitly, or
`--device cpu` if an operation is unsupported by your installed PyTorch MPS
backend. For docking and evaluation, omit `--device` to select the available
backend automatically. For training use `uv run eff-dock train --device mps`;
confidence training also accepts `--device mps`. Training on a Mac runs in
float32 and can be much slower than the CUDA setup. Distributed training and
the retained large-scale benchmark shard scripts remain Linux/CUDA workflows.

The LFS download contains approximately 100 MB of model files. If cloning was
done with LFS downloads disabled, `git lfs pull` replaces the pointer files
with the checkpoints. The verifier checks their SHA-256 hashes and loads both
models on CPU. No dataset, API key or experiment-tracking login is needed for
inference.

## Dock a ligand

Run from the repository root. Supply a prepared receptor PDB, a ligand SDF or
SMILES string, and a pocket center in the receptor's coordinate frame:

```bash
uv run eff-dock dock \
  --protein receptor.pdb \
  --ligand ligand.sdf \
  --pocket-center '12.4,-3.1,8.7' \
  --orientation-injection rw \
  --confidence-frame-policy contextual_v1 \
  --rank-by confidence \
  --no-score \
  --seed 42 \
  --output-dir outputs/docked
```

Replace the input filenames and example center with your own. For SMILES, use
an argument such as `--ligand 'CCO'`. The default budget is 100 poses, 10
late-biased ODE steps, a 2 Å translation prior and a 10 Å receptor crop. The
explicit convention options above select the evaluated `rw` and
`contextual_v1` paths; checkpoint-compatible historical defaults remain
available as documented in [coordinate conventions](docs/ORIENTATION_INJECTION.md).

The command generates poses and ranks them by the confidence model's predicted
RMSD. `--no-score` disables the optional Vina/DG scoring, while confidence
scoring remains active. Outputs are `docked_poses.sdf` (best-ranked pose first)
and `results.pt`; a single-pose run writes `docked.sdf` instead. This command
does not automatically apply energy refinement or chirality-filtered reselection; those are separate workflows.
It requires a known pocket and does not predict binding affinity.

For a small installation check using a synthetic receptor and one pose:

```bash
uv run python scripts/verify_release.py --smoke
```

This check chooses CUDA, MPS or CPU automatically (or accepts `--device cpu`)
and exercises preparation, sampling, confidence scoring and SDF export with
the released checkpoints. It is not an accuracy benchmark; the command without
`--smoke` only loads the models on CPU.

<details>
<summary>Python API</summary>

```python
from pathlib import Path
import torch
from effdock.inference import DockingOptions, dock

dock(DockingOptions(
    protein=Path("receptor.pdb"),
    ligand="ligand.sdf",
    pocket_center=torch.tensor([12.4, -3.1, 8.7]),
    orientation_injection="rw",
    confidence_frame_policy="contextual_v1",
    rank_by="confidence",
    score=False,
    seed=42,
    out_dir=Path("outputs/docked"),
))
```

</details>

## Architecture

[![EFF-Dock architecture: docking and confidence networks, an interaction layer with residual connections, equivariant convolution, and normalization and activation.](docs/assets/architecture.png)](docs/assets/architecture.png)

The docking network predicts fragment translation and rotation through a
Newton–Euler readout. The confidence network combines atom and protein–ligand
contact features to predict pose RMSD for ranking. The figure expands the
shared interaction layers, equivariant convolution, normalization and activation.
See the [model contract](docs/MODEL.md) and
[implementation details](docs/methods/README.md) for the released checkpoints
and coordinate conventions.

## Documentation

- [Checkpoints and model cards](weights/MANIFEST.md)
- [Data, architecture and training](docs/README.md)
- [Evaluation protocol and benchmark source values](docs/EVALUATION.md)
- [Reproducibility](docs/REPRODUCIBILITY.md)
- [Workflow helpers](scripts/README.md) and [external-model adapters](benchmarks/README.md)

Figure provenance is recorded [here](docs/assets/README.md). Manuscripts,
full figure bundles and local study outputs are kept separately.

Training and evaluation require separately prepared data. They are exposed
through `eff-dock data`, `eff-dock train`, `eff-dock confidence` and
`eff-dock evaluate`; run each command with `--help` for its options.

<details>
<summary>Development checks</summary>

```bash
uv sync --frozen --python 3.12 --group dev
uv run pytest -q -m 'not gpu'
bash scripts/check.sh fast
```

</details>

## License

Source code and the two released EFF-Dock checkpoints are provided under
[Apache License 2.0](LICENSE). Third-party software and datasets retain their
own terms.
