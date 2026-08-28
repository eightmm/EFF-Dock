# External docking benchmark protocol

This document freezes how EFF-Dock installs, names, classifies, and later
evaluates external docking baselines. A model name alone is not a benchmark
method: every reported result is an **executed pipeline arm** with independently
recorded site information, pose generation, scoring, and refinement.

## Scope

- Primary redocking sets: Astex Diverse (`N=85`) and PoseBusters Benchmark v2
  (`N=308`).
- Keep the official full denominator. Preparation or inference failures count as
  failures; they are not dropped from the denominator.
- Exclude co-folding methods from the primary docking comparison.
- Use released source code and released checkpoints at the revisions recorded in
  `configs/external_models.json`.
- Keep each upstream environment isolated. Do not install external-model
  dependencies into EFF-Dock's `.venv`.
- Preserve native pose generation and native ranking as one reported arm. Add a
  second arm only when changing the scorer or refinement stage intentionally.
- Evaluate every emitted SDF with one EFF-Dock evaluator and the official
  PoseBusters 27-check configuration. A method's internal validity estimate is
  never substituted for this common evaluation.

## Pipeline schema

Every run row must contain these fields:

| Field | Allowed interpretation |
|---|---|
| `site_information` | `pocket_supplied` or `blind_site` |
| `site_engine` | `supplied`, `classical_pocket_prediction`, `learned_pocket_prediction`, or `joint_learned_pose` |
| `pose_engine` | `classical_search`, `learned_regression`, `learned_generative`, or `learned_energy_search` |
| `scoring_engine` | `none`, `classical_energy`, `learned_confidence`, `learned_energy`, or a named deterministic heuristic |
| `refinement` | `none`, `geometry_projection`, `classical_minimization`, `learned_refinement`, or `in_repo_physical_gradient` |
| `family` | Derived `Classical`, `Hybrid`, or `DL` label for this exact arm |

`blind_site` here means holo full-receptor blind-site redocking unless a run
explicitly declares an apo/predicted receptor. It must not be described as apo
docking.

## Family decision rule

- **Classical:** site detection (if any), pose generation, ranking, and
  refinement have no learned component.
- **DL:** the runtime pose/selection pipeline is learned and uses no explicit
  classical energy, classical search engine, or physical-gradient refinement.
- **Hybrid:** the runtime arm combines a learned site/scoring/pose component with classical search,
  explicit geometry projection, classical/physical energy, classical
  minimization, or EFF-Dock's in-repository physical-gradient
  guidance/refinement.

The label belongs to a pipeline arm, not permanently to a paper or repository.
Training-time physics rewards do not by themselves make runtime inference
Hybrid. For example, RLDiff without runtime smina/GNINA remains DL; RL++ with
smina/GNINA is Hybrid.

Examples:

| Arm | Pose | Score | Refine | Family |
|---|---|---|---|---|
| Vina | classical search | Vina | none | Classical |
| Vina using a DiffDock-predicted site | classical search | Vina | none | Hybrid |
| GNINA | classical search | CNN | none | Hybrid |
| DiffDock-Pocket | learned diffusion | learned confidence | none | DL |
| SigmaDock + Vinardo | learned diffusion | Vinardo | none | Hybrid |
| DiffBindFR + MDN | learned diffusion | learned MDN | none | DL |
| DiffBindFR + smina | learned diffusion | smina | smina minimization | Hybrid |
| SurfDock | learned diffusion | learned MDN | none | DL |
| SurfDock + force optimization | learned diffusion | learned MDN | force-field minimization | Hybrid |
| FABind + post-optimization | learned regression | none | intraligand geometry projection | Hybrid |
| EFF-Dock ODE + U50k | learned flow | learned confidence | none | DL |
| EFF-Dock + physical guidance/refinement + U50k | learned flow | learned confidence | in-repo physical gradient | Hybrid |

## Fair-comparison arms

- Main fixed-compute arms: `S10 N100` and `S25 N40`, both with total model NFE
  budget 1000.
- Top-1 uses each declared scoring engine. EFF-Dock uses the U50k confidence
  checkpoint unless an ablation explicitly says otherwise.
- Oracle@40 is reported only when the method actually emitted at least 40 valid
  candidate poses. Do not fabricate Oracle@40 for deterministic single-pose
  methods.
- Run stochastic external methods with three recorded seeds when their official
  interface permits it. Preserve per-seed rows before computing mean/variation.
- Report both RMSD-only success (`RMSD < 2 A`) and Joint success
  (`RMSD < 2 A` and all official PoseBusters checks pass).

## Installation and provenance

Each active comparison model owns an independent uv project below `others/`.
Only dependency contracts and documentation are tracked; `.venv`, cache,
upstream source, weights, and native tools remain ignored inside that model's
directory. The legacy `external_models/` directory is a retained archive and
may seed an initial source/weight link, but it is not an active shared Python
environment.

```bash
MODEL=surfdock sbatch scripts/slurm/others_uv_sync.sbatch
bash scripts/others/run_model.sh surfdock python -c \
  'import torch; print(torch.__version__)'
```

An install is complete only when:

1. the source HEAD equals the pinned revision;
2. `others/<model>/.venv` was produced from the tracked per-model `uv.lock`;
3. the model's import or CLI smoke check passes;
4. any non-Python executable/library is model-local below
   `others/<model>/bin`, not inherited from another model environment;
5. required checkpoint files and their checksums are recorded.

SurfDock's environment requests `dimorphite-dl==1.3.2`, a release absent from
PyPI. Its uv project pins the upstream 1.3.2 fork at immutable commit
`ee006cc6344ca57b71777e99853014f38a816cf5`. It also pins Meta ESM 2.0.1 at
commit `2b369911bb5b4b0dda914521b9475cad1656b2ac`. The no-refinement benchmark arm
uses a fail-closed import shim for eagerly imported OpenFF optimization code;
requesting that optional path raises. The wrapper supplies only the constants
from the pinned checkout's malformed `global_vars.py`, executes APBS beside its
basename-referenced PQR, and keeps SO(3)/torsion lookup arrays in the model's
ignored cache. These corrections do not change weights, diffusion equations,
ranking, or refinement policy.

FABind checkpoints are accepted only after Git-LFS materialization. A checked
out pointer file is not a weight: `best_model.bin` must be exactly `145251173`
bytes and match SHA-256
`549d6f1cef6f8fcbc0c068afa572fa99df58886440f67a124c3bb0fbebe09622`,
the object ID declared by the pinned official LFS repository.

Interformer's PyVina extension is built inside its uv environment against a
model-local copy of the upstream Boost 1.84 headers/runtime. The Reduce binary,
`obrms`, and required compiler runtime libraries are likewise materialized
below `others/interformer/bin` from checksum-pinned packages when an archived
copy is unavailable. PLIP's distribution-name dependency is overridden so
only one pinned OpenBabel distribution provides the `openbabel` module. A
marker symlink exposes Python 3.12's single-file `dbm.gnu` database to the
released `dbm.dumb` discovery glob. The model-local `obrms` wrapper exposes its
matching OpenBabel shared library only to that subprocess, avoiding ABI
interposition into the Python wheel.

DiffBindFR keeps Torch 1.13.1 and the matching official cu117 PyG wheels.
`scikit-learn==1.4.1.post1` requires a compatible `joblib==1.3.2`; ProDy's
undeclared isolated-build requirements are explicitly supplied as NumPy 1.26.4
and Cython below 3. Its `--no_error_correction` arm uses a fail-closed PyMOL
import shim because supplied crystal ligands do not call PyMOL and no compatible
Python 3.9 uv wheel exists.

The not-yet-migrated PoseBench DiffDock archive compiles an OpenFold CUDA extension
against CUDA 11.8 but leaves the host compiler unconstrained. CUDA 11.8 rejects
the cluster's GCC 13. The installer deterministically splits the upstream conda
and pip phases, adds environment-local `gcc_linux-64==11` and
`gxx_linux-64==11`, then builds the unchanged upstream pip requirements with
those compiler wrappers. This avoids the unsafe `-allow-unsupported-compiler`
override and is captured in the resolved environment lock.

Environment installation runs only on the `cpu_only` Slurm partition. GPU smoke
inference is a separate later gate and is not implied by a successful install.

## Plot encoding

Order methods by supplied-pocket arms first, then draw a vertical dashed divider
before blind-site/full-receptor arms. Use family color only:

- Classical: pastel mint
- Hybrid: pastel peach
- DL: pastel blue

Use a solid bar for Top-1 Joint and a hatched extension for Joint Oracle@40.
Emphasize EFF-Dock with label weight and outline, not a fourth family color.
