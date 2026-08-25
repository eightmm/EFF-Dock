# Recent external guided/refined benchmark protocol

Protocol ID: `EFFDOCK-EXTERNAL-TEMPORAL-GUIDED-REFINED-V1`

Status: frozen before smoke or benchmark outcomes were inspected.

## Scope and claim boundary

This is one descriptive EFF-Dock pocket-redocking run on three newly prepared
cohorts. It is not a hyperparameter sweep. PhiBench is an EFF-Dock-derived
203-system cohort, FoldBench is a 66-system pocket-redocking adaptation rather
than its native leaderboard task, and OpenBind is a clean 860-system
non-covalent EV-A71/CVA16 2A cohort. Dataset results remain separate; the one
PhiBench/FoldBench PDB overlap is not removed from either table.

## Frozen inputs

| Dataset | N | Manifest SHA-256 | SMILES SHA-256 | Pocket-center SHA-256 |
|---|---:|---|---|---|
| PhiBench | 203 | `2697ecc14a83646a26aac319193f7ad98c202349836fda3bcac4e533f1a10633` | `2ed3c80c1c0c4736314a9149a3fc8933ef33b6077b45467ac8f03fca4a098e37` | `0ad64e2da8fb94cce64dd9e245b215bd30cdddd18ae709da2c1bb423ebda1ecd` |
| FoldBench | 66 | `7f6a77670d28103afc5eb08509a946b35d2b29cf5b17223e7832ff83fd5cb845` | `fce9550a5739649d0236fd6bc5a95fcdc492470767afd794ab3163425ac90989` | `bad7b200e75cb41b56945de3d4ee309432136a5f08877a5a0199d63a7c165d77` |
| OpenBind | 860 | `f5f8424698fc30970676c52d4e9d4f1b725e8127e540697d25a2d2822982b81d` | `0fbda14dcaa25ff2f48ecb7a923d34e9e2dae0b609c31241d8e20829af9ab194` | `ce0102b6126966a78338abc72502be66704790d235c9be6dd49d029c470d04f4` |

- Docking checkpoint: S50 EMA U50k, SHA-256
  `65be44d7dc8f0867eb9fc5d22214b80f93971ea4702679a527c665046e91e6b6`.
- Confidence checkpoint: internally selected symmetry-confidence U25k,
  SHA-256
  `1c59034172fb925cc8a70777dcba236be349f1a1de1775d49cc17d492b17c030`.
- Model config SHA-256:
  `39aa62e4a48ed6f3aa4ff59345fb43a81220e2baba22edfd5beb0c4981b307ec`.
- Source verification is mandatory. All 1,129 normalized complexes passed
  loader discovery and complete heavy-atom element/connectivity mapping before
  sampling.

## Sampling and guidance

- One deterministic run per complex with seed root 42.
- `N=100` candidate poses and `S=10` learned ODE steps: 1,000 learned
  pose-steps per complex.
- Translation prior `sigma=2.0 A`, pocket cutoff `10 A`, no center jitter, and
  the late schedule with power 3.
- Unified in-repository GuidanceEnergy only; Vina guidance is disabled.
- Guidance acts within admitted ODE intervals as normalized direct drift with
  `eta=2.0`, beginning at `t=0.5`, using the frozen geometry-only receptor
  policy and registered force, velocity, angular-velocity, displacement, and
  backtracking caps.

The frozen pocket center is computed before inference from the crystal complex:
it is the centroid of receptor residue virtual nodes within 8 A of the
reference ligand, with the ligand centroid as fallback. That derived
three-vector, prepared receptor coordinates, and ligand chemistry may enter
generation. Reference ligand atom coordinates do not otherwise enter the model
or GuidanceEnergy; they are used directly after generation for RMSD.
PoseBusters outcomes are evaluation-only information.

## Refinement and selection

- Refine all 100 saved poses independently in rigid-fragment SE(3) coordinates
  with the same in-repository GuidanceEnergy and Torch autograd.
- Maximum 100 update iterations, pose batch size 10, maximum accepted atom displacement
  `0.10 A`, monotone pose-wise backtracking, and materialized steps
  `0, 25, 50, 75, 100`.
- Adaptive stopping starts at step 25. Stop after five accepted updates with

  `delta_E <= 0.02 kcal/mol + 1e-3 * max(1 kcal/mol, abs(E))`.

- Re-score step 0 and the terminal/step-100 bank in chunks of 20 at confidence
  sigma 2.0. Select by stable argmin predicted symmetry-aware RMSD from U25k;
  ties use original pose index. RMSD and validity never enter selection.

## Endpoints and completion gates

- Top-1 RMSD: symmetry-aware, no-alignment heavy-atom RMSD `<2 A` for the
  confidence-selected pose.
- Oracle RMSD: best RMSD among the same 100 candidates; diagnostic only.
- PB-valid: all 27 non-RMSD PoseBusters 0.6.5 `redock` checks pass; missing or
  non-finite checks fail.
- PL-valid: the 21-check view that excludes only organic-cofactor,
  inorganic-cofactor, and water-contact checks.
- Joint: the same confidence-selected pose is PB-valid and has RMSD `<2 A`.

One fixed complex from each dataset must pass sampling, refinement, confidence,
finite-pose, mapping, and hash smoke gates before full execution. Aggregation
requires exactly 203, 66, and 860 unique result rows, complete disjoint shard
coverage, exact checkpoint and input hashes, and no dropped failure rows. No
outcome from these cohorts may change this run's checkpoint, coefficients,
stopping rule, selector, or cohort.
