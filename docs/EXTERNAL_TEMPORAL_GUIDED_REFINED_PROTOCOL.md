# Recent external guided/refined benchmark protocol

Protocol ID: `EFFDOCK-EXTERNAL-TEMPORAL-GUIDED-REFINED-V1`

Status: frozen before smoke or benchmark outcomes are inspected.

## Scope and claim boundary

This is one descriptive EFF-Dock pocket-redocking run on three newly prepared
cohorts. It is not a hyperparameter sweep. PhiBench is an EFF-Dock-derived
203-system cohort, FoldBench is a 66-system pocket-redocking adaptation rather
than its native leaderboard task, and OpenBind is the clean 860-system
non-covalent EV-A71/CVA16 2A cohort. Dataset results remain separate; the one
PhiBench/FoldBench PDB overlap is not removed from either table.

## Frozen inputs

| Dataset | N | Local manifest SHA-256 | SMILES SHA-256 | Pocket centers SHA-256 |
| --- | ---: | --- | --- | --- |
| PhiBench | 203 | `2697ecc14a83646a26aac319193f7ad98c202349836fda3bcac4e533f1a10633` | `2ed3c80c1c0c4736314a9149a3fc8933ef33b6077b45467ac8f03fca4a098e37` | `0ad64e2da8fb94cce64dd9e245b215bd30cdddd18ae709da2c1bb423ebda1ecd` |
| FoldBench | 66 | `7f6a77670d28103afc5eb08509a946b35d2b29cf5b17223e7832ff83fd5cb845` | `fce9550a5739649d0236fd6bc5a95fcdc492470767afd794ab3163425ac90989` | `bad7b200e75cb41b56945de3d4ee309432136a5f08877a5a0199d63a7c165d77` |
| OpenBind | 860 | `f5f8424698fc30970676c52d4e9d4f1b725e8127e540697d25a2d2822982b81d` | `0fbda14dcaa25ff2f48ecb7a923d34e9e2dae0b609c31241d8e20829af9ab194` | `ce0102b6126966a78338abc72502be66704790d235c9be6dd49d029c470d04f4` |

- Docking checkpoint: S50 EMA U50k
  `step50000_ema_common_init.pt`, SHA-256
  `65be44d7dc8f0867eb9fc5d22214b80f93971ea4702679a527c665046e91e6b6`.
- Confidence checkpoint: internally selected symmetry-confidence U25k
  `best.pt`, SHA-256
  `1c59034172fb925cc8a70777dcba236be349f1a1de1775d49cc17d492b17c030`.
- Model config SHA-256:
  `39aa62e4a48ed6f3aa4ff59345fb43a81220e2baba22edfd5beb0c4981b307ec`.
- Every source manifest has `source_verification_performed=true`; all 1,129
  normalized complexes passed loader discovery and a complete heavy-atom
  element/connectivity mapping audit before submission.

## Sampling and guidance

- One deterministic run per complex, seed root 42.
- `N=100` candidate poses and `S=10` learned ODE steps (`1,000` learned
  pose-steps per complex).
- Translation prior `sigma=2.0 Angstrom`, pocket cutoff `10 Angstrom`, no
  center jitter, late schedule with power 3.
- Unified in-repository GuidanceEnergy only; Vina guidance is disabled.
- Guidance acts inside every admitted ODE interval as normalized direct drift,
  `eta=2.0`, from `t=0.5`, with the frozen force, velocity, angular-velocity,
  displacement, backtracking, and `geometry_only` receptor-policy caps used by
  the completed Astex/PoseBusters sigma-2 study.

## Refinement and selection

- Refine all 100 saved poses independently in rigid-fragment SE(3) coordinates
  with the same in-repository GuidanceEnergy and Torch autograd.
- Maximum 100 attempts, batch size 10, maximum accepted atom displacement
  `0.10 Angstrom`, monotone pose-wise backtracking, saved steps
  `0,25,50,75,100`.
- Adaptive stopping begins at step 25 and stops after five accepted updates
  satisfying

  `delta_E <= 0.02 kcal/mol + 1e-3 * max(1 kcal/mol, abs(E))`.

- Re-score step 0 and step 100 in chunks of 20 at confidence sigma 2.0. Select
  by stable argmin predicted symmetry-aware RMSD from the frozen U25k model.
  RMSD and validity are outcomes and never selector inputs.

## Evaluation and execution gate

- Primary descriptive endpoints per dataset: raw and refined Top-1 ligand RMSD
  below 2 Angstrom, refined RMSD oracle, refined protein-ligand validity, and
  refined joint protein-ligand-valid plus RMSD below 2 Angstrom.
- Also retain all-27-check PoseBusters redock validity. Protein-ligand validity
  uses its 21-check view that excludes only organic-cofactor,
  inorganic-cofactor, and water contact checks.
- Run one fixed complex from each dataset as a smoke gate. The full 72-task GPU
  array starts only if all three complete sampling, refinement, confidence
  rescoring, finite-pose, mapping, and hash checks. PoseBusters runs afterward
  on `cpu_only`, followed by a coverage-enforcing aggregate report.
- No outcome from these external cohorts may change this run's checkpoint,
  coefficients, stopping rule, selector, or cohort.
