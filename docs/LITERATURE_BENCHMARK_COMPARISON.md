# Literature benchmark comparison

Last source audit: 2026-08-26

This ledger records published docking results next to the current EFF-Dock
U50k report. EFF-Dock values are local measurements; every other value is
transcribed from the cited paper, source-data workbook, or official model
card. No external method was rerun for this table.

## How to read the tables

- `Top-1 RMSD` is the percentage of complexes whose selected pose has the
  source-defined symmetry-aware ligand RMSD below, or at, 2 A.
- `Top-1 joint` requires that same selected pose to pass the source's
  PoseBusters validity definition as well as the RMSD threshold.
- `Oracle` asks whether any pose in the source's candidate set succeeds; it is
  not a deployable Top-1 result.
- `NR` means that the source did not report that endpoint for that cohort.
- Candidate count, pocket construction, receptor input, training set,
  minimization, ranking, RMSD implementation, inequality (`<` versus `<=`),
  and PoseBusters version differ across sources. These are contextual
  comparisons, not a single controlled leaderboard.

EFF-Dock rows use 100 poses, 10 ODE steps, translation prior `sigma=2`, direct
GuidanceEnergy drift with `eta=2`, adaptive in-repository refinement, and the
terminal U50k confidence checkpoint. Its SHA-256 is
`fd49fa86f67187bf26d6c1bcf2daf925ba3e3b19dfeae733e57535d183280469`.

## Astex Diverse (`N=85`)

The EFF-Dock row is pocket-conditioned holo redocking. Matcha's main row is a
blind-docking result with GNINA minimization and validity filtering. The other
external rows are pocket/holo results or reproductions in the SurfDock source
data unless the setting says otherwise.

| Method | Setting in source | Top-1 RMSD (%) | Top-1 joint (%) | Source |
|---|---|---:|---:|---|
| SurfDock | minimized | 95.29 | 91.76 | S1 |
| Uni-Mol Docking V2 | pocket specified | 95.29 | NR | S2 |
| SurfDock | unminimized | 92.86 | 63.10 | S1 |
| SigmaDock | pocket, 40 seeds, energy/PB ranker, no minimization | 90.60 | 90.60 | S3 |
| **EFF-Dock U50k** | pocket, N100/S10, guidance + adaptive refinement | **85.88** | **81.18** | EFF |
| Matcha | blind, 20 poses, GNINA minimization/filter/ranking | 85.90 | 82.40 | S4 |
| DiffDock-L | SurfDock source-data reproduction | 85.19 | 77.78 | S1 |
| GNINA | SurfDock source-data reproduction | 83.53 | 80.00 | S1 |
| GLIDE SP | SurfDock source-data reproduction | 83.53 | 81.18 | S1 |
| KarmaDock | SurfDock source-data reproduction | 74.12 | 0.00 | S1 |
| DiffDock | SurfDock source-data reproduction | 71.76 | 47.06 | S1 |
| KarmaDock (force-field) | SurfDock source-data reproduction | 69.41 | 3.53 | S1 |
| KarmaDock (aligned) | SurfDock source-data reproduction | 68.24 | 11.76 | S1 |
| GOLD | SurfDock source-data reproduction | 67.06 | 63.53 | S1 |
| Smina | SurfDock source-data reproduction | 64.29 | 64.29 | S1 |
| TankBind | SurfDock source-data reproduction | 58.82 | 5.88 | S1 |
| AutoDock Vina | SurfDock source-data reproduction | 57.65 | 56.47 | S1 |
| Uni-Mol | SurfDock source-data reproduction | 44.71 | 11.76 | S1 |
| DeepDock | SurfDock source-data reproduction | 34.52 | 11.76 | S1 |
| EquiBind | SurfDock source-data reproduction | 7.06 | 1.18 | S1 |

The `0.00` KarmaDock joint value above is reproduced literally from SurfDock's
Figure 2 source workbook. It should not be generalized beyond that preparation
and evaluation pipeline.

## PoseBusters v2 (`N=308`)

This is the closest literature panel to the current EFF-Dock evaluation, but
the evaluation contracts are still heterogeneous. In particular, AlphaFold 3
is a co-folding model supplied with pocket information, PocketXMol reports
only the joint endpoint for PBv2, Matcha is blind, and RLDiff RL++ calls Vina
scoring through Smina minimization and then GNINA reranking.

| Method | Setting in source | Top-1 RMSD (%) | Top-1 joint (%) | RMSD oracle (%) | Joint oracle (%) | Source |
|---|---|---:|---:|---:|---:|---|
| AlphaFold 3 (2019 cutoff) | co-folding, pocket specified | 93.20 | 84.40 | NR | NR | S5 |
| PocketXMol (self-ranking) | pocket, 100 poses; PBv2 joint only | NR | 84.70 | NR | NR | S6 |
| PocketXMol (tuned ranker) | pocket, 100 poses; PBv2 joint only | NR | 84.40 | NR | NR | S6 |
| **EFF-Dock U50k** | pocket, N100/S10, guidance + adaptive refinement | **84.09** | **81.17** | **95.78** | NR | EFF |
| SigmaDock | pocket, 40 seeds, energy/PB ranker, no minimization | 80.50 | 79.90 | NR | NR | S3 |
| DiffDock-Pocket RL++ | pocket, RL + Smina/Vina minimization + GNINA reranking | 80.2 +/- 1.2 | 78.2 +/- 1.0 | 88.5 +/- 0.9 | 87.7 +/- 0.9 | S7 |
| GNINA | pocket, hybrid search/ranking | 73.70 | 72.40 | NR | NR | S7 |
| DiffDock-Pocket RL | pocket, RL, no external refinement | 69.0 +/- 2.8 | 58.8 +/- 1.7 | 84.8 +/- 0.4 | 79.9 +/- 0.3 | S7 |
| DiffDock-Pocket | pocket baseline, no external refinement | 66.8 +/- 0.5 | 46.2 +/- 1.0 | 78.0 +/- 1.4 | 66.1 +/- 0.7 | S7 |
| Matcha | blind, 20 poses, GNINA minimization/filter/ranking | 65.90 | 63.00 | NR | NR | S4 |
| AutoDock Vina | pocket | 59.70 | 58.10 | NR | NR | S5/S7 |
| GOLD | pocket | 58.10 | 54.20 | NR | NR | S7 |
| Matcha-lite | blind, 10 poses, two stages | 54.20 | 51.30 | NR | NR | S4 |
| RapidDock | blind | 52.10 | NR | NR | NR | S9 |
| Re-Dock | pocket | 50.70 | 32.80 | NR | NR | S3 |
| DiffDock-L | blind, original author report | 50.00 | NR | NR | NR | S8 |
| DiffDock | holo receptor | 38.00 | 12.30 | NR | NR | S7 |
| Uni-Mol | pocket | 21.80 | 1.90 | NR | NR | S5 |
| DeepDock | pocket | 19.50 | 5.20 | NR | NR | S5/S7 |
| TankBind | holo receptor | 15.90 | 3.20 | NR | NR | S5 |
| EquiBind | holo receptor | 1.90 | 0.00 | NR | NR | S5/S7 |

RLDiff reports means and standard deviations across repeated runs, which are
retained above. Later re-evaluations differ slightly for some older baselines
(for example, GOLD joint 54.2 versus 54.5, Uni-Mol 20.8 versus 21.8, and
TankBind joint 3.9 versus 3.2). The main table uses RLDiff for GOLD and the
AlphaFold 3/PoseBusters Figure 4 values for Uni-Mol and TankBind; it does not
average incompatible evaluations.

### Closely related results excluded from the `N=308` panel

| Method | Reported denominator/task | Top-1 RMSD (%) | Top-1 joint (%) | RMSD oracle (%) | Why separate | Source |
|---|---|---:|---:|---:|---|---|
| nvDock | official model card says 306 PoseBusters complexes | 81.85 | NR | 94.51 | paper/model-card denominator mismatch; workshop/model release | S10 |
| ArtiDock | PoseBusters, `N=306` | 78.00 | NR | NR | two PBv2 entries omitted | S10/S11 |
| RobustDock | PBv2 `N=308`, predicted apo/flexible-receptor task | 54.20 | NR | NR | receptor and task differ from holo pocket redocking | S14 |

These values are useful context, but inserting them into the main `N=308`
holo-redocking ranking would conceal a changed denominator or task.

## Legacy PoseBusters (`N=428`)

PoseBusters v1/legacy results must not be mixed with the 308-complex PBv2
panel. The following table preserves the complete docking-method matrix in the
SurfDock Figure 2 source workbook.

| Method | Top-1 RMSD (%) | Top-1 joint (%) | Source |
|---|---:|---:|---|
| SurfDock | 81.54 | 74.07 | S1 |
| SurfDock (unminimized) | 78.04 | 40.42 | S1 |
| GNINA | 66.51 | 65.34 | S1 |
| GLIDE SP | 65.19 | 58.88 | S1 |
| AutoDock Vina | 52.34 | 51.17 | S1 |
| GOLD | 51.53 | 48.36 | S1 |
| Smina | 51.42 | 51.18 | S1 |
| DiffDock-L | 50.00 | 22.59 | S1 |
| KarmaDock | 46.73 | 0.47 | S1 |
| KarmaDock (force-field) | 42.99 | 0.70 | S1 |
| DiffDock | 38.03 | 14.02 | S1 |
| KarmaDock (aligned) | 33.64 | 7.01 | S1 |
| Uni-Mol | 22.95 | 2.10 | S1 |
| DeepDock | 17.80 | 4.91 | S1 |
| TankBind | 14.95 | 2.57 | S1 |
| EquiBind | 2.58 | 0.70 | S1 |

Additional author-reported legacy results are:

| Method | Setting | Top-1 RMSD (%) | Top-1 joint (%) | RMSD oracle (%) | Source |
|---|---|---:|---:|---:|---|
| Interformer | reference ligand conformation supplied | 84.09 | NR | NR | S12 |
| PocketXMol (tuned ranker) | pocket, 100 poses | 83.40 | 79.40 | 96.50 | S6 |
| PocketXMol (self-ranking) | pocket, 100 poses | 82.50 | 78.50 | 96.50 | S6 |
| Uni-Mol Docking V2 | pocket specified | 77.60 | NR | NR | S2 |

## OpenBind EV-A71 2A official-style Top-25 (`N=802`)

OpenBind is an any-pose Top-25 comparison after public filtering, not a Top-1
docking selector leaderboard. EFF-Dock has predictions for 786 complexes and
keeps all 16 missing predictions in the denominator.

| Method | Setting | PB-valid + BiSyRMSD <=2 A (%) | + LDDT-PLI >=0.8 (%) | Source |
|---|---|---:|---:|---|
| GNINA multi | redocking | 92.14 | 85.29 | S13 |
| **EFF-Dock U50k** | pocket redocking | **86.66** | **72.44** | EFF |
| Best co-folding | per-complex union, not one deployable model | 85.66 | 72.19 | S13 |
| Protenix | co-folding | 68.33 | 54.49 | S13 |
| OpenFold3-p2 | co-folding | 51.12 | 35.54 | S13 |
| AlphaFold 3 | co-folding | 44.64 | 25.81 | S13 |
| GNINA multi | fragment cross-docking | 37.78 | 13.59 | S13 |
| RosettaFold 3 | co-folding | 17.83 | 3.37 | S13 |
| Boltz-1 | co-folding | 14.21 | 7.48 | S13 |
| Boltz-2 | co-folding | 10.10 | 7.98 | S13 |

Counts, prediction coverage, and artifact identities are in the
[EFF-Dock OpenBind report](OPENBIND_OFFICIAL_TOP25_RESULTS.md).

## PhiBench and FoldBench

No external model values are placed beside the current EFF-Dock PhiBench or
FoldBench rows. The EFF-Dock `phibench` cohort is a 203-complex derived archive
cohort, not the authors' hidden/native split, and the 66-complex FoldBench
protein-ligand cohort is a pocket-redocking adaptation rather than FoldBench's
native co-folding/LDDT-PLI task. Native leaderboard values would therefore
compare different samples and endpoints. Their exact derivations are recorded
in the [external benchmark registry](EXTERNAL_TEMPORAL_BENCHMARKS.md).

## Sources

- **EFF**: [EFF-Dock benchmark results](BENCHMARK_RESULTS.md) and
  [OpenBind report](OPENBIND_OFFICIAL_TOP25_RESULTS.md).
- **S1**: [SurfDock, Nature Methods](https://www.nature.com/articles/s41592-024-02516-y),
  exact values from the article's Figure 2 source-data workbook.
- **S2**: [Uni-Mol Docking V2](https://arxiv.org/abs/2405.11769), Table 1.
- **S3**: [SigmaDock](https://arxiv.org/abs/2511.04854), Figure 4 and Table 1.
- **S4**: [Matcha](https://arxiv.org/abs/2510.14586), Figures 3 and 7.
- **S5**: [AlphaFold 3](https://www.nature.com/articles/s41586-024-07487-w),
  Extended Data Figure 4.
- **S6**: [PocketXMol](https://doi.org/10.1016/j.cell.2026.01.003), Figure 6.
- **S7**: [Teaching Diffusion Models Physics / RLDiff](https://doi.org/10.64898/2026.03.25.714128),
  Tables 1-3.
- **S8**: [DiffDock-L, ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/file/db334db287337b2a365120b524300ef3-Paper-Conference.pdf).
- **S9**: [RapidDock](https://arxiv.org/abs/2411.00004).
- **S10**: [nvDock official model card](https://huggingface.co/nvidia/nvDock).
- **S11**: [ArtiDock](https://doi.org/10.1021/acs.jcim.5c02777), which
  documents omission of two unprocessable PBv2 entries; the numeric value is
  transcribed from the S10 comparison table.
- **S12**: [Interformer](https://www.nature.com/articles/s41467-024-54440-6).
- **S13**: [OpenBind public Top-25 table](https://github.com/OpenBind-Consortium/EV-A71_2A_benchmark/blob/8849566aeb6b22c39589918d8ac00c24c0983aba/plotting/tables/allmethods_plot_data_scaffolds_filtered_top25.csv).
- **S14**: [RobustDock](https://openreview.net/forum?id=KVQpIbVzSm).

The machine-readable Astex/PBv2 plotting subset and metric annotations are in
[`LITERATURE_RMSD_COMPARISON.json`](LITERATURE_RMSD_COMPARISON.json).
