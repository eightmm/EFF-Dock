# PhiBench and FoldBench literature context

Source audit: 2026-09-02

This note records published results related to EFF-Dock's temporal cohorts.
It deliberately separates source-native results from EFF-Dock's derived
pocket-redocking evaluations. The numbers below are useful context, but they
must not be placed in a single ranked table. FoldBench is shown as a protocol
matrix plus separate source-native and EFF-Dock panels; PhiBench rows retain
their contract columns and explicit non-comparability flag.

## EFF-Dock derived cohorts

EFF-Dock U70k uses 100 generated poses, 10 sampling steps, `sigma=2`, a known
crystal-pocket center, adaptive refinement, and confidence Top-1 selection.
RMSD is symmetry-aware ligand heavy-atom RMSD.

| Cohort | N | Raw Top-1 `<2 A` | Refined Top-1 `<2 A` | PB-valid | Joint PB-valid + `<2 A` |
|---|---:|---:|---:|---:|---:|
| PhiBench-derived | 203 | 63.05% | 64.53% | 90.64% | 59.11% |
| FoldBench-Pocket full | 558 | 72.04% | 75.63% | 95.16% | 72.94% |
| FoldBench-Pocket fully observed | 556 | 72.12% | 75.72% | 95.14% | 73.02% |
| FoldBench-Pocket post-cutoff | 66 | 68.18% | 71.21% | 89.39% | 66.67% |
| FoldBench fixed-66 legacy bank | 66 | 63.64% | 68.18% | 90.91% | 66.67% |

These values are the released EFF-Dock results. The PhiBench cohort is a
deterministic 203-system reconstruction from the official archive rather than
the authors' 206-system set. FoldBench-Pocket evaluates all 558 released
protein-ligand interfaces as crystal-pocket redocking; its post-cutoff slice
contains the 66 interfaces released strictly after 2024-06-30. The legacy
fixed-66 bank is retained because it used different per-complex seeds and is
not an interchangeable repeat of the full-run slice. The public full-run
ledger is [`foldbench_pocket_558.json`](foldbench_pocket_558.json).

The separately evaluated U70k PhiBench Top-5 endpoint recovers `152/203`
(`74.88%`) raw RMSD successes and `156/203` (`76.85%`) refined RMSD successes.
Its refined Top-5 PB-valid rate is `196/203` (`96.55%`) and its same-pose
PB-valid/RMSD joint success is `150/203` (`73.89%`). The immutable result is
[`phibench_u70k_top5.json`](phibench_u70k_top5.json).

## Contract-aware comparison views

### FoldBench protocol matrix

The FoldBench panels remain separate because cofolding methods are not valid
baseline rows for supplied-pocket docking.

| Contract | EFF-Dock FoldBench-Pocket | FoldBench source-native leaderboard |
|---|---|---|
| Prediction task | Holo-receptor redocking | Complete-complex cofolding |
| Receptor | Experimental holo coordinates supplied | Predicted by the model |
| Pocket | Crystal pocket supplied | No crystal-pocket input |
| Selection | U70k confidence Top-1 | Model-native rank |
| Success endpoint | Symmetry LRMSD `<2 A`; separate PB conjunction | LRMSD `<2 A` and LDDT-PLI `>0.8` |
| Directly comparable | No | No |

### PhiBench pocket-guided view

| Method | Cohort | Receptor/input | Endpoint | RMSD `<2 A` | Joint PB-valid | Comparable |
|---|---:|---|---|---:|---:|:---:|
| EFF-Dock U70k | Derived 203 | Experimental holo receptor + crystal pocket | Refined confidence Top-1 | 64.53% | 59.11% | No |
| EFF-Dock U70k | Derived 203 | Experimental holo receptor + crystal pocket | Refined confidence Top-5 | 76.85% | 73.89% | No |
| PhysDock | Source-native 206 | Paper pocket prior | Paper pocket-guided Top-5 | 83.0% | 77.7% | No |
| SurfDock | Source-native 206 | Paper pocket prior | Paper pocket-guided Top-5 | 71.7% | 71.2% | No |
| Interformer | Source-native 206 | Paper pocket prior | Paper pocket-guided Top-5 | 68.3% | 63.3% | No |
| Uni-Mol Docking V2 | Source-native 206 | Paper pocket prior | Paper pocket-guided Top-5 | 53.3% | 52.2% | No |
| DiffDock-L | Source-native 206 | Paper pocket prior | Paper pocket-guided Top-5 | 39.8% | 35.5% | No |
| Glide | Source-native 206 | Paper pocket prior | Paper pocket-guided Top-5 | 25.4% | 24.3% | No |
| AutoDock Vina | Source-native 206 | Paper pocket prior | Paper pocket-guided Top-5 | 23.7% | 22.7% | No |
| DiffDock | Source-native 206 | Paper pocket prior | Paper pocket-guided Top-5 | 30.7% | 25.8% | No |

## PhiBench source-native results

PhysDock Figure 2 reports results on the authors' PhiBench set (`N=206`).
`Pocket prior` follows the asterisk notation in that figure. The paper
describes the pocket-guided endpoint as Top-5 docking success, so it is not an
EFF-Dock confidence-selected Top-1 comparison.

| Method | Pocket prior | PAL-RMSD `<2 A` | PB-valid + PAL-RMSD `<2 A` |
|---|:---:|---:|---:|
| PhysDock | No | 61.2% | 49.5% |
| AlphaFold 3 | No | 51.0% | 47.6% |
| Chai-1 | No | 44.7% | 42.2% |
| DynamicBind | No | 54.3% | 53.3% |
| NeuralPLexer | No | 19.9% | 16.7% |
| PhysDock | Yes | 83.0% | 77.7% |
| SurfDock | Yes | 71.7% | 71.2% |
| Interformer | Yes | 68.3% | 63.3% |
| Uni-Mol Docking V2 | Yes | 53.3% | 52.2% |
| DiffDock-L | Yes | 39.8% | 35.5% |
| Glide | Yes | 25.4% | 24.3% |
| AutoDock Vina | Yes | 23.7% | 22.7% |
| DiffDock | Yes | 30.7% | 25.8% |

Source: [PhysDock preprint](https://doi.org/10.1101/2025.04.28.650887) and
[official Figure 2 at commit `7c26bff`](https://github.com/KexinZhangResearch/PhysDock/blob/7c26bffddde30856cd180a499be93f60d93aadfa/figs/F2.pdf).
The PDF SHA-256 used for extraction is
`a9a1ffdac55e817ac28d4e19675fc8640fee5a20e5e96b241eea7d04e24e59ff`.
Values were recovered from the vector bar extents and rounded to one decimal;
the four PhysDock headline values (`61.2`, `49.5`, `83.0`, `77.7`) were also
cross-checked against the manuscript text.

### Comparison boundary

EFF-Dock's `64.53%` refined Top-1 and `76.85%` refined Top-5 values cannot be
claimed to
beat or trail the table above because the cohort (`203` versus `206`),
candidate budget, ranking endpoint, pocket construction, and receptor output
contract differ. A defensible direct comparison requires rerunning each method
on the frozen EFF-Dock 203-system manifest or evaluating EFF-Dock under the
authors' exact 206-system protocol.

## FoldBench source-native leaderboard

FoldBench evaluates full complex cofolding, not crystal-pocket redocking.
Protein-ligand success requires both ligand RMSD `<2 A` and LDDT-PLI `>0.8`.

### Targets released after 2023-01 (full set)

| Model | Protein-ligand success |
|---|---:|
| AlphaFold 3 | 64.90% |
| Boltz-1 | 55.04% |
| Chai-1 | 51.23% |
| HelixFold 3 | 51.82% |
| Protenix | 50.70% |
| OpenFold 3 preview | 44.49% |

### Targets released after 2024-01

| Model | Protein-ligand success |
|---|---:|
| AlphaFold 3 | 67.59% |
| Boltz-1 | 51.33% |
| Chai-1 | 49.28% |
| HelixFold 3 | 50.68% |
| Protenix | 53.25% |
| OpenFold 3 preview | 40.85% |
| Boltz-2* | 53.90% |
| RosettaFold3* | 57.28% |

`*` FoldBench flags these models because their training cutoff is later than
the benchmark reference date (2023-01-13); their values are reference-only
under the benchmark's leakage policy.

Source: [FoldBench official leaderboard at commit `4273f68`](https://github.com/BEAM-Labs/FoldBench/blob/4273f6877d82bd0b2fa476d1b2f34d121cbccc70/README.md)
and the [FoldBench paper](https://doi.org/10.1038/s41467-025-67127-3).

### Comparison boundary

EFF-Dock's `75.63%` full-cohort value is symmetry-aware refined Top-1 RMSD
success with an experimental holo receptor and crystal pocket. Its stricter
PB-valid conjunction is `72.94%`, but PoseBusters validity is not LDDT-PLI.
FoldBench's values are joint LRMSD/LDDT-PLI success from source-native complex
cofolding. The values must not be used as a direct ranked comparison.

## Recommended paper usage

- Main table: report only EFF-Dock's frozen derived-cohort results and state
  the exact pocket-redocking contract.
- Related-work or appendix table: include the source-native tables above with
  separate `N`, receptor input, pocket prior, ranking endpoint, and metric
  columns.
- Strong direct baseline claim: run selected external models on the exact
  EFF-Dock manifests and score every output with the same symmetry/PoseBusters
  pipeline.
