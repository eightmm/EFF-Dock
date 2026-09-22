# Benchmark results

The current manuscript analysis uses three seeds with unguided N100/S10,
translation sigma 2.0, the released docking/U70k confidence pair, explicit
physical refinement and input-chirality selection. These postprocessing steps
are separate from the default `dock()` call.

Success uses symmetry-aware heavy-atom RMSD <2 Å without alignment. PB-valid
success requires both the RMSD cutoff and PoseBusters validity for the same
selected pose. Values below are percentages, mean ± sample SD over three seeds.

| Dataset | N per seed | RMSD <2 Å | PB-valid poses | PB-valid success |
|---|---:|---:|---:|---:|
| Astex Diverse Set | 85 | 82.35 ± 2.04 | 95.69 ± 0.68 | 79.22 ± 2.72 |
| PoseBusters v2 | 308 | 81.60 ± 0.94 | 95.56 ± 0.50 | 79.22 ± 1.42 |
| PhiBench reconstructed full | 206 | 62.62 ± 3.79 | 94.01 ± 0.28 | 60.19 ± 3.36 |
| FoldBench-Pocket full | 558 | 74.49 ± 0.37 | 96.36 ± 0.37 | 72.82 ± 0.63 |
| OpenBind full | 925 | 52.58 ± 0.61 | 99.64 ± 0.17 | 52.58 ± 0.61 |

These supplied-pocket redocking cohorts were inspected during development;
the results are descriptive. U70k was selected on the fixed 1,035-complex
PLINDER validation bank. OpenBind is an auxiliary single-protease cohort.
FoldBench-Pocket is a holo-receptor redocking adaptation, not the native
cofolding benchmark. PhiBench uses the reconstructed 206-complex cohort.

The [manuscript captions](paper/FIGURE_CAPTIONS.md) define the conditions,
coverage and limitations of each figure. Guided/budget and pocket/prior analyses
are separate ablations, not interchangeable with the table above. FoldBench
PoseBusters evaluation includes the disclosed energy-reference InChI
compatibility repair.

- [Model and environment identities](REPRODUCIBILITY.md)
- [Manuscript figure PDF](paper/paper_figures.pdf)
- [External model comparison artifacts](../benchmarks/results/external_models/README.md)
- [Literature comparison conditions](../benchmarks/results/external_models/TEMPORAL_LITERATURE.md)

Historical single-bank and intermediate experiment reports are not the current
manuscript baseline.
