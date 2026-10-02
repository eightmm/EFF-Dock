# Benchmark results

Completed corrected Rw, frozen released docking/confidence weights, unguided N100/S10, 10 Å supplied pocket, physical/interaction energy refinement, input-chirality-filtered minimum-pRMSD selection. Seeds 42, 100042 and 200042. Every original complex is retained; benchmark statistics are derived from the completed study without new cohort inference, training, reference-based selection or tuned thresholds. The separate fixed-complex Rw N1 workflow illustration is not included in benchmark statistics.

## Primary selected-pose performance

| Dataset | N per seed | RMSD <2 Å (%) | PB-valid (%) | RMSD <2 Å & PB-valid (%) |
|---|---:|---:|---:|---:|
| Astex Diverse Set | 85 | 83.14 ± 1.80 | 95.69 ± 0.68 | 80.00 ± 2.04 |
| PoseBusters v2 | 308 | 81.82 ± 0.56 | 95.02 ± 0.50 | 79.00 ± 0.82 |
| PhiBench | 206 | 62.94 ± 3.23 | 93.69 ± 0.00 | 60.36 ± 3.08 |
| FoldBench | 558 | 75.27 ± 0.54 | 96.54 ± 0.10 | 73.78 ± 0.63 |
| OpenBind | 925 | 53.19 ± 0.56 | 99.68 ± 0.19 | 53.19 ± 0.56 |

Values are three-seed means and sample SD on fixed cohorts. The endpoint uses the same selected pose for RMSD and all 27 non-RMSD PoseBusters 0.6.5 checks. Temporal cohorts retain their audited InChI energy-reference compatibility path; Astex/PoseBusters use native official redock.

Current primary figures use corrected Rw with unchanged released weights. Legacy guided/budget and pocket/prior panels are excluded from the current submission package; runtime and illustrations also use Rw.

[Figure PDF](paper/paper_figures.pdf) · [Captions](paper/FIGURE_CAPTIONS.md) · [Evidence and limitations](paper/EVIDENCE.md)
