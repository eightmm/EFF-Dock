# OpenBind official-style Top-25 U50 reporting addendum

Protocol ID: `EFFDOCK-OPENBIND-OFFICIAL-TOP25-U50-REPORT-V1`

The metric implementation and denominator remain
`EFFDOCK-OPENBIND-OFFICIAL-TOP25-V1`. This addendum changes the source ranking
ledger only:

- selector checkpoint SHA-256:
  `fd49fa86f67187bf26d6c1bcf2daf925ba3e3b19dfeae733e57535d183280469`;
- ranking: ascending U50 `after_confidence_rmsd`, stable by pose index;
- frozen candidate coordinates and adaptive-refinement outputs are identical
  to the earlier U25 aggregation;
- PoseBusters 0.6.5 and OpenStructure 2.11.1 are rerun on the U50-ranked Top-25
  poses because the ranked subset can change.

The earlier U25 OpenBind outcomes were already visible. This U50 reranking is
therefore a descriptive reporting update, not a blind selector-selection
experiment.
