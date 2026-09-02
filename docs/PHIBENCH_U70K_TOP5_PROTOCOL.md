# PhiBench U70k Top-5 descriptive reanalysis

Protocol ID: `EFFDOCK-PHIBENCH-U70K-TOP5-V1`

Status: frozen before aggregate Top-5 outcomes are computed.

## Purpose

Report EFF-Dock under a Top-5 output budget because the PhysDock PhiBench
pocket-guided panel uses a paper-defined Top-5 endpoint. This is a descriptive
reanalysis of the already generated U70k 100-pose bank; it does not change the
production Top-1 selector or checkpoint.

## Frozen inputs

- Cohort: the existing deterministic PhiBench-derived `N=203` reconstruction.
- Source report SHA-256:
  `3365e59753b13464f4911d28f0983e121f3f7be3ea00521f6942a17e167071bf`.
- Candidate bank: the same saved `N=100`, `S=10`, `sigma=2` bank used by the
  released PhiBench U70k result.
- Confidence checkpoint: U70k, SHA-256
  `ce59be42f0ca613871ca079127c3296f5ca9a4ec72e44a9e5cf61878351c2638`.
- Refinement: the same saved adaptive trajectories with at most 100 steps.
- RMSD: symmetry-aware ligand heavy-atom RMSD without alignment.
- Validity: PoseBusters `0.6.5`, `redock`, with RMSD excluded from the
  pass-all validity conjunction.

Every confidence summary, score CSV, refinement summary, trajectory SDF,
protein, and reference ligand is hash-checked before use.

## Ranking and endpoints

Raw and refined poses are ranked independently by ascending predicted RMSD,
with pose index as the stable tie-breaker. Reference RMSD and PoseBusters
outcomes never enter ranking.

For each complex, report:

- Top-1 and Top-5 RMSD success, where Top-5 succeeds if any of the first five
  ranked poses has symmetry RMSD `<2 A`;
- refined Top-5 PB-validity, where any of the first five ranked poses may pass;
- refined Top-5 joint success, where the same one of the first five poses must
  be both PB-valid and symmetry RMSD `<2 A`;
- 100-pose refined RMSD oracle as a separate headroom diagnostic.

The full `N=203` denominator is retained. Missing, changed, malformed, or
unevaluable inputs fail the job rather than shrinking the denominator.

## Gates

- Exactly 203 unique complexes and 100 finite score rows per complex.
- Recomputed U70k Top-1 counts must reproduce raw `128/203`, refined
  `131/203`, refined PB-valid `184/203`, and refined joint `120/203`.
- Top-1 success must be no greater than Top-5 success, which must be no
  greater than the 100-pose oracle.
- All 1,015 refined Top-5 poses must receive complete PoseBusters results.

## Interpretation boundary

The PhysDock source-native panel contains 206 systems, while this reproducible
reconstruction contains 203. Pocket construction and exact output/ranking
contracts may also differ. The resulting Top-5 values are therefore the
closest available endpoint-aligned context, not a direct head-to-head claim.
Because PhiBench outcomes were already opened, this analysis cannot tune or
select a checkpoint, candidate budget, refinement, or selector.
