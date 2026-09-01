# Raw GuidanceEnergy and confidence selection results

- Protocol: `EFFDOCK-GUIDANCE-ENERGY-CONFIDENCE-RAW-SELECTION-V1`
- Status: complete post-hoc descriptive characterization; not production-admitted
- Cohort: 85 Astex Diverse and 308 PoseBusters v2 complexes, 100 poses each
- Runtime job: `53641` (`cpu_only`, one CPU, no GPU, exit 0, 21 seconds)
- Full fixed-grid artifact:
  `outputs/benchmarks/guidance_sdf_post_refinement_runs/20260812T044100Z/full/raw_energy_confidence_selection_v1`

## Candidate-count-invariant score

The size-normalized raw selector minimizes

`S_i = confidence_predicted_RMSD_i + lambda * GuidanceEnergy_i / N_heavy`.

It uses no candidate rank, percentile, mean, variance, minimum, maximum, or
candidate count. An existing pose therefore keeps exactly the same score when
the candidate set changes. Adding or removing poses can change Top-1 only by
adding a better-scoring pose or removing the selected pose.

The direct-total control instead minimizes

`S_i = confidence_predicted_RMSD_i + lambda * GuidanceEnergy_i`.

It is also candidate-count invariant, but makes the effective energy influence
grow with ligand size.

## Primary step-100 characterization

| Dataset | Selector | RMSD <2A | PL-valid | Joint | Median RMSD |
|---|---|---:|---:|---:|---:|
| Astex | confidence | 70/85 (82.35%) | 80/85 (94.12%) | 67/85 (78.82%) | 0.959 A |
| Astex | per-atom lambda 0.05 | 69/85 (81.18%) | 80/85 (94.12%) | 66/85 (77.65%) | 1.017 A |
| Astex | per-atom lambda 0.25 | 69/85 (81.18%) | 81/85 (95.29%) | 66/85 (77.65%) | 0.997 A |
| Astex | per-atom lambda 1.00 | 67/85 (78.82%) | 81/85 (95.29%) | 64/85 (75.29%) | 0.987 A |
| PoseBusters | confidence | 217/308 (70.45%) | 282/308 (91.56%) | 204/308 (66.23%) | 1.188 A |
| PoseBusters | per-atom lambda 0.05 | 219/308 (71.10%) | 285/308 (92.53%) | 209/308 (67.86%) | 1.196 A |
| PoseBusters | per-atom lambda 0.25 | 222/308 (72.08%) | 293/308 (95.13%) | 217/308 (70.45%) | 1.181 A |
| PoseBusters | per-atom lambda 1.00 | 225/308 (73.05%) | 293/308 (95.13%) | 219/308 (71.10%) | 1.100 A |

Even the weakest tested per-atom correction changes only five Astex indices,
but one is a native-pose success loss, giving `-1.18 pp` RMSD and joint. On
PoseBusters it gives five RMSD gains versus three losses and eight joint gains
versus three losses. At lambda 0.25, PoseBusters has 11 RMSD gains versus six
losses and 18 joint gains versus five losses, while Astex still has one joint
loss.

The largest tested direct-total coefficient produces the strongest
PoseBusters row in that family (`73.70%` RMSD SR, `95.78%` PL-valid, `71.75%`
joint) but loses two Astex RMSD/joint successes and remains ligand-size
sensitive. It is therefore an informative control rather than a portable
score definition.

## Comparison with within-complex rank fusion

Light geometric rank fusion previously preserved Astex RMSD/joint counts and
raised PoseBusters joint from `66.23%` to `70.78%`, but its numerical score
changes with candidate count. Raw per-atom selection removes that dependency
and still improves PoseBusters, but every nonzero frozen coefficient loses at
least one Astex success. The two methods therefore expose a real trade-off:
rank fusion is empirically more stable on these two opened datasets, while raw
per-atom scoring has the cleaner deployment invariant.

## Interpretation and next gate

Raw combination is feasible, but total GuidanceEnergy mixes very large
repulsive penalties, ligand-internal geometry, and favorable interaction
terms. A single global coefficient cannot distinguish a clash penalty from an
interaction reward. The next internal calibration should retain the
candidate-count-invariant form while using fixed, per-term size/site
normalization or a simple validity calibrator fitted only on PLINDER. Astex and
PoseBusters cannot choose the coefficient because their outcomes were already
opened before this characterization.
