# GuidanceEnergy and confidence selection results

- Protocol: `EFFDOCK-GUIDANCE-ENERGY-CONFIDENCE-SELECTION-V1`
- Status: complete post-hoc descriptive characterization; not production-admitted
- Cohort: 85 Astex Diverse and 308 PoseBusters v2 complexes, 100 poses each
- Runtime job: `53637` (`cpu_only`, one CPU, no GPU, exit 0, 26 seconds)
- Full fixed-grid artifact:
  `outputs/benchmarks/guidance_sdf_post_refinement_runs/20260812T044100Z/full/energy_confidence_selection_v1`

## Why compare a combined selector

The current selector minimizes confidence-predicted RMSD. It is trained to
recognize native-like geometry, but does not explicitly optimize steric or
protein-ligand validity. GuidanceEnergy provides that complementary signal,
but energy alone can prefer a physically comfortable non-native pose. The
comparison therefore asks whether energy can act as a bounded secondary signal
without replacing confidence.

Confidence is in predicted Angstrom RMSD while GuidanceEnergy is a
ligand- and contact-dependent energy sum. Adding or multiplying their raw
values would make the coefficient depend on molecular size and score scale.
The frozen comparison instead converts each signal to a within-complex ordinal
quality and evaluates additive, geometric, and filter combinations.

## Primary step-100 comparison

`rank_geo_a10` assigns 90% of the ordinal weight to confidence and 10% to
GuidanceEnergy. It is shown as a descriptive light-fusion reference, not a
selected production setting.

| Dataset | Selector | RMSD <2A | PL-valid | Joint | Median RMSD |
|---|---|---:|---:|---:|---:|
| Astex | confidence | 70/85 (82.35%) | 80/85 (94.12%) | 67/85 (78.82%) | 0.959 A |
| Astex | rank_geo_a10 | 70/85 (82.35%) | 81/85 (95.29%) | 67/85 (78.82%) | 0.927 A |
| PoseBusters | confidence | 217/308 (70.45%) | 282/308 (91.56%) | 204/308 (66.23%) | 1.188 A |
| PoseBusters | rank_geo_a10 | 224/308 (72.73%) | 296/308 (96.10%) | 218/308 (70.78%) | 1.133 A |

Against confidence alone, light geometric fusion leaves Astex RMSD and joint
counts unchanged while adding one PL-valid complex. On PoseBusters it changes
126 selected indices, produces 19 RMSD-success gains versus 12 losses, and 24
joint-success gains versus 10 losses. The net changes are `+2.28 pp` RMSD SR,
`+4.55 pp` PL-validity, and `+4.55 pp` joint success.

The closely related additive 10% fusion gives Astex 82.35% RMSD SR and 78.82%
joint, and PoseBusters 72.40% RMSD SR and 70.45% joint. Thus the signal is not
unique to one arithmetic operator in this fixed grid.

## Why energy should remain secondary

Energy-only step-100 selection raises PL-validity but loses native-pose
selection on Astex: compared with confidence it changes RMSD SR from 82.35% to
71.76% and joint success from 78.82% to 68.24%. On PoseBusters it changes RMSD
SR from 70.45% to 68.83%, although joint success rises from 66.23% to 67.53%.
Likewise, increasing the energy share to 75% lowers Astex joint success to
72.94% for additive fusion and 71.76% for geometric fusion. The measured role
of GuidanceEnergy is therefore a soft tie-break/reranking signal, not a
replacement objective.

At step 0, before physical refinement, energy-heavy selectors substantially
increase validity and joint success because the pose pool still contains many
physical failures. At step 100, most poses have already been repaired, so the
remaining useful information is smaller and must be balanced against native
pose recognition.

## Decision boundary

These external benchmark outcomes were already visible when this analysis was
designed. They characterize the available signal but cannot select a formula,
coefficient, filter, or production default. A separately frozen internal
PLINDER calibration should compare confidence-only with the complete light
fusion candidates, choose the selector there, and then use Astex and
PoseBusters only as confirmation sets.
