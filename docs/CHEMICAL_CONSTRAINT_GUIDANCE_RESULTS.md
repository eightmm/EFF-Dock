# Chemical-constraint guidance mechanistic results

Status: complete mechanistic characterization; **not a benchmark estimate or
production-setting selection**.

## Information taxonomy

`ChemicalConstraintEnergy` is limited to coordinate states explicitly declared
by the inference ligand graph or isomeric SMILES. V1 implements
heavy-atom-observable tetrahedral chirality and E/Z geometry.

- Exact ligand-declared invariants -> `ChemicalConstraintEnergy`
- Generic coordinate feasibility, strain, and sterics -> `PhysicalEnergy`
- Protein-conditioned pose preferences -> `InteractionEnergy`

SMILES does not determine a rotatable-bond conformer or protein-frame binding
orientation. Those quantities are never encoded as chemical constraints.

## Paired setup

- Complexes: `7sdd_4ip`, `6xht_v2v`, `8f4j_pho`
- Sampling: S10, N100, prior sigma 2.0, seed 42
- Checkpoint: `weights/effdock_docking_early_time_t0p10_50k.pt`
- Existing guidance: normalized drift, eta 2.0, start t=0.5, ramp power 1
- Chemical constraint: separately normalized drift, the same shared start
  t=0.5, ramp power 1, per-step displacement cap 0.10 A
- Strength sweep: 0, 0.10, 0.25, 0.50
- Pairing: identical prior-pool SHA256 across every arm within each complex
- Completed run: Slurm array `65274`, all tasks exit 0 on `heavy`

The three cases were selected after external outcomes were opened. The sweep is
therefore descriptive only and cannot select a production strength. An internal
PLINDER validation/confirmation split remains required.

## V3 separate-drift result

| Strength | Stereo-valid / 300 | RMSD < 2 A / 300 | Stereo + RMSD < 2 A / 300 | Fast-valid / 300 | Fast + stereo / 300 |
|---:|---:|---:|---:|---:|---:|
| 0.00 | 50 | 22 | 5 | 47 | 7 |
| 0.10 | 62 | 22 | 6 | 47 | 10 |
| 0.25 | 64 | 22 | 6 | 47 | 10 |
| 0.50 | 65 | 22 | 6 | 47 | 10 |

Per-complex stereo-valid counts were:

| Complex | 0.00 | 0.10 | 0.25 | 0.50 |
|---|---:|---:|---:|---:|
| `7sdd_4ip` | 20 | 24 | 25 | 26 |
| `6xht_v2v` | 23 | 30 | 31 | 31 |
| `8f4j_pho` | 7 | 8 | 8 | 8 |

Strength 0.10/0.25/0.50 caused 1,531/2,049/2,119 constraint-channel
cap triggers across the three runs. All reached the frozen 0.10 A displacement
cap. The weak incremental gain above 0.10 is therefore a saturated-cap response,
not evidence that 0.50 is a better setting.

The exact symmetry-aware RMSD-under-2-A candidate count stayed at 22 for every
arm, and mean RMSD changed by less than 0.003 A per complex. The intervention
therefore improved declared-stereo preservation in these cases without a visible
pose-accuracy change, but it did not repair their broader docking failures.

## Unit-weights paired inference

Protocol: `EFFDOCK-CHEMICAL-CONSTRAINT-UNIT-WEIGHTS-SMOKE-V1`.
Slurm sampling array `65281` and CPU report job `65283` completed with exit 0.
This comparison holds the physical-plus-interaction outer normalized-drift
strength at 1 and changes only the separately normalized chemical-constraint
strength from 0 to 1. Term-level physical and interaction constants remain the
versioned values; they are not replaced with unit constants.

| Arm | Stereo-valid / 300 | RMSD < 2 A / 300 | Stereo + RMSD < 2 A / 300 | Fast-valid / 300 | Fast + stereo / 300 |
|---|---:|---:|---:|---:|---:|
| Physical + Interaction = 1, Chemical = 0 | 50 | 22 | 5 | 47 | 7 |
| Physical + Interaction = 1, Chemical = 1 | 70 | 21 | 6 | 45 | 11 |

Per-complex changes from chemical 0 to chemical 1 were:

| Complex | Stereo-valid | RMSD < 2 A | Fast-valid |
|---|---:|---:|---:|
| `7sdd_4ip` | 20 -> 31 | 4 -> 4 | 36 -> 33 |
| `6xht_v2v` | 23 -> 31 | 18 -> 17 | 11 -> 12 |
| `8f4j_pho` | 7 -> 8 | 0 -> 0 | 0 -> 0 |

The paired mean coordinate RMSD between the two arms was 0.0508 A and the
largest candidate displacement was 0.4232 A. The chemical channel triggered
its 0.10-A per-step cap on all 2,258 active pose-step applications. Thus the
observed unit-strength response is completely cap-saturated. These are fast
validity diagnostics, not official 27-check PoseBusters results, and no
confidence selector was applied.

Machine-readable record:
`outputs/benchmarks/chemical_constraint_unit_weights_v1/report.json`.

## Superseded V1 summed-energy result

The earlier implementation added a scale-1 stereo barrier before the common
physical/interaction normalization. It changed the outputs by only 0.011--0.020 A
on average and moved total stereo-valid count from 50 to 53 of 300. That result
motivated the V3 separately normalized channel and must not be compared as if the
two scale parameters had the same meaning.

## Operational notes

- Job `65269` was cancelled while pending on `6000ada`; it produced no result.
- Job `65271` on `test` encountered 11 GB GPU OOM before completing the paired
  arms; its partial V2 output is non-claim-bearing and ignored.
- Job `65277` completed an exploratory chemical-start-t=0 run before the shared
  t=0.5 decision. Its V4 output is superseded and is not used for any claim.
- Machine-readable V3 records:
  `outputs/benchmarks/chemical_constraint_guidance_smoke_v3/report.json`.

## Decision

The separate channel is retained as an opt-in diagnostic because its mechanism
works and is more effective than summed-energy scaling. Its default strength
remains zero, and every unified-guidance channel now shares the t=0.5 start gate.
No nonzero value is production-admitted until an internal validation protocol
tests stereo preservation, PoseBusters validity, pose accuracy, cap rate, and
runtime together.
