# EFF-Dock benchmark results

Protocol: `EFFDOCK-REDOCK-EMA-N40-S25-V1`

Completed: 2026-07-19 (Asia/Seoul)

Scope: reference-defined oracle-pocket redocking diagnostic.

These are not target-independent pocket-finding or prospective inference
results. The exact frozen protocol and source hashes are in
`BENCHMARK_PROTOCOL.md`.

## Physical-selector compatibility baseline

All rates are percentages of the full frozen dataset. RMSD success uses
symmetry-aware heavy-atom RMSD <2A. `Vina+DG` is the selected top-1 pose;
`oracle-40` is the best RMSD among the same 40 samples and is not a deployable
selector.

| Dataset | N | First pose <2A | Vina+DG top-1 <2A | Improvement | Oracle-40 <2A | Vina median RMSD | Vina fast-valid | Final failures |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Astex Diverse | 85 | 35.29 | **64.71** | +29.41 pp | 94.12 | 1.432A | 90.59 | 0 |
| PoseBusters v2 | 308 | 31.82 | **64.29** | +32.47 pp | 91.56 | 1.446A | 74.68 | 0 |
| CASF-2016 | 285 | 31.58 | **59.30** | +27.72 pp | 88.07 | 1.562A | 84.21 | 0 |

Absolute Vina+DG <2A successes are 55/85, 198/308, and 169/285. Oracle-40
successes are 80/85, 282/308, and 251/285. All 678 complexes have final rows.

| Dataset | Vina <1A | Vina <3A | Vina mean RMSD |
|---|---:|---:|---:|
| Astex Diverse | 41.18 | 77.65 | 2.106A |
| PoseBusters v2 | 31.49 | 80.52 | 2.062A |
| CASF-2016 | 34.04 | 77.19 | 2.180A |

## Official PoseBusters validity

PoseBusters 0.6.5 `redock` was run on all 308 Vina+DG-selected PoseBusters
poses. The pass-all structural/chemical validity rate, excluding the RMSD
criterion, is **60.71% (187/308)**. The official RMSD criterion independently
reproduces 64.29%.

Important component pass rates are protein minimum-distance 76.30%,
tetrahedral chirality 73.05%, internal steric clash 97.73%, internal energy
98.70%, bond lengths 99.03%, and bond angles 99.03%. Protein volume-overlap
passes 100%. The self-contained fast-valid subset is reported separately and
must not be called the official PoseBusters pass rate.

## Pre-registered decision

The hypothesis passed all frozen thresholds:

- PoseBusters oracle-40 91.56% is above the pre-registered 85% threshold.
- Astex oracle-40 94.12% is above the pre-registered 90% threshold.
- PoseBusters Vina+DG improves over first-pose order by 32.47 percentage
  points, above the pre-registered 5-point threshold.
- Every dataset has 0% unresolved failures, below the 2% invalidation limit.

The retained EMA model therefore has strong sampling coverage even when the
trained confidence stack is ablated. This is an ablation/compatibility baseline,
not the final EFF-Dock selector result. The remaining Vina-to-oracle gap is the
pose-selection opportunity addressed by the retained confidence model below.

## Retained trained confidence model

The selected confidence asset is
`weights/effdock_confidence_extmatch_n80_s25_step42500.pt`, paired with
`weights/effdock_geometry_ft_100k_best.pt`. It was trained on generated PLINDER
pose sets matched to N80/S25/sigma0.5/pocket10 inference. A 500-step hard-pair
fine-tune did not improve the frozen validation metric, so step 42500 remains
the selected checkpoint.

Historical full single-run evidence, recovered from the original per-pose JSONL
and frozen selector artifacts, is:

| Dataset | N | Pure confidence <2A | Frozen confidence selector <2A | Oracle-80 <2A |
|---|---:|---:|---:|---:|
| Astex Diverse | 85 | 78.82 | **81.18** | 92.94 |
| PoseBusters v2 | 308 | 73.38 | **77.60** | 94.16 |

These historical values used the same model pair and N80 preset but predated
the active EFF-Dock frozen pocket manifests. They are context, not substituted
for the active three-dataset rerun defined in
`CONFIDENCE_BENCHMARK_PROTOCOL.md`.

The later cluster-free filter study applied its already-frozen rules to these
same historical single-run candidate artifacts:

| Dataset | N | Pure confidence <2A | Strict filter <2A | Atom-RMSD guard <2A | Historical composite <2A |
|---|---:|---:|---:|---:|---:|
| Astex Diverse | 85 | 78.82 (67) | 78.82 (67) | 78.82 (67) | **81.18 (69)** |
| PoseBusters v2 | 308 | 73.38 (226) | 73.38 (226) | 74.03 (228) | **77.60 (239)** |

The strict filter changed 3/85 Astex and 10/308 PoseBusters selections without
changing either <2A count. The atom guard changed 29/85 and 113/308, preserving
Astex and adding two net PoseBusters successes, but it had already failed the
PLINDER validation gate and is not promoted. These are RMSD-selection results,
not official PoseBusters pass-all validity; the stored fast-validity proxy was
100% for all selectors and is not substituted for an official SDF-based run.
Machine artifacts are under
`outputs/eff-dock/confidence-filter-v1-external/`.

### Active frozen-manifest N80 rerun

All selectors below consume the same 80 sampled poses per complex. The
composite is the pre-registered
`pair_gate_density_rank_vote_plclash_ambig` policy; oracle-80 is diagnostic.

| Dataset | N | First <2A | Vina+DG <2A | Pure confidence <2A | Frozen composite <2A | Composite vs Vina | Oracle-80 <2A |
|---|---:|---:|---:|---:|---:|---:|---:|
| Astex Diverse | 85 | 32.94 | 77.65 | 76.47 | **78.82** | +1.18 pp | 95.29 |
| PoseBusters v2 | 308 | 35.71 | 71.10 | **73.05** | 72.73 | +1.62 pp | 94.81 |
| CASF-2016 | 285 | 32.98 | **69.47** | **69.47** | 68.42 | -1.05 pp | 91.93 |

Absolute composite successes are 67/85, 224/308, and 195/285. Pure confidence
has 488 <2A successes across all 678 complexes, versus 486 for the composite
and 483 for Vina+DG. Composite median RMSD is 1.074A, 1.293A, and 1.299A;
failure rate is 0% after two recorded Astex numerical rescues.

Official PoseBusters 0.6.5 `redock` on all 308 composite-selected poses gives
**54.87% (169/308)** pass-all validity excluding RMSD. Component pass rates
include protein minimum-distance 69.16%, tetrahedral chirality 78.25%, internal
steric clash 95.78%, internal energy 98.70%, bond lengths 100%, and bond angles
97.73%. The official RMSD criterion reproduces 72.73%.

The pre-registered primary prediction passed: PoseBusters composite improved
over same-candidate Vina by 1.62 percentage points. The historical-reproduction
prediction failed: Astex was 2.35 points below 81.18%, and PoseBusters was 4.87
points below 77.60%, outside the ±2-point criterion. CASF also shows that the
composite is not uniformly stronger than Vina or pure confidence. Therefore the
trained checkpoint remains active, while any selector recalibration must be a
new validation-only study rather than post-hoc tuning on these benchmarks.

Machine artifacts are `outputs/benchmarks/confidence/summary.json`, combined
rows under `outputs/benchmarks/confidence/combined/`, and ledger metrics under
`outputs/benchmarks/confidence/ledger_metrics.json`. Inference arrays were
`38752`, `38760`, and `38761`; Astex rescues were `38770`/`38771`; official
PoseBusters validity was `38781`. The recorded active code-tree SHA256 is
`5ffe2c0d3abec7748e59c134efd61997893727632304409d5559577eaf886cf9`.

## Slices and applicability

The machine summary records per-dataset heavy-atom, active fragment,
rotatable-bond, and receptor cofactor slices. On PoseBusters, Vina+DG <2A is
70.00% for ligands with at most 20 heavy atoms but 36.00% above 40 atoms; it is
68.14% for at most 3 rotatable bonds and 48.48% for 8 or more. This identifies
large/flexible ligands as the clearest compatibility weakness.

Train-neighbor ligand similarity and pocket similarity are deliberately not
reported yet: the preserved compatibility split predates the strict
three-benchmark exclusion contract. Those applicability-domain slices require
the new strict split and will belong to a publishable target-independent
evaluation, not this compatibility diagnostic.

## Execution and artifacts

- EMA checkpoint SHA256:
  `3ee604ec2338532532fa23a2ae91d0d540322defc32f5e453c8e7e12e389d36a`
- Config SHA256:
  `39aa62e4a48ed6f3aa4ff59345fb43a81220e2baba22edfd5beb0c4981b307ec`
- Final active code-tree SHA256:
  `9ec1ff3edfc61abc2b8127a3800293aaab4c2bba2005428a24b6846440846101`
- Slurm arrays: Astex `38717`, PoseBusters `38718`, CASF `38719`, official
  PoseBusters `38727`; CASF `1mq6` rescue `38749`.
- Runtime: PyTorch 2.10.0+cu130 / CUDA 13.0 on RTX 6000 Ada and H100 PCIe.
- Machine summary: `outputs/benchmarks/summary.json`.
- Per-complex combined rows: `outputs/benchmarks/combined/{astex,posebusters,casf}.csv`.
- Selected pose SDFs, shard summaries, official PoseBusters rows, and Slurm
  logs remain under ignored `outputs/benchmarks/`.

One CASF complex (`1mq6`) initially hit a CUDA `linalg.eigh` convergence error
on an RTX 6000 Ada shard. It was rerun alone on H100 with the identical global
dataset seed and hashes and succeeded; both the original failure and rescue
are retained in the raw shard summaries and aggregate `rescued_failures`.

CASF uses frozen CASF ligand/reference coordinates with current RCSB receptors;
two structures (`4tmn`, `5tmn`) required recorded rigid transforms into the
CASF ligand frame before exact reference-residue removal. This receptor source
difference is another reason to treat CASF here as a compatibility diagnostic,
not a canonical CASF leaderboard submission.
