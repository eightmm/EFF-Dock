# External-model runtime accounting — 2026-08-31

Status: active; append results only after the corresponding coverage inventory
and Slurm task states have been reconciled. Queue wait time is never included.

This document accompanies
[`EXTERNAL_MODEL_OFFICIAL_INFERENCE_PROTOCOL.md`](EXTERNAL_MODEL_OFFICIAL_INFERENCE_PROTOCOL.md).
Runtime numbers use the same official model settings, full dataset denominators,
candidate counts, selectors, and refinements as the accuracy benchmark.

## Reporting contract

Runtime is not represented by a single ambiguous wall-clock number. Every model
is reported with the following quantities:

| Quantity | Definition |
|---|---|
| Input/preprocessing time | Protein, pocket, ligand, graph, and conformer preparation performed at inference time. Cached and uncached values are reported separately. |
| Pose-generation time | Learned or classical search that emits candidate coordinates. For EFF-Dock this is the fragment-prior construction plus ODE integration, before post-ODE refinement and confidence. |
| In-trajectory guidance time | Energy, autograd, projection, and correction evaluated inside the pose generator. For EFF-Dock it is timed separately from learned-model forward time even though both occur inside each ODE step. |
| Minimization/refinement time | Classical minimization, geometry projection, or EFF-Dock post-ODE physical autograd refinement, divided by the number of input poses. |
| Scoring time | Native affinity, energy, or confidence forward evaluation for all generated/refined poses. |
| Reranking time | Selector/filter/sort operation applied to already-computed scores. Score computation is not hidden in this row. |
| Serialization time | SDF/PT/JSON writing after selection. This is recorded but excluded from model-compute comparisons. |
| End-to-end resource time | Sum of preprocessing, pose generation, admitted guidance, native refinement, native scoring, reranking, and required serialization. Common RMSD and PoseBusters evaluation are excluded. |
| Resource time per complex | Sum of admitted task elapsed seconds divided by the frozen dataset denominator. This is throughput-normalized work, not interactive latency. |
| Parallel stage latency | Maximum elapsed task time in the stage when all recorded shards are allowed to run concurrently, excluding scheduler wait and dependencies. |

Additional rules:

- Use actual emitted and parseable pose counts. A requested `N=40` does not
  become 40 poses when an engine emits fewer modes.
- A failed target remains in time accounting because it consumed resources.
  Report attempted-pose and successful-output denominators separately when they
  differ.
- Keep generation, classical minimization, learned scoring, and official
  PoseBusters evaluation separate. Do not hide expensive reranking inside the
  generator's time.
- If an upstream program genuinely fuses search and scoring, report the fused
  stage explicitly; do not invent a split by subtraction. Add internal timers
  only when they do not change the model's algorithm or settings.
- EFF-Dock's primary runtime ledger uses the fixed sequence `preprocess ->
  prior/ODE -> in-ODE guidance -> post-ODE refinement -> confidence forward ->
  selector/sort -> serialization`. The official RMSD and PoseBusters passes are
  evaluation costs and never part of inference latency.
- EFF-Dock guidance is interleaved with the ODE. Its exact timer therefore
  surrounds the guidance energy/gradient/projection call at every active step;
  `guided wall time - unguided wall time` is only a secondary paired overhead
  estimate. Refinement records attempted steps, terminal/adaptive step, and
  seconds per pose. Confidence forward and the subsequent stable selection are
  different timing fields.
- Record the resource class and allocation, but do not record private node
  names. Current GPU rows use one `6000ada` accelerator per task. CPU-only rows
  below use one allocated CPU per task.
- Report each complete repeat before computing mean and sample standard
  deviation (`ddof=1`). Seed labels are provenance only; they are not treated
  as meaningful model choices.
- The final comparison will show both seconds per pose and seconds required to
  produce and select one complete candidate set per complex. This prevents an
  `N=1` method from looking artificially comparable to an `N=40` method.

## Reconciled measurements

### Astex Diverse (`N=85` complexes)

| Executed arm / stage | Candidate inventory | Repeats | Resource seconds per pose, mean ± SD | Resource seconds per complex, mean | Parallel stage latency | State |
|---|---:|---:|---:|---:|---:|---|
| RLDiff RL++ generation | 3,400 poses/repeat (`N=40`) | 3 | `1.041 ± 0.009` GPU-s | `41.64` GPU-s | `15.94 min` mean maximum shard | generation complete |
| RLDiff RL++ smina pre-score | 3,360 successful poses/repeat | 2 complete repeats measured | `0.00564 ± 0.00116` CPU-s/successful pose | `0.223` CPU-s | not isolated by existing shard logs | provisional: successful targets only |
| RLDiff RL++ smina minimization | 3,360 successful poses/repeat | 2 complete repeats measured | `0.00982 ± 0.00021` CPU-s/successful pose | `0.388` CPU-s | not isolated by existing shard logs | provisional: successful targets only |
| RLDiff RL++ GNINA CNN scoring/reranking | 3,360 successful poses/repeat | 2 complete repeats measured | `2.650 ± 0.003` CPU-s/successful pose | `104.74` CPU-s | not isolated by existing shard logs | provisional: successful targets only |
| RLDiff RL++ postprocess total | 3,400 attempted; 3,360 successful outputs/repeat | 2 complete repeats measured | `2.654 ± 0.003` CPU-s/attempted pose; `2.686` CPU-s/successful pose | `106.15` CPU-s | `39.08 min` mean maximum shard | provisional: includes launch/I/O and `1YV3_BIT` failure |
| DiffBindFR smina error correction + MDN | 3,400 poses/repeat (`N=40`) | 3 | `1.490 ± 0.051` GPU-s | `59.59` GPU-s | `23.15 min` mean maximum shard | complete; upstream task currently fuses refinement and learned scoring |
| PoseBench FABind post-optimized inference | 85 poses/repeat (`N=1`) | 3 | `1.788 ± 0.418` GPU-s | `1.788` GPU-s | `0.92 min` mean maximum shard | complete; upstream task currently fuses regression and post-optimization |
| PoseBench Vina search + native affinity | actual `2,954`, `2,960`, `2,916` poses | 3 | `3.156 ± 0.062` CPU-s | `109.26` CPU-s | `47.92 min` mean maximum shard | complete; upstream engine fuses search and scoring |

RLDiff generation repeat values were `1.043`, `1.031`, and `1.049`
GPU-s/pose. The first two currently completed RL++ postprocessing repeats were
`2.656` and `2.651` CPU-s per attempted pose. All three hit the same
`1YV3_BIT` smina incompatibility because the retained vanadate contains the
unsupported AutoDock atom type `V`; that row is not an admitted end-to-end
runtime until the receptor-policy decision and recovery are frozen.

The RLDiff split above comes from the successful-target stage lines in jobs
`61845` and `61847`: smina pre-score sums were `21.7` and `16.2` seconds,
smina minimization sums were `33.5` and `32.5` seconds, and GNINA scoring sums
were `8,910.5` and `8,895.3` seconds. GNINA therefore accounts for about
`99.4%` of the measured successful-pose postprocessing stage time. The existing
logs do not expose a trustworthy parallel critical path for each substage, so
that column remains unclaimed rather than being reconstructed from averages.

Vina repeat values were `3.100`, `3.145`, and `3.222` CPU-s per actually
emitted pose. Only `49`, `52`, and `46` of the 85 complexes emitted all 40
requested modes, so dividing by the requested `85 * 40` would understate its
true per-pose time.

### PoseBusters v2 (`N=308` complexes)

| Executed arm / stage | Candidate inventory | Repeats | Resource seconds per pose, mean ± SD | Resource seconds per complex, mean | Parallel stage latency | State |
|---|---:|---:|---:|---:|---:|---|
| DiffBindFR smina error correction + MDN | 12,320 poses/repeat (`N=40`) | 3 | `1.745 ± 0.011` GPU-s | `69.80` GPU-s | `55.58 min` mean maximum shard | complete; upstream task currently fuses refinement and learned scoring |
| PoseBench FABind post-optimized inference | 308 poses/repeat (`N=1`) | 3 | `1.443 ± 0.025` GPU-s | `1.443` GPU-s | `0.72 min` mean maximum shard | complete; upstream task currently fuses regression and post-optimization |
| RLDiff RL++ | `N=40` | — | — | — | — | running |
| PoseBench Vina | up to `N=40` | — | — | — | — | running |

### EFF-Dock fixed-NFE reconstruction (`S25/N40`, one frozen run)

This is a descriptive reconstruction of the completed `sigma=2`, direct-drift
`eta=2`, adaptive-refinement, U50k run across Astex plus PoseBusters
(`393` complexes, `15,720` generated poses). It is not yet a same-device
three-repeat speed benchmark.

| Stage | Device/cohort | Resource seconds per pose | Resource seconds per complex | Parallel stage latency | Scope note |
|---|---|---:|---:|---:|---|
| Guided ODE pose generation | RTX A5000, all `393` | `0.857` GPU-s/generated pose | `34.27` GPU-s | `22.97 min` | includes shard startup, preprocessing, SDF/CSV output, learned ODE, and in-ODE guidance; old logs do not split guidance from model forward |
| Adaptive autograd refinement | mixed devices, all `393` | `1.049` GPU-s/input pose | `41.94` GPU-s | `14.83 min` | `282` complexes on 6000 Ada and `111` recovery complexes on RTX A5000 |
| Adaptive autograd refinement | 6000 Ada subset, `282` | `0.870` GPU-s/input pose | `34.80` GPU-s | `8.08 min` | hardware-consistent subset; not a full-denominator row |
| Adaptive autograd refinement | RTX A5000 recovery subset, `111` | `1.503` GPU-s/input pose | `60.10` GPU-s | `14.83 min` | recovery subset only |
| U50k confidence stage | 6000 Ada, all `393` | `0.155` GPU-s/scored pose | `12.43` GPU-s | `3.17 min` | scores both raw and refined banks (`80` poses/complex); includes per-complex model load, preprocessing, selection, and serialization |

The adaptive refiner terminated at mean step `62.36` on Astex and `65.57` on
PoseBusters (`64.87` over all poses; maximum 100). Summing the actually
executed heterogeneous stages gives `88.64` resource-seconds per complex, but
that workflow scores both the raw and refined 40-pose banks. A normal deployed
path scores only the final 40-pose bank; halving the old combined confidence
stage gives a rough `~82.4`-second total with the mixed-device refinement, or
`~75.3` seconds using the observed 6000 Ada refinement subset. These two
deployment totals are estimates, not admitted measurements, because historical
confidence logs did not split model load, forward, selection, and output.

Sampling elapsed values come from Slurm array `59606`; refinement and
confidence stage ledgers come from arrays `59608` and `59609`. New runs use the
direct internal stage timers added after this historical run, so guidance,
refinement compute, confidence forward, and selector/sort will no longer need
this reconstruction.

## Pending runtime rows

The same accounting will be appended for SigmaDock, DiffDock-Pocket, SurfDock,
Interformer plus PoseScore, PoseBench DiffDock, PoseBench DynamicBind, the
remaining running PoseBusters arms, and EFF-Dock's same-device, three-repeat
split runtime ledger. For EFF-Dock, the still-pending direct rows are
preprocessing, prior construction, learned ODE forward/integration, in-ODE
physical guidance, post-ODE autograd refinement, confidence forward,
selector/sort, and serialization. A row remains pending until task success,
output coverage, actual pose counts, and the timing schema agree.

## Provenance

- Slurm elapsed time source: `sacct`, `ElapsedRaw`, one row per array task.
- Resource source: `sacct`, `AllocTRES`.
- Pose counts: each run's reconciled `coverage.json`; RLDiff fixed-N generation
  additionally requires all rank files.
- Current official job and output roots:
  [`EXTERNAL_MODEL_SUBMISSION_20260831.md`](EXTERNAL_MODEL_SUBMISSION_20260831.md).
