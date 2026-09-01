# External-model official submission — 2026-08-31

> Historical execution ledger. The current EFF-Dock comparison is
> supplied-pocket-only. PoseBench DiffDock, DynamicBind, FABind, and other
> blind/full-receptor or co-folding runs below are non-comparison archives and
> must not be plotted or reported as EFF-Dock baselines. Remaining global
> DiffDock jobs were cancelled on 2026-09-01.

Status: inference submitted; results are not admitted until the per-seed
coverage, selector, RMSD, and three-repeat aggregation gates complete.

This campaign follows
[`EXTERNAL_MODEL_OFFICIAL_INFERENCE_PROTOCOL.md`](EXTERNAL_MODEL_OFFICIAL_INFERENCE_PROTOCOL.md).
Every stochastic model uses three frozen repeats. Missing targets remain
denominator failures; no result is copied from another seed. Resource-only
batch-size reductions do not change model checkpoints, diffusion steps,
candidate counts, receptor policy, or ranking.

Runtime is tracked separately in
[`EXTERNAL_MODEL_RUNTIME_20260831.md`](EXTERNAL_MODEL_RUNTIME_20260831.md).
It reports actual resource-seconds per emitted pose, resource-seconds per
complex, and queue-free parallel stage latency. Generation and native
selection/refinement are kept separate.

## Compatibility corrections

- SigmaDock: PoseBusters target `7XPO_UPG` contains a terminal TYR residue with
  `N/C/O` but no `CA`. The runtime adapter now removes only the complete PDB
  residue group lacking `CA` before SigmaDock's residue-to-CA graph lookup. It
  never invents a coordinate. GPU smoke `61819` completed `1/1`, including
  Vinardo, with one residue and three atoms removed. The final adapter applies
  this rule only to non-hetero protein residues; waters, metals, and cofactors
  are preserved. Initial r5 jobs `61833`, `61892`, and `61893` were cancelled
  after seed 0--3 had loaded the pre-HETATM-guard adapter. Their partial outputs
  are superseded and never merged.
- DiffBindFR: its flexible receptor export may omit crystallographic waters,
  cofactors, or unresolved side-chain atoms while relabeling chains beyond
  `Z` as two-character tokens. The adapter now accepts a chain relabel only
  when at least 95% of exported coordinate-independent atom identities occur
  in the mapped source chain. Real `8F4J_PHO` sample 1 passed all 40 chains;
  minimum exported-atom overlap was `0.967871`, and chain `A` was `2622/2622`.
  Original raw PDBs remain unchanged and compatible siblings are audited.
- DynamicBind: failed/incomplete shards are regenerated under a new run tag,
  `batch_size=1`, and a fresh output root. The first `test` submission
  (`61823`, `61826`, `61829`--`61832`) was cancelled before admission because
  `8F4J_PHO` can require more GPU memory than the partition provides. The
  replacement uses `6000ada`; cancelled partial outputs are never merged.

## Corrective recovery — 2026-09-01

- PoseBench DiffDock: scheduler-provided input/output paths are canonicalized
  before the pinned upstream entry point changes its working directory. GPU
  smoke `62579` completed `1/1`. The first full submission `62608`–`62614`
  was stopped and excluded after inspection showed that it used PoseBench's
  multi-component released SMILES rather than the preregistered primary
  crystal ligand. The admitted replacement uses the already-audited
  primary-only manifests and all source shards—not the truncated earlier
  range—covering Astex `85/85` in 8 shards and PoseBusters v2 `308/308` in 24
  shards for every seed. The replacement jobs are `62681`, `62684`, `62687` (Astex) and
  `62688`, `62690`, `62691` (PoseBusters), on supported `test` A5000 GPUs.
  Diagnostic execution also exposed an upstream zero-edge shape bug on
  `1L2S_STC`: when no torsion-neighbour edges exist, the convolution returned
  one zero row per ligand atom instead of one per requested rotatable bond.
  Traceback job `62667` localized the fault; compatibility smoke `62671`
  completed with exact N=5 after honoring `out_nodes` only in that empty-edge
  branch. Isolated seed-0/1 confirmation jobs `62676` and `62677` also passed
  exact N=5; the replacement full campaign loads the corrected wrapper. The
  first four completed full shards (`62681_0`, `62681_1`, `62684_0`, and
  `62684_1`) each passed exact coverage (`11/11` targets, five poses per
  target), while subsequent shards continued automatically. Pending tasks
  were moved to the 12-hour `short` QOS after measured shard runtimes of about
  11 minutes.
  A later exact-coverage gate separated two additional failure classes. The
  pinned DiffDock v1.1 graph builder deliberately rejects receptors above
  3,000 residues: `7FRX_O88` (3,405), `7M31_TDR` (4,047), and `8F4J_PHO`
  (5,308). These three targets remain zero-pose failures in the frozen 308
  denominator; the limit is not removed because that would change the official
  model implementation. `7PJQ_OWH`, by contrast, reached the released
  all-atom confidence model and exhausted a 24 GB A5000. The caught OOM then
  poisoned the shared CUDA context and caused a later ESM call to fail. The
  corrected runner isolates manifest rows in fresh processes, checks complete
  outputs using the exact full target ID (the upstream `--skip_existing` check
  truncates IDs at `_`), records per-target failure classes, and accepts only
  the three audited native-limit cases as terminal denominator failures.
  Regression smoke `62912_4` completed in 33 seconds with `12/12` pose-complete
  targets plus one native-limit target and exit 0. Seed-0/1 native-limit shard
  recoveries are `62991` and `62989`; replacement seed-2 shards are `62994`.
  Isolated 48 GB recovery of shard 009 is `62988` (seed 1) and `62990`
  (seed 2). Seed-1 job `62988_9` completed in 6:03 with exit 0, restored
  `7PJQ_OWH` at exact N=5, filled every target lost after the poisoned CUDA
  context, and closed shard coverage at `13/13`; its structured failure ledger
  is empty. Seed-2 job `62990_9` also completed with exact `13/13` coverage
  and an empty failure ledger. Seed-2 native-limit replacements `62994_4` and
  `62994_5` closed at `12` pose-complete plus one audited native-limit target
  each; `62994_10` is the final corresponding replacement task. No successful
  target is regenerated by the admitted recoveries.
- SurfDock and Interformer: the fixed-holo benchmark manifests expose
  `holo_protein`; the failed jobs incorrectly requested the absent
  `experimental_protein` column. Corrected GPU smokes `62580` and `62581`
  completed `1/1`. SurfDock full jobs are `62627`–`62629` (Astex) and
  `62630`–`62632` (PoseBusters). Interformer generation jobs are
  `62615`, `62617`, `62619` (Astex) and `62621`, `62623`, `62625`
  (PoseBusters), with dependent native PoseScore jobs `62616`, `62618`,
  `62620` and `62622`, `62624`, `62626` respectively. The original `long` QOS
  caused only a per-user scheduling hold, not a model failure. All generation
  and dependent PoseScore arrays were moved to the `short` QOS with a 12-hour
  limit after the one-target smokes completed in 3–4 minutes. Interformer full
  shards `62615_0` and `62615_1` then entered `RUNNING`; the remaining tasks
  are resource/priority or array-limit waits.
- RLDiff RL++: the raw diffusion stage was already complete. smina rejected
  receptor hetero-residues containing unsupported AutoDock atom types
  (B/V/Mo/Xe). The compatibility adapter now removes only the complete
  unsupported HETATM residue, never a protein `ATOM` residue, and writes a
  per-target `receptor_compatibility.json`. Astex recovery `62582`–`62584`
  and PoseBusters recovery arrays `62589`–`62591` all completed; every one of
  the 15 recovered seed-target pairs produced 40 minimized poses and a GNINA
  reranking score.
- DynamicBind: four otherwise successful shards emitted only four of five
  requested poses. A fresh `62593` run reproduced the missing rank and exposed
  an upstream in-place rename collision: native confidence reranking can move
  one file onto a rank filename that has not yet been moved, silently
  overwriting that pose. The runtime adapter now performs the same native sort
  through a two-phase temporary rename and records the source/patched hashes.
  Exact-target A5000 job `62643` validated the fix on `6Z14_Q4Z` with all five
  ranks preserved. The three remaining targets run independently as `62647`,
  `62648`, and `62650`; superseded full-shard jobs `62594`, `62596`, and
  unstarted `62640` were cancelled. No partial output is merged unless the
  target passes the exact N=5 coverage gate.

## Submitted jobs

| Model | Astex Diverse | PoseBusters v2 | Frozen setting | Output root |
|---|---|---|---|---|
| SigmaDock | completed earlier: `61249`, `61257`, `61258` | final generation `61897` (seed pools `0–39`, `40–79`, `80–119`); seven-check postprocess `61898`; evaluation/aggregation `61899` | Euler S25, N40, Vinardo plus paper seven-check selector, fixed holo | `outputs/external_models/runs/sigmadock/official_holo_protocol_r6_parser_ca_hetero_safe_20260831` |
| RLDiff RL++ | raw `61844`, `61846`, `61848`; corrected CPU recovery `62582`–`62584` | raw `61850`, `61852`, `61854`; corrected CPU recovery `62589`–`62591` | S20, N40, seeds 0/1/2, fixed holo pocket crop | base `outputs/external_models/runs/rldiff/official_rlpp_r2_20260831`; audited recovery `outputs/external_models/runs/rldiff/rlpp_recovery_20260901` |
| DiffDock-Pocket | `61856`–`61858` | `61859`–`61861` | S30, N40, seeds 0/1/2, released predicted-receptor pocket crop | `outputs/external_models/runs/diffdock_pocket/official_s30_n40_r2_20260831` |
| SurfDock | corrected full `62627`–`62629` | corrected full `62630`–`62632` | S20, N40, seeds 0/1/2, MDN, fixed holo receptor | `outputs/external_models/runs/surfdock/official_holo_s20_n40_r3_20260901` |
| DiffBindFR | complete earlier | recovery `61820`–`61822`, shard 009 only | native S20/N40, seeds 0/1/2, official smina EC then MDN, fixed holo | `outputs/external_models/runs/diffbindfr/official_holo_ec_r1_20260830` |
| Interformer | corrected generation/score `62615/62616`, `62617/62618`, `62619/62620` | corrected generation/score `62621/62622`, `62623/62624`, `62625/62626` | N20, seeds 0/1/2, PyVina 64×2000 plus BFGS, four-checkpoint PoseScore, fixed holo | `outputs/external_models/runs/interformer/official_holo_n20_r3_20260901`; `outputs/external_models/runs/interformer_pose_score/official_holo_v02_r3_20260901` |
| PoseBench DiffDock | corrected full `62681`, `62684`, `62687` | corrected full `62688`, `62690`, `62691` | released S20/actual-19, N5, seeds 0/1/2, primary-ligand input, holo-aligned predicted receptor | `outputs/external_models/runs/posebench_diffdock/official_primary_s20_n5_r4_20260901` |
| PoseBench FABind | already complete, seeds 128/129/130, 85/85 each | already complete, seeds 128/129/130, 308/308 each | released one-pose post-optimized redocking | `outputs/external_models/runs/posebench_fabind/official_r1_20260830` |
| PoseBench DynamicBind | `61841`–`61843` | `61838`–`61840` | S20, N5, seeds 42/43/44, fresh-tag batch-1 recovery on failed shards | `outputs/external_models/runs/posebench_dynamicbind/official_s20_n5_batch1_recovery_v2_20260831` |
| PoseBench Vina | `61886`–`61888` | `61889`–`61891` | exhaustiveness 32, requested N40, seeds 1/2/3, CPU-only, clean ADFR receptor writer | `outputs/external_models/runs/posebench_vina/official_ex32_n40_r3_20260831` |

## Admission gates

1. Reconcile every array task and inspect stderr; scheduler completion alone is
   insufficient.
2. Freeze per-target pose counts and missing/error reasons for each seed.
3. Evaluate native Top-1 and Oracle with symmetry-aware no-alignment heavy-atom
   RMSD against the complete denominator.
4. Run official PoseBusters v2 on the selected pose when a PB-valid joint result
   is reported.
5. Report mean and sample standard deviation across the three complete repeats.
6. Do not merge cancelled DynamicBind outputs, the cancelled SigmaDock r5
   partial output, or the superseded SigmaDock `306/308` runs into this
   campaign.
7. Reconcile `ElapsedRaw`, allocated resources, and actual emitted pose counts;
   append per-stage and end-to-end runtime to
   `EXTERNAL_MODEL_RUNTIME_20260831.md` before final comparison.
