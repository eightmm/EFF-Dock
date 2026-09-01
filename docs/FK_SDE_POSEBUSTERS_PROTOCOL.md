# FK Translation-SDE PoseBusters Protocol

- Protocol ID: `EFFDOCK-FK-TRANSLATION-SDE-POSEBUSTERS-V1`
- Frozen: 2026-08-12, after the paired Astex result and before this run
- Status: queued paired descriptive external evaluation
- Claim boundary: PoseBusters v2 has already been opened in this project; this
  run cannot tune or admit any method or parameter

## Question and prediction

Question: when constraint-only Feynman--Kac resampling is held fixed, does the
score-corrected translation SDE turn restored post-resampling exploration into
better official PoseBusters validity and near-native docking utility than the
deterministic FK flow?

The evidence-informed prediction, frozen after seeing the Astex result, is that
FK-SDE raises candidate-level and confidence-selected official PB validity by
roughly 1--5 percentage points, while a material gain in the joint near-native
metric is uncertain. A utility-positive outcome requires all of:

- confidence-selected `PB-valid AND official RMSD <= 2 A` improves by at least
  2 percentage points versus FK-ODE;
- confidence-selected official RMSD `<= 2 A` does not decrease by more than
  2 percentage points;
- oracle `PB-valid AND official RMSD <= 2 A` does not decrease by more than
  2 percentage points;
- FK-SDE rounded terminal-coordinate uniqueness exceeds `0.90`.

Anything weaker remains diagnostic evidence only. No outcome from this run may
change `g_0`, beta, resampling times, checkpoint, selector, sampling budget, or
the metrics below.

## Frozen arms

| Arm | FK beta | FK times | Translation SDE `g_0` |
|---|---:|---|---:|
| FK-ODE | 0.01 | 0.3, 0.6, 0.8 | 0 |
| FK-SDE | 0.01 | 0.3, 0.6, 0.8 | 0.3 A |

Both arms use systematic resampling, zero post-resampling translation and
rotation jitter, and deterministic SO(3) dynamics. FK-SDE changes only the
score-corrected stochastic translation dynamics described in
`docs/FK_SDE_ASTEX_PROTOCOL.md`.

## Frozen evaluation

- Dataset: all `308` audited PoseBusters v2 complexes with a complete
  element- and connectivity-preserving heavy-atom bijection
- Checkpoint: `weights/effdock_geometry_ft_100k_best.pt`
- Selector: pure minimum predicted RMSD from
  `weights/effdock_confidence_extmatch_n80_s25_step42500.pt`
  (`confidence_cluster_free`)
- Budget: `N40/S25`, exactly `1,000` learned-model pose steps per complex/arm
- Prior: translation sigma `0.5 A`, shared deterministic pool size `100`, first
  `40` poses
- Time grid: late schedule, power `3`
- Pocket: frozen reference-defined center and `10 A` crop
- Seed: `42`, preserving the full-dataset per-complex seed mapping
- Receptor policy: explicit `geometry_only`
- Refinement: none
- Official evaluation: PoseBusters `0.6.5`, `redock`; PB validity is the
  conjunction of all `27` non-RMSD checks. Its separate symmetry-aware
  `RMSD <= 2 A` check is never folded into the PB-valid definition.

## Metrics

Primary metric: paired percentage-point difference, FK-SDE minus FK-ODE, in
confidence-selected `PB-valid AND official RMSD <= 2 A`.

Secondary metrics:

- confidence-selected PB-valid and official RMSD `<= 2 A` separately;
- confidence-selected numeric mapped RMSD and evaluator oracle RMSD;
- any-candidate official RMSD `<= 2 A` and any-candidate joint success;
- pooled all-candidate PB-valid and joint-valid fractions;
- all 27 individual official check pass rates;
- FK ESS and final unique initial-ancestor fraction.

Terminal diversity is reported three ways, all before looking at PB outcomes:

1. rounded terminal-coordinate uniqueness: unique complete coordinate arrays
   after rounding to `0.001 A`, divided by 40;
2. receptor-frame direct heavy-atom RMSD over all `40 choose 2 = 780` candidate
   pairs: per-complex median and fraction at least `2 A`;
3. per-pose nearest-neighbor receptor-frame heavy-atom RMSD, summarized by the
   per-complex median.

The diversity RMSDs do not align poses or permute symmetric atoms: placement
relative to the fixed receptor is the quantity of interest, and every record
retains identical atom ordering. Rounded uniqueness is specifically a clone-
collapse diagnostic, not a complete measure of chemically distinct modes.

## Execution and failure policy

One fixed `7b2c_tp7` technical smoke runs both arms from identical priors,
verifies three finite FK events and all-pose/ancestry hashes, then obtains
exactly `80` official pose rows. The full chain is smoke sampling -> smoke
manifest -> smoke official PB -> smoke audit -> paired full sampling -> frozen
manifest -> official all-pose PB -> report, with `afterok` dependencies.

The full run requires exact `308/308` success in each arm, `616` all-pose SDF
cells, `24,640` official pose rows, identical prior hashes/seeds between paired
arms, and no per-complex failure. Software defects may be fixed and rerun with
this exact protocol; numerical or scientific failures cannot trigger parameter
changes.
