# FK Translation-SDE Astex Protocol

- Protocol ID: `EFFDOCK-FK-TRANSLATION-SDE-ASTEX-V1`
- Frozen: 2026-08-12, before opening results from this run
- Status: completed paired descriptive external evaluation; hypothesis not
  supported; never a tuning or production-admission study
- Results: `docs/FK_SDE_ASTEX_RESULTS.md`
- Paper: Mark et al., *Feynman-Kac-Flow*, arXiv:2509.01543v1, Eq. 12
- Reference implementation inspected at commit
  `867df8301f45b65d87d6043249ed1b30a0912bdc`

## Question and hypothesis

Question: under the already-fixed constraint-only FK potential, does a
score-corrected stochastic translation flow restore post-resampling exploration
and improve Astex docking outcomes relative to deterministic FK?

Hypothesis: FK-SDE will produce more distinct terminal candidates than FK-ODE
and improve confidence-selected Astex RMSD `<2 A` by at least 2 percentage
points, without reducing oracle RMSD `<2 A` by more than 2 percentage points.
The directional prediction is a 2--5 percentage-point confidence-selected gain
and a terminal unique-coordinate fraction above 0.90.

A disconfirming outcome is a smaller selected gain, an oracle decrease larger
than 2 percentage points, a numerical/coverage failure, or no clear recovery
of terminal diversity. Such an outcome keeps the method diagnostic only. No
parameter is changed after results are opened.

## Dynamics

EFF-Dock's linear translation path is

```text
T_t = t T_1 + (1-t) T_0,       T_0 ~ N(0, sigma_0^2 I).
```

For the pocket-centred translation state, the analytic joint translation score
is

```text
score_T(T_t,t) = (t v_T(T_t,t) - T_t) / ((1-t) sigma_0^2).
```

The SDE arm uses Euler--Maruyama with

```text
g(t) = g_0 (1-t)
dT = [v_T + 0.5 g(t)^2 score_T] dt + g(t) dW_T.
```

`g_0=0.3 A` is copied from the paper's main chemical FK setting and is frozen
without an Astex sweep. The formula is generalized only for EFF-Dock's known
per-particle Gaussian prior scale `sigma_0`. Fragment rotations retain the
learned deterministic SO(3) flow: the uniform SO(3) prior is non-Gaussian and
no manifold score model is available. SDE randomness uses a generator separated
from the FK resampling generator by the fixed seed rule
`sampling_seed XOR 0x54534445`.

## Frozen arms

| Arm | FK beta | FK times | Translation SDE `g_0` |
|---|---:|---|---:|
| ODE | 0 | none | 0 |
| SDE | 0 | none | 0.3 A |
| FK-ODE | 0.01 | 0.3, 0.6, 0.8 | 0 |
| FK-SDE | 0.01 | 0.3, 0.6, 0.8 | 0.3 A |

The primary contrast is FK-SDE versus FK-ODE; it changes only the translation
dynamics. ODE and SDE are contextual controls. FK uses the difference potential
schedule, systematic resampling, the existing constraint-only term whitelist,
and no post-resampling translation or rotation jitter.

## Frozen evaluation

- Dataset: all 85 frozen Astex Diverse inputs with complete heavy-atom mapping
- Checkpoint: `weights/effdock_geometry_ft_100k_best.pt`
- Selector: pure minimum predicted RMSD from
  `weights/effdock_confidence_extmatch_n80_s25_step42500.pt`
  (`confidence_cluster_free`)
- Budget: `N40/S25`, exactly 1,000 learned-model pose steps per complex and arm
- Prior: scalar translation sigma `0.5 A`, shared nested pool size 100, first 40
- Time grid: late schedule, power 3
- Pocket: frozen reference-defined centre, 10 A crop; this limitation is
  carried in the output provenance
- Seed: 42 with the existing full-dataset per-complex seed mapping
- Receptor policy: explicit `geometry_only`
- Refinement: none

The `N40/S25` registered fixed-budget cell is chosen before outcomes because
Euler--Maruyama needs a finer grid than the deployment `N100/S10` cell while
holding model compute fixed.

## Metrics and gates

Primary scientific metric: paired difference in confidence-selected Astex
RMSD `<2 A` success rate, FK-SDE minus FK-ODE.

Secondary metrics:

- oracle RMSD `<2 A` success and median oracle RMSD;
- median confidence-selected RMSD;
- confidence-selected fast validity and its conjunction with RMSD `<2 A`;
- number of fast-valid candidates;
- rounded terminal-coordinate uniqueness;
- FK ESS fractions and final unique initial-ancestor fraction.

Before the full cohort, one fixed `1jje` smoke must have complete finite output,
identical prior hashes across all arms, three FK events, correct dynamics
provenance, and distinct FK-SDE descendants when a parent is duplicated. The
full run requires exactly 85 successful complexes per arm and matching prior
hashes per complex. A software defect may be fixed and rerun with this exact
protocol; numerical or scientific failure may not trigger parameter changes.

Astex outcomes were previously opened in this project. Every value from this
protocol is therefore paired descriptive evidence only and cannot select
`g_0`, beta, resampling times, checkpoint, selector, or production defaults.
