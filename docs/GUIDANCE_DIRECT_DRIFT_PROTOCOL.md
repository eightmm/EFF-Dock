# Normalized direct GuidanceEnergy drift protocol

Protocol: `EFFDOCK-UNIFIED-GUIDANCE-DIRECT-DRIFT-BUDGET1000-V1`

Status: completed diagnostic implementation and execution protocol. Frozen
values are recorded without an automatic selection in
[`GUIDANCE_DIRECT_DRIFT_RESULTS.md`](GUIDANCE_DIRECT_DRIFT_RESULTS.md).

## Question and claim boundary

This run measures what changes when the same in-repository
`GuidanceEnergy = PhysicalEnergy + InteractionEnergy` force direction is added
directly to the learned fragment SE(3) ODE rather than applied as a post-step
operator-split corrector.

Astex Diverse and PoseBusters v2 results were already opened in earlier
studies. This is therefore a paired descriptive reference-pocket redocking
experiment. External outcomes cannot select or tune the energy formula,
active terms, strength, ramp, caps, checkpoint, pose budget, or production
sampler. The report records all values without automatically choosing a mode.

## Frozen intervention

At the current pose,

```text
x_bi = R(q_bf) l_i + T_bf
F_bi = -d GuidanceEnergy / d x_bi
```

The existing mass/inertia Newton--Euler projection gives one raw fragment
direction `(dT_bf, domega_bf)`. Translation and rotation are placed in the
same Angstrom-per-normalized-time space through their induced atom velocity:

```text
A_i(v, omega) = v_f + omega_f cross (x_i - T_f)
m_b = sqrt(mean_i ||A_i(v_model, omega_model)||^2)
g_b = sqrt(mean_i ||A_i(dT, domega)||^2)
r_b = m_b / (g_b + eps), with r_b = 0 when either direction is zero/non-finite
```

For ODE interval `[t_k,t_{k+1}]`, start `s=0.5`, and ramp power `p=1`,

```text
ramp_bar_k = 1/(t_{k+1}-t_k) * integral_[t_k,t_{k+1}]
             clamp((t-s)/(1-s), 0, 1)^p dt
c_b = eta * ramp_bar_k * r_b
```

with frozen strength `eta=0.1`. One additional positive scalar `gamma_b <= 1`
jointly bounds `c_b(dT,domega)` to maximum guide-only translation velocity
`5`, angular velocity `5`, and conservative per-step atom displacement
`0.25 Angstrom`. The same scalar is used for translation and rotation.

The learned and direct fields are then integrated exactly once:

```text
T_{k+1} = T_k + dt * (v_model + gamma*c*dT)
q_{k+1} = Exp(dt * (omega_model + gamma*c*domega)) tensor_product q_k
```

No post-step energy acceptance, backtracking, confidence, crystal coordinate,
RMSD, PoseBusters outcome, or selector value enters sampling. This is a
generative ODE control field, not molecular dynamics, free energy, or affinity.

## Frozen data and model

- checkpoint: `weights/effdock_geometry_ft_100k_best.pt`, step 100,000;
- checkpoint SHA-256:
  `6932fb3ba6ebac770f714453529656a44b8f33cf15119d23c9e675d2d60b36db`;
- config: `configs/train.yaml`, SHA-256
  `39aa62e4a48ed6f3aa4ff59345fb43a81220e2baba22edfd5beb0c4981b307ec`;
- input manifest: `docs/GUIDANCE_BUDGET1000_FULL_INPUTS.json`, SHA-256
  `99f15f557644cc51c3dd1f559b0dd97dd4259c1de3e1403fb761b7c7e079f668`;
- cohorts: all 85 Astex and all 308 PoseBusters v2 complexes admitted by the
  complete heavy-atom mapping contract;
- receptor policy: `geometry_only`, guidance shell `18 Angstrom`;
- prior sigma `0.5 Angstrom`, pocket cutoff `10 Angstrom`, center jitter `0`;
- global seed `42` plus the frozen sorted-ID offset;
- exact deterministic prior pool of 100 poses per complex;
- confidence disabled and refinement `none`.

## Arms and budget cells

Each shard runs paired `unguided` and `direct` arms from the same prior pool.
All three cells use exactly 1,000 learned model pose-steps:

| Cell | Poses | ODE steps | Active ramp-overlap intervals | Direct pose evaluations per complex |
|---|---:|---:|---:|---:|
| N100/S10 | 100 | 10 | 8 | 800 |
| N50/S20 | 50 | 20 | 16 | 800 |
| N40/S25 | 40 | 25 | 20 | 800 |

The interval-average ramp has the same integrated continuous strength in all
three cells. Guidance energy/gradient work is excluded from the learned-model
pose-step budget and is reported separately.

## Outputs and metrics

Sampling reports, without an automatic performance decision:

- symmetry-aware RMSD oracle `<2 Angstrom` and median RMSD;
- `any(fast-valid and RMSD <2 Angstrom)`;
- exact paired deltas and 10,000-resample complex-ID bootstrap 95% intervals;
- direct call, finite, zero-direction, norm, cap, displacement, CUDA-memory,
  failure, seed, prior-pool, input, parameter, implementation, and receptor
  provenance;
- official PoseBusters 0.6.5 `redock` pass-all over all 27 non-RMSD checks for
  each saved RMSD-oracle pose.

Technical completion requires exact `393/393` coverage in every arm/cell,
zero sampling/PoseBusters failures, zero non-finite guidance poses, exact
paired prior hashes, and no declared cap violation. These are execution gates,
not performance-selection rules.

## Post-completion rerun hardening

The completed result retains its original implementation and audit hashes.
The fail-closed launcher and sampling-to-official file-hash binding below were
added afterward for future reruns; they do not retroactively change the frozen
result identity or its recorded values. Because these changes alter the
current implementation identity, every new run must regenerate both dataset
audits and their combined manifest rather than reuse the completed-run audit.

## Execution

Submit the complete dependency chain with one command:

```bash
scripts/slurm/submit_guidance_direct_drift_full.sh
```

The launcher creates a fresh ignored output root and chains:

```text
fresh Astex/PoseBusters audit (2 tasks)
  -> strict audit merge
  -> paired GPU sampling (48 tasks)
  -> official PoseBusters checks (96 tasks)
  -> strict aggregate reports
```

Every sampling shard requires discovery of exactly `85` Astex or `308`
PoseBusters v2 inputs and exits nonzero after writing diagnostics if any
assigned complex fails. Official PoseBusters shards enforce the same input
counts, re-hash the protein, reference ligand, and selected pose against their
sampling-time SHA-256 values, and fail on any per-pose exception. `afterok`
dependencies therefore
prevent survivor-only downstream execution. The fresh output-root requirement
also prevents stale shard files from an earlier run satisfying a later report.
The `308` count is the frozen official PoseBusters v2 membership; 120 older
complex directories present in the local 428-target archive are outside this
protocol rather than failed or skipped v2 cases.

The individual Slurm scripts remain available for diagnosis:
`guidance_direct_drift_audit.sbatch`,
`guidance_direct_drift_merge_audit.sbatch`,
`guidance_direct_drift_array.sbatch`,
`guidance_direct_drift_posebusters_array.sbatch`, and
`guidance_direct_drift_report.sbatch`.
