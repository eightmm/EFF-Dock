# External-model official inference protocol

Status: frozen on 2026-08-30 before opening the corrected three-repeat results.

## Benchmark scope

The claim-bearing comparison is restricted to **pocket-conditioned docking**:
every method must consume the supplied benchmark pocket or a cognate-ligand
defined pocket. Blind/full-receptor docking, learned pocket discovery, and
co-folding methods are not plotted or included in benchmark tables. Existing
DiffDock, DynamicBind, FABind, and FlowDock runs are retained only as local
non-comparison archives.

## Resolution rule

For every model, the inference contract is selected in this order:

1. the paper's released benchmark command or an official benchmark wrapper;
2. the repository README's recommended inference command;
3. the repository's checked-in inference configuration;
4. raw parser defaults only when none of the above exists.

Batch size and worker count are resource controls only when the upstream code does
not use them to change the sampled distribution. All scientific parameters,
checkpoints, input receptor policy, generated pose count, selector, and native
refinement remain fixed. Every repeat uses the same frozen cohort and denominator.

Reported uncertainty is the mean and sample standard deviation (`ddof=1`) of three
complete repeats. Failed or missing targets remain denominator failures.

## Frozen model contracts

| Model | Official generation | Official selection / refinement | Receptor and site | Three repeats | Primary evidence |
|---|---|---|---|---|---|
| SigmaDock | Euler S25; one independent pose per seed; 40 seeds per reported run; canonical fragmentation; sampled conformer; cutoff 5 A; official jitter/noise defaults; EMA checkpoint | Vinardo plus seven PoseBusters checks, `score_i = -Vinardo_i * p_i^4`; no coordinate minimization | fixed holo receptor; cognate ligand defines the pocket | pools 0–39, 40–79, 80–119 | paper Appendix F.2; `external_models/src/sigmadock/conf/sampling/base.yaml`; `external_models/src/sigmadock/src/sigmadock/chem/statistics.py` |
| RLDiff RL++ | S20, N40, official RL checkpoint | released `--minimize_and_rerank`: smina minimization then GNINA ranking | benchmark-set fixed holo receptor; cognate pocket | seeds 0, 1, 2 | `external_models/src/rldiff/README.md` |
| RLDiff raw | same generated candidates as RL++ | native confidence only; no minimization | same as RL++ | seeds 0, 1, 2 | retained as an ablation, not the paper's primary RL++ result |
| DiffDock-Pocket | S30, N40, `keep_local_structures`, official score and confidence checkpoints | released confidence ranker; no optional relaxation | aligned predicted receptor used by the released model; cognate supplied pocket crop | seeds 0, 1, 2 | `external_models/src/diffdock-pocket/README.md` |
| SurfDock | S20, N40, MDN distance threshold 3.0, official docking and posepredict checkpoints | posepredict MDN; no optional force optimization | fixed processed holo receptor; cognate 8 A surface pocket | seeds 0, 1, 2 | `external_models/src/surfdock/bash_scripts/test_scripts/eval_samples.sh` |
| DiffBindFR | N40; native 20 effective diffusion steps; batch 16; paper checkpoints | official smina error correction enabled, followed by MDN ranking; `results_ec_mdn_top1.csv` | fixed holo receptor; cognate ligand; upstream 12 A pocket radius | seeds 0, 1, 2 | `external_models/src/diffbindfr/README.md`; `DiffBindFR/app/predict.py` |
| Interformer | N20; v0.2 energy ensemble; native PyVina 64 Monte Carlo repeats × 2000 steps plus BFGS | four-checkpoint PoseScore ensemble; appended input pose excluded | fixed holo receptor; cognate supplied pocket | seeds 0, 1, 2 | `external_models/src/interformer/inference.py`; native reconstruction defaults |
| PoseBench Vina | exhaustiveness 32, box 25 × 25 × 25 A, spacing 1 A, N40 | native Vina affinity | PoseBench predicted receptor; supplied cognate site for this pocket-specified arm | seeds 1, 2, 3 because upstream leaves seed random | `external_models/src/posebench/configs/model/vina_inference.yaml` |

The frozen settings for excluded global-model diagnostics remain recoverable in
the historical submission/audit records, but those rows are outside this
protocol's primary comparison.

## SigmaDock correction

The earlier `common_rmsd_v8` SigmaDock row is a robustness experiment, not an
official reproduction. It used a holo-aligned predicted receptor and ranked only
by minimum Vinardo affinity. The paper instead evaluates fixed holo redocking and
ranks 40 independent samples by

\[
s_i=-b_i p_i^4,
\]

where `b_i` is the Vinardo binding energy and `p_i` is the mean of:

- minimum distance to protein;
- tetrahedral chirality;
- internal energy;
- internal steric clash;
- double-bond flatness;
- bond lengths;
- bond angles.

The old values, Astex 62.35% and PoseBusters 50.97% RMSD-only Top-1, must
therefore not be compared with the paper's 90.6% and 80.5%. The old oracle values
were 89.41% and 79.22%, but they also used the predicted receptor and are not the
paper's selector result.

Corrected implementation:

- holo inputs: `outputs/external_models/inputs/sigmadock_official_holo/`;
- generation: `benchmarks/external_models/slurm/external_sigmadock_inference.sbatch`;
- official PoseBusters postprocess: `benchmarks/external_models/slurm/external_sigmadock_posebusters.sbatch`;
- official selector: `benchmarks/external_models/evaluate_sigmadock_official.py`;
- three-repeat aggregation: `benchmarks/external_models/aggregate_repeat_metrics.py`.

One-target end-to-end smoke jobs `61184` and `61185` completed successfully,
including PoseBusters 0.6.5 `redock`, but that test-node run did not expose a
6000ada-only external-scorer runtime problem. The first full attempt (`61186`,
`61192`, with dependants `61193`, `61194`, `61200`, and `61201`) was cancelled:
the pinned GNINA executable could not resolve `libcudart.so.12` on 6000ada and
therefore returned status 127 after sampling. Those outputs are invalid and are
not evaluated.

`scripts/others/run_model.sh` now exposes every NVIDIA runtime directory from
SigmaDock's pinned uv environment to external subprocesses, and the Slurm entry
point fails before sampling unless `gnina --version` succeeds. The 6000ada
one-target regression job `61248` completed with GNINA 1.3.2 and coverage 1/1,
with zero Vinardo failures. The corrected full chain is:

- generation: `61249` (Astex) and `61255` (PoseBusters), seeds 0--119;
- PoseBusters selector checks: `61257` and `61256`, respectively;
- three-repeat selector/evaluation/aggregation: `61258` and `61259`.

The corrected output root is
`outputs/external_models/runs/sigmadock/official_holo_protocol_r4_20260830`.

## Corrections to the preliminary table

- DynamicBind's preliminary N40/seed-0 result is superseded by the official
  N5/seed-42 contract.
- FABind's preliminary output directory was labelled seed 0 although the
  PoseBench wrapper hard-coded seed 128. The new runner records the true seed and
  changes only that seed for repeats.
- DiffBindFR's preliminary result disabled official error correction. The new
  contract enables smina correction and evaluates the error-corrected MDN output.
- Preliminary SigmaDock, SurfDock, DiffBindFR, and Interformer predicted-receptor
  runs remain available as robustness runs. Paper-style redocking uses the fixed
  holo receptor where specified above.

No preliminary single-run value is promoted to the final comparison until its
contract check, full-coverage check, and three-repeat aggregation all pass.
