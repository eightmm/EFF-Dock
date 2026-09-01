# S50 raw+refined confidence external results

Protocol ID: `EFFDOCK-S50-RAW-REFINED-CONFIDENCE-EXTERNAL-V1`

Status: completed. Smoke array `61906`, full array `61907`, and report job
`61908` all completed with exit code 0. The full inventory contains 64/64
shards and 786/786 arm-complex summaries covering 85 Astex Diverse and 308
PoseBusters v2 complexes for both U70k and U100k.

## Aggregate result

All rows reuse the identical sigma-2 N100 candidate bank. `raw` is refinement
step 0 and `refined` is deterministic refinement step 100. Top-1 and Top-5 use
symmetry-aware no-alignment RMSD below 2 Angstrom. Joint additionally requires
the existing official validity ledger to pass.

| Dataset | Stage | Checkpoint | Top-1 <2A | Top-5 <2A | Official valid | Joint valid+<2A |
|---|---|---|---:|---:|---:|---:|
| Astex | raw | U70k | 69/85 (81.18%) | 94.12% | n/a | n/a |
| Astex | raw | U100k | 71/85 (83.53%) | 94.12% | n/a | n/a |
| Astex | refined | U70k | 73/85 (85.88%) | 95.29% | 94.12% | 69/85 (81.18%) |
| Astex | refined | U100k | 72/85 (84.71%) | 94.12% | 94.12% | 68/85 (80.00%) |
| PoseBusters | raw | U70k | 241/308 (78.25%) | 89.29% | n/a | n/a |
| PoseBusters | raw | U100k | 240/308 (77.92%) | 89.61% | n/a | n/a |
| PoseBusters | refined | U70k | 259/308 (84.09%) | 91.56% | 95.13% | 250/308 (81.17%) |
| PoseBusters | refined | U100k | 260/308 (84.42%) | 90.91% | 94.16% | 248/308 (80.52%) |

## Paired U100k minus U70k changes after refinement

- Astex Top-1: 0 gains and 1 loss (`-1.18 pp`). Joint: 0 gains and 1 loss
  (`-1.18 pp`). Official validity was unchanged.
- PoseBusters Top-1: 4 gains and 3 losses (`+0.32 pp`). Official validity: 0
  gains and 3 losses (`-0.97 pp`). Joint: 2 gains and 4 losses (`-0.65 pp`).

U100k therefore provides only a one-complex PoseBusters RMSD-only gain while
losing Astex success and both refined joint endpoints. This descriptive
external result is consistent with retaining the internally selected U70k
`best.pt`, but it does not independently select or promote that checkpoint.

## Artifact

- Report:
  `outputs/benchmarks/s50_raw_refined_confidence_external_runs/4184b0087a19d1138d578b123fb11ba4658fa149bf2d8087f8dd96c4beb4d4dc/report.json`
- Report SHA-256:
  `ed65159344ad1c320bc8b81c01e809880a3e0a54120914bc35682f7ec9591946`
