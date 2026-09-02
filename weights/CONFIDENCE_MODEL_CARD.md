# EFF-Dock S50 raw+refined pose-confidence checkpoint

- File: `effdock_confidence_s50_raw_refined_u70k.pt`
- SHA-256: `ce59be42f0ca613871ca079127c3296f5ca9a4ec72e44a9e5cf61878351c2638`
- Model type: `docking_graph_pose_confidence`
- Training update: 70,000 of the registered 100,000-update run
- Selection metric: validation Top-1 symmetry-aware RMSD `<2A`
- Paired docking checkpoint: `effdock_docking_early_time_t0p10_50k.pt`
- Default sampler: N100/S10/sigma-2/late-power-3/pocket-10A
- Default selector: stable minimum predicted pose RMSD

## Intended use

Rank multiple poses generated for the same receptor, ligand, and explicit
pocket. The promoted contract uses 100 poses, 10 ODE steps, translation sigma
2.0, and a 10A protein crop. The checkpoint predicts pose RMSD and success plus
per-atom displacement and success heads. These are within-complex ranking
signals; they are not calibrated across targets and do not predict binding
affinity.

The confidence model consumes t=1 ligand hidden representations from the
paired docking model. The default files, sampling preset, and pure-confidence
selector form one versioned stack. Using a different generator, sigma, pose
count, ODE budget, or protein crop is a distribution shift and must be reported
explicitly.

## Training and internal selection

The run initialized from the terminal U50k symmetry-confidence state and used
one balanced per-complex mixture of 32 raw sigma-2 poses, 32 deterministically
refined poses, and one mapped crystal anchor. Pose-level training and selection
labels use RDKit `CalcRMS` symmetry-aware no-alignment heavy-atom RMSD.

The checkpoint was selected only on the fixed 1,035-complex PLINDER validation
bank. U70k reached 622/1,035 (60.10%) Top-1 `<2A`, the best registered value in
the 100k run. U100k reached 617/1,035 (59.61%) and remains the terminal training
state, not the deployment checkpoint.

## External characterization

All rows reuse the same N100/S10/sigma-2 candidates. `Raw` is the sampled
ensemble. `Refined` uses the separately evaluated deterministic physical
refinement. `Joint` additionally requires official PoseBusters validity.

| Dataset | N | Raw Top-1 `<2A` | Refined Top-1 `<2A` | Refined PB-valid | Refined joint valid + `<2A` |
|---|---:|---:|---:|---:|---:|
| Astex Diverse | 85 | 81.18% | 85.88% | 94.12% | 81.18% |
| PoseBusters v2 | 308 | 78.25% | 84.09% | 95.13% | 81.17% |
| PhiBench | 203 | 63.05% | 64.53% | 90.64% | 59.11% |
| FoldBench-Pocket full | 558 | 72.04% | 75.63% | 95.16% | 72.94% |
| FoldBench-Pocket post-cutoff | 66 | 68.18% | 71.21% | 89.39% | 66.67% |
| OpenBind | 860 | 49.07% | 55.47% | 98.60% | 54.65% |

As a separately reported endpoint-aligned PhiBench diagnostic, confidence
Top-5 reaches `156/203` (`76.85%`) refined RMSD success and `150/203`
(`73.89%`) same-pose PB-valid/RMSD joint success. This does not change the
production Top-1 selector.

U70k was not chosen from these external results. The cohorts had already been
used during development, so the figures are descriptive. PhiBench and
FoldBench are the core temporal checks; OpenBind is a dense single-protease
auxiliary cohort. The refinement numbers do not imply that `eff-dock dock`
silently performs refinement.

## Limitations

- Requires an explicit pocket and compatible EFF-Dock hidden features.
- Performance can shift with receptor preparation, ligand protonation or
  stereochemistry, pose count, sigma, ODE budget, crop, or generator weights.
- Reference structures are used only for evaluation labels, not ranking.
- The external studies are pocket-redocking evaluations, not blind pocket
  discovery or prospective screening.
- The model does not predict affinity or binder/non-binder status.

Exact protocols and paired comparisons are in
`docs/S50_RAW_REFINED_CONFIDENCE_100K_PROTOCOL.md`,
`docs/S50_RAW_REFINED_CONFIDENCE_EXTERNAL_RESULTS.md`, and
`docs/S50_RAW_REFINED_CONFIDENCE_TEMPORAL_EXTERNAL_RESULTS.md`. The current
cross-model and Top-5 context is in `docs/BENCHMARK_RESULTS.md` and
`benchmarks/results/external_models/phibench_u70k_top5.json`.

This checkpoint is released together with the paired docking checkpoint under
Apache-2.0. See `DOCKING_MODEL_CARD.md` and `MANIFEST.md` for the complete
deployment identity.
