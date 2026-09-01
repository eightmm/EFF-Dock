# Sigma-2 Saved-Candidate Fragment-Template Headroom Protocol

Protocol ID: `EFFDOCK-FRAGMENT-TEMPLATE-SWAP-HEADROOM-V1`

Status: frozen before the full 85-system Astex and 308-system PoseBusters
outputs were opened by this analysis. Engineering smokes on Astex `1jje` and
PoseBusters `7b2c_tp7`/`5sak_zry` were opened first and are not confirmation
results.

## Question

For the already generated sigma-2, N=100 candidate ensembles, how much
near-native candidate density is lost solely because every fragment retains
its seeded RDKit internal geometry rather than the corresponding crystal
internal geometry?

This is an optimistic endpoint counterfactual, not a new inference run and not
evidence by itself that RDKit-local fine-tuning will improve the model.

## Frozen inputs

- Cohort: all 85 Astex Diverse and 308 PoseBusters v2 complexes in
  `docs/GUIDANCE_BUDGET1000_FULL_INPUTS.json`.
- Saved ensembles: completed `sigma=2.0`, `N=100`, `S=10`, late-power-3,
  normalized-drift `eta=2.0` runs under
  `outputs/benchmarks/guidance_sigma_sweep_eta2_runs/20260809T031535Z`.
- The exact per-complex `sampling_seed` is read from both saved SDF properties
  and bound shard CSVs. Seed 0 is not substituted.
- Frozen SMILES, crystal reference ligand, and every all-pose SDF hash are
  validated. All 100 pose indices and the original atom order must be complete.

The source candidates used normalized-drift guidance and therefore do not
represent a frozen unguided deployment solver. Results diagnose endpoint
geometry headroom in this candidate population only.

## Counterfactual

For every saved candidate and every fragment defined by the regenerated
production RDKit template:

1. recover the fragment's saved rigid frame by a proper Kabsch fit;
2. replace only its internal coordinates with the mapped crystal fragment;
3. retain the saved fragment centroid and orientation;
4. repeat independently for every fragment.

Whole-ligand crystal placement and inter-fragment crystal geometry are never
copied. A complete stereo-preserving atom map and complete symmetry enumeration
are required. The mapping used for template transport is the symmetry-equivalent
map with the smallest fragment-local template floor and is fixed across all 100
poses of that complex.

## Metrics and decision rule

Primary metric per dataset is the change in macro mean `K2`, where `K2` is the
number of 100 candidates with symmetry-aware RMSD strictly below 2 Angstrom.
Also report median K2, `P(K>=1)`, `P(K>=5)`, `P(K>=10)`, best RMSD, and all
candidate/complex threshold crossings.

All 393 complexes must complete. The diagnostic headroom is considered strong
enough to justify implementing a short paired RDKit-local fine-tuning ablation
only if:

- the 393-complex weighted mean improvement is at least `+1.0 K2` per 100
  candidates; and
- neither Astex nor PoseBusters has a negative macro mean K2 change.

If the absolute weighted mean change is below `0.25 K2` and the net number of
complexes newly reaching `K>=1` is below two, geometry fine-tuning is not
prioritized. Other outcomes are intermediate and require a direct internal
paired rollout or conservative short fine-tune before any long run.

These thresholds are screening rules. External benchmark outcomes were already
opened elsewhere in the project and cannot admit a production model.

## Registered outputs

```text
outputs/analysis/fragment_template_swap_headroom_v1/astex_sigma2_n100.json
outputs/analysis/fragment_template_swap_headroom_v1/posebusters_sigma2_n100.json
```

Post-swap confidence and the saved `fast_valid` flags are not reused because
they were computed on the original RDKit geometry. The primary probe reports
only recomputed symmetry-aware RMSD and K2-family metrics.
