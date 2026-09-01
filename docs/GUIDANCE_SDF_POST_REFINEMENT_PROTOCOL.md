# Saved-pose GuidanceEnergy post-refinement

- Protocol: `EFFDOCK-GUIDANCE-SDF-POST-REFINEMENT-V1`
- Status: paired descriptive diagnostic completed; see
  `docs/GUIDANCE_SDF_POST_REFINEMENT_RESULTS.md`.
- Purpose: determine whether fixed-budget autograd descent under the current
  in-repository `GuidanceEnergy` can improve the physical validity of already
  generated poses after the learned ODE has finished.

## Frozen comparison

- Start from the complete saved `eta=0` candidate ensembles produced by
  `guidance_steric_high_eta_confidence_runs/20260807T045916Z`.
- Use exactly `100` poses per complex. Pose order and `sample_index` remain
  unchanged.
- Reconstruct the exact benchmark ligand input from
  `docs/GUIDANCE_BUDGET1000_FULL_INPUTS.json` and the original per-complex
  sampling seed. The saved SDF coordinates must have the same atom order,
  elements, and connectivity as this input.
- Use the frozen explicit reference-pocket center, the current EFF-FF v2.1
  physical terms, all seven default interaction terms, and the
  `geometry_only` receptor policy. Vina and external minimization engines are
  excluded.
- Optimize rigid fragment SE(3) transforms, not independent atom coordinates.
  Forces are obtained by Torch autograd and projected to fragment translation
  and rotation. Each pose has an independent monotone backtracking line
  search.
- Run at most `100` accepted/proposed iterations with a maximum translation,
  rotation, and induced atom displacement of `0.10 Angstrom`, `5 degrees`,
  and `0.10 Angstrom` per iteration. Save steps `0, 25, 50, 75, 100`.

The crystal ligand is used only for symmetry-aware RMSD reporting and never
enters the minimized objective. The explicit benchmark pocket centers are
reference-ligand-derived and this external-set experiment is descriptive; it
cannot tune or admit the refinement as a production default.

## Gates and outputs

- Fail before optimization on any input hash, pose count, atom-order,
  connectivity, pocket-center, or ligand-input identity mismatch.
- Fail a pose explicitly on non-finite energy/gradient. A finite exhausted
  line search is retained at its last accepted coordinates, marked
  `line_search_failed`, counted separately from max-step/converged poses, and
  never relabeled as successful optimization.
- Persist step SDFs, a trajectory tensor, per-pose initial/final energy and
  RMSD, terminal status, backtracking counts, implementation and parameter
  identities, and SHA-256 hashes.
- Official PoseBusters 0.6.5 `redock` is run separately on step 0 and step 100.
  Primary descriptive quantities are all-pose 27-check validity, per-check
  pass rates, RMSD `<2 Angstrom`, and paired invalid-to-valid / valid-to-invalid
  transitions.

## Full-cohort confidence reselection extension V2

- Coverage is exactly 85 Astex Diverse plus 308 PoseBusters v2 complexes from
  the frozen eta=0 manifest, with 100 poses per complex.
- Both step 0 and step 100 pose sets are rescored at sigma 0.5 using frozen
  docking checkpoint `effdock_geometry_ft_100k_best.pt` and frozen confidence
  checkpoint `effdock_confidence_extmatch_n80_s25_step42500.pt`.
- Confidence inference uses stable pose-order chunks of 20 on the `6000ada`
  partition. Step 0 and step 100 are both freshly scored with the same code,
  chunking, device family, and stable global argmin over all 100 concatenated
  scores.
- Exact agreement with the historical step-0 selector is diagnostic, not a
  completion gate. The rejected V1 attempts showed that both chunks of 20 and
  a batch of 100 could change a small number of historical Top-1 indices across
  current CUDA hardware. Mixing historical step-0 selection with fresh
  step-100 selection is therefore excluded from the primary paired comparison.
- Top-1 is fresh `argmin confidence_rmsd`, independently before and after
  refinement under the identical V2 scoring runtime.
  PL-valid, official PB-valid, and crystal RMSD are never selector inputs.
- Primary descriptive outcomes per dataset are Top-1 RMSD `<2 Angstrom`, Top-1
  21-check PL-valid, their conjunction, median selected RMSD, all-pose PL-valid,
  RMSD and PL-valid oracles, and selector-index change rate. The official
  27-check PoseBusters validity is retained as a secondary outcome.
- The final analysis decomposes the selected-pose change into three frozen
  states: fresh step-0 confidence Top-1, that exact pose index after
  refinement, and fresh step-100 confidence Top-1. The first transition is the
  same-pose refinement effect; the second is the incremental reselection
  effect. Their signed changes must add to the full-pipeline change. This
  decomposition is aggregation-only and does not rerun refinement, confidence,
  or PoseBusters.
- Step-0 official PoseBusters rows are reused from the content-addressed
  all-pose eta=0 run. Only step 100 is recomputed. This external benchmark
  extension is post-hoc descriptive and cannot tune the energy, optimizer,
  confidence model, selector, or production defaults.
- Slurm scripts do not request an explicit CPU count. PoseBusters workers and
  BLAS/OpenMP threads follow the scheduler-provided `SLURM_CPUS_PER_TASK`, or
  use one worker when the scheduler leaves it unset.

## Smoke result: PoseBusters `7b2c_tp7`

- Run date: 2026-08-12; refinement Slurm job `52516`, PB job `52517`.
- Scope: the 100 frozen `eta=0` poses for one deliberately difficult complex;
  this is not a benchmark-level estimate.
- Refinement completed for 100/100 poses in 3m57s on one RTX 2080 Ti. Every
  pose decreased its GuidanceEnergy; the median changed from 293.49 to -20.59.
- Official all-27-check PB validity changed from 0/100 to 14/100, consisting of
  14 invalid-to-valid and zero valid-to-invalid transitions.
- Protein minimum-distance, ligand internal-clash, bond-angle, internal-energy,
  and tetrahedral-chirality pass counts changed from 43, 40, 73, 65, and 27 to
  100/100. The diagnostic improper-inversion count was nonzero for 73/100
  inputs and zero for every refined pose.
- The remaining bottleneck was `minimum_distance_to_organic_cofactors`, which
  stayed at 14/100; `volume_overlap_with_organic_cofactors` changed 99 to 98.
- Symmetry RMSD `<2 Angstrom` changed from 65/100 to 57/100. Joint RMSD `<2
  Angstrom` plus PB-valid changed from 0/100 to 9/100. Thus the current pure
  energy descent repairs several validity failures but can move poses away
  from the crystal basin, and its organic-cofactor boundary is not yet strong
  enough for this complex.
