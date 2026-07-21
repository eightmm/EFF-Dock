# Vina-guided Sampling Protocol

## Registration

- Protocol ID: `EFFDOCK-VINA-GUIDANCE-V1`
- Registered: 2026-07-19, before implementation-dependent benchmark runs
- Study-board claim: same ID
- Question: does inference-time differentiable Vina plus ligand distance-
  geometry guidance improve physical validity of EFF-Dock poses without
  materially reducing docking accuracy?

## Scientific Basis

AutoDock Vina 1.2 uses a united-atom, five-term empirical intermolecular score
with a rotatable-bond normalization. EFF-Dock implements that score in Torch so
its coordinate gradient can be added to the learned flow field. This is an
inference-time intervention; it does not retrain or alter either frozen model.

ForceFM and related force-guided flow/diffusion work support adding a
differentiable physical correction vector field during sampling. Their results
also show the expected trade-off: stronger guidance can reduce clashes but can
overconstrain pose exploration. Therefore the scale is not copied from another
model and external benchmark results are not used to tune it.

## Frozen Comparison

- Geometry checkpoint: retained confidence-matched geometry-FT checkpoint
- Confidence checkpoint: retained extmatch step-42500 checkpoint
- Sampler preset: N80, S25, sigma 0.5, late schedule, pocket cutoff 10A
- Baseline: identical command, weights, candidate count, selector, and per-ID
  seed with guidance scale zero
- Intervention: Vina+DG guidance only, frozen before opening results at scale
  0.05, start_t 0.5, linear ramp, atom-force cap 10, translation/angular
  velocity caps 5, DG weight 1.0, and receptor shell 18A. Smoke checks may
  reject the run for numerical failure but do not select among scales.
- Final selector: frozen EFF-Dock confidence composite
- External evaluation: PoseBusters v2 frozen snapshot

## Pre-registered Outcomes

- Primary validity outcome: official PoseBusters selected-pose pass-all rate;
  higher is better.
- Accuracy guardrail: symmetry-aware selected top-1 heavy-atom RMSD <2A;
  non-inferiority margin -2 percentage points.
- Secondary outcomes: per-term PoseBusters failures, fast protein/ligand clash
  validity, first-pose and Vina+DG-selected RMSD, oracle top-80 RMSD, Vina
  energy, DG strain, failure rate, and wall-clock overhead.
- Success: pass-all improves by at least +3 percentage points and the accuracy
  guardrail is met.
- Falsification: improvement below +3 points, accuracy loss beyond -2 points,
  non-finite trajectories, or unacceptable runtime/memory overhead.

## Implementation and Parity Contract

- Source of truth: AutoDock Vina v1.2.7 scoring implementation and its published
  Vina 1.2 coefficients.
- Pair cutoff is atom-center distance <8A; the scoring terms use surface
  distance after the cutoff mask.
- The exact Torch kernel accepts explicit Vina XS atom types. Exact parity is
  claimed only when those types come from a Vina-compatible PDBQT preparation.
- PDB/RDKit convenience typing is a declared best-effort fallback because a
  plain PDB does not fully encode protonation or AutoDock donor/acceptor types.
- The active benchmark fallback remains frozen for comparability; exact PDBQT
  parity is tested separately rather than silently changing input chemistry.
- Guidance uses dense `[samples, ligand atoms, receptor atoms]` Torch scoring,
  finite-gradient checks, equivariant norm clipping, and a late-time ramp.

## Run Provenance

Every run records protocol ID, commands, code state, checkpoint/config hashes,
dataset snapshot, seeds, sampler parameters, guidance scale/start/ramp, force
and velocity caps, strain weight, device, Slurm job ID, runtime, failures, and
all reported metrics. Failed and diagnostic runs remain in ignored outputs and
are not promoted to the final result.

## References

- [Trott and Olson, AutoDock Vina, *J. Comput. Chem.* 2010](https://pmc.ncbi.nlm.nih.gov/articles/PMC3041641/)
- [AutoDock Vina documentation](https://vina.scripps.edu/manual/) and
  [v1.2.7 source release](https://github.com/ccsb-scripps/AutoDock-Vina/releases/tag/v1.2.7)
- [ForceFM, force-guided flow matching for physically plausible molecular
  docking, NeurIPS 2025](https://papers.nips.cc/paper_files/paper/2025/hash/b3ee701cea4872c41356f32592a72289-Abstract-Conference.html)
- [Improving protein-ligand complex generation with force field guidance,
  *Journal of Cheminformatics* 2026](https://doi.org/10.1186/s13321-026-01198-2)
- [A fully differentiable ligand pose optimization framework with
  DeepRMSD+Vina](https://arxiv.org/abs/2206.13345)
