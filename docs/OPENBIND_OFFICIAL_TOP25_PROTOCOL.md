# OpenBind official-style Top-25 aggregation protocol

Protocol ID: `EFFDOCK-OPENBIND-OFFICIAL-TOP25-V1`

Status: frozen before the Top-25 PoseBusters or OpenStructure outcomes were
inspected.

This file preserves the original U25-ranked aggregation contract. Current
public reporting changes only the confidence ranking to U50 and reruns the
pose-level evaluators under
[`OPENBIND_OFFICIAL_TOP25_U50_PROTOCOL.md`](OPENBIND_OFFICIAL_TOP25_U50_PROTOCOL.md).

## Scope and claim boundary

This protocol re-aggregates a completed EFF-Dock pocket-redocking run under the
public OpenBind EV-A71 2A figure contract. It does not regenerate poses, tune a
selector, or use an outcome metric during ranking.

This is a pocket-conditioned redocking diagnostic. It is not blind pocket
prediction, cross-docking, co-folding, or an official leaderboard submission.
Cross-method values are reported only when their input setting is labelled.

## Frozen inference stack

- Candidate generation: 100 poses, 10 ODE steps, translation prior
  `sigma=2.0`, and normalized direct-drift GuidanceEnergy coupling with
  `eta=2.0`.
- Docking checkpoint SHA-256:
  `65be44d7dc8f0867eb9fc5d22214b80f93971ea4702679a527c665046e91e6b6`.
- Physical refinement: the adaptive, in-repository rigid-fragment SE(3)
  corrector described in
  [`S50_SYMMETRY_CONFIDENCE_REFINED_EXTERNAL_PROTOCOL.md`](S50_SYMMETRY_CONFIDENCE_REFINED_EXTERNAL_PROTOCOL.md).
  No Vina or external force-field/minimization engine is called.
- Confidence checkpoint: internally selected U25k symmetry-confidence model,
  SHA-256
  `1c59034172fb925cc8a70777dcba236be349f1a1de1775d49cc17d492b17c030`.
- Ranking signal: ascending post-refinement confidence-predicted RMSD, with
  original zero-based pose index as the stable tie-breaker.
- Information boundary: the frozen pocket center is computed from the crystal
  complex as the centroid of receptor residue virtual nodes within 8 A of the
  reference ligand, with the ligand centroid as fallback. That derived
  three-vector, receptor coordinates, and ligand chemistry may enter inference.
  Reference ligand atom coordinates do not otherwise enter the model or
  GuidanceEnergy; their direct use, PoseBusters outcomes, RMSD, and LDDT-PLI
  are evaluation-only.

## Denominator

Reproduce the public OpenBind `filtered=True, scaffold_only=True` cohort from
the official metadata:

- prepared ground truth must be PoseBusters-valid;
- suspected artefacts are excluded;
- fragment-screen structures are excluded;
- covalent ligands are excluded.

The resulting denominator is 802 complexes. The frozen EFF-Dock source run has
predictions for 786. Its stricter reference-preparation filter excluded 16
official-denominator complexes; those complexes remain in every denominator
and count as failures.

- Official cohort ID SHA-256:
  `a5ba75493d58fe5744a8c96552e7aa5cd339d7fd867b8189b687533a530418b2`.
- OpenBind metadata SHA-256:
  `389a7edca3ac8034d6533da5a3f3235619e7206aef7284441fd52d350bb1c652`.

## Pose-level measurements

### PoseBusters validity

Use PoseBusters 0.6.5 `redock`. A pose is valid only when all 27 non-RMSD
binary checks pass. The separate PoseBusters RMSD module is not part of
validity; OpenStructure supplies the endpoint RMSD. Missing or non-finite
checks count as failures.

### OpenStructure BiSyRMSD and LDDT-PLI

Use OpenStructure 2.11.1 `compare-ligand-structures` with:

- the same prepared protein as model and reference;
- explicit predicted and reference ligand SDFs;
- `--rmsd --lddt-pli -of csv -csvm`.

For one pose:

- `rmsd_valid = PoseBusters-valid and BiSyRMSD <= 2 A`;
- `success_valid = rmsd_valid and LDDT-PLI >= 0.8`.

The repository's RDKit symmetry-aware, no-alignment RMSD is retained as an
independent diagnostic and does not define these two endpoints.

## Complex-level aggregation

For `N` in `{1, 5, 25}`, a complex passes an endpoint when any pose with stable
confidence rank `<N` passes. Missing predictions are failures. OpenStructure
evaluation may stop after the first pose satisfying `success_valid`, because
that pose also establishes every later Top-N endpoint containing its rank;
otherwise every PB-valid Top-25 pose is evaluated.

Top-1 is the deployable confidence-selection result. Top-5 and Top-25 are
candidate-coverage measurements. The public OpenBind cross-method figure uses
the any-pose Top-25 endpoint, so only EFF-Dock Top-25 is aligned with that
comparison.

## Verification gates

- Require exactly 925 source metadata rows and exactly 802 admitted cohort IDs.
- Require exactly 100 readable candidate poses and 100 finite confidence rows
  for every predicted complex.
- Require complete, disjoint shard coverage and unique `(complex, rank)` rows.
- Require exact PoseBusters 0.6.5 and OpenStructure 2.11.1 versions.
- Require all missing predictions, failed checks, and failed evaluator calls to
  remain explicit; no denominator repair or row dropping is permitted.
- Generate the aggregate only after every PoseBusters and OpenStructure shard
  completes successfully.

The official comparison table and plotting code are versioned in the
[OpenBind benchmark repository](https://github.com/OpenBind-Consortium/EV-A71_2A_benchmark).
