# Chemical-constraint unit-weights smoke protocol

Status: frozen descriptive mechanism check; not a production-setting selection.

- Question: what changes when the existing physical-plus-interaction direct
  drift and the separately normalized chemical-constraint drift both use
  outer strength 1?
- Paired baseline: physical-plus-interaction outer strength 1 and chemical
  strength 0.
- Intervention: physical-plus-interaction outer strength 1 and chemical
  strength 1.
- Inputs: `7sdd_4ip`, `6xht_v2v`, and `8f4j_pho`; the same cases and frozen
  input manifest used by the earlier mechanism check.
- Sampling: S10, N100, translation sigma 2, seed 42, shared prior pool,
  late-power-3 schedule, and a common t=0.5 guidance start gate.
- Checkpoint: `weights/effdock_docking_early_time_t0p10_50k.pt`.
- Primary readouts: declared-stereo-valid candidate count and paired coordinate
  displacement.
- Guard readouts: symmetry-aware RMSD-under-2-A count, fast-valid count,
  joint counts, nonfinite events, and chemical-channel cap triggers.
- Prediction: chemical strength 1 increases declared-stereo preservation over
  the paired chemical-zero arm, with a cap-saturated response and little change
  in RMSD-under-2-A candidate count.
- Interpretation boundary: these cases and earlier external outcomes were
  already opened. Results are descriptive only and cannot select a strength or
  admit guidance to production.

“Weight 1” means the outer normalized-drift strengths. Versioned term-level
physical and interaction constants retain their units and are not overwritten
with ones.
