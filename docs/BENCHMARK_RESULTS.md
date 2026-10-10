# Benchmark results

The canonical article,
Supplementary Information and
[numerical Source Data](../papers/data) contain the manuscript results and
evaluation definitions. A compact primary-result table is in the
[repository README](../README.md#main-results).
Figure captions specify the protocol for each panel.

## Primary protocol

The primary evaluation uses the frozen released docking/confidence weights,
explicit `rw` angular features, `contextual_v1` confidence frames and ligand
reference conformers reconstructed from the generation preparation. Sampling
is unguided N100/S10 with translation sigma 2 Å and a supplied 10 Å pocket,
followed by energy refinement and input-chirality-filtered minimum-predicted-RMSD
selection, including the all-ineligible fallback.

RMSD–PoseBusters success requires the same selected pose to have symmetry-aware
heavy-atom RMSD <2 Å without structural alignment and pass all 27 non-RMSD
PoseBusters 0.6.5 checks. An unavailable required check is conservatively
nonpassing. Every complex remains in the binary-rate denominator. Means and
sample SD use three fixed-weight inference repeats; paired bootstrap intervals
are identified separately.

## Benchmark-specific analyses

PhiBench-derived comprises 206 locally selected cases; identity with the
published PhiBench cohort and its PAL-RMSD/18-check endpoint is not established.
FoldBench uses 558 supplied-holo-receptor redocking interfaces. Its separate
native-score analysis uses binding-site-superposed symmetry-aware RMSD <2 Å
and LDDT-PLI >0.8 on the same pose. This does not reproduce the source cofolding task.
OpenBind retains all 925 local cases in the primary analysis; the separate
benchmark-specific analysis uses the official 802-case follow-on subset.
Its RMSD ≤2 Å, PoseBusters and additional LDDT-PLI ≥0.8 conjunctions are reported
in the Supplementary Information. Cohorts, preparation, training exposure,
candidate budgets and tasks differ between these analyses and published models.

Initial acquisition timings exclude the later conformer-template and
confidence-frame correction stages. They are not the end-to-end cost of the
current primary pipeline. Guided controls, sampling-input perturbations and
illustrative trajectories retain their individually specified protocols.
