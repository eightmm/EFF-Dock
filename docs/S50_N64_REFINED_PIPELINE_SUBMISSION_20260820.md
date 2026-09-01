# S50 N64 + refined confidence pipeline submission

Status: historical/superseded submission ledger. The fixed-map N64 and
refined-only continuation chains below are preserved as provenance; they are
not the current confidence-training contract. The completed raw source is the
symmetry-target U50k job `56642`, documented in
`docs/S50_SYMMETRY_CONFIDENCE_RESULTS.md`. The current continuation contract
mixes raw, refined, and crystal-anchor poses under
`docs/S50_RAW_REFINED_CONFIDENCE_FINETUNE_PROTOCOL.md`.

- Failed raw N80 training: 55336 (preserved; OOM at U1960 before first latest).
- Mechanical N64 schema-error smoke: 56288 (preserved; no model update).
- Corrected N64 smoke: 56291 (completed U0/U1 on four heavy GPUs).
- Superseded short-wall full attempt: 56292 (cancelled at U0 and replaced).
- Raw N64 U50,000: 56312 (`verylong`, latest every 500 updates).
- Superseded refined coordinate smoke/full/aggregate: 56309 -> 56310 -> 56311.
  The full array exposed processed-receptor fallback tokens and a Ru ligand;
  its partial outputs remain preserved under the old content root.
- EFF-FF 2.2 recovery chain: label-free full-cohort preflight 56616 -> targeted
  Ru/token smoke 56617 -> full refined coordinates 56618 -> aggregate 56619 ->
  refined confidence feature smoke/full/aggregate 56620 -> 56621 -> 56622.
- Refined train symmetry-RMSD smoke/full/aggregate: 56623 -> 56624 -> 56625.
- Refined symmetry-target confidence continuation: `(56496 AND 56625) ->`
  four-GPU smoke 56626 -> full 10k job 56627. This branch warm-starts the
  completed 50k symmetry-label checkpoint, not the superseded fixed-map N64
  branch.
- Recovery content root:
  `outputs/eff-dock/s50-refined-pose-runs/207611a7fed8805b7f52cdf4648da2ea03e7d1e699c7d47106b632215634fdc2`.
- Refined N64 10k smoke/full: `(56312 AND 56320) -> 56324 -> 56325`.

The raw N100 pose bank is reused unchanged. Raw training samples stratified N64
and validates N100. Refinement transforms all 100 poses for every 43,092 train
and 1,035 validation complex. The refined continuation warm-starts raw U50,000
weights with a fresh lower-rate optimizer and writes both atomic `latest.pt`
and validation-selected `best.pt` in a separate directory.

## Dependency recovery (2026-08-20)

The original symmetry-target job `56496` was superseded by recovery job `56642`
after the large-graph stall investigation. The already-pending refined smoke
`56626` still referenced `afterok:56496`, which made the branch permanently
unsatisfiable even though no refined-training work had started. Its dependency
was updated in place, without changing code, inputs, or scientific settings, to
`afterok:56642,afterok:56625`. Full refined continuation `56627` remains
unchanged behind `afterok:56626`.

## Generic-obstacle label recovery (2026-08-21)

The EFF-FF 2.2 full coordinate array `56618` was cancelled after tasks 0--73
all failed with `geometry obstacle labels must be unique`; jobs `56619`--`56627`
were cancelled with it, while the independent raw symmetry-confidence training
job `56642` remained active. A controlled reproduction on
`1l6g__1__1.A_1.B__1.C__1.C` showed that distinct catch-all processed receptor
atoms in one residue were all reconstructed as the same synthetic PDB atom
name `X`. This was a provenance-label collision, not a coordinate, ligand
element, or energy failure.

The recovery assigns each catch-all atom a stable four-character identifier
derived from its one-based processed-atom order while retaining element `X`,
coordinates, atom inventory, and bounded generic repulsion. The original
partial root is preserved. The fresh content root is
`outputs/eff-dock/s50-refined-pose-runs/2682fdc5f37c517116cea7b722ecfe6d2530ef21bf1cfb6ad2dc58941bf49ebc`.
The new afterok chain is label-free preflight `57051` -> targeted GPU smoke
`57052` -> full 136-task array `57053` -> exact aggregate `57055`.
