# Frozen confidence benchmark protocol

Protocol ID: `EFFDOCK-CONFIDENCE-EXTMATCH-N80-S25-V1`

Frozen: 2026-07-19, before the active EFF-Dock confidence rerun.

## Scope

This is the same reference-defined, oracle-pocket redocking diagnostic described
in `BENCHMARK_PROTOCOL.md`. It is not a target-independent pocket-finding or
prospective screening result. Reference coordinates define the frozen pocket
center and the RMSD label only; the selector never receives RMSD.

Prediction unit is one receptor/ligand/pocket complex. The identical 80 sampled
poses feed every selector so first, Vina+DG, learned confidence, frozen composite
confidence, and oracle results are paired within each complex.

## Pre-registration

- Question: does the retained trained confidence model improve pose selection
  over the physical Vina+DG baseline on the same N80 candidate sets?
- Hypothesis: the extmatch confidence model, evaluated on its matched sampling
  distribution, closes part of the Vina-to-oracle selection gap.
- Prediction: the frozen confidence selector exceeds Vina+DG top-1 RMSD <2A on
  PoseBusters v2; Astex and PoseBusters reproduce the historical single-run
  results within 2 percentage points (81.18% and 77.60%, respectively).
- Baselines: first sampled pose, Vina+DG on the same candidates, and pure minimum
  predicted RMSD from the confidence head.
- Primary metric: PoseBusters v2 frozen-confidence selected top-1 symmetry-aware
  heavy-atom RMSD <2A.
- Secondary metrics: Astex selected top-1, pure-confidence top-1,
  Vina+DG, oracle top-80, median RMSD, fast validity, and failure rate.
- Failure threshold: invalidate a dataset if more than 2% of frozen IDs fail or
  if shard checkpoint/config/center hashes differ.

## Frozen execution

- Docking checkpoint: `weights/effdock_geometry_ft_100k_best.pt`
  (`sha256:6932fb3ba6ebac770f714453529656a44b8f33cf15119d23c9e675d2d60b36db`).
- Confidence checkpoint:
  `weights/effdock_confidence_extmatch_n80_s25_step42500.pt`
  (`sha256:e31fde6f351284205c78f7a1510002779c43312e94d9f82003d47a14d72bc78f`).
- Candidate generation: 80 poses, 25 ODE steps, sigma 0.5, late schedule power
  3, pocket cutoff 10A, no refinement, base seed 42 with deterministic global-ID
  offsets.
- Frozen selector:
  `pair_gate_density_rank_vote_plclash_ambig`, preserved exactly from the final
  historical single-run selector. It combines learned pose/atom heads, sample
  density, rank voting, and a protein-ligand clash fallback.
- Oracle: minimum RMSD among the same 80 poses; diagnostic only.

The prior N40/EMA protocol remains a separate physical compatibility baseline;
its results are not overwritten or presented as the confidence model result.

## Frozen outcome

Completed 2026-07-19 with 393/393 final rows. Frozen-composite <2A rates were
78.82% Astex and 72.73% PoseBusters; same-candidate Vina rates were 77.65% and
71.10%. The PoseBusters improvement prediction passed
(+1.62pp), while historical reproduction failed the ±2pp criterion on Astex
(-2.35pp) and PoseBusters (-4.87pp). Official PoseBusters pass-all validity was
54.87%. Full results and rescue provenance are in `BENCHMARK_RESULTS.md` and
`outputs/benchmarks/confidence/summary.json`.
