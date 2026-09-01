# S50 refinement budget calibration protocol

Protocol ID: `EFFDOCK-S50-REFINEMENT-BUDGET-CALIBRATION-V4`.

## Objective

Reduce the wall time of the sealed S50 pose-bank rigid-fragment refinement
without using crystal RMSD, confidence, success, or validation outcomes to tune
the solver. This calibration changes only the numerical stopping/budget policy;
the source poses, receptor construction, GuidanceEnergy, coordinate variables,
step caps, and line search remain fixed.

## Frozen inputs

- Source: the sealed 43,092-train / 1,035-validation S50 confidence bank.
- Poses: the same 100 stored sigma-2 poses per complex.
- Calibration IDs, in order:
  `1h08__1__1.A__1.B_1.C__1.C`,
  `3o7u__1__1.A__1.C_1.E__1.E`, and
  `1a5s__1__1.A__1.C__1.C`.
- The second ID is an observed line-search stress case in the superseded
  100-step partial run; this terminal-status information is numerical solver
  provenance, not an RMSD/confidence outcome.

## Arms

All arms use custom rigid-fragment SE(3) gradient descent: GuidanceEnergy
autograd forces are projected to fragment translation and rotation, clipped to
0.10 Angstrom / 5 degrees / 0.10 Angstrom maximum atom motion, and accepted by
pose-wise monotone backtracking with at most 12 backtracks. No Adam, AdamW,
SGD, or learned optimizer is used.

- `baseline_100`: maximum 100 steps; displacement below `1e-5 Angstrom` for
  20 consecutive accepted updates; energy-plateau stopping disabled.
- `adaptive_75`: maximum 75 steps; displacement below `0.01 Angstrom` for
  5 consecutive accepted updates; from step 25, energy decrease below
  `0.02 kcal/mol + 1e-3 * max(1 kcal/mol, abs(E))` for 5 consecutive accepted
  updates.
- `adaptive_50`: identical adaptive rules with maximum 50 steps.

The completed V1 label-free calibration found that `adaptive_50` and
`adaptive_75` passed runtime, p95-energy, p95-coordinate, and line-search
gates, but narrowly missed the frozen median gates. V2 does not relax any V1
gate. It adds one conservative rescue arm:

- `adaptive_90`: identical adaptive rules with maximum 90 steps.

The V2 execution reruns only `baseline_100` and `adaptive_90` on the same
ordered calibration IDs. V1 remains immutable negative numerical evidence.

V2 found that `adaptive_90` passed five of six unchanged gates and missed only
the median-energy gate (`0.544` versus `0.5 kcal/mol`). V3 again leaves every
gate unchanged and isolates the source of the difference while retaining 100
steps as a hard safety ceiling:

- `adaptive_100`: maximum 100 steps with both the `0.01 Angstrom` displacement
  and the V1 energy-plateau stop;
- `displacement_100`: maximum 100 steps with only the `0.01 Angstrom`
  displacement stop; energy-plateau stopping disabled.

V3 reruns `baseline_100` and these two arms on the same ordered IDs. It selects
the fastest arm passing all unchanged gates. V1 and V2 reports remain immutable
negative numerical evidence; no RMSD, confidence, or success outcome is used.

V3 found `displacement_100` physically near-identical to the baseline (median
energy increase `0.00011 kcal/mol`, median coordinate RMSD `0.00092 Angstrom`)
but its runtime ratio `0.879` narrowly missed the unchanged `0.85` gate.
`adaptive_100` was faster but narrowly missed the unchanged median-energy gate.
V4 therefore changes no step budget or physical stopping threshold and tests
the independent implementation lever suggested by the user:

- `batch20_100`: the exact `baseline_100` solver with pose batch size 20 rather
  than 10;
- `displacement_100_b20`: batch size 20 plus the physically near-identical
  `0.01 Angstrom` displacement stop, with energy-plateau stopping disabled.

V4 compares both arms directly with the rerun batch-10 `baseline_100` using the
same unchanged energy, coordinate, line-search, and runtime gates. Batched
solver acceptance and stopping remain pose-specific; no batch reduction decides
whether another pose advances.

## Label-free gates and decision

Each arm must return all 100 poses per complex with finite coordinates and
energies and only declared terminal statuses. Relative to `baseline_100`, a
candidate must satisfy every gate over the pooled frozen calibration poses:

- no increase in `line_search_failed` count;
- elapsed-time ratio at most `0.85`;
- median / p95 terminal-energy increase at most `0.5 / 5.0 kcal/mol`;
- median / p95 same-index coordinate RMSD at most `0.1 / 0.5 Angstrom`.

Within each protocol iteration, select the shortest passing candidate. If no
candidate passes, retain `baseline_100`. The calibration report is sealed before
any full rerun. A changed solver uses a fresh content-addressed root; partial
V1 coordinates are preserved and never mixed with the new bank.
