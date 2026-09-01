# S50 refined pose bank protocol

Protocol ID: `EFFDOCK-S50-REFINED-POSE-BANK-V2`.

## Objective

Create a paired refined-pose bank for every eligible complex in the sealed S50
confidence bank while the raw N64 confidence model trains. The source inventory
is unchanged: 43,092 train and 1,035 validation complexes, each with the exact
stored 100-pose S50/sigma-2 ensemble.

## Frozen transform

- Input coordinates: the sealed pocket-centred N100 raw-bank tensor.
- Variables: rigid fragment SE(3) translation and rotation only.
- Energy: the repository's unified `GuidanceEnergy` implementation.
- Receptor: the sealed processed-protein atom coordinates and `(residue, atom)`
  tokens reconstructed into a deterministic PDB representation; no crystal
  ligand coordinates are exposed to the energy.
- Receptor policy: `geometry_only`.
- Processed receptor compatibility: canonical residue/metal tokens retain their
  exact identities; unknown-residue backbone tokens retain their known atom
  element under an `UNK` residue; catch-all atom/metal tokens become bounded
  generic geometry obstacles with no attractive term. Because multiple lossy
  catch-all atoms can occur in one processed residue, each receives a stable
  synthetic atom identifier derived only from its one-based processed-atom
  order. This identifier affects provenance labels only; coordinates, element
  dispatch, energy, and atom inventory are unchanged.
- Solver: at most 100 steps, batch 20, step size 1.0, max translation 0.10 Angstrom,
  max rotation 5 degrees, max atom motion 0.10 Angstrom, 12 backtracks,
  displacement convergence 0.01 Angstrom for 5 consecutive accepted steps,
  energy-plateau stopping disabled, 18-Angstrom receptor
  shell and 8-Angstrom physical cutoff.
- Output: terminal coordinates, terminal status/step/backtracks, and exact
  source/refined coordinate identities. RMSD, success and confidence labels are
  not used to admit or exclude records.

Tokens outside the frozen vocabulary, non-finite energies/forces/coordinates,
unusable terminal statuses, source hash drift, missing records, or any partial
inventory fail the shard/aggregate. The two declared catch-all vocabulary
tokens follow the geometry-only downgrade above. No record is silently dropped
or replaced.

The worker snapshots its implementation identity once at process start. For
the completed V2 recovery bank, older workers re-read repository provenance for
every complex; 1,454/44,127 records therefore observed a changed
`pyproject.toml` hash while the already-imported executable modules and frozen
parameters remained unchanged. Recovery aggregation requires every executable
file list, runtime version, `uv.lock`, parameter identity, and Torch identity to
match the label-free preflight exactly. It normalizes only the observational
`pyproject.toml` hash and its derived guidance digest, and records all observed
hash/count pairs in the sealed manifest. Any other drift remains fail-closed.

The EFF-FF 2.2 element table covers all 33 atomic numbers observed in the
frozen 44,127-ligand S50 train/validation cohort, including Ru and the other
rare metal/metalloid elements. UFF x/D constants and versioned vdW radii give
finite diagnostic repulsion; this is not a claim that transition-metal
coordination chemistry is physically complete.

## Execution and continuation

A CPU-only, label-free preflight scans all 44,127 frozen records before any new
GPU output is admitted. It requires exact source hashes, parses every canonical
SMILES, proves that the observed atomic-number set equals the 33-entry EFF-FF
table, reloads and hashes every processed receptor, and reconstructs every
protein token into the declared synthetic-PDB geometry policy.

A targeted three-train/two-validation GPU smoke must pass before the full
128-train and 8-validation arrays start. The train probes include the observed
Ru ligand and the processed-receptor unknown-residue/catch-all-token failures
from the superseded array, including a repeated-catch-all case that previously
triggered duplicate geometry-obstacle labels. The refinement bank is independent of confidence
training and may run concurrently with raw N64 training. Refined confidence
features and labels are materialized only after the coordinate bank is sealed.
The refined-confidence training branch may start only after both the raw N64
50k job and the refined confidence bank complete; it warm-starts the selected
raw N64 checkpoint and uses a fresh optimizer/scheduler in a separate output
directory.

## V2 numerical calibration

The superseded V1 full array was stopped after 423/44,127 complexes because
99.4% of observed poses exhausted all 100 steps; its partial root is preserved
and is never mixed with V2. The label-free calibration protocol and immutable
reports are recorded in `docs/S50_REFINEMENT_BUDGET_CALIBRATION_PROTOCOL.md`.

Hard caps of 50, 75, and 90 steps all missed an unchanged terminal-energy gate.
Keeping 100 as a safety ceiling while enabling only the 0.01-Angstrom
displacement stop was physically near-identical but saved about 12% at batch
10. Increasing the pose batch from 10 to 20 and using the same displacement
stop passed every unchanged gate on a normal train complex, an observed
line-search stress complex, and a validation complex. Relative to batch-10
fixed-100, the selected calibration arm had runtime ratio 0.447, median/p95
energy increase 0.000072/0.1469 kcal/mol, and median/p95 same-index coordinate
RMSD 0.0010/0.0935 Angstrom. It also passed on a 24-GB A5000, providing a
conservative memory gate for the 48-GB full-run device family.
