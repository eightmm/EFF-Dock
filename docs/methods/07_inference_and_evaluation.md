# Production inference and evaluation contract

Implementation and evidence sources: `README.md`, `docs/EVALUATION.md`,
`docs/BENCHMARK_RESULTS.md`, model cards in `weights/`, and the released
checkpoint manifest.

## 1. Fixed public sampler

The released pair is
`effdock_docking_early_time_t0p10_50k.pt` plus
`effdock_confidence_s50_raw_refined_u70k.pt`. Given a receptor, ligand, and an
explicit pocket centre, production inference uses:

| quantity | fixed value |
|---|---:|
| receptor crop | 10 A |
| translation prior sigma | 2.0 A |
| candidates | N100 |
| ODE steps per candidate | S10 |
| integration times | late schedule, power 3 |
| ranking | minimum U70k predicted RMSD |

The ODE evolves all fragment translations and rotations from the priors in
`02_fragment_se3_flow.md`. It emits all 100 candidates in generation order;
confidence fields, ensemble identifiers, and diagnostic energy fields (when
computed) are persisted rather than overwriting a pose. The selected output is
the candidate with the smallest predicted RMSD. No score mixing, energy
reranking, FK-SDE, Vina, or differentiable-energy guidance is part of this
production contract.

## 2. Raw and refined conditions

`Raw` denotes direct ODE samples. `Refined` denotes a separately executed,
labelled post-sampling condition. It is not implicit in the public API and its
poses must never be pooled silently with raw poses. Each reported row names
the condition, candidate budget, confidence checkpoint, ranking rule, and
pocket definition.

Diagnostic SDF energy fields are metadata only under the current release:
total energy, physical term, signed interaction term, non-negative violation
term, and energy change. The public ranker remains U70k predicted RMSD. These
fields cannot be interpreted as affinity or free energy.

## 3. RMSD and validity endpoints

The primary docking endpoint is selected Top-1 heavy-atom RMSD `<2 A` using
symmetry-aware, no-alignment RDKit `CalcRMS`. Top-k success applies the same
label to the best member among the top `k` confidence-ranked poses; oracle
success uses the best RMSD among all generated candidates and is explicitly
not a deployable selector.

PoseBusters validity is calculated with the official evaluator on the same
selected coordinates. The joint endpoint requires both the RMSD condition and
validity for that exact pose--it is not a conjunction across different poses.
Failures to construct inputs, sample all candidates, map ligand atoms, or
evaluate a required endpoint are reported as coverage failures, never silently
dropped from a denominator.

## 4. Evaluation scope

Completed cohorts are Astex Diverse (85), PoseBusters v2 (308), PhiBench (203
reconstructed systems), FoldBench-Pocket full (558), and auxiliary OpenBind
(860). All headline comparisons are supplied-pocket redocking. They do not
support claims about blind pocket discovery, co-folding, affinity prediction,
or virtual-screening classification.

Astex and PoseBusters have been inspected during development and are
descriptive. U70k was selected only on the fixed internal PLINDER bank
described in `06_training_and_checkpoint_selection.md`. PoseX-SD/CD remains a
separate, ongoing official-evaluation track; fixed-receptor relaxation is
labelled as such and must not be reported as unmodified official relaxation.
