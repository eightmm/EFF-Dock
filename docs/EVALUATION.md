# Evaluation contract

The active external suite is PoseBusters v2, the Astex Diverse Set, a locally
selected PhiBench-derived cohort, FoldBench-Pocket, and the auxiliary OpenBind
cohort. Raw structures stay ignored
locally, while snapshot identity, molecule mappings, and explicit pocket
definitions are frozen in manifests. PhiBench-derived and FoldBench-Pocket
adapt their source tasks to supplied-pocket redocking; OpenBind is a dense
single-protease-family cohort.

The primary metric is selected top-1 symmetry-aware ligand RMSD success at
`<2A`. Secondary reporting includes oracle success, official PoseBusters
validity, the conjunction of validity and `<2A`, and slices by ligand size,
fragment count, rotatable bonds, ligand similarity to train, and binding-chain
sequence identity to train. Sequence relatedness does not establish pocket identity.

Evaluation requires a JSON mapping each benchmark complex ID to an explicit
`[x, y, z]` pocket center. Missing IDs fail before sampling. Frozen benchmark
pocket centres may be derived from reference ligands in this
retrospective redocking setting. The reference pose is used for mapping and RMSD
evaluation, not as a learned model/scorer input or an energy-refinement target.

## Interaction recovery

Each evaluated complex also records distance-based interaction-fingerprint
recovery for the first sampled pose and the RMSD oracle pose, when the ligand
has a complete connectivity-preserving atom mapping. The metric compares typed
ligand–protein contacts in the pose with contacts in the crystal pose, using
the same fixed receptor and ligand atom order. Equivalent ligand atoms are
grouped by RDKit canonical symmetry rank. It reports per-class and micro
precision/recall/F1, plus macro-F1 over interaction classes present in the
reference. A class with no reference contacts has undefined recall and F1 and
does not contribute to macro-F1.

The initial classes and inclusive distance ranges are hydrogen bonds (2.2–3.5
Å), salt bridges (2.0–4.0 Å), hydrophobic contacts (3.0–4.5 Å), and ligand
donor/acceptor-to-metal contacts (1.8–2.8 Å). Hydrogen bonds are distance-only:
the receptor representation has no hydrogens, so donor-angle geometry is not
evaluated. Waters are omitted by the current receptor parser, and this metric
does not yet score water bridges, aromatic stacking, or covalent interactions.
These are declared geometric proxies, not a complete interaction-energy model.

Results are stored in `interaction_recovery_by_selector_json`, keyed by the
pose selectors in that evaluation row (including `first` and `oracle`).
`interaction_recovery_available` and `interaction_recovery_unavailable_reason`
explain missing measurements. This diagnostic supplements RMSD and PoseBusters
and does not affect pose ranking.

```bash
uv run eff-dock evaluate \
  --dataset astex \
  --data-dir data/external_test/astex \
  --pocket-centers data/external_test/astex_reference_pocket_centers.json
```

The promoted deployment pair is
`weights/effdock_docking_early_time_t0p10_50k.pt` with
`weights/effdock_confidence_s50_raw_refined_u70k.pt`. N100/S10, translation
sigma 2.0, a 10A pocket crop, late-power-3 time sampling, and pure minimum
predicted-RMSD ranking are the defaults for both `dock` and `evaluate`. Every
value remains overridable; `--no-confidence` runs generator-only diagnostics.
U70k was selected only on the fixed 1,035-complex PLINDER validation bank at
622/1,035 (60.10%) Top-1 `<2A`.

Active selection choices are sampling order, Vina+strain, pure learned
confidence, the cluster-free confidence filter diagnostic, and the frozen
historical composite. When a confidence checkpoint is supplied, `auto` uses
pure predicted-RMSD confidence; it does not use pose clusters, candidate ranks,
or pairwise pose distances. All selectors consume the identical sampled pose
set and the evaluator records each result separately. External test results
must not drive further training, hyperparameter choice, or pocket-center
construction.

The retained confidence outputs are ranking signals, not calibrated RMSD or
success probabilities. `confidence_filter_v1` is an explicit conservative
diagnostic; it did not meet its PLINDER validation deployment gate. The frozen
`pair_gate_density_rank_vote_plclash_ambig` selector is retained for historical
reproduction. The completed active diagnostic found pure confidence slightly
stronger overall than this composite, so all results remain separate and none
may be used to tune a new external-test selector post hoc.

The public sampler writes raw poses. The benchmark's `Refined` columns use a
separate deterministic physical-refinement pass and must not be interpreted as
an implicit `eff-dock dock` step. Equations, numerical settings and the
chirality-selector fallback are specified in
[the inference methods](methods/07_inference_and_evaluation.md). Internal
checkpoint selection is documented in the
[confidence model card](../weights/CONFIDENCE_MODEL_CARD.md).

## Reported benchmark results

The [README table](../README.md#benchmark-results) summarizes the refined,
chirality-filtered primary condition for five fixed cohorts. Its full-precision
means and sample SDs are in [benchmark_summary.csv](benchmark_summary.csv).
Values are percentages and use all complexes in each cohort as the denominator.

- **Top-1 RMSD:** the selected pose has symmetry-aware, no-alignment heavy-atom
  RMSD strictly below 2 Å in the fixed receptor frame.
- **Top-1 RMSD + PB:** that same pose also passes all 27 non-RMSD checks in
  PoseBusters 0.6.5, `redock` mode. The PhiBench-derived, FoldBench and OpenBind
  evaluations normalize the internal-energy reference molecule through a
  mapped SMILES round trip, preserving atom identities, connectivity,
  canonical isomeric SMILES and InChI. Pose coordinates, thresholds and the
  27 check definitions are unchanged.
- **Oracle@100:** at least one refined candidate has RMSD below 2 Å. This
  retrospective endpoint does not require PoseBusters validity and is not
  selected-pose performance.

All rows use the frozen released docking/confidence pair, explicit `rw`
orientation injection and `contextual_v1` frames, 100 candidates, 10 late-power-3
steps, translation-prior sigma 2 Å, a 10 Å generation crop, no center jitter
and no in-ODE guidance. Each candidate is refined for at most 100 iterations.
Selection takes the lowest predicted RMSD among input-chirality-compatible
candidates, falling back to the unfiltered top-ranked pose when none qualifies.
This is a separate post-generation workflow, not the raw-pose quick-start command.

The three repeats use inference seeds 42, 100042 and 200042 with the same
trained weights. Astex and PoseBusters fix the starting conformer; the remaining
cohorts also vary its generation seed. SD measures inference variability,
not independent training runs or confidence intervals.

Cohorts are Astex Diverse Set (85 complexes), PoseBusters v2 (308), the locally
selected PhiBench-derived cohort (206), FoldBench-Pocket (558 interfaces), and
the primary OpenBind cohort (925 complexes). The PhiBench-derived cohort is
not established to be identical to the published 206-case set. FoldBench
results here use fixed-receptor RMSD, not its native cofolding endpoint;
OpenBind uses the full primary cohort, not the official 802-case follow-on
subset. The latter is a single-protease-family auxiliary evaluation.

Astex and PoseBusters were repeatedly inspected during development. The
historical ligand-and-pocket exclusions do not establish complete independence
from every benchmark; see the
[training-membership audit](../benchmarks/inputs/training_membership/README.md).
These protocol-specific rates are not a matched-compute cross-model comparison
or evidence of performance on wholly unrelated targets.

### Numerical provenance

The table is an extract of the manuscript's `data/source_data.json`, schema
`effdock.ncs_source_data.v1`. Selected-pose values come from
`values.benchmark_results.rows` with `stage=refined` and `policy=filtered`;
oracle values come from `values.candidate_metrics.rows` with `stage=refined`.
Rows are joined by dataset and checked for matching cohort size and candidate
budget. The CSV retains unrounded means and sample SDs; the README rounds
each to one decimal place.

Source SHA-256: `4b41dc2d5cbb1f8430bfd0c45ed6f497cbcf383142894cc9ad495936423fb7cf`.

## Candidate SDF outputs

Candidate persistence is part of the inference contract. Public `dock` writes
all generated poses to `docked_poses.sdf` (or the one-record `docked.sdf` for a
single-pose request). By default, benchmark `evaluate` writes one
multi-record `all_poses/<complex-id>.sdf` beside its selector-specific pose
directories. Records stay in sampling order and carry `sample_index`, the
candidate-ensemble hash, and every available confidence output as SDF molecule
properties. The active `confidence_*` names and historical
`confidence_pred_*` aliases are both written. This complete candidate artifact
is required for post-hoc, label-blind reranking without repeating sampling.
`--no-save-selected-poses` explicitly opts out of both the complete ensemble
and selector-specific pose artifacts.

## Archived Vina-guided sampling (inactive)

This path is retained only for historical reproduction and is not the unified
`GuidanceEnergy` or recommended inference path. It can act inside the ODE
sampler, which is distinct from scoring a finished candidate set. The
inference-time callback differentiates the Torch Vina+DG energy with respect to
current ligand coordinates, converts atom forces to fragment
translation/torque, and adds the capped late-time correction to the learned
vector field. Scale zero is the exact unguided path.

```bash
uv run eff-dock dock ... \
  --vina-guidance-scale 0.05 \
  --vina-guidance-start-t 0.5 \
  --vina-guidance-max-force 10
```

The scorer's explicit XS-type path matches the official Vina 1.2 potential,
weights, radii, flags, center-distance cutoff, and torsion normalization. The
current benchmark inputs are PDB/RDKit, so their automatic atom typing is a
declared best-effort compatibility path rather than exact PDBQT protonation.
See `VINA_GUIDANCE_PROTOCOL.md` for the frozen experiment and attribution.
