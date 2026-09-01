# Evaluation contract

The active external suite is PoseBusters v2, Astex Diverse, PhiBench,
FoldBench, and the auxiliary OpenBind cohort. Raw structures stay ignored
locally, while snapshot identity, molecule mappings, and explicit pocket
definitions are frozen in manifests. PhiBench and FoldBench are the core
temporal checks; OpenBind is reported separately because it is a dense
single-protease cohort.

The primary metric is selected top-1 symmetry-aware ligand RMSD success at
`<2A`. Secondary reporting includes oracle success, official PoseBusters
validity, the conjunction of validity and `<2A`, and slices by ligand size,
fragment count, rotatable bonds, ligand similarity to train, and pocket
similarity to train.

Evaluation requires a JSON mapping each benchmark complex ID to an explicit
`[x, y, z]` pocket center. Missing IDs fail before sampling. Crystal ligand
coordinates are used only as RMSD targets and never to define the model input
pocket.

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
an implicit `eff-dock dock` step. Exact U70k counts, the U50k/U100k comparisons,
and the repeated-use evaluation boundary are in `BENCHMARK_RESULTS.md`.

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
