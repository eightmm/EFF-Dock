# Evaluation contract

External benchmarks are PoseBusters v2, Astex Diverse, and CASF-2016. Their raw
structures stay ignored locally, while snapshot identity, molecule mappings,
and target-independent pocket definitions must be frozen in manifests.

Until target-independent centers are supplied, the compatibility benchmark in
`BENCHMARK_PROTOCOL.md` is reported separately as a reference-defined
oracle-pocket redocking diagnostic. It does not satisfy the prospective/public
inference pocket contract and is not substituted for it.

The primary metric is PoseBusters selected top-1 symmetry-aware ligand RMSD
success at <2 Angstrom. Secondary reporting includes Astex and CASF top-1,
oracle top-k, PoseBusters validity, and slices by ligand size, fragment count,
rotatable bonds, ligand similarity to train, and pocket similarity to train.

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

The promoted geometry/confidence checkpoints and the N80/S25/sigma0.5/pocket10
preset are the defaults for both `dock` and `evaluate`. Every value remains
overridable; `--no-confidence` runs generator-only diagnostics.

Active selection choices are sampling order, Vina+strain, pure learned
confidence, the cluster-free confidence filter diagnostic, and the frozen
historical composite. When a confidence checkpoint is supplied, `auto` uses
pure predicted-RMSD confidence; it does not use pose clusters, candidate ranks,
or pairwise pose distances. All selectors consume the identical sampled pose
set and the evaluator records each result separately. The N40/EMA physical
baseline and N80/extmatch confidence protocol are distinct frozen studies and
must not be merged into one claimed ablation. External test results must not
drive further training, hyperparameter choice, or pocket-center construction.

The retained confidence outputs are ranking signals, not calibrated RMSD or
success probabilities. `confidence_filter_v1` is an explicit conservative
diagnostic; it did not meet its PLINDER validation deployment gate. The frozen
`pair_gate_density_rank_vote_plclash_ambig` selector is retained for historical
reproduction. The completed active diagnostic found pure confidence slightly
stronger overall than this composite, so all results remain separate and none
may be used to tune a new external-test selector post hoc.

The completed S50 symmetry-confidence experiment does not alter public
defaults. Its internal rule selected U25k, while U50k is the terminal state;
both require explicit checkpoint paths. Their Astex/PoseBusters comparison is
repeated-use and descriptive and cannot promote either checkpoint. Exact
metrics and hashes are in `S50_SYMMETRY_CONFIDENCE_RESULTS.md`.

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
