# Supplementary Information: pocket-input contracts

This note fixes the site-conditioning definition for every row in
`docs/results/current_overview/figures/pocket_model_comparison.pdf`.  It covers
the exact final variants in that figure: the force-optimized SurfDock arm is
shown simply as **SurfDock**; the unoptimized SurfDock diagnostic is not a
figure row.

All locally executed learned methods are **supplied-pocket redocking** arms:
the reference ligand is used to form the method's site/pocket representation,
but never supplied as the candidate pose to be reported.  None of these rows
performs blind pocket discovery.  The distance values below are deliberately not
treated as interchangeable: depending on the method, an Å value can be a
ligand-contact shell, a receptor crop radius, a surface-face mask, or a
scoring-graph radius.

| Figure row | Receptor / pocket source | Pocket definition | Center used for sampling or crop | Reference-ligand use |
|---|---|---|---|---|
| **EFF-Dock** | Fixed holo receptor | The frozen site center is the centroid of receptor residue virtual nodes whose minimum distance to a reference-ligand atom is **≤8 Å**.  The final benchmark inference crops the receptor at **10 Å** about this frozen center. | **Pocket center, not ligand center.** The stored point is the 8-Å contact-residue centroid; it falls back to the ligand-coordinate centroid only if no residue is selected. | Defines the frozen site center once per complex.  It is not used as a bound-pose input during ODE sampling. |
| **SigmaDock** | Fixed holo receptor | Official `pocket_distance_cutoff=5 Å`: residues are defined by ligand proximity.  Official pocket-COM noise (0.25 Å), distance noise (0.5 Å), and coordinate jitter (0.02 Å) remain enabled. | **No evaluator-supplied scalar center.** The upstream graph builder receives the reference ligand and derives the pocket residue set / internal pocket coordinates. | Atomwise reference-ligand coordinates define the 5-Å pocket residues. |
| **SurfDock** | Fixed processed holo receptor and molecular surface | A receptor pocket is first cropped at **8 Å** around the cognate ligand.  SurfDock then keeps surface faces only when every vertex is within **3 Å** of the ligand; the latter is a surface-face criterion, not a spherical pocket radius. | **No scalar center.** The model consumes a ligand-defined local surface patch rather than a supplied point center. | Defines both the 8-Å local receptor/surface input and the 3-Å surface-face mask.  The plotted arm runs released `--force_optimize` on all candidates before the native MDN selector. |
| **RLDiff RL++** | Fixed holo receptor | First select receptor Cα atoms within **5 Å** of any reference-ligand atom.  Let their centroid be `c`; retain receptor residues within the variable radius `R = max_ligand_atom ||x - c|| + 10 Å`. | **Pocket Cα centroid.** This is a receptor-derived center calculated after the 5-Å ligand-contact selection, not the ligand centroid. | Supplies the atomwise ligand coordinates used for the 5-Å Cα contact rule and for the variable crop radius. |
| **DiffDock-Pocket** | Holo-aligned predicted receptor used by the released pocket model | Uses the same model-native pre-ESM crop as RLDiff: 5-Å ligand-contact Cα selection followed by `R = max_ligand_atom ||x - c|| + 10 Å`. | **Pocket Cα centroid** `c`, derived on the aligned predicted receptor; not a ligand centroid. | Supplies the reference ligand only to define the local crop. |
| **DiffBindFR + MDN/EC** | Fixed holo receptor | The upstream DiffBindFR / MDN scoring graph uses a **12 Å** pocket radius.  In this benchmark frame, `crystal_ligand` is present, so the code supplies its full coordinate set to construct that graph. | **Ligand-defined coordinate cloud, not a frozen point center.** The upstream supports a `center` fallback only when no crystal ligand is provided; that fallback was not used here. | Full reference-ligand coordinates define the 12-Å scoring pocket; the final row uses official smina error correction followed by MDN ranking. |
| **GOLD** | Published PoseBusters aggregate result | **Not recoverable from the deposited aggregate CSV used for this figure.** It does not preserve a per-complex binding-site radius or cavity definition. | **Not recoverable from this result artifact.** | The plotted value is the paper-deposited supplied-pocket result, not a local rerun.  We therefore do not infer a center or cutoff from the aggregate table. |
| **AutoDock Vina** | Published PoseBusters aggregate result | **Not recoverable from the deposited aggregate CSV used for this figure.** It does not encode each docking box or its dimensions. | **Not recoverable from this result artifact.** | No local Vina result is used in the reported comparison; this row is the paper-deposited supplied-pocket value. |

## Interpretation and comparability

`pocket cutoff` therefore means different things across the rows.  EFF-Dock's
8 Å value builds a stable reference-derived **point center**, followed by a
10 Å inference crop.  SigmaDock's 5 Å value is a ligand-proximity rule used to
make the graph.  RLDiff and DiffDock-Pocket use a 5 Å contact rule only to
obtain a receptor Cα centroid, then use a ligand-size-dependent crop.  SurfDock
uses an 8 Å receptor/surface neighborhood plus a separate 3 Å surface mask, and
DiffBindFR's 12 Å value is the MDN graph radius.  Reporting them as a shared
"pocket radius" would be incorrect.

The two classical rows are retained because they are the official deposited
PoseBusters comparison values.  Their source table contains outcome metrics but
not the per-complex site construction fields needed to reconstruct a precise
cutoff or center.  They are consequently reported as published aggregate rows,
with no fabricated site specification.  In particular, the Vina row is not a
locally executed result in this comparison.

## Evidence map

- EFF-Dock frozen centers: `data/external_test/*_reference_pocket_centers.json`;
  center construction: `src/effdock/inference/preprocess.py` and
  `src/effdock/workflows/benchmark_data.py`.
- SigmaDock official graph settings:
  `external_models/src/sigmadock/conf/sampling/base.yaml`; frozen invocation:
  `benchmarks/external_models/slurm/external_sigmadock_inference.sbatch`.
- SurfDock pocket provenance and surface preparation:
  `benchmarks/external_models/prepare_surfdock_runtime.py`; force arm:
  `benchmarks/external_models/slurm/external_surfdock_force_optimize.sbatch`.
- RLDiff and DiffDock-Pocket crop implementation:
  `benchmarks/external_models/prepare_rldiff_inputs.py` and
  `benchmarks/external_models/prepare_diffdock_pocket_inputs.py`.
- DiffBindFR 12-Å MDN graph construction:
  `external_models/src/diffbindfr/DiffBindFR/scoring/dataset/pipeline.py` and
  `external_models/src/diffbindfr/DiffBindFR/app/predict.py`.
- Classical source-table provenance:
  `benchmarks/results/external_models/posebusters_classical_paper_values.json`.
