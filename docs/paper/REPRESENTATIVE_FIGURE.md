# Figure 1 representative image

- [PDF](figures/Fig1_representative.pdf)
- [Editable SVG](figures/Fig1_representative.svg)
- [PNG preview](figures/Fig1_representative.png)

## Manuscript caption

**Fragment-based pose generation, post-generation refinement and confidence selection.**
All model evaluations in this figure use corrected Rw with frozen released weights.
This known-pocket redocking overview combines an illustrative N1 generation/refinement
trajectory and a separate corrected-Rw N100 candidate bank for the same Astex 1T46–STI complex;
it does not depict a single end-to-end run. A supplied pocket center defines the
receptor crop, while the ligand is decomposed into rigid fragments. Fragment-level
SE(3) flow generates a pose (t = 0, 0.488, 0.784 and 1), followed by energy
minimization with physical and interaction terms (steps 0, 50 and 100). Refinement
starts from the displayed generated endpoint. Trajectories run top to bottom;
the workflow proceeds left to right. The confidence panel shows refined candidates
at predicted-RMSD ranks 1, 25, 50 and 100, with intervening candidates omitted.
Only the selected candidate carries a check mark and green border; unmarked poses
are not necessarily incorrect or physically invalid. pRMSD is the model prediction
used for ranking; RMSD is the retrospective symmetry-aware heavy-atom error against
the crystal ligand, in Å, and is not a selection input. The selected pose (blue)
is overlaid with the crystal reference (peach; RMSD 0.96 Å). Generation, refinement
and ranked candidates share the same camera and display scale; the final overlay
is enlarged for readability. Fragment carbon colors
remain consistent before this overlay. This single-case illustration does not
establish refinement convergence, improved RMSD or physical validity.

## Display and print layout

The standalone figure uses five stages, sans-serif typography, white molecular
panels and pale stage backgrounds. It has no lettered panels. Input branches
converge centrally: protein → extracted pocket downward, ligand → fragments upward.
The input labels **Given pocket center** and **Extracted pocket** distinguish a
supplied binding site from pocket prediction. The teal region is the retained
pocket; the terracotta dot marks the supplied center, projected above the surface
for visibility, not a molecular atom.

Generation has four saved states; refinement has three (0/50/100), reducing
visually redundant snapshots without exaggerating structural changes. The saved
step-25 capture is retained for reproducibility but omitted from the figure.
Generation, refinement and ranked candidates retain the same camera, common
crop and 30 mm molecular-image width. The final selected/crystal overlay uses
that same camera and crop at a 35 mm image width (7/6 of the display scale),
solely for readability. The common crop still includes the step-25 image so
omitting it does not alter framing.
Input views use different camera scales to show the complete receptor and pocket.
The same receptor appears behind downstream poses with wider chain context.

Only the selected candidate has a check and green border. The three other
candidates have no outline or rejection symbol: one has reference RMSD below
2 Å. The **Confidence ranking** panel explicitly labels ranks 1, 25, 50 and 100;
pRMSD uses dark text and bold values, while retrospective RMSD uses muted text.
Each rank label is centered vertically beside its two score rows.
The stage subtitle gives their common unit (Å). Scores remain in a reserved
6 mm top strip, clear of molecular geometry. Footers identify the N1 example,
its refinement and the separate N100 bank. The manuscript caption explains
their provenance; there is no sentence beneath the graphic.

The exported canvas is **180 × 115 mm**: stage titles print at 8 pt and every
other label at 7 pt without further scaling. The first four stages are 32 mm
wide, with 3 mm gutters; the selected-pose stage is 37 mm wide. Molecular panels
use white backgrounds without repeated outlines, beneath pale stage fills.
The selected/crystal legend occupies one row. Input branches meet without a
junction dot, and short arrows keep the existing directions explicit.
Use the vector PDF at 180 mm two-column width; narrower placement requires
another layout pass. Molecular captures are raster images; text and ligand
diagrams remain vector elements. The architecture figure is unchanged. The current submission package includes
this Rw illustration and 20 Rw result figures.

## Recorded states and provenance

| Display | Source / meaning |
|---|---|
| Ligand and rigid fragments | Original prepared STI graph; five fragment boundaries produce six fragments; same 2D coordinates before/after removing boundary bonds |
| Given pocket center | Complete supplied 1T46 receptor as gray ribbons, exact retained crop in teal, supplied center marked |
| Extracted pocket | 295 retained heavy atoms in 37 residues as sticks under a translucent molecular surface; unchanged production 10 Å crop |
| Pose generation | Saved N1/S10 frames 0, 2, 4 and 10, at t = 0, 0.488, 0.784 and 1 |
| Pose refinement | Recorded CPU refinement of that exact endpoint, steps 0, 50 and 100; step 0 reuses the generated endpoint image |
| Confidence candidates | Separate corrected-Rw N100/S10 Astex repeat-0 refined bank, same complex; eligible predicted-RMSD ranks 1, 25, 50 and 100 |
| Selected pose | Actual selected bank index 41 (zero-based), crystal overlay in unchanged receptor coordinates; symmetry-aware heavy-atom RMSD 0.9578690776411111 Å |

The four candidates were chosen by predicted rank before inspecting geometry.
All 100 candidates are chirality-eligible. Their ranks, scores, eligibility and
selected index agree with the frozen ledger and scores CSV. Candidate identity,
bank, receptor and reference hashes were checked; complete graph isomorphism
maps atoms while preserving fragment assignments. No coordinates were aligned
or displaced. These four rank-selected examples do not estimate bank diversity.

| Eligible rank | Bank index (zero-based) | pRMSD (Å) | RMSD (Å) |
|---|---|---|---|
| 1 | 41 | 1.59 | 0.96 |
| 25 | 23 | 2.34 | 2.06 |
| 50 | 90 | 2.77 | 1.83 |
| 100 | 84 | 4.42 | 4.31 |

Reference RMSDs for all four were independently reproduced using RDKit `CalcRMS`
without alignment, to within 1e-6 Å. They are retrospective labels, never selector
inputs. The primary selector ranks eligible refined poses by predicted RMSD;
raw and refined banks are scored independently.

### Pocket and molecular rendering

The production `crop_to_pocket` operation retains an entire residue if any of its
heavy atoms lies within 10 Å of the saved center. All 295 retained atoms occur at
identical coordinates in the 2,359-atom A-chain pose display, which is contained
in the original 2,369-atom supplied receptor. The wider chain context is for display,
not an enlarged model input. Full receptor means the supplied structure, including
experimental gaps, not a reconstruction of missing sequence.

The extraction panel uses gray full-protein ribbons, dark-teal pocket ribbons
and a teal pocket molecular surface at 72% opacity. It omits a whole-protein
surface that would obscure the crop. The isolated crop uses all retained heavy
atoms as element-colored sticks under the same surface at 45% opacity, with no
ribbon and no added hydrogens. Both input views preserve source coordinates and
orientation; separate fitted viewports show their complete outlines. Surface
completion, capture-border clipping and rendered atom count are checked. All
pose panels use the original common orientation/center and zoom 0.70.

### Refinement scope and limitations

The recorded refinement uses the prepared ligand, saved rigid-fragment assignments,
a fixed receptor, an 18 Å receptor shell and a 100-step budget. Physical and
interaction terms are active; the separate chemical-constraint channel is not
added. Energy-plateau controls are 0.02 absolute / 0.001 relative, patience 5,
starting at step 25; remaining controls are in the numerical record.

The run reached its **100-step budget**, not a convergence certificate. Combined
diagnostic energy decreased from 1681.374 to −34.513 kcal/mol. The finite-shell
envelope flag is **false**: the geometry exceeds the region in which the 18 Å
shell guarantees complete interactions up to the maximum active cutoff. This
is an illustration of that recorded finite-shell protocol, not a claim of fully
covered receptor interactions. No RMSD/PoseBusters improvement is claimed.
The optimizer's diagnostic reference was the raw endpoint; its displacement
metrics are not crystal-reference RMSD and are not labeled as such.

## Reproduction and verification

Render the committed captures with ordinary figure dependencies:

```bash
python -m benchmarks.figures.representative --output outputs/paper_figures
```

The renderer checks source hashes, cameras, raw-endpoint identity, rigid-fragment
geometry, energy-group accounting, receptor coordinate containment and stored
candidate labels. Print-size label bounds, text intersections and overlap with
molecular images are checked before export. PDF/SVG/PNG exports are checked
for deterministic reproduction.
This Rw refresh runs only the fixed N1 illustration (seed 42), then refines its
exact endpoint. The N100 bank, scores and labels are reused from the completed
Rw study. Standalone rendering itself performs no inference or evaluation.

Optional recapture uses the pinned JavaScript and py3Dmol/Playwright stack
documented in the [trajectory record](../../benchmarks/results/paper/trajectory/README.md):

```bash
python -m benchmarks.figures.representative --javascript /path/to/3Dmol-min.js --work-dir /tmp/representative-views --output outputs/paper_figures
```

With the saved Rw N1 results and original source SDF/PDB files, export first:

```bash
python -m benchmarks.analysis.trajectory_example --root /path/to/source-checkout --output benchmarks/results/paper/trajectory/trace.json
```

Then the dependent data exports are reproducible with:

```bash
python -m benchmarks.analysis.representative_refinement --source-root /path/to/source-checkout
python -m benchmarks.analysis.representative_ligand --source-root /path/to/source-checkout
python -m benchmarks.analysis.representative_selection --source-root /path/to/source-checkout
python -m benchmarks.analysis.representative_pocket --source-root /path/to/source-checkout
```

The first command invokes the existing rigid-fragment optimizer, not the docking
model. The others export the original ligand diagram, stored N100 selection and
production crop. Authoritative records under `benchmarks/results/paper/trajectory/`
are `trace.json`, `representative_refinement.json`, `ligand_diagram.json`,
`selection_example.json`, `candidate_annotations.json`, `selected_reference.json`
and `input_pocket.json`. Captures and source-hash metadata are in `views/overview/`.
The source trace/checkpoint hashes and Rw bank hashes identify the current
illustration. Historical versions remain in Git history and the private archive.
No benchmark outcome or selector is changed by this refresh.
