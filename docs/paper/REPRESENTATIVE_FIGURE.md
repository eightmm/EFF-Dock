# Figure 1 representative image

- [PDF](figures/Fig1_representative.pdf)
- [Editable SVG](figures/Fig1_representative.svg)
- [PNG preview](figures/Fig1_representative.png)

## Manuscript caption

**Fragment-based pose generation, post-generation refinement and confidence selection.**
This known-pocket redocking overview combines an illustrative N1 generation/refinement
trajectory and a separate N100 candidate bank for the same Astex 1T46–STI complex;
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
is overlaid with the crystal reference (peach; RMSD 0.96 Å). Fragment carbon colors
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
All pose panels retain their common camera, crop, scale and dimensions; the
common crop still includes the step-25 image so removing it does not alter framing.
Input views use different camera scales to show the complete receptor and pocket.
The same receptor appears behind downstream poses with wider chain context.

Only the selected candidate has a check. The three other candidates have neutral
borders and no rejection symbol: two have reference RMSD below 2 Å. The compact
score labels remain inside each panel in a reserved top strip, clear of the
molecular image. All panels gain the same annotation space, preserving molecular
image dimensions and avoiding score/structure overlap. Footers identify the N1 example and N100
selection, and an explicit figure note states that these are separate runs.

At the 262.89 mm source width, stage titles are 10 pt and other annotations are
at least 8.8 pt. At a 180 mm manuscript width these become approximately 6.85 pt
and 6.03 pt. Use the vector PDF at two-column width; narrower placement requires
another layout pass. The molecular captures are raster images; text and ligand
diagrams are vector elements. The existing 21-page reference package is unchanged.

## Recorded states and provenance

| Display | Source / meaning |
|---|---|
| Ligand and rigid fragments | Original prepared STI graph; five fragment boundaries produce six fragments; same 2D coordinates before/after removing boundary bonds |
| Given pocket center | Complete supplied 1T46 receptor as gray ribbons, exact retained crop in teal, supplied center marked |
| Extracted pocket | 295 retained heavy atoms in 37 residues as sticks under a translucent molecular surface; unchanged production 10 Å crop |
| Pose generation | Saved N1/S10 frames 0, 2, 4 and 10, at t = 0, 0.488, 0.784 and 1 |
| Pose refinement | Recorded CPU refinement of that exact endpoint, steps 0, 50 and 100; step 0 reuses the generated endpoint image |
| Confidence candidates | Separate primary N100/S10 Astex repeat-0 refined bank, same complex; eligible predicted-RMSD ranks 1, 25, 50 and 100 |
| Selected pose | Actual selected bank index 41 (zero-based), crystal overlay in unchanged receptor coordinates; symmetry-aware heavy-atom RMSD 0.9585406947604571 Å |

The four candidates were chosen by predicted rank before inspecting geometry.
All 100 candidates are chirality-eligible. Their ranks, scores, eligibility and
selected index agree with the frozen ledger and scores CSV. Candidate identity,
bank, receptor and reference hashes were checked; complete graph isomorphism
maps atoms while preserving fragment assignments. No coordinates were aligned
or displaced. These four rank-selected examples do not estimate bank diversity.

| Eligible rank | Bank index (zero-based) | pRMSD (Å) | RMSD (Å) |
|---|---|---|---|
| 1 | 41 | 1.59 | 0.96 |
| 25 | 98 | 2.33 | 1.84 |
| 50 | 3 | 2.82 | 1.81 |
| 100 | 84 | 4.44 | 4.31 |

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
diagnostic energy decreased from 1473.786 to −34.909 kcal/mol. The finite-shell
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
candidate labels. PDF/SVG/PNG exports are checked for deterministic reproduction.
This layout revision performs no inference, optimization, scoring or evaluation.

Optional recapture uses the pinned JavaScript and py3Dmol/Playwright stack
documented in the [trajectory record](../../benchmarks/results/paper/trajectory/README.md):

```bash
python -m benchmarks.figures.representative --javascript /path/to/3Dmol-min.js --work-dir /tmp/representative-views --output outputs/paper_figures
```

With original source SDF/PDB files, the data exports are reproducible with:

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
Original captures, numerical records, candidate identities and scores are unchanged.
