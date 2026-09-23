# Figure 1 representative image

A standalone left-to-right overview for known-pocket redocking: ligand
fragmentation + given pocket → candidate generation → post-generation refinement
→ confidence selection → one output pose. Refinement is downstream of the
completed generative flow. No η-based guidance is depicted.
The ligand, fragments, full supplied receptor and given pocket occupy equally sized panels in a vertical
input column enclosed in an Input preparation box with the same height as
the generation, refinement and confidence stages. Names sit inside
the upper-left corners. Separate arrows show ligand → fragments and full
receptor → given-pocket crop; the pocket center is supplied, not predicted. Generation and refinement are enclosed in separate large boxes;
each contains four states ordered from top to bottom with downward arrows.
The stages progress from left to right. Method subtitles sit inside each box
under its heading. Confidence displays four distinct recorded poses in a matching large box, with
selected/not-selected marks overlaid in the upper-left image corners.
Time and step labels use that same corner. Stage boxes have equal widths and
uniform gaps, with aligned connectors and white backgrounds with neutral gray borders.
Typography uses Matplotlib-bundled DejaVu Sans fonts for both text and mathematical
labels. The final panel omits the PDB identifier; source identity remains here.
Empty stacked backplates and × N denote candidate multiplicity in the generation
and refinement stages; confidence ellipses omit intervening candidates, and
N → 1 denotes selection. Here N = 100 in the primary candidate-bank protocol.
Each confidence image has a small two-line pRMSD/RMSD annotation inside its
upper-left corner, beside the selection glyph,
with both values in ångströms. All pose panels
have equal dimensions.
This draft does not change the existing 21-page reference package.

- [PDF](figures/Fig1_representative.pdf)
- [Editable SVG](figures/Fig1_representative.svg)
- [PNG preview](figures/Fig1_representative.png)

## Reproduction

```bash
python -m benchmarks.figures.representative --output outputs/paper_figures
```

Rendering uses committed molecular captures and ordinary figure dependencies.
It checks source hashes, identical molecular cameras, raw-endpoint identity,
rigid-fragment geometry and energy-group accounting. PDF/SVG text and the ligand
fragment schematic remain vector elements; the molecular views are embedded
3Dmol.js captures. PNG previews use 400 dpi.

To regenerate the molecular captures (pocket, refinement and confidence candidates)
with the pinned JavaScript and optional
py3Dmol/Playwright stack described in the trajectory README:

```bash
python -m benchmarks.figures.representative --javascript /path/to/3Dmol-min.js --work-dir /tmp/representative-views --output outputs/paper_figures
```

With the original source SDF/PDB files available, the one-pose CPU refinement
is reproducible with:

```bash
python -m benchmarks.analysis.representative_refinement --source-root /path/to/source-checkout
python -m benchmarks.analysis.representative_ligand --source-root /path/to/source-checkout
python -m benchmarks.analysis.representative_selection --source-root /path/to/source-checkout
python -m benchmarks.analysis.representative_pocket --source-root /path/to/source-checkout
```

The first command invokes the existing rigid-fragment optimizer, not the docking
model. The second exports a 2D depiction of the original ligand graph with the
saved fragment boundaries; it does not generate a new 3D conformer. The third
exports existing N100 candidate coordinates and confidence ranks, with source
checksums, graph isomorphism and fragment-assignment checks.
The fourth command exports the original full receptor and verifies the actual
residue-aware crop used by the saved N1 illustration.
The numerical record is
`benchmarks/results/paper/trajectory/representative_refinement.json`; separate
capture metadata record its hash and the unchanged camera. Source-file hashes,
energy parameter identities and implementation checksums are retained.

## Displayed states and interpretation

| Display | Source / meaning |
|---|---|
| Ligand | Original prepared STI graph; five fragment boundaries marked |
| Rigid fragments | Same 2D coordinates with those five bonds omitted; six saved fragment identities |
| Protein | Full supplied 1T46 receptor structure, retained pocket residues highlighted in darker gray |
| Given pocket | Only retained residues from the saved center and 10 Å cutoff; unchanged coordinates; wider camera than the pose panels to show the cropped region |
| Pose generation | Saved frames 0, 2, 4 and 10: t = 0, 0.488, 0.784 and 1 |
| Post-refinement | Steps 0, 25, 50 and 100 from the recorded CPU refinement of that exact endpoint |
| Confidence candidates | Astex repeat 0, same 1T46 complex, existing N100 refined bank; eligible ranks 1, 25, 50 and 100 |
| Selected pose | Actual confidence-selected candidate, zero-based index 41, overlaid with its crystal reference; symmetry-aware heavy-atom RMSD 0.96 Å |

**Selection is measured; the whole pipeline is an overview assembled from two
runs.** The generation/refinement trajectory is the saved N1 example. The
confidence panels come from the separate primary N100/S10 Astex repeat-0 bank
for the same complex. All 100 candidates are chirality-eligible. Candidates were
chosen before inspecting their structures, using best, lower-quartile, lower-median and worst
predicted-RMSD ranks (1, 25, 50, 100). The best is
the recorded selected index. Scores and selection were checked against the frozen
ledger and scores CSV, with bank and receptor hashes verified. Atom ordering
was mapped by complete molecular-graph isomorphism; all symmetric mappings
preserve the fragment assignments. No coordinates were aligned or displaced.

**✓ means selected; × means not selected.** Neither mark indicates PB validity,
RMSD correctness or a probability. The three unselected structures are distinct
candidates, not intermediate states of the selected trajectory. The figure does
not imply that they were generated by the displayed N1 run. These four
rank-based illustrations are not an estimate of candidate-bank diversity.
`selection_example.json` records all 100 scores, eligibility, the selected index,
the four displayed coordinates and their source provenance. The primary selector
ranks eligible
refined poses by predicted RMSD; raw and refined banks are scored independently.

All pose views use the same camera, crop and display scale.
The full-receptor overview deliberately uses a wider camera scale while keeping
the same orientation, so the complete supplied structure fits inside its panel. Pose captures in the overview
gallery (`views/overview`) retain the original camera center and orientation
with a common 0.70 zoom factor so that all four confidence candidates are fully
visible. Original captures for the other manuscript figures remain unchanged.
The supplied-pocket cartoon now hides non-pocket residues using the verified
model crop, and uses neutral gray at 0.65 opacity. Its retained atom coordinates
remain unchanged; its input camera is widened to display more of the crop. The full receptor is light gray with the
same retained residues highlighted in darker gray.
The four input panels have identical width and height; the ligand diagrams preserve their 2D
aspect ratio inside these bounds. They are not docking conformers. Refinement
step 0 reuses the generated t = 1 capture, preserving the exact transition between
stages. The overview gallery uses stored coordinates; no optimization was rerun.
Generation times and refinement iterations are labeled separately inside the
upper-left corner of each image. The four
panels within each trajectory stage are successive states of one example, not four
independently sampled candidates. The empty backplates and × N are schematic
cues for applying these stages across a candidate bank, not extra captured
trajectories. Only the front sequence is a recorded trajectory.

Refinement used the original prepared ligand for parameterization, stored
fragment assignments, a fixed receptor, an 18 Å receptor shell, and 100 maximum
rigid-fragment optimization steps. Physics and interaction terms are both
active; the separate chemical-constraint channel is not added. Production-like
energy-plateau thresholds are 0.02 absolute / 0.001 relative, patience 5, starting
at step 25. Other numerical controls are recorded in the JSON configuration.

The run reached its **100-step budget**, not a convergence certificate. Its
combined diagnostic energy decreased from 1473.786 to −34.909 kcal/mol. The
finite-shell envelope flag is **false**: the geometry exceeds the region in
which the 18 Å shell guarantees complete interactions up to the maximum active
cutoff. This is an illustration of the recorded finite-shell protocol, not a
claim of a fully covered receptor environment. No RMSD/PoseBusters improvement
or validity claim is made. The optimizer's diagnostic reference argument was
the raw endpoint; its displacement-to-reference metrics are deliberately not
published as crystal RMSD. No new docking inference or confidence selection
was performed.

## Supplied-pocket crop

`input_pocket.json` stores the original receptor PDB, the retained atom records,
residue identities, supplied center, 10 Å cutoff and source hashes. The production
`crop_to_pocket` function keeps an entire residue when at least one of its heavy
atoms lies within 10 Å of the recorded center. Export checks that the atom records
shown in the crop exactly match the production crop coordinates. Full protein
means the complete supplied receptor structure, including its experimental gaps,
not a reconstruction of the full-length biological sequence. This input export performs
no ligand or pocket prediction. The overview and cropped panel share orientation but
use different camera scales. The input pocket uses a wider view than the trajectory panels; hiding
non-pocket residues is an input illustration and does not change other panels.

## Confidence candidate annotations

`pRMSD` is the frozen confidence model prediction used to rank eligible candidates.
`RMSD` is the retrospective symmetry-aware heavy-atom error against the crystal
ligand, in the receptor frame without alignment; it is not used for selection.
The displayed candidates and their order remain unchanged. Stored reference
RMSDs for all four were independently reproduced with `CalcRMS` to within
1e-6 Å. The paired values and source-record hashes are stored separately in
`candidate_annotations.json`; original geometry and capture records are unchanged.

| Eligible rank | Bank index (zero-based) | pRMSD (Å) | RMSD (Å) |
|---|---|---|---|
| 1 | 41 | 1.59 | 0.96 |
| 25 | 98 | 2.33 | 1.84 |
| 50 | 3 | 2.82 | 1.81 |
| 100 | 84 | 4.44 | 4.31 |

## Selected-pose reference overlay

The output panel compares the frozen selected candidate (index 41) with the exact
crystal ligand used by its Astex repeat-0 refined evaluation. The reference SDF
hash agrees between the trajectory provenance and that run's refinement inputs.
Its saved symmetry-aware heavy-atom RMSD is 0.9585406947604571 Å (displayed as
0.96 Å), independently reproduced with RDKit `CalcRMS` without alignment. This is
an observed reference RMSD, not the confidence model's predicted RMSD. Reference
coordinates, source hashes and the verified value are stored in
`selected_reference.json`; its capture is `views/overview/selected_overlay.png`.
Existing selected coordinates, confidence scores and candidate choice are unchanged.
The crystal ligand is used only for retrospective visualization and evaluation;
it is not provided to the confidence selector. This illustration is a single
case and does not establish a success rate.

## Caption

**Fragment-based pose generation, post-generation refinement and confidence selection.**
The ligand is decomposed into six rigid fragments and combined with a given
receptor pocket. Fragment-level SE(3) flow matching generates poses; the four
panels show recorded states at t = 0, 0.488, 0.784 and 1. Physics- and
interaction-based post-generation refinement is illustrated at iterations 0,
25, 50 and 100. Each boxed sequence proceeds from top to bottom, while the complete
workflow progresses from left to right. Input preparation groups the ligand,
fragmentation and full supplied receptor → pocket crop, with inset labels.
The crop retains complete residues within 10 Å of the supplied center and is
not a pocket prediction. The full receptor highlights those residues in gray. Time and iteration labels are overlaid inside
the corresponding images. Refinement step 0 equals the generated t = 1 pose.
Method subtitles appear beneath the stage headings; η-based guidance is omitted.
Empty backplates and × N schematically indicate parallel candidate processing
(N = 100 in the primary protocol); the four temporal frames on the front card
are successive states of one illustrative trajectory. Ellipses between the
confidence examples denote omitted candidates, and N → 1 indicates selection.
The confidence stage displays four distinct refined candidates from the existing
N100 bank for the same 1T46–STI complex, at eligible predicted-RMSD ranks 1, 25, 50
and 100. The best-ranked candidate carries an overlaid ✓ and is shown as the
selected output, overlaid with the crystal ligand in the unchanged receptor
coordinate frame (symmetry-aware heavy-atom RMSD 0.96 Å);
× marks the three unselected candidates, not physical invalidity. Each candidate
is annotated inside the image with pRMSD (the ranking prediction) and RMSD (the retrospective
crystal-reference symmetry-aware heavy-atom error), both in Å. Reference RMSD
is displayed for comparison and is not used to select the pose. All
pose panels share camera, crop and scale; the full-receptor and pocket-input
views use wider scales to show their respective structures. Carbon colors identify fragments in the trajectory and candidate panels;
the final overlay instead uses blue for selected-pose carbons and peach for
crystal carbons. Heteroatoms retain element colors. The generation/refinement trajectory is the
saved N1/S10 illustration; the confidence examples are from the separate primary
N100/S10 Astex repeat-0 run. Thus the overview combines complementary records
for one complex, rather than depicting a single end-to-end sampled bank.
Candidate coordinates and scores are stored results, with graph-verified atom
mapping and no geometric alignment, new scoring or inference. The refinement
example's finite-shell
limitations are documented above; the illustration is not a convergence claim.
