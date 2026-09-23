# Figure 1 representative image

A standalone left-to-right overview for known-pocket redocking: ligand
fragmentation + given pocket → candidate generation → post-generation refinement
→ confidence selection → one output pose. Refinement is downstream of the
completed generative flow. No η-based guidance is depicted.
The ligand, fragments and given pocket occupy equally sized panels in a vertical
input column. Generation and refinement each show four states in a 2×2 grid,
read left to right across the top row, then across the bottom row.
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

To regenerate the supplied-pocket and three refinement captures with the pinned JavaScript and optional
py3Dmol/Playwright stack described in the trajectory README:

```bash
python -m benchmarks.figures.representative --javascript /path/to/3Dmol-min.js --work-dir /tmp/representative-views --output outputs/paper_figures
```

With the original source SDF/PDB files available, the one-pose CPU refinement
is reproducible with:

```bash
python -m benchmarks.analysis.representative_refinement --source-root /path/to/source-checkout
python -m benchmarks.analysis.representative_ligand --source-root /path/to/source-checkout
```

The first command invokes the existing rigid-fragment optimizer, not the docking
model. The second exports a 2D depiction of the original ligand graph with the
saved fragment boundaries; it does not generate a new 3D conformer.
The numerical record is
`benchmarks/results/paper/trajectory/representative_refinement.json`; separate
capture metadata record its hash and the unchanged camera. Source-file hashes,
energy parameter identities and implementation checksums are retained.

## Displayed states and interpretation

| Display | Source / meaning |
|---|---|
| Ligand | Original prepared STI graph; five fragment boundaries marked |
| Rigid fragments | Same 2D coordinates with those five bonds omitted; six saved fragment identities |
| Given pocket | Stored 1T46 receptor, ligand hidden, matching close-up crop |
| Pose generation | Saved frames 0, 2, 4 and 10: t = 0, 0.488, 0.784 and 1 |
| Post-refinement | Steps 0, 25, 50 and 100 from the recorded CPU refinement of that exact endpoint |
| Stacked cards | Conceptual candidate bank; backing cards contain no invented poses |
| Confidence ranks and selected pose | Conceptual selection; the same refined view is reused to illustrate the output |

**Selection is schematic.** The displayed trajectory contains one candidate.
No additional candidate coordinates, confidence scores or measured ranking are
introduced by this figure. The small rank glyphs describe the selection operation,
not observed ranks of this molecule. The output view is not claimed to be a
measured best-of-N result. In the full workflow, confidence ranks eligible refined
candidates by predicted RMSD; the primary benchmark also applies its documented
chirality eligibility filter. Raw and refined banks are scored independently.

All molecular views use the same camera, crop and display scale. The three input
panels have identical width and height; the ligand diagrams preserve their 2D
aspect ratio inside these bounds. They are not docking conformers. Refinement
step 0 reuses the generated t = 1 capture, preserving the exact transition between
stages. Two new captures (steps 25 and 50) use already saved coordinates with the
original camera; no optimization was rerun. Generation times and refinement
iterations are labeled separately. The endpoint backing cards indicate candidate
multiplicity; the four panels within each stage are successive states of one
example, not four independently sampled candidates.

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

## Caption

**Fragment-based pose generation, post-generation refinement and confidence selection.**
The input ligand and its rigid fragments are arranged vertically above a given
receptor pocket, with equal panel dimensions. The ligand diagrams show the five
fragment boundaries and six fragment assignments of STI; the decomposition is
a representation, not a chemical bond-breaking reaction. Given these inputs,
unguided SE(3) flow generates candidate poses; four saved states of the 1T46–STI
example are shown at t = 0, 0.488, 0.784 and 1. After generation, physical and
protein–ligand interaction energies refine each candidate; the example shows
recorded iterations 0, 25, 50 and 100. Each 2×2 sequence is read across the top
row and then across the bottom row. Refinement step 0 is the same pose as the
final generation state. A confidence model ranks eligible refined candidates by
predicted RMSD and selects one pose; the primary benchmark's chirality eligibility
filter is omitted from the drawing for clarity. η-based guidance is not depicted.
All molecular views use the same camera, crop and scale. Carbon colors identify
fragments and heteroatoms retain element colors. The endpoint backing cards and
rank glyphs schematically represent candidate multiplicity and selection; the
four states of each sequence belong to one N1/S10 trajectory. The output reuses
its final refined pose, rather than representing a measured best-of-N result.
The molecular coordinates are recorded states; no interpolation, new generation,
refinement or confidence evaluation was performed for this layout. The example
illustrates the workflow rather than convergence or an accuracy gain; the
finite-shell refinement limitations are documented above.
