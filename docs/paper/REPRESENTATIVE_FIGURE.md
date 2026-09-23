# Figure 1 representative image

A standalone left-to-right overview for known-pocket redocking: ligand
fragmentation + given pocket → candidate generation → post-generation refinement
→ confidence selection → one output pose. Refinement is downstream of the
completed generative flow. No η-based guidance is depicted.
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

To regenerate the two added captures with the pinned JavaScript and optional
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
| Pose generation | Actual saved t = 1 endpoint of the unguided N1/S10 trajectory |
| Post-refinement | Recorded 100-step CPU refinement of that exact endpoint |
| Stacked cards | Conceptual candidate bank; backing cards contain no invented poses |
| Confidence ranks and selected pose | Conceptual selection; the same refined view is reused to illustrate the output |

**Selection is schematic.** The displayed trajectory contains one candidate.
No additional candidate coordinates, confidence scores or measured ranking are
introduced by this figure. The small rank glyphs describe the selection operation,
not observed ranks of this molecule. The output view is not claimed to be a
measured best-of-N result. In the full workflow, confidence ranks eligible refined
candidates by predicted RMSD; the primary benchmark also applies its documented
chirality eligibility filter. Raw and refined banks are scored independently.

All molecular views use the same camera and crop. The pocket thumbnail is
smaller; the generation, refinement and selected-pose views share display scale.
The intact and fragmented ligand diagrams share a 2D depiction, not a docking
conformer. The full time-series remains in the separate trajectory figure.

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

**Fragment-based docking with post-generation refinement and confidence selection.**
Left to right: the input ligand is decomposed into rigid fragments and combined
with a given receptor pocket to generate a bank of candidate poses through
unguided SE(3) flow from t = 0 to t = 1. After generation, candidates undergo
refinement using physical and protein–ligand interaction energies. A confidence
model ranks eligible refined candidates by predicted RMSD and selects one pose;
the primary benchmark's chirality eligibility filter is omitted from the drawing
for clarity. Refinement is a post-generation operation, distinct from the learned
flow; η-based guidance is not shown. Carbon colors identify fragments, while
heteroatoms retain element colors. The molecular views reuse the saved 1T46–STI
N1/S10 endpoint and its actual 100-step refinement, with matched cameras and scale.
Stacked cards and rank glyphs schematically illustrate candidate multiplicity and
selection; the final view reuses the refined pose and does not represent a measured
best-of-N outcome. The 2D ligand diagrams show the actual five fragment boundaries
and six fragment assignments. Bonds are omitted only to depict decomposition;
no chemical bond-breaking or formation reaction is simulated. The numerical
example illustrates the workflow rather than convergence or an accuracy gain;
its finite-shell refinement limitations are documented above.
