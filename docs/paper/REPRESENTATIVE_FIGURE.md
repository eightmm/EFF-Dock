# Figure 1 representative image

A standalone concept figure for known-pocket redocking: intact ligand → rigid
fragments, combined with a supplied pocket → unguided SE(3) generation →
post-generation energy refinement.
The layout was reviewed with Claude Opus 5.5 and implemented/verified locally.
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

| State | Source | Time / iteration |
|---|---|---|
| Ligand | Original prepared STI molecular graph; five fragment boundaries marked | 2D input diagram |
| Rigid fragments | Same 2D coordinates, with those five bonds omitted; six saved fragment identities | Schematic decomposition |
| Supplied pocket | Stored 1T46 receptor; ligand models hidden | Input |
| Initial state | Saved ODE frame 0 | t = 0 |
| Intermediate state | Saved ODE frame 2 | t = 0.488 |
| Generated pose | Saved ODE frame 10 | t = 1 |
| Refined pose | CPU refinement of that exact endpoint | Step 100 |

The receptor-only thumbnail uses the same orientation and camera. Its dashed
rectangle marks the image region displayed in the molecular panels, not a
predicted pocket, selected residue set or a radial cutoff. The thumbnail is
shown at a smaller display scale; all four trajectory/refinement panels share
one camera, crop and display scale. Darker receptor coloring is used only for
the supplied-pocket thumbnail.

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

**Pocket-conditioned fragment generation followed by energy refinement.**
A ligand is decomposed into rigid fragments and combined with a supplied receptor
pocket to condition unguided SE(3) pose generation. Top: a 2D depiction of STI
marks the five bonds crossing the saved fragment boundaries; omitting these
bonds reveals the six color-coded fragments. The two depictions share the same
2D coordinates and are neither docking poses nor newly generated conformers.
The supplied-pocket thumbnail shows the stored 1T46 receptor without ligand;
its dashed rectangle marks the close-up region, not a predicted pocket or
distance-cutoff boundary. Bottom: actual saved ODE states at t = 0, 0.488
and 1 (frames 0, 2 and 10 of 11). The generated t = 1 pose then undergoes
100 CPU optimization steps using physical and protein–ligand interaction
energies, with fixed receptor coordinates and rigid fragment internal geometry.
The refined panel shows the resulting coordinates of this same pose. Physical
terms describe covalent geometry and nonbonded/steric contributions; interaction
terms describe typed protein–ligand contacts. These postprocessing energies are
distinct from the learned vector field and are not binding affinities. All
molecular trajectory/refinement panels share a camera, crop and scale. Carbon colors track fragments;
heteroatoms retain element colors. Interfragment bonds are hidden in the early
ODE panels and drawn at the endpoint and after refinement solely for display;
no chemical bond formation is simulated. Flow time is dimensionless and is not
physical time or optimization iteration. This separate N1/S10 illustration uses
prior sigma 2 Å and seed 42. The refinement reaches the step budget and uses a
finite 18 Å receptor shell whose full-neighborhood envelope is not satisfied;
it does not establish convergence, docking accuracy, PB validity or representative
improvement. Confidence/chirality selection is downstream and is not shown.
