# Corrected-Rw ODE trajectory illustration

Figure S10 and Fig1 use a fixed-complex corrected-Rw run of Astex Diverse Set
1T46–STI: N1/S10, seed 42, late schedule with power 3, positional prior sigma
2 Å, supplied pocket cutoff 10 Å, no guidance, scoring or confidence selection.
This single illustration uses frozen released weights; it is not an N100
benchmark-selected pose or a new benchmark evaluation. Its case and seed were
fixed before inspecting the Rw outcome.

`trace.json` preserves all 11 actual frames, six rigid fragments, ordered
37-heavy-atom graph, receptor coordinates, pocket translation, options and
source hashes. Frames 0, 1, 2, 4 and 10 are nearest the five target times
0, 0.25, 0.5, 0.75 and 1; actual times are displayed. No coordinates are
interpolated. The exporter verifies ligand/checkpoint/config hashes, rigid
intrafragment geometry and equality of t=1, the stored pose and docked SDF
within SDF precision. `orientation_injection` is explicitly `rw`; the released
checkpoint's training operator remains `legacy_rt_w`.

`representative_refinement.json` records the exact N1 endpoint's CPU energy
refinement, with the original 100-step protocol. It is an illustration, not
an RMSD/PB-validity or convergence claim. `selection_example.json` and the
reference/annotation records instead reuse the completed Rw Astex repeat-0
N100 refined bank. These two runs are explicitly separate in Fig1. Candidate
ranks, pRMSD, chirality eligibility and selected index follow frozen scores;
crystal RMSD is retrospective and never used for selection.

ODE time is dimensionless flow time, not physical time or refinement progress.
Interfragment bonds are hidden before t=1 and displayed at t=1 for clarity;
the graph is known throughout and no chemical reaction is simulated. The
camera, receptor and fragment-carbon palette are identical across captures.

Public rendering needs only ordinary figure dependencies:

```bash
python -m benchmarks.figures.trajectory --output outputs/paper_figures
python -m benchmarks.figures.representative --output outputs/paper_figures
```

Capture manifests record input/image hashes, frame times, camera vectors and
software versions. Captures use the pinned 3Dmol.js with py3Dmol/Playwright
and CPU SwiftShader. Add `--javascript /path/to/3Dmol-min.js --work-dir /tmp/views`
to recapture. Rendering performs no inference or refinement.

With private saved results available, export the actual trace using:

```bash
python -m benchmarks.analysis.trajectory_example --root /path/to/source-checkout --output benchmarks/results/paper/trajectory/trace.json
```

The default source is `outputs/figures/rw_actual_ode_trace_1t46/results.pt`;
`--trace` can name another saved Rw trace for the same fixed illustration.
The docking checkpoint SHA-256 is
`65be44d7dc8f0867eb9fc5d22214b80f93971ea4702679a527c665046e91e6b6`.
Full source mappings are retained in the JSON records. Historical versions
remain in Git history; current S10 is PDF page 19 of the 20-page result bundle.
Fig1 is supplied separately as the representative workflow illustration.
