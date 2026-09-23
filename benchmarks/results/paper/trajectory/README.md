# Saved ODE trajectory illustration

Figure S10 uses the existing 1T46–STI N1/S10 trace, not a new docking run.
`trace.json` preserves all 11 frames, the six original fragment assignments,
ordered element/bond identities, supplied receptor chain coordinates, pocket
translation, recorded options and source hashes. Frame indices 0, 1, 2, 4, 10
are nearest to targets 0, 0.25, 0.5, 0.75, 1. Actual times are displayed.
The exporter verifies saved graph identity, original ligand input checksum,
rigid intrafragment distances and equality of the t=1 frame, stored output
coordinates and saved docked SDF (within SDF coordinate precision).

This is an illustration, with no endpoint accuracy or PB-validity claim. ODE
time is not physical time or energy refinement. Bonds joining fragments are
hidden before t=1 and displayed at t=1 for clarity; the molecular graph is
known throughout, and no chemical reaction is being simulated. The camera,
receptor and fragment-carbon palette are identical across the five captures.

Public PDF/PNG regeneration needs only the ordinary figure dependencies:

```bash
python -m benchmarks.figures.trajectory --output outputs/paper_figures
```

`views/manifest.json` records exact input/capture hashes, frame times, camera
vectors and software-renderer versions. Captures use the same pinned 3Dmol.js
and optional py3Dmol/Playwright stack as [S8](../evidence/README.md).
To recapture, add `--javascript /path/to/3Dmol-min.js --work-dir /tmp/ode-views`.
No training or inference is performed. With the private source trace available,
re-export coordinates using:

```bash
python -m benchmarks.analysis.trajectory_example --root /path/to/source-checkout --output benchmarks/results/paper/trajectory/trace.json
```

Checkpoint filename and options are recorded by the original run. A historical
checkpoint binary checksum was not embedded in that trace; source-file hashes
here identify the saved trace and geometry, not proof of historical weight bytes.

The time-only artwork is intended as the manuscript Figure 1 representative
image. Its existing S10 filename and page 21 position are working-bundle IDs.
