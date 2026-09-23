# Figure 1 representative image

Concept figure, not the architecture diagram. Rendered by
`benchmarks/figures/representative.py` from the committed 1T46–STI trace and
captures already used by Figure S10; no inference or recapture.

```bash
python -m benchmarks.figures.representative --output outputs/paper_figures
```

Outputs `Fig1_representative.pdf` (manuscript), `.svg` (editable, text kept as
text) and `.png` (400 dpi preview). The renderer first calls
`benchmarks.figures.trajectory.verify()` (trace/capture SHA-256, frame times,
identical cameras) and additionally checks that every fragment's intra-fragment
distances are unchanged across all 11 saved frames (≤1e-3 Å) and that no
ligand pixel touches a capture border.

## Displayed states

| Stage | Saved frame | Recorded t | Capture |
|---|---|---|---|
| SE(3) flow | 0 | 0.000 | `views/frame_00.png` |
| SE(3) flow | 2 | 0.488 | `views/frame_02.png` |
| SE(3) flow | 4 | 0.784 | `views/frame_04.png` |
| Generated pose | 10 | 1.000 | `views/frame_10.png` |

Frame 1 (t=0.271) is omitted; it is visually close to t=0 at this size.

## Design

- Three stages left to right: Rigid fragments → SE(3) flow → Generated pose.
  Stage names are small semibold headers, with minimal explanatory text. There
  are no panel letters, figure title, legend or metrics.
- All four molecular frames share one crop and one displayed size. The crop is
  the union bounding box of ligand pixels across all five verified captures
  plus a 4.5% margin, so camera and scale stay identical. Pixels are never
  altered; vector outputs embed them unresampled. Rounded clipping and a
  darker border on the endpoint set the visual hierarchy.
- The left schematic is drawn in vector form from saved coordinates. Each
  fragment's local geometry is projected onto its principal plane. Orientation
  and grid placement are arbitrary, so it is neither an input pose nor a
  conformer. One fragment carries a qualitative rotation and
  translation glyph.
- Connectors are thin pale-grey arrows between stages, plus one flow-time
  baseline under the four frames labelled with the recorded times.
- Colors are the fragment-carbon palette from `trajectory.COLORS`. N and O use
  `ELEMENT_COLORS`. There is no swatch legend.

## Caption

**EFF-Dock generates a ligand pose by moving rigid ligand fragments with an
SE(3) flow inside a fixed receptor pocket.** Left (schematic): the six rigid
fragments of STI from Astex Diverse Set 1T46–STI, drawn from the
saved intra-fragment atom geometry, which is identical at every saved time.
Each fragment is shown as an orthographic projection onto its principal plane
with arbitrary orientation and placement; this depiction is not an input pose
or conformer. The curved and straight arrows on one fragment are qualitative
rotation and translation glyphs, not measured motion. Middle and right: actual
saved states of one generative ODE trajectory at flow times t = 0.000, 0.488,
0.784 and 1.000 (saved frames 0, 2, 4 and 10 of 11), rendered with an
identical camera, receptor and crop; no coordinates are interpolated. At t = 0
fragments are sampled around the supplied pocket centre. The receptor is fixed
and shown as a pale cartoon. Carbon colours identify the same six fragments
throughout; nitrogen is blue and oxygen red. Bonds joining different fragments
are hidden before t = 1 and drawn in grey at t = 1 as a display convention
using the known ligand connectivity; they do not represent chemical bond
formation. t is the dimensionless generative-flow time, not physical time or
energy refinement. This is a separate illustrative N1 run (one sample, ten ODE
steps, positional prior σ = 2 Å, seed 42, no guidance or confidence
selection), distinct from the N100 benchmark runs. The generated pose is shown
with no claim of accuracy, PoseBusters validity or representativeness.

## Draft files

- [PDF](figures/Fig1_representative.pdf)
- [Editable SVG](figures/Fig1_representative.svg)
- [PNG preview](figures/Fig1_representative.png)

This is a separate Figure 1 design draft; the existing 21-page reference PDF
and its figure numbering are unchanged.
