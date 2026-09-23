"""Render the Figure 1 concept image from the saved 1T46-STI fragment trajectory."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from benchmarks.figures.structure_views import ELEMENT_COLORS
from benchmarks.figures.trajectory import COLORS, DATA, ROOT, verify

NAME = "Fig1_representative"
# Saved frames nearest to t = 0, 0.5 and 0.75, plus the t = 1 endpoint (views/manifest.json).
FLOW_INDICES = (0, 2, 4)
POSE_INDEX = 10
# Arbitrary schematic placement, independent of the trajectory coordinates.
SCHEMATIC_GRID = ((0, 1, 5), (4, 2, 3))
DARK = "#3E434A"
MUTED = "#7B838D"
CONNECTOR = "#B4BBC4"
FRAME_EDGE = "#D6DBE1"
POSE_EDGE = "#8C96A3"

# Figure geometry in inches; two-column width.
WIDTH = 7.2
FRAME_W = 1.25
FRAME_GAP = 0.07
ARROW_GAP = 0.3
SCHEMATIC_W = 1.3
MARGIN = 0.05
BOTTOM = 0.44
TOP = 0.46


def load():
    verify()
    row = json.loads((DATA / "trace.json").read_text())
    xyz = np.asarray(row["coordinates"])
    frag = np.asarray(row["fragment_id"])
    for f in range(len(COLORS)):
        ids = np.flatnonzero(frag == f)
        d = np.linalg.norm(xyz[:, ids, None] - xyz[:, None, ids], axis=-1)
        if len(ids) < 2 or np.abs(d - d[0]).max() > 1e-3:
            raise ValueError(f"Fragment {f} is not rigid across saved frames")
    if set(FLOW_INDICES + (POSE_INDEX,)) - set(row["shown_indices"]):
        raise ValueError("Displayed frame lacks a verified capture")
    if POSE_INDEX != len(row["times"]) - 1:
        raise ValueError("Generated pose must be the t = 1 state")
    return row


def molecule_mask(pixels):
    """Ligand pixels: saturated carbon/heteroatom colors or dark sphere shading."""
    rgb = pixels[..., :3]
    return (rgb.max(-1) - rgb.min(-1) > 0.18) | (rgb.max(-1) < 0.55)


def common_crop(images, pad=0.045):
    masks = [molecule_mask(im) for im in images]
    for m in masks:
        if m[0].any() or m[-1].any() or m[:, 0].any() or m[:, -1].any():
            raise ValueError("Ligand touches the capture border; atoms may be clipped")
    ys, xs = np.nonzero(np.logical_or.reduce(masks))
    h, w = masks[0].shape
    p = int(round(pad * w))
    return (
        max(ys.min() - p, 0),
        min(ys.max() + p + 1, h),
        max(xs.min() - p, 0),
        min(xs.max() + p + 1, w),
    )


def frame(fig, rect, image, crop, edge, lw):
    ax = fig.add_axes(rect)
    y0, y1, x0, x1 = crop
    art = ax.imshow(image[y0:y1, x0:x1], interpolation="none", aspect="auto")
    border = FancyBboxPatch(
        (0, 0),
        1,
        1,
        boxstyle="round,pad=0,rounding_size=0.06",
        transform=ax.transAxes,
        facecolor="none",
        edgecolor=edge,
        linewidth=lw,
        mutation_aspect=ax.bbox.width / ax.bbox.height,
        zorder=5,
    )
    ax.add_patch(border)
    art.set_clip_path(border)
    ax.set_axis_off()
    return art


def schematic(ax, row):
    """Six rigid bodies drawn from saved intra-fragment geometry; placement is arbitrary."""
    xyz = np.asarray(row["coordinates"])[0]
    frag = np.asarray(row["fragment_id"])
    elements = row["elements"]
    step_x, step_y = 5.2, 5.6
    anchors = {
        f: np.array([c * step_x, -r * step_y])
        for r, line in enumerate(SCHEMATIC_GRID)
        for c, f in enumerate(line)
    }
    all_points = []
    for f, carbon in enumerate(COLORS):
        ids = np.flatnonzero(frag == f)
        local = xyz[ids] - xyz[ids].mean(0)
        # Orthographic projection onto the fragment's principal plane.
        flat = local @ np.linalg.svd(local, full_matrices=False)[2][:2].T
        flat *= np.sign(np.sum(flat**3, axis=0) + 1e-12)
        flat += anchors[f]
        all_points.append(flat)
        colors = [carbon if elements[i] == "C" else ELEMENT_COLORS[elements[i]] for i in ids]
        where = {int(i): k for k, i in enumerate(ids)}
        for a, b in row["bonds"]:
            if a in where and b in where:
                p, q = flat[where[a]], flat[where[b]]
                mid = (p + q) / 2
                for s, e, color in ((p, mid, colors[where[a]]), (mid, q, colors[where[b]])):
                    ax.plot(*zip(s, e), color=color, lw=2.1, solid_capstyle="butt", zorder=2)
        ax.scatter(*flat.T, s=15, c=colors, edgecolors="#00000033", linewidths=0.4, zorder=3)
    cue(ax, anchors[0])
    ax.set_aspect("equal")
    points = np.concatenate(all_points)
    ax.set_xlim(min(-2.9, points[:, 0].min() - 0.65), points[:, 0].max() + 0.65)
    ax.set_ylim(min(-step_y - 3.0, points[:, 1].min() - 0.65), 3.0)
    ax.set_axis_off()


def cue(ax, centre):
    """Qualitative rotation and translation glyph on one fragment; not a measured motion."""
    style = dict(
        color=MUTED,
        lw=0.8,
        shrinkA=0,
        shrinkB=0,
        zorder=4,
        arrowstyle="-|>,head_length=2.4,head_width=1.3",
    )
    x, y = centre
    ax.add_patch(
        FancyArrowPatch(
            (x - 1.9, y + 1.8), (x + 1.9, y + 1.8), connectionstyle="arc3,rad=-0.45", **style
        )
    )
    ax.add_patch(FancyArrowPatch((x + 2.0, y - 2.3), (x + 3.1, y - 1.2), **style))


def connector(fig, x0, x1, y, width, height):
    fig.add_artist(
        FancyArrowPatch(
            (x0 / width, y / height),
            (x1 / width, y / height),
            transform=fig.transFigure,
            arrowstyle="-|>,head_length=4,head_width=2.2",
            color=CONNECTOR,
            lw=1.0,
            shrinkA=0,
            shrinkB=0,
        )
    )


def compose(row):
    images = {i: plt.imread(DATA / f"views/frame_{i:02d}.png") for i in row["shown_indices"]}
    crop = common_crop(list(images.values()))
    frame_h = FRAME_W * (crop[1] - crop[0]) / (crop[3] - crop[2])
    height = BOTTOM + frame_h + TOP
    fig = plt.figure(figsize=(WIDTH, height))

    def rect(x, y, w, h):
        return [x / WIDTH, y / height, w / WIDTH, h / height]

    def header(x, title, note):
        fig.text(
            x / WIDTH,
            (top + 0.24) / height,
            title,
            fontsize=9,
            fontweight="semibold",
            color=DARK,
            ha="center",
            va="baseline",
        )
        fig.text(
            x / WIDTH,
            (top + 0.1) / height,
            note,
            fontsize=7,
            color=MUTED,
            ha="center",
            va="baseline",
        )

    mid = BOTTOM + frame_h / 2
    top = BOTTOM + frame_h

    schematic(fig.add_axes(rect(MARGIN, BOTTOM - 0.12, SCHEMATIC_W, frame_h + 0.12)), row)
    sx = MARGIN + SCHEMATIC_W / 2
    header(sx, "Rigid fragments", "fixed internal geometry")
    fig.text(
        sx / WIDTH,
        (BOTTOM - 0.3) / height,
        "schematic",
        fontsize=7,
        style="italic",
        color=MUTED,
        ha="center",
        va="baseline",
    )

    start = x = MARGIN + SCHEMATIC_W + ARROW_GAP
    connector(fig, x - ARROW_GAP + 0.06, x - 0.06, mid, WIDTH, height)
    arts, centres = [], []
    for index in FLOW_INDICES:
        arts.append(
            frame(fig, rect(x, BOTTOM, FRAME_W, frame_h), images[index], crop, FRAME_EDGE, 0.6)
        )
        centres.append((x + FRAME_W / 2, row["times"][index]))
        x += FRAME_W + FRAME_GAP
    x -= FRAME_GAP
    header((start + x) / 2, "SE(3) flow", "Fragment rotation and translation")

    x += ARROW_GAP
    connector(fig, x - ARROW_GAP + 0.06, x - 0.06, mid, WIDTH, height)
    arts.append(
        frame(fig, rect(x, BOTTOM, FRAME_W, frame_h), images[POSE_INDEX], crop, POSE_EDGE, 1.0)
    )
    centres.append((x + FRAME_W / 2, row["times"][POSE_INDEX]))
    header(x + FRAME_W / 2, "Generated pose", "")

    # Flow-time baseline under the four saved states; labels are the recorded times.
    first, last = centres[0][0] - FRAME_W / 2, centres[-1][0] + FRAME_W / 2
    axis_y = BOTTOM - 0.3
    connector(fig, first, last, axis_y, WIDTH, height)
    for cx, t in centres:
        fig.text(
            cx / WIDTH,
            (BOTTOM - 0.16) / height,
            f"$t$ = {t:.3f}",
            fontsize=7.5,
            color=DARK,
            ha="center",
            va="baseline",
        )
    fig.text(
        (first - 0.04) / WIDTH,
        axis_y / height,
        "flow time",
        fontsize=7,
        color=MUTED,
        ha="right",
        va="center",
    )
    return fig, arts


def render(out):
    row = load()
    out.mkdir(parents=True, exist_ok=True)
    style = {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "svg.hashsalt": NAME,
        "image.composite_image": False,
    }
    with plt.rc_context(style):
        fig, arts = compose(row)
        # Vector outputs embed the source capture pixels unresampled.
        fig.savefig(out / f"{NAME}.pdf", metadata={"CreationDate": None, "ModDate": None})
        fig.savefig(out / f"{NAME}.svg", metadata={"Date": None})
        svg = out / f"{NAME}.svg"
        svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
        for art in arts:
            art.set_interpolation("antialiased")
        fig.savefig(out / f"{NAME}.png", dpi=400, metadata={"Software": None})
        plt.close(fig)
    shown = ", ".join(f"{i} (t={row['times'][i]:.3f})" for i in FLOW_INDICES + (POSE_INDEX,))
    print(f"Rendered {NAME} PDF/SVG/PNG in {out}; frames {shown}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/paper_figures")
    args = parser.parse_args()
    render(args.output)


if __name__ == "__main__":
    main()
