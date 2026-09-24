"""Publication architecture schematic, audited against released source/config files."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc, Circle, FancyArrowPatch, FancyBboxPatch, Polygon

ROOT = Path(__file__).resolve().parents[2]
SPEC = ROOT / "benchmarks/results/paper/architecture/spec.json"
NAME = "Fig2_architecture"
WIDTH, HEIGHT = 12.0, 9.35
INK, MUTED, LINE = "#252525", "#666666", "#666666"
BLUE, MINT, PEACH, VIOLET = "#70A9CA", "#71AD9B", "#DCA179", "#A895C4"
PALE_BLUE, PALE_MINT, PALE_PEACH, PALE_VIOLET = "#E8F1F7", "#EAF3EE", "#F8EEE5", "#F0ECF6"


def load():
    spec = json.loads(SPEC.read_text())
    for rel, sha in spec["source_sha256"].items():
        if hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() != sha:
            raise ValueError(f"Architecture source changed; re-audit the diagram: {rel}")
    irreps = spec["irreps"]
    assert sum((2 * int(k[0]) + 1) * n for k, n in irreps.items()) == spec["feature_width"]
    assert sum(irreps.values()) == spec["invariant_width"]
    assert spec["global_pool_width"] + spec["contact_pool_width"] == spec["pose_readout_width"]
    return spec


class Canvas:
    def __init__(self):
        self.fig = plt.figure(figsize=(WIDTH, HEIGHT))
        self.ax = self.fig.add_axes((0, 0, 1, 1))
        self.ax.set(xlim=(0, WIDTH), ylim=(0, HEIGHT), aspect="equal")
        self.ax.set_axis_off()
        self.texts = []
        self.box_texts = []

    def text(self, x, y, text, size=11, color=INK, bold=False, ha="center", **kwargs):
        artist = self.ax.text(
            x,
            y,
            text,
            fontsize=size,
            color=color,
            ha=ha,
            va="center",
            weight="bold" if bold else "normal",
            linespacing=1.3,
            zorder=6,
            **kwargs,
        )
        self.texts.append(artist)
        return artist

    def box(self, x, y, w, h, fill="white", edge="#858585", lw=0.65, radius=0.015):
        self.ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle=f"round,pad=0,rounding_size={radius}",
                facecolor=fill,
                edgecolor=edge,
                linewidth=lw,
                zorder=1,
            )
        )

    def block(self, x, y, w, h, text, fill="white", edge="#858585", size=11):
        self.box(x, y, w, h, fill, edge)
        artist = self.text(x + w / 2, y + h / 2, text, size=size)
        self.box_texts.append((artist, (x, y, w, h)))

    def arrow(self, points, color=LINE, lw=0.85, dashed=False):
        for a, b in zip(points[:-2], points[1:-1]):
            self.ax.plot(*zip(a, b), color=color, lw=lw, ls="--" if dashed else "-", zorder=2)
        self.ax.add_patch(
            FancyArrowPatch(
                points[-2],
                points[-1],
                arrowstyle="-|>",
                mutation_scale=10,
                linewidth=lw,
                color=color,
                linestyle="--" if dashed else "-",
                shrinkA=0,
                shrinkB=0,
                zorder=3,
            )
        )

    def line(self, a, b, color=LINE, lw=1, dashed=False):
        self.ax.plot(*zip(a, b), color=color, lw=lw, ls=(0, (3, 2)) if dashed else "-", zorder=2)

    def node(self, x, y, color, kind="atom", r=0.068):
        if kind == "atom":
            patch = Circle((x, y), r, facecolor=color, edgecolor="white", lw=0.9, zorder=4)
        elif kind == "fragment":
            patch = Polygon(
                [(x, y + r * 1.35), (x + r * 1.35, y), (x, y - r * 1.35), (x - r * 1.35, y)],
                facecolor=color,
                edgecolor="white",
                lw=0.9,
                zorder=4,
            )
        else:
            patch = FancyBboxPatch(
                (x - r, y - r),
                2 * r,
                2 * r,
                boxstyle="round,pad=0,rounding_size=0.025",
                facecolor=color,
                edgecolor="white",
                lw=0.9,
                zorder=4,
            )
        self.ax.add_patch(patch)

    def panel(self, x, y, w, h, letter, title, tint):
        self.text(x + 0.08, y + h - 0.20, letter, size=15, bold=True, ha="left")
        self.text(x + 0.45, y + h - 0.20, title, size=13, bold=True, ha="left")


def graph_panel(c, s):
    c.panel(0.20, 4.85, 5.65, 4.30, "A", "Graph construction", PALE_MINT)
    c.text(1.48, 8.50, "Ligand fragments", bold=True)
    c.text(4.40, 8.50, "Receptor pocket", bold=True)
    lig = [
        (0.65, 7.75),
        (0.92, 8.06),
        (1.31, 7.98),
        (1.48, 7.61),
        (1.93, 7.76),
        (2.24, 8.04),
        (2.54, 7.74),
    ]
    for i, j in ((0, 1), (1, 2), (2, 3), (3, 0), (3, 4), (4, 5), (5, 6)):
        c.line(lig[i], lig[j], PEACH if i >= 4 else BLUE, lw=2)
    for i, xy in enumerate(lig):
        c.node(*xy, BLUE if i < 4 else PEACH, r=0.085)
    frags = [(1.10, 7.33), (2.23, 7.33)]
    for k, xy in enumerate(frags):
        for i in range(4) if k == 0 else range(4, 7):
            c.line(xy, lig[i], MUTED, lw=0.8, dashed=True)
        c.node(*xy, VIOLET, "fragment", r=0.10)
    c.line(*frags, VIOLET, lw=1.4)
    protein = [(3.46, 7.96), (3.85, 8.10), (4.18, 7.80), (4.55, 8.03), (4.87, 7.83), (5.16, 8.08)]
    residues = [(3.85, 7.30), (4.76, 7.30)]
    for i, j in ((0, 1), (1, 2), (3, 4), (4, 5)):
        c.line(protein[i], protein[j], MINT, lw=1.8)
    for k, xy in enumerate(residues):
        for i in range(k * 3, k * 3 + 3):
            c.line(xy, protein[i], MINT, lw=0.8)
        c.node(*xy, "#90A4B4", "residue", r=0.10)
    c.line(*residues, "#90A4B4", lw=1.2)
    for xy in protein:
        c.node(*xy, MINT, r=0.085)
    for i, j in ((5, 0), (6, 1), (6, 2)):
        c.line(lig[i], protein[j], BLUE, lw=0.9, dashed=True)
    c.text(2.98, 7.02, "Dynamic contacts ≤ 5 Å", color=MUTED)
    legends = [
        (0.54, "Ligand atom", BLUE, "atom"),
        (1.93, "Fragment", VIOLET, "fragment"),
        (3.19, "Protein atom", MINT, "atom"),
        (4.58, "Residue", "#90A4B4", "residue"),
    ]
    for x, label, color, kind in legends:
        c.node(x, 6.69, color, kind, r=0.060)
        c.text(x + 0.12, 6.69, label, ha="left")
    c.arrow([(2.98, 6.47), (2.98, 6.39), (1.57, 6.39), (1.57, 6.29)])
    c.block(0.48, 5.65, 2.18, 0.61, "Node features\nEmbedding MLP", PALE_MINT)
    c.arrow([(2.70, 5.95), (3.03, 5.95)])
    c.block(3.08, 5.65, 2.48, 0.61, "Equivariant features", PALE_BLUE)
    c.text(4.32, 6.44, "Coordinates + frames", color=MUTED)
    c.arrow([(4.32, 6.34), (4.32, 6.28)])
    c.text(3.02, 5.38, r"Scalars ($\ell=0$) · Vectors ($\ell=1$) · Tensors ($\ell=2$)")
    c.text(3.02, 5.05, r"Conditioning: time $t$ and prior scale $\sigma$", color=MUTED)


def interaction_panel(c, s):
    c.panel(6.15, 4.85, 5.65, 4.30, "B", "Equivariant message passing", PALE_BLUE)
    c.text(11.45, 8.95, "×6", size=12, color=MUTED)
    mx, mw = 8.63, 2.04
    c.text(mx + mw / 2, 8.55, r"$h^{(\ell)}$", size=12)
    c.arrow([(mx + mw / 2, 8.43), (mx + mw / 2, 8.32)])
    c.block(mx, 7.92, mw, 0.40, "RMSNorm")
    c.block(mx, 7.18, mw, 0.49, "Tensor product\nRadial scaling", PALE_BLUE)
    c.block(mx, 6.47, mw, 0.49, "Gated aggregation", PALE_BLUE)
    c.block(mx, 5.88, mw, 0.39, "Linear · gate · dropout")
    for ya, yb in ((7.92, 7.67), (7.18, 6.96), (6.47, 6.27)):
        c.arrow([(mx + mw / 2, ya - 0.025), (mx + mw / 2, yb + 0.025)])
    c.ax.add_patch(
        Circle((mx + mw / 2, 5.68), 0.10, facecolor="white", edgecolor=LINE, lw=1, zorder=4)
    )
    c.text(mx + mw / 2, 5.68, "+", size=13)
    c.arrow([(mx + mw / 2, 5.86), (mx + mw / 2, 5.79)])
    c.arrow([(mx + mw / 2 + 0.20, 8.55), (11.24, 8.55), (11.24, 5.68), (mx + mw / 2 + 0.12, 5.68)])
    c.text(11.44, 7.02, "Residual", color=MUTED, rotation=90)
    c.block(mx, 5.17, mw, 0.36, "AdaLN", PALE_VIOLET)
    c.arrow([(mx + mw / 2, 5.57), (mx + mw / 2, 5.54)])
    c.text(mx + mw / 2, 4.995, r"$h^{(\ell+1)}$", size=12)
    c.text(7.36, 8.52, "Edge attributes\n+ node scalars", color=MUTED)
    c.block(6.43, 7.56, 1.83, 0.68, "Edge MLP", PALE_MINT)
    c.arrow([(7.345, 8.32), (7.345, 8.25)])
    c.arrow([(8.29, 7.88), (8.44, 7.88), (8.44, 7.43), (8.60, 7.43)])
    c.block(
        6.43,
        6.58,
        1.83,
        0.57,
        "Spherical harmonics\n" + r"$Y_{\ell\leq2}(\hat r_{ij})$",
        PALE_PEACH,
    )
    c.arrow([(8.29, 6.86), (8.49, 6.86), (8.49, 7.27), (8.60, 7.27)])
    c.block(6.43, 5.15, 1.83, 0.58, r"$t$, $\log\sigma$ embeddings", PALE_VIOLET)
    c.arrow([(8.29, 5.35), (8.60, 5.35)])


def readout_panel(c, s):
    c.panel(0.20, 0.20, 5.65, 4.35, "C", "Fragment velocity readout", PALE_PEACH)
    c.block(0.43, 3.31, 1.08, 0.49, "Atom\nfeatures", PALE_BLUE)
    c.arrow([(1.55, 3.555), (1.74, 3.555)])
    c.block(1.78, 3.31, 1.27, 0.49, "Linear\n+ gate", PALE_BLUE)
    c.block(3.39, 3.69, 1.06, 0.35, "Linear", PALE_PEACH)
    c.block(3.39, 3.15, 1.06, 0.35, "Self TP", PALE_PEACH)
    c.arrow([(3.08, 3.555), (3.23, 3.555), (3.23, 3.865), (3.36, 3.865)])
    c.arrow([(3.23, 3.555), (3.23, 3.325), (3.36, 3.325)])
    c.ax.add_patch(Circle((4.80, 3.555), 0.10, facecolor="white", edgecolor=LINE, lw=1, zorder=4))
    c.text(4.80, 3.555, "+", size=13)
    c.arrow([(4.48, 3.865), (4.80, 3.865), (4.80, 3.68)])
    c.arrow([(4.48, 3.325), (4.80, 3.325), (4.80, 3.43)])
    c.arrow([(4.93, 3.555), (5.22, 3.555)])
    c.text(5.40, 3.555, r"$f_a$", size=13, bold=True)
    c.text(5.25, 3.92, "Atom field", color=MUTED)
    c.text(1.25, 3.03, "Rigid fragment", bold=True)
    pts = np.array([[0.73, 2.17], [1.19, 2.00], [1.59, 2.28], [1.41, 2.70], [0.91, 2.65]])
    for i in range(5):
        c.line(pts[i], pts[(i + 1) % 5], BLUE, lw=2.2)
    for i, (x, y) in enumerate(pts):
        c.node(x, y, BLUE, r=0.070)
        vec = np.array([0.18 + 0.04 * i, 0.22 - 0.035 * i])
        c.arrow([(x, y), tuple(np.array([x, y]) + vec)], PEACH, lw=1.2)
    c.arrow([(1.22, 2.36), (1.95, 2.36)], MINT, lw=1.8)
    c.text(2.06, 2.39, r"$v_f$", color="#438574", size=12)
    c.ax.add_patch(Arc((1.19, 2.35), 1.45, 1.25, theta1=195, theta2=295, color=VIOLET, lw=1.5))
    c.arrow([(1.39, 1.75), (1.53, 1.81)], VIOLET, lw=1.5)
    c.text(0.62, 1.78, r"$\omega_f$", color="#8B71B3", size=12)
    c.box(2.53, 1.60, 3.02, 1.45)
    c.box(2.53, 2.68, 3.02, 0.37, PALE_PEACH)
    c.text(4.04, 2.86, "Newton–Euler readout", bold=True)
    c.text(4.04, 2.46, r"$v_f=\mathrm{mean}_{a\in f}\,f_a$", size=12)
    c.text(4.04, 2.12, r"$\tau_f=\sum_{a\in f}(x_a-T_f)\times f_a$", size=11.5)
    c.text(4.04, 1.79, r"$\omega_f=I_f^{+}\tau_f$", size=12)
    c.arrow([(5.40, 3.41), (5.40, 3.08)])
    c.text(3.04, 1.40, r"$(v_f,\omega_f)$ → rigid SE(3) update", size=11)
    c.box(0.43, 0.39, 5.10, 0.74, "#FAFAFA", edge="#BBBBBB")
    c.text(2.98, 0.95, "Training objective", bold=True)
    c.text(
        2.98,
        0.65,
        r"$\mathcal{L}=\mathcal{L}_v+8\mathcal{L}_\omega+0.3\mathcal{L}_{\rm atom}+3\mathcal{L}_{\rm DG}$",
        size=12,
    )


def confidence_panel(c, s):
    c.panel(6.15, 0.20, 5.65, 4.35, "D", "Confidence prediction", PALE_VIOLET)
    c.text(8.98, 4.04, "Separate network · one graph per candidate", color=MUTED)
    c.block(6.42, 3.39, 1.30, 0.45, "Candidate\npose graph", PALE_MINT)
    c.arrow([(7.75, 3.615), (7.92, 3.615)])
    c.block(7.96, 3.39, 1.39, 0.45, "4 interaction\nlayers (B)", PALE_BLUE)
    c.arrow([(9.39, 3.615), (9.59, 3.615)])
    c.block(9.63, 3.39, 1.87, 0.45, "Invariant features\nScalars + irrep norms", PALE_VIOLET)
    c.arrow([(10.565, 3.36), (10.565, 3.20), (7.545, 3.20), (7.545, 3.06)])
    c.arrow([(10.565, 3.20), (10.275, 3.20), (10.275, 3.06)])
    c.box(6.42, 2.24, 2.25, 0.79)
    c.box(6.42, 2.69, 2.25, 0.34, PALE_VIOLET)
    c.text(7.545, 2.87, "Global pooling", bold=True)
    c.text(7.545, 2.47, "Mean / max\nby node type")
    c.box(9.05, 2.24, 2.45, 0.79)
    c.box(9.05, 2.69, 2.45, 0.34, PALE_PEACH)
    c.text(10.275, 2.87, "Contact-aware pooling", bold=True)
    c.text(10.275, 2.47, "Contact features + atom MLP\nAttention / mean / max")
    c.text(10.45, 1.96, "Atom error · success", color=MUTED)
    c.arrow([(11.27, 2.22), (11.27, 2.09)], dashed=True)
    c.block(6.80, 1.29, 4.30, 0.43, "Concatenate → pose MLP", PALE_VIOLET)
    c.arrow([(7.545, 2.21), (7.545, 1.75)])
    c.arrow([(9.26, 2.21), (9.26, 1.75)])
    c.block(6.42, 0.64, 2.25, 0.40, "Predicted RMSD", PALE_MINT)
    c.block(9.25, 0.64, 2.25, 0.40, "Pose success logit", PALE_VIOLET)
    c.arrow([(8.95, 1.26), (8.95, 1.15), (7.545, 1.15), (7.545, 1.07)])
    c.arrow([(8.95, 1.15), (10.375, 1.15), (10.375, 1.07)])
    c.text(7.545, 0.395, "Select minimum pRMSD", color=INK)
    c.text(10.375, 0.395, "Auxiliary prediction", color=MUTED)


def compose(spec):
    c = Canvas()
    graph_panel(c, spec)
    interaction_panel(c, spec)
    readout_panel(c, spec)
    confidence_panel(c, spec)
    return c


def check_layout(c):
    c.fig.canvas.draw()
    renderer = c.fig.canvas.get_renderer()
    for i, t in enumerate(c.texts):
        box = t.get_window_extent(renderer)
        if not c.fig.bbox.contains(box.x0, box.y0) or not c.fig.bbox.contains(box.x1, box.y1):
            raise ValueError(f"Text outside canvas: {t.get_text()}")
        for u in c.texts[i + 1 :]:
            if box.overlaps(u.get_window_extent(renderer)):
                raise ValueError(f"Overlapping text: {t.get_text()} / {u.get_text()}")

    for t, (x, y, w, h) in c.box_texts:
        box = t.get_window_extent(renderer).transformed(c.ax.transData.inverted())
        if (
            box.x0 < x + 0.035
            or box.x1 > x + w - 0.035
            or box.y0 < y + 0.02
            or box.y1 > y + h - 0.02
        ):
            raise ValueError(f"Insufficient module padding: {t.get_text()}")


def render(output):
    spec = load()
    output.mkdir(parents=True, exist_ok=True)
    style = {
        "font.family": "DejaVu Sans",
        "mathtext.fontset": "dejavusans",
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "svg.hashsalt": NAME,
    }
    with plt.rc_context(style):
        c = compose(spec)
        check_layout(c)
        c.fig.savefig(output / f"{NAME}.pdf", metadata={"CreationDate": None, "ModDate": None})
        c.fig.savefig(output / f"{NAME}.svg", metadata={"Date": None})
        svg = output / f"{NAME}.svg"
        svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
        c.fig.savefig(output / f"{NAME}.png", dpi=300, metadata={"Software": None})
        plt.close(c.fig)
    print(f"Rendered {NAME}: source hashes, dimensions and text bounds verified")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/paper_figures")
    args = parser.parse_args()
    render(args.output)


if __name__ == "__main__":
    main()
