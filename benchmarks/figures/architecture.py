"""Publication architecture schematic, audited against released source/config files."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
SPEC = ROOT / "benchmarks/results/paper/architecture/spec.json"
NAME = "Fig2_architecture"
WIDTH, HEIGHT = 12.0, 14.05
BOTTOM, TOP = -3.05, 11.0
INK, MUTED, LINE = "#252525", "#666666", "#666666"
BLUE, MINT, PEACH, VIOLET = "#70A9CA", "#71AD9B", "#DCA179", "#A895C4"
PALE_GRAY = "#F2F3F4"
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
        self.ax.set(xlim=(0, WIDTH), ylim=(BOTTOM, TOP), aspect="equal")
        self.ax.set_axis_off()
        self.texts = []
        self.box_texts = []
        self.connectors = []
        self.junctions = []

    def text(self, x, y, text, size=12.5, color=INK, bold=False, ha="center", **kwargs):
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

    def block(self, x, y, w, h, text, fill="white", edge="#858585", size=12.5):
        self.box(x, y, w, h, fill, edge)
        artist = self.text(x + w / 2, y + h / 2, text, size=size)
        self.box_texts.append((artist, (x, y, w, h)))

    def arrow(self, points, color=LINE, lw=0.85):
        if np.linalg.norm(np.subtract(points[-1], points[-2])) < 0.10 - 1e-8:
            raise ValueError(f"Arrowhead without sufficient shaft: {points}")
        self.connectors.extend(zip(points[:-1], points[1:]))
        for a, b in zip(points[:-2], points[1:-1]):
            self.ax.plot(*zip(a, b), color=color, lw=lw, solid_capstyle="butt", zorder=2)
        self.ax.add_patch(
            FancyArrowPatch(
                points[-2],
                points[-1],
                arrowstyle="-|>",
                mutation_scale=9,
                linewidth=lw,
                color=color,
                shrinkA=0,
                shrinkB=0,
                zorder=3,
            )
        )

    def line(self, a, b, color=LINE, lw=0.85):
        self.connectors.append((a, b))
        self.ax.plot(*zip(a, b), color=color, lw=lw, solid_capstyle="butt", zorder=2)

    def junction(self, x, y, color=LINE):
        self.junctions.append((x, y))
        # Register intentional branches for routing checks without drawing a marker.

    def op(self, x, y, symbol, r=0.10):
        """Operator node: + addition, ‖ concatenation, × multiplication."""
        self.ax.add_patch(Circle((x, y), r, facecolor="white", edgecolor=LINE, lw=0.8, zorder=4))
        self.text(x, y, symbol, size=12 if symbol == "‖" else 13)

    def panel(self, x, top, letter, title):
        self.text(x + 0.08, top - 0.20, letter, size=15, bold=True, ha="left")
        self.text(x + 0.45, top - 0.20, title, size=13.5, bold=True, ha="left")


def overview_panel(c, s):
    c.panel(0.20, 10.95, "A", "Model architecture")
    c.text(0.45, 10.30, "Docking network", bold=True, ha="left")
    c.text(3.55, 10.40, r"$\mathbf{c}=E_t(t)+E_\sigma(\log\sigma)$", color=MUTED)
    c.arrow([(3.55, 10.20), (3.55, 10.00)], color=VIOLET)
    y = 9.62
    for x, w, text, fill in (
        (0.45, 1.60, "Equivariant\nembedding", PALE_GRAY),
        (2.75, 1.60, f"Interaction layer\n×{s['docking_layers']}  (B)", PALE_BLUE),
    ):
        if text.startswith("Interaction layer"):
            for offset in (0.08, 0.04):
                c.box(x + offset, y - 0.30 + offset, w, 0.60, fill, edge=BLUE)
        c.block(x, y - 0.30, w, 0.60, text, fill)
    c.text(5.45, 10.25, "Ligand atoms", color=MUTED)
    c.block(4.70, 9.60, 1.50, 0.50, "Equivariant\nlinear (E)", PALE_PEACH)
    c.arrow([(5.45, 9.60), (5.45, 9.36)])
    c.block(4.70, 9.02, 1.50, 0.34, "Activation (D)", PALE_PEACH)
    c.arrow([(2.05, y), (2.75, y)])
    c.arrow([(4.43, y), (4.55, y), (4.55, 9.85), (4.70, 9.85)])
    c.line((6.20, 9.19), (6.32, 9.19))
    c.line((6.32, 9.19), (6.32, y))
    c.line((6.32, y), (6.45, y))
    for branch_y, label in ((y + 0.41, "Equivariant linear"), (y - 0.41, "Self tensor product")):
        c.arrow([(6.45, y), (6.45, branch_y), (6.65, branch_y)])
        c.block(6.65, branch_y - 0.19, 1.80, 0.38, label, PALE_PEACH)
        end = y + 0.10 if branch_y > y else y - 0.10
        c.arrow([(8.45, branch_y), (8.60, branch_y), (8.60, end)])
    c.junction(6.45, y)
    c.op(8.60, y, "+")
    c.text(10.45, 10.40, "Newton–Euler readout", bold=True)
    c.line((8.70, y), (9.10, y))
    c.junction(9.10, y)
    for branch_y, label in ((y + 0.41, "Mean"), (y - 0.41, "Torque")):
        c.arrow([(9.10, y), (9.10, branch_y), (9.35, branch_y)])
        c.block(9.35, branch_y - 0.19, 1.05, 0.38, label, PALE_PEACH)
    c.arrow([(10.40, y + 0.41), (11.45, y + 0.41)])
    c.text(11.70, y + 0.41, r"$v_f$", size=13)
    c.arrow([(10.40, y - 0.41), (10.80, y - 0.41)])
    c.block(10.80, y - 0.60, 0.50, 0.38, r"$I_f^+$", PALE_PEACH)
    c.arrow([(11.30, y - 0.41), (11.48, y - 0.41)])
    c.text(11.70, y - 0.41, r"$\omega_f$", size=13)
    c.text(3.55, 9.07, r"$\ell=0,1,2$", color=MUTED)

    c.text(0.45, 9.10, "Confidence network", bold=True, ha="left")
    c.text(3.55, 8.65, r"$\mathbf{c}=0$", color=MUTED)
    c.arrow([(3.55, 8.46), (3.55, 8.28)], color=VIOLET)
    y = 7.90
    for x, w, text, fill in (
        (0.45, 1.60, "Equivariant\nembedding", PALE_GRAY),
        (2.75, 1.60, f"Interaction layer\n×{s['confidence_layers']}  (B)", PALE_BLUE),
        (4.70, 1.50, r"$\ell=0$ features" + "\n" + r"$\ell>0$ norms", PALE_GRAY),
        (10.23, 1.55, "Pose MLP\npRMSD · logit", PALE_PEACH),
    ):
        if text.startswith("Interaction layer"):
            for offset in (0.08, 0.04):
                c.box(x + offset, y - 0.30 + offset, w, 0.60, fill, edge=BLUE)
        c.block(x, y - 0.30, w, 0.60, text, fill)
    c.block(0.45, 8.38, 1.60, 0.50, "Docking states\nPer pose, t = 1", PALE_PEACH)
    c.arrow([(2.05, 8.63), (2.30, 8.63)], color=PEACH)
    c.op(2.40, 8.63, "×")
    c.text(2.40, 8.90, r"$\alpha$", color=MUTED)
    c.arrow([(2.40, 8.53), (2.40, 8.00)], color=PEACH)
    c.arrow([(2.05, y), (2.30, y)])
    c.op(2.40, y, "+")
    c.arrow([(2.50, y), (2.75, y)])
    c.arrow([(4.43, y), (4.70, y)])
    c.line((6.20, y), (6.45, y))
    c.arrow([(6.45, y), (6.45, 8.35), (6.70, 8.35)])
    c.arrow([(6.45, y), (6.45, 7.42), (6.70, 7.42)])
    c.junction(6.45, y)
    c.block(6.70, 8.13, 2.45, 0.44, "Global pooling", PALE_PEACH)
    c.block(6.70, 7.14, 1.10, 0.56, "Atom MLP", PALE_PEACH)
    c.arrow([(7.80, 7.42), (8.05, 7.42)])
    c.block(8.05, 7.14, 1.10, 0.56, "Contact\nreadout", PALE_PEACH)
    for branch_y, end in ((8.35, 8.09), (7.42, 7.71)):
        c.arrow([(9.15, branch_y), (9.70, branch_y), (9.70, end)])
    c.block(9.35, 7.71, 0.70, 0.38, "concat", PALE_GRAY)
    c.arrow([(10.05, y), (10.23, y)])
    c.text(7.94, 6.63, "Pose–protein contacts", color=MUTED)
    c.arrow([(7.94, 6.77), (7.94, 6.90)])
    c.line((7.25, 6.90), (8.60, 6.90))
    c.junction(7.94, 6.90)
    c.arrow([(7.25, 6.90), (7.25, 7.14)])
    c.arrow([(8.60, 6.90), (8.60, 7.14)])


def interaction_panel(c, s):
    c.panel(0.20, 6.38, "B", "Interaction layer")
    y = 5.23
    c.text(0.25, y, r"$\mathbf{h}^{(k)}$", size=13)
    c.arrow([(0.48, y), (0.75, y)])
    modules = (
        (0.75, 1.10, "RMSNorm\n(D)", PALE_VIOLET),
        (2.50, 1.65, "Equivariant\nconvolution (C)", PALE_BLUE),
        (4.78, 1.22, "Equivariant\nlinear (E)", PALE_GRAY),
        (6.20, 1.10, "Activation\n(D)", PALE_BLUE),
        (7.50, 1.05, "Channel\ndropout", PALE_GRAY),
    )
    previous = None
    for x, w, label, fill in modules:
        c.block(x, y - 0.30, w, 0.60, label, fill)
        if previous is not None:
            c.arrow([(previous, y), (x, y)])
        previous = x + w
    c.arrow([(8.55, y), (8.83, y)])
    c.op(8.93, y, "+")
    c.arrow([(9.03, y), (9.50, y)])
    c.block(9.50, y - 0.30, 1.30, 0.60, "AdaLN\n(D)", PALE_VIOLET)
    c.arrow([(10.80, y), (11.16, y)])
    c.text(11.50, y, r"$\mathbf{h}^{(k+1)}$", size=13)
    c.arrow([(0.60, y), (0.60, 5.83), (8.93, 5.83), (8.93, y + 0.10)], color=BLUE, lw=1.2)
    c.junction(0.60, y, BLUE)
    c.text(4.76, 6.05, "Identity skip", color=MUTED)
    c.text(6.665, 4.70, "Post-convolution transform", color=MUTED)
    c.ax.plot([4.78, 4.78, 8.55, 8.55], [4.90, 4.83, 4.83, 4.90], color=LINE, lw=0.65)
    c.text(2.175, 5.42, r"$\mathbf{h}_{\mathrm{in}}$", color=MUTED)
    c.text(4.50, 5.42, r"$\mathbf{h}_{\mathrm{conv}}$", color=MUTED)
    c.text(10.15, 5.94, r"$\mathbf{c}$", color=MUTED)
    c.arrow([(10.15, 5.76), (10.15, 5.53)], color=VIOLET)


def convolution_panel(c, s):
    c.panel(0.20, 4.45, "C", "Equivariant convolution")
    x, w, mid = 3.05, 2.50, 4.30
    c.text(mid, 4.01, r"$\mathbf{h}_{\mathrm{in}}$", size=13)
    c.arrow([(mid, 3.84), (mid, 3.68)])
    modules = [
        (3.33, 0.35, r"Input scale: $1+\delta_{\mathrm{in}}$"),
        (2.43, 0.58, "Shared tensor product\n" + r"$\otimes\,Y_{\ell\leq2}(\hat r_{ij})$"),
        (1.76, 0.35, r"Output scale: $1+\delta_{\mathrm{out}}$"),
        (1.24, 0.35, "Activation  (D)"),
        (0.55, 0.52, "Gate-normalized\naggregation"),
    ]
    previous = None
    for y, height, label in modules:
        c.block(x, y, w, height, label, PALE_BLUE)
        if previous is not None:
            c.arrow([(mid, previous), (mid, y + height)])
        previous = y
    c.arrow([(mid, 0.55), (mid, 0.38)])
    c.text(mid, 0.20, r"$\mathbf{h}_{\mathrm{conv}}$", size=13)
    c.text(1.43, 3.73, "Edge descriptors\n" + r"$\hat{\mathbf{h}}_{i,0},\,\hat{\mathbf{h}}_{j,0},\,\mathbf{c}$", color=MUTED)
    c.arrow([(1.43, 3.38), (1.43, 3.00)])
    c.block(0.48, 2.52, 1.90, 0.48, "Radial MLP", PALE_MINT)
    c.line((2.38, 2.76), (2.65, 2.76))
    c.arrow([(2.65, 2.76), (2.65, 3.505), (3.05, 3.505)])
    c.arrow([(2.65, 2.76), (2.65, 1.935), (3.05, 1.935)])
    c.junction(2.65, 2.76)
    c.line((1.43, 3.13), (0.25, 3.13))
    c.line((0.25, 3.13), (0.25, 0.375))
    c.junction(1.43, 3.13)
    c.junction(0.25, 1.95)
    c.arrow([(0.25, 1.95), (0.48, 1.95)])
    c.block(0.48, 1.76, 1.90, 0.38, "Gate MLP", PALE_MINT)
    c.arrow([(1.43, 1.76), (1.43, 1.51)])
    c.block(0.48, 1.16, 1.90, 0.35, "Sigmoid", PALE_MINT)
    c.arrow([(1.43, 1.16), (1.43, 0.94)])
    c.arrow([(0.25, 0.375), (0.48, 0.375)])
    c.block(0.48, 0.08, 1.90, 0.52, "Distance decay\n" + r"$e^{-d_{ij}/\sigma_{\mathrm{type}}}$", PALE_MINT)
    c.arrow([(1.43, 0.60), (1.43, 0.74)])
    c.op(1.43, 0.84, "×")
    c.arrow([(1.53, 0.84), (3.05, 0.84)])


def operations_panel(c, s):
    c.panel(6.10, 4.45, "D", "Normalization and activation")
    c.text(6.55, 3.92, "AdaLN", bold=True, ha="left")
    c.text(7.95, 4.00, r"$\mathbf{h}$", size=13)
    c.text(10.60, 4.00, r"$\mathbf{c}$", size=13)
    c.arrow([(7.95, 3.87), (7.95, 3.72)])
    c.arrow([(10.60, 3.87), (10.60, 3.72)], color=VIOLET)
    c.block(7.15, 3.20, 1.60, 0.52, "RMSNorm", PALE_VIOLET)
    c.block(9.65, 3.20, 1.95, 0.52,
            "Scalar linear\n" + r"$\gamma_0,\,\beta_0,\,\gamma_{>0}$", PALE_VIOLET)
    c.arrow([(7.95, 3.20), (7.95, 2.95)])
    c.text(7.63, 3.08, r"$\hat{\mathbf{h}}$", size=13)
    c.arrow([(10.60, 3.20), (10.60, 2.95)], color=VIOLET)
    # A shared boundary groups mutually exclusive degree rules, not serial stages.
    c.box(6.62, 1.85, 5.13, 1.10, PALE_VIOLET)
    c.ax.plot([6.62, 11.75], [2.40, 2.40], color="#858585", lw=0.65)
    c.ax.plot([7.38, 7.38], [1.85, 2.95], color="#858585", lw=0.65)
    for y, degree, formula in (
        (2.40, r"$\ell=0$", r"$\mathbf{h}'_0=(1+\gamma_0)\odot\hat{\mathbf{h}}_0+\beta_0$"),
        (1.85, r"$\ell>0$", r"$\mathbf{h}'_{>0}=[1+0.1\tanh(\gamma_{>0})]\odot\hat{\mathbf{h}}_{>0}$"),
    ):
        label = c.text(7.00, y + 0.275, degree)
        equation = c.text(9.565, y + 0.275, formula, size=13)
        c.box_texts.extend(((label, (6.62, y, 0.76, 0.55)),
                            (equation, (7.38, y, 4.37, 0.55))))
    c.arrow([(9.185, 1.85), (9.185, 1.65)])
    c.text(9.185, 1.48, r"$\mathbf{h}'$", size=13)
    c.text(6.55, 1.38, "Equivariant activation", bold=True, ha="left")
    ya, yb = 1.02, 0.47
    c.text(6.53, ya, r"$\mathbf{h}_0$", size=13)
    c.arrow([(6.78, ya), (7.13, ya)])
    c.block(7.13, ya - 0.175, 2.60, 0.35, r"SiLU  ($\ell=0$)", PALE_BLUE)
    c.arrow([(9.73, ya), (10.63, ya)])
    c.text(10.99, ya, r"$\mathbf{h}'_0$", size=13)
    c.text(6.50, yb, r"$\mathbf{h}_{>0}$", size=13)
    c.arrow([(6.78, yb), (7.13, yb)])
    c.junction(6.94, yb)
    c.block(7.13, yb - 0.175, 2.60, 0.35, "Norms → MLP → sigmoid", PALE_BLUE)
    c.arrow([(9.73, yb), (10.15, yb)])
    c.text(9.95, yb + 0.19, r"$g$")
    c.op(10.25, yb, "×")
    c.arrow([(6.94, yb), (6.94, 0.10), (10.25, 0.10), (10.25, yb - 0.10)])
    c.arrow([(10.35, yb), (10.63, yb)])
    c.text(11.02, yb, r"$\mathbf{h}'_{>0}$", size=13)


def linear_panel(c):
    c.panel(0.20, -0.18, "E", "Equivariant linear")
    c.text(9.55, -0.38, r"Illustrative $\ell=1$ block", color=MUTED)
    # Schematic dimensions: 3 input channels, 2 output channels, 3 components.
    cell = 0.36
    colors = (PALE_BLUE, PALE_MINT, PALE_PEACH)
    for x, rows, weights in ((1.20, 2, True), (4.20, 3, False), (7.20, 2, False)):
        top = -1.69 + rows * cell / 2
        for row in range(rows):
            for col in range(3):
                c.box(x + col * cell, top - (row + 1) * cell,
                      cell, cell, PALE_GRAY if weights else colors[col], radius=0)
    for x, label in ((1.74, "Channel weights"), (4.74, "Input features"), (7.74, "Output features")):
        c.text(x, -0.91, label)
    c.text(3.15, -1.69, r"$\times$", size=18)
    c.text(6.15, -1.69, r"$=$", size=18)
    c.text(3.89, -1.69, "Channels", rotation=90)
    c.text(4.74, -2.40, "Components", color=MUTED)
    c.text(1.74, -2.31, r"$W_{\ell,p}$", size=13)
    c.text(7.74, -2.31, r"$\mathbf{h}'_{\ell,p}$", size=13)
    c.text(10.05, -1.41, "Same weights across\nall components", color=MUTED)
    c.text(10.05, -2.14, r"Independent weights" + "\n" + r"for each $(\ell,p)$", color=MUTED)
    c.text(4.74, -2.79, r"$\mathbf{h}'_{\ell,p}=W_{\ell,p}\,\mathbf{h}_{\ell,p}$", size=13)


def compose(spec):
    c = Canvas()
    overview_panel(c, spec)
    interaction_panel(c, spec)
    convolution_panel(c, spec)
    operations_panel(c, spec)
    linear_panel(c)
    return c


def check_connectors(c, renderer):
    """Reject routing regressions across every registered dataflow segment."""

    def cross(a, b):
        return a[0] * b[1] - a[1] * b[0]

    def hits_box(p, q, bounds):
        x0, y0, x1, y1 = bounds
        # Exact border ports may differ by floating-point roundoff.
        eps = 1e-9
        x0, y0, x1, y1 = x0 + eps, y0 + eps, x1 - eps, y1 - eps
        if abs(p[0] - q[0]) < 1e-8:
            return x0 < p[0] < x1 and max(p[1], q[1]) > y0 and min(p[1], q[1]) < y1
        return y0 < p[1] < y1 and max(p[0], q[0]) > x0 and min(p[0], q[0]) < x1

    for i, (start, end) in enumerate(c.connectors):
        p, q = np.array(start), np.array(end)
        if min(abs(q - p)) > 1e-8:
            raise ValueError("Dataflow connector is not orthogonal")
        for text in c.texts:
            box = text.get_window_extent(renderer).transformed(c.ax.transData.inverted())
            if hits_box(p, q, (box.x0, box.y0, box.x1, box.y1)):
                raise ValueError(f"Connector intersects text: {text.get_text()}")
        for text, (x, y, w, h) in c.box_texts:
            if hits_box(p, q, (x, y, x + w, y + h)):
                raise ValueError(f"Connector enters module: {text.get_text()}")
        for start2, end2 in c.connectors[i + 1 :]:
            a, b = np.array(start2), np.array(end2)
            d, e = q - p, b - a
            denom = cross(d, e)
            if abs(denom) < 1e-8:
                if abs(cross(a - p, d)) < 1e-8:
                    axis = int(np.argmax(abs(d)))
                    overlap = min(max(p[axis], q[axis]), max(a[axis], b[axis])) - max(
                        min(p[axis], q[axis]), min(a[axis], b[axis])
                    )
                    if overlap > 1e-8:
                        raise ValueError("Collinear connectors overlap")
                continue
            t, u = cross(a - p, e) / denom, cross(a - p, d) / denom
            if not (-1e-8 <= t <= 1 + 1e-8 and -1e-8 <= u <= 1 + 1e-8):
                continue
            point = p + t * d
            both_ends = min(abs(t), abs(t - 1)) < 1e-8 and min(abs(u), abs(u - 1)) < 1e-8
            junction = any(np.linalg.norm(point - j) < 1e-8 for j in c.junctions)
            if not both_ends and not junction:
                raise ValueError(f"Unmarked connector intersection at {point}")


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

    check_connectors(c, renderer)

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
    print(f"Rendered {NAME}: source hashes, dimensions, text bounds and connector routing verified")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/paper_figures")
    args = parser.parse_args()
    render(args.output)


if __name__ == "__main__":
    main()
