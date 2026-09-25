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
WIDTH, HEIGHT = 12.0, 11.0
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
        self.ax.set(xlim=(0, WIDTH), ylim=(0, HEIGHT), aspect="equal")
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
            self.ax.plot(*zip(a, b), color=color, lw=lw, zorder=2)
        self.ax.add_patch(
            FancyArrowPatch(
                points[-2],
                points[-1],
                arrowstyle="-|>",
                mutation_scale=10,
                linewidth=lw,
                color=color,
                shrinkA=0,
                shrinkB=0,
                zorder=3,
            )
        )

    def line(self, a, b, color=LINE, lw=0.85):
        self.connectors.append((a, b))
        self.ax.plot(*zip(a, b), color=color, lw=lw, zorder=2)

    def junction(self, x, y, color=LINE):
        self.junctions.append((x, y))
        self.ax.add_patch(Circle((x, y), 0.025, facecolor=color, edgecolor="none", zorder=4))

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
    c.text(3.55, 10.40, r"$c=E_t(t)+E_\sigma(\log\sigma)$", color=MUTED)
    c.arrow([(3.55, 10.20), (3.55, 10.03)], color=VIOLET)
    y = 9.62
    for x, w, text, fill in (
        (0.45, 1.60, "Equivariant\nembedding", PALE_GRAY),
        (2.75, 1.60, f"Interaction layer\n×{s['docking_layers']}  (B)", PALE_BLUE),
        (4.70, 1.50, "Linear → act.\nLigand atoms", PALE_PEACH),
    ):
        if text.startswith("Interaction layer"):
            for offset in (0.08, 0.04):
                c.box(x + offset, y - 0.30 + offset, w, 0.60, fill, edge=BLUE)
        c.block(x, y - 0.30, w, 0.60, text, fill)
    c.arrow([(2.08, y), (2.72, y)])
    c.arrow([(4.46, y), (4.67, y)])
    c.line((6.23, y), (6.45, y))
    for branch_y, label in ((y + 0.41, "Linear"), (y - 0.41, "Self tensor product")):
        c.arrow([(6.45, y), (6.45, branch_y), (6.67, branch_y)])
        c.block(6.70, branch_y - 0.19, 1.80, 0.38, label, PALE_PEACH)
        end = y + 0.13 if branch_y > y else y - 0.13
        c.arrow([(8.53, branch_y), (8.74, branch_y), (8.74, end)])
    c.junction(6.45, y)
    c.op(8.74, y, "+")
    c.text(10.45, 10.40, "Newton–Euler readout", bold=True)
    c.line((8.87, y), (9.00, y))
    c.junction(9.00, y)
    for branch_y, label in ((y + 0.41, "Mean"), (y - 0.41, "Torque")):
        c.arrow([(9.00, y), (9.00, branch_y), (9.17, branch_y)])
        c.block(9.20, branch_y - 0.19, 1.20, 0.38, label, PALE_PEACH)
    c.arrow([(10.43, y + 0.41), (11.45, y + 0.41)])
    c.text(11.70, y + 0.41, r"$v_f$", size=13)
    c.arrow([(10.43, y - 0.41), (10.77, y - 0.41)])
    c.block(10.80, y - 0.60, 0.50, 0.38, r"$I_f^+$", PALE_PEACH)
    c.arrow([(11.33, y - 0.41), (11.48, y - 0.41)])
    c.text(11.70, y - 0.41, r"$\omega_f$", size=13)
    c.text(3.75, 9.07, "Scalar · Vector · Rank-2", color=MUTED)

    c.text(0.45, 9.10, "Confidence network", bold=True, ha="left")
    c.text(3.55, 8.65, r"$c=0$", color=MUTED)
    c.arrow([(3.55, 8.46), (3.55, 8.31)], color=VIOLET)
    y = 7.90
    for x, w, text, fill in (
        (0.45, 1.60, "Equivariant\nembedding", PALE_GRAY),
        (2.75, 1.60, f"Interaction layer\n×{s['confidence_layers']}  (B)", PALE_BLUE),
        (4.70, 1.50, "Scalars and\nirrep norms", PALE_GRAY),
        (10.23, 1.55, "Pose MLP\npRMSD · logit", PALE_PEACH),
    ):
        if text.startswith("Interaction layer"):
            for offset in (0.08, 0.04):
                c.box(x + offset, y - 0.30 + offset, w, 0.60, fill, edge=BLUE)
        c.block(x, y - 0.30, w, 0.60, text, fill)
    c.block(0.45, 8.38, 1.60, 0.50, "Docking states\nPer pose, t = 1", PALE_PEACH)
    c.arrow([(2.08, 8.63), (2.27, 8.63)], color=PEACH)
    c.op(2.40, 8.63, "×")
    c.text(2.40, 8.90, r"$\alpha$", color=MUTED)
    c.arrow([(2.40, 8.50), (2.40, 8.03)], color=PEACH)
    c.arrow([(2.08, y), (2.27, y)])
    c.op(2.40, y, "+")
    c.arrow([(2.53, y), (2.72, y)])
    c.arrow([(4.46, y), (4.67, y)])
    c.line((6.23, y), (6.45, y))
    c.arrow([(6.45, y), (6.45, 8.35), (6.67, 8.35)])
    c.arrow([(6.45, y), (6.45, 7.42), (6.67, 7.42)])
    c.junction(6.45, y)
    c.block(6.70, 8.13, 2.45, 0.44, "Global pooling", PALE_PEACH)
    c.block(6.70, 7.20, 1.10, 0.44, "Atom MLP", PALE_PEACH)
    c.arrow([(7.83, 7.42), (8.02, 7.42)])
    c.block(8.05, 7.14, 1.10, 0.56, "Contact\nreadout", PALE_PEACH)
    for branch_y, end in ((8.35, 8.12), (7.42, 7.68)):
        c.arrow([(9.18, branch_y), (9.70, branch_y), (9.70, end)])
    c.block(9.35, 7.71, 0.70, 0.38, "concat", PALE_GRAY)
    c.arrow([(10.08, y), (10.20, y)])
    c.text(7.94, 6.63, "Pose–protein contacts", color=MUTED)
    c.arrow([(7.94, 6.77), (7.94, 6.90)])
    c.line((7.25, 6.90), (8.60, 6.90))
    c.junction(7.94, 6.90)
    c.arrow([(7.25, 6.90), (7.25, 7.17)])
    c.arrow([(8.60, 6.90), (8.60, 7.11)])


def interaction_panel(c, s):
    c.panel(0.20, 6.10, "B", "Interaction layer")
    y = 4.95
    c.text(0.53, y, r"$h^{(k)}$", size=13)
    c.arrow([(0.74, y), (1.17, y)])
    for x, w, label, fill in (
        (1.20, 1.43, "RMSNorm\n(D)", PALE_VIOLET),
        (3.00, 2.30, "Equivariant\nconvolution (C)", PALE_BLUE),
        (5.68, 2.30, "Linear → activation (D)\nChannel dropout", PALE_GRAY),
        (9.02, 1.60, "AdaLN\n(D)", PALE_VIOLET),
    ):
        c.block(x, y - 0.30, w, 0.60, label, fill)
    c.arrow([(2.66, y), (2.97, y)])
    c.arrow([(5.33, y), (5.65, y)])
    c.arrow([(8.01, y), (8.27, y)])
    c.op(8.40, y, "+")
    c.arrow([(8.53, y), (8.99, y)])
    c.arrow([(10.65, y), (10.96, y)])
    c.text(11.32, y, r"$h^{(k+1)}$", size=13)
    c.arrow([(0.96, y), (0.96, 5.55), (8.40, 5.55), (8.40, y + 0.13)], color=BLUE, lw=1.2)
    c.junction(0.96, y, BLUE)
    c.text(5.05, 5.77, "Identity skip", color=MUTED)
    c.text(9.82, 4.32, r"$c$", color=MUTED)
    c.arrow([(9.82, 4.48), (9.82, 4.62)], color=VIOLET)


def convolution_panel(c, s):
    c.panel(0.20, 4.15, "C", "Equivariant convolution")
    x, w, mid = 3.05, 2.50, 4.30
    c.text(mid, 3.71, r"$\hat h$", size=13)
    c.arrow([(mid, 3.54), (mid, 3.41)])
    modules = [
        (3.03, 0.35, "Input radial scale"),
        (2.28, 0.58, "Shared tensor product\n" + r"$\otimes\,Y_{\ell\leq2}(\hat r_{ij})$"),
        (1.76, 0.35, "Output radial scale"),
        (1.24, 0.35, "Activation  (D)"),
        (0.55, 0.52, "Gate-normalized\naggregation"),
    ]
    previous = None
    for y, height, label in modules:
        c.block(x, y, w, height, label, PALE_BLUE)
        if previous is not None:
            c.arrow([(mid, previous - 0.03), (mid, y + height + 0.03)])
        previous = y
    c.arrow([(mid, 0.52), (mid, 0.38)])
    c.text(mid, 0.20, r"$m$", size=13)
    c.text(1.43, 3.43, "Edge descriptors\n" + r"$\hat s_i,\,\hat s_j,\,c$", color=MUTED)
    c.arrow([(1.43, 3.08), (1.43, 2.88)])
    c.block(0.48, 2.37, 1.90, 0.48, "Radial MLP", PALE_MINT)
    c.line((2.41, 2.61), (2.65, 2.61))
    c.arrow([(2.65, 2.61), (2.65, 3.205), (3.02, 3.205)])
    c.arrow([(2.65, 2.61), (2.65, 1.935), (3.02, 1.935)])
    c.junction(2.65, 2.61)
    c.arrow([(1.43, 2.98), (0.25, 2.98), (0.25, 0.81), (0.45, 0.81)])
    c.junction(1.43, 2.98)
    c.block(0.48, 0.55, 1.90, 0.52, "Gate MLP → sigmoid\n× distance decay", PALE_MINT)
    c.arrow([(2.41, 0.81), (3.02, 0.81)])


def operations_panel(c, s):
    c.panel(6.10, 4.15, "D", "Normalization and activation")
    c.text(6.55, 3.53, "AdaLN", bold=True, ha="left")
    c.text(6.61, 3.13, r"$h$", size=13)
    c.arrow([(6.77, 3.13), (7.12, 3.13)])
    c.block(7.15, 2.93, 1.35, 0.40, "RMSNorm", PALE_VIOLET)
    c.text(9.82, 3.13, r"$c$", size=13)
    c.arrow([(9.98, 3.13), (10.27, 3.13)], color=VIOLET)
    c.block(10.30, 2.93, 1.15, 0.40, "Linear", PALE_VIOLET)
    c.line((7.825, 2.90), (7.825, 2.75))
    c.line((7.825, 2.75), (6.70, 2.75))
    c.line((6.70, 2.75), (6.70, 1.80))
    c.line((10.875, 2.90), (10.875, 2.75), color=VIOLET)
    c.line((10.875, 2.75), (11.65, 2.75), color=VIOLET)
    c.line((11.65, 2.75), (11.65, 1.90), color=VIOLET)
    for y, label, feature, parameters in (
        (2.36, "Scalar affine", r"$\hat s$", r"$\gamma_s,\beta_s$"),
        (1.80, "Bounded non-scalar scale", r"$\hat u$", r"$\gamma_u$"),
    ):
        c.block(7.30, y - 0.20, 2.85, 0.40, label, PALE_VIOLET)
        c.arrow([(6.70, y), (7.27, y)])
        c.arrow([(11.65, y + 0.10), (10.18, y + 0.10)], color=VIOLET)
        c.text(6.98, y + 0.16, feature, size=12.5)
        c.text(10.95, y + 0.26, parameters, size=12.5)
        c.arrow([(10.18, y - 0.10), (10.70, y - 0.10)])
        c.text(10.95, y - 0.10, r"$s'$" if y == 2.36 else r"$u'$", size=13)
        if y == 2.36:
            c.junction(6.70, y)
            c.junction(11.65, y + 0.10, VIOLET)
    c.text(6.55, 1.38, "Activation", bold=True, ha="left")
    ya, yb = 1.02, 0.47
    c.text(6.62, ya, r"$s$", size=13)
    c.arrow([(6.78, ya), (7.10, ya)])
    c.block(7.13, ya - 0.175, 2.60, 0.35, "SiLU", PALE_BLUE)
    c.arrow([(9.76, ya), (10.63, ya)])
    c.text(10.85, ya, r"$s'$", size=13)
    c.text(6.62, yb, r"$u$", size=13)
    c.arrow([(6.78, yb), (7.10, yb)])
    c.junction(6.94, yb)
    c.block(7.13, yb - 0.175, 2.60, 0.35, "Norms → MLP → sigmoid", PALE_BLUE)
    c.arrow([(9.76, yb), (10.12, yb)])
    c.text(9.95, yb + 0.19, r"$g$")
    c.op(10.25, yb, "×")
    c.arrow([(6.94, yb), (6.94, 0.10), (10.25, 0.10), (10.25, yb - 0.13)])
    c.arrow([(10.38, yb), (10.63, yb)])
    c.text(10.85, yb, r"$u'$", size=13)


def compose(spec):
    c = Canvas()
    overview_panel(c, spec)
    interaction_panel(c, spec)
    convolution_panel(c, spec)
    operations_panel(c, spec)
    return c


def check_connectors(c, renderer):
    """Reject routing regressions across every registered dataflow segment."""

    def cross(a, b):
        return a[0] * b[1] - a[1] * b[0]

    def hits_box(p, q, bounds):
        x0, y0, x1, y1 = bounds
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
