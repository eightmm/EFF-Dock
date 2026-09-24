"""Publication architecture schematic, audited against released source/config files."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon

ROOT = Path(__file__).resolve().parents[2]
SPEC = ROOT / "benchmarks/results/paper/architecture/spec.json"
NAME = "Fig2_architecture"
WIDTH, HEIGHT = 12.0, 12.2
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
        self.connectors = []
        self.junctions = []

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

    def arrow(self, points, color=LINE, lw=0.85, dashed=False, connector=True):
        if connector:
            if np.linalg.norm(np.subtract(points[-1], points[-2])) < 0.10 - 1e-8:
                raise ValueError(f"Arrowhead without sufficient shaft: {points}")
            self.connectors.extend(zip(points[:-1], points[1:]))
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

    def line(self, a, b, color=LINE, lw=1, dashed=False, connector=False):
        if connector:
            self.connectors.append((a, b))
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

    def junction(self, x, y, color=LINE):
        self.junctions.append((x, y))
        self.ax.add_patch(Circle((x, y), 0.025, facecolor=color, edgecolor="none", zorder=4))

    def panel(self, x, y, w, h, letter, title, tint):
        self.text(x + 0.08, y + h - 0.20, letter, size=15, bold=True, ha="left")
        self.text(x + 0.45, y + h - 0.20, title, size=13, bold=True, ha="left")


def overview_panel(c, s):
    c.panel(0.20, 7.75, 11.6, 4.25, "A", "Model architecture", PALE_BLUE)
    c.text(0.48, 11.43, "Docking network", bold=True, ha="left")
    c.text(3.495, 11.43, r"$c=E_t(t)+E_\sigma(\log\sigma)$", color=MUTED)
    c.arrow([(3.495, 11.27), (3.495, 11.10)], color=VIOLET)
    y = 10.79
    for x, w, text, fill in (
        (0.48, 1.50, "Equivariant\nembedding", PALE_MINT),
        (2.69, 1.61, f"Interaction layer\n×{s['docking_layers']}  (B)", PALE_BLUE),
        (4.72, 1.55, "Linear → act.\n(ligand atoms)", PALE_PEACH),
        (9.06, 1.47, "Newton–Euler\nreadout", PALE_PEACH),
    ):
        c.block(x, y - 0.28, w, 0.56, text, fill)
    c.arrow([(2.01, y), (2.66, y)])
    c.arrow([(4.33, y), (4.69, y)])
    c.line((6.30, y), (6.50, y), connector=True)
    for branch_y, label in ((11.20, "Linear"), (10.38, "Self tensor product")):
        c.arrow([(6.50, y), (6.50, branch_y), (6.76, branch_y)])
        c.block(6.79, branch_y - 0.17, 1.55, 0.34, label, PALE_PEACH, size=10.5)
        end = y + 0.13 if branch_y > y else y - 0.13
        c.arrow([(8.37, branch_y), (8.68, branch_y), (8.68, end)])
    c.junction(6.50, y)
    c.ax.add_patch(Circle((8.68, y), 0.10, facecolor="white", edgecolor=LINE, lw=0.8, zorder=4))
    c.text(8.68, y, "+", size=13)
    c.arrow([(8.81, y), (9.03, y)])
    c.arrow([(10.56, y), (10.83, y)])
    c.text(11.28, y, r"$v_f,\,\omega_f$", size=13)

    c.text(0.48, 10.01, "Confidence network", bold=True, ha="left")
    c.text(3.495, 9.49, r"$c=0$", color=MUTED)
    c.arrow([(3.495, 9.32), (3.495, 9.12)], color=VIOLET)
    y = 8.81
    for x, w, text, fill in (
        (0.48, 1.50, "Equivariant\nembedding", PALE_MINT),
        (2.69, 1.61, f"Interaction layer\n×{s['confidence_layers']}  (B)", PALE_BLUE),
        (4.72, 1.55, "Invariant\nfeatures", PALE_VIOLET),
        (9.35, 1.01, "Pose MLP", PALE_VIOLET),
        (10.65, 1.17, "pRMSD\nSuccess logit", PALE_MINT),
    ):
        c.block(x, y - 0.28, w, 0.56, text, fill)
    c.block(0.48, 9.27, 1.50, 0.48, "Docking states\nat t = 1", PALE_PEACH)
    c.arrow([(2.01, 9.51), (2.33, 9.51), (2.33, 8.94)], color=PEACH)
    c.text(2.33, 9.71, r"$\times g$", size=11, color=MUTED)
    c.arrow([(2.01, y), (2.20, y)])
    c.ax.add_patch(Circle((2.33, y), 0.10, facecolor="white", edgecolor=LINE, lw=0.8, zorder=4))
    c.text(2.33, y, "+", size=13)
    c.arrow([(2.46, y), (2.66, y)])
    c.arrow([(4.33, y), (4.69, y)])
    c.line((6.30, y), (6.50, y), connector=True)
    for branch_y, label in ((9.30, "Global pooling"), (8.32, "Contact pooling")):
        c.arrow([(6.50, y), (6.50, branch_y), (6.76, branch_y)])
        c.block(6.79, branch_y - 0.20, 1.71, 0.40, label, PALE_VIOLET)
        end = y + 0.15 if branch_y > y else y - 0.15
        c.arrow([(8.53, branch_y), (8.93, branch_y), (8.93, end)])
    c.junction(6.50, y)
    c.ax.add_patch(Circle((8.93, y), 0.12, facecolor="white", edgecolor=LINE, lw=0.8, zorder=4))
    c.text(8.93, y, "‖", size=12)
    c.text(9.18, 9.56, "Concat.", size=10.5, color=MUTED)
    c.arrow([(9.08, y), (9.32, y)])
    c.arrow([(10.39, y), (10.62, y)])


def interaction_panel(c, s):
    c.panel(0.20, 2.00, 5.65, 5.57, "B", "Interaction layer", PALE_BLUE)
    x, w, mid = 2.94, 2.24, 4.06
    c.text(mid, 7.04, r"$h^{(k)}$", size=12)
    c.block(x, 6.40, w, 0.34, "RMSNorm  (C)")
    c.arrow([(mid, 6.91), (mid, 6.77)])
    modules = [
        (5.73, 0.49, "Shared tensor product\nInput / output radial scaling", PALE_BLUE),
        (5.23, 0.32, "Activation  (E)", "white"),
        (4.73, 0.32, "Gated aggregation", PALE_BLUE),
        (4.23, 0.32, "Equivariant linear", "white"),
        (3.73, 0.32, "Activation  (E)", "white"),
        (3.25, 0.32, "Channel dropout", "white"),
    ]
    previous = 6.40
    for y, height, label, fill in modules:
        c.block(x, y, w, height, label, fill)
        c.arrow([(mid, previous - 0.03), (mid, y + height + 0.03)])
        previous = y
    c.block(
        0.48,
        5.81,
        1.80,
        0.50,
        "Spherical harmonics\n" + r"$Y_{\ell\leq2}(\hat r_{ij})$",
        PALE_PEACH,
    )
    c.arrow([(2.31, 6.06), (2.91, 6.06)])
    c.text(1.38, 5.52, "Edge features\nNode scalars, c", color=MUTED)
    c.arrow([(1.38, 5.31), (1.38, 5.17)])
    c.block(0.48, 4.64, 1.80, 0.50, "Radial / gate MLPs", PALE_MINT)
    c.arrow([(2.31, 4.89), (2.91, 4.89)])
    c.arrow([(2.60, 4.89), (2.60, 5.87), (2.91, 5.87)])
    c.junction(2.60, 4.89)
    c.ax.add_patch(Circle((mid, 2.99), 0.09, facecolor="white", edgecolor=LINE, lw=0.8, zorder=4))
    c.text(mid, 2.99, "+", size=13)
    c.arrow([(mid, 3.22), (mid, 3.11)])
    c.arrow([(mid, 6.86), (5.57, 6.86), (5.57, 2.99), (mid + 0.12, 2.99)], color=BLUE, lw=1.2)
    c.junction(mid, 6.86, BLUE)
    c.text(5.73, 4.40, "Identity skip", rotation=90, color=MUTED)
    c.block(x, 2.40, w, 0.32, "AdaLN  (D)", PALE_VIOLET)
    c.arrow([(mid, 2.87), (mid, 2.75)])
    c.arrow([(mid, 2.37), (mid, 2.25)])
    c.text(1.32, 2.56, "Condition c", color=MUTED)
    c.arrow([(2.13, 2.56), (2.91, 2.56)], color=VIOLET)
    c.text(mid, 2.10, r"$h^{(k+1)}$", size=12)


def vector_channels(c, x, y, scales):
    directions = np.array([[0.28, 0.37], [0.39, 0.17], [0.16, 0.41]])
    for i, (direction, scale) in enumerate(zip(directions, scales)):
        origin = np.array([x + i * 0.47, y])
        c.node(*origin, "#888888", r=0.023)
        c.arrow(
            [tuple(origin), tuple(origin + scale * direction)],
            [BLUE, MINT, VIOLET][i],
            lw=1.6,
            connector=False,
        )


def norm_panel(c, s):
    c.panel(6.12, 4.80, 5.68, 2.77, "C", "Equivariant RMSNorm", PALE_BLUE)
    c.text(7.10, 6.91, "One irrep block", color=MUTED)
    c.text(10.99, 6.91, "Normalized block", color=MUTED)
    vector_channels(c, 6.54, 6.13, [1.25, 0.95, 1.10])
    vector_channels(c, 10.43, 6.13, [0.72, 0.55, 0.64])
    c.block(8.22, 6.09, 1.71, 0.55, "RMS scale\n+ channel gain", PALE_BLUE)
    c.arrow([(7.89, 6.37), (8.19, 6.37)])
    c.arrow([(9.96, 6.37), (10.27, 6.37)])
    c.text(9.075, 5.65, r"$r_b=\sqrt{\frac{1}{C_b}\sum_c\Vert h_{b,c}\Vert^2+\epsilon}$", size=12)
    c.text(9.075, 5.12, r"$\hat h_{b,c}=a_{b,c}\,h_{b,c}/r_b$", size=12)


def adaln_panel(c, s):
    c.panel(6.12, 2.02, 5.68, 2.86, "D", "Equivariant AdaLN", PALE_VIOLET)
    c.text(6.61, 4.13, r"$h$", size=13)
    c.arrow([(6.76, 4.13), (7.02, 4.13)])
    c.block(7.05, 3.90, 1.58, 0.46, "RMSNorm  (C)", PALE_BLUE)
    c.block(9.39, 3.90, 1.99, 0.46, "Condition c\nLinear projection", PALE_VIOLET)
    c.line((7.84, 3.87), (7.84, 3.68), connector=True)
    c.line((7.84, 3.68), (6.62, 3.68), connector=True)
    c.line((6.62, 3.68), (6.62, 2.34), connector=True)
    c.line((10.385, 3.87), (10.385, 3.68), color=VIOLET, connector=True)
    c.line((10.385, 3.68), (11.62, 3.68), color=VIOLET, connector=True)
    c.line((11.62, 3.68), (11.62, 2.34), color=VIOLET, connector=True)
    for y, title, formula, feature, parameters in (
        (
            3.22,
            "Scalar affine modulation",
            r"$s'=(1+\gamma_s)\hat s+\beta_s$",
            r"$\hat s$",
            r"$\gamma_s,\beta_s$",
        ),
        (
            2.34,
            "Vector / tensor modulation",
            r"$u'=(1+0.1\tanh\gamma_u)\hat u$",
            r"$\hat u$",
            r"$\gamma_u$",
        ),
    ):
        c.block(7.61, y - 0.27, 2.74, 0.54, title + "\n" + formula, PALE_VIOLET)
        c.arrow([(6.62, y), (7.58, y)])
        c.arrow([(11.62, y), (10.38, y)], color=VIOLET)
        c.text(7.12, y + 0.17, feature, size=12)
        c.text(11.00, y + 0.17, parameters, size=11.5)
        if y == 3.22:
            c.junction(6.62, y)
            c.junction(11.62, y, VIOLET)


def activation_panel(c, s):
    c.panel(0.20, 0.12, 11.60, 1.78, "E", "Equivariant activation", PALE_BLUE)
    c.text(2.62, 1.37, "Scalars", bold=True)
    c.text(0.85, 0.92, r"$s$", size=13)
    c.arrow([(1.03, 0.92), (1.40, 0.92)])
    c.block(1.43, 0.72, 2.38, 0.40, "SiLU")
    c.arrow([(3.84, 0.92), (4.20, 0.92)])
    c.text(4.70, 0.92, r"$s'=\mathrm{SiLU}(s)$", size=12)
    c.text(8.50, 1.37, "Vectors / tensors", bold=True)
    c.text(6.65, 0.72, r"$u$", size=13)
    c.arrow([(6.83, 0.72), (7.03, 0.72), (7.03, 0.92), (7.30, 0.92)])
    c.block(7.33, 0.70, 2.37, 0.44, "Norms → MLP → sigmoid", PALE_BLUE)
    c.arrow([(9.73, 0.92), (10.30, 0.92)])
    c.text(10.02, 1.12, r"$g$", size=12)
    c.ax.add_patch(Circle((10.43, 0.92), 0.10, facecolor="white", edgecolor=LINE, lw=0.8, zorder=4))
    c.text(10.43, 0.92, "×", size=13)
    c.arrow([(7.03, 0.72), (7.03, 0.25), (10.43, 0.25), (10.43, 0.79)])
    c.junction(7.03, 0.72)
    c.arrow([(10.56, 0.92), (10.93, 0.92)])
    c.text(11.36, 0.92, r"$u'=g\odot u$", size=12)


def compose(spec):
    c = Canvas()
    overview_panel(c, spec)
    interaction_panel(c, spec)
    norm_panel(c, spec)
    adaln_panel(c, spec)
    activation_panel(c, spec)
    return c


def check_connectors(c, renderer):
    """Reject routing regressions; illustrative vector arrows are excluded."""

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
