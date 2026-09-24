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
WIDTH, HEIGHT = 12.0, 10.2
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
        self.text(x + 0.45, top - 0.20, title, size=13, bold=True, ha="left")


def fork(c, y, labels, fill, height, symbol, r):
    """Two-branch readout after the block ending at x=6.20, merged by one operator node."""
    c.line((6.23, y), (6.45, y))
    for branch_y, label in zip((y + 0.40, y - 0.40), labels):
        c.arrow([(6.45, y), (6.45, branch_y), (6.67, branch_y)])
        c.block(6.70, branch_y - height / 2, 1.65, height, label, fill)
        end = y + r + 0.03 if branch_y > y else y - r - 0.03
        c.arrow([(8.38, branch_y), (8.65, branch_y), (8.65, end)])
    c.junction(6.45, y)
    c.op(8.65, y, symbol, r)
    c.arrow([(8.68 + r, y), (8.92, y)])


def overview_panel(c, s):
    c.panel(0.20, 10.15, "A", "Model overview")
    c.text(0.45, 9.53, "Docking network", bold=True, ha="left")
    c.text(3.55, 9.53, r"$c=E_t(t)+E_\sigma(\log\sigma)$", color=MUTED)
    c.arrow([(3.55, 9.36), (3.55, 9.16)], color=VIOLET)
    y = 8.85
    for x, w, text, fill in (
        (0.45, 1.60, "Equivariant\nembedding", PALE_MINT),
        (2.75, 1.60, f"Interaction layer\n×{s['docking_layers']}  (B)", PALE_BLUE),
        (4.70, 1.50, "Linear → act.\n(ligand atoms)", PALE_PEACH),
        (8.95, 1.30, "Newton–Euler\nreadout", PALE_PEACH),
    ):
        c.block(x, y - 0.28, w, 0.56, text, fill)
    c.arrow([(2.08, y), (2.72, y)])
    c.arrow([(4.38, y), (4.67, y)])
    fork(c, y, ("Linear", "Self tensor product"), PALE_PEACH, 0.34, "+", 0.10)
    c.arrow([(10.28, y), (10.55, y)])
    c.text(11.05, y, r"$v_f,\,\omega_f$", size=13)

    c.text(0.45, 7.97, "Confidence network", bold=True, ha="left")
    c.text(3.55, 7.40, r"$c=0$", color=MUTED)
    c.arrow([(3.55, 7.26), (3.55, 7.06)], color=VIOLET)
    y = 6.75
    for x, w, text, fill in (
        (0.45, 1.60, "Equivariant\nembedding", PALE_MINT),
        (2.75, 1.60, f"Interaction layer\n×{s['confidence_layers']}  (B)", PALE_BLUE),
        (4.70, 1.50, "Invariant\nfeatures", PALE_VIOLET),
        (8.95, 1.30, "Pose MLP", PALE_VIOLET),
        (10.60, 1.20, "pRMSD\nSuccess logit", PALE_MINT),
    ):
        c.block(x, y - 0.28, w, 0.56, text, fill)
    # Ligand states come from a fresh docking forward at t = 1 on each scored pose.
    c.block(0.45, 7.22, 1.60, 0.50, "Docking net (t = 1)\nligand states", PALE_PEACH)
    c.arrow([(2.08, 7.47), (2.27, 7.47)], color=PEACH)
    c.op(2.40, 7.47, "×")
    c.text(2.40, 7.71, r"$g$", color=MUTED)
    c.arrow([(2.40, 7.34), (2.40, 6.88)], color=PEACH)
    c.arrow([(2.08, y), (2.27, y)])
    c.op(2.40, y, "+")
    c.arrow([(2.53, y), (2.72, y)])
    c.arrow([(4.38, y), (4.67, y)])
    fork(c, y, ("Global pooling", "Contact pooling"), PALE_VIOLET, 0.40, "‖", 0.12)
    c.arrow([(10.28, y), (10.57, y)])


def interaction_panel(c, s):
    c.panel(0.20, 5.85, "B", "Interaction layer")
    x, w, mid = 2.90, 2.40, 4.10
    c.text(mid, 5.32, r"$h^{(k)}$", size=12)
    c.arrow([(mid, 5.16), (mid, 5.02)])
    c.block(x, 4.65, w, 0.34, "RMSNorm  (C)")
    modules = [
        (3.98, 0.49, "Shared tensor product\nInput / output radial scaling", PALE_BLUE),
        (3.48, 0.32, "Activation  (D)", "white"),
        (2.98, 0.32, "Gated aggregation", PALE_BLUE),
        (2.48, 0.32, "Equivariant linear", "white"),
        (1.98, 0.32, "Activation  (D)", "white"),
        (1.48, 0.32, "Channel dropout", "white"),
    ]
    previous = 4.65
    for y, height, label, fill in modules:
        c.block(x, y, w, height, label, fill)
        c.arrow([(mid, previous - 0.03), (mid, y + height + 0.03)])
        previous = y
    c.block(
        0.45,
        4.10,
        1.90,
        0.50,
        "Spherical harmonics\n" + r"$Y_{\ell\leq2}(\hat r_{ij})$",
        PALE_PEACH,
    )
    c.arrow([(2.38, 4.35), (2.87, 4.35)])
    c.text(1.40, 3.83, "Edge features\nNode scalars, c", color=MUTED)
    c.arrow([(1.40, 3.62), (1.40, 3.42)])
    c.block(0.45, 2.89, 1.90, 0.50, "Radial / gate MLPs", PALE_MINT)
    c.arrow([(2.38, 3.14), (2.87, 3.14)])
    c.arrow([(2.62, 3.14), (2.62, 4.12), (2.87, 4.12)])
    c.junction(2.62, 3.14)
    c.op(mid, 1.22, "+", r=0.09)
    c.arrow([(mid, 1.45), (mid, 1.34)])
    c.arrow([(mid, 5.09), (5.65, 5.09), (5.65, 1.22), (mid + 0.12, 1.22)], color=BLUE, lw=1.2)
    c.junction(mid, 5.09, BLUE)
    c.text(5.83, 3.20, "Identity skip", rotation=90, color=MUTED)
    c.block(x, 0.63, w, 0.32, "AdaLN  (C)", PALE_VIOLET)
    c.arrow([(mid, 1.10), (mid, 0.98)])
    c.arrow([(mid, 0.60), (mid, 0.48)])
    c.text(1.35, 0.79, "Condition c", color=MUTED)
    c.arrow([(2.15, 0.79), (2.87, 0.79)], color=VIOLET)
    c.text(mid, 0.33, r"$h^{(k+1)}$", size=12)


def norm_panel(c, s):
    c.panel(6.12, 5.85, "C", "Equivariant RMSNorm and AdaLN")
    c.text(6.40, 5.22, "RMSNorm", bold=True, ha="left")
    c.text(7.35, 5.22, r"per irrep block $b$", color=MUTED, ha="left")
    c.text(7.75, 4.72, r"$r_b=\sqrt{\frac{1}{C_b}\sum_c\Vert h_{b,c}\Vert^2+\epsilon}$", size=12)
    c.text(10.30, 4.72, r"$\hat h_{b,c}=a_{b,c}\,h_{b,c}/r_b$", size=12)

    c.text(6.40, 4.18, "AdaLN", bold=True, ha="left")
    y1, y2, y3, xf, xp = 3.72, 3.08, 2.50, 7.45, 11.10
    c.text(6.47, y1, r"$h$", size=13)
    c.arrow([(6.58, y1), (6.82, y1)])
    c.block(6.85, y1 - 0.19, 1.20, 0.38, "RMSNorm", PALE_BLUE)
    c.text(10.28, y1, r"$c$", size=13)
    c.arrow([(10.38, y1), (10.62, y1)], color=VIOLET)
    c.block(10.65, y1 - 0.19, 0.90, 0.38, "Linear", PALE_VIOLET)
    c.arrow([(xf, y1 - 0.22), (xf, y3), (7.82, y3)])
    c.arrow([(xf, y2), (7.82, y2)])
    c.junction(xf, y2)
    c.arrow([(xp, y1 - 0.22), (xp, y3), (10.18, y3)], color=VIOLET)
    c.arrow([(xp, y2), (10.18, y2)], color=VIOLET)
    c.junction(xp, y2, VIOLET)
    for y, formula, feature, parameters in (
        (y2, r"$s'=(1+\gamma_s)\hat s+\beta_s$", r"$\hat s$", r"$\gamma_s,\beta_s$"),
        (y3, r"$u'=(1+0.1\tanh\gamma_u)\hat u$", r"$\hat u$", r"$\gamma_u$"),
    ):
        c.block(7.85, y - 0.20, 2.30, 0.40, formula, PALE_VIOLET)
        c.text(7.63, y + 0.17, feature, size=12)
        c.text(10.64, y + 0.17, parameters)


def activation_panel(c, s):
    c.panel(6.12, 2.18, "D", "Equivariant activation")
    ya, yb = 1.52, 0.98
    c.text(6.47, ya, r"$s$", size=13)
    c.arrow([(6.58, ya), (7.10, ya)])
    c.block(7.13, ya - 0.18, 2.60, 0.36, "SiLU")
    c.arrow([(9.76, ya), (10.63, ya)])
    c.text(10.80, ya, r"$s'$", size=13)
    c.text(6.47, yb, r"$u$", size=13)
    c.arrow([(6.58, yb), (7.10, yb)])
    c.junction(6.80, yb)
    c.block(7.13, yb - 0.18, 2.60, 0.36, "Norms → MLP → sigmoid", PALE_BLUE)
    c.arrow([(9.76, yb), (10.12, yb)])
    c.text(9.95, yb + 0.17, r"$g$")
    c.op(10.25, yb, "×")
    c.arrow([(6.80, yb), (6.80, 0.50), (10.25, 0.50), (10.25, yb - 0.13)])
    c.arrow([(10.38, yb), (10.63, yb)])
    c.text(10.80, yb, r"$u'$", size=13)


def compose(spec):
    c = Canvas()
    overview_panel(c, spec)
    interaction_panel(c, spec)
    norm_panel(c, spec)
    activation_panel(c, spec)
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
