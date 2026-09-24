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
WIDTH, HEIGHT = 12.0, 10.5
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
    c.panel(0.20, 7.75, 11.6, 2.55, "A", "Model architecture", PALE_BLUE)
    c.text(0.48, 9.73, "Docking network", bold=True, ha="left")
    c.text(4.74, 9.73, r"$c=E_t(t)+E_\sigma(\log\sigma)$", color=MUTED)
    c.text(0.48, 8.76, "Confidence network", bold=True, ha="left")
    columns = [(0.48, 1.35), (2.07, 1.46), (3.77, 1.94), (5.95, 1.55), (7.74, 1.69), (9.67, 1.83)]
    rows = [
        (
            9.02,
            [
                "Node features",
                "Equivariant\nembedding",
                "Interaction layer\n×6  (B)",
                "Atom vector\nhead",
                "Newton–Euler\nreadout",
                "Fragment velocities\n" + r"$(v_f,\omega_f)$",
            ],
            [PALE_MINT, PALE_BLUE, PALE_BLUE, PALE_PEACH, PALE_PEACH, "white"],
        ),
        (
            8.05,
            [
                "Pose features",
                "Equivariant\nembedding",
                "Interaction layer\n×4  (B)",
                "Invariant\nfeatures",
                "Global + contact\npooling",
                "Pose head\npRMSD · success",
            ],
            [PALE_MINT, PALE_BLUE, PALE_BLUE, PALE_VIOLET, PALE_VIOLET, PALE_MINT],
        ),
    ]
    for y, labels, fills in rows:
        for i, ((x, w), label, fill) in enumerate(zip(columns, labels, fills)):
            c.block(x, y, w, 0.51, label, fill)
            if i:
                xp, wp = columns[i - 1]
                c.arrow([(xp + wp + 0.03, y + 0.255), (x - 0.03, y + 0.255)])
    c.text(4.74, 8.76, "Independent weights; c = 0", color=MUTED)


def interaction_panel(c, s):
    c.panel(0.20, 0.15, 5.65, 7.42, "B", "Interaction layer", PALE_BLUE)
    x, w, mid = 2.94, 2.24, 4.06
    c.text(mid, 7.04, r"$h^{(k)}$", size=12)
    c.block(x, 6.29, w, 0.42, "RMSNorm  (C)")
    c.arrow([(mid, 6.91), (mid, 6.74)])
    c.block(x, 5.57, w, 0.50, "Shared tensor product\nInput / output radial scaling", PALE_BLUE)
    c.block(x, 5.00, w, 0.35, "Equivariant activation")
    c.block(x, 4.38, w, 0.40, "Gated aggregation", PALE_BLUE)
    c.block(x, 3.73, w, 0.43, "Equivariant linear")
    for top, bottom in ((6.26, 6.10), (5.54, 5.38), (4.97, 4.81), (4.35, 4.19), (3.70, 3.54)):
        c.arrow([(mid, top), (mid, bottom)])
    c.block(
        0.48,
        5.66,
        1.80,
        0.50,
        "Spherical harmonics\n" + r"$Y_{\ell\leq2}(\hat r_{ij})$",
        PALE_PEACH,
    )
    c.arrow([(2.31, 5.91), (2.91, 5.91)])
    c.text(1.38, 5.28, "Edge attributes\n+ node scalars", color=MUTED)
    c.arrow([(1.38, 5.03), (1.38, 4.86)])
    c.block(0.48, 4.33, 1.80, 0.50, "Radial / gate MLPs", PALE_MINT)
    c.arrow([(2.31, 4.58), (2.91, 4.58)])
    c.arrow([(2.60, 4.58), (2.60, 5.73), (2.91, 5.73)])
    c.junction(2.60, 4.58)
    c.box(0.48, 2.11, 4.70, 1.40, fill="#FAFAFA", edge="#AAAAAA")
    c.text(2.83, 3.31, "Equivariant activation", bold=True)
    c.text(0.76, 2.96, "Scalars", ha="left", color=MUTED)
    c.block(1.95, 2.765, 1.18, 0.39, "SiLU")
    c.text(4.03, 2.96, r"$s'=\mathrm{SiLU}(s)$")
    c.arrow([(1.60, 2.96), (1.92, 2.96)])
    c.arrow([(3.16, 2.96), (3.38, 2.96)])
    c.text(0.76, 2.43, "Vectors /\ntensors", ha="left", color=MUTED)
    c.block(1.95, 2.23, 1.18, 0.40, "Norms → MLP\n→ sigmoid")
    c.text(4.03, 2.43, r"$u_c'=g_c u_c$")
    c.arrow([(1.60, 2.43), (1.92, 2.43)])
    c.arrow([(3.16, 2.43), (3.38, 2.43)])
    c.block(x, 1.49, w, 0.40, "Channel dropout")
    c.arrow([(mid, 2.08), (mid, 1.92)])
    c.ax.add_patch(Circle((mid, 1.16), 0.10, facecolor="white", edgecolor=LINE, lw=0.8, zorder=4))
    c.text(mid, 1.16, "+", size=13)
    c.arrow([(mid, 1.46), (mid, 1.29)])
    c.arrow([(mid, 6.86), (5.57, 6.86), (5.57, 1.16), (mid + 0.13, 1.16)])
    c.junction(mid, 6.86)
    c.text(5.73, 4.04, "Residual", rotation=90, color=MUTED)
    c.block(x, 0.44, w, 0.40, "AdaLN  (D)", PALE_VIOLET)
    c.arrow([(mid, 1.03), (mid, 0.87)])
    c.text(1.32, 0.64, "Condition c", color=MUTED)
    c.arrow([(2.13, 0.64), (2.91, 0.64)])
    c.text(mid, 0.19, r"$h^{(k+1)}$", size=12)


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
    c.panel(6.12, 0.15, 5.68, 4.32, "D", "Equivariant AdaLN", PALE_VIOLET)
    c.text(6.61, 3.72, r"$h$", size=13)
    c.arrow([(6.76, 3.72), (7.02, 3.72)])
    c.block(7.05, 3.47, 1.58, 0.50, "RMSNorm  (C)", PALE_BLUE)
    c.block(9.39, 3.47, 1.99, 0.50, "Condition c\nLinear projection", PALE_VIOLET)
    # Separate feature and conditioning trunks terminate at the lower row.
    c.line((7.84, 3.44), (7.84, 3.23), connector=True)
    c.line((7.84, 3.23), (6.62, 3.23), connector=True)
    c.line((6.62, 3.23), (6.62, 1.76), connector=True)
    c.line((10.385, 3.44), (10.385, 3.23), color=VIOLET, connector=True)
    c.line((10.385, 3.23), (11.62, 3.23), color=VIOLET, connector=True)
    c.line((11.62, 3.23), (11.62, 1.76), color=VIOLET, connector=True)
    for y, title, formula, feature, parameters in (
        (
            2.71,
            "Scalar affine modulation",
            r"$s'=(1+\gamma_s)\hat s+\beta_s$",
            r"$\hat s$",
            r"$\gamma_s,\beta_s$",
        ),
        (
            1.76,
            "Vector / tensor modulation",
            r"$u'=(1+0.1\tanh\gamma_u)\hat u$",
            r"$\hat u$",
            r"$\gamma_u$",
        ),
    ):
        c.block(7.61, y - 0.29, 2.74, 0.58, title + "\n" + formula, PALE_VIOLET)
        c.arrow([(6.62, y), (7.58, y)])
        c.arrow([(11.62, y), (10.38, y)], color=VIOLET)
        c.text(7.12, y + 0.17, feature, size=12)
        c.text(11.00, y + 0.17, parameters, size=11.5)
        if y == 2.71:
            c.junction(6.62, y)
            c.junction(11.62, y, VIOLET)
    for origin, shift in ((6.99, 0.0), (8.09, 0.10)):
        c.line((origin - 0.10, 0.81), (origin + 0.63, 0.81), lw=0.6)
        for i, value in enumerate((-0.15, 0.20, 0.35)):
            x = origin + i * 0.25
            c.line((x, 0.81), (x, 0.81 + value + shift), BLUE, lw=5)
    c.arrow([(7.72, 0.91), (7.93, 0.91)], lw=0.8)
    c.text(7.89, 0.36, "Scale + shift", color=MUTED)
    c.arrow([(9.67, 0.65), (9.94, 1.16)], VIOLET, lw=1.6, connector=False)
    c.arrow([(10.62, 0.65), (10.92, 1.21)], VIOLET, lw=1.6, connector=False)
    c.arrow([(10.10, 0.91), (10.36, 0.91)], lw=0.8)
    c.text(10.33, 0.36, "Direction preserved", color=MUTED)


def compose(spec):
    c = Canvas()
    overview_panel(c, spec)
    interaction_panel(c, spec)
    norm_panel(c, spec)
    adaln_panel(c, spec)
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
