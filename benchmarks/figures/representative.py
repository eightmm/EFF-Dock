"""Render the Figure 1 concept image from the saved 1T46-STI fragment trajectory."""

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from benchmarks.figures.structure_views import ELEMENT_COLORS, JS_SHA256, JS_URL
from benchmarks.figures.trajectory import COLORS, DATA, ROOT, scene, verify

NAME = "Fig1_representative"
POSE_INDEX = 10
DARK = "#3E434A"
MUTED = "#7B838D"
CONNECTOR = "#B4BBC4"
FRAME_EDGE = "#D6DBE1"
POSE_EDGE = "#8C96A3"

# Landscape overview; preserve vector labels when scaling for the manuscript.
WIDTH = 9.4


def verify_extra_views():
    captures = json.loads((DATA / "views/manifest.json").read_text())
    for stem in ("supplied_pocket", "refined_endpoint"):
        metadata = json.loads((DATA / f"views/{stem}.json").read_text())
        sources = [
            ("trace.json", metadata["trace_sha256"]),
            (f"views/{stem}.png", metadata["sha256"]),
        ]
        if stem == "refined_endpoint":
            sources.append(("representative_refinement.json", metadata["refinement_sha256"]))
        for name, expected in sources:
            if hashlib.sha256((DATA / name).read_bytes()).hexdigest() != expected:
                raise ValueError(f"Stale molecular capture: {stem}")
        if not np.allclose(metadata["camera"], captures["records"][0]["camera"], atol=1e-7):
            raise ValueError(f"Camera differs from the trajectory: {stem}")


async def capture_views(javascript, work):
    from playwright.async_api import async_playwright

    verify()
    row = json.loads((DATA / "trace.json").read_text())
    refined = json.loads((DATA / "representative_refinement.json").read_text())
    camera = json.loads((DATA / "views/manifest.json").read_text())["records"][0]["camera"]
    final = dict(row, coordinates=row["coordinates"][:-1] + [refined["coordinates"][-1]])
    work.mkdir(parents=True, exist_ok=True)
    async with async_playwright() as api:
        browser = await api.chromium.launch(
            args=[
                "--use-gl=angle",
                "--use-angle=swiftshader",
                "--enable-unsafe-swiftshader",
                "--disable-dev-shm-usage",
            ]
        )
        for stem, source, index, pocket in (
            ("supplied_pocket", row, 0, True),
            ("refined_endpoint", final, 10, False),
        ):
            html = work / f"{stem}.html"
            html.write_text(scene(source, index, javascript, pocket_only=pocket, camera=camera))
            output = DATA / f"views/{stem}.png"
            page = await browser.new_page(
                viewport={"width": 900, "height": 680}, device_scale_factor=2
            )
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            await page.goto(html.resolve().as_uri(), wait_until="load")
            await page.wait_for_function("window.sceneReady === true", timeout=60000)
            info = await page.evaluate("window.sceneInfo")
            actual = await page.evaluate("window.sceneCamera")
            if (
                errors
                or "SwiftShader" not in info["renderer"]
                or not np.allclose(actual, camera, atol=1e-7)
            ):
                raise ValueError(f"Molecular capture failed: {stem}, {errors}, {info}")
            await page.screenshot(path=str(output), animations="disabled")
            metadata = dict(
                trace_sha256=hashlib.sha256((DATA / "trace.json").read_bytes()).hexdigest(),
                sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                camera=actual,
                scene=info,
                javascript_url=JS_URL,
                javascript_sha256=JS_SHA256,
                chromium_version=browser.version,
                py3dmol_version=importlib.metadata.version("py3Dmol"),
                playwright_version=importlib.metadata.version("playwright"),
                description=(
                    "Stored receptor only; ligand hidden; darker input cartoon. No predicted pocket or cutoff boundary."
                    if pocket
                    else "CPU refinement of the exact stored N1 endpoint; original camera and fragment colors."
                ),
            )
            if not pocket:
                metadata["refinement_sha256"] = hashlib.sha256(
                    (DATA / "representative_refinement.json").read_bytes()
                ).hexdigest()
            (DATA / f"views/{stem}.json").write_text(json.dumps(metadata, indent=2) + "\n")
            await page.close()
        await browser.close()
    verify_extra_views()
    print("Captured supplied pocket and matched refined endpoint")


def verify_refinement(row):
    record = json.loads((DATA / "representative_refinement.json").read_text())
    expected = hashlib.sha256((DATA / "trace.json").read_bytes()).hexdigest()
    raw = np.asarray(row["coordinates"][-1])
    coords = np.asarray(record["coordinates"])
    if record["trace_sha256"] != expected or not np.array_equal(record["raw_coordinates"], raw):
        raise ValueError("Refinement does not start from this trajectory")
    if (
        coords.shape[1:] != raw.shape
        or not np.isfinite(coords).all()
        or not np.allclose(coords[0], raw, atol=1e-4)
    ):
        raise ValueError("Invalid refinement geometry")
    for f in range(6):
        xyz = coords[:, np.asarray(row["fragment_id"]) == f]
        distances = np.linalg.norm(xyz[:, :, None] - xyz[:, None, :], axis=-1)
        if np.max(np.abs(distances - distances[0])) > 1e-3:
            raise ValueError("Non-rigid refined fragment")
    energies = np.asarray(
        [[r[k] for k in ("physical", "interaction", "combined")] for r in record["energies"]]
    )
    if not np.isfinite(energies).all() or not np.allclose(
        energies[:, 0] + energies[:, 1], energies[:, 2], atol=1e-4
    ):
        raise ValueError("Invalid refinement energy groups")
    for name, digest in record["source_code"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise ValueError("Refinement implementation differs from the recorded source")
    return record


def load():
    verify()
    verify_extra_views()
    row = json.loads((DATA / "trace.json").read_text())
    verify_refinement(row)
    xyz = np.asarray(row["coordinates"])
    frag = np.asarray(row["fragment_id"])
    for f in range(len(COLORS)):
        ids = np.flatnonzero(frag == f)
        d = np.linalg.norm(xyz[:, ids, None] - xyz[:, None, ids], axis=-1)
        if len(ids) < 2 or np.abs(d - d[0]).max() > 1e-3:
            raise ValueError(f"Fragment {f} is not rigid across saved frames")
    if POSE_INDEX not in row["shown_indices"]:
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


def ligand_diagram(ax, row, fragmented=False):
    diagram = json.loads((DATA / "ligand_diagram.json").read_text())
    if diagram["trace_sha256"] != hashlib.sha256((DATA / "trace.json").read_bytes()).hexdigest():
        raise ValueError("Stale ligand decomposition diagram")
    if diagram["elements"] != row["elements"] or diagram["fragment_id"] != row["fragment_id"]:
        raise ValueError("Diagram fragment identity differs from trajectory")
    if sorted(sorted(b["atoms"]) for b in diagram["bonds"]) != sorted(
        sorted(b) for b in row["bonds"]
    ):
        raise ValueError("Diagram connectivity mismatch")
    xy = np.asarray(diagram["coordinates"])
    colors = [
        ELEMENT_COLORS.get(e, COLORS[f] if fragmented else "#7D8791")
        for e, f in zip(diagram["elements"], diagram["fragment_id"], strict=True)
    ]
    for bond in diagram["bonds"]:
        a, b = bond["atoms"]
        cut = diagram["fragment_id"][a] != diagram["fragment_id"][b]
        if cut != bond["cut"]:
            raise ValueError("Diagram cut-bond assignment mismatch")
        if cut and fragmented:
            continue
        p, q = xy[a], xy[b]
        d = q - p
        normal = np.array([-d[1], d[0]]) / np.linalg.norm(d) * 0.11
        offsets = (-normal, normal) if bond["order"] in (1.5, 2) else (np.zeros(2),)
        for k, offset in enumerate(offsets):
            mid = (p + q) / 2 + offset
            style = "--" if bond["aromatic"] and k == 1 else "-"
            for start, end, color in ((p + offset, mid, colors[a]), (mid, q + offset, colors[b])):
                ax.plot(
                    *zip(start, end),
                    color=color,
                    linewidth=1.25,
                    linestyle=style,
                    solid_capstyle="round",
                    zorder=2,
                )
        if cut and not fragmented:
            mid = (p + q) / 2
            ax.plot(
                *zip(mid - 2.7 * normal, mid + 2.7 * normal),
                color="#C58A73",
                linewidth=1.3,
                zorder=4,
            )
    ax.scatter(*xy.T, s=8, c=colors, linewidths=0, zorder=3)
    ax.set_xlim(xy[:, 0].min() - 0.7, xy[:, 0].max() + 0.7)
    ax.set_ylim(xy[:, 1].min() - 0.7, xy[:, 1].max() + 0.7)
    ax.set_aspect("equal")
    ax.set_axis_off()


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
    raw = plt.imread(DATA / f"views/frame_{POSE_INDEX:02d}.png")
    refined = plt.imread(DATA / "views/refined_endpoint.png")
    pocket = plt.imread(DATA / "views/supplied_pocket.png")
    crop = common_crop([raw, refined])
    height = 2.2
    fw = 1.43
    fh = fw * (crop[1] - crop[0]) / (crop[3] - crop[2])
    middle = 1.23
    bottom = middle - fh / 2
    fig = plt.figure(figsize=(WIDTH, height))
    arts = []

    def rect(x, y, w, h):
        return [x / WIDTH, y / height, w / WIDTH, h / height]

    def label(x, y, text, size=9.5, weight="semibold", color=DARK):
        fig.text(
            x / WIDTH,
            y / height,
            text,
            fontsize=size,
            weight=weight,
            color=color,
            ha="center",
            va="center",
        )

    def card(x, y, w, h, edge=FRAME_EDGE, fill="white", lw=0.7):
        patch = FancyBboxPatch(
            (x / WIDTH, y / height),
            w / WIDTH,
            h / height,
            boxstyle="round,pad=0,rounding_size=0.006",
            transform=fig.transFigure,
            facecolor=fill,
            edgecolor=edge,
            linewidth=lw,
            mutation_aspect=WIDTH / height,
            zorder=-1,
        )
        fig.add_artist(patch)

    def candidates(x, pixels):
        # Empty backing cards denote multiplicity, not fabricated molecular poses.
        for offset, fill in ((0.14, "#F3F6F8"), (0.07, "#FAFBFC")):
            card(x + offset, bottom + offset, fw, fh, fill=fill)
        arts.append(frame(fig, rect(x, bottom, fw, fh), pixels, crop, FRAME_EDGE, 0.7))

    label(0.60, 1.96, "Ligand")
    label(2.04, 1.96, "Rigid fragments")
    ligand_diagram(fig.add_axes(rect(0.03, middle - 0.32, 1.13, 0.64)), row)
    ligand_diagram(fig.add_axes(rect(1.47, middle - 0.32, 1.13, 0.64)), row, fragmented=True)
    connector(fig, 1.20, 1.42, middle, WIDTH, height)

    pw = 1.05
    ph = pw * (crop[1] - crop[0]) / (crop[3] - crop[2])
    arts.append(frame(fig, rect(1.515, 0.10, pw, ph), pocket, crop, FRAME_EDGE, 0.6))
    label(2.04, 0.10 + ph + 0.13, "Given pocket", size=8, weight="normal")
    # A short local join combines the two supplied inputs before generation.
    fig.add_artist(
        plt.Line2D(
            [2.62 / WIDTH, 2.81 / WIDTH, 2.81 / WIDTH],
            [(0.10 + ph / 2) / height, (0.10 + ph / 2) / height, middle / height],
            color=CONNECTOR,
            linewidth=0.9,
        )
    )
    connector(fig, 2.65, 2.97, middle, WIDTH, height)

    candidates(3.02, raw)
    label(3.79, 1.96, "Pose generation")
    label(3.79, 0.56, r"SE(3) flow · $t$: 0 → 1", size=8, weight="normal")
    connector(fig, 4.63, 4.89, middle, WIDTH, height)

    candidates(4.94, refined)
    label(5.71, 1.96, "Post-refinement")
    label(5.71, 0.56, "Physical + interaction energy", size=8, weight="normal")
    connector(fig, 6.55, 6.79, middle, WIDTH, height)

    label(7.14, 1.96, "Confidence\nselection", size=9)
    # Rank glyphs are conceptual. No confidence scores were computed for this N1 example.
    for y, text, edge, fill in (
        (middle - 0.10, "1", "#78A797", "#E6F1EA"),
        (middle - 0.37, "2", FRAME_EDGE, "#F4F6F8"),
        (middle - 0.64, "⋯", FRAME_EDGE, "#F4F6F8"),
    ):
        card(6.87, y, 0.54, 0.20, edge=edge, fill=fill)
        label(7.14, y + 0.10, text, size=8, weight="normal")
    label(7.14, 0.30, "Lowest predicted\nRMSD", size=8, weight="normal")
    connector(fig, 7.47, 7.84, middle, WIDTH, height)

    arts.append(frame(fig, rect(7.90, bottom, fw, fh), refined, crop, "#78A797", 1.0))
    label(8.615, 1.96, "Selected pose")
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
    print(f"Rendered {NAME} PDF/SVG/PNG in {out}; saved endpoint and conceptual selection")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/paper_figures")
    parser.add_argument(
        "--javascript", type=Path, help="Recapture the pocket and refined pose with pinned 3Dmol.js"
    )
    parser.add_argument("--work-dir", type=Path, default=ROOT / "outputs/representative_webgl")
    parser.add_argument("--capture-only", action="store_true")
    args = parser.parse_args()
    if args.capture_only and not args.javascript:
        parser.error("--capture-only requires --javascript")
    if args.javascript:
        blob = args.javascript.read_bytes()
        if hashlib.sha256(blob).hexdigest() != JS_SHA256:
            raise ValueError("3Dmol.js identity mismatch")
        asyncio.run(capture_views(blob.decode(), args.work_dir))
    if not args.capture_only:
        render(args.output)


if __name__ == "__main__":
    main()
