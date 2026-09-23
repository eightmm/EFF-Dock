"""Render a saved ODE time series with a fixed camera and fragment colors."""

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from benchmarks.figures.evidence import save
from benchmarks.figures.structure_views import ELEMENT_COLORS, JS_SHA256, JS_URL, atoms, viewer_html

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "benchmarks/results/paper/trajectory"
COLORS = ["#77AFD1", "#E6AC83", "#85BDA3", "#AF98C5", "#C8BC78", "#D38EAE"]


def scene(row, frame_index, javascript, *, pocket_only=False, camera=None, zoom_factor=1.0):
    import py3Dmol

    view = py3Dmol.view(width=900, height=680)
    view.setBackgroundColor("white")
    view.addModel(row["protein_display"]["pdb"], "pdb", {"keepH": False})
    view.setStyle({"model": 0}, {"cartoon": {"color": "#B7C7D8", "opacity": 0.35, "arrows": True}})
    xyz = np.asarray(row["coordinates"])[frame_index]
    fragment_ids = np.asarray(row["fragment_id"])
    for frag, carbon in enumerate(COLORS):
        ids = np.flatnonzero(fragment_ids == frag)
        mapping = {int(i): j for j, i in enumerate(ids)}
        molecule = dict(
            coordinates=xyz[ids].tolist(),
            elements=[row["elements"][i] for i in ids],
            bonds=[
                [mapping[a], mapping[b]] for a, b in row["bonds"] if a in mapping and b in mapping
            ],
        )
        view.addModel()
        view.getModel(frag + 1).addAtoms(atoms(molecule))
        scheme = dict(ELEMENT_COLORS, C=carbon)
        view.setStyle(
            {"model": frag + 1},
            {
                "stick": {"colorscheme": scheme, "radius": 0.20, "singleBonds": True},
                "sphere": {"colorscheme": scheme, "scale": 0.23},
            },
        )
    if not pocket_only and frame_index == len(row["times"]) - 1:
        for a, b in row["bonds"]:
            if fragment_ids[a] != fragment_ids[b]:
                view.addCylinder(
                    dict(
                        start=dict(zip("xyz", xyz[a].tolist())),
                        end=dict(zip("xyz", xyz[b].tolist())),
                        radius=0.13,
                        color="#99A2AA",
                        fromCap=1,
                        toCap=1,
                    )
                )
    # Invisible union of actual coordinates fixes framing across all five states.
    view.addModel()
    union = np.asarray(row["coordinates"])[row["shown_indices"]].reshape(-1, 3)
    view.getModel(7).addAtoms(
        [dict(elem="C", x=float(x), y=float(y), z=float(z)) for x, y, z in union]
    )
    view.setStyle({"model": 7}, {})
    view.setProjection("orthographic")
    view.rotate(30, "y")
    view.rotate(-15, "x")
    view.zoomTo({"model": 7})
    view.zoom(2.0)
    if pocket_only:
        view.setStyle({"model": list(range(1, 7))}, {})
        view.setStyle(
            {"model": 0},
            {"cartoon": {"color": "#94ABC3", "opacity": 0.85, "arrows": True}},
        )
    if camera is not None:
        values = np.asarray(camera, dtype=float)
        if values.shape != (8,) or not np.isfinite(values).all():
            raise ValueError("Invalid fixed molecular camera")
        view.setView(values.tolist())
    if not np.isfinite(zoom_factor) or zoom_factor <= 0:
        raise ValueError("Invalid camera zoom factor")
    if zoom_factor != 1.0:
        view.zoom(float(zoom_factor))
    view.render()
    return viewer_html(view, javascript)


async def capture(javascript, work):
    from PIL import Image
    from playwright.async_api import async_playwright

    row = json.loads((DATA / "trace.json").read_text())
    work.mkdir(parents=True, exist_ok=True)
    out = DATA / "views"
    out.mkdir(exist_ok=True)
    records = []
    async with async_playwright() as api:
        browser = await api.chromium.launch(
            args=[
                "--use-gl=angle",
                "--use-angle=swiftshader",
                "--enable-unsafe-swiftshader",
                "--disable-dev-shm-usage",
            ]
        )
        version = browser.version
        for index in row["shown_indices"]:
            html = work / f"frame_{index:02d}.html"
            html.write_text(scene(row, index, javascript))
            page = await browser.new_page(
                viewport={"width": 900, "height": 680}, device_scale_factor=2
            )
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            await page.goto(html.resolve().as_uri(), wait_until="load")
            await page.wait_for_function("window.sceneReady === true", timeout=60000)
            info = await page.evaluate("window.sceneInfo")
            camera = await page.evaluate("window.sceneCamera")
            if errors or "SwiftShader" not in info["renderer"]:
                raise ValueError(f"Molecular scene failed: {errors}, {info}")
            filename = f"frame_{index:02d}.png"
            await page.screenshot(path=str(out / filename), animations="disabled")
            pixels = np.asarray(Image.open(out / filename).convert("RGB"))
            if np.mean(np.min(pixels, axis=-1) < 230) < 0.015:
                raise ValueError("Blank trajectory image")
            records.append(
                dict(
                    index=index,
                    time=row["times"][index],
                    file=filename,
                    sha256=hashlib.sha256((out / filename).read_bytes()).hexdigest(),
                    camera=camera,
                    scene=info,
                )
            )
            await page.close()
        await browser.close()
    for record in records:
        if not np.allclose(record["camera"], records[0]["camera"], atol=1e-7):
            raise ValueError("Camera differs between frames")
    manifest = dict(
        trace_sha256=hashlib.sha256((DATA / "trace.json").read_bytes()).hexdigest(),
        javascript_url=JS_URL,
        javascript_sha256=JS_SHA256,
        chromium_version=version,
        py3dmol_version=importlib.metadata.version("py3Dmol"),
        playwright_version=importlib.metadata.version("playwright"),
        fragment_carbon_colors=COLORS,
        records=records,
    )
    (DATA / "views/manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Captured five trajectory frames with identical cameras")


def verify():
    row = json.loads((DATA / "trace.json").read_text())
    xyz = np.asarray(row["coordinates"])
    times = np.asarray(row["times"])
    if xyz.shape != (11, 37, 3) or not np.isfinite(xyz).all():
        raise ValueError("Invalid trajectory coordinates")
    if times[0] != 0 or times[-1] != 1 or not (np.diff(times) > 0).all():
        raise ValueError("Invalid trajectory times")
    expected = [int(np.abs(times - t).argmin()) for t in (0, 0.25, 0.5, 0.75, 1)]
    if row["shown_indices"] != expected:
        raise ValueError("Unexpected frame selection")
    manifest = json.loads((DATA / "views/manifest.json").read_text())
    if manifest["trace_sha256"] != hashlib.sha256((DATA / "trace.json").read_bytes()).hexdigest():
        raise ValueError("Stale trajectory images")
    if [r["index"] for r in manifest["records"]] != expected:
        raise ValueError("Missing trajectory frame")
    for r in manifest["records"]:
        if (
            r["time"] != times[r["index"]]
            or hashlib.sha256((DATA / "views" / r["file"]).read_bytes()).hexdigest() != r["sha256"]
        ):
            raise ValueError("Trajectory capture mismatch")
        if not np.allclose(r["camera"], manifest["records"][0]["camera"], atol=1e-7):
            raise ValueError("Trajectory camera drift")


def render(out):
    verify()
    row = json.loads((DATA / "trace.json").read_text())
    out.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"font.size": 10, "pdf.fonttype": 42}):
        fig, axes = plt.subplots(1, 5, figsize=(18, 3.3))
        for ax, index in zip(axes, row["shown_indices"], strict=True):
            ax.imshow(plt.imread(DATA / f"views/frame_{index:02d}.png"))
            ax.set_axis_off()
            ax.set_title(
                f"$t = {row['times'][index]:.2f}$",
                loc="left",
                weight="bold",
                fontsize=12,
            )
        fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.03, wspace=0.04)
        save(fig, out, "S10_fragment_trajectory", dpi=400)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--javascript", type=Path)
    parser.add_argument("--work-dir", type=Path, default=ROOT / "outputs/trajectory_webgl")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/paper/figures")
    args = parser.parse_args()
    if args.javascript:
        blob = args.javascript.read_bytes()
        if hashlib.sha256(blob).hexdigest() != JS_SHA256:
            raise ValueError("3Dmol.js identity mismatch")
        asyncio.run(capture(blob.decode(), args.work_dir))
    render(args.output)


if __name__ == "__main__":
    main()
