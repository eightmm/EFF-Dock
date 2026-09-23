"""Capture real receptor cartoons and element-colored poses with software 3Dmol.js."""

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "benchmarks/results/paper/evidence"
JS_URL = "https://cdn.jsdelivr.net/npm/3dmol@2.5.5/build/3Dmol-min.js"
JS_SHA256 = "f7cc78921ae72e7623e89cdd111434f58c2efddd2ffda1cd212644b406fb8016"
POSE_COLORS = {"reference": "#78BAA1", "selected": "#73ACD1", "comparator": "#E7AA7C"}
ELEMENT_COLORS = {
    "N": "#365BD4",
    "O": "#E44545",
    "S": "#D9BC32",
    "P": "#B877CB",
    "F": "#69B871",
    "Cl": "#55A862",
    "Br": "#A86842",
    "I": "#9158AF",
}


def atoms(mol):
    result = []
    for i, (element, xyz) in enumerate(zip(mol["elements"], mol["coordinates"], strict=True)):
        bonds = [b if a == i else a for a, b in mol["bonds"] if i in (a, b)]
        result.append(
            dict(
                elem=element, x=xyz[0], y=xyz[1], z=xyz[2], bonds=bonds, bondOrder=[1] * len(bonds)
            )
        )
    return result


def make_html(row, javascript):
    import py3Dmol

    view = py3Dmol.view(width=900, height=680)
    view.addModel(row["protein_display"]["pdb"], "pdb", {"keepH": False})
    view.setStyle(
        {"model": 0},
        {
            "cartoon": {
                "color": "#B7C7D8",
                "opacity": 0.45,
                "arrows": True,
                "thickness": 0.35,
            }
        },
    )
    models = []
    for key, carbon in POSE_COLORS.items():
        if row[key] is None:
            continue
        model = len(models) + 1
        view.addModel()
        view.getModel(model).addAtoms(atoms(row[key]))
        scheme = dict(ELEMENT_COLORS, C=carbon)
        view.setStyle(
            {"model": model},
            {
                "stick": {
                    "colorscheme": scheme,
                    "radius": 0.19,
                    "singleBonds": True,
                },
                "sphere": {"colorscheme": scheme, "scale": 0.20},
            },
        )
        models.append(model)
    view.setProjection("orthographic")
    view.setBackgroundColor("white")
    # One common camera rotation for every molecule; no coordinate alignment.
    view.rotate(30, "y")
    view.rotate(-15, "x")
    view.zoomTo({"model": models})
    view.zoom(1.1)
    view.render()
    return viewer_html(view, javascript)


def viewer_html(view, javascript):
    content = view._make_html()
    variable = re.search(r"var (viewer_[A-Za-z0-9_]+) = null;", content).group(1)
    return (
        '<!doctype html><html><head><meta charset="utf-8"><style>html,body{margin:0;background:white;width:900px;height:680px;overflow:hidden}</style><script>'
        + javascript
        + "</script><script>var $3Dmolpromise=Promise.resolve();</script></head><body>"
        + content
        + """<script>
$3Dmolpromise.then(function(){
const viewer=VIEWER;
window.sceneCamera=viewer.getView();
const gl=viewer.getRenderer().getContext();
const info=gl.getExtension('WEBGL_debug_renderer_info');
window.sceneInfo={renderer:info ? gl.getParameter(info.UNMASKED_RENDERER_WEBGL) : 'unavailable', atoms:viewer.getModel(0).selectedAtoms({}).length, secondary:viewer.getModel(0).selectedAtoms({atom:'CA'}).reduce((acc,a)=>{acc[a.ss]=(acc[a.ss]||0)+1;return acc;},{})};
viewer.render();
requestAnimationFrame(()=>requestAnimationFrame(()=>{window.sceneReady=true;}));
});</script></body></html>""".replace("VIEWER", variable)
    )


async def capture(javascript, work, out):
    from playwright.async_api import async_playwright

    source = DATA / "structures.json"
    rows = json.loads(source.read_text())
    work.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
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
        for row in rows:
            mode = "pocket"
            stem = f"{row['id']}_{mode}"
            html = work / f"{stem}.html"
            html.write_text(make_html(row, javascript))
            page = await browser.new_page(
                viewport={"width": 900, "height": 680}, device_scale_factor=2
            )
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            await page.goto(html.resolve().as_uri(), wait_until="load")
            await page.wait_for_function("window.sceneReady === true", timeout=60000)
            scene = await page.evaluate("window.sceneInfo")
            if errors or "SwiftShader" not in scene["renderer"]:
                raise ValueError(
                    f"Scene failed or did not use software rendering: {errors}, {scene}"
                )
            if not scene["secondary"].get("h", 0) + scene["secondary"].get("s", 0):
                raise ValueError("No helix/sheet assignment in receptor cartoon")
            output = out / f"{stem}.png"
            await page.screenshot(path=str(output), animations="disabled")
            # Reject blank captures, not small camera/color differences across GPUs.
            from PIL import Image

            pixels = np.asarray(Image.open(output).convert("RGB"))
            nonwhite = np.mean(np.min(pixels, axis=-1) < 230)
            if nonwhite < 0.015:
                raise ValueError(f"Empty molecular scene: {stem}")
            records.append(
                dict(
                    id=row["id"],
                    mode=mode,
                    file=output.name,
                    sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                    scene=scene,
                    nonwhite_fraction=float(nonwhite),
                )
            )
            print(stem, scene, f"visible fraction={nonwhite:.3f}", flush=True)
            await page.close()
        await browser.close()
    manifest = dict(
        structures_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        javascript_url=JS_URL,
        javascript_sha256=JS_SHA256,
        py3dmol_version=importlib.metadata.version("py3Dmol"),
        playwright_version=importlib.metadata.version("playwright"),
        chromium_version=version,
        pose_carbon_colors=POSE_COLORS,
        element_colors=ELEMENT_COLORS,
        records=records,
    )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--javascript", type=Path, required=True, help=f"Local copy of {JS_URL}")
    parser.add_argument("--work-dir", type=Path, default=ROOT / "outputs/structure_webgl")
    parser.add_argument("--output", type=Path, default=DATA / "structure_views")
    args = parser.parse_args()
    blob = args.javascript.read_bytes()
    if hashlib.sha256(blob).hexdigest() != JS_SHA256:
        raise ValueError("3Dmol.js version/checksum mismatch")
    asyncio.run(capture(blob.decode(), args.work_dir, args.output))


if __name__ == "__main__":
    main()
