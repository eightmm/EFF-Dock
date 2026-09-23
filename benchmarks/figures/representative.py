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
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch

from benchmarks.figures.structure_views import ELEMENT_COLORS, JS_SHA256, JS_URL, atoms, viewer_html
from benchmarks.figures.trajectory import COLORS, DATA, ROOT, scene, verify

NAME = "Fig1_representative"
POSE_INDEX = 10
FLOW_INDICES = (0, 2, 4, 10)
REFINEMENT_STEPS = (0, 25, 50, 100)
EXTRA_VIEWS = {
    "supplied_pocket": None,
    "refined_endpoint": 100,
    "refined_step_025": 25,
    "refined_step_050": 50,
}
CANDIDATE_VIEWS = {
    "candidate_best": 0,
    "candidate_quartile": 1,
    "candidate_median": 2,
    "candidate_worst": 3,
}
OUTPUT_VIEWS = {"selected_overlay": 0}
OUTPUT_COLORS = {"selected": "#79AEC8", "crystal": "#D9A080"}
FLOW_VIEWS = {f"flow_{i:02d}": i for i in FLOW_INDICES}
OVERVIEW_DIR = DATA / "views/overview"
OVERVIEW_ZOOM = 0.70
DARK = "#3E434A"
MUTED = "#7B838D"
CONNECTOR = "#9AAEBB"
FRAME_EDGE = "#D6DBE1"
POSE_EDGE = "#8C96A3"

# Landscape overview; preserve vector labels when scaling for the manuscript.
WIDTH = 10.35


def selection_data():
    record = json.loads((DATA / "selection_example.json").read_text())
    if record["trace_sha256"] != hashlib.sha256((DATA / "trace.json").read_bytes()).hexdigest():
        raise ValueError("Selection illustration has stale trace identity")
    predictions = np.asarray(record["predicted_rmsd"])
    if predictions.shape != (100,) or not np.isfinite(predictions).all():
        raise ValueError("Invalid stored confidence scores")
    order = sorted(range(100), key=lambda i: (predictions[i], i))
    eligible = [i for i in order if record["chirality_valid"][i]]
    ranks = (0, (len(eligible) - 1) // 4, (len(eligible) - 1) // 2, len(eligible) - 1)
    if record["selected_index"] != eligible[0] or len(record["candidates"]) != 4:
        raise ValueError("Stored confidence selection differs")
    for candidate, rank in zip(record["candidates"], ranks, strict=True):
        index = eligible[rank]
        coords = np.asarray(candidate["coordinates"])
        if (
            candidate["index"] != index
            or candidate["rank"] != rank + 1
            or candidate["selected"] != (rank == 0)
            or candidate["predicted_rmsd"] != predictions[index]
            or coords.shape != (37, 3)
            or not np.isfinite(coords).all()
            or sorted(candidate["atom_mapping"]) != list(range(37))
        ):
            raise ValueError("Invalid confidence candidate record")
    return record


def annotation_data():
    record = json.loads((DATA / "candidate_annotations.json").read_text())
    for filename, key in (
        ("selection_example.json", "selection_sha256"),
        ("selected_reference.json", "reference_sha256"),
    ):
        if hashlib.sha256((DATA / filename).read_bytes()).hexdigest() != record[key]:
            raise ValueError("Stale candidate RMSD annotations")
    if record["unit"] != "angstrom":
        raise ValueError("Unexpected candidate RMSD units")
    for annotation, candidate in zip(
        record["candidates"], selection_data()["candidates"], strict=True
    ):
        if any(annotation[key] != candidate[key] for key in ("index", "rank", "predicted_rmsd")):
            raise ValueError("Candidate annotation identity differs")
        values = [annotation[key] for key in ("predicted_rmsd", "symmetry_rmsd", "verified_rmsd")]
        if (
            not np.isfinite(values).all()
            or min(values) < 0
            or not np.isclose(values[1], values[2], atol=1e-6, rtol=0)
        ):
            raise ValueError("Invalid candidate RMSD annotation")
    return record["candidates"]


def reference_data():
    record = json.loads((DATA / "selected_reference.json").read_text())
    selection = selection_data()
    if (
        record["selection_sha256"]
        != hashlib.sha256((DATA / "selection_example.json").read_bytes()).hexdigest()
    ):
        raise ValueError("Stale selected/crystal overlay")
    if record["selected_index"] != selection["selected_index"]:
        raise ValueError("Overlay does not show the selected pose")
    coords = np.asarray(record["reference"]["coordinates"])
    rmsd = record["symmetry_rmsd_angstrom"]
    if (
        coords.shape != (37, 3)
        or not np.isfinite(coords).all()
        or not np.isfinite(rmsd)
        or rmsd < 0
    ):
        raise ValueError("Invalid crystal reference or RMSD")
    if not np.isclose(rmsd, record["verified_rmsd_angstrom"], atol=1e-6, rtol=0):
        raise ValueError("Overlay RMSD verification differs")
    return record


def input_scene(row, javascript, camera):
    import py3Dmol

    view = py3Dmol.view(width=900, height=680)
    view.setBackgroundColor("white")
    view.addModel(row["protein_display"]["pdb"], "pdb", {"keepH": False})
    view.setStyle({"model": 0}, {"cartoon": {"color": "#ABC4D3", "opacity": 0.55, "arrows": True}})
    view.setProjection("orthographic")
    view.setView(camera)
    view.zoom(OVERVIEW_ZOOM)
    view.render()
    return viewer_html(view, javascript)


def output_scene(row, selection, javascript, camera):
    import py3Dmol

    view = py3Dmol.view(width=900, height=680)
    view.setBackgroundColor("white")
    view.addModel(row["protein_display"]["pdb"], "pdb", {"keepH": False})
    view.setStyle({"model": 0}, {"cartoon": {"color": "#B7C7D8", "opacity": 0.35, "arrows": True}})
    chosen = dict(
        coordinates=selection["candidates"][0]["coordinates"],
        elements=row["elements"],
        bonds=row["bonds"],
    )
    for index, (key, molecule) in enumerate(
        (("selected", chosen), ("crystal", reference_data()["reference"])), start=1
    ):
        view.addModel()
        view.getModel(index).addAtoms(atoms(molecule))
        scheme = dict(ELEMENT_COLORS, C=OUTPUT_COLORS[key])
        view.setStyle(
            {"model": index},
            {
                "stick": {"colorscheme": scheme, "radius": 0.15, "singleBonds": True},
                "sphere": {"colorscheme": scheme, "scale": 0.16},
            },
        )
    view.setProjection("orthographic")
    view.setView(camera)
    view.zoom(OVERVIEW_ZOOM)
    view.render()
    return viewer_html(view, javascript)


def verify_extra_views():
    captures = json.loads((DATA / "views/manifest.json").read_text())
    candidates = selection_data()
    for stem, step in (EXTRA_VIEWS | CANDIDATE_VIEWS | FLOW_VIEWS | OUTPUT_VIEWS).items():
        metadata = json.loads((OVERVIEW_DIR / f"{stem}.json").read_text())
        sources = [
            ("trace.json", metadata["trace_sha256"]),
            (f"views/overview/{stem}.png", metadata["sha256"]),
        ]
        if stem in OUTPUT_VIEWS:
            reference_data()
            sources.append(("selected_reference.json", metadata["reference_sha256"]))
        elif stem in CANDIDATE_VIEWS:
            if metadata.get("candidate_index") != candidates["candidates"][step]["index"]:
                raise ValueError(f"Incorrect confidence candidate: {stem}")
            sources.append(("selection_example.json", metadata["selection_sha256"]))
        elif stem in FLOW_VIEWS:
            if metadata.get("flow_index") != step:
                raise ValueError(f"Incorrect flow frame: {stem}")
        elif step is not None:
            if stem != "refined_endpoint" and metadata.get("refinement_step") != step:
                raise ValueError(f"Incorrect refinement step: {stem}")
            sources.append(("representative_refinement.json", metadata["refinement_sha256"]))
        for name, expected in sources:
            if hashlib.sha256((DATA / name).read_bytes()).hexdigest() != expected:
                raise ValueError(f"Stale molecular capture: {stem}")
        reference = captures["records"][0]["camera"]
        camera = metadata["camera"]
        if not np.allclose(camera[:3] + camera[4:], reference[:3] + reference[4:], atol=1e-7):
            raise ValueError(f"Overview camera orientation/center changed: {stem}")
        first = json.loads((OVERVIEW_DIR / "supplied_pocket.json").read_text())
        if (
            not np.allclose(camera, first["camera"], atol=1e-7)
            or metadata.get("zoom_factor") != OVERVIEW_ZOOM
        ):
            raise ValueError(f"Overview camera differs between panels: {stem}")


async def capture_views(javascript, work, *, stems=None):
    from playwright.async_api import async_playwright

    verify()
    row = json.loads((DATA / "trace.json").read_text())
    refined = json.loads((DATA / "representative_refinement.json").read_text())
    camera = json.loads((DATA / "views/manifest.json").read_text())["records"][0]["camera"]
    verify_refinement(row)
    candidates = selection_data()
    if (
        stems is not None
        and set(stems) - (EXTRA_VIEWS | CANDIDATE_VIEWS | FLOW_VIEWS | OUTPUT_VIEWS).keys()
    ):
        raise ValueError("Unknown molecular capture request")
    work.mkdir(parents=True, exist_ok=True)
    OVERVIEW_DIR.mkdir(parents=True, exist_ok=True)
    async with async_playwright() as api:
        browser = await api.chromium.launch(
            args=[
                "--use-gl=angle",
                "--use-angle=swiftshader",
                "--enable-unsafe-swiftshader",
                "--disable-dev-shm-usage",
            ]
        )
        for stem, step in (EXTRA_VIEWS | CANDIDATE_VIEWS | FLOW_VIEWS | OUTPUT_VIEWS).items():
            if stems is not None and stem not in stems:
                continue
            pocket = step is None
            index = 0 if pocket else POSE_INDEX
            source = row
            is_flow = stem in FLOW_VIEWS
            if is_flow:
                index = step
            is_candidate = stem in CANDIDATE_VIEWS
            if is_candidate:
                source = dict(
                    row,
                    coordinates=row["coordinates"][:-1]
                    + [candidates["candidates"][step]["coordinates"]],
                )
            elif not pocket and not is_flow:
                saved = refined["saved_steps"].index(step)
                source = dict(
                    row, coordinates=row["coordinates"][:-1] + [refined["coordinates"][saved]]
                )
            html = work / f"{stem}.html"
            html.write_text(
                input_scene(row, javascript, camera)
                if pocket
                else output_scene(row, candidates, javascript, camera)
                if stem in OUTPUT_VIEWS
                else scene(
                    source,
                    index,
                    javascript,
                    pocket_only=pocket,
                    camera=camera,
                    zoom_factor=OVERVIEW_ZOOM,
                )
            )
            output = OVERVIEW_DIR / f"{stem}.png"
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
                or not np.allclose(actual[:3] + actual[4:], camera[:3] + camera[4:], atol=1e-7)
            ):
                raise ValueError(f"Molecular capture failed: {stem}, {errors}, {info}")
            await page.screenshot(path=str(output), animations="disabled")
            metadata = dict(
                trace_sha256=hashlib.sha256((DATA / "trace.json").read_bytes()).hexdigest(),
                sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                camera=actual,
                zoom_factor=OVERVIEW_ZOOM,
                scene=info,
                javascript_url=JS_URL,
                javascript_sha256=JS_SHA256,
                chromium_version=browser.version,
                py3dmol_version=importlib.metadata.version("py3Dmol"),
                playwright_version=importlib.metadata.version("playwright"),
                description=(
                    "Stored receptor only; pale blue-gray input cartoon. No predicted pocket or cutoff boundary."
                    if pocket
                    else "Recorded refinement of the stored N1 endpoint; shared overview camera and original fragment colors."
                ),
            )
            if stem in OUTPUT_VIEWS:
                metadata["description"] = (
                    "Selected pose and evaluated crystal reference; receptor frame; no alignment"
                )
                metadata["reference_sha256"] = hashlib.sha256(
                    (DATA / "selected_reference.json").read_bytes()
                ).hexdigest()
            elif is_candidate:
                metadata["description"] = (
                    "Existing N100 refined candidate; matched receptor frame and camera; selected/not-selected marks are not PB validity"
                )
                metadata["candidate_index"] = candidates["candidates"][step]["index"]
                metadata["selection_sha256"] = hashlib.sha256(
                    (DATA / "selection_example.json").read_bytes()
                ).hexdigest()
            elif is_flow:
                metadata["description"] = "Stored ODE frame; overview camera; unchanged coordinates"
                metadata["flow_index"] = step
            elif not pocket:
                metadata["refinement_step"] = step
                metadata["refinement_sha256"] = hashlib.sha256(
                    (DATA / "representative_refinement.json").read_bytes()
                ).hexdigest()
            (OVERVIEW_DIR / f"{stem}.json").write_text(json.dumps(metadata, indent=2) + "\n")
            await page.close()
        await browser.close()
    verify_extra_views()
    print("Captured requested pocket/refinement views with matched cameras")


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
    if set(FLOW_INDICES) - set(row["shown_indices"]):
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
            arrowstyle="-|>",
            mutation_scale=9,
            color=CONNECTOR,
            lw=0.9,
            shrinkA=0,
            shrinkB=0,
        )
    )


def compose(row):
    flow = [plt.imread(OVERVIEW_DIR / f"flow_{i:02d}.png") for i in FLOW_INDICES]
    refinement = [flow[-1]] + [
        plt.imread(OVERVIEW_DIR / f"{stem}.png")
        for stem in ("refined_step_025", "refined_step_050", "refined_endpoint")
    ]
    pocket = plt.imread(OVERVIEW_DIR / "supplied_pocket.png")
    candidates = [plt.imread(OVERVIEW_DIR / f"{stem}.png") for stem in CANDIDATE_VIEWS]
    overlay = plt.imread(OVERVIEW_DIR / "selected_overlay.png")
    selection = selection_data()
    annotations = annotation_data()
    reference = reference_data()
    crop = common_crop([*flow, *refinement, *candidates, overlay])
    height, fw, box_width = 5.25, 1.30, 1.75
    fh = fw * (crop[1] - crop[0]) / (crop[3] - crop[2])
    positions = (3.80, 2.73, 1.66, 0.59)
    columns = (0.10, 2.20, 4.30, 6.40, 8.50)
    middle = 2.21 + fh / 2
    fig = plt.figure(figsize=(WIDTH, height))
    arts = []

    def rect(x, y, w, h):
        return [x / WIDTH, y / height, w / WIDTH, h / height]

    def label(x, y, text, size=9.5, weight="bold", color=DARK):
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

    def card(x, y, w, h, edge=FRAME_EDGE, fill="white", lw=0.6):
        fig.add_artist(
            FancyBboxPatch(
                (x / WIDTH, y / height),
                w / WIDTH,
                h / height,
                boxstyle="round,pad=0,rounding_size=0.008",
                transform=fig.transFigure,
                facecolor=fill,
                edgecolor=edge,
                linewidth=lw,
                mutation_aspect=WIDTH / height,
                zorder=-1,
            )
        )

    def down(x, top, bottom):
        fig.add_artist(
            FancyArrowPatch(
                (x / WIDTH, top / height),
                (x / WIDTH, bottom / height),
                transform=fig.transFigure,
                arrowstyle="-|>",
                mutation_scale=9,
                color=CONNECTOR,
                linewidth=0.9,
                shrinkA=0,
                shrinkB=0,
            )
        )

    def corner(x, y, text):
        fig.text(
            (x + 0.07) / WIDTH,
            (y + fh - 0.095) / height,
            text,
            fontsize=7,
            color=DARK,
            ha="left",
            va="center",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.92, "pad": 1.5},
            zorder=8,
        )

    palettes = (
        ("#F3F7FA", "#BDCDD8"),
        ("#F5F3FA", "#C9C0D9"),
        ("#FBF5F0", "#DCCBBE"),
        ("#F1F8F5", "#BBD3C9"),
    )
    for x, (fill, edge) in zip(columns[:4], palettes, strict=True):
        card(x, 0.30, box_width, 4.80, edge=edge, fill=fill, lw=0.85)
    for left, right in zip(columns[:-1], columns[1:], strict=True):
        connector(fig, left + box_width + 0.065, right - 0.065, middle, WIDTH, height)

    center = columns[0] + box_width / 2
    px = center - fw / 2
    label(center, 4.95, "Input preparation", size=9)
    for y, title, fragmented in ((3.80, "Ligand", False), (2.73, "Rigid fragments", True)):
        card(px, y, fw, fh, fill="#FCFDFE")
        ligand_diagram(fig.add_axes(rect(px, y, fw, fh)), row, fragmented=fragmented)
        corner(px, y, title)
    down(center, 3.80 - 0.07, 2.73 + fh + 0.07)
    label(center, (2.73 + 1.66 + fh) / 2, "+", size=14, weight="normal", color=MUTED)
    arts.append(frame(fig, rect(px, 1.66, fw, fh), pocket, crop, FRAME_EDGE, 0.6))
    corner(px, 1.66, "Given pocket")

    def trajectory(box_x, title, method, images, labels):
        center = box_x + box_width / 2
        px = center - fw / 2
        label(center, 4.95, title)
        label(center, 4.73, method, size=8, weight="normal")
        # Empty backplates denote parallel candidates, not additional saved traces.
        for dx, dy in ((0.07, 0.04), (0.025, 0.02), (-0.04, 0.0)):
            card(px + dx, 0.53 + dy, fw + 0.08, 4.01, edge="#D4D6DE", fill="#FDFDFE", lw=0.55)
        label(center, 0.415, r"$\times\,N$", size=8.5, weight="normal", color=MUTED)
        for i, (py, pixels, text) in enumerate(zip(positions, images, labels, strict=True)):
            arts.append(frame(fig, rect(px, py, fw, fh), pixels, crop, FRAME_EDGE, 0.55))
            corner(px, py, text)
            if i < 3:
                down(center, py - 0.07, positions[i + 1] + fh + 0.07)

    trajectory(
        columns[1],
        "Pose generation",
        "Fragment SE(3)\nflow matching",
        flow,
        [f"$t$ = {row['times'][i]:.2f}" for i in FLOW_INDICES],
    )
    trajectory(
        columns[2],
        "Post-refinement",
        "Physics- and\ninteraction-based",
        refinement,
        [f"Step {step}" for step in REFINEMENT_STEPS],
    )

    center = columns[3] + box_width / 2
    px = center - fw / 2
    label(center, 4.95, "Confidence selection", size=9)
    label(center, 4.70, "Predicted RMSD\nranking", size=8, weight="normal")
    for upper in positions[:-1]:
        label(center, upper - 0.26, "…", size=10, weight="normal", color=MUTED)
    label(center, 0.375, r"$N\,\to\,1$", size=8.5, weight="normal", color=MUTED)
    for pixels, candidate, annotation, y in zip(
        candidates, selection["candidates"], annotations, positions, strict=True
    ):
        label(
            center,
            y - 0.09,
            f"pRMSD {annotation['predicted_rmsd']:.2f} Å  /  RMSD {annotation['symmetry_rmsd']:.2f} Å",
            size=6.5,
            weight="normal",
        )
        selected = candidate["selected"]
        edge = "#82B7A4" if selected else FRAME_EDGE
        arts.append(frame(fig, rect(px, y, fw, fh), pixels, crop, edge, 0.9 if selected else 0.6))
        cx, cy = px + 0.11, y + fh - 0.10
        fig.add_artist(
            Ellipse(
                (cx / WIDTH, cy / height),
                0.18 / WIDTH,
                0.18 / height,
                transform=fig.transFigure,
                facecolor="white",
                edgecolor="none",
                zorder=8,
            )
        )
        color = "#619D86" if selected else "#C28F8F"
        paths = (
            [[(-0.05, 0), (-0.01, -0.04), (0.055, 0.05)]]
            if selected
            else [[(-0.04, -0.04), (0.04, 0.04)], [(-0.04, 0.04), (0.04, -0.04)]]
        )
        for path in paths:
            fig.add_artist(
                plt.Line2D(
                    [(cx + dx) / WIDTH for dx, _ in path],
                    [(cy + dy) / height for _, dy in path],
                    transform=fig.transFigure,
                    color=color,
                    linewidth=1.5,
                    solid_capstyle="round",
                    zorder=9,
                )
            )

    center = columns[4] + box_width / 2
    px = center - fw / 2
    card(columns[4], 1.57, box_width, 1.90, edge="#BBD3C9", fill="#F7FBF9", lw=0.85)
    label(center, 3.25, "Selected pose")
    arts.append(frame(fig, rect(px, 2.21, fw, fh), overlay, crop, "#82B7A4", 0.9))
    corner(px, 2.21, f"RMSD {reference['symmetry_rmsd_angstrom']:.2f} Å")
    for y, key, text in ((1.98, "selected", "Selected"), (1.76, "crystal", "Crystal")):
        fig.add_artist(
            plt.Line2D(
                [(center - 0.46) / WIDTH, (center - 0.24) / WIDTH],
                [y / height] * 2,
                transform=fig.transFigure,
                color=OUTPUT_COLORS[key],
                linewidth=2.8,
                solid_capstyle="round",
            )
        )
        fig.text(
            (center - 0.15) / WIDTH,
            y / height,
            text,
            fontsize=8,
            color=DARK,
            ha="left",
            va="center",
        )
    return fig, arts


def render(out):
    row = load()
    out.mkdir(parents=True, exist_ok=True)
    style = {
        "font.family": "DejaVu Sans",
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
    print(
        f"Rendered {NAME} PDF/SVG/PNG in {out}; four-state trajectories and recorded confidence selection"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/paper_figures")
    parser.add_argument(
        "--javascript",
        type=Path,
        help="Recapture the pocket and refinement states with pinned 3Dmol.js",
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
