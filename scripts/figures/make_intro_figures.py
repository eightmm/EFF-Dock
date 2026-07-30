#!/usr/bin/env python3
"""Create slide-ready introduction figures for EFF-Dock.

The script deliberately uses the repository's production fragmentation rule
(``decompose_fragments``) instead of RDKit BRICS/RECAP.  It only depends on
packages already pinned by the project (RDKit, torch, NumPy, and Pillow).
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import rdkit
import torch
from PIL import Image, ImageDraw, ImageFont
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D

from effdock.data.dataset import crop_to_pocket
from effdock.preprocess.fragments import decompose_fragments
from effdock.preprocess.ligand import load_molecule
from effdock.preprocess.protein import parse_pocket_atoms

ROOT = Path(__file__).resolve().parents[2]
ASTEX_ROOT = ROOT / "data/external_benchmarks/data/astex_diverse_set"
POCKET_CENTERS = ROOT / "data/external_test/astex_reference_pocket_centers.json"
DEFAULT_OUTPUT = ROOT / "docs/assets/intro"

FRAGMENT_CASE = "1T46_STI"
POCKET_CASE = "1T46_STI"
POCKET_CUTOFF = 10.0

# Color-blind-safe palette (Okabe-Ito plus two compatible extensions).
FRAGMENT_COLORS = (
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#CC79A7",
    "#56B4E9",
    "#D55E00",
    "#7A68A6",
    "#B59A00",
)
INK = "#172033"
MUTED = "#64748B"
PANEL_BG = "#F8FAFC"
POCKET_BLUE = "#38BDF8"
PROTEIN_GRAY = "#94A3B8"
CUT_RED = "#EF4444"


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    )
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def _rgb01(value: str) -> tuple[float, float, float]:
    value = value.lstrip("#")
    return tuple(int(value[i : i + 2], 16) / 255.0 for i in (0, 2, 4))


def _rgba01(value: str, alpha: float) -> tuple[float, float, float, float]:
    return (*_rgb01(value), alpha)


def _load_astex_mol(case_id: str) -> Chem.Mol:
    sdf = ASTEX_ROOT / case_id / f"{case_id}_ligand.sdf"
    mol, _, sanitize_ok = load_molecule(sdf)
    if mol is None:
        raise RuntimeError(f"Could not load ligand: {sdf}")
    if not sanitize_ok:
        raise RuntimeError(f"Ligand did not fully sanitize: {sdf}")
    return mol


def _coords(mol: Chem.Mol) -> torch.Tensor:
    return torch.tensor(mol.GetConformer().GetPositions(), dtype=torch.float32)


def _fragment(mol: Chem.Mol) -> dict:
    result = decompose_fragments(mol, _coords(mol))
    if result is None:
        raise RuntimeError("Fragment decomposition failed")
    return result


def _cut_bond_indices(mol: Chem.Mol, frag: dict) -> list[int]:
    result: list[int] = []
    for i, j in frag["rot_bonds"]:
        bond = mol.GetBondBetweenAtoms(int(i), int(j))
        if bond is not None:
            result.append(bond.GetIdx())
    return result


def _draw_plain_molecule(
    mol_input: Chem.Mol,
    width: int,
    height: int,
    *,
    highlight_bonds: list[int] | None = None,
) -> Image.Image:
    """Draw a conventional 2D structure with optional red cut bonds."""
    mol = Chem.Mol(mol_input)
    rdDepictor.Compute2DCoords(mol)
    highlight_bonds = highlight_bonds or []

    drawer = rdMolDraw2D.MolDraw2DCairo(width, height)
    options = drawer.drawOptions()
    options.clearBackground = True
    options.fillHighlights = False
    options.highlightBondWidthMultiplier = 14
    options.bondLineWidth = 3.0
    options.padding = 0.08
    options.fixedFontSize = 25
    drawer.DrawMolecule(
        mol,
        highlightAtoms=[],
        highlightBonds=highlight_bonds,
        highlightBondColors={bond_idx: _rgb01(CUT_RED) for bond_idx in highlight_bonds},
    )
    drawer.FinishDrawing()
    return Image.open(io.BytesIO(drawer.GetDrawingText())).convert("RGB")


def _draw_fragment_regions(
    mol_input: Chem.Mol,
    fragment_ids: list[int],
    width: int,
    height: int,
    *,
    cut_bonds: list[int] | None = None,
) -> Image.Image:
    """Draw fragment regions with RDKit's native continuous highlighting."""
    mol = Chem.Mol(mol_input)
    rdDepictor.Compute2DCoords(mol)
    cut_set = set(cut_bonds or [])

    atom_colors = {
        atom_idx: _rgba01(FRAGMENT_COLORS[fragment_ids[atom_idx] % len(FRAGMENT_COLORS)], 0.30)
        for atom_idx in range(mol.GetNumAtoms())
    }
    bond_colors: dict[int, tuple[float, float, float, float]] = {}
    for bond in mol.GetBonds():
        bond_idx = bond.GetIdx()
        if bond_idx in cut_set:
            bond_colors[bond_idx] = _rgba01(CUT_RED, 0.95)
        else:
            begin = bond.GetBeginAtomIdx()
            end = bond.GetEndAtomIdx()
            if fragment_ids[begin] == fragment_ids[end]:
                color = FRAGMENT_COLORS[fragment_ids[begin] % len(FRAGMENT_COLORS)]
                bond_colors[bond_idx] = _rgba01(color, 0.30)

    drawer = rdMolDraw2D.MolDraw2DCairo(width, height)
    options = drawer.drawOptions()
    options.clearBackground = True
    options.fillHighlights = True
    options.continuousHighlight = True
    options.atomHighlightsAreCircles = False
    options.highlightBondWidthMultiplier = 14
    options.bondLineWidth = 3.0
    options.padding = 0.06
    options.fixedFontSize = 25
    options.useBWAtomPalette()
    drawer.DrawMolecule(
        mol,
        highlightAtoms=list(atom_colors),
        highlightBonds=list(bond_colors),
        highlightAtomColors=atom_colors,
        highlightBondColors=bond_colors,
        highlightAtomRadii={atom_idx: 0.34 for atom_idx in atom_colors},
    )
    cut_segments: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for bond_idx in cut_set:
        bond = mol.GetBondWithIdx(bond_idx)
        begin = drawer.GetDrawCoords(bond.GetBeginAtomIdx())
        end = drawer.GetDrawCoords(bond.GetEndAtomIdx())
        # Inset the overlay slightly so atom labels remain readable.
        p0 = (0.84 * begin.x + 0.16 * end.x, 0.84 * begin.y + 0.16 * end.y)
        p1 = (0.16 * begin.x + 0.84 * end.x, 0.16 * begin.y + 0.84 * end.y)
        cut_segments.append((p0, p1))
    drawer.FinishDrawing()
    image = Image.open(io.BytesIO(drawer.GetDrawingText())).convert("RGB")
    image_draw = ImageDraw.Draw(image)
    for p0, p1 in cut_segments:
        image_draw.line((p0, p1), fill=CUT_RED, width=10)
    return image


def _separate_fragments(mol: Chem.Mol, frag: dict) -> list[Chem.Mol]:
    """Remove the project-selected cut bonds and return ordered components."""
    editable = Chem.RWMol(Chem.Mol(mol))
    for i, j in frag["rot_bonds"]:
        editable.RemoveBond(int(i), int(j))
    disconnected = editable.GetMol()
    disconnected.UpdatePropertyCache(strict=False)

    atom_mappings: list[tuple[int, ...]] = []
    pieces = Chem.GetMolFrags(
        disconnected,
        asMols=True,
        sanitizeFrags=True,
        fragsMolAtomMapping=atom_mappings,
    )
    ordered = sorted(
        zip(pieces, atom_mappings),
        key=lambda item: int(frag["fragment_id"][item[1][0]].item()),
    )
    return [piece for piece, _ in ordered]


def _draw_molecule(mol_3d: Chem.Mol, frag: dict, width: int, height: int) -> Image.Image:
    """Draw a fragment-colored ligand for the 3D pocket panel."""
    mol = Chem.Mol(mol_3d)
    rdDepictor.Compute2DCoords(mol)

    atom_colors = {
        i: _rgb01(FRAGMENT_COLORS[int(frag["fragment_id"][i].item()) % len(FRAGMENT_COLORS)])
        for i in range(mol.GetNumAtoms())
    }
    atom_radii = {i: 0.34 for i in range(mol.GetNumAtoms())}
    cut_bonds: list[int] = []
    for i, j in frag["rot_bonds"]:
        bond = mol.GetBondBetweenAtoms(int(i), int(j))
        if bond is not None:
            cut_bonds.append(bond.GetIdx())

    drawer = rdMolDraw2D.MolDraw2DCairo(width, height)
    options = drawer.drawOptions()
    options.clearBackground = True
    options.fillHighlights = True
    options.continuousHighlight = False
    options.highlightBondWidthMultiplier = 18
    options.bondLineWidth = 3.0
    options.padding = 0.08
    options.fixedFontSize = 28
    drawer.DrawMolecule(
        mol,
        highlightAtoms=list(range(mol.GetNumAtoms())),
        highlightBonds=cut_bonds,
        highlightAtomColors=atom_colors,
        highlightBondColors={b: _rgb01(CUT_RED) for b in cut_bonds},
        highlightAtomRadii=atom_radii,
    )
    drawer.FinishDrawing()
    return Image.open(io.BytesIO(drawer.GetDrawingText())).convert("RGB")


def _rounded_panel(
    canvas: Image.Image,
    box: tuple[int, int, int, int],
    radius: int = 34,
    fill: str = PANEL_BG,
) -> None:
    ImageDraw.Draw(canvas).rounded_rectangle(box, radius=radius, fill=fill, outline="#E2E8F0", width=3)


def make_fragmentation_panel(output_dir: Path) -> tuple[Path, list[dict]]:
    canvas = Image.new("RGB", (2400, 1350), "white")
    draw = ImageDraw.Draw(canvas)
    mol = _load_astex_mol(FRAGMENT_CASE)
    frag = _fragment(mol)
    pieces = _separate_fragments(mol, frag)
    cut_bonds = _cut_bond_indices(mol, frag)

    fragment_ids = [int(x) for x in frag["fragment_id"].tolist()]
    ligand_image = _draw_fragment_regions(
        mol,
        fragment_ids,
        1070,
        760,
        cut_bonds=cut_bonds,
    )
    canvas.paste(ligand_image, (35, 285))

    draw.line((120, 180, 250, 180), fill=CUT_RED, width=12)
    draw.text((275, 180), "rotatable bond", font=_font(28, True), fill=CUT_RED, anchor="lm")
    arrow_y = 675
    draw.line((1110, arrow_y, 1310, arrow_y), fill=INK, width=9)
    draw.polygon(((1350, arrow_y), (1298, 643), (1298, 707)), fill=INK)

    slots = (
        (1370, 70, 1840, 430),
        (1880, 70, 2350, 430),
        (1370, 485, 1840, 845),
        (1880, 485, 2350, 845),
        (1370, 900, 1840, 1260),
        (1880, 900, 2350, 1260),
    )
    for idx, (piece, slot) in enumerate(zip(pieces, slots)):
        x0, y0, x1, y1 = slot
        fragment_image = _draw_fragment_regions(
            piece,
            [idx] * piece.GetNumAtoms(),
            x1 - x0,
            y1 - y0,
        )
        canvas.paste(fragment_image, (x0, y0))

    records = [
        {
            "case_id": FRAGMENT_CASE,
            "label": "STI / imatinib",
            "heavy_atoms": mol.GetNumAtoms(),
            "n_fragments": int(frag["n_frags"]),
            "fragment_sizes": [int(x) for x in frag["frag_sizes"].tolist()],
            "cut_bonds": [[int(a), int(b)] for a, b in frag["rot_bonds"]],
        }
    ]

    path = output_dir / "fragmentation_examples.png"
    canvas.save(path, optimize=True)
    return path, records


def _camera_basis(protein: np.ndarray, pocket_center: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centroid = protein.mean(axis=0)
    view = pocket_center - centroid
    view_norm = np.linalg.norm(view)
    if view_norm < 1e-8:
        view = np.array([0.0, 0.0, 1.0])
    else:
        view /= view_norm

    centered = protein - centroid
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    x_axis = None
    for candidate in vh:
        projected = candidate - np.dot(candidate, view) * view
        if np.linalg.norm(projected) > 1e-6:
            x_axis = projected / np.linalg.norm(projected)
            break
    if x_axis is None:
        trial = np.array([1.0, 0.0, 0.0])
        x_axis = trial - np.dot(trial, view) * view
        x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(view, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    return x_axis, y_axis, view


def _project(
    points: np.ndarray,
    focus: np.ndarray,
    basis: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    centered = points - focus
    x_axis, y_axis, view = basis
    xy = np.stack((centered @ x_axis, centered @ y_axis), axis=1)
    depth = centered @ view
    return xy, depth


def _screen_transform(
    xy: np.ndarray,
    box: tuple[int, int, int, int],
    padding: int,
    fixed_half_span: float | None = None,
) -> tuple[np.ndarray, float, tuple[float, float]]:
    x0, y0, x1, y1 = box
    width, height = x1 - x0 - 2 * padding, y1 - y0 - 2 * padding
    if fixed_half_span is None:
        lo, hi = xy.min(axis=0), xy.max(axis=0)
        span = np.maximum(hi - lo, 1e-6)
        scale = min(width / span[0], height / span[1])
        center = (lo + hi) / 2.0
    else:
        scale = min(width, height) / (2.0 * fixed_half_span)
        center = np.zeros(2)
    mapped = np.empty_like(xy)
    mapped[:, 0] = x0 + (x1 - x0) / 2.0 + (xy[:, 0] - center[0]) * scale
    mapped[:, 1] = y0 + (y1 - y0) / 2.0 - (xy[:, 1] - center[1]) * scale
    return mapped, float(scale), (float(center[0]), float(center[1]))


def _depth_color(depth: float, low: float, high: float) -> tuple[int, int, int, int]:
    t = 0.5 if high <= low else (depth - low) / (high - low)
    value = int(205 - 85 * t)
    return value, value + 5, min(255, value + 14), 210


def _draw_dashed_ellipse(
    draw: ImageDraw.ImageDraw,
    box: tuple[float, float, float, float],
    fill: str,
    width: int,
    dash_degrees: int = 10,
) -> None:
    for start in range(0, 360, dash_degrees * 2):
        draw.arc(box, start=start, end=start + dash_degrees, fill=fill, width=width)


def _draw_ligand_sticks(
    layer: Image.Image,
    mol: Chem.Mol,
    frag: dict,
    screen_xy: np.ndarray,
    depth: np.ndarray,
    atom_radius: int,
    bond_width: int,
) -> None:
    draw = ImageDraw.Draw(layer)
    frag_ids = frag["fragment_id"].tolist()
    cut_set = {tuple(sorted((int(a), int(b)))) for a, b in frag["rot_bonds"]}
    bonds = sorted(mol.GetBonds(), key=lambda b: float((depth[b.GetBeginAtomIdx()] + depth[b.GetEndAtomIdx()]) / 2))
    for bond in bonds:
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        pi, pj = screen_xy[i], screen_xy[j]
        mid = (pi + pj) / 2.0
        draw.line((tuple(pi), tuple(pj)), fill="#FFFFFF", width=bond_width + 8)
        if tuple(sorted((i, j))) in cut_set:
            draw.line((tuple(pi), tuple(pj)), fill=CUT_RED, width=bond_width + 3)
        else:
            draw.line((tuple(pi), tuple(mid)), fill=FRAGMENT_COLORS[frag_ids[i] % len(FRAGMENT_COLORS)], width=bond_width)
            draw.line((tuple(mid), tuple(pj)), fill=FRAGMENT_COLORS[frag_ids[j] % len(FRAGMENT_COLORS)], width=bond_width)
    order = np.argsort(depth)
    for i in order:
        x, y = screen_xy[i]
        color = FRAGMENT_COLORS[frag_ids[i] % len(FRAGMENT_COLORS)]
        r = atom_radius
        draw.ellipse((x - r - 3, y - r - 3, x + r + 3, y + r + 3), fill="white")
        draw.ellipse((x - r, y - r, x + r, y + r), fill=color, outline=INK, width=2)


def make_pocket_panel(output_dir: Path) -> tuple[Path, dict]:
    case_dir = ASTEX_ROOT / POCKET_CASE
    mol = _load_astex_mol(POCKET_CASE)
    frag = _fragment(mol)
    ligand_xyz = _coords(mol).numpy()
    protein_pdb = case_dir / f"{POCKET_CASE}_protein.pdb"
    protein = parse_pocket_atoms(protein_pdb)
    if protein is None:
        raise RuntimeError(f"Could not parse protein: {protein_pdb}")

    centers = json.loads(POCKET_CENTERS.read_text())
    pdb_id = POCKET_CASE[:4].lower()
    center_record = centers[pdb_id]
    pocket_center = np.asarray(center_record["center"], dtype=np.float32)
    center_t = torch.tensor(pocket_center, dtype=torch.float32)
    cropped = crop_to_pocket(protein, center_t, cutoff=POCKET_CUTOFF)
    if cropped is None:
        raise RuntimeError("Reference pocket crop is empty")

    pres = protein["pres_coords"].numpy()
    patom = protein["patom_coords"]
    atom_res = protein["patom_residue_id"]
    active_atoms = torch.linalg.vector_norm(patom - center_t, dim=1) <= POCKET_CUTOFF
    active_res_ids = set(int(x) for x in atom_res[active_atoms].unique().tolist())
    active_res = np.array([i in active_res_ids for i in range(pres.shape[0])])

    basis = _camera_basis(pres, pocket_center)
    full_xy, full_depth = _project(pres, pres.mean(axis=0), basis)
    center_xy_full, _ = _project(pocket_center[None, :], pres.mean(axis=0), basis)
    lig_xy_full, lig_depth_full = _project(ligand_xyz, pres.mean(axis=0), basis)

    canvas = Image.new("RGB", (3200, 1800), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((120, 68), "Docking happens inside an explicit pocket context", font=_font(80, True), fill=INK)
    draw.text(
        (124, 170),
        "Example: Astex 1T46 · STI/imatinib · reference-defined holo pocket",
        font=_font(40),
        fill=MUTED,
    )

    left = (100, 290, 1550, 1660)
    right = (1650, 290, 3100, 1660)
    _rounded_panel(canvas, left, radius=42)
    _rounded_panel(canvas, right, radius=42)
    draw.text((left[0] + 58, left[1] + 48), "A  Receptor overview", font=_font(44, True), fill=INK)
    draw.text((right[0] + 58, right[1] + 48), "B  Residue-aware pocket crop", font=_font(44, True), fill=INK)

    # Full receptor: a depth-shaded residue trace plus highlighted pocket residues.
    full_box = (left[0] + 60, left[1] + 150, left[2] - 60, left[3] - 150)
    combined_full = np.concatenate((full_xy, center_xy_full, lig_xy_full), axis=0)
    mapped_all, full_scale, _ = _screen_transform(combined_full, full_box, padding=35)
    mapped_pres = mapped_all[: len(full_xy)]
    mapped_center = mapped_all[len(full_xy)]
    mapped_lig = mapped_all[len(full_xy) + 1 :]

    layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    ldraw = ImageDraw.Draw(layer)
    pocket_r = POCKET_CUTOFF * full_scale
    ldraw.ellipse(
        (mapped_center[0] - pocket_r, mapped_center[1] - pocket_r, mapped_center[0] + pocket_r, mapped_center[1] + pocket_r),
        fill=(56, 189, 248, 34),
        outline=(14, 165, 233, 170),
        width=5,
    )
    lo_d, hi_d = float(full_depth.min()), float(full_depth.max())
    for i in range(len(pres) - 1):
        if np.linalg.norm(pres[i + 1] - pres[i]) > 5.6:
            continue
        color = _depth_color(float((full_depth[i] + full_depth[i + 1]) / 2), lo_d, hi_d)
        width = 13 if active_res[i] or active_res[i + 1] else 7
        if active_res[i] or active_res[i + 1]:
            color = (14, 165, 233, 225)
        ldraw.line((tuple(mapped_pres[i]), tuple(mapped_pres[i + 1])), fill=color, width=width)
    for i in np.where(active_res)[0]:
        x, y = mapped_pres[i]
        ldraw.ellipse((x - 10, y - 10, x + 10, y + 10), fill=(2, 132, 199, 235), outline="white", width=2)
    canvas = Image.alpha_composite(canvas.convert("RGBA"), layer)
    _draw_ligand_sticks(canvas, mol, frag, mapped_lig, lig_depth_full, atom_radius=10, bond_width=10)
    draw = ImageDraw.Draw(canvas)
    draw.line((mapped_center[0] - 18, mapped_center[1], mapped_center[0] + 18, mapped_center[1]), fill=INK, width=5)
    draw.line((mapped_center[0], mapped_center[1] - 18, mapped_center[0], mapped_center[1] + 18), fill=INK, width=5)
    draw.rounded_rectangle((left[0] + 80, left[3] - 115, left[2] - 80, left[3] - 38), radius=30, fill="#E0F2FE")
    draw.text(
        ((left[0] + left[2]) / 2, left[3] - 78),
        f"explicit center  +  {POCKET_CUTOFF:g} Å crop radius",
        anchor="mm",
        font=_font(32, True),
        fill="#075985",
    )

    # Zoomed pocket: protein heavy-atom bonds, fragment-colored ligand, and crop boundary.
    zoom_box = (right[0] + 75, right[1] + 155, right[2] - 75, right[3] - 155)
    crop_xyz = cropped["patom_coords"].numpy()
    crop_xy, crop_depth = _project(crop_xyz, pocket_center, basis)
    lig_xy_zoom, lig_depth_zoom = _project(ligand_xyz, pocket_center, basis)
    combined_zoom = np.concatenate((crop_xy, lig_xy_zoom, np.zeros((1, 2), dtype=np.float32)), axis=0)
    mapped_zoom, zoom_scale, _ = _screen_transform(combined_zoom, zoom_box, padding=30, fixed_half_span=11.5)
    mapped_crop = mapped_zoom[: len(crop_xy)]
    mapped_lig_zoom = mapped_zoom[len(crop_xy) : len(crop_xy) + len(lig_xy_zoom)]
    mapped_center_zoom = mapped_zoom[-1]

    zoom_layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    zdraw = ImageDraw.Draw(zoom_layer)
    zoom_r = POCKET_CUTOFF * zoom_scale
    _draw_dashed_ellipse(
        zdraw,
        (
            mapped_center_zoom[0] - zoom_r,
            mapped_center_zoom[1] - zoom_r,
            mapped_center_zoom[0] + zoom_r,
            mapped_center_zoom[1] + zoom_r,
        ),
        fill="#0EA5E9",
        width=5,
    )
    pbonds = cropped["pbond_index"].numpy()
    undirected = [(int(i), int(j)) for i, j in pbonds.T if int(i) < int(j)]
    undirected.sort(key=lambda ij: float((crop_depth[ij[0]] + crop_depth[ij[1]]) / 2))
    for i, j in undirected:
        zdraw.line((tuple(mapped_crop[i]), tuple(mapped_crop[j])), fill=(125, 162, 183, 140), width=4)
    order = np.argsort(crop_depth)
    d_lo, d_hi = float(crop_depth.min()), float(crop_depth.max())
    for idx in order:
        t = 0.5 if d_hi <= d_lo else float((crop_depth[idx] - d_lo) / (d_hi - d_lo))
        r = int(4 + 3 * t)
        x, y = mapped_crop[idx]
        zdraw.ellipse((x - r, y - r, x + r, y + r), fill=(56, 146, 190, int(105 + 105 * t)))
    canvas = Image.alpha_composite(canvas, zoom_layer)
    _draw_ligand_sticks(canvas, mol, frag, mapped_lig_zoom, lig_depth_zoom, atom_radius=14, bond_width=13)
    draw = ImageDraw.Draw(canvas)
    cx, cy = mapped_center_zoom
    draw.ellipse((cx - 15, cy - 15, cx + 15, cy + 15), fill="white", outline=INK, width=5)
    draw.line((cx - 28, cy, cx + 28, cy), fill=INK, width=5)
    draw.line((cx, cy - 28, cx, cy + 28), fill=INK, width=5)
    draw.rounded_rectangle((right[0] + 80, right[3] - 115, right[2] - 80, right[3] - 38), radius=30, fill="#ECFDF5")
    draw.text(
        ((right[0] + right[2]) / 2, right[3] - 78),
        f"{cropped['pres_coords'].shape[0]} residues · {mol.GetNumAtoms()} ligand atoms · {frag['n_frags']} rigid fragments",
        anchor="mm",
        font=_font(32, True),
        fill="#065F46",
    )

    # Compact legend.
    legend_y = 255
    draw.line((2210, legend_y, 2310, legend_y), fill=PROTEIN_GRAY, width=9)
    draw.text((2330, legend_y), "receptor", anchor="lm", font=_font(29, True), fill=MUTED)
    draw.ellipse((2550, legend_y - 17, 2584, legend_y + 17), fill=POCKET_BLUE)
    draw.text((2600, legend_y), "pocket", anchor="lm", font=_font(29, True), fill="#0284C7")
    draw.line((2780, legend_y, 2860, legend_y), fill=CUT_RED, width=10)
    draw.text((2880, legend_y), "fragment boundary", anchor="lm", font=_font(29, True), fill=CUT_RED)

    path = output_dir / "protein_pocket_context.png"
    canvas.convert("RGB").save(path, optimize=True)
    record = {
        "case_id": POCKET_CASE,
        "pdb_id": pdb_id,
        "ligand_label": "STI / imatinib",
        "pocket_definition": center_record["definition"],
        "pocket_center_angstrom": [float(x) for x in pocket_center],
        "pocket_cutoff_angstrom": POCKET_CUTOFF,
        "full_protein_atoms": int(protein["patom_coords"].shape[0]),
        "full_protein_residues": int(protein["pres_coords"].shape[0]),
        "pocket_atoms": int(cropped["patom_coords"].shape[0]),
        "pocket_residues": int(cropped["pres_coords"].shape[0]),
        "ligand_atoms": mol.GetNumAtoms(),
        "n_fragments": int(frag["n_frags"]),
        "fragment_sizes": [int(x) for x in frag["frag_sizes"].tolist()],
    }
    return path, record


def _existing_paths(paths: Iterable[Path]) -> list[str]:
    return [str(path.relative_to(ROOT)) for path in paths if path.exists()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    fragmentation_path, fragmentation_records = make_fragmentation_panel(output_dir)
    pocket_path, pocket_record = make_pocket_panel(output_dir)
    manifest = {
        "figure_set": "effdock_intro_visuals_v1",
        "fragmentation_implementation": "src/effdock/preprocess/fragments.py::decompose_fragments",
        "software": {"rdkit": rdkit.__version__, "torch": torch.__version__},
        "coordinate_unit": "angstrom",
        "sources": {
            "dataset": "Astex Diverse Set (local frozen benchmark snapshot)",
            "pocket_centers": str(POCKET_CENTERS.relative_to(ROOT)),
        },
        "figures": _existing_paths((fragmentation_path, pocket_path)),
        "fragmentation_examples": fragmentation_records,
        "protein_pocket_example": pocket_record,
        "scientific_boundary": (
            "The pocket panel is a reference-defined holo visualization. It illustrates the explicit "
            "pocket-conditioning interface and is not evidence of blind pocket detection."
        ),
    }
    (output_dir / "intro_visuals_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(f"Wrote {fragmentation_path}")
    print(f"Wrote {pocket_path}")
    print(f"Wrote {output_dir / 'intro_visuals_manifest.json'}")


if __name__ == "__main__":
    main()
