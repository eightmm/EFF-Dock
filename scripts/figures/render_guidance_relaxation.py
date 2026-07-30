#!/usr/bin/env python3
"""Render guidance-only fragment relaxation artifacts without external runtimes."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw, ImageFont
from torch import Tensor

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
GRID = "#DCE3EA"
PANEL = "#FFFFFF"
BACKGROUND = "#F8FAFC"
PROTEIN = "#CBD5E1"
CRYSTAL = "#7B8794"
CUT_BOND = "#D64B4B"
PHYSICAL_COLOR = "#E69F00"
UNIFIED_COLOR = "#0072B2"
INTERACTION_COLOR = "#009E73"
POCKET_CENTER_COLOR = "#7A68A6"
INLINE_TEMPLATE_PATH = Path(__file__).with_name("guidance_relaxation_inline.html")


MetricPath = tuple[str, ...]

METRIC_PATHS: dict[str, tuple[MetricPath, ...]] = {
    "total": (
        ("energy_groups", "combined"),
        ("energies", "total"),
        ("energy", "combined"),
        ("energy", "total"),
        ("combined_energy",),
        ("total_energy",),
        ("total",),
    ),
    "physical": (
        ("energy_groups", "physical"),
        ("energies", "physical"),
        ("energy", "physical"),
        ("physical_energy",),
        ("physical",),
    ),
    "interaction": (
        ("energy_groups", "interaction"),
        ("energies", "interaction"),
        ("energy", "interaction"),
        ("interaction_energy",),
        ("interaction",),
    ),
    "screened_charge": (
        ("energies", "interaction_screened_formal_charge"),
        ("energy", "interaction_screened_formal_charge"),
        ("interaction_screened_formal_charge",),
    ),
    "rmsd": (
        ("raw_rmsd_angstrom",),
        ("rmsd_to_crystal_angstrom",),
        ("rmsd_angstrom",),
        ("rmsd",),
    ),
    "cut_bond_error": (
        ("cut_bonds", "max_abs_error_angstrom"),
        ("cut_bond", "max_abs_error_angstrom"),
        ("cut_bond_max_abs_error_angstrom",),
        ("cut_bond_max_error_angstrom",),
        ("max_cut_bond_error_angstrom",),
        ("max_cut_bond_error",),
    ),
    "step": (
        ("saved_step",),
        ("accepted_step",),
        ("step",),
    ),
}


@dataclass
class RunData:
    key: str
    label: str
    frames: Tensor
    saved_steps: list[int]
    metrics: dict[str, list[float | None]]


@dataclass
class RenderData:
    case_id: str
    crystal_coords: Tensor
    protein_coords: Tensor
    pocket_center: Tensor | None
    fragment_id: Tensor
    bonds: list[tuple[int, int]]
    initialization_label: str
    runs: dict[str, RunData]


@dataclass(frozen=True)
class Camera:
    focus: Tensor
    basis: Tensor
    half_span: float


@dataclass(frozen=True)
class ChartSeries:
    label: str
    x: list[int]
    y: list[float | None]
    color: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path, help="Relaxation summary JSON.")
    parser.add_argument("trajectory", type=Path, help="Relaxation trajectory PT bundle.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: summary JSON directory).",
    )
    parser.add_argument(
        "--inline-html",
        type=Path,
        default=None,
        help="Optionally write a dependency-free inline HTML fragment.",
    )
    parser.add_argument("--max-gif-frames", type=int, default=72)
    return parser.parse_args()


def _as_coords(value: Any, name: str) -> Tensor:
    coords = torch.as_tensor(value, dtype=torch.float64).detach().cpu()
    if coords.ndim != 2 or coords.shape[-1] != 3:
        raise ValueError(f"{name} must have shape [N,3], got {tuple(coords.shape)}")
    if not bool(torch.isfinite(coords).all()):
        raise ValueError(f"{name} contains non-finite coordinates")
    return coords


def _as_frames(value: Any, name: str) -> Tensor:
    frames = torch.as_tensor(value, dtype=torch.float64).detach().cpu()
    if frames.ndim == 2:
        frames = frames.unsqueeze(0)
    if frames.ndim != 3 or frames.shape[-1] != 3:
        raise ValueError(f"{name} must have shape [T,N,3], got {tuple(frames.shape)}")
    if not bool(torch.isfinite(frames).all()):
        raise ValueError(f"{name} contains non-finite coordinates")
    if frames.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one frame")
    return frames


def _as_optional_point(value: Any, name: str) -> Tensor | None:
    if value is None:
        return None
    point = torch.as_tensor(value, dtype=torch.float64).detach().cpu().reshape(-1)
    if point.numel() != 3:
        raise ValueError(f"{name} must contain three coordinates, got {point.numel()}")
    if not bool(torch.isfinite(point).all()):
        raise ValueError(f"{name} contains non-finite coordinates")
    return point


def _normalise_bonds(value: Any, n_atoms: int) -> list[tuple[int, int]]:
    if value is None:
        return []
    rows: list[list[int | float]]
    if isinstance(value, Tensor):
        tensor = value.detach().cpu()
        if tensor.ndim != 2:
            raise ValueError("bonds tensor must be two-dimensional")
        if tensor.shape[0] in (2, 3) and tensor.shape[1] > tensor.shape[0]:
            tensor = tensor.T
        rows = tensor.tolist()
    else:
        rows = [list(row) for row in value]
    bonds: list[tuple[int, int]] = []
    for row in rows:
        if len(row) < 2:
            raise ValueError("every bond row must contain at least two atom indices")
        atom_i, atom_j = int(row[0]), int(row[1])
        if not 0 <= atom_i < n_atoms or not 0 <= atom_j < n_atoms:
            raise ValueError(f"bond index out of range: {(atom_i, atom_j)}")
        if atom_i != atom_j:
            bonds.append((atom_i, atom_j))
    return bonds


def _pick_key(mapping: Mapping[str, Any], aliases: tuple[str, ...]) -> str:
    for alias in aliases:
        if alias in mapping:
            return alias
    for key in mapping:
        lowered = key.lower().replace("-", "_")
        if any(alias in lowered for alias in aliases):
            return key
    raise KeyError(f"none of {aliases!r} found in run keys {tuple(mapping)}")


def _nested_get(value: Any, path: MetricPath) -> Any:
    current = value
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, Tensor):
        if value.numel() != 1:
            return None
        value = value.detach().cpu().item()
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _humanize_initialization(value: Any) -> str | None:
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    lowered = raw.lower().replace("_", " ").replace("-", " ")
    if "tear" in lowered:
        return "Local fragment tear"
    if "pocket" in lowered and any(token in lowered for token in ("center", "gaussian", "prior")):
        return "Pocket-centered fragment prior"
    if "ode" in lowered:
        return "ODE fragment prior"
    words = " ".join(raw.replace("_", " ").replace("-", " ").split())
    return words[0].upper() + words[1:] if words else None


def _initialization_label(summary: Mapping[str, Any]) -> str:
    initialization = summary.get("initialization")
    if not isinstance(initialization, Mapping):
        return "Initialization not recorded"

    kind = _humanize_initialization(initialization.get("kind"))
    mode = _humanize_initialization(initialization.get("mode"))
    parts = [kind or mode or "Initialization"]
    if mode is not None and mode.casefold() != parts[0].casefold():
        parts.append(mode)

    sigma = next(
        (
            number
            for key in (
                "sigma",
                "std",
                "standard_deviation",
                "translation_sigma",
                "sigma_angstrom",
            )
            if (number := _to_float(initialization.get(key))) is not None
        ),
        None,
    )
    if sigma is not None:
        parts.append(f"σ={sigma:g} Å")
    elif kind is not None and "tear" in kind.casefold():
        maximum = _to_float(
            initialization.get(
                "maximum_fragment_displacement_angstrom",
                initialization.get("fragment_displacement_angstrom"),
            )
        )
        if maximum is not None:
            parts.append(f"max {maximum:g} Å")
    return " · ".join(parts)


def _sequence_values(value: Any) -> list[float | None] | None:
    if isinstance(value, Tensor):
        if value.ndim == 0:
            return [_to_float(value)]
        return [_to_float(item) for item in value.detach().cpu().flatten()]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        numbers = [_to_float(item) for item in value]
        if any(number is not None for number in numbers):
            return numbers
    return None


def _mapping_rows(metrics: Mapping[str, Any]) -> list[Mapping[str, Any]] | None:
    rows = metrics.get("rows")
    if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)):
        result = [row for row in rows if isinstance(row, Mapping)]
        return result or None
    if metrics and all(isinstance(value, Mapping) for value in metrics.values()):
        keys = list(metrics)
        if all(str(key).lstrip("-").isdigit() for key in keys):
            keys.sort(key=lambda key: int(str(key)))
            return [metrics[key] for key in keys]
    return None


def _raw_metric_series(metrics: Any, paths: tuple[MetricPath, ...]) -> list[float | None]:
    if isinstance(metrics, Mapping):
        rows = _mapping_rows(metrics)
        if rows is not None:
            metrics = rows
        else:
            for path in paths:
                values = _sequence_values(_nested_get(metrics, path))
                if values is not None:
                    return values
            return []
    if isinstance(metrics, Sequence) and not isinstance(metrics, (str, bytes)):
        result: list[float | None] = []
        for row in metrics:
            if not isinstance(row, Mapping):
                result.append(None)
                continue
            result.append(
                next(
                    (
                        number
                        for path in paths
                        if (number := _to_float(_nested_get(row, path))) is not None
                    ),
                    None,
                )
            )
        return result
    return []


def _align_series(
    values: list[float | None],
    n_frames: int,
    saved_steps: list[int],
) -> list[float | None]:
    if not values:
        return [None] * n_frames
    if len(values) == n_frames:
        return values
    if (
        len(saved_steps) == n_frames
        and saved_steps
        and min(saved_steps) >= 0
        and max(saved_steps) < len(values)
    ):
        return [values[step] for step in saved_steps]
    if n_frames == 1:
        return [values[-1]]
    indices = [round(index * (len(values) - 1) / max(n_frames - 1, 1)) for index in range(n_frames)]
    return [values[index] for index in indices]


def _saved_steps(
    bundle: Mapping[str, Any],
    run_bundle: Mapping[str, Any],
    summary: Mapping[str, Any],
    run_summary: Mapping[str, Any],
    metrics: Any,
    mode_key: str,
    n_frames: int,
) -> list[int]:
    candidates = (
        run_bundle.get("saved_steps"),
        bundle.get("saved_steps"),
        run_summary.get("saved_steps"),
        summary.get("saved_steps"),
    )
    for candidate in candidates:
        if isinstance(candidate, Mapping):
            candidate = candidate.get(mode_key)
        values = _sequence_values(candidate)
        if values is None:
            continue
        steps = [int(value) for value in values if value is not None]
        if len(steps) == n_frames:
            return steps
    metric_steps = _raw_metric_series(metrics, METRIC_PATHS["step"])
    if len(metric_steps) == n_frames and all(value is not None for value in metric_steps):
        return [int(value) for value in metric_steps if value is not None]
    return list(range(n_frames))


def _raw_rmsd(frames: Tensor, crystal: Tensor) -> list[float]:
    return (frames - crystal.unsqueeze(0)).square().sum(dim=-1).mean(dim=-1).sqrt().tolist()


def _cut_bond_error(
    frames: Tensor,
    crystal: Tensor,
    fragment_id: Tensor,
    bonds: list[tuple[int, int]],
) -> list[float]:
    cut_bonds = [
        (atom_i, atom_j)
        for atom_i, atom_j in bonds
        if int(fragment_id[atom_i]) != int(fragment_id[atom_j])
    ]
    if not cut_bonds:
        return [0.0] * frames.shape[0]
    index_i = torch.tensor([bond[0] for bond in cut_bonds], dtype=torch.long)
    index_j = torch.tensor([bond[1] for bond in cut_bonds], dtype=torch.long)
    reference = (crystal[index_i] - crystal[index_j]).norm(dim=-1)
    observed = (frames[:, index_i] - frames[:, index_j]).norm(dim=-1)
    return (observed - reference).abs().amax(dim=-1).tolist()


def _load_run(
    mode: str,
    key: str,
    label: str,
    bundle: Mapping[str, Any],
    summary: Mapping[str, Any],
    crystal: Tensor,
    fragment_id: Tensor,
    bonds: list[tuple[int, int]],
) -> RunData:
    bundle_runs = bundle["runs"]
    run_bundle = bundle_runs[key]
    if not isinstance(run_bundle, Mapping):
        raise ValueError(f"trajectory run {key!r} must be a mapping")
    frames = _as_frames(run_bundle["frames"], f"runs[{key!r}].frames")
    if frames.shape[1:] != crystal.shape:
        raise ValueError(
            f"runs[{key!r}].frames atom shape {tuple(frames.shape[1:])} "
            f"does not match crystal {tuple(crystal.shape)}"
        )

    summary_runs = summary.get("runs", {})
    run_summary: Mapping[str, Any] = {}
    if isinstance(summary_runs, Mapping):
        summary_key = key if key in summary_runs else None
        if summary_key is None:
            aliases = (mode, "physical_only") if mode == "physical" else (mode, "full", "guidance")
            try:
                summary_key = _pick_key(summary_runs, aliases)
            except KeyError:
                summary_key = None
        if summary_key is not None and isinstance(summary_runs[summary_key], Mapping):
            run_summary = summary_runs[summary_key]
    metrics = run_summary.get("metrics", run_bundle.get("metrics", []))
    steps = _saved_steps(
        bundle,
        run_bundle,
        summary,
        run_summary,
        metrics,
        key,
        frames.shape[0],
    )
    extracted = {
        name: _align_series(
            _raw_metric_series(metrics, paths),
            frames.shape[0],
            steps,
        )
        for name, paths in METRIC_PATHS.items()
        if name != "step"
    }
    if not any(value is not None for value in extracted["rmsd"]):
        extracted["rmsd"] = _raw_rmsd(frames, crystal)
    if not any(value is not None for value in extracted["cut_bond_error"]):
        extracted["cut_bond_error"] = _cut_bond_error(
            frames,
            crystal,
            fragment_id,
            bonds,
        )
    if mode == "physical" and not any(value is not None for value in extracted["total"]):
        extracted["total"] = extracted["physical"]
    return RunData(key=key, label=label, frames=frames, saved_steps=steps, metrics=extracted)


def load_render_data(summary_path: Path, trajectory_path: Path) -> RenderData:
    summary = json.loads(summary_path.read_text())
    bundle = torch.load(trajectory_path, map_location="cpu", weights_only=False)
    if not isinstance(summary, Mapping) or not isinstance(bundle, Mapping):
        raise ValueError("summary and trajectory roots must be mappings")
    if not isinstance(bundle.get("runs"), Mapping):
        raise ValueError("trajectory bundle must contain a runs mapping")

    crystal = _as_coords(bundle["crystal_coords"], "crystal_coords")
    protein = _as_coords(bundle.get("protein_coords", torch.empty(0, 3)), "protein_coords")
    pocket_center = _as_optional_point(bundle.get("pocket_center"), "pocket_center")
    fragment_id = torch.as_tensor(bundle["fragment_id"], dtype=torch.long).detach().cpu().view(-1)
    if fragment_id.numel() != crystal.shape[0]:
        raise ValueError("fragment_id length must match crystal atom count")
    if fragment_id.numel() == 0 or int(fragment_id.min()) < 0:
        raise ValueError("fragment_id must contain non-negative fragment indices")
    bonds = _normalise_bonds(bundle.get("bonds"), crystal.shape[0])

    bundle_runs = bundle["runs"]
    physical_key = _pick_key(bundle_runs, ("physical", "physical_only"))
    unified_key = _pick_key(bundle_runs, ("unified", "full", "guidance"))
    runs = {
        "physical": _load_run(
            "physical",
            physical_key,
            "Physical only",
            bundle,
            summary,
            crystal,
            fragment_id,
            bonds,
        ),
        "unified": _load_run(
            "unified",
            unified_key,
            "Unified guidance",
            bundle,
            summary,
            crystal,
            fragment_id,
            bonds,
        ),
    }
    case_id = str(
        summary.get(
            "case_id",
            summary.get("complex_id", summary.get("sample_id", trajectory_path.parent.name)),
        )
    )
    return RenderData(
        case_id=case_id,
        crystal_coords=crystal,
        protein_coords=protein,
        pocket_center=pocket_center,
        fragment_id=fragment_id,
        bonds=bonds,
        initialization_label=_initialization_label(summary),
        runs=runs,
    )


def _normalise(vector: Tensor, fallback: Tensor) -> Tensor:
    norm = vector.norm()
    return vector / norm if float(norm) > 1e-10 else fallback.clone()


def _camera(data: RenderData) -> Camera:
    focus = data.crystal_coords.mean(dim=0)
    if data.protein_coords.numel():
        protein_centroid = data.protein_coords.mean(dim=0)
        view = _normalise(focus - protein_centroid, focus.new_tensor([0.0, 0.0, 1.0]))
        source = data.protein_coords - protein_centroid
    else:
        view = focus.new_tensor([0.0, 0.0, 1.0])
        source = data.crystal_coords - focus
    try:
        _, _, vh = torch.linalg.svd(source, full_matrices=False)
        candidates = list(vh)
    except RuntimeError:
        candidates = []
    candidates.extend(
        [
            focus.new_tensor([1.0, 0.0, 0.0]),
            focus.new_tensor([0.0, 1.0, 0.0]),
        ]
    )
    x_axis = candidates[-1]
    for candidate in candidates:
        projected = candidate - torch.dot(candidate, view) * view
        if float(projected.norm()) > 1e-8:
            x_axis = projected / projected.norm()
            break
    y_axis = _normalise(
        torch.linalg.cross(view, x_axis),
        focus.new_tensor([0.0, 1.0, 0.0]),
    )
    basis = torch.stack((x_axis, y_axis, view))
    all_ligand = [data.crystal_coords]
    all_ligand.extend(run.frames.reshape(-1, 3) for run in data.runs.values())
    if data.pocket_center is not None:
        all_ligand.append(data.pocket_center.unsqueeze(0))
    camera_coords = _to_camera(torch.cat(all_ligand), focus, basis)
    half_span = max(4.0, float(camera_coords.abs().amax()) + 1.5)
    return Camera(focus=focus, basis=basis, half_span=half_span)


def _to_camera(points: Tensor, focus: Tensor, basis: Tensor) -> Tensor:
    return (points - focus) @ basis.T


def _fragment_centers(frames: Tensor, fragment_id: Tensor) -> Tensor:
    n_fragments = int(fragment_id.max()) + 1
    centers = frames.new_zeros(frames.shape[0], n_fragments, 3)
    centers.index_add_(1, fragment_id, frames)
    counts = torch.bincount(fragment_id, minlength=n_fragments).to(frames.dtype)
    return centers / counts.clamp_min(1).view(1, -1, 1)


def _sample_protein(points: Tensor, focus: Tensor, max_points: int = 700) -> Tensor:
    if not points.numel():
        return points
    distance = (points - focus).norm(dim=-1)
    selected = points[distance <= 12.0]
    if not selected.numel():
        selected = points[torch.argsort(distance)[:max_points]]
    if selected.shape[0] > max_points:
        stride = math.ceil(selected.shape[0] / max_points)
        selected = selected[::stride][:max_points]
    return selected


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    names = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    )
    for name in names:
        if Path(name).exists():
            return ImageFont.truetype(name, size=size)
    return ImageFont.load_default()


def _map_points(
    points: Tensor,
    box: tuple[int, int, int, int],
    half_span: float,
) -> list[tuple[float, float, float]]:
    x0, y0, x1, y1 = box
    width, height = x1 - x0, y1 - y0
    scale = min(width, height) / (2.0 * half_span)
    center_x, center_y = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    return [
        (
            center_x + float(point[0]) * scale,
            center_y - float(point[1]) * scale,
            float(point[2]),
        )
        for point in points
    ]


def _draw_dashed_line(
    draw: ImageDraw.ImageDraw,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    fill: str,
    width: int,
    dash: float = 9.0,
) -> None:
    dx, dy = end[0] - start[0], end[1] - start[1]
    length = math.hypot(dx, dy)
    if length < 1e-8:
        return
    ux, uy = dx / length, dy / length
    position = 0.0
    while position < length:
        stop = min(position + dash, length)
        draw.line(
            (
                start[0] + ux * position,
                start[1] + uy * position,
                start[0] + ux * stop,
                start[1] + uy * stop,
            ),
            fill=fill,
            width=width,
        )
        position += 2.0 * dash


def _draw_scene(
    image: Image.Image,
    box: tuple[int, int, int, int],
    *,
    frame: Tensor,
    crystal: Tensor,
    protein: Tensor,
    pocket_center: Tensor | None,
    center_path: Tensor,
    fragment_id: Tensor,
    bonds: list[tuple[int, int]],
    half_span: float,
) -> None:
    draw = ImageDraw.Draw(image)
    protein_screen = _map_points(protein, box, half_span)
    for x, y, _ in sorted(protein_screen, key=lambda point: point[2]):
        if box[0] <= x <= box[2] and box[1] <= y <= box[3]:
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=PROTEIN)

    if pocket_center is not None:
        x, y, _ = _map_points(pocket_center.unsqueeze(0), box, half_span)[0]
        draw.ellipse(
            (x - 11, y - 11, x + 11, y + 11),
            fill=BACKGROUND,
            outline=POCKET_CENTER_COLOR,
            width=3,
        )
        draw.line((x - 15, y, x + 15, y), fill=POCKET_CENTER_COLOR, width=2)
        draw.line((x, y - 15, x, y + 15), fill=POCKET_CENTER_COLOR, width=2)
        draw.polygon(
            ((x, y - 5), (x + 5, y), (x, y + 5), (x - 5, y)),
            fill=POCKET_CENTER_COLOR,
        )
        label = "pocket center"
        label_font = _font(13)
        label_width = draw.textlength(label, font=label_font)
        label_x = min(max(x + 17, box[0] + 4), box[2] - label_width - 4)
        label_y = min(max(y - 9, box[1] + 4), box[3] - 20)
        draw.text((label_x, label_y), label, fill=INK, font=label_font)

    crystal_screen = _map_points(crystal, box, half_span)
    for atom_i, atom_j in bonds:
        pi, pj = crystal_screen[atom_i], crystal_screen[atom_j]
        draw.line((pi[0], pi[1], pj[0], pj[1]), fill=CRYSTAL, width=2)
    for x, y, _ in sorted(crystal_screen, key=lambda point: point[2]):
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=BACKGROUND, outline=CRYSTAL, width=2)

    if center_path.numel():
        for fragment in range(center_path.shape[1]):
            mapped = _map_points(center_path[:, fragment], box, half_span)
            color = FRAGMENT_COLORS[fragment % len(FRAGMENT_COLORS)]
            if len(mapped) > 1:
                draw.line(
                    [(point[0], point[1]) for point in mapped],
                    fill=color,
                    width=3,
                )
            start = mapped[0]
            current = mapped[-1]
            draw.rectangle(
                (start[0] - 5, start[1] - 5, start[0] + 5, start[1] + 5),
                fill=BACKGROUND,
                outline=color,
                width=2,
            )
            draw.ellipse(
                (current[0] - 5, current[1] - 5, current[0] + 5, current[1] + 5),
                fill=color,
                outline=INK,
                width=1,
            )

    frame_screen = _map_points(frame, box, half_span)
    ordered_bonds = sorted(
        bonds,
        key=lambda bond: (frame_screen[bond[0]][2] + frame_screen[bond[1]][2]) / 2.0,
    )
    for atom_i, atom_j in ordered_bonds:
        pi, pj = frame_screen[atom_i], frame_screen[atom_j]
        if int(fragment_id[atom_i]) != int(fragment_id[atom_j]):
            _draw_dashed_line(
                draw,
                (pi[0], pi[1]),
                (pj[0], pj[1]),
                fill=CUT_BOND,
                width=4,
            )
        else:
            midpoint = ((pi[0] + pj[0]) / 2.0, (pi[1] + pj[1]) / 2.0)
            left = FRAGMENT_COLORS[int(fragment_id[atom_i]) % len(FRAGMENT_COLORS)]
            right = FRAGMENT_COLORS[int(fragment_id[atom_j]) % len(FRAGMENT_COLORS)]
            draw.line((pi[0], pi[1], *midpoint), fill=left, width=5)
            draw.line((*midpoint, pj[0], pj[1]), fill=right, width=5)
    for atom in sorted(range(len(frame_screen)), key=lambda index: frame_screen[index][2]):
        x, y, _ = frame_screen[atom]
        color = FRAGMENT_COLORS[int(fragment_id[atom]) % len(FRAGMENT_COLORS)]
        draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=BACKGROUND)
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color, outline=INK, width=1)

    scale = min(box[2] - box[0], box[3] - box[1]) / (2.0 * half_span)
    bar_pixels = 2.0 * scale
    bar_y = box[3] - 18
    bar_x = box[2] - 24 - bar_pixels
    draw.line((bar_x, bar_y, bar_x + bar_pixels, bar_y), fill=INK, width=3)
    draw.text(
        (bar_x, bar_y - 22),
        "2 Å",
        fill=MUTED,
        font=_font(15),
    )


def _format_number(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return "n/a"
    absolute = abs(value)
    if absolute >= 10_000 or (0 < absolute < 0.001):
        return f"{value:.2e}"
    if absolute >= 100:
        return f"{value:.1f}"
    return f"{value:.3f}"


def _draw_panel(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title: str,
    subtitle: str = "",
) -> tuple[int, int, int, int]:
    draw.rounded_rectangle(box, radius=18, fill=PANEL, outline=GRID, width=2)
    draw.text((box[0] + 22, box[1] + 16), title, fill=INK, font=_font(25, bold=True))
    if subtitle:
        draw.text((box[0] + 22, box[1] + 48), subtitle, fill=MUTED, font=_font(17))
    top = box[1] + (78 if subtitle else 60)
    return box[0] + 18, top, box[2] - 18, box[3] - 18


def _finite_chart_values(series: list[ChartSeries]) -> list[tuple[int, float]]:
    return [
        (x_value, y_value)
        for item in series
        for x_value, y_raw in zip(item.x, item.y, strict=False)
        if (y_value := _to_float(y_raw)) is not None
    ]


def _draw_line_chart(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    series: list[ChartSeries],
    *,
    title: str,
    unit: str,
) -> None:
    x0, y0, x1, y1 = box
    draw.text((x0, y0), title, fill=INK, font=_font(18, bold=True))
    legend_x = x0
    legend_y = y0 + 34
    for item in series:
        draw.line((legend_x, legend_y, legend_x + 20, legend_y), fill=item.color, width=4)
        draw.text(
            (legend_x + 27, legend_y),
            item.label,
            fill=INK,
            font=_font(13),
            anchor="lm",
        )
        legend_x += 35 + int(draw.textlength(item.label, font=_font(13)))
    plot = (x0 + 72, y0 + 58, x1 - 14, y1 - 28)
    values = _finite_chart_values(series)
    if not values:
        draw.text((plot[0], plot[1] + 20), "Metric not recorded", fill=MUTED, font=_font(16))
        return
    x_values = [value[0] for value in values]
    y_values = [value[1] for value in values]
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)
    if x_min == x_max:
        x_max = x_min + 1
    if y_min == y_max:
        pad = max(abs(y_min) * 0.05, 1.0)
        y_min, y_max = y_min - pad, y_max + pad
    else:
        pad = 0.06 * (y_max - y_min)
        y_min, y_max = y_min - pad, y_max + pad

    for tick in range(4):
        fraction = tick / 3.0
        y = plot[3] - fraction * (plot[3] - plot[1])
        value = y_min + fraction * (y_max - y_min)
        draw.line((plot[0], y, plot[2], y), fill=GRID, width=1)
        label = _format_number(value)
        draw.text(
            (plot[0] - 8, y),
            label,
            fill=MUTED,
            font=_font(13),
            anchor="rm",
        )
    draw.line((plot[0], plot[1], plot[0], plot[3]), fill=MUTED, width=1)
    draw.line((plot[0], plot[3], plot[2], plot[3]), fill=MUTED, width=1)

    for item in series:
        segments: list[list[tuple[float, float]]] = [[]]
        for x_value, y_raw in zip(item.x, item.y, strict=False):
            y_value = _to_float(y_raw)
            if y_value is None:
                if segments[-1]:
                    segments.append([])
                continue
            x = plot[0] + (x_value - x_min) / (x_max - x_min) * (plot[2] - plot[0])
            y = plot[3] - (y_value - y_min) / (y_max - y_min) * (plot[3] - plot[1])
            segments[-1].append((x, y))
        for points in segments:
            if len(points) >= 2:
                draw.line(points, fill=item.color, width=4, joint="curve")
            elif points:
                x, y = points[0]
                draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=item.color)

    draw.text((plot[0], plot[3] + 8), f"saved step · {unit}", fill=MUTED, font=_font(12))


def _metric_at(run: RunData, name: str, index: int) -> float | None:
    values = run.metrics.get(name, [])
    return values[index] if index < len(values) else None


def _sample_frame_indices(frames: Tensor, maximum: int) -> list[int]:
    if maximum < 2:
        raise ValueError("--max-gif-frames must be at least 2")
    n_frames = frames.shape[0]
    if n_frames <= maximum:
        return list(range(n_frames))
    motion = (frames[1:] - frames[:-1]).square().sum(dim=-1).mean(dim=-1).sqrt()
    cumulative = torch.cat((motion.new_zeros(1), motion.cumsum(dim=0)))
    total = float(cumulative[-1])
    if total <= 1e-12:
        indices = torch.linspace(0, n_frames - 1, maximum).round().to(torch.long)
    else:
        targets = torch.linspace(0.0, total, maximum, dtype=cumulative.dtype)
        indices = torch.searchsorted(cumulative, targets).clamp(max=n_frames - 1)
    result = sorted(set(int(index) for index in indices.tolist()) | {0, n_frames - 1})
    return result


def render_gif(
    data: RenderData,
    camera: Camera,
    path: Path,
    *,
    maximum_frames: int,
) -> None:
    run = data.runs["unified"]
    indices = _sample_frame_indices(run.frames, maximum_frames)
    crystal = _to_camera(data.crystal_coords, camera.focus, camera.basis)
    pocket_center = (
        _to_camera(data.pocket_center.unsqueeze(0), camera.focus, camera.basis)[0]
        if data.pocket_center is not None
        else None
    )
    protein = _to_camera(
        _sample_protein(data.protein_coords, camera.focus),
        camera.focus,
        camera.basis,
    )
    frames = _to_camera(run.frames.reshape(-1, 3), camera.focus, camera.basis).reshape(
        run.frames.shape
    )
    centers = _fragment_centers(frames, data.fragment_id)
    rendered: list[Image.Image] = []
    for frame_index in indices:
        image = Image.new("RGB", (960, 640), BACKGROUND)
        draw = ImageDraw.Draw(image)
        draw.text(
            (28, 14),
            "Unified guidance relaxation (+ screened charge)",
            fill=INK,
            font=_font(28, bold=True),
        )
        draw.text(
            (28, 47),
            data.initialization_label,
            fill=MUTED,
            font=_font(16),
        )
        draw.text(
            (932, 27),
            f"step {run.saved_steps[frame_index]}",
            fill=MUTED,
            font=_font(18),
            anchor="ra",
        )
        _draw_scene(
            image,
            (28, 78, 932, 570),
            frame=frames[frame_index],
            crystal=crystal,
            protein=protein,
            pocket_center=pocket_center,
            center_path=centers[: frame_index + 1],
            fragment_id=data.fragment_id,
            bonds=data.bonds,
            half_span=camera.half_span,
        )
        total = _metric_at(run, "total", frame_index)
        interaction = _metric_at(run, "interaction", frame_index)
        screened_charge = _metric_at(run, "screened_charge", frame_index)
        rmsd = _metric_at(run, "rmsd", frame_index)
        status = (
            f"total {_format_number(total)}   "
            f"interaction {_format_number(interaction)}   "
            f"charge {_format_number(screened_charge)}   "
            f"raw RMSD {_format_number(rmsd)} Å"
        )
        draw.text((28, 596), status, fill=INK, font=_font(18))
        rendered.append(image)

    palette = [image.convert("P", palette=Image.Palette.ADAPTIVE, colors=255) for image in rendered]
    durations = [90] * len(palette)
    durations[0] = 450
    durations[-1] = 1200
    path.parent.mkdir(parents=True, exist_ok=True)
    palette[0].save(
        path,
        save_all=True,
        append_images=palette[1:],
        duration=durations,
        loop=0,
        disposal=2,
        optimize=False,
    )
    for image in rendered:
        image.close()


def render_dashboard(data: RenderData, camera: Camera, path: Path) -> None:
    image = Image.new("RGB", (1600, 1040), BACKGROUND)
    draw = ImageDraw.Draw(image)
    draw.text(
        (50, 24),
        "Guidance-only fragment relaxation",
        fill=INK,
        font=_font(38, bold=True),
    )
    draw.text(
        (50, 68),
        data.initialization_label,
        fill=MUTED,
        font=_font(18),
    )
    draw.text(
        (1550, 44),
        data.case_id,
        fill=MUTED,
        font=_font(22),
        anchor="ra",
    )

    crystal = _to_camera(data.crystal_coords, camera.focus, camera.basis)
    pocket_center = (
        _to_camera(data.pocket_center.unsqueeze(0), camera.focus, camera.basis)[0]
        if data.pocket_center is not None
        else None
    )
    protein = _to_camera(
        _sample_protein(data.protein_coords, camera.focus),
        camera.focus,
        camera.basis,
    )
    scene_boxes = {
        "physical": (50, 105, 775, 555),
        "unified": (825, 105, 1550, 555),
    }
    for mode, panel_box in scene_boxes.items():
        run = data.runs[mode]
        final_rmsd = _metric_at(run, "rmsd", len(run.frames) - 1)
        content = _draw_panel(
            draw,
            panel_box,
            run.label,
            f"final raw RMSD {_format_number(final_rmsd)} Å",
        )
        frames = _to_camera(run.frames.reshape(-1, 3), camera.focus, camera.basis).reshape(
            run.frames.shape
        )
        centers = _fragment_centers(frames, data.fragment_id)
        _draw_scene(
            image,
            content,
            frame=frames[-1],
            crystal=crystal,
            protein=protein,
            pocket_center=pocket_center,
            center_path=centers,
            fragment_id=data.fragment_id,
            bonds=data.bonds,
            half_span=camera.half_span,
        )

    physical = data.runs["physical"]
    unified = data.runs["unified"]
    energy_content = _draw_panel(
        draw,
        (50, 590, 775, 990),
        "Energy",
    )
    split = (energy_content[1] + energy_content[3]) // 2
    _draw_line_chart(
        draw,
        (energy_content[0], energy_content[1], energy_content[2], split),
        [
            ChartSeries(
                "Physical only",
                physical.saved_steps,
                physical.metrics["total"],
                PHYSICAL_COLOR,
            ),
            ChartSeries(
                "Unified",
                unified.saved_steps,
                unified.metrics["total"],
                UNIFIED_COLOR,
            ),
        ],
        title="Total",
        unit="energy",
    )
    _draw_line_chart(
        draw,
        (energy_content[0], split + 5, energy_content[2], energy_content[3]),
        [
            ChartSeries(
                "All interaction",
                unified.saved_steps,
                unified.metrics["interaction"],
                INTERACTION_COLOR,
            ),
            ChartSeries(
                "Screened charge",
                unified.saved_steps,
                unified.metrics["screened_charge"],
                CUT_BOND,
            ),
        ],
        title="Interaction contribution",
        unit="energy",
    )

    diagnostic_content = _draw_panel(
        draw,
        (825, 590, 1550, 990),
        "Geometric convergence",
    )
    split = (diagnostic_content[1] + diagnostic_content[3]) // 2
    _draw_line_chart(
        draw,
        (diagnostic_content[0], diagnostic_content[1], diagnostic_content[2], split),
        [
            ChartSeries(
                "Physical only",
                physical.saved_steps,
                physical.metrics["rmsd"],
                PHYSICAL_COLOR,
            ),
            ChartSeries(
                "Unified",
                unified.saved_steps,
                unified.metrics["rmsd"],
                UNIFIED_COLOR,
            ),
        ],
        title="Raw atom-index RMSD",
        unit="Å",
    )
    _draw_line_chart(
        draw,
        (diagnostic_content[0], split + 5, diagnostic_content[2], diagnostic_content[3]),
        [
            ChartSeries(
                "Physical only",
                physical.saved_steps,
                physical.metrics["cut_bond_error"],
                PHYSICAL_COLOR,
            ),
            ChartSeries(
                "Unified",
                unified.saved_steps,
                unified.metrics["cut_bond_error"],
                UNIFIED_COLOR,
            ),
        ],
        title="Maximum cut-bond error",
        unit="Å",
    )
    draw.text(
        (50, 1012),
        "Crystal pose is a diagnostic reference only; RMSD is raw atom-index RMSD without alignment.",
        fill=MUTED,
        font=_font(15),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, optimize=True)


def _rounded_coords(points: Tensor) -> list[list[float]]:
    return [[round(float(value), 3) for value in point] for point in points]


def _inline_run(run: RunData, data: RenderData, camera: Camera) -> dict[str, Any]:
    indices = _sample_frame_indices(run.frames, 80)
    frames = _to_camera(run.frames.reshape(-1, 3), camera.focus, camera.basis).reshape(
        run.frames.shape
    )
    centers = _fragment_centers(frames, data.fragment_id)
    return {
        "label": run.label,
        "frames": [_rounded_coords(frames[index]) for index in indices],
        "centers": [_rounded_coords(centers[index]) for index in indices],
        "steps": [run.saved_steps[index] for index in indices],
        "total": [_metric_at(run, "total", index) for index in indices],
        "interaction": [_metric_at(run, "interaction", index) for index in indices],
        "screenedCharge": [_metric_at(run, "screened_charge", index) for index in indices],
        "rmsd": [_metric_at(run, "rmsd", index) for index in indices],
    }


def write_inline_html(data: RenderData, camera: Camera, path: Path) -> None:
    protein = _sample_protein(data.protein_coords, camera.focus, max_points=550)
    payload = {
        "crystal": _rounded_coords(_to_camera(data.crystal_coords, camera.focus, camera.basis)),
        "protein": _rounded_coords(_to_camera(protein, camera.focus, camera.basis)),
        "pocketCenter": (
            _rounded_coords(
                _to_camera(data.pocket_center.unsqueeze(0), camera.focus, camera.basis)
            )[0]
            if data.pocket_center is not None
            else None
        ),
        "fragmentId": data.fragment_id.tolist(),
        "bonds": [list(bond) for bond in data.bonds],
        "halfSpan": round(camera.half_span, 3),
        "initializationLabel": data.initialization_label,
        "runs": {mode: _inline_run(run, data, camera) for mode, run in data.runs.items()},
    }
    encoded = json.dumps(payload, separators=(",", ":"), allow_nan=False).replace(
        "</",
        "<\\/",
    )
    template = INLINE_TEMPLATE_PATH.read_text()
    if template.count("__PAYLOAD__") != 1:
        raise ValueError(f"{INLINE_TEMPLATE_PATH} must contain exactly one __PAYLOAD__ token")
    root_suffix = "".join(
        character if character.isalnum() else "-"
        for character in path.stem.lower()
    ).strip("-")
    root_id = f"effdock-{root_suffix}-vis"
    if not root_suffix or "__ROOT_ID__" not in template:
        raise ValueError(f"{INLINE_TEMPLATE_PATH} must contain a valid __ROOT_ID__ token")
    fragment = template.replace("__PAYLOAD__", encoded).replace("__ROOT_ID__", root_id)
    if "__PAYLOAD__" in fragment or "__ROOT_ID__" in fragment:
        raise ValueError("inline visualization template replacement is incomplete")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(fragment)


def main() -> None:
    args = parse_args()
    if args.max_gif_frames < 2:
        raise SystemExit("--max-gif-frames must be at least 2")
    data = load_render_data(args.summary, args.trajectory)
    camera = _camera(data)
    output_dir = args.output_dir or args.summary.parent
    gif_path = output_dir / "relaxation.gif"
    dashboard_path = output_dir / "dashboard.png"
    render_gif(data, camera, gif_path, maximum_frames=args.max_gif_frames)
    render_dashboard(data, camera, dashboard_path)
    print(gif_path)
    print(dashboard_path)
    if args.inline_html is not None:
        write_inline_html(data, camera, args.inline_html)
        print(args.inline_html)


if __name__ == "__main__":
    main()
