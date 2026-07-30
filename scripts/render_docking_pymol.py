#!/usr/bin/env python3
"""Render an actual docking trajectory with PyMOL cartoon and molecular surface."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

FRAGMENT_COLORS = (
    (0.00, 0.45, 0.70),
    (0.90, 0.62, 0.00),
    (0.00, 0.62, 0.45),
    (0.80, 0.47, 0.65),
    (0.34, 0.71, 0.91),
    (0.84, 0.37, 0.00),
    (0.49, 0.38, 0.66),
    (0.42, 0.56, 0.14),
)
ELEMENT_SYMBOLS = {
    1: "H",
    6: "C",
    7: "N",
    8: "O",
    9: "F",
    15: "P",
    16: "S",
    17: "CL",
    35: "BR",
    53: "I",
}


def _pdb_atom_line(serial: int, atomic_number: int, xyz: np.ndarray) -> str:
    element = ELEMENT_SYMBOLS.get(atomic_number, "C")
    atom_name = f"{element}{serial}"[:4]
    x, y, z = (float(value) for value in xyz)
    return (
        f"HETATM{serial:5d} {atom_name:>4s} LIG X   1    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00          {element:>2s}"
    )


def _write_ligand_state(
    path: Path,
    xyz: np.ndarray,
    atomic_numbers: np.ndarray,
    bonds: list[tuple[int, int, float]],
) -> None:
    lines = [
        _pdb_atom_line(index + 1, int(atomic_numbers[index]), xyz[index])
        for index in range(len(atomic_numbers))
    ]
    adjacency: dict[int, list[int]] = {index + 1: [] for index in range(len(atomic_numbers))}
    for begin, end, order in bonds:
        begin_serial = int(begin) + 1
        end_serial = int(end) + 1
        repeats = max(1, int(round(float(order))))
        adjacency[begin_serial].extend([end_serial] * repeats)
        adjacency[end_serial].extend([begin_serial] * repeats)
    lines.append("TER")
    for serial, neighbors in adjacency.items():
        if neighbors:
            lines.append(f"CONECT{serial:5d}" + "".join(f"{other:5d}" for other in neighbors))
    lines.append("END")
    path.write_text("\n".join(lines) + "\n")


def _selection_for_ids(atom_ids: list[int]) -> str:
    return "+".join(str(atom_id) for atom_id in atom_ids)


def _pml_script(
    protein: Path,
    reference: Path | None,
    ligand_states: list[Path],
    fragment_id: np.ndarray,
    bonds: list[tuple[int, int, float]],
    pocket_center: np.ndarray,
    pocket_cutoff: float,
    frame_dir: Path,
    width: int,
    height: int,
) -> str:
    lines = [
        "reinitialize",
        "set quiet, 1",
        "set max_threads, 4",
        f"load {protein.as_posix()}, protein",
        "remove protein and solvent",
        "remove protein and not polymer.protein",
    ]
    if reference is not None:
        lines.extend(
            [
                f"load {reference.as_posix()}, reference",
                "remove reference and hydro",
            ]
        )
    for state, ligand_path in enumerate(ligand_states, start=1):
        lines.append(f"load {ligand_path.as_posix()}, ligand, state={state}")
    center = ",".join(f"{float(value):.6f}" for value in pocket_center)
    focus_selection = "ligand or reference" if reference is not None else "pocket_surface or ligand"
    focus_buffer = 4.0 if reference is not None else 1.5
    lines.extend(
        [
            f"pseudoatom pocket_center, pos=[{center}]",
            f"select pocket, byres (protein within {pocket_cutoff:.3f} of pocket_center)",
            f"select local_cartoon, byres (protein within {pocket_cutoff + 7.0:.3f} of pocket_center)",
            "create pocket_surface, pocket",
            "disable pocket_center",
            "hide everything, all",
            "dss protein",
            "show cartoon, local_cartoon",
            "show surface, pocket_surface",
            "show sticks, ligand",
            "show spheres, ligand",
            "set_color protein_cartoon_color, [0.52, 0.63, 0.73]",
            "set_color pocket_surface_color, [0.48, 0.77, 0.92]",
            "color protein_cartoon_color, protein",
            "set surface_color, pocket_surface_color, pocket_surface",
            "set cartoon_transparency, 0.18, protein",
            "set transparency, 0.70, pocket_surface",
            "set surface_quality, 1",
            "set solvent_radius, 1.4",
            "set stick_radius, 0.22, ligand",
            "set stick_quality, 24",
            "set sphere_scale, 0.31, ligand",
            "set sphere_quality, 2",
            "set valence, 1",
        ]
    )
    if reference is not None:
        lines.extend(
            [
                "show sticks, reference",
                "show spheres, reference",
                "set stick_radius, 0.16, reference",
                "set stick_transparency, 0.00, reference",
                "set sphere_scale, 0.21, reference",
                "set sphere_transparency, 0.00, reference",
                "set_color crystal_charcoal, [0.18, 0.20, 0.23]",
                "color crystal_charcoal, reference",
                "set transparency, 0.82, pocket_surface",
                "set surface_clear_selection, ligand or reference, pocket_surface",
                "set surface_clear_cutoff, 2.5, pocket_surface",
                "rebuild pocket_surface",
            ]
        )
    num_fragments = int(fragment_id.max()) + 1
    for fragment in range(num_fragments):
        color = FRAGMENT_COLORS[fragment % len(FRAGMENT_COLORS)]
        atom_ids = [int(index) + 1 for index in np.flatnonzero(fragment_id == fragment)]
        lines.extend(
            [
                f"set_color fragment_{fragment}, [{color[0]:.3f},{color[1]:.3f},{color[2]:.3f}]",
                f"select fragment_{fragment}_atoms, ligand and id {_selection_for_ids(atom_ids)}",
                f"color fragment_{fragment}, fragment_{fragment}_atoms",
            ]
        )
    lines.extend(
        [
            "color blue, ligand and elem N",
            "color red, ligand and elem O",
            "color yellow, ligand and elem S",
            "color orange, ligand and elem P",
            "color green, ligand and elem F+CL",
            "color brown, ligand and elem BR",
            "color purple, ligand and elem I",
        ]
    )
    for bond_index, (begin, end, _) in enumerate(bonds):
        begin, end = int(begin), int(end)
        if int(fragment_id[begin]) == int(fragment_id[end]):
            continue
        lines.extend(
            [
                f"select cut_a_{bond_index}, ligand and id {begin + 1}",
                f"select cut_b_{bond_index}, ligand and id {end + 1}",
                f"set_bond stick_color, red, cut_a_{bond_index}, cut_b_{bond_index}",
            ]
        )
    lines.extend(
        [
            "bg_color white",
            "set orthoscopic, on",
            "set depth_cue, 0",
            "set ray_shadows, 0",
            "set antialias, 2",
            "set ambient, 0.42",
            "set direct, 0.58",
            "set specular, 0.22",
            "set shininess, 22",
            "set ray_trace_mode, 1",
            "set ray_trace_color, gray70",
            "set opaque_background, on",
            "set cartoon_fancy_helices, on",
            "set cartoon_smooth_loops, on",
            f"set state, {len(ligand_states)}",
            f"orient {focus_selection}",
            f"zoom {focus_selection}, {focus_buffer:.1f}, complete=1",
            "turn x, 8",
            "turn y, -12",
        ]
    )
    for state in range(1, len(ligand_states) + 1):
        frame_path = frame_dir / f"frame_{state - 1:03d}.png"
        lines.extend(
            [
                f"set state, {state}",
                f"png {frame_path.as_posix()}, width={width}, height={height}, dpi=150, ray=1, quiet=1",
            ]
        )
    lines.append("quit")
    return "\n".join(lines) + "\n"


def _write_gif(frame_paths: list[Path], output: Path) -> None:
    images = [Image.open(path).convert("RGB") for path in frame_paths]
    palette_images = [
        image.convert("P", palette=Image.Palette.ADAPTIVE, colors=255) for image in images
    ]
    durations = [110] * len(palette_images)
    durations[0] = 350
    durations[-1] = 1100
    output.parent.mkdir(parents=True, exist_ok=True)
    palette_images[0].save(
        output,
        save_all=True,
        append_images=palette_images[1:],
        duration=durations,
        loop=0,
        disposal=2,
        optimize=False,
    )
    images[-1].save(output.with_suffix(".final.png"))
    chosen = [0, len(images) // 3, 2 * len(images) // 3, len(images) - 1]
    width, height = images[0].size
    contact = Image.new("RGB", (width, height), "white")
    for slot, index in enumerate(chosen):
        thumbnail = images[index].resize((width // 2, height // 2), Image.Resampling.LANCZOS)
        contact.paste(thumbnail, ((slot % 2) * width // 2, (slot // 2) * height // 2))
    contact.save(output.with_suffix(".contact.png"))
    for image in images:
        image.close()


def render(
    bundle_path: Path,
    output: Path,
    pymol_executable: Path,
    width: int,
    height: int,
    show_reference: bool,
) -> Path:
    bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
    frames = [torch.as_tensor(frame).numpy().astype(np.float64) for frame in bundle["traj"]]
    fragment_id = torch.as_tensor(bundle["fragment_id"]).numpy().astype(np.int64)
    atomic_numbers = torch.as_tensor(bundle["atomic_numbers"]).numpy().astype(np.int64)
    bonds = [tuple(bond) for bond in bundle["bonds"]]
    if any(len(frame) != len(atomic_numbers) for frame in frames):
        raise ValueError("trajectory atom count does not match ligand topology")

    with tempfile.TemporaryDirectory(prefix="effdock-pymol-") as tmp_name:
        tmp_dir = Path(tmp_name)
        frame_dir = tmp_dir / "rendered"
        frame_dir.mkdir()
        ligand_states = []
        for index, frame in enumerate(frames):
            state_path = tmp_dir / f"ligand_state_{index:03d}.pdb"
            _write_ligand_state(state_path, frame, atomic_numbers, bonds)
            ligand_states.append(state_path)
        pml_path = tmp_dir / "render.pml"
        pml_path.write_text(
            _pml_script(
                Path(bundle["protein"]).resolve(),
                Path(bundle["ligand_ref"]).resolve() if show_reference else None,
                ligand_states,
                fragment_id,
                bonds,
                torch.as_tensor(bundle["pocket_center"]).numpy(),
                float(bundle["pocket_cutoff"]),
                frame_dir,
                width,
                height,
            )
        )
        result = subprocess.run(
            [str(pymol_executable), "-cq", str(pml_path)],
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"PyMOL failed with exit code {result.returncode}:\n{result.stdout}\n{result.stderr}"
            )
        frame_paths = sorted(frame_dir.glob("frame_*.png"))
        if len(frame_paths) != len(frames):
            raise RuntimeError(
                f"PyMOL rendered {len(frame_paths)} frames; expected {len(frames)}\n"
                f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
            )
        _write_gif(frame_paths, output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--pymol",
        type=Path,
        default=Path(shutil.which("pymol") or "/tmp/effdock-pymol/bin/pymol"),
    )
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument(
        "--show-reference",
        action="store_true",
        help="Overlay the experimental reference ligand as semi-transparent gray sticks.",
    )
    args = parser.parse_args()
    if not args.pymol.exists():
        raise SystemExit(f"PyMOL executable not found: {args.pymol}")
    print(
        render(
            args.bundle,
            args.output,
            args.pymol,
            args.width,
            args.height,
            args.show_reference,
        )
    )


if __name__ == "__main__":
    main()
