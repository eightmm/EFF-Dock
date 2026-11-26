"""Score or minimize poses that are already in the receptor's coordinate frame."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

import torch
from rdkit import Chem

from effdock.checkpoint import ORIENTATION_INJECTIONS
from effdock.confidence.features import FRAME_RECOVERY_POLICIES
from effdock.device import resolve_device
from effdock.inference.defaults import (
    DEFAULT_CONFIDENCE_CHECKPOINT,
    DEFAULT_CONFIG,
    DEFAULT_DOCKING_CHECKPOINT,
    DEFAULT_POCKET_CUTOFF,
    DEFAULT_SIGMA,
)
from effdock.inference.docking import load_model, parse_center
from effdock.inference.io import _clear_pose_scores, _set_coords, _tag_props
from effdock.inference.minimization import minimize_poses
from effdock.inference.preprocess import load_ligand, preprocess_complex


@dataclass(frozen=True)
class _SavedPoseOptions:
    protein: Path
    poses: Path
    pocket_center: torch.Tensor
    output: Path
    ligand_template: str | None = None
    device: str | None = None


@dataclass(frozen=True)
class ConfidenceOptions(_SavedPoseOptions):
    """Learned confidence settings for a saved SDF bank."""

    config: Path = DEFAULT_CONFIG
    checkpoint: Path = DEFAULT_DOCKING_CHECKPOINT
    confidence_checkpoint: Path = DEFAULT_CONFIDENCE_CHECKPOINT
    pocket_cutoff: float = DEFAULT_POCKET_CUTOFF
    sigma: float = DEFAULT_SIGMA
    orientation_injection: str | None = None
    confidence_frame_policy: str = "historical_kabsch"
    batch_size: int = 20


@dataclass(frozen=True)
class MinimizationOptions(_SavedPoseOptions):
    """Physical optimization settings for a saved SDF bank."""

    steps: int = 100
    batch_size: int = 10


def _ordered_graph(mol: Chem.Mol) -> tuple:
    # Score pose coordinates even when their geometry disagrees with the
    # declared stereo on the template, but require the same atom indexing.
    atoms = tuple(
        (a.GetAtomicNum(), a.GetIsotope(), a.GetFormalCharge(), a.GetIsAromatic())
        for a in mol.GetAtoms()
    )
    bonds = tuple(sorted(
        (min(b.GetBeginAtomIdx(), b.GetEndAtomIdx()), max(b.GetBeginAtomIdx(), b.GetEndAtomIdx()),
         b.GetBondTypeAsDouble())
        for b in mol.GetBonds()
    ))
    return atoms, bonds


def _read_poses(
    opts: _SavedPoseOptions, output_suffix: str,
) -> tuple[list[Chem.Mol], Chem.Mol, torch.Tensor, torch.Tensor]:
    if not opts.protein.is_file():
        raise FileNotFoundError(f"Receptor PDB not found: {opts.protein}")
    if opts.output.suffix.lower() != output_suffix:
        raise ValueError(f"Output must be a {output_suffix} file")
    inputs = [opts.protein, opts.poses]
    if opts.ligand_template is not None:
        inputs.append(Path(opts.ligand_template))
    if opts.output.resolve() in {path.resolve() for path in inputs}:
        raise ValueError("Output must be separate from the input files")
    center = torch.as_tensor(opts.pocket_center, dtype=torch.float32, device="cpu").detach()
    if center.shape != (3,) or not bool(torch.isfinite(center).all()):
        raise ValueError("Pocket center must contain three finite coordinates")
    molecules = []
    coordinates = []
    with opts.poses.open("rb") as stream:
        for index, mol in enumerate(Chem.ForwardSDMolSupplier(stream, removeHs=False)):
            if mol is None:
                raise ValueError(f"Unreadable SDF record {index + 1}: {opts.poses}")
            mol = Chem.RemoveAllHs(mol)
            if mol.GetNumAtoms() == 0 or mol.GetNumConformers() != 1:
                raise ValueError(f"SDF record {index + 1} must contain a heavy-atom conformer")
            xyz = torch.as_tensor(mol.GetConformer().GetPositions(), dtype=torch.float32)
            if not mol.GetConformer().Is3D() or not bool(torch.isfinite(xyz).all()):
                raise ValueError(f"SDF record {index + 1} must contain finite 3D coordinates")
            molecules.append(mol)
            coordinates.append(xyz - center)
    if not molecules:
        raise ValueError(f"No poses in SDF: {opts.poses}")
    if opts.ligand_template is None:
        template = molecules[0]
    else:
        template, _ = load_ligand(opts.ligand_template)
        template = Chem.RemoveAllHs(template)
    expected = _ordered_graph(template)
    for index, mol in enumerate(molecules):
        if _ordered_graph(mol) != expected:
            raise ValueError(f"SDF record {index + 1} does not match the ordered ligand graph")
    return molecules, template, torch.stack(coordinates), center


@contextmanager
def _atomic_output(path: Path) -> Iterator[Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(prefix=f".{path.name}.", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
    try:
        yield temporary
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def score_saved_poses(opts: ConfidenceOptions) -> None:
    """Write confidence scores and ranks for supplied poses, in input order."""
    from effdock.confidence.runtime import load_pose_confidence_model, score_poses_with_confidence

    if opts.batch_size < 1:
        raise ValueError("Batch size must be positive")
    if not math.isfinite(opts.sigma) or opts.sigma <= 0:
        raise ValueError("Sigma must be finite and positive")
    if not math.isfinite(opts.pocket_cutoff) or opts.pocket_cutoff <= 0:
        raise ValueError("Pocket cutoff must be finite and positive")
    if opts.confidence_frame_policy not in FRAME_RECOVERY_POLICIES:
        raise ValueError(f"Unknown confidence frame policy: {opts.confidence_frame_policy}")
    molecules, template, poses, center = _read_poses(opts, ".json")
    device = resolve_device(opts.device)
    graph, ligand_data, meta = preprocess_complex(
        opts.protein, template, pocket_center=center, pocket_cutoff=opts.pocket_cutoff,
    )
    docking_model, _, _ = load_model(
        opts.config, opts.checkpoint, device, orientation_injection=opts.orientation_injection,
    )
    confidence_model, _ = load_pose_confidence_model(opts.confidence_checkpoint, device)
    scores = []
    for batch in poses.split(opts.batch_size):
        scores.extend(score_poses_with_confidence(
            confidence_model, docking_model, graph, ligand_data, meta, list(batch),
            sigma=opts.sigma, device=device, frame_policy=opts.confidence_frame_policy,
        ))
    order = sorted(range(len(scores)), key=lambda index: scores[index]["confidence_rmsd"])
    ranks = {index: rank for rank, index in enumerate(order, 1)}
    records = []
    for index, (mol, score) in enumerate(zip(molecules, scores, strict=True)):
        entry = {"input_index": index, "rank": ranks[index], **score}
        if mol.HasProp("sample_index"):
            entry["sample_index"] = mol.GetProp("sample_index")
        records.append(entry)
    report = {
        "schema_version": "effdock.saved_pose_confidence.v1",
        "pocket_center": center.tolist(),
        "pocket_cutoff": opts.pocket_cutoff,
        "sigma": opts.sigma,
        "docking_checkpoint": str(opts.checkpoint),
        "confidence_checkpoint": str(opts.confidence_checkpoint),
        "orientation_injection": docking_model.orientation_injection,
        "confidence_frame_policy": opts.confidence_frame_policy,
        "poses": records,
    }
    with _atomic_output(opts.output) as temporary:
        temporary.write_text(json.dumps(report, allow_nan=False, indent=2) + "\n", encoding="utf-8")


@torch.inference_mode(False)
@torch.enable_grad()
def minimize(opts: MinimizationOptions) -> None:
    """Lower GuidanceEnergy for each supplied pose in a fixed receptor."""
    from effdock.preprocess.fragments import decompose_fragments

    molecules, template, poses, center = _read_poses(opts, ".sdf")
    template_coords = torch.as_tensor(template.GetConformer().GetPositions(), dtype=torch.float32)
    fragments = decompose_fragments(template, template_coords)
    if fragments is None:
        raise ValueError("Ligand fragment decomposition failed")
    minimized, diagnostics = minimize_poses(
        template, opts.protein, poses, center, fragments["fragment_id"],
        resolve_device(opts.device), steps=opts.steps, batch_size=opts.batch_size,
    )
    with _atomic_output(opts.output) as temporary, Chem.SDWriter(str(temporary)) as writer:
        writer.SetForceV3000(True)
        for mol, coords, properties in zip(molecules, minimized, diagnostics, strict=True):
            mol = _set_coords(mol, coords + center)
            _clear_pose_scores(mol)
            _tag_props(mol, properties)
            writer.write(mol)


def _pose_parser(prog: str, description: str, *, output: str, batch_size: int) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog, description=description)
    parser.add_argument("--protein", type=Path, required=True, help="Receptor PDB.")
    parser.add_argument("--poses", type=Path, required=True, help="Single- or multi-record pose SDF.")
    parser.add_argument("--pocket-center", type=parse_center, required=True, metavar="X,Y,Z")
    parser.add_argument("--output", type=Path, required=True, help=output)
    parser.add_argument("--ligand-template", help="Optional SDF, MOL2 or SMILES in the same atom order.")
    parser.add_argument("--device", help="Torch device. Defaults to CUDA, MPS or CPU if available.")
    parser.add_argument("--batch-size", type=int, default=batch_size)
    return parser


def confidence_main(argv: list[str] | None = None) -> None:
    parser = _pose_parser(
        "eff-dock confidence score", "Score saved poses with EFF-Dock confidence.",
        output="Path for the confidence JSON.", batch_size=20,
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_DOCKING_CHECKPOINT)
    parser.add_argument("--confidence-checkpoint", type=Path, default=DEFAULT_CONFIDENCE_CHECKPOINT)
    parser.add_argument("--pocket-cutoff", type=float, default=DEFAULT_POCKET_CUTOFF)
    parser.add_argument("--sigma", type=float, default=DEFAULT_SIGMA)
    parser.add_argument("--orientation-injection", choices=sorted(ORIENTATION_INJECTIONS))
    parser.add_argument("--confidence-frame-policy", choices=sorted(FRAME_RECOVERY_POLICIES),
                        default="historical_kabsch")
    opts = ConfidenceOptions(**vars(parser.parse_args(argv)))
    score_saved_poses(opts)
    print(f"Saved pose confidence: {opts.output}")


def minimize_main(argv: list[str] | None = None) -> None:
    parser = _pose_parser(
        "eff-dock minimize", "Minimize saved poses using EFF-Dock GuidanceEnergy.",
        output="Path for the minimized pose SDF.", batch_size=10,
    )
    parser.add_argument("--steps", type=int, default=100, help="Maximum descent steps per pose.")
    opts = MinimizationOptions(**vars(parser.parse_args(argv)))
    minimize(opts)
    print(f"Saved minimized poses: {opts.output}")
