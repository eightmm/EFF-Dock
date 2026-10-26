#!/usr/bin/env python3
"""Verify checkpoint hashes/loading; optionally run one synthetic pose."""

from __future__ import annotations

import argparse
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from rdkit import Chem

from effdock.confidence import load_pose_confidence_model
from effdock.inference import DockingOptions, dock
from effdock.inference.docking import load_model

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPOSITORY_ROOT / "configs/train.yaml"
DOCKING_CHECKPOINT = (
    REPOSITORY_ROOT / "weights/effdock_docking_early_time_t0p10_50k.pt"
)
CONFIDENCE_CHECKPOINT = (
    REPOSITORY_ROOT / "weights/effdock_confidence_s50_raw_refined_u70k.pt"
)
EXPECTED_SHA256 = {
    DOCKING_CHECKPOINT: "65be44d7dc8f0867eb9fc5d22214b80f93971ea4702679a527c665046e91e6b6",
    CONFIDENCE_CHECKPOINT: "ce59be42f0ca613871ca079127c3296f5ca9a4ec72e44a9e5cf61878351c2638",
}


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def smoke_inference(device: str) -> None:
    """Exercise the released pipeline without downloading a benchmark structure."""
    with TemporaryDirectory(prefix="effdock-release-") as directory:
        root = Path(directory)
        receptor = root / "receptor.pdb"
        atoms = [
            ("N", "N", (-4.0, 0.0, 0.0)),
            ("CA", "C", (-2.55, 0.0, 0.0)),
            ("C", "C", (-1.5, 1.0, 0.0)),
            ("O", "O", (-1.8, 2.2, 0.0)),
        ]
        receptor.write_text(
            "".join(
                f"ATOM  {index:5d} {name:^4s} GLY A   1    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00          {element:>2s}\n"
                for index, (name, element, (x, y, z)) in enumerate(atoms, start=1)
            )
            + "TER\nEND\n"
        )
        output = root / "docked"
        dock(
            DockingOptions(
                protein=receptor,
                ligand="CCO",
                pocket_center=torch.zeros(3),
                config=CONFIG,
                checkpoint=DOCKING_CHECKPOINT,
                confidence_checkpoint=CONFIDENCE_CHECKPOINT,
                orientation_injection="rw",
                confidence_frame_policy="contextual_v1",
                num_samples=1,
                num_steps=1,
                rank_by="confidence",
                score=False,
                device=device,
                seed=42,
                out_dir=output,
            )
        )
        poses = list(Chem.SDMolSupplier(str(output / "docked.sdf"), removeHs=False))
        if len(poses) != 1 or poses[0] is None or poses[0].GetNumAtoms() != 3:
            raise RuntimeError("release smoke did not export the complete three-atom ligand")
        pose = poses[0]
        coordinates = torch.as_tensor(pose.GetConformer().GetPositions())
        score = torch.tensor(pose.GetDoubleProp("confidence_rmsd"))
        if not bool(torch.isfinite(coordinates).all() and torch.isfinite(score) and score >= 0):
            raise RuntimeError("release smoke produced invalid coordinates or confidence")
        if not (output / "results.pt").is_file():
            raise RuntimeError("release smoke did not export inference provenance")
    print("release smoke passed: preparation, one ODE step, confidence and SDF export")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="also run one synthetic docking pose")
    parser.add_argument(
        "--device",
        default="auto",
        help="smoke device: auto, cuda, mps, or cpu (default: auto)",
    )
    args = parser.parse_args()
    torch.set_num_threads(1)
    for path, expected in EXPECTED_SHA256.items():
        if not path.is_file():
            raise FileNotFoundError(f"release artifact is missing: {path}; run git lfs pull")
        observed = file_sha256(path)
        if observed != expected:
            with path.open("rb") as handle:
                if handle.read(128).startswith(b"version https://git-lfs.github.com/spec/v1"):
                    raise RuntimeError(f"{path.name} is an LFS pointer; run git lfs pull")
            raise RuntimeError(
                f"release artifact hash mismatch for {path.name}: "
                f"observed={observed}, expected={expected}"
            )

    if args.smoke:
        from effdock.inference.docking import resolve_device

        device = str(resolve_device(None if args.device == "auto" else args.device))
        smoke_inference(device)
        return

    device = torch.device("cpu")
    docking_model, _, docking_payload = load_model(
        CONFIG, DOCKING_CHECKPOINT, device, orientation_injection="rw"
    )
    confidence_model, confidence_payload = load_pose_confidence_model(
        CONFIDENCE_CHECKPOINT, device
    )

    print(
        "release verification passed: "
        f"docking={type(docking_model).__name__} "
        f"step={docking_payload.get('step', '?')}, "
        f"confidence={type(confidence_model).__name__} "
        f"step={confidence_payload.get('step', '?')}"
    )


if __name__ == "__main__":
    main()
