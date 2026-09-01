#!/usr/bin/env python3
"""Verify the published EFF-Dock checkpoint pair on CPU."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import torch

from effdock.confidence import load_pose_confidence_model
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


def main() -> None:
    for path, expected in EXPECTED_SHA256.items():
        if not path.is_file():
            raise FileNotFoundError(f"release artifact is missing: {path}")
        observed = file_sha256(path)
        if observed != expected:
            raise RuntimeError(
                f"release artifact hash mismatch for {path.name}: "
                f"observed={observed}, expected={expected}"
            )

    device = torch.device("cpu")
    docking_model, _, docking_payload = load_model(CONFIG, DOCKING_CHECKPOINT, device)
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
