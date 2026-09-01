#!/usr/bin/env python3
"""Run the frozen rigid-fragment refinement contract on a 40-pose bank."""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_guidance_sdf_post_refinement as refinement  # noqa: E402

SOURCE_PROTOCOL_ID = "EFFDOCK-FIXED-NFE-STEP-POSE-REFINEMENT-INPUT-V1"
REFINEMENT_PROTOCOL_ID = "EFFDOCK-FIXED-NFE-STEP-POSE-REFINEMENT-V1"
EXPECTED_POSES = 40


def configure() -> None:
    """Apply only the frozen 40-pose/protocol specialization."""
    refinement.EXPECTED_POSES = EXPECTED_POSES
    refinement.PROTOCOL_ID = REFINEMENT_PROTOCOL_ID
    refinement.SUPPORTED_SOURCE_PROTOCOLS.add(SOURCE_PROTOCOL_ID)


def main() -> None:
    configure()
    refinement.main()


if __name__ == "__main__":
    main()
