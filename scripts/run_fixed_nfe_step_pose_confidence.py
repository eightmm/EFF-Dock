#!/usr/bin/env python3
"""Run frozen U50 symmetry-confidence scoring on a 40-pose refined bank."""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_guidance_sdf_post_refinement as refinement  # noqa: E402

EXPECTED_POSES = 40
REFINEMENT_PROTOCOL_ID = "EFFDOCK-FIXED-NFE-STEP-POSE-REFINEMENT-V1"
CONFIDENCE_PROTOCOL_ID = "EFFDOCK-FIXED-NFE-STEP-POSE-U50-CONFIDENCE-V1"


def configure():
    """Import and specialize the existing scorer without changing its math."""
    refinement.EXPECTED_POSES = EXPECTED_POSES
    refinement.PROTOCOL_ID = REFINEMENT_PROTOCOL_ID

    import score_guidance_sdf_post_refinement_confidence as confidence

    confidence.EXPECTED_POSES = EXPECTED_POSES
    confidence.REFINEMENT_PROTOCOL_ID = REFINEMENT_PROTOCOL_ID
    confidence.PROTOCOL_ID = CONFIDENCE_PROTOCOL_ID
    return confidence


def main() -> None:
    configure().main()


if __name__ == "__main__":
    main()
