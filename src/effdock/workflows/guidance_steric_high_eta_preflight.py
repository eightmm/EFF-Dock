#!/usr/bin/env python3
"""Verify current runtime identities before submitting the high-eta profile."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from effdock.guidance.parameterization import guidance_parameter_identity
from effdock.guidance.provenance import guidance_implementation_identity
from effdock.guidance.system import receptor_policy_identity
from effdock.workflows.guidance_eta_sweep_standalone_spec import STERIC_HIGH_ETA_V1


def build_preflight_identity() -> dict[str, Any]:
    spec = STERIC_HIGH_ETA_V1
    guidance = guidance_parameter_identity()
    receptor = receptor_policy_identity("geometry_only")
    physical = guidance.get("physical")
    interaction = guidance.get("interaction")
    exact = {
        "combined guidance SHA-256": (
            guidance.get("sha256"),
            spec.guidance_parameter_sha256,
        ),
        "physical SHA-256": (
            physical.get("sha256") if isinstance(physical, dict) else None,
            spec.physical_parameter_sha256,
        ),
        "physical version": (
            physical.get("version") if isinstance(physical, dict) else None,
            spec.physical_parameter_version,
        ),
        "physical formula version": (
            physical.get("formula_version") if isinstance(physical, dict) else None,
            spec.physical_formula_version,
        ),
        "interaction SHA-256": (
            interaction.get("sha256") if isinstance(interaction, dict) else None,
            spec.interaction_parameter_sha256,
        ),
        "geometry_only receptor policy SHA-256": (
            receptor.get("sha256"),
            spec.receptor_policy_sha256,
        ),
    }
    mismatches = {
        label: {"observed": observed, "expected": expected}
        for label, (observed, expected) in exact.items()
        if observed != expected
    }
    if mismatches:
        raise RuntimeError(f"frozen high-eta identity mismatch: {mismatches}")
    return {
        "protocol_id": spec.protocol_id,
        "profile": spec.key,
        "status": "passed",
        "eta_values": list(spec.eta_values),
        "eta_tags": list(spec.eta_tags),
        "guidance_parameter_set": guidance,
        "receptor_policy_identity": receptor,
        "guidance_implementation": guidance_implementation_identity(),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    identity = build_preflight_identity()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(identity, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(identity, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
