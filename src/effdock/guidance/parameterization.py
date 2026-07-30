"""Load versioned, in-repository EFF-Dock guidance parameter sets."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
from importlib.resources import files

import torch
from torch import Tensor

from .errors import UnsupportedPhysicalChemistryError


@dataclass(frozen=True)
class ElementTensorParameters:
    mass: Tensor
    uff_x: Tensor
    uff_d: Tensor


@lru_cache(maxsize=1)
def load_effff_v2() -> dict:
    path = files("effdock.guidance.parameters").joinpath("effff_v2.json")
    return json.loads(path.read_text())


@lru_cache(maxsize=1)
def load_interaction_v1() -> dict:
    path = files("effdock.guidance.parameters").joinpath("interaction_v1.json")
    return json.loads(path.read_text())


def _parameter_identity(raw: dict) -> dict[str, str]:
    canonical = json.dumps(raw, sort_keys=True, separators=(",", ":")).encode()
    return {
        "name": str(raw["name"]),
        "version": str(raw["version"]),
        "formula_version": str(raw["formula_version"]),
        "sha256": sha256(canonical).hexdigest(),
        "energy_unit": str(raw["energy_unit"]),
        "distance_unit": str(raw["distance_unit"]),
        "claim": str(raw["claim"]),
    }


def parameter_identity() -> dict[str, str]:
    """Return the physical EFF-FF-v2 identity (backward-compatible name)."""
    return _parameter_identity(load_effff_v2())


def interaction_parameter_identity() -> dict[str, str]:
    return _parameter_identity(load_interaction_v1())


def guidance_parameter_identity() -> dict[str, object]:
    physical = parameter_identity()
    interaction = interaction_parameter_identity()
    payload = {
        "schema_version": "effdock.guidance_parameter_set.v1",
        "physical": physical,
        "interaction": interaction,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return {
        **payload,
        "name": "EFF-Dock-Guidance",
        "version": "1.1.0",
        "formula_version": "physical-v2_plus_interaction-v1.3",
        "sha256": sha256(canonical).hexdigest(),
        "energy_unit": "kcal/mol",
        "distance_unit": "angstrom",
        "claim": (
            "Unified self-contained diagnostic GuidanceEnergy = "
            "PhysicalEnergy + InteractionEnergy; Vina is excluded."
        ),
    }


def element_parameters(
    atomic_numbers: Tensor,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
) -> ElementTensorParameters:
    """Resolve supported element parameters or fail explicitly."""
    raw = load_effff_v2()["elements"]
    flat = atomic_numbers.detach().cpu().to(torch.long).view(-1).tolist()
    missing = sorted({int(z) for z in flat if str(int(z)) not in raw})
    if missing:
        raise UnsupportedPhysicalChemistryError(
            "unsupported_element",
            f"EFF-FF-v2 unsupported atomic numbers: {missing}",
            details={"atomic_numbers": missing},
        )

    def values(key: str) -> Tensor:
        return torch.tensor(
            [float(raw[str(int(z))][key]) for z in flat],
            dtype=dtype,
            device=device,
        ).view(atomic_numbers.shape)

    return ElementTensorParameters(
        mass=values("mass"),
        uff_x=values("uff_x"),
        uff_d=values("uff_d"),
    )


__all__ = [
    "ElementTensorParameters",
    "element_parameters",
    "guidance_parameter_identity",
    "interaction_parameter_identity",
    "load_effff_v2",
    "load_interaction_v1",
    "parameter_identity",
]
