"""Input-defined chemical constraints for fragment guidance.

This layer is reserved for coordinate constraints whose desired state is
declared by the inference ligand itself (for example, isomeric SMILES).  It
must not infer a binding pose, conformer preference, or protein interaction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

from .parameterization import load_chemical_constraints_v1
from .system import PhysicalSystem

_DEFAULTS = load_chemical_constraints_v1()["defaults"]


@dataclass(frozen=True)
class ChemicalConstraintEnergyConfig:
    """Numerical contract for opt-in input-defined chemical constraints."""

    scale: float = float(_DEFAULTS["runtime_scale"])
    tetrahedral_k: float = float(_DEFAULTS["tetrahedral_k"])
    double_bond_k: float = float(_DEFAULTS["double_bond_k"])
    margin_fraction: float = float(_DEFAULTS["margin_fraction"])
    softness: float = float(_DEFAULTS["softness"])

    def __post_init__(self) -> None:
        for name in ("scale", "tetrahedral_k", "double_bond_k"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not math.isfinite(self.margin_fraction) or not 0 < self.margin_fraction < 1:
            raise ValueError("margin_fraction must be in (0,1)")
        if not math.isfinite(self.softness) or self.softness <= 0:
            raise ValueError("softness must be finite and positive")


def _as_batch(coords: Tensor) -> tuple[Tensor, bool]:
    if coords.ndim == 2 and coords.shape[-1] == 3:
        return coords.unsqueeze(0), True
    if coords.ndim == 3 and coords.shape[-1] == 3:
        return coords, False
    raise ValueError(f"coords must have shape [N,3] or [B,N,3], got {tuple(coords.shape)}")


def _dihedral(p0: Tensor, p1: Tensor, p2: Tensor, p3: Tensor, eps: float = 1e-12) -> Tensor:
    b0 = p1 - p0
    b1 = p2 - p1
    b2 = p3 - p2
    b1_hat = b1 / b1.norm(dim=-1, keepdim=True).clamp_min(eps)
    v = b0 - (b0 * b1_hat).sum(dim=-1, keepdim=True) * b1_hat
    w = b2 - (b2 * b1_hat).sum(dim=-1, keepdim=True) * b1_hat
    x = (v * w).sum(dim=-1)
    y = (torch.linalg.cross(b1_hat, v, dim=-1) * w).sum(dim=-1)
    valid = (
        b1.square().sum(dim=-1).gt(eps)
        & v.square().sum(dim=-1).gt(eps)
        & w.square().sum(dim=-1).gt(eps)
    )
    return torch.atan2(
        torch.where(valid, y, torch.zeros_like(y)),
        torch.where(valid, x, torch.ones_like(x)),
    )


def _tetrahedral_alignment(coords: Tensor, system: PhysicalSystem) -> Tensor:
    topology = system.topology
    if not topology.stereo_tetra_index.numel():
        return coords.new_empty((coords.shape[0], 0))
    center, first, second, third = topology.stereo_tetra_index
    a = coords[:, first] - coords[:, center]
    b = coords[:, second] - coords[:, center]
    c = coords[:, third] - coords[:, center]
    denominator = (a.norm(dim=-1) * b.norm(dim=-1) * c.norm(dim=-1)).clamp_min(1e-12)
    signed_volume = (torch.linalg.cross(a, b, dim=-1) * c).sum(dim=-1) / denominator
    return signed_volume * topology.stereo_tetra_reference_sign.unsqueeze(0)


def _double_bond_alignment(coords: Tensor, system: PhysicalSystem) -> Tensor:
    topology = system.topology
    if not topology.stereo_double_index.numel():
        return coords.new_empty((coords.shape[0], 0))
    first, begin, end, fourth = topology.stereo_double_index
    value = _dihedral(
        coords[:, first],
        coords[:, begin],
        coords[:, end],
        coords[:, fourth],
    )
    return torch.cos(value) * topology.stereo_double_reference_cos_sign.unsqueeze(0)


def _barrier(
    alignment: Tensor,
    reference_magnitude: Tensor,
    *,
    force_constant: float,
    config: ChemicalConstraintEnergyConfig,
) -> Tensor:
    margin = float(config.margin_fraction) * reference_magnitude.unsqueeze(0)
    softness = float(config.softness)
    penetration = softness * F.softplus((margin - alignment) / softness)
    return 0.5 * float(force_constant) * penetration.square()


def chemical_constraint_energy(
    coords: Tensor,
    system: PhysicalSystem,
    config: ChemicalConstraintEnergyConfig = ChemicalConstraintEnergyConfig(),
) -> dict[str, Tensor]:
    """Return input-defined constraint barriers without choosing a conformer."""
    batched, squeeze = _as_batch(coords)
    if batched.shape[1] != system.topology.num_atoms:
        raise ValueError(
            f"coordinate atom count {batched.shape[1]} != topology {system.topology.num_atoms}"
        )
    zero = batched.new_zeros(batched.shape[0])
    if config.scale == 0:
        components = {
            "chemical_stereo_tetrahedral_barrier": zero,
            "chemical_stereo_double_bond_barrier": zero,
            "total": zero,
        }
    else:
        tetrahedral = _tetrahedral_alignment(batched, system)
        double_bond = _double_bond_alignment(batched, system)
        tetrahedral_energy = (
            _barrier(
                tetrahedral,
                system.topology.stereo_tetra_reference_magnitude,
                force_constant=config.tetrahedral_k,
                config=config,
            ).sum(dim=1)
            if tetrahedral.shape[1]
            else zero
        )
        double_bond_energy = (
            _barrier(
                double_bond,
                system.topology.stereo_double_reference_cos_magnitude,
                force_constant=config.double_bond_k,
                config=config,
            ).sum(dim=1)
            if double_bond.shape[1]
            else zero
        )
        components = {
            "chemical_stereo_tetrahedral_barrier": float(config.scale)
            * tetrahedral_energy,
            "chemical_stereo_double_bond_barrier": float(config.scale)
            * double_bond_energy,
            "total": float(config.scale) * (tetrahedral_energy + double_bond_energy),
        }
    if squeeze:
        return {name: value.squeeze(0) for name, value in components.items()}
    return components


def chemical_constraint_diagnostics(
    coords: Tensor,
    system: PhysicalSystem,
) -> dict[str, Tensor]:
    """Return per-pose signed margins and strict declared-stereo inversion counts."""
    batched, squeeze = _as_batch(coords)
    tetrahedral = _tetrahedral_alignment(batched, system)
    double_bond = _double_bond_alignment(batched, system)
    one = batched.new_ones(batched.shape[0])
    result = {
        "tetrahedral_min_alignment": tetrahedral.amin(dim=1) if tetrahedral.shape[1] else one,
        "tetrahedral_inversion_count": (
            tetrahedral.le(0).sum(dim=1) if tetrahedral.shape[1] else one.to(torch.long) * 0
        ),
        "double_bond_min_alignment": double_bond.amin(dim=1) if double_bond.shape[1] else one,
        "double_bond_inversion_count": (
            double_bond.le(0).sum(dim=1) if double_bond.shape[1] else one.to(torch.long) * 0
        ),
    }
    if squeeze:
        return {name: value.squeeze(0) for name, value in result.items()}
    return result


__all__ = [
    "ChemicalConstraintEnergyConfig",
    "chemical_constraint_diagnostics",
    "chemical_constraint_energy",
]
