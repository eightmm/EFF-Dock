"""Vectorized Torch physical terms for the EFF-FF diagnostic parameter set."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

from .parameterization import load_effff_v2
from .system import (
    _GENERIC_OBSTACLE_CUTOFF,
    _GENERIC_OBSTACLE_RADIUS,
    _GENERIC_OBSTACLE_REPULSION_MAX,
    _GENERIC_OBSTACLE_SOFTCORE,
    _GENERIC_OBSTACLE_SWITCH_DISTANCE,
    PhysicalSystem,
)


@dataclass(frozen=True)
class PhysicalEnergyConfig:
    softcore: float = float(load_effff_v2()["defaults"]["softcore_angstrom"])
    switch_distance: float = float(
        load_effff_v2()["defaults"]["switch_distance_angstrom"]
    )
    cutoff: float = float(load_effff_v2()["defaults"]["cutoff_angstrom"])
    protein_ligand_steric_barrier_enabled: bool = bool(
        load_effff_v2()["defaults"]["protein_ligand_steric_barrier_enabled"]
    )
    steric_radius_scale: float = float(
        load_effff_v2()["defaults"]["steric_radius_scale"]
    )
    steric_k: float = float(load_effff_v2()["defaults"]["steric_k"])
    steric_tau: float = float(
        load_effff_v2()["defaults"]["steric_tau_angstrom"]
    )
    steric_cutoff_margin: float = float(
        load_effff_v2()["defaults"]["steric_cutoff_margin_angstrom"]
    )
    # Dimensionless runtime ablation scale.  The versioned EFF-FF equilibrium
    # geometry and force constants remain unchanged; this scale applies only
    # to non-planar (stereochemical) improper restraints.
    chiral_improper_scale: float = 1.0
    protein_chunk_size: int = 512
    generic_obstacle_repulsion_max: float = _GENERIC_OBSTACLE_REPULSION_MAX
    generic_obstacle_softcore: float = _GENERIC_OBSTACLE_SOFTCORE
    generic_obstacle_radius: float = _GENERIC_OBSTACLE_RADIUS
    generic_obstacle_switch_distance: float = _GENERIC_OBSTACLE_SWITCH_DISTANCE
    generic_obstacle_cutoff: float = _GENERIC_OBSTACLE_CUTOFF

    def __post_init__(self) -> None:
        if self.softcore <= 0:
            raise ValueError("softcore must be positive")
        if not 0 < self.switch_distance < self.cutoff:
            raise ValueError("require 0 < switch_distance < cutoff")
        if self.protein_chunk_size <= 0:
            raise ValueError("protein_chunk_size must be positive")
        if not isinstance(self.protein_ligand_steric_barrier_enabled, bool):
            raise ValueError("protein_ligand_steric_barrier_enabled must be bool")
        for name in (
            "steric_radius_scale",
            "steric_tau",
            "steric_cutoff_margin",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.steric_k < 0:
            raise ValueError("steric_k must be non-negative")
        if self.chiral_improper_scale < 0:
            raise ValueError("chiral_improper_scale must be non-negative")
        for name in (
            "generic_obstacle_repulsion_max",
            "generic_obstacle_softcore",
            "generic_obstacle_radius",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0 < self.generic_obstacle_switch_distance < self.generic_obstacle_cutoff:
            raise ValueError(
                "require 0 < generic_obstacle_switch_distance < "
                "generic_obstacle_cutoff"
            )


def _as_batch(coords: Tensor) -> tuple[Tensor, bool]:
    if coords.ndim == 2 and coords.shape[-1] == 3:
        return coords.unsqueeze(0), True
    if coords.ndim == 3 and coords.shape[-1] == 3:
        return coords, False
    raise ValueError(f"coords must have shape [N,3] or [B,N,3], got {tuple(coords.shape)}")


def _zero(coords: Tensor) -> Tensor:
    return coords.new_zeros(coords.shape[0])


def _pairwise_distance(left: Tensor, right: Tensor) -> Tensor:
    """Return stable pair distances without ``cdist``'s MM cancellation.

    ``torch.cdist`` may select a matrix-multiplication implementation based on
    the point counts.  At large absolute PDB coordinates that makes the same
    close contact differ slightly when the receptor shell or chunk layout
    changes.  Direct relative-coordinate norms keep the diagnostic energy
    independent of those bookkeeping choices.
    """
    return (left.unsqueeze(-2) - right.unsqueeze(-3)).norm(dim=-1)


def _angle(p0: Tensor, p1: Tensor, p2: Tensor, eps: float = 1e-12) -> Tensor:
    left = p0 - p1
    right = p2 - p1
    cross = torch.linalg.cross(left, right, dim=-1).norm(dim=-1)
    dot = (left * right).sum(dim=-1)
    valid = left.square().sum(dim=-1).gt(eps) & right.square().sum(dim=-1).gt(eps)
    safe_cross = torch.where(valid, cross, torch.zeros_like(cross))
    safe_dot = torch.where(valid, dot, torch.ones_like(dot))
    return torch.atan2(safe_cross, safe_dot)


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
    safe_x = torch.where(valid, x, torch.ones_like(x))
    safe_y = torch.where(valid, y, torch.zeros_like(y))
    return torch.atan2(safe_y, safe_x)


def _wrapped_delta(value: Tensor, target: Tensor) -> Tensor:
    return torch.atan2(torch.sin(value - target), torch.cos(value - target))


def _switch(distance: Tensor, switch_distance: float, cutoff: float) -> Tensor:
    u = ((float(cutoff) - distance) / (float(cutoff) - float(switch_distance))).clamp(0, 1)
    # Quintic smoothstep: value, first derivative, and second derivative are
    # continuous at both the switch and cutoff boundaries.
    return u.pow(3) * (10.0 - 15.0 * u + 6.0 * u.square())


def _lj_components(
    distance: Tensor,
    x_ij: Tensor,
    d_ij: Tensor,
    *,
    scale: Tensor,
    config: PhysicalEnergyConfig,
) -> tuple[Tensor, Tensor]:
    r_eff = (distance.square() + float(config.softcore) ** 2).sqrt()
    ratio6 = (x_ij / r_eff.clamp_min(1e-6)).pow(6)
    switch = _switch(distance, config.switch_distance, config.cutoff)
    repulsive = scale * switch * d_ij * ratio6.square()
    attractive = scale * switch * (-2.0 * d_ij * ratio6)
    return repulsive, attractive


def _steric_barrier(
    distance: Tensor,
    ligand_vdw_radius: Tensor,
    protein_vdw_radius: Tensor,
    *,
    config: PhysicalEnergyConfig,
) -> Tensor:
    """Compact smooth vdW-overlap barrier for protein-ligand heavy atoms."""
    safe_distance = float(config.steric_radius_scale) * (
        ligand_vdw_radius.view(1, -1, 1)
        + protein_vdw_radius.view(1, 1, -1)
    )
    tau = float(config.steric_tau)
    penetration = tau * F.softplus((safe_distance - distance) / tau)
    outer = safe_distance + float(config.steric_cutoff_margin)
    u = ((outer - distance) / float(config.steric_cutoff_margin)).clamp(0, 1)
    compact_switch = u.pow(3) * (10.0 - 15.0 * u + 6.0 * u.square())
    return 0.5 * float(config.steric_k) * penetration.square() * compact_switch


def physical_energy(
    coords: Tensor,
    system: PhysicalSystem,
    config: PhysicalEnergyConfig = PhysicalEnergyConfig(),
) -> dict[str, Tensor]:
    """Return per-pose energy components in the declared diagnostic units."""
    batched, squeeze = _as_batch(coords)
    topology = system.topology
    if batched.shape[1] != topology.num_atoms:
        raise ValueError(
            f"coordinate atom count {batched.shape[1]} != topology {topology.num_atoms}"
        )
    components: dict[str, Tensor] = {}

    if topology.bond_index.numel():
        i, j = topology.bond_index
        distance = (batched[:, i] - batched[:, j]).norm(dim=-1)
        components["ligand_intra_bond"] = (
            0.5 * topology.bond_k.unsqueeze(0) * (distance - topology.bond_r0) ** 2
        ).sum(dim=1)
    else:
        components["ligand_intra_bond"] = _zero(batched)

    if topology.angle_index.numel():
        i, j, k = topology.angle_index
        value = _angle(batched[:, i], batched[:, j], batched[:, k])
        components["ligand_intra_angle"] = (
            0.5 * topology.angle_k.unsqueeze(0) * (value - topology.angle_theta0) ** 2
        ).sum(dim=1)
    else:
        components["ligand_intra_angle"] = _zero(batched)

    if topology.proper_index.numel():
        i, j, k, l = topology.proper_index
        value = _dihedral(batched[:, i], batched[:, j], batched[:, k], batched[:, l])
        components["ligand_intra_proper"] = (
            topology.proper_k.unsqueeze(0)
            * topology.proper_weight.unsqueeze(0)
            * (
                1.0
                + torch.cos(
                    topology.proper_periodicity.unsqueeze(0) * value
                    - topology.proper_phase.unsqueeze(0)
                )
            )
        ).sum(dim=1)
    else:
        components["ligand_intra_proper"] = _zero(batched)

    if topology.improper_index.numel():
        i, j, k, l = topology.improper_index
        value = _dihedral(batched[:, i], batched[:, j], batched[:, k], batched[:, l])
        delta = _wrapped_delta(value, topology.improper_phi0.unsqueeze(0))
        harmonic = 0.5 * topology.improper_k.unsqueeze(0) * delta.square()
        planar = 0.5 * topology.improper_k.unsqueeze(0) * (
            1.0 - torch.cos(2.0 * value)
        )
        components["ligand_intra_improper"] = torch.where(
            topology.improper_planar.unsqueeze(0),
            planar,
            float(config.chiral_improper_scale) * harmonic,
        ).sum(dim=1)
    else:
        components["ligand_intra_improper"] = _zero(batched)

    if topology.ligand_pair_index.numel():
        i, j = topology.ligand_pair_index
        distance = (batched[:, i] - batched[:, j]).norm(dim=-1)
        x_ij = torch.sqrt(topology.uff_x[i] * topology.uff_x[j]).unsqueeze(0)
        d_ij = torch.sqrt(topology.uff_d[i] * topology.uff_d[j]).unsqueeze(0)
        scale = topology.ligand_pair_scale.unsqueeze(0)
        rep, attr = _lj_components(
            distance,
            x_ij,
            d_ij,
            scale=scale,
            config=config,
        )
        components["ligand_intra_lj_repulsive"] = rep.sum(dim=1)
        components["ligand_intra_lj_attractive"] = attr.sum(dim=1)
    else:
        components["ligand_intra_lj_repulsive"] = _zero(batched)
        components["ligand_intra_lj_attractive"] = _zero(batched)

    pl_repulsive = _zero(batched)
    pl_attractive = _zero(batched)
    pl_steric_barrier = _zero(batched)
    protein_coords = system.protein_coords
    for start in range(0, protein_coords.shape[0], config.protein_chunk_size):
        stop = min(start + config.protein_chunk_size, protein_coords.shape[0])
        protein_chunk = protein_coords[start:stop]
        distance = _pairwise_distance(batched, protein_chunk)
        x_ij = torch.sqrt(
            topology.uff_x.view(1, -1, 1)
            * system.protein_uff_x[start:stop].view(1, 1, -1)
        )
        d_ij = torch.sqrt(
            topology.uff_d.view(1, -1, 1)
            * system.protein_uff_d[start:stop].view(1, 1, -1)
        )
        rep, attr = _lj_components(
            distance,
            x_ij,
            d_ij,
            scale=torch.ones_like(distance),
            config=config,
        )
        pl_repulsive = pl_repulsive + rep.sum(dim=(1, 2))
        pl_attractive = pl_attractive + attr.sum(dim=(1, 2))
        if config.protein_ligand_steric_barrier_enabled and config.steric_k > 0:
            steric = _steric_barrier(
                distance,
                topology.vdw_radius,
                system.protein_vdw_radius[start:stop],
                config=config,
            )
            pl_steric_barrier = pl_steric_barrier + steric.sum(dim=(1, 2))
    components["protein_ligand_lj_repulsive"] = pl_repulsive
    components["protein_ligand_lj_attractive"] = pl_attractive
    components["protein_ligand_steric_barrier"] = pl_steric_barrier

    obstacle_count = int(system.geometry_obstacle_coords.shape[0])
    if obstacle_count:
        generic_mask = system.geometry_obstacle_is_generic
        uff_mask = ~generic_mask
        if bool(uff_mask.any()):
            obstacle_coords = system.geometry_obstacle_coords[uff_mask]
            distance = _pairwise_distance(batched, obstacle_coords)
            x_ij = torch.sqrt(
                topology.uff_x.view(1, -1, 1)
                * system.geometry_obstacle_uff_x[uff_mask].view(1, 1, -1)
            )
            d_ij = torch.sqrt(
                topology.uff_d.view(1, -1, 1)
                * system.geometry_obstacle_uff_d[uff_mask].view(1, 1, -1)
            )
            repulsive, _ = _lj_components(
                distance,
                x_ij,
                d_ij,
                scale=torch.ones_like(distance),
                config=config,
            )
            components["receptor_geometry_obstacle_uff_repulsive"] = repulsive.sum(
                dim=(1, 2)
            )
        if bool(generic_mask.any()):
            obstacle_coords = system.geometry_obstacle_coords[generic_mask]
            distance = _pairwise_distance(batched, obstacle_coords)
            rho = (
                distance.square() + float(config.generic_obstacle_softcore) ** 2
            ).sqrt()
            ratio = (float(config.generic_obstacle_radius) / rho).pow(12)
            bounded = ratio / (1.0 + ratio)
            switch = _switch(
                distance,
                config.generic_obstacle_switch_distance,
                config.generic_obstacle_cutoff,
            )
            components["receptor_geometry_obstacle_generic_repulsive"] = (
                float(config.generic_obstacle_repulsion_max) * bounded * switch
            ).sum(dim=(1, 2))
    components["total"] = sum(components.values(), start=_zero(batched))

    if squeeze:
        return {name: value.squeeze(0) for name, value in components.items()}
    return components


__all__ = ["PhysicalEnergyConfig", "physical_energy"]
