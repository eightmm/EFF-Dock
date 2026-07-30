"""Unified guidance energy and inference-time fragment force projection."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .interaction import InteractionEnergyConfig, interaction_energy
from .physical import PhysicalEnergyConfig, physical_energy
from .system import PhysicalSystem


@dataclass(frozen=True)
class GuidanceEnergyConfig:
    physical: PhysicalEnergyConfig = PhysicalEnergyConfig()
    interaction: InteractionEnergyConfig = InteractionEnergyConfig()


def guidance_energy(
    coords: Tensor,
    system: PhysicalSystem,
    config: GuidanceEnergyConfig = GuidanceEnergyConfig(),
) -> dict[str, Tensor]:
    """Return leaf terms plus one combined ``total``.

    Group totals are deliberately omitted from this flat mapping so summing
    every value except ``total`` cannot double-count physical or interaction
    energy.
    """
    physical = physical_energy(coords, system, config.physical)
    interaction = interaction_energy(coords, system, config.interaction)
    components = {
        name: value
        for name, value in physical.items()
        if name != "total"
    }
    for name, value in interaction.items():
        if name == "total":
            continue
        if name in components:
            raise RuntimeError(f"guidance energy term collision: {name}")
        components[name] = value
    components["total"] = physical["total"] + interaction["total"]
    return components


def _stable_symmetric_eigh(matrix: Tensor) -> tuple[Tensor, Tensor]:
    work = matrix.to(torch.float64)
    work = 0.5 * (work + work.transpose(-1, -2))
    if not bool(torch.isfinite(work).all()):
        raise FloatingPointError("physical projection inertia is non-finite")
    try:
        return torch.linalg.eigh(work)
    except RuntimeError as exc:
        message = str(exc).lower()
        convergence_failure = (
            "failed to converge" in message or "ill-conditioned" in message
        )
        if work.device.type != "cuda" or not convergence_failure:
            raise
        eigenvalues, eigenvectors = torch.linalg.eigh(work.cpu())
        return eigenvalues.to(work.device), eigenvectors.to(work.device)


def _clip_vectors(vectors: Tensor, maximum: float, eps: float = 1e-8) -> Tensor:
    norm = vectors.norm(dim=-1, keepdim=True)
    return vectors * (float(maximum) / norm.clamp_min(eps)).clamp(max=1.0)


def project_atom_forces(
    atom_force: Tensor,
    coords: Tensor,
    centers: Tensor,
    fragment_id: Tensor,
    masses: Tensor,
) -> tuple[Tensor, Tensor]:
    """Mass/inertia-precondition atom forces into fragment corrections."""
    if atom_force.ndim != 3 or coords.shape != atom_force.shape:
        raise ValueError("atom_force and coords must have shape [B,N,3]")
    batch_size, n_atoms, _ = coords.shape
    if fragment_id.numel() != n_atoms or masses.numel() != n_atoms:
        raise ValueError("fragment_id and masses must match atom count")
    n_fragments = centers.shape[1]
    if centers.shape != (batch_size, n_fragments, 3):
        raise ValueError("centers must have shape [B,F,3]")
    fragment_id = fragment_id.to(device=coords.device, dtype=torch.long)
    masses = masses.to(device=coords.device, dtype=coords.dtype)
    if int(fragment_id.min().item()) < 0 or int(fragment_id.max().item()) >= n_fragments:
        raise ValueError("fragment_id values must index the supplied centers")

    resultant = coords.new_zeros(batch_size, n_fragments, 3)
    resultant.index_add_(1, fragment_id, atom_force)
    fragment_mass = coords.new_zeros(n_fragments)
    fragment_mass.index_add_(0, fragment_id, masses)
    translation_at_com = resultant / fragment_mass.clamp_min(1e-8).view(1, -1, 1)

    weighted_position = coords.new_zeros(batch_size, n_fragments, 3)
    weighted_position.index_add_(
        1,
        fragment_id,
        coords * masses.view(1, -1, 1),
    )
    center_of_mass = weighted_position / fragment_mass.clamp_min(1e-8).view(1, -1, 1)
    lever = coords - center_of_mass[:, fragment_id]
    torque_atom = torch.linalg.cross(lever, atom_force, dim=-1)
    torque = coords.new_zeros(batch_size, n_fragments, 3)
    torque.index_add_(1, fragment_id, torque_atom)
    angular = torch.zeros_like(torque)
    eye = torch.eye(3, device=coords.device, dtype=torch.float64)
    for fragment in range(n_fragments):
        mask = fragment_id == fragment
        if int(mask.sum()) <= 1:
            continue
        r = lever[:, mask].to(torch.float64)
        m = masses[mask].to(torch.float64).view(1, -1, 1, 1)
        rr = r.unsqueeze(-1) * r.unsqueeze(-2)
        r2 = r.square().sum(dim=-1).unsqueeze(-1).unsqueeze(-1)
        inertia = (m * (r2 * eye - rr)).sum(dim=1)
        eigenvalues, eigenvectors = _stable_symmetric_eigh(inertia)
        maximum = eigenvalues.amax(dim=-1, keepdim=True).clamp_min(1e-8)
        observable = eigenvalues > (0.01 * maximum)
        torque_eig = torch.einsum(
            "bij,bj->bi", eigenvectors.transpose(-1, -2), torque[:, fragment].to(torch.float64)
        )
        omega_eig = torch.where(
            observable,
            torque_eig / eigenvalues.clamp_min(1e-8),
            torch.zeros_like(torque_eig),
        )
        angular[:, fragment] = torch.einsum(
            "bij,bj->bi", eigenvectors, omega_eig
        ).to(coords.dtype)
    translation = translation_at_com + torch.linalg.cross(
        angular,
        centers - center_of_mass,
        dim=-1,
    )
    return translation, angular


@dataclass(frozen=True)
class PhysicalGuidanceConfig:
    start_t: float = 0.4
    ramp_power: float = 1.0
    softcore_start: float = 1.5
    softcore_end: float = 0.75
    max_atom_force: float = 20.0
    max_translation_velocity: float = 5.0
    max_angular_velocity: float = 5.0

    def __post_init__(self) -> None:
        if not 0 <= self.start_t < 1:
            raise ValueError("start_t must be in [0,1)")
        for name in (
            "ramp_power",
            "softcore_start",
            "softcore_end",
            "max_atom_force",
            "max_translation_velocity",
            "max_angular_velocity",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")


class PhysicalGuidance:
    """Experimental callback; not admitted until an operator-split corrector passes."""

    def __init__(
        self,
        system: PhysicalSystem,
        config: PhysicalGuidanceConfig = PhysicalGuidanceConfig(),
    ) -> None:
        self.system = system
        self.config = config
        self.n_atoms = system.topology.num_atoms
        self.n_fragments = int(system.topology.fragment_id.max().item()) + 1
        self.last_components: dict[str, Tensor] | None = None

    def _progress(self, t: float) -> float:
        if t < self.config.start_t:
            return 0.0
        value = (float(t) - self.config.start_t) / (1.0 - self.config.start_t)
        return max(0.0, min(1.0, value))

    def __call__(
        self,
        atom_pos_flat: Tensor,
        frag_id_flat: Tensor,
        centers_flat: Tensor,
        t: float,
    ) -> tuple[Tensor, Tensor]:
        del frag_id_flat
        progress = self._progress(t)
        if progress == 0:
            zero = torch.zeros_like(centers_flat)
            return zero, zero
        if atom_pos_flat.shape[0] % self.n_atoms:
            raise ValueError("flattened physical-guidance coordinates are not whole poses")
        batch_size = atom_pos_flat.shape[0] // self.n_atoms
        if centers_flat.shape[0] != batch_size * self.n_fragments:
            raise ValueError("fragment centers do not match physical-guidance batch")
        ramp = progress**self.config.ramp_power
        softcore = self.config.softcore_start + progress * (
            self.config.softcore_end - self.config.softcore_start
        )

        with torch.enable_grad():
            coords = atom_pos_flat.detach().view(batch_size, self.n_atoms, 3).requires_grad_(True)
            components = physical_energy(
                coords,
                self.system,
                PhysicalEnergyConfig(softcore=softcore),
            )
            atom_force = -torch.autograd.grad(components["total"].sum(), coords)[0]
        if not torch.isfinite(atom_force).all():
            raise FloatingPointError("non-finite physical guidance force")
        atom_force = _clip_vectors(atom_force, self.config.max_atom_force)
        centers = centers_flat.detach().view(batch_size, self.n_fragments, 3)
        translation, angular = project_atom_forces(
            atom_force,
            coords.detach(),
            centers,
            self.system.topology.fragment_id,
            self.system.topology.mass,
        )
        translation = _clip_vectors(translation, self.config.max_translation_velocity)
        angular = _clip_vectors(angular, self.config.max_angular_velocity)
        self.last_components = {name: value.detach() for name, value in components.items()}
        return (
            (ramp * translation).reshape(batch_size * self.n_fragments, 3),
            (ramp * angular).reshape(batch_size * self.n_fragments, 3),
        )


__all__ = [
    "GuidanceEnergyConfig",
    "PhysicalGuidance",
    "PhysicalGuidanceConfig",
    "guidance_energy",
    "project_atom_forces",
]
