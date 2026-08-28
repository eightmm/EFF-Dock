"""Unified guidance energy and inference-time fragment force projection."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import torch
from torch import Tensor

from effdock.geometry.flow_matching import integrate_se3_step
from effdock.geometry.se3 import quaternion_to_matrix

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


def _induced_atom_velocity(
    translation: Tensor,
    angular: Tensor,
    coords: Tensor,
    centers: Tensor,
    fragment_id: Tensor,
) -> Tensor:
    """Map fragment translation/rotation velocities into atom velocity space."""
    fragment_id = fragment_id.to(device=coords.device, dtype=torch.long)
    lever = coords - centers[:, fragment_id]
    return translation[:, fragment_id] + torch.linalg.cross(
        angular[:, fragment_id],
        lever,
        dim=-1,
    )


def _interval_average_ramp(
    t_start: float,
    t_end: float,
    *,
    guidance_start: float,
    power: float,
) -> float:
    """Average the continuous late-time ramp over one ODE interval."""
    left = float(t_start)
    right = float(t_end)
    if not 0.0 <= left < right <= 1.0 + 1e-6:
        raise ValueError("direct-guidance interval must satisfy 0 <= t_start < t_end <= 1")
    if right <= guidance_start:
        return 0.0
    active_left = max(left, guidance_start)
    numerator = (right - guidance_start) ** (power + 1.0) - (
        active_left - guidance_start
    ) ** (power + 1.0)
    denominator = (
        (power + 1.0)
        * (1.0 - guidance_start) ** power
        * (right - left)
    )
    return max(0.0, min(1.0, numerator / denominator))


def _direct_trace_distribution(
    name: str,
    values: Tensor,
    valid: Tensor,
) -> dict[str, float | int | None]:
    """Summarize one pose-wise direct-drift metric for compact JSON tracing."""
    selected = values.detach()[valid].to(device="cpu", dtype=torch.float64)
    count = int(selected.numel())
    result: dict[str, float | int | None] = {
        f"{name}_sum": float(selected.sum().item()) if count else 0.0,
        f"{name}_valid_count": count,
    }
    if count:
        quantiles = torch.quantile(
            selected,
            torch.tensor((0.05, 0.50, 0.95, 0.99), dtype=torch.float64),
        ).tolist()
    else:
        quantiles = (None, None, None, None)
    for suffix, value in zip(("p05", "p50", "p95", "p99"), quantiles):
        result[f"{name}_{suffix}"] = None if value is None else float(value)
    return result


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


@dataclass(frozen=True)
class UnifiedGuidanceConfig:
    """Numerical contract for experimental operator-split ODE guidance."""

    start_t: float = 0.5
    ramp_power: float = 1.0
    softcore_start: float = 1.5
    softcore_end: float = 0.75
    max_atom_force: float = 20.0
    max_translation_velocity: float = 5.0
    max_angular_velocity: float = 5.0
    max_atom_displacement: float = 0.25
    backtrack_factor: float = 0.5
    max_backtracks: int = 8
    descent_atol: float = 1e-6
    descent_rtol: float = 1e-6
    energy: GuidanceEnergyConfig = field(default_factory=GuidanceEnergyConfig)

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
            "max_atom_displacement",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0 < self.backtrack_factor < 1:
            raise ValueError("backtrack_factor must be in (0,1)")
        if self.max_backtracks < 0:
            raise ValueError("max_backtracks must be non-negative")
        if self.descent_atol < 0 or self.descent_rtol < 0:
            raise ValueError("descent tolerances must be non-negative")


def _fragment_pose_coords(
    T_flat: Tensor,
    q_flat: Tensor,
    local_pos: Tensor,
    fragment_id: Tensor,
    *,
    batch_size: int,
    n_fragments: int,
) -> Tensor:
    """Reconstruct ``[B,N,3]`` ligand coordinates from fragment SE(3) state."""
    n_atoms = int(local_pos.shape[0])
    if local_pos.shape != (n_atoms, 3):
        raise ValueError("local_pos must have shape [N,3]")
    if fragment_id.shape != (n_atoms,):
        raise ValueError("fragment_id must have shape [N]")
    if T_flat.shape != (batch_size * n_fragments, 3):
        raise ValueError("fragment translations do not match batch topology")
    if q_flat.shape != (batch_size * n_fragments, 4):
        raise ValueError("fragment rotations do not match batch topology")
    fragment_id = fragment_id.to(device=T_flat.device, dtype=torch.long)
    local_pos = local_pos.to(device=T_flat.device, dtype=T_flat.dtype)
    T = T_flat.view(batch_size, n_fragments, 3)
    R = quaternion_to_matrix(q_flat).view(batch_size, n_fragments, 3, 3)
    rotated = torch.einsum("bnij,nj->bni", R[:, fragment_id], local_pos)
    return rotated + T[:, fragment_id]


class UnifiedGuidance:
    """Two explicit experimental couplings for the same ``GuidanceEnergy``.

    ``correct`` is the bounded post-ODE operator-split corrector.  It proposes
    a guidance-only fragment step after the learned ODE proposal, then
    independently accepts, shrinks, or rejects each pose using finite,
    energy-descent, and maximum atom-displacement gates.

    ``direct_velocity`` instead adds a normalized guidance drift to the learned
    ODE right-hand side.  It uses a single pose-wise scale in induced atom
    velocity space so fragment translation and rotation remain coupled.  This
    is a generative control field, not molecular dynamics or physical time.
    """

    def __init__(
        self,
        system: PhysicalSystem,
        config: UnifiedGuidanceConfig = UnifiedGuidanceConfig(),
    ) -> None:
        self.system = system
        self.config = config
        self.n_atoms = system.topology.num_atoms
        self.n_fragments = int(system.topology.fragment_id.max().item()) + 1
        self.last_components: dict[str, Tensor] | None = None
        self._stats: dict[str, float | int | None] = {
            "steps_attempted": 0,
            "pose_corrections_attempted": 0,
            "pose_corrections_accepted": 0,
            "pose_corrections_rejected": 0,
            "total_backtracks": 0,
            "nonfinite_base_poses": 0,
            "nonfinite_trials": 0,
            "batched_energy_evaluations": 0,
            "pose_energy_evaluations": 0,
            "max_accepted_atom_displacement": 0.0,
            "min_accepted_energy_drop": None,
            "direct_steps_attempted": 0,
            "direct_pose_evaluations": 0,
            "direct_pose_applied": 0,
            "direct_nonfinite_poses": 0,
            "direct_zero_raw_direction_poses": 0,
            "direct_zero_reference_velocity_poses": 0,
            "direct_batched_energy_evaluations": 0,
            "direct_pose_energy_evaluations": 0,
            "direct_reference_atom_speed_rms_sum": 0.0,
            "direct_model_atom_speed_rms_sum": 0.0,
            "direct_raw_atom_speed_rms_sum": 0.0,
            "direct_applied_atom_speed_rms_sum": 0.0,
            "direct_total_atom_speed_rms_sum": 0.0,
            "direct_atom_speed_rms_valid_count": 0,
            "direct_applied_to_model_rms_ratio_sum": 0.0,
            "direct_applied_to_model_rms_ratio_valid_count": 0,
            "direct_model_guide_cosine_sum": 0.0,
            "direct_model_guide_cosine_valid_count": 0,
            "direct_guide_parallel_to_model_ratio_sum": 0.0,
            "direct_guide_parallel_to_model_ratio_valid_count": 0,
            "direct_model_rms_path_proxy_sum": 0.0,
            "direct_applied_rms_path_proxy_sum": 0.0,
            "direct_total_rms_path_proxy_sum": 0.0,
            "direct_cap_scale_sum": 0.0,
            "direct_cap_scale_valid_count": 0,
            "direct_translation_cap_trigger_count": 0,
            "direct_angular_cap_trigger_count": 0,
            "direct_displacement_cap_trigger_count": 0,
            "direct_any_cap_trigger_count": 0,
            "direct_multiple_cap_trigger_count": 0,
            "direct_max_reference_atom_speed_rms": 0.0,
            "direct_max_model_atom_speed_rms": 0.0,
            "direct_max_raw_atom_speed_rms": 0.0,
            "direct_max_applied_atom_speed_rms": 0.0,
            "direct_max_total_atom_speed_rms": 0.0,
            "direct_max_normalization_factor": 0.0,
            "direct_max_translation_velocity": 0.0,
            "direct_max_angular_velocity": 0.0,
            "direct_max_estimated_atom_displacement": 0.0,
        }
        self._direct_step_trace: list[dict[str, float | int | None]] = []

    def _progress(self, t: float) -> float:
        if t < self.config.start_t:
            return 0.0
        value = (float(t) - self.config.start_t) / (1.0 - self.config.start_t)
        return max(0.0, min(1.0, value))

    def _energy_config(self, progress: float) -> GuidanceEnergyConfig:
        softcore = self.config.softcore_start + progress * (
            self.config.softcore_end - self.config.softcore_start
        )
        return replace(
            self.config.energy,
            physical=replace(self.config.energy.physical, softcore=softcore),
        )

    def _direction(
        self,
        coords: Tensor,
        centers: Tensor,
        *,
        progress: float,
        apply_schedule_and_caps: bool = True,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        energy_config = self._energy_config(progress)
        with torch.enable_grad():
            variable = coords.detach().requires_grad_(True)
            components = guidance_energy(variable, self.system, energy_config)
            total = components["total"]
            atom_force = -torch.autograd.grad(total.sum(), variable)[0]
        finite = (
            torch.isfinite(total)
            & torch.isfinite(atom_force).all(dim=(1, 2))
            & torch.isfinite(variable).all(dim=(1, 2))
            & torch.isfinite(centers).all(dim=(1, 2))
        )
        safe_force = torch.where(
            finite.view(-1, 1, 1),
            atom_force,
            torch.zeros_like(atom_force),
        )
        safe_coords = torch.where(
            finite.view(-1, 1, 1),
            variable.detach(),
            torch.zeros_like(variable),
        )
        safe_centers = torch.where(
            finite.view(-1, 1, 1),
            centers,
            torch.zeros_like(centers),
        )
        safe_force = _clip_vectors(safe_force, self.config.max_atom_force)
        translation, angular = project_atom_forces(
            safe_force,
            safe_coords,
            safe_centers,
            self.system.topology.fragment_id,
            self.system.topology.mass,
        )
        if apply_schedule_and_caps:
            ramp = progress**self.config.ramp_power
            translation = ramp * _clip_vectors(
                translation,
                self.config.max_translation_velocity,
            )
            angular = ramp * _clip_vectors(
                angular,
                self.config.max_angular_velocity,
            )
        translation = torch.where(
            finite.view(-1, 1, 1),
            translation,
            torch.zeros_like(translation),
        )
        angular = torch.where(
            finite.view(-1, 1, 1),
            angular,
            torch.zeros_like(angular),
        )
        self.last_components = {
            name: value.detach() for name, value in components.items()
        }
        return translation, angular, total.detach(), finite.detach()

    def direct_velocity(
        self,
        atom_pos_flat: Tensor,
        centers_flat: Tensor,
        learned_translation_flat: Tensor,
        learned_angular_flat: Tensor,
        frag_sizes_flat: Tensor,
        *,
        t_start: float,
        t_end: float,
        strength: float,
    ) -> tuple[Tensor, Tensor]:
        """Return the actual drift increment added to the learned ODE velocity.

        Raw Newton--Euler guidance is normalized per pose against the learned
        fragment field after both are mapped into atom velocity space.  One
        scalar multiplies translation and rotation together, preserving their
        coupled rigid-body direction.  ``strength`` is applied exactly once
        here; the sampler integrates the returned velocity exactly once over
        its normal ``dt``.
        """
        if strength < 0:
            raise ValueError("unified direct-guidance strength must be non-negative")
        ramp = _interval_average_ramp(
            t_start,
            t_end,
            guidance_start=self.config.start_t,
            power=self.config.ramp_power,
        )
        if strength == 0 or ramp == 0:
            return torch.zeros_like(centers_flat), torch.zeros_like(centers_flat)
        if atom_pos_flat.shape[0] % self.n_atoms:
            raise ValueError("flattened direct-guidance coordinates are not whole poses")
        if centers_flat.shape[0] % self.n_fragments:
            raise ValueError("flattened direct-guidance centers are not whole poses")
        batch_size = atom_pos_flat.shape[0] // self.n_atoms
        if centers_flat.shape != (batch_size * self.n_fragments, 3):
            raise ValueError("direct-guidance fragment centers do not match atom batch")
        expected_fragment_shape = (batch_size * self.n_fragments, 3)
        if learned_translation_flat.shape != expected_fragment_shape:
            raise ValueError("learned translation does not match direct-guidance batch")
        if learned_angular_flat.shape != expected_fragment_shape:
            raise ValueError("learned angular velocity does not match direct-guidance batch")
        if frag_sizes_flat.shape != (batch_size * self.n_fragments,):
            raise ValueError("fragment sizes do not match direct-guidance batch")
        if not bool(torch.isfinite(learned_translation_flat).all()) or not bool(
            torch.isfinite(learned_angular_flat).all()
        ):
            raise FloatingPointError("learned ODE velocity is non-finite before guidance")

        coords = atom_pos_flat.detach().view(batch_size, self.n_atoms, 3)
        centers = centers_flat.detach().view(batch_size, self.n_fragments, 3)
        learned_translation = learned_translation_flat.detach().view(
            batch_size,
            self.n_fragments,
            3,
        )
        learned_angular = learned_angular_flat.detach().view(
            batch_size,
            self.n_fragments,
            3,
        )
        movable_rotation = frag_sizes_flat.view(batch_size, self.n_fragments) > 1
        learned_angular = torch.where(
            movable_rotation.unsqueeze(-1),
            learned_angular,
            torch.zeros_like(learned_angular),
        )

        active_left = max(float(t_start), self.config.start_t)
        active_midpoint = 0.5 * (active_left + float(t_end))
        progress = self._progress(active_midpoint)
        raw_translation, raw_angular, _, finite = self._direction(
            coords,
            centers,
            progress=progress,
            apply_schedule_and_caps=False,
        )
        raw_angular = torch.where(
            movable_rotation.unsqueeze(-1),
            raw_angular,
            torch.zeros_like(raw_angular),
        )

        fragment_id = self.system.topology.fragment_id
        learned_atom_velocity = _induced_atom_velocity(
            learned_translation,
            learned_angular,
            coords,
            centers,
            fragment_id,
        )
        raw_atom_velocity = _induced_atom_velocity(
            raw_translation,
            raw_angular,
            coords,
            centers,
            fragment_id,
        )
        reference_rms = learned_atom_velocity.square().sum(dim=-1).mean(dim=1).sqrt()
        raw_rms = raw_atom_velocity.square().sum(dim=-1).mean(dim=1).sqrt()
        eps = 1e-8
        has_reference = reference_rms > eps
        has_direction = raw_rms > eps
        normalization = torch.where(
            finite & has_reference & has_direction,
            reference_rms / raw_rms.clamp_min(eps),
            torch.zeros_like(raw_rms),
        )
        coefficient = float(strength) * float(ramp) * normalization
        translation = coefficient.view(-1, 1, 1) * raw_translation
        angular = coefficient.view(-1, 1, 1) * raw_angular

        # Apply all post-normalization limits through one positive pose scalar.
        # This keeps the Newton--Euler translation/rotation coupling intact.
        translation_max = translation.norm(dim=-1).amax(dim=1)
        angular_max = angular.norm(dim=-1).amax(dim=1)
        fragment_id_d = fragment_id.to(device=coords.device, dtype=torch.long)
        lever_radius = (coords - centers[:, fragment_id_d]).norm(dim=-1)
        endpoint_speed_bound = (
            translation.norm(dim=-1)[:, fragment_id_d]
            + angular.norm(dim=-1)[:, fragment_id_d] * lever_radius
        ).amax(dim=1)
        dt = float(t_end) - float(t_start)
        cap_eligible = finite & has_reference & has_direction
        translation_cap_trigger = cap_eligible & (
            translation_max > float(self.config.max_translation_velocity)
        )
        angular_cap_trigger = cap_eligible & (
            angular_max > float(self.config.max_angular_velocity)
        )
        displacement_cap_trigger = cap_eligible & (
            dt * endpoint_speed_bound > float(self.config.max_atom_displacement)
        )
        cap_trigger_count = (
            translation_cap_trigger.to(torch.int8)
            + angular_cap_trigger.to(torch.int8)
            + displacement_cap_trigger.to(torch.int8)
        )
        any_cap_trigger = cap_trigger_count > 0
        multiple_cap_trigger = cap_trigger_count > 1

        cap = torch.ones_like(reference_rms)
        cap = torch.minimum(
            cap,
            float(self.config.max_translation_velocity)
            / translation_max.clamp_min(eps),
        )
        cap = torch.minimum(
            cap,
            float(self.config.max_angular_velocity) / angular_max.clamp_min(eps),
        )
        cap = torch.minimum(
            cap,
            float(self.config.max_atom_displacement)
            / (dt * endpoint_speed_bound).clamp_min(eps),
        ).clamp(max=1.0)
        translation = cap.view(-1, 1, 1) * translation
        angular = cap.view(-1, 1, 1) * angular
        translation = torch.where(
            finite.view(-1, 1, 1), translation, torch.zeros_like(translation)
        )
        angular = torch.where(finite.view(-1, 1, 1), angular, torch.zeros_like(angular))

        applied_atom_velocity = _induced_atom_velocity(
            translation,
            angular,
            coords,
            centers,
            fragment_id,
        )
        applied_rms = applied_atom_velocity.square().sum(dim=-1).mean(dim=1).sqrt()
        total_atom_velocity = learned_atom_velocity + applied_atom_velocity
        total_rms = total_atom_velocity.square().sum(dim=-1).mean(dim=1).sqrt()
        ratio_valid = finite & has_reference
        applied_to_model_ratio = torch.where(
            ratio_valid,
            applied_rms / reference_rms.clamp_min(eps),
            torch.zeros_like(reference_rms),
        )
        model_squared_norm = learned_atom_velocity.square().sum(dim=(1, 2))
        applied_squared_norm = applied_atom_velocity.square().sum(dim=(1, 2))
        model_guide_dot = (learned_atom_velocity * applied_atom_velocity).sum(dim=(1, 2))
        cosine_valid = finite & has_reference & (applied_rms > eps)
        model_guide_cosine = torch.where(
            cosine_valid,
            model_guide_dot
            / (
                model_squared_norm.clamp_min(eps).sqrt()
                * applied_squared_norm.clamp_min(eps).sqrt()
            ),
            torch.zeros_like(reference_rms),
        ).clamp(min=-1.0, max=1.0)
        guide_parallel_to_model_ratio = torch.where(
            ratio_valid,
            model_guide_dot / model_squared_norm.clamp_min(eps),
            torch.zeros_like(reference_rms),
        )
        translation_max_after = translation.norm(dim=-1).amax(dim=1)
        angular_max_after = angular.norm(dim=-1).amax(dim=1)
        endpoint_bound_after = dt * (
            translation.norm(dim=-1)[:, fragment_id_d]
            + angular.norm(dim=-1)[:, fragment_id_d] * lever_radius
        ).amax(dim=1)
        applied = finite & (applied_rms > eps)

        self._stats["direct_steps_attempted"] += 1
        self._stats["direct_pose_evaluations"] += batch_size
        self._stats["direct_pose_applied"] += int(applied.sum().item())
        self._stats["direct_nonfinite_poses"] += int((~finite).sum().item())
        self._stats["direct_zero_raw_direction_poses"] += int(
            (finite & ~has_direction).sum().item()
        )
        self._stats["direct_zero_reference_velocity_poses"] += int(
            (finite & ~has_reference).sum().item()
        )
        self._stats["direct_batched_energy_evaluations"] += 1
        self._stats["direct_pose_energy_evaluations"] += batch_size
        finite_reference = torch.where(finite, reference_rms, torch.zeros_like(reference_rms))
        finite_raw = torch.where(finite, raw_rms, torch.zeros_like(raw_rms))
        finite_applied = torch.where(finite, applied_rms, torch.zeros_like(applied_rms))
        finite_total = torch.where(finite, total_rms, torch.zeros_like(total_rms))
        finite_cap = torch.where(finite, cap, torch.zeros_like(cap))
        self._stats["direct_reference_atom_speed_rms_sum"] += float(
            finite_reference.sum().item()
        )
        self._stats["direct_model_atom_speed_rms_sum"] += float(
            finite_reference.sum().item()
        )
        self._stats["direct_raw_atom_speed_rms_sum"] += float(finite_raw.sum().item())
        self._stats["direct_applied_atom_speed_rms_sum"] += float(
            finite_applied.sum().item()
        )
        self._stats["direct_total_atom_speed_rms_sum"] += float(finite_total.sum().item())
        self._stats["direct_atom_speed_rms_valid_count"] += int(finite.sum().item())
        self._stats["direct_applied_to_model_rms_ratio_sum"] += float(
            applied_to_model_ratio[ratio_valid].sum().item()
        )
        self._stats["direct_applied_to_model_rms_ratio_valid_count"] += int(
            ratio_valid.sum().item()
        )
        self._stats["direct_model_guide_cosine_sum"] += float(
            model_guide_cosine[cosine_valid].sum().item()
        )
        self._stats["direct_model_guide_cosine_valid_count"] += int(
            cosine_valid.sum().item()
        )
        self._stats["direct_guide_parallel_to_model_ratio_sum"] += float(
            guide_parallel_to_model_ratio[ratio_valid].sum().item()
        )
        self._stats["direct_guide_parallel_to_model_ratio_valid_count"] += int(
            ratio_valid.sum().item()
        )
        self._stats["direct_model_rms_path_proxy_sum"] += float(
            (dt * finite_reference).sum().item()
        )
        self._stats["direct_applied_rms_path_proxy_sum"] += float(
            (dt * finite_applied).sum().item()
        )
        self._stats["direct_total_rms_path_proxy_sum"] += float(
            (dt * finite_total).sum().item()
        )
        self._stats["direct_cap_scale_sum"] += float(finite_cap.sum().item())
        self._stats["direct_cap_scale_valid_count"] += int(finite.sum().item())
        cap_counts = {
            "direct_translation_cap_trigger_count": translation_cap_trigger,
            "direct_angular_cap_trigger_count": angular_cap_trigger,
            "direct_displacement_cap_trigger_count": displacement_cap_trigger,
            "direct_any_cap_trigger_count": any_cap_trigger,
            "direct_multiple_cap_trigger_count": multiple_cap_trigger,
        }
        for name, mask in cap_counts.items():
            self._stats[name] += int(mask.sum().item())
        maxima = {
            "direct_max_reference_atom_speed_rms": finite_reference.max(),
            "direct_max_model_atom_speed_rms": finite_reference.max(),
            "direct_max_raw_atom_speed_rms": finite_raw.max(),
            "direct_max_applied_atom_speed_rms": finite_applied.max(),
            "direct_max_total_atom_speed_rms": finite_total.max(),
            "direct_max_normalization_factor": normalization.max(),
            "direct_max_translation_velocity": translation_max_after.max(),
            "direct_max_angular_velocity": angular_max_after.max(),
            "direct_max_estimated_atom_displacement": endpoint_bound_after.max(),
        }
        for name, value in maxima.items():
            self._stats[name] = max(float(self._stats[name]), float(value.item()))
        finite_count = int(finite.sum().item())
        self._direct_step_trace.append(
            {
                "t": float(t_start),
                "t_end": float(t_end),
                "dt": dt,
                "ramp": float(ramp),
                "eta": float(strength),
                "pose_count": batch_size,
                "finite_count": finite_count,
                "applied_count": int(applied.sum().item()),
                "model_atom_speed_rms_sum": float(finite_reference.sum().item()),
                "applied_atom_speed_rms_sum": float(finite_applied.sum().item()),
                "total_atom_speed_rms_sum": float(finite_total.sum().item()),
                "atom_speed_rms_valid_count": finite_count,
                "model_rms_path_proxy_sum": float((dt * finite_reference).sum().item()),
                "applied_rms_path_proxy_sum": float((dt * finite_applied).sum().item()),
                "total_rms_path_proxy_sum": float((dt * finite_total).sum().item()),
                "translation_cap_trigger_count": int(
                    translation_cap_trigger.sum().item()
                ),
                "angular_cap_trigger_count": int(angular_cap_trigger.sum().item()),
                "displacement_cap_trigger_count": int(
                    displacement_cap_trigger.sum().item()
                ),
                "any_cap_trigger_count": int(any_cap_trigger.sum().item()),
                "multiple_cap_trigger_count": int(multiple_cap_trigger.sum().item()),
                **_direct_trace_distribution(
                    "applied_to_model_rms_ratio",
                    applied_to_model_ratio,
                    ratio_valid,
                ),
                **_direct_trace_distribution(
                    "model_guide_cosine",
                    model_guide_cosine,
                    cosine_valid,
                ),
                **_direct_trace_distribution(
                    "guide_parallel_to_model_ratio",
                    guide_parallel_to_model_ratio,
                    ratio_valid,
                ),
                **_direct_trace_distribution("cap_scale", cap, finite),
            }
        )
        return translation.reshape_as(centers_flat), angular.reshape_as(centers_flat)

    def _total(self, coords: Tensor, *, progress: float) -> Tensor:
        with torch.no_grad():
            total = guidance_energy(
                coords,
                self.system,
                self._energy_config(progress),
            )["total"]
        return total

    def correct(
        self,
        T_flat: Tensor,
        q_flat: Tensor,
        local_pos: Tensor,
        fragment_id: Tensor,
        frag_sizes_flat: Tensor,
        *,
        dt: Tensor | float,
        t: float,
        scale: float,
    ) -> tuple[Tensor, Tensor]:
        """Apply one trust-region guidance substep after a learned ODE step."""
        if scale < 0:
            raise ValueError("unified guidance scale must be non-negative")
        progress = self._progress(t)
        if scale == 0 or progress == 0:
            return T_flat, q_flat
        if T_flat.shape[0] % self.n_fragments:
            raise ValueError("fragment state is not a whole-pose batch")
        batch_size = T_flat.shape[0] // self.n_fragments
        if int(local_pos.shape[0]) != self.n_atoms:
            raise ValueError("unified guidance ligand atom count mismatch")

        base_coords = _fragment_pose_coords(
            T_flat,
            q_flat,
            local_pos,
            fragment_id,
            batch_size=batch_size,
            n_fragments=self.n_fragments,
        )
        base_centers = T_flat.view(batch_size, self.n_fragments, 3)
        translation, angular, base_total, direction_finite = self._direction(
            base_coords,
            base_centers,
            progress=progress,
        )
        self._stats["steps_attempted"] += 1
        self._stats["pose_corrections_attempted"] += batch_size
        self._stats["batched_energy_evaluations"] += 1
        self._stats["pose_energy_evaluations"] += batch_size
        self._stats["nonfinite_base_poses"] += int((~direction_finite).sum().item())

        accepted = torch.zeros(batch_size, dtype=torch.bool, device=T_flat.device)
        pending = direction_finite.clone()
        backtrack_count = torch.zeros(
            batch_size,
            dtype=torch.long,
            device=T_flat.device,
        )
        accepted_displacement = base_total.new_zeros(batch_size)
        accepted_drop = base_total.new_zeros(batch_size)
        base_T = T_flat.view(batch_size, self.n_fragments, 3)
        base_q = q_flat.view(batch_size, self.n_fragments, 4)
        sizes_by_pose = frag_sizes_flat.view(batch_size, self.n_fragments)
        final_T = base_T.clone()
        final_q = base_q.clone()

        for backtrack in range(self.config.max_backtracks + 1):
            pending_index = torch.nonzero(pending, as_tuple=False).flatten()
            if pending_index.numel() == 0:
                break
            pending_size = int(pending_index.numel())
            alpha = self.config.backtrack_factor**backtrack
            trial_T_flat, trial_q_flat = integrate_se3_step(
                base_T.index_select(0, pending_index).reshape(
                    pending_size * self.n_fragments,
                    3,
                ),
                base_q.index_select(0, pending_index).reshape(
                    pending_size * self.n_fragments,
                    4,
                ),
                translation.index_select(0, pending_index).reshape(
                    pending_size * self.n_fragments,
                    3,
                ),
                angular.index_select(0, pending_index).reshape(
                    pending_size * self.n_fragments,
                    3,
                ),
                torch.as_tensor(dt, device=T_flat.device, dtype=T_flat.dtype)
                * float(scale)
                * alpha,
                frag_sizes=sizes_by_pose.index_select(0, pending_index).reshape(-1),
            )
            trial_coords = _fragment_pose_coords(
                trial_T_flat,
                trial_q_flat,
                local_pos,
                fragment_id,
                batch_size=pending_size,
                n_fragments=self.n_fragments,
            )
            trial_total = self._total(trial_coords, progress=progress)
            self._stats["batched_energy_evaluations"] += 1
            self._stats["pose_energy_evaluations"] += pending_size
            pending_base_coords = base_coords.index_select(0, pending_index)
            displacement = (trial_coords - pending_base_coords).norm(dim=-1).amax(dim=1)
            finite = (
                torch.isfinite(trial_total)
                & torch.isfinite(trial_coords).all(dim=(1, 2))
                & torch.isfinite(displacement)
            )
            self._stats["nonfinite_trials"] += int((~finite).sum().item())
            pending_base_total = base_total.index_select(0, pending_index)
            tolerance = (
                self.config.descent_atol
                + self.config.descent_rtol * pending_base_total.abs()
            )
            descends = trial_total <= pending_base_total + tolerance
            within_trust = displacement <= self.config.max_atom_displacement
            take_local = finite & descends & within_trust
            if bool(take_local.any()):
                accepted_index = pending_index[take_local]
                trial_T = trial_T_flat.view(pending_size, self.n_fragments, 3)
                trial_q = trial_q_flat.view(pending_size, self.n_fragments, 4)
                final_T[accepted_index] = trial_T[take_local]
                final_q[accepted_index] = trial_q[take_local]
                accepted[accepted_index] = True
                pending[accepted_index] = False
                backtrack_count[accepted_index] = backtrack
                accepted_displacement[accepted_index] = displacement[take_local]
                accepted_drop[accepted_index] = (
                    pending_base_total[take_local] - trial_total[take_local]
                )

        accepted_count = int(accepted.sum().item())
        rejected_count = batch_size - accepted_count
        backtrack_count[pending] = self.config.max_backtracks
        self._stats["pose_corrections_accepted"] += accepted_count
        self._stats["pose_corrections_rejected"] += rejected_count
        self._stats["total_backtracks"] += int(backtrack_count.sum().item())
        if accepted_count:
            maximum = float(accepted_displacement[accepted].max().item())
            self._stats["max_accepted_atom_displacement"] = max(
                float(self._stats["max_accepted_atom_displacement"]),
                maximum,
            )
            minimum_drop = float(accepted_drop[accepted].min().item())
            previous = self._stats["min_accepted_energy_drop"]
            self._stats["min_accepted_energy_drop"] = (
                minimum_drop
                if previous is None
                else min(float(previous), minimum_drop)
            )
        return final_T.reshape_as(T_flat), final_q.reshape_as(q_flat)

    def diagnostics(self) -> dict[str, float | int | None]:
        """Return only JSON-serializable cumulative scalar diagnostics."""
        return dict(self._stats)

    def direct_step_trace(self) -> list[dict[str, float | int | None]]:
        """Return a defensive copy of compact per-active-step drift telemetry."""
        return [dict(step) for step in self._direct_step_trace]


__all__ = [
    "GuidanceEnergyConfig",
    "PhysicalGuidance",
    "PhysicalGuidanceConfig",
    "UnifiedGuidance",
    "UnifiedGuidanceConfig",
    "guidance_energy",
    "project_atom_forces",
]
