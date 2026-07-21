"""Flow-matching utilities on fragment poses in SE(3).

This module uses a single, explicit convention throughout:
- Angular velocities are global-frame vectors.
- Pose integration uses left multiplication:
  ``q_{t + dt} = exp(dt * omega) ⊗ q_t``.

That matches the active rotation convention in :mod:`effdock.geometry.se3` and is
the natural choice for an SE(3)-equivariant network that predicts vectors in
the ambient coordinate frame.
"""

from __future__ import annotations

import torch

from .se3 import (
    axis_angle_to_quaternion,
    normalize_quaternion,
    quaternion_inverse,
    quaternion_multiply,
    quaternion_slerp,
    quaternion_to_axis_angle,
    sample_uniform_quaternion,
    standardize_quaternion,
)


def sample_prior_translations(
    num_fragments: int,
    pocket_center: torch.Tensor,
    sigma: float,
    *,
    generator: torch.Generator | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Sample Gaussian translation priors around a pocket center.

    Args:
        num_fragments: Number of fragment poses to sample.
        pocket_center: Tensor of shape ``[3]`` or ``[num_fragments, 3]`` in
            Angstroms.
        sigma: Standard deviation of the Gaussian translation prior in Angstroms.
        generator: Optional torch random generator.
        device: Optional output device. Defaults to ``pocket_center.device``.
        dtype: Optional output dtype. Defaults to ``pocket_center.dtype``.

    Returns:
        Tensor of shape ``[num_fragments, 3]``.
    """
    if pocket_center.shape[-1] != 3:
        raise ValueError(f"Expected pocket_center with last dim 3, got {pocket_center.shape}.")

    if device is None:
        device = pocket_center.device
    if dtype is None:
        dtype = pocket_center.dtype

    center = pocket_center.to(device=device, dtype=dtype)
    center = torch.broadcast_to(center, (num_fragments, 3))
    noise = sigma * torch.randn(
        num_fragments,
        3,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    return center + noise


def sample_prior_rotations(
    num_fragments: int,
    *,
    frag_sizes: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Sample rotation priors as quaternions.

    Multi-atom fragments are sampled uniformly on SO(3). Single-atom fragments
    are clamped to identity because their orientation is physically irrelevant
    and should not become a nuisance variable for the model.

    Args:
        num_fragments: Number of fragment poses to sample.
        frag_sizes: Optional tensor of shape ``[num_fragments]``. Entries
            ``<= 1`` are treated as single-atom fragments.
        generator: Optional torch random generator.
        device: Optional output device.
        dtype: Optional floating output dtype.

    Returns:
        Tensor of shape ``[num_fragments, 4]``.
    """
    q0 = sample_uniform_quaternion(
        num_fragments,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    return _mask_single_atom_quaternions(q0, frag_sizes=frag_sizes)


def sample_prior_poses(
    num_fragments: int,
    pocket_center: torch.Tensor,
    translation_sigma: float,
    *,
    frag_sizes: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample translation and rotation priors for fragment poses.

    Args:
        num_fragments: Number of fragment poses to sample.
        pocket_center: Tensor of shape ``[3]`` or ``[num_fragments, 3]``.
        translation_sigma: Gaussian translation prior scale in Angstroms.
        frag_sizes: Optional tensor of shape ``[num_fragments]`` used to clamp
            single-atom fragment rotations to identity.
        generator: Optional torch random generator.
        device: Optional output device.
        dtype: Optional output dtype.

    Returns:
        Tuple ``(T_0, q_0)`` with shapes ``[num_fragments, 3]`` and
        ``[num_fragments, 4]``.
    """
    T_0 = sample_prior_translations(
        num_fragments,
        pocket_center,
        translation_sigma,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    q_0 = sample_prior_rotations(
        num_fragments,
        frag_sizes=frag_sizes,
        generator=generator,
        device=device,
        dtype=dtype if dtype is not None else T_0.dtype,
    )
    return T_0, q_0


def interpolate_translations(
    T_0: torch.Tensor,
    T_1: torch.Tensor,
    t: torch.Tensor | float,
) -> torch.Tensor:
    """Linearly interpolate fragment translations.

    Args:
        T_0: Start translations of shape ``[..., 3]``.
        T_1: End translations of shape ``[..., 3]``.
        t: Interpolation time in ``[0, 1]``, broadcastable to the batch shape.

    Returns:
        Tensor of shape ``[..., 3]`` with
        ``T_t = (1 - t) * T_0 + t * T_1``.
    """
    if T_0.shape[-1] != 3 or T_1.shape[-1] != 3:
        raise ValueError(
            f"Expected translation tensors with last dim 3, got {T_0.shape} and {T_1.shape}."
        )

    t = _broadcast_last_dim_like(t, T_0)
    return (1.0 - t) * T_0 + t * T_1


def interpolate_rotations(
    q_0: torch.Tensor,
    q_1: torch.Tensor,
    t: torch.Tensor | float,
    *,
    frag_sizes: torch.Tensor | None = None,
) -> torch.Tensor:
    """Interpolate fragment rotations with SLERP.

    Single-atom fragments are forced to identity throughout the path.

    Args:
        q_0: Start quaternions of shape ``[..., 4]``.
        q_1: End quaternions of shape ``[..., 4]``.
        t: Interpolation time in ``[0, 1]``, broadcastable to the batch shape.
        frag_sizes: Optional fragment sizes used to identify single-atom
            fragments.

    Returns:
        Tensor of shape ``[..., 4]``.
    """
    q_0 = _mask_single_atom_quaternions(q_0, frag_sizes=frag_sizes)
    q_1 = _mask_single_atom_quaternions(q_1, frag_sizes=frag_sizes)
    return quaternion_slerp(q_0, q_1, t)


def interpolate_poses(
    T_0: torch.Tensor,
    q_0: torch.Tensor,
    T_1: torch.Tensor,
    q_1: torch.Tensor,
    t: torch.Tensor | float,
    *,
    frag_sizes: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Interpolate full fragment poses.

    Args:
        T_0: Start translations of shape ``[..., 3]``.
        q_0: Start quaternions of shape ``[..., 4]``.
        T_1: End translations of shape ``[..., 3]``.
        q_1: End quaternions of shape ``[..., 4]``.
        t: Interpolation time in ``[0, 1]``.
        frag_sizes: Optional fragment sizes used to clamp single-atom fragment
            rotations to identity.

    Returns:
        Tuple ``(T_t, q_t)``.
    """
    T_t = interpolate_translations(T_0, T_1, t)
    q_t = interpolate_rotations(q_0, q_1, t, frag_sizes=frag_sizes)
    return T_t, q_t


def compute_translation_velocity(T_0: torch.Tensor, T_1: torch.Tensor) -> torch.Tensor:
    """Compute the exact translation velocity for linear interpolation.

    For ``T_t = (1 - t) * T_0 + t * T_1``, the time derivative is constant:
    ``dT_t / dt = T_1 - T_0``.

    Args:
        T_0: Start translations of shape ``[..., 3]``.
        T_1: End translations of shape ``[..., 3]``.

    Returns:
        Tensor of shape ``[..., 3]``.
    """
    if T_0.shape[-1] != 3 or T_1.shape[-1] != 3:
        raise ValueError(
            f"Expected translation tensors with last dim 3, got {T_0.shape} and {T_1.shape}."
        )

    return T_1 - T_0


def compute_angular_velocity(
    q_0: torch.Tensor,
    q_1: torch.Tensor,
    *,
    frag_sizes: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute the exact global-frame angular velocity for shortest-path SLERP.

    The returned vector ``omega`` is defined by
    ``q_t = exp(t * omega) ⊗ q_0`` in quaternion form, so it is constant in
    ``t`` for the chosen interpolation path.

    Args:
        q_0: Start quaternions of shape ``[..., 4]``.
        q_1: End quaternions of shape ``[..., 4]``.
        frag_sizes: Optional fragment sizes. Entries ``<= 1`` receive zero
            angular velocity.

    Returns:
        Tensor of shape ``[..., 3]`` in radians per unit time.
    """
    if q_0.shape[-1] != 4 or q_1.shape[-1] != 4:
        raise ValueError(
            f"Expected quaternion tensors with last dim 4, got {q_0.shape} and {q_1.shape}."
        )

    q_0 = normalize_quaternion(q_0)
    q_1 = normalize_quaternion(q_1)
    q_0 = _mask_single_atom_quaternions(q_0, frag_sizes=frag_sizes)
    q_1 = _mask_single_atom_quaternions(q_1, frag_sizes=frag_sizes)

    q_1 = standardize_quaternion(q_1, reference=q_0)
    q_delta = quaternion_multiply(q_1, quaternion_inverse(q_0))
    omega = quaternion_to_axis_angle(q_delta, shortest_path=True)
    return _mask_single_atom_vectors(omega, frag_sizes=frag_sizes)


def compute_flow_matching_targets(
    T_0: torch.Tensor,
    q_0: torch.Tensor,
    T_1: torch.Tensor,
    q_1: torch.Tensor,
    t: torch.Tensor | float,
    *,
    frag_sizes: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Build interpolated states and exact flow targets for SE(3) fragments.

    Args:
        T_0: Start translations of shape ``[..., 3]``.
        q_0: Start quaternions of shape ``[..., 4]``.
        T_1: End translations of shape ``[..., 3]``.
        q_1: End quaternions of shape ``[..., 4]``.
        t: Interpolation time in ``[0, 1]``.
        frag_sizes: Optional fragment sizes used to clamp single-atom fragment
            rotations to identity and their angular velocities to zero.

    Returns:
        Dict with keys:
        - ``"T_t"``: interpolated translations, shape ``[..., 3]``
        - ``"q_t"``: interpolated quaternions, shape ``[..., 4]``
        - ``"v_t"``: translation velocity targets, shape ``[..., 3]``
        - ``"omega_t"``: angular velocity targets, shape ``[..., 3]``
    """
    T_t, q_t = interpolate_poses(T_0, q_0, T_1, q_1, t, frag_sizes=frag_sizes)
    v_t = compute_translation_velocity(T_0, T_1)
    omega_t = compute_angular_velocity(q_0, q_1, frag_sizes=frag_sizes)
    return {"T_t": T_t, "q_t": q_t, "v_t": v_t, "omega_t": omega_t}


def compute_vp_flow_targets(
    T_0: torch.Tensor,
    q_0: torch.Tensor,
    T_1: torch.Tensor,
    q_1: torch.Tensor,
    t: torch.Tensor | float,
    *,
    frag_sizes: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Build a cosine VP-style probability path and exact flow targets.

    The translation path is
        T_t = sin(pi t / 2) T_1 + cos(pi t / 2) T_0,
    where ``T_0`` is the Gaussian prior sample. Rotations follow the same
    eased time through SLERP. This keeps the existing ODE sampler and model
    outputs compatible while testing a diffusion-style curved path.
    """
    half_pi = torch.tensor(0.5 * torch.pi, device=T_1.device, dtype=T_1.dtype)
    t_b = _broadcast_last_dim_like(t, T_1)
    theta = half_pi * t_b
    alpha = torch.sin(theta)
    beta = torch.cos(theta)
    alpha_dot = half_pi * torch.cos(theta)
    beta_dot = -half_pi * torch.sin(theta)

    T_t = alpha * T_1 + beta * T_0
    v_t = alpha_dot * T_1 + beta_dot * T_0

    s = torch.sin(_broadcast_last_dim_like(t, q_1[..., :1]) * half_pi)
    q_t = interpolate_rotations(q_0, q_1, s, frag_sizes=frag_sizes)
    base_omega = compute_angular_velocity(q_0, q_1, frag_sizes=frag_sizes)
    ds_dt = half_pi * torch.cos(_broadcast_last_dim_like(t, base_omega))
    omega_t = ds_dt * base_omega
    return {"T_t": T_t, "q_t": q_t, "v_t": v_t, "omega_t": omega_t}


def compute_vp_score_targets(
    T_0: torch.Tensor,
    q_0: torch.Tensor,
    T_1: torch.Tensor,
    q_1: torch.Tensor,
    t: torch.Tensor | float,
    *,
    translation_sigma: torch.Tensor | float,
    frag_sizes: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Build VP noising states with normalized translation-noise targets.

    This uses the same clean-time convention as the flow models:
    ``t=0`` is the Gaussian prior and ``t=1`` is the crystal pose.

    Translation follows
        ``T_t = sin(pi t / 2) T_1 + cos(pi t / 2) T_0``,
    but the model target is the normalized Gaussian noise
        ``eps = T_0 / translation_sigma``.

    Rotation currently keeps the VP-SLERP angular velocity target. That makes
    this a translation score/noise baseline while preserving the existing
    SO(3) integration machinery.
    """
    half_pi = torch.tensor(0.5 * torch.pi, device=T_1.device, dtype=T_1.dtype)
    t_b = _broadcast_last_dim_like(t, T_1)
    theta = half_pi * t_b
    alpha = torch.sin(theta)
    beta = torch.cos(theta)

    T_t = alpha * T_1 + beta * T_0
    sigma = _broadcast_last_dim_like(translation_sigma, T_1).clamp_min(1e-6)
    eps_t = T_0 / sigma

    s = torch.sin(_broadcast_last_dim_like(t, q_1[..., :1]) * half_pi)
    q_t = interpolate_rotations(q_0, q_1, s, frag_sizes=frag_sizes)
    base_omega = compute_angular_velocity(q_0, q_1, frag_sizes=frag_sizes)
    ds_dt = half_pi * torch.cos(_broadcast_last_dim_like(t, base_omega))
    omega_t = ds_dt * base_omega
    return {"T_t": T_t, "q_t": q_t, "v_t": eps_t, "omega_t": omega_t}


def compute_vp_score_full_targets(
    T_0: torch.Tensor,
    T_1: torch.Tensor,
    q_1: torch.Tensor,
    t: torch.Tensor | float,
    *,
    translation_sigma: torch.Tensor | float,
    rot_noise: torch.Tensor,
    rot_sigma_max: float = 3.141592653589793,
    score_alpha_min: float = 0.0,
    frag_sizes: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Build VP denoising-score targets for translation and rotation.

    Translation predicts normalized Gaussian noise in the VP path. With
    ``score_alpha_min > 0``, the clean coefficient is lower-bounded so DDIM
    reconstruction remains numerically stable near t=0.

    Rotation predicts tangent-space noise from
    ``q_t = exp(sigma_rot(t) eps_rot) q_1`` with
    ``sigma_rot(t) = rot_sigma_max * beta(t)``.
    """
    t_T = _broadcast_last_dim_like(t, T_1)
    alpha, beta = vp_score_alpha_beta(t_T, score_alpha_min=score_alpha_min)

    T_t = alpha * T_1 + beta * T_0
    sigma = _broadcast_last_dim_like(translation_sigma, T_1).clamp_min(1e-6)
    eps_t = T_0 / sigma

    t_R = _broadcast_last_dim_like(t, rot_noise)
    _, beta_R = vp_score_alpha_beta(t_R, score_alpha_min=score_alpha_min)
    sigma_rot = float(rot_sigma_max) * beta_R
    rot_noise = _mask_single_atom_vectors(rot_noise, frag_sizes=frag_sizes)
    delta_q = axis_angle_to_quaternion(sigma_rot * rot_noise)
    q_t = quaternion_multiply(delta_q, normalize_quaternion(q_1))
    q_t = standardize_quaternion(q_t, reference=q_1)
    q_t = _mask_single_atom_quaternions(q_t, frag_sizes=frag_sizes)
    return {"T_t": T_t, "q_t": q_t, "v_t": eps_t, "omega_t": rot_noise}


def vp_score_alpha_beta(
    t: torch.Tensor | float,
    *,
    score_alpha_min: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return VP score path coefficients with optional finite-SNR floor."""
    if not torch.is_tensor(t):
        t = torch.as_tensor(t, dtype=torch.float32)
    half_pi = torch.tensor(0.5 * torch.pi, device=t.device, dtype=t.dtype)
    base = torch.sin(half_pi * t)
    amin = float(score_alpha_min)
    if amin < 0.0 or amin >= 1.0:
        raise ValueError(f"score_alpha_min must be in [0, 1), got {score_alpha_min}.")
    if amin == 0.0:
        alpha = base
    else:
        alpha = torch.sqrt(amin * amin + (1.0 - amin * amin) * base.square())
    beta = torch.sqrt((1.0 - alpha.square()).clamp_min(0.0))
    return alpha, beta


def integrate_se3_step(
    T: torch.Tensor,
    q: torch.Tensor,
    v: torch.Tensor,
    omega: torch.Tensor,
    dt: torch.Tensor | float,
    *,
    frag_sizes: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Advance fragment poses by one ODE step.

    Translation uses explicit Euler because the vector field is Euclidean.
    Rotation uses the exact exponential-map update for a constant angular
    velocity over the step:
    ``q_{t + dt} = exp(dt * omega) ⊗ q_t``.

    Args:
        T: Current translations of shape ``[..., 3]``.
        q: Current quaternions of shape ``[..., 4]``.
        v: Global translation velocities of shape ``[..., 3]``.
        omega: Global angular velocities of shape ``[..., 3]``.
        dt: Step size, broadcastable to the batch shape.
        frag_sizes: Optional fragment sizes. Entries ``<= 1`` keep identity
            orientation regardless of the incoming quaternion.

    Returns:
        Tuple ``(T_next, q_next)``.
    """
    if T.shape[-1] != 3 or v.shape[-1] != 3 or omega.shape[-1] != 3:
        raise ValueError(
            f"Expected translation/vector tensors with last dim 3, got {T.shape}, {v.shape}, {omega.shape}."
        )
    if q.shape[-1] != 4:
        raise ValueError(f"Expected quaternion tensor with last dim 4, got {q.shape}.")

    q = normalize_quaternion(q)
    q = _mask_single_atom_quaternions(q, frag_sizes=frag_sizes)
    omega = _mask_single_atom_vectors(omega, frag_sizes=frag_sizes)

    dt_T = _broadcast_last_dim_like(dt, T)
    dt_omega = _broadcast_last_dim_like(dt, omega)

    T_next = T + dt_T * v
    delta_q = axis_angle_to_quaternion(dt_omega * omega)
    q_next = quaternion_multiply(delta_q, q)
    q_next = normalize_quaternion(q_next)
    q_next = standardize_quaternion(q_next, reference=q)
    q_next = _mask_single_atom_quaternions(q_next, frag_sizes=frag_sizes)
    return T_next, q_next


def _mask_single_atom_quaternions(
    q: torch.Tensor,
    *,
    frag_sizes: torch.Tensor | None,
) -> torch.Tensor:
    """Clamp single-atom fragment quaternions to identity."""
    if frag_sizes is None:
        return normalize_quaternion(q)

    mask = _single_atom_mask(frag_sizes).to(device=q.device)
    identity = torch.zeros_like(q)
    identity[..., 0] = 1.0
    q = normalize_quaternion(q)
    return torch.where(mask.unsqueeze(-1), identity, q)


def _mask_single_atom_vectors(
    value: torch.Tensor,
    *,
    frag_sizes: torch.Tensor | None,
) -> torch.Tensor:
    """Zero out vector quantities for single-atom fragments."""
    if frag_sizes is None:
        return value

    mask = _single_atom_mask(frag_sizes).to(device=value.device)
    return torch.where(mask.unsqueeze(-1), torch.zeros_like(value), value)


def _single_atom_mask(frag_sizes: torch.Tensor) -> torch.Tensor:
    """Return a boolean mask selecting fragments with size <= 1."""
    return frag_sizes <= 1


def _broadcast_last_dim_like(value: torch.Tensor | float, like: torch.Tensor) -> torch.Tensor:
    """Broadcast a scalar or batch tensor to match the rank of ``like``."""
    if torch.is_tensor(value):
        value_t = value.to(device=like.device, dtype=like.dtype)
    else:
        value_t = torch.tensor(value, device=like.device, dtype=like.dtype)

    while value_t.ndim < like.ndim:
        value_t = value_t.unsqueeze(-1)

    return value_t


__all__ = [
    "compute_angular_velocity",
    "compute_flow_matching_targets",
    "compute_translation_velocity",
    "compute_vp_score_full_targets",
    "vp_score_alpha_beta",
    "compute_vp_flow_targets",
    "compute_vp_score_targets",
    "integrate_se3_step",
    "interpolate_poses",
    "interpolate_rotations",
    "interpolate_translations",
    "sample_prior_poses",
    "sample_prior_rotations",
    "sample_prior_translations",
]
