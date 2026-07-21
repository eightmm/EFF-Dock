"""Quaternion and SO(3) utilities for EFFDock.

Conventions:
- Quaternions are scalar-first: ``(w, x, y, z)``.
- Rotations are active and act on column vectors on the left:
  ``x_global = R @ x_local``.
- ``quaternion_multiply(q1, q2)`` corresponds to composition
  ``R(q1 ⊗ q2) = R(q1) @ R(q2)``.
"""

from __future__ import annotations

import math

import torch


def normalize_quaternion(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalize quaternions to unit norm.

    Args:
        q: Tensor of shape ``[..., 4]``.
        eps: Minimum norm clamp for numerical stability.

    Returns:
        Tensor of shape ``[..., 4]`` with unit-norm quaternions.
    """
    if q.shape[-1] != 4:
        raise ValueError(f"Expected quaternion tensor with last dim 4, got {q.shape}.")

    norm = torch.linalg.vector_norm(q, dim=-1, keepdim=True)
    return q / norm.clamp_min(eps)


def standardize_quaternion(
    q: torch.Tensor,
    reference: torch.Tensor | None = None,
) -> torch.Tensor:
    """Resolve quaternion sign ambiguity by flipping to a consistent hemisphere.

    Args:
        q: Tensor of shape ``[..., 4]``.
        reference: Optional reference quaternion with the same broadcastable shape.
            If provided, the output is flipped so ``dot(q, reference) >= 0``.
            Otherwise the output is flipped so the scalar part is non-negative.

    Returns:
        Tensor of shape ``[..., 4]`` with a standardized sign.
    """
    if q.shape[-1] != 4:
        raise ValueError(f"Expected quaternion tensor with last dim 4, got {q.shape}.")

    if reference is None:
        flip = q[..., :1] < 0
    else:
        if reference.shape[-1] != 4:
            raise ValueError(
                f"Expected reference quaternion tensor with last dim 4, got {reference.shape}."
            )
        flip = (q * reference).sum(dim=-1, keepdim=True) < 0

    sign = torch.where(flip, -torch.ones_like(q[..., :1]), torch.ones_like(q[..., :1]))
    return q * sign


def quaternion_conjugate(q: torch.Tensor) -> torch.Tensor:
    """Return quaternion conjugates.

    Args:
        q: Tensor of shape ``[..., 4]``.

    Returns:
        Tensor of shape ``[..., 4]``.
    """
    if q.shape[-1] != 4:
        raise ValueError(f"Expected quaternion tensor with last dim 4, got {q.shape}.")

    return torch.cat((q[..., :1], -q[..., 1:]), dim=-1)


def quaternion_inverse(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Return quaternion inverses.

    Args:
        q: Tensor of shape ``[..., 4]``.
        eps: Minimum squared norm clamp for numerical stability.

    Returns:
        Tensor of shape ``[..., 4]``.
    """
    if q.shape[-1] != 4:
        raise ValueError(f"Expected quaternion tensor with last dim 4, got {q.shape}.")

    norm_sq = (q * q).sum(dim=-1, keepdim=True)
    return quaternion_conjugate(q) / norm_sq.clamp_min(eps)


def quaternion_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Compose two quaternions with the Hamilton product.

    Args:
        q1: Tensor of shape ``[..., 4]``.
        q2: Tensor of shape ``[..., 4]``.

    Returns:
        Tensor of shape ``[..., 4]`` representing ``q1 ⊗ q2``.
    """
    if q1.shape[-1] != 4 or q2.shape[-1] != 4:
        raise ValueError(
            f"Expected quaternion tensors with last dim 4, got {q1.shape} and {q2.shape}."
        )

    w1, x1, y1, z1 = q1.unbind(dim=-1)
    w2, x2, y2, z2 = q2.unbind(dim=-1)

    return torch.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        dim=-1,
    )


def axis_angle_to_quaternion(axis_angle: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Convert axis-angle rotation vectors to quaternions.

    The input is a rotation vector whose direction is the rotation axis and whose
    norm is the rotation angle in radians.

    Args:
        axis_angle: Tensor of shape ``[..., 3]``.
        eps: Small-angle threshold used for a stable Taylor expansion.

    Returns:
        Tensor of shape ``[..., 4]`` containing unit quaternions.
    """
    if axis_angle.shape[-1] != 3:
        raise ValueError(f"Expected axis-angle tensor with last dim 3, got {axis_angle.shape}.")

    angle = torch.linalg.vector_norm(axis_angle, dim=-1, keepdim=True)
    half_angle = 0.5 * angle
    angle_sq = angle * angle

    imag_scale = torch.where(
        angle > eps,
        torch.sin(half_angle) / angle,
        0.5 - angle_sq / 48.0 + angle_sq * angle_sq / 3840.0,
    )

    q = torch.cat((torch.cos(half_angle), axis_angle * imag_scale), dim=-1)
    return normalize_quaternion(q, eps=eps)


def quaternion_to_axis_angle(
    q: torch.Tensor,
    *,
    shortest_path: bool = True,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Convert quaternions to axis-angle rotation vectors.

    Args:
        q: Tensor of shape ``[..., 4]``.
        shortest_path: If ``True``, flip quaternion signs so the returned angle
            lies in ``[0, pi]``.
        eps: Small-angle threshold for stable normalization.

    Returns:
        Tensor of shape ``[..., 3]`` containing rotation vectors in radians.
    """
    if q.shape[-1] != 4:
        raise ValueError(f"Expected quaternion tensor with last dim 4, got {q.shape}.")

    q = normalize_quaternion(q, eps=eps)
    if shortest_path:
        q = standardize_quaternion(q)

    w = q[..., :1].clamp(-1.0, 1.0)
    xyz = q[..., 1:]
    sin_half = torch.linalg.vector_norm(xyz, dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(sin_half, w)

    scale = torch.where(
        sin_half > eps,
        angle / sin_half,
        2.0 * torch.ones_like(sin_half),
    )
    return xyz * scale


def quaternion_to_matrix(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Convert quaternions to rotation matrices.

    Args:
        q: Tensor of shape ``[..., 4]``.
        eps: Minimum norm clamp for numerical stability.

    Returns:
        Tensor of shape ``[..., 3, 3]``.
    """
    if q.shape[-1] != 4:
        raise ValueError(f"Expected quaternion tensor with last dim 4, got {q.shape}.")

    q = normalize_quaternion(q, eps=eps)
    w, x, y, z = q.unbind(dim=-1)

    xx = x * x
    yy = y * y
    zz = z * z
    wx = w * x
    wy = w * y
    wz = w * z
    xy = x * y
    xz = x * z
    yz = y * z

    return torch.stack(
        (
            1.0 - 2.0 * (yy + zz),
            2.0 * (xy - wz),
            2.0 * (xz + wy),
            2.0 * (xy + wz),
            1.0 - 2.0 * (xx + zz),
            2.0 * (yz - wx),
            2.0 * (xz - wy),
            2.0 * (yz + wx),
            1.0 - 2.0 * (xx + yy),
        ),
        dim=-1,
    ).reshape(q.shape[:-1] + (3, 3))


def matrix_to_quaternion(matrix: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Convert rotation matrices to quaternions.

    Args:
        matrix: Tensor of shape ``[..., 3, 3]``.
        eps: Small clamp to avoid dividing by zero in branch formulas.

    Returns:
        Tensor of shape ``[..., 4]`` with standardized unit quaternions.
    """
    if matrix.shape[-2:] != (3, 3):
        raise ValueError(
            f"Expected rotation matrix tensor with shape [..., 3, 3], got {matrix.shape}."
        )

    m00 = matrix[..., 0, 0]
    m01 = matrix[..., 0, 1]
    m02 = matrix[..., 0, 2]
    m10 = matrix[..., 1, 0]
    m11 = matrix[..., 1, 1]
    m12 = matrix[..., 1, 2]
    m20 = matrix[..., 2, 0]
    m21 = matrix[..., 2, 1]
    m22 = matrix[..., 2, 2]

    q_abs = torch.sqrt(
        torch.clamp(
            torch.stack(
                (
                    1.0 + m00 + m11 + m22,
                    1.0 + m00 - m11 - m22,
                    1.0 - m00 + m11 - m22,
                    1.0 - m00 - m11 + m22,
                ),
                dim=-1,
            ),
            min=0.0,
        )
    )
    denom = 2.0 * q_abs.clamp_min(eps)

    candidates = torch.stack(
        (
            torch.stack(
                (
                    0.5 * q_abs[..., 0],
                    (m21 - m12) / denom[..., 0],
                    (m02 - m20) / denom[..., 0],
                    (m10 - m01) / denom[..., 0],
                ),
                dim=-1,
            ),
            torch.stack(
                (
                    (m21 - m12) / denom[..., 1],
                    0.5 * q_abs[..., 1],
                    (m01 + m10) / denom[..., 1],
                    (m02 + m20) / denom[..., 1],
                ),
                dim=-1,
            ),
            torch.stack(
                (
                    (m02 - m20) / denom[..., 2],
                    (m01 + m10) / denom[..., 2],
                    0.5 * q_abs[..., 2],
                    (m12 + m21) / denom[..., 2],
                ),
                dim=-1,
            ),
            torch.stack(
                (
                    (m10 - m01) / denom[..., 3],
                    (m02 + m20) / denom[..., 3],
                    (m12 + m21) / denom[..., 3],
                    0.5 * q_abs[..., 3],
                ),
                dim=-1,
            ),
        ),
        dim=-2,
    )

    best = q_abs.argmax(dim=-1)
    gather_index = best.unsqueeze(-1).unsqueeze(-1).expand(best.shape + (1, 4))
    q = candidates.gather(dim=-2, index=gather_index).squeeze(dim=-2)
    q = normalize_quaternion(q, eps=eps)
    return standardize_quaternion(q)


def quaternion_slerp(
    q0: torch.Tensor,
    q1: torch.Tensor,
    t: torch.Tensor | float,
    *,
    shortest_path: bool = True,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Spherical linear interpolation between two quaternions.

    Args:
        q0: Start quaternions of shape ``[..., 4]``.
        q1: End quaternions of shape ``[..., 4]``.
        t: Interpolation time in ``[0, 1]``. Can be a scalar or any tensor
            broadcastable to the batch shape of ``q0``.
        shortest_path: If ``True``, interpolate along the shorter geodesic by
            flipping ``q1`` when ``dot(q0, q1) < 0``.
        eps: Minimum norm clamp for numerical stability.

    Returns:
        Tensor of shape ``[..., 4]`` with interpolated unit quaternions.
    """
    if q0.shape[-1] != 4 or q1.shape[-1] != 4:
        raise ValueError(
            f"Expected quaternion tensors with last dim 4, got {q0.shape} and {q1.shape}."
        )

    q0 = normalize_quaternion(q0, eps=eps)
    q1 = normalize_quaternion(q1, eps=eps)
    if shortest_path:
        q1 = standardize_quaternion(q1, reference=q0)

    q_delta = quaternion_multiply(q1, quaternion_inverse(q0, eps=eps))
    rotvec = quaternion_to_axis_angle(q_delta, shortest_path=shortest_path, eps=eps)
    t = _broadcast_last_dim_like(t, rotvec)

    q_step = axis_angle_to_quaternion(t * rotvec, eps=eps)
    q_t = quaternion_multiply(q_step, q0)
    q_t = normalize_quaternion(q_t, eps=eps)
    return standardize_quaternion(q_t, reference=q0)


def sample_uniform_quaternion(
    num_samples: int | tuple[int, ...],
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample quaternions uniformly over SO(3).

    Args:
        num_samples: Sample count or batch shape.
        device: Optional output device.
        dtype: Optional floating output dtype.
        generator: Optional torch random generator.

    Returns:
        Tensor of shape ``[num_samples, 4]`` or ``[*num_samples, 4]``.
    """
    if isinstance(num_samples, int):
        sample_shape = (num_samples,)
    else:
        sample_shape = tuple(num_samples)

    if dtype is None:
        dtype = torch.get_default_dtype()

    u = torch.rand(
        sample_shape + (3,),
        device=device,
        dtype=dtype,
        generator=generator,
    )
    u1, u2, u3 = u.unbind(dim=-1)

    sqrt_u1 = torch.sqrt(u1)
    sqrt_one_minus_u1 = torch.sqrt(1.0 - u1)
    theta1 = 2.0 * math.pi * u2
    theta2 = 2.0 * math.pi * u3

    q = torch.stack(
        (
            sqrt_u1 * torch.cos(theta2),
            sqrt_one_minus_u1 * torch.sin(theta1),
            sqrt_one_minus_u1 * torch.cos(theta1),
            sqrt_u1 * torch.sin(theta2),
        ),
        dim=-1,
    )
    q = normalize_quaternion(q)
    return standardize_quaternion(q)


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
    "axis_angle_to_quaternion",
    "matrix_to_quaternion",
    "normalize_quaternion",
    "quaternion_conjugate",
    "quaternion_inverse",
    "quaternion_multiply",
    "quaternion_slerp",
    "quaternion_to_axis_angle",
    "quaternion_to_matrix",
    "sample_uniform_quaternion",
    "standardize_quaternion",
]
