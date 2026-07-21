from __future__ import annotations

import pytest
import torch

from effdock.models.effdock import _stable_symmetric_eigh, newton_euler_aggregate


def _proper_rotation(dtype: torch.dtype = torch.float64) -> torch.Tensor:
    matrix = torch.tensor(
        [[0.36, -0.48, 0.80], [0.80, 0.60, 0.00], [-0.48, 0.64, 0.60]],
        dtype=dtype,
    )
    torch.testing.assert_close(torch.det(matrix), torch.tensor(1.0, dtype=dtype))
    return matrix


def _mixed_fragments(dtype: torch.dtype = torch.float32) -> tuple[torch.Tensor, ...]:
    # One single atom, one collinear two-atom fragment, and one non-degenerate
    # tetrahedral fragment exercise zero, rank-deficient, and full-rank inertia.
    atom_pos = torch.tensor(
        [
            [2.0, -1.0, 0.5],
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=dtype,
    )
    frag_id = torch.tensor([0, 1, 1, 2, 2, 2, 2], dtype=torch.long)
    centers = torch.stack(
        [atom_pos[0], atom_pos[1:3].mean(0), atom_pos[3:7].mean(0)]
    )
    frag_sizes = torch.tensor([1, 2, 4], dtype=torch.long)
    forces = torch.tensor(
        [
            [0.3, -0.2, 0.1],
            [0.1, 0.3, -0.4],
            [-0.2, 0.4, 0.5],
            [0.5, -0.1, 0.2],
            [-0.2, 0.7, -0.3],
            [0.4, 0.1, 0.6],
            [-0.3, -0.5, 0.2],
        ],
        dtype=dtype,
    )
    return forces, atom_pos, centers, frag_id, frag_sizes


def test_newton_euler_handles_single_and_collinear_fragments() -> None:
    inputs = _mixed_fragments()
    velocity, omega, projector = newton_euler_aggregate(*inputs[:3], inputs[3], 3, inputs[4])

    assert torch.isfinite(velocity).all()
    assert torch.isfinite(omega).all()
    assert torch.isfinite(projector).all()
    torch.testing.assert_close(omega[0], torch.zeros(3))
    torch.testing.assert_close(projector[0], torch.zeros(3, 3))
    torch.testing.assert_close(projector[1], torch.diag(torch.tensor([0.0, 1.0, 1.0])))
    torch.testing.assert_close(omega[1, 0], torch.tensor(0.0), atol=1e-6, rtol=0.0)


def test_newton_euler_is_rotation_translation_equivariant() -> None:
    forces, atom_pos, centers, frag_id, frag_sizes = _mixed_fragments(dtype=torch.float64)
    velocity, omega, projector = newton_euler_aggregate(
        forces, atom_pos, centers, frag_id, 3, frag_sizes
    )
    rotation = _proper_rotation()
    translation = torch.tensor([3.0, -4.0, 2.0], dtype=torch.float64)
    velocity_rt, omega_rt, projector_rt = newton_euler_aggregate(
        forces @ rotation.T,
        atom_pos @ rotation.T + translation,
        centers @ rotation.T + translation,
        frag_id,
        3,
        frag_sizes,
    )

    torch.testing.assert_close(velocity_rt, velocity @ rotation.T, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(omega_rt, omega @ rotation.T, atol=1e-10, rtol=1e-10)
    expected_projector = rotation @ projector @ rotation.T
    torch.testing.assert_close(projector_rt, expected_projector, atol=1e-10, rtol=1e-10)


def test_newton_euler_force_gradient_is_finite_for_degenerate_fragments() -> None:
    forces, atom_pos, centers, frag_id, frag_sizes = _mixed_fragments()
    forces.requires_grad_(True)
    velocity, omega, _ = newton_euler_aggregate(
        forces, atom_pos, centers, frag_id, 3, frag_sizes
    )
    (velocity.square().sum() + omega.square().sum()).backward()

    assert forces.grad is not None
    assert torch.isfinite(forces.grad).all()


def test_stable_eigh_rejects_nonfinite_inertia() -> None:
    matrix = torch.eye(3, dtype=torch.float32).unsqueeze(0)
    matrix[0, 1, 1] = torch.nan

    with pytest.raises(FloatingPointError, match="non-finite"):
        _stable_symmetric_eigh(matrix)
