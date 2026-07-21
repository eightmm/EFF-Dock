"""Tests for quaternion math and SE(3) flow-matching utilities."""

import math

import torch

from effdock.geometry.flow_matching import (
    compute_angular_velocity,
    compute_flow_matching_targets,
    compute_translation_velocity,
    integrate_se3_step,
    interpolate_rotations,
    interpolate_translations,
    sample_prior_rotations,
)
from effdock.geometry.se3 import (
    axis_angle_to_quaternion,
    matrix_to_quaternion,
    normalize_quaternion,
    quaternion_multiply,
    quaternion_slerp,
    quaternion_to_matrix,
)


def _assert_same_rotation(q_a: torch.Tensor, q_b: torch.Tensor, atol: float = 1e-6) -> None:
    """Assert that two quaternions represent the same rotation."""
    R_a = quaternion_to_matrix(normalize_quaternion(q_a))
    R_b = quaternion_to_matrix(normalize_quaternion(q_b))
    assert torch.allclose(R_a, R_b, atol=atol)


def test_quaternion_multiply_matches_matrix_composition():
    q_x = axis_angle_to_quaternion(torch.tensor([math.pi / 2, 0.0, 0.0], dtype=torch.float64))
    q_z = axis_angle_to_quaternion(torch.tensor([0.0, 0.0, math.pi / 2], dtype=torch.float64))

    q_composed = quaternion_multiply(q_z, q_x)
    R_composed = quaternion_to_matrix(q_composed)
    R_expected = quaternion_to_matrix(q_z) @ quaternion_to_matrix(q_x)

    assert torch.allclose(R_composed, R_expected, atol=1e-6)


def test_matrix_quaternion_round_trip_preserves_rotation():
    q = axis_angle_to_quaternion(torch.tensor([0.3, -0.5, 1.1], dtype=torch.float64))
    R = quaternion_to_matrix(q)
    q_round_trip = matrix_to_quaternion(R)

    _assert_same_rotation(q, q_round_trip)


def test_slerp_uses_shortest_path_under_sign_ambiguity():
    q0 = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
    q1 = axis_angle_to_quaternion(torch.tensor([0.0, 0.0, math.pi / 2], dtype=torch.float64))
    q_mid = axis_angle_to_quaternion(torch.tensor([0.0, 0.0, math.pi / 4], dtype=torch.float64))

    q_t_pos = quaternion_slerp(q0, q1, 0.5)
    q_t_neg = quaternion_slerp(q0, -q1, 0.5)

    _assert_same_rotation(q_t_pos, q_mid)
    _assert_same_rotation(q_t_neg, q_mid)


def test_translation_velocity_matches_linear_interpolation_derivative():
    T_0 = torch.tensor([[0.0, 1.0, 2.0], [-2.0, 0.5, 4.0]], dtype=torch.float64)
    T_1 = torch.tensor([[3.0, -1.0, 5.0], [1.0, 1.5, -2.0]], dtype=torch.float64)
    v = compute_translation_velocity(T_0, T_1)

    assert torch.allclose(v, T_1 - T_0, atol=1e-8)

    t = 0.3
    dt = 1e-4
    T_t = interpolate_translations(T_0, T_1, t)
    T_next = interpolate_translations(T_0, T_1, t + dt)
    finite_diff = (T_next - T_t) / dt

    assert torch.allclose(finite_diff, v, atol=1e-6)


def test_global_frame_angular_velocity_matches_left_multiplication_update():
    T_0 = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float64)
    T_1 = torch.tensor([[4.0, 6.0, 8.0]], dtype=torch.float64)
    q_0 = axis_angle_to_quaternion(torch.tensor([[math.pi / 2, 0.0, 0.0]], dtype=torch.float64))
    q_delta = axis_angle_to_quaternion(torch.tensor([[0.0, 0.0, math.pi / 2]], dtype=torch.float64))
    q_1 = quaternion_multiply(q_delta, q_0)

    v = compute_translation_velocity(T_0, T_1)
    omega = compute_angular_velocity(q_0, q_1)

    assert torch.allclose(
        omega,
        torch.tensor([[0.0, 0.0, math.pi / 2]], dtype=torch.float64),
        atol=1e-6,
    )

    T_next, q_next = integrate_se3_step(T_0, q_0, v, omega, dt=1.0)
    assert torch.allclose(T_next, T_1, atol=1e-6)
    _assert_same_rotation(q_next, q_1)


def test_flow_matching_targets_return_interpolated_state_and_constant_velocity():
    T_0 = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64)
    T_1 = torch.tensor([[2.0, -1.0, 4.0]], dtype=torch.float64)
    q_0 = axis_angle_to_quaternion(torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64))
    q_1 = axis_angle_to_quaternion(torch.tensor([[0.0, math.pi / 2, 0.0]], dtype=torch.float64))
    t = 0.25

    targets = compute_flow_matching_targets(T_0, q_0, T_1, q_1, t)

    assert torch.allclose(targets["T_t"], interpolate_translations(T_0, T_1, t), atol=1e-6)
    _assert_same_rotation(targets["q_t"], interpolate_rotations(q_0, q_1, t))
    assert torch.allclose(targets["v_t"], T_1 - T_0, atol=1e-6)
    assert torch.allclose(
        targets["omega_t"],
        torch.tensor([[0.0, math.pi / 2, 0.0]], dtype=torch.float64),
        atol=1e-6,
    )


def test_single_atom_fragments_use_identity_and_zero_omega():
    frag_sizes = torch.tensor([1, 3], dtype=torch.int64)
    q_0 = axis_angle_to_quaternion(
        torch.tensor(
            [
                [0.8, 0.1, -0.2],
                [0.0, 0.0, math.pi / 2],
            ],
            dtype=torch.float64,
        )
    )
    q_1 = axis_angle_to_quaternion(
        torch.tensor(
            [
                [0.4, -0.3, 0.2],
                [0.0, math.pi / 2, 0.0],
            ],
            dtype=torch.float64,
        )
    )

    q_t = interpolate_rotations(q_0, q_1, t=0.4, frag_sizes=frag_sizes)
    omega = compute_angular_velocity(q_0, q_1, frag_sizes=frag_sizes)

    assert torch.allclose(
        q_t[0],
        torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64),
        atol=1e-6,
    )
    assert torch.allclose(omega[0], torch.zeros(3, dtype=torch.float64), atol=1e-6)

    generator = torch.Generator().manual_seed(7)
    q_prior = sample_prior_rotations(
        2, frag_sizes=frag_sizes, generator=generator, dtype=torch.float64
    )
    assert torch.allclose(
        q_prior[0],
        torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64),
        atol=1e-6,
    )
    assert torch.allclose(
        torch.linalg.vector_norm(q_prior[1]), torch.tensor(1.0, dtype=torch.float64)
    )
