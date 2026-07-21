"""Tests for Phase 1 loss improvements: body-frame, direction/magnitude, atom aux."""

import torch

from effdock.training.losses import atom_position_auxiliary_loss, flow_matching_loss

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rand(shape, **kw):
    return torch.randn(shape, **kw)


def _identity_R(n):
    return torch.eye(3).unsqueeze(0).expand(n, -1, -1)


def _random_rotation(n):
    """Random rotation matrices via QR decomposition."""
    A = torch.randn(n, 3, 3)
    Q, _ = torch.linalg.qr(A)
    # Ensure proper rotation (det=+1)
    det = torch.linalg.det(Q)
    Q = Q * det.sign().unsqueeze(-1).unsqueeze(-1)
    return Q


# ---------------------------------------------------------------------------
# 1. Baseline flow_matching_loss regression
# ---------------------------------------------------------------------------


def test_flow_matching_loss_baseline():
    """Existing MSE behavior is preserved with default args."""
    v_p, v_t = _rand((4, 3)), _rand((4, 3))
    w_p, w_t = _rand((4, 3)), _rand((4, 3))
    sizes = torch.tensor([3, 1, 5, 2])

    result = flow_matching_loss(v_p, w_p, v_t, w_t, sizes)
    assert "loss" in result
    assert "loss_v" in result
    assert "loss_omega" in result
    assert "cos_v" in result
    assert "cos_omega" in result

    # Manual check: loss_v = MSE of all fragments
    expected_v = ((v_p - v_t) ** 2).mean()
    assert torch.allclose(result["loss_v"], expected_v, atol=1e-6)

    # loss_omega only from multi-atom fragments (indices 0, 2, 3)
    multi = torch.tensor([True, False, True, True])
    expected_w = ((w_p[multi] - w_t[multi]) ** 2).mean()
    assert torch.allclose(result["loss_omega"], expected_w, atol=1e-6)


# ---------------------------------------------------------------------------
# 2. Body-frame loss with R_t = I matches world-frame
# ---------------------------------------------------------------------------


def test_body_frame_identity_matches_world():
    """When R_t = I, body-frame loss must equal world-frame loss."""
    v_p, v_t = _rand((5, 3)), _rand((5, 3))
    w_p, w_t = _rand((5, 3)), _rand((5, 3))
    sizes = torch.tensor([3, 2, 4, 1, 6])
    R_t = _identity_R(5)

    world = flow_matching_loss(v_p, w_p, v_t, w_t, sizes)
    body = flow_matching_loss(v_p, w_p, v_t, w_t, sizes, R_t=R_t, omega_loss_frame="body")

    assert torch.allclose(world["loss"], body["loss"], atol=1e-5)
    assert torch.allclose(world["loss_omega"], body["loss_omega"], atol=1e-5)
    assert torch.allclose(world["cos_omega"], body["cos_omega"], atol=1e-5)


# ---------------------------------------------------------------------------
# 3. Body-frame loss with known R_t
# ---------------------------------------------------------------------------


def test_body_frame_known_rotation():
    """Body-frame omega should equal R_t^T @ world-frame omega."""
    n = 6
    sizes = torch.full((n,), 3)  # all multi-atom
    R_t = _random_rotation(n)

    w_t_world = _rand((n, 3))
    w_p_world = _rand((n, 3))

    # Body-frame versions
    Rt_inv = R_t.transpose(-1, -2)
    w_t_body = torch.einsum("nij,nj->ni", Rt_inv, w_t_world)
    w_p_body = torch.einsum("nij,nj->ni", Rt_inv, w_p_world)

    body_result = flow_matching_loss(
        _rand((n, 3)),
        w_p_world,
        _rand((n, 3)),
        w_t_world,
        sizes,
        R_t=R_t,
        omega_loss_frame="body",
    )

    # Manual body-frame MSE
    expected_omega_loss = ((w_p_body - w_t_body) ** 2).mean()
    assert torch.allclose(body_result["loss_omega"], expected_omega_loss, atol=1e-5)


# ---------------------------------------------------------------------------
# 4. Body-frame zeros omega loss for single-atom fragments
# ---------------------------------------------------------------------------


def test_body_frame_single_atom_zero():
    """Single-atom fragments must have zero omega loss."""
    sizes = torch.tensor([1, 1, 1])
    R_t = _random_rotation(3)

    result = flow_matching_loss(
        _rand((3, 3)),
        _rand((3, 3)),
        _rand((3, 3)),
        _rand((3, 3)),
        sizes,
        R_t=R_t,
        omega_loss_frame="body",
    )
    assert result["loss_omega"].item() == 0.0


# ---------------------------------------------------------------------------
# 5. Direction/magnitude: parallel vectors → zero direction loss
# ---------------------------------------------------------------------------


def test_direction_magnitude_parallel():
    """Parallel pred and target should have zero direction loss."""
    n = 4
    sizes = torch.full((n,), 3)
    w_t = _rand((n, 3))
    w_p = w_t * 2.0  # same direction, different magnitude

    result = flow_matching_loss(
        _rand((n, 3)),
        w_p,
        _rand((n, 3)),
        w_t,
        sizes,
        omega_loss_type="direction_magnitude",
    )
    assert "loss_omega_dir" in result
    assert "loss_omega_mag" in result
    assert result["loss_omega_dir"].item() < 1e-5  # direction loss ~0
    assert result["loss_omega_mag"].item() > 0  # magnitude differs


# ---------------------------------------------------------------------------
# 6. Direction/magnitude: orthogonal vectors → direction loss = 1.0
# ---------------------------------------------------------------------------


def test_direction_magnitude_orthogonal():
    """Orthogonal pred and target should have direction loss = 1.0."""
    # Construct orthogonal pairs
    w_t = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    w_p = torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    sizes = torch.tensor([3, 3])

    result = flow_matching_loss(
        _rand((2, 3)),
        w_p,
        _rand((2, 3)),
        w_t,
        sizes,
        omega_loss_type="direction_magnitude",
    )
    # 1 - cos(90°) = 1.0
    assert abs(result["loss_omega_dir"].item() - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# 7. Atom aux loss: known 3-atom fragment
# ---------------------------------------------------------------------------


def test_atom_aux_known_fragment():
    """For a known fragment, verify atom velocities match cross(omega, r)."""
    # 1 fragment, 3 atoms
    T = torch.tensor([[0.0, 0.0, 0.0]])
    atoms = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    frag_id = torch.tensor([0, 0, 0])
    sizes = torch.tensor([3])

    v_pred = torch.tensor([[0.0, 0.0, 0.0]])
    omega_pred = torch.tensor([[0.0, 0.0, 1.0]])  # rotation around z
    v_target = torch.tensor([[0.0, 0.0, 0.0]])
    omega_target = torch.tensor([[0.0, 0.0, 1.0]])  # same

    result = atom_position_auxiliary_loss(
        v_pred,
        omega_pred,
        v_target,
        omega_target,
        atoms,
        T,
        frag_id,
        sizes,
    )
    # pred == target → loss should be ~0
    assert result["loss_atom_aux"].item() < 1e-10


# ---------------------------------------------------------------------------
# 8. Atom aux loss: single-atom fragment contributes zero rotation component
# ---------------------------------------------------------------------------


def test_atom_aux_single_atom():
    """Single-atom fragment at centroid has zero lever arm → zero rotation effect."""
    T = torch.tensor([[1.0, 2.0, 3.0]])
    atoms = torch.tensor([[1.0, 2.0, 3.0]])  # at centroid
    frag_id = torch.tensor([0])
    sizes = torch.tensor([1])

    v_pred = torch.tensor([[1.0, 0.0, 0.0]])
    omega_pred = torch.tensor([[5.0, 5.0, 5.0]])  # any omega
    v_target = torch.tensor([[1.0, 0.0, 0.0]])
    omega_target = torch.tensor([[0.0, 0.0, 0.0]])  # different omega

    result = atom_position_auxiliary_loss(
        v_pred,
        omega_pred,
        v_target,
        omega_target,
        atoms,
        T,
        frag_id,
        sizes,
    )
    # Lever arm = 0, so cross(omega, r) = 0 regardless of omega
    # v_pred == v_target, so loss = 0
    assert result["loss_atom_aux"].item() < 1e-10


# ---------------------------------------------------------------------------
# 9. Atom aux loss: gradient flows to omega_pred
# ---------------------------------------------------------------------------


def test_atom_aux_gradient_flows():
    """omega_pred.grad must be non-None after backward on atom aux loss."""
    T = torch.tensor([[0.0, 0.0, 0.0]])
    atoms = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    frag_id = torch.tensor([0, 0])
    sizes = torch.tensor([2])

    v_pred = torch.zeros(1, 3, requires_grad=True)
    omega_pred = torch.randn(1, 3, requires_grad=True)
    v_target = torch.zeros(1, 3)
    omega_target = torch.randn(1, 3)

    result = atom_position_auxiliary_loss(
        v_pred,
        omega_pred,
        v_target,
        omega_target,
        atoms,
        T,
        frag_id,
        sizes,
    )
    result["loss_atom_aux"].backward()
    assert omega_pred.grad is not None
    assert omega_pred.grad.abs().sum() > 0
