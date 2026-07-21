"""SE(3) rotation equivariance tests for src/models/equivariant.py.

These tests verify that layers produce outputs that transform correctly under
rotations of their inputs:

    f(Rx) == R f(x)    for every SE(3)-equivariant layer f.

The rotation R acts on a feature vector via Wigner D-matrices assembled by
e3nn (which uses the same ``mul_ir`` layout as cuEquivariance when we build
the irreps through the mapping below).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

pytest.importorskip("e3nn")
import cuequivariance as cue  # noqa: E402
from e3nn import o3  # noqa: E402

from effdock.models.equivariant import (  # noqa: E402
    EquivariantActivation,
    EquivariantAdaLN,
    EquivariantBlock,
    EquivariantDropout,
    EquivariantMLP,
    EquivariantRMSNorm,
    GatedEquivariantConv,
    IrrepsLayout,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def cue_to_e3nn(cue_irreps: cue.Irreps) -> o3.Irreps:
    """Convert a cue.Irreps('O3', ...) to an equivalent e3nn o3.Irreps."""
    parts = []
    for mul, ir in cue_irreps:
        parity = "e" if ir.p == 1 else "o"
        parts.append(f"{mul}x{ir.l}{parity}")
    return o3.Irreps(" + ".join(parts))


def rotate_features(x: torch.Tensor, cue_irreps: cue.Irreps, R: torch.Tensor) -> torch.Tensor:
    """Apply the representation of ``R`` to ``x`` structured by ``cue_irreps``."""
    e3 = cue_to_e3nn(cue_irreps)
    # e3nn's Wigner generators live on CPU — compute D on CPU, then move.
    D = e3.D_from_matrix(R.detach().cpu()).to(device=x.device, dtype=x.dtype)
    return x @ D.T


def random_rotation(seed: int = 0) -> torch.Tensor:
    # e3nn's rand_matrix uses the global torch RNG — seed it deterministically.
    torch.manual_seed(seed)
    return o3.rand_matrix().to(torch.float64)


def set_inference_mode(module: nn.Module) -> nn.Module:
    """Put a module in inference mode (equivalent to ``.eval()``)."""
    module.train(False)
    return module


DTYPE = torch.float64
ATOL = 1e-9


# ---------------------------------------------------------------------------
# IrrepsLayout utility tests
# ---------------------------------------------------------------------------


def test_irreps_layout_dimensions():
    irreps = cue.Irreps("O3", "4x0e + 2x1o + 3x2e + 1x2o + 1x3o")
    layout = IrrepsLayout(irreps)
    # 4*1 + 2*3 + 3*5 + 1*5 + 1*7 = 4 + 6 + 15 + 5 + 7 = 37
    assert layout.feat_dim == 37
    assert layout.n_scalar_channels == 4
    assert layout.n_nonscalar_channels == 2 + 3 + 1 + 1
    assert layout.n_channels == 11
    # l-grouping: l=1 → [0], l=2 → [1, 2], l=3 → [3]
    assert layout.l_groups[1] == (0,)
    assert layout.l_groups[2] == (1, 2)
    assert layout.l_groups[3] == (3,)


def test_expand_channel_scale_matches_irreps_layout():
    irreps = cue.Irreps("O3", "4x0e + 2x1o + 3x2e")
    layout = IrrepsLayout(irreps)
    scale = torch.tensor(
        [
            [
                0.0,
                1.0,
                2.0,
                3.0,  # scalar ch 0..3
                4.0,
                5.0,  # 1o ch 0..1
                6.0,
                7.0,
                8.0,
            ]
        ]
    )  # 2e ch 0..2
    expanded = layout.expand_channel_scale(scale)
    assert torch.equal(expanded[0, :4], torch.tensor([0.0, 1.0, 2.0, 3.0]))
    assert torch.equal(expanded[0, 4:10], torch.tensor([4.0, 4.0, 4.0, 5.0, 5.0, 5.0]))
    assert torch.equal(expanded[0, 10:15], torch.full((5,), 6.0))
    assert torch.equal(expanded[0, 15:20], torch.full((5,), 7.0))
    assert torch.equal(expanded[0, 20:25], torch.full((5,), 8.0))


def test_gather_nonscalar_norms_rotation_invariant():
    irreps = cue.Irreps("O3", "2x0e + 3x1o + 2x2e")
    layout = IrrepsLayout(irreps)
    x = torch.randn(4, irreps.dim, dtype=DTYPE)
    R = random_rotation(seed=1)
    x_rot = rotate_features(x, irreps, R)
    assert torch.allclose(
        layout.gather_nonscalar_norms(x), layout.gather_nonscalar_norms(x_rot), atol=ATOL
    )


# ---------------------------------------------------------------------------
# Layer equivariance tests
# ---------------------------------------------------------------------------


def test_equivariant_activation_equivariance_high_l():
    """EquivariantActivation must preserve equivariance for l ≥ 1 (incl. l=3)."""
    torch.manual_seed(0)
    irreps = cue.Irreps("O3", "16x0e + 4x1o + 4x1e + 2x2e + 2x2o + 1x3o")
    act = set_inference_mode(EquivariantActivation(irreps).to(DTYPE))
    # Randomize the norm MLP so gates are non-trivial.
    with torch.no_grad():
        for p in act.parameters():
            if p.dim() > 0:
                nn.init.normal_(p, std=0.2)
    x = torch.randn(5, irreps.dim, dtype=DTYPE)
    R = random_rotation(seed=6)

    lhs = act(rotate_features(x, irreps, R))
    rhs = rotate_features(act(x), irreps, R)
    assert torch.allclose(lhs, rhs, atol=1e-8)


def test_equivariant_rmsnorm_rotation_equivariance():
    torch.manual_seed(0)
    irreps = cue.Irreps("O3", "16x0e + 4x1o + 4x1e + 2x2e + 2x2o + 1x3o")
    norm = set_inference_mode(EquivariantRMSNorm(irreps).to(DTYPE))
    # Randomize gains so it's non-trivial
    with torch.no_grad():
        norm.gain.uniform_(0.5, 2.0)
    x = torch.randn(5, irreps.dim, dtype=DTYPE)
    R = random_rotation(seed=9)

    lhs = norm(rotate_features(x, irreps, R))
    rhs = rotate_features(norm(x), irreps, R)
    assert torch.allclose(lhs, rhs, atol=1e-8)


def test_equivariant_dropout_eval_equivariance():
    torch.manual_seed(0)
    irreps = cue.Irreps("O3", "8x0e + 4x1o + 2x2e")
    drop = set_inference_mode(EquivariantDropout(irreps, p=0.3).to(DTYPE))
    x = torch.randn(4, irreps.dim, dtype=DTYPE)
    R = random_rotation(seed=3)
    assert torch.allclose(
        drop(rotate_features(x, irreps, R)),
        rotate_features(drop(x), irreps, R),
        atol=ATOL,
    )


def test_equivariant_adaln_equivariance_random_weights():
    torch.manual_seed(0)
    irreps = cue.Irreps("O3", "16x0e + 4x1o + 2x2e + 2x2o")
    cond_dim = 8
    ada = set_inference_mode(EquivariantAdaLN(irreps, cond_dim=cond_dim).to(DTYPE))
    # Break zero-init so we test a non-trivial transform.
    with torch.no_grad():
        for p in ada.parameters():
            if p.dim() > 0:
                nn.init.normal_(p, std=0.05)
    x = torch.randn(5, irreps.dim, dtype=DTYPE)
    cond = torch.randn(5, cond_dim, dtype=DTYPE)  # rotation-invariant scalar cond
    R = random_rotation(seed=4)

    lhs = ada(rotate_features(x, irreps, R), cond)
    rhs = rotate_features(ada(x, cond), irreps, R)
    assert torch.allclose(lhs, rhs, atol=1e-8)


def test_equivariant_block_rotation_equivariance():
    torch.manual_seed(0)
    irreps_in = cue.Irreps("O3", "16x0e + 4x1o + 2x2e")
    irreps_out = cue.Irreps("O3", "8x0e + 4x1o + 2x2e + 2x2o")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        pytest.skip("EquivariantBlock uses cuEquivariance CUDA kernels")

    block = set_inference_mode(EquivariantBlock(irreps_in, irreps_out, dropout=0.0).to(device))
    x = torch.randn(6, irreps_in.dim, device=device)
    R = random_rotation(seed=7).to(device=device, dtype=torch.float32)

    lhs = block(rotate_features(x, irreps_in, R))
    rhs = rotate_features(block(x), irreps_out, R)
    assert torch.allclose(lhs, rhs, atol=1e-3, rtol=1e-3)


def test_equivariant_mlp_rotation_equivariance_with_residual():
    torch.manual_seed(0)
    irreps = cue.Irreps("O3", "16x0e + 4x1o + 2x2e + 2x2o")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        pytest.skip("EquivariantMLP uses cuEquivariance CUDA kernels")

    mlp = set_inference_mode(
        EquivariantMLP(irreps, irreps, irreps, num_layers=3, dropout=0.0).to(device)
    )
    x = torch.randn(6, irreps.dim, device=device)
    R = random_rotation(seed=8).to(device=device, dtype=torch.float32)

    lhs = mlp(rotate_features(x, irreps, R))
    rhs = rotate_features(mlp(x), irreps, R)
    assert torch.allclose(lhs, rhs, atol=1e-3, rtol=1e-3)


def test_gated_equivariant_conv_rotation_equivariance():
    torch.manual_seed(0)
    irreps = cue.Irreps("O3", "32x0e + 8x1o + 8x1e + 4x2e + 4x2o")
    edge_scalar_dim = 16

    conv = set_inference_mode(
        GatedEquivariantConv(
            irreps,
            sh_lmax=2,
            edge_scalar_dim=edge_scalar_dim,
        )
    )
    # cuEquivariance fused_tp kernels require CUDA + fp32.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        pytest.skip("GatedEquivariantConv relies on cuEquivariance CUDA kernels")
    conv = conv.to(device)

    n, e = 8, 24
    h = torch.randn(n, irreps.dim, device=device)
    edge_vec = torch.randn(e, 3, device=device)
    edge_scalars = torch.randn(e, edge_scalar_dim, device=device)
    src = torch.randint(0, n, (e,), device=device)
    dst = torch.randint(0, n, (e,), device=device)
    edge_dist = edge_vec.norm(dim=-1)

    R = random_rotation(seed=5).to(device=device, dtype=torch.float32)

    h_rot = rotate_features(h, irreps, R)
    edge_vec_rot = edge_vec @ R.T

    out_from_rot = conv(h_rot, edge_vec_rot, edge_scalars, src, dst, n, edge_dist)
    out_orig = conv(h, edge_vec, edge_scalars, src, dst, n, edge_dist)
    rot_out = rotate_features(out_orig, irreps, R)

    # fp32 + many ops → looser tolerance.
    assert torch.allclose(out_from_rot, rot_out, atol=1e-3, rtol=1e-3)
