"""SE(3)-equivariant layers built on cuEquivariance primitives.

All modules in this file accept an arbitrary ``cue.Irreps`` (any combination
of l/parity/multiplicity) and auto-detect the layout via :class:`IrrepsLayout`.
They follow the ``mul_ir`` memory layout (irrep dimensions contiguous).

Public API:
    IrrepsLayout          — utility: parse irreps + expand per-channel scales, gather norms
    EquivariantActivation — nonlinearity for any irreps (scalar f(x) + norm-gated vectors)
    EquivariantDropout    — direction-preserving dropout for mixed irreps
    EquivariantAdaLN      — DiT-style conditional norm for mixed irreps
    GatedEquivariantConv  — TP message passing with dual radial + attention gating
"""

from __future__ import annotations

from dataclasses import dataclass

import cuequivariance as cue
import cuequivariance_torch as cuet
import torch
import torch.nn as nn
from torch import Tensor

# ---------------------------------------------------------------------------
# Irreps layout helper
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IrrepBlock:
    """One (mul × irrep) block inside a flat ``mul_ir`` feature vector."""

    offset: int  # flat start index
    mul: int  # multiplicity (number of channels)
    ir_dim: int  # 2l + 1
    l: int
    parity: int  # +1 (even) or -1 (odd)

    @property
    def span(self) -> int:
        return self.mul * self.ir_dim

    @property
    def end(self) -> int:
        return self.offset + self.span

    @property
    def is_scalar(self) -> bool:
        return self.l == 0 and self.parity == 1


class IrrepsLayout:
    """Parse ``cue.Irreps`` once; expose helpers reused across modules.

    Channel order (used by ``expand_channel_scale`` / ``gather_nonscalar_norms``):
    scalar blocks first (in irreps order), then non-scalar blocks (in irreps
    order). Per-channel scales are broadcast across each block's ``ir_dim``,
    preserving direction → SE(3)-equivariant.
    """

    def __init__(self, irreps: cue.Irreps) -> None:
        blocks: list[IrrepBlock] = []
        offset = 0
        for mul, ir in irreps:
            blocks.append(IrrepBlock(offset=offset, mul=mul, ir_dim=ir.dim, l=ir.l, parity=ir.p))
            offset += mul * ir.dim

        self.irreps = irreps
        self.blocks: tuple[IrrepBlock, ...] = tuple(blocks)
        self.feat_dim: int = offset

        self.scalar_blocks: tuple[IrrepBlock, ...] = tuple(b for b in blocks if b.is_scalar)
        self.nonscalar_blocks: tuple[IrrepBlock, ...] = tuple(b for b in blocks if not b.is_scalar)

        self.n_scalar_channels: int = sum(b.mul for b in self.scalar_blocks)
        self.n_nonscalar_channels: int = sum(b.mul for b in self.nonscalar_blocks)
        self.n_channels: int = self.n_scalar_channels + self.n_nonscalar_channels

        # Group non-scalar block indices by angular momentum l (for per-l rescaling).
        groups: dict[int, list[int]] = {}
        for idx, b in enumerate(self.nonscalar_blocks):
            groups.setdefault(b.l, []).append(idx)
        self.l_groups: dict[int, tuple[int, ...]] = {l: tuple(ids) for l, ids in groups.items()}

    # ---- channel/feature transforms ----

    def expand_channel_scale(self, scale_ch: Tensor) -> Tensor:
        """Broadcast a per-channel scale ``[..., n_channels]`` to ``[..., feat_dim]``.

        Each scalar channel takes 1 slot; each vector/tensor channel's scale is
        replicated over its ``ir_dim`` components, so ``v → α·v`` — direction is
        preserved and the operation is SE(3)-equivariant.
        """
        parts: list[Tensor] = []
        ch = 0
        lead = scale_ch.shape[:-1]
        for b in self.scalar_blocks:
            parts.append(scale_ch[..., ch : ch + b.mul])
            ch += b.mul
        for b in self.nonscalar_blocks:
            s = scale_ch[..., ch : ch + b.mul].unsqueeze(-1).expand(*lead, b.mul, b.ir_dim)
            parts.append(s.reshape(*lead, b.span))
            ch += b.mul
        return torch.cat(parts, dim=-1)

    def gather_nonscalar_norms(self, x: Tensor) -> Tensor:
        """Return per-channel vector norms ``[..., n_nonscalar_channels]``.

        Rotation-invariant scalar summary of the non-scalar content of ``x``.
        """
        if not self.nonscalar_blocks:
            return x.new_zeros(*x.shape[:-1], 0)
        parts: list[Tensor] = []
        for b in self.nonscalar_blocks:
            v = x[..., b.offset : b.end].reshape(*x.shape[:-1], b.mul, b.ir_dim)
            parts.append(v.norm(dim=-1))
        return torch.cat(parts, dim=-1)


# ---------------------------------------------------------------------------
# Scatter helper (shape-agnostic)
# ---------------------------------------------------------------------------


def _scatter_add(src: Tensor, idx: Tensor, n: int) -> Tensor:
    """Row-wise scatter_add: ``[E, D] → [n, D]``."""
    out = torch.zeros(n, src.shape[-1], device=src.device, dtype=src.dtype)
    out.scatter_add_(0, idx.unsqueeze(-1).expand_as(src), src)
    return out


# ---------------------------------------------------------------------------
# Activations / dropout / norm
# ---------------------------------------------------------------------------


class EquivariantActivation(nn.Module):
    """Nonlinearity that works on any ``cue.Irreps``.

    - Scalars (l=0, p=+1): apply ``activation`` element-wise.
    - Non-scalars (l≥1 or p=-1): apply a per-channel scalar gate derived from
      the vector norm: ``v → v · σ(MLP(||v||))``. Direction-preserving, so
      SE(3)-equivariant. The norm MLP is *shared* across all non-scalar blocks
      (input = all per-channel norms, output = all per-channel gates).

    This is the standard "gated nonlinearity" pattern (Weiler et al. 2018 /
    NEQUIP / e3nn).
    """

    def __init__(
        self,
        irreps: cue.Irreps,
        activation: nn.Module | None = None,
        norm_mlp_hidden: int | None = None,
    ) -> None:
        super().__init__()
        self.layout = IrrepsLayout(irreps)
        self.scalar_activation = activation if activation is not None else nn.SiLU()

        nv = self.layout.n_nonscalar_channels
        if nv > 0:
            hidden = norm_mlp_hidden if norm_mlp_hidden is not None else max(nv, 16)
            last_linear = nn.Linear(hidden, nv)
            # Identity init: sigmoid(2.0) ≈ 0.88, so gates start near 1 (not 0.5).
            nn.init.zeros_(last_linear.weight)
            nn.init.constant_(last_linear.bias, 2.0)
            self.norm_mlp = nn.Sequential(
                nn.Linear(nv, hidden),
                nn.SiLU(),
                last_linear,
            )

    def forward(self, x: Tensor) -> Tensor:
        layout = self.layout
        has_vec = bool(layout.nonscalar_blocks)
        has_scalar = bool(layout.scalar_blocks)
        if not has_vec and not has_scalar:
            return x

        out = x.clone()

        # Scalars: elementwise activation.
        for b in layout.scalar_blocks:
            out[..., b.offset : b.end] = self.scalar_activation(x[..., b.offset : b.end])

        # Non-scalars: norm-gated per-channel rescale (direction preserved).
        if has_vec:
            norms = layout.gather_nonscalar_norms(x)  # [..., n_vec_ch]
            vec_gate = torch.sigmoid(self.norm_mlp(norms))  # [..., n_vec_ch]
            scale_ch = torch.cat(
                [x.new_ones(*x.shape[:-1], layout.n_scalar_channels), vec_gate],
                dim=-1,
            )
            out = out * layout.expand_channel_scale(scale_ch)

        return out


class EquivariantRMSNorm(nn.Module):
    """Per-block RMSNorm for any ``cue.Irreps``.

    - **Scalars (l=0, p=+1)**: standard RMSNorm over the channel dimension.
      ``x → x / rms(x) * γ`` where ``rms = √(mean(x²) + ε)``.
    - **Non-scalars (l≥1)**: RMSNorm on per-channel *norms* within each
      ``(l, parity)`` block. ``v → v / rms(‖v‖) * γ`` where
      ``rms = √(mean(‖v_c‖²) + ε)`` over channels ``c`` in the block.
      Direction is preserved (divided by a per-sample scalar).

    Learnable per-channel gain ``γ`` (init = 1) for every block.

    Replaces / complements ``LayerNorm`` in the scalar path and fills the
    normalization gap in the non-scalar path.
    """

    def __init__(self, irreps: cue.Irreps, eps: float = 1e-6) -> None:
        super().__init__()
        self.layout = IrrepsLayout(irreps)
        self.eps = eps
        # One gain per channel, ordered: scalar blocks first, then non-scalar.
        self.gain = nn.Parameter(torch.ones(self.layout.n_channels))

    def forward(self, x: Tensor) -> Tensor:
        layout = self.layout
        out = x.clone()
        ch = 0

        for b in layout.scalar_blocks:
            g = self.gain[ch : ch + b.mul]
            s = x[..., b.offset : b.end]  # [..., mul]
            rms = (s.pow(2).mean(dim=-1, keepdim=True) + self.eps).sqrt()  # [..., 1]
            out[..., b.offset : b.end] = s / rms * g
            ch += b.mul

        for b in layout.nonscalar_blocks:
            g = self.gain[ch : ch + b.mul]
            lead = x.shape[:-1]
            v = x[..., b.offset : b.end].reshape(*lead, b.mul, b.ir_dim)  # [..., mul, ir_dim]
            norms_sq = v.pow(2).sum(dim=-1)  # [..., mul]
            rms = (norms_sq.mean(dim=-1, keepdim=True) + self.eps).sqrt()  # [..., 1]
            v_normed = v / rms.unsqueeze(-1)  # [..., mul, ir_dim]
            out[..., b.offset : b.end] = (v_normed * g.unsqueeze(-1)).reshape(*lead, b.span)
            ch += b.mul

        return out


class EquivariantDropout(nn.Module):
    """Dropout that preserves SE(3)-equivariance.

    - Scalars: standard element-wise Bernoulli mask (per-channel).
    - Non-scalars: one Bernoulli mask per channel, broadcast over the irrep
      dimension → each vector/tensor is kept or zeroed as a whole, so direction
      is preserved.
    """

    def __init__(self, irreps: cue.Irreps, p: float = 0.1) -> None:
        super().__init__()
        self.layout = IrrepsLayout(irreps)
        self.p = float(p)

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.p <= 0:
            return x
        out = x.clone()
        inv_p = 1.0 / (1.0 - self.p)
        lead = x.shape[:-1]
        for b in self.layout.scalar_blocks:
            mask = (torch.rand(lead + (b.mul,), device=x.device) > self.p).to(x.dtype)
            out[..., b.offset : b.end] = x[..., b.offset : b.end] * mask * inv_p
        for b in self.layout.nonscalar_blocks:
            mask = (torch.rand(lead + (b.mul, 1), device=x.device) > self.p).to(x.dtype)
            mask = mask.expand(*lead, b.mul, b.ir_dim).reshape(lead + (b.span,))
            out[..., b.offset : b.end] = x[..., b.offset : b.end] * mask * inv_p
        return out


class EquivariantAdaLN(nn.Module):
    """DiT-style adaptive normalization for mixed scalar + non-scalar irreps.

    Uses :class:`EquivariantRMSNorm` to normalize ALL channels (scalars AND
    non-scalars) before applying time-conditioned affine/gating:

    - Scalars: ``RMSNorm(x) · (1 + γ) + β`` with ``γ, β`` from ``cond``.
    - Non-scalars: ``RMSNorm(v) · (1 + 0.1 · tanh(γ_v))`` with ``γ_v`` from ``cond``.

    Zero-init on the conditioning linear → identity at initialization.
    """

    def __init__(self, irreps: cue.Irreps, cond_dim: int) -> None:
        super().__init__()
        self.layout = IrrepsLayout(irreps)
        ns = self.layout.n_scalar_channels
        nv = self.layout.n_nonscalar_channels

        self.norm = EquivariantRMSNorm(irreps)
        self.cond = nn.Linear(cond_dim, 2 * ns + nv)
        nn.init.zeros_(self.cond.weight)
        nn.init.zeros_(self.cond.bias)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        layout = self.layout
        ns = layout.n_scalar_channels
        nv = layout.n_nonscalar_channels
        cond_out = self.cond(cond)

        # Normalize ALL channels (scalar + non-scalar) uniformly
        out = self.norm(x)

        # Scalar conditioning: affine (scale + shift)
        if ns > 0:
            scalars = torch.cat([out[..., b.offset : b.end] for b in layout.scalar_blocks], dim=-1)
            scalars = scalars * (1.0 + cond_out[..., :ns]) + cond_out[..., ns : 2 * ns]
            cur = 0
            for b in layout.scalar_blocks:
                out[..., b.offset : b.end] = scalars[..., cur : cur + b.mul]
                cur += b.mul

        # Non-scalar conditioning: bounded per-channel gate (direction-preserving)
        if nv > 0:
            vec_gate_ch = 1.0 + 0.1 * torch.tanh(cond_out[..., 2 * ns :])
            scale_full = layout.expand_channel_scale(
                torch.cat([x.new_ones(*vec_gate_ch.shape[:-1], ns), vec_gate_ch], dim=-1)
            )
            out = out * scale_full

        return out


# ---------------------------------------------------------------------------
# Equivariant building blocks (Linear → Activation → Dropout)
# ---------------------------------------------------------------------------


class EquivariantBlock(nn.Module):
    """Equivariant Linear → Activation → Dropout.

    A composable building block that applies a single equivariant linear map
    followed by a nonlinearity and dropout. Residual wiring is the caller's
    responsibility — compose with ``x + block(x)`` (internal residual) or
    ``h + block(msg)`` (external residual on a different tensor) as needed.

    Layout-agnostic: any ``cue.Irreps`` for input / output.
    """

    def __init__(
        self,
        irreps_in: cue.Irreps,
        irreps_out: cue.Irreps,
        dropout: float = 0.0,
        activation: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.linear = cuet.Linear(
            irreps_in,
            irreps_out,
            layout=cue.mul_ir,
            method="fused_tp",
        )
        self.activation = EquivariantActivation(irreps_out, activation=activation)
        self.dropout = EquivariantDropout(irreps_out, p=dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(self.activation(self.linear(x)))


class EquivariantMLP(nn.Module):
    """Stack of :class:`EquivariantBlock` with residual connections at matching widths.

    Structure:
        in → hidden → hidden → ... → out

    Each block applies ``Linear → Activation → Dropout``. Blocks whose input
    and output irreps match get an internal residual connection automatically.
    The final block has no nonlinearity on its output (``EquivariantLinear``
    only), matching standard MLP conventions where the last layer is pre-norm.

    Layout-agnostic.
    """

    def __init__(
        self,
        irreps_in: cue.Irreps,
        irreps_hidden: cue.Irreps,
        irreps_out: cue.Irreps,
        num_layers: int = 2,
        dropout: float = 0.0,
        activation: nn.Module | None = None,
    ) -> None:
        super().__init__()
        assert num_layers >= 1, "num_layers must be >= 1"

        blocks: list[nn.Module] = []
        if num_layers == 1:
            blocks.append(cuet.Linear(irreps_in, irreps_out, layout=cue.mul_ir, method="fused_tp"))
            self._residual_targets: list[bool] = [False]
        else:
            blocks.append(EquivariantBlock(irreps_in, irreps_hidden, dropout, activation))
            self._residual_targets = [irreps_in == irreps_hidden]
            for _ in range(num_layers - 2):
                blocks.append(EquivariantBlock(irreps_hidden, irreps_hidden, dropout, activation))
                self._residual_targets.append(True)
            # Final layer: linear only (no activation/dropout).
            blocks.append(
                cuet.Linear(irreps_hidden, irreps_out, layout=cue.mul_ir, method="fused_tp")
            )
            self._residual_targets.append(irreps_hidden == irreps_out)

        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: Tensor) -> Tensor:
        for block, add_residual in zip(self.blocks, self._residual_targets):
            h = block(x)
            x = x + h if add_residual else h
        return x


# ---------------------------------------------------------------------------
# Gated equivariant convolution (message passing)
# ---------------------------------------------------------------------------


class GatedEquivariantConv(nn.Module):
    """SE(3)-equivariant graph convolution.

    Pipeline (per edge, all ``cue.Irreps``-aware):

    1. ``h_src ← h[src] * radial_in(edge_scalars)``     — input-side radial scale
    2. ``msg   ← TP(h_src, SH(edge_vec))``              — shared-weight TP
    3. ``msg   ← ScalarAct(msg * radial_out(edge_scalars))`` — output-side scale + SiLU
    4. ``s_gate = σ(gate_mlp) · exp(-d/σ_learn)``       — edge attention
    5. ``scalars = Σ(s_gate · msg_s) / Σ s_gate``       — gate-normalized mean
    6. For each non-scalar block: per-channel gate ``v_gate_ch``, gate-normalized
       scatter, direction/magnitude split, per-l-group norm rescaling.
    7. ``out = aggregated + self_linear(h)``            — self-interaction residual

    Layout-agnostic: any combination of l/parity/multiplicity in ``node_irreps``.
    """

    def __init__(
        self,
        node_irreps: cue.Irreps,
        sh_lmax: int,
        edge_scalar_dim: int,
        n_edge_types: int = 10,
        radial_hidden: int = 128,
        dist_decay_init: float = 8.0,
    ) -> None:
        super().__init__()
        self.layout = IrrepsLayout(node_irreps)

        # Spherical harmonics: 1x0e + 1x1o + 1x2e + ...
        ls = list(range(sh_lmax + 1))
        sh_irreps = cue.Irreps(
            "O3",
            " + ".join(f"1x{l}{'e' if l % 2 == 0 else 'o'}" for l in ls),
        )
        self.sh = cuet.SphericalHarmonics(ls=ls, normalize=True)

        # Shared-weight fully connected TP (internal params).
        self.tp = cuet.FullyConnectedTensorProduct(
            node_irreps,
            sh_irreps,
            node_irreps,
            layout_in1=cue.mul_ir,
            layout_in2=cue.mul_ir,
            layout_out=cue.mul_ir,
            internal_weights=True,
            shared_weights=True,
        )

        # Dual radial MLP (MACE/NEQUIP-style): per-channel scale on input & output.
        # Identity init: zero weight/bias → output 0 → scale = 1 + 0 = 1.
        n_ch = self.layout.n_channels
        self.radial_trunk = nn.Sequential(
            nn.Linear(edge_scalar_dim, radial_hidden),
            nn.SiLU(),
        )
        self.radial_in = nn.Linear(radial_hidden, n_ch)
        self.radial_out = nn.Linear(radial_hidden, n_ch)
        nn.init.zeros_(self.radial_in.weight)
        nn.init.zeros_(self.radial_in.bias)
        nn.init.zeros_(self.radial_out.weight)
        nn.init.zeros_(self.radial_out.bias)

        # Post-TP nonlinearity (scalars + norm-gated vectors).
        self.msg_act = EquivariantActivation(node_irreps)

        # Edge attention: 1 scalar gate + per-non-scalar-channel gate.
        gate_hidden = max(radial_hidden // 2, 16)
        self.gate_mlp = nn.Sequential(
            nn.Linear(edge_scalar_dim, gate_hidden),
            nn.SiLU(),
            nn.Linear(gate_hidden, 1 + self.layout.n_nonscalar_channels),
        )

        # Per-edge-type distance decay: gate *= exp(-d / σ_etype).
        self.log_dist_sigma = nn.Embedding(n_edge_types, 1)
        nn.init.constant_(self.log_dist_sigma.weight, float(torch.tensor(dist_decay_init).log()))

        # Per-l-group norm rescaling (direction-magnitude split output stage).
        self.vec_rescale = nn.ModuleDict()
        for l, indices in self.layout.l_groups.items():
            n_group_ch = sum(self.layout.nonscalar_blocks[i].mul for i in indices)
            self.vec_rescale[str(l)] = nn.Linear(n_group_ch, n_group_ch)

    def forward(
        self,
        h: Tensor,
        edge_vec: Tensor,
        edge_scalars: Tensor,
        src_idx: Tensor,
        dst_idx: Tensor,
        n_dst: int,
        edge_dist: Tensor | None = None,
        edge_type: Tensor | None = None,
    ) -> Tensor:
        device, dtype = h.device, h.dtype
        layout = self.layout

        # --- 1) TP with dual radial scaling (identity init: scale = 1 + δ) ---
        edge_sh = self.sh(edge_vec)
        trunk = self.radial_trunk(edge_scalars)
        scale_in = 1.0 + layout.expand_channel_scale(self.radial_in(trunk))
        scale_out = 1.0 + layout.expand_channel_scale(self.radial_out(trunk))

        h_src_scaled = h[src_idx] * scale_in
        msg = self.tp(h_src_scaled, edge_sh)
        msg = self.msg_act(msg * scale_out)

        # --- 2) Edge attention gates with per-edge-type distance decay ---
        gate_out = self.gate_mlp(edge_scalars)
        s_gate = torch.sigmoid(gate_out[:, :1])
        if edge_dist is not None and edge_type is not None:
            sigma = self.log_dist_sigma(edge_type.long()).exp()  # [E, 1]
            s_gate = s_gate * torch.exp(-edge_dist.unsqueeze(-1) / sigma)
        elif edge_dist is not None:
            # Fallback: use mean of all edge-type sigmas
            sigma = self.log_dist_sigma.weight.mean().exp()
            s_gate = s_gate * torch.exp(-edge_dist.unsqueeze(-1) / sigma)

        out = torch.zeros(n_dst, layout.feat_dim, device=device, dtype=dtype)

        # --- 3) Scalar aggregation: gate-normalized mean ---
        gate_sum = _scatter_add(s_gate, dst_idx, n_dst)  # [n_dst, 1]
        for b in layout.scalar_blocks:
            agg = _scatter_add(s_gate * msg[:, b.offset : b.end], dst_idx, n_dst)
            out[:, b.offset : b.end] = agg / (gate_sum + 1e-6)

        # --- 4) Non-scalar aggregation: gate-normalized + per-l norm rescale ---
        if layout.nonscalar_blocks:
            v_gate_all = torch.sigmoid(gate_out[:, 1 : 1 + layout.n_nonscalar_channels])

            dir_per_block: list[Tensor | None] = [None] * len(layout.nonscalar_blocks)
            norm_per_block: list[Tensor | None] = [None] * len(layout.nonscalar_blocks)

            vch = 0
            for i, b in enumerate(layout.nonscalar_blocks):
                v_gate_b = v_gate_all[:, vch : vch + b.mul]  # [E, mul]
                vch += b.mul

                v = msg[:, b.offset : b.end].view(-1, b.mul, b.ir_dim)
                g = s_gate.unsqueeze(-1) * v_gate_b.unsqueeze(-1)  # [E, mul, 1]
                v_gated = (g * v).reshape(-1, b.span)

                agg = _scatter_add(v_gated, dst_idx, n_dst).view(n_dst, b.mul, b.ir_dim)
                g_sum_ch = _scatter_add(s_gate * v_gate_b, dst_idx, n_dst)  # [n_dst, mul]
                agg = agg / (g_sum_ch.unsqueeze(-1) + 1e-6)

                v_norm = agg.norm(dim=-1, keepdim=True).clamp(min=1e-6)  # [n_dst, mul, 1]
                dir_per_block[i] = agg / v_norm
                norm_per_block[i] = v_norm.squeeze(-1)  # [n_dst, mul]

            for l, indices in layout.l_groups.items():
                group_norms = torch.cat([norm_per_block[i] for i in indices], dim=-1)
                rescale = torch.nn.functional.silu(self.vec_rescale[str(l)](group_norms))
                gch = 0
                for i in indices:
                    b = layout.nonscalar_blocks[i]
                    scale = rescale[:, gch : gch + b.mul].unsqueeze(-1).expand(-1, -1, b.ir_dim)
                    out[:, b.offset : b.end] = (dir_per_block[i] * scale).reshape(n_dst, b.span)
                    gch += b.mul

        return out


__all__ = [
    "IrrepBlock",
    "IrrepsLayout",
    "EquivariantActivation",
    "EquivariantRMSNorm",
    "EquivariantDropout",
    "EquivariantAdaLN",
    "EquivariantBlock",
    "EquivariantMLP",
    "GatedEquivariantConv",
]
