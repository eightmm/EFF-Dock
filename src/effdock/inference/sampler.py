"""ODE sampler utilities for the unified fragment-flow pipeline.

Primary entry point: :func:`sample_unified` runs a batched ODE integration
from the SE(3) prior to t=1 for one complex at a time; :func:`build_batched_graph`
replicates a single-complex graph across samples for parallel sampling.

Additional helper :func:`build_time_grid` produces non-uniform t-grids
(``late`` schedule concentrates steps near t=1 to match the main model's
late-biased training distribution).
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from effdock.geometry.flow_matching import (
    integrate_se3_step,
    sample_prior_rotations,
    vp_score_alpha_beta,
)
from effdock.geometry.se3 import (
    axis_angle_to_quaternion,
    quaternion_inverse,
    quaternion_multiply,
    quaternion_to_axis_angle,
    quaternion_to_matrix,
    standardize_quaternion,
)
from effdock.inference.defaults import DEFAULT_NUM_STEPS


# ---------------------------------------------------------------------------
# Time grid
# ---------------------------------------------------------------------------
def build_time_grid(
    num_steps: int,
    *,
    schedule: str = "uniform",
    power: float = 3.0,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """Construct a monotone time grid on ``[0, 1]`` for ODE integration."""
    if num_steps <= 0:
        raise ValueError(f"num_steps must be positive, got {num_steps}.")
    if power <= 0:
        raise ValueError(f"power must be positive, got {power}.")

    if dtype is None:
        dtype = torch.float32

    u = torch.linspace(0.0, 1.0, num_steps + 1, device=device, dtype=dtype)
    if schedule == "uniform" or power == 1.0:
        return u
    if schedule == "late":
        return 1.0 - (1.0 - u) ** power
    if schedule == "early":
        return u**power
    raise ValueError(f"Unknown time schedule '{schedule}'.")


def linear_fm_gaussian_translation_score(
    position: Tensor,
    velocity: Tensor,
    time: Tensor | float,
    prior_sigma: Tensor | float,
) -> Tensor:
    """Recover the linear-CFM translation score for an isotropic Gaussian prior.

    For ``X_t = t X_1 + (1 - t) X_0`` with
    ``X_0 ~ N(0, prior_sigma**2 I)``, Proposition 2 of Feynman-Kac-Flow gives

    ``grad log p_t(X_t) = (t * v_t(X_t) - X_t) / ((1 - t) * prior_sigma**2)``.

    EFF-Dock fragment translations are pocket-centred before integration, so
    ``position`` is expressed in the zero-mean prior frame required by the
    identity. Rotation is deliberately excluded because its prior is uniform
    on SO(3), not Gaussian.
    """
    if position.shape != velocity.shape or position.shape[-1] != 3:
        raise ValueError(
            "position and velocity must have identical [..., 3] shapes, got "
            f"{position.shape} and {velocity.shape}"
        )
    t = torch.as_tensor(time, device=position.device, dtype=position.dtype)
    sigma = torch.as_tensor(prior_sigma, device=position.device, dtype=position.dtype)
    if not bool(torch.isfinite(t).all()) or bool((t < 0.0).any()) or bool((t >= 1.0).any()):
        raise ValueError("linear-FM score time must be finite and lie in [0, 1)")
    if not bool(torch.isfinite(sigma).all()) or bool((sigma <= 0.0).any()):
        raise ValueError("linear-FM Gaussian prior sigma must be finite and positive")
    return (t * velocity - position) / ((1.0 - t) * sigma.square())


def score_corrected_translation_drift(
    position: Tensor,
    velocity: Tensor,
    time: Tensor | float,
    prior_sigma: Tensor | float,
    diffusion_sigma: Tensor | float,
) -> Tensor:
    """Return ``v + 0.5 * g(t)^2 * score`` for the translation SDE drift."""
    score = linear_fm_gaussian_translation_score(position, velocity, time, prior_sigma)
    diffusion = torch.as_tensor(
        diffusion_sigma,
        device=position.device,
        dtype=position.dtype,
    )
    if not bool(torch.isfinite(diffusion).all()) or bool((diffusion < 0.0).any()):
        raise ValueError("translation diffusion sigma must be finite and non-negative")
    return velocity + 0.5 * diffusion.square() * score


def vp_score_noise_to_velocity(
    T: Tensor,
    eps_pred: Tensor,
    t: Tensor,
    dt: Tensor,
    sigma_per_frag: Tensor,
    *,
    score_t_min: float = 1e-3,
    score_alpha_min: float = 0.0,
) -> Tensor:
    """Convert normalized VP noise prediction to a DDIM-style velocity.

    The model predicts ``eps`` for a VP denoising path.
    We estimate ``T_1`` and take the deterministic update to ``t + dt``.
    """
    t_eff = t.to(device=T.device, dtype=T.dtype).clamp(min=score_t_min, max=1.0)
    t_next = (t.to(device=T.device, dtype=T.dtype) + dt.to(device=T.device, dtype=T.dtype)).clamp(
        max=1.0
    )
    a, b = vp_score_alpha_beta(t_eff, score_alpha_min=score_alpha_min)
    a_next, b_next = vp_score_alpha_beta(t_next, score_alpha_min=score_alpha_min)
    a = a.clamp_min(1e-6)
    sigma = sigma_per_frag.to(device=T.device, dtype=T.dtype).view(-1, 1)
    eps = eps_pred.to(dtype=T.dtype)
    T1_hat = (T - b * sigma * eps) / a
    T_next = a_next * T1_hat + b_next * sigma * eps
    return (T_next - T) / dt.to(device=T.device, dtype=T.dtype).clamp_min(1e-6)


def vp_score_rotation_noise_to_angular_velocity(
    q: Tensor,
    eps_pred: Tensor,
    t: Tensor,
    dt: Tensor,
    *,
    rot_sigma_max: float = 3.141592653589793,
    frag_sizes: Tensor | None = None,
    score_t_min: float = 1e-3,
    score_alpha_min: float = 0.0,
) -> Tensor:
    """Convert SO(3) tangent-noise prediction to an integration velocity."""
    t_eff = t.to(device=q.device, dtype=eps_pred.dtype).clamp(min=score_t_min, max=1.0)
    t_next = (
        t.to(device=q.device, dtype=eps_pred.dtype) + dt.to(device=q.device, dtype=eps_pred.dtype)
    ).clamp(max=1.0)
    _, beta_t = vp_score_alpha_beta(t_eff, score_alpha_min=score_alpha_min)
    _, beta_next = vp_score_alpha_beta(t_next, score_alpha_min=score_alpha_min)
    sigma_t = float(rot_sigma_max) * beta_t
    sigma_next = float(rot_sigma_max) * beta_next

    eps = eps_pred
    if frag_sizes is not None:
        mask = (frag_sizes <= 1).to(device=q.device)
        eps = torch.where(mask.unsqueeze(-1), torch.zeros_like(eps), eps)

    q_clean_hat = quaternion_multiply(
        axis_angle_to_quaternion(-sigma_t * eps),
        q,
    )
    q_next = quaternion_multiply(
        axis_angle_to_quaternion(sigma_next * eps),
        q_clean_hat,
    )
    q_next = standardize_quaternion(q_next, reference=q)
    delta_q = quaternion_multiply(q_next, quaternion_inverse(q))
    rotvec = quaternion_to_axis_angle(delta_q, shortest_path=True)
    omega = rotvec / dt.to(device=q.device, dtype=eps_pred.dtype).clamp_min(1e-6)
    if frag_sizes is not None:
        omega = torch.where(mask.unsqueeze(-1), torch.zeros_like(omega), omega)
    return omega


# ---------------------------------------------------------------------------
# Batched graph replication
# ---------------------------------------------------------------------------
_SKIP_REPLICATE_KEYS = frozenset(
    (
        "edge_index",
        "node_fragment_id",  # offsetted below
        "num_nodes",
        "num_prot_atom",
        "num_prot_res",
        "lig_frag_slice",
        "lig_atom_slice",  # metadata
    )
)


def build_batched_graph(
    graph: dict[str, Tensor],
    B: int,
    n_frags_per: int,
    device: torch.device,
) -> dict[str, Tensor]:
    """Replicate a single-complex graph ``B`` times with per-sample offsets.

    Auto-detects which tensors to replicate from their leading dim (n_nodes or
    n_edges). Index tensors referencing node ids (``edge_index``) or fragment
    ids (``node_fragment_id``) get per-sample offsets so sample blocks do not
    cross-talk.
    """
    n_nodes = graph["node_coords"].shape[0]
    n_edges = graph["edge_index"].shape[1]
    out: dict[str, Tensor] = {}

    for k, v in graph.items():
        if k in _SKIP_REPLICATE_KEYS:
            out[k] = v
            continue
        if not isinstance(v, torch.Tensor):
            out[k] = v
            continue
        t = v.to(device)
        if t.ndim >= 1 and (t.shape[0] == n_nodes or t.shape[0] == n_edges):
            out[k] = t.repeat(B, *([1] * (t.ndim - 1))) if t.ndim > 1 else t.repeat(B)
        else:
            out[k] = t

    ei = graph["edge_index"].to(device)
    offsets = torch.arange(B, device=device, dtype=ei.dtype).repeat_interleave(n_edges) * n_nodes
    out["edge_index"] = ei.repeat(1, B) + offsets.unsqueeze(0)

    if "node_fragment_id" in graph:
        nf = graph["node_fragment_id"].to(device).repeat(B).clone()
        for i in range(B):
            sl_lo, sl_hi = i * n_nodes, (i + 1) * n_nodes
            seg = nf[sl_lo:sl_hi]
            pos = seg >= 0
            seg[pos] = seg[pos] + i * n_frags_per
            nf[sl_lo:sl_hi] = seg
        out["node_fragment_id"] = nf

    out["batch"] = torch.arange(B, device=device).repeat_interleave(n_nodes)
    out["frag_batch"] = torch.arange(B, device=device).repeat_interleave(n_frags_per)
    return out


def sample_shared_prior_states(
    num_samples: int,
    n_fragments: int,
    frag_sizes: Tensor,
    *,
    translation_sigma: float,
    seed: int,
) -> tuple[Tensor, Tensor]:
    """Create a deterministic CPU prior pool that can be sliced across budgets.

    Sampling one common pool before choosing ``num_samples`` prevents a change
    in candidate count from shifting the translation/rotation RNG streams.
    """
    if num_samples <= 0 or n_fragments <= 0:
        raise ValueError("num_samples and n_fragments must be positive")
    if translation_sigma <= 0:
        raise ValueError("translation_sigma must be positive")
    if frag_sizes.shape != (n_fragments,):
        raise ValueError("frag_sizes must have one entry per fragment")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    repeated_sizes = frag_sizes.detach().cpu().to(torch.long).repeat(num_samples)
    translations = float(translation_sigma) * torch.randn(
        num_samples * n_fragments,
        3,
        dtype=torch.float32,
        generator=generator,
    )
    rotations = sample_prior_rotations(
        num_samples * n_fragments,
        frag_sizes=repeated_sizes,
        generator=generator,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    return (
        translations.view(num_samples, n_fragments, 3),
        rotations.view(num_samples, n_fragments, 4),
    )


# ---------------------------------------------------------------------------
# Unified ODE sampler (batched across N samples of one complex)
# ---------------------------------------------------------------------------
def sample_unified(
    model: torch.nn.Module,
    graph: dict[str, Tensor],
    lig_data: dict,
    meta: dict,
    num_samples: int = 1,
    *,
    num_steps: int = DEFAULT_NUM_STEPS,
    translation_sigma: float | Tensor = 5.0,
    time_schedule: str = "late",
    schedule_power: float = 3.0,
    device: torch.device = torch.device("cpu"),
    save_traj: bool = False,
    stochastic_gamma: float = 0.0,
    translation_sde_base_sigma: float = 0.0,
    translation_sde_generator: torch.Generator | None = None,
    pose_objective: str = "linear_fm",
    score_rot_sigma_max: float = 3.141592653589793,
    score_alpha_min: float = 0.0,
    guidance_fn=None,
    guidance_scale: float = 0.0,
    guidance_min_t: float = 0.0,
    guidance_operator_split: bool = False,
    guidance_direct_drift: bool = False,
    start_t: float = 0.0,
    initial_T_frag: Tensor | None = None,
    initial_q_frag: Tensor | None = None,
    fk_resample_times: list[float] | tuple[float, ...] | None = None,
    fk_resampler=None,
    fk_resample_trans_sigma: float = 0.0,
    fk_resample_rot_sigma: float = 0.0,
    particle_resample_times: list[float] | tuple[float, ...] | None = None,
    particle_resample_fn=None,
    particle_resample_trans_sigma: float = 0.0,
    particle_resample_rot_sigma: float = 0.0,
) -> list[dict[str, Tensor]]:
    """Run batched ODE integration for ``num_samples`` poses of one complex.

    All samples share the same protein/ligand graph but have independent SE(3)
    priors; the graph is replicated ``num_samples`` times and fed through the
    model once per ODE step instead of once per sample, for a ~4x speedup.

    ``translation_sigma`` may be a scalar (all samples share σ) **or** a
    tensor of shape ``[num_samples]`` carrying a per-sample prior σ. The
    per-sample form fuses multi-σ inference into a single batched ODE call,
    avoiding the sequential overhead of ``sample_unified_multi_sigma``. The
    σ value is also fed to the model via ``batch["prior_sigma"]`` so the
    σ-conditional time embedding learnt during training is exercised.

    ``stochastic_gamma > 0`` turns the deterministic ODE into an annealed SDE by
    adding per-step Gaussian noise of scale ``γ · √(dt · (1 - t))`` to both
    translation and angular velocity. The ``√(1 - t)`` factor guarantees the
    perturbation vanishes as t → 1 so the trajectory still converges to the
    target manifold; mid-trajectory it broadens the sample distribution and
    helps escape local modes of the learnt drift.

    ``translation_sde_base_sigma > 0`` instead enables the paper-derived,
    marginal-preserving SDE on the Gaussian translation subspace. It uses
    ``g(t) = translation_sde_base_sigma * (1 - t)`` and the analytic score
    recovered from the linear flow velocity. Fragment rotations remain on the
    deterministic SO(3) flow because their uniform prior does not admit this
    Gaussian score identity.

    ``particle_resample_times`` optionally performs sequential Monte Carlo style
    resampling at specified integration times. ``particle_resample_fn`` receives
    the current pose set and returns source particle indices of length
    ``num_samples``; selected states are copied, lightly jittered, and continued.
    This is intentionally callback-based so scoring can live in benchmark code
    without coupling pose generation to an auxiliary reranker.

    ``fk_resample_times`` enables the experimental constraint-only
    Feynman--Kac path. At each resolved grid time, the already-computed learned
    velocity is extrapolated once to ``t=1`` and ``fk_resampler`` scores that
    endpoint. This path makes no additional model call and is deliberately
    restricted to ``linear_fm`` dynamics. It may use either the deterministic
    flow or the translation-only score-corrected SDE described above.

    Returns a list of length ``num_samples``. Each entry has
    ``atom_pos_pred: [N_atoms, 3]`` (pocket-centered frame) and, if
    ``save_traj=True``, ``traj: list[Tensor]`` + ``traj_times: list[float]``
    covering the ``num_steps + 1`` recorded frames. ``initial_sample_index``
    records particle ancestry after any resampling.
    """
    objective = (pose_objective or "linear_fm").lower()
    if objective not in ("linear_fm", "vp_flow", "vp_score", "vp_score_full"):
        raise ValueError(f"unknown pose_objective={pose_objective!r}")

    if start_t < 0.0 or start_t >= 1.0:
        raise ValueError(f"start_t must be in [0, 1), got {start_t}.")
    if not math.isfinite(translation_sde_base_sigma) or translation_sde_base_sigma < 0.0:
        raise ValueError("translation_sde_base_sigma must be finite and non-negative")
    translation_sde_enabled = translation_sde_base_sigma > 0.0
    if translation_sde_enabled and objective != "linear_fm":
        raise ValueError("score-corrected translation SDE currently requires linear_fm")
    if translation_sde_enabled and stochastic_gamma != 0.0:
        raise ValueError(
            "score-corrected translation SDE and legacy stochastic_gamma are mutually exclusive"
        )
    if translation_sde_enabled and guidance_scale != 0.0:
        raise ValueError("score-corrected translation SDE and gradient guidance are mutually exclusive")
    if guidance_operator_split and guidance_direct_drift:
        raise ValueError("operator-split and direct-drift guidance are mutually exclusive")
    if guidance_operator_split and guidance_scale != 0.0:
        if guidance_fn is None:
            raise ValueError("nonzero operator-split guidance requires a guidance callback")
        if not hasattr(guidance_fn, "correct"):
            raise TypeError("operator-split guidance requires a correct(...) callback")
    if guidance_direct_drift and guidance_scale != 0.0:
        if guidance_fn is None:
            raise ValueError("nonzero direct-drift guidance requires a guidance callback")
        if not hasattr(guidance_fn, "direct_velocity"):
            raise TypeError("direct-drift guidance requires a direct_velocity(...) callback")

    B = int(num_samples)
    if B < 1:
        raise ValueError("num_samples must be positive")
    requested_fk_times = tuple(float(value) for value in (fk_resample_times or ()))
    fk_enabled = bool(requested_fk_times) or fk_resampler is not None
    if bool(requested_fk_times) != (fk_resampler is not None):
        raise ValueError("FK resample times and fk_resampler must be provided together")
    if fk_enabled:
        if objective != "linear_fm":
            raise ValueError("experimental FK resampling currently requires linear_fm")
        if stochastic_gamma != 0.0:
            raise ValueError(
                "experimental FK resampling requires deterministic dynamics or the "
                "score-corrected translation SDE"
            )
        if guidance_scale != 0.0:
            raise ValueError("FK resampling and gradient guidance are mutually exclusive")
        if particle_resample_fn is not None or particle_resample_times:
            raise ValueError("FK and callback particle resampling are mutually exclusive")
        if not callable(getattr(fk_resampler, "resample", None)):
            raise TypeError("fk_resampler must provide a resample(...) method")
        if any(
            not math.isfinite(value) or value <= float(start_t) or value >= 1.0
            for value in requested_fk_times
        ):
            raise ValueError("FK resample times must lie strictly inside (start_t, 1)")
        if any(right <= left for left, right in zip(requested_fk_times, requested_fk_times[1:])):
            raise ValueError("FK resample times must be strictly increasing")
        if (
            not math.isfinite(fk_resample_trans_sigma)
            or not math.isfinite(fk_resample_rot_sigma)
            or fk_resample_trans_sigma < 0.0
            or fk_resample_rot_sigma < 0.0
        ):
            raise ValueError("FK post-resampling jitter scales must be finite and non-negative")
        if translation_sde_enabled and (
            fk_resample_trans_sigma != 0.0 or fk_resample_rot_sigma != 0.0
        ):
            raise ValueError(
                "score-corrected translation SDE cannot be combined with heuristic FK jitter"
            )

    n_frags = int(meta["num_frag"])
    pocket_center = meta["pocket_center"]
    frag_sizes = lig_data["frag_sizes"]
    frag_id = lig_data["fragment_id"]
    local_pos = lig_data["frag_local_coords"]
    n_real_atoms = local_pos.shape[0]

    batch = build_batched_graph(graph, B, n_frags, device)
    batch["node_coords"] = batch["node_coords"] - pocket_center.to(device)

    frag_sizes_flat = frag_sizes.to(device).repeat(B)
    local_pos_d = local_pos.to(device)
    frag_id_d = frag_id.to(device)
    frag_id_flat = (
        frag_id_d.repeat(B)
        + torch.arange(B, device=device).repeat_interleave(n_real_atoms) * n_frags
    )

    # Per-sample σ. Scalar broadcasts to all B samples; tensor of length B
    # encodes multi-σ fusion (one σ per sample).
    if isinstance(translation_sigma, Tensor):
        sigma_per_sample = translation_sigma.view(-1).to(dtype=torch.float32)
        if sigma_per_sample.numel() != B:
            raise ValueError(
                f"translation_sigma tensor must have {B} entries, got {sigma_per_sample.numel()}"
            )
    else:
        sigma_per_sample = torch.full((B,), float(translation_sigma), dtype=torch.float32)
    if translation_sde_enabled and (
        not bool(torch.isfinite(sigma_per_sample).all())
        or bool((sigma_per_sample <= 0.0).any())
    ):
        raise ValueError(
            "score-corrected translation SDE requires finite positive translation priors"
        )
    sigma_per_frag = sigma_per_sample.repeat_interleave(n_frags)  # [B * n_frags]

    # Prior translations (CPU sampling preserves the sequential RNG stream),
    # or caller-provided late-start states for local resampling/refinement.
    if initial_T_frag is None:
        T_flat = sigma_per_frag.view(-1, 1) * torch.randn(B * n_frags, 3, dtype=torch.float32)
    else:
        T0 = initial_T_frag.detach().to(dtype=torch.float32).reshape(B, n_frags, 3)
        T_flat = T0.reshape(B * n_frags, 3)
    if initial_q_frag is None:
        q_flat = sample_prior_rotations(
            B * n_frags,
            frag_sizes=frag_sizes_flat.cpu(),
            dtype=torch.float32,
        )
    else:
        q0 = initial_q_frag.detach().to(dtype=torch.float32).reshape(B, n_frags, 4)
        q_flat = q0.reshape(B * n_frags, 4)
    T_flat = T_flat.to(device)
    q_flat = q_flat.to(device)
    prior_sigma_d = sigma_per_sample.to(device)
    resample_times = sorted(
        float(t)
        for t in (particle_resample_times or [])
        if float(t) > float(start_t) and float(t) < 1.0
    )
    resample_idx = 0

    base_time_grid = build_time_grid(
        num_steps,
        schedule=time_schedule,
        power=schedule_power,
        device=device,
        dtype=torch.float32,
    )
    time_grid = float(start_t) + (1.0 - float(start_t)) * base_time_grid
    fk_step_schedule: dict[int, float] = {}
    for requested_time in requested_fk_times:
        resolved_step = next(
            (
                index
                for index in range(1, num_steps)
                if float(time_grid[index].item()) >= requested_time
            ),
            None,
        )
        if resolved_step is None:
            raise ValueError(
                f"FK resample time {requested_time:g} has no integration-grid state before t=1"
            )
        if resolved_step in fk_step_schedule:
            previous = fk_step_schedule[resolved_step]
            raise ValueError(
                f"FK resample times {previous:g} and {requested_time:g} resolve "
                f"to the same integration step at t={float(time_grid[resolved_step]):g}"
            )
        fk_step_schedule[resolved_step] = requested_time
    if fk_enabled and callable(getattr(fk_resampler, "reset", None)):
        fk_resampler.reset()

    frag_start, frag_end = graph["lig_frag_slice"][0].item(), graph["lig_frag_slice"][1].item()
    atom_start = graph["lig_atom_slice"][0].item()
    n_nodes = graph["node_coords"].shape[0]

    frag_slots = torch.cat(
        [torch.arange(frag_start, frag_end, device=device) + i * n_nodes for i in range(B)]
    )
    atom_slots = torch.cat(
        [
            torch.arange(atom_start, atom_start + n_real_atoms, device=device) + i * n_nodes
            for i in range(B)
        ]
    )

    traj_frames: list[Tensor] = []
    traj_times: list[float] = []
    initial_sample_index = torch.arange(B, device=device, dtype=torch.long)

    def validated_source_indices(source, *, label: str) -> Tensor:
        source_idx = torch.as_tensor(source, device=device, dtype=torch.long).view(-1)
        if source_idx.numel() != B:
            raise ValueError(
                f"{label} must return one source index per sample ({B}), got {source_idx.numel()}"
            )
        if int(source_idx.min().item()) < 0 or int(source_idx.max().item()) >= B:
            raise ValueError(f"{label} returned out-of-range source indices")
        return source_idx

    for step_idx in range(num_steps):
        t = time_grid[step_idx]
        dt = time_grid[step_idx + 1] - time_grid[step_idx]

        R_flat = quaternion_to_matrix(q_flat)
        atom_pos_flat = (
            torch.einsum("nij,nj->ni", R_flat[frag_id_flat], local_pos_d.repeat(B, 1))
            + T_flat[frag_id_flat]
        )

        nc = batch["node_coords"].clone()
        nc[frag_slots] = T_flat
        nc[atom_slots] = atom_pos_flat
        batch["node_coords"] = nc

        if save_traj:
            traj_frames.append(atom_pos_flat.view(B, n_real_atoms, 3).cpu())
            traj_times.append(t.item())

        batch["T_frag"] = T_flat
        batch["q_frag"] = q_flat
        batch["frag_sizes"] = frag_sizes_flat
        batch["t"] = t.view(1, 1).expand(B, 1).contiguous()
        batch["frag_id_for_atoms"] = frag_id_flat
        batch["prior_sigma"] = prior_sigma_d

        with torch.no_grad():
            out = model(batch)

        if objective in ("vp_score", "vp_score_full"):
            v_use = vp_score_noise_to_velocity(
                T_flat,
                out["v_pred"],
                t,
                dt,
                sigma_per_frag.to(device),
                score_alpha_min=score_alpha_min,
            )
        else:
            v_use = out["v_pred"]
        if objective == "vp_score_full":
            omega_use = vp_score_rotation_noise_to_angular_velocity(
                q_flat,
                out["omega_pred"],
                t,
                dt,
                rot_sigma_max=score_rot_sigma_max,
                score_alpha_min=score_alpha_min,
                frag_sizes=frag_sizes_flat,
            )
        else:
            omega_use = out["omega_pred"]

        t_next = float(time_grid[step_idx + 1].item())

        requested_fk_time = fk_step_schedule.get(step_idx)
        if requested_fk_time is not None:
            endpoint_horizon = 1.0 - float(t.item())
            terminal_T, terminal_q = integrate_se3_step(
                T_flat,
                q_flat,
                v_use,
                omega_use,
                endpoint_horizon,
                frag_sizes=frag_sizes_flat,
            )
            terminal_R = quaternion_to_matrix(terminal_q)
            terminal_atom_pos = (
                torch.einsum(
                    "nij,nj->ni",
                    terminal_R[frag_id_flat],
                    local_pos_d.repeat(B, 1),
                )
                + terminal_T[frag_id_flat]
            ).view(B, n_real_atoms, 3)
            with torch.no_grad():
                source_idx = fk_resampler.resample(
                    terminal_atom_pos.detach(),
                    prior_sigma=prior_sigma_d.detach(),
                    requested_time=requested_fk_time,
                    actual_time=float(t.item()),
                )
            source_idx = validated_source_indices(
                source_idx,
                label="fk_resampler.resample",
            )
            T_state = T_flat.view(B, n_frags, 3).index_select(0, source_idx).contiguous()
            q_state = q_flat.view(B, n_frags, 4).index_select(0, source_idx).contiguous()
            v_use = (v_use.view(B, n_frags, 3).index_select(0, source_idx).contiguous()).view(
                B * n_frags, 3
            )
            omega_use = (
                omega_use.view(B, n_frags, 3).index_select(0, source_idx).contiguous()
            ).view(B * n_frags, 3)
            if fk_resample_trans_sigma > 0.0:
                T_state = T_state + float(fk_resample_trans_sigma) * torch.randn_like(T_state)
            if fk_resample_rot_sigma > 0.0:
                rot_noise = float(fk_resample_rot_sigma) * torch.randn(
                    q_state.shape[:-1] + (3,),
                    device=device,
                    dtype=q_state.dtype,
                )
                dq = axis_angle_to_quaternion(rot_noise)
                q_state = standardize_quaternion(
                    quaternion_multiply(dq, q_state),
                    reference=q_state,
                )
            T_flat = T_state.view(B * n_frags, 3)
            q_flat = q_state.view(B * n_frags, 4)
            source_cpu = source_idx.cpu()
            sigma_per_sample = sigma_per_sample.index_select(0, source_cpu).contiguous()
            sigma_per_frag = sigma_per_sample.repeat_interleave(n_frags)
            prior_sigma_d = sigma_per_sample.to(device)
            initial_sample_index = initial_sample_index.index_select(0, source_idx)
            if save_traj:
                traj_frames = [
                    frame.index_select(0, source_cpu).contiguous() for frame in traj_frames
                ]

        # Direct normalized GuidanceEnergy drift.  The callback receives the
        # objective-converted learned field, maps both learned and physical
        # fragment velocities into atom-velocity space, and returns the actual
        # (already strength-scaled) correction.  The normal SE(3) integrator
        # below applies dt exactly once.
        if (
            guidance_fn is not None
            and guidance_scale != 0.0
            and guidance_direct_drift
            and t_next > guidance_min_t
        ):
            v_g, omega_g = guidance_fn.direct_velocity(
                atom_pos_flat.detach(),
                T_flat.detach(),
                v_use.detach(),
                omega_use.detach(),
                frag_sizes_flat,
                t_start=float(t.item()),
                t_end=t_next,
                strength=float(guidance_scale),
            )
            v_use = v_use + v_g
            omega_use = omega_use + omega_g

        # Physical guidance: nudge (v, omega) along -grad of an energy on the
        # current atom positions (e.g. PL-clash repulsion). guidance_fn returns
        # per-fragment (v_guide, omega_guide) already aggregated (Newton-Euler).
        if (
            guidance_fn is not None
            and guidance_scale != 0.0
            and not guidance_operator_split
            and not guidance_direct_drift
            and t.item() >= guidance_min_t
        ):
            v_g, omega_g = guidance_fn(
                atom_pos_flat.detach(), frag_id_flat, T_flat.detach(), t.item()
            )
            v_use = v_use + guidance_scale * v_g
            omega_use = omega_use + guidance_scale * omega_g

        if translation_sde_enabled:
            diffusion_sigma = float(translation_sde_base_sigma) * (1.0 - float(t.item()))
            prior_sigma_flat = sigma_per_frag.to(device=device, dtype=v_use.dtype).view(-1, 1)
            v_use = score_corrected_translation_drift(
                T_flat,
                v_use,
                t,
                prior_sigma_flat,
                diffusion_sigma,
            )
            noise_velocity_scale = diffusion_sigma / math.sqrt(max(float(dt.item()), 1e-12))
            translation_noise = torch.randn(
                v_use.shape,
                device=v_use.device,
                dtype=v_use.dtype,
                generator=translation_sde_generator,
            )
            v_use = v_use + noise_velocity_scale * translation_noise
            if not bool(torch.isfinite(v_use).all()):
                raise FloatingPointError("score-corrected translation SDE produced non-finite drift")
        elif stochastic_gamma > 0.0 and t.item() < 1.0:
            # Annealed Langevin correction: perturb velocity by γ·√((1-t)/dt)·N(0,I).
            # The 1/√dt normalization makes the noise kick size γ·√((1-t)·dt) once
            # integrated over ``dt`` via the Euler step below — matching the standard
            # Euler-Maruyama discretization of dX = v dt + σ(t) dW.
            sigma_t = stochastic_gamma * ((1.0 - t.item()) / max(dt.item(), 1e-6)) ** 0.5
            v_use = v_use + sigma_t * torch.randn_like(v_use)
            omega_use = omega_use + sigma_t * torch.randn_like(omega_use)

        T_flat, q_flat = integrate_se3_step(
            T_flat,
            q_flat,
            v_use,
            omega_use,
            dt,
            frag_sizes=frag_sizes_flat,
        )
        if (
            guidance_fn is not None
            and guidance_scale != 0.0
            and guidance_operator_split
            and t_next >= guidance_min_t
        ):
            T_flat, q_flat = guidance_fn.correct(
                T_flat,
                q_flat,
                local_pos_d,
                frag_id_d,
                frag_sizes_flat,
                dt=dt,
                t=t_next,
                scale=guidance_scale,
            )

        while (
            particle_resample_fn is not None
            and resample_idx < len(resample_times)
            and t_next >= resample_times[resample_idx]
        ):
            R_resample = quaternion_to_matrix(q_flat)
            atom_pos_resample = (
                torch.einsum("nij,nj->ni", R_resample[frag_id_flat], local_pos_d.repeat(B, 1))
                + T_flat[frag_id_flat]
            ).view(B, n_real_atoms, 3)
            with torch.no_grad():
                source_idx = particle_resample_fn(
                    atom_pos_resample.detach(),
                    T_flat.view(B, n_frags, 3).detach(),
                    q_flat.view(B, n_frags, 4).detach(),
                    prior_sigma_d.detach(),
                    resample_times[resample_idx],
                )
            source_idx = validated_source_indices(
                source_idx,
                label="particle_resample_fn",
            )
            T_state = T_flat.view(B, n_frags, 3).index_select(0, source_idx).contiguous()
            q_state = q_flat.view(B, n_frags, 4).index_select(0, source_idx).contiguous()
            if particle_resample_trans_sigma > 0.0:
                T_state = T_state + float(particle_resample_trans_sigma) * torch.randn_like(T_state)
            if particle_resample_rot_sigma > 0.0:
                rot_noise = float(particle_resample_rot_sigma) * torch.randn(
                    q_state.shape[:-1] + (3,),
                    device=device,
                    dtype=q_state.dtype,
                )
                dq = axis_angle_to_quaternion(rot_noise)
                q_state = standardize_quaternion(
                    quaternion_multiply(dq, q_state), reference=q_state
                )
            T_flat = T_state.view(B * n_frags, 3)
            q_flat = q_state.view(B * n_frags, 4)
            sigma_per_sample = sigma_per_sample.index_select(0, source_idx.cpu()).contiguous()
            sigma_per_frag = sigma_per_sample.repeat_interleave(n_frags)
            prior_sigma_d = sigma_per_sample.to(device)
            initial_sample_index = initial_sample_index.index_select(0, source_idx)
            if save_traj:
                source_cpu = source_idx.cpu()
                traj_frames = [
                    frame.index_select(0, source_cpu).contiguous() for frame in traj_frames
                ]
            resample_idx += 1

    R_final = quaternion_to_matrix(q_flat)
    atom_pos_pred_flat = (
        torch.einsum("nij,nj->ni", R_final[frag_id_flat], local_pos_d.repeat(B, 1))
        + T_flat[frag_id_flat]
    )
    final_per_sample = atom_pos_pred_flat.view(B, n_real_atoms, 3).cpu()

    if save_traj:
        traj_frames.append(final_per_sample)
        traj_times.append(1.0)

    results: list[dict[str, Tensor]] = []
    T_final = T_flat.view(B, n_frags, 3).detach().cpu()
    q_final = q_flat.view(B, n_frags, 4).detach().cpu()
    sigma_final = sigma_per_sample.detach().cpu()
    initial_sample_index_final = initial_sample_index.detach().cpu()
    for i in range(B):
        res: dict[str, Tensor] = {
            "atom_pos_pred": final_per_sample[i],
            "T_frag": T_final[i],
            "q_frag": q_final[i],
            "prior_sigma": sigma_final[i],
            "initial_sample_index": initial_sample_index_final[i],
        }
        if save_traj:
            res["traj"] = [frame[i] for frame in traj_frames]
            res["traj_times"] = list(traj_times)
        results.append(res)
    return results


def sample_unified_multi_sigma(
    model: torch.nn.Module,
    graph: dict[str, Tensor],
    lig_data: dict,
    meta: dict,
    *,
    sigma_list: list[float],
    samples_per_sigma: list[int] | int,
    num_steps: int = DEFAULT_NUM_STEPS,
    time_schedule: str = "late",
    schedule_power: float = 3.0,
    device: torch.device = torch.device("cpu"),
    save_traj: bool = False,
    stochastic_gamma: float = 0.0,
    translation_sde_base_sigma: float = 0.0,
    translation_sde_generator: torch.Generator | None = None,
    pose_objective: str = "linear_fm",
    score_rot_sigma_max: float = 3.141592653589793,
    score_alpha_min: float = 0.0,
    guidance_fn=None,
    guidance_scale: float = 0.0,
    guidance_min_t: float = 0.0,
    guidance_operator_split: bool = False,
    guidance_direct_drift: bool = False,
    fk_resample_times: list[float] | tuple[float, ...] | None = None,
    fk_resampler=None,
    fk_resample_trans_sigma: float = 0.0,
    fk_resample_rot_sigma: float = 0.0,
    particle_resample_times: list[float] | tuple[float, ...] | None = None,
    particle_resample_fn=None,
    particle_resample_trans_sigma: float = 0.0,
    particle_resample_rot_sigma: float = 0.0,
) -> list[dict[str, Tensor]]:
    """Run ``sample_unified`` once per σ in ``sigma_list`` and concatenate.

    The trained model is σ-conditional (``log(σ)`` is fed into the time
    embedding), so each prior σ produces a different vector field and a
    distinct ODE trajectory. Mixing several σ values in a single inference
    call gives downstream scoring a richer pose set to choose
    from — small σ → tight refinement, large σ → wider exploration of the
    pocket basin.

    Args:
        sigma_list: σ values to sample at, e.g. ``[2.0, 3.0, 4.0, 5.0]``.
        samples_per_sigma: int (uniform) or list[int] of len(sigma_list).
        Other args identical to :func:`sample_unified`.

    Returns:
        Flat list of length ``sum(samples_per_sigma)``. Each entry has the
        same keys as ``sample_unified`` plus ``"sigma": float`` so callers
        can inspect / weight by which prior produced the pose.
    """
    if not sigma_list:
        raise ValueError("sigma_list must be non-empty")
    if isinstance(samples_per_sigma, int):
        per = [samples_per_sigma] * len(sigma_list)
    else:
        per = list(samples_per_sigma)
    if len(per) != len(sigma_list):
        raise ValueError(
            f"samples_per_sigma length ({len(per)}) must match "
            f"sigma_list length ({len(sigma_list)})"
        )

    # Fuse all σ buckets into one batched ODE call. Build a per-sample σ
    # tensor (length = total samples); the order matches sigma_list so callers
    # can map samples back to their σ for downstream analysis.
    sigma_vec: list[float] = []
    for sigma, n in zip(sigma_list, per):
        if n <= 0:
            continue
        sigma_vec.extend([float(sigma)] * int(n))
    if not sigma_vec:
        return []
    sigma_tensor = torch.tensor(sigma_vec, dtype=torch.float32)

    chunk = sample_unified(
        model,
        graph,
        lig_data,
        meta,
        num_samples=len(sigma_vec),
        num_steps=num_steps,
        translation_sigma=sigma_tensor,
        time_schedule=time_schedule,
        schedule_power=schedule_power,
        device=device,
        save_traj=save_traj,
        stochastic_gamma=stochastic_gamma,
        translation_sde_base_sigma=translation_sde_base_sigma,
        translation_sde_generator=translation_sde_generator,
        pose_objective=pose_objective,
        score_rot_sigma_max=score_rot_sigma_max,
        score_alpha_min=score_alpha_min,
        guidance_fn=guidance_fn,
        guidance_scale=guidance_scale,
        guidance_min_t=guidance_min_t,
        guidance_operator_split=guidance_operator_split,
        guidance_direct_drift=guidance_direct_drift,
        fk_resample_times=fk_resample_times,
        fk_resampler=fk_resampler,
        fk_resample_trans_sigma=fk_resample_trans_sigma,
        fk_resample_rot_sigma=fk_resample_rot_sigma,
        particle_resample_times=particle_resample_times,
        particle_resample_fn=particle_resample_fn,
        particle_resample_trans_sigma=particle_resample_trans_sigma,
        particle_resample_rot_sigma=particle_resample_rot_sigma,
    )
    for i, r in enumerate(chunk):
        r["sigma"] = float(r.get("prior_sigma", torch.tensor(sigma_vec[i])).item())
    return chunk


def parse_sigma_list(spec: str | None, num_samples: int) -> tuple[list[float], list[int]]:
    """Parse a CLI ``--sigma_list`` spec into (sigmas, per-sigma counts).

    Accepts:
        "2,3,4,5"          — split num_samples evenly across 4 σ values
        "2:10,3:10,4:20"   — explicit per-σ counts (10 + 10 + 20 = 40)
    Returns ([], []) if spec is None.
    """
    if not spec:
        return [], []
    sigmas: list[float] = []
    counts: list[int] = []
    has_explicit = ":" in spec
    if has_explicit:
        for part in spec.split(","):
            s, n = part.split(":")
            sigmas.append(float(s.strip()))
            counts.append(int(n.strip()))
        if sum(counts) != num_samples:
            raise ValueError(
                f"explicit sigma counts must sum to num_samples ({sum(counts)} != {num_samples})"
            )
    else:
        sigmas = [float(s.strip()) for s in spec.split(",")]
        # Distribute num_samples ≈ evenly (last bucket absorbs remainder).
        base = num_samples // len(sigmas)
        rem = num_samples - base * len(sigmas)
        counts = [base] * len(sigmas)
        counts[-1] += rem
    return sigmas, counts


__all__ = [
    "build_time_grid",
    "sample_shared_prior_states",
    "sample_unified",
    "sample_unified_multi_sigma",
    "parse_sigma_list",
    "build_batched_graph",
]
