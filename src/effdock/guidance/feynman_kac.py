"""Experimental Feynman--Kac resampling for constraint-only flow steering.

This module implements the derivative-free particle reweighting part of
Feynman--Kac Flow. The sampler may pair it with the paper-derived
score-corrected SDE on EFF-Dock's Gaussian translation subspace. Rotations
remain deterministic because their uniform SO(3) prior requires a separate
manifold score model. Optional post-resampling SE(3) jitter is classified as a
heuristic and is kept distinct from the continuous score-corrected dynamics.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field

import torch
from torch import Tensor

from .physical import PhysicalEnergyConfig, physical_energy
from .system import PhysicalSystem

DEFAULT_FK_CONSTRAINT_TERMS = (
    "ligand_intra_bond",
    "ligand_intra_angle",
    "ligand_intra_improper",
    "ligand_intra_lj_repulsive",
    "protein_ligand_steric_barrier",
    "receptor_geometry_obstacle_uff_repulsive",
    "receptor_geometry_obstacle_generic_repulsive",
)

SUPPORTED_FK_CONSTRAINT_TERMS = frozenset(DEFAULT_FK_CONSTRAINT_TERMS)
_OPTIONAL_ZERO_TERMS = frozenset(
    {
        "receptor_geometry_obstacle_uff_repulsive",
        "receptor_geometry_obstacle_generic_repulsive",
    }
)
_SUPPORTED_DYNAMICS = frozenset(
    {
        "deterministic_flow_without_score_corrected_sde",
        "translation_score_corrected_sde_deterministic_so3",
    }
)


def parse_fk_resample_times(spec: str | None) -> tuple[float, ...]:
    """Parse a strict, increasing comma-separated schedule in ``(0, 1)``."""
    if spec is None or not spec.strip():
        return ()
    try:
        values = tuple(float(value.strip()) for value in spec.split(","))
    except ValueError as exc:
        raise ValueError("FK resample times must be comma-separated numbers") from exc
    if any(not 0.0 < value < 1.0 for value in values):
        raise ValueError("FK resample times must lie strictly inside (0, 1)")
    if any(right <= left for left, right in zip(values, values[1:])):
        raise ValueError("FK resample times must be strictly increasing")
    return values


@dataclass(frozen=True)
class FKConstraintConfig:
    """Numerical contract for constraint-only FK resampling."""

    beta: float
    resample_method: str = "systematic"
    terms: tuple[str, ...] = DEFAULT_FK_CONSTRAINT_TERMS
    seed: int = 0
    group_by_prior_sigma: bool = True
    negative_tolerance: float = 1e-6
    dynamics: str = "deterministic_flow_without_score_corrected_sde"
    translation_sde_base_sigma: float = 0.0
    energy: PhysicalEnergyConfig = field(default_factory=PhysicalEnergyConfig)

    def __post_init__(self) -> None:
        if not math.isfinite(self.beta) or self.beta < 0.0:
            raise ValueError("FK beta must be finite and non-negative")
        if self.resample_method not in {"systematic", "multinomial"}:
            raise ValueError("FK resample_method must be systematic or multinomial")
        if not self.terms:
            raise ValueError("FK constraint terms must be non-empty")
        if len(set(self.terms)) != len(self.terms):
            raise ValueError("FK constraint terms must be unique")
        unsupported = sorted(set(self.terms) - SUPPORTED_FK_CONSTRAINT_TERMS)
        if unsupported:
            raise ValueError(f"unsupported FK constraint terms: {unsupported}")
        if not math.isfinite(self.negative_tolerance) or self.negative_tolerance < 0.0:
            raise ValueError("FK negative_tolerance must be finite and non-negative")
        if self.dynamics not in _SUPPORTED_DYNAMICS:
            raise ValueError(f"unsupported FK dynamics: {self.dynamics!r}")
        if (
            not math.isfinite(self.translation_sde_base_sigma)
            or self.translation_sde_base_sigma < 0.0
        ):
            raise ValueError("translation_sde_base_sigma must be finite and non-negative")
        expects_sde = self.dynamics == "translation_score_corrected_sde_deterministic_so3"
        if expects_sde != (self.translation_sde_base_sigma > 0.0):
            raise ValueError("FK dynamics and translation_sde_base_sigma must agree")


def constraint_potential(
    coords: Tensor,
    system: PhysicalSystem,
    config: FKConstraintConfig,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Return a non-negative potential assembled from explicit constraint terms.

    Attractive physical terms and every interaction term are intentionally
    outside this whitelist.  Optional receptor-obstacle terms are exactly zero
    when the corresponding admitted obstacle class is absent.
    """
    components = physical_energy(coords, system, config.energy)
    reference = components["total"]
    selected: dict[str, Tensor] = {}
    potential = torch.zeros_like(reference)
    for name in config.terms:
        value = components.get(name)
        if value is None:
            if name not in _OPTIONAL_ZERO_TERMS:
                raise KeyError(f"physical energy did not return FK term {name!r}")
            value = torch.zeros_like(reference)
        if value.shape != reference.shape:
            raise ValueError(
                f"FK term {name!r} shape {tuple(value.shape)} does not match "
                f"per-pose shape {tuple(reference.shape)}"
            )
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError(f"FK constraint term {name!r} is non-finite")
        minimum = float(value.detach().amin().cpu())
        if minimum < -float(config.negative_tolerance):
            raise FloatingPointError(
                f"FK constraint term {name!r} must be non-negative, got {minimum}"
            )
        safe = value.clamp_min(0.0)
        selected[name] = safe
        potential = potential + safe
    if not bool(torch.isfinite(potential).all()):
        raise FloatingPointError("FK constraint potential is non-finite")
    return potential, selected


def _index_sha256(indices: Tensor) -> str:
    array = indices.detach().cpu().to(torch.int64).contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(b"EFFDOCK_FK_SOURCE_INDEX_V1\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


class FeynmanKacConstraintResampler:
    """Stateful difference-schedule FK resampler with auditable genealogy."""

    def __init__(
        self,
        system: PhysicalSystem,
        config: FKConstraintConfig,
    ) -> None:
        self.system = system
        self.config = config
        self._generator = torch.Generator(device="cpu")
        self._generator.manual_seed(int(config.seed))
        self._previous_potential: Tensor | None = None
        self._initial_ancestor: Tensor | None = None
        self._events: list[dict[str, object]] = []

    @property
    def is_active(self) -> bool:
        return self.config.beta > 0.0

    def reset(self) -> None:
        self._generator.manual_seed(int(self.config.seed))
        self._previous_potential = None
        self._initial_ancestor = None
        self._events.clear()

    def _sample_local(self, weights: Tensor) -> Tensor:
        count = int(weights.numel())
        if self.config.resample_method == "multinomial":
            return torch.multinomial(
                weights,
                count,
                replacement=True,
                generator=self._generator,
            )
        # One random offset gives lower-variance systematic resampling.  Exact
        # uniform weights map to the identity ordering.
        offset = torch.rand((), generator=self._generator, dtype=torch.float64) / count
        positions = offset + torch.arange(count, dtype=torch.float64) / count
        cumulative = torch.cumsum(weights, dim=0)
        cumulative[-1] = 1.0
        return torch.searchsorted(cumulative, positions).clamp_max(count - 1)

    def resample(
        self,
        terminal_atom_pos: Tensor,
        *,
        prior_sigma: Tensor,
        requested_time: float,
        actual_time: float,
    ) -> Tensor:
        """Return source indices for particles scored at a one-shot endpoint."""
        if terminal_atom_pos.ndim != 3 or terminal_atom_pos.shape[-1] != 3:
            raise ValueError("terminal_atom_pos must have shape [B,N,3]")
        batch_size = int(terminal_atom_pos.shape[0])
        if batch_size < 1:
            raise ValueError("FK resampling requires at least one particle")
        if prior_sigma.shape != (batch_size,):
            raise ValueError("prior_sigma must have one entry per FK particle")
        if not bool(torch.isfinite(prior_sigma).all()):
            raise ValueError("prior_sigma must be finite for FK resampling")
        if not math.isfinite(requested_time) or not 0.0 < requested_time < 1.0:
            raise ValueError("requested_time must lie strictly inside (0, 1)")
        if not math.isfinite(actual_time) or not 0.0 <= actual_time < 1.0:
            raise ValueError("actual_time must lie inside [0, 1)")
        if actual_time < requested_time:
            raise ValueError("actual FK resample time must not precede requested_time")
        device = terminal_atom_pos.device
        identity = torch.arange(batch_size, device=device, dtype=torch.long)
        if not self.is_active:
            return identity

        potential_device, components = constraint_potential(
            terminal_atom_pos,
            self.system,
            self.config,
        )
        potential = potential_device.detach().cpu().to(torch.float64).view(-1)
        if potential.numel() != batch_size:
            raise ValueError("FK potential must contain one value per particle")
        sigma = prior_sigma.detach().cpu().to(torch.float64).view(-1)
        if self._previous_potential is None:
            previous = torch.zeros_like(potential)
            self._initial_ancestor = torch.arange(batch_size, dtype=torch.long)
        else:
            previous = self._previous_potential
            if previous.shape != potential.shape:
                raise ValueError("FK particle count changed between resampling events")
        if self._initial_ancestor is None or self._initial_ancestor.shape != potential.shape:
            raise RuntimeError("FK genealogy state is inconsistent")

        delta = potential - previous
        source = torch.arange(batch_size, dtype=torch.long)
        group_values = torch.unique(sigma) if self.config.group_by_prior_sigma else sigma[:1]
        ess_total = 0.0
        entropy_total = 0.0
        maximum_weight = 0.0
        group_sizes: list[int] = []
        for group_value in group_values:
            if self.config.group_by_prior_sigma:
                members = torch.nonzero(sigma == group_value, as_tuple=False).view(-1)
            else:
                members = torch.arange(batch_size, dtype=torch.long)
            if not int(members.numel()):
                continue
            log_weight = -float(self.config.beta) * delta.index_select(0, members)
            if not bool(torch.isfinite(log_weight).all()):
                raise FloatingPointError("FK log weights are non-finite")
            weights = torch.softmax(log_weight, dim=0)
            if not bool(torch.isfinite(weights).all()) or float(weights.sum()) <= 0.0:
                raise FloatingPointError("FK normalized weights are invalid")
            local = self._sample_local(weights)
            source[members] = members.index_select(0, local)
            ess_total += float((1.0 / weights.square().sum()).item())
            entropy_total += float(
                (-(weights * weights.clamp_min(torch.finfo(weights.dtype).tiny).log()).sum()).item()
            )
            maximum_weight = max(maximum_weight, float(weights.max().item()))
            group_sizes.append(int(members.numel()))
            if not self.config.group_by_prior_sigma:
                break

        self._previous_potential = potential.index_select(0, source).contiguous()
        self._initial_ancestor = self._initial_ancestor.index_select(0, source).contiguous()
        component_means = {
            name: float(value.detach().to(torch.float64).mean().cpu())
            for name, value in components.items()
        }
        self._events.append(
            {
                "event_index": len(self._events),
                "requested_time": float(requested_time),
                "actual_time": float(actual_time),
                "potential_min": float(potential.min().item()),
                "potential_median": float(potential.median().item()),
                "potential_max": float(potential.max().item()),
                "delta_min": float(delta.min().item()),
                "delta_median": float(delta.median().item()),
                "delta_max": float(delta.max().item()),
                "ess": ess_total,
                "ess_fraction": ess_total / batch_size,
                "weight_entropy": entropy_total,
                "max_group_weight": maximum_weight,
                "group_sizes": group_sizes,
                "unique_parent_count": int(torch.unique(source).numel()),
                "unique_initial_ancestor_count": int(torch.unique(self._initial_ancestor).numel()),
                "source_index_sha256": _index_sha256(source),
                "constraint_term_means": component_means,
            }
        )
        return source.to(device=device)

    def final_initial_ancestors(self) -> Tensor | None:
        if self._initial_ancestor is None:
            return None
        return self._initial_ancestor.clone()

    def diagnostics(self) -> dict[str, object]:
        final_ancestors = self.final_initial_ancestors()
        return {
            "schema_version": "effdock.fk_constraint_resampling.v2",
            "estimator": "constant_velocity_euler_terminal",
            "potential_schedule": "difference",
            "dynamics": self.config.dynamics,
            "translation_sde_base_sigma": float(self.config.translation_sde_base_sigma),
            "beta": float(self.config.beta),
            "resample_method": self.config.resample_method,
            "seed": int(self.config.seed),
            "group_by_prior_sigma": bool(self.config.group_by_prior_sigma),
            "constraint_terms": list(self.config.terms),
            "num_resampling_events": len(self._events),
            "final_unique_initial_ancestors": (
                int(torch.unique(final_ancestors).numel()) if final_ancestors is not None else None
            ),
            "final_ancestor_sha256": (
                _index_sha256(final_ancestors) if final_ancestors is not None else None
            ),
            "events": [dict(event) for event in self._events],
        }


__all__ = [
    "DEFAULT_FK_CONSTRAINT_TERMS",
    "FKConstraintConfig",
    "FeynmanKacConstraintResampler",
    "SUPPORTED_FK_CONSTRAINT_TERMS",
    "constraint_potential",
    "parse_fk_resample_times",
]
