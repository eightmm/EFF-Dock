"""Differentiable, self-contained protein-ligand interaction guidance.

RDKit is used only while building the static ligand SMARTS masks. Every
coordinate-dependent value in this module is computed with Torch. These terms
are pose-guidance diagnostics, not affinity or free-energy predictions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields

import torch
from torch import Tensor

from .errors import UnsupportedPhysicalChemistryError
from .parameterization import load_interaction_v1
from .system import PhysicalSystem

PLANNED_INTERACTION_TERMS = (
    "hydrophobic",
    "hydrogen_bond",
    "screened_formal_charge",
    "pi_stacking",
    "cation_pi",
    "halogen_bond",
    "metal_coordination",
)
ACTIVE_INTERACTION_TERMS = (
    "hydrophobic",
    "hydrogen_bond",
    "screened_formal_charge",
    "pi_stacking",
    "cation_pi",
    "halogen_bond",
    "metal_coordination",
)


def metal_coordination_v0_contract() -> dict[str, object]:
    """Return the historical narrow Zn(II) V0 contract."""
    return {
        "status": "superseded_by_profile_dispatched_v1",
        "supported_scope": {
            "metal": "Zn(II)",
            "pdb_identity": "residue ZN and element ZN",
            "site": "mononuclear_tetrahedral",
            "fixed_receptor_donors": 3,
            "maximum_ligand_donors": 1,
            "ligand_donor_elements": ["N", "O", "S"],
        },
        "geometry": {
            "vacant_direction": ("v = -normalize(sum_j normalize(x_receptor_donor_j - x_Zn))"),
            "alignment": ("A = exp(-(1 - dot(normalize(x_ligand_donor - x_Zn), v)) / tau_theta)"),
        },
        "pair_energy": {
            "formula": ("E_pair = S(r) * D * (exp(-2*a*(r-r0)) - 2*A*exp(-a*(r-r0)))"),
            "trace_components": [
                "metal_radial_repulsion",
                "metal_directional_attraction",
            ],
        },
        "occupancy": {
            "formula": ("q = S_CN(r) * exp(-0.5*((r-r0)/sigma_r)^2) * A"),
            "coordination_number": "CN = N_fixed + sum_i(q_i)",
            "penalty": ("E_CN = k_over*relu(CN-CN_max)^2 + k_under*relu(CN_target-CN)^2"),
            "v0_defaults": {"CN_max": 4, "CN_target": 4, "k_under": 0.0},
            "slot_exclusion": "E_slot = k_slot * sum_{i<j}(q_i*q_j)",
        },
        "pair_masking": {
            "active_metal_donor_pair": "replace generic protein-ligand LJ",
            "metal_non_donor_pair": "short-range repulsion only",
            "metal_coulomb_v0": "inactive",
        },
        "unsupported": [
            "metal other than Zn(II)",
            "unknown oxidation state",
            "binuclear or polynuclear site",
            "non-tetrahedral or ambiguous receptor geometry",
            "receptor donor count other than three",
            "bridging donor or unresolved coordination water",
            "unsupported ligand donor protonation or resonance assignment",
        ],
        "parameter_status": (
            "r0 is literature-anchored; remaining constants are frozen "
            "EFF-Dock diagnostic priors, not affinity-fit parameters"
        ),
        "frozen_constants": {
            "r0_N_O_S_angstrom": [2.0, 2.1, 2.25],
            "D_kcal_mol": 1.5,
            "a_per_angstrom": 2.0,
            "tau_theta": 0.06,
            "sigma_r_angstrom": 0.30,
            "pair_switch_angstrom": [3.5, 4.5],
            "coordination_switch_angstrom": [2.8, 3.4],
            "k_over_kcal_mol": 4.0,
            "k_slot_kcal_mol": 4.0,
        },
        "references": [
            {
                "name": "AutoDock4Zn",
                "doi": "10.1021/ci500209e",
                "role": "published precedent for zinc-specific energetic and geometric docking terms",
            },
            {
                "name": "MetalPDB 2024",
                "doi": "10.1107/S2059798324003152",
                "role": "metal-site geometry and coordination provenance",
            },
        ],
    }


def metal_coordination_v1_contract() -> dict[str, object]:
    """Return the active profile-dispatched metal coordination contract."""
    profiles = load_interaction_v1()["metal_coordination_profiles"]
    return {
        "status": "user_requested_default_on_diagnostic",
        "default_term": "metal_coordination",
        "automatic_site_detection": True,
        "supported_standalone_elements": list(profiles),
        "attraction_enabled_profiles": [
            element
            for element, profile in profiles.items()
            if bool(profile["auto_attraction"])
        ],
        "repulsion_only_profiles": [
            element
            for element, profile in profiles.items()
            if not bool(profile["auto_attraction"])
        ],
        "site_boundary": {
            "identity": "standalone monatomic residue name must equal element",
            "attraction": (
                "complete retained mononuclear shell, exactly one vacancy, "
                "profile-supported donors, and admitted geometry"
            ),
            "repulsion_only": (
                "bounded metal-ligand clash guard with no Morse attraction; "
                "the disabled-attraction reason is traced"
            ),
            "hard_failures": [
                "metal-containing cofactor or identity mismatch",
                "nearby or multinuclear metal cluster",
                "shared or alternate-location donor",
                "partial occupancy",
                "unsupported standalone element",
            ],
        },
        "coordinate_engine": "torch_only",
        "external_engine": None,
        "profiles": {
            element: {
                "atomic_number": int(profile["atomic_number"]),
                "assumed_oxidation_state": str(profile["assumed_oxidation_state"]),
                "auto_attraction": bool(profile["auto_attraction"]),
                "attraction_status": str(profile["attraction_status"]),
                "target_coordination": int(profile["target_coordination"]),
                "geometries": list(profile["geometries"]),
                "allowed_ligand_elements": list(profile["allowed_ligand_elements"]),
                "r0_angstrom": dict(profile["r0_angstrom"]),
                "water_policy": str(profile["water_policy"]),
            }
            for element, profile in profiles.items()
        },
        "parameter_status": (
            "r0 values are structural distance targets; Morse depth/width and "
            "penalties are EFF-Dock diagnostic guidance priors, not affinity-fit constants"
        ),
        "references": [
            {
                "name": "MetalPDB 2024",
                "doi": "10.1107/S2059798324003152",
                "role": "coordination-number, geometry, nuclearity, and site-provenance boundary",
            },
            {
                "name": "Dokmanic et al. 2008",
                "doi": "10.1107/S090744490706595X",
                "role": "high-resolution PDB metal-donor structural-distance targets",
            },
            {
                "name": "Zheng et al. 2008",
                "doi": "10.1016/j.jinorgbio.2008.05.006",
                "role": "metal identity, oxidation, and geometry misassignment guardrail",
            },
        ],
    }


@dataclass(frozen=True)
class InteractionEnergyConfig:
    """Configuration for admitted interaction terms in kcal/mol and Å."""

    active_terms: tuple[str, ...] = tuple(load_interaction_v1()["active_terms"])
    hydrophobic_epsilon: float = float(load_interaction_v1()["defaults"]["hydrophobic_epsilon"])
    hydrophobic_contact_on: float = float(
        load_interaction_v1()["defaults"]["hydrophobic_contact_on_angstrom"]
    )
    hydrophobic_cutoff: float = float(
        load_interaction_v1()["defaults"]["hydrophobic_cutoff_angstrom"]
    )
    hydrogen_bond_epsilon: float = float(load_interaction_v1()["defaults"]["hydrogen_bond_epsilon"])
    hydrogen_bond_optimal_distance: float = float(
        load_interaction_v1()["defaults"]["hydrogen_bond_optimal_distance_angstrom"]
    )
    hydrogen_bond_distance_sigma: float = float(
        load_interaction_v1()["defaults"]["hydrogen_bond_distance_sigma_angstrom"]
    )
    hydrogen_bond_switch_distance: float = float(
        load_interaction_v1()["defaults"]["hydrogen_bond_switch_distance_angstrom"]
    )
    hydrogen_bond_cutoff: float = float(
        load_interaction_v1()["defaults"]["hydrogen_bond_cutoff_angstrom"]
    )
    hydrogen_bond_donor_cone_half_width_degrees: float = float(
        load_interaction_v1()["defaults"]["hydrogen_bond_donor_cone_half_width_degrees"]
    )
    hydrogen_bond_acceptor_cone_half_width_degrees: float = float(
        load_interaction_v1()["defaults"]["hydrogen_bond_acceptor_cone_half_width_degrees"]
    )
    formal_charge_coulomb_constant: float = float(
        load_interaction_v1()["defaults"]["formal_charge_coulomb_constant_kcal_mol_angstrom_per_e2"]
    )
    formal_charge_relative_dielectric: float = float(
        load_interaction_v1()["defaults"]["formal_charge_relative_dielectric"]
    )
    formal_charge_screening_kappa: float = float(
        load_interaction_v1()["defaults"]["formal_charge_screening_kappa_per_angstrom"]
    )
    formal_charge_softcore: float = float(
        load_interaction_v1()["defaults"]["formal_charge_softcore_angstrom"]
    )
    formal_charge_switch_distance: float = float(
        load_interaction_v1()["defaults"]["formal_charge_switch_distance_angstrom"]
    )
    formal_charge_cutoff: float = float(
        load_interaction_v1()["defaults"]["formal_charge_cutoff_angstrom"]
    )
    pi_stacking_epsilon: float = float(load_interaction_v1()["defaults"]["pi_stacking_epsilon"])
    pi_stacking_distance_on: float = float(
        load_interaction_v1()["defaults"]["pi_stacking_distance_on_angstrom"]
    )
    pi_stacking_cutoff: float = float(
        load_interaction_v1()["defaults"]["pi_stacking_cutoff_angstrom"]
    )
    aromatic_offset_on: float = float(
        load_interaction_v1()["defaults"]["aromatic_offset_on_angstrom"]
    )
    aromatic_offset_cutoff: float = float(
        load_interaction_v1()["defaults"]["aromatic_offset_cutoff_angstrom"]
    )
    aromatic_area_quality_zero_fraction: float = float(
        load_interaction_v1()["defaults"]["aromatic_area_quality_zero_fraction"]
    )
    aromatic_area_quality_full_fraction: float = float(
        load_interaction_v1()["defaults"]["aromatic_area_quality_full_fraction"]
    )
    pi_stacking_parallel_cosine_squared_on: float = float(
        load_interaction_v1()["defaults"]["pi_stacking_parallel_cosine_squared_on"]
    )
    pi_stacking_t_shaped_cosine_squared_off: float = float(
        load_interaction_v1()["defaults"]["pi_stacking_t_shaped_cosine_squared_off"]
    )
    cation_pi_epsilon: float = float(load_interaction_v1()["defaults"]["cation_pi_epsilon"])
    cation_pi_distance_on: float = float(
        load_interaction_v1()["defaults"]["cation_pi_distance_on_angstrom"]
    )
    cation_pi_cutoff: float = float(load_interaction_v1()["defaults"]["cation_pi_cutoff_angstrom"])
    halogen_bond_epsilon: float = float(load_interaction_v1()["defaults"]["halogen_bond_epsilon"])
    halogen_contact_scale: float = float(load_interaction_v1()["defaults"]["halogen_contact_scale"])
    halogen_cutoff_scale: float = float(load_interaction_v1()["defaults"]["halogen_cutoff_scale"])
    halogen_sigma_half_width_degrees: float = float(
        load_interaction_v1()["defaults"]["halogen_sigma_half_width_degrees"]
    )
    halogen_acceptor_half_width_degrees: float = float(
        load_interaction_v1()["defaults"]["halogen_acceptor_half_width_degrees"]
    )
    halogen_vdw_radius_nitrogen: float = float(
        load_interaction_v1()["halogen_bond_vdw_radii_angstrom"]["N"]
    )
    halogen_vdw_radius_oxygen: float = float(
        load_interaction_v1()["halogen_bond_vdw_radii_angstrom"]["O"]
    )
    halogen_vdw_radius_sulfur: float = float(
        load_interaction_v1()["halogen_bond_vdw_radii_angstrom"]["S"]
    )
    halogen_vdw_radius_chlorine: float = float(
        load_interaction_v1()["halogen_bond_vdw_radii_angstrom"]["CL"]
    )
    halogen_vdw_radius_bromine: float = float(
        load_interaction_v1()["halogen_bond_vdw_radii_angstrom"]["BR"]
    )
    halogen_vdw_radius_iodine: float = float(
        load_interaction_v1()["halogen_bond_vdw_radii_angstrom"]["I"]
    )
    metal_pair_depth: float = float(load_interaction_v1()["defaults"]["metal_pair_depth_kcal_mol"])
    metal_morse_a: float = float(load_interaction_v1()["defaults"]["metal_morse_a_per_angstrom"])
    metal_alignment_tau: float = float(load_interaction_v1()["defaults"]["metal_alignment_tau"])
    metal_occupancy_sigma: float = float(
        load_interaction_v1()["defaults"]["metal_occupancy_sigma_angstrom"]
    )
    metal_pair_switch_distance: float = float(
        load_interaction_v1()["defaults"]["metal_pair_switch_distance_angstrom"]
    )
    metal_pair_cutoff: float = float(
        load_interaction_v1()["defaults"]["metal_pair_cutoff_angstrom"]
    )
    metal_cn_switch_distance: float = float(
        load_interaction_v1()["defaults"]["metal_cn_switch_distance_angstrom"]
    )
    metal_cn_cutoff: float = float(
        load_interaction_v1()["defaults"]["metal_cn_cutoff_angstrom"]
    )
    metal_overcoordination_k: float = float(
        load_interaction_v1()["defaults"]["metal_overcoordination_k_kcal_mol"]
    )
    metal_slot_k: float = float(load_interaction_v1()["defaults"]["metal_slot_k_kcal_mol"])
    metal_non_donor_repulsion: float = float(
        load_interaction_v1()["defaults"]["metal_non_donor_repulsion_kcal_mol"]
    )
    metal_non_donor_softcore: float = float(
        load_interaction_v1()["defaults"]["metal_non_donor_softcore_angstrom"]
    )
    metal_non_donor_radius: float = float(
        load_interaction_v1()["defaults"]["metal_non_donor_radius_angstrom"]
    )
    metal_non_donor_switch_distance: float = float(
        load_interaction_v1()["defaults"]["metal_non_donor_switch_distance_angstrom"]
    )
    metal_non_donor_cutoff: float = float(
        load_interaction_v1()["defaults"]["metal_non_donor_cutoff_angstrom"]
    )
    polar_proxy_burial_on: float = float(
        load_interaction_v1()["defaults"]["polar_proxy_burial_on_angstrom"]
    )
    polar_proxy_burial_cutoff: float = float(
        load_interaction_v1()["defaults"]["polar_proxy_burial_cutoff_angstrom"]
    )
    direction_epsilon: float = float(
        load_interaction_v1()["defaults"]["direction_epsilon_angstrom"]
    )
    direction_quality_zero_below: float = float(
        load_interaction_v1()["defaults"]["direction_quality_zero_below"]
    )
    direction_quality_full_above: float = float(
        load_interaction_v1()["defaults"]["direction_quality_full_above"]
    )
    bond_direction_quality_zero_below: float = float(
        load_interaction_v1()["defaults"]["bond_direction_quality_zero_below_angstrom"]
    )
    bond_direction_quality_full_above: float = float(
        load_interaction_v1()["defaults"]["bond_direction_quality_full_above_angstrom"]
    )

    def __post_init__(self) -> None:
        for item in fields(self):
            if item.name == "active_terms":
                continue
            value = getattr(self, item.name)
            if not math.isfinite(float(value)):
                raise ValueError(f"{item.name} must be finite")
        unknown = sorted(set(self.active_terms) - set(ACTIVE_INTERACTION_TERMS))
        if unknown:
            raise ValueError(f"unsupported InteractionGuidance terms requested: {unknown}")
        if len(set(self.active_terms)) != len(self.active_terms):
            raise ValueError("active interaction terms must be unique")
        for name in (
            "hydrophobic_epsilon",
            "hydrogen_bond_epsilon",
            "hydrogen_bond_distance_sigma",
            "formal_charge_coulomb_constant",
            "formal_charge_relative_dielectric",
            "formal_charge_softcore",
            "pi_stacking_epsilon",
            "cation_pi_epsilon",
            "halogen_bond_epsilon",
            "halogen_vdw_radius_nitrogen",
            "halogen_vdw_radius_oxygen",
            "halogen_vdw_radius_sulfur",
            "halogen_vdw_radius_chlorine",
            "halogen_vdw_radius_bromine",
            "halogen_vdw_radius_iodine",
            "metal_pair_depth",
            "metal_morse_a",
            "metal_alignment_tau",
            "metal_occupancy_sigma",
            "metal_overcoordination_k",
            "metal_slot_k",
            "metal_non_donor_repulsion",
            "metal_non_donor_softcore",
            "metal_non_donor_radius",
            "direction_epsilon",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.formal_charge_screening_kappa < 0:
            raise ValueError("formal_charge_screening_kappa must be non-negative")
        if not 0 < self.hydrophobic_contact_on < self.hydrophobic_cutoff:
            raise ValueError("require 0 < hydrophobic_contact_on < hydrophobic_cutoff")
        if not (0 < self.hydrogen_bond_switch_distance < self.hydrogen_bond_cutoff):
            raise ValueError("require 0 < hydrogen_bond_switch_distance < hydrogen_bond_cutoff")
        if not (0 < self.formal_charge_switch_distance < self.formal_charge_cutoff):
            raise ValueError("require 0 < formal_charge_switch_distance < formal_charge_cutoff")
        for prefix in ("pi_stacking", "cation_pi"):
            if not (0 < getattr(self, f"{prefix}_distance_on") < getattr(self, f"{prefix}_cutoff")):
                raise ValueError(f"require 0 < {prefix}_distance_on < {prefix}_cutoff")
        if not 0 < self.aromatic_offset_on < self.aromatic_offset_cutoff:
            raise ValueError("require 0 < aromatic_offset_on < aromatic_offset_cutoff")
        if not (
            0
            < self.aromatic_area_quality_zero_fraction
            < self.aromatic_area_quality_full_fraction
            <= 1
        ):
            raise ValueError("invalid aromatic area-quality fractions")
        if not (
            0 < self.pi_stacking_parallel_cosine_squared_on < 1
            and 0 < self.pi_stacking_t_shaped_cosine_squared_off < 1
            and self.pi_stacking_t_shaped_cosine_squared_off
            < self.pi_stacking_parallel_cosine_squared_on
        ):
            raise ValueError("invalid pi-stacking orientation thresholds")
        if not 0 < self.halogen_contact_scale < self.halogen_cutoff_scale:
            raise ValueError("require 0 < halogen_contact_scale < halogen_cutoff_scale")
        for prefix in ("metal_pair", "metal_cn", "metal_non_donor"):
            if not (
                0 < getattr(self, f"{prefix}_switch_distance") < getattr(self, f"{prefix}_cutoff")
            ):
                raise ValueError(f"invalid {prefix} switch/cutoff")
        if not 0 < self.polar_proxy_burial_on < self.polar_proxy_burial_cutoff:
            raise ValueError("invalid polar proxy burial switch/cutoff")
        if not (0 < self.direction_quality_zero_below < self.direction_quality_full_above):
            raise ValueError(
                "require 0 < direction_quality_zero_below < direction_quality_full_above"
            )
        if not (
            0 < self.bond_direction_quality_zero_below < self.bond_direction_quality_full_above
        ):
            raise ValueError(
                "require 0 < bond_direction_quality_zero_below < bond_direction_quality_full_above"
            )
        for name in (
            "hydrogen_bond_donor_cone_half_width_degrees",
            "hydrogen_bond_acceptor_cone_half_width_degrees",
            "halogen_sigma_half_width_degrees",
            "halogen_acceptor_half_width_degrees",
        ):
            if not 0 < getattr(self, name) < 90:
                raise ValueError(f"{name} must be in (0,90)")


def _as_batch(coords: Tensor) -> tuple[Tensor, bool]:
    if coords.ndim == 2 and coords.shape[-1] == 3:
        return coords.unsqueeze(0), True
    if coords.ndim == 3 and coords.shape[-1] == 3:
        return coords, False
    raise ValueError("interaction coordinates must have shape [N,3] or [B,N,3]")


def _zero(coords: Tensor) -> Tensor:
    return coords.new_zeros(coords.shape[0])


def _quintic(value: Tensor) -> Tensor:
    value = value.clamp(0.0, 1.0)
    result = value.pow(3) * (10.0 - 15.0 * value + 6.0 * value.square())
    return result.clamp(0.0, 1.0)


def _decreasing_switch(
    distance: Tensor,
    switch_distance: float,
    cutoff: float,
) -> Tensor:
    scaled = (float(cutoff) - distance) / (float(cutoff) - float(switch_distance))
    return _quintic(scaled)


def _squared_decreasing_switch(
    distance_squared: Tensor,
    switch_distance: float,
    cutoff: float,
) -> Tensor:
    """C2 radial switch that is also smooth at exact coordinate overlap."""
    switch_squared = float(switch_distance) ** 2
    cutoff_squared = float(cutoff) ** 2
    scaled = (cutoff_squared - distance_squared) / (cutoff_squared - switch_squared)
    return _quintic(scaled)


def _cone_gate(
    cosine: Tensor,
    target_cosine: Tensor,
    half_width_degrees: float,
) -> Tensor:
    """C2 gate around an idealized missing-valence cone."""
    cosine = cosine.clamp(-1.0, 1.0)
    target_cosine = target_cosine.clamp(-1.0, 1.0)
    target_angle = torch.acos(target_cosine)
    half_width = math.radians(float(half_width_degrees))
    lower = torch.cos((target_angle + half_width).clamp(max=math.pi))
    upper = torch.cos((target_angle - half_width).clamp(min=0.0))
    epsilon = torch.finfo(cosine.dtype).eps
    rising = _quintic((cosine - lower) / (target_cosine - lower).clamp_min(epsilon))
    falling = _quintic((upper - cosine) / (upper - target_cosine).clamp_min(epsilon))
    return torch.where(cosine <= target_cosine, rising, falling)


def _ligand_outward_directions(
    coords: Tensor,
    neighbor_index: Tensor,
    epsilon: float,
    quality_zero_below: float,
    quality_full_above: float,
    bond_quality_zero_below: float,
    bond_quality_full_above: float,
) -> tuple[Tensor, Tensor, Tensor]:
    direction_sum = torch.zeros_like(coords)
    site_bond_quality = coords.new_ones(coords.shape[:2])
    if neighbor_index.numel():
        source, target = neighbor_index
        bond_direction = coords[:, source] - coords[:, target]
        bond_length = bond_direction.norm(
            dim=-1,
            keepdim=True,
        )
        bond_quality = _quintic(
            (
                (bond_length.squeeze(-1) - float(bond_quality_zero_below))
                / (float(bond_quality_full_above) - float(bond_quality_zero_below))
            )
        )
        bond_direction = (
            bond_direction / bond_length.clamp_min(float(epsilon)) * bond_quality.unsqueeze(-1)
        )
        direction_sum.index_add_(
            1,
            source,
            bond_direction,
        )
        site_bond_quality = site_bond_quality.scatter_reduce(
            1,
            source.view(1, -1).expand(coords.shape[0], -1),
            bond_quality,
            reduce="prod",
            include_self=True,
        )
    norm = direction_sum.norm(dim=-1)
    valid = norm > float(epsilon)
    quality = (
        _quintic(
            (
                (norm - float(quality_zero_below))
                / (float(quality_full_above) - float(quality_zero_below))
            )
        )
        * site_bond_quality
    )
    direction = direction_sum / norm.clamp_min(float(epsilon)).unsqueeze(-1)
    direction = torch.where(
        valid.unsqueeze(-1),
        direction,
        torch.zeros_like(direction),
    )
    return direction, valid, quality


def _site_saturated_energy(
    weights: Tensor,
    epsilon: float,
    delta: float = 1e-7,
) -> Tensor:
    """Soft-OR each row so duplicated receptor partners cannot pay linearly."""
    weights = weights.clamp(0.0, 1.0)
    log_unoccupied = torch.log1p(-(1.0 - float(delta)) * weights).sum(dim=2)
    occupancy = -torch.expm1(log_unoccupied)
    return -float(epsilon) * occupancy.sum(dim=1)


def _pair_vectors(left: Tensor, right: Tensor, epsilon: float) -> tuple[Tensor, Tensor]:
    vector = right.unsqueeze(1) - left.unsqueeze(2)
    distance = vector.norm(dim=-1)
    unit = vector / distance.clamp_min(float(epsilon)).unsqueeze(-1)
    unit = torch.where(
        (distance > float(epsilon)).unsqueeze(-1),
        unit,
        torch.zeros_like(unit),
    )
    return distance, unit


def _charge_group_coords(coords: Tensor, membership: Tensor) -> Tensor:
    if membership.ndim != 2:
        raise ValueError("charge-site membership must have shape [S,N]")
    if membership.shape[0] == 0:
        return coords[:, :0, :]
    if membership.shape[1] != coords.shape[1]:
        raise ValueError("charge-site membership atom count does not match coordinates")
    weights = membership.to(device=coords.device, dtype=coords.dtype)
    return torch.einsum("sn,bnd->bsd", weights, coords)


def _soft_or(values: Tensor, dim: int | tuple[int, ...], delta: float = 1e-7) -> Tensor:
    """Stable differentiable union for contact probabilities."""
    values = values.clamp(0.0, 1.0)
    return -torch.expm1(torch.log1p(-(1.0 - float(delta)) * values).sum(dim=dim))


def _scatter_soft_or(
    values: Tensor,
    group_index: Tensor,
    group_count: int,
    *,
    dim: int,
    delta: float = 1e-7,
) -> Tensor:
    """Stable differentiable soft union over indexed groups."""
    clamped = values.clamp(0.0, 1.0)
    log_unoccupied = torch.log1p(-(1.0 - float(delta)) * clamped)
    index_shape = [1] * values.ndim
    index_shape[dim] = group_index.numel()
    index = (
        group_index.to(device=values.device, dtype=torch.long)
        .reshape(index_shape)
        .expand_as(values)
    )
    output_shape = list(values.shape)
    output_shape[dim] = group_count
    grouped_log_unoccupied = values.new_zeros(output_shape).scatter_add(
        dim,
        index,
        log_unoccupied,
    )
    return -torch.expm1(grouped_log_unoccupied)


def _ring_geometry(
    coords: Tensor,
    membership: Tensor,
    triplet: Tensor,
    reference_area: Tensor,
    config: InteractionEnergyConfig,
) -> dict[str, Tensor]:
    """Return differentiable aromatic centroids, unoriented normals, and quality."""
    if membership.shape[0] == 0:
        empty_coord = coords[:, :0]
        return {
            "center": empty_coord,
            "normal": empty_coord,
            "quality": coords.new_empty((coords.shape[0], 0)),
            "area": coords.new_empty((coords.shape[0], 0)),
        }
    center = _charge_group_coords(coords, membership)
    ring_triplet = triplet.to(device=coords.device, dtype=torch.long)
    a = coords[:, ring_triplet[:, 0]]
    b = coords[:, ring_triplet[:, 1]]
    c = coords[:, ring_triplet[:, 2]]
    area_vector = torch.linalg.cross(b - a, c - a, dim=-1)
    area_squared = area_vector.square().sum(dim=-1)
    epsilon_squared = float(config.direction_epsilon) ** 2
    normal = area_vector / (area_squared + epsilon_squared).sqrt().unsqueeze(-1)
    reference = reference_area.to(
        device=coords.device,
        dtype=coords.dtype,
    ).view(1, -1)
    low_squared = (float(config.aromatic_area_quality_zero_fraction) * reference).square()
    high_squared = (float(config.aromatic_area_quality_full_fraction) * reference).square()
    quality = _quintic(
        (area_squared - low_squared)
        / (high_squared - low_squared).clamp_min(torch.finfo(coords.dtype).eps)
    )
    return {
        "center": center,
        "normal": normal,
        "quality": quality,
        "area": area_squared.sqrt(),
    }


def _aromatic_pair_components(
    ligand_ring: dict[str, Tensor],
    protein_ring: dict[str, Tensor],
    config: InteractionEnergyConfig,
) -> dict[str, Tensor]:
    displacement = protein_ring["center"].unsqueeze(1) - ligand_ring["center"].unsqueeze(2)
    distance_squared = displacement.square().sum(dim=-1)
    normal_cosine_squared = (
        (ligand_ring["normal"].unsqueeze(2) * protein_ring["normal"].unsqueeze(1))
        .sum(dim=-1)
        .square()
        .clamp(0.0, 1.0)
    )
    ligand_axial = (displacement * ligand_ring["normal"].unsqueeze(2)).sum(dim=-1).square()
    protein_axial = (displacement * protein_ring["normal"].unsqueeze(1)).sum(dim=-1).square()
    ligand_offset_squared = (distance_squared - ligand_axial).clamp_min(0.0)
    protein_offset_squared = (distance_squared - protein_axial).clamp_min(0.0)
    radial = _squared_decreasing_switch(
        distance_squared,
        config.pi_stacking_distance_on,
        config.pi_stacking_cutoff,
    )
    ligand_offset_gate = _squared_decreasing_switch(
        ligand_offset_squared,
        config.aromatic_offset_on,
        config.aromatic_offset_cutoff,
    )
    protein_offset_gate = _squared_decreasing_switch(
        protein_offset_squared,
        config.aromatic_offset_on,
        config.aromatic_offset_cutoff,
    )
    # For a T motif one centroid-to-centroid vector lies within one ring plane,
    # so either ring projection may carry the admissible offset.
    offset_gate = 1.0 - (1.0 - ligand_offset_gate) * (1.0 - protein_offset_gate)
    parallel_on = float(config.pi_stacking_parallel_cosine_squared_on)
    t_off = float(config.pi_stacking_t_shaped_cosine_squared_off)
    parallel_gate = _quintic((normal_cosine_squared - parallel_on) / (1.0 - parallel_on))
    t_gate = _quintic((t_off - normal_cosine_squared) / t_off)
    quality = ligand_ring["quality"].unsqueeze(2) * protein_ring["quality"].unsqueeze(1)
    weight = radial * offset_gate * (parallel_gate + t_gate) * quality
    return {
        "distance_squared": distance_squared,
        "distance": distance_squared.sqrt(),
        "normal_cosine_squared": normal_cosine_squared,
        "ligand_offset_squared": ligand_offset_squared,
        "protein_offset_squared": protein_offset_squared,
        "radial": radial,
        "offset_gate": offset_gate,
        "parallel_gate": parallel_gate,
        "t_gate": t_gate,
        "quality": quality,
        "weight": weight.clamp(0.0, 1.0),
    }


def _aggregate_ring_system_pairs(
    weights: Tensor,
    left_system: Tensor,
    right_system: Tensor,
) -> Tensor:
    """Soft-union ring-pair weights into fused aromatic-system pairs."""
    if weights.shape[1] == 0 or weights.shape[2] == 0:
        left_count = int(left_system.max().item()) + 1 if left_system.numel() else 0
        right_count = int(right_system.max().item()) + 1 if right_system.numel() else 0
        return weights.new_empty((weights.shape[0], left_count, right_count))
    left_count = int(left_system.max().item()) + 1
    right_count = int(right_system.max().item()) + 1
    pair_system = (
        left_system.to(device=weights.device, dtype=torch.long).unsqueeze(1) * right_count
        + right_system.to(device=weights.device, dtype=torch.long).unsqueeze(0)
    )
    grouped = _scatter_soft_or(
        weights.flatten(1),
        pair_system.flatten(),
        left_count * right_count,
        dim=1,
    )
    return grouped.reshape(weights.shape[0], left_count, right_count)


def _symmetric_saturated_energy(weights: Tensor, epsilon: float) -> Tensor:
    if weights.shape[1] == 0 or weights.shape[2] == 0:
        return weights.new_zeros(weights.shape[0])
    left = _soft_or(weights, dim=2).sum(dim=1)
    right = _soft_or(weights, dim=1).sum(dim=1)
    return -0.5 * float(epsilon) * (left + right)


def _cation_pi_components(
    ring: dict[str, Tensor],
    cation_coords: Tensor,
    config: InteractionEnergyConfig,
) -> dict[str, Tensor]:
    displacement = cation_coords.unsqueeze(1) - ring["center"].unsqueeze(2)
    distance_squared = displacement.square().sum(dim=-1)
    axial_squared = (displacement * ring["normal"].unsqueeze(2)).sum(dim=-1).square()
    offset_squared = (distance_squared - axial_squared).clamp_min(0.0)
    radial = _squared_decreasing_switch(
        distance_squared,
        config.cation_pi_distance_on,
        config.cation_pi_cutoff,
    )
    offset_gate = _squared_decreasing_switch(
        offset_squared,
        config.aromatic_offset_on,
        config.aromatic_offset_cutoff,
    )
    quality = ring["quality"].unsqueeze(2)
    return {
        "distance_squared": distance_squared,
        "distance": distance_squared.sqrt(),
        "offset_squared": offset_squared,
        "radial": radial,
        "offset_gate": offset_gate,
        "quality": quality.expand_as(distance_squared),
        "weight": (radial * offset_gate * quality).clamp(0.0, 1.0),
    }


def _aggregate_ring_to_system(weights: Tensor, ring_system: Tensor) -> Tensor:
    if weights.shape[1] == 0:
        system_count = int(ring_system.max().item()) + 1 if ring_system.numel() else 0
        return weights.new_empty((weights.shape[0], system_count, weights.shape[2]))
    system_count = int(ring_system.max().item()) + 1
    return _scatter_soft_or(
        weights,
        ring_system,
        system_count,
        dim=1,
    )


def _halogen_bond_components(
    coords: Tensor,
    protein_coords: Tensor,
    protein_direction: Tensor,
    protein_target_cosine: Tensor,
    protein_direction_quality: Tensor,
    topology,
    system: PhysicalSystem,
    config: InteractionEnergyConfig,
) -> dict[str, Tensor]:
    halogen_index = topology.ligand_halogen_donor_index
    parent_index = topology.ligand_halogen_parent_index
    acceptor_index = topology.protein_halogen_acceptor_index
    halogen = coords[:, halogen_index]
    parent = coords[:, parent_index]
    acceptor = protein_coords[:, acceptor_index]
    distance, halogen_to_acceptor = _pair_vectors(
        halogen,
        acceptor,
        config.direction_epsilon,
    )
    donor_axis = halogen - parent
    donor_bond_length = donor_axis.norm(dim=-1)
    donor_valid = donor_bond_length > float(config.direction_epsilon)
    donor_direction = donor_axis / donor_bond_length.clamp_min(
        float(config.direction_epsilon)
    ).unsqueeze(-1)
    donor_direction = torch.where(
        donor_valid.unsqueeze(-1),
        donor_direction,
        torch.zeros_like(donor_direction),
    )
    donor_bond_quality = _quintic(
        (donor_bond_length - float(config.bond_direction_quality_zero_below))
        / (
            float(config.bond_direction_quality_full_above)
            - float(config.bond_direction_quality_zero_below)
        )
    )
    sigma_cosine = (donor_direction.unsqueeze(2) * halogen_to_acceptor).sum(dim=-1)
    sigma_gate = _cone_gate(
        sigma_cosine,
        sigma_cosine.new_tensor(1.0),
        config.halogen_sigma_half_width_degrees,
    )
    acceptor_cosine = (
        protein_direction[:, acceptor_index].unsqueeze(1) * -halogen_to_acceptor
    ).sum(dim=-1)
    acceptor_target = protein_target_cosine[acceptor_index].view(1, 1, -1)
    acceptor_gate = _cone_gate(
        acceptor_cosine,
        acceptor_target,
        config.halogen_acceptor_half_width_degrees,
    )
    ligand_z = system.topology.atomic_numbers[halogen_index].to(
        device=coords.device,
    )
    protein_z = system.protein_atomic_numbers[acceptor_index].to(
        device=coords.device,
    )
    ligand_radius = torch.where(
        ligand_z == 17,
        coords.new_tensor(config.halogen_vdw_radius_chlorine),
        torch.where(
            ligand_z == 35,
            coords.new_tensor(config.halogen_vdw_radius_bromine),
            coords.new_tensor(config.halogen_vdw_radius_iodine),
        ),
    )
    protein_radius = torch.where(
        protein_z == 7,
        coords.new_tensor(config.halogen_vdw_radius_nitrogen),
        torch.where(
            protein_z == 8,
            coords.new_tensor(config.halogen_vdw_radius_oxygen),
            coords.new_tensor(config.halogen_vdw_radius_sulfur),
        ),
    )
    radius_sum = ligand_radius.view(1, -1, 1) + protein_radius.view(1, 1, -1)
    normalized_distance = distance / radius_sum
    radial = _decreasing_switch(
        normalized_distance,
        config.halogen_contact_scale,
        config.halogen_cutoff_scale,
    )
    acceptor_quality = protein_direction_quality[:, acceptor_index].unsqueeze(1)
    weight = (
        radial
        * sigma_gate
        * acceptor_gate
        * donor_bond_quality.unsqueeze(2)
        * acceptor_quality
        * donor_valid.unsqueeze(2)
    )
    return {
        "distance": distance,
        "normalized_distance": normalized_distance,
        "radius_sum": radius_sum.expand_as(distance),
        "sigma_cosine": sigma_cosine,
        "sigma_gate": sigma_gate,
        "acceptor_cosine": acceptor_cosine,
        "acceptor_target_cosine": acceptor_target.expand_as(acceptor_cosine),
        "acceptor_gate": acceptor_gate,
        "donor_bond_quality": donor_bond_quality.unsqueeze(2).expand_as(distance),
        "acceptor_axis_quality": acceptor_quality.expand_as(distance),
        "weight": weight.clamp(0.0, 1.0),
        "halogen_index": halogen_index,
        "parent_index": parent_index,
        "acceptor_index": acceptor_index,
    }


def _metal_components(
    coords: Tensor,
    topology,
    system: PhysicalSystem,
    config: InteractionEnergyConfig,
) -> dict[str, Tensor]:
    """Evaluate profile-dispatched metal terms with site-specific pair masks.

    Generic topology is canonical.  The legacy Zn tensors remain readable so
    archived fixtures and serialized diagnostics do not change semantics.
    """
    generic_metal_coords = getattr(topology, "metal_coords", coords.new_empty((0, 3)))
    if generic_metal_coords.shape[0]:
        metal_coords = generic_metal_coords
        donor_index = topology.ligand_metal_donor_index
        donor_element = topology.ligand_metal_donor_element
        vacant_direction = topology.metal_vacant_direction
        fixed_coordination = topology.metal_fixed_coordination
        target_coordination = topology.metal_target_coordination
        ligand_r0 = topology.metal_ligand_r0
        donor_allowed = topology.metal_ligand_donor_allowed
        attraction_enabled = topology.metal_attraction_enabled
        metal_atomic_number = topology.metal_atomic_number
    else:
        metal_coords = topology.zinc_coords
        donor_index = topology.ligand_zinc_donor_index
        donor_element = topology.ligand_zinc_donor_element
        vacant_direction = topology.zinc_vacant_direction
        metal_count = int(metal_coords.shape[0])
        fixed_coordination = torch.full(
            (metal_count,),
            3,
            dtype=torch.long,
            device=metal_coords.device,
        )
        target_coordination = torch.full(
            (metal_count,),
            4,
            dtype=torch.long,
            device=metal_coords.device,
        )
        legacy_zinc_r0 = load_interaction_v1()["metal_coordination_profiles"]["ZN"][
            "r0_angstrom"
        ]
        ligand_r0 = metal_coords.new_tensor(
            [
                float(legacy_zinc_r0["N"]),
                float(legacy_zinc_r0["O"]),
                float(legacy_zinc_r0["S"]),
            ]
        ).view(1, 3).expand(metal_count, -1)
        donor_allowed = torch.ones(
            (metal_count, 3),
            dtype=torch.bool,
            device=metal_coords.device,
        )
        attraction_enabled = torch.ones(
            metal_count,
            dtype=torch.bool,
            device=metal_coords.device,
        )
        metal_atomic_number = torch.full(
            (metal_count,),
            30,
            dtype=torch.long,
            device=metal_coords.device,
        )

    metal = metal_coords.unsqueeze(0).expand(coords.shape[0], -1, -1)
    donor = coords[:, donor_index]
    distance, metal_to_donor = _pair_vectors(
        metal,
        donor,
        config.direction_epsilon,
    )
    alignment_cosine = (
        (metal_to_donor * vacant_direction.unsqueeze(0).unsqueeze(2))
        .sum(dim=-1)
        .clamp(-1.0, 1.0)
    )
    alignment = torch.exp(-(1.0 - alignment_cosine) / float(config.metal_alignment_tau))
    donor_column = torch.where(
        donor_element == 7,
        torch.zeros_like(donor_element),
        torch.where(
            donor_element == 8,
            torch.ones_like(donor_element),
            torch.full_like(donor_element, 2),
        ),
    )
    r0_by_pair = ligand_r0[:, donor_column]
    eligible = donor_allowed[:, donor_column] & attraction_enabled.unsqueeze(1)
    pair_mask = eligible.unsqueeze(0).to(dtype=coords.dtype)
    r0 = r0_by_pair.unsqueeze(0)
    delta = distance - r0
    pair_switch = _decreasing_switch(
        distance,
        config.metal_pair_switch_distance,
        config.metal_pair_cutoff,
    )
    exponential = torch.exp((-float(config.metal_morse_a) * delta).clamp(max=40.0))
    radial_repulsion = (
        pair_mask * pair_switch * float(config.metal_pair_depth) * exponential.square()
    )
    directional_attraction = (
        pair_mask
        * pair_switch
        * (-2.0 * float(config.metal_pair_depth))
        * alignment
        * exponential
    )
    pair_energy = radial_repulsion + directional_attraction
    occupancy = (
        pair_mask
        *
        _decreasing_switch(
            distance,
            config.metal_cn_switch_distance,
            config.metal_cn_cutoff,
        )
        * torch.exp(-0.5 * ((distance - r0) / float(config.metal_occupancy_sigma)).square())
        * alignment
    )
    coordination_number = fixed_coordination.to(dtype=coords.dtype).unsqueeze(0) + occupancy.sum(
        dim=2
    )
    overcoordination = (
        float(config.metal_overcoordination_k)
        * torch.relu(
            coordination_number
            - target_coordination.to(dtype=coords.dtype).unsqueeze(0)
        ).square()
        * attraction_enabled.to(dtype=coords.dtype).unsqueeze(0)
    )
    occupancy_sum = occupancy.sum(dim=2)
    slot = (
        0.5 * float(config.metal_slot_k) * (occupancy_sum.square() - occupancy.square().sum(dim=2))
    )

    site_eligible_atom = torch.zeros(
        (metal_coords.shape[0], system.topology.num_atoms),
        dtype=torch.bool,
        device=coords.device,
    )
    if donor_index.numel():
        site_eligible_atom[:, donor_index] = eligible
    non_donor_index = torch.arange(
        system.topology.num_atoms,
        dtype=torch.long,
        device=coords.device,
    )
    non_donor = coords
    non_donor_distance, _ = _pair_vectors(
        metal,
        non_donor,
        config.direction_epsilon,
    )
    rho = (non_donor_distance.square() + float(config.metal_non_donor_softcore) ** 2).sqrt()
    ratio = (float(config.metal_non_donor_radius) / rho).pow(12)
    bounded = ratio / (1.0 + ratio)
    non_donor_switch = _decreasing_switch(
        non_donor_distance,
        config.metal_non_donor_switch_distance,
        config.metal_non_donor_cutoff,
    )
    non_donor_repulsion = (
        float(config.metal_non_donor_repulsion)
        * bounded
        * non_donor_switch
        * (~site_eligible_atom).to(dtype=coords.dtype).unsqueeze(0)
    )
    return {
        "distance": distance,
        "alignment_cosine": alignment_cosine,
        "alignment": alignment,
        "r0": r0.expand_as(distance),
        "eligible": eligible,
        "attraction_enabled": attraction_enabled,
        "metal_atomic_number": metal_atomic_number,
        "pair_switch": pair_switch,
        "radial_repulsion": radial_repulsion,
        "directional_attraction": directional_attraction,
        "pair_energy": pair_energy,
        "occupancy": occupancy,
        "coordination_number": coordination_number,
        "overcoordination_energy": overcoordination,
        "slot_energy": slot,
        "non_donor_index": non_donor_index,
        "non_donor_distance": non_donor_distance,
        "non_donor_repulsion": non_donor_repulsion,
        "donor_index": donor_index,
    }


def _zinc_components(
    coords: Tensor,
    topology,
    system: PhysicalSystem,
    config: InteractionEnergyConfig,
) -> dict[str, Tensor]:
    """Backward-compatible private alias for the generic metal evaluator."""
    return _metal_components(coords, topology, system, config)


def _screened_formal_charge_components(
    ligand_coords: Tensor,
    protein_coords: Tensor,
    ligand_charge: Tensor,
    protein_charge: Tensor,
    config: InteractionEnergyConfig,
) -> dict[str, Tensor]:
    displacement = protein_coords.unsqueeze(1) - ligand_coords.unsqueeze(2)
    distance_squared = displacement.square().sum(dim=-1)
    distance = distance_squared.sqrt()
    softcore_distance = (distance_squared + float(config.formal_charge_softcore) ** 2).sqrt()
    screening = torch.exp(-float(config.formal_charge_screening_kappa) * softcore_distance)
    switch = _squared_decreasing_switch(
        distance_squared,
        config.formal_charge_switch_distance,
        config.formal_charge_cutoff,
    )
    charge_product = ligand_charge.to(
        device=ligand_coords.device,
        dtype=ligand_coords.dtype,
    ).view(1, -1, 1) * protein_charge.to(
        device=protein_coords.device,
        dtype=protein_coords.dtype,
    ).view(1, 1, -1)
    prefactor = float(config.formal_charge_coulomb_constant) / float(
        config.formal_charge_relative_dielectric
    )
    pair_energy = prefactor * charge_product * screening * switch / softcore_distance
    return {
        "distance_squared": distance_squared,
        "distance": distance,
        "softcore_distance": softcore_distance,
        "screening": screening,
        "switch": switch,
        "charge_product": charge_product.expand_as(distance),
        "pair_energy": pair_energy,
    }


def _hydrogen_bond_components(
    donor_coords: Tensor,
    acceptor_coords: Tensor,
    donor_direction: Tensor,
    acceptor_direction: Tensor,
    donor_target_cosine: Tensor,
    acceptor_target_cosine: Tensor,
    donor_axis_quality: Tensor,
    acceptor_axis_quality: Tensor,
    donor_valid: Tensor,
    acceptor_valid: Tensor,
    config: InteractionEnergyConfig,
) -> dict[str, Tensor]:
    distance, donor_to_acceptor = _pair_vectors(
        donor_coords,
        acceptor_coords,
        config.direction_epsilon,
    )
    radial = torch.exp(
        -0.5
        * (
            (distance - float(config.hydrogen_bond_optimal_distance))
            / float(config.hydrogen_bond_distance_sigma)
        ).square()
    )
    radial = radial * _decreasing_switch(
        distance,
        config.hydrogen_bond_switch_distance,
        config.hydrogen_bond_cutoff,
    )
    donor_cosine = (donor_direction.unsqueeze(2) * donor_to_acceptor).sum(dim=-1)
    acceptor_cosine = (acceptor_direction.unsqueeze(1) * -donor_to_acceptor).sum(dim=-1)
    donor_target = donor_target_cosine.view(1, -1, 1)
    acceptor_target = acceptor_target_cosine.view(1, 1, -1)
    donor_gate = _cone_gate(
        donor_cosine,
        donor_target,
        config.hydrogen_bond_donor_cone_half_width_degrees,
    )
    acceptor_gate = _cone_gate(
        acceptor_cosine,
        acceptor_target,
        config.hydrogen_bond_acceptor_cone_half_width_degrees,
    )
    donor_quality = donor_axis_quality.unsqueeze(2)
    acceptor_quality = acceptor_axis_quality.unsqueeze(1)
    geometry = radial * donor_gate * acceptor_gate * donor_quality * acceptor_quality
    valid = donor_valid.unsqueeze(2) & acceptor_valid.unsqueeze(1)
    weight = torch.where(valid, geometry, torch.zeros_like(geometry))
    return {
        "distance": distance,
        "radial": radial,
        "donor_cosine": donor_cosine,
        "donor_target_cosine": donor_target.expand_as(donor_cosine),
        "donor_gate": donor_gate,
        "donor_axis_quality": donor_quality.expand_as(donor_cosine),
        "acceptor_cosine": acceptor_cosine,
        "acceptor_target_cosine": acceptor_target.expand_as(acceptor_cosine),
        "acceptor_gate": acceptor_gate,
        "acceptor_axis_quality": acceptor_quality.expand_as(acceptor_cosine),
        "weight": weight,
    }


def _interaction_weights(
    coords: Tensor,
    system: PhysicalSystem,
    config: InteractionEnergyConfig,
    *,
    include_diagnostics: bool = True,
) -> dict[str, dict[str, Tensor]]:
    topology = system.interaction_topology
    if topology is None:
        raise ValueError("active interaction terms require system interaction typing")
    requested = set(ACTIVE_INTERACTION_TERMS) if include_diagnostics else set(config.active_terms)
    need_hydrophobic = "hydrophobic" in requested
    need_hydrogen_bond = "hydrogen_bond" in requested or include_diagnostics
    need_formal_charge = "screened_formal_charge" in requested
    need_pi_stacking = "pi_stacking" in requested
    need_cation_pi = "cation_pi" in requested
    need_halogen_bond = "halogen_bond" in requested
    need_metal_coordination = "metal_coordination" in requested or include_diagnostics
    need_charge_coordinates = need_formal_charge or need_cation_pi
    need_aromatic_geometry = need_pi_stacking or need_cation_pi
    weights: dict[str, dict[str, Tensor]] = {}
    batch_size = coords.shape[0]
    protein_coords = system.protein_coords.unsqueeze(0).expand(
        batch_size,
        -1,
        -1,
    )
    need_direction_geometry = need_hydrogen_bond or need_halogen_bond or include_diagnostics
    if need_direction_geometry:
        (
            ligand_direction,
            dynamic_ligand_direction_valid,
            ligand_direction_quality,
        ) = _ligand_outward_directions(
            coords,
            topology.ligand_neighbor_index,
            config.direction_epsilon,
            config.direction_quality_zero_below,
            config.direction_quality_full_above,
            config.bond_direction_quality_zero_below,
            config.bond_direction_quality_full_above,
        )
        ligand_direction_valid = (
            dynamic_ligand_direction_valid & topology.ligand_direction_geometry_valid.unsqueeze(0)
        )
        protein_direction = topology.protein_outward_direction.unsqueeze(0).expand(
            batch_size,
            -1,
            -1,
        )
        protein_direction_valid = topology.protein_direction_valid.unsqueeze(0).expand(
            batch_size,
            -1,
        )
        protein_direction_quality = topology.protein_direction_quality.unsqueeze(0).expand(
            batch_size,
            -1,
        )
        ligand_target_cosine = topology.ligand_direction_target_cosine
        protein_target_cosine = topology.protein_direction_target_cosine

    if need_charge_coordinates:
        ligand_charge_coords = _charge_group_coords(
            coords,
            topology.ligand_charge_site_membership,
        )
        protein_charge_coords = _charge_group_coords(
            protein_coords,
            topology.protein_charge_site_membership,
        )
    if need_formal_charge:
        weights["screened_formal_charge"] = {
            **_screened_formal_charge_components(
                ligand_charge_coords,
                protein_charge_coords,
                topology.ligand_charge_site_charge,
                topology.protein_charge_site_charge,
                config,
            ),
            "ligand_charge": topology.ligand_charge_site_charge,
            "protein_charge": topology.protein_charge_site_charge,
        }

    if need_aromatic_geometry:
        ligand_ring = _ring_geometry(
            coords,
            topology.ligand_aromatic_ring_membership,
            topology.ligand_aromatic_ring_triplet,
            topology.ligand_aromatic_ring_reference_area,
            config,
        )
        protein_ring = _ring_geometry(
            protein_coords,
            topology.protein_aromatic_ring_membership,
            topology.protein_aromatic_ring_triplet,
            topology.protein_aromatic_ring_reference_area,
            config,
        )
    if need_pi_stacking:
        pi_stacking = _aromatic_pair_components(
            ligand_ring,
            protein_ring,
            config,
        )
        weights["pi_stacking"] = {
            **pi_stacking,
            "system_weight": _aggregate_ring_system_pairs(
                pi_stacking["weight"],
                topology.ligand_aromatic_ring_system,
                topology.protein_aromatic_ring_system,
            ),
        }

    if need_cation_pi:
        ligand_cation_ring_mask = topology.ligand_aromatic_ring_is_cation_pi_acceptor
        protein_cation_ring_mask = topology.protein_aromatic_ring_is_cation_pi_acceptor

        def slice_ring(data: dict[str, Tensor], mask: Tensor) -> dict[str, Tensor]:
            return {name: value[:, mask] for name, value in data.items()}

        ligand_cation_ring = slice_ring(ligand_ring, ligand_cation_ring_mask)
        protein_cation_ring = slice_ring(protein_ring, protein_cation_ring_mask)
        ligand_ring_system = topology.ligand_aromatic_ring_system[ligand_cation_ring_mask]
        protein_ring_system = topology.protein_aromatic_ring_system[protein_cation_ring_mask]
        if ligand_ring_system.numel():
            ligand_ring_system = torch.unique(
                ligand_ring_system,
                sorted=True,
                return_inverse=True,
            )[1]
        if protein_ring_system.numel():
            protein_ring_system = torch.unique(
                protein_ring_system,
                sorted=True,
                return_inverse=True,
            )[1]
        ligand_positive = torch.isclose(
            topology.ligand_charge_site_charge,
            topology.ligand_charge_site_charge.new_tensor(1.0),
            rtol=0.0,
            atol=1e-6,
        )
        protein_positive = torch.isclose(
            topology.protein_charge_site_charge,
            topology.protein_charge_site_charge.new_tensor(1.0),
            rtol=0.0,
            atol=1e-6,
        )
        ligand_ring_to_protein_cation = _cation_pi_components(
            ligand_cation_ring,
            protein_charge_coords[:, protein_positive],
            config,
        )
        protein_ring_to_ligand_cation = _cation_pi_components(
            protein_cation_ring,
            ligand_charge_coords[:, ligand_positive],
            config,
        )
        ligand_ring_to_protein_cation["system_weight"] = _aggregate_ring_to_system(
            ligand_ring_to_protein_cation["weight"],
            ligand_ring_system,
        )
        protein_ring_to_ligand_cation["system_weight"] = _aggregate_ring_to_system(
            protein_ring_to_ligand_cation["weight"],
            protein_ring_system,
        )
        ligand_ring_to_protein_cation["ring_mask"] = ligand_cation_ring_mask
        protein_ring_to_ligand_cation["ring_mask"] = protein_cation_ring_mask
        ligand_ring_to_protein_cation["cation_mask"] = protein_positive
        protein_ring_to_ligand_cation["cation_mask"] = ligand_positive
        weights["ligand_ring_to_protein_cation"] = ligand_ring_to_protein_cation
        weights["protein_ring_to_ligand_cation"] = protein_ring_to_ligand_cation

    if need_halogen_bond:
        weights["halogen_bond"] = _halogen_bond_components(
            coords,
            protein_coords,
            protein_direction,
            protein_target_cosine,
            protein_direction_quality,
            topology,
            system,
            config,
        )
    if need_metal_coordination:
        weights["metal_coordination"] = _metal_components(
            coords,
            topology,
            system,
            config,
        )

    if need_hydrophobic:
        ligand_hydrophobe = topology.ligand_is_hydrophobe.nonzero(as_tuple=False).flatten()
        protein_hydrophobe = topology.protein_is_hydrophobe.nonzero(as_tuple=False).flatten()
        hydrophobic_distance, _ = _pair_vectors(
            coords[:, ligand_hydrophobe],
            protein_coords[:, protein_hydrophobe],
            config.direction_epsilon,
        )
        hydrophobic_weight = _decreasing_switch(
            hydrophobic_distance,
            config.hydrophobic_contact_on,
            config.hydrophobic_cutoff,
        )
        if (
            include_diagnostics
            and weights["pi_stacking"]["weight"].shape[1] > 0
            and weights["pi_stacking"]["weight"].shape[2] > 0
            and ligand_hydrophobe.numel()
            and protein_hydrophobe.numel()
        ):
            ring_log_unoccupied = torch.log1p(
                -(1.0 - 1e-7) * weights["pi_stacking"]["weight"].clamp(0.0, 1.0)
            )
            ligand_ring_member = (
                topology.ligand_aromatic_ring_membership[:, ligand_hydrophobe] > 0
            ).to(dtype=coords.dtype)
            protein_ring_member = (
                topology.protein_aromatic_ring_membership[:, protein_hydrophobe] > 0
            ).to(dtype=coords.dtype)
            atom_pair_log_unoccupied = torch.einsum(
                "brs,rl,sp->blp",
                ring_log_unoccupied,
                ligand_ring_member,
                protein_ring_member,
            )
            hydrophobic_pi_coverage = -torch.expm1(atom_pair_log_unoccupied)
        else:
            hydrophobic_pi_coverage = torch.zeros_like(hydrophobic_weight)
        weights["hydrophobic"] = {
            "distance": hydrophobic_distance,
            "raw_weight": hydrophobic_weight,
            "pi_coverage": hydrophobic_pi_coverage,
            "weight": hydrophobic_weight,
            "ligand_index": ligand_hydrophobe,
            "protein_index": protein_hydrophobe,
        }

    if need_hydrogen_bond:
        ligand_donor = topology.ligand_is_donor.nonzero(as_tuple=False).flatten()
        ligand_acceptor = topology.ligand_is_acceptor.nonzero(as_tuple=False).flatten()
        protein_donor = topology.protein_is_donor.nonzero(as_tuple=False).flatten()
        protein_acceptor = topology.protein_is_acceptor.nonzero(as_tuple=False).flatten()
        ligand_donor_to_protein_acceptor = _hydrogen_bond_components(
            coords[:, ligand_donor],
            protein_coords[:, protein_acceptor],
            ligand_direction[:, ligand_donor],
            protein_direction[:, protein_acceptor],
            ligand_target_cosine[ligand_donor],
            protein_target_cosine[protein_acceptor],
            ligand_direction_quality[:, ligand_donor],
            protein_direction_quality[:, protein_acceptor],
            ligand_direction_valid[:, ligand_donor],
            protein_direction_valid[:, protein_acceptor],
            config,
        )
        protein_donor_to_ligand_acceptor = _hydrogen_bond_components(
            protein_coords[:, protein_donor],
            coords[:, ligand_acceptor],
            protein_direction[:, protein_donor],
            ligand_direction[:, ligand_acceptor],
            protein_target_cosine[protein_donor],
            ligand_target_cosine[ligand_acceptor],
            protein_direction_quality[:, protein_donor],
            ligand_direction_quality[:, ligand_acceptor],
            protein_direction_valid[:, protein_donor],
            ligand_direction_valid[:, ligand_acceptor],
            config,
        )
        weights["ligand_donor_to_protein_acceptor"] = {
            **ligand_donor_to_protein_acceptor,
            "donor_index": ligand_donor,
            "acceptor_index": protein_acceptor,
        }
        weights["protein_donor_to_ligand_acceptor"] = {
            **protein_donor_to_ligand_acceptor,
            "donor_index": protein_donor,
            "acceptor_index": ligand_acceptor,
        }

    if include_diagnostics:
        ligand_polar_mask = topology.ligand_is_donor | topology.ligand_is_acceptor
        ligand_polar_index = ligand_polar_mask.nonzero(as_tuple=False).flatten()
        polar_distance, _ = _pair_vectors(
            coords[:, ligand_polar_index],
            protein_coords,
            config.direction_epsilon,
        )
        polar_burial_pair = _decreasing_switch(
            polar_distance,
            config.polar_proxy_burial_on,
            config.polar_proxy_burial_cutoff,
        )
        polar_burial = _soft_or(polar_burial_pair, dim=2)

        def atom_site_values(indices: Tensor, values: Tensor) -> Tensor:
            base = coords.new_zeros((coords.shape[0], system.topology.num_atoms))
            if not indices.numel():
                return base
            return base.index_copy(1, indices, values.clamp(0.0, 1.0))

        donor_satisfaction = atom_site_values(
            ligand_donor,
            _soft_or(
                weights["ligand_donor_to_protein_acceptor"]["weight"],
                dim=2,
            ),
        )
        acceptor_satisfaction = atom_site_values(
            ligand_acceptor,
            _soft_or(
                weights["protein_donor_to_ligand_acceptor"]["weight"],
                dim=1,
            ),
        )
        metal_donor_index = (
            topology.ligand_metal_donor_index
            if getattr(topology, "metal_coords", coords.new_empty((0, 3))).shape[0]
            else topology.ligand_zinc_donor_index
        )
        metal_satisfaction = atom_site_values(
            metal_donor_index,
            _soft_or(weights["metal_coordination"]["occupancy"], dim=1),
        )
        ligand_satisfaction = 1.0 - (
            (1.0 - donor_satisfaction)
            * (1.0 - acceptor_satisfaction)
            * (1.0 - metal_satisfaction)
        )
        polar_satisfaction = ligand_satisfaction[:, ligand_polar_index]
        weights["polar_unsatisfied_proxy"] = {
            "ligand_index": ligand_polar_index,
            "distance": polar_distance,
            "burial_pair": polar_burial_pair,
            "burial": polar_burial,
            "satisfaction": polar_satisfaction,
            "value": polar_burial * (1.0 - polar_satisfaction),
        }
    return weights


def interaction_energy(
    coords: Tensor,
    system: PhysicalSystem,
    config: InteractionEnergyConfig = InteractionEnergyConfig(),
) -> dict[str, Tensor]:
    """Return active, per-pose interaction components."""
    batched, squeeze = _as_batch(coords)
    if batched.shape[-2] != system.topology.num_atoms:
        raise ValueError("interaction coordinate atom count does not match topology")
    topology = system.interaction_topology
    generic_metal_count = (
        int(topology.metal_coords.shape[0])
        if topology is not None and hasattr(topology, "metal_coords")
        else 0
    )
    legacy_zinc_count = (
        int(topology.zinc_coords.shape[0])
        if topology is not None
        else 0
    )
    if (
        topology is not None
        and (generic_metal_count or legacy_zinc_count)
        and "metal_coordination" not in config.active_terms
    ):
        site_labels = (
            list(topology.metal_site_labels)
            if generic_metal_count
            else list(topology.zinc_site_labels)
        )
        raise UnsupportedPhysicalChemistryError(
            "required_interaction_term_inactive",
            "an admitted active-shell metal site requires explicit "
            "metal_coordination activation; it cannot be silently omitted",
            details={
                "required_term": "metal_coordination",
                "metal_sites": site_labels,
                "active_terms": list(config.active_terms),
            },
        )
    if not config.active_terms:
        total = _zero(batched)
        return {"total": total.squeeze(0) if squeeze else total}
    weights = _interaction_weights(
        batched,
        system,
        config,
        include_diagnostics=False,
    )
    components: dict[str, Tensor] = {}
    if "hydrophobic" in config.active_terms:
        components["interaction_hydrophobic"] = _site_saturated_energy(
            weights["hydrophobic"]["weight"],
            config.hydrophobic_epsilon,
        )
    if "hydrogen_bond" in config.active_terms:
        components["interaction_hydrogen_bond"] = _site_saturated_energy(
            weights["ligand_donor_to_protein_acceptor"]["weight"],
            config.hydrogen_bond_epsilon,
        ) + _site_saturated_energy(
            weights["protein_donor_to_ligand_acceptor"]["weight"],
            config.hydrogen_bond_epsilon,
        )
    if "screened_formal_charge" in config.active_terms:
        components["interaction_screened_formal_charge"] = weights["screened_formal_charge"][
            "pair_energy"
        ].sum(dim=(1, 2))
    if "pi_stacking" in config.active_terms:
        components["interaction_pi_stacking"] = _symmetric_saturated_energy(
            weights["pi_stacking"]["system_weight"],
            config.pi_stacking_epsilon,
        )
    if "cation_pi" in config.active_terms:
        components["interaction_cation_pi"] = _symmetric_saturated_energy(
            weights["ligand_ring_to_protein_cation"]["system_weight"],
            config.cation_pi_epsilon,
        ) + _symmetric_saturated_energy(
            weights["protein_ring_to_ligand_cation"]["system_weight"],
            config.cation_pi_epsilon,
        )
    if "halogen_bond" in config.active_terms:
        components["interaction_halogen_bond"] = _site_saturated_energy(
            weights["halogen_bond"]["weight"],
            config.halogen_bond_epsilon,
        )
    if "metal_coordination" in config.active_terms:
        metal = weights["metal_coordination"]
        components["interaction_metal_coordination"] = (
            metal["pair_energy"].sum(dim=(1, 2))
            + metal["overcoordination_energy"].sum(dim=1)
            + metal["slot_energy"].sum(dim=1)
            + metal["non_donor_repulsion"].sum(dim=(1, 2))
        )
    components["total"] = sum(components.values(), start=_zero(batched))
    if squeeze:
        return {name: value.squeeze(0) for name, value in components.items()}
    return components


def interaction_contact_stats(
    coords: Tensor,
    system: PhysicalSystem,
    config: InteractionEnergyConfig = InteractionEnergyConfig(),
) -> dict[str, object]:
    """Return typing and continuous contact-weight diagnostics for one pose."""
    batched, squeeze = _as_batch(coords)
    if not squeeze:
        raise ValueError("interaction contact stats require one pose [N,3]")
    weights = _interaction_weights(batched, system, config)

    def summarize(data: dict[str, Tensor]) -> dict[str, float | int]:
        value = data["weight"][0]
        return {
            "candidate_pairs": int(value.numel()),
            "nonzero_pairs": int((value > 0).sum().item()),
            "weight_sum": float(value.sum().detach().cpu()),
            "maximum_weight": (float(value.max().detach().cpu()) if value.numel() else 0.0),
        }

    topology = system.interaction_topology
    if topology is None:
        raise AssertionError("interaction weights require an interaction topology")
    has_generic_metals = bool(
        hasattr(topology, "metal_coords") and topology.metal_coords.shape[0]
    )
    metal_site_labels = (
        topology.metal_site_labels if has_generic_metals else topology.zinc_site_labels
    )
    metal_profile_labels = (
        topology.metal_profile_labels
        if has_generic_metals
        else tuple("ZN:v0" for _ in topology.zinc_site_labels)
    )
    ligand_metal_donor_element = (
        topology.ligand_metal_donor_element
        if has_generic_metals
        else topology.ligand_zinc_donor_element
    )
    ligand_metal_donor_exclusion_labels = (
        topology.ligand_metal_donor_exclusion_labels
        if has_generic_metals
        else topology.ligand_zinc_donor_exclusion_labels
    )
    metal_typing_exclusion_labels = (
        topology.metal_typing_exclusion_labels
        if has_generic_metals
        else topology.zinc_typing_exclusion_labels
    )

    def formal_charge_site_inventory(
        membership: Tensor,
        charge: Tensor,
        site_labels: tuple[str, ...],
        atom_labels: tuple[str, ...],
    ) -> list[dict[str, object]]:
        membership_cpu = membership.detach().cpu()
        charge_cpu = charge.detach().cpu()
        rows: list[dict[str, object]] = []
        for site_index, site_label in enumerate(site_labels):
            member_indices = (
                (membership_cpu[site_index] > 0).nonzero(as_tuple=False).flatten().tolist()
            )
            rows.append(
                {
                    "index": site_index,
                    "label": site_label,
                    "charge_e": float(charge_cpu[site_index]),
                    "members": [
                        {
                            "atom_index": atom_index,
                            "atom_label": atom_labels[atom_index],
                            "weight": float(membership_cpu[site_index, atom_index]),
                        }
                        for atom_index in member_indices
                    ],
                }
            )
        return rows

    def top_hydrophobic_pairs(
        data: dict[str, Tensor],
        limit: int = 8,
    ) -> list[dict[str, object]]:
        weight = data["weight"][0]
        if not weight.numel():
            return []
        nonzero = (weight > 0).flatten().nonzero(as_tuple=False).flatten()
        if not nonzero.numel():
            return []
        selected = nonzero[torch.argsort(weight.flatten()[nonzero], descending=True)[:limit]]
        rows: list[dict[str, object]] = []
        for flat in selected.tolist():
            ligand_slot = flat // weight.shape[1]
            protein_slot = flat % weight.shape[1]
            ligand_index = int(data["ligand_index"][ligand_slot])
            protein_index = int(data["protein_index"][protein_slot])
            rows.append(
                {
                    "ligand": {
                        "index": ligand_index,
                        "label": topology.ligand_atom_labels[ligand_index],
                    },
                    "protein": {
                        "index": protein_index,
                        "label": topology.protein_atom_labels[protein_index],
                    },
                    "distance_angstrom": float(
                        data["distance"][0, ligand_slot, protein_slot].detach().cpu()
                    ),
                    "weight": float(weight[ligand_slot, protein_slot].detach().cpu()),
                }
            )
        return rows

    def top_hydrogen_bond_pairs(
        data: dict[str, Tensor],
        *,
        donor_side: str,
        ranking_field: str = "weight",
        limit: int = 8,
    ) -> list[dict[str, object]]:
        weight = data["weight"][0]
        ranking = data[ranking_field][0]
        if not ranking.numel():
            return []
        nonzero = (ranking > 0).flatten().nonzero(as_tuple=False).flatten()
        if not nonzero.numel():
            return []
        selected = nonzero[
            torch.argsort(
                ranking.flatten()[nonzero],
                descending=True,
            )[:limit]
        ]
        rows: list[dict[str, object]] = []
        for flat in selected.tolist():
            donor_slot = flat // weight.shape[1]
            acceptor_slot = flat % weight.shape[1]
            donor_index = int(data["donor_index"][donor_slot])
            acceptor_index = int(data["acceptor_index"][acceptor_slot])
            if donor_side == "ligand":
                donor_label = topology.ligand_atom_labels[donor_index]
                acceptor_label = topology.protein_atom_labels[acceptor_index]
                acceptor_side = "protein"
            else:
                donor_label = topology.protein_atom_labels[donor_index]
                acceptor_label = topology.ligand_atom_labels[acceptor_index]
                acceptor_side = "ligand"
            pair = (0, donor_slot, acceptor_slot)
            rows.append(
                {
                    "donor": {
                        "side": donor_side,
                        "index": donor_index,
                        "label": donor_label,
                    },
                    "acceptor": {
                        "side": acceptor_side,
                        "index": acceptor_index,
                        "label": acceptor_label,
                    },
                    "distance_angstrom": float(data["distance"][pair].detach().cpu()),
                    "radial_gate": float(data["radial"][pair].detach().cpu()),
                    "donor_cosine": float(data["donor_cosine"][pair].detach().cpu()),
                    "donor_target_cosine": float(data["donor_target_cosine"][pair].detach().cpu()),
                    "donor_cone_gate": float(data["donor_gate"][pair].detach().cpu()),
                    "donor_axis_quality": float(data["donor_axis_quality"][pair].detach().cpu()),
                    "acceptor_cosine": float(data["acceptor_cosine"][pair].detach().cpu()),
                    "acceptor_target_cosine": float(
                        data["acceptor_target_cosine"][pair].detach().cpu()
                    ),
                    "acceptor_cone_gate": float(data["acceptor_gate"][pair].detach().cpu()),
                    "acceptor_axis_quality": float(
                        data["acceptor_axis_quality"][pair].detach().cpu()
                    ),
                    "weight": float(weight[donor_slot, acceptor_slot]),
                }
            )
        return rows

    def top_formal_charge_pairs(
        data: dict[str, Tensor],
        *,
        attractive: bool,
        limit: int = 8,
    ) -> list[dict[str, object]]:
        pair_energy = data["pair_energy"][0]
        if not pair_energy.numel():
            return []
        active = data["switch"][0] > 0
        signed = pair_energy < 0 if attractive else pair_energy > 0
        selected_flat = (active & signed).flatten().nonzero(as_tuple=False).flatten()
        if not selected_flat.numel():
            return []
        ranking = pair_energy.flatten()[selected_flat]
        order = torch.argsort(ranking, descending=not attractive)[:limit]
        selected_flat = selected_flat[order]
        rows: list[dict[str, object]] = []
        for flat in selected_flat.tolist():
            ligand_slot = flat // pair_energy.shape[1]
            protein_slot = flat % pair_energy.shape[1]
            pair = (0, ligand_slot, protein_slot)
            ligand_charge = float(data["ligand_charge"][ligand_slot].detach().cpu())
            protein_charge = float(data["protein_charge"][protein_slot].detach().cpu())
            rows.append(
                {
                    "ligand_site": {
                        "index": ligand_slot,
                        "label": topology.ligand_charge_site_labels[ligand_slot],
                        "charge_e": ligand_charge,
                    },
                    "protein_site": {
                        "index": protein_slot,
                        "label": topology.protein_charge_site_labels[protein_slot],
                        "charge_e": protein_charge,
                    },
                    "charge_product_e2": float(data["charge_product"][pair].detach().cpu()),
                    "distance_angstrom": float(data["distance"][pair].detach().cpu()),
                    "softcore_distance_angstrom": float(
                        data["softcore_distance"][pair].detach().cpu()
                    ),
                    "screening": float(data["screening"][pair].detach().cpu()),
                    "switch": float(data["switch"][pair].detach().cpu()),
                    "energy_kcal_mol": float(pair_energy[ligand_slot, protein_slot].detach().cpu()),
                }
            )
        return rows

    def top_pi_pairs(
        data: dict[str, Tensor],
        limit: int = 8,
    ) -> list[dict[str, object]]:
        weight = data["weight"][0]
        if not weight.numel():
            return []
        selected = (weight > 0).flatten().nonzero(as_tuple=False).flatten()
        if not selected.numel():
            return []
        selected = selected[torch.argsort(weight.flatten()[selected], descending=True)[:limit]]
        rows: list[dict[str, object]] = []
        for flat in selected.tolist():
            ligand_ring = flat // weight.shape[1]
            protein_ring = flat % weight.shape[1]
            pair = (0, ligand_ring, protein_ring)
            cosine = float(data["normal_cosine_squared"][pair].detach().cpu()) ** 0.5
            rows.append(
                {
                    "ligand_ring": {
                        "index": ligand_ring,
                        "label": topology.ligand_aromatic_ring_labels[ligand_ring],
                    },
                    "protein_ring": {
                        "index": protein_ring,
                        "label": topology.protein_aromatic_ring_labels[protein_ring],
                    },
                    "distance_angstrom": float(data["distance"][pair].detach().cpu()),
                    "normal_angle_degrees": math.degrees(math.acos(max(-1.0, min(1.0, cosine)))),
                    "ligand_offset_angstrom": float(
                        data["ligand_offset_squared"][pair].sqrt().detach().cpu()
                    ),
                    "protein_offset_angstrom": float(
                        data["protein_offset_squared"][pair].sqrt().detach().cpu()
                    ),
                    "parallel_gate": float(data["parallel_gate"][pair].detach().cpu()),
                    "t_gate": float(data["t_gate"][pair].detach().cpu()),
                    "weight": float(weight[ligand_ring, protein_ring].detach().cpu()),
                }
            )
        return rows

    def top_cation_pi_pairs(
        data: dict[str, Tensor],
        *,
        ring_side: str,
        limit: int = 8,
    ) -> list[dict[str, object]]:
        weight = data["weight"][0]
        if not weight.numel():
            return []
        selected = (weight > 0).flatten().nonzero(as_tuple=False).flatten()
        if not selected.numel():
            return []
        selected = selected[torch.argsort(weight.flatten()[selected], descending=True)[:limit]]
        ring_mask = data["ring_mask"]
        cation_mask = data["cation_mask"]
        ring_indices = ring_mask.nonzero(as_tuple=False).flatten().tolist()
        cation_indices = cation_mask.nonzero(as_tuple=False).flatten().tolist()
        ring_labels = (
            topology.ligand_aromatic_ring_labels
            if ring_side == "ligand"
            else topology.protein_aromatic_ring_labels
        )
        cation_labels = (
            topology.protein_charge_site_labels
            if ring_side == "ligand"
            else topology.ligand_charge_site_labels
        )
        rows: list[dict[str, object]] = []
        for flat in selected.tolist():
            ring_slot = flat // weight.shape[1]
            cation_slot = flat % weight.shape[1]
            ring_index = ring_indices[ring_slot]
            cation_index = cation_indices[cation_slot]
            pair = (0, ring_slot, cation_slot)
            rows.append(
                {
                    "ring": {
                        "index": ring_index,
                        "label": ring_labels[ring_index],
                        "side": ring_side,
                    },
                    "cation": {
                        "index": cation_index,
                        "label": cation_labels[cation_index],
                        "side": "protein" if ring_side == "ligand" else "ligand",
                    },
                    "distance_angstrom": float(data["distance"][pair].detach().cpu()),
                    "offset_angstrom": float(data["offset_squared"][pair].sqrt().detach().cpu()),
                    "radial_gate": float(data["radial"][pair].detach().cpu()),
                    "offset_gate": float(data["offset_gate"][pair].detach().cpu()),
                    "weight": float(weight[ring_slot, cation_slot].detach().cpu()),
                }
            )
        return rows

    def top_halogen_pairs(
        data: dict[str, Tensor],
        limit: int = 8,
    ) -> list[dict[str, object]]:
        weight = data["weight"][0]
        if not weight.numel():
            return []
        selected = (weight > 0).flatten().nonzero(as_tuple=False).flatten()
        if not selected.numel():
            return []
        selected = selected[torch.argsort(weight.flatten()[selected], descending=True)[:limit]]
        rows: list[dict[str, object]] = []
        for flat in selected.tolist():
            donor_slot = flat // weight.shape[1]
            acceptor_slot = flat % weight.shape[1]
            halogen_index = int(data["halogen_index"][donor_slot])
            parent_index = int(data["parent_index"][donor_slot])
            acceptor_index = int(data["acceptor_index"][acceptor_slot])
            pair = (0, donor_slot, acceptor_slot)
            sigma_angle = 180.0 - math.degrees(
                math.acos(
                    max(
                        -1.0,
                        min(1.0, float(data["sigma_cosine"][pair].detach().cpu())),
                    )
                )
            )
            rows.append(
                {
                    "halogen": {
                        "index": halogen_index,
                        "label": topology.ligand_atom_labels[halogen_index],
                    },
                    "parent": {
                        "index": parent_index,
                        "label": topology.ligand_atom_labels[parent_index],
                    },
                    "acceptor": {
                        "index": acceptor_index,
                        "label": topology.protein_atom_labels[acceptor_index],
                    },
                    "distance_angstrom": float(data["distance"][pair].detach().cpu()),
                    "normalized_distance": float(data["normalized_distance"][pair].detach().cpu()),
                    "c_x_acceptor_angle_degrees": sigma_angle,
                    "sigma_gate": float(data["sigma_gate"][pair].detach().cpu()),
                    "acceptor_gate": float(data["acceptor_gate"][pair].detach().cpu()),
                    "weight": float(weight[donor_slot, acceptor_slot].detach().cpu()),
                }
            )
        return rows

    def top_metal_pairs(
        data: dict[str, Tensor],
        limit: int = 16,
    ) -> list[dict[str, object]]:
        occupancy = data["occupancy"][0]
        if not occupancy.numel():
            return []
        active = (occupancy > 0).flatten().nonzero(as_tuple=False).flatten()
        if not active.numel():
            return []
        selected = active[torch.argsort(occupancy.flatten()[active], descending=True)[:limit]]
        rows: list[dict[str, object]] = []
        for flat in selected.tolist():
            donor_count = occupancy.shape[1]
            metal_slot = flat // donor_count
            donor_slot = flat % donor_count
            donor_index = int(data["donor_index"][donor_slot])
            pair = (0, metal_slot, donor_slot)
            rows.append(
                {
                    "metal": {
                        "index": metal_slot,
                        "label": metal_site_labels[metal_slot],
                        "profile": metal_profile_labels[metal_slot],
                        "atomic_number": int(data["metal_atomic_number"][metal_slot]),
                    },
                    "ligand_donor": {
                        "index": donor_index,
                        "label": topology.ligand_atom_labels[donor_index],
                        "atomic_number": int(ligand_metal_donor_element[donor_slot]),
                    },
                    "distance_angstrom": float(data["distance"][pair].detach().cpu()),
                    "r0_angstrom": float(data["r0"][pair].detach().cpu()),
                    "alignment_cosine": float(data["alignment_cosine"][pair].detach().cpu()),
                    "alignment": float(data["alignment"][pair].detach().cpu()),
                    "occupancy": float(occupancy[metal_slot, donor_slot].detach().cpu()),
                    "radial_repulsion_kcal_mol": float(
                        data["radial_repulsion"][pair].detach().cpu()
                    ),
                    "directional_attraction_kcal_mol": float(
                        data["directional_attraction"][pair].detach().cpu()
                    ),
                    "pair_energy_kcal_mol": float(data["pair_energy"][pair].detach().cpu()),
                }
            )
        return rows

    def top_metal_non_donor_repulsions(
        data: dict[str, Tensor],
        limit: int = 8,
    ) -> list[dict[str, object]]:
        repulsion = data["non_donor_repulsion"][0]
        if not repulsion.numel():
            return []
        active = (repulsion > 0).flatten().nonzero(as_tuple=False).flatten()
        if not active.numel():
            return []
        selected = active[torch.argsort(repulsion.flatten()[active], descending=True)[:limit]]
        rows: list[dict[str, object]] = []
        for flat in selected.tolist():
            atom_count = repulsion.shape[1]
            metal_slot = flat // atom_count
            atom_slot = flat % atom_count
            atom_index = int(data["non_donor_index"][atom_slot])
            rows.append(
                {
                    "metal": {
                        "index": metal_slot,
                        "label": metal_site_labels[metal_slot],
                        "profile": metal_profile_labels[metal_slot],
                        "atomic_number": int(data["metal_atomic_number"][metal_slot]),
                    },
                    "ligand_atom": {
                        "index": atom_index,
                        "label": topology.ligand_atom_labels[atom_index],
                    },
                    "distance_angstrom": float(
                        data["non_donor_distance"][0, metal_slot, atom_slot].detach().cpu()
                    ),
                    "repulsion_kcal_mol": float(repulsion[metal_slot, atom_slot].detach().cpu()),
                }
            )
        return rows

    ambiguous_histidine_labels = [
        label
        for label, excluded in zip(
            topology.protein_atom_labels,
            topology.protein_is_ambiguous_histidine.tolist(),
            strict=True,
        )
        if excluded
    ]
    unsupported_variant_labels = [
        label
        for label, excluded in zip(
            topology.protein_atom_labels,
            topology.protein_is_unsupported_variant.tolist(),
            strict=True,
        )
        if excluded
    ]
    geometry_excluded_labels = [
        label
        for label, excluded in zip(
            topology.protein_atom_labels,
            topology.protein_is_geometry_excluded_hbond_site.tolist(),
            strict=True,
        )
        if excluded
    ]
    ligand_geometry_excluded_labels = [
        label
        for label, excluded in zip(
            topology.ligand_atom_labels,
            topology.ligand_is_geometry_excluded_hbond_site.tolist(),
            strict=True,
        )
        if excluded
    ]
    charge_data = weights["screened_formal_charge"]
    charge_pair_energy = charge_data["pair_energy"][0]
    active_charge_pair = charge_data["switch"][0] > 0
    attractive_charge_pair = active_charge_pair & (charge_pair_energy < 0)
    repulsive_charge_pair = active_charge_pair & (charge_pair_energy > 0)
    ligand_charge_sites = int(topology.ligand_charge_site_charge.numel())
    protein_charge_sites = int(topology.protein_charge_site_charge.numel())
    if ligand_charge_sites == 0:
        charge_eligibility = "ineligible_no_ligand_formal_charge"
    elif protein_charge_sites == 0:
        charge_eligibility = "ineligible_no_protein_charge_site"
    else:
        charge_eligibility = "eligible"
    return {
        "typing_counts": topology.term_counts(),
        "exclusions": {
            "plain_histidine_tautomer_ambiguous": (ambiguous_histidine_labels),
            "unsupported_explicit_residue_variants": (unsupported_variant_labels),
            "missing_or_mismatched_hbond_geometry": (geometry_excluded_labels),
            "ligand_unsupported_hbond_geometry": (ligand_geometry_excluded_labels),
            "ligand_formal_charge_sites": list(topology.ligand_charge_site_exclusion_labels),
            "protein_formal_charge_sites": list(topology.protein_charge_site_exclusion_labels),
            "ligand_aromatic_rings": list(topology.ligand_aromatic_ring_exclusion_labels),
            "protein_aromatic_rings": list(topology.protein_aromatic_ring_exclusion_labels),
            "ligand_halogen_donors": list(topology.ligand_halogen_exclusion_labels),
            "protein_halogen_acceptors": list(topology.protein_halogen_exclusion_labels),
            "ligand_metal_donors": list(ligand_metal_donor_exclusion_labels),
            "metal_sites": list(metal_typing_exclusion_labels),
        },
        "hydrophobic": {
            **summarize(weights["hydrophobic"]),
            "raw_weight_sum": float(weights["hydrophobic"]["raw_weight"][0].sum().detach().cpu()),
            "pi_overlap_weight_sum": float(
                (weights["hydrophobic"]["raw_weight"][0] * weights["hydrophobic"]["pi_coverage"][0])
                .sum()
                .detach()
                .cpu()
            ),
            "top_pairs": top_hydrophobic_pairs(weights["hydrophobic"]),
        },
        "hydrogen_bond": {
            "ligand_donor_to_protein_acceptor": {
                **summarize(weights["ligand_donor_to_protein_acceptor"]),
                "top_pairs": top_hydrogen_bond_pairs(
                    weights["ligand_donor_to_protein_acceptor"],
                    donor_side="ligand",
                ),
                "top_radial_candidates": top_hydrogen_bond_pairs(
                    weights["ligand_donor_to_protein_acceptor"],
                    donor_side="ligand",
                    ranking_field="radial",
                ),
            },
            "protein_donor_to_ligand_acceptor": {
                **summarize(weights["protein_donor_to_ligand_acceptor"]),
                "top_pairs": top_hydrogen_bond_pairs(
                    weights["protein_donor_to_ligand_acceptor"],
                    donor_side="protein",
                ),
                "top_radial_candidates": top_hydrogen_bond_pairs(
                    weights["protein_donor_to_ligand_acceptor"],
                    donor_side="protein",
                    ranking_field="radial",
                ),
            },
        },
        "screened_formal_charge": {
            "eligibility": charge_eligibility,
            "ligand_site_count": ligand_charge_sites,
            "protein_site_count": protein_charge_sites,
            "ligand_sites": formal_charge_site_inventory(
                topology.ligand_charge_site_membership,
                topology.ligand_charge_site_charge,
                topology.ligand_charge_site_labels,
                topology.ligand_atom_labels,
            ),
            "protein_sites": formal_charge_site_inventory(
                topology.protein_charge_site_membership,
                topology.protein_charge_site_charge,
                topology.protein_charge_site_labels,
                topology.protein_atom_labels,
            ),
            "candidate_pairs": int(charge_pair_energy.numel()),
            "active_pairs": int(active_charge_pair.sum().item()),
            "attractive_pairs": int(attractive_charge_pair.sum().item()),
            "repulsive_pairs": int(repulsive_charge_pair.sum().item()),
            "attractive_energy_kcal_mol": float(
                charge_pair_energy.masked_select(attractive_charge_pair).sum().detach().cpu()
            ),
            "repulsive_energy_kcal_mol": float(
                charge_pair_energy.masked_select(repulsive_charge_pair).sum().detach().cpu()
            ),
            "total_energy_kcal_mol": float(charge_pair_energy.sum().detach().cpu()),
            "top_attractive_pairs": top_formal_charge_pairs(
                charge_data,
                attractive=True,
            ),
            "top_repulsive_pairs": top_formal_charge_pairs(
                charge_data,
                attractive=False,
            ),
        },
        "pi_stacking": {
            **summarize(weights["pi_stacking"]),
            "ligand_ring_count": len(topology.ligand_aromatic_ring_labels),
            "protein_ring_count": len(topology.protein_aromatic_ring_labels),
            "ligand_system_count": (
                int(topology.ligand_aromatic_ring_system.max().item()) + 1
                if topology.ligand_aromatic_ring_system.numel()
                else 0
            ),
            "protein_system_count": (
                int(topology.protein_aromatic_ring_system.max().item()) + 1
                if topology.protein_aromatic_ring_system.numel()
                else 0
            ),
            "top_pairs": top_pi_pairs(weights["pi_stacking"]),
        },
        "cation_pi": {
            "ligand_ring_to_protein_cation": {
                **summarize(weights["ligand_ring_to_protein_cation"]),
                "top_pairs": top_cation_pi_pairs(
                    weights["ligand_ring_to_protein_cation"],
                    ring_side="ligand",
                ),
            },
            "protein_ring_to_ligand_cation": {
                **summarize(weights["protein_ring_to_ligand_cation"]),
                "top_pairs": top_cation_pi_pairs(
                    weights["protein_ring_to_ligand_cation"],
                    ring_side="protein",
                ),
            },
        },
        "halogen_bond": {
            **summarize(weights["halogen_bond"]),
            "top_pairs": top_halogen_pairs(weights["halogen_bond"]),
        },
        "metal_coordination": {
            "metal_site_count": len(metal_site_labels),
            "site_profiles": list(metal_profile_labels),
            "attraction_enabled": (
                weights["metal_coordination"]["attraction_enabled"].detach().cpu().tolist()
            ),
            "ligand_donor_count": int(weights["metal_coordination"]["donor_index"].numel()),
            "candidate_pairs": int(weights["metal_coordination"]["distance"].numel()),
            "pair_energy_kcal_mol": float(
                weights["metal_coordination"]["pair_energy"][0].sum().detach().cpu()
            ),
            "overcoordination_energy_kcal_mol": float(
                weights["metal_coordination"]["overcoordination_energy"][0].sum().detach().cpu()
            ),
            "slot_energy_kcal_mol": float(
                weights["metal_coordination"]["slot_energy"][0].sum().detach().cpu()
            ),
            "non_donor_repulsion_kcal_mol": float(
                weights["metal_coordination"]["non_donor_repulsion"][0].sum().detach().cpu()
            ),
            "coordination_number": (
                weights["metal_coordination"]["coordination_number"][0].detach().cpu().tolist()
            ),
            "top_donor_pairs": top_metal_pairs(weights["metal_coordination"]),
            "top_non_donor_repulsions": top_metal_non_donor_repulsions(
                weights["metal_coordination"]
            ),
        },
        "polar_unsatisfied_proxy": {
            "status": "trace_only_unitless_no_force",
            "site_count": int(weights["polar_unsatisfied_proxy"]["ligand_index"].numel()),
            "sum": float(weights["polar_unsatisfied_proxy"]["value"][0].sum().detach().cpu()),
            "sites": [
                {
                    "ligand_index": int(atom_index),
                    "label": topology.ligand_atom_labels[int(atom_index)],
                    "burial": float(
                        weights["polar_unsatisfied_proxy"]["burial"][0, site_index].detach().cpu()
                    ),
                    "satisfaction": float(
                        weights["polar_unsatisfied_proxy"]["satisfaction"][0, site_index]
                        .detach()
                        .cpu()
                    ),
                    "value": float(
                        weights["polar_unsatisfied_proxy"]["value"][0, site_index].detach().cpu()
                    ),
                }
                for site_index, atom_index in enumerate(
                    weights["polar_unsatisfied_proxy"]["ligand_index"].tolist()
                )
            ],
        },
    }


def interaction_profile_metadata(
    config: InteractionEnergyConfig = InteractionEnergyConfig(),
) -> dict[str, object]:
    raw = load_interaction_v1()
    active = list(config.active_terms)
    return {
        "status": "active_diagnostic",
        "active_terms": active,
        "callable_energy_terms": list(ACTIVE_INTERACTION_TERMS),
        "inactive_terms": sorted(set(PLANNED_INTERACTION_TERMS) - set(active)),
        "trace_only_terms": ["polar_unsatisfied_proxy"],
        "formula_version": raw["formula_version"],
        "typing_policy": raw["typing_policy"],
        "external_engine": None,
        "vina": "excluded_from_guidance",
        "metal_coordination_v0": metal_coordination_v0_contract(),
        "metal_coordination_v1": metal_coordination_v1_contract(),
        "polar_unsatisfied_proxy": "trace_only_unitless_no_force",
        "claim": (
            "Default-on self-contained interaction diagnostics include hydrophobic, "
            "idealized heavy-atom hydrogen bond, screened formal charge, "
            "aromatic stacking, cation-pi, ligand-to-protein halogen bond, and "
            "profile-dispatched metal coordination. Metal attraction remains "
            "site-admission gated, and repulsion-only profiles are traced. "
            "None is an affinity or free-energy estimate."
        ),
    }


__all__ = [
    "ACTIVE_INTERACTION_TERMS",
    "InteractionEnergyConfig",
    "PLANNED_INTERACTION_TERMS",
    "interaction_contact_stats",
    "interaction_energy",
    "interaction_profile_metadata",
    "metal_coordination_v0_contract",
    "metal_coordination_v1_contract",
]
