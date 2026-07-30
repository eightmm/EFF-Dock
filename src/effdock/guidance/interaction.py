"""Differentiable, self-contained protein-ligand interaction guidance.

RDKit is used only while building the static ligand SMARTS masks. Every
coordinate-dependent value in this module is computed with Torch. These terms
are pose-guidance diagnostics, not affinity or free-energy predictions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

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
)


def metal_coordination_v0_contract() -> dict[str, object]:
    """Return the narrow, inactive Zn(II) coordination admission contract.

    The formula is recorded now so later parameter fitting cannot silently
    change the scientific target.  It is deliberately not callable as an
    energy term until site and donor typing plus constants are admitted.
    """
    return {
        "status": "contract_only_inactive",
        "supported_scope": {
            "metal": "Zn(II)",
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
        "parameter_status": "r0, D, a, tau_theta, sigma_r, switches, and penalties are not frozen",
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
) -> dict[str, dict[str, Tensor]]:
    topology = system.interaction_topology
    if topology is None:
        raise ValueError("active interaction terms require system interaction typing")
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
    batch_size = coords.shape[0]
    protein_coords = system.protein_coords.unsqueeze(0).expand(
        batch_size,
        -1,
        -1,
    )
    protein_direction = topology.protein_outward_direction.unsqueeze(0).expand(
        batch_size,
        -1,
        -1,
    )
    protein_direction_valid = topology.protein_direction_valid.unsqueeze(0).expand(batch_size, -1)
    protein_direction_quality = topology.protein_direction_quality.unsqueeze(0).expand(
        batch_size, -1
    )
    ligand_target_cosine = topology.ligand_direction_target_cosine
    protein_target_cosine = topology.protein_direction_target_cosine
    ligand_charge_coords = _charge_group_coords(
        coords,
        topology.ligand_charge_site_membership,
    )
    protein_charge_coords = _charge_group_coords(
        protein_coords,
        topology.protein_charge_site_membership,
    )
    screened_formal_charge = _screened_formal_charge_components(
        ligand_charge_coords,
        protein_charge_coords,
        topology.ligand_charge_site_charge,
        topology.protein_charge_site_charge,
        config,
    )

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
    return {
        "screened_formal_charge": {
            **screened_formal_charge,
            "ligand_charge": topology.ligand_charge_site_charge,
            "protein_charge": topology.protein_charge_site_charge,
        },
        "hydrophobic": {
            "distance": hydrophobic_distance,
            "weight": hydrophobic_weight,
            "ligand_index": ligand_hydrophobe,
            "protein_index": protein_hydrophobe,
        },
        "ligand_donor_to_protein_acceptor": (
            {
                **ligand_donor_to_protein_acceptor,
                "donor_index": ligand_donor,
                "acceptor_index": protein_acceptor,
            }
        ),
        "protein_donor_to_ligand_acceptor": (
            {
                **protein_donor_to_ligand_acceptor,
                "donor_index": protein_donor,
                "acceptor_index": ligand_acceptor,
            }
        ),
    }


def interaction_energy(
    coords: Tensor,
    system: PhysicalSystem,
    config: InteractionEnergyConfig = InteractionEnergyConfig(),
) -> dict[str, Tensor]:
    """Return active, per-pose interaction components."""
    batched, squeeze = _as_batch(coords)
    if batched.shape[-2] != system.topology.num_atoms:
        raise ValueError("interaction coordinate atom count does not match topology")
    if not config.active_terms:
        total = _zero(batched)
        return {"total": total.squeeze(0) if squeeze else total}
    weights = _interaction_weights(batched, system, config)
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
        },
        "hydrophobic": {
            **summarize(weights["hydrophobic"]),
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
    }


def interaction_profile_metadata(
    config: InteractionEnergyConfig = InteractionEnergyConfig(),
) -> dict[str, object]:
    raw = load_interaction_v1()
    active = list(config.active_terms)
    return {
        "status": "active_diagnostic",
        "active_terms": active,
        "inactive_terms": sorted(set(PLANNED_INTERACTION_TERMS) - set(active)),
        "formula_version": raw["formula_version"],
        "typing_policy": raw["typing_policy"],
        "external_engine": None,
        "vina": "excluded_from_guidance",
        "metal_coordination_v0": metal_coordination_v0_contract(),
        "claim": (
            "Hydrophobic and idealized missing-valence-cone heavy-atom "
            "hydrogen-bond terms plus a conservative screened formal-charge-"
            "group term are active diagnostics inside unified GuidanceEnergy. "
            "They are not explicit-hydrogen geometry, partial-charge "
            "electrostatics, solvation, affinity, or free-energy estimates."
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
]
