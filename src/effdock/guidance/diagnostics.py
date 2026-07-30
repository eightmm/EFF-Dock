"""Diagnostics for crystal poses and saved EFF-Dock trajectories.

This module deliberately reports energies and gradients without optimizing a
pose.  Crystal coordinates are diagnostic inputs only and are never exposed to
the inference model or used to select guidance hyperparameters.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from .interaction import (
    InteractionEnergyConfig,
    interaction_contact_stats,
)
from .physical import PhysicalEnergyConfig, _pairwise_distance
from .runtime import (
    GuidanceEnergyConfig,
    guidance_energy,
    project_atom_forces,
)
from .system import PhysicalSystem

TRACE_SCHEMA_VERSION = "effdock.guidance_trace.v4"


@dataclass(frozen=True)
class PoseState:
    name: str
    coords: Tensor
    details: dict[str, object]


def fragment_centers(coords: Tensor, fragment_id: Tensor) -> Tensor:
    """Return geometric centers with shape ``[B,F,3]``."""
    if coords.ndim == 2:
        coords = coords.unsqueeze(0)
    if coords.ndim != 3 or coords.shape[-1] != 3:
        raise ValueError("coords must have shape [N,3] or [B,N,3]")
    fragment_id = fragment_id.to(device=coords.device, dtype=torch.long)
    n_fragments = int(fragment_id.max().item()) + 1
    centers = coords.new_zeros(coords.shape[0], n_fragments, 3)
    centers.index_add_(1, fragment_id, coords)
    counts = torch.bincount(fragment_id, minlength=n_fragments).to(coords.dtype)
    return centers / counts.clamp_min(1).view(1, -1, 1)


def _vector_stats(vectors: Tensor) -> dict[str, float]:
    norms = vectors.norm(dim=-1)
    return {
        "max_norm": float(norms.max().detach().cpu()) if norms.numel() else 0.0,
        "mean_norm": float(norms.mean().detach().cpu()) if norms.numel() else 0.0,
        "rms_norm": (float(norms.square().mean().sqrt().detach().cpu()) if norms.numel() else 0.0),
    }


def _protein_ligand_contact_stats(coords: Tensor, system: PhysicalSystem) -> dict[str, object]:
    distance = _pairwise_distance(coords, system.protein_coords)
    equilibrium = torch.sqrt(system.topology.uff_x.view(-1, 1) * system.protein_uff_x.view(1, -1))
    ratio = distance / equilibrium.clamp_min(1e-8)
    flat_ratio = int(ratio.argmin().item())
    ratio_ligand = flat_ratio // ratio.shape[1]
    ratio_protein = flat_ratio % ratio.shape[1]
    flat_distance = int(distance.argmin().item())
    distance_ligand = flat_distance // distance.shape[1]
    distance_protein = flat_distance % distance.shape[1]
    return {
        "pair_count": int(distance.numel()),
        "minimum_distance_angstrom": float(distance.flatten()[flat_distance].detach().cpu()),
        "minimum_distance_pair": {
            "ligand_atom_index": distance_ligand,
            "protein_shell_atom_index": distance_protein,
        },
        "minimum_distance_over_uff_x": float(ratio.flatten()[flat_ratio].detach().cpu()),
        "largest_overlap_pair": {
            "ligand_atom_index": ratio_ligand,
            "protein_shell_atom_index": ratio_protein,
            "distance_angstrom": float(distance[ratio_ligand, ratio_protein].detach().cpu()),
            "uff_x_angstrom": float(equilibrium[ratio_ligand, ratio_protein].detach().cpu()),
        },
    }


def trace_guidance_pose(
    coords: Tensor,
    system: PhysicalSystem,
    *,
    energy_config: PhysicalEnergyConfig = PhysicalEnergyConfig(),
    interaction_config: InteractionEnergyConfig = InteractionEnergyConfig(),
    pose_kind: str,
    pose_index: int = 0,
    step: int | None = None,
    t: float | None = None,
) -> dict[str, object]:
    """Trace unified physical + interaction energy and coordinate gradients."""
    work = coords.detach().to(
        device=system.protein_coords.device,
        dtype=system.protein_coords.dtype,
    )
    if work.ndim != 2 or work.shape != (system.topology.num_atoms, 3):
        raise ValueError(
            "trace coordinates must have shape "
            f"[{system.topology.num_atoms},3], got {tuple(work.shape)}"
        )
    work = work.clone().requires_grad_(True)
    components = guidance_energy(
        work,
        system,
        GuidanceEnergyConfig(
            physical=energy_config,
            interaction=interaction_config,
        ),
    )
    if not all(bool(torch.isfinite(value)) for value in components.values()):
        raise FloatingPointError("non-finite guidance energy component")

    physical_total = sum(
        (
            value
            for name, value in components.items()
            if name != "total" and not name.startswith("interaction_")
        ),
        start=work.new_zeros(()),
    )
    interaction_total = sum(
        (value for name, value in components.items() if name.startswith("interaction_")),
        start=work.new_zeros(()),
    )

    force_stats: dict[str, dict[str, float]] = {}
    total_force: Tensor | None = None
    for name, value in components.items():
        if value.requires_grad:
            gradient = torch.autograd.grad(
                value,
                work,
                retain_graph=True,
                allow_unused=True,
            )[0]
        else:
            gradient = None
        force = torch.zeros_like(work) if gradient is None else -gradient
        if not bool(torch.isfinite(force).all()):
            raise FloatingPointError(f"non-finite force for guidance term {name!r}")
        force_stats[name] = _vector_stats(force)
        if name == "total":
            total_force = force
    if total_force is None:
        raise RuntimeError("guidance energy did not return a total term")

    centers = fragment_centers(work.detach(), system.topology.fragment_id)
    translation, angular = project_atom_forces(
        total_force.unsqueeze(0),
        work.detach().unsqueeze(0),
        centers,
        system.topology.fragment_id,
        system.topology.mass,
    )
    row: dict[str, object] = {
        "pose_kind": pose_kind,
        "pose_index": int(pose_index),
        "step": None if step is None else int(step),
        "t": None if t is None else float(t),
        "energies": {name: float(value.detach().cpu()) for name, value in components.items()},
        "energy_groups": {
            "physical": float(physical_total.detach().cpu()),
            "interaction": float(interaction_total.detach().cpu()),
            "combined": float(components["total"].detach().cpu()),
        },
        "force_by_term": force_stats,
        "fragment_projection": {
            "mass_preconditioned_translation": _vector_stats(translation[0]),
            "inertia_preconditioned_rotation": _vector_stats(angular[0]),
        },
        "protein_ligand_contacts": _protein_ligand_contact_stats(work.detach(), system),
        "interaction_contacts": interaction_contact_stats(
            work.detach(),
            system,
            interaction_config,
        ),
    }
    return row


def trace_physical_pose(
    coords: Tensor,
    system: PhysicalSystem,
    *,
    energy_config: PhysicalEnergyConfig = PhysicalEnergyConfig(),
    interaction_config: InteractionEnergyConfig = InteractionEnergyConfig(),
    pose_kind: str,
    pose_index: int = 0,
    step: int | None = None,
    t: float | None = None,
) -> dict[str, object]:
    """Backward-compatible alias for the unified guidance trace."""
    return trace_guidance_pose(
        coords,
        system,
        energy_config=energy_config,
        interaction_config=interaction_config,
        pose_kind=pose_kind,
        pose_index=pose_index,
        step=step,
        t=t,
    )


def _fragment_component(
    fragment_id: Tensor,
    bond_index: Tensor,
    start_fragment: int,
    blocked_edge: tuple[int, int],
) -> set[int]:
    adjacency: dict[int, set[int]] = {
        int(fragment): set() for fragment in torch.unique(fragment_id).tolist()
    }
    for atom_i, atom_j in bond_index.T.tolist():
        frag_i = int(fragment_id[atom_i])
        frag_j = int(fragment_id[atom_j])
        if frag_i == frag_j:
            continue
        if {frag_i, frag_j} == set(blocked_edge):
            continue
        adjacency[frag_i].add(frag_j)
        adjacency[frag_j].add(frag_i)
    visited = {start_fragment}
    queue = [start_fragment]
    for fragment in queue:
        for neighbor in adjacency[fragment]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return visited


def _rotate_about_axis(coords: Tensor, origin: Tensor, axis: Tensor, angle: float) -> Tensor:
    axis = axis / axis.norm().clamp_min(1e-12)
    relative = coords - origin
    axis_expanded = axis.view(1, 3).expand_as(relative)
    return (
        relative * math.cos(angle)
        + torch.linalg.cross(axis_expanded, relative, dim=-1) * math.sin(angle)
        + axis * (relative * axis).sum(dim=-1, keepdim=True) * (1.0 - math.cos(angle))
        + origin
    )


def make_crystal_perturbations(
    crystal_coords: Tensor,
    system: PhysicalSystem,
    *,
    stretch_angstrom: float = 0.5,
    torsion_degrees: float = 30.0,
    overlap_distance_angstrom: float = 0.5,
) -> list[PoseState]:
    """Create deterministic, interpretable rigid-fragment diagnostic failures."""
    if stretch_angstrom <= 0 or overlap_distance_angstrom <= 0:
        raise ValueError("perturbation distances must be positive")
    coords = crystal_coords.detach().to(
        device=system.protein_coords.device,
        dtype=system.protein_coords.dtype,
    )
    topology = system.topology
    states = [PoseState("crystal", coords.clone(), {"operation": "none"})]

    if topology.bond_index.shape[1]:
        atom_i = int(topology.bond_index[0, 0])
        atom_j = int(topology.bond_index[1, 0])
        frag_i = int(topology.fragment_id[atom_i])
        frag_j = int(topology.fragment_id[atom_j])
        side_i = _fragment_component(
            topology.fragment_id,
            topology.bond_index,
            frag_i,
            (frag_i, frag_j),
        )
        side_j = _fragment_component(
            topology.fragment_id,
            topology.bond_index,
            frag_j,
            (frag_i, frag_j),
        )
        moving_fragments = side_j if len(side_j) <= len(side_i) else side_i
        if frag_j in moving_fragments:
            fixed_atom, moving_atom = atom_i, atom_j
        else:
            fixed_atom, moving_atom = atom_j, atom_i
        moving_mask = torch.tensor(
            [
                int(fragment) in moving_fragments
                for fragment in topology.fragment_id.detach().cpu().tolist()
            ],
            device=coords.device,
            dtype=torch.bool,
        )
        axis = coords[moving_atom] - coords[fixed_atom]
        unit_axis = axis / axis.norm().clamp_min(1e-12)

        stretched = coords.clone()
        stretched[moving_mask] += float(stretch_angstrom) * unit_axis
        states.append(
            PoseState(
                "cut_bond_stretch",
                stretched,
                {
                    "operation": "translate_fragment_component_along_cut_bond",
                    "cut_bond_atom_indices": [fixed_atom, moving_atom],
                    "moving_fragment_ids": sorted(moving_fragments),
                    "distance_angstrom": float(stretch_angstrom),
                },
            )
        )

        rotated = coords.clone()
        rotated[moving_mask] = _rotate_about_axis(
            coords[moving_mask],
            coords[fixed_atom],
            unit_axis,
            math.radians(float(torsion_degrees)),
        )
        states.append(
            PoseState(
                "cut_bond_torsion",
                rotated,
                {
                    "operation": "rotate_fragment_component_about_cut_bond",
                    "cut_bond_atom_indices": [fixed_atom, moving_atom],
                    "moving_fragment_ids": sorted(moving_fragments),
                    "angle_degrees": float(torsion_degrees),
                },
            )
        )

    distance = _pairwise_distance(coords, system.protein_coords)
    equilibrium = torch.sqrt(topology.uff_x.view(-1, 1) * system.protein_uff_x.view(1, -1))
    overlap_flat = int((distance / equilibrium.clamp_min(1e-8)).argmin().item())
    ligand_atom = overlap_flat // distance.shape[1]
    protein_atom = overlap_flat % distance.shape[1]
    direction = coords[ligand_atom] - system.protein_coords[protein_atom]
    if float(direction.norm()) < 1e-8:
        direction = coords.new_tensor([1.0, 0.0, 0.0])
    direction = direction / direction.norm()
    target = system.protein_coords[protein_atom] + float(overlap_distance_angstrom) * direction
    translated = coords + (target - coords[ligand_atom])
    states.append(
        PoseState(
            "protein_ligand_overlap",
            translated,
            {
                "operation": "rigid_ligand_translation_to_protein_overlap",
                "ligand_atom_index": ligand_atom,
                "protein_shell_atom_index": protein_atom,
                "target_distance_angstrom": float(overlap_distance_angstrom),
            },
        )
    )
    return states


__all__ = [
    "PoseState",
    "TRACE_SCHEMA_VERSION",
    "fragment_centers",
    "make_crystal_perturbations",
    "trace_guidance_pose",
    "trace_physical_pose",
]
