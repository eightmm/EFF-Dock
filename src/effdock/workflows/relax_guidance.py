"""Naive rigid-fragment relaxation under the unified EFF-Dock guidance energy.

This workflow deliberately does not call the learned docking model or the
production sampler.  It supports both the original crystal-basin fragment tear
and inference-valid, explicit-pocket prior initializations.  The latter
reproduce the active sampler's CPU float32 translation/rotation draw order, but
still test guidance-only descent rather than the learned ODE.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from effdock.geometry.flow_matching import (
    integrate_se3_step,
    sample_prior_rotations,
)
from effdock.geometry.se3 import quaternion_to_matrix
from effdock.guidance import (
    GuidanceEnergyConfig,
    InteractionEnergyConfig,
    PhysicalEnergyConfig,
    UnsupportedPhysicalChemistryError,
    build_physical_system,
    guidance_energy,
    interaction_contact_stats,
)
from effdock.guidance.diagnostics import fragment_centers
from effdock.guidance.parameterization import guidance_parameter_identity
from effdock.guidance.runtime import project_atom_forces
from effdock.guidance.topology import build_physical_topology
from effdock.inference.io import write_traj_pdb, write_traj_sdf
from effdock.preprocess.fragments import decompose_fragments
from effdock.preprocess.ligand import featurize_ligand
from effdock.workflows.trace_physical import (
    _file_sha256,
    _implementation_identity,
    _load_trace_ligand,
)

RELAXATION_SCHEMA_VERSION = "effdock.guidance_relaxation.v3"
PROTOCOL_ID = "EFFDOCK-CRYSTAL-BASIN-REASSEMBLY-V1"
PRIOR_RELAXATION_PROTOCOL_ID = "EFFDOCK-GUIDANCE-PRIOR-RELAXATION-V1"
INITIALIZATION_MODES = (
    "crystal_tear",
    "pocket_gaussian",
    "model_prior",
)


def _maximum_active_interaction_cutoff() -> float:
    config = InteractionEnergyConfig()
    return max(
        config.hydrophobic_cutoff,
        config.hydrogen_bond_cutoff,
        config.formal_charge_cutoff,
    )


def _protocol_id_for_initialization(initialization_mode: str) -> str:
    if initialization_mode == "crystal_tear":
        return PROTOCOL_ID
    if initialization_mode in {"pocket_gaussian", "model_prior"}:
        return PRIOR_RELAXATION_PROTOCOL_ID
    raise ValueError(f"unknown initialization mode: {initialization_mode!r}")


@dataclass(frozen=True)
class RigidRelaxationConfig:
    """Pre-registered numerical controls for gradient-only SE(3) descent."""

    initialization_mode: str = "crystal_tear"
    tear_distance_angstrom: float = 3.0
    prior_sigma_angstrom: float = 0.5
    seed: int = 20260730
    max_steps: int = 500
    save_every: int = 5
    base_step_size: float = 1.0
    max_translation_step_angstrom: float = 0.10
    max_rotation_step_degrees: float = 5.0
    max_atom_step_angstrom: float = 0.10
    backtrack_factor: float = 0.5
    max_backtracks: int = 12
    convergence_displacement_angstrom: float = 1e-5
    convergence_patience: int = 20
    energy_increase_tolerance: float = 1e-10
    physical_cutoff_angstrom: float = 8.0
    protein_shell_cutoff_angstrom: float = 13.0

    def __post_init__(self) -> None:
        if self.initialization_mode not in INITIALIZATION_MODES:
            raise ValueError(
                "initialization_mode must be one of "
                f"{INITIALIZATION_MODES}, got {self.initialization_mode!r}"
            )
        positive = (
            "tear_distance_angstrom",
            "prior_sigma_angstrom",
            "max_steps",
            "save_every",
            "base_step_size",
            "max_translation_step_angstrom",
            "max_rotation_step_degrees",
            "max_atom_step_angstrom",
            "max_backtracks",
            "convergence_displacement_angstrom",
            "convergence_patience",
            "physical_cutoff_angstrom",
            "protein_shell_cutoff_angstrom",
        )
        for name in positive:
            if float(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0 < self.backtrack_factor < 1:
            raise ValueError("backtrack_factor must be in (0,1)")
        active_cutoff = max(
            self.physical_cutoff_angstrom,
            _maximum_active_interaction_cutoff(),
        )
        if self.initialization_mode == "crystal_tear":
            required_shell = active_cutoff + self.tear_distance_angstrom
            if self.protein_shell_cutoff_angstrom + 1e-12 < required_shell:
                raise ValueError(
                    "protein shell cutoff must cover every active guidance "
                    "cutoff plus the declared crystal displacement envelope"
                )
        elif self.protein_shell_cutoff_angstrom + 1e-12 < 18.0:
            raise ValueError(
                "pocket-prior initialization requires a fixed receptor shell "
                "cutoff of at least 18 A"
            )
        if self.protein_shell_cutoff_angstrom <= active_cutoff:
            raise ValueError("protein shell cutoff must exceed every active guidance cutoff")


@dataclass
class RelaxationRun:
    mode: str
    status: str
    metrics: list[dict[str, Any]]
    frames: list[Tensor]
    saved_steps: list[int]
    total_backtracks: int
    shell_envelope_valid: bool


def _fragment_masses(mass: Tensor, fragment_id: Tensor) -> Tensor:
    n_fragments = int(fragment_id.max().item()) + 1
    result = mass.new_zeros(n_fragments)
    result.index_add_(0, fragment_id, mass)
    return result


def make_torn_fragment_pose(
    crystal_coords: Tensor,
    fragment_id: Tensor,
    atom_mass: Tensor,
    *,
    distance_angstrom: float,
    seed: int,
) -> tuple[Tensor, Tensor]:
    """Translate fragments independently while preserving the ligand mass COM.

    Centered Gaussian directions are rescaled so the largest fragment
    translation is exactly ``distance_angstrom``.  No rotation is applied.
    """
    if crystal_coords.ndim != 2 or crystal_coords.shape[-1] != 3:
        raise ValueError("crystal_coords must have shape [N,3]")
    if distance_angstrom <= 0:
        raise ValueError("distance_angstrom must be positive")
    fragment_id = fragment_id.to(
        device=crystal_coords.device,
        dtype=torch.long,
    )
    atom_mass = atom_mass.to(
        device=crystal_coords.device,
        dtype=crystal_coords.dtype,
    )
    if fragment_id.numel() != crystal_coords.shape[0]:
        raise ValueError("fragment_id must match crystal atom count")
    if atom_mass.numel() != crystal_coords.shape[0]:
        raise ValueError("atom_mass must match crystal atom count")
    n_fragments = int(fragment_id.max().item()) + 1
    if n_fragments < 2:
        raise ValueError("fragment tearing requires at least two fragments")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    displacement = torch.randn(
        n_fragments,
        3,
        generator=generator,
        dtype=torch.float64,
    ).to(device=crystal_coords.device, dtype=crystal_coords.dtype)
    fragment_mass = _fragment_masses(atom_mass, fragment_id)
    weighted_mean = (displacement * fragment_mass.unsqueeze(-1)).sum(
        dim=0
    ) / fragment_mass.sum().clamp_min(1e-12)
    displacement = displacement - weighted_mean
    maximum = displacement.norm(dim=-1).max()
    if float(maximum) < 1e-12:
        raise RuntimeError("deterministic fragment tear produced zero directions")
    displacement = displacement * (float(distance_angstrom) / maximum)
    torn = crystal_coords + displacement[fragment_id]
    return torn, displacement


def make_pocket_prior_fragment_pose(
    fragment_local_coords: Tensor,
    fragment_id: Tensor,
    pocket_center: Tensor,
    *,
    sigma_angstrom: float,
    seed: int,
    rotation_mode: str,
) -> tuple[Tensor, dict[str, Any]]:
    """Construct an absolute pose from the active sampler's ``t=0`` prior.

    The random draws intentionally occur on CPU in float32 and in the same
    order as :func:`effdock.inference.sampler.sample_batched`: one standard
    normal translation tensor followed by the public Uniform(SO3) quaternion
    sampler. ``rotation_mode="identity"`` is a translation-matched control,
    while ``"uniform"`` is the exact model-prior initialization.
    """
    if fragment_local_coords.ndim != 2 or fragment_local_coords.shape[-1] != 3:
        raise ValueError("fragment_local_coords must have shape [N,3]")
    if sigma_angstrom <= 0:
        raise ValueError("sigma_angstrom must be positive")
    if rotation_mode not in {"identity", "uniform"}:
        raise ValueError("rotation_mode must be 'identity' or 'uniform'")

    local = fragment_local_coords.detach().to(
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    fragment_id_cpu = fragment_id.detach().to(
        device=torch.device("cpu"),
        dtype=torch.long,
    )
    if fragment_id_cpu.numel() != local.shape[0]:
        raise ValueError("fragment_id must match fragment_local_coords")
    if fragment_id_cpu.numel() == 0 or int(fragment_id_cpu.min()) < 0:
        raise ValueError("fragment_id must be non-empty and non-negative")
    unique = torch.unique(fragment_id_cpu)
    expected = torch.arange(
        int(fragment_id_cpu.max()) + 1,
        dtype=torch.long,
    )
    if not torch.equal(unique, expected):
        raise ValueError("fragment_id must use contiguous IDs starting at zero")

    center = (
        torch.as_tensor(pocket_center)
        .detach()
        .to(
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
    )
    if center.shape != (3,) or not bool(torch.isfinite(center).all()):
        raise ValueError("pocket_center must be a finite tensor of shape [3]")

    n_fragments = int(fragment_id_cpu.max()) + 1
    frag_sizes = torch.bincount(
        fragment_id_cpu,
        minlength=n_fragments,
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    eps = torch.randn(
        n_fragments,
        3,
        generator=generator,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    translation_relative = float(sigma_angstrom) * eps
    if rotation_mode == "uniform":
        quaternions = sample_prior_rotations(
            n_fragments,
            frag_sizes=frag_sizes,
            generator=generator,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
    else:
        quaternions = _identity_quaternions(
            n_fragments,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
    translation_absolute = center.unsqueeze(0) + translation_relative
    pocket_centered_coords = _reconstruct_coordinates(
        local,
        fragment_id_cpu,
        translation_relative,
        quaternions,
    )
    initial_coords = pocket_centered_coords + center.unsqueeze(0)
    metadata: dict[str, Any] = {
        "standard_normal_translation_eps": eps,
        "translation_relative_to_pocket_angstrom": translation_relative,
        "translation_absolute_angstrom": translation_absolute,
        "quaternion_scalar_first": quaternions,
        "pocket_center_angstrom": center,
        "fragment_sizes": frag_sizes,
        "rotation_mode": rotation_mode,
        "cpu_float32_rng_draw_order": (
            "translation_standard_normal_then_uniform_so3"
            if rotation_mode == "uniform"
            else "translation_standard_normal_only"
        ),
        "exact_active_sampler_model_prior": rotation_mode == "uniform",
    }
    return initial_coords, metadata


def load_explicit_pocket_center(
    path: Path,
    *,
    sample_id: str,
    key: str | None = None,
) -> tuple[Tensor, dict[str, Any]]:
    """Load one explicit pocket center without consulting crystal coordinates."""
    source = path.resolve()
    if not source.is_file():
        raise FileNotFoundError(f"pocket-center JSON not found: {source}")
    payload = json.loads(source.read_text())
    if not isinstance(payload, dict):
        raise ValueError("pocket-center JSON must contain a mapping")
    selected_key = str(key) if key is not None else str(sample_id).lower().split("_", maxsplit=1)[0]
    if selected_key not in payload:
        raise KeyError(f"pocket center key {selected_key!r} is absent from {source}")
    entry = payload[selected_key]
    source_entry_metadata: dict[str, Any] = {}
    if isinstance(entry, dict):
        available = [name for name in ("center", "pocket_center") if name in entry]
        if not available:
            raise ValueError(
                f"pocket center entry {selected_key!r} must contain 'center' or 'pocket_center'"
            )
        if len(available) == 2 and entry["center"] != entry["pocket_center"]:
            raise ValueError(
                f"pocket center entry {selected_key!r} has conflicting "
                "'center' and 'pocket_center' values"
            )
        value = entry[available[0]]
        value_field = available[0]
        source_entry_metadata = {
            name: item for name, item in entry.items() if name not in {"center", "pocket_center"}
        }
    elif isinstance(entry, list):
        value = entry
        value_field = "list"
    else:
        raise ValueError(f"pocket center entry {selected_key!r} must be a list or mapping")
    center = torch.as_tensor(value, dtype=torch.float32)
    if center.shape != (3,) or not bool(torch.isfinite(center).all()):
        raise ValueError(f"pocket center entry {selected_key!r} must have three finite values")
    definition = source_entry_metadata.get("definition")
    reference_ligand_derived = isinstance(definition, str) and definition.startswith(
        "reference_ligand"
    )
    provenance = {
        "source_path": str(source),
        "source_sha256": _file_sha256(source),
        "selected_key": selected_key,
        "value_field": value_field,
        "center_angstrom": center.tolist(),
        "source_entry_metadata": source_entry_metadata,
        "definition": definition,
        "reference_ligand_derived": reference_ligand_derived,
        "derived_from_crystal": (True if reference_ligand_derived else None),
        "derived_at_runtime_from_input_ligand": False,
    }
    return center, provenance


def _identity_quaternions(
    n_fragments: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    result = torch.zeros(n_fragments, 4, device=device, dtype=dtype)
    result[:, 0] = 1.0
    return result


def _fragment_local_coordinates(
    coords: Tensor,
    fragment_id: Tensor,
) -> tuple[Tensor, Tensor]:
    centers = fragment_centers(coords, fragment_id)[0]
    return coords - centers[fragment_id], centers


def _reconstruct_coordinates(
    local_coords: Tensor,
    fragment_id: Tensor,
    translations: Tensor,
    quaternions: Tensor,
) -> Tensor:
    rotation = quaternion_to_matrix(quaternions)
    rotated = torch.einsum(
        "nij,nj->ni",
        rotation[fragment_id],
        local_coords,
    )
    return rotated + translations[fragment_id]


def _clip_vectors(vectors: Tensor, maximum: float) -> Tensor:
    norm = vectors.norm(dim=-1, keepdim=True)
    return vectors * (float(maximum) / norm.clamp_min(1e-12)).clamp(max=1.0)


def _rms_norm(vectors: Tensor) -> float:
    return float(vectors.norm(dim=-1).square().mean().sqrt().detach().cpu())


def _max_norm(vectors: Tensor) -> float:
    return float(vectors.norm(dim=-1).max().detach().cpu())


def _aligned_rmsd(coords: Tensor, reference: Tensor) -> float:
    """Return proper-rotation Kabsch RMSD for diagnostic decomposition only."""
    moving = coords - coords.mean(dim=0, keepdim=True)
    target = reference - reference.mean(dim=0, keepdim=True)
    covariance = moving.transpose(0, 1) @ target
    left, _, right_h = torch.linalg.svd(covariance)
    correction = torch.eye(
        3,
        device=coords.device,
        dtype=coords.dtype,
    )
    correction[-1, -1] = torch.sign(torch.det(left @ right_h))
    rotation = left @ correction @ right_h
    aligned = moving @ rotation
    return float((aligned - target).square().sum(dim=-1).mean().sqrt().detach().cpu())


def _mass_center(coords: Tensor, mass: Tensor) -> Tensor:
    weights = mass.to(device=coords.device, dtype=coords.dtype)
    return (coords * weights.unsqueeze(-1)).sum(dim=0) / weights.sum().clamp_min(1e-12)


def _group_energies(components: dict[str, Tensor]) -> dict[str, float]:
    physical = sum(
        (
            value
            for name, value in components.items()
            if name != "total" and not name.startswith("interaction_")
        ),
        start=components["total"].new_zeros(()),
    )
    interaction = sum(
        (value for name, value in components.items() if name.startswith("interaction_")),
        start=components["total"].new_zeros(()),
    )
    return {
        "physical": float(physical.detach().cpu()),
        "interaction": float(interaction.detach().cpu()),
        "combined": float(components["total"].detach().cpu()),
    }


def _pose_metrics(
    coords: Tensor,
    crystal_coords: Tensor,
    system,
    components: dict[str, Tensor],
    *,
    pocket_center: Tensor | None = None,
    shell_valid_radius_angstrom: float | None = None,
) -> dict[str, Any]:
    topology = system.topology
    raw_rmsd = (coords - crystal_coords).square().sum(dim=-1).mean().sqrt()
    displacement = (coords - crystal_coords).norm(dim=-1)

    if topology.bond_index.shape[1]:
        atom_i, atom_j = topology.bond_index
        current_bond = (coords[atom_i] - coords[atom_j]).norm(dim=-1)
        bond_error = (current_bond - topology.bond_r0).abs()
        cut_bond_mean = float(bond_error.mean().detach().cpu())
        cut_bond_max = float(bond_error.max().detach().cpu())
        cut_bond_rmse = float(bond_error.square().mean().sqrt().detach().cpu())
    else:
        cut_bond_mean = 0.0
        cut_bond_max = 0.0
        cut_bond_rmse = 0.0

    distance = (coords.unsqueeze(1) - system.protein_coords.unsqueeze(0)).norm(dim=-1)
    equilibrium = torch.sqrt(topology.uff_x.view(-1, 1) * system.protein_uff_x.view(1, -1))
    ratio = distance / equilibrium.clamp_min(1e-12)
    ligand_mass_com = _mass_center(coords, topology.mass)
    crystal_mass_com = _mass_center(crystal_coords, topology.mass)
    result: dict[str, Any] = {
        "raw_rmsd_angstrom": float(raw_rmsd.detach().cpu()),
        "aligned_rmsd_angstrom": _aligned_rmsd(coords, crystal_coords),
        "ligand_mass_com_distance_to_crystal_mass_com_angstrom": float(
            (ligand_mass_com - crystal_mass_com).norm().detach().cpu()
        ),
        "maximum_atom_displacement_from_crystal_angstrom": float(displacement.max().detach().cpu()),
        "cut_bond_mean_abs_error_angstrom": cut_bond_mean,
        "cut_bond_max_abs_error_angstrom": cut_bond_max,
        "cut_bond_rmse_angstrom": cut_bond_rmse,
        "minimum_protein_ligand_distance_angstrom": float(distance.min().detach().cpu()),
        "minimum_distance_over_uff_x": float(ratio.min().detach().cpu()),
        "energies": {name: float(value.detach().cpu()) for name, value in components.items()},
        "energy_groups": _group_energies(components),
    }
    if pocket_center is not None:
        center = pocket_center.to(device=coords.device, dtype=coords.dtype)
        maximum_radius = (coords - center.unsqueeze(0)).norm(dim=-1).max()
        result.update(
            {
                "maximum_ligand_atom_radius_from_pocket_center_angstrom": float(
                    maximum_radius.detach().cpu()
                ),
                "ligand_mass_com_distance_to_pocket_center_angstrom": float(
                    (ligand_mass_com - center).norm().detach().cpu()
                ),
                "fixed_shell_valid_radius_angstrom": (
                    float(shell_valid_radius_angstrom)
                    if shell_valid_radius_angstrom is not None
                    else None
                ),
                "within_fixed_shell_valid_radius": (
                    bool(maximum_radius <= float(shell_valid_radius_angstrom) + 1e-8)
                    if shell_valid_radius_angstrom is not None
                    else None
                ),
            }
        )
    return result


def _contact_summary(coords: Tensor, system) -> dict[str, Any]:
    contacts = interaction_contact_stats(
        coords,
        system,
        InteractionEnergyConfig(),
    )
    hydrophobic = contacts["hydrophobic"]
    ligand_to_protein = contacts["hydrogen_bond"]["ligand_donor_to_protein_acceptor"]
    protein_to_ligand = contacts["hydrogen_bond"]["protein_donor_to_ligand_acceptor"]
    formal_charge = contacts["screened_formal_charge"]
    return {
        "hydrophobic_weight_sum": float(hydrophobic["weight_sum"]),
        "hydrophobic_nonzero_pairs": int(hydrophobic["nonzero_pairs"]),
        "hydrogen_bond_weight_sum": float(
            ligand_to_protein["weight_sum"] + protein_to_ligand["weight_sum"]
        ),
        "hydrogen_bond_nonzero_pairs": int(
            ligand_to_protein["nonzero_pairs"] + protein_to_ligand["nonzero_pairs"]
        ),
        "screened_formal_charge_eligibility": formal_charge["eligibility"],
        "screened_formal_charge_active_pairs": int(formal_charge["active_pairs"]),
        "screened_formal_charge_energy_kcal_mol": float(formal_charge["total_energy_kcal_mol"]),
    }


def _mode_interaction_config(mode: str) -> InteractionEnergyConfig:
    if mode == "unified":
        return InteractionEnergyConfig()
    if mode == "physical_only":
        return InteractionEnergyConfig(active_terms=())
    raise ValueError(f"unknown relaxation mode: {mode}")


def relax_rigid_fragments(
    crystal_coords: Tensor,
    initial_coords: Tensor,
    system,
    *,
    config: RigidRelaxationConfig,
    mode: str,
    pocket_center: Tensor | None = None,
) -> RelaxationRun:
    """Run monotone, gradient-only rigid-fragment relaxation."""
    device = system.protein_coords.device
    dtype = system.protein_coords.dtype
    crystal_coords = crystal_coords.to(device=device, dtype=dtype)
    initial_coords = initial_coords.to(device=device, dtype=dtype)
    if config.initialization_mode != "crystal_tear" and pocket_center is None:
        raise ValueError("pocket_center is required for pocket-prior relaxation")
    metric_pocket_center = (
        pocket_center.to(device=device, dtype=dtype)
        if (config.initialization_mode != "crystal_tear" and pocket_center is not None)
        else None
    )
    active_guidance_cutoff = max(
        config.physical_cutoff_angstrom,
        _maximum_active_interaction_cutoff(),
    )
    shell_valid_radius = (
        config.protein_shell_cutoff_angstrom - active_guidance_cutoff
        if metric_pocket_center is not None
        else None
    )
    fragment_id = system.topology.fragment_id
    n_fragments = int(fragment_id.max().item()) + 1
    frag_sizes = torch.bincount(fragment_id, minlength=n_fragments)
    local_coords, translations = _fragment_local_coordinates(
        initial_coords,
        fragment_id,
    )
    quaternions = _identity_quaternions(
        n_fragments,
        device=device,
        dtype=dtype,
    )
    physical_config = PhysicalEnergyConfig(
        cutoff=config.physical_cutoff_angstrom,
    )
    interaction_config = _mode_interaction_config(mode)
    guidance_config = GuidanceEnergyConfig(
        physical=physical_config,
        interaction=interaction_config,
    )

    metrics: list[dict[str, Any]] = []
    frames: list[Tensor] = []
    saved_steps: list[int] = []
    total_backtracks = 0
    status = "max_steps"
    low_displacement_steps = 0
    previous_step: dict[str, Any] = {
        "accepted_alpha": None,
        "backtracks": 0,
        "accepted_max_atom_step_angstrom": None,
        "energy_decrease": None,
    }

    for step in range(config.max_steps + 1):
        work = (
            _reconstruct_coordinates(
                local_coords,
                fragment_id,
                translations,
                quaternions,
            )
            .detach()
            .requires_grad_(True)
        )
        components = guidance_energy(work, system, guidance_config)
        centers = fragment_centers(work.detach(), fragment_id)
        energy_finite = all(bool(torch.isfinite(value)) for value in components.values())
        atom_force: Tensor | None = None
        translation_direction: Tensor | None = None
        angular_direction: Tensor | None = None
        force_finite = False
        if energy_finite:
            gradient = torch.autograd.grad(components["total"], work)[0]
            atom_force = -gradient
            force_finite = bool(torch.isfinite(atom_force).all())
            if force_finite:
                translation_direction, angular_direction = project_atom_forces(
                    atom_force.unsqueeze(0),
                    work.detach().unsqueeze(0),
                    centers,
                    fragment_id,
                    system.topology.mass,
                )
                translation_direction = translation_direction[0]
                angular_direction = angular_direction[0]

        row = {
            "step": step,
            **_pose_metrics(
                work.detach(),
                crystal_coords,
                system,
                components,
                pocket_center=metric_pocket_center,
                shell_valid_radius_angstrom=shell_valid_radius,
            ),
            "force_rms_kcal_mol_angstrom": (_rms_norm(atom_force) if force_finite else None),
            "force_max_kcal_mol_angstrom": (_max_norm(atom_force) if force_finite else None),
            "projected_translation_rms": (
                _rms_norm(translation_direction) if force_finite else None
            ),
            "projected_translation_max": (
                _max_norm(translation_direction) if force_finite else None
            ),
            "projected_rotation_rms": (_rms_norm(angular_direction) if force_finite else None),
            "projected_rotation_max": (_max_norm(angular_direction) if force_finite else None),
            **previous_step,
        }
        metrics.append(row)
        if (
            step % config.save_every == 0
            or step == config.max_steps
            or not energy_finite
            or not force_finite
        ):
            frames.append(work.detach().cpu())
            saved_steps.append(step)
            row["contacts"] = _contact_summary(
                work.detach(),
                system,
            )
            row["fragment_centers_angstrom"] = centers[0].detach().cpu().tolist()
        if not energy_finite:
            status = "nonfinite_energy"
            break
        if not force_finite:
            status = "nonfinite_force"
            break
        if step == config.max_steps:
            status = "max_steps"
            break
        if atom_force is None or translation_direction is None or angular_direction is None:
            raise AssertionError("finite force projection lacks a direction")

        translation_step = _clip_vectors(
            config.base_step_size * translation_direction,
            config.max_translation_step_angstrom,
        )
        angular_step = _clip_vectors(
            config.base_step_size * angular_direction,
            math.radians(config.max_rotation_step_degrees),
        )

        for _ in range(8):
            trial_translation, trial_quaternion = integrate_se3_step(
                translations,
                quaternions,
                translation_step,
                angular_step,
                1.0,
                frag_sizes=frag_sizes,
            )
            trial_coords = _reconstruct_coordinates(
                local_coords,
                fragment_id,
                trial_translation,
                trial_quaternion,
            )
            proposed_atom_step = (trial_coords - work.detach()).norm(dim=-1).max()
            if float(proposed_atom_step) <= config.max_atom_step_angstrom:
                break
            scale = 0.999999 * config.max_atom_step_angstrom / float(proposed_atom_step)
            translation_step = translation_step * scale
            angular_step = angular_step * scale
        else:
            raise RuntimeError("failed to enforce the rigid atom-displacement cap")

        accepted = False
        accepted_alpha = 0.0
        accepted_coords: Tensor | None = None
        accepted_translation: Tensor | None = None
        accepted_quaternion: Tensor | None = None
        accepted_energy: Tensor | None = None
        current_energy = components["total"].detach()
        backtracks = 0
        for backtracks in range(config.max_backtracks + 1):
            alpha = config.backtrack_factor**backtracks
            trial_translation, trial_quaternion = integrate_se3_step(
                translations,
                quaternions,
                translation_step,
                angular_step,
                alpha,
                frag_sizes=frag_sizes,
            )
            trial_coords = _reconstruct_coordinates(
                local_coords,
                fragment_id,
                trial_translation,
                trial_quaternion,
            )
            with torch.no_grad():
                trial_energy = guidance_energy(
                    trial_coords,
                    system,
                    guidance_config,
                )["total"]
            finite = bool(torch.isfinite(trial_energy))
            nonincreasing = bool(trial_energy <= current_energy + config.energy_increase_tolerance)
            if finite and nonincreasing:
                accepted = True
                accepted_alpha = alpha
                accepted_coords = trial_coords.detach()
                accepted_translation = trial_translation.detach()
                accepted_quaternion = trial_quaternion.detach()
                accepted_energy = trial_energy.detach()
                break

        total_backtracks += backtracks
        if not accepted:
            status = "line_search_failed"
            break
        if (
            accepted_coords is None
            or accepted_translation is None
            or accepted_quaternion is None
            or accepted_energy is None
        ):
            raise AssertionError("accepted line-search step lacks state")
        maximum_atom_step = float((accepted_coords - work.detach()).norm(dim=-1).max().cpu())
        if maximum_atom_step > config.max_atom_step_angstrom + 1e-10:
            raise AssertionError("accepted rigid update exceeded the atom-displacement cap")
        energy_decrease = float((current_energy - accepted_energy).cpu())
        translations = accepted_translation
        quaternions = accepted_quaternion
        previous_step = {
            "accepted_alpha": accepted_alpha,
            "backtracks": backtracks,
            "accepted_max_atom_step_angstrom": maximum_atom_step,
            "energy_decrease": energy_decrease,
        }
        if maximum_atom_step < config.convergence_displacement_angstrom:
            low_displacement_steps += 1
        else:
            low_displacement_steps = 0
        if low_displacement_steps >= config.convergence_patience:
            status = "converged_displacement"
            final_coords = _reconstruct_coordinates(
                local_coords,
                fragment_id,
                translations,
                quaternions,
            ).detach()
            if saved_steps[-1] != step + 1:
                with torch.no_grad():
                    final_components = guidance_energy(
                        final_coords,
                        system,
                        guidance_config,
                    )
                final_row = {
                    "step": step + 1,
                    **_pose_metrics(
                        final_coords,
                        crystal_coords,
                        system,
                        final_components,
                        pocket_center=metric_pocket_center,
                        shell_valid_radius_angstrom=shell_valid_radius,
                    ),
                    "force_rms_kcal_mol_angstrom": None,
                    "force_max_kcal_mol_angstrom": None,
                    "projected_translation_rms": None,
                    "projected_translation_max": None,
                    "projected_rotation_rms": None,
                    "projected_rotation_max": None,
                    **previous_step,
                    "contacts": _contact_summary(final_coords, system),
                    "fragment_centers_angstrom": (
                        fragment_centers(
                            final_coords,
                            fragment_id,
                        )[0]
                        .detach()
                        .cpu()
                        .tolist()
                    ),
                }
                metrics.append(final_row)
                frames.append(final_coords.cpu())
                saved_steps.append(step + 1)
            break

    if metrics and saved_steps and saved_steps[-1] != metrics[-1]["step"]:
        final_coords = _reconstruct_coordinates(
            local_coords,
            fragment_id,
            translations,
            quaternions,
        ).detach()
        frames.append(final_coords.cpu())
        saved_steps.append(int(metrics[-1]["step"]))

    if config.initialization_mode == "crystal_tear":
        shell_envelope_valid = all(
            row["maximum_atom_displacement_from_crystal_angstrom"]
            <= config.tear_distance_angstrom + 1e-8
            for row in metrics
        )
    else:
        shell_envelope_valid = all(bool(row["within_fixed_shell_valid_radius"]) for row in metrics)
    return RelaxationRun(
        mode=mode,
        status=status,
        metrics=metrics,
        frames=frames,
        saved_steps=saved_steps,
        total_backtracks=total_backtracks,
        shell_envelope_valid=shell_envelope_valid,
    )


def _success_assessment(
    run: RelaxationRun,
    *,
    crystal_minimum_distance_over_uff_x: float,
    initialization_mode: str,
) -> dict[str, Any]:
    initial = run.metrics[0]
    final = run.metrics[-1]
    initial_rmsd = float(initial["raw_rmsd_angstrom"])
    final_rmsd = float(final["raw_rmsd_angstrom"])
    reduction = 1.0 - final_rmsd / initial_rmsd if initial_rmsd > 0 else 0.0
    pose_threshold = 1.0 if initialization_mode == "crystal_tear" else 2.0
    pose_recovery = (
        final_rmsd <= pose_threshold
        if initialization_mode == "crystal_tear"
        else final_rmsd < pose_threshold
    )
    pose_gate_name = (
        "final_raw_rmsd_le_1_angstrom"
        if initialization_mode == "crystal_tear"
        else "final_raw_rmsd_lt_2_angstrom"
    )
    gates = {
        "finite_and_completed": run.status not in {"nonfinite_energy", "nonfinite_force"},
        "shell_envelope_valid": run.shell_envelope_valid,
        pose_gate_name: pose_recovery,
        "raw_rmsd_reduction_ge_70_percent": reduction >= 0.70,
        "final_cut_bond_max_error_le_0_2_angstrom": (
            float(final["cut_bond_max_abs_error_angstrom"]) <= 0.20
        ),
        "final_minimum_distance_over_uff_x_ge_0_65": (
            float(final["minimum_distance_over_uff_x"]) >= 0.65
        ),
        "final_clash_not_worse_than_crystal_minus_0_05": (
            float(final["minimum_distance_over_uff_x"])
            >= crystal_minimum_distance_over_uff_x - 0.05
        ),
    }
    if initialization_mode == "crystal_tear":
        primary_joint_gates = dict(gates)
        primary_joint_success = all(primary_joint_gates.values())
        if primary_joint_success:
            classification = "success"
        elif (
            final_rmsd >= 2.0
            or reduction < 0.30
            or not gates["finite_and_completed"]
            or not gates["shell_envelope_valid"]
        ):
            classification = "failure"
        else:
            classification = "partial_or_inconclusive"
    else:
        # Keep the frozen global-prior protocol exact: reduction relative to a
        # random t=0 pose and crystal-relative clash are useful diagnostics,
        # but are not part of its pre-registered joint success endpoint.
        primary_joint_gates = {
            "finite": gates["finite_and_completed"],
            "shell_valid": gates["shell_envelope_valid"],
            "final_raw_rmsd_lt_2_angstrom": pose_recovery,
            "final_cut_bond_max_error_le_0_2_angstrom": gates[
                "final_cut_bond_max_error_le_0_2_angstrom"
            ],
            "final_minimum_distance_over_uff_x_ge_0_65": gates[
                "final_minimum_distance_over_uff_x_ge_0_65"
            ],
        }
        primary_joint_success = all(primary_joint_gates.values())
        classification = "success" if primary_joint_success else "failure"
    best = min(
        run.metrics,
        key=lambda row: float(row["raw_rmsd_angstrom"]),
    )
    return {
        "classification": classification,
        "gates": gates,
        "primary_joint_gates": primary_joint_gates,
        "primary_joint_success": primary_joint_success,
        "assembly_gates": {
            "final_cut_bond_max_error_le_0_2_angstrom": gates[
                "final_cut_bond_max_error_le_0_2_angstrom"
            ],
            "final_minimum_distance_over_uff_x_ge_0_65": gates[
                "final_minimum_distance_over_uff_x_ge_0_65"
            ],
        },
        "pose_recovery_threshold_angstrom": pose_threshold,
        "pose_recovery": pose_recovery,
        "initial_raw_rmsd_angstrom": initial_rmsd,
        "final_raw_rmsd_angstrom": final_rmsd,
        "raw_rmsd_reduction_fraction": reduction,
        "best_seen_raw_rmsd_angstrom": float(best["raw_rmsd_angstrom"]),
        "best_seen_step": int(best["step"]),
        "best_seen_is_oracle_diagnostic_only": True,
    }


def _interaction_ablation_assessment(
    unified: RelaxationRun,
    physical_only: RelaxationRun,
    *,
    pose_threshold_angstrom: float,
) -> dict[str, Any]:
    unified_final = float(unified.metrics[-1]["raw_rmsd_angstrom"])
    physical_final = float(physical_only.metrics[-1]["raw_rmsd_angstrom"])

    def first_step_at_threshold(run: RelaxationRun) -> int | None:
        for row in run.metrics:
            if float(row["raw_rmsd_angstrom"]) <= pose_threshold_angstrom:
                return int(row["step"])
        return None

    unified_step = first_step_at_threshold(unified)
    physical_step = first_step_at_threshold(physical_only)
    faster_fraction: float | None = None
    if unified_step is not None and physical_step is not None and physical_step > 0:
        faster_fraction = 1.0 - unified_step / physical_step
    confirmed = physical_final - unified_final >= 0.25 or (
        faster_fraction is not None and faster_fraction >= 0.20
    )
    return {
        "interaction_contribution_confirmed_in_this_demo": confirmed,
        "criterion": (
            "unified final RMSD at least 0.25 A below physical-only, or "
            "at least 20% fewer steps to the initialization-specific pose "
            f"threshold ({pose_threshold_angstrom:g} A)"
        ),
        "unified_minus_physical_final_rmsd_angstrom": (unified_final - physical_final),
        "pose_threshold_angstrom": pose_threshold_angstrom,
        "unified_first_step_at_pose_threshold": unified_step,
        "physical_first_step_at_pose_threshold": physical_step,
        "unified_step_reduction_fraction": faster_fraction,
    }


def _workflow_sha256() -> str:
    return sha256(Path(__file__).read_bytes()).hexdigest()


def _bond_pairs(mol) -> list[list[int]]:
    return [[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()] for bond in mol.GetBonds()]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _run_case(args: argparse.Namespace) -> dict[str, Any]:
    protein = args.protein.resolve()
    ligand = args.ligand.resolve()
    if not protein.is_file():
        raise FileNotFoundError(f"protein PDB not found: {protein}")
    if not ligand.is_file():
        raise FileNotFoundError(f"ligand file not found: {ligand}")
    protocol_file = args.protocol_file.resolve() if args.protocol_file is not None else None
    if protocol_file is not None and not protocol_file.is_file():
        raise FileNotFoundError(f"protocol file not found: {protocol_file}")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol_id = _protocol_id_for_initialization(args.initialization)

    protein_shell_cutoff = args.protein_shell_cutoff
    if protein_shell_cutoff is None:
        protein_shell_cutoff = 13.0 if args.initialization == "crystal_tear" else 18.0
    config = RigidRelaxationConfig(
        initialization_mode=args.initialization,
        tear_distance_angstrom=args.tear_distance,
        prior_sigma_angstrom=args.prior_sigma,
        seed=args.seed,
        max_steps=args.steps,
        save_every=args.save_every,
        base_step_size=args.base_step_size,
        max_translation_step_angstrom=args.max_translation_step,
        max_rotation_step_degrees=args.max_rotation_step_degrees,
        max_atom_step_angstrom=args.max_atom_step,
        max_backtracks=args.max_backtracks,
        convergence_displacement_angstrom=args.convergence_displacement,
        convergence_patience=args.convergence_patience,
        physical_cutoff_angstrom=args.physical_cutoff,
        protein_shell_cutoff_angstrom=protein_shell_cutoff,
    )
    mol, used_mol2_fallback = _load_trace_ligand(ligand)
    ligand_data = featurize_ligand(mol)
    if ligand_data is None:
        raise ValueError("ligand featurization failed")
    crystal_coords = torch.tensor(
        mol.GetConformer().GetPositions(),
        dtype=torch.float64,
    )
    # Match the active inference preprocessor exactly: its sampler local frame
    # is built from the featurizer's float32 coordinates, not the float64
    # diagnostic reference tensor below.
    fragment_data = decompose_fragments(
        mol,
        ligand_data["atom_coords"],
    )
    if fragment_data is None:
        raise ValueError("ligand fragment decomposition failed")
    fragment_id = fragment_data["fragment_id"].to(torch.long)
    provisional_topology = build_physical_topology(
        mol,
        fragment_id,
    ).to(torch.device("cpu"), torch.float64)
    pocket_center: Tensor | None = None
    pocket_center_provenance: dict[str, Any] | None = None
    prior_metadata: dict[str, Any] | None = None
    displacement: Tensor | None = None
    if config.initialization_mode == "crystal_tear":
        initial_coords, displacement = make_torn_fragment_pose(
            crystal_coords,
            fragment_id,
            provisional_topology.mass,
            distance_angstrom=config.tear_distance_angstrom,
            seed=config.seed,
        )
        shell_near_coords = crystal_coords
    else:
        if args.pocket_centers_json is None:
            raise ValueError(
                "--pocket-centers-json is required for "
                f"--initialization {config.initialization_mode}"
            )
        pocket_center, pocket_center_provenance = load_explicit_pocket_center(
            args.pocket_centers_json,
            sample_id=args.sample_id,
            key=args.pocket_center_key,
        )
        rotation_mode = "uniform" if config.initialization_mode == "model_prior" else "identity"
        initial_coords, prior_metadata = make_pocket_prior_fragment_pose(
            fragment_data["frag_local_coords"],
            fragment_id,
            pocket_center,
            sigma_angstrom=config.prior_sigma_angstrom,
            seed=config.seed,
            rotation_mode=rotation_mode,
        )
        shell_near_coords = pocket_center.to(dtype=torch.float64).unsqueeze(0)
    system = build_physical_system(
        mol,
        protein,
        fragment_id=fragment_id,
        near_coords=shell_near_coords,
        protein_cutoff=config.protein_shell_cutoff_angstrom,
    ).to(device=torch.device("cpu"), dtype=torch.float64)

    physical_config = PhysicalEnergyConfig(
        cutoff=config.physical_cutoff_angstrom,
    )
    with torch.no_grad():
        crystal_components = guidance_energy(
            crystal_coords,
            system,
            GuidanceEnergyConfig(
                physical=physical_config,
                interaction=InteractionEnergyConfig(),
            ),
        )
    crystal_metrics = _pose_metrics(
        crystal_coords,
        crystal_coords,
        system,
        crystal_components,
        pocket_center=pocket_center,
        shell_valid_radius_angstrom=(
            config.protein_shell_cutoff_angstrom
            - max(
                config.physical_cutoff_angstrom,
                _maximum_active_interaction_cutoff(),
            )
            if pocket_center is not None
            else None
        ),
    )
    crystal_metrics["contacts"] = _contact_summary(
        crystal_coords,
        system,
    )

    modes = ("unified", "physical_only") if args.mode == "both" else (args.mode,)
    runs: dict[str, RelaxationRun] = {}
    for mode in modes:
        runs[mode] = relax_rigid_fragments(
            crystal_coords,
            initial_coords,
            system,
            config=config,
            mode=mode,
            pocket_center=pocket_center,
        )

    if config.initialization_mode == "crystal_tear":
        if displacement is None:
            raise AssertionError("crystal tear lacks fragment displacement")
        bundle_initialization: dict[str, Any] = {
            "mode": config.initialization_mode,
            "fragment_displacement_angstrom": displacement.cpu(),
            "initial_rotation": "identity",
        }
    else:
        if prior_metadata is None or pocket_center_provenance is None:
            raise AssertionError("pocket prior lacks initialization metadata")
        bundle_initialization = {
            "mode": config.initialization_mode,
            "pocket_center_provenance": pocket_center_provenance,
            **{
                name: (value.cpu() if isinstance(value, Tensor) else value)
                for name, value in prior_metadata.items()
            },
        }

    bundle_path = output_dir / "trajectory.pt"
    torch.save(
        {
            "schema_version": RELAXATION_SCHEMA_VERSION,
            "protocol_id": protocol_id,
            "coordinate_frame": "absolute_pdb_angstrom",
            "crystal_coords": crystal_coords.cpu(),
            "initial_coords": initial_coords.cpu(),
            "pocket_center": (pocket_center.cpu() if pocket_center is not None else None),
            "protein_coords": system.protein_coords.cpu(),
            "protein_atomic_numbers": (system.protein_atomic_numbers.cpu()),
            "fragment_id": fragment_id.cpu(),
            "bonds": torch.tensor(
                _bond_pairs(mol),
                dtype=torch.long,
            ),
            "cut_bonds": system.topology.bond_index.T.cpu(),
            "fragment_displacement_angstrom": (
                displacement.cpu() if displacement is not None else None
            ),
            "initialization": bundle_initialization,
            "protocol_file": (
                {
                    "path": str(protocol_file),
                    "sha256": _file_sha256(protocol_file),
                }
                if protocol_file is not None
                else None
            ),
            "runs": {
                mode: {
                    "frames": torch.stack(run.frames),
                    "saved_steps": torch.tensor(
                        run.saved_steps,
                        dtype=torch.long,
                    ),
                }
                for mode, run in runs.items()
            },
        },
        bundle_path,
    )

    zero_center = torch.zeros(3, dtype=torch.float64)
    artifact_paths: dict[str, Path] = {"trajectory_pt": bundle_path}
    for mode, run in runs.items():
        sdf_path = output_dir / f"{mode}_trajectory.sdf"
        pdb_path = output_dir / f"{mode}_trajectory.pdb"
        write_traj_sdf(
            mol,
            run.frames,
            [float(step) for step in run.saved_steps],
            zero_center,
            sdf_path,
        )
        write_traj_pdb(
            mol,
            run.frames,
            zero_center,
            pdb_path,
        )
        artifact_paths[f"{mode}_trajectory_sdf"] = sdf_path
        artifact_paths[f"{mode}_trajectory_pdb"] = pdb_path

    if config.initialization_mode == "crystal_tear":
        if displacement is None:
            raise AssertionError("crystal tear lacks fragment displacement")
        summary_initialization: dict[str, Any] = {
            "mode": config.initialization_mode,
            "kind": "mass-COM-preserving translation-only fragment tear",
            "fragment_count": int(fragment_id.max().item()) + 1,
            "fragment_sizes": torch.bincount(fragment_id).tolist(),
            "fragment_displacement_angstrom": displacement.cpu().tolist(),
            "maximum_fragment_displacement_angstrom": float(displacement.norm(dim=-1).max()),
            "initial_rotation": "identity for every fragment",
            "crystal_coordinates_used_to_construct_initial_pose": True,
            "result_not_reseeded_after_inspection": True,
        }
        claim_boundary = (
            "Crystal-basin rigid-fragment reassembly under guidance-only "
            "gradient descent; not blind docking, 180-degree pose recovery, "
            "affinity prediction, molecular dynamics, or production ODE."
        )
        shell_construction = (
            "single fixed crystal-correspondence receptor shell; never rebuilt during relaxation"
        )
        shell_validity: dict[str, Any] = {
            "maximum_valid_crystal_atom_envelope_angstrom": (config.tear_distance_angstrom),
        }
        initialization_warnings = [
            "The torn pose preserves each fragment's crystal orientation.",
            "The ligand mass COM is preserved by the initial translation tear.",
        ]
    else:
        if prior_metadata is None or pocket_center_provenance is None:
            raise AssertionError("pocket prior lacks initialization metadata")
        summary_initialization = {
            "mode": config.initialization_mode,
            "kind": (
                "active-sampler t=0 translation and Uniform(SO3) prior"
                if config.initialization_mode == "model_prior"
                else "active-sampler translation prior with identity-rotation control"
            ),
            "fragment_count": int(fragment_id.max().item()) + 1,
            "fragment_sizes": prior_metadata["fragment_sizes"].tolist(),
            "prior_sigma_angstrom": config.prior_sigma_angstrom,
            "pocket_center_provenance": pocket_center_provenance,
            "pocket_center_angstrom": prior_metadata["pocket_center_angstrom"].tolist(),
            "standard_normal_translation_eps": prior_metadata[
                "standard_normal_translation_eps"
            ].tolist(),
            "fragment_translation_relative_to_pocket_angstrom": (
                prior_metadata["translation_relative_to_pocket_angstrom"].tolist()
            ),
            "fragment_translation_absolute_angstrom": prior_metadata[
                "translation_absolute_angstrom"
            ].tolist(),
            "fragment_quaternion_scalar_first": prior_metadata["quaternion_scalar_first"].tolist(),
            "rotation_mode": prior_metadata["rotation_mode"],
            "cpu_float32_rng_draw_order": prior_metadata["cpu_float32_rng_draw_order"],
            "exact_active_sampler_model_prior": prior_metadata["exact_active_sampler_model_prior"],
            "pocket_center_derived_at_runtime_from_input_ligand": False,
            "pocket_center_reference_ligand_derived": (
                pocket_center_provenance["reference_ligand_derived"]
            ),
            "pocket_center_derived_from_crystal": (
                pocket_center_provenance["derived_from_crystal"]
            ),
            "crystal_fragment_local_geometry_used": True,
            "result_not_reseeded_after_inspection": True,
        }
        claim_boundary = (
            "Explicit-pocket guidance-only relaxation from "
            f"{config.initialization_mode}; the model_prior mode exactly "
            "matches the active sampler's CPU float32 t=0 prior draws, but "
            "the learned ODE is not run. Crystal coordinates define local "
            "fragment/reference geometry and evaluation. The explicit frozen "
            "pocket-center provenance is recorded separately and may be "
            "reference-ligand-derived; it is not inferred at runtime. This is "
            "not blind pocket discovery, affinity prediction, molecular "
            "dynamics, or production guided sampling."
        )
        shell_construction = (
            "single fixed explicit-pocket-centered receptor shell; never rebuilt during relaxation"
        )
        shell_validity = {
            "maximum_valid_ligand_atom_radius_from_pocket_center_angstrom": (
                config.protein_shell_cutoff_angstrom
                - max(
                    config.physical_cutoff_angstrom,
                    _maximum_active_interaction_cutoff(),
                )
            ),
        }
        initialization_warnings = [
            "The model_prior run tests guidance alone; the learned ODE is absent.",
            "The pocket_gaussian run is an identity-rotation control, not the exact model prior.",
            "Crystal local fragment geometry is retained while crystal placement is not.",
        ]

    summary: dict[str, Any] = {
        "schema_version": RELAXATION_SCHEMA_VERSION,
        "protocol_id": protocol_id,
        "protocol_file": (
            {
                "path": str(protocol_file),
                "sha256": _file_sha256(protocol_file),
            }
            if protocol_file is not None
            else None
        ),
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "diagnostic_only",
        "claim_boundary": claim_boundary,
        "sample_id": args.sample_id,
        "coordinate_frame": "absolute_pdb_angstrom",
        "inputs": {
            "protein": str(protein),
            "protein_sha256": _file_sha256(protein),
            "ligand": str(ligand),
            "ligand_sha256": _file_sha256(ligand),
            "ligand_used_mol2_fallback": bool(used_mol2_fallback),
            "crystal_coordinates_are_diagnostic_only": True,
            "input_reference_geometry": (
                "the supplied crystal SDF defines cut-bond lengths, "
                "cross-fragment angles, and chiral references"
            ),
        },
        "implementation": {
            "active_guidance": _implementation_identity(),
            "workflow_sha256": _workflow_sha256(),
            "torch": torch.__version__,
        },
        "parameter_set": guidance_parameter_identity(),
        "config": asdict(config),
        "initialization": summary_initialization,
        "shell_contract": {
            "construction": shell_construction,
            "active_physical_cutoff_angstrom": (config.physical_cutoff_angstrom),
            "active_interaction_cutoff_angstrom": (_maximum_active_interaction_cutoff()),
            **shell_validity,
            "protein_shell_cutoff_angstrom": (config.protein_shell_cutoff_angstrom),
            "protein_shell_heavy_atoms": int(system.protein_coords.shape[0]),
            "excluded_nonprotein_atoms": (system.excluded_nonprotein_atoms),
            "excluded_nonprotein_residues": list(system.excluded_nonprotein_residues),
        },
        "crystal_reference": crystal_metrics,
        "runs": {},
        "warnings": [
            "Energy descent alone is not a success criterion.",
            *initialization_warnings,
            "Raw atom-index RMSD is evaluation-only and never enters optimization.",
            (
                "Partial-charge electrostatics, solvation, metal coordination, "
                "and receptor flexibility are absent; screened formal-charge "
                "groups are active in InteractionGuidance."
            ),
            "Vina is not imported, evaluated, or combined.",
        ],
    }
    for mode, run in runs.items():
        summary["runs"][mode] = {
            "status": run.status,
            "total_backtracks": run.total_backtracks,
            "accepted_steps": int(run.metrics[-1]["step"]),
            "saved_frames": len(run.frames),
            "shell_envelope_valid": run.shell_envelope_valid,
            "assessment": _success_assessment(
                run,
                crystal_minimum_distance_over_uff_x=float(
                    crystal_metrics["minimum_distance_over_uff_x"]
                ),
                initialization_mode=config.initialization_mode,
            ),
            "metrics": run.metrics,
        }
    if set(runs) == {"unified", "physical_only"}:
        summary["interaction_ablation"] = _interaction_ablation_assessment(
            runs["unified"],
            runs["physical_only"],
            pose_threshold_angstrom=(1.0 if config.initialization_mode == "crystal_tear" else 2.0),
        )

    summary["artifacts"] = {
        name: {
            "path": str(path),
            "sha256": _file_sha256(path),
        }
        for name, path in artifact_paths.items()
    }
    summary_path = output_dir / "summary.json"
    _write_json(summary_path, summary)
    print(
        json.dumps(
            {
                "summary": str(summary_path),
                "runs": {
                    mode: {
                        "status": run.status,
                        "initial_rmsd": run.metrics[0]["raw_rmsd_angstrom"],
                        "final_rmsd": run.metrics[-1]["raw_rmsd_angstrom"],
                        "final_energy": run.metrics[-1]["energy_groups"]["combined"],
                        "shell_envelope_valid": (run.shell_envelope_valid),
                    }
                    for mode, run in runs.items()
                },
            },
            indent=2,
        )
    )
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Initialize rigid ligand fragments from a crystal tear or an "
            "explicit-pocket model prior, then relax them using only the "
            "self-contained EFF-Dock guidance gradient."
        )
    )
    parser.add_argument("--protein", type=Path, required=True)
    parser.add_argument("--ligand", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument(
        "--initialization",
        choices=INITIALIZATION_MODES,
        default="crystal_tear",
    )
    parser.add_argument(
        "--pocket-centers-json",
        type=Path,
        help=(
            "Required for pocket_gaussian/model_prior; crystal coordinates "
            "are never used to derive this center."
        ),
    )
    parser.add_argument(
        "--pocket-center-key",
        help=(
            "Mapping key in --pocket-centers-json; defaults to the lowercase "
            "sample-id prefix before the first underscore."
        ),
    )
    parser.add_argument("--prior-sigma", type=float, default=0.5)
    parser.add_argument(
        "--protocol-file",
        type=Path,
        help="Optional frozen experiment protocol recorded by path and SHA256.",
    )
    parser.add_argument(
        "--mode",
        choices=("both", "unified", "physical_only"),
        default="both",
    )
    parser.add_argument("--tear-distance", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--base-step-size", type=float, default=1.0)
    parser.add_argument(
        "--max-translation-step",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--max-rotation-step-degrees",
        type=float,
        default=5.0,
    )
    parser.add_argument("--max-atom-step", type=float, default=0.10)
    parser.add_argument("--max-backtracks", type=int, default=12)
    parser.add_argument(
        "--convergence-displacement",
        type=float,
        default=1e-5,
    )
    parser.add_argument(
        "--convergence-patience",
        type=int,
        default=20,
    )
    parser.add_argument("--physical-cutoff", type=float, default=8.0)
    parser.add_argument(
        "--protein-shell-cutoff",
        type=float,
        default=None,
        help=(
            "Defaults to 11 A for crystal_tear and 18 A for explicit-pocket "
            "priors; pocket-prior values below 18 A are rejected."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    try:
        _run_case(args)
    except UnsupportedPhysicalChemistryError as exc:
        output_dir = args.output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        failure = {
            "schema_version": RELAXATION_SCHEMA_VERSION,
            "protocol_id": _protocol_id_for_initialization(args.initialization),
            "created_utc": datetime.now(UTC).isoformat(),
            "status": "unsupported",
            "sample_id": args.sample_id,
            "failure": exc.as_dict(),
        }
        _write_json(output_dir / "unsupported.json", failure)
        raise


if __name__ == "__main__":
    main()
