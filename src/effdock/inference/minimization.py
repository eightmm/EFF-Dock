"""Fixed-receptor minimization shared by docking and saved-pose inference."""

from __future__ import annotations

import math
from pathlib import Path

import torch
from rdkit import Chem


@torch.inference_mode(False)
@torch.enable_grad()
def minimize_poses(
    mol: Chem.Mol,
    protein: Path,
    poses: torch.Tensor,
    pocket_center: torch.Tensor,
    fragment_id: torch.Tensor,
    device: torch.device,
    *,
    steps: int = 100,
    batch_size: int = 10,
) -> tuple[list[torch.Tensor], list[dict]]:
    """Return minimized pocket-centered poses and their per-pose diagnostics."""
    from effdock.guidance import InteractionEnergyConfig, build_physical_system
    from effdock.workflows.relax_guidance import RigidRelaxationConfig, relax_rigid_fragments_batch

    if steps < 1 or batch_size < 1:
        raise ValueError("Steps and batch size must be positive")
    config = RigidRelaxationConfig(
        initialization_mode="model_prior",
        max_steps=steps,
        save_every=steps,
        physical_cutoff_angstrom=8.0,
        protein_shell_cutoff_angstrom=18.0,
        convergence_energy_absolute_kcal_mol=0.02,
        convergence_energy_relative=0.001,
        convergence_energy_min_steps=25,
    )
    center = pocket_center.detach().cpu().clone()
    system = build_physical_system(
        mol, protein, fragment_id=fragment_id.detach().cpu().clone(),
        near_coords=center.view(1, 3), coordinate_origin=center,
        protein_cutoff=config.protein_shell_cutoff_angstrom, receptor_policy="geometry_only",
    ).to(device=device, dtype=torch.float32)
    interaction = InteractionEnergyConfig()
    origin = torch.zeros(3, device=device)
    minimized, diagnostics = [], []
    for batch in poses.detach().split(batch_size):
        batch = batch.clone()
        # The first pose anchors diagnostic displacements for the batch.
        result = relax_rigid_fragments_batch(
            batch[0], batch, system, config=config, mode="unified", pocket_center=origin,
            interaction_config=interaction, collect_every_step_metrics=False,
            collect_contact_stats=False,
        )
        for coords, trace, status, iterations, in_shell in zip(
            result.frames[-1], result.metrics, result.statuses, result.terminal_steps,
            result.shell_envelope_valid, strict=True,
        ):
            before = trace[0]["energy_groups"]["combined"]
            after = trace[-1]["energy_groups"]["combined"]
            index = len(minimized)
            if (
                status.startswith("nonfinite") or not bool(torch.isfinite(coords).all())
                or not math.isfinite(before) or not math.isfinite(after)
            ):
                raise ValueError(f"Non-finite minimization for pose {index + 1}: {status}")
            minimized.append(coords)
            diagnostics.append(dict(
                minimization_input_index=index,
                minimization_status=status,
                minimization_steps=iterations,
                minimization_initial_energy=before,
                minimization_final_energy=after,
                minimization_shell_envelope_valid=int(in_shell),
            ))
    return minimized, diagnostics
