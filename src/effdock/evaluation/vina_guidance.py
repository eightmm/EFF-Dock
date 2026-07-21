"""Inference-time differentiable Vina+DG guidance for EFF-Dock sampling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from .pose_scoring import build_protein_vina_inputs
from .vina import (
    ligand_dg_reference,
    vina_atom_radii,
    vina_atom_types,
    vina_score_with_strain_batched,
)

__all__ = ["VinaGuidance", "VinaGuidanceConfig", "build_vina_guidance"]


@dataclass(frozen=True)
class VinaGuidanceConfig:
    """Numerical controls for the physical correction vector field."""

    start_t: float = 0.5
    ramp_power: float = 1.0
    max_atom_force: float = 10.0
    max_translation_velocity: float = 5.0
    max_angular_velocity: float = 5.0
    w_strain: float = 1.0
    clash_scale: float = 0.75

    def __post_init__(self) -> None:
        if not 0.0 <= self.start_t < 1.0:
            raise ValueError(f"start_t must be in [0,1), got {self.start_t}")
        for name in (
            "ramp_power",
            "max_atom_force",
            "max_translation_velocity",
            "max_angular_velocity",
        ):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if self.w_strain < 0.0:
            raise ValueError("w_strain must be non-negative")


def _clip_vectors(vectors: Tensor, max_norm: float, eps: float = 1e-8) -> Tensor:
    """Rotation-equivariant per-vector norm clipping."""
    norm = vectors.norm(dim=-1, keepdim=True)
    scale = (float(max_norm) / norm.clamp_min(eps)).clamp(max=1.0)
    return vectors * scale


class VinaGuidance:
    """Callable that maps current atom positions to fragment Vina velocities.

    The receptor and chemistry tensors are static. Coordinates passed by the
    sampler are pocket-centered and shaped as a flattened batch of identical
    ligands. The callback differentiates one dense batched Vina+DG objective,
    then aggregates ``-dE/dx`` into fragment translation and angular velocity.
    """

    def __init__(
        self,
        *,
        prot_coords: Tensor,
        prot_radii: Tensor,
        prot_is_hydrophobic: Tensor,
        prot_is_donor: Tensor,
        prot_is_acceptor: Tensor,
        lig_radii: Tensor,
        lig_is_hydrophobic: Tensor,
        lig_is_donor: Tensor,
        lig_is_acceptor: Tensor,
        num_rotatable_bonds: int,
        bond_index: Tensor,
        bond_ref_len: Tensor,
        frag_id: Tensor,
        config: VinaGuidanceConfig = VinaGuidanceConfig(),
    ) -> None:
        self.config = config
        tensors = {
            "prot_coords": prot_coords,
            "prot_radii": prot_radii,
            "prot_is_hydrophobic": prot_is_hydrophobic,
            "prot_is_donor": prot_is_donor,
            "prot_is_acceptor": prot_is_acceptor,
            "lig_radii": lig_radii,
            "lig_is_hydrophobic": lig_is_hydrophobic,
            "lig_is_donor": lig_is_donor,
            "lig_is_acceptor": lig_is_acceptor,
            "bond_index": bond_index,
            "bond_ref_len": bond_ref_len,
            "frag_id": frag_id,
        }
        device = prot_coords.device
        for name, value in tensors.items():
            setattr(self, name, value.to(device))
        self.num_rotatable_bonds = int(num_rotatable_bonds)
        self.n_atoms = int(lig_radii.numel())
        self.n_frags = int(frag_id.max().item()) + 1
        if self.n_atoms <= 0 or self.n_frags <= 0:
            raise ValueError("guidance requires at least one ligand atom and fragment")
        if int(frag_id.numel()) != self.n_atoms:
            raise ValueError("frag_id length must match ligand atoms")
        if int(prot_coords.shape[0]) == 0:
            raise ValueError("guidance receptor shell is empty")

    def _ramp(self, t: float) -> float:
        if t < self.config.start_t:
            return 0.0
        u = (float(t) - self.config.start_t) / (1.0 - self.config.start_t)
        return max(0.0, min(1.0, u)) ** self.config.ramp_power

    def _aggregate(self, atom_force: Tensor, coords: Tensor, centers: Tensor) -> tuple[Tensor, Tensor]:
        """Aggregate atom forces using mean force and isotropic fragment torque."""
        batch_size = coords.shape[0]
        frag_id = self.frag_id.long()
        counts = torch.bincount(frag_id, minlength=self.n_frags).to(coords.dtype).clamp_min(1.0)
        translation = torch.zeros(
            batch_size, self.n_frags, 3, device=coords.device, dtype=coords.dtype
        )
        translation.index_add_(1, frag_id, atom_force)
        translation = translation / counts[None, :, None]

        lever = coords - centers[:, frag_id]
        atom_torque = torch.linalg.cross(lever, atom_force, dim=-1)
        torque = torch.zeros_like(translation)
        torque.index_add_(1, frag_id, atom_torque)
        moment = torch.zeros(batch_size, self.n_frags, device=coords.device, dtype=coords.dtype)
        moment.index_add_(1, frag_id, lever.square().sum(dim=-1))
        angular = torque / moment.clamp_min(1e-6).unsqueeze(-1)
        angular = torch.where((moment > 1e-6).unsqueeze(-1), angular, torch.zeros_like(angular))
        return translation, angular

    def __call__(
        self,
        atom_pos_flat: Tensor,
        frag_id_flat: Tensor,
        centers_flat: Tensor,
        t: float,
    ) -> tuple[Tensor, Tensor]:
        del frag_id_flat  # topology is static and validated at construction
        if atom_pos_flat.shape[0] % self.n_atoms:
            raise ValueError("flattened atom positions do not contain whole ligand poses")
        batch_size = atom_pos_flat.shape[0] // self.n_atoms
        if centers_flat.shape[0] != batch_size * self.n_frags:
            raise ValueError("fragment-center batch does not match ligand pose batch")
        ramp = self._ramp(t)
        if ramp == 0.0:
            zero = torch.zeros_like(centers_flat)
            return zero, zero

        with torch.enable_grad():
            coords = atom_pos_flat.detach().view(batch_size, self.n_atoms, 3).requires_grad_(True)
            objective = vina_score_with_strain_batched(
                coords,
                self.prot_coords,
                self.lig_radii,
                self.prot_radii,
                lig_is_hydrophobic=self.lig_is_hydrophobic,
                prot_is_hydrophobic=self.prot_is_hydrophobic,
                lig_is_donor=self.lig_is_donor,
                prot_is_donor=self.prot_is_donor,
                lig_is_acceptor=self.lig_is_acceptor,
                prot_is_acceptor=self.prot_is_acceptor,
                bond_index=self.bond_index,
                bond_ref_len=self.bond_ref_len,
                num_rotatable_bonds=self.num_rotatable_bonds,
                frag_id=self.frag_id,
                w_strain=self.config.w_strain,
                clash_scale=self.config.clash_scale,
            )
            atom_force = -torch.autograd.grad(objective.sum(), coords, create_graph=False)[0]

        if not torch.isfinite(atom_force).all():
            raise FloatingPointError("non-finite Vina guidance gradient")
        atom_force = _clip_vectors(atom_force, self.config.max_atom_force)
        centers = centers_flat.detach().view(batch_size, self.n_frags, 3)
        translation, angular = self._aggregate(atom_force, coords.detach(), centers)
        translation = _clip_vectors(translation, self.config.max_translation_velocity)
        angular = _clip_vectors(angular, self.config.max_angular_velocity)
        translation = (ramp * translation).reshape(batch_size * self.n_frags, 3)
        angular = (ramp * angular).reshape(batch_size * self.n_frags, 3)
        return translation, angular


def build_vina_guidance(
    mol,
    protein_pdb: str | Path,
    *,
    pocket_center: Tensor,
    frag_id: Tensor,
    device: torch.device,
    protein_shell_cutoff: float = 18.0,
    config: VinaGuidanceConfig = VinaGuidanceConfig(),
) -> VinaGuidance:
    """Build the active PDB/RDKit-typed guidance callback for one complex.

    The exact score kernel is shared with the XS-typed public route. This
    convenience factory uses the project's declared best-effort PDB/RDKit atom
    typing because benchmark receptors are currently stored as PDB, not PDBQT.
    """
    center_cpu = pocket_center.detach().cpu().to(torch.float32)
    prot = build_protein_vina_inputs(
        protein_pdb,
        center_cpu.view(1, 3),
        cutoff=float(protein_shell_cutoff),
    )
    lig = vina_atom_types(mol)
    ref = ligand_dg_reference(mol)
    return VinaGuidance(
        prot_coords=prot["coords"].to(device) - center_cpu.to(device),
        prot_radii=prot["radii"].to(device),
        prot_is_hydrophobic=prot["is_hydrophobic"].to(device),
        prot_is_donor=prot["is_donor"].to(device),
        prot_is_acceptor=prot["is_acceptor"].to(device),
        lig_radii=vina_atom_radii(lig["atomic_nums"], device=device),
        lig_is_hydrophobic=lig["is_hydrophobic"].to(device),
        lig_is_donor=lig["is_donor"].to(device),
        lig_is_acceptor=lig["is_acceptor"].to(device),
        num_rotatable_bonds=int(lig["num_rotatable_bonds"]),
        bond_index=ref["bond_index"].to(device),
        bond_ref_len=ref["bond_ref_len"].to(device),
        frag_id=frag_id.to(device),
        config=config,
    )
