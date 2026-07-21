"""Dataset utilities for ligand-pose confidence training.

The generated confidence shards are intentionally ligand-only.  Each item here
reloads the matching ``protein.pt`` and builds protein contact context around
the sampled ligand poses at training time.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from effdock.data.dataset import (
    _crop_protein_by_atom_mask,
    crop_to_nearest_residues,
    crop_to_pocket,
)
from effdock.preprocess.graph import build_static_complex_graph

PROTEIN_FLAG_KEYS = [
    "patom_is_backbone",
    "patom_is_metal",
    "patom_is_donor",
    "patom_is_acceptor",
    "patom_is_positive",
    "patom_is_negative",
    "patom_is_hydrophobic",
]

DEFAULT_CONFIDENCE_POSE_TAG = "conf_ligonly_v1"

POSE_TENSOR_KEYS = {
    "pose_atom_coords",
    "h_lig_node",
    "atom_disp",
    "pose_rmsd",
}


class LigandPoseConfidenceDataset(Dataset):
    """One item = one processed complex with K sampled ligand poses."""

    def __init__(
        self,
        *,
        split_file: Path,
        split: str,
        processed_dir: Path,
        pose_tag: str | None = None,
        tag: str | None = None,
        protein_crop_mode: str = "center",
        protein_contact_cutoff: float = 5.0,
        protein_crop_cutoff: float = 8.0,
        protein_crop_cutoff_min: float | None = 6.0,
        protein_crop_cutoff_max: float | None = 12.0,
        protein_crop_jitter_sigma: float = 2.0,
        protein_crop_jitter_max: float = 4.0,
        stochastic_crop: bool | None = None,
        max_protein_atoms: int = 4096,
        max_poses_per_complex: int | None = None,
        pose_sample_strategy: str = "best_random",
        limit: int | None = None,
        start: int = 0,
    ) -> None:
        with split_file.open() as handle:
            split_map = json.load(handle)
        pids = list(split_map[split])
        if start:
            pids = pids[start:]
        if limit is not None:
            pids = pids[:limit]

        self.pids = pids
        self.split = split
        self.processed_dir = Path(processed_dir)
        self.pose_tag = pose_tag or tag or DEFAULT_CONFIDENCE_POSE_TAG
        self.protein_crop_mode = protein_crop_mode
        self.protein_contact_cutoff = float(protein_contact_cutoff)
        self.protein_crop_cutoff = float(protein_crop_cutoff)
        self.protein_crop_cutoff_min = protein_crop_cutoff_min
        self.protein_crop_cutoff_max = protein_crop_cutoff_max
        self.protein_crop_jitter_sigma = float(protein_crop_jitter_sigma)
        self.protein_crop_jitter_max = float(protein_crop_jitter_max)
        self.stochastic_crop = (
            (split == "train") if stochastic_crop is None else bool(stochastic_crop)
        )
        self.max_protein_atoms = int(max_protein_atoms)
        self.max_poses_per_complex = (
            None if max_poses_per_complex is None else int(max_poses_per_complex)
        )
        self.pose_sample_strategy = str(pose_sample_strategy)

    def __len__(self) -> int:
        return len(self.pids)

    def _shard_path(self, pid: str) -> Path:
        return self.processed_dir / pid / "confidence_poses" / f"confposes_{self.pose_tag}.pt"

    def _sample_center_crop(self) -> tuple[float, torch.Tensor]:
        cutoff = self.protein_crop_cutoff
        center_offset = torch.zeros(3, dtype=torch.float32)
        if self.stochastic_crop:
            lo = self.protein_crop_cutoff_min
            hi = self.protein_crop_cutoff_max
            if lo is not None and hi is not None and hi > lo:
                cutoff = float(
                    torch.empty((), dtype=torch.float32).uniform_(float(lo), float(hi)).item()
                )
            if self.protein_crop_jitter_sigma > 0:
                center_offset = torch.randn(3, dtype=torch.float32) * self.protein_crop_jitter_sigma
                if self.protein_crop_jitter_max > 0:
                    norm = center_offset.norm().clamp_min(1e-6)
                    scale = min(1.0, self.protein_crop_jitter_max / float(norm.item()))
                    center_offset = center_offset * scale
        return float(cutoff), center_offset

    def _cap_protein_atoms(
        self,
        prot_data: dict[str, torch.Tensor],
        ref_coords: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        max_atoms = self.max_protein_atoms
        if max_atoms <= 0 or prot_data["patom_coords"].shape[0] <= max_atoms:
            return prot_data
        if ref_coords.ndim == 1:
            ref_coords = ref_coords.unsqueeze(0)

        patom_coords = prot_data["patom_coords"]
        patom_residue_id = prot_data["patom_residue_id"]
        ref_coords = ref_coords.to(device=patom_coords.device, dtype=patom_coords.dtype)
        atom_dist = torch.cdist(patom_coords, ref_coords).amin(dim=1)
        n_res = int(patom_residue_id.max().item()) + 1
        res_dist = torch.full((n_res,), float("inf"), dtype=atom_dist.dtype)
        res_dist.scatter_reduce_(0, patom_residue_id, atom_dist, reduce="amin", include_self=True)
        res_order = torch.argsort(res_dist)

        keep = torch.zeros_like(patom_residue_id, dtype=torch.bool)
        kept_atoms = 0
        for res_id in res_order.tolist():
            res_mask = patom_residue_id == int(res_id)
            n_atoms = int(res_mask.sum().item())
            if n_atoms == 0:
                continue
            if kept_atoms > 0 and kept_atoms + n_atoms > max_atoms:
                continue
            keep |= res_mask
            kept_atoms += n_atoms
            if kept_atoms >= max_atoms:
                break

        capped = _crop_protein_by_atom_mask(prot_data, keep)
        return prot_data if capped is None else capped

    def _sample_pose_indices(self, shard: dict[str, torch.Tensor]) -> torch.Tensor | None:
        max_poses = self.max_poses_per_complex
        if max_poses is None or max_poses <= 0:
            return None
        n_poses = int(shard["pose_atom_coords"].shape[0])
        if n_poses <= max_poses:
            return None

        if self.split == "train":
            if self.pose_sample_strategy == "stratified":
                idx = self._sample_pose_indices_stratified(shard, max_poses)
            else:
                best = int(shard["pose_rmsd"].argmin().item())
                all_idx = torch.arange(n_poses, dtype=torch.long)
                pool = all_idx[all_idx != best]
                extra = pool[torch.randperm(pool.numel())[: max_poses - 1]]
                idx = torch.cat([torch.tensor([best], dtype=torch.long), extra])
            return idx[torch.randperm(idx.numel())]

        return torch.arange(max_poses, dtype=torch.long)

    @staticmethod
    def _take_random(candidates: torch.Tensor, n: int) -> torch.Tensor:
        if n <= 0 or candidates.numel() == 0:
            return candidates.new_empty((0,))
        order = torch.randperm(candidates.numel())[: min(n, candidates.numel())]
        return candidates.index_select(0, order)

    def _sample_pose_indices_stratified(
        self, shard: dict[str, torch.Tensor], max_poses: int
    ) -> torch.Tensor:
        rmsd = shard["pose_rmsd"].to(torch.float32)
        n_poses = int(rmsd.numel())
        all_idx = torch.arange(n_poses, dtype=torch.long)
        chosen: list[torch.Tensor] = [torch.tensor([int(rmsd.argmin().item())], dtype=torch.long)]

        source = shard.get("pose_source_type")
        if source is not None:
            source = source.to(torch.long)
            crystal = all_idx[source == 1]
            near_native = all_idx[(source >= 2) & (source <= 4)]
            hard_partial = all_idx[source >= 5]
            if crystal.numel() > 0 and not bool((crystal == chosen[0][0]).any()):
                chosen.append(self._take_random(crystal, 1))
            chosen.append(self._take_random(near_native, 4))
            chosen.append(self._take_random(hard_partial, 5))

        chosen.append(self._take_random(all_idx[rmsd < 1.0], 3))
        chosen.append(self._take_random(all_idx[(rmsd >= 1.0) & (rmsd < 2.0)], 5))
        chosen.append(self._take_random(all_idx[(rmsd >= 2.0) & (rmsd < 3.0)], 5))
        chosen.append(self._take_random(all_idx[(rmsd >= 3.0) & (rmsd < 5.0)], 4))
        chosen.append(self._take_random(all_idx[rmsd >= 5.0], 4))

        idx = torch.unique(torch.cat([x for x in chosen if x.numel() > 0], dim=0), sorted=False)
        if idx.numel() < max_poses:
            used = torch.zeros(n_poses, dtype=torch.bool)
            used[idx] = True
            pool = all_idx[~used]
            idx = torch.cat([idx, self._take_random(pool, max_poses - idx.numel())], dim=0)
        if idx.numel() > max_poses:
            idx = idx[torch.randperm(idx.numel())[:max_poses]]
        return idx

    @staticmethod
    def _pose_value(shard: dict[str, Any], key: str, pose_idx: torch.Tensor | None) -> torch.Tensor:
        value = shard[key]
        if pose_idx is not None and key in POSE_TENSOR_KEYS:
            value = value.index_select(0, pose_idx)
        return value

    def _load_docking_graph_context(
        self,
        protein_path: Path,
        ligand_path: Path,
        pocket_center: torch.Tensor,
        pose_atom_coords: torch.Tensor | None = None,
        ligand_coords: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        prot = torch.load(protein_path, map_location="cpu", weights_only=True)
        lig = torch.load(ligand_path, map_location="cpu", weights_only=True)
        pocket_center = pocket_center.to(torch.float32)

        if self.protein_crop_mode == "center":
            cutoff, center_offset = self._sample_center_crop()
            ref = pocket_center + center_offset
        elif (
            self.protein_crop_mode == "pose_residue"
            and pose_atom_coords is not None
            and pose_atom_coords.numel() > 0
        ):
            ref = pose_atom_coords.to(torch.float32).reshape(-1, 3) + pocket_center
            cutoff = self.protein_contact_cutoff
        elif (
            self.protein_crop_mode == "ligand_residue"
            and ligand_coords is not None
            and ligand_coords.numel() > 0
        ):
            ref = ligand_coords.to(torch.float32) + pocket_center
            cutoff = self.protein_contact_cutoff
        else:
            cutoff, center_offset = self._sample_center_crop()
            ref = pocket_center + center_offset

        cropped = crop_to_pocket(prot, ref, cutoff=cutoff)
        if cropped is None:
            cropped = crop_to_pocket(prot, pocket_center, cutoff=self.protein_crop_cutoff + 5.0)
        if cropped is None:
            cropped = crop_to_nearest_residues(prot, pocket_center, max_residues=32)
        if cropped is None:
            raise RuntimeError(f"No protein residues found for graph crop: {protein_path}")
        cropped = self._cap_protein_atoms(cropped, ref if ref.ndim == 2 else ref.unsqueeze(0))

        graph = build_static_complex_graph(lig, cropped)
        graph = {k: v for k, v in graph.items() if torch.is_tensor(v)}
        graph["node_coords"] = graph["node_coords"].to(torch.float32) - pocket_center
        return graph

    def __getitem__(self, index: int) -> dict[str, Any]:
        pid = self.pids[index]
        shard_path = self._shard_path(pid)
        shard = torch.load(shard_path, map_location="cpu", weights_only=True)
        pose_idx = self._sample_pose_indices(shard)
        pocket_center = shard["pocket_center_used"].to(torch.float32)
        protein_path = Path(shard.get("protein_pt", self.processed_dir / pid / "protein.pt"))
        pose_atom_coords = self._pose_value(shard, "pose_atom_coords", pose_idx).to(torch.float32)

        item = {
            "pid": pid,
            "split": self.split,
            "pose_atom_coords": pose_atom_coords,
            "h_lig_node": self._pose_value(shard, "h_lig_node", pose_idx).to(torch.float32),
            "lig_node_type": shard["lig_node_type"].to(torch.long),
            "fragment_id": shard["fragment_id"].to(torch.long),
            "frag_sizes": shard["frag_sizes"].to(torch.long),
            "atom_disp": self._pose_value(shard, "atom_disp", pose_idx).to(torch.float32),
            "pose_rmsd": self._pose_value(shard, "pose_rmsd", pose_idx).to(torch.float32),
            "pocket_center_used": pocket_center,
            "shard_path": str(shard_path),
        }
        ligand_path = Path(shard.get("ligand_pt", self.processed_dir / pid / "ligand.pt"))
        item["graph"] = self._load_docking_graph_context(
            protein_path,
            ligand_path,
            pocket_center,
            pose_atom_coords,
            shard.get("lig_atom_coords_crystal_centered"),
        )
        return item


def collate_complexes(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep variable-size complexes as a list."""
    return items


def to_device(item: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in item.items():
        if torch.is_tensor(value):
            out[key] = value.to(device, non_blocking=True)
        else:
            out[key] = value
    return out


__all__ = [
    "LigandPoseConfidenceDataset",
    "DEFAULT_CONFIDENCE_POSE_TAG",
    "PROTEIN_FLAG_KEYS",
    "collate_complexes",
    "to_device",
]
