"""Dataset utilities for ligand-pose confidence training.

The generated confidence shards are intentionally ligand-only.  Each item here
reloads the matching ``protein.pt`` and builds protein contact context around
the sampled ligand poses at training time.
"""

from __future__ import annotations

import hashlib
import io
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
from effdock.preprocess.graph_types import NTYPE_FRAGMENT, NTYPE_LIG_ATOM

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
    "pose_rmsd_symmetry_no_align",
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
        max_pose_node_product: int | None = None,
        large_graph_node_threshold: int | None = None,
        large_graph_max_poses: int | None = None,
        pose_sample_strategy: str = "best_random",
        pose_target_key: str = "pose_rmsd",
        eval_target_key: str | None = None,
        external_pose_targets: dict[str, dict[str, Any]] | None = None,
        shard_paths: dict[str, Path] | None = None,
        system_ids: dict[str, str] | None = None,
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
        self.max_pose_node_product = (
            None if max_pose_node_product is None else int(max_pose_node_product)
        )
        if self.max_pose_node_product is not None and self.max_pose_node_product < 1:
            raise ValueError("max_pose_node_product must be positive when provided")
        self.large_graph_node_threshold = (
            None if large_graph_node_threshold is None else int(large_graph_node_threshold)
        )
        self.large_graph_max_poses = (
            None if large_graph_max_poses is None else int(large_graph_max_poses)
        )
        if (self.large_graph_node_threshold is None) != (self.large_graph_max_poses is None):
            raise ValueError(
                "large_graph_node_threshold and large_graph_max_poses must be set together"
            )
        if self.large_graph_node_threshold is not None and self.large_graph_node_threshold < 1:
            raise ValueError("large_graph_node_threshold must be positive")
        if self.large_graph_max_poses is not None and self.large_graph_max_poses < 1:
            raise ValueError("large_graph_max_poses must be positive")
        self.pose_sample_strategy = str(pose_sample_strategy)
        self.pose_target_key = str(pose_target_key)
        if not self.pose_target_key:
            raise ValueError("pose_target_key must be non-empty")
        self.eval_target_key = eval_target_key
        self.external_pose_targets = external_pose_targets
        self.shard_paths = (
            None
            if shard_paths is None
            else {str(pid): Path(path) for pid, path in shard_paths.items()}
        )
        self.system_ids = (
            None
            if system_ids is None
            else {str(pid): str(system_id) for pid, system_id in system_ids.items()}
        )
        if self.shard_paths is not None:
            missing = [pid for pid in self.pids if pid not in self.shard_paths]
            if missing:
                raise ValueError(
                    f"sealed bank shard path inventory is missing {len(missing)} {split} IDs; "
                    f"first={missing[:3]}"
                )
        if self.system_ids is not None:
            missing_systems = [pid for pid in self.pids if pid not in self.system_ids]
            if missing_systems:
                raise ValueError(
                    f"sealed bank system_id inventory is missing {len(missing_systems)} "
                    f"{split} IDs; first={missing_systems[:3]}"
                )
        # Each persistent DataLoader worker validates a sealed shard once.  The
        # bank is immutable, so repeating full finite scans every epoch only
        # adds CPU latency without strengthening the contract.
        self._validated_shards: set[str] = set()
        self._external_sidecar_cache: dict[str, dict[str, Any]] = {}

    def __len__(self) -> int:
        return len(self.pids)

    def _shard_path(self, pid: str) -> Path:
        if self.shard_paths is not None:
            return self.shard_paths[pid]
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

    def _sample_pose_indices(
        self,
        shard: dict[str, torch.Tensor],
        graph: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor | None:
        max_poses = self.max_poses_per_complex
        if max_poses is None or max_poses <= 0:
            return None
        if self.split == "train" and self.max_pose_node_product is not None:
            if not isinstance(graph, dict):
                raise ValueError(
                    "max_pose_node_product requires a saved graph for every training complex"
                )
            node_coords = graph.get("node_coords")
            if not torch.is_tensor(node_coords) or node_coords.ndim != 2:
                raise ValueError("saved graph node_coords are required for pose-node capping")
            graph_nodes = int(node_coords.shape[0])
            if graph_nodes < 1:
                raise ValueError("saved graph must contain at least one node")
            max_poses = min(
                max_poses,
                max(1, self.max_pose_node_product // graph_nodes),
            )
            if (
                self.large_graph_node_threshold is not None
                and graph_nodes > self.large_graph_node_threshold
            ):
                max_poses = min(max_poses, self.large_graph_max_poses or 1)
        n_poses = int(shard["pose_atom_coords"].shape[0])
        if n_poses <= max_poses:
            return None

        if self.split == "train":
            if self.pose_sample_strategy == "stratified":
                idx = self._sample_pose_indices_stratified(shard, max_poses)
            else:
                best = int(shard[self.pose_target_key].argmin().item())
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
        rmsd = shard[self.pose_target_key].to(torch.float32)
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

    def _pose_value(
        self, shard: dict[str, Any], key: str, pose_idx: torch.Tensor | None
    ) -> torch.Tensor:
        value = shard[key]
        if pose_idx is not None and (
            key in POSE_TENSOR_KEYS
            or key == self.eval_target_key
            or key == self.pose_target_key
        ):
            value = value.index_select(0, pose_idx)
        return value

    @staticmethod
    def _require_tensor(
        shard: dict[str, Any], key: str, *, shard_path: Path
    ) -> torch.Tensor:
        value = shard.get(key)
        if not torch.is_tensor(value):
            raise ValueError(f"{shard_path}: {key} must be a tensor")
        return value

    def _validate_shard(
        self,
        shard: dict[str, Any],
        *,
        shard_path: Path,
        graph: dict[str, Any] | None,
    ) -> None:
        """Fail early on bank corruption instead of reaching the model with bad shapes."""
        poses = self._require_tensor(shard, "pose_atom_coords", shard_path=shard_path)
        if poses.ndim != 3 or poses.shape[-1] != 3:
            raise ValueError(
                f"{shard_path}: pose_atom_coords must have shape [K,A,3], got {tuple(poses.shape)}"
            )
        n_poses, n_atoms = int(poses.shape[0]), int(poses.shape[1])
        if n_poses < 1 or n_atoms < 1:
            raise ValueError(f"{shard_path}: confidence shard must contain poses and atoms")

        atom_disp = self._require_tensor(shard, "atom_disp", shard_path=shard_path)
        pose_rmsd = self._require_tensor(shard, self.pose_target_key, shard_path=shard_path)
        h_lig_node = self._require_tensor(shard, "h_lig_node", shard_path=shard_path)
        lig_node_type = self._require_tensor(shard, "lig_node_type", shard_path=shard_path)
        fragment_id = self._require_tensor(shard, "fragment_id", shard_path=shard_path)
        frag_sizes = self._require_tensor(shard, "frag_sizes", shard_path=shard_path)

        for key, value in (
            ("pose_atom_coords", poses),
            ("atom_disp", atom_disp),
            (self.pose_target_key, pose_rmsd),
            ("h_lig_node", h_lig_node),
        ):
            if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
                raise ValueError(f"{shard_path}: {key} must be a finite floating tensor")

        if atom_disp.shape != (n_poses, n_atoms):
            raise ValueError(
                f"{shard_path}: atom_disp shape {tuple(atom_disp.shape)} != {(n_poses, n_atoms)}"
            )
        if pose_rmsd.shape != (n_poses,):
            raise ValueError(
                f"{shard_path}: {self.pose_target_key} shape "
                f"{tuple(pose_rmsd.shape)} != {(n_poses,)}"
            )
        if h_lig_node.ndim != 3 or int(h_lig_node.shape[0]) != n_poses:
            raise ValueError(
                f"{shard_path}: h_lig_node must have shape [K,L,D], got {tuple(h_lig_node.shape)}"
            )

        crystal_anchor_keys = (
            "crystal_anchor_pose_atom_coords",
            "crystal_anchor_h_lig_node",
            "crystal_anchor_atom_disp",
            "crystal_anchor_pose_rmsd",
        )
        present_anchor_keys = [key for key in crystal_anchor_keys if key in shard]
        if present_anchor_keys and len(present_anchor_keys) != len(crystal_anchor_keys):
            raise ValueError(f"{shard_path}: crystal-anchor tensor inventory is incomplete")
        if present_anchor_keys:
            anchor_pose = self._require_tensor(
                shard, "crystal_anchor_pose_atom_coords", shard_path=shard_path
            )
            anchor_hidden = self._require_tensor(
                shard, "crystal_anchor_h_lig_node", shard_path=shard_path
            )
            anchor_disp = self._require_tensor(
                shard, "crystal_anchor_atom_disp", shard_path=shard_path
            )
            anchor_rmsd = self._require_tensor(
                shard, "crystal_anchor_pose_rmsd", shard_path=shard_path
            )
            expected_shapes = {
                "crystal_anchor_pose_atom_coords": (1, n_atoms, 3),
                "crystal_anchor_h_lig_node": (1, h_lig_node.shape[1], h_lig_node.shape[2]),
                "crystal_anchor_atom_disp": (1, n_atoms),
                "crystal_anchor_pose_rmsd": (1,),
            }
            for key, value in (
                ("crystal_anchor_pose_atom_coords", anchor_pose),
                ("crystal_anchor_h_lig_node", anchor_hidden),
                ("crystal_anchor_atom_disp", anchor_disp),
                ("crystal_anchor_pose_rmsd", anchor_rmsd),
            ):
                if value.shape != expected_shapes[key]:
                    raise ValueError(
                        f"{shard_path}: {key} shape {tuple(value.shape)} "
                        f"!= {expected_shapes[key]}"
                    )
                if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
                    raise ValueError(f"{shard_path}: {key} must be finite floating values")
            if bool(anchor_disp.abs().max() > 1e-6) or bool(anchor_rmsd.abs().max() > 1e-6):
                raise ValueError(f"{shard_path}: crystal anchor must have exact zero labels")
        if lig_node_type.ndim != 1 or int(lig_node_type.numel()) != int(h_lig_node.shape[1]):
            raise ValueError(
                f"{shard_path}: lig_node_type length must match h_lig_node ligand nodes"
            )
        if fragment_id.shape != (n_atoms,):
            raise ValueError(
                f"{shard_path}: fragment_id shape {tuple(fragment_id.shape)} != {(n_atoms,)}"
            )
        if frag_sizes.ndim != 1 or frag_sizes.numel() < 1:
            raise ValueError(f"{shard_path}: frag_sizes must be a non-empty vector")
        n_frags = int(frag_sizes.numel())
        if bool((frag_sizes <= 0).any()) or int(frag_sizes.sum().item()) != n_atoms:
            raise ValueError(f"{shard_path}: frag_sizes must be positive and sum to atom count")
        frag_id_long = fragment_id.to(torch.long)
        if bool((frag_id_long < 0).any()) or bool((frag_id_long >= n_frags).any()):
            raise ValueError(f"{shard_path}: fragment_id is outside [0,{n_frags})")
        observed_sizes = torch.bincount(frag_id_long, minlength=n_frags)
        if not torch.equal(observed_sizes.cpu(), frag_sizes.to(torch.long).cpu()):
            raise ValueError(f"{shard_path}: fragment_id counts do not match frag_sizes")
        atom_nodes = int((lig_node_type == NTYPE_LIG_ATOM).sum().item())
        frag_nodes = int((lig_node_type == NTYPE_FRAGMENT).sum().item())
        if atom_nodes != n_atoms or frag_nodes != n_frags:
            raise ValueError(
                f"{shard_path}: hidden ligand node counts atoms/fragments="
                f"{atom_nodes}/{frag_nodes}, expected {n_atoms}/{n_frags}"
            )

        target_key = self.eval_target_key
        if target_key is not None:
            target = self._require_tensor(shard, target_key, shard_path=shard_path)
            if target.shape != (n_poses,):
                raise ValueError(
                    f"{shard_path}: {target_key} shape {tuple(target.shape)} != {(n_poses,)}"
                )
            if not target.is_floating_point() or not bool(torch.isfinite(target).all()):
                raise ValueError(
                    f"{shard_path}: {target_key} must be a finite floating tensor"
                )

        if graph is None:
            return
        if not isinstance(graph, dict):
            raise ValueError(f"{shard_path}: saved centered graph must be a mapping")
        node_coords = graph.get("node_coords")
        node_type = graph.get("node_type")
        edge_index = graph.get("edge_index")
        if not torch.is_tensor(node_coords) or node_coords.ndim != 2 or node_coords.shape[1] != 3:
            raise ValueError(f"{shard_path}: graph node_coords must have shape [N,3]")
        if not node_coords.is_floating_point() or not bool(torch.isfinite(node_coords).all()):
            raise ValueError(f"{shard_path}: graph node_coords must be finite floating values")
        n_nodes = int(node_coords.shape[0])
        if not torch.is_tensor(node_type) or node_type.shape != (n_nodes,):
            raise ValueError(f"{shard_path}: graph node_type must have shape [N]")
        if not torch.is_tensor(edge_index) or edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError(f"{shard_path}: graph edge_index must have shape [2,E]")
        if edge_index.numel() and (
            int(edge_index.min().item()) < 0 or int(edge_index.max().item()) >= n_nodes
        ):
            raise ValueError(f"{shard_path}: graph edge_index references a missing node")

        for key, expected, expected_type in (
            ("lig_atom_slice", n_atoms, NTYPE_LIG_ATOM),
            ("lig_frag_slice", n_frags, NTYPE_FRAGMENT),
        ):
            bounds = graph.get(key)
            if not torch.is_tensor(bounds) or bounds.numel() != 2:
                raise ValueError(f"{shard_path}: graph {key} must contain two bounds")
            lo, hi = (int(value) for value in bounds.reshape(-1).tolist())
            if lo < 0 or hi < lo or hi > n_nodes or hi - lo != expected:
                raise ValueError(
                    f"{shard_path}: graph {key} bounds {(lo, hi)} do not match count {expected}"
                )
            if not bool((node_type[lo:hi] == expected_type).all()):
                raise ValueError(f"{shard_path}: graph {key} node types are inconsistent")

    @staticmethod
    def _sha256_bytes(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    @staticmethod
    def _symmetry_label_digest(sample_key: str, labels: torch.Tensor) -> str:
        values = labels.detach().cpu().contiguous().to(torch.float32)
        return hashlib.sha256(
            b"EFFDOCK_SYMMETRY_RMSD_LABEL_V1\0"
            + sample_key.encode()
            + b"\0"
            + values.numpy().tobytes(order="C")
        ).hexdigest()

    def _load_external_pose_target(
        self,
        pid: str,
    ) -> torch.Tensor:
        if self.external_pose_targets is None:
            raise RuntimeError("external target lookup requested without an inventory")
        record = self.external_pose_targets.get(pid)
        if not isinstance(record, dict):
            raise ValueError(f"external pose-target inventory is missing {pid}")
        sidecar_path = Path(str(record["sidecar_path"])).resolve()
        cache_key = str(sidecar_path)
        artifact = self._external_sidecar_cache.get(cache_key)
        if artifact is None:
            if not sidecar_path.is_file() or sidecar_path.is_symlink():
                raise FileNotFoundError(
                    f"external pose-target sidecar must be a regular file: {sidecar_path}"
                )
            data = sidecar_path.read_bytes()
            actual_sha = self._sha256_bytes(data)
            if actual_sha != record["sidecar_sha256"]:
                raise ValueError(
                    f"{pid}: external pose-target sidecar SHA mismatch: "
                    f"expected {record['sidecar_sha256']}, got {actual_sha}"
                )
            loaded = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
            if not isinstance(loaded, dict):
                raise ValueError(f"{sidecar_path}: sidecar payload must be a mapping")
            if (
                loaded.get("schema_version") != "EFFDOCK_S50_SYMMETRY_RMSD_SIDECAR_V1"
                or loaded.get("status") != "complete"
                or loaded.get("method") != "rdkit_calc_rms_symmetry_no_align"
                or loaded.get("split") != "train"
                or loaded.get("bank_manifest_sha256") != record["bank_manifest_sha256"]
                or loaded.get("input_manifest_sha256") != record["input_manifest_sha256"]
            ):
                raise ValueError(f"{sidecar_path}: unexpected sidecar contract")
            labels = loaded.get("pose_rmsd_symmetry_no_align")
            keys = loaded.get("sample_keys")
            n = len(keys) if isinstance(keys, list) else -1
            if (
                not torch.is_tensor(labels)
                or labels.shape != (n, 100)
                or not labels.is_floating_point()
                or not bool(torch.isfinite(labels).all())
            ):
                raise ValueError(f"{sidecar_path}: invalid symmetry target tensor")
            for key in (
                "system_ids",
                "split_indices",
                "source_pt_sha256",
                "pose_ensemble_sha256",
                "label_sha256",
            ):
                values = loaded.get(key)
                if not isinstance(values, list) or len(values) != n:
                    raise ValueError(f"{sidecar_path}: invalid {key} inventory")
            artifact = loaded
            self._external_sidecar_cache[cache_key] = artifact

        row_index = int(record["row_index"])
        if row_index < 0 or row_index >= len(artifact["sample_keys"]):
            raise ValueError(f"{pid}: external target row_index is out of range")
        expected_values = {
            "sample_keys": pid,
            "system_ids": record["system_id"],
            "split_indices": int(record["split_index"]),
            "source_pt_sha256": record["source_pt_sha256"],
            "pose_ensemble_sha256": record["pose_ensemble_sha256"],
            "label_sha256": record["label_sha256"],
        }
        for key, expected in expected_values.items():
            if artifact[key][row_index] != expected:
                raise ValueError(f"{pid}: sidecar {key} row does not match sealed manifest")
        labels = artifact["pose_rmsd_symmetry_no_align"][row_index].to(torch.float32)
        if self._symmetry_label_digest(pid, labels) != record["label_sha256"]:
            raise ValueError(f"{pid}: external symmetry label digest mismatch")
        return labels

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
        if not isinstance(shard, dict):
            raise ValueError(f"{shard_path}: confidence shard must be a mapping")
        if self.external_pose_targets is not None:
            shard = dict(shard)
            shard[self.pose_target_key] = self._load_external_pose_target(
                pid,
            )
        saved_graph = shard.get("graph_centered", shard.get("graph"))
        shard_cache_key = str(shard_path.resolve())
        if shard_cache_key not in self._validated_shards:
            self._validate_shard(shard, shard_path=shard_path, graph=saved_graph)
            self._validated_shards.add(shard_cache_key)
        system_id = shard.get("system_id")
        if self.system_ids is not None:
            expected_system_id = self.system_ids[pid]
            if not isinstance(system_id, str) or system_id != expected_system_id:
                raise ValueError(
                    f"{shard_path}: shard system_id {system_id!r} != sealed manifest "
                    f"{expected_system_id!r}"
                )
        pose_idx = self._sample_pose_indices(shard, saved_graph)
        pocket_center = shard["pocket_center_used"].to(torch.float32)
        protein_path = Path(shard.get("protein_pt", self.processed_dir / pid / "protein.pt"))
        pose_atom_coords = self._pose_value(shard, "pose_atom_coords", pose_idx).to(torch.float32)

        item = {
            "pid": pid,
            "split": self.split,
            "system_id": system_id,
            "pose_atom_coords": pose_atom_coords,
            "h_lig_node": self._pose_value(shard, "h_lig_node", pose_idx).to(torch.float32),
            "lig_node_type": shard["lig_node_type"].to(torch.long),
            "fragment_id": shard["fragment_id"].to(torch.long),
            "frag_sizes": shard["frag_sizes"].to(torch.long),
            "atom_disp": self._pose_value(shard, "atom_disp", pose_idx).to(torch.float32),
            "pose_rmsd": self._pose_value(
                shard, self.pose_target_key, pose_idx
            ).to(torch.float32),
            "pocket_center_used": pocket_center,
            "shard_path": str(shard_path),
        }
        if self.eval_target_key is not None:
            item[self.eval_target_key] = self._pose_value(
                shard, self.eval_target_key, pose_idx
            ).to(torch.float32)
        for key in (
            "crystal_anchor_pose_atom_coords",
            "crystal_anchor_h_lig_node",
            "crystal_anchor_atom_disp",
            "crystal_anchor_pose_rmsd",
        ):
            if key in shard:
                item[key] = shard[key].to(torch.float32)
        if saved_graph is not None:
            # S50 banks persist the exact deployment graph in its already
            # pocket-centered frame.  Do not recrop or rebuild it here.
            item["graph"] = saved_graph
        else:
            ligand_path = Path(shard.get("ligand_pt", self.processed_dir / pid / "ligand.pt"))
            item["graph"] = self._load_docking_graph_context(
                protein_path,
                ligand_path,
                pocket_center,
                pose_atom_coords,
                shard.get("lig_atom_coords_crystal_centered"),
            )
        return item


class PairedLigandPoseConfidenceDataset(Dataset):
    """Mix two sealed pose banks without double-weighting their shared complexes.

    Each component dataset must expose the same ordered complex inventory and
    sample the same number of poses.  One item remains one complex: pose-level
    tensors are concatenated and shuffled, while molecular topology and the
    receptor graph are required to agree across banks.
    """

    _POSE_KEYS = ("pose_atom_coords", "h_lig_node", "atom_disp", "pose_rmsd")
    _STATIC_KEYS = (
        "lig_node_type",
        "fragment_id",
        "frag_sizes",
        "pocket_center_used",
    )

    def __init__(
        self,
        primary: LigandPoseConfidenceDataset,
        auxiliary: LigandPoseConfidenceDataset,
    ) -> None:
        if primary.split != "train" or auxiliary.split != "train":
            raise ValueError("paired pose-bank mixing is training-only")
        if primary.pids != auxiliary.pids:
            raise ValueError("paired pose banks must have the same ordered complex inventory")
        self.primary = primary
        self.auxiliary = auxiliary
        self.pids = list(primary.pids)
        self.split = "train"
        self._validated_pairs: set[str] = set()

    def __len__(self) -> int:
        return len(self.pids)

    @staticmethod
    def _same_tensor(left: torch.Tensor, right: torch.Tensor) -> bool:
        if left.shape != right.shape or left.dtype != right.dtype:
            return False
        if left.is_floating_point():
            return bool(torch.allclose(left, right, rtol=1e-6, atol=1e-6))
        return bool(torch.equal(left, right))

    @classmethod
    def _require_matching_graphs(
        cls,
        primary: dict[str, torch.Tensor],
        auxiliary: dict[str, torch.Tensor],
        *,
        pid: str,
    ) -> None:
        if set(primary) != set(auxiliary):
            raise ValueError(f"{pid}: paired pose-bank graph fields differ")
        for key in primary:
            left = primary[key]
            right = auxiliary[key]
            if not torch.is_tensor(left) or not torch.is_tensor(right):
                raise ValueError(f"{pid}: paired pose-bank graph field {key!r} is not a tensor")
            if not cls._same_tensor(left, right):
                raise ValueError(f"{pid}: paired pose-bank graph field {key!r} differs")

    def __getitem__(self, index: int) -> dict[str, Any]:
        primary = self.primary[index]
        auxiliary = self.auxiliary[index]
        pid = self.pids[index]
        if primary["pid"] != pid or auxiliary["pid"] != pid:
            raise ValueError(f"{pid}: paired pose-bank item identity mismatch")
        if primary.get("system_id") != auxiliary.get("system_id"):
            raise ValueError(f"{pid}: paired pose-bank system identity mismatch")

        if pid not in self._validated_pairs:
            for key in self._STATIC_KEYS:
                left = primary.get(key)
                right = auxiliary.get(key)
                if not torch.is_tensor(left) or not torch.is_tensor(right):
                    raise ValueError(f"{pid}: paired pose-bank field {key!r} is missing")
                if not self._same_tensor(left, right):
                    raise ValueError(f"{pid}: paired pose-bank field {key!r} differs")
            self._require_matching_graphs(primary["graph"], auxiliary["graph"], pid=pid)
            self._validated_pairs.add(pid)

        primary_count = int(primary["pose_rmsd"].numel())
        auxiliary_count = int(auxiliary["pose_rmsd"].numel())
        if primary_count < 1 or auxiliary_count != primary_count:
            raise ValueError(
                f"{pid}: paired pose banks must contribute equal non-empty pose counts"
            )

        anchor_values = {
            "pose_atom_coords": primary.get("crystal_anchor_pose_atom_coords"),
            "h_lig_node": primary.get("crystal_anchor_h_lig_node"),
            "atom_disp": primary.get("crystal_anchor_atom_disp"),
            "pose_rmsd": primary.get("crystal_anchor_pose_rmsd"),
        }
        if any(value is not None for value in anchor_values.values()) and not all(
            torch.is_tensor(value) for value in anchor_values.values()
        ):
            raise ValueError(f"{pid}: primary pose bank has an incomplete crystal anchor")
        has_crystal_anchor = all(torch.is_tensor(value) for value in anchor_values.values())

        item = dict(primary)
        for key in self._POSE_KEYS:
            left = primary.get(key)
            right = auxiliary.get(key)
            if not torch.is_tensor(left) or not torch.is_tensor(right):
                raise ValueError(f"{pid}: paired pose tensor {key!r} is missing")
            if left.shape[1:] != right.shape[1:]:
                raise ValueError(f"{pid}: paired pose tensor {key!r} shape mismatch")
            values = [left, right]
            if has_crystal_anchor:
                anchor = anchor_values[key]
                if not torch.is_tensor(anchor) or anchor.shape[1:] != left.shape[1:]:
                    raise ValueError(f"{pid}: crystal anchor tensor {key!r} shape mismatch")
                values.append(anchor)
            item[key] = torch.cat(values, dim=0)
        item["pose_bank_component"] = torch.cat(
            (
                torch.zeros(primary_count, dtype=torch.long),
                torch.ones(auxiliary_count, dtype=torch.long),
                torch.full((1 if has_crystal_anchor else 0,), 2, dtype=torch.long),
            )
        )
        permutation = torch.randperm(
            primary_count + auxiliary_count + (1 if has_crystal_anchor else 0)
        )
        for key in (*self._POSE_KEYS, "pose_bank_component"):
            item[key] = item[key].index_select(0, permutation)
        item["shard_path"] = [primary["shard_path"], auxiliary["shard_path"]]
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
    "PairedLigandPoseConfidenceDataset",
    "DEFAULT_CONFIDENCE_POSE_TAG",
    "PROTEIN_FLAG_KEYS",
    "collate_complexes",
    "to_device",
]
