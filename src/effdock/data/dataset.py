"""Dataset: loads protein.pt + ligand.pt + meta.pt, crops pocket at runtime,
builds the unified graph on-the-fly, and samples flow matching state.

Returns a flat dict with all node/edge tensors from the graph plus flow
matching targets (T_frag, q_frag at time t, velocity targets).  Fragment
and atom node coordinates are updated to reflect the interpolated pose.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import Dataset

from effdock.geometry.flow_matching import (
    compute_angular_velocity,
    compute_flow_matching_targets,
    compute_vp_flow_targets,
    compute_vp_score_full_targets,
    compute_vp_score_targets,
    sample_prior_poses,
)
from effdock.geometry.se3 import (
    axis_angle_to_quaternion,
    matrix_to_quaternion,
    quaternion_inverse,
    quaternion_multiply,
    quaternion_to_matrix,
    sample_uniform_quaternion,
)
from effdock.preprocess.graph import build_static_complex_graph

_INDEX_VERSION = 1


def _load_sample_metadata(path: Path) -> tuple[int, int, int, int, int]:
    meta = torch.load(path, map_location="cpu", weights_only=True)
    n_atom = int(meta["num_atom"])
    n_frag = int(meta["num_frag"])
    n_res = int(meta["num_res"])
    n_pocket_res = int(meta.get("num_pocket_res", n_res))
    n_prot_atom = int(meta.get("num_prot_atom", n_res * 8))
    return n_atom, n_frag, n_res, n_pocket_res, n_prot_atom


def _dataset_index(
    root: Path,
    sample_ids: list[str],
    *,
    min_atoms: int,
    max_atoms: int,
    max_frags: int,
    min_protein_res: int,
) -> tuple[list[str], list[float]]:
    """Build or read the filtered sample index without an N-rank metadata storm."""
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {
                "version": _INDEX_VERSION,
                "root": str(root.resolve()),
                "min_atoms": min_atoms,
                "max_atoms": max_atoms,
                "max_frags": max_frags,
                "min_protein_res": min_protein_res,
            },
            sort_keys=True,
        ).encode()
    )
    for sample_id in sample_ids:
        digest.update(b"\0")
        digest.update(sample_id.encode())

    cache_dir = root / ".effdock_index"
    cache_path = cache_dir / f"{digest.hexdigest()}.json"
    lock_path = cache_path.with_suffix(".lock")
    cache_dir.mkdir(parents=True, exist_ok=True)

    def read_cache() -> tuple[list[str], list[float]]:
        with cache_path.open() as handle:
            cached = json.load(handle)
        return cached["sample_ids"], cached["sample_costs"]

    if cache_path.exists():
        return read_cache()

    # File locking makes concurrent DDP ranks wait for one cache builder.
    import fcntl

    with lock_path.open("w") as lock_handle:
        fcntl.flock(lock_handle, fcntl.LOCK_EX)
        if cache_path.exists():
            return read_cache()

        def inspect(sample_id: str) -> tuple[str, float] | None:
            directory = root / sample_id
            if not all(
                (directory / filename).exists()
                for filename in ("protein.pt", "ligand.pt", "meta.pt")
            ):
                return None
            n_atom, n_frag, n_res, n_pocket_res, n_prot_atom = _load_sample_metadata(
                directory / "meta.pt"
            )
            if n_atom < min_atoms or n_atom > max_atoms or n_frag > max_frags:
                return None
            if n_res < min_protein_res:
                return None
            cost = float(
                n_atom * 32 + n_frag * 128 + n_pocket_res * 256 + n_res * 16 + n_prot_atom * 2
            )
            return sample_id, cost

        workers = min(32, max(4, os.cpu_count() or 4))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            records = [record for record in executor.map(inspect, sample_ids) if record]
        ids = [record[0] for record in records]
        costs = [record[1] for record in records]
        temporary = cache_path.with_suffix(f".tmp.{os.getpid()}")
        temporary.write_text(json.dumps({"sample_ids": ids, "sample_costs": costs}))
        os.replace(temporary, cache_path)
        return ids, costs


def crop_to_pocket(
    prot_data: dict[str, Tensor],
    ref_coords: Tensor,
    cutoff: float = 8.0,
) -> dict[str, Tensor] | None:
    """Crop full protein tensors to pocket around reference coordinates.

    Residue-aware: if any atom of a residue is within *cutoff* of any
    reference point, the entire residue is kept.  Atom/bond/virtual-node
    indices are compacted.

    Args:
        prot_data: Full protein dict (from protein.pt / parse_pocket_atoms).
        ref_coords: [N_ref, 3] reference coordinates (e.g. crystal ligand atoms)
                    or [3] single point (e.g. predicted pocket center).
        cutoff: Distance cutoff in Angstroms.

    Returns cropped dict with the same keys, or None if nothing survives.
    """
    if ref_coords.ndim == 1:
        ref_coords = ref_coords.unsqueeze(0)

    # Atom-level distance filter
    patom_coords = prot_data["patom_coords"]
    patom_residue_id = prot_data["patom_residue_id"]
    dmat = torch.cdist(patom_coords, ref_coords)
    in_range = dmat.min(dim=1).values <= cutoff

    # Residue-aware: keep every atom whose residue has ≥1 atom in range
    active_res = patom_residue_id[in_range].unique()
    atom_mask = torch.isin(patom_residue_id, active_res)
    return _crop_protein_by_atom_mask(prot_data, atom_mask)


def _random_rotation_matrix(
    rot_sigma_deg: float, gen: torch.Generator | None, dtype: torch.dtype
) -> Tensor:
    """Sample a small SO(3) perturbation from an axis-angle Gaussian."""
    rot_sigma = math.radians(float(rot_sigma_deg))
    if rot_sigma <= 0.0:
        return torch.eye(3, dtype=dtype)
    rot_vec = torch.randn(3, generator=gen, dtype=dtype) * rot_sigma
    q = axis_angle_to_quaternion(rot_vec.view(1, 3))
    return quaternion_to_matrix(q)[0]


def _fragment_component(
    fragment_adj_index: Tensor,
    start_frag: int,
    blocked_frag: int,
    n_frags: int,
) -> set[int]:
    adj: list[list[int]] = [[] for _ in range(n_frags)]
    if fragment_adj_index.numel() > 0:
        for s, d in fragment_adj_index.T.tolist():
            adj[int(s)].append(int(d))
    seen = {int(start_frag)}
    stack = [int(start_frag)]
    while stack:
        cur = stack.pop()
        for nxt in adj[cur]:
            if nxt == int(blocked_frag) or nxt in seen:
                continue
            seen.add(nxt)
            stack.append(nxt)
    return seen


def _choose_torsion_mask(
    ligand: dict[str, Tensor],
    gen: torch.Generator | None,
    *,
    prefer_side: str,
) -> tuple[int, int, Tensor] | None:
    cut_bonds = ligand.get("cut_bond_index")
    if cut_bonds is None or cut_bonds.numel() == 0:
        return None
    frag_id = ligand["fragment_id"].to(torch.long)
    n_frags = int(ligand["frag_sizes"].numel())
    order = torch.randperm(cut_bonds.shape[1], generator=gen).tolist()
    for cut_i in order:
        a = int(cut_bonds[0, cut_i].item())
        b = int(cut_bonds[1, cut_i].item())
        fa = int(frag_id[a].item())
        fb = int(frag_id[b].item())
        if fa == fb:
            continue
        comp_b = _fragment_component(
            ligand["fragment_adj_index"].to(torch.long),
            fb,
            fa,
            n_frags,
        )
        comp_a = set(range(n_frags)) - comp_b
        if not comp_a or not comp_b:
            continue
        if prefer_side == "larger":
            choose_b = len(comp_b) > len(comp_a) or (
                len(comp_b) == len(comp_a) and bool(torch.rand((), generator=gen) < 0.5)
            )
        elif prefer_side == "random":
            choose_b = bool(torch.rand((), generator=gen) < 0.5)
        else:
            choose_b = len(comp_b) < len(comp_a) or (
                len(comp_b) == len(comp_a) and bool(torch.rand((), generator=gen) < 0.5)
            )
        chosen = comp_b if choose_b else comp_a
        mask = torch.zeros(frag_id.shape[0], dtype=torch.bool)
        for frag in chosen:
            mask |= frag_id == int(frag)
        if bool(mask.any()) and not bool(mask.all()):
            return a, b, mask
    return None


def _axis_angle_rotation(axis: Tensor, angle_rad: Tensor | float, dtype: torch.dtype) -> Tensor:
    axis = axis.to(dtype)
    axis = axis / axis.norm().clamp_min(1e-6)
    angle = torch.as_tensor(angle_rad, dtype=dtype)
    kx = torch.stack(
        (
            torch.stack((axis.new_tensor(0.0), -axis[2], axis[1])),
            torch.stack((axis[2], axis.new_tensor(0.0), -axis[0])),
            torch.stack((-axis[1], axis[0], axis.new_tensor(0.0))),
        )
    )
    eye = torch.eye(3, dtype=dtype)
    return eye + torch.sin(angle) * kx + (1.0 - torch.cos(angle)) * (kx @ kx)


def _apply_torsion_perturb(
    coords: Tensor,
    ligand: dict[str, Tensor],
    *,
    torsion_degrees: tuple[float, ...],
    max_rot_bonds: int,
    prefer_side: str,
    gen: torch.Generator | None,
) -> Tensor:
    if max_rot_bonds <= 0 or not torsion_degrees:
        return coords
    out = coords.clone()
    n_apply = int(torch.randint(1, max_rot_bonds + 1, (1,), generator=gen).item())
    applied = 0
    for i in range(n_apply):
        choice = _choose_torsion_mask(ligand, gen, prefer_side=prefer_side)
        if choice is None:
            break
        a, b, mask = choice
        axis = out[b] - out[a]
        if float(axis.norm().item()) < 1e-6:
            continue
        max_deg = float(torsion_degrees[(applied + i) % len(torsion_degrees)])
        angle = (torch.rand((), generator=gen, dtype=out.dtype) * 2.0 - 1.0) * max_deg
        rot = _axis_angle_rotation(axis, angle * math.pi / 180.0, out.dtype)
        pivot = out[a].view(1, 3)
        out[mask] = (out[mask] - pivot) @ rot.T + pivot
        applied += 1
    return out


def _recover_frag_state(
    atom_pos: Tensor,
    local_pos: Tensor,
    frag_id: Tensor,
    n_frags: int,
) -> tuple[Tensor, Tensor]:
    T = torch.zeros(n_frags, 3, dtype=atom_pos.dtype)
    R = torch.eye(3, dtype=atom_pos.dtype).expand(n_frags, 3, 3).clone()
    for frag in range(n_frags):
        mask = frag_id == frag
        if not bool(mask.any()):
            continue
        y = atom_pos[mask]
        x = local_pos[mask]
        T[frag] = y.mean(dim=0)
        if int(mask.sum().item()) <= 1:
            continue
        y_c = y - T[frag]
        H = x.T @ y_c
        U, _, Vh = torch.linalg.svd(H)
        d = torch.sign(torch.linalg.det(Vh.T @ U.T))
        D = torch.eye(3, dtype=atom_pos.dtype)
        D[2, 2] = d
        R[frag] = Vh.T @ D @ U.T
    return T, matrix_to_quaternion(R)


def crop_to_nearest_residues(
    prot_data: dict[str, Tensor],
    ref_coords: Tensor,
    max_residues: int = 32,
) -> dict[str, Tensor] | None:
    """Crop to the closest residues when radius-based pocket crop is empty."""
    if ref_coords.ndim == 1:
        ref_coords = ref_coords.unsqueeze(0)

    patom_coords = prot_data["patom_coords"]
    patom_residue_id = prot_data["patom_residue_id"]
    if patom_coords.numel() == 0 or patom_residue_id.numel() == 0:
        return None

    dmat = torch.cdist(patom_coords, ref_coords)
    atom_dist = dmat.min(dim=1).values
    n_res = int(patom_residue_id.max().item()) + 1
    res_dist = torch.full((n_res,), float("inf"), dtype=atom_dist.dtype)
    res_dist.scatter_reduce_(0, patom_residue_id, atom_dist, reduce="amin", include_self=True)
    k = min(max_residues, n_res)
    if k <= 0:
        return None
    active_res = torch.topk(res_dist, k=k, largest=False).indices
    atom_mask = torch.isin(patom_residue_id, active_res)
    return _crop_protein_by_atom_mask(prot_data, atom_mask)


def _crop_protein_by_atom_mask(
    prot_data: dict[str, Tensor],
    atom_mask: Tensor,
) -> dict[str, Tensor] | None:
    """Return a schema-preserving protein crop from an atom mask."""
    patom_coords = prot_data["patom_coords"]
    patom_residue_id = prot_data["patom_residue_id"]
    if not atom_mask.any():
        return None

    # Old → new atom index mapping
    old_indices = atom_mask.nonzero(as_tuple=True)[0]
    remap = torch.full((patom_coords.shape[0],), -1, dtype=torch.int64)
    remap[old_indices] = torch.arange(old_indices.shape[0], dtype=torch.int64)

    # Atom tensors
    new_residue_id = patom_residue_id[atom_mask]
    _, new_residue_id = torch.unique(new_residue_id, return_inverse=True)

    # Bond filter + remap
    pbond = prot_data["pbond_index"]
    if pbond.numel() > 0:
        keep = atom_mask[pbond[0]] & atom_mask[pbond[1]]
        new_pbond = torch.stack([remap[pbond[0][keep]], remap[pbond[1][keep]]])
    else:
        new_pbond = torch.zeros(2, 0, dtype=torch.int64)

    # Virtual-node filter (anchor atom must be kept)
    pres_mask = atom_mask[prot_data["pres_atom_index"]]

    out = {
        "patom_coords": patom_coords[atom_mask],
        "patom_token": prot_data["patom_token"][atom_mask],
        "patom_residue_id": new_residue_id,
        "patom_is_backbone": prot_data["patom_is_backbone"][atom_mask],
        "patom_is_metal": prot_data["patom_is_metal"][atom_mask],
        "pbond_index": new_pbond,
        "pres_coords": prot_data["pres_coords"][pres_mask],
        "pres_residue_type": prot_data["pres_residue_type"][pres_mask],
        "pres_atom_index": remap[prot_data["pres_atom_index"][pres_mask]],
        "pres_is_pseudo": prot_data["pres_is_pseudo"][pres_mask],
    }
    # schema_v2 per-atom pharmacophore (donor/acceptor/+/-/hydrophobic).
    # Pass through cropping when present in the source protein.pt.
    for k in (
        "patom_is_donor",
        "patom_is_acceptor",
        "patom_is_positive",
        "patom_is_negative",
        "patom_is_hydrophobic",
    ):
        if k in prot_data:
            out[k] = prot_data[k][atom_mask]
    return out


class EFFDockDataset(Dataset):
    """Dataset for the unified equivariant model.

    Loads per-complex protein.pt (full protein) and ligand.pt, crops the
    protein to a pocket at runtime, builds the unified graph, then applies
    flow matching sampling.

    Args:
        root: Directory with per-complex subdirs containing protein.pt, ligand.pt, meta.pt.
        split_file: JSON or text file with PDB IDs.
        split_key: Key for JSON split files.
        pocket_cutoff: Residue-aware distance cutoff for pocket cropping (Å).
        pocket_jitter_sigma: Gaussian jitter on pocket center (training augmentation).
        pocket_cutoff_noise: Uniform noise on cutoff (training augmentation).
        translation_sigma: Gaussian std for translation prior (Å).
        max_atoms / max_frags / min_atoms / min_protein_res: size filters.
        rotation_augmentation: "none", "ligand_uniform", or "per_fragment".
        deterministic: Fix all random sampling per sample (reproducible eval).
        seed: Base seed for deterministic mode.
    """

    def __init__(
        self,
        root: str | Path,
        split_file: str | Path | None = None,
        split_key: str = "train",
        pocket_cutoff: float = 8.0,
        pocket_jitter_sigma: float = 2.0,
        pocket_cutoff_noise: float = 2.0,
        translation_sigma: float = 10.0,
        max_atoms: int = 80,
        max_frags: int = 20,
        min_atoms: int = 5,
        min_protein_res: int = 50,
        rotation_augmentation: str = "none",
        deterministic: bool = False,
        seed: int = 42,
        receptor_aug_prob: float = 0.0,
        alt_receptor_root: str | Path | None = None,
        alt_receptor_mapping: str | Path | None = None,
        # Wider, range-based augmentation (overrides the ± noise / single
        # sigma when set). Set both to None to fall back to the symmetric
        # ``pocket_cutoff_noise`` and fixed ``translation_sigma`` behavior.
        pocket_cutoff_range: tuple[float, float] | None = None,
        prior_sigma_range: tuple[float, float] | None = None,
        prior_sigma_log_uniform: bool = True,
        prior_sigma_values: tuple[float, ...] | list[float] | None = None,
        prior_sigma_weights: tuple[float, ...] | list[float] | None = None,
        # Time sampling distribution. Default = "simplefold":
        #   p(t) = 0.02·U(0,1) + 0.98·LN(m=0.8, s=1.7)
        # Shifted logit-normal peaked near t≈0.69 (late refinement focus)
        # with a 2 % uniform floor so both endpoints stay above zero —
        # matches what SimpleFold and AF3-style structure models use, and
        # aligns naturally with our late-biased ODE inference schedule.
        # Other options:
        #   "uniform"      — Lipman et al. 2023 baseline (flat)
        #   "logit_normal" — SD3-style (peaked at t≈0.5)
        #   "mixture"      — 70 % logit-normal + 10 % U[0.02,0.20] + 20 % U[0.75,0.98]
        time_distribution: str = "simplefold",
        local_refine_prob: float = 0.0,
        local_refine_trans_sigmas: tuple[float, ...] | list[float] | None = None,
        local_refine_trans_weights: tuple[float, ...] | list[float] | None = None,
        local_refine_rot_sigma_deg: float = 15.0,
        local_refine_horizon_range: tuple[float, float] | list[float] = (0.12, 0.35),
        local_refine_mode: str = "fragment",
        local_refine_torsion_degrees: tuple[float, ...] | list[float] | None = None,
        local_refine_max_torsion_bonds: int = 2,
        local_refine_torsion_side: str = "smaller",
        pose_objective: str = "linear_fm",
        score_rot_sigma_max: float = 3.141592653589793,
        score_alpha_min: float = 0.0,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.pocket_cutoff = pocket_cutoff
        self.pocket_jitter_sigma = pocket_jitter_sigma
        self.pocket_cutoff_noise = pocket_cutoff_noise
        self.translation_sigma = translation_sigma
        self.rotation_augmentation = rotation_augmentation
        self.deterministic = deterministic
        self.pocket_cutoff_range = (
            tuple(pocket_cutoff_range) if pocket_cutoff_range is not None else None
        )
        self.prior_sigma_range = tuple(prior_sigma_range) if prior_sigma_range is not None else None
        self.prior_sigma_log_uniform = prior_sigma_log_uniform
        self.prior_sigma_values = (
            tuple(float(v) for v in prior_sigma_values) if prior_sigma_values is not None else None
        )
        self.prior_sigma_weights = (
            tuple(float(w) for w in prior_sigma_weights)
            if prior_sigma_weights is not None
            else None
        )
        if self.prior_sigma_values is not None:
            if not self.prior_sigma_values:
                raise ValueError("prior_sigma_values must be non-empty when set")
            if any(v <= 0 for v in self.prior_sigma_values):
                raise ValueError("prior_sigma_values must be positive")
            if self.prior_sigma_weights is None:
                self.prior_sigma_weights = tuple(1.0 for _ in self.prior_sigma_values)
            if len(self.prior_sigma_weights) != len(self.prior_sigma_values):
                raise ValueError("prior_sigma_weights must match prior_sigma_values length")
            if any(w < 0 for w in self.prior_sigma_weights) or sum(self.prior_sigma_weights) <= 0:
                raise ValueError("prior_sigma_weights must be non-negative with positive sum")
        td = (time_distribution or "simplefold").lower()
        if td not in ("uniform", "logit_normal", "mixture", "simplefold"):
            raise ValueError(
                f"time_distribution must be 'uniform', 'logit_normal', "
                f"'mixture' or 'simplefold', got {time_distribution!r}"
            )
        self.time_distribution = td
        objective = (pose_objective or "linear_fm").lower()
        if objective not in ("linear_fm", "vp_flow", "vp_score", "vp_score_full"):
            raise ValueError(
                f"pose_objective must be 'linear_fm', 'vp_flow', 'vp_score' or 'vp_score_full', got {pose_objective!r}"
            )
        self.pose_objective = objective
        self.score_rot_sigma_max = float(score_rot_sigma_max)
        self.score_alpha_min = float(score_alpha_min)
        self.local_refine_prob = float(local_refine_prob or 0.0)
        if not (0.0 <= self.local_refine_prob <= 1.0):
            raise ValueError("local_refine_prob must be in [0, 1]")
        self.local_refine_trans_sigmas = tuple(
            float(v) for v in (local_refine_trans_sigmas or (0.25, 0.5, 1.0, 1.5))
        )
        if any(v <= 0 for v in self.local_refine_trans_sigmas):
            raise ValueError("local_refine_trans_sigmas must be positive")
        self.local_refine_trans_weights = tuple(
            float(w)
            for w in (local_refine_trans_weights or (1.0 for _ in self.local_refine_trans_sigmas))
        )
        if len(self.local_refine_trans_weights) != len(self.local_refine_trans_sigmas):
            raise ValueError("local_refine_trans_weights must match local_refine_trans_sigmas")
        if (
            any(w < 0 for w in self.local_refine_trans_weights)
            or sum(self.local_refine_trans_weights) <= 0
        ):
            raise ValueError("local_refine_trans_weights must be non-negative with positive sum")
        self.local_refine_rot_sigma_deg = float(local_refine_rot_sigma_deg)
        if self.local_refine_rot_sigma_deg < 0:
            raise ValueError("local_refine_rot_sigma_deg must be non-negative")
        self.local_refine_horizon_range = tuple(float(v) for v in local_refine_horizon_range)
        if len(self.local_refine_horizon_range) != 2:
            raise ValueError("local_refine_horizon_range must contain [min, max]")
        h_lo, h_hi = self.local_refine_horizon_range
        if not (0.0 < h_lo <= h_hi < 1.0):
            raise ValueError("local_refine_horizon_range must satisfy 0 < min <= max < 1")
        self.local_refine_mode = (local_refine_mode or "fragment").lower()
        if self.local_refine_mode not in ("fragment", "torsion"):
            raise ValueError("local_refine_mode must be 'fragment' or 'torsion'")
        self.local_refine_torsion_degrees = tuple(
            float(v) for v in (local_refine_torsion_degrees or (5.0, 10.0, 15.0))
        )
        if any(v <= 0 for v in self.local_refine_torsion_degrees):
            raise ValueError("local_refine_torsion_degrees must be positive")
        self.local_refine_max_torsion_bonds = int(local_refine_max_torsion_bonds)
        if self.local_refine_max_torsion_bonds < 0:
            raise ValueError("local_refine_max_torsion_bonds must be non-negative")
        self.local_refine_torsion_side = (local_refine_torsion_side or "smaller").lower()
        if self.local_refine_torsion_side not in ("smaller", "larger", "random"):
            raise ValueError("local_refine_torsion_side must be 'smaller', 'larger' or 'random'")
        self.seed = seed
        self.receptor_aug_prob = receptor_aug_prob
        self.alt_receptor_root: Path | None = Path(alt_receptor_root) if alt_receptor_root else None
        self.alt_receptor_mapping: dict[str, list[dict]] = {}
        if alt_receptor_mapping is not None and receptor_aug_prob > 0:
            with open(alt_receptor_mapping) as f:
                self.alt_receptor_mapping = json.load(f).get("mapping", {})

        # Collect PDB IDs
        if split_file is not None:
            sf = Path(split_file)
            if sf.suffix == ".json":
                with open(sf) as f:
                    split_data = json.load(f)
                pdb_ids = split_data[split_key]
            else:
                with open(sf) as f:
                    pdb_ids = [line.strip() for line in f if line.strip()]
        else:
            pdb_ids = sorted(d.name for d in self.root.iterdir() if d.is_dir())

        self.pdb_ids, self.sample_costs = _dataset_index(
            self.root,
            list(pdb_ids),
            min_atoms=min_atoms,
            max_atoms=max_atoms,
            max_frags=max_frags,
            min_protein_res=min_protein_res,
        )

    def __len__(self) -> int:
        return len(self.pdb_ids)

    def _make_generator(self, idx: int, stream_offset: int) -> torch.Generator | None:
        """Return a seeded generator for deterministic mode, else None (global RNG)."""
        if not self.deterministic:
            return None
        generator = torch.Generator()
        generator.manual_seed(self.seed + idx * 9_973 + stream_offset * 104_729)
        return generator

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        pdb_id = self.pdb_ids[idx]
        data_dir = self.root / pdb_id

        ligand = torch.load(data_dir / "ligand.pt", weights_only=True)
        meta = torch.load(data_dir / "meta.pt", weights_only=True)

        # --- Receptor augmentation: optionally swap holo with apo / predicted ---
        # Same chain space (PLINDER guarantees pocket_lddt ≥ 80 alignment), so
        # holo's pocket_center can crop the alt protein at runtime.
        prot_data = None
        sys_id = meta.get("plinder_system_id", pdb_id)
        if (
            self.receptor_aug_prob > 0
            and self.alt_receptor_root is not None
            and sys_id in self.alt_receptor_mapping
        ):
            aug_gen = self._make_generator(idx, stream_offset=4)
            roll = (
                torch.rand(1, generator=aug_gen).item()
                if aug_gen is not None
                else torch.rand(1).item()
            )
            if roll < self.receptor_aug_prob:
                alts = self.alt_receptor_mapping[sys_id]
                pick_idx = (
                    int(torch.randint(0, len(alts), (1,), generator=aug_gen).item())
                    if aug_gen is not None
                    else int(torch.randint(0, len(alts), (1,)).item())
                )
                alt_path = self.alt_receptor_root / f"{alts[pick_idx]['id']}.pt"
                if alt_path.exists():
                    prot_data = torch.load(alt_path, weights_only=True)

        if prot_data is None:
            prot_data = torch.load(data_dir / "protein.pt", weights_only=True)

        pocket_center = meta["pocket_center"]
        n_frags = meta["num_frag"].item()

        # --- Crop protein to pocket & build graph -------------------------
        # Always use pocket_center (protein residue centroid) as reference,
        # never ligand coords — matches inference where ligand position is unknown.
        # Both jitter and cutoff noise honor `deterministic` mode via
        # _make_generator so that val/debug runs are bit-identical.
        ref_center = pocket_center.clone()
        cutoff = self.pocket_cutoff
        if self.pocket_jitter_sigma > 0:
            jitter_gen = self._make_generator(idx, stream_offset=5)
            ref_center = (
                ref_center
                + torch.randn(3, generator=jitter_gen, dtype=ref_center.dtype)
                * self.pocket_jitter_sigma
            )
        if self.pocket_cutoff_range is not None:
            # Wider explicit Uniform[lo, hi] sampling. Lets training see a
            # broader pocket-context distribution (e.g. (5.0, 12.0) for
            # cofactor-heavy ligands that need more residues vs tight
            # drug-like that need fewer).
            cutoff_gen = self._make_generator(idx, stream_offset=6)
            lo, hi = self.pocket_cutoff_range
            u = torch.rand(1, generator=cutoff_gen).item()
            cutoff = lo + u * (hi - lo)
            cutoff = max(cutoff, 4.0)
        elif self.pocket_cutoff_noise > 0:
            cutoff_gen = self._make_generator(idx, stream_offset=6)
            # Symmetric Uniform[-noise, +noise] around the configured cutoff
            # (e.g. cutoff=8, noise=2 → 6..10 Å). The previous formula was
            # asymmetric (-0.5..1.5) and produced 6..14 Å.
            u = torch.rand(1, generator=cutoff_gen).item()
            cutoff = cutoff + (u * 2.0 - 1.0) * self.pocket_cutoff_noise
            cutoff = max(cutoff, 4.0)
        cropped_prot = crop_to_pocket(prot_data, ref_center, cutoff=cutoff)
        if cropped_prot is None:
            cropped_prot = crop_to_pocket(prot_data, pocket_center, cutoff=self.pocket_cutoff + 5.0)
        if cropped_prot is None:
            cropped_prot = crop_to_pocket(
                prot_data, pocket_center, cutoff=max(40.0, self.pocket_cutoff + 10.0)
            )
        if cropped_prot is None:
            cropped_prot = crop_to_nearest_residues(prot_data, pocket_center, max_residues=32)
        assert cropped_prot is not None, f"No pocket residues found for {pdb_id}"
        graph = build_static_complex_graph(ligand, cropped_prot)

        # --- Fragment target (crystal pose, pocket-centered) --------------
        T_1 = ligand["frag_centers"] - pocket_center  # [N_frag, 3]

        # Fragment sizes + atom→fragment assignment
        frag_id_for_atoms = ligand["fragment_id"]
        frag_sizes = torch.zeros(n_frags, dtype=torch.int64)
        frag_sizes.scatter_add_(0, frag_id_for_atoms, torch.ones_like(frag_id_for_atoms))
        local_pos_orig = ligand["frag_local_coords"]

        # --- Rotation augmentation ----------------------------------------
        identity_q = torch.zeros(n_frags, 4, dtype=T_1.dtype)
        identity_q[:, 0] = 1.0

        if self.rotation_augmentation == "none":
            R_aug_q, q_1 = identity_q, identity_q.clone()
        else:
            aug_gen = self._make_generator(idx, stream_offset=1)
            if self.rotation_augmentation == "ligand_uniform":
                R_aug_q = sample_uniform_quaternion(1, dtype=T_1.dtype, generator=aug_gen)
                R_aug_q = R_aug_q.expand(n_frags, -1).clone()
            else:
                R_aug_q = sample_uniform_quaternion(n_frags, dtype=T_1.dtype, generator=aug_gen)
            q_1 = quaternion_inverse(R_aug_q)

        # Mask single-atom fragments
        single_mask = frag_sizes <= 1
        q_1[single_mask] = identity_q[single_mask]

        # Augment local coords
        R_aug = quaternion_to_matrix(R_aug_q)
        local_pos = torch.einsum("nij,nj->ni", R_aug[frag_id_for_atoms], local_pos_orig)

        # --- Flow matching sampling ---------------------------------------
        prior_gen = self._make_generator(idx, stream_offset=2)
        time_gen = self._make_generator(idx, stream_offset=3)

        # Time sampling — dispatch on self.time_distribution.
        #
        # "simplefold" (default) — p(t) = 0.02·U + 0.98·LN(m=0.8, s=1.7).
        #   Shifted logit-normal peaked near t≈0.69 with a 2 % uniform
        #   floor: emphasizes late refinement (matches inference late
        #   schedule) while keeping endpoints non-zero. Source: SimpleFold
        #   protein-folding paper.
        # "logit_normal" — t = sigmoid(N(0, 1)); SD3-style.
        # "mixture"      — 70 % logit-normal + 10 % U[0.02,0.20] + 20 % U[0.75,0.98].
        # "uniform"      — t ~ U(0, 1); Lipman et al. 2023 baseline.
        import math

        if self.time_distribution == "logit_normal":
            z = torch.randn(1, generator=time_gen, dtype=T_1.dtype).item()
            t = 1.0 / (1.0 + math.exp(-z))
        elif self.time_distribution == "mixture":
            r = torch.rand(1, generator=time_gen).item()
            if r < 0.7:
                z = torch.randn(1, generator=time_gen, dtype=T_1.dtype).item()
                t = 1.0 / (1.0 + math.exp(-z))
            elif r < 0.8:
                t = 0.02 + torch.rand(1, generator=time_gen).item() * (0.20 - 0.02)
            else:
                t = 0.75 + torch.rand(1, generator=time_gen).item() * (0.98 - 0.75)
        elif self.time_distribution == "simplefold":
            r = torch.rand(1, generator=time_gen).item()
            if r < 0.02:
                t = torch.rand(1, generator=time_gen).item()
            else:
                z = torch.randn(1, generator=time_gen, dtype=T_1.dtype).item()
                t = 1.0 / (1.0 + math.exp(-(0.8 + 1.7 * z)))
        else:  # "uniform"
            t = torch.rand(1, generator=time_gen).item()

        # Per-sample translation prior σ. Range-based sampling lets the model
        # see priors of varying width — at inference we then have one
        # network that handles σ ∈ [σ_min, σ_max] without retraining.
        # Log-uniform is the natural choice for a scale parameter; uniform
        # is offered as a fallback when prior_sigma_log_uniform=False.
        if self.prior_sigma_values is not None:
            sigma_gen = self._make_generator(idx, stream_offset=7)
            weights = torch.tensor(self.prior_sigma_weights, dtype=torch.float32)
            probs = weights / weights.sum()
            u = torch.rand(1, generator=sigma_gen).item()
            cumulative = torch.cumsum(probs, dim=0)
            sigma_idx = int(torch.searchsorted(cumulative, torch.tensor(u), right=False).item())
            sigma_idx = min(sigma_idx, len(self.prior_sigma_values) - 1)
            trans_sigma = self.prior_sigma_values[sigma_idx]
        elif self.prior_sigma_range is not None:
            sigma_gen = self._make_generator(idx, stream_offset=7)
            lo, hi = self.prior_sigma_range
            u = torch.rand(1, generator=sigma_gen).item()
            if self.prior_sigma_log_uniform:
                trans_sigma = math.exp(math.log(lo) + u * (math.log(hi) - math.log(lo)))
            else:
                trans_sigma = lo + u * (hi - lo)
        else:
            trans_sigma = self.translation_sigma

        refine_gen = self._make_generator(idx, stream_offset=8)
        refine_roll = (
            torch.rand(1, generator=refine_gen).item()
            if refine_gen is not None
            else torch.rand(1).item()
        )
        use_local_refine = refine_roll < self.local_refine_prob

        if use_local_refine:
            weights = torch.tensor(self.local_refine_trans_weights, dtype=torch.float32)
            probs = weights / weights.sum()
            u = torch.rand(1, generator=refine_gen).item()
            cumulative = torch.cumsum(probs, dim=0)
            sigma_idx = int(torch.searchsorted(cumulative, torch.tensor(u), right=False).item())
            sigma_idx = min(sigma_idx, len(self.local_refine_trans_sigmas) - 1)
            trans_sigma = self.local_refine_trans_sigmas[sigma_idx]

            h_lo, h_hi = self.local_refine_horizon_range
            horizon = h_lo + torch.rand(1, generator=refine_gen).item() * (h_hi - h_lo)
            t = 1.0 - horizon

            if self.local_refine_mode == "torsion":
                R_1 = quaternion_to_matrix(q_1)
                atom_pos_1 = (
                    torch.einsum("nij,nj->ni", R_1[frag_id_for_atoms], local_pos)
                    + T_1[frag_id_for_atoms]
                )
                center = atom_pos_1.mean(dim=0, keepdim=True)
                R_delta = _random_rotation_matrix(
                    self.local_refine_rot_sigma_deg,
                    refine_gen,
                    T_1.dtype,
                )
                trans = torch.randn(3, generator=refine_gen, dtype=T_1.dtype) * trans_sigma
                atom_pos_pert = (atom_pos_1 - center) @ R_delta.T + center + trans.view(1, 3)
                atom_pos_pert = _apply_torsion_perturb(
                    atom_pos_pert,
                    ligand,
                    torsion_degrees=self.local_refine_torsion_degrees,
                    max_rot_bonds=self.local_refine_max_torsion_bonds,
                    prefer_side=self.local_refine_torsion_side,
                    gen=refine_gen,
                )
                T_t, q_t = _recover_frag_state(atom_pos_pert, local_pos, frag_id_for_atoms, n_frags)
            else:
                T_t = (
                    T_1
                    + torch.randn(
                        T_1.shape,
                        generator=refine_gen,
                        dtype=T_1.dtype,
                    )
                    * trans_sigma
                )

                rot_sigma = math.radians(self.local_refine_rot_sigma_deg)
                if rot_sigma > 0:
                    rot_vec = (
                        torch.randn(
                            n_frags,
                            3,
                            generator=refine_gen,
                            dtype=T_1.dtype,
                        )
                        * rot_sigma
                    )
                    delta_q = axis_angle_to_quaternion(rot_vec)
                    q_t = quaternion_multiply(delta_q, q_1)
                else:
                    q_t = q_1.clone()
            q_t[single_mask] = identity_q[single_mask]

            v_t = (T_1 - T_t) / horizon
            omega_t = compute_angular_velocity(q_t, q_1, frag_sizes=frag_sizes) / horizon
            targets = {"T_t": T_t, "q_t": q_t, "v_t": v_t, "omega_t": omega_t}
        else:
            T_0, q_0 = sample_prior_poses(
                n_frags,
                pocket_center=torch.zeros(3, dtype=T_1.dtype),
                translation_sigma=trans_sigma,
                frag_sizes=frag_sizes,
                generator=prior_gen,
            )

            if self.pose_objective == "vp_flow":
                targets = compute_vp_flow_targets(T_0, q_0, T_1, q_1, t, frag_sizes=frag_sizes)
            elif self.pose_objective == "vp_score":
                targets = compute_vp_score_targets(
                    T_0,
                    q_0,
                    T_1,
                    q_1,
                    t,
                    translation_sigma=trans_sigma,
                    frag_sizes=frag_sizes,
                )
            elif self.pose_objective == "vp_score_full":
                rot_noise = torch.randn(
                    n_frags,
                    3,
                    generator=prior_gen,
                    dtype=T_1.dtype,
                )
                targets = compute_vp_score_full_targets(
                    T_0,
                    T_1,
                    q_1,
                    t,
                    translation_sigma=trans_sigma,
                    rot_noise=rot_noise,
                    rot_sigma_max=self.score_rot_sigma_max,
                    score_alpha_min=self.score_alpha_min,
                    frag_sizes=frag_sizes,
                )
            else:
                targets = compute_flow_matching_targets(
                    T_0, q_0, T_1, q_1, t, frag_sizes=frag_sizes
                )

        # --- Update node coordinates for flow matching state --------------
        node_coords = graph["node_coords"].clone()

        # Pocket-center the static protein/ligand coords
        node_coords -= pocket_center

        # Fragment nodes: use T_t
        frag_slice = graph["lig_frag_slice"]
        frag_start, frag_end = frag_slice[0].item(), frag_slice[1].item()
        node_coords[frag_start:frag_end] = targets["T_t"]

        # Ligand atom nodes: use R_t @ local_pos + T_t
        atom_slice = graph["lig_atom_slice"]
        atom_start = atom_slice[0].item()
        R_t = quaternion_to_matrix(targets["q_t"])
        atom_pos_t = (
            torch.einsum("nij,nj->ni", R_t[frag_id_for_atoms], local_pos)
            + targets["T_t"][frag_id_for_atoms]
        )
        node_coords[atom_start : atom_start + atom_pos_t.shape[0]] = atom_pos_t

        # --- Build output dict --------------------------------------------
        out: dict[str, Tensor] = {}
        for k, v in graph.items():
            if isinstance(v, Tensor):
                out[k] = v
        out["node_coords"] = node_coords

        # Flow matching state
        out["T_frag"] = targets["T_t"]
        out["q_frag"] = targets["q_t"]
        out["v_target"] = targets["v_t"]
        out["omega_target"] = targets["omega_t"]
        out["frag_sizes"] = frag_sizes
        out["T_target"] = T_1
        out["q_target"] = q_1
        out["t"] = torch.tensor([t], dtype=T_1.dtype)
        # Prior σ used for this sample's T_0 noise. Made available so the
        # model can condition on log(σ) (multi-σ training).
        out["prior_sigma"] = torch.tensor([trans_sigma], dtype=T_1.dtype)
        out["pdb_id"] = pdb_id

        # Atom-level velocity target: v_atom = v_frag + omega × r
        r = atom_pos_t - targets["T_t"][frag_id_for_atoms]
        v_atom = targets["v_t"][frag_id_for_atoms] + torch.cross(
            targets["omega_t"][frag_id_for_atoms],
            r,
            dim=-1,
        )
        out["atom_pos_t"] = atom_pos_t
        out["v_atom_target"] = v_atom
        out["local_pos"] = local_pos
        out["frag_id_for_atoms"] = frag_id_for_atoms

        return out


def effdock_collate(batch: list[dict[str, Tensor]]) -> dict[str, Tensor]:
    """Collate a list of dataset samples into a batched dict.

    Concatenates node/edge tensors and adjusts indices (edge_index, slices)
    with appropriate offsets per sample.
    """
    out: dict[str, Tensor] = {}
    keys = [k for k in batch[0] if isinstance(batch[0][k], Tensor)]
    str_keys = [k for k in batch[0] if isinstance(batch[0][k], str)]

    # Categorize keys
    node_keys = [k for k in keys if k.startswith("node_") or k == "node_coords"]
    edge_keys = [k for k in keys if k.startswith("edge_")]
    frag_keys = [
        "T_frag",
        "q_frag",
        "v_target",
        "omega_target",
        "frag_sizes",
        "T_target",
        "q_target",
    ]
    atom_keys = ["atom_pos_t", "v_atom_target", "local_pos", "frag_id_for_atoms"]
    scalar_keys = ["t", "prior_sigma"]
    count_keys = [k for k in keys if k.startswith("num_")]
    slice_keys = [k for k in keys if k.endswith("_slice")]

    # Node tensors: concatenate along dim 0
    for k in node_keys:
        out[k] = torch.cat([b[k] for b in batch], dim=0)

    # Edge tensors: concatenate, offset edge_index
    node_offsets = [0]
    for b in batch[:-1]:
        node_offsets.append(node_offsets[-1] + b["num_nodes"].item())

    if "edge_index" in keys:
        edge_indices = []
        for i, b in enumerate(batch):
            edge_indices.append(b["edge_index"] + node_offsets[i])
        out["edge_index"] = torch.cat(edge_indices, dim=1)

    for k in edge_keys:
        if k == "edge_index":
            continue
        out[k] = torch.cat([b[k] for b in batch], dim=0)

    # Fragment tensors: concatenate, offset frag_id_for_atoms
    frag_offsets = [0]
    for b in batch[:-1]:
        frag_offsets.append(frag_offsets[-1] + b["num_lig_frag"].item())

    for k in frag_keys:
        if k in keys:
            out[k] = torch.cat([b[k] for b in batch], dim=0)

    for k in atom_keys:
        if k not in keys:
            continue
        if k == "frag_id_for_atoms":
            parts = []
            for i, b in enumerate(batch):
                parts.append(b[k] + frag_offsets[i])
            out[k] = torch.cat(parts, dim=0)
        else:
            out[k] = torch.cat([b[k] for b in batch], dim=0)

    # Scalar: stack
    for k in scalar_keys:
        if k in keys:
            out[k] = torch.stack([b[k] for b in batch], dim=0)

    # Counts: stack
    for k in count_keys:
        out[k] = torch.stack([b[k] for b in batch], dim=0)

    # Slices: offset and stack
    for k in slice_keys:
        parts = []
        for i, b in enumerate(batch):
            parts.append(b[k] + node_offsets[i])
        out[k] = torch.stack(parts, dim=0)

    # Batch index for nodes (which sample each node belongs to)
    batch_idx = []
    for i, b in enumerate(batch):
        batch_idx.append(torch.full((b["num_nodes"].item(),), i, dtype=torch.long))
    out["batch"] = torch.cat(batch_idx, dim=0)

    # Batch index for fragments
    frag_batch_idx = []
    for i, b in enumerate(batch):
        frag_batch_idx.append(torch.full((b["num_lig_frag"].item(),), i, dtype=torch.long))
    out["frag_batch"] = torch.cat(frag_batch_idx, dim=0)

    # Batch index for ligand atoms (frag_id_for_atoms has one entry per atom)
    atom_batch_idx = []
    atom_offsets = [0]
    for i, b in enumerate(batch):
        n_atoms = b["frag_id_for_atoms"].shape[0]
        atom_batch_idx.append(torch.full((n_atoms,), i, dtype=torch.long))
        if i < len(batch) - 1:
            atom_offsets.append(atom_offsets[-1] + n_atoms)
    out["atom_batch"] = torch.cat(atom_batch_idx, dim=0)

    # String keys
    for k in str_keys:
        out[k] = [b[k] for b in batch]

    return out


__all__ = ["EFFDockDataset", "effdock_collate"]
