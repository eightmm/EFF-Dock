"""Training loop for EFFDock with DDP, Muon+AdamW, and cosine-cooldown LR."""

from __future__ import annotations

import math
import os
import random
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, DistributedSampler, Sampler, Subset

from effdock.checkpoint import (
    atomic_torch_save,
    load_checkpoint_file,
    load_portable_model_state,
)
from effdock.data.dataset import EFFDockDataset, effdock_collate
from effdock.geometry.flow_matching import integrate_se3_step, sample_prior_poses
from effdock.geometry.se3 import quaternion_to_matrix
from effdock.models.effdock import EFFDock
from effdock.training.losses import flow_matching_loss

# ---------------------------------------------------------------------------
# DDP helpers
# ---------------------------------------------------------------------------


def setup_ddp() -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    # When scripts/train.py masks CUDA_VISIBLE_DEVICES per rank (workaround for
    # cuEquivariance DDP), each process only sees one GPU as cuda:0.
    if world_size > 1:
        cuda_idx = 0 if torch.cuda.device_count() == 1 else local_rank
        torch.cuda.set_device(cuda_idx)
        timeout_minutes = int(os.environ.get("EFFDOCK_DDP_TIMEOUT_MIN", "10"))
        if timeout_minutes <= 0:
            raise ValueError("EFFDOCK_DDP_TIMEOUT_MIN must be positive")
        dist.init_process_group(backend="nccl", timeout=timedelta(minutes=timeout_minutes))
    return rank, local_rank, world_size


def cleanup_ddp(world_size: int) -> None:
    if world_size > 1:
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------


def configure_optimizers(
    model: nn.Module,
    lr: float = 3e-4,
    muon_lr: float = 0.02,
    weight_decay: float = 0.01,
    use_muon: bool = False,
) -> list:
    """Build optimizers with semantic, exhaustive parameter ownership."""
    muon_params, adamw_params = [], []
    modules = dict(model.named_modules())
    assigned: set[int] = set()
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        owner_name, _, leaf_name = name.rpartition(".")
        owner = modules[owner_name]
        use_for_muon = (
            use_muon
            and isinstance(owner, nn.Linear)
            and leaf_name == "weight"
            and param.ndim == 2
            and min(param.shape) >= 2
        )
        if use_for_muon:
            muon_params.append(param)
        else:
            adamw_params.append(param)
        if id(param) in assigned:
            raise RuntimeError(f"trainable parameter assigned twice: {name}")
        assigned.add(id(param))

    expected = {id(param) for param in model.parameters() if param.requires_grad}
    if assigned != expected:
        raise RuntimeError("optimizer parameter grouping is not exhaustive")

    optimizers = []
    if muon_params:
        from torch.optim import Muon

        optimizers.append(
            Muon(
                muon_params,
                lr=muon_lr,
                momentum=0.95,
                weight_decay=weight_decay,
            )
        )
    if adamw_params:
        optimizers.append(AdamW(adamw_params, lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95)))
    return optimizers


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------


def get_warmup_stable_cosine_scheduler(
    optimizer,
    total_steps: int,
    warmup_ratio: float = 0.1,
    cooldown_ratio: float = 0.3,
    min_lr_ratio: float = 0.05,
) -> LambdaLR:
    warmup = int(total_steps * warmup_ratio)
    cooldown = int(total_steps * cooldown_ratio)
    stable = total_steps - warmup - cooldown

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return step / max(warmup, 1)
        elif step < warmup + stable:
            return 1.0
        else:
            progress = (step - warmup - stable) / max(cooldown, 1)
            progress = min(max(progress, 0.0), 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Size-aware DDP sampling
# ---------------------------------------------------------------------------


def _sample_costs(dataset) -> list[float] | None:
    """Return per-sample cost estimates, unwrapping Subset when needed."""
    if hasattr(dataset, "sample_costs"):
        return list(dataset.sample_costs)
    if isinstance(dataset, Subset):
        parent_costs = _sample_costs(dataset.dataset)
        if parent_costs is None:
            return None
        return [parent_costs[i] for i in dataset.indices]
    return None


def _balanced_shard_indices(dataset, *, n_items: int, num_replicas: int, rank: int) -> list[int]:
    """Greedily balance deterministic evaluation indices by static graph cost."""
    if num_replicas <= 0 or rank < 0 or rank >= num_replicas:
        raise ValueError("invalid distributed shard specification")
    indices = list(range(max(int(n_items), 0)))
    if num_replicas == 1:
        return indices

    costs = _sample_costs(dataset)
    if costs is None or len(costs) < len(indices):
        return indices[rank::num_replicas]
    selected_costs = [float(costs[index]) for index in indices]
    if any(not math.isfinite(cost) or cost < 0.0 for cost in selected_costs):
        return indices[rank::num_replicas]

    shards: list[list[int]] = [[] for _ in range(num_replicas)]
    totals = [0.0] * num_replicas
    for index in sorted(indices, key=lambda item: (-selected_costs[item], item)):
        target = min(
            range(num_replicas),
            key=lambda replica: (totals[replica], len(shards[replica]), replica),
        )
        shards[target].append(index)
        totals[target] += selected_costs[index]
    return sorted(shards[rank])


def _all_ranks_finite(value: torch.Tensor, world_size: int) -> bool:
    """Return one synchronized finite decision so DDP ranks never skip alone."""
    finite = torch.isfinite(value.detach()).all().to(dtype=torch.int32)
    if world_size > 1:
        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
    return bool(finite.item())


def _require_finite_metrics(
    metrics: dict[str, float],
    *,
    context: str,
    device: torch.device,
    world_size: int,
) -> None:
    """Fail every rank together when an evaluation emits NaN or infinity."""
    if not metrics:
        raise RuntimeError(f"{context} produced no metrics")
    values = torch.tensor(
        [float(value) for value in metrics.values()],
        dtype=torch.float64,
        device=device,
    )
    if not _all_ranks_finite(values, world_size):
        raise FloatingPointError(f"{context} produced non-finite metrics: {metrics!r}")


class DistributedSizeAwareSampler(Sampler[int]):
    """DDP sampler that balances variable-size graph batches across ranks.

    It shuffles indices, sorts within large random buckets by static graph
    cost, then greedily assigns each global batch group to rank-local batches
    with similar total cost. This avoids one rank receiving several unusually
    large protein-ligand graphs while other ranks wait at DDP barriers.
    """

    def __init__(
        self,
        dataset,
        *,
        batch_size: int,
        num_replicas: int,
        rank: int,
        shuffle: bool = True,
        seed: int = 42,
        bucket_mult: int = 16,
        drop_last: bool = True,
    ) -> None:
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.shuffle = shuffle
        self.seed = int(seed)
        self.bucket_mult = max(int(bucket_mult), 1)
        self.drop_last = bool(drop_last)
        self.epoch = 0
        costs = _sample_costs(dataset)
        self.costs = costs if costs is not None else [1.0] * len(dataset)
        self.group_size = self.batch_size * self.num_replicas
        if self.drop_last:
            self.num_global_batches = len(dataset) // self.group_size
        else:
            self.num_global_batches = math.ceil(len(dataset) / self.group_size)

    def __len__(self) -> int:
        return self.num_global_batches * self.batch_size

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        n = len(self.dataset)
        if self.shuffle:
            gen = torch.Generator()
            gen.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(n, generator=gen).tolist()
        else:
            indices = list(range(n))

        usable = self.num_global_batches * self.group_size
        if self.drop_last:
            indices = indices[:usable]
        elif usable > n:
            # Retain every split member, padding only enough for equal complete
            # batches on every DDP rank. The duplicate tail rotates by epoch.
            if n == 0:
                return iter([])
            repeats = math.ceil((usable - n) / n)
            indices.extend((indices * repeats)[: usable - n])
        bucket_size = self.group_size * self.bucket_mult
        rank_indices: list[int] = []

        for start in range(0, len(indices), bucket_size):
            bucket = indices[start : start + bucket_size]
            bucket.sort(key=lambda i: self.costs[i], reverse=True)
            for group_start in range(0, len(bucket), self.group_size):
                group = bucket[group_start : group_start + self.group_size]
                if len(group) < self.group_size:
                    continue
                bins: list[list[int]] = [[] for _ in range(self.num_replicas)]
                bin_costs = [0.0 for _ in range(self.num_replicas)]
                for idx in group:
                    candidates = [
                        r for r in range(self.num_replicas) if len(bins[r]) < self.batch_size
                    ]
                    r = min(candidates, key=lambda x: bin_costs[x])
                    bins[r].append(idx)
                    bin_costs[r] += self.costs[idx]
                rank_indices.extend(bins[self.rank])

        return iter(rank_indices)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class Trainer:
    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.rank, self.local_rank, self.world_size = setup_ddp()
        # cuda:0 inside each rank (per-rank CUDA_VISIBLE_DEVICES) or local_rank
        # when unmasked (e.g. single-process run with all GPUs visible).
        if torch.cuda.is_available():
            cuda_idx = 0 if torch.cuda.device_count() == 1 else self.local_rank
            self.device = torch.device(f"cuda:{cuda_idx}")
        else:
            self.device = torch.device("cpu")
        self.is_main = self.rank == 0

        torch.manual_seed(cfg["training"].get("seed", 42))
        random.seed(cfg["training"].get("seed", 42))
        np.random.seed(cfg["training"].get("seed", 42))

        # Dirs
        self.output_dir = Path(cfg["logging"]["output_dir"])
        self.ckpt_dir = self.output_dir / "checkpoints"
        if self.is_main:
            self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Data
        self._build_dataloaders()

        # Model
        model_kwargs = {k: v for k, v in cfg["model"].items() if k != "model_type"}
        self.model = EFFDock(**model_kwargs).to(self.device)
        if self.world_size > 1:
            ddp_device = self.device.index
            self.model = DDP(
                self.model,
                device_ids=[ddp_device],
                output_device=ddp_device,
                find_unused_parameters=False,
                gradient_as_bucket_view=True,
            )

        # Training params (needed before _total_steps)
        tcfg = cfg["training"]
        self.global_step = 0
        self.start_epoch = 0
        self.resume_batch_idx = 0
        self._data_pass_epoch = 0
        self._next_batch_idx = 0
        self._epoch_loader_rng_state: torch.Tensor | None = None
        self._resume_loader_rng_state: torch.Tensor | None = None
        self.grad_accum = tcfg.get("gradient_accumulation_steps", 1)

        # Optimizers & schedulers
        raw_model = self.model.module if isinstance(self.model, DDP) else self.model
        self.optimizers = configure_optimizers(
            raw_model,
            lr=tcfg["lr"],
            muon_lr=tcfg.get("muon_lr", 0.02),
            weight_decay=tcfg.get("weight_decay", 0.01),
            use_muon=tcfg.get("use_muon", False),
        )
        total_steps = self._total_steps()
        self.schedulers = [
            get_warmup_stable_cosine_scheduler(
                opt,
                total_steps,
                warmup_ratio=tcfg.get("warmup_ratio", 0.1),
                cooldown_ratio=tcfg.get("cooldown_ratio", 0.3),
                min_lr_ratio=tcfg.get("min_lr_ratio", 0.05),
            )
            for opt in self.optimizers
        ]
        # EMA (used for val/rollout only)
        self.use_ema = tcfg.get("use_ema", True)
        self.ema_decay = tcfg.get("ema_decay", 0.999)
        if self.use_ema:
            from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

            self.ema_model = AveragedModel(
                raw_model,
                multi_avg_fn=get_ema_multi_avg_fn(self.ema_decay),
                use_buffers=True,
            )
        else:
            self.ema_model = None

        self.max_grad_norm = tcfg.get("max_grad_norm", 1.0)
        self.omega_weight = tcfg.get("omega_weight", 1.0)
        self.omega_loss_frame = tcfg.get("omega_loss_frame", "world")
        self.omega_loss_type = tcfg.get("omega_loss_type", "mse")
        self.omega_dir_weight = tcfg.get("omega_dir_weight", 1.0)
        self.omega_mag_weight = tcfg.get("omega_mag_weight", 0.1)
        self.atom_aux_weight = tcfg.get("atom_aux_weight", 0.0)
        self.dg_weight = tcfg.get("dg_weight", 0.0)
        self.pose_objective = cfg["data"].get("pose_objective", "linear_fm")
        self.score_rot_sigma_max = float(cfg["data"].get("score_rot_sigma_max", math.pi))
        self.score_alpha_min = float(cfg["data"].get("score_alpha_min", 0.0))

        # Mixed precision: disabled by default because cuEquivariance's
        # fused_tp kernels only support FP32 inputs (they raise on BF16).
        # TF32 is enabled globally in scripts/train.py and already accelerates
        # the scalar path of the model.
        self.use_amp = tcfg.get("use_amp", False)
        self.amp_dtype = torch.bfloat16
        self.dummy_weight = tcfg.get("dummy_weight", 0.0)
        self.use_time_weighting = tcfg.get("use_time_weighting", True)

        # Wandb (initialized lazily after potential checkpoint load)
        self.use_wandb = cfg["logging"].get("use_wandb", False) and self.is_main
        self.wandb_run_id: str | None = None
        self._wandb_initialized = False
        self._best_rmsd = float("inf")
        self.rollout_selection_metric = str(
            cfg["logging"].get("rollout_selection_metric", "rollout/rmsd_median")
        )
        self.rollout_selection_mode = str(cfg["logging"].get("rollout_selection_mode", "min"))
        if self.rollout_selection_mode not in ("min", "max"):
            raise ValueError("logging.rollout_selection_mode must be 'min' or 'max'")
        self._best_selection_value = (
            float("inf") if self.rollout_selection_mode == "min" else -float("inf")
        )

    # ---- Data ----

    def _build_dataloaders(self) -> None:
        dcfg = self.cfg["data"]
        tcfg = self.cfg["training"]
        overfit_batches = int(tcfg.get("overfit_batches", 0))
        overfit_mode = overfit_batches > 0
        bs = tcfg["batch_size"]
        ds_kwargs = dict(
            root=dcfg["data_dir"],
            pocket_cutoff=dcfg.get("pocket_cutoff", 8.0),
            pocket_jitter_sigma=dcfg.get("pocket_jitter_sigma", 0.0),
            pocket_cutoff_noise=dcfg.get("pocket_cutoff_noise", 0.0),
            translation_sigma=dcfg.get("prior_sigma", 5.0),
            max_atoms=dcfg.get("max_atoms", 80),
            max_frags=dcfg.get("max_frags", 20),
            min_atoms=dcfg.get("min_atoms", 5),
            min_protein_res=dcfg.get("min_protein_res", 50),
            rotation_augmentation=dcfg.get("rotation_augmentation", "none"),
            deterministic=dcfg.get("deterministic", False),
            seed=tcfg.get("seed", 42),
            # Receptor augmentation (apo / AF2-predicted alt receptor swap).
            # Reads alt_receptor_root + alt_receptor_mapping JSON; train-only.
            receptor_aug_prob=dcfg.get("receptor_aug_prob", 0.0),
            alt_receptor_root=dcfg.get("alt_receptor_root"),
            alt_receptor_mapping=dcfg.get("alt_receptor_mapping"),
            # Range-based pocket cutoff and prior σ sampling (None → use
            # legacy single-value behavior). Setting these lets the model
            # see a wider distribution at training time and stay calibrated
            # under any single-value choice at inference.
            pocket_cutoff_range=tuple(dcfg["pocket_cutoff_range"])
            if dcfg.get("pocket_cutoff_range")
            else None,
            prior_sigma_range=tuple(dcfg["prior_sigma_range"])
            if dcfg.get("prior_sigma_range")
            else None,
            prior_sigma_log_uniform=dcfg.get("prior_sigma_log_uniform", True),
            prior_sigma_values=tuple(dcfg["prior_sigma_values"])
            if dcfg.get("prior_sigma_values")
            else None,
            prior_sigma_weights=tuple(dcfg["prior_sigma_weights"])
            if dcfg.get("prior_sigma_weights")
            else None,
            time_distribution=dcfg.get("time_distribution", "uniform"),
            time_early_replay_weights=tuple(
                dcfg.get("time_early_replay_weights", (0.80, 0.10, 0.10))
            ),
            time_early_replay_max=dcfg.get("time_early_replay_max", 0.30),
            pose_objective=dcfg.get("pose_objective", "linear_fm"),
            score_rot_sigma_max=dcfg.get("score_rot_sigma_max", math.pi),
            score_alpha_min=dcfg.get("score_alpha_min", 0.0),
            local_refine_prob=dcfg.get("local_refine_prob", 0.0),
            local_refine_trans_sigmas=tuple(dcfg["local_refine_trans_sigmas"])
            if dcfg.get("local_refine_trans_sigmas")
            else None,
            local_refine_trans_weights=tuple(dcfg["local_refine_trans_weights"])
            if dcfg.get("local_refine_trans_weights")
            else None,
            local_refine_rot_sigma_deg=dcfg.get("local_refine_rot_sigma_deg", 15.0),
            local_refine_horizon_range=tuple(dcfg["local_refine_horizon_range"])
            if dcfg.get("local_refine_horizon_range")
            else (0.12, 0.35),
            local_refine_mode=dcfg.get("local_refine_mode", "fragment"),
            local_refine_torsion_degrees=tuple(dcfg["local_refine_torsion_degrees"])
            if dcfg.get("local_refine_torsion_degrees")
            else None,
            local_refine_max_torsion_bonds=dcfg.get("local_refine_max_torsion_bonds", 2),
            local_refine_torsion_side=dcfg.get("local_refine_torsion_side", "smaller"),
        )

        split_file = dcfg.get("split_file")
        DatasetClass = EFFDockDataset

        # Val dataset: no augmentation, no jitter (deterministic crystal eval)
        val_kwargs = dict(ds_kwargs)
        val_kwargs["rotation_augmentation"] = "none"
        val_kwargs["pocket_jitter_sigma"] = 0.0
        val_kwargs["pocket_cutoff_noise"] = 0.0
        val_kwargs["receptor_aug_prob"] = 0.0  # crystal-only val
        val_kwargs["pocket_cutoff_range"] = None
        val_kwargs["prior_sigma_range"] = None
        val_kwargs["prior_sigma_values"] = None
        val_kwargs["prior_sigma_weights"] = None
        val_kwargs["time_distribution"] = dcfg.get(
            "val_time_distribution", dcfg.get("time_distribution", "uniform")
        )
        val_kwargs["local_refine_prob"] = 0.0

        if split_file is not None:
            train_ds = DatasetClass(split_file=split_file, split_key="train", **ds_kwargs)
            val_ds = DatasetClass(split_file=split_file, split_key="val", **val_kwargs)
            if len(val_ds) == 0:
                val_ds = None
        else:
            full_ds = DatasetClass(**ds_kwargs)
            val_ratio = dcfg.get("val_split", 0.05)
            n_val = int(len(full_ds) * val_ratio)
            n_train = len(full_ds) - n_val
            if n_val > 0:
                train_ds, val_ds = torch.utils.data.random_split(
                    full_ds,
                    [n_train, n_val],
                    generator=torch.Generator().manual_seed(self.cfg["training"].get("seed", 42)),
                )
            else:
                train_ds, val_ds = full_ds, None

        expected_train_samples = dcfg.get("expected_train_samples")
        if expected_train_samples is not None and len(train_ds) != int(expected_train_samples):
            raise RuntimeError(
                "filtered train split size changed: "
                f"expected {int(expected_train_samples)}, got {len(train_ds)}"
            )
        expected_val_samples = dcfg.get("expected_val_samples")
        actual_val_samples = len(val_ds) if val_ds is not None else 0
        if expected_val_samples is not None and actual_val_samples != int(expected_val_samples):
            raise RuntimeError(
                "filtered validation split size changed: "
                f"expected {int(expected_val_samples)}, got {actual_val_samples}"
            )

        if overfit_mode:
            max_samples = min(len(train_ds), bs * overfit_batches)
            if max_samples <= 0:
                raise ValueError("overfit_batches > 0 but the training dataset is empty.")
            train_ds = Subset(train_ds, list(range(max_samples)))

        if self.is_main:
            train_msg = f"Dataset: train={len(train_ds)}, val={len(val_ds) if val_ds else 0}"
            if overfit_mode:
                train_msg += f" (strict overfit subset, {len(train_ds)} samples)"
            print(train_msg)

        nw = dcfg.get("num_workers", 4)
        collate_fn = effdock_collate

        if self.world_size > 1:
            if tcfg.get("size_aware_batches", True) and not overfit_mode:
                self.train_sampler = DistributedSizeAwareSampler(
                    train_ds,
                    batch_size=bs,
                    num_replicas=self.world_size,
                    rank=self.rank,
                    shuffle=True,
                    seed=tcfg.get("seed", 42),
                    bucket_mult=tcfg.get("size_aware_bucket_mult", 16),
                    drop_last=tcfg.get("size_aware_drop_last", True),
                )
            else:
                self.train_sampler = DistributedSampler(
                    train_ds,
                    num_replicas=self.world_size,
                    rank=self.rank,
                    shuffle=not overfit_mode,
                    seed=42,
                )
            shuffle = False
        else:
            self.train_sampler = (
                None
                if overfit_mode
                else DistributedSampler(
                    train_ds,
                    num_replicas=1,
                    rank=0,
                    shuffle=True,
                    seed=tcfg.get("seed", 42),
                    drop_last=False,
                )
            )
            shuffle = False

        loader_kwargs = dict(
            collate_fn=collate_fn,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=False,
            prefetch_factor=4 if nw > 0 else None,
        )
        self.train_loader_generator = torch.Generator().manual_seed(tcfg.get("seed", 42))
        self.train_loader = DataLoader(
            train_ds,
            batch_size=bs,
            shuffle=shuffle,
            sampler=self.train_sampler,
            num_workers=nw,
            drop_last=True,
            generator=self.train_loader_generator,
            **loader_kwargs,
        )
        if val_ds is not None:
            if self.world_size > 1:
                self.val_sampler = DistributedSampler(
                    val_ds,
                    num_replicas=self.world_size,
                    rank=self.rank,
                    shuffle=False,
                    drop_last=False,
                )
            else:
                self.val_sampler = None
            self.val_loader = DataLoader(
                val_ds,
                batch_size=bs,
                shuffle=False,
                sampler=self.val_sampler,
                num_workers=nw,
                **loader_kwargs,
            )
        else:
            self.val_loader = None
            self.val_sampler = None

    @staticmethod
    def _dict_batch_to_device(batch: dict, device: torch.device) -> dict:
        """Move all tensors in a dict batch to device."""
        out = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                out[k] = v.to(device, non_blocking=True)
            else:
                out[k] = v
        return out

    def _compute_loss_unified(self, out: dict, batch: dict) -> dict:
        """Compute flow matching loss for unified model."""
        from effdock.training.losses import (
            atom_position_auxiliary_loss,
            compute_time_weight,
            distance_geometry_loss,
        )

        R_t = None
        if self.omega_loss_frame == "body":
            R_t = quaternion_to_matrix(batch["q_frag"])

        # Per-fragment time weight
        if self.use_time_weighting:
            t_per_frag = batch["t"].view(-1)[batch["frag_batch"]]  # [N_frag]
            time_weight = compute_time_weight(t_per_frag)
        else:
            time_weight = None

        counts = torch.tensor(
            [
                batch["frag_sizes"].numel(),
                (batch["frag_sizes"] > 1).sum().item(),
                batch["atom_pos_t"].numel(),
            ],
            dtype=torch.float64,
            device=self.device,
        )
        local_counts = counts.clone()
        if self.world_size > 1:
            dist.all_reduce(counts)
        scales = torch.where(
            counts > 0,
            local_counts * self.world_size / counts,
            torch.zeros_like(counts),
        )

        losses = flow_matching_loss(
            out["v_pred"],
            out["omega_pred"],
            batch["v_target"],
            batch["omega_target"],
            batch["frag_sizes"],
            omega_weight=self.omega_weight,
            R_t=R_t,
            omega_loss_frame=self.omega_loss_frame,
            omega_loss_type=self.omega_loss_type,
            omega_dir_weight=self.omega_dir_weight,
            omega_mag_weight=self.omega_mag_weight,
            time_weight=time_weight,
            P_observable=out.get("P_observable"),
            mean_scale_v=float(scales[0]),
            mean_scale_omega=float(scales[1]),
        )

        # Atom-level auxiliary loss: v_atom = v_frag + omega × r
        if self.atom_aux_weight > 0:
            aux = atom_position_auxiliary_loss(
                out["v_pred"],
                out["omega_pred"],
                batch["v_target"],
                batch["omega_target"],
                atom_pos_t=batch["atom_pos_t"],
                T_frag=batch["T_frag"],
                fragment_id=batch["frag_id_for_atoms"],
                frag_sizes=batch["frag_sizes"],
            )
            aux_loss = aux["loss_atom_aux"] * float(scales[2])
            losses["loss"] = losses["loss"] + self.atom_aux_weight * aux_loss
            losses["loss_atom_aux"] = aux_loss.detach()

        # Distance geometry loss: one-step Euler → pairwise distance MSE
        # (weighted by t² — strict near t=1, loose near t=0)
        if self.dg_weight > 0:
            dg = distance_geometry_loss(
                v_pred=out["v_pred"],
                omega_pred=out["omega_pred"],
                T_t=batch["T_frag"],
                q_t=batch["q_frag"],
                t_per_sample=batch["t"],
                frag_batch=batch["frag_batch"],
                T_target=batch["T_target"],
                q_target=batch["q_target"],
                local_pos=batch["local_pos"],
                frag_id_for_atoms=batch["frag_id_for_atoms"],
                atom_batch=batch["atom_batch"],
                lig_atom_slice=batch["lig_atom_slice"],
                lig_frag_slice=batch["lig_frag_slice"],
            )
            dg_normalizer = torch.tensor(
                float(dg["normalizer"]), dtype=torch.float64, device=self.device
            )
            global_dg_normalizer = dg_normalizer.clone()
            if self.world_size > 1:
                dist.all_reduce(global_dg_normalizer)
            dg_scale = (
                float(dg_normalizer * self.world_size / global_dg_normalizer)
                if global_dg_normalizer > 0
                else 0.0
            )
            dg_loss = dg["loss_dg"] * dg_scale
            losses["loss"] = losses["loss"] + self.dg_weight * dg_loss
            losses["loss_dg"] = dg_loss.detach()

        return losses

    def _total_steps(self) -> int:
        """Total optimizer steps from config (step-based schedule)."""
        tcfg = self.cfg["training"]
        assert "max_steps" in tcfg, "config must specify training.max_steps"
        return int(tcfg["max_steps"])

    # ---- Checkpoint ----

    @staticmethod
    def _validate_resume_config(current: dict, saved: object) -> None:
        """Fail closed when an exact-resume run no longer has the same config."""
        if not isinstance(saved, dict):
            raise RuntimeError("strict resume requires an embedded checkpoint config")
        if saved == current:
            return
        sections = sorted(
            key for key in set(current) | set(saved) if current.get(key) != saved.get(key)
        )
        raise RuntimeError(
            "resume checkpoint config differs from the requested config; "
            f"changed top-level sections={sections}"
        )

    def load_checkpoint(self, path: str) -> None:
        ckpt = load_checkpoint_file(path)
        if self.cfg["training"].get("strict_resume_config", False):
            self._validate_resume_config(self.cfg, ckpt.get("config"))
        raw_model = self.model.module if isinstance(self.model, DDP) else self.model
        load_portable_model_state(raw_model, ckpt["model_state_dict"])
        if self.ema_model is not None and "ema_state_dict" in ckpt:
            load_portable_model_state(self.ema_model, ckpt["ema_state_dict"])
        optimizer_states = ckpt.get("optimizer_state_dicts", [])
        scheduler_states = ckpt.get("scheduler_state_dicts", [])
        if len(optimizer_states) != len(self.optimizers):
            raise RuntimeError(
                "resume checkpoint optimizer layout differs from the current config; "
                "use --init-from for weights-only migration"
            )
        if len(scheduler_states) != len(self.schedulers):
            raise RuntimeError("resume checkpoint scheduler layout differs from the current config")
        for opt, opt_sd in zip(self.optimizers, optimizer_states, strict=True):
            opt.load_state_dict(opt_sd)
        for sched, sched_sd in zip(self.schedulers, scheduler_states, strict=True):
            sched.load_state_dict(sched_sd)
        self.global_step = ckpt.get("step", 0)
        if "next_batch_idx" in ckpt:
            self.start_epoch = int(ckpt["data_pass_epoch"])
            self.resume_batch_idx = int(ckpt["next_batch_idx"])
            self._resume_loader_rng_state = ckpt.get("epoch_loader_rng_state")
        else:
            # Historical checkpoints only support restart at the next data pass.
            self.start_epoch = ckpt.get("epoch", 0) + 1
            self.resume_batch_idx = 0
        self.wandb_run_id = ckpt.get("wandb_run_id")
        self._best_rmsd = float(ckpt.get("best_rmsd", float("inf")))
        saved_selection_metric = ckpt.get("best_selection_metric")
        saved_selection_mode = ckpt.get("best_selection_mode")
        if saved_selection_metric is None:
            if (
                self.rollout_selection_metric == "rollout/rmsd_median"
                and self.rollout_selection_mode == "min"
            ):
                self._best_selection_value = self._best_rmsd
        else:
            if saved_selection_metric != self.rollout_selection_metric:
                raise RuntimeError(
                    "resume checkpoint rollout selection metric differs from config: "
                    f"{saved_selection_metric!r} != {self.rollout_selection_metric!r}"
                )
            if saved_selection_mode != self.rollout_selection_mode:
                raise RuntimeError(
                    "resume checkpoint rollout selection mode differs from config: "
                    f"{saved_selection_mode!r} != {self.rollout_selection_mode!r}"
                )
            self._best_selection_value = float(ckpt["best_selection_value"])
        rank_rng_states = ckpt.get("rank_rng_states")
        if rank_rng_states is not None:
            if not isinstance(rank_rng_states, list) or len(rank_rng_states) != self.world_size:
                raise RuntimeError(
                    "resume checkpoint rank RNG layout differs from the current world size: "
                    f"saved={len(rank_rng_states) if isinstance(rank_rng_states, list) else 'invalid'}, "
                    f"current={self.world_size}"
                )
            rng_state = rank_rng_states[self.rank]
        else:
            if self.world_size > 1 and self.cfg["training"].get("strict_resume_config", False):
                raise RuntimeError(
                    "strict distributed resume requires per-rank RNG states; "
                    "use --init-from for legacy checkpoints"
                )
            rng_state = ckpt.get("rng_state")
        self._restore_rng_state(rng_state)
        if self.is_main:
            print(
                f"Resumed from {path} (data pass {self.start_epoch}, "
                f"next batch {self.resume_batch_idx}, step {self.global_step})"
            )

    def load_model_weights(self, path: str) -> None:
        """Initialize model and EMA weights from a checkpoint without optimizer state."""
        ckpt = load_checkpoint_file(path)
        raw_model = self.model.module if isinstance(self.model, DDP) else self.model
        load_portable_model_state(raw_model, ckpt["model_state_dict"])
        if self.ema_model is not None and "ema_state_dict" in ckpt:
            load_portable_model_state(self.ema_model, ckpt["ema_state_dict"])
        self.global_step = 0
        self.start_epoch = 0
        self.resume_batch_idx = 0
        self.wandb_run_id = None
        if self.is_main:
            step = ckpt.get("step", "unknown")
            print(
                f"Initialized model weights from {path} (source step {step}); optimizer/scheduler reset"
            )

    def _build_checkpoint_state(
        self,
        epoch: int,
        metrics: dict,
        *,
        rank_rng_states: list[dict] | None = None,
    ) -> dict:
        raw_model = self.model.module if isinstance(self.model, DDP) else self.model
        state = {
            "format_version": 1,
            "epoch": self._data_pass_epoch,
            "data_pass_epoch": self._data_pass_epoch,
            "sampler_epoch": self._data_pass_epoch,
            "next_batch_idx": self._next_batch_idx,
            "epoch_loader_rng_state": self._epoch_loader_rng_state,
            "step": self.global_step,
            "model_state_dict": raw_model.state_dict(),
            "optimizer_state_dicts": [o.state_dict() for o in self.optimizers],
            "scheduler_state_dicts": [s.state_dict() for s in self.schedulers],
            "metrics": metrics,
            "config": self.cfg,
            "best_rmsd": self._best_rmsd,
            "best_selection_metric": self.rollout_selection_metric,
            "best_selection_mode": self.rollout_selection_mode,
            "best_selection_value": self._best_selection_value,
            # Keep the legacy rank-0 field readable while exact DDP resume
            # restores the distinct RNG stream for every rank.
            "rng_state": (
                rank_rng_states[0] if rank_rng_states is not None else self._capture_rng_state()
            ),
            "wandb_run_id": self.wandb_run_id,
        }
        if rank_rng_states is not None:
            state["rank_rng_states"] = rank_rng_states
        if self.ema_model is not None:
            state["ema_state_dict"] = self.ema_model.state_dict()
        return state

    @staticmethod
    def _capture_rng_state() -> dict:
        numpy_state = np.random.get_state()
        state = {
            "python": random.getstate(),
            "numpy": {
                "algorithm": numpy_state[0],
                "keys": numpy_state[1].tolist(),
                "position": numpy_state[2],
                "has_gauss": numpy_state[3],
                "cached_gaussian": numpy_state[4],
            },
            "torch": torch.random.get_rng_state(),
        }
        if torch.cuda.is_available():
            state["cuda"] = torch.cuda.get_rng_state_all()
        return state

    @staticmethod
    def _restore_rng_state(state: dict | None) -> None:
        if not state:
            return
        random.setstate(state["python"])
        numpy_state = state["numpy"]
        np.random.set_state(
            (
                numpy_state["algorithm"],
                np.asarray(numpy_state["keys"], dtype=np.uint32),
                numpy_state["position"],
                numpy_state["has_gauss"],
                numpy_state["cached_gaussian"],
            )
        )
        torch.random.set_rng_state(state["torch"])
        if torch.cuda.is_available() and "cuda" in state:
            torch.cuda.set_rng_state_all(state["cuda"])

    def _collect_rank_rng_states(self) -> list[dict] | None:
        """Collect exact rank-local RNG streams at a synchronized save point."""
        local_state = self._capture_rng_state()
        if self.world_size <= 1:
            return [local_state]
        gathered: list[dict] | None = [None] * self.world_size if self.is_main else None  # type: ignore[list-item]
        dist.gather_object(local_state, gathered, dst=0)
        return gathered

    def _save_latest(self, epoch: int, metrics: dict) -> None:
        """Save latest.pt (overwritten every val). Used for resume."""
        rank_rng_states = self._collect_rank_rng_states()
        if not self.is_main:
            return
        assert rank_rng_states is not None
        state = self._build_checkpoint_state(
            epoch,
            metrics,
            rank_rng_states=rank_rng_states,
        )
        atomic_torch_save(state, self.ckpt_dir / "latest.pt")

    def _save_rollout(self, epoch: int, metrics: dict) -> None:
        """Save a named rollout checkpoint + update best.pt if RMSD improved."""
        # Update selection state before serialization so named and best
        # checkpoints and every rank carry the exact threshold used on resume.
        rmsd = metrics.get("rollout/rmsd_median", float("inf"))
        if math.isfinite(float(rmsd)) and rmsd < self._best_rmsd:
            self._best_rmsd = rmsd
        if self.rollout_selection_metric not in metrics:
            raise KeyError(
                f"rollout selection metric missing from evaluation: {self.rollout_selection_metric}"
            )
        selection_value = float(metrics[self.rollout_selection_metric])
        if not math.isfinite(selection_value):
            raise FloatingPointError(
                f"non-finite rollout selection metric {self.rollout_selection_metric}"
            )
        improved = (
            selection_value < self._best_selection_value
            if self.rollout_selection_mode == "min"
            else selection_value > self._best_selection_value
        )
        if improved:
            self._best_selection_value = selection_value

        rank_rng_states = self._collect_rank_rng_states()
        if not self.is_main:
            return
        assert rank_rng_states is not None
        state = self._build_checkpoint_state(
            epoch,
            metrics,
            rank_rng_states=rank_rng_states,
        )
        path = self.ckpt_dir / f"rollout_step{self.global_step:07d}.pt"
        atomic_torch_save(state, path)

        if improved:
            atomic_torch_save(state, self.ckpt_dir / "best.pt")
            print(
                f"  New best {self.rollout_selection_metric}: {selection_value:.6g} → saved best.pt"
            )

    def _init_wandb(self) -> None:
        if not self.use_wandb or self._wandb_initialized:
            return
        try:
            import wandb

            init_kwargs: dict = dict(
                project=self.cfg["logging"].get("wandb_project", "eff-dock"),
                config=self.cfg,
            )
            run_name = self.cfg["logging"].get("wandb_run_name")
            if run_name is not None:
                init_kwargs["name"] = run_name
            # Resume existing run if we have a saved run_id
            if self.wandb_run_id is not None:
                init_kwargs["id"] = self.wandb_run_id
                init_kwargs["resume"] = "must"

            wandb.init(**init_kwargs)
            self.wandb_run_id = wandb.run.id  # type: ignore[union-attr]

            # Define metric groupings — hide internal step counter
            wandb.define_metric("global_step", hidden=True)
            wandb.define_metric("step/*", step_metric="global_step")
            wandb.define_metric("epoch/*", step_metric="global_step")
            wandb.define_metric("val/*", step_metric="global_step")
            wandb.define_metric("rollout/*", step_metric="global_step")
            wandb.define_metric("meta/*", step_metric="global_step")

            self._wandb_initialized = True
        except Exception as e:
            print(f"WARNING: wandb init failed: {e}")
            self.use_wandb = False

    # ---- Train ----

    def train(self) -> None:
        self._init_wandb()

        tcfg = self.cfg["training"]
        lcfg = self.cfg["logging"]
        log_every = lcfg.get("log_every", 50)
        val_every = lcfg.get("val_every", 0)
        rollout_every = lcfg.get("rollout_every", 0)
        overfit_mode = tcfg.get("overfit_batches", 0) > 0

        total_steps = self._total_steps()
        if self.is_main:
            effective_bs = tcfg["batch_size"] * self.grad_accum * self.world_size
            print(
                f"Training: total_steps={total_steps}, {len(self.train_loader)} batches/data_pass"
            )
            print(f"  grad_accum={self.grad_accum}, effective_bs={effective_bs}")

        # data_pass = one full iteration through the train dataset (was called
        # "epoch" before we moved to step-based). Used for DistributedSampler
        # seeding and log grouping.
        epoch = self.start_epoch
        epoch_loss = 0.0
        epoch_steps = 0
        avg_loss = 0.0
        final_checkpoint_metrics: dict[str, float] | None = None

        try:
            if lcfg.get("eval_on_start", False) and self.global_step == 0:
                initial_metrics: dict[str, float] = {}
                if val_every > 0:
                    initial_metrics.update(self._validate(epoch))
                if rollout_every > 0:
                    initial_metrics.update(self._validate_rollout(epoch))
                    self._save_rollout(epoch, initial_metrics)
                if initial_metrics:
                    self._save_latest(epoch, initial_metrics)

            while self.global_step < total_steps:
                if self.train_sampler is not None:
                    self.train_sampler.set_epoch(epoch)
                self._data_pass_epoch = epoch
                self._next_batch_idx = self.resume_batch_idx if epoch == self.start_epoch else 0
                if epoch == self.start_epoch and self._resume_loader_rng_state is not None:
                    self.train_loader_generator.set_state(self._resume_loader_rng_state)
                self._epoch_loader_rng_state = self.train_loader_generator.get_state()

                self.model.train()
                epoch_loss = 0.0
                epoch_loss_v = 0.0
                epoch_loss_w = 0.0
                epoch_cos_v = 0.0
                epoch_cos_w = 0.0
                epoch_steps = 0
                t0 = time.time()

                for opt in self.optimizers:
                    opt.zero_grad()

                max_batches = tcfg.get("overfit_batches", 0) if overfit_mode else 0
                for batch_idx, batch in enumerate(self.train_loader):
                    if epoch == self.start_epoch and batch_idx < self.resume_batch_idx:
                        continue
                    if overfit_mode and batch_idx >= max_batches:
                        break

                    batch = self._dict_batch_to_device(batch, self.device)
                    with torch.autocast(
                        device_type="cuda", dtype=self.amp_dtype, enabled=self.use_amp
                    ):
                        out = self.model(batch)
                    losses = self._compute_loss_unified(out, batch)
                    raw_loss = losses["loss"]
                    if not _all_ranks_finite(raw_loss, self.world_size):
                        message = f"non-finite loss on at least one rank at E{epoch} B{batch_idx}"
                        for opt in self.optimizers:
                            opt.zero_grad()
                        if tcfg.get("fail_on_nonfinite", False):
                            raise FloatingPointError(message)
                        if self.is_main:
                            print(f"  WARNING: {message}; skipping on every rank")
                        continue
                    loss = raw_loss / self.grad_accum
                    loss.backward()

                    step_loss = losses["loss"].item()
                    epoch_loss += step_loss
                    epoch_loss_v += losses["loss_v"].item()
                    epoch_loss_w += losses["loss_omega"].item()
                    epoch_cos_v += losses["cos_v"].item()
                    epoch_cos_w += losses["cos_omega"].item()
                    epoch_steps += 1

                    if (batch_idx + 1) % self.grad_accum == 0:
                        grad_norm_tensor = nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.max_grad_norm if self.max_grad_norm > 0 else float("inf"),
                        )
                        if not _all_ranks_finite(grad_norm_tensor, self.world_size):
                            message = (
                                "non-finite gradient norm on at least one rank "
                                f"at E{epoch} B{batch_idx}"
                            )
                            for opt in self.optimizers:
                                opt.zero_grad()
                            if tcfg.get("fail_on_nonfinite", False):
                                raise FloatingPointError(message)
                            if self.is_main:
                                print(f"  WARNING: {message}; skipping on every rank")
                            continue
                        grad_norm = grad_norm_tensor.item()
                        for opt in self.optimizers:
                            opt.step()
                            opt.zero_grad()
                        for sched in self.schedulers:
                            sched.step()
                        self.global_step += 1
                        self._next_batch_idx = batch_idx + 1
                        if self.ema_model is not None:
                            raw = self.model.module if isinstance(self.model, DDP) else self.model
                            self.ema_model.update_parameters(raw)
                        if self.global_step >= total_steps:
                            # Preserve every final metric in latest.pt; the old
                            # unconditional final save used to overwrite them.
                            final_checkpoint_metrics = {
                                "train_loss": epoch_loss / max(epoch_steps, 1)
                            }
                            if val_every > 0:
                                val_metrics = self._validate(epoch)
                                final_checkpoint_metrics.update(val_metrics)
                            if rollout_every > 0:
                                rollout_metrics = self._validate_rollout(epoch)
                                final_checkpoint_metrics.update(rollout_metrics)
                                self._save_rollout(
                                    epoch,
                                    final_checkpoint_metrics,
                                )
                            self._save_latest(epoch, final_checkpoint_metrics)
                            break
                    else:
                        grad_norm = None

                    # Logging (triggered on optimizer step boundaries)
                    log_trigger = (
                        self.is_main
                        and log_every > 0
                        and grad_norm is not None  # optimizer step just happened
                        and self.global_step % log_every == 0
                    )
                    if log_trigger:
                        avg = epoch_loss / epoch_steps
                        lr_vals = [opt.param_groups[0]["lr"] for opt in self.optimizers]
                        print(
                            f"  [S{self.global_step}] loss={step_loss:.4f} avg={avg:.4f} "
                            f"loss_v={losses['loss_v'].item():.4f} loss_w={losses['loss_omega'].item():.4f} "
                            f"lr={lr_vals}"
                        )
                        if self.use_wandb:
                            import wandb

                            log_dict = {
                                "step/loss": step_loss,
                                "step/loss_v": losses["loss_v"].item(),
                                "step/loss_omega": losses["loss_omega"].item(),
                                "step/cos_v": losses["cos_v"].item(),
                                "step/cos_omega": losses["cos_omega"].item(),
                                "meta/lr_adamw": lr_vals[-1],
                                "meta/epoch": epoch,
                            }
                            if len(lr_vals) > 1:
                                log_dict["meta/lr_muon"] = lr_vals[0]
                            if grad_norm is not None:
                                log_dict["step/grad_norm"] = grad_norm
                            for extra_key in (
                                "loss_omega_dir",
                                "loss_omega_mag",
                                "loss_atom_aux",
                                "loss_dg",
                                "cos_omega_world",
                            ):
                                if extra_key in losses:
                                    log_dict[f"step/{extra_key}"] = losses[extra_key].item()
                            wandb.log(log_dict, step=self.global_step)

                    val_trigger = (
                        val_every > 0 and self.global_step > 0 and self.global_step % val_every == 0
                    )
                    rollout_trigger = (
                        rollout_every > 0
                        and self.global_step > 0
                        and self.global_step % rollout_every == 0
                    )
                    if val_trigger or rollout_trigger:
                        checkpoint_metrics = {"train_loss": epoch_loss / max(epoch_steps, 1)}
                        if val_trigger:
                            checkpoint_metrics.update(self._validate(epoch))
                        if rollout_trigger:
                            checkpoint_metrics.update(self._validate_rollout(epoch))
                            self._save_rollout(epoch, checkpoint_metrics)
                        # Save only after rollout has updated the registered best
                        # threshold, so latest.pt is a lossless resume point.
                        self._save_latest(epoch, checkpoint_metrics)

                # Flush leftover gradients at epoch end
                n_batches = batch_idx + 1 if epoch_steps > 0 else 0
                if n_batches > 0 and n_batches % self.grad_accum != 0:
                    leftover_grad_norm = nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.max_grad_norm if self.max_grad_norm > 0 else float("inf"),
                    )
                    if not _all_ranks_finite(leftover_grad_norm, self.world_size):
                        message = (
                            f"non-finite leftover gradient norm on at least one rank at E{epoch}"
                        )
                        for opt in self.optimizers:
                            opt.zero_grad()
                        if tcfg.get("fail_on_nonfinite", False):
                            raise FloatingPointError(message)
                        if self.is_main:
                            print(f"  WARNING: {message}; skipping on every rank")
                    else:
                        for opt in self.optimizers:
                            opt.step()
                            opt.zero_grad()
                        for sched in self.schedulers:
                            sched.step()
                        self.global_step += 1

                # Epoch summary
                elapsed = time.time() - t0
                avg_loss = epoch_loss / epoch_steps if epoch_steps > 0 else float("nan")
                if self.is_main:
                    if epoch_steps > 0:
                        avg_v = epoch_loss_v / epoch_steps
                        avg_w = epoch_loss_w / epoch_steps
                        cv = epoch_cos_v / epoch_steps
                        cw = epoch_cos_w / epoch_steps
                        print(
                            f"Epoch {epoch} done: loss={avg_loss:.4f} v={avg_v:.4f} w={avg_w:.4f} "
                            f"cos_v={cv:.3f} cos_w={cw:.3f} ({elapsed:.1f}s)"
                        )
                        # Epoch-level wandb logging removed — step-level (every 50 steps)
                        # provides finer granularity; epoch averages add chart clutter.
                    else:
                        print(f"Epoch {epoch}: ALL BATCHES SKIPPED ({elapsed:.1f}s)")

                if overfit_mode and (epoch + 1) % 50 == 0:
                    self._save_latest(epoch, {"train_loss": avg_loss})

                if self.global_step < total_steps:
                    epoch += 1
                    self.resume_batch_idx = 0
                    self._resume_loader_rng_state = None
                    self._data_pass_epoch = epoch
                    self._next_batch_idx = 0

            # Final save
            if final_checkpoint_metrics is None:
                final_checkpoint_metrics = {"train_loss": avg_loss}
                self._save_latest(epoch, final_checkpoint_metrics)
            if self.is_main:
                print("Training complete.")

        except KeyboardInterrupt:
            if self.is_main:
                print(
                    "Interrupted. Keeping the last synchronized latest.pt; "
                    "an asynchronous rank-local save would not be exactly resumable."
                )
        finally:
            cleanup_ddp(self.world_size)
            if self.use_wandb:
                import wandb

                wandb.finish()

    # ---- Validation ----

    def _eval_model(self) -> nn.Module:
        """Return EMA model for evaluation if enabled, else unwrapped live model."""
        if self.ema_model is not None:
            return self.ema_model
        return self.model.module if isinstance(self.model, DDP) else self.model

    def _validate(self, epoch: int) -> dict[str, float]:
        if self.val_loader is None:
            return {}

        em = self._eval_model()
        em.train(False)
        total_loss = 0.0
        total_v = 0.0
        total_w = 0.0
        n = 0
        n_samples = 0

        with torch.no_grad():
            for batch in self.val_loader:
                batch = self._dict_batch_to_device(batch, self.device)
                with torch.autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self.use_amp):
                    out = em(batch)
                losses = self._compute_loss_unified(out, batch)
                total_loss += losses["loss"].item()
                total_v += losses["loss_v"].item()
                total_w += losses["loss_omega"].item()
                n += 1
                n_samples += int(batch["t"].shape[0])

        if self.world_size > 1:
            tensors = torch.tensor(
                [total_loss, total_v, total_w, n, n_samples],
                dtype=torch.float64,
                device=self.device,
            )
            dist.all_reduce(tensors)
            total_loss, total_v, total_w, n, n_samples = tensors.tolist()

        if self.cfg["logging"].get("require_exact_eval_coverage", False):
            expected_samples = len(self.val_loader.dataset)
            if int(n_samples) != expected_samples:
                raise RuntimeError(
                    "validation coverage mismatch: "
                    f"expected {expected_samples}, evaluated {int(n_samples)}"
                )

        avg_loss = total_loss / max(n, 1)
        avg_v = total_v / max(n, 1)
        avg_w = total_w / max(n, 1)

        metrics = {
            "val_loss": avg_loss,
            "val_loss_v": avg_v,
            "val_loss_omega": avg_w,
        }
        _require_finite_metrics(
            metrics,
            context=f"validation at step {self.global_step}",
            device=self.device,
            world_size=self.world_size,
        )

        if self.is_main:
            print(
                f"  [Val E{epoch} S{self.global_step}] loss={avg_loss:.4f} v={avg_v:.4f} w={avg_w:.4f}"
            )
            if self.use_wandb:
                import wandb

                wandb.log(
                    {
                        "val/loss": avg_loss,
                        "val/loss_v": avg_v,
                        "val/loss_omega": avg_w,
                    },
                    step=self.global_step,
                )

        self.model.train()
        return metrics

    @torch.no_grad()
    def _rollout_single_unified(
        self,
        raw_model: nn.Module,
        sample: dict[str, torch.Tensor | str],
        *,
        sigma: float,
        num_steps: int,
        time_schedule: str,
        schedule_power: float,
        seed: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        from effdock.inference.sampler import (
            build_time_grid,
            vp_score_noise_to_velocity,
            vp_score_rotation_noise_to_angular_velocity,
        )

        n_frag = sample["num_lig_frag"].item()
        frag_sizes = sample["frag_sizes"].to(self.device)
        frag_id = sample["frag_id_for_atoms"].to(self.device)
        local_pos = sample["local_pos"].to(self.device)

        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)
        T, q = sample_prior_poses(
            n_frag,
            torch.zeros(3),
            sigma,
            frag_sizes=frag_sizes.cpu(),
            dtype=torch.float32,
            generator=gen,
        )
        T, q = T.to(self.device), q.to(self.device)

        time_grid = build_time_grid(
            num_steps,
            schedule=time_schedule,
            power=schedule_power,
            device=self.device,
            dtype=torch.float32,
        )

        batch = effdock_collate([sample])
        batch_gpu = {
            k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        frag_slice = batch_gpu["lig_frag_slice"][0]
        frag_start, frag_end = frag_slice[0].item(), frag_slice[1].item()
        atom_slice = batch_gpu["lig_atom_slice"][0]
        atom_start = atom_slice[0].item()

        for step_idx in range(num_steps):
            t_val = time_grid[step_idx]
            dt = time_grid[step_idx + 1] - time_grid[step_idx]

            R = quaternion_to_matrix(q)
            atom_pos = torch.einsum("nij,nj->ni", R[frag_id], local_pos) + T[frag_id]

            node_coords = batch_gpu["node_coords"].clone()
            node_coords[frag_start:frag_end] = T
            node_coords[atom_start : atom_start + atom_pos.shape[0]] = atom_pos

            batch_gpu["node_coords"] = node_coords
            batch_gpu["T_frag"] = T
            batch_gpu["q_frag"] = q
            batch_gpu["frag_sizes"] = frag_sizes
            batch_gpu["t"] = t_val.view(1, 1)

            with torch.autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self.use_amp):
                out = raw_model(batch_gpu)
            if self.pose_objective in ("vp_score", "vp_score_full"):
                sigma_per_frag = torch.full(
                    (n_frag,),
                    float(sigma),
                    device=self.device,
                    dtype=T.dtype,
                )
                v_use = vp_score_noise_to_velocity(
                    T,
                    out["v_pred"],
                    t_val,
                    dt,
                    sigma_per_frag,
                    score_alpha_min=self.score_alpha_min,
                )
            else:
                v_use = out["v_pred"]
            if self.pose_objective == "vp_score_full":
                omega_use = vp_score_rotation_noise_to_angular_velocity(
                    q,
                    out["omega_pred"],
                    t_val,
                    dt,
                    rot_sigma_max=self.score_rot_sigma_max,
                    score_alpha_min=self.score_alpha_min,
                    frag_sizes=frag_sizes,
                )
            else:
                omega_use = out["omega_pred"]
            T, q = integrate_se3_step(
                T,
                q,
                v_use,
                omega_use,
                dt,
                frag_sizes=frag_sizes,
            )

        R_final = quaternion_to_matrix(q)
        atom_pos_final = torch.einsum("nij,nj->ni", R_final[frag_id], local_pos) + T[frag_id]

        R_target = quaternion_to_matrix(sample["q_target"].to(self.device))
        true_pos = (
            torch.einsum("nij,nj->ni", R_target[frag_id], local_pos)
            + sample["T_target"].to(self.device)[frag_id]
        )

        return T, atom_pos_final, true_pos

    def _validate_rollout(self, epoch: int) -> dict[str, float]:
        """Run ODE rollout on val set and compute docking metrics."""
        if self.val_loader is None:
            return {}

        from effdock.evaluation.metrics import centroid_distance, frag_centroid_rmsd, ligand_rmsd

        raw_model = self._eval_model()
        raw_model.train(False)

        dcfg = self.cfg["data"]
        lcfg = self.cfg["logging"]
        num_steps = lcfg.get("rollout_steps", 20)
        time_schedule = lcfg.get("rollout_time_schedule", "uniform")
        schedule_power = lcfg.get("rollout_schedule_power", 3.0)
        max_samples = lcfg.get("rollout_max_samples", 0)  # 0 = full val set
        sigma = dcfg.get("prior_sigma", 5.0)
        seed_base = self.cfg["training"].get("seed", 42)

        rmsds, cent_dists, frag_rmsds = [], [], []
        n_done = 0

        # Iterate over the optionally capped val set. Static graph costs keep
        # variable-size rollout work balanced across ranks; seeds remain tied
        # to immutable dataset indices regardless of shard assignment.
        val_ds = self.val_loader.dataset
        n_val = len(val_ds) if max_samples <= 0 else min(len(val_ds), max_samples)
        rollout_indices = _balanced_shard_indices(
            val_ds,
            n_items=n_val,
            num_replicas=self.world_size,
            rank=self.rank,
        )
        progress_every = int(lcfg.get("rollout_progress_every", 0))
        for local_index, i in enumerate(rollout_indices, start=1):
            data = val_ds[i]
            T_pred, atom_pos_pred, true_pos = self._rollout_single_unified(
                raw_model,
                data,
                sigma=sigma,
                num_steps=num_steps,
                time_schedule=time_schedule,
                schedule_power=schedule_power,
                seed=seed_base + i,
            )
            T_target = data["T_target"].to(self.device)

            rmsds.append(ligand_rmsd(atom_pos_pred, true_pos).item())
            cent_dists.append(centroid_distance(atom_pos_pred, true_pos).item())
            frag_rmsds.append(frag_centroid_rmsd(T_pred, T_target).item())
            n_done += 1
            if progress_every > 0 and (
                local_index % progress_every == 0 or local_index == len(rollout_indices)
            ):
                print(
                    f"  [Rollout progress S{self.global_step} rank={self.rank}] "
                    f"{local_index}/{len(rollout_indices)}",
                    flush=True,
                )

        if self.world_size > 1:
            gathered: list[tuple[list, list, list]] = [None] * self.world_size  # type: ignore
            dist.all_gather_object(gathered, (rmsds, cent_dists, frag_rmsds))
            rmsds = [x for shard in gathered for x in shard[0]]
            cent_dists = [x for shard in gathered for x in shard[1]]
            frag_rmsds = [x for shard in gathered for x in shard[2]]
            n_done = len(rmsds)

        if n_done != n_val:
            raise RuntimeError(f"rollout coverage mismatch: expected {n_val}, evaluated {n_done}")
        if n_done == 0:
            raise RuntimeError("rollout validation set is empty")

        rmsds_t = torch.tensor(rmsds)
        metrics = {
            "rollout/rmsd_median": rmsds_t.median().item(),
            "rollout/rmsd_mean": rmsds_t.mean().item(),
            "rollout/rmsd_p25": rmsds_t.quantile(0.25).item(),
            "rollout/rmsd_p75": rmsds_t.quantile(0.75).item(),
            "rollout/success_2A": (rmsds_t < 2.0).float().mean().item(),
            "rollout/success_5A": (rmsds_t < 5.0).float().mean().item(),
            "rollout/centroid_dist": torch.tensor(cent_dists).mean().item(),
            "rollout/frag_rmsd": torch.tensor(frag_rmsds).mean().item(),
        }
        _require_finite_metrics(
            metrics,
            context=f"rollout at step {self.global_step}",
            device=self.device,
            world_size=self.world_size,
        )

        if self.is_main:
            print(
                f"  [Rollout E{epoch} S{self.global_step}] "
                f"RMSD={metrics['rollout/rmsd_median']:.2f}A (median) "
                f"<2A={metrics['rollout/success_2A']:.1%} "
                f"<5A={metrics['rollout/success_5A']:.1%} "
                f"({n_done} samples, {num_steps} steps)"
            )
            if self.use_wandb:
                import wandb

                wandb.log(metrics, step=self.global_step)

        self.model.train()
        return metrics


__all__ = ["Trainer"]
