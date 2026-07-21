#!/usr/bin/env python3
"""Train the docking-graph pose confidence model."""

from __future__ import annotations

import argparse
import json
import os
import random
from datetime import timedelta
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from effdock.confidence.dataset import (
    DEFAULT_CONFIDENCE_POSE_TAG,
    LigandPoseConfidenceDataset,
    collate_complexes,
    to_device,
)
from effdock.confidence.losses import pose_confidence_loss
from effdock.confidence.model import DockingGraphPoseConfidence
from effdock.confidence.selectors import select_confidence_poses
from effdock.training.trainer import configure_optimizers, get_warmup_stable_cosine_scheduler

torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def _load_checkpoint(path: Path, device: torch.device | str) -> dict[str, Any]:
    """Safely load retained confidence checkpoints containing Path metadata."""
    with torch.serialization.safe_globals([type(Path())]):
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"confidence checkpoint must be a mapping, got {type(checkpoint).__name__}")
    return checkpoint


def load_config_sections(
    config_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    import yaml

    with config_path.open() as handle:
        cfg = yaml.safe_load(handle)
    model = cfg["model"]
    model_cfg = {
        "flow_hidden_dim": int(model["hidden_dim"]),
        "flow_hidden_vec_dim": int(model["hidden_vec_dim"]),
        "flow_l2_dim": int(model["l2_dim"]),
        "flow_l2o_dim": int(model["l2o_dim"]),
        "use_saved_ligand_hidden": bool(model.get("use_saved_ligand_hidden", True)),
    }
    return (
        model_cfg,
        dict(cfg.get("training", {})),
        dict(cfg.get("confidence", {})),
        dict(cfg.get("loss", {})),
    )


def _arg_or_cfg(value: Any, cfg: dict[str, Any], key: str, default: Any) -> Any:
    return cfg.get(key, default) if value is None else value


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def setup_ddp() -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("DDP confidence training requires CUDA.")
        torch.cuda.set_device(0 if torch.cuda.device_count() == 1 else local_rank)
        timeout_minutes = int(os.environ.get("CONFIDENCE_DDP_TIMEOUT_MIN", "180"))
        dist.init_process_group(backend="nccl", timeout=timedelta(minutes=timeout_minutes))
    return rank, local_rank, world_size


def cleanup_ddp(world_size: int) -> None:
    if world_size > 1:
        dist.destroy_process_group()


def evaluate(
    model: DockingGraphPoseConfidence,
    loader: DataLoader,
    device: torch.device,
    *,
    max_complexes: int | None,
) -> dict[str, float]:
    model.eval()
    selected = []
    selected_success = []
    selected_frozen = []
    selected_rank_vote = []
    oracle = []
    pose_loss = []
    n_seen = 0
    with torch.no_grad():
        for batch in loader:
            for raw in batch:
                item = to_device(raw, device)
                out = model.forward_complex(item)
                true = item["pose_rmsd"]
                pred_idx = int(out["pose_rmsd"].argmin().item())
                succ_idx = int(out["pose_success_logit"].argmax().item())
                selected.append(float(true[pred_idx].item()))
                selected_success.append(float(true[succ_idx].item()))
                pred_rmsd = out["pose_rmsd"].detach().cpu().to(torch.float32)
                pred_success = torch.sigmoid(
                    out["pose_success_logit"].detach().cpu().to(torch.float32)
                )
                atom_disp = torch.expm1(
                    out["atom_disp_log1p"].detach().cpu().to(torch.float32).clamp(-2.0, 5.0)
                ).clamp_min(0.0)
                atom_ok = torch.sigmoid(out["atom_ok_logit"].detach().cpu().to(torch.float32))
                atom_rmsd = torch.sqrt(atom_disp.square().mean(dim=1).clamp_min(0.0))
                atom_q90 = torch.quantile(atom_disp, 0.9, dim=1)
                atom_ok_mean = atom_ok.mean(dim=1)
                scores = [
                    {
                        "confidence_rmsd": float(pred_rmsd[i]),
                        "confidence_success": float(pred_success[i]),
                        "confidence_atom_rmsd": float(atom_rmsd[i]),
                        "confidence_atom_q90": float(atom_q90[i]),
                        "confidence_atom_ok": float(atom_ok_mean[i]),
                    }
                    for i in range(int(true.numel()))
                ]
                # Dataset graph and pose coordinates are already pocket-centered,
                # so the frozen selector receives a zero center here.
                selector_indices = select_confidence_poses(
                    [pose for pose in item["pose_atom_coords"]],
                    scores,
                    item["graph"],
                    torch.zeros(3, device=device),
                )
                selected_frozen.append(
                    float(
                        true[
                            selector_indices["pair_gate_density_rank_vote_plclash_ambig"]
                        ].item()
                    )
                )
                selected_rank_vote.append(
                    float(true[selector_indices["rank_vote"]].item())
                )
                oracle.append(float(true.min().item()))
                pose_loss.append(
                    float(
                        torch.nn.functional.huber_loss(
                            out["pose_rmsd_log1p"],
                            torch.log1p(true),
                        ).item()
                    )
                )
                n_seen += 1
                if max_complexes is not None and n_seen >= max_complexes:
                    break
            if max_complexes is not None and n_seen >= max_complexes:
                break
    sel = torch.tensor(selected)
    sel_s = torch.tensor(selected_success)
    sel_frozen = torch.tensor(selected_frozen)
    sel_rank_vote = torch.tensor(selected_rank_vote)
    ora = torch.tensor(oracle)
    return {
        "n": float(n_seen),
        "selected_mean": float(sel.mean().item()),
        "selected_median": float(sel.median().item()),
        "selected_lt1": float((sel < 1).float().mean().item() * 100),
        "selected_lt2": float((sel < 2).float().mean().item() * 100),
        "selected_lt5": float((sel < 5).float().mean().item() * 100),
        "success_selected_median": float(sel_s.median().item()),
        "success_selected_lt2": float((sel_s < 2).float().mean().item() * 100),
        "frozen_selected_median": float(sel_frozen.median().item()),
        "frozen_selected_lt2": float((sel_frozen < 2).float().mean().item() * 100),
        "rank_vote_selected_lt2": float((sel_rank_vote < 2).float().mean().item() * 100),
        "oracle_mean": float(ora.mean().item()),
        "oracle_median": float(ora.median().item()),
        "oracle_lt2": float((ora < 2).float().mean().item() * 100),
        "pose_loss": float(sum(pose_loss) / max(1, len(pose_loss))),
    }


def fmt_metrics(prefix: str, metrics: dict[str, float]) -> str:
    return (
        f"{prefix} n={metrics['n']:.0f} "
        f"sel_mean={metrics['selected_mean']:.3f} sel_med={metrics['selected_median']:.3f} "
        f"sel<1={metrics['selected_lt1']:.1f}% sel<2={metrics['selected_lt2']:.1f}% "
        f"sel<5={metrics['selected_lt5']:.1f}% "
        f"succ_med={metrics['success_selected_median']:.3f} succ<2={metrics['success_selected_lt2']:.1f}% "
        f"frozen_med={metrics['frozen_selected_median']:.3f} "
        f"frozen<2={metrics['frozen_selected_lt2']:.1f}% "
        f"oracle_med={metrics['oracle_median']:.3f} oracle<2={metrics['oracle_lt2']:.1f}%"
    )


def _wandb_safe_config(args: argparse.Namespace) -> dict[str, Any]:
    cfg: dict[str, Any] = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            cfg[key] = str(value)
        else:
            cfg[key] = value
    return cfg


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/train_confidence.yaml"))
    parser.add_argument("--split_file", type=Path, default=None)
    parser.add_argument("--processed_dir", type=Path, default=None)
    parser.add_argument("--pose_tag", type=str, default=None)
    parser.add_argument("--tag", type=str, default=None, help="Deprecated alias for --pose_tag.")
    parser.add_argument("--out_dir", type=Path, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--use_wandb", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--reset-optimizer", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--finetune-steps", type=int, default=None)
    parser.add_argument("--eval-on-start", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--total_steps", type=int, default=None)
    parser.add_argument("--batch_complexes", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--muon_lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--use_muon", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--scheduler_type", choices=["warmup_stable_cosine", "none"], default=None)
    parser.add_argument("--warmup_ratio", type=float, default=None)
    parser.add_argument("--cooldown_ratio", type=float, default=None)
    parser.add_argument("--min_lr_ratio", type=float, default=None)
    parser.add_argument("--max_grad_norm", type=float, default=None)
    parser.add_argument("--hidden", type=int, default=None)
    parser.add_argument("--num_layers", type=int, default=None)
    parser.add_argument("--n_rbf", type=int, default=None)
    parser.add_argument("--sh_lmax", type=int, default=None)
    parser.add_argument("--cond_dim", type=int, default=None)
    parser.add_argument("--contact_cutoff", type=float, default=None)
    parser.add_argument(
        "--pose_readout",
        type=str,
        default=None,
        choices=["global_pool", "contact_attention", "global_contact_attention"],
    )
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument(
        "--protein_crop_mode",
        type=str,
        default=None,
        choices=["pose_residue", "ligand_residue", "center"],
    )
    parser.add_argument("--protein_contact_cutoff", type=float, default=None)
    parser.add_argument("--protein_crop_cutoff", type=float, default=None)
    parser.add_argument("--protein_crop_cutoff_min", type=float, default=None)
    parser.add_argument("--protein_crop_cutoff_max", type=float, default=None)
    parser.add_argument("--protein_crop_jitter_sigma", type=float, default=None)
    parser.add_argument("--protein_crop_jitter_max", type=float, default=None)
    parser.add_argument("--stochastic_crop", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--disable_stochastic_crop", action="store_false", dest="stochastic_crop")
    parser.add_argument("--max_protein_atoms", type=int, default=None)
    parser.add_argument("--max_train_poses_per_complex", type=int, default=None)
    parser.add_argument("--max_val_poses_per_complex", type=int, default=None)
    parser.add_argument(
        "--pose_sample_strategy", type=str, default=None, choices=["best_random", "stratified"]
    )
    parser.add_argument("--train_limit", type=int, default=None)
    parser.add_argument("--val_limit", type=int, default=None)
    parser.add_argument("--val_start", type=int, default=None)
    parser.add_argument("--eval_every", type=int, default=None)
    parser.add_argument("--eval_complexes", type=int, default=None)
    parser.add_argument("--save_every", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--listwise-weight", type=float, default=None)
    parser.add_argument("--success-listwise-weight", type=float, default=None)
    parser.add_argument("--setwise-success-weight", type=float, default=None)
    parser.add_argument("--setwise-success-temperature", type=float, default=None)
    parser.add_argument("--pairwise-success-weight", type=float, default=None)
    parser.add_argument("--pairwise-success-temperature", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args(argv)

    model_cfg, train_cfg, conf_cfg, loss_cfg = load_config_sections(args.config)
    loss_kwargs = {
        "atom_weight": float(loss_cfg.get("atom_weight", 0.2)),
        "atom_bce_weight": float(loss_cfg.get("atom_bce_weight", 0.2)),
        "pose_weight": float(loss_cfg.get("pose_weight", 1.0)),
        "pose_bce_weight": float(loss_cfg.get("pose_bce_weight", 0.4)),
        "rank_weight": float(loss_cfg.get("rank_weight", 0.5)),
        "listwise_weight": float(
            _arg_or_cfg(args.listwise_weight, loss_cfg, "listwise_weight", 0.0)
        ),
        "listwise_tau": float(loss_cfg.get("listwise_tau", 1.0)),
        "listwise_pred_temp": float(loss_cfg.get("listwise_pred_temp", 1.0)),
        "success_listwise_weight": float(
            _arg_or_cfg(
                args.success_listwise_weight,
                loss_cfg,
                "success_listwise_weight",
                0.0,
            )
        ),
        "success_listwise_margin": float(loss_cfg.get("success_listwise_margin", 0.5)),
        "setwise_success_weight": float(
            _arg_or_cfg(args.setwise_success_weight, loss_cfg, "setwise_success_weight", 0.0)
        ),
        "setwise_success_temperature": float(
            _arg_or_cfg(
                args.setwise_success_temperature,
                loss_cfg,
                "setwise_success_temperature",
                1.0,
            )
        ),
        "pairwise_success_weight": float(
            _arg_or_cfg(args.pairwise_success_weight, loss_cfg, "pairwise_success_weight", 0.0)
        ),
        "pairwise_success_temperature": float(
            _arg_or_cfg(
                args.pairwise_success_temperature,
                loss_cfg,
                "pairwise_success_temperature",
                1.0,
            )
        ),
        "hard_pair_weight": float(loss_cfg.get("hard_pair_weight", 0.0)),
        "hard_pair_margin": float(loss_cfg.get("hard_pair_margin", 0.2)),
        "hard_pair_positive_threshold": float(loss_cfg.get("hard_pair_positive_threshold", 2.0)),
        "hard_pair_negative_threshold": float(loss_cfg.get("hard_pair_negative_threshold", 2.0)),
        "hard_pair_min_gap": float(loss_cfg.get("hard_pair_min_gap", 0.5)),
        "success_threshold": float(loss_cfg.get("success_threshold", 2.0)),
        "rank_margin": float(loss_cfg.get("rank_margin", 0.05)),
        "min_rank_gap": float(loss_cfg.get("min_rank_gap", 0.3)),
    }
    args.loss = loss_kwargs
    args.split_file = Path(
        _arg_or_cfg(args.split_file, conf_cfg, "split_file", "data/splits/plinder.json")
    )
    args.processed_dir = Path(
        _arg_or_cfg(args.processed_dir, conf_cfg, "processed_dir", "data/plinder_processed")
    )
    cfg_pose_tag = conf_cfg.get("pose_tag", conf_cfg.get("tag", DEFAULT_CONFIDENCE_POSE_TAG))
    args.pose_tag = str(args.pose_tag or args.tag or cfg_pose_tag)
    args.out_dir = Path(
        _arg_or_cfg(args.out_dir, conf_cfg, "out_dir", "outputs/pose_confidence_full_graph")
    )
    args.run_name = str(
        _arg_or_cfg(args.run_name, conf_cfg, "run_name", "pose_confidence_full_graph")
    )
    cfg_resume = conf_cfg.get("resume", None)
    args.resume = args.resume or (Path(cfg_resume) if cfg_resume else None)
    args.reset_optimizer = bool(
        _arg_or_cfg(args.reset_optimizer, train_cfg, "reset_optimizer_on_resume", False)
    )
    args.eval_on_start = bool(
        _arg_or_cfg(args.eval_on_start, conf_cfg, "eval_on_start", False)
    )
    args.batch_complexes = int(_arg_or_cfg(args.batch_complexes, conf_cfg, "batch_complexes", 1))
    args.num_workers = int(_arg_or_cfg(args.num_workers, conf_cfg, "num_workers", 2))
    args.hidden = int(_arg_or_cfg(args.hidden, conf_cfg, "hidden", 512))
    args.num_layers = int(_arg_or_cfg(args.num_layers, conf_cfg, "num_layers", 4))
    args.n_rbf = int(_arg_or_cfg(args.n_rbf, conf_cfg, "n_rbf", 32))
    args.sh_lmax = int(_arg_or_cfg(args.sh_lmax, conf_cfg, "sh_lmax", 2))
    args.cond_dim = int(_arg_or_cfg(args.cond_dim, conf_cfg, "cond_dim", 128))
    args.contact_cutoff = float(_arg_or_cfg(args.contact_cutoff, conf_cfg, "contact_cutoff", 5.0))
    args.pose_readout = str(_arg_or_cfg(args.pose_readout, conf_cfg, "pose_readout", "global_pool"))
    args.dropout = float(_arg_or_cfg(args.dropout, conf_cfg, "dropout", 0.1))
    args.protein_crop_mode = str(
        _arg_or_cfg(args.protein_crop_mode, conf_cfg, "protein_crop_mode", "center")
    )
    args.protein_contact_cutoff = float(
        _arg_or_cfg(args.protein_contact_cutoff, conf_cfg, "protein_contact_cutoff", 5.0)
    )
    args.protein_crop_cutoff = float(
        _arg_or_cfg(args.protein_crop_cutoff, conf_cfg, "protein_crop_cutoff", 8.0)
    )
    args.protein_crop_cutoff_min = float(
        _arg_or_cfg(args.protein_crop_cutoff_min, conf_cfg, "protein_crop_cutoff_min", 6.0)
    )
    args.protein_crop_cutoff_max = float(
        _arg_or_cfg(args.protein_crop_cutoff_max, conf_cfg, "protein_crop_cutoff_max", 12.0)
    )
    args.protein_crop_jitter_sigma = float(
        _arg_or_cfg(args.protein_crop_jitter_sigma, conf_cfg, "protein_crop_jitter_sigma", 2.0)
    )
    args.protein_crop_jitter_max = float(
        _arg_or_cfg(args.protein_crop_jitter_max, conf_cfg, "protein_crop_jitter_max", 4.0)
    )
    args.stochastic_crop = bool(
        _arg_or_cfg(args.stochastic_crop, conf_cfg, "stochastic_crop", True)
    )
    args.max_protein_atoms = int(
        _arg_or_cfg(args.max_protein_atoms, conf_cfg, "max_protein_atoms", 2048)
    )
    args.max_train_poses_per_complex = _optional_int(
        _arg_or_cfg(
            args.max_train_poses_per_complex,
            conf_cfg,
            "max_train_poses_per_complex",
            None,
        )
    )
    args.max_val_poses_per_complex = _optional_int(
        _arg_or_cfg(
            args.max_val_poses_per_complex,
            conf_cfg,
            "max_val_poses_per_complex",
            None,
        )
    )
    args.pose_sample_strategy = str(
        _arg_or_cfg(
            args.pose_sample_strategy,
            conf_cfg,
            "pose_sample_strategy",
            "best_random",
        )
    )
    args.train_limit = _optional_int(_arg_or_cfg(args.train_limit, conf_cfg, "train_limit", None))
    args.val_limit = _optional_int(_arg_or_cfg(args.val_limit, conf_cfg, "val_limit", None))
    args.val_start = int(_arg_or_cfg(args.val_start, conf_cfg, "val_start", 0))
    args.eval_every = int(_arg_or_cfg(args.eval_every, conf_cfg, "eval_every", 1000))
    raw_eval_complexes = _arg_or_cfg(args.eval_complexes, conf_cfg, "eval_complexes", 256)
    args.eval_complexes = (
        None
        if raw_eval_complexes is None or int(raw_eval_complexes) <= 0
        else int(raw_eval_complexes)
    )
    best_metric_name = str(conf_cfg.get("best_metric", "success_selected_lt2"))
    best_metric_mode = str(conf_cfg.get("best_metric_mode", "max")).lower()
    if best_metric_mode not in {"min", "max"}:
        raise ValueError(f"best_metric_mode must be 'min' or 'max', got {best_metric_mode!r}")
    args.save_every = int(_arg_or_cfg(args.save_every, conf_cfg, "save_every", 5000))
    args.seed = int(_arg_or_cfg(args.seed, conf_cfg, "seed", 42))
    use_wandb = bool(_arg_or_cfg(args.use_wandb, conf_cfg, "use_wandb", True))
    wandb_project = str(conf_cfg.get("wandb_project", "eff-dock"))
    wandb_run_name = str(conf_cfg.get("wandb_run_name", args.run_name))

    args.total_steps = int(_arg_or_cfg(args.total_steps, train_cfg, "max_steps", 50000))
    args.lr = float(_arg_or_cfg(args.lr, train_cfg, "lr", 2e-4))
    args.muon_lr = float(_arg_or_cfg(args.muon_lr, train_cfg, "muon_lr", 0.02))
    args.weight_decay = float(_arg_or_cfg(args.weight_decay, train_cfg, "weight_decay", 0.01))
    args.use_muon = bool(_arg_or_cfg(args.use_muon, train_cfg, "use_muon", True))
    args.scheduler_type = str(
        _arg_or_cfg(args.scheduler_type, train_cfg, "scheduler_type", "warmup_stable_cosine")
    )
    args.warmup_ratio = float(_arg_or_cfg(args.warmup_ratio, train_cfg, "warmup_ratio", 0.02))
    args.cooldown_ratio = float(_arg_or_cfg(args.cooldown_ratio, train_cfg, "cooldown_ratio", 0.5))
    args.min_lr_ratio = float(_arg_or_cfg(args.min_lr_ratio, train_cfg, "min_lr_ratio", 0.05))
    args.max_grad_norm = float(_arg_or_cfg(args.max_grad_norm, train_cfg, "max_grad_norm", 1.0))
    if args.protein_crop_mode not in {"pose_residue", "ligand_residue", "center"}:
        raise ValueError(f"unknown protein_crop_mode: {args.protein_crop_mode}")

    rank, local_rank, world_size = setup_ddp()
    is_main = rank == 0

    random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = LigandPoseConfidenceDataset(
        split_file=args.split_file,
        split="train",
        processed_dir=args.processed_dir,
        pose_tag=args.pose_tag,
        protein_crop_mode=args.protein_crop_mode,
        protein_contact_cutoff=args.protein_contact_cutoff,
        protein_crop_cutoff=args.protein_crop_cutoff,
        protein_crop_cutoff_min=args.protein_crop_cutoff_min,
        protein_crop_cutoff_max=args.protein_crop_cutoff_max,
        protein_crop_jitter_sigma=args.protein_crop_jitter_sigma,
        protein_crop_jitter_max=args.protein_crop_jitter_max,
        stochastic_crop=args.stochastic_crop,
        max_protein_atoms=args.max_protein_atoms,
        max_poses_per_complex=args.max_train_poses_per_complex,
        pose_sample_strategy=args.pose_sample_strategy,
        limit=args.train_limit,
    )
    val_ds = LigandPoseConfidenceDataset(
        split_file=args.split_file,
        split="val",
        processed_dir=args.processed_dir,
        pose_tag=args.pose_tag,
        protein_crop_mode=args.protein_crop_mode,
        protein_contact_cutoff=args.protein_contact_cutoff,
        protein_crop_cutoff=args.protein_crop_cutoff,
        protein_crop_cutoff_min=args.protein_crop_cutoff_min,
        protein_crop_cutoff_max=args.protein_crop_cutoff_max,
        protein_crop_jitter_sigma=args.protein_crop_jitter_sigma,
        protein_crop_jitter_max=args.protein_crop_jitter_max,
        stochastic_crop=False,
        max_protein_atoms=args.max_protein_atoms,
        max_poses_per_complex=args.max_val_poses_per_complex,
        pose_sample_strategy="best_random",
        limit=args.val_limit,
        start=args.val_start,
    )
    train_sampler = (
        DistributedSampler(
            train_ds,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=args.seed,
            drop_last=False,
        )
        if world_size > 1
        else None
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_complexes,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.num_workers,
        collate_fn=collate_complexes,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_complexes,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_complexes,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    raw_model = DockingGraphPoseConfidence(
        **model_cfg,
        n_rbf=args.n_rbf,
        contact_cutoff=args.contact_cutoff,
        hidden=args.hidden,
        num_layers=args.num_layers,
        sh_lmax=args.sh_lmax,
        cond_dim=args.cond_dim,
        dropout=args.dropout,
        pose_readout=args.pose_readout,
    ).to(device)
    resume_step = 0
    if args.resume is not None:
        ckpt = _load_checkpoint(args.resume, device)
        raw_model.load_state_dict(ckpt["state_dict"])
        resume_step = int(ckpt.get("step", 0))
    if args.finetune_steps is not None:
        if args.resume is None:
            raise ValueError("--finetune-steps requires --resume")
        if args.finetune_steps < 1:
            raise ValueError("--finetune-steps must be at least 1")
        args.total_steps = resume_step + int(args.finetune_steps)
    model = DDP(raw_model, device_ids=[0], output_device=0) if world_size > 1 else raw_model
    optimizers = configure_optimizers(
        raw_model,
        lr=args.lr,
        muon_lr=args.muon_lr,
        weight_decay=args.weight_decay,
        use_muon=args.use_muon,
    )
    if not optimizers:
        raise RuntimeError("No trainable parameters found for optimizer configuration.")
    scheduler_total_steps = (
        int(args.finetune_steps)
        if args.resume is not None and args.reset_optimizer and args.finetune_steps is not None
        else args.total_steps
    )
    if args.scheduler_type == "warmup_stable_cosine":
        schedulers = [
            get_warmup_stable_cosine_scheduler(
                opt,
                scheduler_total_steps,
                warmup_ratio=args.warmup_ratio,
                cooldown_ratio=args.cooldown_ratio,
                min_lr_ratio=args.min_lr_ratio,
            )
            for opt in optimizers
        ]
    elif args.scheduler_type == "none":
        schedulers = []
    else:
        raise ValueError(f"unknown scheduler_type: {args.scheduler_type}")
    if args.resume is not None and not args.reset_optimizer:
        optimizer_states = ckpt.get("optimizer_state_dicts", [])
        scheduler_states = ckpt.get("scheduler_state_dicts", [])
        for opt, state in zip(optimizers, optimizer_states):
            opt.load_state_dict(state)
        for sched, state in zip(schedulers, scheduler_states):
            sched.load_state_dict(state)

    wandb_run = None
    if is_main and use_wandb:
        try:
            import wandb

            wandb_init = {
                "project": wandb_project,
                "name": wandb_run_name,
                "config": _wandb_safe_config(args),
                "resume": "allow",
            }
            if args.resume is not None and "ckpt" in locals() and ckpt.get("wandb_run_id"):
                wandb_init["id"] = ckpt["wandb_run_id"]
            wandb_run = wandb.init(**wandb_init)
            wandb.define_metric("global_step", hidden=True)
            wandb.define_metric("step/*", step_metric="global_step")
            wandb.define_metric("val/*", step_metric="global_step")
            wandb.define_metric("meta/*", step_metric="global_step")
        except Exception as exc:
            print(f"WARNING: wandb init failed: {exc!r}", flush=True)
            use_wandb = False

    if is_main:
        print(
            f"device={device} rank={rank} local_rank={local_rank} world_size={world_size} "
            f"train={len(train_ds)} val={len(val_ds)}"
        )
        print(
            f"model_params={sum(p.numel() for p in raw_model.parameters() if p.requires_grad):,}",
            flush=True,
        )
        print(
            f"optimizers={','.join(type(opt).__name__ for opt in optimizers)} "
            f"lr={args.lr:g} muon_lr={args.muon_lr:g} weight_decay={args.weight_decay:g} "
            f"use_muon={args.use_muon}",
            flush=True,
        )
        print(
            f"scheduler={args.scheduler_type} total_steps={args.total_steps} "
            f"scheduler_steps={scheduler_total_steps} "
            f"warmup_ratio={args.warmup_ratio:g} cooldown_ratio={args.cooldown_ratio:g} "
            f"min_lr_ratio={args.min_lr_ratio:g}",
            flush=True,
        )
        print(
            "loss=" + " ".join(f"{key}={value:g}" for key, value in loss_kwargs.items()),
            flush=True,
        )
        print(
            f"eval_complexes={'all' if args.eval_complexes is None else args.eval_complexes} "
            f"max_protein_atoms={args.max_protein_atoms} "
            f"max_train_poses={args.max_train_poses_per_complex} "
            f"max_val_poses={args.max_val_poses_per_complex} "
            f"pose_readout={args.pose_readout}",
            flush=True,
        )
        if args.resume is not None:
            print(
                f"resumed from {args.resume} at step={resume_step} "
                f"reset_optimizer={args.reset_optimizer}",
                flush=True,
            )

    try:
        best_score = -float("inf") if best_metric_mode == "max" else float("inf")
        best_path = args.out_dir / "best.pt"
        if best_path.exists():
            try:
                best_ckpt = _load_checkpoint(best_path, "cpu")
                best_metrics = best_ckpt.get("metrics") or {}
                if best_metric_name in best_metrics:
                    best_score = float(best_metrics[best_metric_name])
            except Exception as exc:
                if is_main:
                    print(f"warning: failed to read existing best.pt metrics: {exc!r}", flush=True)
        step = resume_step
        last_metrics: dict[str, float] | None = None

        def checkpoint_payload(metrics: dict[str, float] | None = None) -> dict[str, Any]:
            payload: dict[str, Any] = {
                "step": step,
                "state_dict": raw_model.state_dict(),
                "model_cfg": model_cfg,
                "args": vars(args),
                "model_type": "docking_graph_pose_confidence",
                "wandb_run_id": getattr(wandb_run, "id", None)
                if wandb_run is not None
                else None,
                "optimizer_state_dicts": [opt.state_dict() for opt in optimizers],
                "scheduler_state_dicts": [sched.state_dict() for sched in schedulers],
            }
            if metrics is not None:
                payload.update(
                    {
                        "metrics": metrics,
                        "best_metric": best_metric_name,
                        "best_metric_mode": best_metric_mode,
                    }
                )
            return payload

        if args.eval_on_start:
            if is_main:
                last_metrics = evaluate(
                    raw_model, val_loader, device, max_complexes=args.eval_complexes
                )
                print(fmt_metrics(f"[Val S{step} initial]", last_metrics), flush=True)
                best_score = float(last_metrics[best_metric_name])
                torch.save(checkpoint_payload(last_metrics), best_path)
                print(
                    f"  saved initial best.pt {best_metric_name}={best_score:.3f} "
                    f"mode={best_metric_mode}",
                    flush=True,
                )
            if world_size > 1:
                dist.barrier()
        epoch = step // max(1, len(train_loader))
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        train_iter = iter(train_loader)
        while step < args.total_steps:
            try:
                batch = next(train_iter)
            except StopIteration:
                epoch += 1
                if train_sampler is not None:
                    train_sampler.set_epoch(epoch)
                train_iter = iter(train_loader)
                batch = next(train_iter)

            model.train()
            for opt in optimizers:
                opt.zero_grad(set_to_none=True)
            logs: dict[str, float] = {}
            for raw in batch:
                item = to_device(raw, device)
                out = model(item)
                losses = pose_confidence_loss(out, item, **loss_kwargs)
                loss = losses["loss"] / len(batch)
                loss.backward()
                for key, value in losses.items():
                    logs[key] = logs.get(key, 0.0) + float(value.detach().item()) / len(batch)
            grad_norm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), args.max_grad_norm)
            for opt in optimizers:
                opt.step()
            for sched in schedulers:
                sched.step()
            step += 1

            if is_main and (step == 1 or step % 20 == 0):
                lr_vals = [opt.param_groups[0]["lr"] for opt in optimizers]
                lr_text = " ".join(f"lr{i}={lr:.3e}" for i, lr in enumerate(lr_vals))
                print(
                    f"[S{step}] loss={logs['loss']:.4f} pose={logs['loss_pose']:.4f} "
                    f"rank={logs['loss_rank']:.4f} listwise={logs['loss_listwise']:.4f} "
                    f"succ_listwise={logs['loss_success_listwise']:.4f} "
                    f"setwise={logs['loss_setwise_success']:.4f} "
                    f"pairwise={logs['loss_pairwise_success']:.4f} "
                    f"hard_pair={logs['loss_hard_pair']:.4f} "
                    f"atom={logs['loss_atom']:.4f} "
                    f"grad={float(grad_norm):.3f} {lr_text}",
                    flush=True,
                )
                if use_wandb:
                    import wandb

                    log_dict = {
                        "global_step": step,
                        "step/loss": logs["loss"],
                        "step/loss_pose": logs["loss_pose"],
                        "step/loss_pose_bce": logs["loss_pose_bce"],
                        "step/loss_rank": logs["loss_rank"],
                        "step/loss_listwise": logs["loss_listwise"],
                        "step/loss_success_listwise": logs["loss_success_listwise"],
                        "step/loss_setwise_success": logs["loss_setwise_success"],
                        "step/loss_pairwise_success": logs["loss_pairwise_success"],
                        "step/loss_hard_pair": logs["loss_hard_pair"],
                        "step/loss_atom": logs["loss_atom"],
                        "step/loss_atom_bce": logs["loss_atom_bce"],
                        "step/grad_norm": float(grad_norm),
                    }
                    for i, lr in enumerate(lr_vals):
                        log_dict[f"meta/lr_{i}"] = lr
                    wandb.log(log_dict, step=step)

            if is_main and (step % args.save_every == 0 or step == args.total_steps):
                torch.save(checkpoint_payload(), args.out_dir / "latest.pt")
            if world_size > 1 and (step % args.save_every == 0 or step == args.total_steps):
                dist.barrier()

            if step % args.eval_every == 0:
                if is_main:
                    metrics = evaluate(
                        raw_model, val_loader, device, max_complexes=args.eval_complexes
                    )
                    last_metrics = metrics
                    print(fmt_metrics(f"[Val S{step}]", metrics), flush=True)
                    if use_wandb:
                        import wandb

                        wandb.log(
                            {
                                "global_step": step,
                                **{f"val/{key}": value for key, value in metrics.items()},
                            },
                            step=step,
                        )
                    metric_value = float(metrics[best_metric_name])
                    improved = (
                        metric_value > best_score
                        if best_metric_mode == "max"
                        else metric_value < best_score
                    )
                    if improved:
                        best_score = metric_value
                        torch.save(checkpoint_payload(metrics), args.out_dir / "best.pt")
                        print(
                            f"  saved best.pt {best_metric_name}={best_score:.3f} "
                            f"mode={best_metric_mode}",
                            flush=True,
                        )
                if world_size > 1:
                    dist.barrier()

        if is_main:
            best_metrics: dict[str, float] = {}
            if best_path.exists():
                best_metrics = dict(_load_checkpoint(best_path, "cpu").get("metrics") or {})
            metrics_summary = {
                "run_name": args.run_name,
                "resume": str(args.resume) if args.resume is not None else None,
                "resume_step": resume_step,
                "final_step": step,
                "best_metric": best_metric_name,
                "best_score": best_score,
                "best_checkpoint": str(best_path),
                "best_metrics": best_metrics,
                "last_metrics": last_metrics or {},
            }
            (args.out_dir / "metrics.json").write_text(
                json.dumps(metrics_summary, indent=2) + "\n"
            )
            print("done")
    finally:
        if is_main and use_wandb:
            try:
                import wandb

                wandb.finish()
            except Exception:
                pass
        cleanup_ddp(world_size)


if __name__ == "__main__":
    main()
