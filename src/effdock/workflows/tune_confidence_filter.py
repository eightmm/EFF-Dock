#!/usr/bin/env python3
"""Fit a cluster-free confidence filter on PLINDER train and confirm on validation."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from effdock.confidence.dataset import LigandPoseConfidenceDataset, collate_complexes, to_device
from effdock.confidence.runtime import load_pose_confidence_model
from effdock.confidence.selectors import protein_ligand_clash_rates
from effdock.workflows.train_confidence import load_config_sections


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dataset(
    config_path: Path,
    *,
    split: str,
    limit: int | None,
) -> LigandPoseConfidenceDataset:
    _, _, confidence, _ = load_config_sections(config_path)
    return LigandPoseConfidenceDataset(
        split_file=Path(confidence["split_file"]),
        split=split,
        processed_dir=Path(confidence["processed_dir"]),
        pose_tag=str(confidence["pose_tag"]),
        protein_crop_mode=str(confidence.get("protein_crop_mode", "center")),
        protein_contact_cutoff=float(confidence.get("protein_contact_cutoff", 5.0)),
        protein_crop_cutoff=float(confidence.get("protein_crop_cutoff", 10.0)),
        protein_crop_cutoff_min=float(confidence.get("protein_crop_cutoff_min", 10.0)),
        protein_crop_cutoff_max=float(confidence.get("protein_crop_cutoff_max", 10.0)),
        protein_crop_jitter_sigma=0.0,
        protein_crop_jitter_max=0.0,
        stochastic_crop=False,
        max_protein_atoms=int(confidence.get("max_protein_atoms", 1024)),
        max_poses_per_complex=int(confidence.get("max_val_poses_per_complex", 80)),
        pose_sample_strategy="best_random",
        limit=limit,
    )


def _collect(
    model: torch.nn.Module,
    dataset: LigandPoseConfidenceDataset,
    *,
    device: torch.device,
    num_workers: int,
) -> dict[str, Any]:
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_complexes,
        pin_memory=device.type == "cuda",
    )
    records: list[dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            raw = batch[0]
            item = to_device(raw, device)
            out = model.forward_complex(item)
            atom_disp = torch.expm1(
                out["atom_disp_log1p"].detach().cpu().to(torch.float32).clamp(-2.0, 5.0)
            ).clamp_min(0.0)
            atom_ok = torch.sigmoid(out["atom_ok_logit"].detach().cpu().to(torch.float32))
            poses = [pose for pose in raw["pose_atom_coords"]]
            clash = protein_ligand_clash_rates(poses, raw["graph"], torch.zeros(3))
            records.append(
                {
                    "pid": raw["pid"],
                    "true_rmsd": raw["pose_rmsd"].detach().cpu().to(torch.float32),
                    "pred_rmsd": out["pose_rmsd"].detach().cpu().to(torch.float32),
                    "success": torch.sigmoid(
                        out["pose_success_logit"].detach().cpu().to(torch.float32)
                    ),
                    "atom_ok": atom_ok.mean(dim=1),
                    "atom_rmsd": torch.sqrt(atom_disp.square().mean(dim=1).clamp_min(0.0)),
                    "clash": clash,
                }
            )
    if len(records) != len(dataset):
        raise RuntimeError(f"loaded {len(records)} records from dataset of length {len(dataset)}")
    return {
        "split": dataset.split,
        "pose_tag": dataset.pose_tag,
        "num_complexes": len(records),
        "records": records,
    }


def _stack(cache: dict[str, Any], key: str) -> torch.Tensor:
    values = [record[key].to(torch.float32) for record in cache["records"]]
    lengths = {int(value.numel()) for value in values}
    if len(lengths) != 1:
        raise ValueError(f"{key} has variable candidate counts: {sorted(lengths)}")
    tensor = torch.stack(values)
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{key} contains NaN or Inf")
    return tensor


def _select_batch(cache: dict[str, Any], config: dict[str, float | str | None]) -> torch.Tensor:
    pred = _stack(cache, "pred_rmsd")
    success = _stack(cache, "success")
    atom_ok = _stack(cache, "atom_ok")
    clash = _stack(cache, "clash")
    n = pred.shape[0]
    rows = torch.arange(n)
    base = pred.argmin(dim=1)
    base_pred = pred[rows, base]
    base_success = success[rows, base]
    base_atom_ok = atom_ok[rows, base]
    base_clash = clash[rows, base]
    within = pred <= base_pred[:, None] + float(config["pred_rmsd_margin"])

    mode = str(config.get("mode", "strict_both"))
    if mode == "strict_both":
        consensus = (
            within
            & (clash <= base_clash[:, None])
            & (success >= base_success[:, None] + float(config["success_gain"]))
            & (atom_ok >= base_atom_ok[:, None] + float(config["atom_ok_gain"]))
        )
    elif mode == "atom_rmsd_guard":
        atom_rmsd = _stack(cache, "atom_rmsd")
        base_atom_rmsd = atom_rmsd[rows, base]
        consensus = (
            within
            & (clash <= base_clash[:, None])
            & (
                atom_rmsd
                <= base_atom_rmsd[:, None] - float(config.get("atom_rmsd_gain", 0.0))
            )
            & (
                success
                >= base_success[:, None] - float(config.get("success_tolerance", 0.0))
            )
            & (
                atom_ok
                >= base_atom_ok[:, None] - float(config.get("atom_ok_tolerance", 0.0))
            )
        )
    else:
        raise ValueError(f"unknown confidence filter mode: {mode!r}")
    consensus[rows, base] = False
    consensus_pred = pred.masked_fill(~consensus, float("inf"))
    consensus_i = consensus_pred.argmin(dim=1)
    has_consensus = consensus.any(dim=1)
    selected = torch.where(has_consensus, consensus_i, base)

    clash_limit = config["clash_limit"]
    if clash_limit is not None:
        tolerance = float(config["fallback_head_tolerance"])
        physical = (
            within
            & (clash <= float(clash_limit))
            & (success >= base_success[:, None] - tolerance)
            & (atom_ok >= base_atom_ok[:, None] - tolerance)
        )
        physical_pred = pred.masked_fill(~physical, float("inf"))
        physical_i = physical_pred.argmin(dim=1)
        use_physical = (base_clash > float(clash_limit)) & physical.any(dim=1)
        selected = torch.where(use_physical, physical_i, selected)
    return selected


def _metrics(cache: dict[str, Any], selected: torch.Tensor) -> dict[str, float]:
    true = _stack(cache, "true_rmsd")
    pred = _stack(cache, "pred_rmsd")
    rows = torch.arange(true.shape[0])
    base = pred.argmin(dim=1)
    chosen = true[rows, selected]
    return {
        "n": float(chosen.numel()),
        "lt2_pct": float((chosen < 2.0).float().mean().item() * 100.0),
        "median_rmsd": float(chosen.median().item()),
        "mean_rmsd": float(chosen.mean().item()),
        "switch_pct": float((selected != base).float().mean().item() * 100.0),
    }


def _grid() -> list[dict[str, float | str | None]]:
    configs = []
    for margin, success_gain, atom_gain, clash_limit, tolerance in itertools.product(
        (0.03, 0.05, 0.10, 0.20),
        (0.00, 0.02, 0.05, 0.10),
        (0.00, 0.02, 0.05, 0.10),
        (0.0, 0.05, 0.10, None),
        (0.00, 0.02, 0.05),
    ):
        if clash_limit is None and tolerance != 0.0:
            continue
        configs.append(
            {
                "mode": "strict_both",
                "pred_rmsd_margin": margin,
                "success_gain": success_gain,
                "atom_ok_gain": atom_gain,
                "clash_limit": clash_limit,
                "fallback_head_tolerance": tolerance,
            }
        )
    for margin, atom_gain, success_tolerance, atom_tolerance in itertools.product(
        (0.03, 0.05, 0.10, 0.20),
        (0.00, 0.01, 0.02, 0.05, 0.10),
        (0.00, 0.02, 0.05, 0.10),
        (0.00, 0.02, 0.05, 0.10),
    ):
        configs.append(
            {
                "mode": "atom_rmsd_guard",
                "pred_rmsd_margin": margin,
                "success_gain": 0.0,
                "atom_ok_gain": 0.0,
                "clash_limit": None,
                "fallback_head_tolerance": 0.0,
                "atom_rmsd_gain": atom_gain,
                "success_tolerance": success_tolerance,
                "atom_ok_tolerance": atom_tolerance,
            }
        )
    return configs


def _fit(train: dict[str, Any]) -> tuple[dict[str, float | str | None], dict[str, float]]:
    best_config: dict[str, float | str | None] | None = None
    best_metrics: dict[str, float] | None = None
    for config in _grid():
        metrics = _metrics(train, _select_batch(train, config))
        if config.get("mode") == "atom_rmsd_guard" and metrics["switch_pct"] > 35.0:
            continue
        key = (metrics["lt2_pct"], -metrics["median_rmsd"], -metrics["switch_pct"])
        if best_metrics is None:
            better = True
        else:
            best_key = (
                best_metrics["lt2_pct"],
                -best_metrics["median_rmsd"],
                -best_metrics["switch_pct"],
            )
            better = key > best_key
        if better:
            best_config, best_metrics = config, metrics
    if best_config is None or best_metrics is None:
        raise RuntimeError("empty confidence-filter grid")
    return best_config, best_metrics


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/train_confidence.yaml"))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("weights/effdock_confidence_extmatch_n80_s25_step42500.pt"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/eff-dock/confidence-filter-v1")
    )
    parser.add_argument("--train-limit", type=int, default=1024)
    parser.add_argument("--val-limit", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--reuse-cache", action="store_true")
    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    train_path = args.output_dir / "train_scores.pt"
    val_path = args.output_dir / "val_scores.pt"
    if args.reuse_cache:
        train = torch.load(train_path, map_location="cpu", weights_only=True)
        val = torch.load(val_path, map_location="cpu", weights_only=True)
    else:
        model, checkpoint = load_pose_confidence_model(args.checkpoint, device)
        train = _collect(
            model,
            _dataset(args.config, split="train", limit=args.train_limit),
            device=device,
            num_workers=args.num_workers,
        )
        torch.save(train, train_path)
        val = _collect(
            model,
            _dataset(args.config, split="val", limit=args.val_limit),
            device=device,
            num_workers=args.num_workers,
        )
        torch.save(val, val_path)
        del checkpoint

    pure_config = {
        "mode": "strict_both",
        "pred_rmsd_margin": 0.0,
        "success_gain": 1.0,
        "atom_ok_gain": 1.0,
        "clash_limit": None,
        "fallback_head_tolerance": 0.0,
    }
    train_baseline = _metrics(train, _select_batch(train, pure_config))
    best_config, train_best = _fit(train)
    val_baseline = _metrics(val, _select_batch(val, pure_config))
    val_best = _metrics(val, _select_batch(val, best_config))

    train_admitted = (
        train_best["lt2_pct"] >= train_baseline["lt2_pct"] + 1.0
        and train_best["median_rmsd"] <= train_baseline["median_rmsd"]
    )
    deployed = (
        train_admitted
        and val_best["lt2_pct"] >= val_baseline["lt2_pct"] + 1.0
        and val_best["median_rmsd"] <= val_baseline["median_rmsd"]
        and val_best["switch_pct"] <= 35.0
        and int(val_best["n"]) == int(val["num_complexes"])
    )
    result = {
        "protocol_id": "EFFDOCK-CONFIDENCE-FILTER-V1",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "pose_tag": train["pose_tag"],
        "fit_split": {"name": "PLINDER train", "start": 0, "limit": args.train_limit},
        "confirm_split": {"name": "PLINDER val", "start": 0, "limit": args.val_limit},
        "grid_size": len(_grid()),
        "train_baseline": train_baseline,
        "selected_config": best_config,
        "train_selected": train_best,
        "train_admitted": train_admitted,
        "val_baseline": val_baseline,
        "val_selected": val_best,
        "decision": "deploy_confidence_filter_v1" if deployed else "retain_pure_confidence",
        "external_benchmarks_opened": False,
    }
    result_path = args.output_dir / "result.json"
    with result_path.open("w") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
