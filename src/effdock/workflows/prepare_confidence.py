"""Generate labeled ligand-pose shards for confidence-model training.

This is the reproducible, fixed-condition data path for the promoted EFF-Dock
confidence model.  It samples poses from a docking checkpoint, extracts the
same t=1 ligand representations consumed by confidence training, and labels
every sampled pose against the processed crystal ligand.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from effdock.confidence.features import extract_t1_ligand_irreps
from effdock.inference.docking import load_model
from effdock.inference.preprocess import build_inference_bundle, load_processed
from effdock.inference.sampler import sample_unified

DEPLOYED_POSE_TAG = "conf_ligonly_extmatch_n80_s25_sig0p5_pc10"


def compute_pose_labels(
    ligand: dict[str, torch.Tensor],
    pocket_center: torch.Tensor,
    poses: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Return per-atom displacement and pose RMSD in the pocket-centered frame."""
    crystal = ligand["atom_coords"].to(torch.float32) - pocket_center.to(torch.float32)
    atom_disp = (poses.to(torch.float32) - crystal.unsqueeze(0)).norm(dim=-1)
    return {
        "atom_disp": atom_disp,
        "pose_rmsd": atom_disp.square().mean(dim=1).sqrt(),
    }


def _output_dir(processed_dir: Path, output_root: Path | None, pid: str) -> Path:
    if output_root is None:
        return processed_dir / pid / "confidence_poses"
    return output_root / pid


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/train.yaml"))
    parser.add_argument("--split_file", type=Path, default=Path("data/splits/plinder.json"))
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--processed_dir", type=Path, default=Path("data/plinder_processed"))
    parser.add_argument(
        "--output_root",
        type=Path,
        default=None,
        help="Optional mirror root; by default shards stay beside each processed complex.",
    )
    parser.add_argument("--num_samples", type=int, default=80)
    parser.add_argument("--num_steps", type=int, default=25)
    parser.add_argument("--sigma", type=float, default=0.5)
    parser.add_argument("--pocket_cutoff", type=float, default=10.0)
    parser.add_argument("--time_schedule", choices=("uniform", "late", "early"), default="late")
    parser.add_argument("--schedule_power", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--pose_tag", type=str, default=None)
    parser.add_argument("--manifest_suffix", type=str, default="")
    parser.add_argument("--hidden_dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args(argv)
    if args.num_samples <= 0 or args.num_steps <= 0:
        parser.error("--num_samples and --num_steps must be positive")
    return args


def _manifest_row(pid: str, status: str, **fields: Any) -> dict[str, Any]:
    return {"pid": pid, "status": status, **fields}


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, cfg, checkpoint = load_model(args.config, args.checkpoint, device)
    checkpoint_step = int(checkpoint.get("step", -1))
    model_cfg = cfg.get("data", {})
    hidden_dtype = torch.float16 if args.hidden_dtype == "float16" else torch.float32

    with args.split_file.open() as handle:
        split_map = json.load(handle)
    all_pids = list(split_map[args.split])
    indexed_pids = list(enumerate(all_pids))[args.start :]
    if args.limit is not None:
        indexed_pids = indexed_pids[: args.limit]

    is_deployed_condition = (
        args.num_samples == 80
        and args.num_steps == 25
        and args.sigma == 0.5
        and args.pocket_cutoff == 10.0
        and args.time_schedule == "late"
        and args.schedule_power == 3.0
    )
    pose_tag = args.pose_tag or (
        DEPLOYED_POSE_TAG
        if is_deployed_condition
        else (
            f"ckpt{checkpoint_step}_{args.checkpoint.stem}"
            f"_sig{args.sigma:g}_steps{args.num_steps}_n{args.num_samples}_ligand"
        )
    )
    manifest_root = args.output_root or args.processed_dir
    manifest_root.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.manifest_suffix}" if args.manifest_suffix else ""
    manifest_path = manifest_root / f"confidence_pose_manifest_{args.split}_{pose_tag}{suffix}.jsonl"

    counts = {"ok": 0, "skip": 0, "fail": 0}
    with manifest_path.open("a", encoding="utf-8") as manifest:
        for global_index, pid in indexed_pids:
            output_dir = _output_dir(args.processed_dir, args.output_root, pid)
            output_path = output_dir / f"confposes_{pose_tag}.pt"
            if output_path.exists() and not args.overwrite:
                counts["skip"] += 1
                manifest.write(json.dumps(_manifest_row(pid, "skip", pt_path=str(output_path))) + "\n")
                manifest.flush()
                continue

            try:
                loaded = load_processed(args.processed_dir / pid)
                if loaded is None:
                    raise RuntimeError("processed tensors missing or unreadable")
                protein, ligand, meta = loaded
                bundle = build_inference_bundle(
                    protein,
                    ligand,
                    meta,
                    pocket_cutoff=args.pocket_cutoff,
                )
                if bundle is None:
                    raise RuntimeError("failed to build inference graph")
                graph, ligand_data, inference_meta = bundle

                sample_seed = args.seed + global_index
                torch.manual_seed(sample_seed)
                if device.type == "cuda":
                    torch.cuda.manual_seed_all(sample_seed)
                results = sample_unified(
                    model,
                    graph,
                    ligand_data,
                    inference_meta,
                    num_samples=args.num_samples,
                    num_steps=args.num_steps,
                    translation_sigma=args.sigma,
                    time_schedule=args.time_schedule,
                    schedule_power=args.schedule_power,
                    device=device,
                    pose_objective=model_cfg.get("pose_objective", "linear_fm"),
                    score_rot_sigma_max=float(model_cfg.get("score_rot_sigma_max", torch.pi)),
                    score_alpha_min=float(model_cfg.get("score_alpha_min", 0.0)),
                )
                poses = torch.stack(
                    [result["atom_pos_pred"].detach().to(torch.float32).cpu() for result in results]
                )
                features = extract_t1_ligand_irreps(
                    model,
                    graph,
                    ligand_data,
                    inference_meta,
                    poses,
                    sigma=args.sigma,
                    device=device,
                    hidden_dtype=hidden_dtype,
                )
                pocket_center = inference_meta["pocket_center"].to(torch.float32).cpu()
                labels = compute_pose_labels(ligand_data, pocket_center, poses)
                processed_pid_dir = args.processed_dir / pid
                shard = {
                    "storage_version": "effdock_confidence_ligand_pose_v1",
                    "pid": pid,
                    "processed_dir": str(processed_pid_dir),
                    "protein_pt": str(processed_pid_dir / "protein.pt"),
                    "ligand_pt": str(processed_pid_dir / "ligand.pt"),
                    "meta_pt": str(processed_pid_dir / "meta.pt"),
                    "protein_features_saved": False,
                    "checkpoint": str(args.checkpoint),
                    "checkpoint_step": checkpoint_step,
                    "config": str(args.config),
                    "split": args.split,
                    "seed": sample_seed,
                    "sigma": float(args.sigma),
                    "num_steps": int(args.num_steps),
                    "num_samples": int(args.num_samples),
                    "time_schedule": args.time_schedule,
                    "schedule_power": float(args.schedule_power),
                    "pocket_cutoff": float(args.pocket_cutoff),
                    "pocket_center_used": pocket_center,
                    "pose_sigma": torch.full((args.num_samples,), float(args.sigma)),
                    "pose_num_steps": torch.full((args.num_samples,), args.num_steps, dtype=torch.long),
                    "hidden_scope": "ligand",
                    "hidden_dtype": args.hidden_dtype,
                    "lig_num_atoms": int(inference_meta["num_atom"]),
                    "lig_num_frags": int(inference_meta["num_frag"]),
                    "lig_atom_coords_crystal_centered": (
                        ligand_data["atom_coords"].to(torch.float32) - pocket_center
                    ).cpu(),
                    "frag_sizes": ligand_data["frag_sizes"].cpu(),
                    "fragment_id": ligand_data["fragment_id"].cpu(),
                    "pose_atom_coords": poses,
                    "h_lig_node": features["h_lig_node"],
                    "lig_node_type": features["lig_node_type"],
                    "atom_disp": labels["atom_disp"],
                    "pose_rmsd": labels["pose_rmsd"],
                }
                output_dir.mkdir(parents=True, exist_ok=True)
                temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
                torch.save(shard, temporary_path)
                temporary_path.replace(output_path)

                row = _manifest_row(
                    pid,
                    "ok",
                    pt_path=str(output_path),
                    num_samples=args.num_samples,
                    mean_rmsd=float(labels["pose_rmsd"].mean()),
                    best_rmsd=float(labels["pose_rmsd"].min()),
                    success_2A=float((labels["pose_rmsd"] < 2.0).float().mean()),
                    seed=sample_seed,
                )
                counts["ok"] += 1
                manifest.write(json.dumps(row) + "\n")
                manifest.flush()
                print(
                    f"[{sum(counts.values())}/{len(indexed_pids)}] {pid} ok "
                    f"best={row['best_rmsd']:.3f} mean={row['mean_rmsd']:.3f}"
                )
            except Exception as exc:
                counts["fail"] += 1
                row = _manifest_row(pid, "fail", error=f"{type(exc).__name__}: {exc}")
                manifest.write(json.dumps(row) + "\n")
                manifest.flush()
                print(f"[{sum(counts.values())}/{len(indexed_pids)}] {pid} FAIL {exc!r}")

    print(
        f"done ok={counts['ok']} skip={counts['skip']} fail={counts['fail']} "
        f"manifest={manifest_path}"
    )


if __name__ == "__main__":
    main()
