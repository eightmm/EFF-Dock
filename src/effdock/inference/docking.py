"""User-facing docking pipeline.

This module owns the runtime path for one protein-ligand docking job:
load model/config, preprocess inputs, sample poses, and write SDF/PT outputs.
Keep CLI wrappers thin.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch
import yaml
from rdkit.Chem import rdMolDescriptors

from effdock.checkpoint import load_checkpoint_file, load_portable_model_state
from effdock.inference.io import (
    write_multi_sdf,
    write_sdf,
    write_traj_pdb,
    write_traj_sdf,
)
from effdock.inference.preprocess import load_ligand, preprocess_complex
from effdock.inference.sampler import (
    parse_sigma_list,
    sample_unified,
    sample_unified_multi_sigma,
)


@dataclass(frozen=True)
class DockingOptions:
    protein: Path
    ligand: str
    checkpoint: Path
    config: Path
    pocket_center: torch.Tensor | None = None
    pocket_cutoff: float = 8.0
    num_steps: int = 25
    time_schedule: str = "late"
    schedule_power: float = 3.0
    sigma: float | None = None
    sigma_list: str | None = None
    num_samples: int = 1
    seed: int | None = None
    save_traj: bool = False
    out_dir: Path = Path("outputs/docked")
    device: str | None = None
    score: bool = True
    w_strain: float = 1.0
    vina_guidance_scale: float = 0.0
    vina_guidance_start_t: float = 0.5
    vina_guidance_ramp_power: float = 1.0
    vina_guidance_max_force: float = 10.0
    vina_guidance_max_velocity: float = 5.0
    vina_guidance_max_angular_velocity: float = 5.0
    vina_guidance_protein_shell: float = 18.0
    confidence_checkpoint: Path | None = None
    rank_by: str = "auto"


def parse_center(value: str | None) -> torch.Tensor | None:
    if value is None:
        return None
    parts = value.replace(",", " ").split()
    if len(parts) != 3:
        raise ValueError(f"--pocket_center expects 3 floats, got {value!r}")
    return torch.tensor([float(p) for p in parts], dtype=torch.float32)


def resolve_device(device: str | None) -> torch.device:
    return torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))


def load_model(config_path: Path, checkpoint_path: Path, device: torch.device):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    from effdock.models.effdock import EFFDock

    model_cfg = {k: v for k, v in cfg["model"].items() if k != "model_type"}
    model = EFFDock(**model_cfg)
    ckpt = load_checkpoint_file(checkpoint_path)
    load_portable_model_state(model, ckpt["model_state_dict"])
    model.to(device)
    model.train(False)
    return model, cfg, ckpt


def sample_docked_poses(
    model, graph, lig_data, meta, opts: DockingOptions, cfg, device, *, guidance_fn=None
):
    sigma = opts.sigma if opts.sigma is not None else cfg["data"].get("prior_sigma", 5.0)
    sigma_list, sigma_counts = parse_sigma_list(opts.sigma_list, opts.num_samples)
    pose_objective = cfg.get("data", {}).get("pose_objective", "linear_fm")
    score_rot_sigma_max = float(cfg.get("data", {}).get("score_rot_sigma_max", 3.141592653589793))
    score_alpha_min = float(cfg.get("data", {}).get("score_alpha_min", 0.0))

    if sigma_list:
        print(
            f"\nGenerating {opts.num_samples} pose(s) (multi-sigma), "
            f"{opts.num_steps} ODE steps, schedule={list(zip(sigma_list, sigma_counts))} ..."
        )
        return sample_unified_multi_sigma(
            model,
            graph,
            lig_data,
            meta,
            sigma_list=sigma_list,
            samples_per_sigma=sigma_counts,
            num_steps=opts.num_steps,
            time_schedule=opts.time_schedule,
            schedule_power=opts.schedule_power,
            device=device,
            save_traj=opts.save_traj,
            pose_objective=pose_objective,
            score_rot_sigma_max=score_rot_sigma_max,
            score_alpha_min=score_alpha_min,
            guidance_fn=guidance_fn,
            guidance_scale=opts.vina_guidance_scale,
            guidance_min_t=opts.vina_guidance_start_t,
        )

    print(f"\nGenerating {opts.num_samples} pose(s), {opts.num_steps} ODE steps, sigma={sigma}...")
    return sample_unified(
        model,
        graph,
        lig_data,
        meta,
        num_samples=opts.num_samples,
        num_steps=opts.num_steps,
        translation_sigma=sigma,
        time_schedule=opts.time_schedule,
        schedule_power=opts.schedule_power,
        device=device,
        save_traj=opts.save_traj,
        pose_objective=pose_objective,
        score_rot_sigma_max=score_rot_sigma_max,
        score_alpha_min=score_alpha_min,
        guidance_fn=guidance_fn,
        guidance_scale=opts.vina_guidance_scale,
        guidance_min_t=opts.vina_guidance_start_t,
    )


def write_docking_outputs(
    mol,
    lig_data,
    meta,
    poses,
    trajs,
    opts: DockingOptions,
    scores=None,
    effective_rank_by: str | None = None,
) -> None:
    opts.out_dir.mkdir(parents=True, exist_ok=True)
    pocket_center = meta["pocket_center"]

    run_props = {
        "protein": str(opts.protein),
        "ligand": str(opts.ligand),
        "checkpoint": str(opts.checkpoint),
        "config": str(opts.config),
        "pocket_center": ",".join(f"{float(x):.6f}" for x in pocket_center.detach().cpu().tolist()),
        "pocket_cutoff": opts.pocket_cutoff,
        "num_samples": opts.num_samples,
        "num_steps": opts.num_steps,
        "time_schedule": opts.time_schedule,
        "schedule_power": opts.schedule_power,
        "sigma": opts.sigma if opts.sigma is not None else "config_default",
        "sigma_list": opts.sigma_list or "",
        "seed": opts.seed if opts.seed is not None else "",
        "score": opts.score,
        "w_strain": opts.w_strain,
        "vina_guidance_scale": opts.vina_guidance_scale,
        "vina_guidance_start_t": opts.vina_guidance_start_t,
        "vina_guidance_ramp_power": opts.vina_guidance_ramp_power,
        "vina_guidance_max_force": opts.vina_guidance_max_force,
        "vina_guidance_max_velocity": opts.vina_guidance_max_velocity,
        "vina_guidance_max_angular_velocity": opts.vina_guidance_max_angular_velocity,
        "vina_guidance_protein_shell": opts.vina_guidance_protein_shell,
        "confidence_checkpoint": str(opts.confidence_checkpoint or ""),
        "rank_by": opts.rank_by,
        "effective_rank_by": effective_rank_by or opts.rank_by,
    }
    if opts.num_samples == 1:
        out_path = opts.out_dir / "docked.sdf"
        write_sdf(
            mol,
            poses[0],
            pocket_center,
            out_path,
            score=scores[0] if scores else None,
            props={**run_props, "sample_index": 0},
        )
        print(f"\nDocked pose saved to {out_path}")
    else:
        out_path = opts.out_dir / "docked_poses.sdf"
        write_multi_sdf(mol, poses, pocket_center, out_path, scores=scores, props=run_props)
        print(f"\n{opts.num_samples} poses saved to {out_path}")

    if opts.save_traj:
        for i, res in enumerate(trajs):
            suffix = f"_{i}" if opts.num_samples > 1 else ""
            write_traj_sdf(
                mol,
                res["traj"],
                res["traj_times"],
                pocket_center,
                opts.out_dir / f"traj{suffix}.sdf",
            )
            write_traj_pdb(mol, res["traj"], pocket_center, opts.out_dir / f"traj{suffix}.pdb")
            print(f"  Trajectory{suffix}: {len(res['traj'])} frames")

    torch.save(
        {
            "pocket_center": pocket_center,
            "frag_centers": lig_data["frag_centers"],
            "frag_sizes": lig_data["frag_sizes"],
            "poses": [
                {"atom_pos_pred": p, **({"score": scores[i]} if scores else {})}
                for i, p in enumerate(poses)
            ],
            "scores": scores,
            "options": run_props,
            "trajectories": [{"traj": r["traj"], "traj_times": r["traj_times"]} for r in trajs]
            if opts.save_traj
            else None,
        },
        opts.out_dir / "results.pt",
    )
    print(f"Raw tensors saved to {opts.out_dir / 'results.pt'}")


def dock(opts: DockingOptions) -> None:
    if not opts.protein.exists():
        raise FileNotFoundError(f"Protein PDB not found: {opts.protein}")

    device = resolve_device(opts.device)
    model, cfg, ckpt = load_model(opts.config, opts.checkpoint, device)
    print(f"Model loaded: {opts.checkpoint} (step {ckpt.get('step', '?')})")

    print(f"Loading ligand: {opts.ligand}")
    mol, has_pose = load_ligand(opts.ligand)
    print(
        f"  Atoms: {mol.GetNumAtoms()}, "
        f"Formula: {rdMolDescriptors.CalcMolFormula(mol)}, has_pose={has_pose}"
    )
    if opts.pocket_center is None:
        raise ValueError(
            "docking requires explicit --pocket-center x,y,z; target ligand "
            "coordinates are not an allowed pocket-definition shortcut"
        )

    print("Preprocessing...")
    graph, lig_data, meta = preprocess_complex(
        opts.protein,
        mol,
        pocket_center=opts.pocket_center,
        pocket_cutoff=opts.pocket_cutoff,
    )
    print(f"  Pocket center: {meta['pocket_center'].tolist()}")
    print(f"  Ligand: {meta['num_atom']} atoms, {meta['num_frag']} fragments")
    print(
        f"  Graph: {graph['num_nodes'].item()} nodes, "
        f"{graph['num_prot_atom'].item()} prot atoms, "
        f"{graph['num_prot_res'].item()} residues, "
        f"{graph['edge_index'].shape[1]} edges"
    )

    if opts.seed is not None:
        torch.manual_seed(opts.seed)

    guidance_fn = None
    if opts.vina_guidance_scale != 0.0:
        if opts.vina_guidance_scale < 0.0:
            raise ValueError("--vina-guidance-scale must be non-negative")
        from effdock.evaluation.vina_guidance import (
            VinaGuidanceConfig,
            build_vina_guidance,
        )

        guidance_config = VinaGuidanceConfig(
            start_t=opts.vina_guidance_start_t,
            ramp_power=opts.vina_guidance_ramp_power,
            max_atom_force=opts.vina_guidance_max_force,
            max_translation_velocity=opts.vina_guidance_max_velocity,
            max_angular_velocity=opts.vina_guidance_max_angular_velocity,
            w_strain=opts.w_strain,
        )
        guidance_fn = build_vina_guidance(
            mol,
            opts.protein,
            pocket_center=meta["pocket_center"],
            frag_id=lig_data["fragment_id"],
            device=device,
            protein_shell_cutoff=opts.vina_guidance_protein_shell,
            config=guidance_config,
        )
        print(
            f"  Vina+DG guidance: scale={opts.vina_guidance_scale:g}, "
            f"start_t={opts.vina_guidance_start_t:g}, "
            f"receptor_atoms={guidance_fn.prot_coords.shape[0]}"
        )

    results = sample_docked_poses(
        model, graph, lig_data, meta, opts, cfg, device, guidance_fn=guidance_fn
    )
    poses = [r["atom_pos_pred"] for r in results]
    trajs = results if opts.save_traj else []

    scores = None
    if opts.score:
        from effdock.evaluation.pose_scoring import score_poses

        print("\nScoring poses (Vina + DG strain)...")
        cpu_poses = [p.detach().cpu() for p in poses]
        scores = score_poses(
            mol,
            cpu_poses,
            opts.protein,
            pocket_center=meta["pocket_center"].detach().cpu(),
            frag_id=lig_data["fragment_id"].detach().cpu(),
            pocket_cutoff=max(opts.pocket_cutoff, 10.0),
            w_strain=opts.w_strain,
        )

    confidence_indices: dict[str, int] = {}
    if opts.confidence_checkpoint is not None:
        from effdock.confidence.runtime import (
            load_pose_confidence_model,
            sample_sigmas,
            score_poses_with_confidence,
        )
        from effdock.confidence.selectors import select_confidence_poses

        print("\nScoring poses (EFF-Dock confidence)...")
        confidence_model, confidence_ckpt = load_pose_confidence_model(
            opts.confidence_checkpoint, device
        )
        default_sigma = float(
            opts.sigma if opts.sigma is not None else cfg["data"].get("prior_sigma", 1.0)
        )
        confidence_scores = score_poses_with_confidence(
            confidence_model,
            model,
            graph,
            lig_data,
            meta,
            poses,
            sigma=sample_sigmas(results, default_sigma),
            device=device,
        )
        if scores is None:
            scores = confidence_scores
        else:
            for physical, learned in zip(scores, confidence_scores, strict=True):
                physical.update(learned)
        confidence_indices = select_confidence_poses(poses, scores, graph, meta["pocket_center"])
        print(
            f"  Confidence loaded: {opts.confidence_checkpoint} "
            f"(step {confidence_ckpt.get('step', '?')})"
        )

    rank_by = opts.rank_by
    if rank_by == "auto":
        if opts.confidence_checkpoint is not None:
            rank_by = "confidence"
        elif opts.score:
            rank_by = "vina"
        else:
            rank_by = "sample"
    if rank_by == "vina" and not opts.score:
        raise ValueError("--rank_by vina requires --score")
    if (
        rank_by
        in {
            "confidence",
            "success",
            "atom_success",
            "rank_vote",
            "confidence_filter_v1",
            "pair_gate_density_rank_vote_plclash_ambig",
        }
        and opts.confidence_checkpoint is None
    ):
        raise ValueError(f"--rank-by {rank_by} requires --confidence-checkpoint")

    if scores is not None and rank_by != "sample":
        if rank_by == "vina":
            order = sorted(range(len(scores)), key=lambda i: scores[i]["total"])
        elif rank_by in confidence_indices:
            first = confidence_indices[rank_by]
            if rank_by == "confidence":
                tail = sorted(range(len(scores)), key=lambda i: scores[i]["confidence_rmsd"])
            elif rank_by == "success":
                tail = sorted(range(len(scores)), key=lambda i: -scores[i]["confidence_success"])
            elif rank_by == "atom_success":
                tail = sorted(range(len(scores)), key=lambda i: -scores[i]["confidence_atom_ok"])
            elif rank_by == "rank_vote":
                tail = sorted(range(len(scores)), key=lambda i: scores[i]["confidence_rank_vote"])
            else:
                tail = sorted(range(len(scores)), key=lambda i: scores[i]["confidence_rmsd"])
            order = [first, *(i for i in tail if i != first)]
        else:
            raise ValueError(f"unsupported rank_by={opts.rank_by!r}")
        poses = [poses[i] for i in order]
        scores = [scores[i] for i in order]
        if opts.save_traj:
            results = [results[i] for i in order]
            trajs = results

    if scores is not None:
        if "confidence_rmsd" in scores[0]:
            print(f"  {'rank':<6}{'pred_rmsd':>12}{'success':>11}{'atom_ok':>10}")
            for rank, s in enumerate(scores):
                print(
                    f"  {rank:<6}{s['confidence_rmsd']:>12.3f}"
                    f"{s['confidence_success']:>11.4f}{s['confidence_atom_ok']:>10.4f}"
                )
        elif "total" in scores[0]:
            print(f"  {'rank':<6}{'total':>10}{'vina':>10}{'strain':>10}")
            for rank, s in enumerate(scores):
                print(f"  {rank:<6}{s['total']:>10.3f}{s['vina']:>10.3f}{s['strain']:>10.3f}")

    write_docking_outputs(
        mol,
        lig_data,
        meta,
        poses,
        trajs,
        opts,
        scores=scores,
        effective_rank_by=rank_by,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Dock a ligand into a protein pocket using EFFDock"
    )
    parser.add_argument("--protein", type=Path, required=True)
    parser.add_argument("--ligand", type=str, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--pocket-center", type=parse_center, required=True, help="Binding site x,y,z"
    )
    parser.add_argument("--pocket-cutoff", type=float, default=8.0)
    parser.add_argument("--num-steps", type=int, default=25)
    parser.add_argument(
        "--time-schedule", type=str, default="late", choices=("uniform", "late", "early")
    )
    parser.add_argument("--schedule-power", type=float, default=3.0)
    parser.add_argument(
        "--sigma",
        type=float,
        default=None,
        help="Single prior sigma. Ignored when --sigma_list is set.",
    )
    parser.add_argument(
        "--sigma-list",
        type=str,
        default=None,
        help='Multi-sigma inference. "2,3,4,5" splits --num_samples '
        'across values; "2:10,3:10,4:20" gives explicit counts.',
    )
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--save-traj", action="store_true")
    parser.add_argument("--output-dir", dest="out_dir", type=Path, default=Path("outputs/docked"))
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--score",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Score poses with Vina + DG strain and rank them (--no-score to skip).",
    )
    parser.add_argument(
        "--w-strain",
        type=float,
        default=1.0,
        help="Weight on the ligand DG strain penalty added to Vina.",
    )
    parser.add_argument(
        "--vina-guidance-scale",
        type=float,
        default=0.0,
        help="Add differentiable Vina+DG guidance during sampling (0 disables it).",
    )
    parser.add_argument("--vina-guidance-start-t", type=float, default=0.5)
    parser.add_argument("--vina-guidance-ramp-power", type=float, default=1.0)
    parser.add_argument("--vina-guidance-max-force", type=float, default=10.0)
    parser.add_argument("--vina-guidance-max-velocity", type=float, default=5.0)
    parser.add_argument("--vina-guidance-max-angular-velocity", type=float, default=5.0)
    parser.add_argument(
        "--vina-guidance-protein-shell",
        type=float,
        default=18.0,
        help="Receptor shell radius around the declared pocket center (A).",
    )
    parser.add_argument(
        "--confidence-checkpoint",
        type=Path,
        default=None,
        help="Optional learned pose-confidence checkpoint used for reranking.",
    )
    parser.add_argument(
        "--rank-by",
        type=str,
        default="auto",
        choices=(
            "auto",
            "sample",
            "vina",
            "confidence",
            "success",
            "atom_success",
            "rank_vote",
            "confidence_filter_v1",
            "pair_gate_density_rank_vote_plclash_ambig",
        ),
        help="Pose ordering for SDF/results.pt.",
    )
    return parser


def options_from_args(args: argparse.Namespace) -> DockingOptions:
    return DockingOptions(
        protein=args.protein,
        ligand=args.ligand,
        checkpoint=args.checkpoint,
        config=args.config,
        pocket_center=args.pocket_center,
        pocket_cutoff=args.pocket_cutoff,
        num_steps=args.num_steps,
        time_schedule=args.time_schedule,
        schedule_power=args.schedule_power,
        sigma=args.sigma,
        sigma_list=args.sigma_list,
        num_samples=args.num_samples,
        seed=args.seed,
        save_traj=args.save_traj,
        out_dir=args.out_dir,
        device=args.device,
        score=args.score,
        w_strain=args.w_strain,
        vina_guidance_scale=args.vina_guidance_scale,
        vina_guidance_start_t=args.vina_guidance_start_t,
        vina_guidance_ramp_power=args.vina_guidance_ramp_power,
        vina_guidance_max_force=args.vina_guidance_max_force,
        vina_guidance_max_velocity=args.vina_guidance_max_velocity,
        vina_guidance_max_angular_velocity=args.vina_guidance_max_angular_velocity,
        vina_guidance_protein_shell=args.vina_guidance_protein_shell,
        confidence_checkpoint=args.confidence_checkpoint,
        rank_by=args.rank_by,
    )


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    dock(options_from_args(args))


__all__ = [
    "DockingOptions",
    "build_arg_parser",
    "dock",
    "load_model",
    "main",
    "parse_center",
]


if __name__ == "__main__":
    main()
