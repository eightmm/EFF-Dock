#!/usr/bin/env python3
"""Evaluate an EFF-Dock checkpoint on frozen external benchmarks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem

from effdock.evaluation.benchmark import (
    apply_refinement,
    compute_pose_rmsd,
    compute_stats,
    detect_complex_files,
    match_atoms,
    select_by_score,
    select_pose,
)
from effdock.evaluation.benchmark import (
    load_ligand as load_ref_ligand,
)
from effdock.evaluation.pose_scoring import build_protein_vina_inputs, score_poses
from effdock.evaluation.pose_validity import check_validity, ligand_bounds, vdw_radii
from effdock.inference.docking import load_model
from effdock.inference.preprocess import load_ligand, preprocess_complex
from effdock.inference.sampler import parse_sigma_list, sample_unified, sample_unified_multi_sigma


@dataclass(frozen=True)
class ComplexInput:
    complex_id: str
    protein: Path
    ligand_ref: Path
    ligand_format: str
    smiles: str | None
    pocket_center: tuple[float, float, float]


def load_smiles(dataset: str, external_dir: Path) -> dict[str, str]:
    if dataset == "astex":
        with open(external_dir / "astex_smiles.json") as handle:
            raw = json.load(handle)
    elif dataset == "posebusters":
        with open(external_dir / "pb_smiles.json") as handle:
            raw = json.load(handle)
        keep_path = external_dir / "posebusters_v2_ids.txt"
        if keep_path.exists():
            keep = {line.strip() for line in keep_path.read_text().splitlines() if line.strip()}
            raw = {k: v for k, v in raw.items() if k in keep}
    elif dataset == "casf":
        with open(external_dir / "casf_smiles.json") as handle:
            raw = json.load(handle)
    else:
        raise ValueError(dataset)
    return {k.lower(): v["smiles"] for k, v in raw.items()}


def load_pocket_centers(path: Path) -> dict[str, tuple[float, float, float]]:
    """Load frozen pocket centers keyed by complex ID.

    The center provenance is deliberately not inferred here. Prospective and
    reference-defined benchmark centers must be kept in distinct manifests.
    """
    with path.open() as handle:
        raw = json.load(handle)
    centers: dict[str, tuple[float, float, float]] = {}
    for complex_id, value in raw.items():
        if isinstance(value, dict):
            value = value.get("pocket_center", value.get("center"))
        if not isinstance(value, list | tuple) or len(value) != 3:
            raise ValueError(f"invalid pocket center for {complex_id!r}: {value!r}")
        centers[complex_id.lower()] = tuple(float(x) for x in value)
    return centers


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def shard_complexes(
    complexes: list[ComplexInput], shard_index: int, num_shards: int
) -> list[ComplexInput]:
    if num_shards < 1:
        raise ValueError("num_shards must be >= 1")
    if not 0 <= shard_index < num_shards:
        raise ValueError("shard_index must satisfy 0 <= shard_index < num_shards")
    return complexes[shard_index::num_shards]


def _write_pose(mol: Chem.Mol, pose: torch.Tensor, center: torch.Tensor, path: Path) -> None:
    pose_mol = Chem.RWMol(mol)
    conf = pose_mol.GetConformer()
    absolute = pose.detach().cpu() + center.detach().cpu()
    for atom_index, xyz in enumerate(absolute.tolist()):
        conf.SetAtomPosition(atom_index, xyz)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = Chem.SDWriter(str(path))
    writer.write(pose_mol)
    writer.close()


def summarize_rows(rows: list[dict]) -> dict:
    summary: dict[str, dict] = {}
    selectors = [
        selector
        for selector in (
            "first",
            "vina",
            "confidence",
            "confidence_filter",
            "confidence_final",
            "oracle",
        )
        if rows and f"{selector}_rmsd" in rows[0]
    ]
    for selector in selectors:
        values = np.asarray([row[f"{selector}_rmsd"] for row in rows], dtype=float)
        summary[selector] = compute_stats(values) if len(values) else {}
    for selector in selectors:
        valid = [bool(row[f"{selector}_fast_valid"]) for row in rows]
        summary[selector]["fast_valid_pct"] = float(np.mean(valid) * 100) if valid else None
    return summary


def candidate_ids(dataset: str, smiles_by_id: dict[str, str]) -> list[str]:
    ids = sorted(smiles_by_id)
    if dataset == "astex":
        return ids
    return ids


def find_file(root: Path, complex_id: str, kind: str) -> Path | None:
    cid = complex_id.lower()
    suffixes = (".sdf", ".mol2") if kind == "ligand" else (".pdb",)
    hits: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        rel = str(path.relative_to(root)).lower()
        stem = path.stem.lower()
        if cid not in rel and cid.split("_")[0] not in rel:
            continue
        if kind == "protein":
            if "protein" in stem and "predicted" not in stem:
                hits.insert(0, path)
            elif "holo_aligned_predicted_protein" in stem:
                hits.append(path)
            elif "protein" in stem:
                hits.append(path)
        else:
            if "ligand" in stem or "crystal" in stem:
                hits.append(path)
    if not hits:
        return None
    return sorted(set(hits), key=lambda p: (len(str(p)), str(p)))[0]


def discover_complexes(
    dataset: str,
    data_dir: Path,
    external_dir: Path,
    pocket_centers: dict[str, tuple[float, float, float]],
) -> list[ComplexInput]:
    smiles_by_id = load_smiles(dataset, external_dir)
    dir_index = {p.name.lower(): p for p in data_dir.iterdir() if p.is_dir()}
    complexes: list[ComplexInput] = []
    missing_centers: list[str] = []
    for cid in candidate_ids(dataset, smiles_by_id):
        if cid not in pocket_centers:
            missing_centers.append(cid)
            continue
        direct = dir_index.get(cid)
        if direct is None:
            prefix = cid.split("_")[0] + "_"
            direct = next((p for name, p in dir_index.items() if name.startswith(prefix)), None)
        detected = detect_complex_files(direct, direct.name) if direct and direct.exists() else None
        if detected is not None:
            protein, ligand, fmt = detected
        else:
            search_root = direct if direct is not None else data_dir
            protein = find_file(search_root, cid, "protein")
            ligand = find_file(search_root, cid, "ligand")
            if protein is None or ligand is None:
                continue
            fmt = ligand.suffix.lower().lstrip(".")
        complexes.append(
            ComplexInput(
                complex_id=cid,
                protein=protein,
                ligand_ref=ligand,
                ligand_format=fmt,
                smiles=smiles_by_id.get(cid),
                pocket_center=pocket_centers[cid],
            )
        )
    if missing_centers:
        preview = ", ".join(missing_centers[:5])
        raise ValueError(
            f"frozen pocket centers missing for {len(missing_centers)} benchmark IDs "
            f"(first: {preview})"
        )
    return complexes


def _sample_center_jitter(seed: int, sigma: float) -> torch.Tensor:
    """Sample a paired center perturbation without consuming the global RNG."""
    jitter_generator = torch.Generator(device="cpu")
    jitter_generator.manual_seed(seed)
    return sigma * torch.randn(3, dtype=torch.float32, generator=jitter_generator)


def evaluate_one(
    model: torch.nn.Module,
    item: ComplexInput,
    *,
    confidence_model: torch.nn.Module | None,
    device: torch.device,
    num_samples: int,
    num_steps: int,
    sigma: float,
    sigma_list: list[float],
    sigma_counts: list[int],
    center_jitter_sigma: float,
    pocket_cutoff: float,
    pose_objective: str,
    score_rot_sigma_max: float,
    score_alpha_min: float,
    time_schedule: str,
    schedule_power: float,
    vina_guidance_scale: float,
    vina_guidance_start_t: float,
    vina_guidance_ramp_power: float,
    vina_guidance_max_force: float,
    vina_guidance_max_velocity: float,
    vina_guidance_max_angular_velocity: float,
    vina_guidance_protein_shell: float,
    vina_guidance_w_strain: float,
    seed: int,
    refine: str,
    pose_dir: Path | None,
) -> dict:
    torch.manual_seed(seed)
    mol_ref = load_ref_ligand(item.ligand_ref, item.ligand_format)
    ligand_input = item.smiles if item.smiles else str(item.ligand_ref)
    mol_in, _ = load_ligand(ligand_input)
    ref_pos_abs = torch.tensor(mol_ref.GetConformer().GetPositions(), dtype=torch.float32)
    pocket_center = torch.tensor(item.pocket_center, dtype=torch.float32)
    if center_jitter_sigma > 0.0:
        # Keep the sampling RNG paired across jitter conditions.  A dedicated
        # CPU generator also makes 1A and 2A use the same perturbation direction
        # for a given complex, differing only in magnitude.
        pocket_center = pocket_center + _sample_center_jitter(seed, center_jitter_sigma)
    graph, lig_data, meta = preprocess_complex(
        item.protein,
        mol_in,
        pocket_center=pocket_center,
        pocket_cutoff=pocket_cutoff,
    )
    guidance_fn = None
    if vina_guidance_scale != 0.0:
        if vina_guidance_scale < 0.0:
            raise ValueError("vina_guidance_scale must be non-negative")
        from effdock.evaluation.vina_guidance import VinaGuidanceConfig, build_vina_guidance

        guidance_fn = build_vina_guidance(
            mol_in,
            item.protein,
            pocket_center=meta["pocket_center"],
            frag_id=lig_data["fragment_id"],
            device=device,
            protein_shell_cutoff=vina_guidance_protein_shell,
            config=VinaGuidanceConfig(
                start_t=vina_guidance_start_t,
                ramp_power=vina_guidance_ramp_power,
                max_atom_force=vina_guidance_max_force,
                max_translation_velocity=vina_guidance_max_velocity,
                max_angular_velocity=vina_guidance_max_angular_velocity,
                w_strain=vina_guidance_w_strain,
            ),
        )
    if sigma_list:
        results = sample_unified_multi_sigma(
            model,
            graph,
            lig_data,
            meta,
            sigma_list=sigma_list,
            samples_per_sigma=sigma_counts,
            num_steps=num_steps,
            time_schedule=time_schedule,
            schedule_power=schedule_power,
            pose_objective=pose_objective,
            score_rot_sigma_max=score_rot_sigma_max,
            score_alpha_min=score_alpha_min,
            device=device,
            guidance_fn=guidance_fn,
            guidance_scale=vina_guidance_scale,
            guidance_min_t=vina_guidance_start_t,
        )
    else:
        results = sample_unified(
            model,
            graph,
            lig_data,
            meta,
            num_samples=num_samples,
            num_steps=num_steps,
            translation_sigma=sigma,
            time_schedule=time_schedule,
            schedule_power=schedule_power,
            pose_objective=pose_objective,
            score_rot_sigma_max=score_rot_sigma_max,
            score_alpha_min=score_alpha_min,
            device=device,
            guidance_fn=guidance_fn,
            guidance_scale=vina_guidance_scale,
            guidance_min_t=vina_guidance_start_t,
        )
    poses = [result["atom_pos_pred"].detach().cpu() for result in results]
    poses = apply_refinement(refine, poses, mol_in, meta["pocket_center"])

    ref_pos = ref_pos_abs - meta["pocket_center"].cpu()
    dock_idx, ref_idx, match_method = match_atoms(mol_ref, mol_in)
    if not dock_idx:
        raise RuntimeError("atom matching failed")
    ref_pos = ref_pos.index_select(0, torch.as_tensor(ref_idx, dtype=torch.long))
    rmsds = [
        compute_pose_rmsd(
            pose.cpu(),
            ref_pos,
            meta["pocket_center"].cpu(),
            dock_idx,
            mol_in,
            mol_ref,
        )
        for pose in poses
    ]
    vina_scores = score_poses(
        mol_in,
        poses,
        item.protein,
        pocket_center=meta["pocket_center"].cpu(),
        frag_id=lig_data["fragment_id"].cpu(),
    )
    first_i = 0
    vina_i = select_by_score([score["total"] for score in vina_scores])
    oracle_i = select_pose("oracle", rmsds)
    selector_indices = {"first": first_i, "vina": vina_i, "oracle": oracle_i}
    confidence_scores: list[dict[str, float]] | None = None
    if confidence_model is not None:
        from effdock.confidence.runtime import sample_sigmas, score_poses_with_confidence
        from effdock.confidence.selectors import select_confidence_poses

        confidence_scores = score_poses_with_confidence(
            confidence_model,
            model,
            graph,
            lig_data,
            meta,
            poses,
            sigma=sample_sigmas(results, sigma),
            device=device,
        )
        confidence_indices = select_confidence_poses(
            poses, confidence_scores, graph, meta["pocket_center"]
        )
        selector_indices["confidence"] = confidence_indices["confidence"]
        selector_indices["confidence_filter"] = confidence_indices["confidence_filter_v1"]
        selector_indices["confidence_final"] = confidence_indices[
            "pair_gate_density_rank_vote_plclash_ambig"
        ]

    absolute_poses = [pose.cpu() + meta["pocket_center"].cpu() for pose in poses]
    prot = build_protein_vina_inputs(
        item.protein,
        torch.cat(absolute_poses, dim=0),
        cutoff=10.0,
    )
    bounds = ligand_bounds(mol_in)
    lig_atomic_nums = torch.tensor([atom.GetAtomicNum() for atom in mol_in.GetAtoms()])
    lig_r = vdw_radii(lig_atomic_nums)
    prot_r = vdw_radii(prot["atomic_nums"])

    validity: dict[str, dict[str, bool]] = {}
    for selector, index in selector_indices.items():
        validity[selector] = check_validity(
            absolute_poses[index],
            bounds,
            prot_xyz=prot["coords"],
            prot_r=prot_r,
            lig_r=lig_r,
            return_terms=True,
        )
        if pose_dir is not None:
            _write_pose(
                mol_in,
                poses[index],
                meta["pocket_center"],
                pose_dir / selector / f"{item.complex_id}.sdf",
            )

    row = {
        "id": item.complex_id,
        "protein": str(item.protein),
        "ligand_ref": str(item.ligand_ref),
        "num_samples": num_samples,
        "first_index": first_i,
        "vina_index": vina_i,
        "oracle_index": oracle_i,
        "first_rmsd": float(rmsds[first_i]),
        "vina_rmsd": float(rmsds[vina_i]),
        "oracle_rmsd": float(rmsds[oracle_i]),
        "vina_score": vina_scores[vina_i]["vina"],
        "vina_strain": vina_scores[vina_i]["strain"],
        "vina_total": vina_scores[vina_i]["total"],
        "mean_sample_rmsd": float(np.mean(rmsds)),
        **{
            f"{selector}_fast_{term}": value
            for selector, terms in validity.items()
            for term, value in terms.items()
        },
        "match_method": match_method,
        "num_match_atoms": len(dock_idx),
        "num_input_atoms": mol_in.GetNumAtoms(),
        "num_ref_atoms": mol_ref.GetNumAtoms(),
    }
    if confidence_scores is not None:
        for selector in ("confidence", "confidence_filter", "confidence_final"):
            index = selector_indices[selector]
            row[f"{selector}_index"] = index
            row[f"{selector}_rmsd"] = float(rmsds[index])
            row[f"{selector}_pred_rmsd"] = confidence_scores[index]["confidence_rmsd"]
            row[f"{selector}_pred_success"] = confidence_scores[index]["confidence_success"]
    return row


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("astex", "posebusters", "casf"), required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--external-dir", type=Path, default=Path("data/external_test"))
    parser.add_argument(
        "--pocket-centers",
        type=Path,
        required=True,
        help="Frozen JSON mapping benchmark IDs to declared [x,y,z] pocket centers.",
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("weights/effdock_legacy_flowfrag_200k_ema.pt")
    )
    parser.add_argument(
        "--confidence-checkpoint",
        type=Path,
        default=None,
        help="Optional learned confidence checkpoint evaluated on the same sampled poses.",
    )
    parser.add_argument("--config", type=Path, default=Path("configs/train.yaml"))
    parser.add_argument(
        "--output-dir", dest="out_dir", type=Path, default=Path("outputs/external_benchmarks")
    )
    parser.add_argument("--num-samples", type=int, default=40)
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--sigma", type=float, default=None)
    parser.add_argument(
        "--sigma-list",
        type=str,
        default=None,
        help='Multi-sigma sampling, e.g. "2.5,3.0,3.5" or "2.5:14,3.0:13,3.5:13".',
    )
    parser.add_argument("--time-schedule", type=str, default="late")
    parser.add_argument("--schedule-power", type=float, default=3.0)
    parser.add_argument("--vina-guidance-scale", type=float, default=0.0)
    parser.add_argument("--vina-guidance-start-t", type=float, default=0.5)
    parser.add_argument("--vina-guidance-ramp-power", type=float, default=1.0)
    parser.add_argument("--vina-guidance-max-force", type=float, default=10.0)
    parser.add_argument("--vina-guidance-max-velocity", type=float, default=5.0)
    parser.add_argument("--vina-guidance-max-angular-velocity", type=float, default=5.0)
    parser.add_argument("--vina-guidance-protein-shell", type=float, default=18.0)
    parser.add_argument("--vina-guidance-w-strain", type=float, default=1.0)
    parser.add_argument("--center-jitter-sigma", type=float, default=0.0)
    parser.add_argument("--pocket-cutoff", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--only-id",
        action="append",
        default=[],
        help="Evaluate only this ID while preserving its full-dataset seed; repeatable.",
    )
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--protocol-id", type=str, default=None)
    parser.add_argument(
        "--save-selected-poses",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--refine", choices=("none", "mmff"), default="none")
    args = parser.parse_args(argv)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, cfg, ckpt = load_model(args.config, args.checkpoint, device)
    confidence_model = None
    confidence_ckpt = None
    if args.confidence_checkpoint is not None:
        from effdock.confidence.runtime import load_pose_confidence_model

        confidence_model, confidence_ckpt = load_pose_confidence_model(
            args.confidence_checkpoint, device
        )
    sigma = float(args.sigma if args.sigma is not None else cfg["data"].get("prior_sigma", 3.5))
    pose_objective = cfg.get("data", {}).get("pose_objective", "linear_fm")
    score_rot_sigma_max = float(cfg.get("data", {}).get("score_rot_sigma_max", torch.pi))
    score_alpha_min = float(cfg.get("data", {}).get("score_alpha_min", 0.0))
    sigma_list, sigma_counts = parse_sigma_list(args.sigma_list, args.num_samples)
    pocket_centers = load_pocket_centers(args.pocket_centers)
    complexes = discover_complexes(args.dataset, args.data_dir, args.external_dir, pocket_centers)
    if args.limit:
        complexes = complexes[: args.limit]
    total_discovered = len(complexes)
    seed_by_id = {
        item.complex_id: args.seed + global_index
        for global_index, item in enumerate(complexes, start=1)
    }
    if args.only_id:
        requested = {complex_id.lower() for complex_id in args.only_id}
        complexes = [item for item in complexes if item.complex_id.lower() in requested]
        found = {item.complex_id.lower() for item in complexes}
        if missing := requested - found:
            raise ValueError(f"requested benchmark IDs not found: {sorted(missing)}")
    complexes = shard_complexes(complexes, args.shard_index, args.num_shards)
    if not complexes:
        raise SystemExit(f"No complexes found in {args.data_dir}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    failures: list[dict] = []
    print(
        f"checkpoint={args.checkpoint} step={ckpt.get('step')} "
        f"dataset={args.dataset} complexes={len(complexes)}/{total_discovered} "
        f"shard={args.shard_index}/{args.num_shards} samples={args.num_samples}"
    )
    for i, item in enumerate(complexes, start=1):
        try:
            row = evaluate_one(
                model,
                item,
                confidence_model=confidence_model,
                device=device,
                num_samples=args.num_samples,
                num_steps=args.num_steps,
                sigma=sigma,
                sigma_list=sigma_list,
                sigma_counts=sigma_counts,
                center_jitter_sigma=args.center_jitter_sigma,
                pocket_cutoff=args.pocket_cutoff,
                pose_objective=pose_objective,
                score_rot_sigma_max=score_rot_sigma_max,
                score_alpha_min=score_alpha_min,
                time_schedule=args.time_schedule,
                schedule_power=args.schedule_power,
                vina_guidance_scale=args.vina_guidance_scale,
                vina_guidance_start_t=args.vina_guidance_start_t,
                vina_guidance_ramp_power=args.vina_guidance_ramp_power,
                vina_guidance_max_force=args.vina_guidance_max_force,
                vina_guidance_max_velocity=args.vina_guidance_max_velocity,
                vina_guidance_max_angular_velocity=args.vina_guidance_max_angular_velocity,
                vina_guidance_protein_shell=args.vina_guidance_protein_shell,
                vina_guidance_w_strain=args.vina_guidance_w_strain,
                seed=seed_by_id[item.complex_id],
                refine=args.refine,
                pose_dir=(
                    args.out_dir / "poses" / (args.run_name or args.dataset) / args.dataset
                    if args.save_selected_poses
                    else None
                ),
            )
            rows.append(row)
            print(
                f"[{i:04d}/{len(complexes)}] {item.complex_id} "
                f"first={row['first_rmsd']:.3f} vina={row['vina_rmsd']:.3f} "
                + (
                    f"confidence={row['confidence_rmsd']:.3f} "
                    f"confidence_filter={row['confidence_filter_rmsd']:.3f} "
                    f"confidence_final={row['confidence_final_rmsd']:.3f} "
                    if confidence_model is not None
                    else ""
                )
                + f"oracle={row['oracle_rmsd']:.3f}"
            )
        except Exception as exc:
            failures.append({"id": item.complex_id, "error": repr(exc)})
            print(f"[{i:04d}/{len(complexes)}] {item.complex_id} FAIL {exc!r}")

    sigma_tag = (
        "mix" + "-".join(f"{s:g}x{n}" for s, n in zip(sigma_list, sigma_counts))
        if sigma_list
        else f"sig{sigma:g}"
    )
    jitter_tag = f"_cj{args.center_jitter_sigma:g}" if args.center_jitter_sigma > 0.0 else ""
    cutoff_tag = f"_pc{args.pocket_cutoff:g}" if args.pocket_cutoff != 8.0 else ""
    base_tag = args.run_name or (
        f"{args.dataset}_{args.checkpoint.stem}_n{args.num_samples}_s{args.num_steps}_"
        f"{sigma_tag}{jitter_tag}{cutoff_tag}"
    )
    shard_tag = f".shard-{args.shard_index:03d}-of-{args.num_shards:03d}"
    tag = f"{base_tag}{shard_tag}" if args.num_shards > 1 else base_tag
    csv_path = args.out_dir / f"{tag}.csv"
    if rows:
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    summary = {
        "dataset": args.dataset,
        "protocol_id": args.protocol_id,
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": ckpt.get("step"),
        "confidence_checkpoint": str(args.confidence_checkpoint)
        if args.confidence_checkpoint is not None
        else None,
        "confidence_step": confidence_ckpt.get("step") if confidence_ckpt else None,
        "config": str(args.config),
        "data_dir": str(args.data_dir),
        "num_discovered_total": total_discovered,
        "num_assigned": len(complexes),
        "num_success": len(rows),
        "num_failed": len(failures),
        "num_samples": args.num_samples,
        "num_steps": args.num_steps,
        "sigma": sigma,
        "sigma_list": sigma_list,
        "sigma_counts": sigma_counts,
        "pose_objective": pose_objective,
        "score_rot_sigma_max": score_rot_sigma_max,
        "score_alpha_min": score_alpha_min,
        "center_jitter_sigma": args.center_jitter_sigma,
        "pocket_cutoff": args.pocket_cutoff,
        "time_schedule": args.time_schedule,
        "schedule_power": args.schedule_power,
        "vina_guidance_scale": args.vina_guidance_scale,
        "vina_guidance_start_t": args.vina_guidance_start_t,
        "vina_guidance_ramp_power": args.vina_guidance_ramp_power,
        "vina_guidance_max_force": args.vina_guidance_max_force,
        "vina_guidance_max_velocity": args.vina_guidance_max_velocity,
        "vina_guidance_max_angular_velocity": args.vina_guidance_max_angular_velocity,
        "vina_guidance_protein_shell": args.vina_guidance_protein_shell,
        "vina_guidance_w_strain": args.vina_guidance_w_strain,
        "refine": args.refine,
        "seed": args.seed,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "pocket_centers": str(args.pocket_centers),
        "pocket_centers_sha256": file_sha256(args.pocket_centers),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "confidence_checkpoint_sha256": file_sha256(args.confidence_checkpoint)
        if args.confidence_checkpoint is not None
        else None,
        "config_sha256": file_sha256(args.config),
        "runtime": {
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
        "stats": summarize_rows(rows),
        "failures": failures,
        "csv": str(csv_path) if rows else None,
    }
    summary_path = args.out_dir / f"{tag}.summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
