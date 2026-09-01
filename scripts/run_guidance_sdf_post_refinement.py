#!/usr/bin/env python3
"""Refine one frozen 100-pose SDF with the self-contained GuidanceEnergy."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from rdkit import Chem

from effdock.evaluation.benchmark import (
    compute_pose_rmsd,
    match_atoms,
)
from effdock.evaluation.benchmark import (
    load_ligand as load_ref_ligand,
)
from effdock.guidance import InteractionEnergyConfig, build_physical_system
from effdock.guidance.parameterization import guidance_parameter_identity
from effdock.guidance.provenance import (
    guidance_implementation_identity,
    physical_system_reference_sha256,
)
from effdock.inference.io import write_multi_sdf
from effdock.inference.preprocess import preprocess_complex
from effdock.workflows.benchmark_inputs import (
    ligand_input_identity,
    load_benchmark_inputs,
    load_benchmark_ligand,
)
from effdock.workflows.evaluate import file_sha256
from effdock.workflows.relax_guidance import (
    RigidRelaxationConfig,
    relax_rigid_fragments_batch,
)

PROTOCOL_ID = "EFFDOCK-GUIDANCE-SDF-POST-REFINEMENT-V1"
SCHEMA_VERSION = "effdock.guidance_sdf_post_refinement.v1"
EXPECTED_POSES = 100
SUPPORTED_SOURCE_PROTOCOLS = {
    "EFFDOCK-GUIDANCE-ALL-POSE-PB-ETA-V1",
    "EFFDOCK-GUIDANCE-SIGMA2-ETA2-REFINEMENT-INPUT-V1",
    "EFFDOCK-EXTERNAL-TEMPORAL-GUIDED-REFINED-V1",
    "EFFDOCK-POCKET-CUTOFF-ROBUSTNESS-MANIFEST-V1",
}


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def _tensor_sha256(value: torch.Tensor) -> str:
    array = value.detach().cpu().to(torch.float32).contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(b"EFFDOCK_GUIDANCE_SDF_REFINEMENT_TENSOR_V1\0")
    digest.update(str(array.dtype).encode())
    digest.update(b"\0")
    digest.update(_canonical_json(list(array.shape)).encode())
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _graph_signature(mol: Chem.Mol) -> tuple[tuple[int, ...], tuple[tuple[int, int], ...]]:
    elements = tuple(atom.GetAtomicNum() for atom in mol.GetAtoms())
    edges = tuple(
        sorted(
            tuple(sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))) for bond in mol.GetBonds()
        )
    )
    return elements, edges


def _load_pose_batch(path: Path, template: Chem.Mol) -> tuple[torch.Tensor, list[dict[str, str]]]:
    # The random-access SDMolSupplier can seek one byte early when the next
    # record begins immediately after a 64-KiB boundary, shifting the mol-block
    # header and returning None for an otherwise valid record.  Sequential
    # parsing avoids that offset path and preserves every source pose.
    with path.open("rb") as stream:
        molecules = list(Chem.ForwardSDMolSupplier(stream, removeHs=False, sanitize=False))
    if len(molecules) != EXPECTED_POSES or any(mol is None for mol in molecules):
        loaded = sum(mol is not None for mol in molecules)
        raise ValueError(
            f"saved SDF must contain {EXPECTED_POSES} readable poses, got "
            f"records={len(molecules)} readable={loaded}"
        )
    expected_graph = _graph_signature(template)
    coords: list[torch.Tensor] = []
    properties: list[dict[str, str]] = []
    for pose_index, raw in enumerate(molecules):
        if raw is None:
            raise AssertionError("unreachable unreadable pose")
        if _graph_signature(raw) != expected_graph:
            raise ValueError(f"pose {pose_index} atom order/connectivity differs from ligand input")
        value = torch.tensor(raw.GetConformer().GetPositions(), dtype=torch.float32)
        if value.shape != (template.GetNumAtoms(), 3) or not bool(torch.isfinite(value).all()):
            raise ValueError(f"pose {pose_index} has invalid coordinates")
        coords.append(value)
        properties.append(
            {
                name: raw.GetProp(name)
                for name in raw.GetPropNames(includePrivate=False, includeComputed=False)
            }
        )
    return torch.stack(coords), properties


def _load_source_row(manifest: dict[str, Any], record: dict[str, Any]) -> dict[str, str]:
    matches: list[dict[str, str]] = []
    for source in manifest.get("source_files", []):
        path = Path(str(source["path"]))
        if not path.is_file() or file_sha256(path) != source["sha256"]:
            raise ValueError(f"changed source sampling CSV: {path}")
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if (
                    row.get("id") == record["id"]
                    and row.get("all_poses_sdf") == record["pose_path"]
                ):
                    matches.append(row)
    if len(matches) != 1:
        raise ValueError(f"expected one source sampling row, found {len(matches)}")
    return matches[0]


def _select_record(
    manifest: dict[str, Any], *, dataset: str, eta: float, complex_id: str
) -> dict[str, Any]:
    matches = [
        row
        for row in manifest.get("records", [])
        if row.get("dataset") == dataset
        and row.get("id") == complex_id.lower()
        and math.isclose(float(row.get("eta", math.nan)), eta, abs_tol=1e-12)
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one manifest record, found {len(matches)}")
    return matches[0]


def _aligned_crystal(
    mol_ref: Chem.Mol,
    mol_input: Chem.Mol,
) -> tuple[torch.Tensor, list[int], list[int], str]:
    dock_indices, ref_indices, method = match_atoms(mol_ref, mol_input)
    atom_count = mol_input.GetNumAtoms()
    if (
        len(dock_indices) != atom_count
        or sorted(dock_indices) != list(range(atom_count))
        or len(ref_indices) != mol_ref.GetNumAtoms()
    ):
        raise ValueError("refinement requires a complete crystal/input atom mapping")
    ref = torch.tensor(mol_ref.GetConformer().GetPositions(), dtype=torch.float32)
    aligned = torch.empty((atom_count, 3), dtype=torch.float32)
    for dock_index, ref_index in zip(dock_indices, ref_indices, strict=True):
        aligned[dock_index] = ref[ref_index]
    return aligned, dock_indices, ref_indices, method


def _step_frames(
    *,
    initial: torch.Tensor,
    crystal: torch.Tensor,
    system,
    pocket_center: torch.Tensor,
    config: RigidRelaxationConfig,
    batch_size: int,
    dense_diagnostics: bool = False,
) -> tuple[dict[int, torch.Tensor], list[dict[str, Any]]]:
    frames_by_step: dict[int, list[torch.Tensor]] = {}
    pose_summaries: list[dict[str, Any]] = []
    interaction = InteractionEnergyConfig()
    for start in range(0, initial.shape[0], batch_size):
        stop = min(start + batch_size, initial.shape[0])
        run = relax_rigid_fragments_batch(
            crystal,
            initial[start:stop],
            system,
            config=config,
            mode="unified",
            pocket_center=pocket_center,
            interaction_config=interaction,
            collect_every_step_metrics=dense_diagnostics,
            collect_contact_stats=dense_diagnostics,
        )
        run_frames = {
            int(saved_step): frame
            for saved_step, frame in zip(run.saved_steps, run.frames, strict=True)
        }
        # The lightweight production trace emits scheduled frames and may stop
        # before max_steps if every pose is inactive. Materialize the frozen
        # scheduled cohort frames by carrying each finite terminal coordinate
        # forward without relabeling its status.
        for target_step in range(0, config.max_steps + 1, config.save_every):
            available = [step for step in run_frames if step <= target_step]
            selected_step = max(available) if available else min(run_frames)
            frames_by_step.setdefault(target_step, []).append(run_frames[selected_step])
        for offset in range(stop - start):
            initial_metrics = run.metrics[offset][0]
            final_metrics = run.metrics[offset][-1]
            metrics_by_step = {int(row["step"]): row for row in run.metrics[offset]}
            saved_total_energy_by_step: dict[str, float] = {}
            for target_step in range(0, config.max_steps + 1, config.save_every):
                available = [step for step in metrics_by_step if step <= target_step]
                selected_step = max(available) if available else min(metrics_by_step)
                saved_total_energy_by_step[str(target_step)] = float(
                    metrics_by_step[selected_step]["energy_groups"]["combined"]
                )
            pose_summaries.append(
                {
                    "pose_index": start + offset,
                    "status": run.statuses[offset],
                    "terminal_step": run.terminal_steps[offset],
                    "total_backtracks": run.total_backtracks[offset],
                    "shell_envelope_valid": run.shell_envelope_valid[offset],
                    "initial_total_energy": initial_metrics["energy_groups"]["combined"],
                    "final_total_energy": final_metrics["energy_groups"]["combined"],
                    "saved_total_energy_by_step": saved_total_energy_by_step,
                    "initial_raw_rmsd_angstrom": initial_metrics["raw_rmsd_angstrom"],
                    "final_raw_rmsd_angstrom": final_metrics["raw_rmsd_angstrom"],
                    "initial_chiral_improper_inversion_count": initial_metrics.get(
                        "chiral_improper_inversion_count", 0
                    ),
                    "final_chiral_improper_inversion_count": final_metrics.get(
                        "chiral_improper_inversion_count", 0
                    ),
                }
            )
    combined = {step: torch.cat(parts, dim=0) for step, parts in frames_by_step.items()}
    if set(combined) != set(range(0, config.max_steps + 1, config.save_every)):
        raise ValueError(f"unexpected saved-step inventory: {sorted(combined)}")
    if any(frame.shape != initial.shape for frame in combined.values()):
        raise ValueError("refined frame shape mismatch")
    return combined, pose_summaries


def main() -> None:
    pipeline_started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--benchmark-input-manifest", type=Path, default=None)
    parser.add_argument("--external-dir", type=Path, default=Path("data/external_test"))
    parser.add_argument("--pocket-centers", type=Path, required=True)
    parser.add_argument("--protocol-file", type=Path, required=True)
    parser.add_argument(
        "--dataset",
        choices=("astex", "posebusters", "phibench", "foldbench", "openbind"),
        required=True,
    )
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--complex-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--pocket-cutoff", type=float, default=10.0)
    parser.add_argument("--protein-shell", type=float, default=18.0)
    parser.add_argument("--energy-convergence-absolute-kcal-mol", type=float)
    parser.add_argument("--energy-convergence-relative", type=float)
    parser.add_argument("--energy-convergence-patience", type=int, default=5)
    parser.add_argument("--energy-convergence-min-steps", type=int, default=20)
    parser.add_argument(
        "--dense-diagnostics",
        action="store_true",
        help="Debug-only legacy trace with per-step metrics and contact decomposition.",
    )
    parser.add_argument("--receptor-policy", choices=("geometry_only",), default="geometry_only")
    args = parser.parse_args()
    if args.steps != 100 or args.save_every != 25:
        raise ValueError("V1 protocol requires --steps 100 --save-every 25")
    if args.batch_size < 2 or EXPECTED_POSES % args.batch_size:
        raise ValueError("batch-size must be a divisor of 100 and at least two")
    if not math.isfinite(args.pocket_cutoff) or args.pocket_cutoff <= 0:
        raise ValueError("pocket-cutoff must be finite and positive")
    if not torch.cuda.is_available() and str(args.device).startswith("cuda"):
        raise RuntimeError("CUDA device requested but unavailable")

    protocol_path = args.protocol_file.resolve()
    if not protocol_path.is_file():
        raise FileNotFoundError(f"missing protocol file: {protocol_path}")

    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("protocol_id") not in SUPPORTED_SOURCE_PROTOCOLS:
        raise ValueError("unexpected all-pose manifest protocol")
    record = _select_record(
        manifest,
        dataset=args.dataset,
        eta=args.eta,
        complex_id=args.complex_id,
    )
    source_row = _load_source_row(manifest, record)
    for path_key, hash_key in (
        ("pose_path", "pose_sha256"),
        ("protein", "protein_sha256"),
        ("ligand_ref", "ligand_ref_sha256"),
    ):
        path = Path(record[path_key])
        if not path.is_file() or file_sha256(path) != record[hash_key]:
            raise ValueError(f"changed frozen input: {path}")

    mapping, mapping_identity = load_benchmark_inputs(
        args.dataset,
        args.external_dir,
        args.benchmark_input_manifest,
    )
    raw_smiles = mapping[record["id"]]
    expected_identity = ligand_input_identity(record["id"], raw_smiles)
    if source_row.get("ligand_input_identity_sha256") != expected_identity["sha256"]:
        raise ValueError("source sampling ligand-input identity mismatch")
    seed = int(record["sampling_seed"])
    mol_input, _ = load_benchmark_ligand(raw_smiles, random_seed=seed)
    initial_absolute, original_properties = _load_pose_batch(Path(record["pose_path"]), mol_input)

    centers = json.loads(args.pocket_centers.read_text())
    center_entry = centers.get(record["id"], centers.get(record["id"].lower()))
    if isinstance(center_entry, dict):
        center_entry = center_entry.get("pocket_center", center_entry.get("center"))
    pocket_center_absolute = torch.as_tensor(center_entry, dtype=torch.float32)
    if pocket_center_absolute.shape != (3,) or not bool(
        torch.isfinite(pocket_center_absolute).all()
    ):
        raise ValueError("missing or invalid frozen pocket center")

    mol_ref = load_ref_ligand(Path(record["ligand_ref"]), "sdf")
    crystal_absolute, dock_indices, ref_indices, match_method = _aligned_crystal(mol_ref, mol_input)
    _, ligand_data, meta = preprocess_complex(
        Path(record["protein"]),
        mol_input,
        pocket_center=pocket_center_absolute,
        pocket_cutoff=args.pocket_cutoff,
    )
    if not torch.equal(meta["pocket_center"], pocket_center_absolute):
        raise AssertionError("preprocessing changed pocket center")
    system = build_physical_system(
        mol_input,
        Path(record["protein"]),
        fragment_id=ligand_data["fragment_id"],
        near_coords=pocket_center_absolute.view(1, 3),
        protein_cutoff=args.protein_shell,
        coordinate_origin=pocket_center_absolute,
        receptor_policy=args.receptor_policy,
    )
    device = torch.device(args.device)
    system = system.to(device=device, dtype=torch.float32)
    initial = (initial_absolute - pocket_center_absolute).to(device)
    crystal = (crystal_absolute - pocket_center_absolute).to(device)
    zero_center = torch.zeros(3, device=device, dtype=torch.float32)
    config = RigidRelaxationConfig(
        initialization_mode="model_prior",
        prior_sigma_angstrom=float(record.get("sigma", 0.5)),
        max_steps=args.steps,
        save_every=args.save_every,
        base_step_size=1.0,
        max_translation_step_angstrom=0.10,
        max_rotation_step_degrees=5.0,
        max_atom_step_angstrom=0.10,
        max_backtracks=12,
        convergence_displacement_angstrom=1e-5,
        convergence_patience=20,
        convergence_energy_absolute_kcal_mol=(args.energy_convergence_absolute_kcal_mol),
        convergence_energy_relative=args.energy_convergence_relative,
        convergence_energy_patience=args.energy_convergence_patience,
        convergence_energy_min_steps=args.energy_convergence_min_steps,
        physical_cutoff_angstrom=8.0,
        protein_shell_cutoff_angstrom=args.protein_shell,
    )
    _synchronize(device)
    refinement_started = time.perf_counter()
    frames, pose_summaries = _step_frames(
        initial=initial,
        crystal=crystal,
        system=system,
        pocket_center=zero_center,
        config=config,
        batch_size=args.batch_size,
        dense_diagnostics=args.dense_diagnostics,
    )
    _synchronize(device)
    refinement_seconds = time.perf_counter() - refinement_started

    rmsd_started = time.perf_counter()
    for pose_index, row in enumerate(pose_summaries):
        for label, value in (("initial", frames[0]), ("final", frames[args.steps])):
            row[f"{label}_symmetry_rmsd_angstrom"] = compute_pose_rmsd(
                value[pose_index].cpu(),
                (crystal_absolute[ref_indices] - pocket_center_absolute),
                pocket_center_absolute,
                dock_indices,
                mol_input,
                mol_ref,
            )
    rmsd_evaluation_seconds = time.perf_counter() - rmsd_started

    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    attempt = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    serialization_started = time.perf_counter()
    artifacts: dict[str, dict[str, str]] = {}
    for step, value in sorted(frames.items()):
        path = attempt / f"step_{step:03d}.sdf"
        per_pose = []
        for pose_index, original in enumerate(original_properties):
            kept = {
                name: original[name]
                for name in (
                    "sample_index",
                    "confidence_pred_rmsd",
                    "confidence_pred_success",
                    "pl_clash_1p6",
                    "fast_valid",
                )
                if name in original
            }
            kept.update(
                {
                    "guidance_refinement_protocol": PROTOCOL_ID,
                    "guidance_refinement_step": step,
                    "guidance_refinement_status": pose_summaries[pose_index]["status"],
                }
            )
            per_pose.append(kept)
        write_multi_sdf(
            mol_input,
            [pose.cpu() for pose in value],
            pocket_center_absolute,
            path,
            props={
                "dataset": args.dataset,
                "complex_id": record["id"],
                "source_eta": args.eta,
            },
            per_pose_props=per_pose,
            force_v3000=True,
        )
        artifacts[f"step_{step:03d}_sdf"] = {
            "path": str(output_dir / path.name),
            "sha256": file_sha256(path),
        }
    trajectory = attempt / "trajectory.pt"
    torch.save(
        {
            "schema_version": SCHEMA_VERSION,
            "protocol_id": PROTOCOL_ID,
            "saved_steps": torch.tensor(sorted(frames)),
            "frames_pocket_centered": torch.stack([frames[step].cpu() for step in sorted(frames)]),
            "pocket_center_absolute": pocket_center_absolute,
            "fragment_id": ligand_data["fragment_id"].cpu(),
        },
        trajectory,
    )
    artifacts["trajectory_pt"] = {
        "path": str(output_dir / trajectory.name),
        "sha256": file_sha256(trajectory),
    }
    serialization_seconds = time.perf_counter() - serialization_started
    summary = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "complete_descriptive",
        "claim_boundary": "saved-pose guidance-only post-refinement; no learned ODE, Vina, or external minimizer",
        "inputs": {
            "protocol_file": str(protocol_path),
            "protocol_file_sha256": file_sha256(protocol_path),
            "manifest": str(manifest_path),
            "manifest_sha256": file_sha256(manifest_path),
            "dataset": args.dataset,
            "complex_id": record["id"],
            "eta": args.eta,
            "source_sampling_sigma_angstrom": record.get("sigma"),
            "source_guidance_mode": source_row.get("guidance_mode"),
            "source_manifest_protocol_id": manifest.get("protocol_id"),
            "pose_sdf": record["pose_path"],
            "pose_sdf_sha256": record["pose_sha256"],
            "protein": record["protein"],
            "protein_sha256": record["protein_sha256"],
            "ligand_reference": record["ligand_ref"],
            "ligand_reference_sha256": record["ligand_ref_sha256"],
            "sampling_seed": seed,
            "ligand_input_identity": expected_identity,
            "benchmark_mapping_identity": mapping_identity,
            "pocket_centers": str(args.pocket_centers.resolve()),
            "pocket_centers_sha256": file_sha256(args.pocket_centers),
            "pocket_center_absolute": pocket_center_absolute.tolist(),
            "pocket_cutoff_angstrom": args.pocket_cutoff,
        },
        "implementation": {
            "guidance": guidance_implementation_identity(),
            "parameters": guidance_parameter_identity(),
            "system_reference_sha256": physical_system_reference_sha256(system),
            "torch": torch.__version__,
            "device": str(device),
        },
        "solver": {
            "coordinate_variables": "rigid fragment SE(3) translation and rotation",
            "gradient": "negative Torch autograd gradient projected by fragment Newton-Euler aggregation",
            "acceptance": "independent pose-wise monotone backtracking line search",
            "diagnostics": (
                "every_step_with_contact_decomposition"
                if args.dense_diagnostics
                else "scheduled_steps_only_without_contact_decomposition"
            ),
            "saved_pose_initialization": True,
            "internal_relaxation_shell_mode": "explicit_pocket (model_prior config label only; no prior resampling)",
            "config": asdict(config),
            "batch_size": args.batch_size,
        },
        "mapping": {
            "method": match_method,
            "dock_indices": dock_indices,
            "ref_indices": ref_indices,
            "saved_pose_graph_matches_input": True,
        },
        "counts": {
            "poses": EXPECTED_POSES,
            "completed_or_converged": sum(
                row["status"] in {"max_steps", "converged_displacement", "converged_energy_plateau"}
                for row in pose_summaries
            ),
            "line_search_failed": sum(
                row["status"] == "line_search_failed" for row in pose_summaries
            ),
            "finite_terminal": sum(
                row["status"]
                in {
                    "max_steps",
                    "converged_displacement",
                    "converged_energy_plateau",
                    "line_search_failed",
                }
                for row in pose_summaries
            ),
            "failed": sum(
                row["status"]
                not in {
                    "max_steps",
                    "converged_displacement",
                    "converged_energy_plateau",
                    "line_search_failed",
                }
                for row in pose_summaries
            ),
        },
        "coordinate_hashes": {
            f"step_{step:03d}": _tensor_sha256(value) for step, value in sorted(frames.items())
        },
        "poses": pose_summaries,
        "artifacts": artifacts,
        "runtime": {
            "stage_seconds": {
                "input_preparation": refinement_started - pipeline_started,
                "minimization_refinement": refinement_seconds,
                "common_rmsd_evaluation_excluded": rmsd_evaluation_seconds,
                "pose_serialization": serialization_seconds,
            },
            "minimization_refinement_seconds_per_pose": (
                refinement_seconds / EXPECTED_POSES
            ),
            "attempted_refinement_steps": sum(
                int(row["terminal_step"]) for row in pose_summaries
            ),
            "wall_seconds_before_summary_write": time.perf_counter() - pipeline_started,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
        },
    }
    summary_path = attempt / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    os.rename(attempt, output_dir)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "output_dir": str(output_dir),
                "poses": EXPECTED_POSES,
                "failed": summary["counts"]["failed"],
                "initial_mean_rmsd": sum(
                    row["initial_symmetry_rmsd_angstrom"] for row in pose_summaries
                )
                / EXPECTED_POSES,
                "final_mean_rmsd": sum(
                    row["final_symmetry_rmsd_angstrom"] for row in pose_summaries
                )
                / EXPECTED_POSES,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
