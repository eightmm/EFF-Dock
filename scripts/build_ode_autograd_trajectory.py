#!/usr/bin/env python3
"""Append self-contained rigid-fragment autograd refinement to an ODE trajectory."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from rdkit import Chem

from effdock.evaluation.benchmark import match_atoms
from effdock.guidance import InteractionEnergyConfig, build_physical_system
from effdock.inference.preprocess import load_ligand, preprocess_complex
from effdock.workflows.relax_guidance import (
    RigidRelaxationConfig,
    relax_rigid_fragments,
)


def _load_reference(path: Path) -> Chem.Mol:
    supplier = Chem.SDMolSupplier(str(path), removeHs=False, sanitize=True)
    molecules = [molecule for molecule in supplier if molecule is not None]
    if len(molecules) != 1:
        raise ValueError(f"expected one readable reference ligand in {path}, got {len(molecules)}")
    return molecules[0]


def _aligned_reference(mol_ref: Chem.Mol, mol_input: Chem.Mol) -> tuple[torch.Tensor, str]:
    dock_indices, ref_indices, method = match_atoms(mol_ref, mol_input)
    atom_count = mol_input.GetNumAtoms()
    if len(dock_indices) != atom_count or sorted(dock_indices) != list(range(atom_count)):
        raise ValueError("autograd movie requires a complete reference/input atom mapping")
    reference = torch.tensor(mol_ref.GetConformer().GetPositions(), dtype=torch.float32)
    aligned = torch.empty((atom_count, 3), dtype=torch.float32)
    for dock_index, ref_index in zip(dock_indices, ref_indices, strict=True):
        aligned[dock_index] = reference[ref_index]
    return aligned, method


def _validate_bundle_topology(bundle: dict[str, Any], mol_input: Chem.Mol) -> None:
    observed_atomic_numbers = torch.tensor(
        [atom.GetAtomicNum() for atom in mol_input.GetAtoms()], dtype=torch.long
    )
    expected_atomic_numbers = torch.as_tensor(bundle["atomic_numbers"], dtype=torch.long)
    if not torch.equal(observed_atomic_numbers, expected_atomic_numbers):
        raise ValueError("reconstructed ligand atom order does not match the ODE trajectory")

    observed_bonds = sorted(
        (
            min(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()),
            max(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()),
            float(bond.GetBondTypeAsDouble()),
        )
        for bond in mol_input.GetBonds()
    )
    expected_bonds = sorted(
        (min(int(begin), int(end)), max(int(begin), int(end)), float(order))
        for begin, end, order in bundle["bonds"]
    )
    if observed_bonds != expected_bonds:
        raise ValueError("reconstructed ligand bonds do not match the ODE trajectory")


def _frame_labels(
    ode_frames: int,
    ode_steps: int,
    refinement_steps: list[int],
) -> tuple[list[str], list[int]]:
    labels = [
        f"ODE sampling  |  step {round(index * ode_steps / max(ode_frames - 1, 1)):02d}/{ode_steps:02d}"
        for index in range(ode_frames)
    ]
    durations = [110] * ode_frames
    durations[0] = 350
    durations[-1] = 650

    # A held endpoint makes the change of optimizer explicit without moving the camera.
    labels.append("ODE endpoint  ->  Autograd refinement")
    durations.append(850)
    for step in refinement_steps:
        labels.append(f"Autograd refinement  |  step {step:03d}/{refinement_steps[-1]:03d}")
        durations.append(130)
    durations[-1] = 1300
    return labels, durations


def build_combined_trajectory(
    input_path: Path,
    output_path: Path,
    *,
    device_name: str,
    steps: int,
    save_every: int,
    protein_shell: float,
) -> dict[str, Any]:
    bundle = torch.load(input_path, map_location="cpu", weights_only=False)
    if not isinstance(bundle, dict) or not bundle.get("traj"):
        raise ValueError("input must be a saved EFF-Dock ODE trajectory bundle")

    protein_path = Path(str(bundle["protein"])).resolve()
    ligand_ref_path = Path(str(bundle["ligand_ref"])).resolve()
    mol_ref = _load_reference(ligand_ref_path)
    smiles = Chem.MolToSmiles(mol_ref, isomericSmiles=True)
    mol_input, _ = load_ligand(smiles, random_seed=int(bundle["seed"]))
    _validate_bundle_topology(bundle, mol_input)

    pocket_center = torch.as_tensor(bundle["pocket_center"], dtype=torch.float32)
    fragment_id = torch.as_tensor(bundle["fragment_id"], dtype=torch.long)
    _, ligand_data, metadata = preprocess_complex(
        protein_path,
        mol_input,
        pocket_center=pocket_center,
        pocket_cutoff=float(bundle["pocket_cutoff"]),
    )
    if not torch.equal(ligand_data["fragment_id"].cpu(), fragment_id):
        raise ValueError("reconstructed fragment decomposition does not match the ODE trajectory")
    if not torch.equal(metadata["pocket_center"].cpu(), pocket_center):
        raise ValueError("preprocessing changed the frozen pocket center")

    system = build_physical_system(
        mol_input,
        protein_path,
        fragment_id=fragment_id,
        near_coords=pocket_center.view(1, 3),
        protein_cutoff=protein_shell,
        coordinate_origin=pocket_center,
        receptor_policy="geometry_only",
    )
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    system = system.to(device=device, dtype=torch.float32)

    reference_absolute, mapping_method = _aligned_reference(mol_ref, mol_input)
    initial_absolute = torch.as_tensor(bundle["traj"][-1], dtype=torch.float32)
    if initial_absolute.shape != reference_absolute.shape:
        raise ValueError("ODE endpoint shape does not match the reference ligand")
    initial = (initial_absolute - pocket_center).to(device)
    reference = (reference_absolute - pocket_center).to(device)

    config = RigidRelaxationConfig(
        initialization_mode="model_prior",
        max_steps=steps,
        save_every=save_every,
        base_step_size=1.0,
        max_translation_step_angstrom=0.10,
        max_rotation_step_degrees=5.0,
        max_atom_step_angstrom=0.10,
        max_backtracks=12,
        convergence_displacement_angstrom=1e-5,
        convergence_patience=20,
        physical_cutoff_angstrom=8.0,
        protein_shell_cutoff_angstrom=protein_shell,
    )
    run = relax_rigid_fragments(
        reference,
        initial,
        system,
        config=config,
        mode="unified",
        pocket_center=torch.zeros(3, device=device, dtype=torch.float32),
        interaction_config=InteractionEnergyConfig(),
    )
    if not run.frames or run.saved_steps[0] != 0:
        raise RuntimeError("autograd refinement did not emit its initial frame")

    ode_frames = [torch.as_tensor(frame, dtype=torch.float32).cpu() for frame in bundle["traj"]]
    refinement_frames = [
        frame.to(dtype=torch.float32).cpu() + pocket_center for frame in run.frames[1:]
    ]
    refinement_steps = [int(step) for step in run.saved_steps[1:]]
    if not refinement_frames:
        raise RuntimeError("autograd refinement terminated without a post-update frame")

    # Duplicate the ODE endpoint once as a visual phase boundary, then append refinement.
    combined_frames = [*ode_frames, ode_frames[-1].clone(), *refinement_frames]
    frame_labels, frame_durations_ms = _frame_labels(
        len(ode_frames), int(bundle["num_steps"]), refinement_steps
    )
    if len(combined_frames) != len(frame_labels):
        raise AssertionError("combined frame and label counts differ")

    first_metrics = run.metrics[0]
    final_metrics = run.metrics[-1]
    refinement_summary = {
        "method": "negative Torch autograd gradient projected to rigid-fragment SE(3)",
        "energy": "unified self-contained physical + protein-ligand interaction guidance",
        "status": run.status,
        "device": str(device),
        "saved_steps": run.saved_steps,
        "total_backtracks": run.total_backtracks,
        "shell_envelope_valid": run.shell_envelope_valid,
        "reference_atom_mapping": mapping_method,
        "config": asdict(config),
        "initial": {
            "raw_rmsd_angstrom": first_metrics["raw_rmsd_angstrom"],
            "minimum_protein_ligand_distance_angstrom": first_metrics[
                "minimum_protein_ligand_distance_angstrom"
            ],
            "energy_groups": first_metrics["energy_groups"],
        },
        "final": {
            "raw_rmsd_angstrom": final_metrics["raw_rmsd_angstrom"],
            "minimum_protein_ligand_distance_angstrom": final_metrics[
                "minimum_protein_ligand_distance_angstrom"
            ],
            "energy_groups": final_metrics["energy_groups"],
        },
    }

    combined = dict(bundle)
    combined.update(
        {
            "schema_version": 2,
            "source_ode_trajectory": str(input_path.resolve()),
            "traj": combined_frames,
            "traj_times": [*bundle.get("traj_times", []), None, *([None] * len(refinement_frames))],
            "frame_labels": frame_labels,
            "frame_durations_ms": frame_durations_ms,
            "autograd_refinement": refinement_summary,
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    torch.save(combined, output_path)

    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(refinement_summary, indent=2, sort_keys=True) + "\n")
    return {
        "output_bundle": str(output_path.resolve()),
        "output_summary": str(summary_path.resolve()),
        "frames": len(combined_frames),
        "ode_frames": len(ode_frames),
        "autograd_frames": len(refinement_frames),
        **refinement_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Saved ODE trajectory PT bundle.")
    parser.add_argument("output", type=Path, help="Combined ODE + autograd PT bundle.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--protein-shell", type=float, default=18.0)
    args = parser.parse_args()
    if args.steps <= 0 or args.save_every <= 0 or args.steps % args.save_every:
        parser.error("--steps must be positive and divisible by --save-every")
    print(
        json.dumps(
            build_combined_trajectory(
                args.input,
                args.output,
                device_name=args.device,
                steps=args.steps,
                save_every=args.save_every,
                protein_shell=args.protein_shell,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
