#!/usr/bin/env python3
"""Re-evaluate saved refinement frames and summarize GuidanceEnergy progress."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from effdock.guidance import InteractionEnergyConfig, PhysicalEnergyConfig
from effdock.guidance.system import build_physical_system
from effdock.inference.preprocess import preprocess_complex
from effdock.workflows.benchmark_inputs import (
    load_benchmark_inputs,
    load_benchmark_ligand,
)
from effdock.workflows.relax_guidance import _relaxation_energy


def _stats(values: torch.Tensor) -> dict[str, float]:
    return {
        "mean": float(values.mean()),
        "median": float(values.median()),
        "q10": float(torch.quantile(values, 0.1)),
        "q90": float(torch.quantile(values, 0.9)),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--benchmark-input-manifest", type=Path, required=True)
    parser.add_argument("--external-dir", type=Path, default=Path("data/external_test"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    inputs = summary["inputs"]
    dataset = str(inputs["dataset"])
    complex_id = str(inputs["complex_id"])
    mapping, _ = load_benchmark_inputs(
        dataset,
        args.external_dir,
        args.benchmark_input_manifest,
    )
    mol_input, _ = load_benchmark_ligand(
        mapping[complex_id],
        random_seed=int(inputs["sampling_seed"]),
    )
    pocket_center = torch.tensor(inputs["pocket_center_absolute"], dtype=torch.float32)
    _, ligand_data, _ = preprocess_complex(
        Path(inputs["protein"]),
        mol_input,
        pocket_center=pocket_center,
        pocket_cutoff=10.0,
    )
    system = build_physical_system(
        mol_input,
        Path(inputs["protein"]),
        fragment_id=ligand_data["fragment_id"],
        near_coords=pocket_center.view(1, 3),
        protein_cutoff=18.0,
        coordinate_origin=pocket_center,
        receptor_policy="geometry_only",
    )
    device = torch.device(args.device)
    system = system.to(device=device, dtype=torch.float32)
    trajectory = torch.load(
        summary["artifacts"]["trajectory_pt"]["path"],
        map_location="cpu",
        weights_only=True,
    )
    steps = trajectory["saved_steps"].to(torch.long).tolist()
    frames = trajectory["frames_pocket_centered"].to(torch.float32)
    energy_by_step: dict[int, torch.Tensor] = {}
    physical = PhysicalEnergyConfig(cutoff=8.0)
    interaction = InteractionEnergyConfig()
    for step, frame in zip(steps, frames, strict=True):
        parts: list[torch.Tensor] = []
        for start in range(0, frame.shape[0], args.batch_size):
            coords = frame[start : start + args.batch_size].to(device)
            with torch.no_grad():
                energy = _relaxation_energy(
                    coords,
                    system,
                    mode="unified",
                    physical_config=physical,
                    interaction_config=interaction,
                )["total"]
            parts.append(energy.detach().cpu().to(torch.float64))
        energy_by_step[int(step)] = torch.cat(parts)
    intervals: dict[str, dict[str, float]] = {}
    for left, right in zip(steps[:-1], steps[1:], strict=True):
        decrease = energy_by_step[left] - energy_by_step[right]
        intervals[f"{left:03d}_to_{right:03d}"] = {
            **_stats(decrease),
            "mean_decrease_per_step": float(decrease.mean() / (right - left)),
            "median_decrease_per_step": float(decrease.median() / (right - left)),
        }
    result = {
        "dataset": dataset,
        "complex_id": complex_id,
        "poses": int(frames.shape[1]),
        "steps": steps,
        "energy_by_step": {
            str(step): _stats(energy_by_step[step]) for step in steps
        },
        "interval_decrease": intervals,
    }
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.exists():
            raise FileExistsError(args.output)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
