#!/usr/bin/env python3
"""Label-free calibration of the S50 rigid-fragment refinement budget."""

from __future__ import annotations

import argparse
import json
import math
import tempfile
import time
from collections import Counter
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from effdock.guidance import (
    InteractionEnergyConfig,
    PhysicalEnergyConfig,
    build_physical_system,
)
from effdock.guidance.parameterization import guidance_parameter_identity
from effdock.guidance.provenance import guidance_implementation_identity
from effdock.workflows.benchmark_inputs import load_benchmark_ligand
from effdock.workflows.relax_guidance import (
    RigidRelaxationConfig,
    _relaxation_energy,
    relax_rigid_fragments_batch,
)
from scripts.refine_s50_confidence_pose_bank import (
    EXPECTED_SAMPLES,
    PROTEIN_SHELL,
    _atomic_write,
    _canonical_bytes,
    _load_source_contract,
    _protein_pdb_text,
    file_sha256,
)

PROTOCOL_ID = "EFFDOCK-S50-REFINEMENT-BUDGET-CALIBRATION-V4"
ACCEPTED_STATUSES = {
    "max_steps",
    "converged_displacement",
    "converged_energy_plateau",
    "line_search_failed",
}


def _config(
    max_steps: int, *, adaptive: bool, energy_stop: bool | None = None
) -> RigidRelaxationConfig:
    if energy_stop is None:
        energy_stop = adaptive
    return RigidRelaxationConfig(
        initialization_mode="model_prior",
        prior_sigma_angstrom=2.0,
        max_steps=max_steps,
        save_every=max_steps,
        base_step_size=1.0,
        max_translation_step_angstrom=0.10,
        max_rotation_step_degrees=5.0,
        max_atom_step_angstrom=0.10,
        max_backtracks=12,
        convergence_displacement_angstrom=0.01 if adaptive else 1e-5,
        convergence_patience=5 if adaptive else 20,
        convergence_energy_absolute_kcal_mol=0.02 if energy_stop else None,
        convergence_energy_relative=0.001 if energy_stop else None,
        convergence_energy_patience=5,
        convergence_energy_min_steps=25,
        physical_cutoff_angstrom=8.0,
        protein_shell_cutoff_angstrom=PROTEIN_SHELL,
    )


def _quantile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("quantile requires at least one value")
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * probability)]


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "mean": math.fsum(values) / len(values),
        "median": _quantile(values, 0.5),
        "p95": _quantile(values, 0.95),
        "max": max(values),
    }


def _build_inputs(
    raw_record: dict[str, Any], frozen_record: dict[str, Any], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, Any]:
    raw_path = Path(str(raw_record["pt_path"])).resolve(strict=True)
    if file_sha256(raw_path) != raw_record["pt_sha256"]:
        raise RuntimeError(f"changed raw payload: {raw_path}")
    raw = torch.load(raw_path, map_location="cpu", weights_only=False)
    sample_id = str(raw_record["sample_key"])
    if raw.get("pid") != sample_id or raw.get("num_samples") != EXPECTED_SAMPLES:
        raise RuntimeError(f"{sample_id}: raw payload identity/count mismatch")
    initial = torch.as_tensor(raw["pose_atom_coords"], dtype=torch.float32)
    crystal = torch.as_tensor(raw["lig_atom_coords_crystal_centered"], dtype=torch.float32)
    fragment_id = torch.as_tensor(raw["fragment_id"], dtype=torch.long)
    pocket_center = torch.as_tensor(raw["pocket_center_used"], dtype=torch.float32)
    if initial.shape[0] != EXPECTED_SAMPLES or initial.shape[1:] != crystal.shape:
        raise RuntimeError(f"{sample_id}: malformed pose tensors")
    protein_path = Path(str(frozen_record["processed_protein"]["path"])).resolve(strict=True)
    if file_sha256(protein_path) != frozen_record["processed_protein"]["sha256"]:
        raise RuntimeError(f"{sample_id}: processed protein changed")
    protein = torch.load(protein_path, map_location="cpu", weights_only=False)
    molecule, _ = load_benchmark_ligand(str(frozen_record["canonical_smiles"]), random_seed=0)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False) as handle:
        handle.write(_protein_pdb_text(protein))
        receptor_path = Path(handle.name)
    try:
        system = build_physical_system(
            molecule,
            receptor_path,
            fragment_id=fragment_id,
            near_coords=pocket_center.view(1, 3),
            protein_cutoff=PROTEIN_SHELL,
            coordinate_origin=pocket_center,
            receptor_policy="geometry_only",
        ).to(device=device, dtype=torch.float32)
    finally:
        receptor_path.unlink(missing_ok=True)
    return initial, crystal, system


def _run_arm(
    initial: torch.Tensor,
    crystal: torch.Tensor,
    system: Any,
    config: RigidRelaxationConfig,
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    interaction = InteractionEnergyConfig()
    final_parts: list[torch.Tensor] = []
    energies: list[float] = []
    statuses: list[str] = []
    terminal_steps: list[int] = []
    backtracks: list[int] = []
    zero_center = torch.zeros(3, dtype=torch.float32, device=device)
    crystal_device = crystal.to(device)
    started = time.monotonic()
    for start in range(0, EXPECTED_SAMPLES, batch_size):
        stop = start + batch_size
        run = relax_rigid_fragments_batch(
            crystal_device,
            initial[start:stop].to(device),
            system,
            config=config,
            mode="unified",
            pocket_center=zero_center,
            interaction_config=interaction,
            collect_every_step_metrics=False,
            collect_contact_stats=False,
        )
        final = run.frames[-1].detach().to(device)
        with torch.no_grad():
            energy = _relaxation_energy(
                final,
                system,
                mode="unified",
                physical_config=PhysicalEnergyConfig(cutoff=config.physical_cutoff_angstrom),
                interaction_config=interaction,
            )["total"]
        final_parts.append(final.cpu().to(torch.float32))
        energies.extend(float(value) for value in energy.detach().cpu().tolist())
        statuses.extend(run.statuses)
        terminal_steps.extend(int(value) for value in run.terminal_steps)
        backtracks.extend(int(value) for value in run.total_backtracks)
    elapsed = time.monotonic() - started
    coords = torch.cat(final_parts, dim=0)
    if not bool(torch.isfinite(coords).all()) or not all(math.isfinite(x) for x in energies):
        raise RuntimeError("calibration arm produced non-finite coordinates or energies")
    if any(status not in ACCEPTED_STATUSES for status in statuses):
        raise RuntimeError(f"calibration arm produced unusable status: {statuses}")
    return {
        "config": asdict(config),
        "batch_size": batch_size,
        "elapsed_seconds": elapsed,
        "coords": coords,
        "energies": energies,
        "statuses": statuses,
        "terminal_steps": terminal_steps,
        "total_backtracks": backtracks,
    }


def _public_arm(arm: dict[str, Any]) -> dict[str, Any]:
    return {
        "config": arm["config"],
        "batch_size": arm["batch_size"],
        "elapsed_seconds": arm["elapsed_seconds"],
        "status_counts": dict(sorted(Counter(arm["statuses"]).items())),
        "terminal_steps": _summary([float(x) for x in arm["terminal_steps"]]),
        "total_backtracks": _summary([float(x) for x in arm["total_backtracks"]]),
        "final_energy_kcal_mol": _summary(arm["energies"]),
    }


def _compare(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    candidate_coords = candidate["coords"]
    baseline_coords = baseline["coords"]
    if torch.is_tensor(candidate_coords):
        coordinate_pairs = [(candidate_coords, baseline_coords)]
    else:
        coordinate_pairs = list(zip(candidate_coords, baseline_coords, strict=True))
    coordinate_delta = torch.cat(
        [
            (candidate_value - baseline_value).square().sum(dim=-1).mean(dim=-1).sqrt()
            for candidate_value, baseline_value in coordinate_pairs
        ]
    )
    energy_increase = [
        candidate_value - baseline_value
        for candidate_value, baseline_value in zip(
            candidate["energies"], baseline["energies"], strict=True
        )
    ]
    coord_values = [float(value) for value in coordinate_delta.tolist()]
    status_counts = Counter(candidate["statuses"])
    baseline_status_counts = Counter(baseline["statuses"])
    gates = {
        "no_new_line_search_failures": status_counts["line_search_failed"]
        <= baseline_status_counts["line_search_failed"],
        "runtime_ratio_le_0p85": candidate["elapsed_seconds"] / baseline["elapsed_seconds"] <= 0.85,
        "median_energy_increase_le_0p5": _quantile(energy_increase, 0.5) <= 0.5,
        "p95_energy_increase_le_5": _quantile(energy_increase, 0.95) <= 5.0,
        "median_coordinate_delta_le_0p1": _quantile(coord_values, 0.5) <= 0.1,
        "p95_coordinate_delta_le_0p5": _quantile(coord_values, 0.95) <= 0.5,
    }
    return {
        "runtime_ratio": candidate["elapsed_seconds"] / baseline["elapsed_seconds"],
        "energy_increase_kcal_mol": _summary(energy_increase),
        "same_index_coordinate_rmsd_angstrom": _summary(coord_values),
        "gates": gates,
        "passed": all(gates.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-bank-manifest", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-id", action="append", required=True)
    parser.add_argument(
        "--arm",
        action="append",
        choices=(
            "baseline_100",
            "batch20_100",
            "displacement_100_b20",
            "adaptive_100",
            "displacement_100",
            "adaptive_90",
            "adaptive_75",
            "adaptive_50",
        ),
    )
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if args.batch_size <= 0 or EXPECTED_SAMPLES % args.batch_size:
        raise ValueError("batch size must be a positive divisor of 100")
    device = torch.device(args.device)
    bank, frozen = _load_source_contract(args.raw_bank_manifest, args.input_manifest)
    records = {str(row["sample_key"]): row for row in bank["records"]}
    requested = list(dict.fromkeys(args.sample_id))
    if any(sample_id not in records for sample_id in requested):
        raise KeyError("calibration sample is absent from the sealed bank")
    available_arms = {
        "baseline_100": _config(100, adaptive=False),
        "batch20_100": _config(100, adaptive=False),
        "displacement_100_b20": _config(100, adaptive=True, energy_stop=False),
        "adaptive_100": _config(100, adaptive=True),
        "displacement_100": _config(100, adaptive=True, energy_stop=False),
        "adaptive_90": _config(90, adaptive=True),
        "adaptive_75": _config(75, adaptive=True),
        "adaptive_50": _config(50, adaptive=True),
    }
    requested_arms = list(dict.fromkeys(args.arm or available_arms))
    if "baseline_100" not in requested_arms or len(requested_arms) < 2:
        raise ValueError("calibration requires baseline_100 and at least one candidate")
    arms = {name: available_arms[name] for name in requested_arms}
    arm_batch_sizes = {
        name: 20 if name in {"batch20_100", "displacement_100_b20"} else args.batch_size
        for name in arms
    }
    candidate_names = [name for name in arms if name != "baseline_100"]
    per_sample: list[dict[str, Any]] = []
    aggregate: dict[str, list[dict[str, Any]]] = {name: [] for name in arms}
    for sample_id in requested:
        initial, crystal, system = _build_inputs(records[sample_id], frozen[sample_id], device)
        runs: dict[str, dict[str, Any]] = {}
        for name, config in arms.items():
            runs[name] = _run_arm(
                initial,
                crystal,
                system,
                config,
                device=device,
                batch_size=arm_batch_sizes[name],
            )
            aggregate[name].append(runs[name])
            print(f"[{sample_id}] {name} complete", flush=True)
        per_sample.append(
            {
                "sample_key": sample_id,
                "arms": {name: _public_arm(run) for name, run in runs.items()},
                "comparisons": {
                    name: _compare(runs[name], runs["baseline_100"]) for name in candidate_names
                },
            }
        )
    combined: dict[str, dict[str, Any]] = {}
    for name, runs in aggregate.items():
        combined[name] = _public_arm(
            {
                "config": runs[0]["config"],
                "batch_size": runs[0]["batch_size"],
                "elapsed_seconds": sum(run["elapsed_seconds"] for run in runs),
                "coords": [run["coords"] for run in runs],
                "energies": [value for run in runs for value in run["energies"]],
                "statuses": [value for run in runs for value in run["statuses"]],
                "terminal_steps": [value for run in runs for value in run["terminal_steps"]],
                "total_backtracks": [value for run in runs for value in run["total_backtracks"]],
            }
        )
    comparisons: dict[str, Any] = {}
    baseline_runs = aggregate["baseline_100"]
    for name in candidate_names:
        comparisons[name] = _compare(
            {
                "elapsed_seconds": sum(run["elapsed_seconds"] for run in aggregate[name]),
                "coords": [run["coords"] for run in aggregate[name]],
                "energies": [value for run in aggregate[name] for value in run["energies"]],
                "statuses": [value for run in aggregate[name] for value in run["statuses"]],
            },
            {
                "elapsed_seconds": sum(run["elapsed_seconds"] for run in baseline_runs),
                "coords": [run["coords"] for run in baseline_runs],
                "energies": [value for run in baseline_runs for value in run["energies"]],
                "statuses": [value for run in baseline_runs for value in run["statuses"]],
            },
        )
    passing = [name for name in candidate_names if comparisons[name]["passed"]]
    selected = (
        min(passing, key=lambda name: comparisons[name]["runtime_ratio"])
        if passing
        else "baseline_100"
    )
    report = {
        "schema_version": "effdock.s50_refinement_budget_calibration.v4",
        "protocol_id": PROTOCOL_ID,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "complete",
        "claim_boundary": "label-free physical/numerical calibration; no RMSD, confidence, or success labels",
        "inputs": {
            "raw_bank_manifest": str(args.raw_bank_manifest.resolve()),
            "raw_bank_manifest_sha256": file_sha256(args.raw_bank_manifest),
            "input_manifest": str(args.input_manifest.resolve()),
            "input_manifest_sha256": file_sha256(args.input_manifest),
            "sample_ids": requested,
        },
        "implementation": {
            "script_sha256": file_sha256(Path(__file__)),
            "guidance": guidance_implementation_identity(),
            "parameters": guidance_parameter_identity(),
            "torch": torch.__version__,
        },
        "arms": combined,
        "comparisons": comparisons,
        "selection_rule": "shortest candidate passing every frozen numerical/physical gate",
        "selected_arm": selected,
        "per_sample": per_sample,
    }
    _atomic_write(args.output, _canonical_bytes(report))
    _atomic_write(
        args.output.with_suffix(args.output.suffix + ".sha256"),
        f"{file_sha256(args.output)}  {args.output.resolve()}\n".encode(),
    )
    print(json.dumps({"selected_arm": selected, "comparisons": comparisons}, indent=2))


if __name__ == "__main__":
    main()
