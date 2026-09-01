"""Run and aggregate the frozen batched interaction-prior diagnostic."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from effdock.geometry import flow_matching as flow_matching_module
from effdock.geometry import se3 as se3_module
from effdock.guidance import (
    GuidanceEnergyConfig,
    InteractionEnergyConfig,
    PhysicalEnergyConfig,
    build_physical_system,
    guidance_energy,
    interaction_profile_metadata,
)
from effdock.guidance.parameterization import guidance_parameter_identity
from effdock.preprocess.fragments import decompose_fragments
from effdock.preprocess.ligand import featurize_ligand
from effdock.workflows import relax_guidance as relaxation
from effdock.workflows.trace_physical import (
    _file_sha256,
    _implementation_identity,
    _load_trace_ligand,
)

SCHEMA_VERSION = "effdock.interaction_prior_probe.v2"
PROTOCOL_ID = "EFFDOCK-INTERACTION-PRIOR-PROBE-V2"
DEFAULT_SEEDS = tuple(range(20260731, 20260739))
# Historical V2 protocol constant.  Do not bind this completed experiment to
# the evolving user-facing default interaction profile.
LEGACY_DEFAULT_TERMS = (
    "hydrophobic",
    "hydrogen_bond",
    "screened_formal_charge",
)
LEGACY_ALL_TERMS = (
    "hydrophobic",
    "hydrogen_bond",
    "screened_formal_charge",
    "pi_stacking",
    "cation_pi",
    "halogen_bond",
    "metal_coordination",
)
LOCAL_POSE_THRESHOLD_ANGSTROM = 1.0
MODEL_POSE_THRESHOLD_ANGSTROM = 2.0
PROTOCOL_PRIORS = ("local", "model")
PROTOCOL_SAMPLE_ID = "1LPZ_CMB"
PROTOCOL_ARMS = (
    "guard_only",
    "guard_default",
    "guard_pi",
    "guard_halogen",
    "guard_all",
    "interaction_all_raw",
)
PROTOCOL_NUMERICAL_CONTRACT = {
    "prior_sigma": 0.5,
    "local_translation_sigma": 0.5,
    "local_rotation_sigma_degrees": 15.0,
    "steps": 500,
    "save_every": 5,
    "base_step_size": 1.0,
    "max_translation_step": 0.10,
    "max_rotation_step_degrees": 5.0,
    "max_atom_step": 0.10,
    "max_backtracks": 12,
    "convergence_displacement": 1e-5,
    "convergence_patience": 20,
    "physical_cutoff": 8.0,
    "protein_shell_cutoff": 22.0,
}


@dataclass(frozen=True)
class ArmContract:
    mode: str
    interaction_terms: tuple[str, ...]
    guard: bool
    admission_role: str


ARM_CONTRACTS: dict[str, ArmContract] = {
    "guard_only": ArmContract(
        mode="guard_only",
        interaction_terms=(),
        guard=True,
        admission_role="paired_guard_baseline",
    ),
    "guard_default": ArmContract(
        mode="guarded_interaction",
        interaction_terms=LEGACY_DEFAULT_TERMS,
        guard=True,
        admission_role="paired_default_interaction_baseline",
    ),
    "guard_pi": ArmContract(
        mode="guarded_interaction",
        interaction_terms=(*LEGACY_DEFAULT_TERMS, "pi_stacking"),
        guard=True,
        admission_role="single_new_term_pi_comparison",
    ),
    "guard_cation_pi": ArmContract(
        mode="guarded_interaction",
        interaction_terms=(*LEGACY_DEFAULT_TERMS, "cation_pi"),
        guard=True,
        admission_role="single_new_term_cation_pi_comparison",
    ),
    "guard_halogen": ArmContract(
        mode="guarded_interaction",
        interaction_terms=(*LEGACY_DEFAULT_TERMS, "halogen_bond"),
        guard=True,
        admission_role="single_new_term_halogen_comparison",
    ),
    "guard_metal": ArmContract(
        mode="guarded_interaction",
        interaction_terms=(*LEGACY_DEFAULT_TERMS, "metal_coordination"),
        guard=True,
        admission_role="target_specific_metal_diagnostic",
    ),
    "guard_all": ArmContract(
        mode="guarded_interaction",
        interaction_terms=LEGACY_ALL_TERMS,
        guard=True,
        admission_role="exploratory_all_term_only",
    ),
    "interaction_all_raw": ArmContract(
        mode="interaction_only",
        interaction_terms=LEGACY_ALL_TERMS,
        guard=False,
        admission_role="raw_capture_basin_negative_control",
    ),
}


def _jsonable(value: Any) -> Any:
    if isinstance(value, Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n")


def _source_identity(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "sha256": _file_sha256(resolved),
    }


def _module_source_identity(module: Any) -> dict[str, str]:
    module_file = getattr(module, "__file__", None)
    if module_file is None:
        raise ValueError(f"module has no source file: {module!r}")
    return _source_identity(Path(module_file))


def _run_key(prior: str, arm: str, seed: int) -> str:
    return f"{prior}__{arm}__seed-{seed}"


def _run_label(prior: str, arm: str, seed: int) -> str:
    prior_label = "Local σ0.5 Å / 15°" if prior == "local" else "Model prior σ0.5 Å"
    return f"{prior_label} · {arm.replace('_', ' ')} · seed {seed}"


def _load_case(
    args: argparse.Namespace,
) -> tuple[
    Any,
    Tensor,
    Tensor,
    dict[str, Tensor],
    Tensor,
    dict[str, Any],
    Any,
]:
    protein = args.protein.resolve()
    ligand = args.ligand.resolve()
    if not protein.is_file():
        raise FileNotFoundError(f"protein PDB not found: {protein}")
    if not ligand.is_file():
        raise FileNotFoundError(f"ligand file not found: {ligand}")
    mol, used_mol2_fallback = _load_trace_ligand(ligand)
    ligand_data = featurize_ligand(mol)
    if ligand_data is None:
        raise ValueError("ligand featurization failed")
    fragment_data = decompose_fragments(
        mol,
        ligand_data["atom_coords"],
    )
    if fragment_data is None:
        raise ValueError("ligand fragment decomposition failed")
    fragment_id = fragment_data["fragment_id"].to(torch.long)
    crystal_coords = torch.tensor(
        mol.GetConformer().GetPositions(),
        dtype=torch.float64,
    )
    pocket_center, pocket_provenance = relaxation.load_explicit_pocket_center(
        args.pocket_centers_json,
        sample_id=args.sample_id,
        key=args.pocket_center_key,
    )
    system = build_physical_system(
        mol,
        protein,
        fragment_id=fragment_id,
        near_coords=pocket_center.to(torch.float64).unsqueeze(0),
        protein_cutoff=args.protein_shell_cutoff,
    ).to(device=torch.device("cpu"), dtype=torch.float64)
    input_metadata = {
        "protein": str(protein),
        "protein_sha256": _file_sha256(protein),
        "ligand": str(ligand),
        "ligand_sha256": _file_sha256(ligand),
        "ligand_used_mol2_fallback": bool(used_mol2_fallback),
        "pocket_center": pocket_provenance,
    }
    return (
        mol,
        crystal_coords,
        fragment_id,
        fragment_data,
        pocket_center,
        input_metadata,
        system,
    )


def _initial_batch(
    *,
    prior: str,
    crystal_coords: Tensor,
    fragment_data: dict[str, Tensor],
    fragment_id: Tensor,
    pocket_center: Tensor,
    seeds: tuple[int, ...],
    prior_sigma: float,
    local_translation_sigma: float,
    local_rotation_sigma_degrees: float,
) -> tuple[Tensor, list[dict[str, Any]]]:
    if prior == "local":
        return relaxation.make_crystal_local_fragment_batch(
            crystal_coords,
            fragment_id,
            translation_sigma_angstrom=local_translation_sigma,
            rotation_sigma_degrees=local_rotation_sigma_degrees,
            seeds=seeds,
        )
    if prior == "model":
        return relaxation.make_pocket_prior_fragment_batch(
            fragment_data["frag_local_coords"],
            fragment_id,
            pocket_center,
            sigma_angstrom=prior_sigma,
            seeds=seeds,
            rotation_mode="uniform",
        )
    raise ValueError(f"unknown prior: {prior!r}")


def _validate_protocol_arguments(args: argparse.Namespace, seeds: tuple[int, ...]) -> None:
    if args.sample_id != PROTOCOL_SAMPLE_ID:
        raise ValueError(
            f"{PROTOCOL_ID} relaxation case must be {PROTOCOL_SAMPLE_ID}, got {args.sample_id!r}"
        )
    if args.arm not in PROTOCOL_ARMS:
        raise ValueError(
            f"{PROTOCOL_ID} relaxation arm must be one of {PROTOCOL_ARMS}, got {args.arm!r}"
        )
    if seeds != DEFAULT_SEEDS:
        raise ValueError(f"{PROTOCOL_ID} requires the ordered seeds {DEFAULT_SEEDS}, got {seeds}")
    for name, expected in PROTOCOL_NUMERICAL_CONTRACT.items():
        observed = getattr(args, name)
        if observed != expected:
            raise ValueError(
                f"{PROTOCOL_ID} requires --{name.replace('_', '-')}={expected}, got {observed}"
            )


def _crystal_reference(
    crystal_coords: Tensor,
    system: Any,
    *,
    pocket_center: Tensor,
    protein_shell_cutoff: float,
    physical_cutoff: float,
) -> dict[str, Any]:
    all_interaction = InteractionEnergyConfig(
        active_terms=LEGACY_ALL_TERMS,
    )
    physical_config = PhysicalEnergyConfig(cutoff=physical_cutoff)
    with torch.no_grad():
        components = guidance_energy(
            crystal_coords,
            system,
            GuidanceEnergyConfig(
                physical=physical_config,
                interaction=all_interaction,
            ),
        )
    shell_radius = protein_shell_cutoff - max(
        physical_cutoff,
        relaxation._maximum_active_interaction_cutoff(all_interaction),
    )
    metrics = relaxation._pose_metrics(
        crystal_coords,
        crystal_coords,
        system,
        components,
        pocket_center=pocket_center,
        shell_valid_radius_angstrom=shell_radius,
    )
    metrics["contacts"] = relaxation._contact_summary(
        crystal_coords,
        system,
        all_interaction,
    )
    return metrics


def run_batch(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol_file = args.protocol_file.resolve()
    if not protocol_file.is_file():
        raise FileNotFoundError(f"protocol file not found: {protocol_file}")
    seeds = tuple(int(seed) for seed in args.seeds)
    _validate_protocol_arguments(args, seeds)
    arm = ARM_CONTRACTS[args.arm]
    (
        mol,
        crystal_coords,
        fragment_id,
        fragment_data,
        pocket_center,
        input_metadata,
        system,
    ) = _load_case(args)
    initial_coords, prior_metadata = _initial_batch(
        prior=args.prior,
        crystal_coords=crystal_coords,
        fragment_data=fragment_data,
        fragment_id=fragment_id,
        pocket_center=pocket_center,
        seeds=seeds,
        prior_sigma=args.prior_sigma,
        local_translation_sigma=args.local_translation_sigma,
        local_rotation_sigma_degrees=args.local_rotation_sigma_degrees,
    )
    selected_interaction = InteractionEnergyConfig(
        active_terms=arm.interaction_terms,
    )
    # Guard-only still uses the largest all-term cutoff for a shell contract
    # identical to every paired arm; the mode ignores interaction energy.
    relaxation_interaction = (
        InteractionEnergyConfig(active_terms=LEGACY_ALL_TERMS)
        if args.arm == "guard_only"
        else selected_interaction
    )
    config = relaxation.RigidRelaxationConfig(
        initialization_mode="model_prior",
        prior_sigma_angstrom=args.prior_sigma,
        seed=seeds[0],
        max_steps=args.steps,
        save_every=args.save_every,
        base_step_size=args.base_step_size,
        max_translation_step_angstrom=args.max_translation_step,
        max_rotation_step_degrees=args.max_rotation_step_degrees,
        max_atom_step_angstrom=args.max_atom_step,
        max_backtracks=args.max_backtracks,
        convergence_displacement_angstrom=args.convergence_displacement,
        convergence_patience=args.convergence_patience,
        physical_cutoff_angstrom=args.physical_cutoff,
        protein_shell_cutoff_angstrom=args.protein_shell_cutoff,
    )
    batch_run = relaxation.relax_rigid_fragments_batch(
        crystal_coords,
        initial_coords,
        system,
        config=config,
        mode=arm.mode,
        pocket_center=pocket_center,
        interaction_config=relaxation_interaction,
    )

    crystal_reference = _crystal_reference(
        crystal_coords,
        system,
        pocket_center=pocket_center,
        protein_shell_cutoff=args.protein_shell_cutoff,
        physical_cutoff=args.physical_cutoff,
    )
    trajectory_runs: dict[str, Any] = {}
    summary_runs: dict[str, Any] = {}
    stacked_frames = torch.stack(batch_run.frames)
    for batch_index, seed in enumerate(seeds):
        key = _run_key(args.prior, args.arm, seed)
        label = _run_label(args.prior, args.arm, seed)
        pose_run = relaxation.RelaxationRun(
            mode=arm.mode,
            status=batch_run.statuses[batch_index],
            metrics=batch_run.metrics[batch_index],
            frames=[frame[batch_index] for frame in batch_run.frames],
            saved_steps=batch_run.saved_steps,
            total_backtracks=batch_run.total_backtracks[batch_index],
            shell_envelope_valid=batch_run.shell_envelope_valid[batch_index],
        )
        assessment = relaxation._success_assessment(
            pose_run,
            crystal_minimum_distance_over_uff_x=float(
                crystal_reference["minimum_distance_over_uff_x"]
            ),
            initialization_mode="model_prior",
            pose_threshold_angstrom=(
                LOCAL_POSE_THRESHOLD_ANGSTROM
                if args.prior == "local"
                else MODEL_POSE_THRESHOLD_ANGSTROM
            ),
        )
        trajectory_runs[key] = {
            "label": label,
            "frames": stacked_frames[:, batch_index],
            "saved_steps": torch.tensor(
                batch_run.saved_steps,
                dtype=torch.long,
            ),
            "seed": seed,
            "prior": args.prior,
            "arm": args.arm,
            "mode": arm.mode,
            "terminal_step": batch_run.terminal_steps[batch_index],
            "post_terminal_padding_frames": sum(
                step > batch_run.terminal_steps[batch_index] for step in batch_run.saved_steps
            ),
        }
        summary_runs[key] = {
            "label": label,
            "seed": seed,
            "prior": args.prior,
            "arm": args.arm,
            "mode": arm.mode,
            "status": batch_run.statuses[batch_index],
            "total_backtracks": batch_run.total_backtracks[batch_index],
            "terminal_step": batch_run.terminal_steps[batch_index],
            "post_terminal_padding_metrics": sum(
                int(row["step"]) > batch_run.terminal_steps[batch_index]
                for row in batch_run.metrics[batch_index]
            ),
            "shell_envelope_valid": batch_run.shell_envelope_valid[batch_index],
            "assessment": assessment,
            "metrics": batch_run.metrics[batch_index],
        }

    trajectory_path = output_dir / "trajectory.pt"
    torch.save(
        {
            "schema_version": SCHEMA_VERSION,
            "protocol_id": PROTOCOL_ID,
            "coordinate_frame": "absolute_pdb_angstrom",
            "crystal_coords": crystal_coords.cpu(),
            "initial_coords": initial_coords.cpu(),
            "pocket_center": pocket_center.cpu(),
            "protein_coords": system.protein_coords.cpu(),
            "protein_atomic_numbers": system.protein_atomic_numbers.cpu(),
            "fragment_id": fragment_id.cpu(),
            "bonds": torch.tensor(
                relaxation._bond_pairs(mol),
                dtype=torch.long,
            ),
            "cut_bonds": system.topology.bond_index.T.cpu(),
            "initialization": {
                "mode": args.prior,
                "seeds": seeds,
                "metadata": prior_metadata,
            },
            "runs": trajectory_runs,
        },
        trajectory_path,
    )
    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "diagnostic_only",
        "sample_id": args.sample_id,
        "claim_boundary": (
            "Crystal-informed rigid-fragment interaction-basin diagnostic; "
            "learned ODE absent; external benchmark report-only."
        ),
        "protocol_file": {
            "path": str(protocol_file),
            "sha256": _file_sha256(protocol_file),
        },
        "inputs": input_metadata,
        "implementation": {
            "active_guidance": _implementation_identity(),
            "probe_workflow": _source_identity(Path(__file__)),
            "relaxation_workflow": _module_source_identity(relaxation),
            "flow_matching": _module_source_identity(flow_matching_module),
            "se3": _module_source_identity(se3_module),
            "torch": torch.__version__,
        },
        "parameter_set": guidance_parameter_identity(),
        "batch": {
            "batch_size": len(seeds),
            "seeds": seeds,
            "independent_pose_line_search": True,
            "gradient_reduction": "sum_of_independent_per_pose_energies",
            "common_horizon_padding": (
                "Stopped poses retain stationary coordinates until the longest "
                "pose finishes; terminal_step marks the scientific endpoint."
            ),
        },
        "objective": {
            "arm": args.arm,
            **asdict(arm),
            "interaction_profile": interaction_profile_metadata(
                selected_interaction,
            ),
        },
        "config": asdict(config),
        "initialization": {
            "mode": args.prior,
            "kind": (
                "crystal-informed local rigid-fragment perturbation"
                if args.prior == "local"
                else "exact active-sampler t=0 translation and Uniform(SO3) prior"
            ),
            "seeds": seeds,
            "prior_sigma_angstrom": args.prior_sigma,
            "local_translation_sigma_angstrom": args.local_translation_sigma,
            "local_rotation_sigma_degrees": args.local_rotation_sigma_degrees,
            "metadata": prior_metadata,
        },
        "shell_contract": {
            "protein_shell_cutoff_angstrom": args.protein_shell_cutoff,
            "protein_shell_heavy_atoms": int(system.protein_coords.shape[0]),
            "fixed_pocket_center": input_metadata["pocket_center"],
            "never_rebuilt_during_relaxation": True,
        },
        "crystal_reference": crystal_reference,
        "runs": summary_runs,
        "warnings": [
            "Energy descent alone is not pose recovery.",
            "Raw interaction-only descent lacks ligand-connectivity and clash guards.",
            "All-term arms are exploratory and cannot admit a new term.",
            "Crystal RMSD is evaluation-only and never enters optimization.",
            "Vina and external force-field engines are absent.",
        ],
        "artifacts": {
            "trajectory_pt": {
                "path": str(trajectory_path),
                "sha256": _file_sha256(trajectory_path),
            },
        },
    }
    summary_path = output_dir / "summary.json"
    _write_json(summary_path, summary)
    print(
        json.dumps(
            {
                "summary": str(summary_path),
                "batch_size": len(seeds),
                "prior": args.prior,
                "arm": args.arm,
                "statuses": batch_run.statuses,
            },
            indent=2,
        )
    )
    return summary


def _resolve_summary(path: Path) -> Path:
    path = path.resolve()
    return path / "summary.json" if path.is_dir() else path


def _validate_static_bundle(reference: dict[str, Any], candidate: dict[str, Any]) -> None:
    for name in (
        "crystal_coords",
        "pocket_center",
        "protein_coords",
        "protein_atomic_numbers",
        "fragment_id",
        "bonds",
        "cut_bonds",
    ):
        left = torch.as_tensor(reference[name])
        right = torch.as_tensor(candidate[name])
        if left.shape != right.shape or not torch.equal(left, right):
            raise ValueError(f"aggregate bundle mismatch for {name}")


def _validate_batch_contents(summary: dict[str, Any], bundle: dict[str, Any]) -> None:
    prior = str(summary["initialization"]["mode"])
    arm = str(summary["objective"]["arm"])
    summary_runs = summary["runs"]
    bundle_runs = bundle["runs"]
    if set(summary_runs) != set(bundle_runs):
        raise ValueError(f"summary/trajectory run-key mismatch for {prior}/{arm}")
    observed_seeds = sorted(int(run["seed"]) for run in summary_runs.values())
    if observed_seeds != sorted(DEFAULT_SEEDS) or len(summary_runs) != len(DEFAULT_SEEDS):
        raise ValueError(f"run seed matrix mismatch for {prior}/{arm}")
    for key, run in summary_runs.items():
        expected_key = _run_key(prior, arm, int(run["seed"]))
        if key != expected_key or run["prior"] != prior or run["arm"] != arm:
            raise ValueError(f"run identity mismatch for {key}")
        trajectory_run = bundle_runs[key]
        if (
            int(trajectory_run["seed"]) != int(run["seed"])
            or trajectory_run["prior"] != prior
            or trajectory_run["arm"] != arm
        ):
            raise ValueError(f"trajectory run identity mismatch for {key}")
    initial_coords = torch.as_tensor(bundle["initial_coords"])
    if initial_coords.ndim != 3 or initial_coords.shape[0] != len(DEFAULT_SEEDS):
        raise ValueError(f"initial pose batch shape mismatch for {prior}/{arm}")
    initialization = bundle["initialization"]
    if initialization["mode"] != prior or tuple(initialization["seeds"]) != DEFAULT_SEEDS:
        raise ValueError(f"trajectory initialization mismatch for {prior}/{arm}")


def _require_same_contract(
    reference: dict[str, Any],
    candidate: dict[str, Any],
    field: str,
) -> None:
    if reference.get(field) != candidate.get(field):
        raise ValueError(f"aggregate contract mismatch for {field}")


def _require_same_numeric_contract(
    reference: Any,
    candidate: Any,
    field: str,
    *,
    relative_tolerance: float = 1e-12,
    absolute_tolerance: float = 1e-12,
) -> None:
    if isinstance(reference, dict) and isinstance(candidate, dict):
        if set(reference) != set(candidate):
            raise ValueError(f"aggregate contract mismatch for {field} keys")
        for key in sorted(reference):
            _require_same_numeric_contract(
                reference[key],
                candidate[key],
                f"{field}.{key}",
                relative_tolerance=relative_tolerance,
                absolute_tolerance=absolute_tolerance,
            )
        return
    if isinstance(reference, list) and isinstance(candidate, list):
        if len(reference) != len(candidate):
            raise ValueError(f"aggregate contract mismatch for {field} length")
        for index, (left, right) in enumerate(zip(reference, candidate)):
            _require_same_numeric_contract(
                left,
                right,
                f"{field}[{index}]",
                relative_tolerance=relative_tolerance,
                absolute_tolerance=absolute_tolerance,
            )
        return
    if (
        isinstance(reference, (int, float))
        and not isinstance(reference, bool)
        and isinstance(candidate, (int, float))
        and not isinstance(candidate, bool)
    ):
        if not math.isclose(
            float(reference),
            float(candidate),
            rel_tol=relative_tolerance,
            abs_tol=absolute_tolerance,
        ):
            raise ValueError(
                f"aggregate numeric contract mismatch for {field}: {reference!r} != {candidate!r}"
            )
        return
    if reference != candidate:
        raise ValueError(f"aggregate contract mismatch for {field}")


def _resolve_declared_artifact(summary_path: Path, declared_path: str) -> Path:
    artifact_path = Path(declared_path)
    if not artifact_path.is_absolute():
        artifact_path = summary_path.parent / artifact_path
    return artifact_path.resolve()


def _protocol_failure(run: dict[str, Any]) -> bool:
    final = run["metrics"][-1]
    return not (
        run["status"]
        in {"max_steps", "converged_displacement", "converged_energy_plateau"}
        and bool(run["shell_envelope_valid"])
        and float(final["cut_bond_max_abs_error_angstrom"]) <= 0.20
        and float(final["minimum_distance_over_uff_x"]) >= 0.65
        and int(final.get("chiral_improper_inversion_count", 0)) == 0
    )


def _paired_comparisons(
    summaries: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    crystal_gates = _crystal_term_gates(summaries)
    indexed: dict[tuple[str, str], dict[int, dict[str, Any]]] = {}
    roles: dict[str, str] = {}
    for summary in summaries:
        prior = str(summary["initialization"]["mode"])
        arm = str(summary["objective"]["arm"])
        roles[arm] = str(summary["objective"]["admission_role"])
        indexed[(prior, arm)] = {int(run["seed"]): run for run in summary["runs"].values()}

    comparisons: dict[str, dict[str, Any]] = {}
    for (prior, arm), term_runs in sorted(indexed.items()):
        if not roles.get(arm, "").startswith("single_new_term_"):
            continue
        baseline_runs = indexed.get((prior, "guard_default"))
        if baseline_runs is None:
            raise ValueError(f"paired comparison requires guard_default for prior={prior}")
        if set(term_runs) != set(baseline_runs):
            raise ValueError(f"paired comparison seed mismatch for prior={prior}, arm={arm}")
        paired: list[dict[str, Any]] = []
        for seed in sorted(term_runs):
            term_run = term_runs[seed]
            baseline_run = baseline_runs[seed]
            term_final = float(term_run["metrics"][-1]["raw_rmsd_angstrom"])
            baseline_final = float(baseline_run["metrics"][-1]["raw_rmsd_angstrom"])
            paired.append(
                {
                    "seed": seed,
                    "baseline_final_rmsd_angstrom": baseline_final,
                    "term_final_rmsd_angstrom": term_final,
                    "delta_rmsd_angstrom": term_final - baseline_final,
                    "improved": term_final < baseline_final,
                    "baseline_protocol_failure": _protocol_failure(baseline_run),
                    "term_protocol_failure": _protocol_failure(term_run),
                }
            )
        median_delta = statistics.median(row["delta_rmsd_angstrom"] for row in paired)
        improved_count = sum(bool(row["improved"]) for row in paired)
        baseline_failures = sum(bool(row["baseline_protocol_failure"]) for row in paired)
        term_failures = sum(bool(row["term_protocol_failure"]) for row in paired)
        additional_failure_seeds = [
            int(row["seed"])
            for row in paired
            if row["term_protocol_failure"] and not row["baseline_protocol_failure"]
        ]
        crystal_gate_pass = bool(crystal_gates.get(arm, {}).get("gate_pass", False))
        comparisons[f"{prior}__{arm}_vs_guard_default"] = {
            "prior": prior,
            "term_arm": arm,
            "baseline_arm": "guard_default",
            "paired_seed_count": len(paired),
            "median_delta_final_rmsd_angstrom": median_delta,
            "improved_seed_count": improved_count,
            "baseline_protocol_failure_count": baseline_failures,
            "term_protocol_failure_count": term_failures,
            "additional_protocol_failure_count": len(additional_failure_seeds),
            "additional_protocol_failure_seeds": additional_failure_seeds,
            "crystal_target_term_gate_pass": crystal_gate_pass,
            "local_admission_signal": (
                prior == "local"
                and crystal_gate_pass
                and median_delta <= -0.25
                and improved_count >= 5
                and not additional_failure_seeds
            ),
            "paired": paired,
        }
    return comparisons


def _crystal_term_gates(
    summaries: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    target_contracts = {
        "guard_pi": (
            "interaction_pi_stacking",
            "pi_stacking_weight_sum",
        ),
        "guard_halogen": (
            "interaction_halogen_bond",
            "halogen_bond_weight_sum",
        ),
    }
    gates: dict[str, dict[str, Any]] = {}
    for summary in summaries:
        if summary["initialization"]["mode"] != "local":
            continue
        arm = str(summary["objective"]["arm"])
        if arm not in target_contracts:
            continue
        energy_name, contact_name = target_contracts[arm]
        crystal = summary["crystal_reference"]
        crystal_energy = float(crystal["energies"][energy_name])
        crystal_contact_weight = float(crystal["contacts"][contact_name])
        initial_by_seed = {
            int(run["seed"]): float(run["metrics"][0]["energies"][energy_name])
            for run in summary["runs"].values()
        }
        lower_than_initial_count = sum(
            crystal_energy < energy for energy in initial_by_seed.values()
        )
        gates[arm] = {
            "target_energy": energy_name,
            "target_contact": contact_name,
            "crystal_energy": crystal_energy,
            "crystal_contact_weight": crystal_contact_weight,
            "local_initial_energy_by_seed": initial_by_seed,
            "crystal_below_local_initial_count": lower_than_initial_count,
            "required_count": 6,
            "contact_nonzero": crystal_contact_weight > 0.0,
            "gate_pass": (crystal_contact_weight > 0.0 and lower_than_initial_count >= 6),
        }
    return gates


def aggregate_batches(args: argparse.Namespace) -> dict[str, Any]:
    summary_paths = [_resolve_summary(path) for path in args.inputs]
    if len(summary_paths) < 2:
        raise ValueError("aggregate requires at least two batch summaries")
    summaries: list[dict[str, Any]] = []
    bundles: list[dict[str, Any]] = []
    artifact_paths: list[Path] = []
    for summary_path in summary_paths:
        summary = json.loads(summary_path.read_text())
        if summary.get("protocol_id") != PROTOCOL_ID:
            raise ValueError(f"unexpected protocol in {summary_path}")
        if summary.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"unexpected schema in {summary_path}")
        trajectory_artifact = summary["artifacts"]["trajectory_pt"]
        trajectory_path = _resolve_declared_artifact(
            summary_path,
            trajectory_artifact["path"],
        )
        if not trajectory_path.is_file():
            raise FileNotFoundError(f"trajectory artifact not found: {trajectory_path}")
        observed_sha256 = _file_sha256(trajectory_path)
        if observed_sha256 != trajectory_artifact["sha256"]:
            raise ValueError(f"trajectory artifact hash mismatch for {summary_path}")
        bundle = torch.load(
            trajectory_path,
            map_location="cpu",
            weights_only=False,
        )
        if bundle.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"unexpected trajectory schema in {trajectory_path}")
        if bundle.get("protocol_id") != PROTOCOL_ID:
            raise ValueError(f"unexpected trajectory protocol in {trajectory_path}")
        _validate_batch_contents(summary, bundle)
        summaries.append(summary)
        bundles.append(bundle)
        artifact_paths.append(trajectory_path)
    first_summary = summaries[0]
    first_bundle = bundles[0]
    observed_matrix = {
        (
            str(summary["initialization"]["mode"]),
            str(summary["objective"]["arm"]),
        )
        for summary in summaries
    }
    expected_matrix = {(prior, arm) for prior in PROTOCOL_PRIORS for arm in PROTOCOL_ARMS}
    if observed_matrix != expected_matrix or len(summaries) != len(expected_matrix):
        missing = sorted(expected_matrix - observed_matrix)
        unexpected = sorted(observed_matrix - expected_matrix)
        raise ValueError(
            "aggregate protocol matrix incomplete or duplicated: "
            f"missing={missing}, unexpected={unexpected}, batches={len(summaries)}"
        )
    for summary, bundle in zip(summaries[1:], bundles[1:]):
        for field in (
            "schema_version",
            "protocol_id",
            "sample_id",
            "protocol_file",
            "inputs",
            "implementation",
            "parameter_set",
            "config",
            "shell_contract",
        ):
            _require_same_contract(first_summary, summary, field)
        _require_same_numeric_contract(
            first_summary["crystal_reference"],
            summary["crystal_reference"],
            "crystal_reference",
        )
        for field in (
            "batch_size",
            "seeds",
            "independent_pose_line_search",
            "gradient_reduction",
            "common_horizon_padding",
        ):
            if summary["batch"].get(field) != first_summary["batch"].get(field):
                raise ValueError(f"aggregate batch contract mismatch for {field}")
        _validate_static_bundle(first_bundle, bundle)
    first_by_prior: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for summary, bundle in zip(summaries, bundles):
        prior = str(summary["initialization"]["mode"])
        if prior not in first_by_prior:
            first_by_prior[prior] = (summary, bundle)
            continue
        prior_summary, prior_bundle = first_by_prior[prior]
        if summary["initialization"] != prior_summary["initialization"]:
            raise ValueError(f"aggregate initialization contract mismatch for prior={prior}")
        left = torch.as_tensor(prior_bundle["initial_coords"])
        right = torch.as_tensor(bundle["initial_coords"])
        if left.shape != right.shape or not torch.equal(left, right):
            raise ValueError(f"aggregate initial pose mismatch for prior={prior}")

    combined_runs: dict[str, Any] = {}
    combined_summary_runs: dict[str, Any] = {}
    batch_contracts: list[dict[str, Any]] = []
    for summary_path, trajectory_path, summary, bundle in zip(
        summary_paths,
        artifact_paths,
        summaries,
        bundles,
    ):
        overlap = set(combined_runs) & set(bundle["runs"])
        if overlap:
            raise ValueError(f"duplicate aggregate run keys: {sorted(overlap)}")
        combined_runs.update(bundle["runs"])
        combined_summary_runs.update(summary["runs"])
        batch_contracts.append(
            {
                "prior": summary["initialization"]["mode"],
                "arm": summary["objective"]["arm"],
                "seeds": summary["batch"]["seeds"],
                "summary": {
                    "path": str(summary_path),
                    "sha256": _file_sha256(summary_path),
                },
                "trajectory_pt": {
                    "path": str(trajectory_path),
                    "sha256": _file_sha256(trajectory_path),
                },
            }
        )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    combined_trajectory_path = output_dir / "trajectory.pt"
    combined_bundle = {
        **{
            key: value
            for key, value in first_bundle.items()
            if key not in {"runs", "initial_coords", "initialization"}
        },
        "initial_coords": None,
        "initialization": {
            "mode": "paired_local_and_model_prior_ensemble",
            "batch_contracts": batch_contracts,
        },
        "runs": combined_runs,
    }
    torch.save(combined_bundle, combined_trajectory_path)

    aggregate_summary = {
        **{
            key: value
            for key, value in first_summary.items()
            if key
            not in {
                "batch",
                "config",
                "initialization",
                "objective",
                "runs",
                "artifacts",
                "created_utc",
            }
        },
        "created_utc": datetime.now(UTC).isoformat(),
        "implementation": {
            **first_summary["implementation"],
            "aggregation_workflow": _source_identity(Path(__file__)),
        },
        "batch": {
            "num_batches": len(summaries),
            "num_trajectories": len(combined_runs),
            "contracts": batch_contracts,
        },
        "initialization": {
            "mode": "paired_local_and_model_prior_ensemble",
            "kind": "paired local and exact model-prior batches",
        },
        "runs": combined_summary_runs,
        "crystal_term_gates": _crystal_term_gates(summaries),
        "paired_comparisons": _paired_comparisons(summaries),
        "artifacts": {
            "trajectory_pt": {
                "path": str(combined_trajectory_path),
                "sha256": _file_sha256(combined_trajectory_path),
            },
        },
    }
    summary_path = output_dir / "summary.json"
    _write_json(summary_path, aggregate_summary)

    rows: list[dict[str, Any]] = []
    for key, run in combined_summary_runs.items():
        initial = run["metrics"][0]
        final = run["metrics"][-1]
        rows.append(
            {
                "run": key,
                "prior": run["prior"],
                "arm": run["arm"],
                "seed": run["seed"],
                "status": run["status"],
                "shell_valid": run["shell_envelope_valid"],
                "initial_rmsd_angstrom": initial["raw_rmsd_angstrom"],
                "final_rmsd_angstrom": final["raw_rmsd_angstrom"],
                "final_cut_bond_max_error_angstrom": final["cut_bond_max_abs_error_angstrom"],
                "final_minimum_distance_over_uff_x": final["minimum_distance_over_uff_x"],
                "final_chiral_improper_inversion_count": final["chiral_improper_inversion_count"],
                "initial_total_energy": initial["energy_groups"]["combined"],
                "final_total_energy": final["energy_groups"]["combined"],
                "initial_interaction_energy": initial["energy_groups"]["interaction"],
                "final_interaction_energy": final["energy_groups"]["interaction"],
                "primary_joint_success": run["assessment"]["primary_joint_success"],
            }
        )
    csv_path = output_dir / "runs.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["prior"], row["arm"])].append(row)
    lines = [
        "# EFF-Dock interaction prior probe",
        "",
        f"Protocol: `{PROTOCOL_ID}`",
        f"Case: `{first_summary['sample_id']}`",
        "",
        "| Prior | Arm | n | Joint pass | Median initial RMSD | Median final RMSD | "
        "Median final interaction |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for (prior, arm), group in sorted(grouped.items()):
        lines.append(
            f"| {prior} | {arm} | {len(group)} | "
            f"{sum(bool(row['primary_joint_success']) for row in group)}/{len(group)} | "
            f"{statistics.median(float(row['initial_rmsd_angstrom']) for row in group):.3f} A | "
            f"{statistics.median(float(row['final_rmsd_angstrom']) for row in group):.3f} A | "
            f"{statistics.median(float(row['final_interaction_energy']) for row in group):.3f} |"
        )
    if aggregate_summary["crystal_term_gates"]:
        lines.extend(
            [
                "",
                "## Crystal target-term gates",
                "",
                "| Arm | Crystal target energy | Contact weight | Below local initials | Gate |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for arm, gate in aggregate_summary["crystal_term_gates"].items():
            lines.append(
                f"| {arm} | {gate['crystal_energy']:+.4f} | "
                f"{gate['crystal_contact_weight']:.4f} | "
                f"{gate['crystal_below_local_initial_count']}/"
                f"{len(gate['local_initial_energy_by_seed'])} | "
                f"{'pass' if gate['gate_pass'] else 'fail'} |"
            )
    if aggregate_summary["paired_comparisons"]:
        lines.extend(
            [
                "",
                "## Paired single-term comparisons",
                "",
                "| Prior | Term arm | Median Δ final RMSD | Improved seeds | "
                "Additional failures | Local admission signal |",
                "|---|---|---:|---:|---:|---:|",
            ]
        )
        for comparison in aggregate_summary["paired_comparisons"].values():
            lines.append(
                f"| {comparison['prior']} | {comparison['term_arm']} | "
                f"{comparison['median_delta_final_rmsd_angstrom']:+.3f} A | "
                f"{comparison['improved_seed_count']}/"
                f"{comparison['paired_seed_count']} | "
                f"{comparison['additional_protocol_failure_count']} | "
                f"{'yes' if comparison['local_admission_signal'] else 'no'} |"
            )
    lines.extend(
        [
            "",
            "All-term and raw-interaction rows are exploratory only. RMSD and "
            "crystal coordinates never entered optimization.",
        ]
    )
    results_path = output_dir / "RESULTS.md"
    results_path.write_text("\n".join(lines) + "\n")
    aggregate_summary["artifacts"].update(
        {
            "runs_csv": {
                "path": str(csv_path),
                "sha256": _file_sha256(csv_path),
            },
            "results_md": {
                "path": str(results_path),
                "sha256": _file_sha256(results_path),
            },
        }
    )
    _write_json(summary_path, aggregate_summary)
    print(
        json.dumps(
            {
                "summary": str(summary_path),
                "trajectory": str(combined_trajectory_path),
                "runs": len(combined_runs),
            },
            indent=2,
        )
    )
    return aggregate_summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Run one prior/arm tensor batch.")
    run.add_argument("--protein", type=Path, required=True)
    run.add_argument("--ligand", type=Path, required=True)
    run.add_argument("--pocket-centers-json", type=Path, required=True)
    run.add_argument("--pocket-center-key")
    run.add_argument("--sample-id", required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--protocol-file", type=Path, required=True)
    run.add_argument("--prior", choices=("local", "model"), required=True)
    run.add_argument("--arm", choices=tuple(ARM_CONTRACTS), required=True)
    run.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    run.add_argument("--prior-sigma", type=float, default=0.5)
    run.add_argument("--local-translation-sigma", type=float, default=0.5)
    run.add_argument("--local-rotation-sigma-degrees", type=float, default=15.0)
    run.add_argument("--steps", type=int, default=500)
    run.add_argument("--save-every", type=int, default=5)
    run.add_argument("--base-step-size", type=float, default=1.0)
    run.add_argument("--max-translation-step", type=float, default=0.10)
    run.add_argument("--max-rotation-step-degrees", type=float, default=5.0)
    run.add_argument("--max-atom-step", type=float, default=0.10)
    run.add_argument("--max-backtracks", type=int, default=12)
    run.add_argument("--convergence-displacement", type=float, default=1e-5)
    run.add_argument("--convergence-patience", type=int, default=20)
    run.add_argument("--physical-cutoff", type=float, default=8.0)
    run.add_argument("--protein-shell-cutoff", type=float, default=22.0)

    aggregate = subparsers.add_parser(
        "aggregate",
        help="Combine completed prior/arm batches.",
    )
    aggregate.add_argument("--inputs", nargs="+", type=Path, required=True)
    aggregate.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        run_batch(args)
    elif args.command == "aggregate":
        aggregate_batches(args)
    else:
        raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    main()
