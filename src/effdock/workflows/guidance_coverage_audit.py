#!/usr/bin/env python3
"""Outcome-blind CPU coverage and numerical preflight for unified guidance."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

import torch

from effdock.evaluation.benchmark import load_ligand as load_reference_ligand
from effdock.evaluation.benchmark import match_atoms
from effdock.guidance import (
    GuidanceEnergyConfig,
    build_physical_system,
    guidance_energy,
)
from effdock.guidance.parameterization import guidance_parameter_identity
from effdock.guidance.provenance import (
    guidance_implementation_identity,
    physical_system_reference_sha256,
)
from effdock.inference.defaults import DEFAULT_POCKET_CUTOFF
from effdock.inference.preprocess import preprocess_complex
from effdock.workflows.benchmark_inputs import (
    BenchmarkInputMismatchError,
    full_heavy_atom_mapping_metadata,
    load_benchmark_inputs,
    load_benchmark_ligand,
)
from effdock.workflows.evaluate import (
    UNIFIED_GUIDANCE_RECEPTOR_POLICIES,
    ComplexInput,
    _canonical_json,
    discover_complexes,
    file_sha256,
    global_seed_by_id,
    load_pocket_centers,
    receptor_guidance_metadata,
    serialize_evaluation_failure,
    sorted_id_sha256,
)

AUDIT_SCHEMA_VERSION = "effdock.guidance_coverage_audit.v2"
AUDIT_PROTOCOL_ID = "EFFDOCK-UNIFIED-GUIDANCE-BUDGET1000-FULL-V2"
FULL_PROTOCOL_EXPECTED_COUNTS = {"astex": 85, "posebusters": 308}
GUIDANCE_SHELL_ANGSTROM = 18.0
ID_HASH_CONTRACT = "SHA-256 of EFFDOCK_SORTED_COMPLEX_IDS_V1\\0 plus each sorted UTF-8 ID and NUL"


def _implementation_identity() -> dict[str, object]:
    """Backward-compatible alias for the shared audit/sampling identity."""
    return guidance_implementation_identity()


def _system_reference_sha256(system) -> str:
    """Backward-compatible alias for the shared audit/sampling system hash."""
    return physical_system_reference_sha256(system)


def _crystal_coords(
    item: ComplexInput,
    mol_in,
    pocket_center: torch.Tensor,
) -> tuple[torch.Tensor, str, dict[str, object]]:
    """Build an exact full-heavy-atom reference-pose numerical probe."""
    mol_ref = load_reference_ligand(item.ligand_ref, item.ligand_format)
    dock_indices, ref_indices, method = match_atoms(mol_ref, mol_in)
    mapping_metadata = full_heavy_atom_mapping_metadata(
        mol_ref,
        mol_in,
        dock_indices,
        ref_indices,
        method,
    )
    if not mapping_metadata["accepted"]:
        raise BenchmarkInputMismatchError(
            "benchmark_ligand_atom_mapping_mismatch",
            "numerical preflight requires a complete connectivity-preserving atom map",
            details=mapping_metadata,
        )
    reference = torch.tensor(
        mol_ref.GetConformer().GetPositions(),
        dtype=torch.float64,
    )
    dock_index = torch.as_tensor(dock_indices, dtype=torch.long)
    ref_index = torch.as_tensor(ref_indices, dtype=torch.long)
    absolute = torch.empty((mol_in.GetNumAtoms(), 3), dtype=torch.float64)
    absolute[dock_index] = reference.index_select(0, ref_index)
    matched_delta = absolute.index_select(0, dock_index) - reference.index_select(
        0,
        ref_index,
    )
    alignment = {
        "coordinate_source": "exact_reference_coordinates",
        "matched_atoms": len(dock_indices),
        "input_atoms": mol_in.GetNumAtoms(),
        "reference_atoms": mol_ref.GetNumAtoms(),
        "matched_alignment_rmsd_angstrom": float(matched_delta.square().sum(dim=-1).mean().sqrt()),
        "alignment_rank": None,
        "proper_rotation_determinant": None,
        **mapping_metadata,
    }
    return (
        absolute - pocket_center.to(torch.float64).view(1, 3),
        method,
        alignment,
    )


def _numerical_preflight(
    coords: torch.Tensor,
    system,
) -> tuple[dict[str, object], dict[str, torch.Tensor]]:
    """Evaluate every default term and one autograd derivative on CPU float64."""
    work = coords.detach().cpu().to(torch.float64).clone().requires_grad_(True)
    system = system.to(device=torch.device("cpu"), dtype=torch.float64)
    components = guidance_energy(work, system, GuidanceEnergyConfig())
    energies: dict[str, float] = {}
    max_abs_gradient: dict[str, float] = {}
    gradients: dict[str, torch.Tensor] = {}
    for name, value in components.items():
        if value.numel() != 1 or not bool(torch.isfinite(value).all()):
            raise FloatingPointError(f"non-finite or nonscalar guidance term: {name}")
        gradient = (
            torch.autograd.grad(
                value,
                work,
                retain_graph=True,
                allow_unused=True,
            )[0]
            if value.requires_grad
            else None
        )
        gradient = torch.zeros_like(work) if gradient is None else gradient
        if not bool(torch.isfinite(gradient).all()):
            raise FloatingPointError(f"non-finite guidance gradient: {name}")
        energies[name] = float(value.detach().cpu())
        max_abs_gradient[name] = (
            float(gradient.detach().abs().max().cpu()) if gradient.numel() else 0.0
        )
        gradients[name] = gradient.detach().cpu()
    return (
        {
            "energy_finite": True,
            "gradient_finite": True,
            "energies": energies,
            "max_abs_gradient": max_abs_gradient,
            "total_energy": energies["total"],
            "total_max_abs_gradient": max_abs_gradient["total"],
        },
        gradients,
    )


def _fallback_summary(receptor: dict[str, object]) -> dict[str, object]:
    provenance = receptor.get("provenance", {})
    if not isinstance(provenance, dict):
        provenance = {}
    obstacle_count = int(
        provenance.get(
            "active_nonprotein_geometry_obstacle_atoms",
            provenance.get("obstacle_count", 0),
        )
        or 0
    )
    raw_fallbacks = provenance.get("metal_fallbacks", [])
    if not isinstance(raw_fallbacks, list):
        raw_fallbacks = []
    reasons = Counter(
        str(fallback.get("code", "unspecified"))
        for fallback in raw_fallbacks
        if isinstance(fallback, dict)
    )
    return {
        "obstacle_count": obstacle_count,
        "obstacle_residues": list(
            provenance.get("active_nonprotein_geometry_obstacle_residues", []) or []
        ),
        "metal_fallback_count": len(raw_fallbacks),
        "metal_fallback_reasons": dict(sorted(reasons.items())),
    }


def _equivalence_check(
    requested_system,
    strict_system,
    requested_policy: str,
    requested_check: dict[str, object],
    strict_check: dict[str, object],
    requested_gradients: dict[str, torch.Tensor],
    strict_gradients: dict[str, torch.Tensor],
) -> dict[str, object]:
    requested_hash = _system_reference_sha256(requested_system)
    strict_hash = _system_reference_sha256(strict_system)
    requested_energy = requested_check["energies"]
    strict_energy = strict_check["energies"]
    term_names_match = set(requested_energy) == set(strict_energy)
    energy_max_abs_delta = (
        max(
            abs(float(requested_energy[name]) - float(strict_energy[name]))
            for name in requested_energy
        )
        if term_names_match and requested_energy
        else float("inf")
    )
    gradient_max_abs_delta = (
        max(
            float((requested_gradients[name] - strict_gradients[name]).abs().max())
            for name in requested_gradients
        )
        if set(requested_gradients) == set(strict_gradients) and requested_gradients
        else float("inf")
    )
    receptor = receptor_guidance_metadata(requested_system, requested_policy)
    fallback = _fallback_summary(receptor)
    passed = bool(
        requested_hash == strict_hash
        and term_names_match
        and energy_max_abs_delta <= 1e-10
        and gradient_max_abs_delta <= 1e-10
        and fallback["obstacle_count"] == 0
        and fallback["metal_fallback_count"] == 0
    )
    return {
        "passed": passed,
        "requested_system_reference_sha256": requested_hash,
        "strict_system_reference_sha256": strict_hash,
        "term_names_match": term_names_match,
        "energy_max_abs_delta": energy_max_abs_delta,
        "gradient_max_abs_delta": gradient_max_abs_delta,
        "requested_obstacle_count": fallback["obstacle_count"],
        "requested_metal_fallback_count": fallback["metal_fallback_count"],
        "absolute_tolerance": 1e-10,
    }


def _audit_one(
    item: ComplexInput,
    *,
    seed: int,
    receptor_policy: str,
    pocket_cutoff: float,
) -> dict[str, object]:
    torch.manual_seed(seed)
    ligand_input = item.smiles if item.smiles else str(item.ligand_ref)
    if item.smiles and item.enforce_benchmark_heavy_atom_policy:
        mol_in, _ = load_benchmark_ligand(ligand_input, random_seed=seed)
    else:
        from effdock.inference.preprocess import load_ligand

        mol_in, _ = load_ligand(ligand_input, random_seed=seed)
    frozen_center = torch.tensor(item.pocket_center, dtype=torch.float32)
    _, lig_data, meta = preprocess_complex(
        item.protein,
        mol_in,
        pocket_center=frozen_center,
        pocket_cutoff=pocket_cutoff,
    )
    builder_kwargs = {
        "fragment_id": lig_data["fragment_id"],
        "near_coords": meta["pocket_center"].view(1, 3),
        "protein_cutoff": GUIDANCE_SHELL_ANGSTROM,
        "coordinate_origin": meta["pocket_center"],
    }
    strict_system = None
    strict_failure = None
    try:
        strict_system = build_physical_system(
            mol_in,
            item.protein,
            receptor_policy="fail_closed",
            **builder_kwargs,
        )
    except Exception as exc:  # all construction failures are retained in provenance
        strict_failure = serialize_evaluation_failure(item.complex_id, exc)

    if receptor_policy == "fail_closed" and strict_system is not None:
        requested_system = strict_system
    else:
        requested_system = build_physical_system(
            mol_in,
            item.protein,
            receptor_policy=receptor_policy,
            **builder_kwargs,
        )
    receptor = receptor_guidance_metadata(requested_system, receptor_policy)
    fallback = _fallback_summary(receptor)
    crystal_coords, match_method, reference_alignment = _crystal_coords(
        item,
        mol_in,
        meta["pocket_center"],
    )
    numerical, gradients = _numerical_preflight(crystal_coords, requested_system)
    entry: dict[str, object] = {
        "status": "success",
        "seed": seed,
        "protein": str(item.protein),
        "protein_sha256": file_sha256(item.protein),
        "ligand_reference": str(item.ligand_ref),
        "ligand_reference_sha256": file_sha256(item.ligand_ref),
        "ligand_input_kind": "smiles" if item.smiles else "reference_file",
        "ligand_input_identity_sha256": item.ligand_input_identity_sha256,
        "ligand_input_canonical_smiles": item.ligand_input_canonical_smiles,
        "pocket_center": list(item.pocket_center),
        "fragment_count": int(meta["num_frag"]),
        "ligand_heavy_atoms": int(meta["num_atom"]),
        "crystal_atom_match_method": match_method,
        "numerical_preflight_reference_alignment": reference_alignment,
        "receptor": {
            **receptor,
            **fallback,
        },
        "system_reference_sha256": _system_reference_sha256(requested_system),
        "topology_reference_sha256": requested_system.topology.reference_sha256(),
        "interaction_reference_sha256": (
            requested_system.interaction_topology.reference_sha256()
            if requested_system.interaction_topology is not None
            else None
        ),
        "crystal_numerical_preflight": numerical,
        "strict_failure": strict_failure,
    }
    if strict_system is not None:
        strict_numerical, strict_gradients = _numerical_preflight(
            crystal_coords,
            strict_system,
        )
        entry["strict_supported_equivalence"] = _equivalence_check(
            requested_system,
            strict_system,
            receptor_policy,
            numerical,
            strict_numerical,
            gradients,
            strict_gradients,
        )
    else:
        entry["strict_supported_equivalence"] = None
    return entry


def _id_group(ids: list[str]) -> dict[str, object]:
    ordered = sorted(ids)
    return {
        "count": len(ordered),
        "ids": ordered,
        "ids_sha256": sorted_id_sha256(ordered),
    }


def _load_legacy_ids(path: Path | None, dataset: str) -> list[str] | None:
    if path is None:
        return None
    payload = json.loads(path.read_text())
    dataset_payload = payload.get("datasets", {}).get(dataset)
    if not isinstance(dataset_payload, dict):
        raise ValueError(f"legacy eligibility manifest has no {dataset!r} dataset")
    ids = dataset_payload.get("eligible_ids")
    if not isinstance(ids, list) or not all(isinstance(value, str) for value in ids):
        raise ValueError("legacy eligibility manifest eligible_ids must be a string list")
    return sorted(value.lower() for value in ids)


def build_audit_report(args: argparse.Namespace) -> dict[str, object]:
    if args.receptor_policy not in UNIFIED_GUIDANCE_RECEPTOR_POLICIES:
        raise ValueError(f"unsupported receptor policy: {args.receptor_policy}")
    pocket_centers = load_pocket_centers(args.pocket_centers)
    _, benchmark_input_identity = load_benchmark_inputs(
        args.dataset,
        args.external_dir,
        args.benchmark_input_manifest,
    )
    all_complexes = discover_complexes(
        args.dataset,
        args.data_dir,
        args.external_dir,
        pocket_centers,
        args.benchmark_input_manifest,
    )
    seed_by_id = global_seed_by_id(all_complexes, args.seed)
    selected = list(all_complexes)
    if args.limit is not None:
        selected = selected[: args.limit]
    if args.only_id:
        requested = {value.lower() for value in args.only_id}
        selected = [item for item in selected if item.complex_id in requested]
        missing = requested - {item.complex_id for item in selected}
        if missing:
            raise ValueError(f"requested benchmark IDs not found: {sorted(missing)}")
    if not selected:
        raise ValueError("coverage audit selected no complexes")

    complexes: dict[str, dict[str, object]] = {}
    failures: dict[str, dict[str, object]] = {}
    for index, item in enumerate(selected, start=1):
        try:
            entry = _audit_one(
                item,
                seed=seed_by_id[item.complex_id],
                receptor_policy=args.receptor_policy,
                pocket_cutoff=args.pocket_cutoff,
            )
            complexes[item.complex_id] = entry
            print(
                f"[{index:04d}/{len(selected)}] {item.complex_id} OK "
                f"obstacles={entry['receptor']['obstacle_count']} "
                f"metal_fallbacks={entry['receptor']['metal_fallback_count']}"
            )
        except Exception as exc:
            failure = serialize_evaluation_failure(item.complex_id, exc)
            failures[item.complex_id] = failure
            complexes[item.complex_id] = {
                "status": "failed",
                "seed": seed_by_id[item.complex_id],
                "failure": failure,
            }
            print(f"[{index:04d}/{len(selected)}] {item.complex_id} FAIL {exc!r}")

    discovered_ids = [item.complex_id for item in all_complexes]
    audited_ids = [item.complex_id for item in selected]
    success_ids = sorted(set(audited_ids) - set(failures))
    failed_ids = sorted(failures)
    slice_ids: dict[str, list[str]] = {
        "strict_supported": [],
        "nonprotein_only": [],
        "metal_only": [],
        "nonprotein_and_metal": [],
    }
    ligand_representation_ids: dict[str, list[str]] = {
        "exact_graph": [],
        "same_connectivity_representation_mismatch": [],
    }
    fallback_reason_ids: dict[str, set[str]] = defaultdict(set)
    fallback_reason_site_counts: Counter[str] = Counter()
    failure_codes: Counter[str] = Counter()
    equivalence_mismatch_ids: list[str] = []
    equivalence_checked_ids: list[str] = []
    for complex_id in audited_ids:
        entry = complexes[complex_id]
        if entry["status"] != "success":
            code = str(entry["failure"].get("code", entry["failure"]["error_type"]))
            failure_codes[code] += 1
            continue
        receptor = entry["receptor"]
        relation = str(
            entry["numerical_preflight_reference_alignment"].get("relation", "")
        )
        if relation not in ligand_representation_ids:
            raise ValueError(
                f"unexpected accepted ligand representation relation: {complex_id}/{relation}"
            )
        ligand_representation_ids[relation].append(complex_id)
        obstacle = int(receptor["obstacle_count"]) > 0
        metal = int(receptor["metal_fallback_count"]) > 0
        if entry["strict_failure"] is None:
            slice_name = "strict_supported"
        elif obstacle and metal:
            slice_name = "nonprotein_and_metal"
        elif obstacle:
            slice_name = "nonprotein_only"
        elif metal:
            slice_name = "metal_only"
        else:
            code = str(entry["strict_failure"].get("code", "strict_failure_unsliced"))
            slice_name = "metal_only" if "metal" in code or "zinc" in code else "nonprotein_only"
        slice_ids[slice_name].append(complex_id)
        for reason, site_count in receptor["metal_fallback_reasons"].items():
            fallback_reason_ids[reason].add(complex_id)
            fallback_reason_site_counts[reason] += int(site_count)
        equivalence = entry.get("strict_supported_equivalence")
        if equivalence is not None:
            equivalence_checked_ids.append(complex_id)
            if not bool(equivalence["passed"]):
                equivalence_mismatch_ids.append(complex_id)

    chemistry_slices = {name: _id_group(ids) for name, ids in slice_ids.items()}
    ligand_representation_slices = {
        name: _id_group(ids) for name, ids in ligand_representation_ids.items()
    }
    fallback_reasons = {
        reason: {
            **_id_group(sorted(ids)),
            "site_count": int(fallback_reason_site_counts[reason]),
        }
        for reason, ids in sorted(fallback_reason_ids.items())
    }
    legacy_ids = _load_legacy_ids(args.legacy_eligibility_manifest, args.dataset)
    strict_supported_ids = sorted(slice_ids["strict_supported"])
    legacy_comparison = None
    if legacy_ids is not None:
        legacy_comparison = {
            "manifest": str(args.legacy_eligibility_manifest),
            "manifest_sha256": file_sha256(args.legacy_eligibility_manifest),
            "expected": _id_group(legacy_ids),
            "observed": _id_group(strict_supported_ids),
            "exact_match": legacy_ids == strict_supported_ids,
            "missing_ids": sorted(set(legacy_ids) - set(strict_supported_ids)),
            "unexpected_ids": sorted(set(strict_supported_ids) - set(legacy_ids)),
        }
    dataset_report = {
        "discovered": len(discovered_ids),
        "ids": discovered_ids,
        "ids_sha256": sorted_id_sha256(discovered_ids),
        "ids_hash_contract": ID_HASH_CONTRACT,
        "audited": len(audited_ids),
        "audited_ids": audited_ids,
        "audited_ids_sha256": sorted_id_sha256(audited_ids),
        "complete": audited_ids == discovered_ids,
        "success": len(success_ids),
        "success_ids": success_ids,
        "success_ids_sha256": sorted_id_sha256(success_ids),
        "failed": len(failed_ids),
        "failed_ids": failed_ids,
        "failed_ids_sha256": sorted_id_sha256(failed_ids),
        "failure_codes": dict(sorted(failure_codes.items())),
        "chemistry_slices": chemistry_slices,
        "ligand_representation_slices": ligand_representation_slices,
        "fallback_reasons": fallback_reasons,
        "strict_supported_equivalence": {
            **_id_group(equivalence_checked_ids),
            "passed": len(equivalence_mismatch_ids) == 0,
            "mismatch": _id_group(equivalence_mismatch_ids),
            "legacy_v1_eligibility": legacy_comparison,
        },
        "inputs": {
            "data_dir": str(args.data_dir),
            "external_dir": str(args.external_dir),
            "pocket_centers": str(args.pocket_centers),
            "pocket_centers_sha256": file_sha256(args.pocket_centers),
            "benchmark_input_identity": benchmark_input_identity,
        },
        "complexes": complexes,
    }
    policy_identities = {
        str(entry["receptor"]["identity_sha256"]): entry["receptor"]["identity"]
        for entry in complexes.values()
        if entry.get("status") == "success"
        and isinstance(entry.get("receptor"), dict)
        and entry["receptor"].get("identity_sha256")
    }
    if len(policy_identities) != 1:
        raise ValueError(
            "coverage audit requires one exact receptor-policy identity, got "
            f"{sorted(policy_identities)}"
        )
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "protocol_id": args.protocol_id,
        "created_utc": datetime.now(UTC).isoformat(),
        "receptor_policy": args.receptor_policy,
        "receptor_policy_identity": next(iter(policy_identities.values())),
        "datasets": {args.dataset: dataset_report},
        "implementation": _implementation_identity(),
        "parameter_set": guidance_parameter_identity(),
        "settings": {
            "seed": args.seed,
            "global_seed_contract": "seed + one-based index in globally sorted discovered IDs",
            "id_hash_contract": ID_HASH_CONTRACT,
            "pocket_cutoff_angstrom": args.pocket_cutoff,
            "guidance_shell_angstrom": GUIDANCE_SHELL_ANGSTROM,
            "device": "cpu",
            "dtype": "torch.float64",
        },
        "information_boundary": {
            "discovery_and_preprocessing": "identical evaluator functions",
            "pocket_centers": "frozen manifest only",
            "reference_pose_use": "numerical energy/gradient preflight only",
            "forbidden_access": ["RMSD", "confidence", "PoseBusters outcome"],
            "coefficient_tuning_allowed": False,
        },
    }


def merge_audit_reports(paths: list[Path], protocol_id: str) -> dict[str, object]:
    datasets: dict[str, object] = {}
    source_reports: list[dict[str, str]] = []
    receptor_policy = None
    shared_identity: dict[str, object] = {}
    for path in paths:
        report = json.loads(path.read_text())
        if report.get("schema_version") != AUDIT_SCHEMA_VERSION:
            raise ValueError(f"unsupported audit schema in {path}")
        policy = report.get("receptor_policy")
        if receptor_policy is None:
            receptor_policy = policy
        elif policy != receptor_policy:
            raise ValueError("cannot merge audit reports with different receptor policies")
        for key in ("implementation", "parameter_set", "receptor_policy_identity"):
            value = report.get(key)
            if not isinstance(value, dict) or not isinstance(value.get("sha256"), str):
                raise ValueError(f"audit report {path} lacks a versioned {key} identity")
            if key not in shared_identity:
                shared_identity[key] = value
            elif _canonical_json(shared_identity[key]) != _canonical_json(value):
                raise ValueError(f"cannot merge audit reports with different {key}")
        for dataset, payload in report.get("datasets", {}).items():
            if dataset in datasets:
                raise ValueError(f"duplicate dataset {dataset!r} while merging audits")
            datasets[dataset] = payload
        source_reports.append({"path": str(path), "sha256": file_sha256(path)})
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "protocol_id": protocol_id,
        "created_utc": datetime.now(UTC).isoformat(),
        "receptor_policy": receptor_policy,
        **shared_identity,
        "datasets": datasets,
        "source_reports": source_reports,
        "merged_only": True,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("astex", "posebusters"))
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--external-dir", type=Path, default=Path("data/external_test"))
    parser.add_argument("--benchmark-input-manifest", type=Path)
    parser.add_argument("--pocket-centers", type=Path)
    parser.add_argument(
        "--receptor-policy",
        "--unified-guidance-receptor-policy",
        dest="receptor_policy",
        choices=UNIFIED_GUIDANCE_RECEPTOR_POLICIES,
        default="geometry_only",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pocket-cutoff", type=float, default=DEFAULT_POCKET_CUTOFF)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--only-id", action="append", default=[])
    parser.add_argument("--legacy-eligibility-manifest", type=Path)
    parser.add_argument("--protocol-id", default=AUDIT_PROTOCOL_ID)
    parser.add_argument("--merge-audit", type=Path, action="append", default=[])
    parser.add_argument(
        "--require-complete-success",
        action="store_true",
        help="Write diagnostics, then exit nonzero unless every discovered complex passed.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.merge_audit:
        if args.dataset or args.data_dir or args.pocket_centers:
            parser.error("--merge-audit is exclusive with dataset audit inputs")
        report = merge_audit_reports(args.merge_audit, args.protocol_id)
    else:
        missing = [
            name
            for name in ("dataset", "data_dir", "pocket_centers")
            if getattr(args, name) is None
        ]
        if missing:
            parser.error(
                "dataset audit requires " + ", ".join(f"--{x.replace('_', '-')}" for x in missing)
            )
        if args.limit is not None and args.limit < 1:
            parser.error("--limit must be positive")
        if args.pocket_cutoff <= 0:
            parser.error("--pocket-cutoff must be positive")
        report = build_audit_report(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.require_complete_success:
        incomplete = []
        for dataset, payload in report.get("datasets", {}).items():
            if not isinstance(payload, dict):
                incomplete.append(str(dataset))
                continue
            discovered = int(payload.get("discovered", -1))
            success = int(payload.get("success", -1))
            failed = int(payload.get("failed", -1))
            equivalence = payload.get("strict_supported_equivalence", {})
            expected = (
                FULL_PROTOCOL_EXPECTED_COUNTS.get(str(dataset))
                if report.get("protocol_id") == AUDIT_PROTOCOL_ID
                else None
            )
            if (
                payload.get("complete") is not True
                or (expected is not None and discovered != expected)
                or success != discovered
                or failed != 0
                or not isinstance(equivalence, dict)
                or equivalence.get("passed") is not True
            ):
                incomplete.append(str(dataset))
        if incomplete:
            raise SystemExit(
                "coverage audit did not satisfy complete-success gate: "
                + ", ".join(incomplete)
            )


if __name__ == "__main__":
    main()
