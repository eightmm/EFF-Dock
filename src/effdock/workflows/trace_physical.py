"""Trace unified self-contained guidance on a crystal pose or saved trajectory."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

import torch
from rdkit import rdBase

from effdock.guidance import (
    TRACE_SCHEMA_VERSION,
    InteractionEnergyConfig,
    PhysicalEnergyConfig,
    PoseState,
    UnsupportedPhysicalChemistryError,
    build_physical_system,
    interaction_profile_metadata,
    make_crystal_perturbations,
    trace_guidance_pose,
)
from effdock.guidance.parameterization import (
    guidance_parameter_identity,
    load_effff_v2,
    load_interaction_v1,
)
from effdock.preprocess.fragments import decompose_fragments
from effdock.preprocess.ligand import (
    featurize_ligand,
    ligand_graph_identity,
    load_molecule,
)


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _implementation_identity() -> dict[str, object]:
    effdock_root = Path(__file__).resolve().parents[1]
    source_paths = [
        effdock_root / "cli.py",
        effdock_root / "preprocess" / "fragments.py",
        effdock_root / "preprocess" / "ligand.py",
        effdock_root / "preprocess" / "protein.py",
        effdock_root / "workflows" / "trace_physical.py",
        *sorted((effdock_root / "guidance").glob("*.py")),
        *sorted((effdock_root / "guidance" / "parameters").glob("*")),
    ]
    source_paths = [path for path in source_paths if path.is_file()]
    digest = sha256()
    relative_paths: list[str] = []
    for path in source_paths:
        relative = path.relative_to(effdock_root).as_posix()
        relative_paths.append(relative)
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    runtime_versions = {
        "rdkit": rdBase.rdkitVersion,
        "torch": torch.__version__,
    }
    digest.update(b"runtime_versions\0")
    digest.update(
        json.dumps(
            runtime_versions,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    )
    digest.update(b"\0")
    return {
        "sha256": digest.hexdigest(),
        "files": relative_paths,
        "runtime_versions": runtime_versions,
    }


def _parse_center(value: str) -> torch.Tensor:
    parts = value.replace(",", " ").split()
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("--pocket-center requires x,y,z")
    try:
        return torch.tensor([float(part) for part in parts], dtype=torch.float64)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--pocket-center requires three floats") from exc


def _load_trace_ligand(path: Path):
    suffix = path.suffix.lower()
    if suffix == ".sdf":
        mol, used_fallback, sanitize_ok = load_molecule(path)
    elif suffix == ".mol2":
        mol, used_fallback, sanitize_ok = load_molecule(None, path)
    else:
        raise ValueError("physical trace requires a sanitized .sdf or .mol2 ligand file")
    if mol is None:
        raise ValueError(f"failed to parse physical-trace ligand: {path}")
    if not sanitize_ok:
        raise UnsupportedPhysicalChemistryError(
            "ligand_sanitization_failed",
            "physical trace requires successful ligand sanitization because atom "
            "hybridization, aromaticity, and stereochemistry define energy terms",
            details={"ligand": str(path)},
        )
    return mol, used_fallback


def _selected_result_coordinates(
    results: dict[str, Any],
    *,
    sample_index: int | None,
    frame_stride: int,
) -> list[torch.Tensor]:
    poses = results.get("poses") or []
    trajectories = results.get("trajectories") or []
    indices = range(len(poses)) if sample_index is None else [sample_index]
    selected: list[torch.Tensor] = []
    for index in indices:
        if not 0 <= index < len(poses):
            raise IndexError(f"sample index {index} is outside saved pose count {len(poses)}")
        pose_entry = poses[index]
        selected.append(
            torch.as_tensor(
                pose_entry["atom_pos_pred"] if isinstance(pose_entry, dict) else pose_entry
            ).view(-1, 3)
        )
        if index < len(trajectories):
            frames = trajectories[index].get("traj") or []
            selected.extend(
                torch.as_tensor(frames[step]).view(-1, 3)
                for step in range(0, len(frames), frame_stride)
            )
    return selected


def _validate_saved_results_ligand_identity(
    results: dict[str, Any],
    *,
    mol,
    fragment_id: torch.Tensor,
    ligand_path: Path,
) -> dict[str, object]:
    if results.get("schema_version") != "effdock.docking_results.v2":
        raise ValueError("saved results must use schema_version effdock.docking_results.v2")
    actual = results.get("ligand_identity")
    if not isinstance(actual, dict):
        raise ValueError(
            "saved results lack required ligand_identity metadata; regenerate "
            "results.pt with EFF-Dock docking-results V2"
        )
    expected = ligand_graph_identity(mol, fragment_id)
    required = tuple(expected)
    missing = [name for name in required if name not in actual]
    if missing:
        raise ValueError(f"saved results ligand_identity is missing fields: {missing}")
    mismatched = [name for name in required if actual[name] != expected[name]]
    if mismatched:
        raise ValueError(
            "saved results ligand identity/order does not match --ligand; "
            f"mismatched fields: {mismatched}"
        )
    source = actual.get("source")
    if not isinstance(source, dict):
        raise ValueError("saved results ligand_identity lacks source metadata")
    if source.get("kind") != "file":
        raise ValueError(
            "saved results were generated from a literal ligand input; exact "
            "file-provenance tracing is unsupported"
        )
    current_sha256 = _file_sha256(ligand_path)
    if source.get("sha256") != current_sha256:
        raise ValueError("saved results ligand source hash does not match --ligand")
    return actual


def _energy_delta(
    energies: dict[str, float],
    crystal_energies: dict[str, float],
) -> dict[str, float]:
    return {name: float(value - crystal_energies[name]) for name, value in energies.items()}


def _trace_saved_results(
    results: dict[str, Any],
    system,
    *,
    energy_config: PhysicalEnergyConfig,
    interaction_config: InteractionEnergyConfig,
    crystal_energies: dict[str, float],
    sample_index: int | None,
    frame_stride: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    poses = results.get("poses") or []
    trajectories = results.get("trajectories") or []
    indices = range(len(poses)) if sample_index is None else [sample_index]
    for index in indices:
        if not 0 <= index < len(poses):
            raise IndexError(f"sample index {index} is outside saved pose count {len(poses)}")
        pose_entry = poses[index]
        pose_coords = pose_entry["atom_pos_pred"] if isinstance(pose_entry, dict) else pose_entry
        row = trace_guidance_pose(
            pose_coords,
            system,
            energy_config=energy_config,
            interaction_config=interaction_config,
            pose_kind="sampled_final",
            pose_index=index,
        )
        row["energy_delta_from_crystal"] = _energy_delta(row["energies"], crystal_energies)
        rows.append(row)

        if index >= len(trajectories):
            continue
        trajectory = trajectories[index]
        frames = trajectory.get("traj") or []
        times = trajectory.get("traj_times") or [None] * len(frames)
        if len(times) != len(frames):
            raise ValueError(f"trajectory {index} has {len(frames)} frames but {len(times)} times")
        for step in range(0, len(frames), frame_stride):
            row = trace_guidance_pose(
                frames[step],
                system,
                energy_config=energy_config,
                interaction_config=interaction_config,
                pose_kind="trajectory",
                pose_index=index,
                step=step,
                t=times[step],
            )
            row["energy_delta_from_crystal"] = _energy_delta(row["energies"], crystal_energies)
            rows.append(row)
    return rows


def build_trace_report(args: argparse.Namespace) -> dict[str, object]:
    protein = args.protein.resolve()
    ligand = args.ligand.resolve()
    if not protein.is_file():
        raise FileNotFoundError(f"protein PDB not found: {protein}")
    if not ligand.is_file():
        raise FileNotFoundError(f"ligand file not found: {ligand}")
    if args.frame_stride <= 0:
        raise ValueError("--frame-stride must be positive")

    mol, used_mol2_fallback = _load_trace_ligand(ligand)
    lig_data = featurize_ligand(mol)
    if lig_data is None:
        raise ValueError("ligand featurization failed")
    frag_data = decompose_fragments(mol, lig_data["atom_coords"])
    if frag_data is None:
        raise ValueError("ligand fragment decomposition failed")

    saved_results: dict[str, Any] | None = None
    pocket_center: torch.Tensor | None = args.pocket_center
    if args.results is not None:
        if not args.results.is_file():
            raise FileNotFoundError(f"saved results not found: {args.results}")
        saved_results = torch.load(
            args.results,
            map_location="cpu",
            weights_only=True,
        )
        if not isinstance(saved_results, dict):
            raise ValueError("--results must contain an EFF-Dock results dictionary")
        _validate_saved_results_ligand_identity(
            saved_results,
            mol=mol,
            fragment_id=frag_data["fragment_id"],
            ligand_path=ligand,
        )
        stored_center = saved_results.get("pocket_center")
        if pocket_center is None and stored_center is not None:
            pocket_center = torch.as_tensor(stored_center, dtype=torch.float64)

    crystal_absolute = torch.tensor(
        mol.GetConformer().GetPositions(),
        dtype=torch.float64,
    )
    results_frame = args.results_frame
    if results_frame == "auto":
        results_frame = "pocket_centered" if args.results is not None else "absolute"
    if results_frame == "pocket_centered":
        if pocket_center is None:
            raise ValueError(
                "pocket-centered tracing requires --pocket-center or a saved pocket center"
            )
        crystal_coords = crystal_absolute - pocket_center.view(1, 3)
        coordinate_origin = pocket_center
        coordinate_frame = "pocket_centered_angstrom"
    else:
        crystal_coords = crystal_absolute
        coordinate_origin = None
        coordinate_frame = "absolute_pdb_angstrom"

    near_parts = [crystal_absolute]
    if saved_results is not None:
        saved_coordinates = _selected_result_coordinates(
            saved_results,
            sample_index=args.sample_index,
            frame_stride=args.frame_stride,
        )
        for saved_coords in saved_coordinates:
            saved_coords = saved_coords.to(torch.float64)
            if saved_coords.shape != crystal_absolute.shape:
                raise ValueError(
                    "saved coordinate atom count does not match the physical-trace ligand"
                )
            if results_frame == "pocket_centered":
                saved_coords = saved_coords + pocket_center.view(1, 3)
            near_parts.append(saved_coords)
    interaction_config = InteractionEnergyConfig()
    required_shell_cutoff = max(
        args.nonbonded_cutoff,
        interaction_config.hydrophobic_cutoff,
        interaction_config.hydrogen_bond_cutoff,
        interaction_config.formal_charge_cutoff,
    )
    if args.protein_cutoff < required_shell_cutoff:
        raise ValueError(
            "--protein-cutoff must cover every active physical/interaction "
            f"cutoff ({required_shell_cutoff:g} angstrom)"
        )

    near_coords = torch.cat(near_parts, dim=0)
    if args.perturbations:
        provisional_system = build_physical_system(
            mol,
            protein,
            fragment_id=frag_data["fragment_id"],
            near_coords=near_coords,
            protein_cutoff=args.protein_cutoff,
            coordinate_origin=coordinate_origin,
        ).to(device=torch.device("cpu"), dtype=torch.float64)
        provisional_coords = crystal_coords.to(torch.float64)
        provisional_states = make_crystal_perturbations(
            provisional_coords,
            provisional_system,
            stretch_angstrom=args.stretch_angstrom,
            torsion_degrees=args.torsion_degrees,
            overlap_distance_angstrom=args.overlap_distance,
        )
        for state in provisional_states:
            absolute_state = state.coords.detach().cpu()
            if coordinate_origin is not None:
                absolute_state = absolute_state + coordinate_origin.view(1, 3)
            near_parts.append(absolute_state)
        near_coords = torch.cat(near_parts, dim=0)

    device = torch.device(args.device)
    system = build_physical_system(
        mol,
        protein,
        fragment_id=frag_data["fragment_id"],
        near_coords=near_coords,
        protein_cutoff=args.protein_cutoff,
        coordinate_origin=coordinate_origin,
    ).to(device=device, dtype=torch.float64)
    crystal_coords = crystal_coords.to(device=device, dtype=torch.float64)
    energy_config = PhysicalEnergyConfig(
        softcore=args.softcore,
        switch_distance=args.switch_distance,
        cutoff=args.nonbonded_cutoff,
        protein_chunk_size=args.protein_chunk_size,
    )

    states = (
        make_crystal_perturbations(
            crystal_coords,
            system,
            stretch_angstrom=args.stretch_angstrom,
            torsion_degrees=args.torsion_degrees,
            overlap_distance_angstrom=args.overlap_distance,
        )
        if args.perturbations
        else []
    )
    if not states:
        states = [PoseState("crystal", crystal_coords, {"operation": "none"})]

    rows: list[dict[str, object]] = []
    for state in states:
        row = trace_guidance_pose(
            state.coords,
            system,
            energy_config=energy_config,
            interaction_config=interaction_config,
            pose_kind=state.name,
        )
        row["state_details"] = state.details
        rows.append(row)
    crystal_energies = rows[0]["energies"]
    for row in rows:
        row["energy_delta_from_crystal"] = _energy_delta(row["energies"], crystal_energies)

    if saved_results is not None:
        rows.extend(
            _trace_saved_results(
                saved_results,
                system,
                energy_config=energy_config,
                interaction_config=interaction_config,
                crystal_energies=crystal_energies,
                sample_index=args.sample_index,
                frame_stride=args.frame_stride,
            )
        )

    raw_parameters = load_effff_v2()
    raw_interaction_parameters = load_interaction_v1()
    warnings = [
        "This parameter set is diagnostic and is not AMBER, GAFF, OpenFF, UFF, or MMFF.",
        "Crystal values are diagnostics only and must not enter inference or tuning.",
        "Hydrophobic, hydrogen-bond, and screened formal-charge values are pose-guidance terms, not affinity or free energy.",
        "Partial-charge electrostatics, solvation, metal coordination, and receptor flexibility are absent.",
        "Vina code is retained elsewhere for legacy evaluation but is excluded from GuidanceEnergy.",
    ]
    if system.excluded_nonprotein_atoms:
        warnings.append(
            f"Excluded {system.excluded_nonprotein_atoms} non-protein records "
            "outside the active shell; residue names are recorded in system "
            "metadata."
        )
    if system.interaction_topology is not None and bool(
        system.interaction_topology.protein_is_unsupported_variant.any()
    ):
        warnings.append(
            "Explicit mapped residue variants outside the admitted "
            "HID/HIE/HIP policy were fail-closed for every interaction mask; "
            "their atom labels are recorded in interaction_contacts.exclusions."
        )
    if system.interaction_topology is not None and bool(
        system.interaction_topology.protein_is_geometry_excluded_hbond_site.any()
    ):
        warnings.append(
            "Protein H-bond sites with a missing or mismatched canonical "
            "heavy-neighbor degree were fail-closed; atom labels are recorded "
            "in interaction_contacts.exclusions."
        )
    report: dict[str, object] = {
        "schema_version": TRACE_SCHEMA_VERSION,
        "created_utc": datetime.now(UTC).isoformat(),
        "protocol_id": "EFFDOCK-GUIDANCE-DIAGNOSTIC-V4",
        "status": "diagnostic_only",
        "supported": True,
        "implementation": _implementation_identity(),
        "coordinate_frame": coordinate_frame,
        "inputs": {
            "protein": str(protein),
            "protein_sha256": _file_sha256(protein),
            "ligand": str(ligand),
            "ligand_sha256": _file_sha256(ligand),
            "ligand_has_input_pose": True,
            "ligand_sanitize_ok": True,
            "ligand_used_mol2_fallback": bool(used_mol2_fallback),
            "ligand_hydrogen_policy": "explicit hydrogens removed",
            "ligand_component_policy": "largest connected component",
            "results": str(args.results.resolve()) if args.results else None,
            "results_sha256": _file_sha256(args.results) if args.results else None,
            "results_ligand_identity_sha256": (
                saved_results["ligand_identity"]["sha256"] if saved_results is not None else None
            ),
            "pocket_center_angstrom": (
                pocket_center.detach().cpu().tolist() if pocket_center is not None else None
            ),
        },
        "parameter_set": {
            **guidance_parameter_identity(),
            "physical_details": {
                "provenance": raw_parameters["provenance"],
                "functional_form": raw_parameters["functional_form"],
                "coefficient_units": raw_parameters["coefficient_units"],
            },
            "interaction_details": {
                "provenance": raw_interaction_parameters["provenance"],
                "functional_form": raw_interaction_parameters["functional_form"],
                "typing_policy": raw_interaction_parameters["typing_policy"],
            },
        },
        "guidance_layers": {
            "physical": {
                "status": "active_diagnostic",
                "active_terms": [
                    "ligand_intra_bond",
                    "ligand_intra_angle",
                    "ligand_intra_proper",
                    "ligand_intra_improper",
                    "ligand_intra_lj_repulsive",
                    "ligand_intra_lj_attractive",
                    "protein_ligand_lj_repulsive",
                    "protein_ligand_lj_attractive",
                ],
                "inactive_terms": {
                    "electrostatic": (
                        "no provenance-tracked partial-charge contract; "
                        "screened formal-charge groups are owned by "
                        "InteractionGuidance"
                    ),
                    "solvation": "no admitted self-contained solvation model",
                    "metal_coordination": (
                        "owned by InteractionGuidance; Zn(II) V0 is contract-only"
                    ),
                },
            },
            "interaction": interaction_profile_metadata(),
        },
        "energy_config": {
            "physical": asdict(energy_config),
            "interaction": asdict(interaction_config),
        },
        "system": {
            "ligand_heavy_atoms": system.topology.num_atoms,
            "ligand_fragments": int(system.topology.fragment_id.max().item()) + 1,
            "topology_reference_sha256": system.topology.reference_sha256(),
            "interaction_reference_sha256": (
                system.interaction_topology.reference_sha256()
                if system.interaction_topology is not None
                else None
            ),
            "protein_source_heavy_atoms": system.protein_source_atoms,
            "protein_parameterized_source_heavy_atoms": (system.protein_parameterized_source_atoms),
            "protein_shell_heavy_atoms": int(system.protein_coords.shape[0]),
            "excluded_nonprotein_atoms": (system.excluded_nonprotein_atoms),
            "excluded_nonprotein_residues": list(system.excluded_nonprotein_residues),
            "receptor_record_policy": system.receptor_policy,
            "protein_shell_cutoff_angstrom": float(args.protein_cutoff),
            "term_counts": {
                **system.topology.term_counts(),
                "protein_ligand_pairs": int(
                    system.topology.num_atoms * system.protein_coords.shape[0]
                ),
            },
            "interaction_term_counts": (
                system.interaction_topology.term_counts()
                if system.interaction_topology is not None
                else {}
            ),
        },
        "term_semantics": {
            "ligand_intra_bond": ("input-conformer reference restraint on covalent cut bonds"),
            "ligand_intra_angle": (
                "input-conformer reference restraint on cross-fragment covalent angles"
            ),
            "ligand_intra_proper": ("generic periodic cut-bond torsion normalized per cut bond"),
            "protein_ligand_lj_attractive": (
                "UFF-style dispersion attraction; not affinity or hydrogen-bond score"
            ),
            "ligand_intra_improper": (
                "periodic planarity or input-stereochemistry restraint; "
                "no AMBER/GAFF compatibility claim"
            ),
            "interaction_hydrophobic": (
                "carbon-only smooth contact with ligand-site soft-OR saturation"
            ),
            "interaction_hydrogen_bond": (
                "N/O heavy-atom radial and idealized donor/acceptor "
                "missing-valence-cone proxy with donor-site soft-OR "
                "saturation"
            ),
            "interaction_screened_formal_charge": (
                "Debye-Huckel-screened canonical formal-charge-group "
                "interaction; not partial-charge electrostatics, solvation, "
                "affinity, or free energy"
            ),
            "total": ("unified GuidanceEnergy = PhysicalEnergy + InteractionEnergy; Vina excluded"),
        },
        "warnings": warnings,
        "rows": rows,
    }
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="eff-dock physical trace",
        description=(
            "Trace in-repository Torch physical and interaction guidance terms "
            "on a crystal pose and optionally on saved EFF-Dock trajectories."
        ),
    )
    parser.add_argument("--protein", type=Path, required=True)
    parser.add_argument("--ligand", type=Path, required=True)
    parser.add_argument("--results", type=Path, default=None)
    parser.add_argument(
        "--results-frame",
        choices=("auto", "absolute", "pocket_centered"),
        default="auto",
        help="Coordinate frame used by --results (auto: saved results are pocket_centered).",
    )
    parser.add_argument("--pocket-center", type=_parse_center, default=None)
    parser.add_argument("--output", type=Path, default=Path("outputs/guidance/trace.json"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--protein-cutoff", type=float, default=10.0)
    parser.add_argument("--softcore", type=float, default=0.75)
    parser.add_argument("--switch-distance", type=float, default=6.0)
    parser.add_argument("--nonbonded-cutoff", type=float, default=8.0)
    parser.add_argument("--protein-chunk-size", type=int, default=512)
    parser.add_argument(
        "--perturbations",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Trace deterministic cut-bond and protein-overlap diagnostic states.",
    )
    parser.add_argument("--stretch-angstrom", type=float, default=0.5)
    parser.add_argument("--torsion-degrees", type=float, default=30.0)
    parser.add_argument("--overlap-distance", type=float, default=0.5)
    parser.add_argument("--sample-index", type=int, default=None)
    parser.add_argument("--frame-stride", type=int, default=1)
    return parser


def _print_summary(report: dict[str, object], output: Path) -> None:
    print(f"Guidance diagnostic trace: {output}")
    print(
        f"  parameter_set={report['parameter_set']['name']} "
        f"version={report['parameter_set']['version']}"
    )
    system = report["system"]
    print(
        f"  ligand_atoms={system['ligand_heavy_atoms']} "
        f"fragments={system['ligand_fragments']} "
        f"protein_shell_atoms={system['protein_shell_heavy_atoms']}"
    )
    print(
        f"  {'pose':<26}{'total':>14}{'physical':>14}"
        f"{'hydrophobe':>14}{'H-bond':>14}{'formal-q':>14}"
    )
    for row in report["rows"]:
        energies = row["energies"]
        label = row["pose_kind"]
        if row["pose_kind"] == "trajectory":
            label = f"traj[{row['pose_index']}]:{row['step']}"
        print(
            f"  {label:<26}"
            f"{energies['total']:>14.4f}"
            f"{row['energy_groups']['physical']:>14.4f}"
            f"{energies['interaction_hydrophobic']:>14.4f}"
            f"{energies['interaction_hydrogen_bond']:>14.4f}"
            f"{energies['interaction_screened_formal_charge']:>14.4f}"
        )


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    try:
        report = build_trace_report(args)
    except UnsupportedPhysicalChemistryError as exc:
        report = {
            "schema_version": TRACE_SCHEMA_VERSION,
            "created_utc": datetime.now(UTC).isoformat(),
            "protocol_id": "EFFDOCK-GUIDANCE-DIAGNOSTIC-V4",
            "status": "unsupported",
            "supported": False,
            "implementation": _implementation_identity(),
            "inputs": {
                "protein": str(args.protein),
                "protein_sha256": (_file_sha256(args.protein) if args.protein.is_file() else None),
                "ligand": str(args.ligand),
                "ligand_sha256": (_file_sha256(args.ligand) if args.ligand.is_file() else None),
                "results": str(args.results) if args.results else None,
            },
            "parameter_set": guidance_parameter_identity(),
            "guidance_layers": {
                "physical": {"status": "unsupported"},
                "interaction": interaction_profile_metadata(),
            },
            "failure": exc.as_dict(),
            "rows": [],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"Guidance diagnostic unsupported: {exc}")
        print(f"  structured report: {args.output}")
        raise SystemExit(2) from exc
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    _print_summary(report, args.output)


if __name__ == "__main__":
    main()
