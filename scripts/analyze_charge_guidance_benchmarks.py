#!/usr/bin/env python
"""Report-only screened-charge characterization on frozen Astex/PB poses.

The external benchmark results produced here must not select a formula,
coefficient, schedule, term, or sampler setting.  Production activation remains
an internal-validation decision.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

import torch
from rdkit import Chem, rdBase

from effdock.evaluation.benchmark import match_atoms
from effdock.guidance import (
    InteractionEnergyConfig,
    UnsupportedPhysicalChemistryError,
    build_physical_system,
    interaction_contact_stats,
    interaction_energy,
    type_ligand_interactions,
)
from effdock.preprocess.fragments import decompose_fragments
from effdock.preprocess.ligand import featurize_ligand, load_molecule

SCHEMA_VERSION = "effdock.guidance_charge_benchmark_characterization.v1"
BASELINE_TERMS = ("hydrophobic", "hydrogen_bond")
AUGMENTED_TERMS = (*BASELINE_TERMS, "screened_formal_charge")


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _list_sha256(values: list[str]) -> str:
    return sha256(("".join(f"{value}\n" for value in values)).encode()).hexdigest()


def _load_reference(path: Path) -> Chem.Mol:
    mol, used_fallback, sanitize_ok = load_molecule(path)
    if mol is None or used_fallback or not sanitize_ok:
        raise ValueError(f"full-sanitize SDF load failed: {path}")
    return mol


def _coords(mol: Chem.Mol) -> torch.Tensor:
    return torch.tensor(mol.GetConformer().GetPositions(), dtype=torch.float64)


def _fragment_id(mol: Chem.Mol) -> torch.Tensor:
    ligand = featurize_ligand(mol)
    if ligand is None:
        raise ValueError("ligand featurization failed")
    fragments = decompose_fragments(mol, ligand["atom_coords"])
    if fragments is None:
        raise ValueError("ligand fragmentation failed")
    return fragments["fragment_id"]


def _formal_charge_summary(mol: Chem.Mol) -> dict[str, int | bool]:
    atom_charges = [int(atom.GetFormalCharge()) for atom in mol.GetAtoms()]
    typing = type_ligand_interactions(mol)
    site_charges = typing["charge_site_charge"]
    return {
        "net_formal_charge_e": sum(atom_charges),
        "nonzero_formal_charge_atoms": sum(charge != 0 for charge in atom_charges),
        "formal_charge_sites": int(site_charges.numel()),
        "has_nonzero_formal_charge_site": any(charge != 0 for charge in atom_charges),
    }


def _bond_signature(mol: Chem.Mol, atom_map: dict[int, int] | None = None) -> tuple[tuple, ...]:
    atom_map = atom_map or {index: index for index in range(mol.GetNumAtoms())}
    return tuple(
        sorted(
            (
                min(atom_map[bond.GetBeginAtomIdx()], atom_map[bond.GetEndAtomIdx()]),
                max(atom_map[bond.GetBeginAtomIdx()], atom_map[bond.GetEndAtomIdx()]),
                str(bond.GetBondType()),
                bool(bond.GetIsAromatic()),
            )
            for bond in mol.GetBonds()
        )
    )


def _pose_coords_in_reference_order(reference: Chem.Mol, pose: Chem.Mol) -> torch.Tensor:
    dock_indices, ref_indices, method = match_atoms(reference, pose)
    if (
        method != "strict"
        or len(dock_indices) != reference.GetNumAtoms()
        or pose.GetNumAtoms() != reference.GetNumAtoms()
        or pose.GetNumBonds() != reference.GetNumBonds()
    ):
        raise ValueError(
            "candidate-to-crystal mapping must be a full strict match; "
            f"got {method} with {len(dock_indices)}/{reference.GetNumAtoms()} atoms"
        )
    atom_map = dict(zip(dock_indices, ref_indices, strict=True))
    for dock_index, ref_index in atom_map.items():
        dock_atom = pose.GetAtomWithIdx(dock_index)
        ref_atom = reference.GetAtomWithIdx(ref_index)
        dock_identity = (
            dock_atom.GetAtomicNum(),
            dock_atom.GetFormalCharge(),
            bool(dock_atom.GetIsAromatic()),
        )
        ref_identity = (
            ref_atom.GetAtomicNum(),
            ref_atom.GetFormalCharge(),
            bool(ref_atom.GetIsAromatic()),
        )
        if dock_identity != ref_identity:
            raise ValueError("mapped candidate atom identity differs from crystal ligand")
    if _bond_signature(pose, atom_map) != _bond_signature(reference):
        raise ValueError("mapped candidate bond graph differs from crystal ligand")
    pose_coords = _coords(pose)
    reordered = torch.empty_like(pose_coords)
    for dock_index, ref_index in zip(dock_indices, ref_indices, strict=True):
        reordered[ref_index] = pose_coords[dock_index]
    return reordered


def _load_pose_sdf(path: Path) -> tuple[list[Chem.Mol], list[float]]:
    supplier = Chem.SDMolSupplier(str(path), removeHs=True, sanitize=True)
    poses: list[Chem.Mol] = []
    rmsds: list[float] = []
    for index, mol in enumerate(supplier):
        if mol is None:
            raise ValueError(f"failed to sanitize candidate {index} from {path}")
        if not mol.HasProp("rmsd"):
            raise ValueError(f"candidate {index} has no frozen rmsd property: {path}")
        poses.append(mol)
        rmsds.append(float(mol.GetProp("rmsd")))
    if not poses:
        raise ValueError(f"candidate SDF is empty: {path}")
    return poses, rmsds


def _dataset_ids(root: Path, ids_path: Path | None) -> list[str]:
    if ids_path is not None:
        ids = [line.strip() for line in ids_path.read_text().splitlines() if line.strip()]
    else:
        ids = [
            path.name
            for path in root.iterdir()
            if path.is_dir()
            and (path / f"{path.name}_ligand.sdf").is_file()
            and (path / f"{path.name}_protein.pdb").is_file()
        ]
    return sorted(ids)


def _dataset_inventory(
    *,
    name: str,
    root: Path,
    ids_path: Path | None,
    protein_cutoff: float,
) -> dict[str, object]:
    ids = _dataset_ids(root, ids_path)
    eligible: list[dict[str, object]] = []
    sanitize_count = 0
    for complex_id in ids:
        complex_dir = root / complex_id
        ligand_path = complex_dir / f"{complex_id}_ligand.sdf"
        protein_path = complex_dir / f"{complex_id}_protein.pdb"
        mol = _load_reference(ligand_path)
        sanitize_count += 1
        charge = _formal_charge_summary(mol)
        if not charge["has_nonzero_formal_charge_site"]:
            continue
        row: dict[str, object] = {
            "complex_id": complex_id,
            **charge,
            "ligand_sha256": _sha256(ligand_path),
            "protein_sha256": _sha256(protein_path),
        }
        try:
            system = build_physical_system(
                mol,
                protein_path,
                fragment_id=_fragment_id(mol),
                near_coords=_coords(mol),
                protein_cutoff=protein_cutoff,
            ).to(torch.device("cpu"), torch.float64)
        except UnsupportedPhysicalChemistryError as exc:
            row["strict_chemistry_supported"] = False
            row["strict_failure"] = exc.as_dict()
        else:
            topology = system.interaction_topology
            if topology is None:
                raise AssertionError("supported system has no interaction topology")
            row["strict_chemistry_supported"] = True
            row["strict_typing_counts"] = topology.term_counts()
        eligible.append(row)

    supported_ids = [
        str(row["complex_id"])
        for row in eligible
        if bool(row["strict_chemistry_supported"])
    ]
    zwitterion_count = sum(
        int(row["net_formal_charge_e"]) == 0 for row in eligible
    )
    return {
        "dataset": name,
        "root": str(root.resolve()),
        "ids_path": None if ids_path is None else str(ids_path.resolve()),
        "ids_sha256": _list_sha256(ids),
        "complexes": len(ids),
        "full_sanitize_ok": sanitize_count,
        "formal_charge_site_bearing": len(eligible),
        "site_bearing_net_neutral": zwitterion_count,
        "strict_chemistry_supported": len(supported_ids),
        "strict_supported_ids": supported_ids,
        "strict_supported_ids_sha256": _list_sha256(supported_ids),
        "site_bearing_complexes": eligible,
    }


def _minimum_margin(energies: list[float], correct: list[bool]) -> float | None:
    correct_energy = [value for value, is_correct in zip(energies, correct, strict=True) if is_correct]
    wrong_energy = [value for value, is_correct in zip(energies, correct, strict=True) if not is_correct]
    if not correct_energy or not wrong_energy:
        return None
    return min(wrong_energy) - min(correct_energy)


def _force_stats(
    coords: torch.Tensor,
    system,
    config: InteractionEnergyConfig,
) -> dict[str, float]:
    work = coords.detach().clone().requires_grad_(True)
    value = interaction_energy(work, system, config)["interaction_screened_formal_charge"]
    force = -torch.autograd.grad(value, work)[0]
    norms = force.norm(dim=-1)
    return {
        "max_atom_force_kcal_mol_per_angstrom": float(norms.max()),
        "rms_atom_force_kcal_mol_per_angstrom": float(norms.square().mean().sqrt()),
    }


def _score_complex(
    *,
    complex_id: str,
    root: Path,
    candidate_root: Path,
    protein_cutoff: float,
    rmsd_threshold: float,
) -> dict[str, object]:
    complex_dir = root / complex_id
    ligand_path = complex_dir / f"{complex_id}_ligand.sdf"
    protein_path = complex_dir / f"{complex_id}_protein.pdb"
    candidate_path = candidate_root / complex_id.lower() / "all_poses.sdf"
    reference = _load_reference(ligand_path)
    poses, rmsds = _load_pose_sdf(candidate_path)
    candidate_coords: list[torch.Tensor] = []
    for pose_index, pose in enumerate(poses):
        try:
            candidate_coords.append(_pose_coords_in_reference_order(reference, pose))
        except ValueError as exc:
            raise ValueError(
                f"{complex_id} candidate {pose_index} failed graph/mapping validation: {exc}"
            ) from exc
    crystal_coords = _coords(reference)
    system = build_physical_system(
        reference,
        protein_path,
        fragment_id=_fragment_id(reference),
        near_coords=torch.cat([crystal_coords, *candidate_coords], dim=0),
        protein_cutoff=protein_cutoff,
    ).to(torch.device("cpu"), torch.float64)

    baseline_config = InteractionEnergyConfig(active_terms=BASELINE_TERMS)
    augmented_config = InteractionEnergyConfig(active_terms=AUGMENTED_TERMS)
    charge_config = InteractionEnergyConfig(active_terms=("screened_formal_charge",))
    batch = torch.stack(candidate_coords)
    baseline_tensor = interaction_energy(batch, system, baseline_config)["total"]
    augmented_components = interaction_energy(batch, system, augmented_config)
    augmented_tensor = augmented_components["total"]
    charge_tensor = augmented_components["interaction_screened_formal_charge"]
    baseline = [float(value) for value in baseline_tensor]
    augmented = [float(value) for value in augmented_tensor]
    charge = [float(value) for value in charge_tensor]
    correct = [rmsd < rmsd_threshold for rmsd in rmsds]
    baseline_top = int(torch.argmin(baseline_tensor))
    augmented_top = int(torch.argmin(augmented_tensor))
    baseline_margin = _minimum_margin(baseline, correct)
    augmented_margin = _minimum_margin(augmented, correct)

    crystal_baseline = float(
        interaction_energy(crystal_coords, system, baseline_config)["total"]
    )
    crystal_augmented_components = interaction_energy(
        crystal_coords,
        system,
        augmented_config,
    )
    crystal_augmented = float(crystal_augmented_components["total"])
    crystal_charge = float(
        crystal_augmented_components["interaction_screened_formal_charge"]
    )
    wrong_indices = [index for index, is_correct in enumerate(correct) if not is_correct]
    crystal_margin_baseline = (
        min(baseline[index] for index in wrong_indices) - crystal_baseline
        if wrong_indices
        else None
    )
    crystal_margin_augmented = (
        min(augmented[index] for index in wrong_indices) - crystal_augmented
        if wrong_indices
        else None
    )
    contacts = interaction_contact_stats(crystal_coords, system, augmented_config)[
        "screened_formal_charge"
    ]

    return {
        "complex_id": complex_id,
        "inputs": {
            "ligand": str(ligand_path.resolve()),
            "ligand_sha256": _sha256(ligand_path),
            "protein": str(protein_path.resolve()),
            "protein_sha256": _sha256(protein_path),
            "candidates": str(candidate_path.resolve()),
            "candidates_sha256": _sha256(candidate_path),
            "candidate_count": len(poses),
            "mapping": "full_strict_substructure_and_mapped_static_graph",
        },
        "crystal": {
            "baseline_energy_kcal_mol": crystal_baseline,
            "charge_energy_kcal_mol": crystal_charge,
            "augmented_energy_kcal_mol": crystal_augmented,
            "charge_contacts": contacts,
            "charge_force": _force_stats(crystal_coords, system, charge_config),
        },
        "candidate_summary": {
            "rmsd_threshold_angstrom": rmsd_threshold,
            "correct_candidates": sum(correct),
            "baseline_top_index": baseline_top,
            "baseline_top_rmsd_angstrom": rmsds[baseline_top],
            "augmented_top_index": augmented_top,
            "augmented_top_rmsd_angstrom": rmsds[augmented_top],
            "top_changed": baseline_top != augmented_top,
            "charge_energy_range_kcal_mol": [min(charge), max(charge)],
            "baseline_margin_kcal_mol": baseline_margin,
            "augmented_margin_kcal_mol": augmented_margin,
            "delta_margin_kcal_mol": (
                None
                if baseline_margin is None or augmented_margin is None
                else augmented_margin - baseline_margin
            ),
            "crystal_vs_wrong_baseline_margin_kcal_mol": crystal_margin_baseline,
            "crystal_vs_wrong_augmented_margin_kcal_mol": crystal_margin_augmented,
            "crystal_vs_wrong_delta_margin_kcal_mol": (
                None
                if crystal_margin_baseline is None or crystal_margin_augmented is None
                else crystal_margin_augmented - crystal_margin_baseline
            ),
        },
        "poses": [
            {
                "index": index,
                "rmsd_angstrom": rmsd,
                "correct": is_correct,
                "baseline_energy_kcal_mol": baseline_energy,
                "charge_energy_kcal_mol": charge_energy,
                "augmented_energy_kcal_mol": augmented_energy,
            }
            for index, (
                rmsd,
                is_correct,
                baseline_energy,
                charge_energy,
                augmented_energy,
            ) in enumerate(
                zip(rmsds, correct, baseline, charge, augmented, strict=True)
            )
        ],
    }


def _characterization_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    candidate_deltas = [
        float(row["candidate_summary"]["delta_margin_kcal_mol"])
        for row in rows
        if row["candidate_summary"]["delta_margin_kcal_mol"] is not None
    ]
    return {
        "complexes_scored": len(rows),
        "candidate_sets_with_both_correct_and_wrong": len(candidate_deltas),
        "median_candidate_delta_margin_kcal_mol": (
            statistics.median(candidate_deltas) if candidate_deltas else None
        ),
        "candidate_margin_nonloss_fraction": (
            sum(value >= 0 for value in candidate_deltas) / len(candidate_deltas)
            if candidate_deltas
            else None
        ),
        "top_pose_unchanged": sum(
            not bool(row["candidate_summary"]["top_changed"]) for row in rows
        ),
        "top_pose_changed": sum(
            bool(row["candidate_summary"]["top_changed"]) for row in rows
        ),
        "interpretation": (
            "report_only_external_characterization; cannot admit, tune, remove, "
            "or activate the term"
        ),
    }


def build_report(args: argparse.Namespace) -> dict[str, object]:
    posebusters = _dataset_inventory(
        name="posebusters_v2",
        root=args.posebusters_root,
        ids_path=args.posebusters_ids,
        protein_cutoff=args.protein_cutoff,
    )
    astex = _dataset_inventory(
        name="astex_diverse",
        root=args.astex_root,
        ids_path=args.astex_ids,
        protein_cutoff=args.protein_cutoff,
    )
    score_rows = [
        _score_complex(
            complex_id=complex_id,
            root=args.posebusters_root,
            candidate_root=args.posebusters_candidates,
            protein_cutoff=args.protein_cutoff,
            rmsd_threshold=args.rmsd_threshold,
        )
        for complex_id in posebusters["strict_supported_ids"]
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(UTC).isoformat(),
        "protocol_id": "EFFDOCK-INTERACTION-GUIDANCE-V1",
        "stage": "1B_external_fixed_coordinate_characterization",
        "status": "complete_report_only",
        "claim_boundary": {
            "external_benchmarks_consumed_for_tuning": False,
            "allowed": [
                "coverage",
                "strict chemistry support",
                "fixed-coordinate energies and forces",
                "descriptive pose-margin and ranking changes",
            ],
            "forbidden": [
                "formula selection",
                "constant or scale fitting",
                "schedule or force-cap selection",
                "term retention or removal",
                "sampler activation",
            ],
            "production_activation_gate": "internal held-out validation only",
        },
        "selection": {
            "eligibility": "any ligand atom formal charge != 0",
            "net_molecular_charge_used": False,
            "strict_support_selected_before_energy_or_rmsd_inspection": True,
            "protein_cutoff_angstrom": args.protein_cutoff,
        },
        "runtime": {
            "device": "cpu",
            "dtype": "float64",
            "torch": torch.__version__,
            "rdkit": rdBase.rdkitVersion,
            "command": " ".join(sys.argv),
        },
        "inventory": {
            "posebusters_v2": posebusters,
            "astex_diverse": astex,
        },
        "fixed_candidate_characterization": {
            "candidate_root": str(args.posebusters_candidates.resolve()),
            "baseline_terms": list(BASELINE_TERMS),
            "augmented_terms": list(AUGMENTED_TERMS),
            "summary": _characterization_summary(score_rows),
            "complexes": score_rows,
        },
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--posebusters-root",
        type=Path,
        default=Path("data/external_benchmarks/data/posebusters_benchmark_set"),
    )
    parser.add_argument(
        "--posebusters-ids",
        type=Path,
        default=Path("data/external_test/posebusters_v2_ids.txt"),
    )
    parser.add_argument(
        "--astex-root",
        type=Path,
        default=Path("data/external_benchmarks/data/astex_diverse_set"),
    )
    parser.add_argument("--astex-ids", type=Path)
    parser.add_argument(
        "--posebusters-candidates",
        type=Path,
        default=Path(
            "outputs/external_benchmarks/"
            "posebusters_latest_n40_s20_confidence_v1_conf200000_rerank_poses"
        ),
    )
    parser.add_argument("--protein-cutoff", type=float, default=10.0)
    parser.add_argument("--rmsd-threshold", type=float, default=2.0)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    report = build_report(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    summary = report["fixed_candidate_characterization"]["summary"]
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
