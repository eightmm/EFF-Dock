#!/usr/bin/env python3
"""Report a paired chemical-constraint ODE guidance smoke."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from types import SimpleNamespace

import torch
from rdkit import Chem

from effdock.guidance.chemical import (
    ChemicalConstraintEnergyConfig,
    chemical_constraint_diagnostics,
    chemical_constraint_energy,
)
from effdock.guidance.topology import build_physical_topology
from effdock.preprocess.fragments import decompose_fragments
from effdock.workflows.benchmark_inputs import load_benchmark_inputs, load_benchmark_ligand
from effdock.workflows.stereo_filter import explicit_stereo_constraints, stereo_compatibility

COMPLEX_IDS = ("7sdd_4ip", "6xht_v2v", "8f4j_pho")


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--run-prefix", required=True)
    parser.add_argument("--arms", required=True, help="Comma-separated arm names; first is baseline")
    parser.add_argument(
        "--benchmark-input-manifest",
        type=Path,
        default=Path("docs/GUIDANCE_BUDGET1000_FULL_INPUTS.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _one_row(path: Path) -> dict[str, str]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError(f"{path}: expected one row, got {len(rows)}")
    return rows[0]


def _pose_coords(path: Path) -> tuple[list[Chem.Mol], torch.Tensor]:
    poses = [mol for mol in Chem.SDMolSupplier(str(path), removeHs=False) if mol]
    if len(poses) != 100:
        raise ValueError(f"{path}: expected 100 poses, got {len(poses)}")
    coords = torch.stack(
        [
            torch.as_tensor(mol.GetConformer().GetPositions(), dtype=torch.float64)
            for mol in poses
        ]
    )
    return poses, coords


def main() -> None:
    args = _args()
    arms = tuple(value.strip() for value in args.arms.split(",") if value.strip())
    if len(arms) < 2 or len(set(arms)) != len(arms):
        raise ValueError("--arms must contain at least two unique names")
    ligand_mapping, mapping_identity = load_benchmark_inputs(
        "posebusters",
        Path("data/external_test"),
        args.benchmark_input_manifest,
    )
    records: list[dict[str, object]] = []
    prior_hashes: dict[str, set[str]] = {complex_id: set() for complex_id in COMPLEX_IDS}
    baseline_coords: dict[str, torch.Tensor] = {}

    for complex_id in COMPLEX_IDS:
        mol_input, _ = load_benchmark_ligand(ligand_mapping[complex_id], random_seed=0)
        input_coords = torch.as_tensor(
            mol_input.GetConformer().GetPositions(),
            dtype=torch.float64,
        )
        fragments = decompose_fragments(mol_input, input_coords)
        if fragments is None:
            raise RuntimeError(f"{complex_id}: fragment decomposition failed")
        topology = build_physical_topology(mol_input, fragments["fragment_id"]).to(
            torch.device("cpu"),
            torch.float64,
        )
        system = SimpleNamespace(topology=topology)
        expected = explicit_stereo_constraints(mol_input)

        for arm_index, arm in enumerate(arms):
            run_name = f"{args.run_prefix}-{complex_id}-{arm}"
            row = _one_row(args.input_dir / f"{run_name}.csv")
            prior_hashes[complex_id].add(row["prior_pool_sha256"])
            pose_path = (
                args.input_dir
                / "poses"
                / run_name
                / "posebusters"
                / "all_poses"
                / f"{complex_id}.sdf"
            )
            poses, pose_coords = _pose_coords(pose_path)
            if arm_index == 0:
                baseline_coords[complex_id] = pose_coords
            paired_delta = (
                (pose_coords - baseline_coords[complex_id])
                .square()
                .sum(dim=-1)
                .mean(dim=-1)
                .sqrt()
            )
            stereo_valid = torch.tensor(
                [stereo_compatibility(expected, mol).valid for mol in poses],
                dtype=torch.bool,
            )
            diagnostics = chemical_constraint_diagnostics(pose_coords, system)
            barrier_valid = (
                diagnostics["tetrahedral_inversion_count"]
                + diagnostics["double_bond_inversion_count"]
            ).eq(0)
            barrier = chemical_constraint_energy(
                pose_coords,
                system,
                ChemicalConstraintEnergyConfig(scale=1.0),
            )["total"]
            rmsds = torch.tensor(json.loads(row["candidate_rmsds_json"]), dtype=torch.float64)
            fast_valid = torch.tensor(
                json.loads(row["candidate_fast_valid_json"]), dtype=torch.bool
            )
            stereo_valid_rmsds = rmsds[stereo_valid]
            invalid_energy = barrier[~stereo_valid]
            metadata = json.loads(row["guidance_metadata_json"])
            coefficients = metadata["runtime_term_coefficients"]
            records.append(
                {
                    "complex_id": complex_id,
                    "arm": arm,
                    "chemical_constraint_strength": coefficients.get(
                        "chemical_constraint_strength", 0.0
                    ),
                    "prior_pool_sha256": row["prior_pool_sha256"],
                    "stereo_term_counts": topology.stereo_term_counts(),
                    "stereo_valid_count": int(stereo_valid.sum()),
                    "barrier_filter_mask_agreement_count": int(
                        barrier_valid.eq(stereo_valid).sum()
                    ),
                    "stereo_barrier_median_valid": (
                        float(barrier[stereo_valid].median())
                        if bool(stereo_valid.any())
                        else None
                    ),
                    "stereo_barrier_median_invalid": (
                        float(invalid_energy.median()) if bool(invalid_energy.numel()) else None
                    ),
                    "rmsd_lt2_count": int(rmsds.lt(2.0).sum()),
                    "stereo_valid_rmsd_lt2_count": int((stereo_valid & rmsds.lt(2.0)).sum()),
                    "fast_valid_count": int(fast_valid.sum()),
                    "fast_and_stereo_valid_count": int((fast_valid & stereo_valid).sum()),
                    "oracle_rmsd": float(rmsds.min()),
                    "stereo_valid_oracle_rmsd": (
                        float(stereo_valid_rmsds.min())
                        if bool(stereo_valid_rmsds.numel())
                        else math.inf
                    ),
                    "mean_rmsd": float(rmsds.mean()),
                    "paired_coordinate_rmsd_mean": float(paired_delta.mean()),
                    "paired_coordinate_rmsd_max": float(paired_delta.max()),
                    "chemical_constraint_pose_applications": int(
                        row["guidance_direct_chemical_constraint_pose_applied"]
                    ),
                    "chemical_constraint_cap_triggers": int(
                        row["guidance_direct_chemical_constraint_cap_trigger_count"]
                    ),
                    "chemical_constraint_max_atom_displacement": float(
                        row[
                            "guidance_direct_chemical_constraint_max_estimated_atom_displacement"
                        ]
                    ),
                }
            )

    unpaired = {
        complex_id: sorted(values)
        for complex_id, values in prior_hashes.items()
        if len(values) != 1
    }
    if unpaired:
        raise RuntimeError(f"prior pools differ across arms: {unpaired}")
    payload = {
        "schema_version": "effdock.chemical_constraint_guidance_smoke.v1",
        "status": "complete",
        "scope": "three preselected mechanistic failure cases; not a benchmark estimate",
        "mapping_identity_sha256": mapping_identity["sha256"],
        "paired_prior_verified": True,
        "baseline_arm": arms[0],
        "arms": list(arms),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
