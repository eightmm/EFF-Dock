#!/usr/bin/env python3
"""Run PoseBench's pinned AutoDock Vina path on an explicit CSV cohort."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
from pathlib import Path

VINA_RESULT_RE = re.compile(
    r"^REMARK VINA RESULT:\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--posebench-root", type=Path, required=True)
    parser.add_argument("--posebench-python", type=Path, required=True)
    parser.add_argument("--adfr-python", type=Path, required=True)
    parser.add_argument("--prepare-receptor", type=Path, required=True)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Nonzero Vina seed; Vina interprets zero as request for a random seed.",
    )
    parser.add_argument("--exhaustiveness", type=int, default=32)
    parser.add_argument("--num-modes", type=int, default=40)
    parser.add_argument(
        "--cpu-workers",
        type=int,
        default=1,
        help="Vina worker count; must not exceed the scheduler allocation.",
    )
    parser.add_argument("--fail-on-incomplete", action="store_true")
    return parser.parse_args()


def parse_vina_results(pdbqt_path: Path) -> list[tuple[float, float, float]]:
    """Return Vina affinity/lower-bound/upper-bound values in MODEL order."""
    values: list[tuple[float, float, float]] = []
    for line in pdbqt_path.read_text(errors="replace").splitlines():
        match = VINA_RESULT_RE.match(line)
        if match:
            values.append(tuple(float(value) for value in match.groups()))
    return values


def write_vina_ensemble(raw_pdbqt: Path, topology_sdf: Path) -> int:
    """Replace PoseBench's single-mode SDF with every raw Vina conformer.

    PoseBench's upstream writer intentionally keeps only conformer zero.  The
    raw PDBQT remains the authoritative ranked artifact, so this conversion
    reuses the upstream SDF only as a bond-order/topology template and writes
    one SDF record per raw MODEL in the original Vina score order.
    """
    from posebench_vina_compat import install_rdkit_six_compat
    from rdkit import Chem
    from rdkit.Chem import AllChem

    install_rdkit_six_compat()
    from meeko import PDBQTMolecule, RDKitMolCreate

    templates = [
        molecule
        for molecule in Chem.SDMolSupplier(str(topology_sdf), removeHs=True)
        if molecule is not None
    ]
    if len(templates) != 1:
        raise ValueError(f"Expected one topology record in {topology_sdf}, got {len(templates)}")
    template = templates[0]
    raw_molecules = RDKitMolCreate.from_pdbqt_mol(
        PDBQTMolecule.from_file(str(raw_pdbqt), skip_typing=True)
    )
    if len(raw_molecules) != 1 or raw_molecules[0] is None:
        raise ValueError(
            f"Expected one ligand molecule in {raw_pdbqt}, got {len(raw_molecules)}"
        )
    raw = Chem.RemoveHs(raw_molecules[0])
    raw = AllChem.AssignBondOrdersFromTemplate(template, raw)
    scores = parse_vina_results(raw_pdbqt)
    conformer_count = raw.GetNumConformers()
    if len(scores) != conformer_count:
        raise ValueError(
            f"Vina score/conformer mismatch for {raw_pdbqt}: "
            f"{len(scores)} != {conformer_count}"
        )

    temporary = topology_sdf.with_suffix(".ensemble.tmp.sdf")
    writer = Chem.SDWriter(str(temporary))
    try:
        for mode_index, (conformer, score) in enumerate(
            zip(raw.GetConformers(), scores), start=1
        ):
            pose = Chem.Mol(raw)
            pose.RemoveAllConformers()
            pose.AddConformer(conformer, assignId=True)
            pose.SetIntProp("vina_mode", mode_index)
            pose.SetDoubleProp("vina_affinity", score[0])
            pose.SetDoubleProp("vina_rmsd_lb", score[1])
            pose.SetDoubleProp("vina_rmsd_ub", score[2])
            writer.write(pose)
    finally:
        writer.close()
    if not temporary.is_file() or temporary.stat().st_size == 0:
        raise RuntimeError(f"Failed to write Vina ensemble: {temporary}")
    temporary.replace(topology_sdf)
    return conformer_count


def main() -> None:
    args = parse_args()
    posebench_root = args.posebench_root.resolve()
    posebench_python = args.posebench_python.resolve()
    adfr_python = args.adfr_python.resolve()
    prepare_receptor = args.prepare_receptor.resolve()
    output_dir = args.output_dir.resolve()
    predictions = output_dir / "predictions"
    predictions.mkdir(parents=True, exist_ok=True)
    with args.input_csv.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No Vina targets in {args.input_csv}")
    if args.seed == 0:
        raise ValueError("Vina seed 0 is nondeterministic; use a nonzero seed")
    if args.cpu_workers < 1:
        raise ValueError("Vina cpu_workers must be at least one")

    coverage: dict[str, dict[str, object]] = {}
    environment = os.environ.copy()
    environment["PROJECT_ROOT"] = str(posebench_root)
    environment["PYTHONHASHSEED"] = str(args.seed)
    compatibility_entry = Path(__file__).with_name("posebench_vina_compat.py").resolve()
    for row in rows:
        target_id = row["complex_name"]
        target_sdf = predictions / target_id / f"{target_id}.sdf"
        return_code = 0
        if not target_sdf.is_file():
            command = [
                str(posebench_python),
                str(compatibility_entry),
                str(posebench_root / "posebench" / "models" / "vina_inference.py"),
                f"dataset={args.dataset}",
                "method=diffdock",
                f"python2_exec_path={adfr_python}",
                f"prepare_receptor_script_path={prepare_receptor}",
                f"output_dir={predictions}",
                f"cpu={args.cpu_workers}",
                f"seed={args.seed}",
                f"exhaustiveness={args.exhaustiveness}",
                f"num_modes={args.num_modes}",
                f"protein_filepath={Path(row['holo_protein']).resolve()}",
                f"ligand_filepaths=[{Path(row['reference_ligand']).resolve()}]",
                f"apo_protein_filepath={Path(row['predicted_receptor']).resolve()}",
                f"input_id={target_id}",
            ]
            completed = subprocess.run(
                command,
                cwd=posebench_root,
                env=environment,
                check=False,
            )
            return_code = completed.returncode
        raw_outputs = sorted(target_sdf.parent.glob(f"{target_id}_*_group_0.pdbqt"))
        if return_code == 0 and target_sdf.is_file() and len(raw_outputs) == 1:
            with target_sdf.open(errors="replace") as handle:
                existing_pose_count = sum(line.strip() == "$$$$" for line in handle)
            raw_pose_count = len(parse_vina_results(raw_outputs[0]))
            if existing_pose_count != raw_pose_count:
                write_vina_ensemble(raw_outputs[0], target_sdf)
        pose_count = 0
        if target_sdf.is_file():
            with target_sdf.open(errors="replace") as handle:
                pose_count = sum(line.strip() == "$$$$" for line in handle)
            if pose_count == 0 and target_sdf.stat().st_size:
                pose_count = 1
        coverage[target_id] = {
            "pose_file_count": int(target_sdf.is_file()),
            "pose_count": pose_count,
            "return_code": return_code,
            "prediction": str(target_sdf),
            "raw_pdbqt": str(raw_outputs[0]) if len(raw_outputs) == 1 else "",
        }

    expected = len(rows)
    any_pose = sum(row["pose_count"] > 0 for row in coverage.values())
    complete = sum(row["pose_count"] == args.num_modes for row in coverage.values())
    summary = {
        "schema_version": 1,
        "dataset": args.dataset,
        "expected_targets": expected,
        "targets_with_any_pose": any_pose,
        "targets_with_expected_pose_count": complete,
        "vina_num_modes_requested": args.num_modes,
        "vina_mode_policy": (
            "up to num_modes within pinned Vina's default 3 kcal/mol energy range; "
            "actual raw MODEL count is authoritative"
        ),
        "seed": args.seed,
        "seed_policy": "explicit nonzero Vina seed; zero is forbidden as nondeterministic",
        "exhaustiveness": args.exhaustiveness,
        "cpu_workers": args.cpu_workers,
        "site_information": "reference-ligand-supplied pocket",
        "docking_receptor": "holo-aligned predicted receptor",
        "receptor_writer": environment.get(
            "EFFDOCK_VINA_RECEPTOR_WRITER", "posebench_adfr_prepare_receptor4"
        ),
        "coverage": coverage,
    }
    (output_dir / "coverage.json").write_text(json.dumps(summary, indent=2) + "\n")
    if args.fail_on_incomplete and complete != expected:
        raise SystemExit(f"Incomplete Vina coverage: {complete}/{expected}")


if __name__ == "__main__":
    main()
