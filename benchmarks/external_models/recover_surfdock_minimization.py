"""Minimize saved SurfDock poses with the upstream fixed-receptor force fields."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from functools import cache
from io import StringIO
from pathlib import Path

import numpy as np
from rdkit import Chem

FORCE_FIELDS = ("openff-2.0.0", "gaff-2.11")
ATOM_LINE = re.compile(
    r"^\s*([+-]?\d+\.\d{4})\s*([+-]?\d+\.\d{4})\s*([+-]?\d+\.\d{4})( +[A-Za-z].*)$"
)


def read_saved_pose(path: Path) -> Chem.Mol:
    """Read one SDF, recovering extended V2000 coordinate fields losslessly."""
    with path.open("rb") as stream:
        molecules = list(Chem.ForwardSDMolSupplier(stream, removeHs=False))
    if len(molecules) != 1:
        raise ValueError("Expected one SDF record")
    if molecules[0] is not None:
        return Chem.RemoveAllHs(molecules[0])
    lines = path.read_text().splitlines()
    if len(lines) < 5 or "V2000" not in lines[3]:
        raise ValueError("Unreadable SDF")
    count = int(lines[3][:3])
    coords = []
    for i in range(4, 4 + count):
        match = ATOM_LINE.fullmatch(lines[i])
        if match is None:
            raise ValueError(f"Unreadable V2000 atom line {i + 1}")
        coords.append(tuple(map(float, match.groups()[:3])))
        lines[i] = f"{0:10.4f}" * 3 + match[4]
    end = lines.index("M  END", 4 + count) + 1
    mol = Chem.MolFromMolBlock("\n".join(lines[:end]) + "\n", removeHs=False)
    if mol is None or mol.GetNumAtoms() != count:
        raise ValueError("V2000 atom graph did not survive decoding")
    for i, xyz in enumerate(coords):
        mol.GetConformer().SetAtomPosition(i, xyz)
    return Chem.RemoveAllHs(mol)


def validate_pose(mol: Chem.Mol) -> None:
    if mol.GetNumAtoms() < 2 or mol.GetNumConformers() != 1:
        raise ValueError("Expected ligand heavy atoms with one conformer")
    coords = mol.GetConformer().GetPositions()
    if not np.isfinite(coords).all():
        raise ValueError("Nonfinite ligand coordinates")
    if np.abs(coords).max() > 100_000:
        raise ValueError("Ligand coordinates are outside the receptor PDB frame")
    for bond in mol.GetBonds():
        delta = coords[bond.GetBeginAtomIdx()] - coords[bond.GetEndAtomIdx()]
        if np.linalg.norm(delta) < 0.1:
            raise ValueError("Collapsed ligand bond below 0.1 Å")


def ordered_graph(mol: Chem.Mol) -> tuple:
    atoms = tuple(
        (atom.GetAtomicNum(), atom.GetFormalCharge(), atom.GetIsotope()) for atom in mol.GetAtoms()
    )
    bonds = tuple(sorted(
        (min(b.GetBeginAtomIdx(), b.GetEndAtomIdx()), max(b.GetBeginAtomIdx(), b.GetEndAtomIdx()),
         b.GetBondTypeAsDouble())
        for b in mol.GetBonds()
    ))
    return atoms, bonds


@cache
def fixed_protein(path: Path):
    from force_optimize import minimize_utils as ff

    alterations = {}
    structure = ff.PdbStructure(StringIO(ff.fix_pdb(str(path), alterations)))
    ff.clean_structure(structure, alterations)
    return ff.PDBFile(structure)


@cache
def generator(forcefield: str):
    from force_optimize import minimize_utils as ff

    return ff.GetFFGenerator(
        small_molecule_forcefield=forcefield, ignoreExternalBonds=forcefield == "gaff-2.11",
    )


def minimize_pose(mol: Chem.Mol, receptor: Path, forcefield: str) -> tuple[Chem.Mol, float, float]:
    """Apply SurfDock's OpenMM protocol on CPU and report total energy in kJ/mol."""
    from force_optimize import minimize_utils as ff

    fixed = fixed_protein(receptor)
    modeller = ff.Modeller(fixed.topology, fixed.positions)
    receptor_atoms = modeller.topology.getNumAtoms()
    ligand = ff.Molecule.from_rdkit(mol, allow_undefined_stereo=True)
    ligand.assign_partial_charges(partial_charge_method="gasteiger")
    ligand_topology = ligand.to_topology()
    modeller.add(ligand_topology.to_openmm(), ligand_topology.get_positions().to_openmm())
    system = generator(forcefield).create_system(modeller.topology, molecules=ligand)
    if system.getNumParticles() != receptor_atoms + ligand.n_atoms:
        raise ValueError("OpenMM particle ordering does not match the prepared complex")
    for index in range(receptor_atoms):
        system.setParticleMass(index, 0.0 * ff.unit.dalton)
    integrator = ff.LangevinIntegrator(
        300 * ff.unit.kelvin, 1 / ff.unit.picosecond, 0.002 * ff.unit.picoseconds,
    )
    simulation = ff.Simulation(
        modeller.topology, system, integrator, ff.Platform.getPlatformByName("CPU"),
    )
    simulation.context.setPositions(modeller.positions)
    units = ff.unit.kilojoule_per_mole
    energy = simulation.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(units)
    if not math.isfinite(energy):
        raise ValueError("Nonfinite starting OpenMM energy")
    simulation.minimizeEnergy()
    final = simulation.context.getState(getEnergy=True, getPositions=True)
    final_energy = final.getPotentialEnergy().value_in_unit(units)
    if not math.isfinite(final_energy) or final_energy > energy + 1e-3:
        raise ValueError(f"Invalid minimized OpenMM energy {energy} -> {final_energy}")
    ligand_positions = final.getPositions(asNumpy=True).value_in_unit(ff.unit.angstrom)[receptor_atoms:]
    rdkit_ligand = ligand.to_rdkit()
    if not np.isfinite(ligand_positions).all():
        raise ValueError("Nonfinite minimized coordinates")
    heavy_positions = [
        tuple(map(float, xyz)) for atom, xyz in zip(rdkit_ligand.GetAtoms(), ligand_positions, strict=True)
        if atom.GetAtomicNum() != 1
    ]
    if ordered_graph(Chem.RemoveAllHs(rdkit_ligand)) != ordered_graph(mol):
        raise ValueError("Minimization changed the ordered ligand graph")
    result = Chem.Mol(mol)
    for index, xyz in enumerate(heavy_positions):
        result.GetConformer().SetAtomPosition(index, xyz)
    validate_pose(result)
    return result, energy, final_energy


def recover(row: dict[str, str], output: Path, forcefields: tuple[str, ...]) -> str:
    dest = output / row["pose_id"]
    result = {**row, "status": "invalid_input"}
    try:
        mol = read_saved_pose(Path(row["pose"]))
        validate_pose(mol)
    except (IndexError, OSError, ValueError) as error:
        result["error"] = str(error)
    else:
        result["status"] = "minimization_failed"
        attempts = {}
        for forcefield in forcefields:
            try:
                pose, initial, final = minimize_pose(mol, Path(row["receptor"]), forcefield)
            except Exception as error:
                attempts[forcefield] = f"{type(error).__name__}: {error}"
                continue
            stage = dest.with_suffix(".partial.sdf")
            with Chem.SDWriter(str(stage)) as writer:
                writer.SetForceV3000(True)
                writer.write(pose)
            stage.replace(dest.with_suffix(".sdf"))
            result.update(status="minimized", forcefield=forcefield, initial_kj=initial, final_kj=final)
            break
        result["failed_attempts"] = attempts
    stage = dest.with_suffix(".partial.json")
    stage.write_text(json.dumps(result, allow_nan=False, indent=2) + "\n")
    stage.replace(dest.with_suffix(".json"))
    return result["status"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="CSV with pose_id, pose and receptor")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--forcefield", choices=("auto", *FORCE_FIELDS), default="auto")
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()
    if not 0 <= args.index < args.shards:
        parser.error("Index must be within the requested shards")
    with args.manifest.open(newline="") as file:
        reader = csv.DictReader(file)
        if not {"pose_id", "pose", "receptor"}.issubset(reader.fieldnames or ()):
            parser.error("Manifest requires pose_id, pose and receptor columns")
        rows = list(reader)
    identities = [row["pose_id"] for row in rows]
    if not rows or len(set(identities)) != len(rows) or not all(
        re.fullmatch(r"[A-Za-z0-9_-]+", name or "") for name in identities
    ):
        parser.error("Manifest must have unique, nonempty alphanumeric pose IDs")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    inputs = {Path(row[key]).resolve() for row in rows for key in ("pose", "receptor")}
    outputs = {(args.output_dir / f"{name}{suffix}").resolve()
               for name in identities for suffix in (".sdf", ".json")}
    if inputs.intersection(outputs):
        parser.error("Output must be separate from the saved poses")
    forcefields = FORCE_FIELDS if args.forcefield == "auto" else (args.forcefield,)
    for row in rows[args.index::args.shards]:
        print(row["pose_id"], recover(row, args.output_dir, forcefields), flush=True)


if __name__ == "__main__":
    main()
