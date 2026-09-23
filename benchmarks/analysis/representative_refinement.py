"""Refine the exact saved illustrative endpoint on CPU without rerunning docking."""

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "benchmarks/results/paper/trajectory"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(source_root, output):
    import torch
    from rdkit import Chem

    from effdock.guidance import InteractionEnergyConfig, build_physical_system
    from effdock.guidance.parameterization import guidance_parameter_identity
    from effdock.workflows.relax_guidance import RigidRelaxationConfig, relax_rigid_fragments

    torch.set_num_threads(1)
    trace = json.loads((DATA / "trace.json").read_text())
    entries = {r["path"]: r["sha256"] for r in trace["sources"]}
    selected = {}
    for suffix in ("_protein.pdb", "_ligand_start_conf.sdf", "/docked.sdf"):
        matches = [p for p in entries if p.endswith(suffix)]
        if len(matches) != 1:
            raise ValueError(f"Ambiguous source {suffix}")
        name = matches[0]
        path = source_root / name
        if digest(path) != entries[name]:
            raise ValueError(f"Frozen source changed: {name}")
        selected[suffix] = path
    template = Chem.SDMolSupplier(str(selected["_ligand_start_conf.sdf"]), removeHs=True)[0]
    endpoint = Chem.SDMolSupplier(str(selected["/docked.sdf"]), removeHs=True)[0]
    for mol in (template, endpoint):
        if mol is None or [a.GetSymbol() for a in mol.GetAtoms()] != trace["elements"]:
            raise ValueError("Ligand atom identity mismatch")
        bonds = sorted(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx())) for b in mol.GetBonds())
        if bonds != sorted(sorted(pair) for pair in trace["bonds"]):
            raise ValueError("Ligand topology mismatch")
    raw = np.asarray(trace["coordinates"][-1])
    if not np.allclose(endpoint.GetConformer().GetPositions(), raw, atol=1e-3, rtol=0):
        raise ValueError("Saved endpoint differs from docked SDF")
    center = torch.tensor(trace["pocket_center"], dtype=torch.float32)
    fragments = torch.tensor(trace["fragment_id"], dtype=torch.long)
    initial = torch.tensor(raw, dtype=torch.float32) - center
    system = build_physical_system(
        template,
        selected["_protein.pdb"],
        fragment_id=fragments,
        near_coords=center[None],
        protein_cutoff=18.0,
        coordinate_origin=center,
        receptor_policy="geometry_only",
    ).to(device=torch.device("cpu"), dtype=torch.float32)
    config = RigidRelaxationConfig(
        initialization_mode="model_prior",
        prior_sigma_angstrom=2.0,
        seed=42,
        max_steps=100,
        save_every=25,
        protein_shell_cutoff_angstrom=18.0,
        convergence_energy_absolute_kcal_mol=0.02,
        convergence_energy_relative=0.001,
        convergence_energy_patience=5,
        convergence_energy_min_steps=25,
    )
    print("Refining one saved endpoint on CPU", flush=True)
    # The reference argument is diagnostic only; do not report it as crystal RMSD.
    result = relax_rigid_fragments(
        initial.clone(),
        initial,
        system,
        config=config,
        mode="unified",
        pocket_center=torch.zeros(3),
        interaction_config=InteractionEnergyConfig(),
    )
    if result.status not in {"max_steps", "converged_displacement", "converged_energy_plateau"}:
        raise ValueError(f"Illustrative refinement failed: {result.status}")
    coords = torch.stack(result.frames).numpy() + center.numpy()
    if not np.isfinite(coords).all() or not np.allclose(coords[0], raw, atol=1e-4, rtol=0):
        raise ValueError("Refinement trajectory identity/finite check failed")
    for f in range(6):
        xyz = coords[:, np.asarray(trace["fragment_id"]) == f]
        dist = np.linalg.norm(xyz[:, :, None] - xyz[:, None, :], axis=-1)
        if np.max(np.abs(dist - dist[0])) > 1e-3:
            raise ValueError("Refinement changed intrafragment geometry")
    energies = [dict(step=r["step"], **r["energy_groups"]) for r in result.metrics]
    combined = np.asarray([r["combined"] for r in energies])
    if (
        not np.isfinite(combined).all()
        or (np.diff(combined) > 1e-4 + 1e-5 * np.abs(combined[:-1])).any()
    ):
        raise ValueError("Refinement energy is nonfinite or increases")
    record = dict(
        purpose="One saved N1 endpoint refined for a method illustration; no accuracy or PB-validity claim",
        trace_sha256=digest(DATA / "trace.json"),
        source_files={p: entries[p] for p in entries if source_root / p in selected.values()},
        raw_coordinates=raw.tolist(),
        coordinates=coords.tolist(),
        saved_steps=result.saved_steps,
        energies=energies,
        status=result.status,
        total_backtracks=result.total_backtracks,
        shell_envelope_valid=result.shell_envelope_valid,
        config=asdict(config),
        parameter_identity=guidance_parameter_identity(),
        diagnostic_reference="raw endpoint, not crystal",
        source_code={
            str(p.relative_to(ROOT)): digest(p)
            for p in [
                Path(__file__),
                ROOT / "src/effdock/workflows/relax_guidance.py",
                ROOT / "src/effdock/guidance/physical.py",
                ROOT / "src/effdock/guidance/interaction.py",
            ]
        },
    )
    output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    print(
        f"{result.status}; step {result.saved_steps[-1]}; energy {combined[0]:.3f} -> {combined[-1]:.3f}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DATA / "representative_refinement.json")
    args = parser.parse_args()
    run(args.source_root.resolve(), args.output)


if __name__ == "__main__":
    main()
