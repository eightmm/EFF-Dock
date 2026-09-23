"""Export an existing illustrative ODE trace; never run or interpolate the model."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from benchmarks.analysis.evidence import binding_chain_display, require


def export(root, output):
    import torch
    from rdkit import Chem

    trace = root / "outputs/figures/actual_ode_trace_1t46/results.pt"
    bundle = torch.load(trace, map_location="cpu", weights_only=True)
    identity = dict(bundle["ligand_identity"])
    source = identity.pop("source")
    digest = identity.pop("sha256")
    canonical = json.dumps(identity, separators=(",", ":"), sort_keys=True).encode()
    require(hashlib.sha256(canonical).hexdigest() == digest, "Ligand graph identity drift")
    options = bundle["options"]
    ligand_input = Path(options["ligand"])
    require(
        source["kind"] == "file"
        and hashlib.sha256(ligand_input.read_bytes()).hexdigest() == source["sha256"],
        "Ligand source drift",
    )
    require(options["num_samples"] == 1 and len(bundle["trajectories"]) == 1, "Ambiguous trace")
    trajectory = bundle["trajectories"][0]
    times = np.asarray(trajectory["traj_times"])
    centered = torch.stack(trajectory["traj"]).numpy()
    require(centered.shape == (11, 37, 3) and np.isfinite(centered).all(), "Trace shape/values")
    require(times[0] == 0 and times[-1] == 1 and (np.diff(times) > 0).all(), "Trace times")
    require(
        np.array_equal(centered[-1], bundle["poses"][0]["atom_pos_pred"].numpy()),
        "Endpoint mismatch",
    )
    center = bundle["pocket_center"].numpy()
    coordinates = centered + center[None, None, :]
    fragments = identity["fragment_id"]
    require(len(fragments) == 37 and sorted(set(fragments)) == list(range(6)), "Fragment inventory")
    for frag in range(6):
        xyz = centered[:, np.asarray(fragments) == frag]
        distances = np.linalg.norm(xyz[:, :, None] - xyz[:, None, :], axis=-1)
        require(np.allclose(distances, distances[0], atol=1e-4), "Non-rigid saved fragment")
    case = root / "data/external_benchmarks/data/astex_diverse_set/1T46_STI"
    receptor = case / "1T46_STI_protein.pdb"
    reference = case / "1T46_STI_ligand.sdf"
    reference_mol = Chem.SDMolSupplier(str(reference), removeHs=True)[0]
    require(reference_mol is not None, "Missing crystal ligand")
    endpoint = Chem.SDMolSupplier(str(trace.parent / "docked.sdf"), removeHs=True)[0]
    require(
        np.allclose(endpoint.GetConformer().GetPositions(), coordinates[-1], atol=1e-3),
        "SDF frame mismatch",
    )
    indices = [int(np.abs(times - t).argmin()) for t in (0, 0.25, 0.5, 0.75, 1)]
    result = dict(
        id="1T46_STI",
        dataset="astex",
        purpose="Existing N1 illustration, not a benchmark-selected pose",
        times=times.tolist(),
        coordinates=coordinates.tolist(),
        fragment_id=fragments,
        elements=[
            Chem.GetPeriodicTable().GetElementSymbol(a["atomic_number"]) for a in identity["atoms"]
        ],
        bonds=[b["atom_indices"] for b in identity["bonds"]],
        shown_indices=indices,
        pocket_center=center.tolist(),
        protein_display=binding_chain_display(
            receptor.read_text(), reference_mol.GetConformer().GetPositions()
        ),
        protocol={
            k: options[k]
            for k in (
                "seed",
                "num_samples",
                "num_steps",
                "time_schedule",
                "schedule_power",
                "sigma",
                "pocket_cutoff",
                "vina_guidance_scale",
                "rank_by",
            )
        },
        checkpoint=Path(options["checkpoint"]).name,
        ligand_identity_sha256=digest,
        recorded_ligand_source=source,
        sources=[
            dict(path=str(f.relative_to(root)), sha256=hashlib.sha256(f.read_bytes()).hexdigest())
            for f in (trace, receptor, reference, ligand_input, trace.parent / "docked.sdf")
        ],
    )
    # The public record must not retain machine-specific paths from the bundle.
    result["recorded_ligand_source"] = {
        k: Path(v).name if k == "path" else v for k, v in source.items()
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        f"Exported {len(times)} real frames, 6 fragments; panels {indices}, times {times[indices]}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export(args.root.resolve(), args.output)


if __name__ == "__main__":
    main()
