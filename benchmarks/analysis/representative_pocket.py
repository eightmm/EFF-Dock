"""Export the saved illustration's supplied-center crop from its original receptor."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from effdock.data.dataset import crop_to_pocket
from effdock.preprocess.protein import _parse_pdb_lines, parse_pocket_atoms

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "benchmarks/results/paper/trajectory"


def export(source_root):
    torch.set_num_threads(1)
    trace_path = DATA / "trace.json"
    trace = json.loads(trace_path.read_text())
    source = next(r for r in trace["sources"] if r["path"].endswith("_protein.pdb"))
    path = source_root / source["path"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != source["sha256"]:
        raise ValueError("Original receptor source changed")
    full = parse_pocket_atoms(path)
    raw = _parse_pdb_lines(path)
    if full is None or not torch.equal(
        full["patom_coords"], torch.tensor([a.coords for a in raw], dtype=torch.float32)
    ):
        raise ValueError("Receptor parser atom ordering differs")
    center = torch.tensor(trace["pocket_center"], dtype=torch.float32)
    cutoff = trace["protocol"]["pocket_cutoff"]
    cropped = crop_to_pocket(full, center, cutoff=cutoff)
    if cropped is None:
        raise ValueError("Empty supplied pocket")
    active = full["patom_residue_id"][
        torch.cdist(full["patom_coords"], center[None]).squeeze(1) <= cutoff
    ].unique()
    keep = torch.isin(full["patom_residue_id"], active)
    if not torch.equal(cropped["patom_coords"], full["patom_coords"][keep]):
        raise ValueError("Crop altered original atom coordinates")
    residues = sorted(
        {
            (a.chain, a.res_num, a.icode.strip())
            for a, yes in zip(raw, keep.tolist(), strict=True)
            if yes
        }
    )
    pdb = path.read_text()
    cropped_lines = [
        line
        for line in pdb.splitlines()
        if line.startswith(("ATOM  ", "HETATM"))
        and (line[21], int(line[22:26]), line[26].strip()) in residues
    ]
    coordinates = torch.tensor(
        [[float(line[a:b]) for a, b in ((30, 38), (38, 46), (46, 54))] for line in cropped_lines],
        dtype=torch.float32,
    )
    if not torch.equal(coordinates, cropped["patom_coords"]):
        raise ValueError("Display crop atom records differ from model crop")
    if any(icode for _, _, icode in residues):
        raise ValueError("Display selector needs explicit insertion-code handling")
    result = dict(
        trace_sha256=hashlib.sha256(trace_path.read_bytes()).hexdigest(),
        source=source,
        full_pdb=pdb,
        cropped_pdb="\n".join(cropped_lines) + "\nEND\n",
        center=trace["pocket_center"],
        cutoff_angstrom=cutoff,
        residues=residues,
        full_atom_count=len(raw),
        cropped_atom_count=int(keep.sum()),
        rule="Keep complete residues having at least one heavy atom within the recorded cutoff of the supplied center; production crop_to_pocket",
        note="Full supplied receptor, not a reconstructed full-length sequence. No pocket prediction or coordinate alignment.",
        source_code={
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in ("src/effdock/data/dataset.py", "src/effdock/preprocess/protein.py")
        },
    )
    (DATA / "input_pocket.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"Verified crop: {len(raw)} -> {int(keep.sum())} atoms; {len(residues)} residues; cutoff {cutoff} A"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    args = parser.parse_args()
    export(args.source_root.resolve())


if __name__ == "__main__":
    main()
