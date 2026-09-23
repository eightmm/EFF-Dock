"""Export the original ligand's 2D graph with its saved fragment boundaries."""

import argparse
import hashlib
import json
from pathlib import Path

import rdkit
from rdkit import Chem
from rdkit.Chem import rdDepictor

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "benchmarks/results/paper/trajectory"


def export(source_root, output):
    trace_path = DATA / "trace.json"
    trace = json.loads(trace_path.read_text())
    source = next(r for r in trace["sources"] if r["path"].endswith("_ligand_start_conf.sdf"))
    path = source_root / source["path"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != source["sha256"]:
        raise ValueError("Prepared input ligand checksum mismatch")
    mol = Chem.SDMolSupplier(str(path), removeHs=True)[0]
    if mol is None or [a.GetSymbol() for a in mol.GetAtoms()] != trace["elements"]:
        raise ValueError("Prepared ligand atom identity mismatch")
    if sorted(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx())) for b in mol.GetBonds()) != sorted(
        sorted(b) for b in trace["bonds"]
    ):
        raise ValueError("Prepared ligand connectivity mismatch")
    rdDepictor.Compute2DCoords(mol, canonOrient=True)
    bonds = [
        dict(
            atoms=[b.GetBeginAtomIdx(), b.GetEndAtomIdx()],
            order=b.GetBondTypeAsDouble(),
            aromatic=b.GetIsAromatic(),
            cut=trace["fragment_id"][b.GetBeginAtomIdx()]
            != trace["fragment_id"][b.GetEndAtomIdx()],
        )
        for b in mol.GetBonds()
    ]
    record = dict(
        description="2D depiction of the original prepared input ligand; coordinates are a diagram, not a docking conformer.",
        trace_sha256=hashlib.sha256(trace_path.read_bytes()).hexdigest(),
        source=source,
        smiles=Chem.MolToSmiles(mol),
        coordinates=mol.GetConformer().GetPositions()[:, :2].tolist(),
        elements=trace["elements"],
        fragment_id=trace["fragment_id"],
        bonds=bonds,
        rdkit_version=rdkit.__version__,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2) + "\n")
    print(f"Exported {len(bonds)} bonds and {sum(b['cut'] for b in bonds)} fragment boundaries")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DATA / "ligand_diagram.json")
    args = parser.parse_args()
    export(args.source_root, args.output)


if __name__ == "__main__":
    main()
