"""Export existing 1T46 candidates selected by frozen confidence ranks."""

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolAlign

from benchmarks.analysis.evidence import selected_indices

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "benchmarks/results/paper/trajectory"
LEDGER = "outputs/benchmarks/astex_pb_unguided_r3_hostmatched_v2/n100_s10/selection/full/astex_repeat_0/records.json"


def export(source_root):
    sources = {}

    def read(name, expected):
        path = Path(name)
        if not path.is_absolute():
            path = source_root / path
        blob = path.read_bytes()
        digest = hashlib.sha256(blob).hexdigest()
        if digest != expected:
            raise ValueError(f"Frozen source mismatch: {path.name}")
        sources[str(path.relative_to(source_root))] = digest
        return blob

    trace_path = DATA / "trace.json"
    trace = json.loads(trace_path.read_text())
    manifest = json.loads((ROOT / "benchmarks/results/paper/candidate_metrics.json").read_text())
    digest = next(r["sha256"] for r in manifest["sources"] if r["path"] == LEDGER)
    rows = json.loads(read(LEDGER, digest))
    record = next(r for r in rows if r["id"] == "1t46" and r["stage"] == "refined")
    _, selected = selected_indices(record)
    confidence = json.loads(read(record["confidence_summary"], record["confidence_summary_sha256"]))
    artifact = confidence["artifacts"]["scores_csv"]
    if artifact["sha256"] != record["scores_sha256"]:
        raise ValueError("Confidence CSV identity mismatch")
    scores = list(csv.DictReader(io.StringIO(read(artifact["path"], artifact["sha256"]).decode())))
    if [int(r["pose_index"]) for r in scores] != list(range(100)) or [
        float(r["after_confidence_rmsd"]) for r in scores
    ] != record["predicted_rmsd"]:
        raise ValueError("Confidence ordering or values mismatch")
    refinement_path = Path(confidence["inputs"]["refinement_summary"])
    if not refinement_path.is_absolute():
        refinement_path = source_root / refinement_path
    refinement = json.loads(
        read(refinement_path, confidence["inputs"]["refinement_summary_sha256"])
    )
    protein = next(r for r in trace["sources"] if r["path"].endswith("_protein.pdb"))
    if refinement["inputs"]["protein_sha256"] != protein["sha256"]:
        raise ValueError("Selection and trajectory receptor frames differ")
    read(protein["path"], protein["sha256"])
    template_source = next(
        r for r in trace["sources"] if r["path"].endswith("_ligand_start_conf.sdf")
    )
    read(template_source["path"], template_source["sha256"])
    template = Chem.SDMolSupplier(str(source_root / template_source["path"]), removeHs=True)[0]
    if [a.GetSymbol() for a in template.GetAtoms()] != trace["elements"]:
        raise ValueError("Trace atom identity mismatch")
    read(record["bank"], record["bank_sha256"])
    bank = list(Chem.SDMolSupplier(record["bank"], removeHs=True))
    if len(bank) != 100 or any(m is None for m in bank):
        raise ValueError("Incomplete candidate bank")
    order = sorted(range(100), key=lambda i: (record["predicted_rmsd"][i], i))
    eligible = [i for i in order if record["chirality_valid"][i]]
    if len(eligible) < 4 or eligible[0] != selected:
        raise ValueError("Unexpected selector eligibility")
    candidates = []
    for rank_index in (0, (len(eligible) - 1) // 4, (len(eligible) - 1) // 2, len(eligible) - 1):
        index = eligible[rank_index]
        mol = bank[index]
        if Chem.MolToSmiles(mol) != Chem.MolToSmiles(template):
            raise ValueError("Candidate molecular graph differs")
        matches = mol.GetSubstructMatches(template, uniquify=False, useChirality=True)
        if not matches or mol.GetNumAtoms() != len(trace["elements"]):
            raise ValueError("No complete atom mapping")
        mapping = min(matches)
        assigned = []
        for match in matches:
            labels = np.empty(len(match), dtype=int)
            labels[list(match)] = trace["fragment_id"]
            assigned.append(labels)
        if any(not np.array_equal(assigned[0], labels) for labels in assigned):
            raise ValueError("Symmetry changes fragment assignment")
        coords = mol.GetConformer().GetPositions()[list(mapping)]
        if coords.shape != (37, 3) or not np.isfinite(coords).all():
            raise ValueError("Invalid candidate geometry")
        candidates.append(
            dict(
                index=index,
                rank=rank_index + 1,
                selected=index == selected,
                predicted_rmsd=record["predicted_rmsd"][index],
                atom_mapping=list(mapping),
                coordinates=coords.tolist(),
            )
        )
    output = dict(
        dataset="astex",
        id="1t46",
        repeat=0,
        bank_size=100,
        trace_sha256=hashlib.sha256(trace_path.read_bytes()).hexdigest(),
        rule="Best, lower-quartile, lower-median and worst predicted-RMSD ranks among chirality-eligible refined candidates",
        note="Same complex, separate N100 run from the N1 illustrative trajectory; check/cross denotes selection, not PB validity",
        predicted_rmsd=record["predicted_rmsd"],
        chirality_valid=record["chirality_valid"],
        selected_index=selected,
        candidates=candidates,
        source_files=sources,
    )
    (DATA / "selection_example.json").write_text(
        json.dumps(output, indent=2, allow_nan=False) + "\n"
    )
    print("Exported existing candidate indices:", [r["index"] for r in candidates])
    reference_source = next(r for r in trace["sources"] if r["path"].endswith("_ligand.sdf"))
    inputs = refinement["inputs"]
    if inputs["ligand_reference_sha256"] != reference_source["sha256"]:
        raise ValueError("Crystal reference differs between the illustration and evaluation")
    read(inputs["ligand_reference"], reference_source["sha256"])
    reference = Chem.SDMolSupplier(str(source_root / inputs["ligand_reference"]), removeHs=True)[0]
    if reference is None:
        raise ValueError("Invalid crystal ligand")
    before = bank[selected].GetConformer().GetPositions().copy()
    measured = float(rdMolAlign.CalcRMS(bank[selected], reference))
    if not np.array_equal(before, bank[selected].GetConformer().GetPositions()):
        raise ValueError("RMSD verification modified the selected pose")
    if not np.isclose(measured, record["symmetry_rmsd"][selected], atol=1e-6, rtol=0):
        raise ValueError("Crystal overlay RMSD differs from frozen evaluation")
    overlay = dict(
        dataset="astex",
        id="1t46",
        repeat=0,
        selected_index=selected,
        symmetry_rmsd_angstrom=record["symmetry_rmsd"][selected],
        verified_rmsd_angstrom=measured,
        metric="Symmetry-aware heavy-atom RMSD in the receptor frame; no alignment",
        note="Crystal reference is displayed retrospectively, not used for candidate selection",
        selection_sha256=hashlib.sha256((DATA / "selection_example.json").read_bytes()).hexdigest(),
        reference=dict(
            coordinates=reference.GetConformer().GetPositions().tolist(),
            elements=[a.GetSymbol() for a in reference.GetAtoms()],
            bonds=[[b.GetBeginAtomIdx(), b.GetEndAtomIdx()] for b in reference.GetBonds()],
        ),
        source_files=sources,
    )
    (DATA / "selected_reference.json").write_text(
        json.dumps(overlay, indent=2, allow_nan=False) + "\n"
    )
    print(f"Verified selected-pose crystal RMSD: {measured:.8f} A")
    annotations = []
    for candidate in candidates:
        index = candidate["index"]
        before = bank[index].GetConformer().GetPositions().copy()
        measured = float(rdMolAlign.CalcRMS(bank[index], reference))
        if not np.array_equal(before, bank[index].GetConformer().GetPositions()):
            raise ValueError("Candidate RMSD verification modified coordinates")
        saved = record["symmetry_rmsd"][index]
        if not np.isclose(measured, saved, atol=1e-6, rtol=0):
            raise ValueError("Candidate annotation differs from frozen RMSD")
        annotations.append(
            dict(
                index=index,
                rank=candidate["rank"],
                predicted_rmsd=candidate["predicted_rmsd"],
                symmetry_rmsd=saved,
                verified_rmsd=measured,
            )
        )
    annotation_record = dict(
        selection_sha256=hashlib.sha256((DATA / "selection_example.json").read_bytes()).hexdigest(),
        reference_sha256=hashlib.sha256(
            (DATA / "selected_reference.json").read_bytes()
        ).hexdigest(),
        unit="angstrom",
        candidates=annotations,
        note="pRMSD is the frozen confidence prediction; RMSD is retrospective symmetry-aware heavy-atom error without alignment, not a selector input",
    )
    (DATA / "candidate_annotations.json").write_text(
        json.dumps(annotation_record, indent=2, allow_nan=False) + "\n"
    )
    print("Verified all four candidate pRMSD/RMSD annotations")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    args = parser.parse_args()
    export(args.source_root.resolve())


if __name__ == "__main__":
    main()
