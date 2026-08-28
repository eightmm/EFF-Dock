#!/usr/bin/env python3
"""Prepare SurfDock surfaces and pocket ESM embeddings for a frozen shard."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import types
from pathlib import Path
from typing import Any

import torch
from Bio.PDB import PDBParser

THREE_TO_ONE = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "MSE": "M",
    "PHE": "F",
    "PRO": "P",
    "PYL": "O",
    "SER": "S",
    "SEC": "U",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "ASX": "B",
    "GLX": "Z",
    "XAA": "X",
    "XLE": "J",
}
MAX_ESM_RESIDUES = 1022


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--surfdock-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--distance-threshold", type=int, default=8)
    return parser.parse_args()


def safe_link(target: Path, link: Path) -> None:
    target = target.resolve()
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.exists() or link.is_symlink():
        if link.resolve() != target:
            raise FileExistsError(f"Conflicting link: {link}")
        return
    link.symlink_to(target)


def valid_residues(chain: Any) -> list[Any]:
    residues = []
    for residue in chain:
        atom_names = {atom.name for atom in residue}
        if {"N", "CA", "C"}.issubset(atom_names):
            residues.append(residue)
    return residues


def window_groups(indices: list[int], sequence_length: int) -> list[tuple[int, int]]:
    if sequence_length <= MAX_ESM_RESIDUES:
        return [(0, sequence_length)]
    remaining = sorted(set(indices))
    groups: list[tuple[int, int]] = []
    while remaining:
        first = remaining[0]
        included = [idx for idx in remaining if idx - first < MAX_ESM_RESIDUES]
        low, high = included[0], included[-1]
        start = max(
            0, min((low + high - MAX_ESM_RESIDUES + 1) // 2, sequence_length - MAX_ESM_RESIDUES)
        )
        if low < start:
            start = low
        if high >= start + MAX_ESM_RESIDUES:
            start = high - MAX_ESM_RESIDUES + 1
        stop = min(sequence_length, start + MAX_ESM_RESIDUES)
        groups.append((start, stop))
        remaining = [idx for idx in remaining if not (start <= idx < stop)]
    return groups


@torch.inference_mode()
def embed_pocket(
    model: Any,
    alphabet: Any,
    protein_path: Path,
    pocket_path: Path,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, Any]]:
    parser = PDBParser(QUIET=True)
    full = parser.get_structure("full", str(protein_path))[0]
    pocket = parser.get_structure("pocket", str(pocket_path))[0]
    batch_converter = alphabet.get_batch_converter()
    per_residue_embedding: dict[tuple[str, tuple[Any, ...]], torch.Tensor] = {}
    per_residue_edge_distance: dict[tuple[str, tuple[Any, ...]], int] = {}
    window_records: list[dict[str, Any]] = []

    for chain in full:
        residues = valid_residues(chain)
        if not residues:
            continue
        pocket_chain = pocket.child_dict.get(chain.id)
        if pocket_chain is None:
            continue
        pocket_ids = {residue.id for residue in valid_residues(pocket_chain)}
        residue_index = {residue.id: index for index, residue in enumerate(residues)}
        required_indices = [
            residue_index[residue_id] for residue_id in pocket_ids if residue_id in residue_index
        ]
        if not required_indices:
            continue
        sequence = "".join(THREE_TO_ONE.get(residue.resname, "-") for residue in residues)
        for start, stop in window_groups(required_indices, len(sequence)):
            label = f"{protein_path.name}_chain_{chain.id}_{start}_{stop}"
            _, _, tokens = batch_converter([(label, sequence[start:stop])])
            tokens = tokens.to(device)
            representation = model(
                tokens,
                repr_layers=[33],
                return_contacts=False,
            )["representations"][33][0, 1 : stop - start + 1].cpu()
            for index in required_indices:
                if start <= index < stop:
                    key = (chain.id, residues[index].id)
                    distance_to_edge = min(index - start, stop - index - 1)
                    if distance_to_edge > per_residue_edge_distance.get(key, -1):
                        per_residue_embedding[key] = representation[index - start].clone()
                        per_residue_edge_distance[key] = distance_to_edge
            window_records.append(
                {
                    "chain": chain.id,
                    "sequence_length": len(sequence),
                    "window_start": start,
                    "window_stop": stop,
                    "windowed": len(sequence) > MAX_ESM_RESIDUES,
                }
            )

    ordered_embeddings = []
    missing = []
    for chain in pocket:
        for residue in valid_residues(chain):
            key = (chain.id, residue.id)
            vector = per_residue_embedding.get(key)
            if vector is None:
                missing.append(f"{chain.id}:{residue.id}")
            else:
                ordered_embeddings.append(vector)
    if missing:
        raise ValueError(f"Missing ESM embeddings for {len(missing)} pocket residues")
    if not ordered_embeddings:
        raise ValueError("Pocket contains no protein residues")
    return torch.stack(ordered_embeddings), {"windows": window_records}


def configure_surface_imports(surfdock_root: Path) -> Any:
    prepare_target = surfdock_root / "comp_surface" / "prepare_target"
    sys.path.insert(0, str(prepare_target))
    tools = surfdock_root / "comp_surface" / "tools"
    apbs_root = tools / "APBS-3.4.1.Linux"
    pdb2pqr_root = tools / "pdb2pqr-linux-bin64-2.1.1"
    tool_paths = {
        "msms_bin": str(apbs_root / "bin" / "msms"),
        "apbs_bin": str(apbs_root / "bin" / "apbs"),
        "multivalue_bin": str(
            apbs_root / "share" / "apbs" / "tools" / "bin" / "multivalue"
        ),
        "pdb2pqr_bin": str(pdb2pqr_root / "pdb2pqr"),
    }

    # The pinned SurfDock revision contains malformed un-commented attribution
    # text at the end of global_vars.py.  Supply only its constants as an
    # import-time compatibility module, leaving the frozen upstream tree intact.
    global_vars = types.ModuleType("default_config.global_vars")
    global_vars.epsilon = 1.0e-6
    global_vars.NoSolutionError = type("NoSolutionError", (Exception,), {})
    for name, value in tool_paths.items():
        setattr(global_vars, name, value)
    sys.modules[global_vars.__name__] = global_vars

    import computeAPBS  # type: ignore[import-not-found]
    import computeMSMS  # type: ignore[import-not-found]

    upstream_compute_apbs = computeAPBS.computeAPBS

    def compute_apbs_in_output_dir(
        vertices: Any,
        pdb_file: str,
        tmp_file_base: str,
        clear: bool = False,
    ) -> Any:
        # PDB2PQR writes the molecule next to tmp_file_base while the generated
        # APBS input references it by basename.  The pinned wrapper launches APBS
        # from the caller's cwd, so run this serial preprocessing call beside the
        # generated input and restore cwd even if the native tool fails.
        previous_cwd = Path.cwd()
        work_dir = Path(tmp_file_base).resolve().parent
        try:
            os.chdir(work_dir)
            return upstream_compute_apbs(vertices, pdb_file, tmp_file_base, clear=clear)
        finally:
            os.chdir(previous_cwd)

    computeAPBS.computeAPBS = compute_apbs_in_output_dir
    import computeTargetMesh_test_samples  # type: ignore[import-not-found]

    computeMSMS.msms_bin = tool_paths["msms_bin"]
    computeAPBS.apbs_bin = tool_paths["apbs_bin"]
    computeAPBS.multivalue_bin = tool_paths["multivalue_bin"]
    computeAPBS.pdb2pqr_bin = tool_paths["pdb2pqr_bin"]
    for executable in (
        computeMSMS.msms_bin,
        computeAPBS.apbs_bin,
        computeAPBS.multivalue_bin,
        computeAPBS.pdb2pqr_bin,
    ):
        if not os.access(executable, os.X_OK):
            raise FileNotFoundError(
                f"SurfDock preprocessing executable is unavailable: {executable}"
            )
    return computeTargetMesh_test_samples


def main() -> None:
    args = parse_args()
    input_csv = args.input_csv.resolve()
    surfdock_root = args.surfdock_root.resolve()
    output_dir = args.output_dir.resolve()
    prepared_data = output_dir / "prepared_data"
    surface_dir = output_dir / "surfaces"
    esm_dir = output_dir / "esm"
    output_csv = output_dir / "surfdock_inputs.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    esm_dir.mkdir(parents=True, exist_ok=True)

    with input_csv.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    required_columns = {"complex_name", "reference_ligand", "predicted_receptor"}
    if not rows or not required_columns.issubset(rows[0]):
        raise ValueError(f"Input CSV must contain {sorted(required_columns)}")

    materialized: dict[str, dict[str, Path]] = {}
    for row in rows:
        target_id = row["complex_name"]
        target_root = prepared_data / target_id
        protein = target_root / f"{target_id}_protein_processed.pdb"
        ligand = target_root / f"{target_id}_ligand.sdf"
        safe_link(Path(row["predicted_receptor"]), protein)
        safe_link(Path(row["reference_ligand"]), ligand)
        materialized[target_id] = {"protein": protein, "ligand": ligand}

    surface_module = configure_surface_imports(surfdock_root)
    surface_status: dict[str, dict[str, Any]] = {}
    prepared_rows: list[dict[str, str]] = []
    for row in rows:
        target_id = row["complex_name"]
        protein = materialized[target_id]["protein"]
        ligand = materialized[target_id]["ligand"]
        target_surface_dir = surface_dir / target_id
        pocket = (
            target_surface_dir / f"{target_id}_protein_processed_{args.distance_threshold}A.pdb"
        )
        surface = (
            target_surface_dir / f"{target_id}_protein_processed_{args.distance_threshold}A.ply"
        )
        error = None
        if not (pocket.is_file() and surface.is_file()):
            result = surface_module.compute_inp_surface(
                str(protein),
                str(ligand),
                str(surface_dir),
                dist_threshold=args.distance_threshold,
            )
            if result != 0:
                error = f"upstream_surface_failure:{result}"
        complete = pocket.is_file() and surface.is_file()
        if not complete and error is None:
            error = "surface_outputs_missing"
        surface_status[target_id] = {
            "complete": complete,
            "error": error,
            "pocket_path": str(pocket),
            "surface_path": str(surface),
        }
        if complete:
            prepared_rows.append(
                {
                    "complex_name": target_id,
                    "protein_path": str(protein),
                    "pocket_path": str(pocket),
                    "ref_ligand": str(ligand),
                    "ligand_path": str(ligand),
                    "protein_surface": str(surface),
                }
            )

    if not prepared_rows:
        raise RuntimeError("SurfDock surface preprocessing failed for every target")

    import esm

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, alphabet = esm.pretrained.load_model_and_alphabet("esm2_t33_650M_UR50D")
    model.eval().to(device)
    embeddings: dict[str, torch.Tensor] = {}
    embedding_status: dict[str, dict[str, Any]] = {}
    inference_rows: list[dict[str, str]] = []
    for prepared_row in prepared_rows:
        target_id = prepared_row["complex_name"]
        try:
            embedding, details = embed_pocket(
                model,
                alphabet,
                Path(prepared_row["protein_path"]),
                Path(prepared_row["pocket_path"]),
                device,
            )
            pocket_key = Path(prepared_row["pocket_path"]).stem
            embeddings[pocket_key] = embedding
            embedding_status[target_id] = {
                "complete": True,
                "shape": list(embedding.shape),
                **details,
            }
            inference_rows.append(prepared_row)
        except Exception as error:  # preserve the full denominator and continue
            embedding_status[target_id] = {
                "complete": False,
                "error": f"{type(error).__name__}: {error}",
            }

    embedding_path = esm_dir / "esm2_t33_650M_pocket_embeddings.pt"
    torch.save(embeddings, embedding_path)
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "complex_name",
                "protein_path",
                "pocket_path",
                "ref_ligand",
                "ligand_path",
                "protein_surface",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(inference_rows)

    provenance = {
        "schema_version": 1,
        "input_csv": str(input_csv),
        "expected_targets": len(rows),
        "surface_ready_targets": len(prepared_rows),
        "inference_ready_targets": len(inference_rows),
        "distance_threshold_angstrom": args.distance_threshold,
        "protein_policy": "holo-aligned predicted receptor",
        "site_policy": "reference-ligand-supplied 8A surface pocket",
        "esm_model": "esm2_t33_650M_UR50D",
        "esm_max_residues_per_window": MAX_ESM_RESIDUES,
        "long_chain_policy": "local windows covering supplied-pocket residues",
        "surface_status": surface_status,
        "embedding_status": embedding_status,
        "surfdock_csv": str(output_csv),
        "esm_embeddings": str(embedding_path),
    }
    (output_dir / "preprocess_metadata.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(
        f"SurfDock preprocessing: surfaces={len(prepared_rows)}/{len(rows)} "
        f"embeddings={len(inference_rows)}/{len(rows)}"
    )
    if not inference_rows:
        raise RuntimeError("SurfDock ESM preprocessing failed for every target")


if __name__ == "__main__":
    main()
