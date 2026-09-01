#!/usr/bin/env python3
"""Freeze the exact FULL-V2 Astex/PoseBusters ligand-input mapping.

The source mappings stay local and ignored by Git.  This command creates the
small, reviewable scientific manifest used by the published workflow.  The one
known source defect, Astex 1meh, is corrected from the deposited reference SDF
*graph*; coordinates are not used for conformer generation or sampling.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq
from rdkit import Chem, rdBase

from effdock.evaluation.benchmark import load_ligand as load_reference_ligand
from effdock.workflows.benchmark_inputs import (
    BENCHMARK_INPUT_MANIFEST_SCHEMA,
    canonical_heavy_smiles,
    file_sha256,
    ligand_input_identity,
    mapping_sha256,
    sorted_id_sha256,
)

PROTOCOL_ID = "EFFDOCK-UNIFIED-GUIDANCE-BUDGET1000-FULL-V2"


def _sample_key(system_id: str, instance_chain: str) -> str:
    return f"{system_id}__{instance_chain}".replace("/", "_")


def _load_local_mapping(dataset: str, external_dir: Path) -> tuple[dict[str, str], dict]:
    filename = "astex_smiles.json" if dataset == "astex" else "pb_smiles.json"
    path = external_dir / filename
    raw = json.loads(path.read_text())
    sources: dict[str, object] = {
        "mapping": {"path": str(path), "sha256": file_sha256(path)}
    }
    if dataset == "posebusters":
        membership = external_dir / "posebusters_v2_ids.txt"
        keep = {
            line.strip().lower()
            for line in membership.read_text().splitlines()
            if line.strip()
        }
        raw = {key: value for key, value in raw.items() if key.lower() in keep}
        sources["membership"] = {
            "path": str(membership),
            "sha256": file_sha256(membership),
        }
    mapping = {
        str(key).lower(): str(value["smiles"] if isinstance(value, dict) else value)
        for key, value in raw.items()
    }
    return mapping, sources


def _correct_1meh(
    mapping: dict[str, str],
    benchmark_root: Path,
) -> dict[str, object]:
    reference = (
        benchmark_root
        / "astex_diverse_set"
        / "1MEH_MOA"
        / "1MEH_MOA_ligand.sdf"
    )
    mol = load_reference_ligand(reference, "sdf")
    corrected = Chem.MolToSmiles(
        Chem.RemoveAllHs(mol),
        canonical=True,
        isomericSmiles=True,
    )
    old_smiles = mapping["1meh"]
    if canonical_heavy_smiles(old_smiles) == corrected:
        raise ValueError("1meh source mapping is unexpectedly already corrected")
    mapping["1meh"] = corrected
    title = ""
    supplier = Chem.SDMolSupplier(str(reference), sanitize=False, removeHs=False)
    raw = next(supplier)
    if raw is not None and raw.HasProp("_Name"):
        title = raw.GetProp("_Name")
    return {
        "complex_id": "1meh",
        "reason": "source mapping labels IMP/nucleotide but deposited Astex complex is MOA",
        "old_input_identity": ligand_input_identity("1meh", old_smiles),
        "corrected_smiles": corrected,
        "reference_sdf": str(reference),
        "reference_sdf_sha256": file_sha256(reference),
        "reference_sdf_title": title,
        "reference_heavy_atoms": mol.GetNumAtoms(),
        "coordinate_use": "graph identity correction only; coordinates forbidden in sampling",
    }


def _legacy_split_overlap(
    datasets: dict[str, dict[str, str]],
    pool_path: Path,
    split_path: Path,
) -> dict[str, object]:
    split = json.loads(split_path.read_text())
    train = set(split["train"])
    val = set(split["val"])
    table = pq.read_table(
        pool_path,
        columns=[
            "system_id",
            "ligand_instance_chain",
            "entry_pdb_id",
            "ligand_rdkit_canonical_smiles",
        ],
    )
    by_canonical: dict[str, list[tuple[str, str, str]]] = {}
    for row in table.to_pylist():
        key = _sample_key(row["system_id"], row["ligand_instance_chain"])
        split_name = "train" if key in train else ("val" if key in val else "")
        if not split_name:
            continue
        try:
            canonical = canonical_heavy_smiles(row["ligand_rdkit_canonical_smiles"])
        except Exception:
            continue
        by_canonical.setdefault(canonical, []).append(
            (key, split_name, str(row["entry_pdb_id"]).lower())
        )

    result: dict[str, object] = {
        "claim": (
            "checkpoint uses the preserved compatibility split; absolute benchmark metrics "
            "are descriptive and are not an independent external-generalization estimate"
        ),
        "pool_parquet": {"path": str(pool_path), "sha256": file_sha256(pool_path)},
        "split": {"path": str(split_path), "sha256": file_sha256(split_path)},
        "datasets": {},
    }
    for dataset, mapping in datasets.items():
        ligand_overlap_ids: list[str] = []
        exact_entry_overlap_ids: list[str] = []
        train_rows = 0
        val_rows = 0
        for complex_id, smiles in mapping.items():
            canonical = canonical_heavy_smiles(smiles)
            rows = by_canonical.get(canonical, [])
            if rows:
                ligand_overlap_ids.append(complex_id)
            train_rows += sum(split_name == "train" for _, split_name, _ in rows)
            val_rows += sum(split_name == "val" for _, split_name, _ in rows)
            entry_id = complex_id.split("_", 1)[0]
            if any(pdb_id == entry_id for _, _, pdb_id in rows):
                exact_entry_overlap_ids.append(complex_id)
        result["datasets"][dataset] = {
            "benchmark_ids_with_split_ligand_identity_overlap": sorted(ligand_overlap_ids),
            "benchmark_ids_with_split_ligand_identity_overlap_count": len(ligand_overlap_ids),
            "benchmark_ids_with_exact_entry_and_ligand_overlap": sorted(
                exact_entry_overlap_ids
            ),
            "benchmark_ids_with_exact_entry_and_ligand_overlap_count": len(
                exact_entry_overlap_ids
            ),
            "matching_train_rows": train_rows,
            "matching_val_rows": val_rows,
        }
    return result


def build_manifest(args: argparse.Namespace) -> dict[str, object]:
    astex, astex_sources = _load_local_mapping("astex", args.external_dir)
    posebusters, pb_sources = _load_local_mapping("posebusters", args.external_dir)
    correction = _correct_1meh(astex, args.benchmark_root)
    mappings = {"astex": astex, "posebusters": posebusters}
    overlap = _legacy_split_overlap(mappings, args.pool_parquet, args.split_file)
    datasets: dict[str, object] = {}
    for dataset, mapping in mappings.items():
        source_manifests = astex_sources if dataset == "astex" else pb_sources
        ids = sorted(mapping)
        integrity = overlap["datasets"][dataset]
        datasets[dataset] = {
            "count": len(ids),
            "ids_sha256": sorted_id_sha256(ids),
            "mapping_sha256": mapping_sha256(dataset, mapping),
            "source_manifests": source_manifests,
            "integrity_boundary": integrity,
            "ligands": {
                complex_id: {
                    "smiles": mapping[complex_id],
                    "input_identity": ligand_input_identity(complex_id, mapping[complex_id]),
                }
                for complex_id in ids
            },
        }
    return {
        "schema_version": BENCHMARK_INPUT_MANIFEST_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "rdkit_version": rdBase.rdkitVersion,
        "hydrogen_policy": "seeded generic loader followed by Chem.RemoveAllHs",
        "datasets": datasets,
        "declared_source_correction": correction,
        "checkpoint_integrity_boundary": overlap,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--external-dir", type=Path, default=Path("data/external_test"))
    parser.add_argument(
        "--benchmark-root",
        type=Path,
        default=Path("data/external_benchmarks/data"),
    )
    parser.add_argument("--pool-parquet", type=Path, default=Path("data/plinder_pool.parquet"))
    parser.add_argument("--split-file", type=Path, default=Path("data/splits/plinder.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = build_manifest(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
                "counts": {
                    key: value["count"] for key, value in manifest["datasets"].items()
                },
                "integrity": manifest["checkpoint_integrity_boundary"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
