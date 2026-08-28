from __future__ import annotations

import json
from pathlib import Path

import pytest
from rdkit import Chem

from effdock.inference.preprocess import load_ligand as load_generic_ligand
from effdock.workflows.benchmark_inputs import (
    BENCHMARK_INPUT_MANIFEST_SCHEMA,
    BENCHMARK_INPUT_PROTOCOL_ID,
    canonical_heavy_smiles,
    full_heavy_atom_mapping_metadata,
    ligand_input_identity,
    load_benchmark_inputs,
    load_benchmark_ligand,
    mapping_sha256,
    sorted_id_sha256,
)


@pytest.mark.parametrize(
    "smiles,expected_atoms",
    [
        ("[H]/N=C(/c1ccccc1)\\N", 9),
        ("[H]/N=C\\1/CCCN1", 6),
    ],
)
def test_benchmark_loader_removes_stereo_defining_explicit_hydrogen(
    smiles: str,
    expected_atoms: int,
) -> None:
    mol, has_pose = load_benchmark_ligand(smiles, random_seed=43)
    assert has_pose is False
    assert mol.GetNumAtoms() == expected_atoms
    assert all(atom.GetAtomicNum() != 1 for atom in mol.GetAtoms())
    assert mol.GetNumConformers() == 1


def test_full_v2_heavy_atom_policy_does_not_change_generic_loader() -> None:
    smiles = "[H]/N=C(/c1ccccc1)\\N"
    generic, _ = load_generic_ligand(smiles, random_seed=43)
    full_v2, _ = load_benchmark_ligand(smiles, random_seed=43)
    assert sum(atom.GetAtomicNum() == 1 for atom in generic.GetAtoms()) == 1
    assert all(atom.GetAtomicNum() != 1 for atom in full_v2.GetAtoms())


def test_frozen_manifest_is_content_addressed(tmp_path: Path) -> None:
    mapping = {"one": "CCO", "two": "[H]/N=C(\\N)c1ccccc1"}
    manifest = {
        "schema_version": BENCHMARK_INPUT_MANIFEST_SCHEMA,
        "protocol_id": BENCHMARK_INPUT_PROTOCOL_ID,
        "datasets": {
            "astex": {
                "count": 2,
                "ids_sha256": sorted_id_sha256(sorted(mapping)),
                "mapping_sha256": mapping_sha256("astex", mapping),
                "source_manifests": {},
                "integrity_boundary": {},
                "ligands": {
                    key: {
                        "smiles": value,
                        "input_identity": ligand_input_identity(key, value),
                    }
                    for key, value in mapping.items()
                },
            }
        },
    }
    path = tmp_path / "inputs.json"
    path.write_text(json.dumps(manifest))

    loaded, identity = load_benchmark_inputs("astex", tmp_path, path)
    assert loaded == mapping
    assert identity["count"] == 2
    assert identity["mapping_sha256"] == manifest["datasets"]["astex"]["mapping_sha256"]
    assert len(identity["sha256"]) == 64
    assert identity["per_id"]["two"]["canonical_heavy_isomeric_smiles"] == canonical_heavy_smiles(
        mapping["two"]
    )

    manifest["datasets"]["astex"]["ligands"]["one"]["smiles"] = "CCN"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="identity mismatch"):
        load_benchmark_inputs("astex", tmp_path, path)


def test_full_atom_mapping_allows_bond_representation_but_not_connectivity_change() -> None:
    reference = Chem.MolFromSmiles("NC=O")
    represented = Chem.MolFromSmiles("N=CO")
    assert reference is not None and represented is not None
    metadata = full_heavy_atom_mapping_metadata(
        reference,
        represented,
        [0, 1, 2],
        [0, 1, 2],
        "mcs(3/3)",
    )
    assert metadata["accepted"] is True
    assert metadata["connectivity_match"] is True
    assert metadata["bond_orders_match"] is False
    assert metadata["relation"] == "same_connectivity_representation_mismatch"

    constitutional_isomer = Chem.MolFromSmiles("CC(O)C")
    linear = Chem.MolFromSmiles("CCCO")
    assert constitutional_isomer is not None and linear is not None
    rejected = full_heavy_atom_mapping_metadata(
        linear,
        constitutional_isomer,
        [0, 1, 2, 3],
        [0, 1, 2, 3],
        "mcs(4/4)",
    )
    assert rejected["accepted"] is False
    assert rejected["connectivity_match"] is False
