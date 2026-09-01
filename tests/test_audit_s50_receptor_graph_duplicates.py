import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch

from scripts.audit_s50_receptor_graph_duplicates import _audit_one, _load_tasks


def _write_sample(root: Path, *, duplicated: bool) -> dict[str, object]:
    sample_key = "sample"
    sample_root = root / sample_key
    sample_root.mkdir()
    coords = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
        if duplicated
        else [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
        dtype=torch.float32,
    )
    protein = {
        "patom_coords": coords,
        "patom_residue_id": torch.tensor([0, 0, 1, 1]),
        "patom_token": torch.tensor([1, 2, 1, 2]),
        "patom_is_backbone": torch.tensor([True, False, True, False]),
        "patom_is_metal": torch.zeros(4, dtype=torch.bool),
        "patom_is_donor": torch.zeros(4, dtype=torch.bool),
        "patom_is_acceptor": torch.zeros(4, dtype=torch.bool),
        "patom_is_positive": torch.zeros(4, dtype=torch.bool),
        "patom_is_negative": torch.zeros(4, dtype=torch.bool),
        "patom_is_hydrophobic": torch.ones(4, dtype=torch.bool),
        "pres_coords": torch.tensor(
            [[0.5, 0.0, 0.0], [0.5, 0.0, 0.0]]
            if duplicated
            else [[0.5, 0.0, 0.0], [2.5, 0.0, 0.0]],
            dtype=torch.float32,
        ),
        "pres_residue_type": torch.tensor([3, 3]),
        "pres_is_pseudo": torch.zeros(2, dtype=torch.bool),
        "pres_atom_index": torch.tensor([0, 2]),
    }
    meta = {
        "plinder_system_id": "system",
        "pocket_center": torch.zeros(3),
        "num_atom": torch.tensor(2),
        "num_frag": torch.tensor(1),
    }
    torch.save(protein, sample_root / "protein.pt")
    torch.save(meta, sample_root / "meta.pt")
    return {
        "sample_key": sample_key,
        "split": "train",
        "system_id": "system",
        "processed_root": str(root),
        "pocket_cutoff": 10.0,
    }


def test_audit_detects_exact_duplicate_receptor_copy(tmp_path: Path) -> None:
    result = _audit_one(_write_sample(tmp_path, duplicated=True))

    assert result["graph_nodes"] == 9
    assert result["deduplicated_graph_nodes"] == 6
    assert result["exact_duplicate_atom_nodes"] == 2
    assert result["exact_duplicate_residue_nodes"] == 1
    assert result["max_atom_multiplicity"] == 2
    assert result["material_exact_duplication"] is True


def test_audit_keeps_distinct_receptor_nodes(tmp_path: Path) -> None:
    result = _audit_one(_write_sample(tmp_path, duplicated=False))

    assert result["graph_nodes"] == result["deduplicated_graph_nodes"] == 9
    assert result["has_any_exact_duplicate"] is False
    assert result["material_exact_duplication"] is False


def test_load_tasks_reads_required_pool_columns(tmp_path: Path) -> None:
    processed_root = tmp_path / "processed"
    processed_root.mkdir()
    bank_manifest = tmp_path / "bank.json"
    bank_manifest.write_text(
        json.dumps(
            {
                "status": "complete",
                "claim_eligible": True,
                "records": [
                    {"sample_key": "sample", "system_id": "system", "split": "train"}
                ],
            }
        )
    )
    pool_parquet = tmp_path / "pool.parquet"
    pq.write_table(
        pa.table(
            {
                "system_id": ["system"],
                "system_id_no_biounit": ["entry__1.A_2.A"],
                "unused": [1],
            }
        ),
        pool_parquet,
    )

    tasks, _ = _load_tasks(bank_manifest, pool_parquet, processed_root, 10.0)

    assert tasks == [
        {
            "sample_key": "sample",
            "split": "train",
            "system_id": "system",
            "system_id_no_biounit": "entry__1.A_2.A",
            "receptor_instance_count": 2,
            "unique_receptor_chain_count": 2,
            "processed_root": str(processed_root),
            "pocket_cutoff": 10.0,
        }
    ]


def test_audit_allows_atom_residue_without_virtual_node(tmp_path: Path) -> None:
    task = _write_sample(tmp_path, duplicated=False)
    sample_root = tmp_path / "sample"
    protein = torch.load(sample_root / "protein.pt", weights_only=True)
    protein["patom_coords"] = torch.cat(
        [protein["patom_coords"], torch.tensor([[4.0, 0.0, 0.0]])]
    )
    protein["patom_residue_id"] = torch.cat(
        [protein["patom_residue_id"], torch.tensor([2])]
    )
    protein["patom_token"] = torch.cat([protein["patom_token"], torch.tensor([3])])
    for key in (
        "patom_is_backbone",
        "patom_is_metal",
        "patom_is_donor",
        "patom_is_acceptor",
        "patom_is_positive",
        "patom_is_negative",
        "patom_is_hydrophobic",
    ):
        protein[key] = torch.cat(
            [protein[key], torch.zeros(1, dtype=protein[key].dtype)]
        )
    protein["pres_atom_index"] = torch.tensor([0, 2])
    torch.save(protein, sample_root / "protein.pt")

    result = _audit_one(task)

    assert result["protein_atom_residue_group_count"] == 3
    assert result["protein_residue_nodes"] == 2
    assert result["atom_residue_groups_without_virtual_node"] == 1


def test_audit_reproduces_plus_five_radius_fallback(tmp_path: Path) -> None:
    task = _write_sample(tmp_path, duplicated=False)
    sample_root = tmp_path / "sample"
    protein = torch.load(sample_root / "protein.pt", weights_only=True)
    protein["patom_coords"] += torch.tensor([12.0, 0.0, 0.0])
    protein["pres_coords"] += torch.tensor([12.0, 0.0, 0.0])
    torch.save(protein, sample_root / "protein.pt")

    result = _audit_one(task)

    assert result["crop_route"] == "radius_plus_5"
    assert result["protein_atom_nodes"] == 4


def test_audit_reproduces_nearest_residue_fallback(tmp_path: Path) -> None:
    task = _write_sample(tmp_path, duplicated=False)
    sample_root = tmp_path / "sample"
    protein = torch.load(sample_root / "protein.pt", weights_only=True)
    protein["patom_coords"] += torch.tensor([30.0, 0.0, 0.0])
    protein["pres_coords"] += torch.tensor([30.0, 0.0, 0.0])
    torch.save(protein, sample_root / "protein.pt")

    result = _audit_one(task)

    assert result["crop_route"] == "nearest_32_residues"
    assert result["protein_atom_nodes"] == 4
