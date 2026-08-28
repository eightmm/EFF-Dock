from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import pytest
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import scripts.prepare_s50_confidence_training_bank as bank

FROZEN_INPUTS = {
    "sampler_checkpoint": {"path": "/sealed/s50.pt", "sha256": "a" * 64},
    "sampler_config": {"path": "/sealed/train.yaml", "sha256": "b" * 64},
}
MAPPING_METADATA = {"accepted": True, "mapping_method": "strict_stereo"}


def _pose_record(sample_key: str, split: str, system_id: str) -> dict[str, Any]:
    return {
        "sample_key": sample_key,
        "system_id": system_id,
        "split": split,
        "status": "complete",
        "split_index": 1,
        "global_index": 1,
        "sampling_seed": 43,
        "pose_count": bank.NUM_SAMPLES,
        "pose_source": bank._expected_pose_source(split),
        "pose_ensemble_sha256": "",
    }


def _valid_pose_payload(record: dict[str, Any]) -> dict[str, Any]:
    n_atoms, n_fragments, hidden_dim = 3, 2, 4
    poses = torch.zeros(bank.NUM_SAMPLES, n_atoms, 3)
    graph = {
        "node_coords": torch.zeros(7, 3),
        "node_type": torch.zeros(7, dtype=torch.long),
    }
    record["pose_ensemble_sha256"] = bank._tensor_sha256(poses)
    payload: dict[str, Any] = {
        "storage_version": bank.POSE_STORAGE_VERSION,
        "protocol_id": bank.PROTOCOL_ID,
        "study_protocol_id": bank.STUDY_PROTOCOL_ID,
        "pid": record["sample_key"],
        "system_id": record["system_id"],
        "split": record["split"],
        "split_index": record["split_index"],
        "seed": record["sampling_seed"],
        "sampling_seed": record["sampling_seed"],
        "ligand_conformer_seed": bank.CONFORMER_SEED,
        "pose_tag": bank.DEFAULT_POSE_TAG,
        "pose_source": record["pose_source"],
        "checkpoint": FROZEN_INPUTS["sampler_checkpoint"]["path"],
        "checkpoint_sha256": FROZEN_INPUTS["sampler_checkpoint"]["sha256"],
        "config": FROZEN_INPUTS["sampler_config"]["path"],
        "config_sha256": FROZEN_INPUTS["sampler_config"]["sha256"],
        "num_samples": bank.NUM_SAMPLES,
        "num_steps": bank.NUM_STEPS,
        "sigma": bank.SIGMA,
        "time_schedule": bank.TIME_SCHEDULE,
        "schedule_power": bank.SCHEDULE_POWER,
        "pocket_cutoff": bank.POCKET_CUTOFF,
        "prior_pool_size": bank.PRIOR_POOL_SIZE,
        "prior_pool_sha256": "c" * 64,
        "sampling_dynamics": "deterministic_ode",
        "hidden_scope": "ligand",
        "hidden_dtype": "float16",
        "hidden_chunk_size": bank.HIDDEN_CHUNK_SIZE,
        "graph_coordinate_frame": "pocket_centered",
        "lig_num_atoms": n_atoms,
        "lig_num_frags": n_fragments,
        "pocket_center_used": torch.zeros(3),
        "pose_sigma": torch.full((bank.NUM_SAMPLES,), bank.SIGMA),
        "pose_num_steps": torch.full(
            (bank.NUM_SAMPLES,), bank.NUM_STEPS, dtype=torch.long
        ),
        "lig_atom_coords_crystal_centered": torch.zeros(n_atoms, 3),
        "frag_sizes": torch.tensor([2, 1]),
        "fragment_id": torch.tensor([0, 0, 1]),
        "pose_atom_coords": poses,
        "h_lig_node": torch.zeros(
            bank.NUM_SAMPLES, n_atoms + n_fragments, hidden_dim, dtype=torch.float16
        ),
        "lig_node_type": torch.zeros(n_atoms + n_fragments, dtype=torch.long),
        "atom_disp": torch.zeros(bank.NUM_SAMPLES, n_atoms),
        "pose_rmsd": torch.zeros(bank.NUM_SAMPLES),
        "input_to_reference": torch.arange(n_atoms),
        "mapping_metadata": MAPPING_METADATA,
        "graph_centered": graph,
        "graph": graph,
    }
    if record["split"] == "val":
        payload["pose_rmsd_symmetry_no_align"] = torch.zeros(bank.NUM_SAMPLES)
        payload["symmetry_rmsd_method"] = "rdkit_calc_rms_symmetry_no_align"
        payload["source_all_poses_sdf"] = {
            "path": "/sealed/val.sdf",
            "sha256": "d" * 64,
            "size_bytes": 1,
        }
    return payload


def _frozen_record(
    tmp_path: Path,
    record: dict[str, Any],
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ligand_path = tmp_path / f"{record['sample_key']}-ligand.pt"
    reference_ligand: dict[str, Any] = {"atom_coords": torch.zeros(3, 3)}
    canonical_smiles: str | None = None
    ligand_identity: dict[str, str] | None = None
    mapping_metadata = MAPPING_METADATA
    input_to_reference = [0, 1, 2]
    if record["split"] == "val":
        canonical_smiles = "CCC"
        molecule_input, _ = bank.load_benchmark_ligand(
            canonical_smiles, random_seed=bank.CONFORMER_SEED
        )
        reference_ligand = bank._canonical_ligand_data(molecule_input)
        reference_ligand["atom_coords"] = torch.zeros(3, 3)
        molecule_reference = bank.reconstruct_processed_reference(
            reference_ligand, molecule_input
        )
        mapping_metadata, input_to_reference, _ = bank._mapping_record(
            molecule_input, molecule_reference, reference_ligand
        )
        ligand_identity = bank.ligand_input_identity(
            record["sample_key"], canonical_smiles
        )
        if payload is not None:
            payload["mapping_metadata"] = mapping_metadata
            payload["input_to_reference"] = torch.tensor(
                input_to_reference, dtype=torch.long
            )
    torch.save(reference_ligand, ligand_path)
    frozen = {
        "sample_key": record["sample_key"],
        "system_id": record["system_id"],
        "split": record["split"],
        "status": "eligible",
        "split_index": record["split_index"],
        "global_index": record["global_index"],
        "sampling_seed": record["sampling_seed"],
        "pocket_center": [0.0, 0.0, 0.0],
        "num_input_atoms": 3,
        "num_fragments": 2,
        "mapping": mapping_metadata,
        "input_to_reference": input_to_reference,
        "processed_ligand_reference": bank._asset(ligand_path),
    }
    if record["split"] == "val":
        frozen["canonical_smiles"] = canonical_smiles
        frozen["ligand_input_identity"] = ligand_identity
        frozen["val_pose_bank"] = {
            "all_poses_sdf": {
                "path": "/sealed/val.sdf",
                "sha256": "d" * 64,
                "size_bytes": 1,
            },
            "prior_pool_sha256": "c" * 64,
        }
    return frozen


def test_fixed_labels_use_one_frozen_input_to_reference_map() -> None:
    reference = {
        "atom_coords": torch.tensor(
            [[10.0, 0.0, 0.0], [12.0, 0.0, 0.0]], dtype=torch.float32
        )
    }
    center = torch.tensor([10.0, 0.0, 0.0])
    # Input atom 0 maps to crystal atom 1; input atom 1 maps to crystal atom 0.
    poses = torch.tensor([[[2.0, 0.0, 0.0], [0.0, 0.0, 0.0]]])

    aligned, atom_disp, pose_rmsd = bank._fixed_labels(
        poses, reference, [1, 0], center
    )

    assert torch.equal(aligned, poses[0])
    assert torch.equal(atom_disp, torch.zeros(1, 2))
    assert torch.equal(pose_rmsd, torch.zeros(1))


def test_mapping_is_strict_stereo_min_floor_and_deterministic() -> None:
    molecule_input, _ = bank.load_benchmark_ligand("C[C@H](O)F", random_seed=0)
    # Mimic a processed reference with a different atom order and crystal coordinates.
    molecule_reordered = Chem.RenumberAtoms(molecule_input, [3, 2, 1, 0])
    reference_ligand = bank._canonical_ligand_data(molecule_reordered)
    reference_raw = bank.reconstruct_processed_reference(reference_ligand)

    first_metadata, first_mapping, first_reference = bank._mapping_record(
        molecule_input, reference_raw, reference_ligand
    )
    second_metadata, second_mapping, _ = bank._mapping_record(
        molecule_input, reference_raw, reference_ligand
    )

    assert first_mapping == second_mapping
    assert first_metadata == second_metadata
    assert first_metadata["accepted"] is True
    assert first_metadata["mapping_method"] == "strict_stereo"
    assert first_metadata["mapping_truncated"] is False
    assert first_metadata["symmetry_complete"] is True
    assert first_reference.GetNumAtoms() == molecule_input.GetNumAtoms()


def test_mapping_tie_window_selects_lexicographically_smallest_map(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    molecule_input, _ = bank.load_benchmark_ligand("CC", random_seed=0)
    reference_ligand = bank._canonical_ligand_data(molecule_input)
    mappings = [(1, 0), (0, 1)]
    monkeypatch.setattr(
        bank,
        "enumerate_full_atom_mappings",
        lambda *_args, **_kwargs: (mappings, "strict_stereo", False),
    )

    def fake_floor(*args: Any) -> dict[str, Any]:
        mapping = tuple(args[-1])
        floor = 0.5 if mapping == (1, 0) else 0.5 + 5e-13
        return {
            "rigid_fragment_floor_rmsd": floor,
            "pair_distance_rmse": 0.0,
            "stored_partition_equal": True,
        }

    monkeypatch.setattr(bank, "fragment_rigid_fit_floor", fake_floor)

    analysis = bank._analyze_mapping_deterministic(
        molecule_input, molecule_input, reference_ligand
    )

    assert analysis["inference_to_crystal"] == [0, 1]
    assert analysis["rigid_fragment_floor_rmsd"] == pytest.approx(0.5 + 5e-13)


def test_mapping_metadata_tolerates_only_tiny_float_diagnostic_drift() -> None:
    expected = {
        "accepted": True,
        "mapping_method": "strict_stereo",
        "mapping_count": 2,
        "inference_to_crystal": [0, 1],
        "rigid_fragment_floor_rmsd": 0.5,
        "pair_distance_rmse": 0.25,
    }
    observed = copy.deepcopy(expected)
    observed["rigid_fragment_floor_rmsd"] += 1e-16
    observed["pair_distance_rmse"] -= 1e-16

    assert bank._mapping_metadata_matches(observed, expected)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("rigid_fragment_floor_rmsd", 0.5 + 1e-8),
        ("pair_distance_rmse", 0.25 + 1e-8),
        ("rigid_fragment_floor_rmsd", float("nan")),
        ("pair_distance_rmse", float("inf")),
        ("mapping_count", 3),
        ("inference_to_crystal", [1, 0]),
    ],
)
def test_mapping_metadata_rejects_material_nonfinite_or_discrete_drift(
    key: str, value: Any
) -> None:
    expected = {
        "accepted": True,
        "mapping_method": "strict_stereo",
        "mapping_count": 2,
        "inference_to_crystal": [0, 1],
        "rigid_fragment_floor_rmsd": 0.5,
        "pair_distance_rmse": 0.25,
    }
    observed = copy.deepcopy(expected)
    observed[key] = value

    assert not bank._mapping_metadata_matches(observed, expected)


def test_processed_other_element_recovers_unambiguous_arsenic_from_canonical() -> None:
    molecule_input = Chem.AddHs(Chem.MolFromSmiles("C[As](C)C"))
    assert AllChem.EmbedMolecule(molecule_input, randomSeed=0) == 0
    molecule_input = Chem.RemoveHs(molecule_input)
    molecule_processed = Chem.RenumberAtoms(molecule_input, [1, 3, 0, 2])
    reference_ligand = bank._canonical_ligand_data(molecule_processed)
    assert reference_ligand["atom_element"].tolist().count(bank.OTHER_ELEMENT_IDX) == 1

    reconstructed = bank.reconstruct_processed_reference(
        reference_ligand, molecule_input
    )
    metadata, mapping, _ = bank._mapping_record(
        molecule_input, reconstructed, reference_ligand
    )

    assert [atom.GetAtomicNum() for atom in reconstructed.GetAtoms()] == [
        atom.GetAtomicNum() for atom in molecule_processed.GetAtoms()
    ]
    assert torch.allclose(
        torch.tensor(reconstructed.GetConformer().GetPositions()),
        reference_ligand["atom_coords"].to(torch.float64),
    )
    assert metadata["mapping_method"] == "strict_stereo"
    assert sorted(mapping) == list(range(molecule_input.GetNumAtoms()))


def test_processed_other_element_remains_fail_closed_for_ambiguous_elements() -> None:
    molecule_input = Chem.AddHs(Chem.MolFromSmiles("[As][Zn]"))
    assert AllChem.EmbedMolecule(molecule_input, randomSeed=0) == 0
    molecule_input = Chem.RemoveHs(molecule_input)
    reference_ligand = bank._canonical_ligand_data(molecule_input)
    assert reference_ligand["atom_element"].tolist() == [
        bank.OTHER_ELEMENT_IDX,
        bank.OTHER_ELEMENT_IDX,
    ]

    with pytest.raises(ValueError, match="ambiguous"):
        bank.reconstruct_processed_reference(reference_ligand, molecule_input)


def test_mapping_enumeration_truncation_is_input_ineligible(monkeypatch: pytest.MonkeyPatch) -> None:
    molecule = Chem.MolFromSmiles("CC")
    assert molecule is not None
    molecule = Chem.AddHs(molecule)
    assert AllChem.EmbedMolecule(molecule, randomSeed=0) == 0
    molecule = Chem.RemoveHs(molecule)
    ligand = bank._canonical_ligand_data(molecule)
    monkeypatch.setattr(
        bank,
        "enumerate_full_atom_mappings",
        lambda *_args, **_kwargs: ([(0, 1)], "strict_stereo", True),
    )

    with pytest.raises(ValueError, match="exceeded 1024"):
        bank._analyze_mapping_deterministic(molecule, molecule, ligand)


def test_pose_payload_gate_checks_shapes_hashes_finiteness_and_system_id(
    tmp_path: Path,
) -> None:
    record = _pose_record("sample__A", "val", "authoritative-system")
    payload = _valid_pose_payload(record)
    frozen_record = _frozen_record(tmp_path, record, payload)
    path = tmp_path / "pose.pt"

    bank._validate_pose_payload(
        payload,
        record=record,
        frozen_record=frozen_record,
        frozen_inputs=FROZEN_INPUTS,
        pose_tag=bank.DEFAULT_POSE_TAG,
        path=path,
    )

    payload["pose_rmsd_symmetry_no_align"][0] = 0.25
    with pytest.raises(bank.BankContractError, match="independent no-align CalcRMS"):
        bank._validate_pose_payload(
            payload,
            record=record,
            frozen_record=frozen_record,
            frozen_inputs=FROZEN_INPUTS,
            pose_tag=bank.DEFAULT_POSE_TAG,
            path=path,
        )
    payload["pose_rmsd_symmetry_no_align"][0] = 0.0

    fallback_record = copy.deepcopy(frozen_record)
    fallback_record["mapping"]["mapping_method"] = "fallback_full:mcs"
    fallback_record["mapping"]["symmetry_complete"] = False
    fallback_payload = copy.deepcopy(payload)
    fallback_payload["mapping_metadata"] = fallback_record["mapping"]
    with pytest.raises(bank.BankContractError, match="fallback or changed"):
        bank._validate_pose_payload(
            fallback_payload,
            record=record,
            frozen_record=fallback_record,
            frozen_inputs=FROZEN_INPUTS,
            pose_tag=bank.DEFAULT_POSE_TAG,
            path=path,
        )

    payload["graph_centered"]["node_coords"][0, 0] = float("nan")
    with pytest.raises(bank.BankContractError, match="non-finite"):
        bank._validate_pose_payload(
            payload,
            record=record,
            frozen_record=frozen_record,
            frozen_inputs=FROZEN_INPUTS,
            pose_tag=bank.DEFAULT_POSE_TAG,
            path=path,
        )


def test_pose_payload_gate_rejects_pose_hash_and_direct_label_mismatch(
    tmp_path: Path,
) -> None:
    record = _pose_record("sample__A", "train", "authoritative-system")
    payload = _valid_pose_payload(record)
    frozen_record = _frozen_record(tmp_path, record)
    path = tmp_path / "pose.pt"
    record["pose_ensemble_sha256"] = "0" * 64
    with pytest.raises(bank.BankContractError, match="ensemble hash"):
        bank._validate_pose_payload(
            payload,
            record=record,
            frozen_record=frozen_record,
            frozen_inputs=FROZEN_INPUTS,
            pose_tag=bank.DEFAULT_POSE_TAG,
            path=path,
        )

    record["pose_ensemble_sha256"] = bank._tensor_sha256(payload["pose_atom_coords"])
    payload["pose_rmsd"][0] = 1.0
    with pytest.raises(bank.BankContractError, match="disagree"):
        bank._validate_pose_payload(
            payload,
            record=record,
            frozen_record=frozen_record,
            frozen_inputs=FROZEN_INPUTS,
            pose_tag=bank.DEFAULT_POSE_TAG,
            path=path,
        )


def test_pose_payload_gate_recomputes_atom_labels_from_poses_and_crystal(
    tmp_path: Path,
) -> None:
    record = _pose_record("sample__A", "train", "authoritative-system")
    payload = _valid_pose_payload(record)
    frozen_record = _frozen_record(tmp_path, record)
    payload["pose_atom_coords"].fill_(10.0)
    record["pose_ensemble_sha256"] = bank._tensor_sha256(payload["pose_atom_coords"])

    with pytest.raises(bank.BankContractError, match="poses/crystal"):
        bank._validate_pose_payload(
            payload,
            record=record,
            frozen_record=frozen_record,
            frozen_inputs=FROZEN_INPUTS,
            pose_tag=bank.DEFAULT_POSE_TAG,
            path=tmp_path / "pose.pt",
        )


def test_summary_record_join_rejects_system_seed_and_map_authority(
    tmp_path: Path,
) -> None:
    record = _pose_record("sample__A", "train", "authoritative-system")
    frozen_record = _frozen_record(tmp_path, record)
    bad_record = {**record, "system_id": "wrong-system", "sampling_seed": 99}

    with pytest.raises(bank.BankContractError, match="join mismatch"):
        bank._validate_summary_record_join(
            bad_record, frozen_record, path=tmp_path / "shard.json"
        )

    payload = _valid_pose_payload(record)
    payload["mapping_metadata"] = {"accepted": True, "mapping_method": "wrong"}
    with pytest.raises(bank.BankContractError, match="atom-map identity"):
        bank._validate_pose_payload(
            payload,
            record=record,
            frozen_record=frozen_record,
            frozen_inputs=FROZEN_INPUTS,
            pose_tag=bank.DEFAULT_POSE_TAG,
            path=tmp_path / "pose.pt",
        )


def test_eligible_record_selection_is_outcome_blind() -> None:
    manifest = {
        "inventory": {"train": {"eligible_ids": ["a", "b"]}},
        "records": [
            {
                "sample_key": "a",
                "split": "train",
                "status": "eligible",
                "candidate_rmsd": 100.0,
            },
            {
                "sample_key": "b",
                "split": "train",
                "status": "eligible",
                "candidate_rmsd": 0.0,
            },
        ],
    }

    selected = bank._eligible_records(manifest, "train")

    assert [record["sample_key"] for record in selected] == ["a", "b"]


def _minimal_preflight_task(tmp_path: Path) -> dict[str, Any]:
    return {
        "sample_key": "sample__A",
        "split": "train",
        "split_index": 1,
        "canonical_smiles": "CC",
        "processed_root": str(tmp_path),
        "val_pose_bank": None,
    }


def test_preflight_only_classifies_explicit_input_compatibility(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        bank,
        "_asset",
        lambda path: {"path": str(path), "sha256": "a" * 64, "size_bytes": 1},
    )
    monkeypatch.setattr(
        bank,
        "_load_processed",
        lambda _paths: ({}, {}, {"pocket_center": torch.zeros(3), "plinder_system_id": "sys"}),
    )
    monkeypatch.setattr(
        bank,
        "load_benchmark_ligand",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("cannot embed")),
    )

    result = bank._preflight_one(_minimal_preflight_task(tmp_path))

    assert result["status"] == "input_ineligible"
    assert result["failure_reason"] == "canonical_ligand_preparation_failed"


def test_preflight_propagates_io_and_unexpected_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        bank,
        "_asset",
        lambda _path: (_ for _ in ()).throw(OSError("storage unavailable")),
    )
    with pytest.raises(OSError, match="storage unavailable"):
        bank._preflight_one(_minimal_preflight_task(tmp_path))

    monkeypatch.setattr(
        bank,
        "_asset",
        lambda path: {"path": str(path), "sha256": "a" * 64, "size_bytes": 1},
    )
    monkeypatch.setattr(
        bank,
        "_load_processed",
        lambda _paths: ({}, {}, {"pocket_center": torch.zeros(3), "plinder_system_id": "sys"}),
    )
    monkeypatch.setattr(
        bank,
        "load_benchmark_ligand",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("builder defect")),
    )
    with pytest.raises(RuntimeError, match="builder defect"):
        bank._preflight_one(_minimal_preflight_task(tmp_path))


def test_full_aggregate_preserves_frozen_order_and_publishes_filtered_split(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    train_ids = ["train-a", "train-b"]
    val_ids = ["val-a"]
    source_records = [
        {
            "sample_key": sample_id,
            "system_id": f"sys-{sample_id}",
            "split": split,
            "status": "eligible",
        }
        for split, ids in (("train", train_ids), ("val", val_ids))
        for sample_id in ids
    ]
    input_manifest = {
        "inputs": {"val_bank_manifest": {"sha256": "a" * 64}},
        "inventory": {
            split: {
                "full_count": len(ids),
                "eligible_count": len(ids),
                "excluded_count": 0,
                "full_ids_sha256": bank._ordered_ids_sha256(ids),
                "eligible_ids_sha256": bank._ordered_ids_sha256(ids),
                "excluded_ids_sha256": bank._ordered_ids_sha256([]),
                "eligible_ids": ids,
            }
            for split, ids in (("train", train_ids), ("val", val_ids))
        },
        "records": source_records,
    }
    monkeypatch.setattr(bank, "_load_input_manifest", lambda *_args, **_kwargs: input_manifest)

    def fake_load_shard(
        _output_root: Path, *, split: str, shard_index: int, **_kwargs: Any
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        ids = train_ids if split == "train" else val_ids
        records = [
            {
                "sample_key": sample_id,
                "system_id": f"sys-{sample_id}",
                "split": split,
                "status": "complete",
                "sampling_seed": 43 + index,
                "pt_path": str(tmp_path / f"{sample_id}.pt"),
                "pt_sha256": "b" * 64,
                "size_bytes": 1,
                "pose_count": bank.NUM_SAMPLES,
                "pose_ensemble_sha256": "c" * 64,
            }
            for index, sample_id in enumerate(ids)
        ]
        return (
            {
                "selection_mode": "full",
                "claim_eligible": True,
                "max_records": None,
                "records": records,
                "record_count": len(records),
            },
            {
                "path": str(tmp_path / f"{split}-{shard_index}.json"),
                "sha256": "d" * 64,
                "size_bytes": 1,
            },
        )

    monkeypatch.setattr(bank, "_load_shard_summary", fake_load_shard)
    args = argparse.Namespace(
        input_manifest=tmp_path / "inputs.json",
        expected_input_manifest_sha256="e" * 64,
        expected_builder_sha256="f" * 64,
        output_root=tmp_path / "bank",
        num_train_shards=1,
        num_val_shards=1,
        pose_tag=bank.DEFAULT_POSE_TAG,
        filtered_split_output=tmp_path / "filtered.json",
        output_manifest=tmp_path / "manifest.json",
        allow_smoke_subset=False,
    )

    # Simulate interruption after the filtered split was linked but before the
    # aggregate manifest commit marker was published.
    expected_filtered = {"train": train_ids, "val": val_ids}
    bank._atomic_write_noreplace(
        args.filtered_split_output, bank._canonical_json_bytes(expected_filtered)
    )
    result = bank.aggregate(args)
    first_manifest_bytes = args.output_manifest.read_bytes()
    recovered = bank.aggregate(args)

    assert result["status"] == "complete"
    assert result["claim_eligible"] is True
    assert result["study_protocol_id"] == bank.STUDY_PROTOCOL_ID
    assert json.loads(args.filtered_split_output.read_text()) == {
        "train": train_ids,
        "val": val_ids,
    }
    assert [record["sample_key"] for record in result["records"]] == train_ids + val_ids
    assert all(record["system_id"] for record in result["records"])
    assert recovered == result
    assert args.output_manifest.read_bytes() == first_manifest_bytes


def test_atomic_json_writer_refuses_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "sealed.json"
    bank._atomic_write_noreplace(output, b"{}\n")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        bank._atomic_write_noreplace(output, b"{}\n")
