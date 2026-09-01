from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign

sys.path.insert(0, str(Path(__file__).parents[1]))

from scripts import audit_early_time_sampler_plinder_k2_full as full_audit


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _asset(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "sha256": _sha(path),
        "size_bytes": path.stat().st_size,
    }


def _versioned(ids: list[str]) -> str:
    return full_audit._versioned_ids_sha256(ids)


def _write_molecule(path: Path, molecule: Chem.Mol) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = Chem.SDWriter(str(path))
    writer.write(molecule)
    writer.close()


def _base_molecule() -> Chem.Mol:
    molecule = Chem.MolFromSmiles("CCO")
    assert molecule is not None
    conformer = Chem.Conformer(molecule.GetNumAtoms())
    for index, xyz in enumerate(((0.0, 0.0, 0.0), (1.4, 0.0, 0.0), (2.1, 0.8, 0.0))):
        conformer.SetAtomPosition(index, xyz)
    molecule.AddConformer(conformer)
    return molecule


def _translated(molecule: Chem.Mol, offset: float) -> Chem.Mol:
    result = Chem.Mol(molecule)
    conformer = result.GetConformer()
    for index in range(result.GetNumAtoms()):
        position = conformer.GetAtomPosition(index)
        conformer.SetAtomPosition(index, (position.x + offset, position.y, position.z))
    return result


def _write_all_poses(
    path: Path,
    *,
    molecule: Chem.Mol,
    offsets: list[float],
    sample_id: str,
    sampling_seed: int,
    ensemble_hash: str,
) -> list[Chem.Mol]:
    path.parent.mkdir(parents=True, exist_ok=True)
    poses = [_translated(molecule, offset) for offset in offsets]
    writer = Chem.SDWriter(str(path))
    for index, pose in enumerate(poses):
        properties = {
            "sample_index": index,
            "complex_id": sample_id,
            "dataset": "plinder_val",
            "sampling_seed": sampling_seed,
            "ligand_conformer_seed": 0,
            "num_samples": len(offsets),
            "num_steps": 2,
            "candidate_ensemble_sha256": ensemble_hash,
            "fast_valid": True,
        }
        for name, value in properties.items():
            pose.SetProp(name, str(value))
        writer.write(pose)
    writer.close()
    return poses


def _csv_bytes(rows: list[dict[str, object]]) -> bytes:
    fields = sorted({field for row in rows for field in row})
    from io import StringIO

    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode()


def _write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _settings() -> dict[str, object]:
    return {
        "stage": "full",
        "selected_count": None,
        "num_samples": 4,
        "num_steps": 2,
        "model_pose_step_budget": 8,
        "sigma": 2.0,
        "prior_pool_size": 4,
        "time_schedule": "late",
        "schedule_power": 3.0,
        "pocket_cutoff_angstrom": 10.0,
        "center_jitter_sigma": 0.0,
        "confidence": False,
        "vina_selection": False,
        "vina_guidance_scale": 0.0,
        "unified_guidance_scale": 0.0,
        "fk_constraint_beta": 0.0,
        "fk_resample_times": [],
        "translation_sde_base_sigma": 0.0,
        "sampling_dynamics": "deterministic_ode",
        "refine": "none",
        "selector_profile": "candidate_only",
        "ligand_conformer_seed": 0,
        "include_s50_replay": False,
    }


@pytest.fixture
def synthetic_full_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    eligible_ids = ["sys0__L", "sys1__L", "sys2__L", "sys2__M"]
    excluded_ids = ["zzz__L"]
    full_ids = sorted(eligible_ids + excluded_ids)
    monkeypatch.setattr(full_audit, "EXPECTED_SHARDS", 2)
    monkeypatch.setattr(full_audit, "EXPECTED_FULL_COUNT", len(full_ids))
    monkeypatch.setattr(full_audit, "EXPECTED_ELIGIBLE_COUNT", len(eligible_ids))
    monkeypatch.setattr(full_audit, "EXPECTED_EXCLUDED_COUNT", len(excluded_ids))
    monkeypatch.setattr(full_audit, "EXPECTED_SYSTEM_COUNT", 3)
    monkeypatch.setattr(full_audit, "EXPECTED_NUM_SAMPLES", 4)
    monkeypatch.setattr(full_audit, "EXPECTED_NUM_STEPS", 2)
    monkeypatch.setattr(full_audit, "EXPECTED_PRIOR_POOL_SIZE", 4)
    monkeypatch.setattr(full_audit, "BOOTSTRAP_RESAMPLES", 256)
    monkeypatch.setattr(
        full_audit,
        "EXPECTED_ELIGIBLE_NEWLINE_SHA256",
        full_audit._newline_ids_sha256(eligible_ids),
    )

    assets = tmp_path / "assets"
    assets.mkdir()
    fixed_asset_names = (
        "protocol_document",
        "split",
        "pool_parquet",
        "config",
        "raw_gate",
        "conformer_mapping_audit",
    )
    fixed: dict[str, object] = {}
    for name in fixed_asset_names:
        path = assets / f"{name}.bin"
        path.write_bytes(name.encode())
        fixed[name] = _asset(path)
    checkpoint_hashes: dict[str, str] = {}
    checkpoint_assets: dict[str, dict[str, object]] = {}
    for arm in full_audit.ARMS:
        path = assets / f"{arm}.pt"
        path.write_bytes(f"checkpoint:{arm}".encode())
        checkpoint_assets[arm] = _asset(path)
        checkpoint_hashes[arm] = str(checkpoint_assets[arm]["sha256"])
    monkeypatch.setattr(full_audit, "CHECKPOINT_SHA256", checkpoint_hashes)
    fixed["checkpoints"] = checkpoint_assets
    code_path = assets / "runner.py"
    code_path.write_text("# synthetic runner\n")
    code_files = {"runner.py": _asset(code_path)}
    code_hashes = {name: str(record["sha256"]) for name, record in code_files.items()}
    code_digest = hashlib.sha256(b"EFFDOCK_PLINDER_PAIRED_CODE_INVENTORY_V1\0")
    code_digest.update(json.dumps(code_hashes, separators=(",", ":"), sort_keys=True).encode())
    fixed["code"] = {
        "contract": "EFFDOCK_PLINDER_PAIRED_CODE_INVENTORY_V1",
        "sha256": code_digest.hexdigest(),
        "files": code_files,
    }

    molecule = _base_molecule()
    records: list[dict[str, object]] = []
    eligible_records: dict[str, dict[str, object]] = {}
    for global_index, sample_id in enumerate(full_ids, start=1):
        sample_root = assets / sample_id
        sample_root.mkdir()
        protein = sample_root / "protein.pdb"
        protein.write_text(
            "ATOM      1  C   ALA A   1     100.000 100.000 100.000  1.00 20.00           C  \n"
        )
        reference = sample_root / "ligand.sdf"
        _write_molecule(reference, molecule)
        meta = sample_root / "meta.pt"
        meta.write_bytes(b"synthetic-meta")
        system_id, chain = sample_id.rsplit("__", 1)
        record: dict[str, object] = {
            "sample_key": sample_id,
            "status": "eligible" if sample_id in eligible_ids else "excluded",
            "global_index": global_index,
            "sampling_seed": 42 + global_index,
            "ligand_conformer_seed": 0,
            "system_id": system_id,
            "ligand_chain": chain,
            "canonical_smiles": "CCO",
            "canonical_smiles_identity_sha256": full_audit._canonical_smiles_identity(
                "CCO"
            ),
            "receptor": _asset(protein),
            "ligand_reference": _asset(reference),
            "processed_meta": _asset(meta),
        }
        if sample_id in eligible_ids:
            record["audit_mapping_method"] = "strict_stereo"
            record["audit_symmetry_complete"] = True
        records.append(record)
        if sample_id in eligible_ids:
            eligible_records[sample_id] = record
    eligibility = {
        "schema_version": full_audit.ELIGIBILITY_SCHEMA,
        "protocol_id": full_audit.PROTOCOL_ID,
        "status": "complete",
        "inputs": {"fixed_identities": fixed},
        "inventory": {
            "full_count": len(full_ids),
            "full_ids": full_ids,
            "full_ids_sha256": _versioned(full_ids),
            "eligible_count": len(eligible_ids),
            "eligible_ids": eligible_ids,
            "eligible_ids_sha256": _versioned(eligible_ids),
            "eligible_ids_newline_sha256": full_audit._newline_ids_sha256(eligible_ids),
            "eligible_system_count": 3,
            "excluded_count": len(excluded_ids),
            "excluded_ids": excluded_ids,
            "excluded_ids_sha256": _versioned(excluded_ids),
            "preflight_error_count": 0,
            "preflight_error_ids": [],
        },
        "records": records,
    }
    eligibility_path = tmp_path / "eligibility.json"
    eligibility_path.write_text(json.dumps(eligibility, sort_keys=True) + "\n")
    eligibility_sha = _sha(eligibility_path)

    run_root = tmp_path / "synthetic-pathfix1" / "full"
    rows_by_key: dict[tuple[int, str, str], dict[str, object]] = {}
    for shard_index in range(2):
        shard_dir = run_root / f"shard-{shard_index:03d}-of-002"
        assigned = eligible_ids[shard_index::2]
        artifact_arms: dict[str, object] = {}
        for arm in full_audit.ARMS:
            rows: list[dict[str, object]] = []
            for sample_id in assigned:
                record = eligible_records[sample_id]
                global_index = int(record["global_index"])
                offsets = (
                    [-0.5, 1.5, 3.5, 5.5]
                    if arm == full_audit.TREATMENT_ARM
                    else [1.0, 3.0, 5.0, 7.0]
                )
                ensemble_hash = hashlib.sha256(f"ensemble:{arm}:{sample_id}".encode()).hexdigest()
                all_poses = (
                    shard_dir / "arms" / arm / "poses" / "all_poses" / f"{sample_id}.sdf"
                )
                poses = _write_all_poses(
                    all_poses,
                    molecule=molecule,
                    offsets=offsets,
                    sample_id=sample_id,
                    sampling_seed=42 + global_index,
                    ensemble_hash=ensemble_hash,
                )
                selected = (
                    shard_dir / "arms" / arm / "poses" / "selected" / f"{sample_id}.sdf"
                )
                _write_molecule(selected, poses[0])
                reference = full_audit._load_reference(Path(str(record["ligand_reference"]["path"])))
                rmsds = [float(rdMolAlign.CalcRMS(pose, reference)) for pose in poses]
                methods = ["rdkit_calc_rms_symmetry_no_align"] * len(poses)
                mapping_metadata = full_audit._full_heavy_atom_graph_metadata(
                    reference,
                    poses[0],
                    list(range(poses[0].GetNumAtoms())),
                    list(range(reference.GetNumAtoms())),
                    "strict",
                )
                coordinates = np.stack(
                    [np.asarray(pose.GetConformer().GetPositions()) for pose in poses], axis=0
                )
                atomic_numbers = np.asarray(
                    [atom.GetAtomicNum() for atom in poses[0].GetAtoms()]
                )
                diversity = full_audit._diversity_metrics(coordinates, atomic_numbers)
                k2 = sum(value < 2.0 for value in rmsds)
                row: dict[str, object] = {
                    "id": sample_id,
                    "arm": arm,
                    "plinder_system_id": sample_id.rsplit("__", 1)[0],
                    "plinder_ligand_chain": sample_id.rsplit("__", 1)[1],
                    "plinder_global_index": global_index,
                    "sampling_seed": 42 + global_index,
                    "ligand_conformer_seed": 0,
                    "num_samples": 4,
                    "all_poses_count": 4,
                    "prior_pool_size": 4,
                    "selector_profile": "candidate_only",
                    "guidance_mode": "none",
                    "sampling_dynamics": "deterministic_ode",
                    "translation_sde_base_sigma": 0.0,
                    "full_heavy_atom_bijection": True,
                    "checkpoint": checkpoint_assets[arm]["path"],
                    "checkpoint_sha256": checkpoint_hashes[arm],
                    "protein": record["receptor"]["path"],
                    "protein_sha256": record["receptor"]["sha256"],
                    "ligand_ref": record["ligand_reference"]["path"],
                    "ligand_reference_sha256": record["ligand_reference"]["sha256"],
                    "processed_meta": record["processed_meta"]["path"],
                    "processed_meta_sha256": record["processed_meta"]["sha256"],
                    "ligand_input_identity_sha256": record[
                        "canonical_smiles_identity_sha256"
                    ],
                    "ligand_input_canonical_smiles": record["canonical_smiles"],
                    "prior_pool_sha256": hashlib.sha256(
                        f"prior:{sample_id}".encode()
                    ).hexdigest(),
                    "candidate_ensemble_sha256": ensemble_hash,
                    "all_poses_sdf": str(all_poses.resolve()),
                    "all_poses_sdf_sha256": _sha(all_poses),
                    "saved_pose_sha256_json": json.dumps({"selected": _sha(selected)}),
                    "candidate_rmsds_json": json.dumps(rmsds),
                    "candidate_rmsd_method_json": json.dumps(methods),
                    "candidate_fast_valid_json": json.dumps([True] * 4),
                    "num_rmsd_lt2_candidates": k2,
                    "num_fast_valid_candidates": 4,
                    "num_fast_valid_rmsd_lt2_candidates": k2,
                    "num_mapped_index_rmsd_fallback_candidates": 0,
                    "first_index": 0,
                    "selected_index": 0,
                    "first_rmsd": rmsds[0],
                    "selected_rmsd": rmsds[0],
                    "oracle_rmsd": min(rmsds),
                    "mean_sample_rmsd": sum(rmsds) / 4,
                    "match_method": "strict",
                    "num_match_atoms": molecule.GetNumAtoms(),
                    "num_input_atoms": molecule.GetNumAtoms(),
                    "num_ref_atoms": reference.GetNumAtoms(),
                    "ligand_graph_relation": mapping_metadata["relation"],
                    "exact_full_heavy_atom_graph": True,
                    "ligand_mapping_metadata_json": json.dumps(mapping_metadata),
                    "pose_diversity_contract": full_audit.POSE_DIVERSITY_CONTRACT,
                    "pose_diversity_round_decimals": 3,
                    **diversity,
                }
                rows.append(row)
                rows_by_key[(shard_index, arm, sample_id)] = row
            results_path = shard_dir / "arms" / arm / "results.csv"
            csv_data = _csv_bytes(rows)
            _write_bytes(results_path, csv_data)
            arm_summary_path = shard_dir / "arms" / arm / "summary.json"
            arm_summary = {
                "arm": arm,
                "count": len(assigned),
                "results_csv": str(results_path.resolve()),
                "results_csv_sha256": hashlib.sha256(csv_data).hexdigest(),
            }
            arm_summary_path.write_text(json.dumps(arm_summary, sort_keys=True) + "\n")
            artifact_arms[arm] = {
                **arm_summary,
                "summary": str(arm_summary_path.resolve()),
            }
        paired_summary = {
            "schema_version": full_audit.SHARD_SCHEMA,
            "protocol_id": full_audit.PROTOCOL_ID,
            "status": "complete",
            "run_id": run_root.parent.name,
            "mode": "full",
            "settings": _settings(),
            "eligibility_manifest": {
                "path": str(eligibility_path.resolve()),
                "sha256": eligibility_sha,
                "eligible_count": len(eligible_ids),
                "eligible_ids_newline_sha256": full_audit._newline_ids_sha256(
                    eligible_ids
                ),
                "eligible_system_count": 3,
                "ineligible_count": len(excluded_ids),
            },
            "inventory": {
                "full_count": len(full_ids),
                "eligible_count": len(eligible_ids),
                "selected_count": len(eligible_ids),
                "selected_ids": eligible_ids,
                "selected_ids_sha256": _versioned(eligible_ids),
                "num_shards": 2,
                "shard_index": shard_index,
                "assigned_count": len(assigned),
                "assigned_ids": assigned,
                "assigned_ids_sha256": _versioned(assigned),
                "arm_success_counts": {arm: len(assigned) for arm in full_audit.ARMS},
            },
            "operational_inventory": {
                "requested_count": len(full_ids),
                "evaluable_count": len(eligible_ids),
                "common_preprocessing_failure_count": len(excluded_ids),
                "common_preprocessing_failure_ids": excluded_ids,
                "operational_sensitivity_assignment": (
                    "common preprocessing failures have K2=0"
                ),
            },
            "arms": [
                {"name": arm, "checkpoint_sha256": checkpoint_hashes[arm]}
                for arm in full_audit.ARMS
            ],
            "fixed_identities": fixed,
            "paired_identity_gate": {"passed": True, "checked_count": len(assigned)},
            "replay_integrity_gate": {"required": False, "passed": True},
            "failures": [],
            "artifacts": {
                "paired_summary": str((shard_dir / "paired_summary.json").resolve()),
                "arms": artifact_arms,
            },
        }
        (shard_dir / "paired_summary.json").write_text(
            json.dumps(paired_summary, sort_keys=True) + "\n"
        )
    return {
        "run_root": run_root,
        "eligibility_path": eligibility_path,
        "eligibility_sha": eligibility_sha,
        "records": eligible_records,
        "rows": rows_by_key,
    }


def _rewrite_sdf(path: Path, molecules: list[Chem.Mol]) -> None:
    writer = Chem.SDWriter(str(path))
    for molecule in molecules:
        writer.write(molecule)
    writer.close()


def _read_sdf(path: Path) -> list[Chem.Mol]:
    with path.open("rb") as handle:
        return [
            molecule
            for molecule in Chem.ForwardSDMolSupplier(
                handle, sanitize=True, removeHs=True, strictParsing=True
            )
            if molecule is not None
        ]


def _direct_audit(
    fixture: dict[str, object], row: dict[str, object], *, sample_id: str, arm: str
) -> full_audit.RowAudit:
    shard_dir = Path(fixture["run_root"]) / "shard-000-of-002"
    return full_audit._audit_row(
        {key: str(value) if not isinstance(value, str) else value for key, value in row.items()},
        arm=arm,
        shard_dir=shard_dir,
        expected_id=sample_id,
        expected_global_index=int(fixture["records"][sample_id]["global_index"]),
        manifest_record=fixture["records"][sample_id],
        hashes=full_audit.FileHashCache(),
        fast_valid_recheck=False,
    )


def test_synthetic_full_audit_recomputes_inventory_coordinates_and_decision(
    synthetic_full_run: dict[str, object],
) -> None:
    report = full_audit.audit_full(
        run_root=Path(synthetic_full_run["run_root"]),
        eligibility_manifest=Path(synthetic_full_run["eligibility_path"]),
        eligibility_manifest_sha256=str(synthetic_full_run["eligibility_sha"]),
        fast_valid_mode="off",
        progress_every=0,
    )
    assert report["status"] == "complete"
    assert report["inventory"]["audited_csv_rows"] == 12
    assert report["inventory"]["parsed_sdf_records"] == 48
    assert report["primary_comparison"]["delta_mean_k2"] == pytest.approx(1.0)
    assert report["operational_full_split_sensitivity"]["primary_comparison"][
        "delta_mean_k2"
    ] == pytest.approx(0.8)
    assert report["decision"]["passed"] is True
    baseline_metrics = report["arms"][full_audit.BASELINE_ARM][
        "saved_coordinate_metrics"
    ]
    assert baseline_metrics["coordinate_unique_fraction"] == 1.0
    assert baseline_metrics["nearest_neighbor_median_mean"] == pytest.approx(2.0)
    assert baseline_metrics["c2_component_mean"] == pytest.approx(4.0)
    assert report["coordinate_quantization_sensitivity"]["ambiguous_pair_edge_count"] > 0
    assert report["coordinate_quantization_sensitivity"]["decision_stable"] is True


def test_coordinate_tamper_fails_even_after_sdf_hash_is_updated(
    synthetic_full_run: dict[str, object],
) -> None:
    sample_id = "sys0__L"
    arm = full_audit.BASELINE_ARM
    row = dict(synthetic_full_run["rows"][(0, arm, sample_id)])
    path = Path(str(row["all_poses_sdf"]))
    molecules = _read_sdf(path)
    position = molecules[0].GetConformer().GetAtomPosition(0)
    molecules[0].GetConformer().SetAtomPosition(
        0, (position.x + 0.02, position.y, position.z)
    )
    _rewrite_sdf(path, molecules)
    row["all_poses_sdf_sha256"] = _sha(path)
    with pytest.raises(full_audit.AuditError, match="coordinate RMSD mismatch"):
        _direct_audit(synthetic_full_run, row, sample_id=sample_id, arm=arm)


@pytest.mark.parametrize("mutation", ["property", "record_count"])
def test_sdf_property_and_record_count_tampering_fail_closed(
    synthetic_full_run: dict[str, object], mutation: str
) -> None:
    sample_id = "sys0__L"
    arm = full_audit.BASELINE_ARM
    row = dict(synthetic_full_run["rows"][(0, arm, sample_id)])
    path = Path(str(row["all_poses_sdf"]))
    molecules = _read_sdf(path)
    if mutation == "property":
        molecules[1].SetProp("sampling_seed", "999")
    else:
        molecules = molecules[:-1]
    _rewrite_sdf(path, molecules)
    row["all_poses_sdf_sha256"] = _sha(path)
    expected = "SDF sampling_seed mismatch" if mutation == "property" else "SDF record count"
    with pytest.raises(full_audit.AuditError, match=expected):
        _direct_audit(synthetic_full_run, row, sample_id=sample_id, arm=arm)


def test_two_angstrom_quantization_flip_is_ledgered_not_rejected(
    synthetic_full_run: dict[str, object],
) -> None:
    sample_id = "sys0__L"
    arm = full_audit.BASELINE_ARM
    row = dict(synthetic_full_run["rows"][(0, arm, sample_id)])
    path = Path(str(row["all_poses_sdf"]))
    molecules = _read_sdf(path)
    reference = full_audit._load_reference(
        Path(str(synthetic_full_run["records"][sample_id]["ligand_reference"]["path"]))
    )
    reference_coordinates = np.asarray(reference.GetConformer().GetPositions())
    conformer = molecules[1].GetConformer()
    for atom_index, xyz in enumerate(reference_coordinates):
        conformer.SetAtomPosition(atom_index, (xyz[0] + 2.0001, xyz[1], xyz[2]))
    _rewrite_sdf(path, molecules)
    row["all_poses_sdf_sha256"] = _sha(path)
    saved_rmsds = [float(rdMolAlign.CalcRMS(molecule, reference)) for molecule in molecules]
    declared_rmsds = list(saved_rmsds)
    declared_rmsds[1] = 1.9998
    row["candidate_rmsds_json"] = json.dumps(declared_rmsds)
    row["num_rmsd_lt2_candidates"] = sum(value < 2.0 for value in declared_rmsds)
    row["num_fast_valid_rmsd_lt2_candidates"] = row["num_rmsd_lt2_candidates"]
    row["first_rmsd"] = declared_rmsds[0]
    row["selected_rmsd"] = declared_rmsds[0]
    row["oracle_rmsd"] = min(declared_rmsds)
    row["mean_sample_rmsd"] = sum(declared_rmsds) / 4
    coordinates = np.stack(
        [np.asarray(molecule.GetConformer().GetPositions()) for molecule in molecules], axis=0
    )
    diversity = full_audit._diversity_metrics(
        coordinates,
        np.asarray([atom.GetAtomicNum() for atom in molecules[0].GetAtoms()]),
    )
    row.update(diversity)
    result = _direct_audit(synthetic_full_run, row, sample_id=sample_id, arm=arm)
    assert result.declared_k2 == result.k2 + 1
    assert any(index == 1 for index, _, _ in result.quantization_ambiguous_candidates)


def test_later_candidate_atom_reordering_is_rejected(
    synthetic_full_run: dict[str, object],
) -> None:
    sample_id = "sys0__L"
    arm = full_audit.BASELINE_ARM
    row = dict(synthetic_full_run["rows"][(0, arm, sample_id)])
    path = Path(str(row["all_poses_sdf"]))
    molecules = _read_sdf(path)
    original = molecules[1]
    renumbered = Chem.RenumberAtoms(original, [1, 0, 2])
    for property_name in original.GetPropNames(includePrivate=True):
        renumbered.SetProp(property_name, original.GetProp(property_name))
    molecules[1] = renumbered
    _rewrite_sdf(path, molecules)
    row["all_poses_sdf_sha256"] = _sha(path)
    with pytest.raises(full_audit.AuditError, match="ordered topology changed"):
        _direct_audit(synthetic_full_run, row, sample_id=sample_id, arm=arm)


def test_later_candidate_hydrogen_radical_state_mutation_is_rejected(
    synthetic_full_run: dict[str, object],
) -> None:
    sample_id = "sys0__L"
    arm = full_audit.BASELINE_ARM
    row = dict(synthetic_full_run["rows"][(0, arm, sample_id)])
    path = Path(str(row["all_poses_sdf"]))
    molecules = _read_sdf(path)
    atom = molecules[1].GetAtomWithIdx(0)
    atom.SetNoImplicit(True)
    atom.SetNumExplicitHs(2)
    atom.SetNumRadicalElectrons(1)
    Chem.SanitizeMol(molecules[1])
    _rewrite_sdf(path, molecules)
    row["all_poses_sdf_sha256"] = _sha(path)
    with pytest.raises(full_audit.AuditError, match="ordered topology changed"):
        _direct_audit(synthetic_full_run, row, sample_id=sample_id, arm=arm)


def test_full_atom_substructure_without_connectivity_identity_is_rejected(
    synthetic_full_run: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    sample_id = "sys0__L"
    arm = full_audit.BASELINE_ARM
    row = dict(synthetic_full_run["rows"][(0, arm, sample_id)])
    cyclic_reference = Chem.MolFromSmiles("C1CO1")
    assert cyclic_reference is not None
    conformer = Chem.Conformer(cyclic_reference.GetNumAtoms())
    for index, xyz in enumerate(((0.0, 0.0, 0.0), (1.4, 0.0, 0.0), (0.7, 1.2, 0.0))):
        conformer.SetAtomPosition(index, xyz)
    cyclic_reference.AddConformer(conformer)
    pose = _read_sdf(Path(str(row["all_poses_sdf"])))[0]
    dock_indices, ref_indices, method = full_audit._match_atoms(cyclic_reference, pose)
    metadata = full_audit._full_heavy_atom_graph_metadata(
        cyclic_reference, pose, dock_indices, ref_indices, method
    )
    assert metadata["full_bijection"] is True
    assert metadata["atom_elements_match"] is True
    assert metadata["connectivity_match"] is False
    assert metadata["accepted"] is False
    monkeypatch.setattr(full_audit, "_load_reference", lambda _: cyclic_reference)
    with pytest.raises(full_audit.AuditError, match="graph/connectivity gate failed"):
        _direct_audit(synthetic_full_run, row, sample_id=sample_id, arm=arm)


def test_saved_sdf_topology_must_match_frozen_canonical_smiles(
    synthetic_full_run: dict[str, object],
) -> None:
    sample_id = "sys0__L"
    arm = full_audit.BASELINE_ARM
    row = dict(synthetic_full_run["rows"][(0, arm, sample_id)])
    record = dict(synthetic_full_run["records"][sample_id])
    record["canonical_smiles"] = "C1CO1"
    record["canonical_smiles_identity_sha256"] = full_audit._canonical_smiles_identity(
        "C1CO1"
    )
    row["ligand_input_canonical_smiles"] = "C1CO1"
    row["ligand_input_identity_sha256"] = record["canonical_smiles_identity_sha256"]
    shard_dir = Path(synthetic_full_run["run_root"]) / "shard-000-of-002"
    with pytest.raises(full_audit.AuditError, match="frozen canonical SMILES"):
        full_audit._audit_row(
            {key: str(value) if not isinstance(value, str) else value for key, value in row.items()},
            arm=arm,
            shard_dir=shard_dir,
            expected_id=sample_id,
            expected_global_index=int(record["global_index"]),
            manifest_record=record,
            hashes=full_audit.FileHashCache(),
            fast_valid_recheck=False,
        )


@pytest.mark.parametrize(
    ("frozen_smiles", "different_smiles"),
    (
        ("CC=O", "C=CO"),
        ("CC(=O)O", "CC(=O)[O-]"),
        ("[13CH3]CO", "CCO"),
        ("[CH3]", "C"),
        ("[O]", "O"),
        ("[NH]", "N"),
    ),
)
def test_frozen_canonical_graph_rejects_non_stereo_chemical_change(
    frozen_smiles: str,
    different_smiles: str,
) -> None:
    frozen = Chem.MolFromSmiles(frozen_smiles)
    different = Chem.MolFromSmiles(different_smiles)
    aromatic = Chem.MolFromSmiles("c1ccccc1")
    kekule = Chem.MolFromSmiles("C1=CC=CC=C1")
    assert frozen is not None and different is not None
    assert aromatic is not None and kekule is not None
    assert not full_audit._matches_frozen_canonical_graph(frozen, different)
    assert full_audit._matches_frozen_canonical_graph(aromatic, kekule)


def test_generated_pose_stereo_is_not_an_ordered_topology_mutation(
    tmp_path: Path,
) -> None:
    canonical = Chem.MolFromSmiles("F[C@H](Cl)Br")
    opposite = Chem.MolFromSmiles("F[C@@H](Cl)Br")
    assert canonical is not None and opposite is not None
    embedded: list[Chem.Mol] = []
    for seed, molecule in enumerate((canonical, opposite), start=7):
        with_hydrogens = Chem.AddHs(molecule)
        assert AllChem.EmbedMolecule(with_hydrogens, randomSeed=seed) == 0
        embedded.append(Chem.RemoveHs(with_hydrogens))
    path = tmp_path / "opposite-stereo-records.sdf"
    writer = Chem.SDWriter(str(path))
    for molecule in embedded:
        writer.write(molecule)
    writer.close()
    records = _read_sdf(path)
    centers = [
        Chem.FindMolChiralCenters(
            molecule, includeUnassigned=True, useLegacyImplementation=False
        )
        for molecule in records
    ]
    assert centers[0] != centers[1]
    assert full_audit._topology_signature(records[0]) == full_audit._topology_signature(
        records[1]
    )
    assert all(
        full_audit._matches_frozen_canonical_graph(canonical, molecule)
        for molecule in records
    )

    trans = Chem.MolFromSmiles("C/C=C/C")
    cis = Chem.MolFromSmiles("C/C=C\\C")
    assert trans is not None and cis is not None
    assert full_audit._topology_signature(trans) == full_audit._topology_signature(cis)
    assert full_audit._matches_frozen_canonical_graph(trans, cis)


@pytest.mark.parametrize("canonical_smiles", ("CC(O)F", "CC=CC"))
def test_unspecified_stereo_survives_3d_sdf_round_trip(
    tmp_path: Path,
    canonical_smiles: str,
) -> None:
    canonical = Chem.MolFromSmiles(canonical_smiles)
    assert canonical is not None
    embedded = Chem.AddHs(canonical)
    assert AllChem.EmbedMolecule(embedded, randomSeed=7) == 0
    embedded = Chem.RemoveHs(embedded)
    path = tmp_path / "roundtrip.sdf"
    _write_molecule(path, embedded)
    round_tripped = _read_sdf(path)[0]
    assert full_audit._matches_frozen_canonical_graph(canonical, round_tripped)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("audit_mapping_method", "loose", "not strict_stereo"),
        ("audit_symmetry_complete", False, "symmetry audit is incomplete"),
        ("canonical_smiles_identity_sha256", "0" * 64, "canonical-SMILES identity"),
    ),
)
def test_eligibility_requires_strict_stereo_symmetry_and_smiles_identity(
    synthetic_full_run: dict[str, object],
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    payload = json.loads(Path(synthetic_full_run["eligibility_path"]).read_text())
    record = next(entry for entry in payload["records"] if entry["sample_key"] == "sys0__L")
    record[field] = value
    path = tmp_path / f"bad-{field}.json"
    path.write_text(json.dumps(payload, sort_keys=True) + "\n")
    with pytest.raises(full_audit.AuditError, match=message):
        full_audit._load_eligibility(
            path,
            _sha(path),
            hashes=full_audit.FileHashCache(),
        )


def test_selected_pose_must_equal_candidate_zero(
    synthetic_full_run: dict[str, object],
) -> None:
    sample_id = "sys0__L"
    arm = full_audit.BASELINE_ARM
    row = dict(synthetic_full_run["rows"][(0, arm, sample_id)])
    selected = (
        Path(synthetic_full_run["run_root"])
        / "shard-000-of-002"
        / "arms"
        / arm
        / "poses"
        / "selected"
        / f"{sample_id}.sdf"
    )
    molecule = _read_sdf(selected)[0]
    position = molecule.GetConformer().GetAtomPosition(0)
    molecule.GetConformer().SetAtomPosition(0, (position.x + 0.1, position.y, position.z))
    _rewrite_sdf(selected, [molecule])
    row["saved_pose_sha256_json"] = json.dumps({"selected": _sha(selected)})
    with pytest.raises(
        full_audit.AuditError, match="selected pose coordinates differ from candidate 0"
    ):
        _direct_audit(synthetic_full_run, row, sample_id=sample_id, arm=arm)


def test_sampled_fast_valid_coordinate_mismatch_is_explicitly_non_gating(
    synthetic_full_run: dict[str, object],
) -> None:
    sample_id = "sys0__L"
    arm = full_audit.BASELINE_ARM
    row = dict(synthetic_full_run["rows"][(0, arm, sample_id)])
    path = Path(str(row["all_poses_sdf"]))
    molecules = _read_sdf(path)
    molecules[0].SetProp("fast_valid", "False")
    _rewrite_sdf(path, molecules)
    row["all_poses_sdf_sha256"] = _sha(path)
    row["candidate_fast_valid_json"] = json.dumps([False, True, True, True])
    row["num_fast_valid_candidates"] = 3
    row["num_fast_valid_rmsd_lt2_candidates"] = 0
    shard_dir = Path(synthetic_full_run["run_root"]) / "shard-000-of-002"
    result = full_audit._audit_row(
        {key: str(value) if not isinstance(value, str) else value for key, value in row.items()},
        arm=arm,
        shard_dir=shard_dir,
        expected_id=sample_id,
        expected_global_index=int(synthetic_full_run["records"][sample_id]["global_index"]),
        manifest_record=synthetic_full_run["records"][sample_id],
        hashes=full_audit.FileHashCache(),
        fast_valid_recheck=True,
    )
    assert result.fast_valid_recheck_mismatch_indices == (0,)


def test_symlink_and_surplus_csv_fail_closed(tmp_path: Path) -> None:
    target = tmp_path / "target.bin"
    target.write_bytes(b"target")
    alias = tmp_path / "alias.bin"
    alias.symlink_to(target)
    with pytest.raises(full_audit.AuditError, match="symlink artifacts are forbidden"):
        full_audit._require_file_identity(
            str(alias),
            _sha(target),
            label="alias",
            hashes=full_audit.FileHashCache(),
        )
    csv_path = tmp_path / "surplus.csv"
    csv_path.write_text("id\na,b\n")
    with pytest.raises(full_audit.AuditError, match="surplus columns"):
        full_audit._read_csv(csv_path)


def test_published_summary_path_rejects_resolving_symlink_alias(
    synthetic_full_run: dict[str, object],
) -> None:
    run_root = Path(synthetic_full_run["run_root"])
    paired_summary_path = run_root / "shard-000-of-002" / "paired_summary.json"
    alias = run_root.parent.parent / "paired-summary-alias.json"
    alias.symlink_to(paired_summary_path)
    summary = json.loads(paired_summary_path.read_text())
    summary["artifacts"]["paired_summary"] = str(alias)
    paired_summary_path.write_text(json.dumps(summary, sort_keys=True) + "\n")
    with pytest.raises(full_audit.AuditError, match="symlink artifacts are forbidden"):
        full_audit.audit_full(
            run_root=run_root,
            eligibility_manifest=Path(synthetic_full_run["eligibility_path"]),
            eligibility_manifest_sha256=str(synthetic_full_run["eligibility_sha"]),
            fast_valid_mode="off",
            progress_every=0,
        )


def test_exact_root_inventory_rejects_extra_artifact(
    synthetic_full_run: dict[str, object],
) -> None:
    (Path(synthetic_full_run["run_root"]) / "unexpected.txt").write_text("unexpected")
    with pytest.raises(full_audit.AuditError, match="unexpected artifact"):
        full_audit.audit_full(
            run_root=Path(synthetic_full_run["run_root"]),
            eligibility_manifest=Path(synthetic_full_run["eligibility_path"]),
            eligibility_manifest_sha256=str(synthetic_full_run["eligibility_sha"]),
            fast_valid_mode="off",
            progress_every=0,
        )


def test_quantization_decision_instability_is_complete_but_never_promotes() -> None:
    coordinate = {
        "passed": False,
        "failed_gates": ["efficacy_mean_k2"],
        "gates": {"efficacy_mean_k2": {"passed": False}},
    }
    csv_bound = {
        "passed": True,
        "failed_gates": [],
        "gates": {"efficacy_mean_k2": {"passed": True}},
    }
    effective, changed, unexplained = full_audit._reconcile_quantized_decisions(
        coordinate,
        csv_bound,
        candidate_flip_count=1,
        c2_changed_row_count=0,
        ambiguous_edge_count=0,
    )
    assert changed == ["efficacy_mean_k2"]
    assert unexplained == []
    assert effective["selection_eligible"] is False
    assert effective["passed"] is False
    assert effective["action"].startswith("keep_s50")


def test_unexplained_quantization_decision_flip_fails() -> None:
    coordinate = {
        "passed": False,
        "failed_gates": ["nearest_neighbor_ratio"],
        "gates": {"nearest_neighbor_ratio": {"passed": False}},
    }
    csv_bound = {
        "passed": True,
        "failed_gates": [],
        "gates": {"nearest_neighbor_ratio": {"passed": True}},
    }
    with pytest.raises(full_audit.AuditError, match="not explained"):
        full_audit._reconcile_quantized_decisions(
            coordinate,
            csv_bound,
            candidate_flip_count=1,
            c2_changed_row_count=0,
            ambiguous_edge_count=0,
        )


def test_no_overwrite_json_output(tmp_path: Path) -> None:
    output = tmp_path / "audit.json"
    first_sha = full_audit._write_noreplace(output, {"status": "complete"})
    assert first_sha == _sha(output)
    with pytest.raises(full_audit.AuditError, match="refusing to overwrite"):
        full_audit._write_noreplace(output, {"status": "different"})
    assert json.loads(output.read_text()) == {"status": "complete"}


def test_cluster_bootstrap_is_fixed_seed_and_clustered(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(full_audit, "BOOTSTRAP_RESAMPLES", 128)
    rows: list[full_audit.RowAudit] = []
    for index, (sample_id, system_id) in enumerate(
        (("a__L", "a"), ("a__M", "a"), ("b__L", "b")), start=1
    ):
        rows.append(
            full_audit.RowAudit(
                sample_id=sample_id,
                system_id=system_id,
                arm="x",
                global_index=index,
                sampling_seed=42 + index,
                k2=1,
                declared_k2=1,
                fast_valid_k2=1,
                declared_fast_valid_k2=1,
                fast_valid_count=4,
                first_rmsd=1.0,
                declared_first_rmsd=1.0,
                oracle_rmsd=1.0,
                declared_oracle_rmsd=1.0,
                coordinate_unique_count=4,
                nearest_neighbor_rmsd_median=2.0,
                declared_nearest_neighbor_rmsd_median=2.0,
                c2_component_count=4,
                declared_c2_component_count=4,
                pair_identity=("same",),
                all_poses_sha256="0" * 64,
                fast_valid_recheck_candidates=0,
                fast_valid_recheck_mismatches=0,
                fast_valid_recheck_mismatch_indices=(),
                quantization_ambiguous_candidates=(),
                quantization_ambiguous_edges=(),
            )
        )
    treatment = [
        full_audit.replace(row, k2=row.k2 + (1 if row.system_id == "a" else 0))
        for row in rows
    ]
    first = full_audit._cluster_bootstrap(rows, treatment)
    second = full_audit._cluster_bootstrap(rows, treatment)
    assert first == second
    assert first["cluster_count"] == 2
    assert first["k2_delta"]["ci95_low"] >= 0.0
    rng = np.random.Generator(np.random.PCG64(full_audit.BOOTSTRAP_SEED))
    indices = rng.integers(0, 2, size=(128, 2))
    counts = np.asarray([2.0, 1.0])[indices].sum(axis=1)
    deltas = np.asarray([2.0, 0.0])[indices].sum(axis=1)
    expected_low, expected_high = np.percentile(deltas / counts, [2.5, 97.5])
    assert first["k2_delta"]["ci95_low"] == pytest.approx(expected_low)
    assert first["k2_delta"]["ci95_high"] == pytest.approx(expected_high)
