from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pytest
from rdkit import Chem

sys.path.insert(0, str(Path(__file__).parents[1]))

from scripts import report_early_time_sampler_plinder_confidence as report


def _sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report._file_sha256(path)


def _write_sdf(path: Path, record: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    molecule = Chem.MolFromSmiles("CC")
    assert molecule is not None
    conformer = Chem.Conformer(molecule.GetNumAtoms())
    for atom_index in range(molecule.GetNumAtoms()):
        conformer.SetAtomPosition(atom_index, (float(atom_index), 0.0, 0.0))
    molecule.AddConformer(conformer)
    writer = Chem.SDWriter(str(path))
    for sample_index in range(report.POSE_COUNT):
        pose = Chem.Mol(molecule)
        for key, value in {
            "sample_index": sample_index,
            "sampling_seed": record["sampling_seed"],
            "ligand_conformer_seed": record["ligand_conformer_seed"],
            "candidate_ensemble_sha256": record["candidate_ensemble_sha256"],
            "sample_sigma": report.SAMPLE_SIGMA,
        }.items():
            pose.SetProp(key, str(value))
        writer.write(pose)
    writer.close()
    return report._file_sha256(path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return report._file_sha256(path)


@pytest.fixture
def frozen_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    monkeypatch.setattr(report, "FULL_SPLIT_COUNT", 5)
    monkeypatch.setattr(report, "FULL_ELIGIBLE_COUNT", 4)
    monkeypatch.setattr(report, "FULL_SYSTEM_COUNT", 3)
    monkeypatch.setattr(report, "FULL_EXCLUDED_COUNT", 1)
    monkeypatch.setattr(report, "EXPECTED_SHARDS", 2)
    monkeypatch.setattr(report, "POSE_COUNT", 4)
    monkeypatch.setattr(report, "SMOKE_COUNT", 2)
    monkeypatch.setattr(report, "BOOTSTRAP_RESAMPLES", 200)
    monkeypatch.setattr(report, "BOOTSTRAP_BATCH_SIZE", 32)

    eligible_ids = ["sys0__L", "sys1__L", "sys2__L", "sys2__M"]
    excluded_ids = ["zz_excluded__L"]
    full_ids = sorted(eligible_ids + excluded_ids)
    system_by_id = {
        "sys0__L": "sys0",
        "sys1__L": "sys1",
        "sys2__L": "sys2",
        "sys2__M": "sys2",
    }
    ligand_by_id = {sample_id: sample_id.rsplit("__", 1)[1] for sample_id in eligible_ids}
    rmsds_by_id = {
        "sys0__L": [1.0, 3.0, 4.0, 5.0],
        "sys1__L": [3.0, 1.0, 4.0, 5.0],
        "sys2__L": [3.0, 4.0, 1.0, 5.0],
        "sys2__M": [3.0, 4.0, 5.0, 6.0],
    }
    eligibility_payload = {
        "schema_version": report.ELIGIBILITY_SCHEMA_VERSION,
        "protocol_id": report.SOURCE_PROTOCOL_ID,
        "status": "complete",
        "inventory": {
            "full_count": len(full_ids),
            "full_ids": full_ids,
            "full_ids_sha256": report._ids_sha256(full_ids),
            "eligible_count": len(eligible_ids),
            "eligible_ids": eligible_ids,
            "eligible_ids_sha256": report._ids_sha256(eligible_ids),
            "eligible_system_count": 3,
            "excluded_count": len(excluded_ids),
            "excluded_ids": excluded_ids,
            "excluded_ids_sha256": report._ids_sha256(excluded_ids),
            "preflight_error_count": 0,
            "preflight_error_ids": [],
        },
    }
    eligibility_path = tmp_path / "eligibility.json"
    eligibility_sha = _write_json(eligibility_path, eligibility_payload)
    monkeypatch.setattr(report, "FROZEN_ELIGIBILITY_MANIFEST_SHA256", eligibility_sha)

    source_run_root = tmp_path / "source_run"
    source_output_root = source_run_root / "full"
    source_output_root.mkdir(parents=True)
    source_report_payload = {
        "schema_version": report.SOURCE_REPORT_SCHEMA_VERSION,
        "protocol_id": report.SOURCE_PROTOCOL_ID,
        "status": "complete_selection_fail",
        "stage": "full",
        "configuration": {
            "expected_shards": report.EXPECTED_SHARDS,
            "num_samples": report.POSE_COUNT,
            "selected_count": report.FULL_ELIGIBLE_COUNT,
            "system_count": report.FULL_SYSTEM_COUNT,
        },
        "integrity": {
            "exact_arm_inventory": True,
            "zero_runtime_failures": True,
            "eligibility_manifest_sha256": eligibility_sha,
            "all_pose_sdf_artifact_audit": {
                "files_verified": report.FULL_ELIGIBLE_COUNT * 3,
                "records_verified": report.FULL_ELIGIBLE_COUNT * report.POSE_COUNT * 3,
                "sha256_recomputed": True,
            },
        },
    }
    source_report_path = source_run_root / "reports" / "full_strict_report.json"
    source_report_sha = _write_json(source_report_path, source_report_payload)
    monkeypatch.setattr(report, "FROZEN_SOURCE_REPORT_SHA256", source_report_sha)

    source_audit_payload = {
        "schema_version": report.SOURCE_AUDIT_SCHEMA_VERSION,
        "protocol_id": report.SOURCE_PROTOCOL_ID,
        "status": "complete",
        "eligibility_manifest": {"sha256": eligibility_sha},
        "inventory": {
            "shard_count": report.EXPECTED_SHARDS,
            "eligible_sample_count": report.FULL_ELIGIBLE_COUNT,
            "excluded_sample_count": report.FULL_EXCLUDED_COUNT,
            "candidate_count_per_sample": report.POSE_COUNT,
            "audited_csv_rows": report.FULL_ELIGIBLE_COUNT * 3,
            "parsed_sdf_records": report.FULL_ELIGIBLE_COUNT * report.POSE_COUNT * 3,
            "system_count": report.FULL_SYSTEM_COUNT,
        },
    }
    source_audit_path = source_run_root / "reports" / "coordinate_audit_v3.json"
    source_audit_sha = _write_json(source_audit_path, source_audit_payload)
    monkeypatch.setattr(report, "FROZEN_SOURCE_AUDIT_SHA256", source_audit_sha)

    protocol_path = tmp_path / "protocol.md"
    protocol_path.write_text("# frozen synthetic protocol\n", encoding="utf-8")
    protocol_sha = report._file_sha256(protocol_path)
    source_protocol_path = tmp_path / "source_protocol.md"
    source_protocol_path.write_text("# frozen synthetic source protocol\n", encoding="utf-8")
    source_protocol_sha = report._file_sha256(source_protocol_path)
    monkeypatch.setattr(report, "FROZEN_SOURCE_PROTOCOL_SHA256", source_protocol_sha)

    frozen_asset_dir = tmp_path / "frozen_assets"
    frozen_asset_dir.mkdir()
    config_path = frozen_asset_dir / "config.yaml"
    config_path.write_text("synthetic: true\n", encoding="utf-8")
    confidence_path = frozen_asset_dir / "confidence.pt"
    confidence_path.write_bytes(b"synthetic confidence checkpoint")
    backbone_paths: dict[str, Path] = {}
    for arm in report.ARMS:
        backbone_path = frozen_asset_dir / f"{arm}.pt"
        backbone_path.write_bytes(f"synthetic {arm}".encode("utf-8"))
        backbone_paths[arm] = backbone_path.resolve()
    scorer_source_path = frozen_asset_dir / "scorer.py"
    scorer_source_path.write_text("# synthetic scorer\n", encoding="utf-8")
    scorer_source_sha = report._file_sha256(scorer_source_path)
    report_source_path = Path(report.__file__).resolve()
    report_source_sha = report._file_sha256(report_source_path)
    runtime_code_sha = _sha_text("synthetic runtime code identity")

    global_index = {sample_id: full_ids.index(sample_id) + 1 for sample_id in eligible_ids}
    bank_records: list[dict[str, Any]] = []
    source_rows_by_id: dict[str, dict[str, Any]] = {}
    for sample_id in eligible_ids:
        source_shard_index = eligible_ids.index(sample_id) % report.EXPECTED_SHARDS
        record = {
            "sample_key": sample_id,
            "system_id": system_by_id[sample_id],
            "ligand_chain": ligand_by_id[sample_id],
            "plinder_global_index": global_index[sample_id],
            "sampling_seed": 42 + global_index[sample_id],
            "ligand_conformer_seed": 0,
            "source_shard_index": source_shard_index,
            "pose_count": report.POSE_COUNT,
            "sample_sigma": report.SAMPLE_SIGMA,
            "num_steps": report.NUM_STEPS,
            "candidate_ensemble_sha256": _sha_text(f"candidate:{sample_id}"),
            "prior_pool_sha256": _sha_text(f"prior:{sample_id}"),
            "prior_pool_size": report.PRIOR_POOL_SIZE,
            "ligand_input_identity_sha256": report._canonical_smiles_identity("CC"),
            "canonical_smiles": "CC",
            "pocket_center": [0.0, 0.0, 0.0],
            "num_input_atoms": 2,
        }
        receptor_path = tmp_path / "assets" / "receptors" / f"{sample_id}.pdb"
        receptor_path.parent.mkdir(parents=True, exist_ok=True)
        receptor_path.write_text(f"RECEPTOR {sample_id}\n", encoding="utf-8")
        receptor_sha = report._file_sha256(receptor_path)
        processed_meta_path = tmp_path / "assets" / "meta" / f"{sample_id}.json"
        processed_meta_sha = _write_json(
            processed_meta_path,
            {"sample_key": sample_id, "kind": "synthetic_non_outcome_metadata"},
        )
        record.update(
            {
                "receptor": {
                    "path": str(receptor_path.resolve()),
                    "sha256": receptor_sha,
                },
                "receptor_sha256": receptor_sha,
                "processed_meta": {
                    "path": str(processed_meta_path.resolve()),
                    "sha256": processed_meta_sha,
                },
                "processed_meta_sha256": processed_meta_sha,
            }
        )
        sdf_path = (
            source_output_root
            / f"shard-{source_shard_index:03d}-of-{report.EXPECTED_SHARDS:03d}"
            / "arms"
            / report.SOURCE_ARM
            / "poses"
            / "all_poses"
            / f"{sample_id}.sdf"
        )
        sdf_sha = _write_sdf(sdf_path, record)
        record["all_poses_sdf"] = {"path": str(sdf_path.resolve()), "sha256": sdf_sha}
        bank_records.append(record)
        rmsds = rmsds_by_id[sample_id]
        source_rows_by_id[sample_id] = {
            "id": sample_id,
            "arm": report.SOURCE_ARM,
            "plinder_system_id": system_by_id[sample_id],
            "plinder_ligand_chain": ligand_by_id[sample_id],
            "plinder_global_index": global_index[sample_id],
            "sampling_seed": record["sampling_seed"],
            "ligand_conformer_seed": 0,
            "num_samples": report.POSE_COUNT,
            "all_poses_count": report.POSE_COUNT,
            "candidate_ensemble_sha256": record["candidate_ensemble_sha256"],
            "prior_pool_sha256": record["prior_pool_sha256"],
            "prior_pool_size": report.PRIOR_POOL_SIZE,
            "protein": str(receptor_path.resolve()),
            "protein_sha256": receptor_sha,
            "processed_meta": str(processed_meta_path.resolve()),
            "processed_meta_sha256": processed_meta_sha,
            "ligand_input_canonical_smiles": record["canonical_smiles"],
            "ligand_input_identity_sha256": record["ligand_input_identity_sha256"],
            "num_input_atoms": record["num_input_atoms"],
            "all_poses_sdf": str(sdf_path.resolve()),
            "all_poses_sdf_sha256": sdf_sha,
            "checkpoint_sha256": report.FROZEN_BACKBONE_SHA256[report.PRIMARY_ARM],
            "selector_profile": "candidate_only",
            "sampling_dynamics": "deterministic_ode",
            "candidate_rmsds_json": json.dumps(rmsds),
            "candidate_fast_valid_json": json.dumps([True, True, True, True]),
            "num_rmsd_lt2_candidates": sum(value < 2.0 for value in rmsds),
        }

    source_shards: list[dict[str, Any]] = []
    for shard_index in range(report.EXPECTED_SHARDS):
        ids = eligible_ids[shard_index:: report.EXPECTED_SHARDS]
        shard_dir = (
            source_output_root
            / f"shard-{shard_index:03d}-of-{report.EXPECTED_SHARDS:03d}"
        )
        paired_path = shard_dir / "paired_summary.json"
        paired_sha = _write_json(
            paired_path,
            {"shard_index": shard_index, "kind": "synthetic_paired_summary"},
        )
        csv_path = shard_dir / "arms" / report.SOURCE_ARM / "results.csv"
        csv_sha = _write_csv(csv_path, [source_rows_by_id[sample_id] for sample_id in ids])
        source_shards.append(
            {
                "shard_index": shard_index,
                "paired_summary": {
                    "path": str(paired_path.resolve()),
                    "sha256": paired_sha,
                },
                "results_csv": {"path": str(csv_path.resolve()), "sha256": csv_sha},
                "assigned_count": len(ids),
                "assigned_ids_sha256": report._ids_sha256(ids),
            }
        )

    bank_payload = {
        "schema_version": report.BANK_SCHEMA_VERSION,
        "protocol_id": report.PROTOCOL_ID,
        "source_protocol_id": report.SOURCE_PROTOCOL_ID,
        "status": "complete_label_free",
        "created_at_utc": "2026-08-16T00:00:00+00:00",
        "information_boundary": {
            "source_csv_allowlist": sorted(report.SOURCE_ROW_ALLOWLIST),
            "outcome_columns_exported": False,
            "score_stage_reads_source_results_csv": False,
            "crystal_reference_exported": False,
        },
        "bank_root": str(source_output_root.resolve()),
        "inputs": {
            "protocol_document": {
                "path": str(protocol_path.resolve()),
                "sha256": protocol_sha,
            },
            "eligibility_manifest": {
                "path": str(eligibility_path.resolve()),
                "sha256": eligibility_sha,
            },
            "source_sampler_report": {
                "path": str(source_report_path.resolve()),
                "sha256": source_report_sha,
            },
            "source_coordinate_audit": {
                "path": str(source_audit_path.resolve()),
                "sha256": source_audit_sha,
            },
            "source_sampler_protocol": {
                "path": str(source_protocol_path.resolve()),
                "sha256": source_protocol_sha,
            },
            "config": {
                "path": str(config_path.resolve()),
                "sha256": report.FROZEN_CONFIG_SHA256,
            },
            "s50_backbone_checkpoint": {
                "path": str(backbone_paths[report.PRIMARY_ARM]),
                "sha256": report.FROZEN_BACKBONE_SHA256[report.PRIMARY_ARM],
            },
            "matched_backbone_checkpoint": {
                "path": str(backbone_paths[report.DIAGNOSTIC_ARM]),
                "sha256": report.FROZEN_BACKBONE_SHA256[report.DIAGNOSTIC_ARM],
            },
            "confidence_checkpoint": {
                "path": str(confidence_path.resolve()),
                "sha256": report.FROZEN_CONFIDENCE_SHA256,
            },
            "scorer_source": {
                "path": str(scorer_source_path.resolve()),
                "sha256": scorer_source_sha,
            },
            "report_source": {
                "path": str(report_source_path),
                "sha256": report_source_sha,
            },
            "source_output_root": str(source_output_root.resolve()),
            "runtime_code_identity": {
                "aggregate_sha256": runtime_code_sha,
                "files": {
                    "scorer": {
                        "path": str(scorer_source_path.resolve()),
                        "sha256": scorer_source_sha,
                        "size_bytes": scorer_source_path.stat().st_size,
                    }
                },
            },
        },
        "backbone_arms": {
            arm: {
                "path": str(backbone_paths[arm]),
                "sha256": report.FROZEN_BACKBONE_SHA256[arm],
                "role": (
                    "primary_deployment_backbone"
                    if arm == report.PRIMARY_ARM
                    else "diagnostic_training_matched_backbone"
                ),
            }
            for arm in report.ARMS
        },
        "fixed_settings": {
            "source_arm": report.SOURCE_ARM,
            "pose_count": report.POSE_COUNT,
            "sample_sigma": report.SAMPLE_SIGMA,
            "num_steps": report.NUM_STEPS,
            "prior_pool_size": report.PRIOR_POOL_SIZE,
            "pose_batch_size": 20,
            "pocket_cutoff_angstrom": 10.0,
            "ligand_conformer_seed": 0,
            "selector": report.SELECTOR,
            "label_blind": True,
            "resampling": False,
        },
        "inventory": {
            "full_count": report.FULL_SPLIT_COUNT,
            "eligible_count": report.FULL_ELIGIBLE_COUNT,
            "eligible_system_count": report.FULL_SYSTEM_COUNT,
            "excluded_count": report.FULL_EXCLUDED_COUNT,
            "source_shard_count": report.EXPECTED_SHARDS,
            "pose_count": report.POSE_COUNT,
            "full_ids_sha256": report._ids_sha256(full_ids),
            "eligible_ids_sha256": report._ids_sha256(eligible_ids),
            "excluded_ids_sha256": report._ids_sha256(excluded_ids),
        },
        "source_shards": source_shards,
        "records": bank_records,
    }
    bank_path = tmp_path / "bank.json"
    bank_sha = _write_json(bank_path, bank_payload)

    base = {
        "root": tmp_path,
        "eligible_ids": eligible_ids,
        "bank_by_id": {row["sample_key"]: row for row in bank_records},
        "protocol_file": protocol_path,
        "protocol_sha256": protocol_sha,
        "label_free_bank_manifest": bank_path,
        "label_free_bank_manifest_sha256": bank_sha,
        "eligibility_manifest": eligibility_path,
        "eligibility_manifest_sha256": eligibility_sha,
        "source_sampler_report": source_report_path,
        "source_sampler_report_sha256": source_report_sha,
        "source_coordinate_audit": source_audit_path,
        "source_coordinate_audit_sha256": source_audit_sha,
        "source_sampler_protocol_sha256": source_protocol_sha,
        "scorer_source_sha256": scorer_source_sha,
        "runtime_code_identity_sha256": runtime_code_sha,
        "backbone_paths": backbone_paths,
        "confidence_path": confidence_path.resolve(),
    }
    return base


def _score_arrays(sample_id: str, arm: str) -> dict[str, list[float]]:
    selected_by_id = {
        report.PRIMARY_ARM: {
            "sys0__L": 0,
            "sys1__L": 1,
            "sys2__L": 0,
            "sys2__M": 0,
        },
        report.DIAGNOSTIC_ARM: {
            "sys0__L": 1,
            "sys1__L": 1,
            "sys2__L": 2,
            "sys2__M": 0,
        },
    }
    selected = selected_by_id[arm][sample_id]
    predicted = [2.0 + index for index in range(report.POSE_COUNT)]
    predicted[selected] = 0.1
    success = [0.1] * report.POSE_COUNT
    success[selected] = 0.9
    return {
        "confidence_rmsd": predicted,
        "confidence_success_logit": [-2.0, -1.0, 0.0, 1.0],
        "confidence_success": success,
        "confidence_atom_rmsd": [1.0, 2.0, 3.0, 4.0],
        "confidence_atom_q90": [1.5, 2.5, 3.5, 4.5],
        "confidence_atom_ok": [0.9, 0.8, 0.7, 0.6],
    }


def _write_score_artifacts(fixture: dict[str, Any], *, stage: str) -> Path:
    scores_root = fixture["root"] / "scores"
    report_sha = report._file_sha256(Path(report.__file__).resolve())
    pins = {
        "protocol_sha256": fixture["protocol_sha256"],
        "label_free_bank_manifest_sha256": fixture["label_free_bank_manifest_sha256"],
        "eligibility_manifest_sha256": fixture["eligibility_manifest_sha256"],
        "source_sampler_report_sha256": fixture["source_sampler_report_sha256"],
        "source_coordinate_audit_sha256": fixture["source_coordinate_audit_sha256"],
        "source_sampler_protocol_sha256": fixture["source_sampler_protocol_sha256"],
        "report_source_sha256": report_sha,
        "scorer_source_sha256": fixture["scorer_source_sha256"],
        "config_sha256": report.FROZEN_CONFIG_SHA256,
        "confidence_checkpoint_sha256": report.FROZEN_CONFIDENCE_SHA256,
        "runtime_code_identity_sha256": fixture["runtime_code_identity_sha256"],
    }
    selected_by_shard = report._expected_ids_by_shard(fixture["eligible_ids"], stage)
    for shard_index in range(report.EXPECTED_SHARDS):
        ids = selected_by_shard[shard_index]
        for arm in report.ARMS:
            records: list[dict[str, Any]] = []
            for sample_id in ids:
                bank = fixture["bank_by_id"][sample_id]
                arrays = _score_arrays(sample_id, arm)
                selected = min(
                    range(report.POSE_COUNT),
                    key=lambda index: (arrays["confidence_rmsd"][index], index),
                )
                records.append(
                    {
                        **bank,
                        "score_arrays": arrays,
                        "selected_index": selected,
                        "score_ledger_sha256": report._score_ledger_sha256(arrays),
                    }
                )
            payload = {
                "schema_version": report.SCORE_SCHEMA_VERSION,
                "protocol_id": report.PROTOCOL_ID,
                "status": "complete",
                "mode": "full_shard" if stage == "full" else "smoke_replay",
                "stage": stage,
                "arm": arm,
                "arm_role": (
                    "primary_deployment_backbone"
                    if arm == report.PRIMARY_ARM
                    else "diagnostic_training_matched_backbone"
                ),
                "selector": report.SELECTOR,
                "created_at_utc": "2026-08-16T00:00:00+00:00",
                "fixed_settings": {
                    "saved_pose_bank_only": True,
                    "resampling": False,
                    "sample_sigma": report.SAMPLE_SIGMA,
                    "pose_count": report.POSE_COUNT,
                    "num_steps": report.NUM_STEPS,
                    "prior_pool_size": report.PRIOR_POOL_SIZE,
                    "pose_batch_size": 20,
                    "t1_hidden_backbone": arm,
                    "pocket_cutoff_angstrom": 10.0,
                    "selector": report.SELECTOR,
                    "label_blind": True,
                },
                "inputs": {
                    **pins,
                    "label_free_bank_manifest": str(
                        Path(fixture["label_free_bank_manifest"]).resolve()
                    ),
                    "backbone_checkpoint": str(fixture["backbone_paths"][arm]),
                    "backbone_checkpoint_sha256": report.FROZEN_BACKBONE_SHA256[arm],
                    "confidence_checkpoint": str(fixture["confidence_path"]),
                },
                "inventory": {
                    "eligible_count": report.FULL_ELIGIBLE_COUNT,
                    "source_shard_count": report.EXPECTED_SHARDS,
                    "source_shard_index": shard_index,
                    "assigned_count": len(ids),
                    "scored_count": len(ids),
                    "assigned_ids_sha256": report._ids_sha256(ids),
                },
                "records": records,
                "replay": (
                    {}
                    if stage == "full"
                    else {
                        "passed": True,
                        "checked_count": len(ids),
                        "selected_index_mismatches": 0,
                        "all_scores_finite": True,
                        "per_field_max_abs_score_delta": {
                            field: 0.0 for field in report.SCORE_ARRAY_FIELDS
                        },
                        "max_abs_score_delta": 0.0,
                        "absolute_tolerance": report.REPLAY_ABS_TOLERANCE,
                        "records": [
                            {
                                "sample_key": sample_id,
                                "first_selected_index": records[index]["selected_index"],
                                "replay_selected_index": records[index]["selected_index"],
                                "selected_index_stable": True,
                                "replay_score_ledger_sha256": records[index][
                                    "score_ledger_sha256"
                                ],
                            }
                            for index, sample_id in enumerate(ids)
                        ],
                    }
                ),
                "runtime": {
                    "elapsed_seconds": 1.0,
                    "finished_at_utc": "2026-08-16T00:00:01+00:00",
                    "slurm_job_id": "synthetic",
                    "slurm_array_job_id": "synthetic",
                    "slurm_array_task_id": str(shard_index),
                    "cuda_device_name": "synthetic-gpu",
                    "torch_version": "synthetic",
                    "torch_cuda_version": "synthetic",
                    "rdkit_version": "synthetic",
                },
            }
            path = report._canonical_score_path(scores_root, stage, shard_index, arm)
            _write_json(path, payload)
    return scores_root


def _kwargs(fixture: dict[str, Any], scores_root: Path, stage: str) -> dict[str, Any]:
    return {
        "stage": stage,
        "scores_root": scores_root,
        **{
            key: fixture[key]
            for key in (
                "label_free_bank_manifest",
                "label_free_bank_manifest_sha256",
                "eligibility_manifest",
                "eligibility_manifest_sha256",
                "source_sampler_report",
                "source_sampler_report_sha256",
                "source_coordinate_audit",
                "source_coordinate_audit_sha256",
                "protocol_file",
                "protocol_sha256",
            )
        },
    }


def test_full_report_strict_join_metrics_and_decision(frozen_fixture: dict[str, Any]) -> None:
    scores_root = _write_score_artifacts(frozen_fixture, stage="full")
    result = report.build_report(**_kwargs(frozen_fixture, scores_root, "full"))

    assert result["schema_version"] == report.REPORT_SCHEMA_VERSION
    assert result["status"] == "complete_diagnostic"
    assert result["integrity_only"] is False
    assert result["integrity"]["score_shard_count"] == 4
    primary = result["arms"][report.PRIMARY_ARM]
    assert primary["top1_success_count"] == 2
    assert primary["oracle_success_count"] == 3
    assert primary["oracle_recovery_fraction"] == pytest.approx(2 / 3)
    assert primary["selection_miss_count"] == 1
    assert primary["sampler_unreachable_count"] == 1
    assert primary["bottleneck_delta_count"] == 0
    assert result["decision"]["oracle_recovery_band"] == "severe_selection_bottleneck"
    assert result["paired_backbone"]["point_pp"] == pytest.approx(0.0)
    json.dumps(result, allow_nan=False)


def test_smoke_report_is_integrity_only(frozen_fixture: dict[str, Any]) -> None:
    scores_root = _write_score_artifacts(frozen_fixture, stage="smoke")
    result = report.build_report(**_kwargs(frozen_fixture, scores_root, "smoke"))

    assert result["status"] == "passed"
    assert result["integrity_only"] is True
    assert result["expected_shards"] == 2
    assert result["integrity"]["efficacy_emitted"] is False
    assert result["integrity"]["source_label_artifacts_opened"] is False
    assert result["integrity"]["source_sampler_reports_parsed"] is False
    assert (
        result["integrity"]["source_csv_shards"]
        == "not_opened_in_outcome_blind_smoke"
    )
    assert "arms" not in result
    assert "decision" not in result
    assert "paired_backbone" not in result


def test_selector_index_is_recomputed_fail_closed(frozen_fixture: dict[str, Any]) -> None:
    scores_root = _write_score_artifacts(frozen_fixture, stage="full")
    path = report._canonical_score_path(scores_root, "full", 0, report.PRIMARY_ARM)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["records"][0]["selected_index"] = 3
    _write_json(path, payload)

    with pytest.raises(ValueError, match="selector recomputation mismatch"):
        report.build_report(**_kwargs(frozen_fixture, scores_root, "full"))


def test_score_arbitrary_label_field_is_rejected(frozen_fixture: dict[str, Any]) -> None:
    scores_root = _write_score_artifacts(frozen_fixture, stage="full")
    path = report._canonical_score_path(scores_root, "full", 0, report.PRIMARY_ARM)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["records"][0]["rmsd_label"] = 1.0
    _write_json(path, payload)

    with pytest.raises(ValueError, match="exact field inventory mismatch"):
        report.build_report(**_kwargs(frozen_fixture, scores_root, "full"))


def test_bank_arbitrary_label_field_is_rejected(frozen_fixture: dict[str, Any]) -> None:
    bank_path = Path(frozen_fixture["label_free_bank_manifest"])
    payload = json.loads(bank_path.read_text(encoding="utf-8"))
    payload["records"][0]["rmsd_label"] = 1.0
    frozen_fixture["label_free_bank_manifest_sha256"] = _write_json(bank_path, payload)
    scores_root = _write_score_artifacts(frozen_fixture, stage="smoke")

    with pytest.raises(ValueError, match="exact field inventory mismatch"):
        report.build_report(**_kwargs(frozen_fixture, scores_root, "smoke"))


def test_score_runtime_identity_must_match_bank(frozen_fixture: dict[str, Any]) -> None:
    scores_root = _write_score_artifacts(frozen_fixture, stage="smoke")
    path = report._canonical_score_path(scores_root, "smoke", 0, report.PRIMARY_ARM)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["inputs"]["runtime_code_identity_sha256"] = _sha_text("different runtime")
    _write_json(path, payload)

    with pytest.raises(ValueError, match="runtime code identity differs"):
        report.build_report(**_kwargs(frozen_fixture, scores_root, "smoke"))


def test_smoke_never_parses_source_outcomes(
    frozen_fixture: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    scores_root = _write_score_artifacts(frozen_fixture, stage="smoke")
    source_json_paths = {
        Path(frozen_fixture["source_sampler_report"]),
        Path(frozen_fixture["source_coordinate_audit"]),
    }
    original_loader = report._load_json_object

    def guarded_loader(path: Path, *, label: str) -> dict[str, Any]:
        if path in source_json_paths:
            raise AssertionError(f"smoke parsed forbidden source JSON: {label}")
        return original_loader(path, label=label)

    def forbidden_source_csv_read(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("smoke opened a source results CSV")

    monkeypatch.setattr(report, "_load_json_object", guarded_loader)
    monkeypatch.setattr(report, "_read_source_rows", forbidden_source_csv_read)

    result = report.build_report(**_kwargs(frozen_fixture, scores_root, "smoke"))
    assert result["status"] == "passed"
    assert result["integrity"]["efficacy_emitted"] is False


def test_mutated_source_sdf_is_rejected(frozen_fixture: dict[str, Any]) -> None:
    scores_root = _write_score_artifacts(frozen_fixture, stage="full")
    sdf_path = Path(frozen_fixture["bank_by_id"]["sys0__L"]["all_poses_sdf"]["path"])
    sdf_path.write_bytes(sdf_path.read_bytes() + b"\n")

    with pytest.raises(ValueError, match="source SDF changed after bank freeze"):
        report.build_report(**_kwargs(frozen_fixture, scores_root, "full"))


def test_missing_paired_backbone_fails(frozen_fixture: dict[str, Any]) -> None:
    scores_root = _write_score_artifacts(frozen_fixture, stage="smoke")
    path = report._canonical_score_path(scores_root, "smoke", 1, report.DIAGNOSTIC_ARM)
    path.unlink()

    with pytest.raises(FileNotFoundError, match="matched_backbone"):
        report.build_report(**_kwargs(frozen_fixture, scores_root, "smoke"))


def test_atomic_writer_refuses_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "report.json"
    report._atomic_write_json_noreplace(output, {"status": "first"})
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        report._atomic_write_json_noreplace(output, {"status": "second"})
    assert json.loads(output.read_text(encoding="utf-8")) == {"status": "first"}
