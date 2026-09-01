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

from scripts import audit_early_time_sampler_plinder_confidence_full as audit


def _sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return audit._file_sha256(path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return audit._file_sha256(path)


def _runtime_aggregate(files: dict[str, dict[str, Any]]) -> str:
    hashes = {name: str(value["sha256"]) for name, value in files.items()}
    digest = hashlib.sha256(audit.RUNTIME_CODE_DOMAIN)
    digest.update(audit._canonical_json_bytes(hashes))
    return digest.hexdigest()


@pytest.fixture
def synthetic_full(tmp_path: Path) -> dict[str, Any]:
    contract = audit.AuditContract(
        full_count=9,
        eligible_count=8,
        eligible_system_count=7,
        excluded_count=1,
        shard_count=8,
        pose_count=4,
        bootstrap_resamples=200,
        bootstrap_batch_size=31,
    )
    eligible_ids = [f"s{index:02d}__L" for index in range(contract.eligible_count)]
    excluded_ids = ["zz_excluded__L"]
    full_ids = sorted(eligible_ids + excluded_ids)
    systems = {
        sample_id: ("shared" if index in {4, 5} else f"system-{index}")
        for index, sample_id in enumerate(eligible_ids)
    }
    rmsds = {
        eligible_ids[0]: [1.0, 3.0, 4.0, 5.0],
        eligible_ids[1]: [3.0, 1.0, 4.0, 5.0],
        eligible_ids[2]: [3.0, 4.0, 1.0, 5.0],
        eligible_ids[3]: [3.0, 4.0, 5.0, 1.0],
        eligible_ids[4]: [1.0, 1.5, 4.0, 5.0],
        eligible_ids[5]: [3.0, 4.0, 5.0, 6.0],
        eligible_ids[6]: [1.9, 3.0, 4.0, 5.0],
        eligible_ids[7]: [3.0, 1.9, 4.0, 5.0],
    }

    protocol = tmp_path / "protocol.md"
    protocol.write_text("# frozen synthetic confidence protocol\n", encoding="utf-8")
    protocol_sha = audit._file_sha256(protocol)
    source_protocol = tmp_path / "source_protocol.md"
    source_protocol.write_text("# frozen synthetic source protocol\n", encoding="utf-8")
    source_protocol_sha = audit._file_sha256(source_protocol)
    config = tmp_path / "assets" / "config.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("synthetic: true\n", encoding="utf-8")
    confidence = tmp_path / "assets" / "confidence.pt"
    confidence.write_bytes(b"synthetic confidence")
    backbones: dict[str, Path] = {}
    for arm in audit.ARMS:
        path = tmp_path / "assets" / f"{arm}.pt"
        path.write_bytes(f"synthetic {arm}".encode())
        backbones[arm] = path.resolve()
    scorer_source = tmp_path / "assets" / "scorer.py"
    scorer_source.write_text("# frozen synthetic scorer\n", encoding="utf-8")
    report_source = tmp_path / "assets" / "reporter.py"
    report_source.write_text("# frozen synthetic reporter\n", encoding="utf-8")

    source_report = tmp_path / "source_run" / "reports" / "strict_source.json"
    source_report_sha = _write_json(source_report, {"status": "synthetic-frozen"})
    source_audit = tmp_path / "source_run" / "reports" / "coordinate_audit.json"
    source_audit_sha = _write_json(source_audit, {"status": "synthetic-frozen"})
    source_root = tmp_path / "source_run" / "full"
    source_root.mkdir(parents=True)

    eligibility = tmp_path / "source_run" / "eligibility.json"
    eligibility_payload = {
        "schema_version": audit.ELIGIBILITY_SCHEMA_VERSION,
        "protocol_id": audit.SOURCE_PROTOCOL_ID,
        "status": "complete",
        "inventory": {
            "full_count": contract.full_count,
            "eligible_count": contract.eligible_count,
            "eligible_system_count": contract.eligible_system_count,
            "excluded_count": contract.excluded_count,
            "preflight_error_count": 0,
            "preflight_error_ids": [],
            "full_ids": full_ids,
            "full_ids_sha256": audit._ids_sha256(full_ids),
            "eligible_ids": eligible_ids,
            "eligible_ids_sha256": audit._ids_sha256(eligible_ids),
            "excluded_ids": excluded_ids,
            "excluded_ids_sha256": audit._ids_sha256(excluded_ids),
        },
    }
    eligibility_sha = _write_json(eligibility, eligibility_payload)

    identities = audit.FrozenIdentities(
        eligibility_manifest_sha256=eligibility_sha,
        config_sha256=audit._file_sha256(config),
        s50_backbone_sha256=audit._file_sha256(backbones[audit.PRIMARY_ARM]),
        matched_backbone_sha256=audit._file_sha256(backbones[audit.DIAGNOSTIC_ARM]),
        confidence_sha256=audit._file_sha256(confidence),
        source_protocol_sha256=source_protocol_sha,
        source_report_sha256=source_report_sha,
        source_audit_sha256=source_audit_sha,
    )

    bank_records: list[dict[str, Any]] = []
    source_rows: dict[str, dict[str, Any]] = {}
    for global_order, sample_id in enumerate(eligible_ids):
        receptor = tmp_path / "assets" / "receptors" / f"{sample_id}.pdb"
        receptor.parent.mkdir(parents=True, exist_ok=True)
        receptor.write_text(f"RECEPTOR {sample_id}\n", encoding="utf-8")
        meta = tmp_path / "assets" / "meta" / f"{sample_id}.json"
        meta_sha = _write_json(meta, {"sample_key": sample_id})
        ensemble_sha = _sha_text(f"ensemble:{sample_id}")
        shard = global_order % contract.shard_count
        sdf = (
            source_root
            / f"shard-{shard:03d}-of-{contract.shard_count:03d}"
            / "arms"
            / audit.SOURCE_ARM
            / "poses"
            / "all_poses"
            / f"{sample_id}.sdf"
        )
        sdf.parent.mkdir(parents=True, exist_ok=True)
        molecule = Chem.MolFromSmiles("CC")
        assert molecule is not None
        conformer = Chem.Conformer(molecule.GetNumAtoms())
        conformer.SetAtomPosition(0, (0.0, 0.0, 0.0))
        conformer.SetAtomPosition(1, (1.0, 0.0, 0.0))
        molecule.AddConformer(conformer)
        writer = Chem.SDWriter(str(sdf))
        for pose_index in range(contract.pose_count):
            pose = Chem.Mol(molecule)
            pose.SetProp("sample_index", str(pose_index))
            pose.SetProp("complex_id", sample_id)
            pose.SetProp("candidate_ensemble_sha256", ensemble_sha)
            pose.SetProp("sample_sigma", str(contract.sample_sigma))
            writer.write(pose)
        writer.close()
        sdf_sha = audit._file_sha256(sdf)
        receptor_sha = audit._file_sha256(receptor)
        record = {
            "sample_key": sample_id,
            "system_id": systems[sample_id],
            "ligand_chain": "L",
            "plinder_global_index": global_order,
            "sampling_seed": 100 + global_order,
            "ligand_conformer_seed": 0,
            "source_shard_index": shard,
            "pose_count": contract.pose_count,
            "sample_sigma": contract.sample_sigma,
            "num_steps": contract.num_steps,
            "candidate_ensemble_sha256": ensemble_sha,
            "prior_pool_sha256": _sha_text(f"prior:{sample_id}"),
            "prior_pool_size": contract.prior_pool_size,
            "all_poses_sdf": {"path": str(sdf.resolve()), "sha256": sdf_sha},
            "receptor": {"path": str(receptor.resolve()), "sha256": receptor_sha},
            "receptor_sha256": receptor_sha,
            "processed_meta": {"path": str(meta.resolve()), "sha256": meta_sha},
            "processed_meta_sha256": meta_sha,
            "canonical_smiles": "CC",
            "ligand_input_identity_sha256": audit._canonical_smiles_identity("CC"),
            "pocket_center": [0.0, 0.0, 0.0],
            "num_input_atoms": 2,
        }
        bank_records.append(record)
        source_rows[sample_id] = {
            "id": sample_id,
            "arm": audit.SOURCE_ARM,
            "plinder_system_id": systems[sample_id],
            "candidate_ensemble_sha256": ensemble_sha,
            "all_poses_sdf": str(sdf.resolve()),
            "all_poses_sdf_sha256": sdf_sha,
            "all_poses_count": contract.pose_count,
            "num_samples": contract.pose_count,
            "checkpoint_sha256": identities.s50_backbone_sha256,
            "candidate_rmsds_json": json.dumps(rmsds[sample_id]),
            "num_rmsd_lt2_candidates": sum(value < 2.0 for value in rmsds[sample_id]),
        }

    source_shards: list[dict[str, Any]] = []
    for shard in range(contract.shard_count):
        ids = eligible_ids[shard :: contract.shard_count]
        shard_root = source_root / f"shard-{shard:03d}-of-{contract.shard_count:03d}"
        paired = shard_root / "paired_summary.json"
        paired_sha = _write_json(paired, {"shard_index": shard})
        results_csv = shard_root / "arms" / audit.SOURCE_ARM / "results.csv"
        results_sha = _write_csv(results_csv, [source_rows[sample_id] for sample_id in ids])
        source_shards.append(
            {
                "shard_index": shard,
                "paired_summary": {
                    "path": str(paired.resolve()),
                    "sha256": paired_sha,
                },
                "results_csv": {
                    "path": str(results_csv.resolve()),
                    "sha256": results_sha,
                },
                "assigned_count": len(ids),
                "assigned_ids_sha256": audit._ids_sha256(ids),
            }
        )

    runtime_files = {
        "scorer": {
            "path": str(scorer_source.resolve()),
            "sha256": audit._file_sha256(scorer_source),
            "size_bytes": scorer_source.stat().st_size,
        }
    }
    runtime_sha = _runtime_aggregate(runtime_files)
    bank_payload = {
        "schema_version": audit.BANK_SCHEMA_VERSION,
        "protocol_id": audit.PROTOCOL_ID,
        "source_protocol_id": audit.SOURCE_PROTOCOL_ID,
        "status": "complete_label_free",
        "created_at_utc": "2026-08-16T00:00:00+00:00",
        "information_boundary": {
            "source_csv_allowlist": sorted(audit.SOURCE_ROW_ALLOWLIST),
            "outcome_columns_exported": False,
            "score_stage_reads_source_results_csv": False,
            "crystal_reference_exported": False,
        },
        "bank_root": str(source_root.resolve()),
        "inputs": {
            "eligibility_manifest": {
                "path": str(eligibility.resolve()),
                "sha256": eligibility_sha,
            },
            "config": {"path": str(config.resolve()), "sha256": identities.config_sha256},
            "s50_backbone_checkpoint": {
                "path": str(backbones[audit.PRIMARY_ARM]),
                "sha256": identities.s50_backbone_sha256,
            },
            "matched_backbone_checkpoint": {
                "path": str(backbones[audit.DIAGNOSTIC_ARM]),
                "sha256": identities.matched_backbone_sha256,
            },
            "confidence_checkpoint": {
                "path": str(confidence.resolve()),
                "sha256": identities.confidence_sha256,
            },
            "source_sampler_protocol": {
                "path": str(source_protocol.resolve()),
                "sha256": source_protocol_sha,
            },
            "protocol_document": {
                "path": str(protocol.resolve()),
                "sha256": protocol_sha,
            },
            "source_sampler_report": {
                "path": str(source_report.resolve()),
                "sha256": source_report_sha,
            },
            "source_coordinate_audit": {
                "path": str(source_audit.resolve()),
                "sha256": source_audit_sha,
            },
            "scorer_source": {
                "path": str(scorer_source.resolve()),
                "sha256": audit._file_sha256(scorer_source),
            },
            "report_source": {
                "path": str(report_source.resolve()),
                "sha256": audit._file_sha256(report_source),
            },
            "source_output_root": str(source_root.resolve()),
            "runtime_code_identity": {
                "aggregate_sha256": runtime_sha,
                "files": runtime_files,
            },
        },
        "backbone_arms": {
            audit.PRIMARY_ARM: {
                "path": str(backbones[audit.PRIMARY_ARM]),
                "sha256": identities.s50_backbone_sha256,
                "role": "primary_deployment_backbone",
            },
            audit.DIAGNOSTIC_ARM: {
                "path": str(backbones[audit.DIAGNOSTIC_ARM]),
                "sha256": identities.matched_backbone_sha256,
                "role": "diagnostic_training_matched_backbone",
            },
        },
        "fixed_settings": {
            "source_arm": audit.SOURCE_ARM,
            "pose_count": contract.pose_count,
            "sample_sigma": contract.sample_sigma,
            "num_steps": contract.num_steps,
            "prior_pool_size": contract.prior_pool_size,
            "pose_batch_size": 20,
            "pocket_cutoff_angstrom": 10.0,
            "ligand_conformer_seed": 0,
            "selector": audit.SELECTOR,
            "label_blind": True,
            "resampling": False,
        },
        "inventory": {
            "full_count": contract.full_count,
            "eligible_count": contract.eligible_count,
            "eligible_system_count": contract.eligible_system_count,
            "excluded_count": contract.excluded_count,
            "source_shard_count": contract.shard_count,
            "pose_count": contract.pose_count,
            "full_ids_sha256": audit._ids_sha256(full_ids),
            "eligible_ids_sha256": audit._ids_sha256(eligible_ids),
            "excluded_ids_sha256": audit._ids_sha256(excluded_ids),
        },
        "source_shards": source_shards,
        "records": bank_records,
    }
    bank = tmp_path / "bank.json"
    bank_sha = _write_json(bank, bank_payload)

    selected_by_arm = {
        audit.PRIMARY_ARM: [0, 1, 0, 3, 1, 0, 0, 0],
        audit.DIAGNOSTIC_ARM: [1, 1, 2, 0, 0, 0, 2, 1],
    }
    scores_root = tmp_path / "scores"
    for shard in range(contract.shard_count):
        ids = eligible_ids[shard :: contract.shard_count]
        for arm in audit.ARMS:
            records: list[dict[str, Any]] = []
            for sample_id in ids:
                selected = selected_by_arm[arm][eligible_ids.index(sample_id)]
                predicted = [float(index + 1) for index in range(contract.pose_count)]
                predicted[selected] = 0.1
                success = [0.1] * contract.pose_count
                success[selected] = 0.9
                logit = [-2.0] * contract.pose_count
                logit[selected] = 2.0
                arrays = {
                    "confidence_rmsd": predicted,
                    "confidence_success_logit": logit,
                    "confidence_success": success,
                    "confidence_atom_rmsd": [1.0, 2.0, 3.0, 4.0],
                    "confidence_atom_q90": [1.5, 2.5, 3.5, 4.5],
                    "confidence_atom_ok": [0.9, 0.8, 0.7, 0.6],
                }
                records.append(
                    {
                        **bank_records[eligible_ids.index(sample_id)],
                        "score_arrays": arrays,
                        "selected_index": selected,
                        "score_ledger_sha256": audit._score_ledger_sha256(arrays),
                    }
                )
            backbone_sha = (
                identities.s50_backbone_sha256
                if arm == audit.PRIMARY_ARM
                else identities.matched_backbone_sha256
            )
            payload = {
                "schema_version": audit.SCORE_SCHEMA_VERSION,
                "protocol_id": audit.PROTOCOL_ID,
                "status": "complete",
                "mode": "full_shard",
                "stage": "full",
                "arm": arm,
                "arm_role": (
                    "primary_deployment_backbone"
                    if arm == audit.PRIMARY_ARM
                    else "diagnostic_training_matched_backbone"
                ),
                "selector": audit.SELECTOR,
                "created_at_utc": "2026-08-16T00:01:00+00:00",
                "fixed_settings": {
                    "saved_pose_bank_only": True,
                    "resampling": False,
                    "sample_sigma": contract.sample_sigma,
                    "pose_count": contract.pose_count,
                    "num_steps": contract.num_steps,
                    "prior_pool_size": contract.prior_pool_size,
                    "pose_batch_size": 20,
                    "t1_hidden_backbone": arm,
                    "pocket_cutoff_angstrom": 10.0,
                    "selector": audit.SELECTOR,
                    "label_blind": True,
                },
                "inputs": {
                    "label_free_bank_manifest": str(bank.resolve()),
                    "label_free_bank_manifest_sha256": bank_sha,
                    "eligibility_manifest_sha256": eligibility_sha,
                    "protocol_sha256": protocol_sha,
                    "source_sampler_report_sha256": source_report_sha,
                    "source_coordinate_audit_sha256": source_audit_sha,
                    "source_sampler_protocol_sha256": source_protocol_sha,
                    "scorer_source_sha256": audit._file_sha256(scorer_source),
                    "report_source_sha256": audit._file_sha256(report_source),
                    "config_sha256": identities.config_sha256,
                    "backbone_checkpoint": str(backbones[arm]),
                    "backbone_checkpoint_sha256": backbone_sha,
                    "confidence_checkpoint": str(confidence.resolve()),
                    "confidence_checkpoint_sha256": identities.confidence_sha256,
                    "runtime_code_identity_sha256": runtime_sha,
                },
                "inventory": {
                    "eligible_count": contract.eligible_count,
                    "source_shard_count": contract.shard_count,
                    "source_shard_index": shard,
                    "assigned_count": len(ids),
                    "scored_count": len(ids),
                    "assigned_ids_sha256": audit._ids_sha256(ids),
                },
                "records": records,
                "replay": {},
                "runtime": {
                    "elapsed_seconds": 1.0,
                    "finished_at_utc": "2026-08-16T00:02:00+00:00",
                    "slurm_job_id": "synthetic",
                    "slurm_array_job_id": "synthetic",
                    "slurm_array_task_id": str(shard),
                    "cuda_device_name": "synthetic-gpu",
                    "torch_version": "synthetic",
                    "torch_cuda_version": "synthetic",
                    "rdkit_version": "synthetic",
                },
            }
            _write_json(audit._score_path(scores_root, shard, arm, contract), payload)

    return {
        "contract": contract,
        "identities": identities,
        "scores_root": scores_root.resolve(),
        "bank": bank.resolve(),
        "bank_sha": bank_sha,
        "protocol": protocol.resolve(),
        "protocol_sha": protocol_sha,
    }


def _audit_kwargs(fixture: dict[str, Any]) -> dict[str, Any]:
    return {
        "scores_root": fixture["scores_root"],
        "label_free_bank_manifest": fixture["bank"],
        "label_free_bank_manifest_sha256": fixture["bank_sha"],
        "protocol_file": fixture["protocol"],
        "protocol_sha256": fixture["protocol_sha"],
        "contract": fixture["contract"],
        "identities": fixture["identities"],
    }


def test_full_independent_audit_and_strict_report_comparison(
    synthetic_full: dict[str, Any],
) -> None:
    result = audit.audit_full(**_audit_kwargs(synthetic_full))
    assert result["status"] == "passed"
    assert result["integrity"]["exact_8_shards_x_2_arms"] is True
    assert len(result["integrity"]["score_artifacts"]) == 16
    assert result["integrity"]["source_sdf_record_count"] == 32
    assert result["arms"][audit.PRIMARY_ARM]["top1_success_count"] == 5
    assert result["arms"][audit.PRIMARY_ARM]["oracle_success_count"] == 7
    assert result["arms"][audit.PRIMARY_ARM]["oracle_recovery_fraction"] == pytest.approx(
        5 / 7
    )

    strict_payload = {
        "schema_version": audit.STRICT_REPORT_SCHEMA_VERSION,
        "protocol_id": audit.PROTOCOL_ID,
        "status": "complete_diagnostic",
        "stage": "full",
        "arms": result["arms"],
        "paired_backbone": result["paired_backbone"],
    }
    strict_path = synthetic_full["bank"].parent / "strict_report.json"
    strict_sha = _write_json(strict_path, strict_payload)
    compared = audit.audit_full(
        **_audit_kwargs(synthetic_full),
        strict_report=strict_path.resolve(),
        strict_report_sha256=strict_sha,
    )
    assert compared["strict_report_comparison"]["passed"] is True
    assert compared["strict_report_comparison"]["maximum_absolute_delta"] == 0.0


def test_score_nested_label_tamper_is_rejected(synthetic_full: dict[str, Any]) -> None:
    contract = synthetic_full["contract"]
    path = audit._score_path(
        synthetic_full["scores_root"], 0, audit.PRIMARY_ARM, contract
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["records"][0]["candidate_rmsds_json"] = [1.0] * contract.pose_count
    _write_json(path, payload)

    with pytest.raises(audit.AuditError, match="exact field inventory mismatch"):
        audit.audit_full(**_audit_kwargs(synthetic_full))


def test_declared_selector_tamper_is_rejected(synthetic_full: dict[str, Any]) -> None:
    contract = synthetic_full["contract"]
    path = audit._score_path(
        synthetic_full["scores_root"], 0, audit.PRIMARY_ARM, contract
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["records"][0]["selected_index"] = 3
    _write_json(path, payload)

    with pytest.raises(audit.AuditError, match="declared selector index"):
        audit.audit_full(**_audit_kwargs(synthetic_full))


def test_cluster_bootstrap_is_seeded_and_cluster_weighted() -> None:
    contract = audit.AuditContract(
        full_count=3,
        eligible_count=3,
        eligible_system_count=2,
        excluded_count=0,
        shard_count=1,
        pose_count=4,
        bootstrap_resamples=20_000,
        bootstrap_batch_size=256,
    )
    primary = [
        {
            "system_id": "A",
            "selected_success": True,
            "oracle_success": True,
            "first_success": True,
            "random_success_expectation": 0.25,
            "top5_success": True,
        },
        {
            "system_id": "A",
            "selected_success": False,
            "oracle_success": True,
            "first_success": False,
            "random_success_expectation": 0.25,
            "top5_success": True,
        },
        {
            "system_id": "B",
            "selected_success": False,
            "oracle_success": True,
            "first_success": False,
            "random_success_expectation": 0.25,
            "top5_success": True,
        },
    ]
    diagnostic = [
        {**row, "selected_success": index == 2}
        for index, row in enumerate(primary)
    ]
    first = audit._cluster_bootstrap(
        {audit.PRIMARY_ARM: primary, audit.DIAGNOSTIC_ARM: diagnostic},
        contract=contract,
    )
    second = audit._cluster_bootstrap(
        {audit.PRIMARY_ARM: primary, audit.DIAGNOSTIC_ARM: diagnostic},
        contract=contract,
    )
    assert first == second
    assert first[1]["point_pp"] == pytest.approx(0.0)
    assert first[0][audit.PRIMARY_ARM]["top1_success_pct"] == {
        "lower": 0.0,
        "upper": 50.0,
    }


def test_atomic_audit_writer_refuses_overwrite(tmp_path: Path) -> None:
    output = (tmp_path / "audit.json").resolve()
    audit._atomic_write_json_noreplace(output, {"status": "first"})
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        audit._atomic_write_json_noreplace(output, {"status": "second"})
    assert json.loads(output.read_text(encoding="utf-8")) == {"status": "first"}
