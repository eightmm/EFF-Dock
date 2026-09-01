from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest
import torch
from rdkit import Chem

_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/score_early_time_sampler_plinder_confidence.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "score_early_time_sampler_plinder_confidence", _SCRIPT
)
assert _SPEC is not None and _SPEC.loader is not None
scorer = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = scorer
_SPEC.loader.exec_module(scorer)


def _write(path: Path, value: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")
    return path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _asset(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": _sha(path)}


def _source_settings() -> dict[str, Any]:
    return {
        "stage": "full",
        "num_samples": 100,
        "num_steps": 10,
        "model_pose_step_budget": 1000,
        "sigma": 2.0,
        "prior_pool_size": 100,
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


def _make_sdf(path: Path, *, bad_sample_index: int | None = None) -> Chem.Mol:
    path.parent.mkdir(parents=True, exist_ok=True)
    ligand = Chem.MolFromSmiles("CCO")
    assert ligand is not None
    conformer = Chem.Conformer(ligand.GetNumAtoms())
    for atom_index in range(ligand.GetNumAtoms()):
        conformer.SetAtomPosition(
            atom_index,
            (10.0 + atom_index, 20.0 + 2 * atom_index, 30.0 + 3 * atom_index),
        )
    ligand.AddConformer(conformer)
    writer = Chem.SDWriter(str(path))
    for pose_index in range(100):
        molecule = Chem.Mol(ligand)
        molecule.SetProp("_Name", f"docked_pose_{pose_index}")
        molecule.SetProp(
            "sample_index",
            str(999 if bad_sample_index == pose_index else pose_index),
        )
        molecule.SetProp("complex_id", "a__L")
        molecule.SetProp("dataset", "plinder_val")
        molecule.SetProp("sampling_seed", "43")
        molecule.SetProp("ligand_conformer_seed", "0")
        molecule.SetProp("num_samples", "100")
        molecule.SetProp("num_steps", "10")
        molecule.SetProp("sample_sigma", "2.0")
        molecule.SetProp("candidate_ensemble_sha256", "a" * 64)
        # Outcome-like SDF properties may exist in the source bank, but the scorer
        # must neither inspect nor copy them to its label-free ledger.
        molecule.SetProp("candidate_rmsds_json", "[0.1]")
        molecule.SetProp("candidate_fast_valid_json", "[true]")
        writer.write(molecule)
    writer.close()
    return ligand


def _bank_record(tmp_path: Path, sdf: Path) -> dict[str, Any]:
    receptor = _write(tmp_path / "receptor.pt", "receptor")
    processed_meta = _write(tmp_path / "meta.json", "meta")
    return {
        "sample_key": "a__L",
        "system_id": "a",
        "ligand_chain": "L",
        "plinder_global_index": 1,
        "sampling_seed": 43,
        "ligand_conformer_seed": 0,
        "source_shard_index": 0,
        "pose_count": 100,
        "sample_sigma": 2.0,
        "num_steps": 10,
        "candidate_ensemble_sha256": "a" * 64,
        "prior_pool_sha256": "b" * 64,
        "prior_pool_size": 100,
        "all_poses_sdf": _asset(sdf),
        "receptor": _asset(receptor),
        "receptor_sha256": _sha(receptor),
        "processed_meta": _asset(processed_meta),
        "processed_meta_sha256": _sha(processed_meta),
        "canonical_smiles": "CCO",
        "ligand_input_identity_sha256": scorer._canonical_smiles_identity("CCO"),
        "pocket_center": [10.0, 20.0, 30.0],
        "num_input_atoms": 3,
    }


def test_freeze_projects_real_schema_without_outcomes_and_preserves_lex_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ids = ["a__L", "b__L"]
    config = _write(tmp_path / "config.yaml", "config")
    s50 = _write(tmp_path / "s50.pt", "s50")
    matched = _write(tmp_path / "matched.pt", "matched")
    confidence = _write(tmp_path / "confidence.pt", "confidence")
    source_protocol = _write(tmp_path / "source_protocol.md", "source protocol")
    protocol = _write(tmp_path / "protocol.md", "new protocol")
    source_report = _write(tmp_path / "source_report.json", "source report")
    source_audit = _write(tmp_path / "source_audit.json", "source audit")
    report_source = _write(tmp_path / "report.py", "report source")
    receptor = _write(tmp_path / "receptor.pt", "receptor")
    processed_meta = _write(tmp_path / "meta.json", "meta")

    monkeypatch.setattr(scorer, "FULL_COUNT", 2)
    monkeypatch.setattr(scorer, "EXPECTED_ELIGIBLE_COUNT", 2)
    monkeypatch.setattr(scorer, "EXPECTED_SOURCE_SHARDS", 2)
    monkeypatch.setattr(scorer, "ELIGIBLE_SYSTEM_COUNT", 2)
    monkeypatch.setattr(scorer, "EXCLUDED_COUNT", 0)
    monkeypatch.setattr(scorer, "FROZEN_CONFIG_SHA256", _sha(config))
    monkeypatch.setattr(scorer, "FROZEN_S50_BACKBONE_SHA256", _sha(s50))
    monkeypatch.setattr(scorer, "FROZEN_MATCHED_BACKBONE_SHA256", _sha(matched))
    monkeypatch.setattr(scorer, "FROZEN_CONFIDENCE_SHA256", _sha(confidence))
    monkeypatch.setattr(scorer, "FROZEN_SOURCE_PROTOCOL_SHA256", _sha(source_protocol))
    monkeypatch.setattr(scorer, "FROZEN_SOURCE_REPORT_SHA256", _sha(source_report))
    monkeypatch.setattr(scorer, "FROZEN_SOURCE_AUDIT_SHA256", _sha(source_audit))

    eligibility_records = []
    for global_index, sample_id in enumerate(ids, start=1):
        system_id, ligand_chain = sample_id.split("__", maxsplit=1)
        eligibility_records.append(
            {
                "sample_key": sample_id,
                "status": "eligible",
                "global_index": global_index,
                "sampling_seed": 42 + global_index,
                "system_id": system_id,
                "ligand_chain": ligand_chain,
                "canonical_smiles": "CCO",
                "canonical_smiles_identity_sha256": scorer._canonical_smiles_identity(
                    "CCO"
                ),
                "receptor": _asset(receptor),
                "processed_meta": _asset(processed_meta),
                "pocket_center": [1.0, 2.0, 3.0],
            }
        )
    eligibility = {
        "schema_version": "effdock.plinder_checkpoint_eligibility.v1",
        "protocol_id": scorer.SOURCE_PROTOCOL_ID,
        "status": "complete",
        "inputs": {
            "fixed_identities": {
                "config": _asset(config),
                "protocol_document": _asset(source_protocol),
                "checkpoints": {scorer.SOURCE_ARM: _asset(s50)},
            }
        },
        "inventory": {
            "full_count": 2,
            "full_ids": ids,
            "full_ids_sha256": scorer.sorted_id_sha256(ids),
            "eligible_count": 2,
            "eligible_ids": ids,
            "eligible_ids_sha256": scorer.sorted_id_sha256(ids),
            "eligible_system_count": 2,
            "excluded_count": 0,
            "excluded_ids": [],
            "excluded_ids_sha256": scorer.sorted_id_sha256([]),
        },
        "records": eligibility_records,
    }
    eligibility_path = tmp_path / "eligibility.json"
    eligibility_path.write_text(json.dumps(eligibility), encoding="utf-8")
    monkeypatch.setattr(scorer, "FROZEN_ELIGIBILITY_SHA256", _sha(eligibility_path))

    bank_root = tmp_path / "source" / "full"
    for shard_index, sample_id in enumerate(ids):
        shard_dir = bank_root / f"shard-{shard_index:03d}-of-002"
        arm_dir = shard_dir / "arms" / scorer.SOURCE_ARM
        sdf = _write(
            arm_dir / "poses" / "all_poses" / f"{sample_id}.sdf",
            f"saved bank {sample_id}",
        )
        global_index = shard_index + 1
        row = {field: "" for field in scorer.SOURCE_ROW_ALLOWLIST}
        row.update(
            {
                "id": sample_id,
                "arm": scorer.SOURCE_ARM,
                "plinder_global_index": str(global_index),
                "sampling_seed": str(42 + global_index),
                "ligand_conformer_seed": "0",
                "all_poses_count": "100",
                "all_poses_sdf": str(sdf),
                "all_poses_sdf_sha256": _sha(sdf),
                "candidate_ensemble_sha256": f"{global_index}" * 64,
                "checkpoint": str(s50),
                "checkpoint_sha256": _sha(s50),
                "ligand_input_canonical_smiles": "CCO",
                "ligand_input_identity_sha256": scorer._canonical_smiles_identity("CCO"),
                "num_input_atoms": "3",
                "num_samples": "100",
                "plinder_ligand_chain": "L",
                "plinder_system_id": sample_id.split("__", maxsplit=1)[0],
                "prior_pool_sha256": f"{global_index + 2}" * 64,
                "prior_pool_size": "100",
                "processed_meta": str(processed_meta),
                "processed_meta_sha256": _sha(processed_meta),
                "protein": str(receptor),
                "protein_sha256": _sha(receptor),
                "sampling_dynamics": "deterministic_ode",
                "selector_profile": "candidate_only",
            }
        )
        csv_path = arm_dir / "results.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        outcome_fields = [
            "candidate_rmsds_json",
            "candidate_fast_valid_json",
            "oracle_rmsd",
        ]
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=[*scorer.SOURCE_ROW_ALLOWLIST, *outcome_fields]
            )
            writer.writeheader()
            writer.writerow(
                {
                    **row,
                    "candidate_rmsds_json": "[0.1]",
                    "candidate_fast_valid_json": "[true]",
                    "oracle_rmsd": "0.1",
                }
            )
        paired = {
            "schema_version": "effdock.plinder_checkpoint_paired_shard.v1",
            "protocol_id": scorer.SOURCE_PROTOCOL_ID,
            "status": "complete",
            "mode": "full",
            "settings": _source_settings(),
            "eligibility_manifest": {
                "path": str(eligibility_path),
                "sha256": _sha(eligibility_path),
            },
            "inventory": {
                "num_shards": 2,
                "eligible_count": 2,
                "selected_ids": ids,
                "assigned_ids": [sample_id],
                "assigned_count": 1,
            },
            "failures": [],
            "arms": [
                {
                    "name": scorer.SOURCE_ARM,
                    "checkpoint": str(s50),
                    "checkpoint_sha256": _sha(s50),
                }
            ],
            "artifacts": {
                "arms": {
                    scorer.SOURCE_ARM: {
                        "results_csv": str(csv_path),
                        "results_csv_sha256": _sha(csv_path),
                        "count": 1,
                    }
                }
            },
        }
        (shard_dir / "paired_summary.json").write_text(
            json.dumps(paired), encoding="utf-8"
        )

    runtime_sha = scorer._runtime_code_identity()["aggregate_sha256"]
    output = tmp_path / "bank.json"
    manifest = scorer.freeze_label_free_inputs(
        bank_root=bank_root,
        eligibility_manifest=eligibility_path,
        expected_eligibility_manifest_sha256=_sha(eligibility_path),
        config=config,
        expected_config_sha256=_sha(config),
        s50_backbone_checkpoint=s50,
        expected_s50_backbone_checkpoint_sha256=_sha(s50),
        matched_backbone_checkpoint=matched,
        expected_matched_backbone_checkpoint_sha256=_sha(matched),
        confidence_checkpoint=confidence,
        expected_confidence_checkpoint_sha256=_sha(confidence),
        source_sampler_protocol=source_protocol,
        expected_source_sampler_protocol_sha256=_sha(source_protocol),
        protocol_document=protocol,
        expected_protocol_sha256=_sha(protocol),
        source_sampler_report=source_report,
        expected_source_sampler_report_sha256=_sha(source_report),
        source_coordinate_audit=source_audit,
        expected_source_coordinate_audit_sha256=_sha(source_audit),
        report_source=report_source,
        expected_report_source_sha256=_sha(report_source),
        expected_scorer_source_sha256=_sha(_SCRIPT),
        expected_runtime_code_sha256=runtime_sha,
        output=output,
        expected_source_shards=2,
        expected_eligible_count=2,
    )

    assert manifest["status"] == "complete_label_free"
    assert [row["sample_key"] for row in manifest["records"]] == ids
    assert set(manifest["records"][0]) == set(scorer.BANK_RECORD_FIELDS)
    assert manifest["inputs"]["source_output_root"] == str(bank_root)
    assert all(
        set(manifest["inputs"][name]) == {"path", "sha256"}
        for name in scorer.BANK_INPUT_ASSET_NAMES
    )
    assert all(
        set(spec) == {"path", "sha256", "role"}
        for spec in manifest["backbone_arms"].values()
    )
    assert manifest["inventory"] == {
        "full_count": 2,
        "eligible_count": 2,
        "eligible_system_count": 2,
        "excluded_count": 0,
        "source_shard_count": 2,
        "pose_count": 100,
        "full_ids_sha256": scorer.sorted_id_sha256(ids),
        "eligible_ids_sha256": scorer.sorted_id_sha256(ids),
        "excluded_ids_sha256": scorer.sorted_id_sha256([]),
    }
    serialized = json.dumps(manifest)
    for forbidden in (
        "candidate_rmsds_json",
        "candidate_fast_valid_json",
        "oracle_rmsd",
        "fast_valid",
    ):
        assert forbidden not in serialized
    scorer._validate_score_bank_manifest(
        manifest,
        expected_manifest_sha256=_sha(output),
        manifest_path=output,
    )


def test_score_asset_verification_does_not_hash_sealed_outcome_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _write(tmp_path / "config.yaml", "config")
    protocol = _write(tmp_path / "protocol.md", "protocol")
    report = _write(tmp_path / "report.py", "report")
    backbone = _write(tmp_path / "backbone.pt", "backbone")
    confidence = _write(tmp_path / "confidence.pt", "confidence")
    blocked_paths = {
        tmp_path / "eligibility_with_labels.json",
        tmp_path / "source_protocol.md",
        tmp_path / "source_report_with_outcomes.json",
        tmp_path / "coordinate_audit_with_labels.json",
    }
    runtime = {"aggregate_sha256": "9" * 64, "files": {}}
    manifest = {
        "inputs": {
            "config": _asset(config),
            "protocol_document": _asset(protocol),
            "scorer_source": _asset(_SCRIPT),
            "report_source": _asset(report),
            "confidence_checkpoint": _asset(confidence),
            "runtime_code_identity": runtime,
            "eligibility_manifest": {
                "path": str(tmp_path / "eligibility_with_labels.json"),
                "sha256": "1" * 64,
            },
            "source_sampler_protocol": {
                "path": str(tmp_path / "source_protocol.md"),
                "sha256": "2" * 64,
            },
            "source_sampler_report": {
                "path": str(tmp_path / "source_report_with_outcomes.json"),
                "sha256": "3" * 64,
            },
            "source_coordinate_audit": {
                "path": str(tmp_path / "coordinate_audit_with_labels.json"),
                "sha256": "4" * 64,
            },
        },
        "backbone_arms": {"s50_backbone": _asset(backbone)},
    }
    original_hash = scorer.file_sha256
    hashed_paths: list[Path] = []

    def guarded_hash(path: Path) -> str:
        resolved = Path(path).resolve()
        if resolved in blocked_paths:
            raise AssertionError(f"score stage opened sealed source artifact {resolved}")
        hashed_paths.append(resolved)
        return original_hash(resolved)

    monkeypatch.setattr(scorer, "file_sha256", guarded_hash)
    monkeypatch.setattr(scorer, "_runtime_code_identity", lambda: runtime)
    verified = scorer._verify_score_stage_assets(
        manifest,
        arm="s50_backbone",
        docking_checkpoint=backbone,
        expected_docking_checkpoint_sha256=_sha(backbone),
        confidence_checkpoint=confidence,
        expected_confidence_checkpoint_sha256=_sha(confidence),
        expected_protocol_sha256=_sha(protocol),
        expected_scorer_source_sha256=_sha(_SCRIPT),
        expected_report_source_sha256=_sha(report),
        expected_runtime_code_sha256="9" * 64,
    )

    assert verified["source_sampler_report"]["sha256"] == "3" * 64
    assert not blocked_paths.intersection(hashed_paths)


def test_saved_sdf_requires_exact_order_properties_hash_and_topology(tmp_path: Path) -> None:
    sdf = tmp_path / "poses.sdf"
    ligand = _make_sdf(sdf)
    record = _bank_record(tmp_path, sdf)

    poses = scorer.read_saved_bank_poses(record, ligand=ligand)

    assert len(poses) == 100
    assert torch.equal(poses[0][0], torch.zeros(3))
    assert scorer._topology_signature(Chem.MolFromSmiles("F[C@H](Cl)Br")) == (
        scorer._topology_signature(Chem.MolFromSmiles("F[C@@H](Cl)Br"))
    )

    bad_sdf = tmp_path / "bad_order.sdf"
    _make_sdf(bad_sdf, bad_sample_index=7)
    bad_record = {**record, "all_poses_sdf": _asset(bad_sdf)}
    with pytest.raises(scorer.ScoreContractError, match="sample_index"):
        scorer.read_saved_bank_poses(bad_record, ligand=ligand)

    old_hash = record["all_poses_sdf"]["sha256"]
    with sdf.open("ab") as handle:
        handle.write(b"tamper")
    assert old_hash != _sha(sdf)
    with pytest.raises(scorer.ScoreContractError, match="SHA-256 mismatch"):
        scorer.read_saved_bank_poses(record, ligand=ligand)


def test_topology_signature_normalizes_explicit_implicit_h_representation() -> None:
    smiles = "O=c1[nH]cnc2c([C@@H]3O[C@H](CO)[C@@H](O)[C@H]3O)n[nH]c12"
    ligand = Chem.RemoveAllHs(Chem.MolFromSmiles(smiles))
    conformer = Chem.Conformer(ligand.GetNumAtoms())
    conformer.Set3D(True)
    for atom_index in range(ligand.GetNumAtoms()):
        conformer.SetAtomPosition(
            atom_index,
            (float(atom_index), float(atom_index % 3), float(atom_index % 5)),
        )
    ligand.AddConformer(conformer)
    serialized = Chem.MolFromMolBlock(
        Chem.MolToMolBlock(ligand),
        removeHs=False,
        sanitize=True,
        strictParsing=True,
    )
    assert ligand is not None and serialized is not None
    assert any(
        left.GetNumExplicitHs() != right.GetNumExplicitHs()
        for left, right in zip(ligand.GetAtoms(), serialized.GetAtoms(), strict=True)
    )
    assert scorer._topology_signature(ligand) == scorer._topology_signature(serialized)

    changed_charge = Chem.RWMol(serialized)
    changed_charge.GetAtomWithIdx(0).SetFormalCharge(1)
    assert scorer._topology_signature(ligand) != scorer._topology_signature(
        changed_charge.GetMol()
    )

    changed_bond = Chem.RWMol(serialized)
    bond = changed_bond.GetBondWithIdx(0)
    bond.SetBondType(Chem.BondType.SINGLE)
    assert scorer._topology_signature(ligand) != scorer._topology_signature(
        changed_bond.GetMol()
    )


def test_chunk20_selection_exact_score_schema_and_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[int] = []

    def fake_score(
        confidence_model: torch.nn.Module,
        docking_model: torch.nn.Module,
        graph: dict[str, torch.Tensor],
        ligand_data: dict[str, torch.Tensor],
        meta: dict[str, Any],
        poses: list[torch.Tensor],
        **kwargs: Any,
    ) -> list[dict[str, float]]:
        start = sum(calls)
        calls.append(len(poses))
        return [
            {
                "confidence_rmsd": 0.25 if start + index == 57 else 3.0,
                "confidence_success_logit": -0.5,
                "confidence_success": 0.4,
                "confidence_atom_rmsd": 1.0,
                "confidence_atom_q90": 1.5,
                "confidence_atom_ok": 0.7,
            }
            for index in range(len(poses))
        ]

    poses = [torch.zeros((3, 3)) for _ in range(100)]
    scores = scorer._score_in_chunks(
        torch.nn.Identity(),
        torch.nn.Identity(),
        {},
        {},
        {},
        poses,
        device=torch.device("cpu"),
        score_fn=fake_score,
    )
    arrays = scorer._score_arrays(scores)
    assert calls == [20, 20, 20, 20, 20]
    assert scorer._selected_index(arrays) == 57

    sdf = _write(tmp_path / "poses.sdf", "unused")
    record = _bank_record(tmp_path, sdf)
    monkeypatch.setattr(scorer, "_prepare_complex", lambda record: ({}, {}, {}, poses))
    calls.clear()
    result = scorer._score_one_record(
        record,
        confidence_model=torch.nn.Identity(),
        docking_model=torch.nn.Identity(),
        device=torch.device("cpu"),
        score_fn=fake_score,
    )
    assert set(result) == set(scorer.SCORE_RECORD_FIELDS)
    assert result["selected_index"] == 57
    assert result["score_ledger_sha256"] == scorer.score_ledger_sha256(
        result["score_arrays"]
    )
    assert not {
        "candidate_rmsds_json",
        "candidate_fast_valid_json",
        "oracle_rmsd",
        "fast_valid",
    }.intersection(result)

    replay = json.loads(json.dumps(result))
    replay["score_arrays"]["confidence_success_logit"][0] += 1e-7
    replay["score_ledger_sha256"] = scorer.score_ledger_sha256(replay["score_arrays"])
    replay_summary = scorer._replay_delta(result, replay)
    assert replay_summary["passed"] is True
    assert replay_summary["checked_count"] == 1
    assert replay_summary["selected_index_mismatches"] == 0
    assert replay_summary["max_abs_score_delta"] == pytest.approx(1e-7)

    replay["selected_index"] = 3
    with pytest.raises(scorer.ScoreContractError, match="selected pose index"):
        scorer._replay_delta(result, replay)


def test_atomic_no_overwrite_and_symlink_paths_fail_closed(tmp_path: Path) -> None:
    output = tmp_path / "nested" / "artifact.json"
    scorer._atomic_write_new_json(output, {"first": True})
    with pytest.raises(FileExistsError, match="overwrite"):
        scorer._atomic_write_new_json(output, {"first": False})
    assert json.loads(output.read_text(encoding="utf-8")) == {"first": True}

    target = _write(tmp_path / "target.txt", "target")
    link = tmp_path / "link.txt"
    link.symlink_to(target)
    with pytest.raises(scorer.ScoreContractError, match="symlink"):
        scorer._canonical_existing_path(link, label="symlink regression")
    with pytest.raises(scorer.ScoreContractError, match="absolute"):
        scorer._canonical_existing_path(Path("relative.txt"), label="relative regression")
