from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

import effdock.workflows.guidance_budget_full_report as full_report
from effdock.guidance.parameterization import guidance_parameter_identity
from effdock.guidance.provenance import guidance_implementation_identity
from effdock.guidance.system import receptor_policy_identity
from effdock.workflows.evaluate import sorted_id_sha256
from effdock.workflows.guidance_budget_full_posebusters_report import (
    build_report as build_official_report,
)
from effdock.workflows.guidance_budget_full_report import (
    PROTOCOL_ID,
    build_report,
    validate_full_cohort_audit_for_dataset,
)
from effdock.workflows.guidance_budget_posebusters_report import (
    EXPECTED_SELECTOR,
    POSEBUSTERS_CONFIG,
    POSEBUSTERS_VERSION,
    VALIDITY_CHECKS,
)
from effdock.workflows.guidance_budget_report import (
    CONDITIONS,
    DATASETS,
    EXPECTED_CHECKPOINT_SHA256,
    EXPECTED_CONFIG_SHA256,
    EXPECTED_POCKET_CENTERS_SHA256,
)
from effdock.workflows.guidance_coverage_audit import (
    AUDIT_SCHEMA_VERSION,
    ID_HASH_CONTRACT,
)


def _id_group(ids: list[str]) -> dict[str, object]:
    ids = sorted(ids)
    return {
        "count": len(ids),
        "ids": ids,
        "ids_sha256": sorted_id_sha256(ids),
    }


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _test_input_identity(dataset: str, ids: list[str]) -> dict[str, object]:
    ordered = sorted(ids)
    ligand_overlap = ordered[:1]
    exact_overlap = ordered[:1] if dataset == "astex" else []
    return {
        "schema_version": "effdock.benchmark_input_identity.v1",
        "mode": "frozen_manifest",
        "dataset": dataset,
        "heavy_atom_policy": "seeded_generic_loader_then_rdkit_remove_all_hs",
        "count": len(ordered),
        "ids_sha256": sorted_id_sha256(ordered),
        "mapping_sha256": _digest(f"mapping-{dataset}"),
        "sources": {
            "frozen_manifest": {
                "path": str(full_report.BENCHMARK_INPUT_MANIFEST),
                "sha256": full_report.EXPECTED_BENCHMARK_INPUT_MANIFEST_SHA256,
            },
            "source_manifests": {},
            "integrity_boundary": {
                "benchmark_ids_with_split_ligand_identity_overlap": ligand_overlap,
                "benchmark_ids_with_split_ligand_identity_overlap_count": len(
                    ligand_overlap
                ),
                "benchmark_ids_with_exact_entry_and_ligand_overlap": exact_overlap,
                "benchmark_ids_with_exact_entry_and_ligand_overlap_count": len(
                    exact_overlap
                ),
                "matching_train_rows": len(ligand_overlap),
                "matching_val_rows": 0,
            },
        },
        "per_id": {
            complex_id: {
                "canonical_heavy_isomeric_smiles": f"canonical-{complex_id}",
                "raw_smiles_sha256": _digest(f"raw-{complex_id}"),
                "sha256": _digest(f"input-{complex_id}"),
            }
            for complex_id in ordered
        },
        "sha256": _digest(f"identity-{dataset}"),
    }


def _audit_dataset(dataset: str, ids: list[str]) -> dict[str, object]:
    ids = sorted(ids)
    strict_ids, fallback_ids = ids[:1], ids[1:]
    slices = {
        "strict_supported": _id_group(strict_ids),
        "nonprotein_only": _id_group(fallback_ids),
        "metal_only": _id_group([]),
        "nonprotein_and_metal": _id_group([]),
    }
    policy_identity = receptor_policy_identity("geometry_only")
    input_identity = _test_input_identity(dataset, ids)
    exact_ids, representation_mismatch_ids = ids[:1], ids[1:]
    representation_slices = {
        "exact_graph": _id_group(exact_ids),
        "same_connectivity_representation_mismatch": _id_group(
            representation_mismatch_ids
        ),
    }
    return {
        "discovered": len(ids),
        "ids": ids,
        "ids_sha256": sorted_id_sha256(ids),
        "ids_hash_contract": ID_HASH_CONTRACT,
        "audited": len(ids),
        "audited_ids": ids,
        "audited_ids_sha256": sorted_id_sha256(ids),
        "complete": True,
        "success": len(ids),
        "success_ids": ids,
        "success_ids_sha256": sorted_id_sha256(ids),
        "failed": 0,
        "failed_ids": [],
        "failed_ids_sha256": sorted_id_sha256([]),
        "failure_codes": {},
        "chemistry_slices": slices,
        "ligand_representation_slices": representation_slices,
        "fallback_reasons": {},
        "inputs": {"benchmark_input_identity": input_identity},
        "strict_supported_equivalence": {
            **_id_group(strict_ids),
            "passed": True,
            "mismatch": _id_group([]),
            "legacy_v1_eligibility": None,
        },
        "complexes": {
            complex_id: {
                "status": "success",
                "protein_sha256": _digest(f"protein-{dataset}-{complex_id}"),
                "ligand_reference_sha256": _digest(f"reference-{dataset}-{complex_id}"),
                "system_reference_sha256": _digest(f"system-{dataset}-{complex_id}"),
                "topology_reference_sha256": _digest(f"topology-{dataset}-{complex_id}"),
                "interaction_reference_sha256": _digest(
                    f"interaction-{dataset}-{complex_id}"
                ),
                "ligand_input_identity_sha256": input_identity["per_id"][complex_id][
                    "sha256"
                ],
                "ligand_input_canonical_smiles": input_identity["per_id"][complex_id][
                    "canonical_heavy_isomeric_smiles"
                ],
                "numerical_preflight_reference_alignment": {
                    "accepted": True,
                    "relation": (
                        "exact_graph"
                        if complex_id in exact_ids
                        else "same_connectivity_representation_mismatch"
                    ),
                    "match_method": (
                        "strict" if complex_id in exact_ids else "mcs(1/1)"
                    ),
                    "matched_atoms": 1,
                    "input_atoms": 1,
                    "reference_atoms": 1,
                    "full_bijection": True,
                    "atom_elements_match": True,
                    "connectivity_match": True,
                    "bond_orders_match": complex_id in exact_ids,
                    "formal_charges_match": complex_id in exact_ids,
                },
                "receptor": {
                    "mode": "geometry_only",
                    "identity_sha256": policy_identity["sha256"],
                    "provenance": {},
                    "obstacle_count": int(complex_id in fallback_ids),
                    "metal_fallback_count": 0,
                    "metal_fallback_reasons": {},
                },
                "crystal_numerical_preflight": {
                    "energy_finite": True,
                    "gradient_finite": True,
                },
            }
            for complex_id in ids
        },
    }


def _write_audit(path: Path, ids_by_dataset: dict[str, list[str]]) -> Path:
    payload = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "receptor_policy": "geometry_only",
        "implementation": guidance_implementation_identity(),
        "parameter_set": guidance_parameter_identity(),
        "receptor_policy_identity": receptor_policy_identity("geometry_only"),
        "datasets": {
            dataset: _audit_dataset(dataset, ids) for dataset, ids in ids_by_dataset.items()
        },
    }
    for dataset, ids in ids_by_dataset.items():
        for complex_id in ids:
            protein = path.parent / f"{dataset}-{complex_id}-protein.pdb"
            reference = path.parent / f"{dataset}-{complex_id}-ligand.sdf"
            protein.write_text(f"protein-{dataset}-{complex_id}")
            reference.write_text(f"reference-{dataset}-{complex_id}")
            record = payload["datasets"][dataset]["complexes"][complex_id]
            record["protein"] = str(protein)
            record["ligand_reference"] = str(reference)
    path.write_text(json.dumps(payload))
    return path


def _write_sampling_fixture(
    root: Path, audit_path: Path, ids_by_dataset: dict[str, list[str]]
) -> Path:
    input_dir = root / "raw"
    input_dir.mkdir()
    audit_sha = hashlib.sha256(audit_path.read_bytes()).hexdigest()
    guidance_parameters = guidance_parameter_identity()
    guidance_sha = guidance_parameters["sha256"]
    implementation = guidance_implementation_identity()
    policy_identity = receptor_policy_identity("geometry_only")
    for dataset in DATASETS:
        input_identity = _test_input_identity(dataset, ids_by_dataset[dataset])
        for condition_index, (_, num_samples, num_steps) in enumerate(CONDITIONS):
            for arm in ("unguided", "guided"):
                scale = 0.0 if arm == "unguided" else 0.1
                run_name = full_report._expected_run_name(dataset, num_samples, num_steps, arm)
                for shard_index, complex_id in enumerate(ids_by_dataset[dataset]):
                    exact_graph = shard_index == 0
                    relation = (
                        "exact_graph"
                        if exact_graph
                        else "same_connectivity_representation_mismatch"
                    )
                    match_method = "strict" if exact_graph else "mcs(1/1)"
                    mapping_metadata = {
                        "accepted": True,
                        "relation": relation,
                        "match_method": match_method,
                        "matched_atoms": 1,
                        "input_atoms": 1,
                        "reference_atoms": 1,
                        "full_bijection": True,
                        "atom_elements_match": True,
                        "connectivity_match": True,
                        "bond_orders_match": exact_graph,
                        "formal_charges_match": exact_graph,
                    }
                    csv_path = input_dir / f"{run_name}.shard-{shard_index}.csv"
                    rmsd = 2.3 - 0.3 * condition_index - (0.5 if arm == "guided" else 0.0)
                    row = {
                        "id": complex_id,
                        "oracle_rmsd": rmsd + 0.05 * shard_index,
                        "oracle_fast_valid": True,
                        "num_fast_valid_candidates": 1,
                        "fast_valid_oracle_rmsd": rmsd + 0.05 * shard_index,
                        "joint_fast_valid_and_rmsd_lt2": rmsd + 0.05 * shard_index < 2,
                        "prior_pool_size": 100,
                        "sampling_seed": 100 + shard_index,
                        "prior_pool_sha256": hashlib.sha256(
                            f"prior-{complex_id}".encode()
                        ).hexdigest(),
                        "guidance_mode": ("unified_operator_split" if arm == "guided" else "none"),
                        "guidance_parameter_sha256": guidance_sha if arm == "guided" else "",
                        "guidance_receptor_policy": ("geometry_only" if arm == "guided" else ""),
                        "guidance_receptor_policy_identity_sha256": (
                            policy_identity["sha256"] if arm == "guided" else ""
                        ),
                        "protein_sha256": _digest(
                            f"protein-{dataset}-{complex_id}"
                        ),
                        "ligand_reference_sha256": _digest(
                            f"reference-{dataset}-{complex_id}"
                        ),
                        "num_match_atoms": 1,
                        "num_input_atoms": 1,
                        "num_ref_atoms": 1,
                        "full_heavy_atom_bijection": True,
                        "ligand_graph_relation": relation,
                        "ligand_mapping_metadata_json": json.dumps(
                            mapping_metadata, separators=(",", ":"), sort_keys=True
                        ),
                        "exact_full_heavy_atom_graph": exact_graph,
                        "ligand_input_identity_sha256": input_identity["per_id"][
                            complex_id
                        ]["sha256"],
                        "ligand_input_canonical_smiles": input_identity["per_id"][
                            complex_id
                        ]["canonical_heavy_isomeric_smiles"],
                        "guidance_system_reference_sha256": (
                            _digest(f"system-{dataset}-{complex_id}")
                            if arm == "guided"
                            else ""
                        ),
                        "guidance_topology_reference_sha256": (
                            _digest(f"topology-{dataset}-{complex_id}")
                            if arm == "guided"
                            else ""
                        ),
                        "guidance_interaction_reference_sha256": (
                            _digest(f"interaction-{dataset}-{complex_id}")
                            if arm == "guided"
                            else ""
                        ),
                    }
                    with csv_path.open("w", newline="") as handle:
                        writer = csv.DictWriter(handle, fieldnames=list(row))
                        writer.writeheader()
                        writer.writerow(row)
                    summary = {
                        "run_name": run_name,
                        "protocol_id": PROTOCOL_ID,
                        "dataset": dataset,
                        "num_discovered_total": len(ids_by_dataset[dataset]),
                        "num_assigned": 1,
                        "num_success": 1,
                        "num_failed": 0,
                        "num_samples": num_samples,
                        "num_steps": num_steps,
                        "model_pose_step_budget": 1000,
                        "num_shards": 2,
                        "shard_index": shard_index,
                        "seed": 42,
                        "unified_guidance_scale": scale,
                        "unified_guidance_receptor_policy": "geometry_only",
                        "guidance_implementation": implementation,
                        "benchmark_input_identity": input_identity,
                        "require_full_ligand_atom_mapping": True,
                        "prior_pool_size": 100,
                        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
                        "confidence_checkpoint_sha256": None,
                        "config_sha256": EXPECTED_CONFIG_SHA256,
                        "pocket_centers_sha256": EXPECTED_POCKET_CENTERS_SHA256[dataset],
                        "eligibility_manifest_sha256": audit_sha,
                        "sigma": 0.5,
                        "time_schedule": "late",
                        "schedule_power": 3.0,
                        "pocket_cutoff": 10.0,
                        "center_jitter_sigma": 0.0,
                        "vina_guidance_scale": 0.0,
                        "unified_guidance_start_t": 0.5,
                        "unified_guidance_ramp_power": 1.0,
                        "unified_guidance_max_force": 20.0,
                        "unified_guidance_max_velocity": 5.0,
                        "unified_guidance_max_angular_velocity": 5.0,
                        "unified_guidance_max_atom_displacement": 0.25,
                        "unified_guidance_max_backtracks": 8,
                        "unified_guidance_protein_shell": 18.0,
                        "refine": "none",
                        "csv": str(csv_path),
                        "failures": [],
                    }
                    if arm == "guided":
                        summary["guidance_parameter_set"] = guidance_parameters
                        summary["guidance_receptor_policy_identities"] = {
                            policy_identity["sha256"]: policy_identity
                        }
                        summary["guidance_receptor_provenance_by_id"] = {
                            complex_id: {"mode": "geometry_only", "shard": shard_index}
                        }
                        summary["guidance_operator_stats"] = {
                            "steps_attempted": 1,
                            "pose_corrections_attempted": 1,
                            "pose_corrections_accepted": 1,
                            "pose_corrections_rejected": 0,
                            "nonfinite_base_poses": 0,
                            "nonfinite_trials": 0,
                            "max_accepted_atom_displacement": 0.1,
                        }
                    (input_dir / f"{run_name}.shard-{shard_index}.summary.json").write_text(
                        json.dumps(summary)
                    )
    return input_dir


def _write_official_fixture(root: Path, ids_by_dataset: dict[str, list[str]]) -> Path:
    input_dir = root / "official"
    input_dir.mkdir()
    rmsd_check = "rmsd_≤_2å"
    for dataset in DATASETS:
        for _, num_samples, num_steps in CONDITIONS:
            for arm in ("unguided", "guided"):
                run_name = full_report._expected_run_name(dataset, num_samples, num_steps, arm)
                run_dir = input_dir / run_name
                run_dir.mkdir()
                for shard_index, complex_id in enumerate(ids_by_dataset[dataset]):
                    checks = {name: True for name in VALIDITY_CHECKS}
                    row = {
                        "id": complex_id,
                        "posebusters_valid": True,
                        **checks,
                        rmsd_check: True,
                    }
                    tag = f"shard-{shard_index:03d}-of-002"
                    csv_path = run_dir / f"{tag}.csv"
                    with csv_path.open("w", newline="") as handle:
                        writer = csv.DictWriter(handle, fieldnames=list(row))
                        writer.writeheader()
                        writer.writerow(row)
                    summary = {
                        "posebusters_version": POSEBUSTERS_VERSION,
                        "config": POSEBUSTERS_CONFIG,
                        "selector": EXPECTED_SELECTOR,
                        "num_discovered_total": 2,
                        "num_assigned": 1,
                        "num_success": 1,
                        "num_failed": 0,
                        "posebusters_valid_pct": 100.0,
                        "failures": [],
                        "csv": str(csv_path),
                    }
                    (run_dir / f"{tag}.summary.json").write_text(json.dumps(summary))
    return input_dir


@pytest.fixture
def tiny_counts(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[str]]:
    ids = {"astex": ["a1", "a2"], "posebusters": ["p1", "p2"]}
    for dataset in DATASETS:
        monkeypatch.setitem(full_report.EXPECTED_DATASET_COUNTS, dataset, 2)
    monkeypatch.setattr(
        full_report,
        "_expected_benchmark_input_identity",
        lambda dataset: _test_input_identity(dataset, ids[dataset]),
    )
    return ids


def test_full_audit_gate_requires_finite_complete_equivalent_cohort(
    tmp_path: Path, tiny_counts: dict[str, list[str]]
) -> None:
    audit = _write_audit(tmp_path / "audit.json", tiny_counts)
    validated = validate_full_cohort_audit_for_dataset(audit, "astex")
    assert validated["discovered"] == 2
    assert validated["chemistry_slices"]["strict_supported"] == ("a1",)
    assert validated["ligand_representation_slices"][
        "same_connectivity_representation_mismatch"
    ] == ("a2",)
    assert validated["integrity_slices"]["exact_entry_and_ligand_overlap"] == (
        "a1",
    )

    payload = json.loads(audit.read_text())
    original_payload = json.loads(audit.read_text())
    payload["datasets"]["astex"]["inputs"]["benchmark_input_identity"][
        "mapping_sha256"
    ] = "0" * 64
    audit.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="benchmark-input identity differs"):
        validate_full_cohort_audit_for_dataset(audit, "astex")

    payload = original_payload
    audit.write_text(json.dumps(payload))
    protein_path = Path(payload["datasets"]["astex"]["complexes"]["a1"]["protein"])
    protein_path.write_text("drifted")
    with pytest.raises(ValueError, match="protein changed after audit"):
        validate_full_cohort_audit_for_dataset(audit, "astex")
    protein_path.write_text("protein-astex-a1")

    payload["datasets"]["astex"]["complexes"]["a2"]["crystal_numerical_preflight"][
        "gradient_finite"
    ] = False
    audit.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="crystal preflight is non-finite"):
        validate_full_cohort_audit_for_dataset(audit, "astex")


def test_full_sampling_report_exact_coverage_slices_and_pairing(
    tmp_path: Path, tiny_counts: dict[str, list[str]]
) -> None:
    audit = _write_audit(tmp_path / "audit.json", tiny_counts)
    input_dir = _write_sampling_fixture(tmp_path, audit, tiny_counts)
    report = build_report(
        input_dir,
        audit,
        expected_shards=2,
        bootstrap_resamples=50,
    )

    assert report["status"] == "complete_strict_full_cohort_paired"
    assert report["coverage_gate"] == {
        "expected_total": 4,
        "covered_total": 4,
        "failed_total": 0,
    }
    astex = report["datasets"]["astex"]
    assert astex["full_cohort_coverage"]["coverage_pct"] == 100.0
    assert astex["prior_pairing"]["verified"] is True
    expected_hash = sorted_id_sha256(tiny_counts["astex"])
    cell = astex["cells"]["n100_s10"]
    for arm in ("unguided", "guided"):
        assert cell[arm]["eligible_ids_sha256"] == expected_hash
        assert cell[arm]["ids_hash_contract"] == ID_HASH_CONTRACT
    assert cell["guided_vs_unguided"]["common_ids_sha256"] == expected_hash
    assert cell["guided_vs_unguided"]["ids_hash_contract"] == ID_HASH_CONTRACT
    budget = astex["guided_budget_comparison"]
    assert budget["common_ids_sha256"] == expected_hash
    assert budget["ids_hash_contract"] == ID_HASH_CONTRACT
    for effect in budget["pairwise_deltas"].values():
        assert effect["common_ids_sha256"] == expected_hash
        assert effect["ids_hash_contract"] == ID_HASH_CONTRACT
    slices = astex["cells"]["n100_s10"]["chemistry_slice_guided_vs_unguided"]
    assert slices["strict_supported"]["count"] == 1
    assert slices["nonprotein_only"]["count"] == 1
    assert "paired_effect" not in slices["metal_only"]
    representation = astex["cells"]["n100_s10"][
        "ligand_representation_slice_guided_vs_unguided"
    ]
    assert representation["same_connectivity_representation_mismatch"]["count"] == 1
    integrity = astex["cells"]["n100_s10"][
        "checkpoint_integrity_slice_guided_vs_unguided"
    ]
    assert integrity["split_ligand_identity_overlap"]["count"] == 1


def test_full_sampling_report_rejects_file_drift_after_audit(
    tmp_path: Path, tiny_counts: dict[str, list[str]]
) -> None:
    audit = _write_audit(tmp_path / "audit.json", tiny_counts)
    input_dir = _write_sampling_fixture(tmp_path, audit, tiny_counts)
    csv_path = next(input_dir.glob("*astex-n100-s10-unguided.shard-0.csv"))
    with csv_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["protein_sha256"] = "0" * 64
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="protein changed after audit"):
        build_report(
            input_dir,
            audit,
            expected_shards=2,
            bootstrap_resamples=10,
        )


def test_full_official_report_uses_same_exact_audit_slices(
    tmp_path: Path, tiny_counts: dict[str, list[str]]
) -> None:
    audit = _write_audit(tmp_path / "audit.json", tiny_counts)
    input_dir = _write_official_fixture(tmp_path, tiny_counts)
    report = build_official_report(
        input_dir,
        audit,
        expected_shards=2,
        bootstrap_resamples=20,
    )
    assert report["status"] == "complete_strict_full_cohort_paired_official_posebusters"
    astex = report["datasets"]["astex"]
    assert astex["full_cohort_coverage"]["official_evaluated_per_cell"] == 2
    cell = astex["cells"]["n100_s10"]
    assert cell["guided"]["posebusters_valid_pct"] == 100.0
    expected_hash = sorted_id_sha256(tiny_counts["astex"])
    for arm in ("unguided", "guided"):
        assert cell[arm]["eligible_ids_sha256"] == expected_hash
        assert cell[arm]["ids_hash_contract"] == ID_HASH_CONTRACT
    assert cell["guided_vs_unguided"]["common_ids_sha256"] == expected_hash
    assert cell["guided_vs_unguided"]["ids_hash_contract"] == ID_HASH_CONTRACT
    for effect in astex["guided_budget_comparison"]["pairwise_deltas"].values():
        assert effect["common_ids_sha256"] == expected_hash
        assert effect["ids_hash_contract"] == ID_HASH_CONTRACT
    assert (
        astex["cells"]["n100_s10"]["chemistry_slice_guided_vs_unguided"]["strict_supported"][
            "count"
        ]
        == 1
    )
