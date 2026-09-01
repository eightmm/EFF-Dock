#!/usr/bin/env python3
"""Strict full-cohort report for unified-guidance budget-1000 V2."""

from __future__ import annotations

import argparse
import json
import math
from itertools import combinations
from pathlib import Path
from typing import Any

from effdock.guidance.parameterization import guidance_parameter_identity
from effdock.guidance.provenance import guidance_implementation_identity
from effdock.guidance.system import receptor_policy_identity
from effdock.workflows.benchmark_inputs import load_benchmark_inputs
from effdock.workflows.evaluate import sorted_id_sha256, summarize_rows
from effdock.workflows.guidance_budget_report import (
    _REQUIRED_SUMMARY_KEYS,
    ARM_SCALES,
    CONDITIONS,
    DATASETS,
    DEFAULT_BOOTSTRAP_RESAMPLES,
    DEFAULT_BOOTSTRAP_SEED,
    DEFAULT_EXPECTED_SHARDS,
    EXPECTED_CHECKPOINT_SHA256,
    EXPECTED_CONFIG_SHA256,
    EXPECTED_POCKET_CENTERS_SHA256,
    _aggregate_cell,
    _arm_for_scale,
    _cell_key,
    _require_keys,
    _sha256_file,
    _target_metrics,
    _validate_nested_prior_pairing,
)
from effdock.workflows.guidance_budget_report import (
    _paired_comparison as _legacy_paired_comparison,
)
from effdock.workflows.guidance_coverage_audit import AUDIT_SCHEMA_VERSION, ID_HASH_CONTRACT

PROTOCOL_ID = "EFFDOCK-UNIFIED-GUIDANCE-BUDGET1000-FULL-V2"
RECEPTOR_POLICY = "geometry_only"
EXPECTED_DATASET_COUNTS = {"astex": 85, "posebusters": 308}
BENCHMARK_INPUT_MANIFEST = Path("docs/GUIDANCE_BUDGET1000_FULL_INPUTS.json")
EXPECTED_BENCHMARK_INPUT_MANIFEST_SHA256 = (
    "99f15f557644cc51c3dd1f559b0dd97dd4259c1de3e1403fb761b7c7e079f668"
)
_FULL_REQUIRED_SUMMARY_KEYS = (
    *_REQUIRED_SUMMARY_KEYS,
    "unified_guidance_receptor_policy",
    "guidance_implementation",
    "benchmark_input_identity",
    "require_full_ligand_atom_mapping",
)
_FULL_REQUIRED_ROW_KEYS = {
    "protein_sha256",
    "ligand_reference_sha256",
    "num_match_atoms",
    "num_input_atoms",
    "num_ref_atoms",
    "full_heavy_atom_bijection",
    "ligand_graph_relation",
    "ligand_mapping_metadata_json",
    "exact_full_heavy_atom_graph",
    "ligand_input_identity_sha256",
    "ligand_input_canonical_smiles",
}


def _paired_comparison(
    baseline_rows: dict[str, dict[str, Any]],
    comparison_rows: dict[str, dict[str, Any]],
    ids: tuple[str, ...],
    *,
    baseline_label: str,
    comparison_label: str,
    seed: int,
    resamples: int,
) -> dict[str, Any]:
    """Use the FULL-V2 versioned ID-hash contract in every paired result."""
    result = _legacy_paired_comparison(
        baseline_rows,
        comparison_rows,
        ids,
        baseline_label=baseline_label,
        comparison_label=comparison_label,
        seed=seed,
        resamples=resamples,
    )
    result["common_ids_sha256"] = sorted_id_sha256(list(ids))
    result["ids_hash_contract"] = ID_HASH_CONTRACT
    return result


def _expected_benchmark_input_identity(dataset: str) -> dict[str, object]:
    """Load the exact tracked FULL-V2 input mapping or fail before aggregation."""
    if _sha256_file(BENCHMARK_INPUT_MANIFEST) != EXPECTED_BENCHMARK_INPUT_MANIFEST_SHA256:
        raise ValueError("tracked FULL-V2 benchmark-input manifest SHA-256 mismatch")
    _, identity = load_benchmark_inputs(
        dataset,
        Path("data/external_test"),
        BENCHMARK_INPUT_MANIFEST,
    )
    return identity


def _expected_run_name(dataset: str, num_samples: int, num_steps: int, arm: str) -> str:
    return f"effdock-guidance-budget1000-full-v2-{dataset}-n{num_samples}-s{num_steps}-{arm}"


def _nonempty_unique_ids(values: Any, *, label: str) -> tuple[str, ...]:
    if not isinstance(values, list) or not values:
        raise ValueError(f"{label} must be a non-empty list")
    if any(not isinstance(value, str) or value != value.strip() or not value for value in values):
        raise ValueError(f"{label} must contain trimmed non-empty strings")
    if len(values) != len(set(values)):
        raise ValueError(f"{label} contains duplicate IDs")
    return tuple(sorted(values))


def _unique_ids(values: Any, *, label: str) -> tuple[str, ...]:
    if not isinstance(values, list):
        raise ValueError(f"{label} must be a list")
    if any(not isinstance(value, str) or value != value.strip() or not value for value in values):
        raise ValueError(f"{label} must contain trimmed non-empty strings")
    if len(values) != len(set(values)):
        raise ValueError(f"{label} contains duplicate IDs")
    return tuple(sorted(values))


def _count_from(raw: dict[str, Any], name: str) -> int | None:
    counts = raw.get("counts", {})
    if isinstance(counts, dict) and name in counts:
        return int(counts[name])
    if name in raw:
        return int(raw[name])
    return None


def _ids_from(raw: dict[str, Any], name: str) -> Any:
    values = raw.get("ids")
    if isinstance(values, dict):
        return values.get(name)
    if name in {"discovered", "audited", "success"} and isinstance(values, list):
        return values
    alias = raw.get(f"{name}_ids")
    if alias is not None:
        return alias
    return None


def _declared_ids_hash(raw: dict[str, Any], name: str) -> str | None:
    hashes = raw.get("id_sha256")
    if isinstance(hashes, dict) and name in hashes:
        return str(hashes[name])
    for key in (f"{name}_ids_sha256", "ids_sha256" if name == "discovered" else ""):
        if key and key in raw:
            return str(raw[key])
    return None


def _policy_from(raw: dict[str, Any]) -> str | None:
    direct = raw.get("receptor_policy")
    if isinstance(direct, str):
        return direct
    settings = raw.get("settings")
    if isinstance(settings, dict):
        for key in ("receptor_policy", "unified_guidance_receptor_policy"):
            if isinstance(settings.get(key), str):
                return str(settings[key])
    return None


def _require_exact_identity(
    raw: Any, expected: dict[str, object], *, label: str
) -> dict[str, object]:
    if not isinstance(raw, dict) or not isinstance(raw.get("sha256"), str):
        raise ValueError(f"{label} must be a versioned identity object")
    if json.dumps(raw, sort_keys=True, separators=(",", ":")) != json.dumps(
        expected, sort_keys=True, separators=(",", ":")
    ):
        raise ValueError(f"{label} differs from the current implementation")
    return raw


def _canonical_payload(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _number(mapping: dict[str, Any], names: tuple[str, ...]) -> int:
    for name in names:
        value = mapping.get(name)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return int(value)
    return 0


def _finite_preflight_gate(value: Any, *, label: str) -> None:
    """Reject explicitly declared non-finite audit values without guessing units."""
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = key.lower()
            child_label = f"{label}.{key}"
            if "nonfinite" in normalized and isinstance(child, (int, float)):
                if int(child) != 0:
                    raise ValueError(f"{child_label} must be zero")
            if normalized in {
                "finite",
                "is_finite",
                "all_finite",
                "energy_finite",
                "gradient_finite",
                "finite_energy",
                "finite_gradient",
                "coordinates_finite",
            } and isinstance(child, bool):
                if not child:
                    raise ValueError(f"{child_label} must be true")
            _finite_preflight_gate(child, label=child_label)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _finite_preflight_gate(child, label=f"{label}[{index}]")


def _manifest_slice_and_fallback_inventory(
    raw: dict[str, Any], ids: tuple[str, ...], *, label: str
) -> tuple[dict[str, tuple[str, ...]], dict[str, dict[str, Any]]]:
    raw_slices = raw.get("chemistry_slices")
    if not isinstance(raw_slices, dict) or not raw_slices:
        raise ValueError(f"{label}.chemistry_slices must be a non-empty object")
    slices: dict[str, tuple[str, ...]] = {}
    seen: set[str] = set()
    for name, group in raw_slices.items():
        if not isinstance(group, dict):
            raise ValueError(f"{label}.chemistry_slices.{name} must be an object")
        group_ids = _unique_ids(group.get("ids"), label=f"{label}.chemistry_slices.{name}.ids")
        if int(group.get("count", -1)) != len(group_ids):
            raise ValueError(f"{label}.chemistry_slices.{name} count mismatch")
        if group.get("ids_sha256") != sorted_id_sha256(list(group_ids)):
            raise ValueError(f"{label}.chemistry_slices.{name} ID hash mismatch")
        if outside := set(group_ids) - set(ids):
            raise ValueError(
                f"{label}.chemistry_slices.{name} contains outside IDs {sorted(outside)}"
            )
        if overlap := seen & set(group_ids):
            raise ValueError(
                f"{label}.chemistry_slices are not mutually exclusive: {sorted(overlap)}"
            )
        seen.update(group_ids)
        slices[str(name)] = group_ids
    if seen != set(ids):
        raise ValueError(f"{label}.chemistry_slices must partition the full cohort")

    raw_fallbacks = raw.get("fallback_reasons", {})
    if not isinstance(raw_fallbacks, dict):
        raise ValueError(f"{label}.fallback_reasons must be an object")
    fallbacks: dict[str, dict[str, Any]] = {}
    for reason, group in raw_fallbacks.items():
        if not isinstance(group, dict):
            raise ValueError(f"{label}.fallback_reasons.{reason} must be an object")
        group_ids = _unique_ids(group.get("ids"), label=f"{label}.fallback_reasons.{reason}.ids")
        count = int(group.get("count", -1))
        site_count = int(group.get("site_count", -1))
        if count != len(group_ids) or site_count < count:
            raise ValueError(f"{label}.fallback_reasons.{reason} count mismatch")
        if group.get("ids_sha256") != sorted_id_sha256(list(group_ids)):
            raise ValueError(f"{label}.fallback_reasons.{reason} ID hash mismatch")
        if outside := set(group_ids) - set(ids):
            raise ValueError(f"{label}.fallback_reasons.{reason} has outside IDs {sorted(outside)}")
        fallbacks[str(reason)] = {
            "site_count": site_count,
            "complex_count": count,
            "ids": list(group_ids),
            "ids_sha256": group["ids_sha256"],
        }
    return slices, fallbacks


def _partition_slices(
    raw: Any,
    ids: tuple[str, ...],
    *,
    label: str,
    expected_names: set[str] | None = None,
) -> dict[str, tuple[str, ...]]:
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"{label} must be a non-empty object")
    if expected_names is not None and set(raw) != expected_names:
        raise ValueError(
            f"{label} requires exactly {sorted(expected_names)}, got {sorted(raw)}"
        )
    slices: dict[str, tuple[str, ...]] = {}
    seen: set[str] = set()
    for name, group in raw.items():
        if not isinstance(group, dict):
            raise ValueError(f"{label}.{name} must be an object")
        group_ids = _unique_ids(group.get("ids"), label=f"{label}.{name}.ids")
        if int(group.get("count", -1)) != len(group_ids):
            raise ValueError(f"{label}.{name} count mismatch")
        if group.get("ids_sha256") != sorted_id_sha256(list(group_ids)):
            raise ValueError(f"{label}.{name} ID hash mismatch")
        if outside := set(group_ids) - set(ids):
            raise ValueError(f"{label}.{name} contains outside IDs {sorted(outside)}")
        if overlap := seen & set(group_ids):
            raise ValueError(f"{label} is not mutually exclusive: {sorted(overlap)}")
        seen.update(group_ids)
        slices[str(name)] = group_ids
    if seen != set(ids):
        raise ValueError(f"{label} must partition the full cohort")
    return slices


def _integrity_disclosure_slices(
    identity: dict[str, Any],
    ids: tuple[str, ...],
    *,
    label: str,
) -> tuple[dict[str, Any], dict[str, tuple[str, ...]]]:
    sources = identity.get("sources")
    boundary = sources.get("integrity_boundary") if isinstance(sources, dict) else None
    if not isinstance(boundary, dict):
        raise ValueError(f"{label}.sources.integrity_boundary must be an object")

    ligand_overlap = _unique_ids(
        boundary.get("benchmark_ids_with_split_ligand_identity_overlap"),
        label=f"{label}.integrity_boundary.ligand_identity_overlap",
    )
    exact_entry_overlap = _unique_ids(
        boundary.get("benchmark_ids_with_exact_entry_and_ligand_overlap"),
        label=f"{label}.integrity_boundary.exact_entry_overlap",
    )
    for name, values, count_key in (
        (
            "ligand_identity_overlap",
            ligand_overlap,
            "benchmark_ids_with_split_ligand_identity_overlap_count",
        ),
        (
            "exact_entry_overlap",
            exact_entry_overlap,
            "benchmark_ids_with_exact_entry_and_ligand_overlap_count",
        ),
    ):
        if int(boundary.get(count_key, -1)) != len(values):
            raise ValueError(f"{label}.integrity_boundary.{name} count mismatch")
        if outside := set(values) - set(ids):
            raise ValueError(
                f"{label}.integrity_boundary.{name} has outside IDs {sorted(outside)}"
            )
    if not set(exact_entry_overlap).issubset(ligand_overlap):
        raise ValueError(f"{label}: exact-entry overlap must be a ligand-identity overlap")
    for count_key in ("matching_train_rows", "matching_val_rows"):
        value = boundary.get(count_key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{label}.integrity_boundary.{count_key} must be non-negative")

    id_set = set(ids)
    slices = {
        "split_ligand_identity_overlap": ligand_overlap,
        "no_split_ligand_identity_overlap": tuple(sorted(id_set - set(ligand_overlap))),
        "exact_entry_and_ligand_overlap": exact_entry_overlap,
        "no_exact_entry_and_ligand_overlap": tuple(
            sorted(id_set - set(exact_entry_overlap))
        ),
    }
    return boundary, slices


def _normalize_audit(
    raw: dict[str, Any],
    *,
    dataset_hint: str | None,
    source_path: Path,
    source_sha256: str,
    verify_current_files: bool = False,
) -> dict[str, Any]:
    dataset = str(raw.get("dataset", dataset_hint or ""))
    if dataset not in DATASETS:
        raise ValueError(f"{source_path}: invalid or missing audit dataset {dataset!r}")
    if raw.get("protocol_id") != PROTOCOL_ID:
        raise ValueError(f"{source_path}: full-cohort audit protocol_id mismatch")
    if raw.get("schema_version") != AUDIT_SCHEMA_VERSION:
        raise ValueError(f"{source_path}: full-cohort audit schema mismatch")
    expected_count = EXPECTED_DATASET_COUNTS[dataset]
    discovered_ids = _nonempty_unique_ids(
        _ids_from(raw, "discovered"), label=f"{source_path}:{dataset}.ids.discovered"
    )
    discovered = _count_from(raw, "discovered")
    if discovered is None:
        discovered = len(discovered_ids)
    if discovered != expected_count or len(discovered_ids) != expected_count:
        raise ValueError(
            f"{source_path}:{dataset} requires exactly {expected_count} discovered IDs"
        )
    digest = sorted_id_sha256(list(discovered_ids))
    declared_digest = _declared_ids_hash(raw, "discovered")
    if declared_digest != digest:
        raise ValueError(f"{source_path}:{dataset} discovered ID hash mismatch")

    for name in ("audited", "success"):
        count = _count_from(raw, name)
        values = _ids_from(raw, name)
        current_ids = (
            discovered_ids
            if values is None
            else _nonempty_unique_ids(values, label=f"{source_path}:{dataset}.ids.{name}")
        )
        if count is None:
            count = len(current_ids)
        if count != expected_count or current_ids != discovered_ids:
            raise ValueError(f"{source_path}:{dataset} requires {name} == discovered")
        declared = _declared_ids_hash(raw, name)
        if declared != digest:
            raise ValueError(f"{source_path}:{dataset} {name} ID hash mismatch")

    failed = _count_from(raw, "failed")
    failed_ids = _ids_from(raw, "failed")
    if failed not in (None, 0) or failed_ids not in (None, []):
        raise ValueError(f"{source_path}:{dataset} full-cohort audit contains failures")
    failed_digest = _declared_ids_hash(raw, "failed")
    if failed_digest != sorted_id_sha256([]):
        raise ValueError(f"{source_path}:{dataset} failed ID hash mismatch")
    failure_codes = raw.get("failure_codes", {})
    if not isinstance(failure_codes, dict) or any(
        int(value) != 0 for value in failure_codes.values()
    ):
        raise ValueError(f"{source_path}:{dataset} failure_codes must be empty/zero")
    if raw.get("complete") is not True:
        raise ValueError(f"{source_path}:{dataset} audit complete must be true")
    if raw.get("ids_hash_contract") != ID_HASH_CONTRACT:
        raise ValueError(f"{source_path}:{dataset} ID hash contract mismatch")

    implementation = _require_exact_identity(
        raw.get("implementation"),
        guidance_implementation_identity(),
        label=f"{source_path}:{dataset}.implementation",
    )
    parameter_set = _require_exact_identity(
        raw.get("parameter_set"),
        guidance_parameter_identity(),
        label=f"{source_path}:{dataset}.parameter_set",
    )
    policy_identity = _require_exact_identity(
        raw.get("receptor_policy_identity"),
        receptor_policy_identity(RECEPTOR_POLICY),
        label=f"{source_path}:{dataset}.receptor_policy_identity",
    )

    inputs = raw.get("inputs")
    benchmark_input_identity = (
        inputs.get("benchmark_input_identity") if isinstance(inputs, dict) else None
    )
    expected_input_identity = _expected_benchmark_input_identity(dataset)
    if not isinstance(benchmark_input_identity, dict):
        raise ValueError(
            f"{source_path}:{dataset}.inputs.benchmark_input_identity must be an object"
        )
    if _canonical_payload(benchmark_input_identity) != _canonical_payload(
        expected_input_identity
    ):
        raise ValueError(
            f"{source_path}:{dataset} benchmark-input identity differs from frozen FULL-V2"
        )
    if (
        benchmark_input_identity.get("mode") != "frozen_manifest"
        or benchmark_input_identity.get("dataset") != dataset
        or int(benchmark_input_identity.get("count", -1)) != expected_count
        or benchmark_input_identity.get("ids_sha256") != digest
    ):
        raise ValueError(f"{source_path}:{dataset} invalid frozen benchmark-input identity")
    for identity_hash_key in ("mapping_sha256", "sha256"):
        value = benchmark_input_identity.get(identity_hash_key)
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(
                f"{source_path}:{dataset} invalid benchmark-input {identity_hash_key}"
            )
    sources = benchmark_input_identity.get("sources")
    frozen_manifest = sources.get("frozen_manifest") if isinstance(sources, dict) else None
    if (
        not isinstance(frozen_manifest, dict)
        or frozen_manifest.get("sha256")
        != EXPECTED_BENCHMARK_INPUT_MANIFEST_SHA256
    ):
        raise ValueError(f"{source_path}:{dataset} benchmark-input manifest hash mismatch")
    per_id_input = benchmark_input_identity.get("per_id")
    if not isinstance(per_id_input, dict) or set(per_id_input) != set(discovered_ids):
        raise ValueError(f"{source_path}:{dataset} benchmark-input per-ID coverage mismatch")
    integrity_boundary, integrity_slices = _integrity_disclosure_slices(
        benchmark_input_identity,
        discovered_ids,
        label=f"{source_path}:{dataset}.benchmark_input_identity",
    )

    representation_slices = _partition_slices(
        raw.get("ligand_representation_slices"),
        discovered_ids,
        label=f"{source_path}:{dataset}.ligand_representation_slices",
        expected_names={"exact_graph", "same_connectivity_representation_mismatch"},
    )

    policy = _policy_from(raw)
    complexes = raw.get("complexes")
    if not isinstance(complexes, dict) or set(complexes) != set(discovered_ids):
        raise ValueError(f"{source_path}:{dataset} complexes must exactly cover discovered IDs")
    for complex_id in discovered_ids:
        record = complexes[complex_id]
        if not isinstance(record, dict) or record.get("status") != "success":
            raise ValueError(f"{source_path}:{dataset}/{complex_id} audit status is not success")
        receptor = record.get("receptor")
        if not isinstance(receptor, dict) or receptor.get("mode") != RECEPTOR_POLICY:
            raise ValueError(
                f"{source_path}:{dataset}/{complex_id} receptor mode must be {RECEPTOR_POLICY}"
            )
        if receptor.get("identity_sha256") != policy_identity["sha256"]:
            raise ValueError(
                f"{source_path}:{dataset}/{complex_id} receptor-policy identity mismatch"
            )
        input_record = per_id_input[complex_id]
        if not isinstance(input_record, dict) or record.get(
            "ligand_input_identity_sha256"
        ) != input_record.get("sha256"):
            raise ValueError(
                f"{source_path}:{dataset}/{complex_id} ligand-input identity mismatch"
            )
        if record.get("ligand_input_canonical_smiles") != input_record.get(
            "canonical_heavy_isomeric_smiles"
        ):
            raise ValueError(
                f"{source_path}:{dataset}/{complex_id} canonical ligand input mismatch"
            )
        for hash_key in (
            "protein_sha256",
            "ligand_reference_sha256",
            "system_reference_sha256",
        ):
            value = record.get(hash_key)
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError(
                    f"{source_path}:{dataset}/{complex_id} lacks {hash_key}"
                )
        if verify_current_files:
            for path_key, hash_key in (
                ("protein", "protein_sha256"),
                ("ligand_reference", "ligand_reference_sha256"),
            ):
                current_path = record.get(path_key)
                if not isinstance(current_path, str) or not current_path:
                    raise ValueError(
                        f"{source_path}:{dataset}/{complex_id} lacks {path_key} path"
                    )
                path_value = Path(current_path)
                if not path_value.is_file():
                    raise ValueError(
                        f"{source_path}:{dataset}/{complex_id} input file is missing: "
                        f"{path_value}"
                    )
                if _sha256_file(path_value) != record[hash_key]:
                    raise ValueError(
                        f"{source_path}:{dataset}/{complex_id} {path_key} changed after audit"
                    )
        alignment = record.get("numerical_preflight_reference_alignment")
        if not isinstance(alignment, dict):
            raise ValueError(
                f"{source_path}:{dataset}/{complex_id} lacks ligand mapping provenance"
            )
        expected_relation = next(
            name for name, values in representation_slices.items() if complex_id in values
        )
        if (
            alignment.get("accepted") is not True
            or alignment.get("full_bijection") is not True
            or alignment.get("atom_elements_match") is not True
            or alignment.get("connectivity_match") is not True
            or alignment.get("relation") != expected_relation
            or int(alignment.get("matched_atoms", -1)) <= 0
            or int(alignment.get("matched_atoms", -1))
            != int(alignment.get("input_atoms", -2))
            or int(alignment.get("matched_atoms", -1))
            != int(alignment.get("reference_atoms", -3))
        ):
            raise ValueError(
                f"{source_path}:{dataset}/{complex_id} incomplete ligand atom bijection"
            )
        preflight = record.get("crystal_numerical_preflight")
        if not isinstance(preflight, dict):
            raise ValueError(f"{source_path}:{dataset}/{complex_id} lacks crystal preflight")
        if (
            preflight.get("energy_finite") is not True
            or preflight.get("gradient_finite") is not True
        ):
            raise ValueError(
                f"{source_path}:{dataset}/{complex_id} crystal preflight is non-finite"
            )
        _finite_preflight_gate(record, label=f"{source_path}:{dataset}/{complex_id}")
    if policy != RECEPTOR_POLICY:
        raise ValueError(f"{source_path}:{dataset} receptor policy must be {RECEPTOR_POLICY}")
    _finite_preflight_gate(raw.get("counts", {}), label=f"{source_path}:{dataset}.counts")
    equivalence = raw.get("strict_supported_equivalence")
    if not isinstance(equivalence, dict) or equivalence.get("passed") is not True:
        raise ValueError(f"{source_path}:{dataset} strict-supported equivalence failed")
    mismatch = equivalence.get("mismatch")
    if not isinstance(mismatch, dict) or int(mismatch.get("count", -1)) != 0:
        raise ValueError(f"{source_path}:{dataset} strict-supported equivalence mismatch")
    mismatch_ids = _unique_ids(
        mismatch.get("ids"),
        label=f"{source_path}:{dataset}.strict_supported_equivalence.mismatch.ids",
    )
    if mismatch_ids or mismatch.get("ids_sha256") != sorted_id_sha256([]):
        raise ValueError(f"{source_path}:{dataset} strict-supported equivalence mismatch IDs")
    slices, fallbacks = _manifest_slice_and_fallback_inventory(
        raw, discovered_ids, label=f"{source_path}:{dataset}"
    )
    return {
        "dataset": dataset,
        "source_path": str(source_path),
        "source_sha256": source_sha256,
        "discovered": expected_count,
        "ids": discovered_ids,
        "ids_sha256": digest,
        "policy": RECEPTOR_POLICY,
        "implementation": implementation,
        "parameter_set": parameter_set,
        "receptor_policy_identity": policy_identity,
        "benchmark_input_identity": benchmark_input_identity,
        "integrity_boundary": integrity_boundary,
        "integrity_slices": integrity_slices,
        "chemistry_slices": slices,
        "ligand_representation_slices": representation_slices,
        "fallback_reasons": fallbacks,
        "failure_codes": failure_codes,
        "complexes": complexes,
    }


def load_full_cohort_audits(paths: Path | list[Path] | tuple[Path, ...]) -> dict[str, Any]:
    """Load one combined audit or one dataset-specific audit per dataset."""
    source_paths = [paths] if isinstance(paths, Path) else list(paths)
    if not source_paths:
        raise ValueError("at least one full-cohort audit path is required")
    result: dict[str, Any] = {}
    for path in source_paths:
        raw = json.loads(path.read_text())
        if not isinstance(raw, dict):
            raise ValueError(f"{path}: audit must be a JSON object")
        digest = _sha256_file(path)
        if isinstance(raw.get("datasets"), dict):
            entries = raw["datasets"]
            for dataset, entry in entries.items():
                if not isinstance(entry, dict):
                    raise ValueError(f"{path}: datasets.{dataset} must be an object")
                merged = dict(entry)
                merged.setdefault("dataset", dataset)
                merged.setdefault("protocol_id", raw.get("protocol_id"))
                merged.setdefault("receptor_policy", raw.get("receptor_policy"))
                merged.setdefault("schema_version", raw.get("schema_version"))
                merged.setdefault("implementation", raw.get("implementation"))
                merged.setdefault("parameter_set", raw.get("parameter_set"))
                merged.setdefault("receptor_policy_identity", raw.get("receptor_policy_identity"))
                if dataset in result:
                    raise ValueError(f"duplicate audit for dataset {dataset}")
                result[dataset] = _normalize_audit(
                    merged,
                    dataset_hint=dataset,
                    source_path=path,
                    source_sha256=digest,
                )
        else:
            dataset = str(raw.get("dataset", ""))
            if dataset in result:
                raise ValueError(f"duplicate audit for dataset {dataset}")
            result[dataset] = _normalize_audit(
                raw,
                dataset_hint=None,
                source_path=path,
                source_sha256=digest,
            )
    if set(result) != set(DATASETS):
        raise ValueError(
            f"full-cohort audits must cover exactly {list(DATASETS)}, got {sorted(result)}"
        )
    return result


def validate_full_cohort_audit_for_dataset(path: Path, dataset: str) -> dict[str, Any]:
    """Validate one dataset entry for the Slurm launch gate."""
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict) or not isinstance(raw.get("datasets"), dict):
        raise ValueError(f"{path}: audit requires a datasets object")
    entry = raw["datasets"].get(dataset)
    if not isinstance(entry, dict):
        raise ValueError(f"{path}: audit does not contain {dataset}")
    merged = dict(entry)
    merged.setdefault("dataset", dataset)
    for key in (
        "protocol_id",
        "receptor_policy",
        "schema_version",
        "implementation",
        "parameter_set",
        "receptor_policy_identity",
    ):
        merged.setdefault(key, raw.get(key))
    return _normalize_audit(
        merged,
        dataset_hint=dataset,
        source_path=path,
        source_sha256=_sha256_file(path),
        verify_current_files=True,
    )


def _validate_cross_cell_metadata(
    grouped: dict[tuple[str, str, str], list[tuple[Path, dict[str, Any]]]],
    audits: dict[str, Any],
) -> str:
    flat = [summary for summaries in grouped.values() for _, summary in summaries]
    for key in ("checkpoint_sha256", "confidence_checkpoint_sha256", "config_sha256", "seed"):
        if len({json.dumps(summary[key], sort_keys=True) for summary in flat}) != 1:
            raise ValueError(f"cross-cell metadata mismatch for {key}")
    if any(summary["checkpoint_sha256"] != EXPECTED_CHECKPOINT_SHA256 for summary in flat):
        raise ValueError("protocol docking checkpoint hash mismatch")
    if any(summary["config_sha256"] != EXPECTED_CONFIG_SHA256 for summary in flat):
        raise ValueError("protocol config hash mismatch")
    if any(summary["confidence_checkpoint_sha256"] is not None for summary in flat):
        raise ValueError("protocol requires confidence to be disabled")
    exact_values: dict[str, Any] = {
        "prior_pool_size": 100,
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
        "unified_guidance_receptor_policy": RECEPTOR_POLICY,
        "require_full_ligand_atom_mapping": True,
        "refine": "none",
    }
    for key, expected in exact_values.items():
        for summary in flat:
            actual = summary[key]
            matches = (
                math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-12)
                if isinstance(expected, float)
                else actual == expected
            )
            if not matches:
                raise ValueError(
                    f"protocol setting mismatch for {key}: expected {expected!r}, got {actual!r}"
                )
    for dataset in DATASETS:
        summaries = [
            summary
            for (current, _, _), values in grouped.items()
            if current == dataset
            for _, summary in values
        ]
        if {summary["eligibility_manifest_sha256"] for summary in summaries} != {
            audits[dataset]["source_sha256"]
        }:
            raise ValueError(f"{dataset}: summary full-cohort audit hash mismatch")
        if {summary["pocket_centers_sha256"] for summary in summaries} != {
            EXPECTED_POCKET_CENTERS_SHA256[dataset]
        }:
            raise ValueError(f"{dataset}: protocol pocket-centers hash mismatch")
        if any(
            int(summary.get("num_discovered_total", -1)) != audits[dataset]["discovered"]
            for summary in summaries
        ):
            raise ValueError(f"{dataset}: discovered count differs from full-cohort audit")
        for summary in summaries:
            if summary["guidance_implementation"] != audits[dataset]["implementation"]:
                raise ValueError(f"{dataset}: sampling implementation differs from audit")
            if summary["benchmark_input_identity"] != audits[dataset][
                "benchmark_input_identity"
            ]:
                raise ValueError(
                    f"{dataset}: sampling benchmark-input identity differs from audit"
                )

    guided_hashes: set[str] = set()
    for (dataset, _, arm), summaries in grouped.items():
        for path, summary in summaries:
            parameter_set = summary.get("guidance_parameter_set")
            if arm == "guided":
                if not isinstance(parameter_set, dict) or not parameter_set.get("sha256"):
                    raise ValueError(f"{path}: guided shard lacks guidance parameter hash")
                if parameter_set != audits[dataset]["parameter_set"]:
                    raise ValueError(f"{path}: guided parameter set differs from audit")
                identities = summary.get("guidance_receptor_policy_identities")
                expected_policy = audits[dataset]["receptor_policy_identity"]
                if identities != {expected_policy["sha256"]: expected_policy}:
                    raise ValueError(f"{path}: receptor-policy identity differs from audit")
                guided_hashes.add(str(parameter_set["sha256"]))
            elif parameter_set not in (None, {}):
                raise ValueError(f"{path}: unguided shard unexpectedly has guidance parameters")
    if len(guided_hashes) != 1:
        raise ValueError("guided parameter hash differs across cells")
    return next(iter(guided_hashes))


def _slice_effects(
    baseline_rows: dict[str, dict[str, Any]],
    comparison_rows: dict[str, dict[str, Any]],
    slices: dict[str, tuple[str, ...]],
    *,
    baseline_label: str,
    comparison_label: str,
    seed: int,
    resamples: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, ids in slices.items():
        entry: dict[str, Any] = {
            "count": len(ids),
            "ids_sha256": sorted_id_sha256(list(ids)),
        }
        if ids:
            baseline_slice = {complex_id: baseline_rows[complex_id] for complex_id in ids}
            comparison_slice = {complex_id: comparison_rows[complex_id] for complex_id in ids}
            entry["paired_effect"] = _paired_comparison(
                baseline_slice,
                comparison_slice,
                ids,
                baseline_label=baseline_label,
                comparison_label=comparison_label,
                seed=seed,
                resamples=resamples,
            )
        result[name] = entry
    return result


def _validate_sampling_input_rows(
    rows: dict[str, dict[str, Any]],
    audit: dict[str, Any],
    *,
    dataset: str,
    arm: str,
) -> None:
    """Bind every sampling row to the exact audited files, mapping, and system."""
    for complex_id, row in rows.items():
        missing = sorted(_FULL_REQUIRED_ROW_KEYS - set(row))
        if missing:
            raise ValueError(f"{dataset}/{complex_id}: missing FULL-V2 row fields {missing}")
        audit_record = audit["complexes"][complex_id]
        input_record = audit["benchmark_input_identity"]["per_id"][complex_id]
        if row["ligand_input_identity_sha256"] != input_record["sha256"]:
            raise ValueError(f"{dataset}/{complex_id}: sampling ligand-input hash mismatch")
        if (
            row["ligand_input_canonical_smiles"]
            != input_record["canonical_heavy_isomeric_smiles"]
        ):
            raise ValueError(f"{dataset}/{complex_id}: sampling canonical input mismatch")
        if row["protein_sha256"] != audit_record["protein_sha256"]:
            raise ValueError(f"{dataset}/{complex_id}: protein changed after audit")
        if row["ligand_reference_sha256"] != audit_record["ligand_reference_sha256"]:
            raise ValueError(f"{dataset}/{complex_id}: reference ligand changed after audit")

        matched = int(row["num_match_atoms"])
        if (
            row["full_heavy_atom_bijection"] is not True
            or matched <= 0
            or matched != int(row["num_input_atoms"])
            or matched != int(row["num_ref_atoms"])
        ):
            raise ValueError(f"{dataset}/{complex_id}: incomplete sampling atom bijection")
        relation = str(row["ligand_graph_relation"])
        if relation not in {"exact_graph", "same_connectivity_representation_mismatch"}:
            raise ValueError(f"{dataset}/{complex_id}: invalid ligand graph relation")
        if bool(row["exact_full_heavy_atom_graph"]) != (relation == "exact_graph"):
            raise ValueError(f"{dataset}/{complex_id}: exact-graph alias disagrees with relation")
        try:
            mapping = json.loads(str(row["ligand_mapping_metadata_json"]))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"{dataset}/{complex_id}: invalid ligand mapping metadata JSON"
            ) from exc
        if not isinstance(mapping, dict):
            raise ValueError(f"{dataset}/{complex_id}: ligand mapping metadata must be an object")
        audit_mapping = audit_record["numerical_preflight_reference_alignment"]
        for key in (
            "accepted",
            "relation",
            "match_method",
            "matched_atoms",
            "input_atoms",
            "reference_atoms",
            "full_bijection",
            "atom_elements_match",
            "connectivity_match",
            "bond_orders_match",
            "formal_charges_match",
        ):
            if mapping.get(key) != audit_mapping.get(key):
                raise ValueError(
                    f"{dataset}/{complex_id}: sampling/audit mapping differs for {key}"
                )

        system_hash = str(row.get("guidance_system_reference_sha256", ""))
        if arm == "guided":
            if system_hash != audit_record["system_reference_sha256"]:
                raise ValueError(
                    f"{dataset}/{complex_id}: guided physical-system hash differs from audit"
                )
            if row.get("guidance_topology_reference_sha256") != audit_record.get(
                "topology_reference_sha256"
            ):
                raise ValueError(f"{dataset}/{complex_id}: topology hash differs from audit")
            expected_interaction = audit_record.get("interaction_reference_sha256") or ""
            if row.get("guidance_interaction_reference_sha256") != expected_interaction:
                raise ValueError(
                    f"{dataset}/{complex_id}: interaction topology differs from audit"
                )
        elif system_hash:
            raise ValueError(f"{dataset}/{complex_id}: unguided row has a guidance system hash")


def build_report(
    input_dir: Path,
    cohort_audits: Path | list[Path] | tuple[Path, ...],
    *,
    expected_shards: int = DEFAULT_EXPECTED_SHARDS,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
) -> dict[str, Any]:
    """Reject incomplete V2 output and aggregate exact full-cohort paired effects."""
    if expected_shards < 1 or bootstrap_resamples < 1:
        raise ValueError("expected_shards and bootstrap_resamples must be >= 1")
    audits = load_full_cohort_audits(cohort_audits)
    grouped: dict[tuple[str, str, str], list[tuple[Path, dict[str, Any]]]] = {}
    for path in sorted(input_dir.rglob("*.summary.json")):
        summary = json.loads(path.read_text())
        if not isinstance(summary, dict) or summary.get("protocol_id") != PROTOCOL_ID:
            continue
        _require_keys(summary, _FULL_REQUIRED_SUMMARY_KEYS, label=str(path))
        dataset = str(summary["dataset"])
        if dataset not in DATASETS:
            raise ValueError(f"{path}: unexpected dataset {dataset!r}")
        num_samples, num_steps = int(summary["num_samples"]), int(summary["num_steps"])
        condition = _cell_key(num_samples, num_steps)
        arm = _arm_for_scale(summary["unified_guidance_scale"])
        expected_name = _expected_run_name(dataset, num_samples, num_steps, arm)
        if summary["run_name"] != expected_name:
            raise ValueError(f"{path}: run_name mismatch; expected {expected_name!r}")
        if int(summary["model_pose_step_budget"]) != 1000 or num_samples * num_steps != 1000:
            raise ValueError(f"{path}: learned-model pose-step budget must equal 1000")
        grouped.setdefault((dataset, condition, arm), []).append((path, summary))
    expected_cells = {
        (dataset, condition, arm)
        for dataset in DATASETS
        for condition, _, _ in CONDITIONS
        for arm in ARM_SCALES
    }
    if set(grouped) != expected_cells:
        raise ValueError(
            f"protocol cell mismatch; missing={sorted(expected_cells - set(grouped))}, "
            f"extra={sorted(set(grouped) - expected_cells)}"
        )
    guidance_hash = _validate_cross_cell_metadata(grouped, audits)

    report: dict[str, Any] = {
        "protocol_id": PROTOCOL_ID,
        "status": "complete_strict_full_cohort_paired",
        "claim_boundary": (
            "coverage-policy completion and paired descriptive rerun; external datasets "
            "were opened in V1 and cannot tune or independently confirm guidance"
        ),
        "receptor_policy": RECEPTOR_POLICY,
        "guidance_implementation": audits["astex"]["implementation"],
        "guidance_parameter_set": audits["astex"]["parameter_set"],
        "guidance_parameter_sha256": guidance_hash,
        "receptor_policy_identity": audits["astex"]["receptor_policy_identity"],
        "benchmark_input_manifest": {
            "path": str(BENCHMARK_INPUT_MANIFEST),
            "sha256": EXPECTED_BENCHMARK_INPUT_MANIFEST_SHA256,
            "dataset_identities": {
                dataset: audits[dataset]["benchmark_input_identity"] for dataset in DATASETS
            },
        },
        "audit_id_hash_contract": ID_HASH_CONTRACT,
        "budget": {
            "model_pose_step_budget": 1000,
            "conditions": [
                {"key": key, "num_samples": samples, "num_steps": steps}
                for key, samples, steps in CONDITIONS
            ],
            "prior_pool_size": 100,
        },
        "bootstrap": {
            "method": "paired complex-ID bootstrap, percentile 95% CI",
            "seed": bootstrap_seed,
            "resamples": bootstrap_resamples,
        },
        "coverage_gate": {
            "expected_total": sum(EXPECTED_DATASET_COUNTS.values()),
            "covered_total": sum(audit["discovered"] for audit in audits.values()),
            "failed_total": 0,
        },
        "expected_shards_per_cell": expected_shards,
        "datasets": {},
    }
    rows: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = {}
    for dataset in DATASETS:
        audit = audits[dataset]
        ids = audit["ids"]
        slices = audit["chemistry_slices"]
        representation_slices = audit["ligand_representation_slices"]
        integrity_slices = audit["integrity_slices"]
        dataset_result: dict[str, Any] = {
            "full_cohort_coverage": {
                "discovered": audit["discovered"],
                "audited": audit["discovered"],
                "sampled_per_cell": audit["discovered"],
                "failed": 0,
                "coverage_pct": 100.0,
                "ids_sha256": audit["ids_sha256"],
                "audit_path": audit["source_path"],
                "audit_sha256": audit["source_sha256"],
            },
            "chemistry_slices": {
                name: {
                    "count": len(slice_ids),
                    "ids_sha256": sorted_id_sha256(list(slice_ids)),
                    "ids_hash_contract": ID_HASH_CONTRACT,
                }
                for name, slice_ids in slices.items()
            },
            "ligand_representation_slices": {
                name: {
                    "count": len(slice_ids),
                    "ids_sha256": sorted_id_sha256(list(slice_ids)),
                    "ids_hash_contract": ID_HASH_CONTRACT,
                }
                for name, slice_ids in representation_slices.items()
            },
            "checkpoint_integrity_boundary": audit["integrity_boundary"],
            "checkpoint_integrity_slices": {
                name: {
                    "count": len(slice_ids),
                    "ids_sha256": sorted_id_sha256(list(slice_ids)),
                    "ids_hash_contract": ID_HASH_CONTRACT,
                }
                for name, slice_ids in integrity_slices.items()
            },
            "fallback_reasons": audit["fallback_reasons"],
            "cells": {},
        }
        for condition, _, _ in CONDITIONS:
            cell: dict[str, Any] = {}
            for arm in ARM_SCALES:
                cell_rows, cell_result = _aggregate_cell(
                    grouped[(dataset, condition, arm)],
                    ids,
                    arm=arm,
                    input_dir=input_dir,
                    expected_shards=expected_shards,
                    manifest_discovered=audit["discovered"],
                )
                _validate_sampling_input_rows(
                    cell_rows,
                    audit,
                    dataset=dataset,
                    arm=arm,
                )
                if arm == "guided":
                    for complex_id, row in cell_rows.items():
                        if row.get("guidance_receptor_policy") != RECEPTOR_POLICY:
                            raise ValueError(
                                f"{dataset}/{complex_id}: guided row receptor policy mismatch"
                            )
                        if (
                            row.get("guidance_receptor_policy_identity_sha256")
                            != audit["receptor_policy_identity"]["sha256"]
                        ):
                            raise ValueError(
                                f"{dataset}/{complex_id}: receptor-policy identity hash mismatch"
                            )
                rows[(dataset, condition, arm)] = cell_rows
                cell_result["eligible_ids_sha256"] = sorted_id_sha256(list(ids))
                cell_result["ids_hash_contract"] = ID_HASH_CONTRACT
                cell[arm] = cell_result
            cell["guided_vs_unguided"] = _paired_comparison(
                rows[(dataset, condition, "unguided")],
                rows[(dataset, condition, "guided")],
                ids,
                baseline_label="unguided",
                comparison_label="guided",
                seed=bootstrap_seed,
                resamples=bootstrap_resamples,
            )
            cell["chemistry_slice_guided_vs_unguided"] = _slice_effects(
                rows[(dataset, condition, "unguided")],
                rows[(dataset, condition, "guided")],
                slices,
                baseline_label="unguided",
                comparison_label="guided",
                seed=bootstrap_seed,
                resamples=bootstrap_resamples,
            )
            cell["ligand_representation_slice_guided_vs_unguided"] = _slice_effects(
                rows[(dataset, condition, "unguided")],
                rows[(dataset, condition, "guided")],
                representation_slices,
                baseline_label="unguided",
                comparison_label="guided",
                seed=bootstrap_seed,
                resamples=bootstrap_resamples,
            )
            cell["checkpoint_integrity_slice_guided_vs_unguided"] = _slice_effects(
                rows[(dataset, condition, "unguided")],
                rows[(dataset, condition, "guided")],
                integrity_slices,
                baseline_label="unguided",
                comparison_label="guided",
                seed=bootstrap_seed,
                resamples=bootstrap_resamples,
            )
            dataset_result["cells"][condition] = cell

        dataset_result["prior_pairing"] = _validate_nested_prior_pairing(rows, dataset, ids)
        guided_budget: dict[str, Any] = {
            "common_ids": len(ids),
            "common_ids_sha256": sorted_id_sha256(list(ids)),
            "ids_hash_contract": ID_HASH_CONTRACT,
            "cell_metrics_on_common_ids": {},
            "pairwise_deltas": {},
        }
        for condition, _, _ in CONDITIONS:
            guided_budget["cell_metrics_on_common_ids"][condition] = _target_metrics(
                summarize_rows([rows[(dataset, condition, "guided")][value] for value in ids])
            )
        for (left, _, _), (right, _, _) in combinations(CONDITIONS, 2):
            guided_budget["pairwise_deltas"][f"{right}_minus_{left}"] = _paired_comparison(
                rows[(dataset, left, "guided")],
                rows[(dataset, right, "guided")],
                ids,
                baseline_label=left,
                comparison_label=right,
                seed=bootstrap_seed,
                resamples=bootstrap_resamples,
            )
        dataset_result["guided_budget_comparison"] = guided_budget
        report["datasets"][dataset] = dataset_result
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument(
        "--cohort-audit",
        type=Path,
        action="append",
        required=True,
        help="Combined audit or repeat once per dataset.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-shards", type=int, default=DEFAULT_EXPECTED_SHARDS)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument("--bootstrap-resamples", type=int, default=DEFAULT_BOOTSTRAP_RESAMPLES)
    args = parser.parse_args(argv)
    result = build_report(
        args.input_dir,
        args.cohort_audit,
        expected_shards=args.expected_shards,
        bootstrap_seed=args.bootstrap_seed,
        bootstrap_resamples=args.bootstrap_resamples,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
