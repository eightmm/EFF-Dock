"""Audit and report the frozen FK translation-SDE Astex experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import Chem

PROTOCOL_ID = "EFFDOCK-FK-TRANSLATION-SDE-ASTEX-V1"
NUM_SAMPLES = 40
NUM_STEPS = 25
NUM_SHARDS = 8
EXPECTED_COMPLEXES = 85
COORDINATE_ROUND_DECIMALS = 3


@dataclass(frozen=True)
class ArmSpec:
    name: str
    slug: str
    sde_sigma: float
    fk_beta: float

    @property
    def dynamics(self) -> str:
        if self.sde_sigma > 0.0:
            return "translation_score_corrected_sde_deterministic_so3"
        return "deterministic_ode"

    @property
    def guidance_mode(self) -> str:
        if self.fk_beta > 0.0 and self.sde_sigma > 0.0:
            return "feynman_kac_constraint_resampling_translation_sde"
        if self.fk_beta > 0.0:
            return "feynman_kac_constraint_resampling"
        if self.sde_sigma > 0.0:
            return "translation_score_corrected_sde"
        return "none"


ARM_SPECS = {
    spec.name: spec
    for spec in (
        ArmSpec("ode", "ode", 0.0, 0.0),
        ArmSpec("sde", "sde", 0.3, 0.0),
        ArmSpec("fk_ode", "fk-ode", 0.0, 0.01),
        ArmSpec("fk_sde", "fk-sde", 0.3, 0.01),
    )
}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _as_bool(value: object, *, label: str) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"{label}: expected a boolean, got {value!r}")


def _finite_float(value: object, *, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label}: expected a finite value")
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_close(actual: object, expected: float, *, label: str) -> None:
    value = _finite_float(actual, label=label)
    if not math.isclose(value, expected, rel_tol=0.0, abs_tol=1.0e-9):
        raise ValueError(f"{label}: expected {expected}, got {value}")


def _cohort_ids(path: Path) -> list[str]:
    manifest = _read_json(path)
    try:
        ids = manifest["datasets"]["astex"]["audited_ids"]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"{path}: missing datasets.astex.audited_ids") from exc
    if not isinstance(ids, list) or any(not isinstance(value, str) for value in ids):
        raise ValueError(f"{path}: Astex audited IDs must be a string list")
    if len(ids) != EXPECTED_COMPLEXES or len(set(ids)) != EXPECTED_COMPLEXES:
        raise ValueError(f"{path}: expected {EXPECTED_COMPLEXES} unique Astex IDs")
    return sorted(ids)


def _validate_summary(summary: dict[str, Any], spec: ArmSpec, *, label: str) -> None:
    expected = {
        "dataset": "astex",
        "protocol_id": PROTOCOL_ID,
        "selector_profile": "confidence_cluster_free",
        "checkpoint": "weights/effdock_geometry_ft_100k_best.pt",
        "confidence_checkpoint": (
            "weights/effdock_confidence_extmatch_n80_s25_step42500.pt"
        ),
        "config": "configs/train.yaml",
        "require_full_ligand_atom_mapping": True,
        "num_samples": NUM_SAMPLES,
        "num_steps": NUM_STEPS,
        "model_pose_step_budget": NUM_SAMPLES * NUM_STEPS,
        "prior_pool_size": 100,
        "time_schedule": "late",
        "schedule_power": 3.0,
        "seed": 42,
        "refine": "none",
        "unified_guidance_receptor_policy": "geometry_only",
        "unified_guidance_scale": 0.0,
        "vina_guidance_scale": 0.0,
        "expected_discovered_count": EXPECTED_COMPLEXES,
        "require_complete_success": True,
    }
    for key, value in expected.items():
        if summary.get(key) != value:
            raise ValueError(f"{label}.{key}: expected {value!r}, got {summary.get(key)!r}")
    _require_close(summary.get("sigma"), 0.5, label=f"{label}.sigma")
    _require_close(summary.get("pocket_cutoff"), 10.0, label=f"{label}.pocket_cutoff")
    _require_close(
        summary.get("center_jitter_sigma"), 0.0, label=f"{label}.center_jitter_sigma"
    )
    _require_close(
        summary.get("translation_sde_base_sigma"),
        spec.sde_sigma,
        label=f"{label}.translation_sde_base_sigma",
    )
    dynamics = summary.get("sampling_dynamics_contract")
    if not isinstance(dynamics, dict) or dynamics.get("mode") != spec.dynamics:
        raise ValueError(f"{label}: sampling-dynamics contract drifted")
    if int(summary.get("num_failed", -1)) != 0:
        raise ValueError(f"{label}: failures were recorded")
    if spec.fk_beta > 0.0:
        _require_close(summary.get("fk_constraint_beta"), spec.fk_beta, label=f"{label}.beta")
        if summary.get("fk_resample_times") != [0.3, 0.6, 0.8]:
            raise ValueError(f"{label}: FK resampling schedule drifted")
        if summary.get("fk_resample_method") != "systematic":
            raise ValueError(f"{label}: FK resampling method drifted")
    elif "fk_constraint_beta" in summary:
        raise ValueError(f"{label}: non-FK arm unexpectedly records FK configuration")


def _coordinate_key(coords: np.ndarray) -> bytes:
    if coords.ndim != 2 or coords.shape[1] != 3 or not np.isfinite(coords).all():
        raise ValueError("SDF record contains invalid coordinates")
    scale = 10**COORDINATE_ROUND_DECIMALS
    rounded = np.rint(coords * scale).astype("<i8", copy=False)
    return rounded.tobytes()


def _load_sdf(path: Path, *, expect_ancestry: bool) -> tuple[list[np.ndarray], list[int]]:
    if not path.is_file():
        raise ValueError(f"missing all-poses SDF: {path}")
    molecules = [mol for mol in Chem.SDMolSupplier(str(path), removeHs=False) if mol is not None]
    if len(molecules) != NUM_SAMPLES:
        raise ValueError(f"{path}: expected {NUM_SAMPLES} records, got {len(molecules)}")
    coordinates: list[np.ndarray] = []
    ancestry: list[int] = []
    for index, molecule in enumerate(molecules):
        coords = np.asarray(molecule.GetConformer().GetPositions(), dtype=np.float64)
        _coordinate_key(coords)
        coordinates.append(coords)
        if expect_ancestry:
            if not molecule.HasProp("fk_initial_sample_index"):
                raise ValueError(f"{path}: record {index} lacks FK ancestry")
            ancestor = int(molecule.GetProp("fk_initial_sample_index"))
            if not 0 <= ancestor < NUM_SAMPLES:
                raise ValueError(f"{path}: record {index} has invalid FK ancestry")
            ancestry.append(ancestor)
    return coordinates, ancestry


def _validate_fk_diagnostics(
    row: dict[str, str], spec: ArmSpec, *, label: str
) -> tuple[list[float], int]:
    diagnostics = json.loads(row["fk_diagnostics_json"])
    if diagnostics.get("schema_version") != "effdock.fk_constraint_resampling.v2":
        raise ValueError(f"{label}: FK diagnostics schema drifted")
    expected_dynamics = (
        "translation_score_corrected_sde_deterministic_so3"
        if spec.sde_sigma > 0.0
        else "deterministic_flow_without_score_corrected_sde"
    )
    if diagnostics.get("dynamics") != expected_dynamics:
        raise ValueError(f"{label}: FK dynamics provenance drifted")
    _require_close(
        diagnostics.get("translation_sde_base_sigma"),
        spec.sde_sigma,
        label=f"{label}.translation_sde_base_sigma",
    )
    events = diagnostics.get("events")
    if diagnostics.get("num_resampling_events") != 3 or not isinstance(events, list):
        raise ValueError(f"{label}: expected exactly three FK events")
    if len(events) != 3:
        raise ValueError(f"{label}: expected exactly three FK events")
    ess: list[float] = []
    for index, (event, requested) in enumerate(zip(events, (0.3, 0.6, 0.8))):
        if not isinstance(event, dict) or event.get("event_index") != index:
            raise ValueError(f"{label}: invalid FK event index")
        _require_close(event.get("requested_time"), requested, label=f"{label}.requested_time")
        actual_time = _finite_float(event.get("actual_time"), label=f"{label}.actual_time")
        if not 0.0 < actual_time < 1.0:
            raise ValueError(f"{label}: FK actual time is outside (0, 1)")
        ess_fraction = _finite_float(event.get("ess_fraction"), label=f"{label}.ess")
        if not 0.0 < ess_fraction <= 1.0:
            raise ValueError(f"{label}: FK ESS fraction is outside (0, 1]")
        ess.append(ess_fraction)
        for key in (
            "potential_min",
            "potential_median",
            "potential_max",
            "delta_min",
            "delta_median",
            "delta_max",
            "max_group_weight",
        ):
            _finite_float(event.get(key), label=f"{label}.{key}")
    final_ancestors = int(diagnostics.get("final_unique_initial_ancestors", 0))
    if not 1 <= final_ancestors <= NUM_SAMPLES:
        raise ValueError(f"{label}: invalid final FK ancestry count")
    return ess, final_ancestors


def _validate_row(
    row: dict[str, str], spec: ArmSpec, *, label: str, load_coordinates: bool
) -> dict[str, Any]:
    if row.get("guidance_mode") != spec.guidance_mode:
        raise ValueError(f"{label}: guidance mode drifted")
    if row.get("sampling_dynamics") != spec.dynamics:
        raise ValueError(f"{label}: row dynamics drifted")
    _require_close(
        row.get("translation_sde_base_sigma"),
        spec.sde_sigma,
        label=f"{label}.translation_sde_base_sigma",
    )
    if int(row.get("num_samples", -1)) != NUM_SAMPLES:
        raise ValueError(f"{label}: sample count drifted")
    if int(row.get("prior_pool_size", -1)) != 100:
        raise ValueError(f"{label}: prior-pool size drifted")
    prior_hash = row.get("prior_pool_sha256", "")
    if len(prior_hash) != 64 or any(char not in "0123456789abcdef" for char in prior_hash):
        raise ValueError(f"{label}: invalid prior-pool SHA-256")
    sampling_seed = int(row["sampling_seed"])
    sde_seed = row.get("translation_sde_seed", "")
    parsed_sde_seed: int | None = None
    if spec.sde_sigma > 0.0:
        parsed_sde_seed = int(sde_seed)
        if parsed_sde_seed != (sampling_seed ^ 0x54534445):
            raise ValueError(f"{label}: translation-SDE seed-domain contract drifted")
    elif sde_seed not in {"", "None"}:
        raise ValueError(f"{label}: deterministic arm unexpectedly has an SDE seed")

    ess: list[float] = []
    final_ancestors: int | None = None
    if spec.fk_beta > 0.0:
        ess, final_ancestors = _validate_fk_diagnostics(row, spec, label=label)
    elif row.get("fk_diagnostics_json"):
        raise ValueError(f"{label}: non-FK row unexpectedly has FK diagnostics")

    coordinate_count: int | None = None
    ancestry_count: int | None = None
    coordinates: list[np.ndarray] = []
    ancestry: list[int] = []
    if load_coordinates:
        sdf_path = Path(row.get("all_poses_sdf", ""))
        if int(row.get("all_poses_count", -1)) != NUM_SAMPLES:
            raise ValueError(f"{label}: all-poses count drifted")
        expected_sha = row.get("all_poses_sdf_sha256", "")
        if _sha256(sdf_path) != expected_sha:
            raise ValueError(f"{label}: all-poses SDF hash mismatch")
        coordinates, ancestry = _load_sdf(sdf_path, expect_ancestry=spec.fk_beta > 0.0)
        coordinate_count = len({_coordinate_key(value) for value in coordinates})
        if spec.fk_beta > 0.0:
            ancestry_count = len(set(ancestry))
            if ancestry_count != final_ancestors:
                raise ValueError(f"{label}: SDF ancestry and FK diagnostics disagree")

    return {
        "id": row["id"],
        "sampling_seed": sampling_seed,
        "translation_sde_seed": parsed_sde_seed,
        "prior_pool_sha256": prior_hash,
        "selected_rmsd": _finite_float(row["confidence_rmsd"], label=f"{label}.confidence"),
        "oracle_rmsd": _finite_float(row["oracle_rmsd"], label=f"{label}.oracle"),
        "selected_fast_valid": _as_bool(
            row["confidence_fast_valid"], label=f"{label}.confidence_fast_valid"
        ),
        "num_fast_valid_candidates": int(row["num_fast_valid_candidates"]),
        "coordinate_unique_count": coordinate_count,
        "ancestry_unique_count": ancestry_count,
        "ess_fractions": ess,
        "coordinates": coordinates,
        "ancestry": ancestry,
    }


def _run_name(spec: ArmSpec, *, smoke: bool = False) -> str:
    prefix = "effdock-fk-sde-astex-v1-smoke" if smoke else "effdock-fk-sde-astex-v1"
    return f"{prefix}-{spec.slug}"


def _load_full_arm(input_root: Path, spec: ArmSpec, expected_ids: list[str]) -> dict[str, dict]:
    rows_by_id: dict[str, dict] = {}
    successes = 0
    assigned = 0
    for shard in range(NUM_SHARDS):
        tag = f"{_run_name(spec)}.shard-{shard:03d}-of-{NUM_SHARDS:03d}"
        arm_dir = input_root / spec.name
        summary = _read_json(arm_dir / f"{tag}.summary.json")
        _validate_summary(summary, spec, label=f"{spec.name}.shard{shard}")
        if summary.get("run_name") != _run_name(spec):
            raise ValueError(f"{spec.name}.shard{shard}: run name drifted")
        if summary.get("num_shards") != NUM_SHARDS or summary.get("shard_index") != shard:
            raise ValueError(f"{spec.name}.shard{shard}: shard contract drifted")
        rows = _read_csv(arm_dir / f"{tag}.csv")
        if len(rows) != int(summary.get("num_success", -1)):
            raise ValueError(f"{spec.name}.shard{shard}: CSV/summary count mismatch")
        successes += len(rows)
        assigned += int(summary.get("num_assigned", -1))
        for row in rows:
            complex_id = row.get("id", "")
            if complex_id in rows_by_id:
                raise ValueError(f"{spec.name}: duplicate complex {complex_id}")
            rows_by_id[complex_id] = _validate_row(
                row,
                spec,
                label=f"{spec.name}.{complex_id}",
                load_coordinates=True,
            )
    if successes != EXPECTED_COMPLEXES or assigned != EXPECTED_COMPLEXES:
        raise ValueError(f"{spec.name}: expected {EXPECTED_COMPLEXES} assigned successes")
    if sorted(rows_by_id) != expected_ids:
        raise ValueError(f"{spec.name}: full-cohort ID set mismatch")
    return rows_by_id


def _arm_metrics(rows: dict[str, dict]) -> dict[str, Any]:
    ordered = [rows[key] for key in sorted(rows)]
    selected = [record["selected_rmsd"] for record in ordered]
    oracle = [record["oracle_rmsd"] for record in ordered]
    selected_success = [value < 2.0 for value in selected]
    oracle_success = [value < 2.0 for value in oracle]
    selected_valid = [record["selected_fast_valid"] for record in ordered]
    coordinate_counts = [record["coordinate_unique_count"] for record in ordered]
    if any(value is None for value in coordinate_counts):
        raise ValueError("terminal-coordinate metrics were not loaded")
    result: dict[str, Any] = {
        "complexes": len(ordered),
        "confidence_selected": {
            "rmsd_lt2_count": sum(selected_success),
            "rmsd_lt2_pct": 100.0 * sum(selected_success) / len(ordered),
            "median_rmsd": float(statistics.median(selected)),
            "fast_valid_count": sum(selected_valid),
            "fast_valid_pct": 100.0 * sum(selected_valid) / len(ordered),
            "fast_valid_and_rmsd_lt2_count": sum(
                success and valid for success, valid in zip(selected_success, selected_valid)
            ),
            "fast_valid_and_rmsd_lt2_pct": 100.0
            * sum(success and valid for success, valid in zip(selected_success, selected_valid))
            / len(ordered),
        },
        "oracle": {
            "rmsd_lt2_count": sum(oracle_success),
            "rmsd_lt2_pct": 100.0 * sum(oracle_success) / len(ordered),
            "median_rmsd": float(statistics.median(oracle)),
        },
        "candidate_set": {
            "mean_fast_valid_candidates": float(
                statistics.fmean(record["num_fast_valid_candidates"] for record in ordered)
            ),
            "terminal_unique_coordinate_fraction": float(
                sum(int(value) for value in coordinate_counts) / (len(ordered) * NUM_SAMPLES)
            ),
            "complexes_with_all_terminal_coordinates_unique": sum(
                int(value) == NUM_SAMPLES for value in coordinate_counts
            ),
        },
    }
    ess = [value for record in ordered for value in record["ess_fractions"]]
    ancestry = [record["ancestry_unique_count"] for record in ordered]
    if ess:
        result["feynman_kac"] = {
            "events": len(ess),
            "ess_fraction_min": min(ess),
            "ess_fraction_median": float(statistics.median(ess)),
            "final_unique_initial_ancestor_fraction_mean": float(
                statistics.fmean(int(value) / NUM_SAMPLES for value in ancestry)
            ),
            "final_unique_initial_ancestor_fraction_median": float(
                statistics.median(int(value) / NUM_SAMPLES for value in ancestry)
            ),
        }
    return result


def paired_metrics(
    baseline: dict[str, dict], comparison: dict[str, dict]
) -> dict[str, float | int]:
    if sorted(baseline) != sorted(comparison):
        raise ValueError("paired arms have different ID sets")
    ids = sorted(baseline)
    base_selected = [baseline[key]["selected_rmsd"] for key in ids]
    comp_selected = [comparison[key]["selected_rmsd"] for key in ids]
    base_oracle = [baseline[key]["oracle_rmsd"] for key in ids]
    comp_oracle = [comparison[key]["oracle_rmsd"] for key in ids]
    base_success = [value < 2.0 for value in base_selected]
    comp_success = [value < 2.0 for value in comp_selected]
    base_oracle_success = [value < 2.0 for value in base_oracle]
    comp_oracle_success = [value < 2.0 for value in comp_oracle]
    return {
        "complexes": len(ids),
        "confidence_selected_rmsd_lt2_count_delta": sum(comp_success) - sum(base_success),
        "confidence_selected_rmsd_lt2_pct_delta": 100.0
        * (sum(comp_success) - sum(base_success))
        / len(ids),
        "confidence_selected_gained_complexes": sum(
            (not before) and after for before, after in zip(base_success, comp_success)
        ),
        "confidence_selected_lost_complexes": sum(
            before and (not after) for before, after in zip(base_success, comp_success)
        ),
        "confidence_selected_paired_median_rmsd_delta": float(
            statistics.median(after - before for before, after in zip(base_selected, comp_selected))
        ),
        "oracle_rmsd_lt2_count_delta": sum(comp_oracle_success) - sum(base_oracle_success),
        "oracle_rmsd_lt2_pct_delta": 100.0
        * (sum(comp_oracle_success) - sum(base_oracle_success))
        / len(ids),
        "oracle_paired_median_rmsd_delta": float(
            statistics.median(after - before for before, after in zip(base_oracle, comp_oracle))
        ),
        "terminal_unique_coordinate_fraction_delta": float(
            statistics.fmean(
                (
                    int(comparison[key]["coordinate_unique_count"])
                    - int(baseline[key]["coordinate_unique_count"])
                )
                / NUM_SAMPLES
                for key in ids
            )
        ),
    }


def _validate_shared_priors(arms: dict[str, dict[str, dict]]) -> None:
    ids = sorted(next(iter(arms.values())))
    for complex_id in ids:
        observations = [
            (rows[complex_id]["sampling_seed"], rows[complex_id]["prior_pool_sha256"])
            for rows in arms.values()
        ]
        if len(set(observations)) != 1:
            raise ValueError(f"{complex_id}: arm prior/seed identity mismatch")


def build_report(input_root: Path, cohort_manifest: Path) -> dict[str, Any]:
    expected_ids = _cohort_ids(cohort_manifest)
    arms = {
        name: _load_full_arm(input_root, spec, expected_ids)
        for name, spec in ARM_SPECS.items()
    }
    _validate_shared_priors(arms)
    metrics = {name: _arm_metrics(rows) for name, rows in arms.items()}
    primary = paired_metrics(arms["fk_ode"], arms["fk_sde"])
    contextual = paired_metrics(arms["ode"], arms["sde"])
    selected_gate = primary["confidence_selected_rmsd_lt2_pct_delta"] >= 2.0
    oracle_gate = primary["oracle_rmsd_lt2_pct_delta"] >= -2.0
    diversity_gate = (
        metrics["fk_sde"]["candidate_set"]["terminal_unique_coordinate_fraction"] > 0.90
    )
    return {
        "schema_version": "effdock.fk_sde_astex_report.v1",
        "protocol_id": PROTOCOL_ID,
        "claim_boundary": "paired_descriptive_external_evidence_only",
        "cohort_manifest": str(cohort_manifest),
        "cohort_manifest_sha256": _sha256(cohort_manifest),
        "checks": {
            "four_arms_complete_85_of_85": True,
            "model_pose_step_budget_exact_1000": True,
            "shared_prior_and_sampling_seed_per_complex": True,
            "all_pose_sdf_hashes_and_counts_exact": True,
            "fk_diagnostics_complete_and_finite": True,
        },
        "arms": metrics,
        "primary_contrast_fk_sde_minus_fk_ode": primary,
        "contextual_contrast_sde_minus_ode": contextual,
        "predeclared_hypothesis": {
            "selected_gain_at_least_2pp": selected_gate,
            "oracle_decrease_not_worse_than_2pp": oracle_gate,
            "fk_sde_terminal_unique_fraction_above_0p90": diversity_gate,
            "supported": selected_gate and oracle_gate and diversity_gate,
        },
    }


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# FK Translation-SDE Astex Results",
        "",
        f"- Protocol: `{report['protocol_id']}`",
        "- Claim boundary: paired descriptive external evidence only",
        "",
        "| Arm | Selected <2 A | Oracle <2 A | Selected median | Terminal unique |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ARM_SPECS:
        arm = report["arms"][name]
        selected = arm["confidence_selected"]
        oracle = arm["oracle"]
        candidates = arm["candidate_set"]
        lines.append(
            f"| {name} | {selected['rmsd_lt2_count']}/85 "
            f"({selected['rmsd_lt2_pct']:.2f}%) | {oracle['rmsd_lt2_count']}/85 "
            f"({oracle['rmsd_lt2_pct']:.2f}%) | {selected['median_rmsd']:.3f} A | "
            f"{candidates['terminal_unique_coordinate_fraction']:.3f} |"
        )
    primary = report["primary_contrast_fk_sde_minus_fk_ode"]
    hypothesis = report["predeclared_hypothesis"]
    lines.extend(
        [
            "",
            "## Primary contrast: FK-SDE minus FK-ODE",
            "",
            f"- Selected <2 A: {primary['confidence_selected_rmsd_lt2_pct_delta']:+.2f} pp",
            f"- Oracle <2 A: {primary['oracle_rmsd_lt2_pct_delta']:+.2f} pp",
            f"- Paired median selected RMSD: "
            f"{primary['confidence_selected_paired_median_rmsd_delta']:+.3f} A",
            f"- Terminal uniqueness: "
            f"{primary['terminal_unique_coordinate_fraction_delta']:+.3f}",
            f"- Predeclared hypothesis supported: `{str(hypothesis['supported']).lower()}`",
            "",
            "Astex was already opened; these values cannot tune settings or admit the method.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def audit_smoke(input_root: Path) -> dict[str, Any]:
    records: dict[str, dict[str, Any]] = {}
    for name, spec in ARM_SPECS.items():
        run_name = _run_name(spec, smoke=True)
        arm_dir = input_root / name
        summary = _read_json(arm_dir / f"{run_name}.summary.json")
        _validate_summary(summary, spec, label=f"smoke.{name}")
        if summary.get("run_name") != run_name:
            raise ValueError(f"smoke.{name}: run name drifted")
        if summary.get("num_success") != 1 or summary.get("num_assigned") != 1:
            raise ValueError(f"smoke.{name}: expected one successful assigned complex")
        rows = _read_csv(arm_dir / f"{run_name}.csv")
        if len(rows) != 1 or rows[0].get("id") != "1jje":
            raise ValueError(f"smoke.{name}: expected exactly 1jje")
        records[name] = _validate_row(
            rows[0], spec, label=f"smoke.{name}.1jje", load_coordinates=True
        )
    _validate_shared_priors({name: {"1jje": record} for name, record in records.items()})

    ode_coords = records["ode"]["coordinates"]
    fk_ode = records["fk_ode"]
    subset_errors = []
    for coords, ancestor in zip(fk_ode["coordinates"], fk_ode["ancestry"]):
        reference = ode_coords[ancestor]
        if coords.shape != reference.shape:
            raise ValueError("smoke: ODE/FK-ODE atom shapes differ")
        subset_errors.append(float(np.linalg.norm(coords - reference, axis=1).max()))
    max_subset_error = max(subset_errors)
    if not math.isfinite(max_subset_error) or max_subset_error > 5.0e-3:
        raise ValueError("smoke: deterministic FK outputs are not an ODE trajectory subset")

    fk_sde = records["fk_sde"]
    groups: dict[int, list[bytes]] = {}
    for coords, ancestor in zip(fk_sde["coordinates"], fk_sde["ancestry"]):
        groups.setdefault(ancestor, []).append(_coordinate_key(coords))
    duplicated = [values for values in groups.values() if len(values) > 1]
    if not duplicated:
        raise ValueError("smoke: FK-SDE did not duplicate a parent, so clone diversity is untested")
    if any(len(set(values)) <= 1 for values in duplicated):
        raise ValueError("smoke: at least one duplicated FK-SDE parent failed to diversify")

    result = {
        "schema_version": "effdock.fk_sde_astex_smoke_audit.v1",
        "protocol_id": PROTOCOL_ID,
        "status": "passed",
        "scientific_outcomes_inspected": False,
        "checks": {
            "four_arms_one_of_one_complete": True,
            "model_pose_step_budget_exact_1000": True,
            "prior_hash_and_sampling_seed_shared": True,
            "sampling_dynamics_provenance_exact": True,
            "three_finite_fk_events_per_fk_arm": True,
            "deterministic_fk_is_ode_trajectory_subset": True,
            "duplicated_fk_sde_descendants_diversify": True,
        },
        "max_fk_ode_subset_atom_error_angstrom": max_subset_error,
        "fk_ode_final_unique_initial_ancestors": fk_ode["ancestry_unique_count"],
        "fk_sde_final_unique_initial_ancestors": fk_sde["ancestry_unique_count"],
        "fk_sde_terminal_unique_coordinates": fk_sde["coordinate_unique_count"],
        "fk_sde_duplicated_parent_groups": len(duplicated),
    }
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    smoke = subparsers.add_parser("smoke")
    smoke.add_argument("--input-root", type=Path, required=True)
    smoke.add_argument("--output", type=Path, required=True)
    report = subparsers.add_parser("report")
    report.add_argument("--input-root", type=Path, required=True)
    report.add_argument("--cohort-manifest", type=Path, required=True)
    report.add_argument("--output-json", type=Path, required=True)
    report.add_argument("--output-markdown", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.command == "smoke":
        result = audit_smoke(args.input_root)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    else:
        result = build_report(args.input_root, args.cohort_manifest)
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        _write_markdown(result, args.output_markdown)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
