"""Audit and compare the paired 10k exact-t0 dose branches."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import yaml
from audit_early_time_ft_50k import (
    REQUIRED_ROLLOUT_METRICS,
    _atomic_write_json,
    _require_state_mapping_equal,
    _success_count,
    audit,
)

from effdock.checkpoint import extract_ema_model_state, load_checkpoint_file

EXPECTED_COMMON_INIT_SHA256 = (
    "65be44d7dc8f0867eb9fc5d22214b80f93971ea4702679a527c665046e91e6b6"
)
EXPECTED_PARENT_SHA256 = (
    "ad6c794851698294b38246dd173035e0d336b9e12bc5a1c91289b241c22b3756"
)
EXPECTED_SPLIT_SHA256 = (
    "3ac570bf08bced053f1ce040b57efca27c3be616f29a82cd66ef887c08860e6b"
)
EXPECTED_CODE_SHA256 = (
    "6b41a2f744b9f9678f870f4121160479ed559546306b134ca808fae40664e3cc"
)
EXPECTED_CONFIG_SHA256 = {
    "control_t0p10": "13d2f9d8b7d64eb4b0286c6cdf84cba0110364991f9d7f16eba94e5f1be1cbde",
    "treatment_t0p15": "e24d3068d8c2d768c1643fec581cca41098acb2edb1bd431b29c16ddde0972f4",
}
EXPECTED_BRANCH_FIELDS = {
    "control_t0p10": {
        "task_id": 0,
        "weights": [0.80, 0.10, 0.10],
        "wandb_run_name": "effdock-t0-dose-control-t0p10-10k-v1",
        "output_dir": (
            "outputs/eff-dock/early-time-t0-dose-control-t0p10-10k-v1-20260814"
        ),
    },
    "treatment_t0p15": {
        "task_id": 1,
        "weights": [0.80, 0.05, 0.15],
        "wandb_run_name": "effdock-t0-dose-treatment-t0p15-10k-v1",
        "output_dir": (
            "outputs/eff-dock/early-time-t0-dose-treatment-t0p15-10k-v1-20260814"
        ),
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"{path} does not contain a config mapping")
    return value


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if not isinstance(value, dict):
        return {prefix: value}
    flattened: dict[str, Any] = {}
    for key, child in value.items():
        child_prefix = f"{prefix}.{key}" if prefix else str(key)
        flattened.update(_flatten(child, child_prefix))
    return flattened


def _require_config_contract(control_path: Path, treatment_path: Path) -> None:
    configs = {
        "control_t0p10": _read_yaml(control_path),
        "treatment_t0p15": _read_yaml(treatment_path),
    }
    for branch, config in configs.items():
        expected = EXPECTED_BRANCH_FIELDS[branch]
        if config["data"].get("time_early_replay_weights") != expected["weights"]:
            raise RuntimeError(f"{branch} has the wrong registered time mixture")
        if config["logging"].get("wandb_run_name") != expected["wandb_run_name"]:
            raise RuntimeError(f"{branch} has the wrong run name")
        if config["logging"].get("output_dir") != expected["output_dir"]:
            raise RuntimeError(f"{branch} has the wrong output directory")

    control_flat = _flatten(configs["control_t0p10"])
    treatment_flat = _flatten(configs["treatment_t0p15"])
    differing = {
        key
        for key in set(control_flat) | set(treatment_flat)
        if control_flat.get(key) != treatment_flat.get(key)
    }
    expected_differences = {
        "data.time_early_replay_weights",
        "logging.output_dir",
        "logging.wandb_run_name",
    }
    if differing != expected_differences:
        raise RuntimeError(
            "paired configs violate the single-change contract: "
            f"differences={sorted(differing)}"
        )


def _require_run_identity(
    *,
    output_dir: Path,
    config_path: Path,
    common_init: Path,
    expected_array_job_id: str,
    branch: str,
) -> dict[str, Any]:
    manifest_path = output_dir / "run_identity.json"
    with manifest_path.open(encoding="utf-8") as handle:
        identity = json.load(handle)
    expected_fields = {
        "schema_version": 1,
        "status": "completed",
        "producer_array_job_id": expected_array_job_id,
        "producer_task_id": EXPECTED_BRANCH_FIELDS[branch]["task_id"],
        "branch": branch,
        "config": str(config_path),
        "config_sha256": EXPECTED_CONFIG_SHA256[branch],
        "split": "data/splits/plinder.json",
        "split_sha256": EXPECTED_SPLIT_SHA256,
        "common_init": str(common_init),
        "common_init_sha256": EXPECTED_COMMON_INIT_SHA256,
        "common_init_step": 50_000,
        "code_sha256": EXPECTED_CODE_SHA256,
        "output_dir": str(output_dir),
        "expected_steps": 10_000,
    }
    mismatches = {
        key: (identity.get(key), expected)
        for key, expected in expected_fields.items()
        if identity.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(f"{branch} run identity mismatch: {mismatches}")
    if not str(identity.get("producer_job_id", "")):
        raise RuntimeError(f"{branch} has no producer job ID")

    expected_artifacts = {
        "baseline": (output_dir / "checkpoints" / "rollout_step0000000.pt", 0),
        "endpoint": (output_dir / "checkpoints" / "rollout_step0010000.pt", 10_000),
        "latest": (output_dir / "checkpoints" / "latest.pt", 10_000),
    }
    records = identity.get("artifacts")
    if not isinstance(records, dict):
        raise RuntimeError(f"{branch} has no completed artifact inventory")
    for name, (path, step) in expected_artifacts.items():
        record = records.get(name)
        if not isinstance(record, dict):
            raise RuntimeError(f"{branch} manifest is missing {name}")
        if record.get("path") != str(path) or record.get("step") != step:
            raise RuntimeError(f"{branch} {name} manifest record is inconsistent")
        if record.get("sha256") != _sha256(path):
            raise RuntimeError(f"{branch} {name} changed after producer completion")
    return identity


def _require_shared_baseline(control: dict, treatment: dict) -> None:
    for key in sorted(REQUIRED_ROLLOUT_METRICS):
        control_value = float(control["final_metrics"][key])
        treatment_value = float(treatment["final_metrics"][key])
        if not math.isclose(control_value, treatment_value, rel_tol=0.0, abs_tol=1e-6):
            raise RuntimeError(
                f"paired S0 mismatch for {key}: {control_value} != {treatment_value}"
            )


def _require_common_initialization(
    common_init_path: Path,
    control_s0: dict[str, Any],
    treatment_s0: dict[str, Any],
) -> None:
    if _sha256(common_init_path) != EXPECTED_COMMON_INIT_SHA256:
        raise RuntimeError("common EMA initialization hash mismatch")
    common = load_checkpoint_file(common_init_path)
    expected_metadata = {
        "artifact_type": "effdock_ema_inference_checkpoint",
        "inference_only": True,
        "weight_source": "ema",
        "source_checkpoint_sha256": EXPECTED_PARENT_SHA256,
        "source_checkpoint_step": 50_000,
        "step": 50_000,
    }
    mismatches = {
        key: (common.get(key), expected)
        for key, expected in expected_metadata.items()
        if common.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(f"common EMA initialization metadata mismatch: {mismatches}")
    _require_state_mapping_equal(
        common.get("model_state_dict"),
        extract_ema_model_state(common),
        context="common raw model versus promoted EMA",
    )
    for branch, s0 in (("control", control_s0), ("treatment", treatment_s0)):
        if int(s0.get("step", -1)) != 0:
            raise RuntimeError(f"{branch} baseline is not S0")
        _require_state_mapping_equal(
            s0.get("model_state_dict"),
            common.get("model_state_dict"),
            context=f"{branch} S0 raw model versus common initialization",
        )
        _require_state_mapping_equal(
            s0.get("ema_state_dict"),
            common.get("ema_state_dict"),
            context=f"{branch} S0 EMA versus common initialization",
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-dir", type=Path, required=True)
    parser.add_argument("--treatment-dir", type=Path, required=True)
    parser.add_argument("--control-config", type=Path, required=True)
    parser.add_argument("--treatment-config", type=Path, required=True)
    parser.add_argument("--common-init", type=Path, required=True)
    parser.add_argument("--expected-producer-array-job-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    common = {
        "expected_steps": 10_000,
        "rollout_every": 2_000,
        "expected_val_samples": 1_076,
        "expected_world_size": 4,
    }
    try:
        _require_config_contract(args.control_config, args.treatment_config)
        if _sha256(args.control_config) != EXPECTED_CONFIG_SHA256["control_t0p10"]:
            raise RuntimeError("control config hash mismatch")
        if _sha256(args.treatment_config) != EXPECTED_CONFIG_SHA256["treatment_t0p15"]:
            raise RuntimeError("treatment config hash mismatch")
        identities = {
            "control": _require_run_identity(
                output_dir=args.control_dir,
                config_path=args.control_config,
                common_init=args.common_init,
                expected_array_job_id=args.expected_producer_array_job_id,
                branch="control_t0p10",
            ),
            "treatment": _require_run_identity(
                output_dir=args.treatment_dir,
                config_path=args.treatment_config,
                common_init=args.common_init,
                expected_array_job_id=args.expected_producer_array_job_id,
                branch="treatment_t0p15",
            ),
        }
        control = audit(
            output_dir=args.control_dir,
            config_path=args.control_config,
            **common,
        )
        treatment = audit(
            output_dir=args.treatment_dir,
            config_path=args.treatment_config,
            **common,
        )

        # Compare the named S0 files, not each branch endpoint summary.
        control_s0 = load_checkpoint_file(
            args.control_dir / "checkpoints" / "rollout_step0000000.pt"
        )
        treatment_s0 = load_checkpoint_file(
            args.treatment_dir / "checkpoints" / "rollout_step0000000.pt"
        )
        _require_common_initialization(args.common_init, control_s0, treatment_s0)
        _require_shared_baseline(
            {"final_metrics": control_s0["metrics"]},
            {"final_metrics": treatment_s0["metrics"]},
        )

        control_final = control["final_metrics"]
        treatment_final = treatment["final_metrics"]
        control_success_2a_count = _success_count(
            control_final["rollout/success_2A"],
            1_076,
            context="control endpoint rollout/success_2A",
        )
        treatment_success_2a_count = _success_count(
            treatment_final["rollout/success_2A"],
            1_076,
            context="treatment endpoint rollout/success_2A",
        )
        control_success_5a_count = _success_count(
            control_final["rollout/success_5A"],
            1_076,
            context="control endpoint rollout/success_5A",
        )
        treatment_success_5a_count = _success_count(
            treatment_final["rollout/success_5A"],
            1_076,
            context="treatment endpoint rollout/success_5A",
        )
        success_2a_count_delta = treatment_success_2a_count - control_success_2a_count
        success_5a_count_delta = treatment_success_5a_count - control_success_5a_count
        median_delta = float(treatment_final["rollout/rmsd_median"]) - float(
            control_final["rollout/rmsd_median"]
        )
        shared_s0_success_2a_count = _success_count(
            control_s0["metrics"]["rollout/success_2A"],
            1_076,
            context="shared S0 rollout/success_2A",
        )
        gates = {
            "primary_treatment_minus_control_ge_11_successes": (
                success_2a_count_delta >= 11
            ),
            "treatment_retains_shared_s0_success": (
                treatment_success_2a_count >= shared_s0_success_2a_count
            ),
            "success_5A_not_worse_than_control_by_over_5_successes": (
                success_5a_count_delta >= -5
            ),
            "median_rmsd_not_worse_than_control_by_over_0p10A": median_delta <= 0.10,
        }
        result = {
            "status": "passed",
            "producer_array_job_id": args.expected_producer_array_job_id,
            "scientific_gate_pass": all(gates.values()),
            "gates": gates,
            "run_identities": identities,
            "shared_s0_success_2A_count": shared_s0_success_2a_count,
            "shared_s0_success_2A": shared_s0_success_2a_count / 1_076,
            "control": control,
            "treatment": treatment,
            "treatment_minus_control": {
                "success_2A_pp": 100.0 * success_2a_count_delta / 1_076,
                "success_2A_count": success_2a_count_delta,
                "success_5A_pp": 100.0 * success_5a_count_delta / 1_076,
                "success_5A_count": success_5a_count_delta,
                "rmsd_median_A": median_delta,
            },
        }
    except Exception as error:
        _atomic_write_json(
            {"status": "failed", "error": f"{type(error).__name__}: {error}"},
            args.output,
        )
        raise

    _atomic_write_json(result, args.output)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
