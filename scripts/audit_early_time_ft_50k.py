"""Audit the registered 50k early-time fine-tune after its Slurm job exits."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import torch
import yaml

from effdock.checkpoint import load_checkpoint_file

REQUIRED_FINAL_METRICS = {
    "train_loss",
    "val_loss",
    "val_loss_v",
    "val_loss_omega",
    "rollout/rmsd_median",
    "rollout/rmsd_mean",
    "rollout/rmsd_p25",
    "rollout/rmsd_p75",
    "rollout/success_2A",
    "rollout/success_5A",
    "rollout/centroid_dist",
    "rollout/frag_rmsd",
}
REQUIRED_ROLLOUT_METRICS = {
    "val_loss",
    "val_loss_v",
    "val_loss_omega",
    "rollout/rmsd_median",
    "rollout/rmsd_mean",
    "rollout/rmsd_p25",
    "rollout/rmsd_p75",
    "rollout/success_2A",
    "rollout/success_5A",
    "rollout/centroid_dist",
    "rollout/frag_rmsd",
}


def _atomic_write_json(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _require_finite(metrics: dict[str, Any], *, context: str) -> None:
    bad = {
        key: value
        for key, value in metrics.items()
        if not isinstance(value, int | float) or not math.isfinite(float(value))
    }
    if bad:
        raise RuntimeError(f"{context} contains non-finite/non-scalar metrics: {bad!r}")


def _rollout_step(path: Path) -> int:
    return int(path.stem.removeprefix("rollout_step"))


def _require_state_mapping_equal(
    left: object,
    right: object,
    *,
    context: str,
) -> None:
    if not isinstance(left, dict) or not isinstance(right, dict):
        raise RuntimeError(f"{context} is missing a state mapping")
    if set(left) != set(right):
        raise RuntimeError(f"{context} state keys differ")
    for key in left:
        left_value = left[key]
        right_value = right[key]
        if not isinstance(left_value, torch.Tensor) or not isinstance(
            right_value, torch.Tensor
        ):
            raise RuntimeError(f"{context} state value {key!r} is not a tensor")
        if not torch.equal(left_value, right_value):
            raise RuntimeError(f"{context} tensor differs at {key!r}")


def _success_count(value: object, expected_val_samples: int, *, context: str) -> int:
    success = float(value)
    count = round(success * expected_val_samples)
    if not 0 <= count <= expected_val_samples:
        raise RuntimeError(f"{context} is outside [0, 1]: {success}")
    if not math.isclose(
        success,
        count / expected_val_samples,
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        raise RuntimeError(
            f"{context} is not an integer count/{expected_val_samples}: {success}"
        )
    return count


def audit(
    *,
    output_dir: Path,
    config_path: Path,
    expected_steps: int,
    rollout_every: int,
    expected_val_samples: int,
    expected_world_size: int,
) -> dict[str, Any]:
    checkpoint_dir = output_dir / "checkpoints"
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    expected_rollout_steps = list(range(0, expected_steps + 1, rollout_every))
    rollout_paths = sorted(checkpoint_dir.glob("rollout_step*.pt"))
    actual_rollout_steps = [_rollout_step(path) for path in rollout_paths]
    if actual_rollout_steps != expected_rollout_steps:
        raise RuntimeError(
            "named rollout checkpoint inventory mismatch: "
            f"expected={expected_rollout_steps}, actual={actual_rollout_steps}"
        )

    successes: dict[int, float] = {}
    running_best = -math.inf
    for step, path in zip(actual_rollout_steps, rollout_paths, strict=True):
        checkpoint = load_checkpoint_file(path)
        if int(checkpoint.get("step", -1)) != step:
            raise RuntimeError(f"{path.name} embeds step={checkpoint.get('step')!r}")
        if checkpoint.get("config") != config:
            raise RuntimeError(f"{path.name} config differs from the registered config")
        rank_rng_states = checkpoint.get("rank_rng_states")
        if not isinstance(rank_rng_states, list) or len(rank_rng_states) != expected_world_size:
            raise RuntimeError(f"{path.name} has the wrong per-rank RNG inventory")
        metrics = checkpoint.get("metrics")
        if not isinstance(metrics, dict):
            raise RuntimeError(f"{path.name} has no metric mapping")
        missing_metrics = sorted(REQUIRED_ROLLOUT_METRICS - set(metrics))
        if missing_metrics:
            raise RuntimeError(f"{path.name} is missing rollout metrics: {missing_metrics}")
        _require_finite(metrics, context=path.name)
        successes[step] = float(metrics["rollout/success_2A"])
        _success_count(
            metrics["rollout/success_2A"],
            expected_val_samples,
            context=f"{path.name} rollout/success_2A",
        )
        _success_count(
            metrics["rollout/success_5A"],
            expected_val_samples,
            context=f"{path.name} rollout/success_5A",
        )
        if checkpoint.get("best_selection_metric") != "rollout/success_2A":
            raise RuntimeError(f"{path.name} has the wrong selection metric")
        if checkpoint.get("best_selection_mode") != "max":
            raise RuntimeError(f"{path.name} has the wrong selection mode")
        running_best = max(running_best, successes[step])
        if not math.isclose(
            float(checkpoint.get("best_selection_value", math.nan)),
            running_best,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise RuntimeError(f"{path.name} has an inconsistent running best value")

    baseline = load_checkpoint_file(rollout_paths[0])
    endpoint = load_checkpoint_file(rollout_paths[-1])
    latest = load_checkpoint_file(checkpoint_dir / "latest.pt")
    best = load_checkpoint_file(checkpoint_dir / "best.pt")

    if int(latest.get("step", -1)) != expected_steps:
        raise RuntimeError(f"latest.pt embeds step={latest.get('step')!r}")
    if latest.get("config") != config:
        raise RuntimeError("latest.pt config differs from the registered config")
    rank_rng_states = latest.get("rank_rng_states")
    if not isinstance(rank_rng_states, list) or len(rank_rng_states) != expected_world_size:
        raise RuntimeError("latest.pt does not contain the expected per-rank RNG inventory")

    final_metrics = latest.get("metrics")
    if not isinstance(final_metrics, dict):
        raise RuntimeError("latest.pt has no final metric mapping")
    missing_metrics = sorted(REQUIRED_FINAL_METRICS - set(final_metrics))
    if missing_metrics:
        raise RuntimeError(f"latest.pt is missing final metrics: {missing_metrics}")
    _require_finite(final_metrics, context="latest.pt")
    if final_metrics != endpoint.get("metrics"):
        raise RuntimeError("latest.pt and the registered endpoint metrics differ")
    _require_state_mapping_equal(
        latest.get("model_state_dict"),
        endpoint.get("model_state_dict"),
        context="latest.pt versus endpoint raw model",
    )
    _require_state_mapping_equal(
        latest.get("ema_state_dict"),
        endpoint.get("ema_state_dict"),
        context="latest.pt versus endpoint EMA",
    )

    best_success = max(successes.values())
    if latest.get("best_selection_metric") != "rollout/success_2A":
        raise RuntimeError("latest.pt selection metric is not rollout/success_2A")
    if latest.get("best_selection_mode") != "max":
        raise RuntimeError("latest.pt selection mode is not max")
    if not math.isclose(
        float(latest.get("best_selection_value", math.nan)),
        best_success,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError("latest.pt lost the maximum registered rollout success")
    best_metrics = best.get("metrics")
    if not isinstance(best_metrics, dict) or not math.isclose(
        float(best_metrics.get("rollout/success_2A", math.nan)),
        best_success,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError("best.pt does not contain the maximum registered rollout success")
    best_step = int(best.get("step", -1))
    best_step_candidates = [
        step
        for step, value in successes.items()
        if math.isclose(value, best_success, rel_tol=0.0, abs_tol=1e-12)
    ]
    if best_step not in best_step_candidates:
        raise RuntimeError(
            f"best.pt embeds step={best_step}, not one of {best_step_candidates}"
        )
    if best.get("config") != config:
        raise RuntimeError("best.pt config differs from the registered config")
    named_best = load_checkpoint_file(
        checkpoint_dir / f"rollout_step{best_step:07d}.pt"
    )
    _require_state_mapping_equal(
        best.get("model_state_dict"),
        named_best.get("model_state_dict"),
        context="best.pt versus named best raw model",
    )
    _require_state_mapping_equal(
        best.get("ema_state_dict"),
        named_best.get("ema_state_dict"),
        context="best.pt versus named best EMA",
    )

    baseline_success = float(baseline["metrics"]["rollout/success_2A"])
    endpoint_success = float(endpoint["metrics"]["rollout/success_2A"])
    baseline_count = _success_count(
        baseline_success,
        expected_val_samples,
        context="baseline rollout/success_2A",
    )
    endpoint_count = _success_count(
        endpoint_success,
        expected_val_samples,
        context="endpoint rollout/success_2A",
    )
    return {
        "status": "passed",
        "expected_steps": expected_steps,
        "rollout_steps": actual_rollout_steps,
        "validation_samples": expected_val_samples,
        "baseline_success_2A": baseline_success,
        "baseline_success_2A_count": baseline_count,
        "endpoint_success_2A": endpoint_success,
        "endpoint_success_2A_count": endpoint_count,
        "endpoint_delta_pp": 100.0 * (endpoint_success - baseline_success),
        "registered_gate_pass": endpoint_count - baseline_count >= 11,
        "best_success_2A": best_success,
        "best_step_candidates": best_step_candidates,
        "final_metrics": final_metrics,
        "rank_rng_states": len(rank_rng_states),
        "config_exact": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-steps", type=int, default=50_000)
    parser.add_argument("--rollout-every", type=int, default=5_000)
    parser.add_argument("--expected-val-samples", type=int, default=1_076)
    parser.add_argument("--expected-world-size", type=int, default=4)
    args = parser.parse_args()

    try:
        result = audit(
            output_dir=args.output_dir,
            config_path=args.config,
            expected_steps=args.expected_steps,
            rollout_every=args.rollout_every,
            expected_val_samples=args.expected_val_samples,
            expected_world_size=args.expected_world_size,
        )
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
