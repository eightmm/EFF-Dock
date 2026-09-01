"""Run one real EFF-Dock optimizer step for the early-time fine-tune config."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path


def _configure_rank_device() -> None:
    """Match the canonical training entry point's per-rank CUDA visibility."""
    local_rank = os.environ.get("LOCAL_RANK")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if local_rank is None or not visible:
        return
    devices = [device.strip() for device in visible.split(",") if device.strip()]
    if len(devices) > 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = devices[int(local_rank)]


_configure_rank_device()


def main() -> int:
    import torch
    import yaml

    from effdock.checkpoint import load_checkpoint_file
    from effdock.training.trainer import Trainer, _all_ranks_finite, cleanup_ddp

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/eff-dock"))
    parser.add_argument("--exercise-rollout", action="store_true")
    parser.add_argument("--resume-only", action="store_true")
    parser.add_argument("--resume-checkpoint", type=Path)
    args = parser.parse_args()

    with args.config.open() as handle:
        config = yaml.safe_load(handle)

    run_id = os.environ.get("SLURM_JOB_ID", "local")
    output_dir = args.output_root / f"early-time-ft-smoke-{run_id}"
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    config["data"]["num_workers"] = 0
    config["training"].update(
        {
            "max_steps": 1,
            "batch_size": 2,
            "gradient_accumulation_steps": 1,
            # Keep one full rank-local batch without duplicating the tiny
            # overfit subset under DistributedSampler.
            "overfit_batches": world_size,
            "size_aware_batches": False,
        }
    )
    config["logging"].update(
        {
            "use_wandb": False,
            "val_every": 0,
            "rollout_every": 0,
            "eval_on_start": False,
            "output_dir": str(output_dir),
        }
    )
    if args.exercise_rollout:
        config["logging"].update(
            {
                "rollout_every": 1,
                "eval_on_start": True,
                "rollout_steps": 1,
                "rollout_max_samples": max(world_size, 4),
                "rollout_progress_every": 1,
            }
        )

    trainer = Trainer(config)
    if args.resume_only:
        if args.resume_checkpoint is None:
            raise ValueError("--resume-only requires --resume-checkpoint")
        trainer.load_checkpoint(str(args.resume_checkpoint))
        if trainer.global_step != 1:
            raise RuntimeError(f"resume smoke loaded step={trainer.global_step}")
        saved = load_checkpoint_file(args.resume_checkpoint)
        rank_rng_states = saved.get("rank_rng_states")
        if not isinstance(rank_rng_states, list) or len(rank_rng_states) != world_size:
            raise RuntimeError("resume smoke checkpoint lacks exact per-rank RNG states")
        expected_rng = rank_rng_states[trainer.rank]
        actual_rng = trainer._capture_rng_state()
        if expected_rng["python"] != actual_rng["python"]:
            raise RuntimeError("resume smoke restored the wrong Python RNG stream")
        if expected_rng["numpy"] != actual_rng["numpy"]:
            raise RuntimeError("resume smoke restored the wrong NumPy RNG stream")
        if not torch.equal(expected_rng["torch"], actual_rng["torch"]):
            raise RuntimeError("resume smoke restored the wrong Torch RNG stream")
        expected_cuda = expected_rng.get("cuda", [])
        actual_cuda = actual_rng.get("cuda", [])
        if len(expected_cuda) != len(actual_cuda) or any(
            not torch.equal(expected, actual)
            for expected, actual in zip(expected_cuda, actual_cuda, strict=True)
        ):
            raise RuntimeError("resume smoke restored the wrong CUDA RNG stream")
        if trainer.rank == 0:
            print("early_time_training_smoke: exact per-rank resume load ok", flush=True)
        cleanup_ddp(trainer.world_size)
        return 0

    trainer.load_model_weights(str(args.checkpoint))

    if world_size > 1:
        local_probe = torch.tensor(
            float("nan") if trainer.rank == 0 else 1.0,
            device=trainer.device,
        )
        if _all_ranks_finite(local_probe, world_size):
            raise RuntimeError("distributed finite probe failed to propagate rank-0 NaN")
        if trainer.rank == 0:
            print("early_time_training_smoke: distributed finite probe ok", flush=True)

    trainer.train()

    checkpoint_path = output_dir / "checkpoints" / "latest.pt"
    if trainer.rank == 0:
        saved = load_checkpoint_file(checkpoint_path)
        if int(saved.get("step", -1)) != 1:
            raise RuntimeError(f"one-step smoke checkpoint has step={saved.get('step')!r}")
        train_loss = float(saved.get("metrics", {}).get("train_loss", math.nan))
        if not math.isfinite(train_loss):
            raise FloatingPointError(f"one-step smoke produced non-finite train_loss={train_loss}")
        if args.exercise_rollout and "rollout/success_2A" not in saved.get("metrics", {}):
            raise RuntimeError("final latest.pt lost rollout metrics")
        if args.exercise_rollout:
            baseline = load_checkpoint_file(output_dir / "checkpoints" / "rollout_step0000000.pt")
            if int(baseline.get("step", -1)) != 0:
                raise RuntimeError("eval_on_start baseline checkpoint was not preserved")
            baseline_success = float(baseline["metrics"]["rollout/success_2A"])
            final_success = float(saved["metrics"]["rollout/success_2A"])
            if float(saved["best_selection_value"]) != max(baseline_success, final_success):
                raise RuntimeError("latest.pt lost the rollout best-selection threshold")
        if torch.cuda.is_available():
            print(
                "early_time_training_smoke: ok "
                f"world_size={world_size} loss={train_loss:.6f} "
                f"max_cuda_gib={torch.cuda.max_memory_allocated() / 2**30:.3f}"
            )
        else:
            print(f"early_time_training_smoke: ok loss={train_loss:.6f} device=cpu")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
