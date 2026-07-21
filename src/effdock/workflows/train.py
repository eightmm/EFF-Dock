"""Canonical EFF-Dock training entry point."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def _configure_rank_device() -> None:
    """Apply the per-rank CUDA visibility workaround before importing torch."""
    local_rank = os.environ.get("LOCAL_RANK")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if local_rank is None or not visible:
        return
    devices = [device.strip() for device in visible.split(",") if device.strip()]
    if len(devices) > 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = devices[int(local_rank)]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train EFF-Dock")
    parser.add_argument("--config", type=Path, default=Path("configs/train.yaml"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--init-from", type=Path)
    args = parser.parse_args(argv)
    if args.resume and args.init_from:
        parser.error("--resume and --init-from are mutually exclusive")

    _configure_rank_device()
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    import torch
    import yaml

    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    from effdock.training.trainer import Trainer

    with args.config.open() as handle:
        config = yaml.safe_load(handle)
    trainer = Trainer(config)
    if args.resume:
        trainer.load_checkpoint(str(args.resume))
    elif args.init_from:
        trainer.load_model_weights(str(args.init_from))
    trainer.train()


if __name__ == "__main__":
    main()
