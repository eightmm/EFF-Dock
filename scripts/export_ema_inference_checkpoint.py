"""Export an EFF-Dock EMA checkpoint for the existing inference loaders."""

from __future__ import annotations

import argparse
from pathlib import Path

from effdock.checkpoint import export_ema_inference_checkpoint, load_checkpoint_file


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    output = export_ema_inference_checkpoint(args.source, args.output)
    checkpoint = load_checkpoint_file(output)
    print(
        f"EMA inference checkpoint: {output} "
        f"(step={checkpoint.get('step')}, n_averaged={checkpoint['ema_n_averaged']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
