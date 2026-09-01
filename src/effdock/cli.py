"""Unified command-line interface for EFF-Dock."""

from __future__ import annotations

import sys
from collections.abc import Callable

_HELP = """usage: eff-dock <command> [options]

commands:
  data curate     curate the PLINDER pool
  data prepare    preprocess PLINDER complexes
  data split      create the strict train/validation split
  data benchmark  prepare frozen external redocking inputs
  data external   normalize PhiBench, FoldBench, and OpenBind inputs
  train           train or resume EFF-Dock
  confidence prepare generate labeled pose shards for confidence training
  confidence train train or resume the learned pose-confidence model
  evaluate        evaluate a checkpoint on an external benchmark
  benchmark       aggregate completed external benchmark shards
  dock            dock one ligand into an explicitly defined pocket
  physical trace  trace unified guidance on crystal/saved trajectory coordinates

Run `eff-dock <command> --help` for command-specific options.
"""


def _resolve(command: str, data_command: str | None = None) -> Callable[[list[str] | None], None]:
    if command == "dock":
        from effdock.inference.docking import main

        return main
    if command == "train":
        from effdock.workflows.train import main

        return main
    if command == "confidence" and data_command == "train":
        from effdock.workflows.train_confidence import main

        return main
    if command == "confidence" and data_command == "prepare":
        from effdock.workflows.prepare_confidence import main

        return main
    if command == "evaluate":
        from effdock.workflows.evaluate import main

        return main
    if command == "benchmark":
        from effdock.workflows.benchmark_report import main

        return main
    if command == "physical" and data_command == "trace":
        from effdock.workflows.trace_physical import main

        return main
    if command == "data" and data_command == "curate":
        from effdock.workflows.curate import main

        return main
    if command == "data" and data_command == "prepare":
        from effdock.workflows.prepare import main

        return main
    if command == "data" and data_command == "split":
        from effdock.workflows.split import main

        return main
    if command == "data" and data_command == "benchmark":
        from effdock.workflows.benchmark_data import main

        return main
    if command == "data" and data_command == "external":
        from effdock.workflows.external_benchmark_data import main

        return main
    raise ValueError(f"unknown command: {' '.join(x for x in (command, data_command) if x)}")


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        print(_HELP)
        return

    command = args.pop(0)
    data_command = None
    if command in {"data", "confidence", "physical"}:
        if not args or args[0] in {"-h", "--help"}:
            if command == "data":
                print("usage: eff-dock data {curate,prepare,split,benchmark,external} [options]")
            elif command == "confidence":
                print("usage: eff-dock confidence {prepare,train} [options]")
            else:
                print("usage: eff-dock physical {trace} [options]")
            return
        data_command = args.pop(0)
    try:
        handler = _resolve(command, data_command)
    except ValueError as exc:
        raise SystemExit(f"{exc}\n\n{_HELP}") from exc
    handler(args)


if __name__ == "__main__":
    main()
