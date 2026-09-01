#!/usr/bin/env python3
"""Run an unchanged DiffDock inference script with a recorded random seed."""

from __future__ import annotations

import argparse
import builtins
import csv
import json
import os
import random
import re
import runpy
import subprocess
import sys
import tempfile
import traceback
from functools import wraps
from pathlib import Path
from typing import Any

NATIVE_UNSUPPORTED_PREFIXES = ("The receptor is too large ",)


def _option_value(arguments: list[str], option: str) -> str:
    try:
        index = arguments.index(option)
    except ValueError as exc:
        raise ValueError(f"Required upstream option is missing: {option}") from exc
    if index + 1 >= len(arguments):
        raise ValueError(f"Upstream option has no value: {option}")
    return arguments[index + 1]


def _replace_option(arguments: list[str], option: str, value: str) -> list[str]:
    updated = list(arguments)
    index = updated.index(option)
    updated[index + 1] = value
    return updated


def _has_complete_output(output_dir: Path, target: str, expected_poses: int) -> bool:
    target_dir = output_dir / target
    if not target_dir.is_dir():
        return False
    ranks: set[int] = set()
    for path in target_dir.glob("rank*_confidence*.sdf"):
        match = re.match(r"rank(\d+)_confidence", path.name)
        if match:
            ranks.add(int(match.group(1)))
    return ranks == set(range(1, expected_poses + 1))


class TargetFailureRecorder:
    """Capture DiffDock's caught per-target failures without patching upstream."""

    def __init__(self, original_print: Any) -> None:
        self.original_print = original_print
        self.failures: dict[str, dict[str, str]] = {}
        self._pending_skip_target: str | None = None

    def __call__(self, *values: object, **kwargs: object) -> None:
        if values:
            first = str(values[0])
            if first.startswith("Skipping ") and first.endswith(" because of the error:"):
                self._pending_skip_target = first[len("Skipping ") : -len(" because of the error:")]
            elif self._pending_skip_target is not None:
                message = " ".join(str(value) for value in values)
                kind = (
                    "native_unsupported"
                    if message.startswith(NATIVE_UNSUPPORTED_PREFIXES)
                    else "preprocessing_failure"
                )
                self.failures[self._pending_skip_target] = {
                    "kind": kind,
                    "message": message,
                }
                self._pending_skip_target = None
            elif first == "Failed on" and len(values) >= 3:
                raw_target = values[1]
                if isinstance(raw_target, (list, tuple)) and raw_target:
                    target = str(raw_target[0])
                else:
                    target = str(raw_target)
                self.failures[target] = {
                    "kind": "inference_failure",
                    "message": " ".join(str(value) for value in values[2:]),
                }
                traceback.print_exc()
        self.original_print(*values, **kwargs)


def _write_failure_log(path: Path | None, failures: dict[str, dict[str, str]]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema_version": 1, "failures": failures}
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _run_isolated_targets(
    *,
    upstream_script: Path,
    seed: int,
    upstream_args: list[str],
    trace_target_failures: bool,
    failure_log: Path | None,
) -> None:
    """Run each manifest row in a fresh process so one CUDA failure cannot cascade."""
    input_csv = Path(_option_value(upstream_args, "--protein_ligand_csv")).resolve()
    output_dir = Path(_option_value(upstream_args, "--out_dir")).resolve()
    expected_poses = int(_option_value(upstream_args, "--samples_per_complex"))
    skip_existing = "--skip_existing" in upstream_args
    with input_csv.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        rows = list(reader)
    if not fieldnames:
        raise ValueError(f"DiffDock input CSV has no header: {input_csv}")

    combined: dict[str, dict[str, str]] = {}
    with tempfile.TemporaryDirectory(prefix="effdock_diffdock_targets_") as temp_dir:
        temp_root = Path(temp_dir)
        for index, row in enumerate(rows):
            target = str(row.get("complex_name", f"row_{index}"))
            if skip_existing and _has_complete_output(
                output_dir, target, expected_poses
            ):
                print(
                    f"EFF-Dock external runner: skipping complete target {target} "
                    f"({expected_poses} poses)",
                    flush=True,
                )
                continue
            target_csv = temp_root / f"target_{index:04d}.csv"
            with target_csv.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(row)
            target_failure_log = temp_root / f"target_{index:04d}_failures.json"
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--upstream-script",
                str(upstream_script),
                "--seed",
                str(seed),
                "--failure-log",
                str(target_failure_log),
            ]
            if trace_target_failures:
                command.append("--trace-target-failures")
            command.extend(
                _replace_option(
                    upstream_args,
                    "--protein_ligand_csv",
                    str(target_csv),
                )
            )
            completed = subprocess.run(command, check=False)
            if target_failure_log.is_file():
                payload = json.loads(target_failure_log.read_text())
                combined.update(payload.get("failures", {}))
            if completed.returncode != 0:
                combined[target] = {
                    "kind": "process_failure",
                    "message": f"isolated DiffDock process exited {completed.returncode}",
                }
    _write_failure_log(failure_log, combined)


def install_empty_torsion_edge_compat() -> None:
    """Honor requested torsion output size when the upstream edge set is empty."""
    import torch
    from models.tensor_layers import TensorProductConvLayer

    original_forward = TensorProductConvLayer.forward

    @wraps(original_forward)
    def compatible_forward(
        layer: object,
        node_attr: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_sh: torch.Tensor,
        out_nodes: object = None,
        reduce: str = "mean",
        edge_weight: object = 1.0,
    ) -> torch.Tensor:
        if edge_index.shape[1] == 0 and out_nodes is not None:
            size = int(out_nodes.item()) if torch.is_tensor(out_nodes) else int(out_nodes)
            return torch.zeros(
                (size, layer.out_size), dtype=node_attr.dtype, device=node_attr.device
            )
        return original_forward(
            layer,
            node_attr,
            edge_index,
            edge_attr,
            edge_sh,
            out_nodes=out_nodes,
            reduce=reduce,
            edge_weight=edge_weight,
        )

    TensorProductConvLayer.forward = compatible_forward


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--upstream-script", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--trace-target-failures", action="store_true")
    parser.add_argument("--isolate-targets", action="store_true")
    parser.add_argument("--failure-log", type=Path)
    args, upstream_args = parser.parse_known_args()

    upstream_script = args.upstream_script.resolve()
    if not upstream_script.is_file():
        raise FileNotFoundError(f"DiffDock inference script not found: {upstream_script}")

    if args.isolate_targets:
        _run_isolated_targets(
            upstream_script=upstream_script,
            seed=args.seed,
            upstream_args=upstream_args,
            trace_target_failures=args.trace_target_failures,
            failure_log=args.failure_log,
        )
        return

    os.environ["PYTHONHASHSEED"] = str(args.seed)
    random.seed(args.seed)

    # DiffDock itself imports both NumPy and Torch, so initializing their RNGs
    # here covers stochastic conformer initialization and diffusion sampling.
    import numpy as np
    import torch

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    upstream_root = upstream_script.parent
    os.chdir(upstream_root)
    sys.path.insert(0, str(upstream_root))
    install_empty_torsion_edge_compat()
    sys.argv = [str(upstream_script), *upstream_args]
    print(f"EFF-Dock external runner: DiffDock seed={args.seed}", flush=True)
    recorder = TargetFailureRecorder(builtins.print)
    if args.trace_target_failures or args.failure_log is not None:
        builtins.print = recorder
    try:
        runpy.run_path(str(upstream_script), run_name="__main__")
    finally:
        builtins.print = recorder.original_print
        _write_failure_log(args.failure_log, recorder.failures)


if __name__ == "__main__":
    main()
