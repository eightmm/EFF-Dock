#!/usr/bin/env python3
"""Seed Interformer's native PyVina generators, then execute its sampler."""

from __future__ import annotations

import argparse
import os
import runpy
import sys
from pathlib import Path

import pyvina_core


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-script", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--normalize-db-markers", action="store_true")
    return parser.parse_known_args()


def normalize_db_markers(upstream_args: list[str]) -> None:
    """Expose gdbm's single-file output to the upstream dbm.dumb glob."""
    try:
        cwd_index = next(
            index for index, value in enumerate(upstream_args) if value in {"-c", "--cwd"}
        )
        energy_dir = Path(upstream_args[cwd_index + 1]).resolve()
    except (StopIteration, IndexError) as error:
        raise ValueError("--normalize-db-markers requires the upstream --cwd argument") from error

    databases = sorted((energy_dir / "gaussian_predict").glob("*_G.db"))
    if not databases:
        raise FileNotFoundError(f"No Interformer energy databases found under {energy_dir}")
    for database in databases:
        marker = Path(f"{database}.dat")
        if marker.exists() or marker.is_symlink():
            if marker.resolve() != database.resolve():
                raise FileExistsError(f"Conflicting Interformer DB marker: {marker}")
            continue
        marker.symlink_to(database.name)


def main() -> None:
    args, upstream_args = parse_args()
    if args.normalize_db_markers:
        normalize_db_markers(upstream_args)
    try:
        thread_count = len(os.sched_getaffinity(0))
    except AttributeError:
        thread_count = os.cpu_count() or 1
    seeds = [args.seed + index for index in range(thread_count)]
    pyvina_core.wrappers_for_random.SetGeneratorsForThreadsGivenSeeds(seeds)
    sys.argv = [str(args.upstream_script), *upstream_args]
    runpy.run_path(str(args.upstream_script), run_name="__main__")


if __name__ == "__main__":
    main()
