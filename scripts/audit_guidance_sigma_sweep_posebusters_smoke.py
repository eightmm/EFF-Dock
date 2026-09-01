#!/usr/bin/env python3
"""Fail-closed audit for the primary 10-cell SigmaDock-compatible PB smoke run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from effdock.workflows.guidance_sigma_sweep_posebusters_report import (
    DATASETS,
    POSEBUSTERS_CONFIG,
    POSEBUSTERS_VERSION,
    SELECTORS,
    SIGMAS,
    run_name,
)

SMOKE_IDS = {"astex": "1jje", "posebusters": "7b2c_tp7"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = []
    for selector in SELECTORS:
        for dataset, expected_count in DATASETS.items():
            for sigma in SIGMAS:
                name = run_name(dataset, sigma)
                summary_path = (
                    args.sweep_root
                    / "sigmadock_posebusters/smoke"
                    / selector
                    / name
                    / "shard-000-of-001.summary.json"
                )
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                expected = {
                    "posebusters_version": POSEBUSTERS_VERSION,
                    "config": POSEBUSTERS_CONFIG,
                    "selector": selector,
                    "only_id": SMOKE_IDS[dataset],
                    "expected_discovered_count": expected_count,
                    "num_discovered_total": expected_count,
                    "num_assigned": 1,
                    "num_success": 1,
                    "num_failed": 0,
                    "failures": [],
                    "input_hashes_verified": True,
                    "num_input_hashes_verified": 1,
                }
                for key, value in expected.items():
                    if summary.get(key) != value:
                        raise ValueError(
                            f"{summary_path}: {key} must be {value!r}, got {summary.get(key)!r}"
                        )
                records.append(str(summary_path.resolve()))
    result = {
        "schema_version": "effdock.sigmadock_compatible_pb_smoke_audit.v1",
        "status": "passed",
        "cells": len(records),
        "summaries": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
