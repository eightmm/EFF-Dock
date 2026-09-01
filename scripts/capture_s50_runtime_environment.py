#!/usr/bin/env python3
"""Emit the deterministic runtime fingerprint for the S50 confidence DAG."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import sys
from pathlib import Path
from typing import Any

import rdkit
import torch

SCHEMA_VERSION = "effdock.s50_confidence_runtime_environment.v1"


def _distribution_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError(f"required runtime distribution is missing: {name}") from exc


def runtime_environment() -> dict[str, Any]:
    """Return only stable identities that affect numerical execution."""

    return {
        "schema_version": SCHEMA_VERSION,
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "executable_realpath": str(Path(sys.executable).resolve(strict=True)),
            "prefix_realpath": str(Path(sys.prefix).resolve(strict=True)),
        },
        "torch": {
            "version": torch.__version__,
            "git_version": torch.version.git_version,
            "cuda_build_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
        },
        "rdkit": {"version": rdkit.__version__},
        "cuequivariance": {
            name: _distribution_version(name)
            for name in (
                "cuequivariance",
                "cuequivariance-torch",
                "cuequivariance-ops-cu13",
                "cuequivariance-ops-torch-cu13",
            )
        },
    }


def main() -> None:
    print(json.dumps(runtime_environment(), indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
