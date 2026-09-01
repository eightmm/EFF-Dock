#!/usr/bin/env python3
"""Run pinned upstream SigmaDock with the narrow predicted-receptor parser fix."""

from __future__ import annotations

import runpy
from pathlib import Path

import sigmadock
from sigmadock_compat import install_sigmadock_parser_compat


def main() -> None:
    install_sigmadock_parser_compat()
    upstream_root = Path(sigmadock.__file__).resolve().parents[2]
    sample_script = upstream_root / "scripts/sample.py"
    if not sample_script.is_file():
        raise FileNotFoundError(sample_script)
    runpy.run_path(str(sample_script), run_name="__main__")


if __name__ == "__main__":
    main()
