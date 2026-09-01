#!/usr/bin/env python3
"""Launch PoseBench Vina with narrowly scoped legacy import compatibility."""

from __future__ import annotations

import io
import runpy
import sys
import types
from pathlib import Path


def install_rdkit_six_compat() -> bool:
    """Provide the sole ``rdkit.six`` symbol required by Meeko 0.6.0a3."""
    try:
        import rdkit.six  # type: ignore[import-not-found]  # noqa: F401
    except ModuleNotFoundError:
        import rdkit

        compatibility_module = types.ModuleType("rdkit.six")
        compatibility_module.StringIO = io.StringIO  # type: ignore[attr-defined]
        sys.modules["rdkit.six"] = compatibility_module
        setattr(rdkit, "six", compatibility_module)
        return True
    return False


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: posebench_vina_compat.py ENTRY_SCRIPT [ARGS ...]")
    entry_script = Path(sys.argv[1]).resolve()
    if not entry_script.is_file():
        raise FileNotFoundError(entry_script)

    install_rdkit_six_compat()
    sys.argv = [str(entry_script), *sys.argv[2:]]
    runpy.run_path(str(entry_script), run_name="__main__")


if __name__ == "__main__":
    main()
