#!/usr/bin/env python3
"""Compatibility replacement for PoseBench's bundled ADFR receptor writer.

PoseBench still invokes the Python-2 ``prepare_receptor4.py`` interface.  Its
bundled NumPy extension is ABI-incompatible on the current cluster, so this
adapter accepts the same ``-r/-o`` arguments and delegates PDBQT writing to the
Meeko installation already pinned in the PoseBench environment.
"""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

try:
    from .posebench_vina_compat import install_rdkit_six_compat
except ImportError:  # direct script execution
    from posebench_vina_compat import install_rdkit_six_compat


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--receptor", type=Path, required=True)
    parser.add_argument("-o", "--output", type=Path, required=True)
    return parser.parse_args()


def meeko_output_prefix(output: Path) -> Path:
    return output.with_suffix("") if output.suffix == ".pdbqt" else output


def select_meeko_receptor_source(receptor: Path) -> Path:
    """Recover PoseBench's original temp PDB before Reduce/OpenBabel rewrites."""
    suffix = "_reduced_prepped.pdb"
    receptor_text = str(receptor)
    if receptor_text.endswith(suffix):
        original = Path(receptor_text[: -len(suffix)] + ".pdb")
        if original.is_file():
            return original
    return receptor


def main() -> None:
    args = parse_args()
    receptor = args.receptor.resolve()
    output = args.output.resolve()
    if not receptor.is_file():
        raise FileNotFoundError(receptor)
    meeko_receptor = select_meeko_receptor_source(receptor)

    install_rdkit_six_compat()
    meeko_entry = Path(sys.executable).resolve().parent / "mk_prepare_receptor.py"
    if not meeko_entry.is_file():
        raise FileNotFoundError(meeko_entry)

    output.parent.mkdir(parents=True, exist_ok=True)
    sys.argv = [
        str(meeko_entry),
        "--pdb",
        str(meeko_receptor),
        "-o",
        str(meeko_output_prefix(output)),
        "--skip_gpf",
        "--allow_bad_res",
    ]
    runpy.run_path(str(meeko_entry), run_name="__main__")
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError(f"Meeko did not create the requested receptor: {output}")


if __name__ == "__main__":
    main()
