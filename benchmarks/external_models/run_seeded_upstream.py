#!/usr/bin/env python3
"""Execute an upstream Python CLI after initializing recorded RNG seeds."""

from __future__ import annotations

import argparse
import os
import random
import runpy
import sys
import types
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--upstream-script", type=Path, required=True)
    parser.add_argument(
        "--upstream-cwd",
        type=Path,
        default=None,
        help="Working directory for an upstream CLI that assumes its repository root.",
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--stub-optional-posebusters-em",
        action="store_true",
        help=(
            "Bypass DiffDock-Pocket's eager OpenFF relaxation import. "
            "The stub raises if relaxation is actually requested."
        ),
    )
    parser.add_argument(
        "--stub-rldiff-postprocess",
        action="store_true",
        help=(
            "Keep RLDiff's --minimize_and_rerank sampling branch but defer "
            "its CPU-heavy smina/GNINA stage to a separate job."
        ),
    )
    parser.add_argument(
        "--stub-surfdock-force-optimize",
        action="store_true",
        help=(
            "Bypass SurfDock's eager OpenFF force-optimization import. "
            "The shim raises if --force_optimize reaches that path."
        ),
    )
    parser.add_argument(
        "--surfdock-compat",
        action="store_true",
        help=(
            "Install EFF-Dock's runtime-only SurfDock PDB and conformer "
            "compatibility adapters without editing the frozen checkout."
        ),
    )
    parser.add_argument(
        "--stub-diffbindfr-pymol",
        action="store_true",
        help=(
            "Bypass DiffBindFR's eager PyMOL import. The shim raises if a "
            "real PyMOL operation is used; smina error correction is allowed."
        ),
    )
    args, upstream_args = parser.parse_known_args()

    # Preserve symlink components: some upstream projects (notably RLDiff)
    # derive sibling package locations from the invocation path itself.
    upstream_script = Path(os.path.abspath(args.upstream_script))
    if not upstream_script.is_file():
        raise FileNotFoundError(f"Upstream script not found: {upstream_script}")

    os.environ["PYTHONHASHSEED"] = str(args.seed)
    random.seed(args.seed)

    import numpy as np
    import torch

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    upstream_root = (
        Path(os.path.abspath(args.upstream_cwd))
        if args.upstream_cwd is not None
        else upstream_script.parent
    )
    if not upstream_root.is_dir():
        raise FileNotFoundError(f"Upstream working directory not found: {upstream_root}")
    os.chdir(upstream_root)
    sys.path.insert(0, str(upstream_root))
    if args.stub_optional_posebusters_em:
        optional_module = types.ModuleType("utils.posebusters_em")
        openmm_module = types.ModuleType("openmm")
        openmm_unit_module = types.ModuleType("openmm.unit")

        def unavailable_relaxation(*_args: object, **_kwargs: object) -> None:
            raise RuntimeError(
                "PoseBusters OpenFF relaxation was disabled by the EFF-Dock "
                "inference compatibility shim"
            )

        optional_module.optimize_ligand_in_pocket = unavailable_relaxation
        # inference.py also imports these two unit symbols solely for the
        # optional relaxation report, before argparse has inspected --relax.
        openmm_unit_module.megajoule = object()
        openmm_unit_module.mole = object()
        openmm_module.unit = openmm_unit_module
        sys.modules[optional_module.__name__] = optional_module
        sys.modules[openmm_module.__name__] = openmm_module
        sys.modules[openmm_unit_module.__name__] = openmm_unit_module
    if args.stub_rldiff_postprocess:
        minimize_module = types.ModuleType("utils.minimize_utils")

        def deferred_postprocess(*_args: object, **_kwargs: object) -> None:
            print(
                "EFF-Dock external runner: deferred RLDiff smina/GNINA "
                "post-processing to CPU-only job",
                flush=True,
            )

        minimize_module.minimize_and_rerank = deferred_postprocess
        sys.modules[minimize_module.__name__] = minimize_module
    if args.stub_surfdock_force_optimize:
        minimize_module = types.ModuleType("force_optimize.minimize_utils")

        def unavailable_force_optimization(
            *_args: object, **_kwargs: object
        ) -> None:
            raise RuntimeError(
                "SurfDock force optimization is not part of the native "
                "no-refinement benchmark arm"
            )

        minimize_module.UpdateGrpah = unavailable_force_optimization
        minimize_module.GetfixedPDB = unavailable_force_optimization
        minimize_module.GetFFGenerator = unavailable_force_optimization
        sys.modules[minimize_module.__name__] = minimize_module
    if args.surfdock_compat:
        from surfdock_compat import install_surfdock_compat

        install_surfdock_compat()
    if args.stub_diffbindfr_pymol:
        pymol_module = types.ModuleType("pymol")
        pymol2_module = types.ModuleType("pymol2")

        class UnavailablePyMOL:
            def __init__(self, *_args: object, **_kwargs: object) -> None:
                raise RuntimeError(
                    "DiffBindFR attempted a PyMOL operation outside the "
                    "official smina error-correction path"
                )

            def __getattr__(self, _name: str) -> object:
                raise RuntimeError(
                    "DiffBindFR attempted a PyMOL command outside the "
                    "official smina error-correction path"
                )

        class UnavailableCommand:
            def __getattr__(self, _name: str) -> object:
                raise RuntimeError(
                    "PyMOL was called from the no-error-correction benchmark arm"
                )

        pymol_module.cmd = UnavailableCommand()
        pymol2_module.PyMOL = UnavailablePyMOL
        sys.modules[pymol_module.__name__] = pymol_module
        sys.modules[pymol2_module.__name__] = pymol2_module
    sys.argv = [str(upstream_script), *upstream_args]
    print(
        f"EFF-Dock external runner: script={upstream_script.name} seed={args.seed}",
        flush=True,
    )
    runpy.run_path(str(upstream_script), run_name="__main__")


if __name__ == "__main__":
    main()
